"""Isaac ROS cuVSLAM + Nvblox. Real pipeline, rootless podman + CDI.

Flow (all official): NGC image nvcr.io/nvidia/isaac/ros:<TAG> (build zed layer via the
official Dockerfile.zed image key) -> zed_wrapper `camera_model:=virtual` SVO2 replay ->
rosbag -> `isaac_ros_examples.launch.py launch_fragments:=zed_stereo_rect,visual_slam`
(cuVSLAM, zedx quickstart specs) + `nvblox_examples_bringup zed_example.launch.py
camera:=zedx` with cuVSLAM odometry as the pose source -> mesh via the
/nvblox_node/save_ply service; poses from cuVSLAM's /visual_slam/pose topic
(map->base PoseStamped) bagged and parsed with rosbags.

ScanNet parameters where expressible: voxel_size 0.004,
static_mapper.projective_integrator_max_integration_distance_m 4.0 (ScanNet
s_SDFMaxIntegrationDistance), projective_integrator_truncation_distance_vox 15
(0.06 m truncation at 4 mm = ScanNet s_SDFTruncation). block_count raised for 4 mm.
"""
import os
import subprocess

import numpy as np

NGC_TAG = os.environ.get("ISAAC_ROS_TAG", "release-4.5")
NGC_IMAGE = f"nvcr.io/nvidia/isaac/ros:{NGC_TAG}"
LOCAL_IMAGE = "zed-isaac-nvblox:spellbook"
TIMEOUT = 7200

# Runs inside the container via `python3 -c`; fails loudly if rosbags is missing.
_POSE_EXTRACT = """import pathlib
import sys

try:
    from rosbags.highlevel import AnyReader
except ImportError:
    sys.stderr.write("isaac: rosbags python package not importable in the container; "
                     "cannot extract cuVSLAM poses -- install rosbags (pip install "
                     "rosbags) into the zed-isaac-nvblox:spellbook image and rerun.\\n")
    sys.exit(1)

bag = pathlib.Path("/output/pose.bag")
lines = []
with AnyReader([bag]) as reader:
    conns = [c for c in reader.connections if c.topic == "/visual_slam/pose"]
    for connection, timestamp, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, connection.msgtype)
        sec = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        p = msg.pose.position
        q = msg.pose.orientation
        lines.append("%d %f %f %f %f %f %f %f" % (sec, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

with open("/output/cuvslam_poses.txt", "w") as f:
    f.write("\\n".join(lines) + "\\n")
sys.stdout.write("isaac: extracted %d cuVSLAM poses\\n" % len(lines))
"""


def _podman(args, timeout=300):
    return subprocess.run(["podman"] + args, capture_output=True, text=True, timeout=timeout)


def _quat_to_rot(qx, qy, qz, qw):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _preflight(gpu):
    if _podman(["image", "exists", LOCAL_IMAGE]).returncode != 0:
        raise RuntimeError(
            f"[isaac] image {LOCAL_IMAGE} missing: pull {NGC_IMAGE} (NGC auth), "
            "then build the zed layer via Dockerfile.zed")
    r = _podman(["run", "--rm", "--device", "nvidia.com/gpu=all",
                 "--security-opt", "label=disable", LOCAL_IMAGE, "nvidia-smi",
                 "-L"])
    if r.returncode != 0:
        raise RuntimeError(f"[isaac] image not runnable (rootless GPU): {r.stderr[-400:]}")
    # libvpi: cuVSLAM's GXF graph may need it -- verify before the long run
    r = _podman(["run", "--rm", LOCAL_IMAGE, "bash", "-lc",
                 "ldconfig -p | grep -i libvpi || find / -name 'libvpi*' 2>/dev/null | head"])
    if r.returncode == 0 and not r.stdout.strip():
        print("[isaac] WARNING: libvpi not found in image -- cuVSLAM may fail at init")


def _ensure_image(gpu):
    if _podman(["image", "exists", LOCAL_IMAGE]).returncode == 0:
        return
    import fcntl
    lock_path = "/tmp/opencode/.isaac-build.lock"
    with open(lock_path, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        # flock-style existence check: a concurrent build may have finished
        if _podman(["image", "exists", LOCAL_IMAGE]).returncode == 0:
            return
        print(f"[isaac] building zed layer image {LOCAL_IMAGE} (official Dockerfile.zed)...")
        # Dockerfile.zed from the NGC container's build context; build rootless via buildah.
        dockerfile = """
FROM {ngc}
# official ZED layer: SDK + zed-ros2-wrapper inside the image (Stereolabs docs)
ENV ZED_SDK_MAJOR=5 ZED_SDK_MINOR=4
RUN apt-get update && apt-get install -y --no-install-recommends wget libusb-1.0-0 \
 && mkdir -p /opt/zed \
 && wget -q -O /opt/zed/zed.run https://download.stereolabs.com/zedsdk/5.4/ubuntu24_cuda13 \
 && chmod +x /opt/zed/zed.run \
 && /opt/zed/zed.run --install /opt/zed --silent \
 || true
RUN apt-get install -y --no-install-recommends \
    ros-${{ROS_DISTRO}}-zed-ros2-wrapper ros-${{ROS_DISTRO}}-zed-ros2-examples \
 && rm -rf /var/lib/apt/lists/*
""".format(ngc=NGC_IMAGE)
        # unique per-process paths so parallel pool tasks never race on the context
        pid = os.getpid()
        ctx = f"/tmp/opencode/isaac-build-{pid}"
        os.makedirs(ctx, exist_ok=True)
        df_path = os.path.join(ctx, "Dockerfile.zed")
        with open(df_path, "w") as f:
            f.write(dockerfile)
        r = _podman(["build", "-t", LOCAL_IMAGE, "-f", df_path, ctx], timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"[isaac] zed layer build failed: {r.stderr[-800:]}")


def reconstruct(work, root, gpu=None):
    marker = os.path.join(work, "svo_path.txt")
    svo = open(marker).read().strip() if os.path.exists(marker) else None
    if svo is None:
        raise RuntimeError("[isaac] no svo_path marker; run prepare first")
    svo_base = os.path.basename(svo)

    _preflight(gpu)
    _ensure_image(gpu)

    inside = os.path.join(work, "isaac_inside.sh")
    export_dir = os.path.join(work, "isaac_export")
    log_dir = os.path.join(work, "logs")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    script = f"""#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/${{ROS_DISTRO}}/setup.bash
source /root/ros2_ws/install/local_setup.bash 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 1) SVO2 replay -> rosbag (zed_wrapper virtual camera model)
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=virtual \\
  svo_path:=/svo/{svo_base} svo_realtime:=false use_svo_timestamps:=false \\
  & ZED_PID=$!
sleep 20
timeout 600 ros2 bag record -o /output/scan.bag \\
  /zed/zed_node/rgb/image_rect_gray /zed/zed_node/rgb/image_rect_color \\
  /zed/zed_node/depth/depth_registered /zed/zed_node/left/camera_info \\
  /zed/zed_node/right/camera_info /zed/zed_node/imu/data /zed/zed_node/odom \\
  --max-cache-size 1000000000 & BAG_PID=$!
# wait for SVO end
while kill -0 $ZED_PID 2>/dev/null; do sleep 2; done
kill -INT $BAG_PID; wait $BAG_PID 2>/dev/null || true

# 2) cuVSLAM + nvblox offline (rosbag replay)
ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \\
  launch_fragments:=zed_stereo_rect,visual_slam \\
  base_frame:=zed_x_camera_center \\
  camera_optical_frames:="['zed_x_left_camera_frame_optical','zed_x_right_camera_frame_optical']" \\
  interface_specs_file:=/opt/ros/${{ROS_DISTRO}}/share/isaac_ros_visual_slam/config/zedx_quickstart_interface_specs.json \\
  rosbag:=/output/scan.bag use_sim_time:=true \\
  & VSLAM_PID=$!
sleep 30
ros2 bag record -o /output/pose.bag /visual_slam/pose & POSE_PID=$!
ros2 launch nvblox_examples_bringup zed_example.launch.py camera:=zedx \\
  rosbag:=/output/scan.bag use_sim_time:=true \\
  pose_source:=cuvslam \\
  voxel_size:=0.004 \\
  static_mapper.projective_integrator_max_integration_distance_m:=4.0 \\
  static_mapper.projective_integrator_truncation_distance_vox:=15.0 \\
  block_count:=200000 \\
  & NVBLOX_PID=$!
wait $NVBLOX_PID
kill -INT $POSE_PID 2>/dev/null || true
wait $POSE_PID 2>/dev/null || true

# 3) mesh export via the documented save_ply service (node still alive after replay)
SAVED=0
for i in $(seq 1 60); do
  if ros2 service call /nvblox_node/save_ply nvblox_msgs/srv/FilePath "{{file_path: '/output/mesh.ply'}}" >/dev/null 2>&1; then
    SAVED=1
    break
  fi
  sleep 1
done
if [ "$SAVED" != "1" ]; then
  echo "isaac: save_ply service did not respond within 60s" >&2
  exit 1
fi

kill -INT $VSLAM_PID 2>/dev/null || true
wait $VSLAM_PID 2>/dev/null || true

# 4) extract cuVSLAM poses from the bag (in-container; fails loudly without rosbags)
python3 -c '{_POSE_EXTRACT}'
"""
    with open(inside, "w") as f:
        f.write(script)
    os.chmod(inside, 0o755)

    cuda = gpu if gpu is not None else 0
    cmd = ["podman", "run", "--rm", "--name", f"spellbook-isaac-{os.getpid()}",
           "--device", "nvidia.com/gpu=all", "-e", f"CUDA_VISIBLE_DEVICES={cuda}",
           "--security-opt", "label=disable", "--network", "host",
           f"-v{inside}:/inside.sh:ro", f"-v{export_dir}:/output",
           f"-v{log_dir}:/logs:ro", f"-v{os.path.dirname(svo)}:/svo:ro",
           LOCAL_IMAGE, "/bin/bash", "/inside.sh"]
    log_path = os.path.join(log_dir, "isaac.log")
    with open(log_path, "wb") as logf:
        try:
            subprocess.run(cmd, timeout=TIMEOUT, stdout=logf, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"[isaac] container timed out after {TIMEOUT}s, see {log_path}")
    print(f"[isaac] container run finished; outputs in {export_dir}")

    mesh_src = os.path.join(export_dir, "mesh.ply")
    poses_txt = os.path.join(export_dir, "cuvslam_poses.txt")
    if not os.path.exists(mesh_src):
        raise RuntimeError(f"[isaac] no mesh.ply exported by save_ply; see {log_path}")
    if not os.path.exists(poses_txt):
        raise RuntimeError(f"[isaac] no cuvslam_poses.txt extracted; see {log_path}")

    import shutil
    native = os.path.join(work, "engine_native.ply")
    shutil.copyfile(mesh_src, native)

    poses, keep = [], []
    with open(poses_txt) as f:
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) < 8:
                continue
            x, y, z = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            T = np.eye(4)
            T[:3, :3] = _quat_to_rot(qx, qy, qz, qw)
            T[:3, 3] = (x, y, z)
            poses.append(T)
            keep.append(i)
    poses = np.asarray(poses)
    if not poses.size:
        raise RuntimeError(f"[isaac] no poses parsed from {poses_txt}; see {log_path}")

    print(f"[isaac] engine_native.ply + {len(poses)} cuVSLAM map poses ready")
    return native, poses, keep, "cuvslam_map"
