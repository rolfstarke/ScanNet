"""RTAB-Map engine: assisted-hybrid baseline `_20` (#70 2026-10-01, params #41 2026-09-15).

"Explicitly not pure SLAM" (#70): the shared frame pool is replayed into rtabmap with per-frame
ack (containers/rtabmap/rtabmap_replay.py), Stage-A ZED poses as odometry, only every 90th frame
in the map (9004: 41/41 frames, span 99.7%). Export is poses only, optimized by GTSAM; no
reprocess, no loop-closure pass. The mesh is a TSDF fusion of the node frames with the shared CPU
tsdf.py values (2 cm, trunc 0.06, depth 6 m, no weight threshold) [inferred: #41 "no mesh flags",
losers named sdf.04/depth3; engines/__init__.py: shared integration lives in tsdf.py]. Open3D B1's
weight_threshold 3 needs three views per voxel: 41 sparse frames shred it (47.19 cm, comp 75.40).

Target: `_20` 20.515862 cm (21.5595/19.472223) with the old mask, recon 91.401 s.

The live zed_wrapper path is gone (master before 2026-08-30), so ZED wrapper parameters such as
#41 grab_resolution=SVGA do not apply: the baseline never opens the SVO.

Docker runs through reconstruct/containers (snap docker: hostfs mounts, explicit GPU args).
Container I/O goes to recon/logs/rtabmap.log and recon/rtabmap/logs/. Overruns are enforced by
run.py (SIGALRM, 3x scene length); the container is killed on the way out.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from .. import preset_env, runtime_cap_s
from ..containers import gpu_args, mount, user_args
from ..tsdf import PARAMS as FUSION

IMAGE = "localhost/zed-rtabmap:jazzy"
TIMEOUT = 7200
CONTAINER_NAME = f"spellbook-rtabmap-{os.getpid()}"  # unique per process: parallel tasks (#23)

GPU_POLICY = "managed"
SERIAL = False

ISSUE = 70
PROFILE = "assisted-hybrid"
DEFAULT_STRIDE = 90
SOURCE_FPS = 15  # #51 run.json "fps 15" (SVO header)
STAMP_ORIGIN = 1_700_000_000  # [inferred] synthetic stamps: node stamp -> pool index
CONVENTION = "rtabmap_map"  # [inferred] exported camera poses in the ROS z-up map frame
FUSE_THREADS = 6  # OpenMP on all cores spin-waits under load: fusion 231 s vs 8 s at 6 threads

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

_REPLAY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "containers", "rtabmap", "rtabmap_replay.py")


def selected_profile():
    return preset_env("rtabmap", "SPELLBOOK_RTABMAP_PROFILE") or PROFILE


def selected_stride():
    return int(os.environ.get("SPELLBOOK_RTABMAP_STRIDE", "").strip() or DEFAULT_STRIDE)


def provenance():
    return {"variant": f"{selected_profile()}-s{selected_stride()}",
            "config": dict(profile=selected_profile(), stride=selected_stride(), image=IMAGE,
                           map_only=True, external_odom=True, fast=True,
                           fusion=dict(FUSION)),
            "fps": SOURCE_FPS}


def provenance_after(work):
    path = os.path.join(work, "effective_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        eff = json.load(f)
    return {k: eff[k] for k in ("keep", "span", "profile", "stride") if k in eff}


def preflight():
    if selected_profile() != PROFILE:
        return f"unknown rtabmap profile {selected_profile()!r} (only {PROFILE})"
    if selected_stride() < 1:
        return f"bad SPELLBOOK_RTABMAP_STRIDE {selected_stride()}"
    r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, text=True)
    if r.returncode:
        return f"docker image {IMAGE} unavailable: {r.stderr.strip()[-200:]}"
    try:
        import open3d  # noqa: F401
    except Exception as ex:
        return f"open3d not importable with {sys.executable} (use the conda env): {ex}"
    return None


def gpu_check(gpu=None, lease_fd=None, hold_seconds=5):
    """Image start under the lease with the real GPU args, in-container NVIDIA probe."""
    if gpu is None or lease_fd is None:
        return dict(status="fail", reason="managed check requires gpu + lease_fd")
    start = time.time()
    try:
        args = gpu_args(gpu)
    except Exception as ex:
        return dict(status="fail", reason=f"gpu_args: {ex}")
    container = f"spellbook-gpu-check-rtabmap-{os.getpid()}"
    cmd = ["docker", "run", "--rm", "--name", container] + user_args() + args + [
        IMAGE, "bash", "-c",
        "echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; "
        "nvidia-smi -L || echo no-nvidia-smi; sleep %d" % hold_seconds,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=hold_seconds + 120, pass_fds=(lease_fd,))
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", container], capture_output=True)
        return dict(status="fail", reason="rtabmap gpu_check timed out")
    if r.returncode != 0:
        return dict(status="fail", reason=f"rtabmap gpu_check rc={r.returncode}: "
                                          f"{(r.stderr or r.stdout)[-400:]}")
    if f"CUDA_VISIBLE_DEVICES={gpu}" not in (r.stdout or ""):
        return dict(status="fail",
                    reason=f"assigned env missing in container output: {r.stdout[-300:]}")
    return dict(status="pass", policy="managed", physical_gpu=gpu, visible_gpu=gpu,
                lease_fd=lease_fd, runtime=f"docker {IMAGE}",
                seconds=round(time.time() - start, 1), reason=None)


_RGBD_SYNC_YAML = """\
/**:
  ros__parameters:
    approx_sync: false
    approx_sync_max_interval: 2.0
    topic_queue_size: 30
    sync_queue_size: 30
"""

_RTABMAP_YAML = """\
/**:
  ros__parameters:
    frame_id: "camera_link"
    map_frame_id: "map"
    subscribe_rgbd: true
    approx_sync: true
    approx_sync_max_interval: 2.0
    wait_for_transform: 0.5
    tf_tolerance: 0.5
    database_path: "/output/rtabmap.db"
    topic_queue_size: 30
    sync_queue_size: 30
    "Mem/IncrementalMemory": "true"
    "Rtabmap/DetectionRate": "0"
    "RGBD/LinearUpdate": "0"
    "RGBD/AngularUpdate": "0"
    "Kp/MaxFeatures": "400"
    "Optimizer/Strategy": "2"
    "Rtabmap/ImagesAlreadyRectified": "true"
    "Rtabmap/ImageBufferSize": "0"
"""

_INSIDE_SH = r"""#!/usr/bin/env bash
# Pool replay -> rtabmap.db -> GTSAM camera poses, one container run.
set -eo pipefail
source /opt/ros/${ROS_DISTRO}/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export RCUTILS_LOGGING_USE_STDOUT=1
STRIDE="$1"; FPS="$2"; STAMP_ORIGIN="$3"
LOGS=/output/logs
mkdir -p "$LOGS" /tmp/ros "$HOME/.ros"

# Real binaries, not `ros2 run`: a backgrounded python launcher would ignore SIGINT.
/opt/ros/jazzy/lib/rtabmap_sync/rgbd_sync --ros-args -r __node:=rgbd_sync_map \
  --params-file /output/rgbd_sync.yaml \
  -r rgb/image:=/map_rgbd/rgb/image -r depth/image:=/map_rgbd/depth/image \
  -r rgb/camera_info:=/map_rgbd/rgb/camera_info -r rgbd_image:=/map_rgbd/rgbd_image \
  > "$LOGS/rgbd_sync.log" 2>&1 &
SYNC_PID=$!
/opt/ros/jazzy/lib/rtabmap_slam/rtabmap --ros-args --params-file /output/rtabmap.yaml \
  -r rgbd_image:=/map_rgbd/rgbd_image -r odom:=/odom -- -d \
  > "$LOGS/rtabmap.log" 2>&1 &
RTAB_PID=$!

stop_nodes() {
  kill -INT "$RTAB_PID" 2>/dev/null || true
  for _ in $(seq 1 180); do
    kill -0 "$RTAB_PID" 2>/dev/null || break
    sleep 0.5
  done
  kill -KILL "$RTAB_PID" 2>/dev/null || true
  wait "$RTAB_PID" 2>/dev/null || true
  kill -INT "$SYNC_PID" 2>/dev/null || true
  wait "$SYNC_PID" 2>/dev/null || true
}

if ! python3 /rtabmap_replay.py --frames /frames --stride "$STRIDE" --fps "$FPS" \
      --stamp-origin "$STAMP_ORIGIN" --result /output/replay.json \
      --watch-pids "$SYNC_PID" "$RTAB_PID"; then
  stop_nodes
  tail -40 "$LOGS/rtabmap.log" "$LOGS/rgbd_sync.log" >&2 || true
  exit 1
fi
stop_nodes
[ -s /output/rtabmap.db ] || { echo "[inside] no database written" >&2; exit 1; }

echo "[inside] export"
rtabmap-export --poses_camera --poses_format 11 --opt 2 --voxel 0.01 \
  --output scene --output_dir /output/export /output/rtabmap.db
echo "[inside] done"
"""


def _quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _parse_poses(path):
    """poses_format 11: 'stamp x y z qx qy qz qw id' -> [(stamp, 4x4 c2w)] in node order."""
    out = []
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 9 or line.startswith("#"):
            continue
        stamp, x, y, z, qx, qy, qz, qw = map(float, parts[:8])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = (x, y, z)
        out.append((int(float(parts[8])), stamp, T))
    out.sort(key=lambda t: t[0])
    return [(stamp, T) for _, stamp, T in out]


def _pool_index(stamp, origin=STAMP_ORIGIN, fps=SOURCE_FPS):
    return int(round((stamp - origin) * fps))


def _parse_export(export_dir, expected):
    """Exported poses -> (poses, keep); every replayed frame must be a graph node (41/41)."""
    files = glob.glob(os.path.join(export_dir, "*_camera_poses.txt"))
    if not files:
        raise RuntimeError(f"no *_camera_poses.txt in export dir {export_dir}")
    parsed = _parse_poses(files[0])
    keep = [_pool_index(stamp) for stamp, _ in parsed]
    if keep != list(expected):
        raise RuntimeError(f"export nodes do not match the replay: {len(keep)} poses vs "
                           f"{len(expected)} frames, first {keep[:5]} vs {list(expected)[:5]}")
    return np.stack([T for _, T in parsed]).astype(np.float64), keep


def _run_container(cmd, name, log_path, lease_fd):
    with open(log_path, "w") as lf:
        lf.write("cmd: " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, pass_fds=(lease_fd,))
        try:
            proc.wait(timeout=TIMEOUT)
        except BaseException as ex:  # includes the run.py cap (a BaseException)
            lf.write(f"{type(ex).__name__}: killing container {name}\n")
            lf.flush()
            subprocess.run(["docker", "kill", name], stdout=lf, stderr=subprocess.STDOUT)
            proc.kill()
            proc.wait()
            raise
    if proc.returncode != 0:
        raise RuntimeError(f"rtabmap container failed rc={proc.returncode} (log: {log_path})")


def reconstruct(work, root, gpu=None, lease_fd=None):
    if gpu is None or lease_fd is None:
        raise RuntimeError("rtabmap is GPU_POLICY managed: gpu + lease_fd required "
                           "(run.py acquires the settings-pool lease)")
    import cv2  # noqa: F401  cv2 fails to load after pymeshlab (glib clash), finalize needs it

    t0 = time.monotonic()
    stride = selected_stride()
    frames = os.path.realpath(os.path.join(work, "frames"))
    states = open(os.path.join(frames, "pose_state.txt")).read().split()
    expected = list(range(0, len(states), stride))
    bad = [i for i in expected if states[i] != "OK"]
    if bad:  # [inferred] external odometry needs a valid Stage-A pose on every replayed frame
        raise RuntimeError(f"replayed pool frames without OK pose: {bad[:10]}")

    rtab_dir = os.path.join(work, "rtabmap")
    log_dir = os.path.join(work, "logs")
    shutil.rmtree(rtab_dir, ignore_errors=True)
    os.makedirs(rtab_dir)
    os.makedirs(log_dir, exist_ok=True)
    for name, text in (("inside.sh", _INSIDE_SH), ("rgbd_sync.yaml", _RGBD_SYNC_YAML),
                       ("rtabmap.yaml", _RTABMAP_YAML)):
        with open(os.path.join(rtab_dir, name), "w") as f:
            f.write(text)

    cmd = ["docker", "run", "--rm", "--name", CONTAINER_NAME] + user_args() + gpu_args(gpu) + [
        "--network=host", "--ipc=host",
        "-e", f"ROS_DOMAIN_ID={40 + gpu}",
        "-e", "HOME=/tmp", "-e", "ROS_LOG_DIR=/tmp/ros",
    ] + mount(frames, "/frames", read_only=True) + mount(rtab_dir, "/output") \
      + mount(_REPLAY, "/rtabmap_replay.py", read_only=True) + [
        IMAGE, "bash", "/output/inside.sh", str(stride), str(SOURCE_FPS), str(STAMP_ORIGIN),
    ]
    t_slam = time.monotonic()
    _run_container(cmd, CONTAINER_NAME, os.path.join(log_dir, "rtabmap.log"), lease_fd)
    slam_s = time.monotonic() - t_slam

    with open(os.path.join(rtab_dir, "replay.json")) as f:
        replay = json.load(f)
    poses, keep = _parse_export(os.path.join(rtab_dir, "export"), expected)
    span = replay["span"]
    print(f"[rtabmap] {len(keep)}/{len(expected)} frames, span {span:.1%}", flush=True)

    eff = dict(sid=os.path.basename(os.path.normpath(root)), issue=ISSUE,
               profile=selected_profile(), stride=stride, keep=f"{len(keep)}/{len(expected)}",
               span=span, convention=CONVENTION, fps=SOURCE_FPS, fusion=dict(FUSION))
    with open(os.path.join(work, "effective_config.json"), "w") as f:
        json.dump(eff, f, indent=2, sort_keys=True)
        f.write("\n")
    np.save(os.path.join(work, "frame_indices.npy"), np.asarray(keep))
    np.save(os.path.join(rtab_dir, "fusion_poses.npy"), poses)

    t_fuse = time.monotonic()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(FUSE_THREADS)  # read when open3d loads, hence the child
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _REPO_ROOT + (os.pathsep + pp if pp else "")
    r = subprocess.run([sys.executable, "-m", "spellbook.reconstruct.engines.rtabmap",
                        "--child", "--work", work], env=env, cwd=_REPO_ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"rtabmap TSDF child failed rc={r.returncode}")
    native = os.path.join(work, "engine_native.ply")
    fuse_s = time.monotonic() - t_fuse

    timings = dict(slam_export=round(slam_s, 3), fuse=round(fuse_s, 3),
                   reconstruct=round(time.monotonic() - t0, 3))
    with open(os.path.join(work, "stage_timings.json"), "w") as f:
        json.dump(timings, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[rtabmap] timings {json.dumps(timings, sort_keys=True)}", flush=True)
    return native, poses, keep, CONVENTION


def _child_main(argv):
    """CPU TSDF fusion of the node frames at the exported poses, thread count set by the parent."""
    from ..tsdf import integrate_tsdf
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--work", required=True)
    args = ap.parse_args(argv)
    keep = [int(i) for i in np.load(os.path.join(args.work, "frame_indices.npy"))]
    poses = np.load(os.path.join(args.work, "rtabmap", "fusion_poses.npy"))
    integrate_tsdf(args.work, keep, poses, min_frames=1, conjugate=False)


if __name__ == "__main__":
    _child_main(sys.argv[1:])
