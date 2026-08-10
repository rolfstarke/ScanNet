"""RTAB-Map RGB-D SLAM on ZED X SVO2, official workflow only, one podman run.

STAGE 1 (capture, in-container): the official launch file
`rtabmap_examples/zed.launch.py` declares ONLY `camera_model` and
`use_zed_odometry` -- it cannot forward `svo_path`/`enable_ipc` and does not
expose `tf_tolerance`, so its documented equivalent node graph is used verbatim:
the exact launch file zed.launch.py includes (`zed_wrapper/zed_camera.launch.py`)
plus rgbd_sync + rtabmap with `odom` remapped to `/zed/zed_node/odom`
(= `use_zed_odometry:=true`; no rgbd_odometry node, no rtabmap_viz headless).
Documented working zed_wrapper params: svo_realtime=true, use_svo_timestamps=false
(wall-clock stamps -- SVO stamps caused a 937502 s TF gap), enable_ipc=false,
camera_model=zedx, depth NEURAL pre-warm waited out in the readiness loop.
RTAB-Map fixes vs the old wrapper: wait_for_transform=3.0, tf_tolerance=3.0
(odom TF arrives ~2.3 s late). Rtabmap/DetectionRate stays at default (1 Hz);
Mem/IncrementalMemory=true; approx_sync_max_interval=0.5 (an RTAB-Map sync param,
NOT a zed_wrapper param -- this wrapper version has none). SVO runs to the end,
then SIGINT with >= 30 s grace (up to 90 s) before SIGKILL so the db saves.

STAGE 2 (offline refinement, maintainer recipe rtabmap issue 1605):
    rtabmap-reprocess -default --Rtabmap/DetectionRate 0 -odom rtabmap.db refined.db
then detect_more_loop_closures via the official CLI rtabmap-detectMoreLoopClosures
(refined.db in place; same code as the rtabmap_slam service -- the GUI-only
databaseViewer post-processing and the ROS service need a running node). If the
binary is absent at runtime: warn and continue with refined.db.

STAGE 3 (export): rtabmap-export --cloud --mesh --texture --images_id
--poses_camera --output scene --output_dir work/rtabmap/export --max_range 6.0
--voxel 0 --decimation 1 --poses_format 11 refined.db (NO --opt 0: export the
optimized graph poses; --opt 0 exported raw graph poses and caused 4.37 m drift).

Parsing (reuses the verified normalize_export.py logic): scene_camera_poses.txt
format 11 lines "stamp x y z qx qy qz qw id" -> 4x4 camera-to-world; images are
keyed by node id (scene_rgb/<id>.jpg, scene_depth/<id>.png); keep = node ids with
both rgb and depth, sorted, 0-based (rtabmap node ids are contiguous from 1, so
keep == 0..M-1). The mesh is the textured OBJ -> engine_native.ply (binary PLY;
vertex colors baked from the texture via UV sampling, else written without
colors). Exported depth images (uint16 mm) are not needed -- Stage A provides
depth. Poses are in the ROS map frame, Z-up -> convention "rtabmap_map".

All container I/O goes to work/logs/rtabmap.log (no DEVNULL); subprocess timeout
7200 s; no caching (db/refined/export removed before every run); no swallowed
exceptions. Podman image localhost/zed-rtabmap:jazzy; heavy storage under
/data/zed-rtabmap (ZED_RTABMAP_DATA overridable); GPU args via the documented
podman_gpu.sh (rootless, host-lib mount).
"""
import glob
import os
import shutil
import subprocess

import numpy as np

PODMAN_IMAGE = "localhost/zed-rtabmap:jazzy"
GPU_SCRIPT = "/home/rolf/GIT/zed-rtabmap/scripts/podman_gpu.sh"
TIMEOUT = 7200
CONTAINER_NAME = f"spellbook-rtabmap-{os.getpid()}"  # unique per process: parallel tasks

_INSIDE_SH = r"""#!/usr/bin/env bash
# SVO2 -> rtabmap.db -> refined.db -> export, one container run.
set -eo pipefail
source /opt/ros/${ROS_DISTRO}/setup.bash
[ -f /root/ros2_ws/install/local_setup.bash ] && source /root/ros2_ws/install/local_setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export OMP_WAIT_POLICY=passive
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1

SVO_PATH="$1"
DB_PATH="$2"
REFINED_PATH="$3"
EXPORT_DIR="$4"
CAMERA_MODEL="$5"

rm -f "$DB_PATH" "$REFINED_PATH"
rm -rf "$EXPORT_DIR"
mkdir -p "$(dirname "$DB_PATH")" "$EXPORT_DIR" /tmp/rtabmap_logs "$(dirname "$DB_PATH")/logs"

cat > /tmp/zed_override.yaml <<'EOF'
---
/**:
  ros__parameters:
    general:
      grab_resolution: 'NATIVE'
    depth:
      depth_mode: 'NEURAL'
      depth_stabilization: 0
    svo:
      use_svo_timestamps: false
      svo_realtime: true
      replay_rate: 1.0
EOF

cat > /tmp/rgbd_sync.yaml <<'EOF'
/**:
  ros__parameters:
    subscribe_rgbd: true
    approx_sync: true
    approx_sync_max_interval: 0.5
    topic_queue_size: 30
    sync_queue_size: 30
    qos: 1
    qos_camera_info: 1
EOF

cat > /tmp/rtabmap.yaml <<EOF
/**:
  ros__parameters:
    frame_id: "zed_camera_link"
    map_frame_id: "map"
    odom_frame_id: "odom"
    subscribe_rgbd: true
    approx_sync: true
    approx_sync_max_interval: 0.5
    wait_imu_to_init: true
    wait_for_transform: 3.0
    tf_tolerance: 3.0
    database_path: "${DB_PATH}"
    topic_queue_size: 30
    sync_queue_size: 30
    qos_image: 1
    qos_camera_info: 1
    qos_odom: 1
    qos_imu: 1
    "Mem/IncrementalMemory": "true",
    # process EVERY frame (default 1 Hz drops ~97% of frames and leaves a sparse graph;
    # 0 = all frames, same value the maintainer's offline reprocess recipe uses)
    "Rtabmap/DetectionRate": "0"
EOF

echo "[inside] starting rgbd_sync + rtabmap (subscribers first)"
stdbuf -oL ros2 run rtabmap_sync rgbd_sync --ros-args \
  --params-file /tmp/rgbd_sync.yaml \
  -r rgb/image:=/zed/zed_node/rgb/color/rect/image \
  -r depth/image:=/zed/zed_node/depth/depth_registered \
  -r rgb/camera_info:=/zed/zed_node/rgb/color/rect/camera_info \
  > /tmp/rtabmap_logs/rgbd_sync.log 2>&1 &
SYNC_PID=$!

stdbuf -oL ros2 run rtabmap_slam rtabmap --ros-args \
  --params-file /tmp/rtabmap.yaml \
  -r odom:=/zed/zed_node/odom \
  -r imu:=/zed/zed_node/imu/data \
  -- -d \
  > /tmp/rtabmap_logs/rtabmap.log 2>&1 &
RTAB_PID=$!

sleep 3

echo "[inside] starting ZED wrapper (model=$CAMERA_MODEL svo=$SVO_PATH)"
stdbuf -oL ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:="$CAMERA_MODEL" \
  camera_name:=zed \
  svo_path:="$SVO_PATH" \
  publish_tf:=true \
  publish_map_tf:=false \
  publish_imu_tf:=true \
  enable_ipc:=false \
  ros_params_override_path:=/tmp/zed_override.yaml \
  > /tmp/rtabmap_logs/zed_wrapper.log 2>&1 &
ZED_PID=$!

echo "[inside] waiting for ZED + first RGBD frames (NEURAL pre-warm allowed)..."
READY=0
for i in $(seq 1 900); do
  if ! kill -0 "$ZED_PID" 2>/dev/null; then
    echo "[inside] ZED wrapper died early" >&2
    tail -120 /tmp/rtabmap_logs/zed_wrapper.log >&2 || true
    exit 1
  fi
  if grep -q "Optimizing model" /tmp/rtabmap_logs/zed_wrapper.log 2>/dev/null; then
    if ! grep -qE "Optimizing model: .*100|zed started" /tmp/rtabmap_logs/zed_wrapper.log 2>/dev/null; then
      sleep 1
      continue
    fi
  fi
  if grep -q "zed started" /tmp/rtabmap_logs/zed_wrapper.log 2>/dev/null; then
    if grep -qE "Update rate|Publishing map|Processed|link" /tmp/rtabmap_logs/rtabmap.log 2>/dev/null; then
      echo "[inside] data flowing after ${i}s"
      READY=1
      break
    fi
    if (( i > 30 )) && ros2 topic list 2>/dev/null | grep -q "/zed/zed_node/rgb/color/rect/image"; then
      sleep 5
      if grep -qE "Update rate|Publishing map|Processed|link" /tmp/rtabmap_logs/rtabmap.log 2>/dev/null; then
        echo "[inside] data flowing after ${i}s"
        READY=1
        break
      fi
    fi
  fi
  sleep 1
done
if [[ "$READY" != "1" ]]; then
  echo "[inside] timed out waiting for pipeline data" >&2
  tail -40 /tmp/rtabmap_logs/zed_wrapper.log >&2 || true
  tail -40 /tmp/rtabmap_logs/rtabmap.log >&2 || true
  tail -20 /tmp/rtabmap_logs/rgbd_sync.log >&2 || true
  exit 1
fi

SVO_DONE=0
while kill -0 "$ZED_PID" 2>/dev/null; do
  if ! kill -0 "$RTAB_PID" 2>/dev/null; then
    echo "[inside] rtabmap died" >&2
    tail -80 /tmp/rtabmap_logs/rtabmap.log >&2 || true
    exit 1
  fi
  if grep -q "SVO reached the end" /tmp/rtabmap_logs/zed_wrapper.log 2>/dev/null; then
    echo "[inside] SVO finished"
    sleep 5
    SVO_DONE=1
    break
  fi
  sleep 2
done
if [[ "$SVO_DONE" != "1" ]]; then
  echo "[inside] ZED wrapper stopped before SVO end" >&2
  tail -60 /tmp/rtabmap_logs/zed_wrapper.log >&2 || true
  tail -40 /tmp/rtabmap_logs/rtabmap.log >&2 || true
  exit 1
fi

shutdown() {
  echo "[inside] SIGINT rtabmap (db save), grace 90s"
  kill -INT "$RTAB_PID" 2>/dev/null || true
  for s in $(seq 1 18); do
    kill -0 "$RTAB_PID" 2>/dev/null || break
    sleep 5
  done
  if kill -0 "$RTAB_PID" 2>/dev/null; then
    echo "[inside] rtabmap still alive after 90s, SIGKILL"
    kill -KILL "$RTAB_PID" 2>/dev/null || true
  fi
  wait "$RTAB_PID" 2>/dev/null || true
  kill -INT "$SYNC_PID" "$ZED_PID" 2>/dev/null || true
  sleep 5
  kill "$SYNC_PID" "$ZED_PID" 2>/dev/null || true
  wait "$SYNC_PID" 2>/dev/null || true
  wait "$ZED_PID" 2>/dev/null || true
}
trap shutdown EXIT
shutdown
trap - EXIT

cp -f /tmp/rtabmap_logs/* "$(dirname "$DB_PATH")/logs/" 2>/dev/null || true

if [[ ! -s "$DB_PATH" ]]; then
  echo "[inside] no database written at $DB_PATH" >&2
  exit 1
fi
echo "[inside] database size: $(stat -c%s "$DB_PATH") bytes"

echo "[inside] stage 2: rtabmap-reprocess (maintainer recipe)"
rtabmap-reprocess -default --Rtabmap/DetectionRate 0 -odom "$DB_PATH" "$REFINED_PATH"
if [[ ! -s "$REFINED_PATH" ]]; then
  echo "[inside] reprocess produced no refined.db" >&2
  exit 1
fi

echo "[inside] stage 2b: detect_more_loop_closures"
if command -v rtabmap-detectMoreLoopClosures >/dev/null 2>&1; then
  rtabmap-detectMoreLoopClosures "$REFINED_PATH"
else
  echo "[inside] WARNING: rtabmap-detectMoreLoopClosures not available, continuing with refined.db"
fi

echo "[inside] stage 3: rtabmap-export"
rtabmap-export --cloud --mesh --texture --images_id --poses_camera \
  --output scene --output_dir "$EXPORT_DIR" \
  --max_range 6.0 --voxel 0 --decimation 1 --poses_format 11 \
  "$REFINED_PATH"

ls -la "$EXPORT_DIR"
echo "[inside] done"
"""


def _podman_gpu_args(gpu):
    if not os.path.exists(GPU_SCRIPT):
        raise RuntimeError(f"podman_gpu.sh not found at {GPU_SCRIPT}")
    r = subprocess.run(["bash", GPU_SCRIPT, str(gpu if gpu is not None else 0)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"podman_gpu.sh failed: {r.stderr.strip() or r.stdout.strip()}")
    return [line for line in r.stdout.splitlines() if line]


def _quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _parse_poses(path):
    """poses_format 11: 'stamp x y z qx qy qz qw id' -> [(node_id, 4x4 c2w)].
    Same parsing as zed-rtabmap/scripts/normalize_export.py (verified correct)."""
    out = []
    for line in open(path).read().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        x, y, z = map(float, parts[1:4])
        qx, qy, qz, qw = map(float, parts[4:8])
        node_id = int(float(parts[8]))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = (x, y, z)
        out.append((node_id, T))
    out.sort(key=lambda t: t[0])
    return out


def _parse_export(export_dir):
    poses_files = glob.glob(os.path.join(export_dir, "*_camera_poses.txt"))
    if not poses_files:
        raise RuntimeError(f"no *_camera_poses.txt in export dir {export_dir}")
    rgb_dir = glob.glob(os.path.join(export_dir, "*_rgb"))
    depth_dir = glob.glob(os.path.join(export_dir, "*_depth"))
    if not rgb_dir or not depth_dir:
        raise RuntimeError(f"export has no *_rgb / *_depth image dirs in {export_dir}")
    rgb_dir, depth_dir = rgb_dir[0], depth_dir[0]

    kept = []
    for node_id, T in _parse_poses(poses_files[0]):
        rgb = os.path.join(rgb_dir, f"{node_id}.jpg")
        if not os.path.exists(rgb):
            rgb = os.path.join(rgb_dir, f"{node_id}.png")
        if os.path.exists(rgb) and os.path.exists(os.path.join(depth_dir, f"{node_id}.png")):
            kept.append((node_id, T))
    if not kept:
        raise RuntimeError(f"no node with both rgb+depth in export {export_dir}")

    ids = [nid for nid, _ in kept]
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise RuntimeError(
            f"rtabmap export node ids not contiguous ({ids[0]}..{ids[-1]}, {len(ids)} nodes); "
            "cannot map node ids to original frame indices without a timestamp "
            "correspondence (stamps are wall-clock, use_svo_timestamps=false)")
    keep = [i - ids[0] for i in ids]
    poses = np.stack([T for _, T in kept]).astype(np.float64)
    return poses, keep


def _bake_vertex_colors(mesh):
    texs = [np.asarray(t) for t in mesh.textures]
    tris = np.asarray(mesh.triangles)
    if tris.size == 0 or len(texs) == 0:
        return
    uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
    mids = np.asarray(mesh.triangle_material_ids)
    n = len(mesh.vertices)
    acc = np.zeros((n, 3), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int64)
    for mat in np.unique(mids):
        sel = np.where(mids == mat)[0]
        tex = texs[int(mat)]
        h, w = tex.shape[:2]
        u = uvs[sel]
        px = np.clip(np.rint(u[:, :, 0] * (w - 1)).astype(np.int64), 0, w - 1)
        py = np.clip(np.rint((1.0 - u[:, :, 1]) * (h - 1)).astype(np.int64), 0, h - 1)
        cols = tex[py, px]
        if cols.shape[-1] == 1:
            cols = np.repeat(cols, 3, axis=-1)
        else:
            cols = cols[..., :3]
        cols = cols.astype(np.float64)
        if cols.max() > 1.5:
            cols /= 255.0
        idx = tris[sel].ravel()
        np.add.at(acc, idx, cols.reshape(-1, 3))
        np.add.at(cnt, idx, 1)
    if cnt.sum() == 0:
        return
    import open3d as o3d
    mesh.vertex_colors = o3d.utility.Vector3dVector(acc / np.maximum(cnt[:, None], 1))


def _to_native_ply(export_dir, out_path):
    import open3d as o3d
    objs = sorted(glob.glob(os.path.join(export_dir, "*_mesh.obj")))
    if not objs:
        raise RuntimeError(f"rtabmap export produced no *_mesh.obj in {export_dir}")
    mesh = o3d.io.read_triangle_mesh(objs[0])
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"empty mesh from {objs[0]}")
    if not mesh.has_vertex_colors() and mesh.has_triangle_uvs() and len(mesh.textures):
        try:
            _bake_vertex_colors(mesh)
        except Exception as e:
            print(f"[rtabmap] texture color bake failed ({e}); writing mesh without colors")
    o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False,
                               write_vertex_normals=False,
                               write_vertex_colors=mesh.has_vertex_colors(),
                               write_triangle_uvs=False)
    return out_path


def reconstruct(work, root, gpu=None):
    marker = os.path.join(work, "svo_path.txt")
    if not os.path.exists(marker):
        raise RuntimeError("run.py must write recon/svo_path.txt before engines run")
    svo = open(marker).read().strip()

    rtab_dir = os.path.join(work, "rtabmap")
    log_dir = os.path.join(work, "logs")
    os.makedirs(rtab_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    db_path = os.path.join(rtab_dir, "rtabmap.db")
    refined_path = os.path.join(rtab_dir, "refined.db")
    export_dir = os.path.join(rtab_dir, "export")
    for stale in (db_path, refined_path):
        if os.path.exists(stale):
            os.unlink(stale)
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)

    inside_sh = os.path.join(rtab_dir, "inside.sh")
    with open(inside_sh, "w") as f:
        f.write(_INSIDE_SH)
    os.chmod(inside_sh, 0o755)

    gpu_args = _podman_gpu_args(gpu)
    data_root = os.environ.get("ZED_RTABMAP_DATA", "/data/zed-rtabmap")
    resources = os.path.join(data_root, "zed-resources")
    settings = os.path.join(data_root, "zed-settings")
    for d in (resources, settings):
        os.makedirs(d, exist_ok=True)

    svo_real = os.path.realpath(svo)
    svo_dir, svo_base = os.path.split(svo_real)
    cmd = ["podman", "run", "--rm", "--name", CONTAINER_NAME] + gpu_args + [
        "--network=host", "--ipc=host",
        "-e", f"ROS_DOMAIN_ID={40 + (gpu if gpu is not None else 0)}",
        "-e", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
        "-v", f"{svo_dir}:/input:ro",
        "-v", f"{rtab_dir}:/output:rw",
        "-v", f"{resources}:/usr/local/zed/resources:rw",
        "-v", f"{settings}:/usr/local/zed/settings:rw",
        "-v", f"{inside_sh}:/inside.sh:ro",
        PODMAN_IMAGE,
        "bash", "/inside.sh",
        f"/input/{svo_base}", "/output/rtabmap.db", "/output/refined.db",
        "/output/export", "zedx",
    ]

    log_path = os.path.join(log_dir, "rtabmap.log")
    with open(log_path, "w") as lf:
        lf.write("cmd: " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            lf.write(f"timeout after {TIMEOUT}s, killing container {CONTAINER_NAME}\n")
            lf.flush()
            subprocess.run(["podman", "kill", CONTAINER_NAME], stdout=lf, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise RuntimeError(f"rtabmap container timed out after {TIMEOUT}s (log: {log_path})")
    if proc.returncode != 0:
        raise RuntimeError(f"rtabmap container failed rc={proc.returncode} (log: {log_path})")

    poses, keep = _parse_export(export_dir)
    native = _to_native_ply(export_dir, os.path.join(work, "engine_native.ply"))
    return native, poses, keep, "rtabmap_map"
