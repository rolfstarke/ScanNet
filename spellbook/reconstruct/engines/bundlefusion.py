"""BundleFusion (FangGet/Ubuntu fork) in Docker -- the ScanNet reference engine.

Runs the fork's headless binary on the Stage-A frames via the PrimeSense dir
emulation (color/ + depth/ subdirs). This is the ONLY sensor path the fork
binary instantiates: its getRGBDSensor() (src/BundleFusion.cpp:480-495) throws
MLIB_EXCEPTION("unkown sensor id") for any s_sensorIdx != 1, so the
SensorDataReader (.sens) path is dead code in this build and CANNOT be fed.

Configs are ScanNet's authoritative files, patched minimally:
  s_SDFVoxelSize 0.010f -> 0.004f   (the plan's single fidelity deviation)
  s_sensorIdx 8 -> 1                (fork requires PrimeSense; .sens unreachable)
  s_cameraIntrinsicFx/Fy/Cx/Cy appended (ScanNet's file lacks them; the fork's
  PrimeSenseSensor reads them and would default them to 0.0, making the
  projection singular). Values are the 640x480-rescaled intrinsics, matching
  the fork's built-in stretch-resize of our 1920x1200 input (depth NEAREST,
  color LINEAR -- we pre-resize identically to shrink the docker payload).
Everything else byte-for-byte from /home/rolf/GIT/ScanNet/Server/tools/recons/.

Poses: <out>/pose/<i>.txt, camera-to-world in BundleFusion's world frame
(first camera = identity), row-major 4x4, written by saveOptimizedPoses BEFORE
the reintegration drain (fork bugfix). The pose filename IS the original frame
index (TrajectoryManager indexed by getCurrFrameNumber; verified
src/BundleFusion.cpp:909-963). Frames without a finite pose are DROPPED (never
identity-backfilled); `keep` = the original frame indices with finite poses.
If the headless example skips any frame (read error), all subsequent indices
shift -- we detect this from the log and fail loudly instead of mislabeling.

Mesh: <out>/mesh.ply, binary_little_endian PLY (MLIB generated), xyz + rgba
vertex colors; copied verbatim to engine_native.ply.

Docker plumbing: snap docker can only bind-mount from $HOME, so input and
output are staged under ~/.bf_tmp and results are copied back to work/.
Timeout 5400 s; on expiry the container is docker-killed by name.
"""
import os
import re
import shutil
import struct
import subprocess
import tempfile

import numpy as np

_IMAGE = "bundlefusion:latest"
_SCANNET_RECONS = "/home/rolf/GIT/ScanNet/Server/tools/recons"
_APP_CFG_SRC = os.path.join(_SCANNET_RECONS, "zParametersScanNet.txt")
_BUNDLE_CFG_SRC = os.path.join(_SCANNET_RECONS, "zParametersBundlingScanNet.txt")
_OPT_ROUNDS = 10
_TIMEOUT = 5400
_HOME_TMP = os.path.join(os.path.expanduser("~"), ".bf_tmp")
_BF_WIDTH, _BF_HEIGHT = 640, 480


def _png_size(path):
    """Return (width, height) of a PNG by parsing the IHDR chunk (no deps)."""
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"not a png: {path}")
        ihdr = f.read(16)  # length(4) + "IHDR"(4) + width(4) + height(4)
    return struct.unpack(">II", ihdr[8:16])


def _patch_app_config(text, fx, fy, cx, cy):
    patched = re.sub(r"s_SDFVoxelSize\s*=\s*0\.010f;", "s_SDFVoxelSize = 0.004f;", text)
    if "s_SDFVoxelSize = 0.004f" not in patched:
        raise RuntimeError("s_SDFVoxelSize patch did not apply")
    patched = re.sub(r"s_sensorIdx\s*=\s*8;", "s_sensorIdx = 1;", patched)
    if "s_sensorIdx = 1;" not in patched:
        raise RuntimeError("s_sensorIdx patch did not apply")
    for key, value in (("s_sensorDepthMax", "6.0f"), ("s_renderDepthMax", "6.0f")):
        if not re.search(key + r"\s*=\s*6\.0f;", patched):
            raise RuntimeError(f"{key} is not 6.0f in the patched config")
    patched += (
        f"\ns_cameraIntrinsicFx = {fx:.6f};\n"
        f"s_cameraIntrinsicFy = {fy:.6f};\n"
        f"s_cameraIntrinsicCx = {cx:.6f};\n"
        f"s_cameraIntrinsicCy = {cy:.6f};\n"
    )
    return patched


def _stage_input(bf_in, frames):
    """Copy frames into bf_in/{color,depth}, pre-resized to 640x480 exactly
    like the fork's PrimeSenseSensor::readDepthAndColor does (NEAREST depth,
    LINEAR color), so the payload is small and the run is deterministic."""
    import cv2

    os.makedirs(os.path.join(bf_in, "color"), exist_ok=True)
    os.makedirs(os.path.join(bf_in, "depth"), exist_ok=True)
    names = sorted(
        int(f[:-4]) for f in os.listdir(os.path.join(frames, "color"))
        if f.endswith(".jpg")
    )
    if not names:
        raise RuntimeError(f"no frames in {os.path.join(frames, 'color')}")
    for i in names:
        rgb = cv2.imread(os.path.join(frames, "color", f"{i}.jpg"))
        dep = cv2.imread(os.path.join(frames, "depth", f"{i}.png"),
                         cv2.IMREAD_UNCHANGED)
        if rgb is None or dep is None:
            raise RuntimeError(f"unreadable frame {i}")
        if rgb.shape[1] != _BF_WIDTH or rgb.shape[0] != _BF_HEIGHT:
            rgb = cv2.resize(rgb, (_BF_WIDTH, _BF_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)
            dep = cv2.resize(dep, (_BF_WIDTH, _BF_HEIGHT),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(bf_in, "color", f"{i}.jpg"), rgb)
        cv2.imwrite(os.path.join(bf_in, "depth", f"{i}.png"), dep)
    return names


def _run_docker(bf_in, bf_out, gpu):
    container = f"bundlefusion_{os.getpid()}"
    cmd = [
        "docker", "run", "--rm", "--name", container, "--privileged",
        "-e", f"CUDA_VISIBLE_DEVICES={gpu if gpu is not None else 0}",
        "-v", f"{bf_in}:/input:ro",
        "-v", f"{bf_out}:/output",
        _IMAGE,
        "/app/build/bundle_fusion_headless",
        "/input/zParametersScanNet.txt",
        "/input/zParametersBundlingScanNet.txt",
        "/input", "/output",
        str(_OPT_ROUNDS),
    ]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", container], capture_output=True)
        raise RuntimeError(
            f"BundleFusion docker timed out after {_TIMEOUT}s; container "
            f"{container} killed"
        )


def _parse_poses(pose_dir, n_frames, log):
    if not os.path.isdir(pose_dir):
        raise RuntimeError(f"no pose/ dir in docker output; see {log}")
    keep = sorted(
        int(f[:-4]) for f in os.listdir(pose_dir) if f.endswith(".txt")
    )
    if any(i >= n_frames for i in keep):
        raise RuntimeError("pose index exceeds frame count; trajectory mismatch")
    if not keep:
        raise RuntimeError("no finite poses in docker output; reconstruction failed")
    poses = np.stack([
        np.loadtxt(os.path.join(pose_dir, f"{i}.txt")).reshape(4, 4)
        for i in keep
    ])
    if not np.all(np.isfinite(poses)):
        raise RuntimeError("non-finite pose found in docker output")
    return poses, keep


def reconstruct(work, root, gpu=None):
    frames = os.path.join(work, "frames")
    K = np.loadtxt(os.path.join(frames, "intrinsic_depth.txt"))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dw, dh = _png_size(os.path.join(
        frames, "depth", f"{0}.png"))
    sx, sy = _BF_WIDTH / dw, _BF_HEIGHT / dh

    os.makedirs(_HOME_TMP, exist_ok=True)
    bf_in = tempfile.mkdtemp(prefix="bf_in_", dir=_HOME_TMP)
    bf_out = tempfile.mkdtemp(prefix="bf_out_", dir=_HOME_TMP)
    try:
        names = _stage_input(bf_in, frames)

        with open(_APP_CFG_SRC) as f:
            app_cfg = _patch_app_config(f.read(), fx * sx, fy * sy, cx * sx, cy * sy)
        with open(os.path.join(bf_in, "zParametersScanNet.txt"), "w") as f:
            f.write(app_cfg)
        shutil.copy2(_BUNDLE_CFG_SRC, os.path.join(bf_in, "zParametersBundlingScanNet.txt"))

        print(f"[bundlefusion] {len(names)} frames, K@640x480 = "
              f"{fx * sx:.3f}/{fy * sy:.3f} f, {cx * sx:.1f}/{cy * sy:.1f} c, gpu={gpu or 0}")
        res = _run_docker(bf_in, bf_out, gpu)
        log_dir = os.path.join(work, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "bundlefusion.log")
        with open(log_path, "w") as f:
            f.write(res.stdout or "")
            f.write(res.stderr or "")
        if res.returncode != 0:
            raise RuntimeError(
                f"BundleFusion docker failed (exit {res.returncode}); see {log_path}")
        log = (res.stdout or "") + (res.stderr or "")
        if "[warn] skipping frame" in log:
            raise RuntimeError(
                "BundleFusion skipped frames (read error); pose indices would "
                f"shift; see {log_path}")
        m = re.search(r"\[info\] processed (\d+)/(\d+) frames", log)
        if not m or int(m.group(1)) != int(m.group(2)):
            raise RuntimeError(
                f"frame count mismatch in run log ({log_path}); cannot map poses")

        mesh_src = os.path.join(bf_out, "mesh.ply")
        if not os.path.exists(mesh_src):
            raise RuntimeError(f"no mesh.ply in docker output; see {log_path}")
        native = os.path.join(work, "engine_native.ply")
        shutil.copy2(mesh_src, native)

        poses, keep = _parse_poses(os.path.join(bf_out, "pose"), len(names), log_path)
        dropped = len(names) - len(keep)
        print(f"[bundlefusion] mesh={native} poses={len(keep)}/{len(names)} "
              f"dropped={dropped} conv=bundlefusion_world")
        return native, poses, keep, "bundlefusion_world"
    finally:
        shutil.rmtree(bf_in, ignore_errors=True)
        shutil.rmtree(bf_out, ignore_errors=True)
