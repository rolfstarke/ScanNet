"""Stage A -- SVO2 -> frames/ + intrinsics.

Extraction runs only from the main checkout under a managed GPU lease (pool 1-4).
The child process sets CUDA_VISIBLE_DEVICES before importing pyzed and never sets
sdk_gpu_id. Scene9004's existing complete frame set is promoted, not re-extracted.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np

from benchmark import load_settings
from utils.gpu import gpu_lease
from utils.scan_lock import exclusive_lock, frames_lock_path

_COLOR_JPEG = 2
_DEPTH_ZLIB = 1
_DEPTH_SHIFT = 1000.0
_DEPTH_MIN, _DEPTH_MAX = 0.1, 6.0
FRAMES_ROOT = "/data/scannet/derived/reconstruction/frames"
_EXTRACT_CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_extract_run.py")


def write_sens(path, camera_to_world, color_bytes, depth_bytes, color_ts, depth_ts, K,
               width, height):
    n = len(camera_to_world)
    name = b"ZED X"

    def _m4(m):
        if m.shape == (4, 4):
            return m
        out = np.eye(4)
        out[:3, :3] = m
        return out

    with open(path, "wb") as f:
        f.write(struct.pack("<I", 4))
        f.write(struct.pack("<Q", len(name)) + name)
        for m in (_m4(K), np.eye(4), _m4(K), np.eye(4)):
            f.write(struct.pack("<16f", *m.T.ravel()))
        f.write(struct.pack("<ii", _COLOR_JPEG, _DEPTH_ZLIB))
        f.write(struct.pack("<IIII", width, height, width, height))
        f.write(struct.pack("<f", _DEPTH_SHIFT))
        f.write(struct.pack("<Q", n))
        for ctw, ct, dt, cb, db in zip(camera_to_world, color_ts, depth_ts, color_bytes, depth_bytes):
            f.write(struct.pack("<16f", *ctw.T.ravel()))
            f.write(struct.pack("<QQ", int(ct), int(dt)))
            f.write(struct.pack("<QQ", len(cb), len(db)))
            f.write(cb)
            f.write(db)
        f.write(struct.pack("<Q", 0))


def frames_complete(frames_dir):
    return (os.path.isfile(os.path.join(frames_dir, "pose_state.txt"))
            and os.path.isdir(os.path.join(frames_dir, "color"))
            and len(os.listdir(os.path.join(frames_dir, "color"))) > 0
            and os.path.isfile(os.path.join(frames_dir, "intrinsic_depth.txt"))
            and os.path.isfile(os.path.join(frames_dir, "gravity.npy"))
            and os.path.isfile(os.path.join(frames_dir, "camera_to_world.npy")))


def frames_manifest(frames_dir):
    import hashlib

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    color_n = len(os.listdir(os.path.join(frames_dir, "color")))
    depth_n = len(os.listdir(os.path.join(frames_dir, "depth")))
    pose_n = len(os.listdir(os.path.join(frames_dir, "pose")))
    states = open(os.path.join(frames_dir, "pose_state.txt")).read().split()
    return {
        "color_n": color_n,
        "depth_n": depth_n,
        "pose_n": pose_n,
        "states_n": len(states),
        "ok_n": sum(1 for s in states if s == "OK"),
        "camera_to_world_sha256": sha(os.path.join(frames_dir, "camera_to_world.npy")),
        "pose_state_sha256": sha(os.path.join(frames_dir, "pose_state.txt")),
        "intrinsic_depth_sha256": sha(os.path.join(frames_dir, "intrinsic_depth.txt")),
        "gravity_sha256": sha(os.path.join(frames_dir, "gravity.npy")),
    }


def write_manifest(frames_dir, path=None):
    path = path or os.path.join(os.path.dirname(frames_dir), "frames_manifest.json")
    man = frames_manifest(frames_dir)
    with open(path, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
        f.write("\n")
    return man


def load_info(frames_dir):
    return {
        "K": np.loadtxt(os.path.join(frames_dir, "intrinsic_depth.txt"))[:3, :3],
        "poses": np.load(os.path.join(frames_dir, "camera_to_world.npy")),
        "states": open(os.path.join(frames_dir, "pose_state.txt")).read().split(),
        "svo_frames": len(open(os.path.join(frames_dir, "pose_state.txt")).read().split()),
        "gravity": np.load(os.path.join(frames_dir, "gravity.npy")),
    }


def extract(svo, work_dir):
    """Play the SVO once; write work_dir/frames/. Called only inside the leased child."""
    import cv2
    import pyzed.sl as sl

    frames = os.path.join(work_dir, "frames")
    for d in ("color", "depth", "pose"):
        os.makedirs(os.path.join(frames, d), exist_ok=True)

    init = sl.InitParameters()
    init.set_from_svo_file(svo)
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init.depth_minimum_distance = _DEPTH_MIN
    init.depth_maximum_distance = _DEPTH_MAX
    # Never set sdk_gpu_id; device comes from CUDA_VISIBLE_DEVICES only.

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"failed to open {svo}")
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    K = np.array([[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1.0]])
    K4 = np.eye(4)
    K4[:3, :3] = K
    for key in ("intrinsic_color", "intrinsic_depth"):
        np.savetxt(os.path.join(frames, f"{key}.txt"), K4, fmt="%.6f")
    for key in ("extrinsic_color", "extrinsic_depth"):
        np.savetxt(os.path.join(frames, f"{key}.txt"), np.eye(4), fmt="%.6f")

    runtime = sl.RuntimeParameters()
    runtime.enable_depth = True
    image, depth, pose = sl.Mat(), sl.Mat(), sl.Pose()
    total = zed.get_svo_number_of_frames()
    ctw, states = [], []
    gravity = None
    i = 0
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"grab failed at frame {i}: {err}")
        if i == total - 1:
            break
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
        d = depth.get_data()
        dmm = np.where(np.isfinite(d) & (d > 0), np.clip(d * 1000.0, 0, 65535), 0).astype(np.uint16)
        cv2.imwrite(os.path.join(frames, "color", f"{i}.jpg"), bgr)
        cv2.imwrite(os.path.join(frames, "depth", f"{i}.png"), dmm)
        np.savetxt(os.path.join(frames, "pose", f"{i}.txt"), pose.pose_data(sl.Transform()).m, fmt="%.6f")
        ctw.append(pose.pose_data(sl.Transform()).m)
        states.append(state.name if isinstance(state, sl.POSITIONAL_TRACKING_STATE) else str(state))
        if gravity is None:
            try:
                sd = sl.SensorsData()
                if zed.get_sensors_data(sd) == sl.ERROR_CODE.SUCCESS:
                    gravity = np.array(sd.get_gravity_vector()).astype(float)
            except Exception:
                gravity = np.zeros(3)
        i += 1

    zed.disable_positional_tracking()
    zed.close()
    if gravity is None:
        gravity = np.zeros(3)

    with open(os.path.join(frames, "pose_state.txt"), "w") as f:
        f.write("\n".join(states))
    np.save(os.path.join(frames, "camera_to_world.npy"), np.stack(ctw))
    np.save(os.path.join(frames, "gravity.npy"), gravity)
    print(f"[extract] {total} svo frames, {len(ctw)} exported")
    return dict(poses=np.stack(ctw), states=states, K=K, gravity=gravity, svo_frames=total)


def ensure_frames(svo, work_dir, replace=False):
    """Reuse complete frames only. New extraction goes through extract_scenes()."""
    frames = os.path.join(work_dir, "frames")
    if replace:
        raise RuntimeError("direct --replace extraction is disabled; use "
                           "`python spellbook/main.py --extract-frames --scene ...`")
    if frames_complete(frames):
        print("[extract] reusing existing frames (complete)")
        return load_info(frames)
    raise RuntimeError(f"incomplete frames at {frames}; run --extract-frames from main checkout")


def pool_frames_dir(scene_num):
    return os.path.join(FRAMES_ROOT, f"scene{int(scene_num):04d}", "frames")


def promote_scene9004_frames(src=None):
    """Move the irreplaceable scene9004 frame set into the shared pool."""
    src = src or "/data/scannet/scans/scene9004_04/recon/frames"
    dst_parent = os.path.join(FRAMES_ROOT, "scene9004")
    dst = os.path.join(dst_parent, "frames")
    with exclusive_lock(frames_lock_path(9004)):
        if frames_complete(dst):
            man = os.path.join(dst_parent, "frames_manifest.json")
            if not os.path.isfile(man):
                write_manifest(dst)
            print(f"[frames] pool already complete: {dst}")
            return dst
        if not os.path.isdir(src):
            raise FileNotFoundError(src)
        if not frames_complete(src):
            raise RuntimeError(f"source frames incomplete: {src}")
        if os.path.exists(dst_parent) and not os.path.isdir(dst_parent):
            raise RuntimeError(f"unexpected path: {dst_parent}")
        os.makedirs(FRAMES_ROOT, exist_ok=True)
        os.makedirs(dst_parent, exist_ok=True)
        if os.path.exists(dst):
            raise RuntimeError(f"partial destination exists: {dst}")
        print(f"[frames] promoting {src} -> {dst}")
        shutil.move(src, dst)
        # temporary symlink so old scan stays operable until reset
        try:
            os.symlink(dst, src)
        except OSError:
            pass
        if not frames_complete(dst):
            raise RuntimeError("promotion produced incomplete pool")
        write_manifest(dst)
        print(f"[frames] promoted scene9004 ({frames_manifest(dst)['color_n']} frames)")
        return dst


def extract_scenes(scene_nums, replace=False):
    """Main-checkout multi-GPU frame extraction into the shared pool.

    One SVO per leased GPU. Existing complete pools are reused unless replace=True.
    Scene9004 is never re-extracted: promote the existing set instead.
    """
    settings = load_settings()
    results = {}
    for scene in scene_nums:
        scene = int(scene)
        if scene == 9004:
            results[scene] = promote_scene9004_frames()
            continue
        svo = f"/data/scannet/custom/raw/scene{scene:04d}.svo2"
        if not os.path.exists(svo):
            raise FileNotFoundError(svo)
        work = os.path.join(FRAMES_ROOT, f"scene{scene:04d}")
        frames = os.path.join(work, "frames")
        with exclusive_lock(frames_lock_path(scene)):
            man_path = os.path.join(work, "frames_manifest.json")
            if frames_complete(frames) and not replace:
                if os.path.isfile(man_path):
                    print(f"[frames] reusing scene{scene:04d}")
                    results[scene] = frames
                    continue
                write_manifest(frames)
                results[scene] = frames
                continue
            if replace and os.path.isfile(man_path):
                # pinned pools reject replace
                raise RuntimeError(f"scene{scene:04d} frame pool is pinned; refuse --replace")

            os.makedirs(work, exist_ok=True)
            tmp = tempfile.mkdtemp(prefix=f".extract_{scene}_", dir=FRAMES_ROOT)
            try:
                with gpu_lease(settings["gpu_pool"], settings["scannet_root"]) as lease:
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(lease.index)
                    env["PYTHONUNBUFFERED"] = "1"
                    cmd = [sys.executable, _EXTRACT_CHILD,
                           "--svo", svo, "--work-dir", tmp,
                           "--lease-fd", str(lease.fileno())]
                    print(f"[frames] extracting scene{scene:04d} on GPU {lease.index}")
                    r = subprocess.run(cmd, env=env, pass_fds=(lease.fileno(),))
                    if r.returncode != 0:
                        raise RuntimeError(f"extract child failed rc={r.returncode} scene={scene}")
                tmp_frames = os.path.join(tmp, "frames")
                if not frames_complete(tmp_frames):
                    raise RuntimeError("extract child produced incomplete frames")
                # publish atomically
                if os.path.isdir(frames):
                    shutil.rmtree(frames)
                elif os.path.islink(frames):
                    os.remove(frames)
                os.replace(tmp_frames, frames)
                write_manifest(frames)
                results[scene] = frames
                print(f"[frames] published scene{scene:04d}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description="multi-GPU SVO frame extraction (main checkout)")
    ap.add_argument("--scene", nargs="+", type=int, required=True)
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()
    extract_scenes(args.scene, replace=args.replace)


if __name__ == "__main__":
    main()
