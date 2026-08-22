"""Stage A -- SVO2 -> frames/ + intrinsics. The .sens and <id>.txt are written at
finalize time (after the engine has produced its optimized poses), see finalize.py.

Format mirrors SensReader/c++/src/sensorData.h:1057-1109 (version 4, jpeg + zlib_ushort,
depth_shift 1000.0), verified against SensReader/python/SensorData.py:52-74 and 6 release
.sens files.

ZED extraction is currently BLOCKED: every ZED SDK open lands on physical GPU 0
(user-reserved, #18; managed remapping not validated). Complete frame sets remain
reusable; no new extraction or --replace is possible until a validated remapped path
exists.
"""
import os
import struct
import zlib

import numpy as np

_COLOR_JPEG = 2
_DEPTH_ZLIB = 1
_DEPTH_SHIFT = 1000.0
_WIDTH, _HEIGHT = 1920, 1200  # ZED X native
_DEPTH_MIN, _DEPTH_MAX = 0.1, 6.0  # ScanNet zParametersScanNet.txt s_sensorDepthMin/Max

ZED_BLOCK_REASON = ("ZED frame extraction disabled: the SDK's default-device path would "
                    "use user-reserved physical GPU 0 (#18); managed remapping not "
                    "validated. Reuse complete frame sets only.")


def _zed_blocked():
    raise RuntimeError(ZED_BLOCK_REASON)


def write_sens(path, camera_to_world, color_bytes, depth_bytes, color_ts, depth_ts, K,
               width, height):
    """camera_to_world: (N,4,4); color/depth_bytes: list of N bytes objects.
    width/height: the actual frame resolution of this recording."""
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
            f.write(struct.pack("<16f", *m.T.ravel()))  # row-major float32
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
        f.write(struct.pack("<Q", 0))  # num_IMU


def _depth_bytes(depth_u16):
    return zlib.compress(depth_u16.astype("<u2").tobytes(), level=6)


def read_gravity(svo):
    """Quick SVO open to read the IMU gravity vector (no frame grab, no depth compute).

    Blocked: any ZED SDK open lands on user-reserved physical GPU 0."""
    _zed_blocked()


def frames_complete(frames_dir):
    """True when a previous extraction is complete (pose_state.txt + color + intrinsics)."""
    return (os.path.isfile(os.path.join(frames_dir, "pose_state.txt"))
            and os.path.isdir(os.path.join(frames_dir, "color"))
            and len(os.listdir(os.path.join(frames_dir, "color"))) > 0
            and os.path.isfile(os.path.join(frames_dir, "intrinsic_depth.txt")))


def ensure_frames(svo, work_dir, replace=False):
    """Extract SVO frames once and reuse complete sets; `replace=True` re-extracts.
    Returns the same info dict as extract().

    Extraction requires the ZED SDK, which is blocked (user-reserved GPU 0); only
    complete existing sets can be reused, and `--replace` always fails."""
    frames = os.path.join(work_dir, "frames")
    if replace:
        _zed_blocked()
    if frames_complete(frames):
        print(f"[extract] reusing existing frames (complete)")
        grav_path = os.path.join(frames, "gravity.npy")
        if not os.path.exists(grav_path):
            _zed_blocked()  # would need a ZED SVO open to read gravity
        return {
            "K": np.loadtxt(os.path.join(frames, "intrinsic_depth.txt"))[:3, :3],
            "poses": np.load(os.path.join(frames, "camera_to_world.npy")),
            "states": open(os.path.join(frames, "pose_state.txt")).read().split(),
            "svo_frames": len(open(os.path.join(frames, "pose_state.txt")).read().split()),
            "gravity": np.load(grav_path),
        }
    _zed_blocked()


def extract(svo, work_dir):
    """Play the SVO once and write frames/. Unreachable while ZED is blocked."""
    _zed_blocked()
