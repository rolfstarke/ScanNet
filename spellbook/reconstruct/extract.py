"""Stage A -- SVO2 -> frames/ + intrinsics. The .sens and <id>.txt are written at
finalize time (after the engine has produced its optimized poses), see finalize.py.

Format mirrors SensReader/c++/src/sensorData.h:1057-1109 (version 4, jpeg + zlib_ushort,
depth_shift 1000.0), verified against SensReader/python/SensorData.py:52-74 and 6 release
.sens files.
"""
import os
import shutil
import struct
import zlib

import numpy as np

_COLOR_JPEG = 2
_DEPTH_ZLIB = 1
_DEPTH_SHIFT = 1000.0
_WIDTH, _HEIGHT = 1920, 1200  # ZED X native
_DEPTH_MIN, _DEPTH_MAX = 0.1, 6.0  # ScanNet zParametersScanNet.txt s_sensorDepthMin/Max


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


def read_gravity(svo, gpu=None):
    """Quick SVO open to read the IMU gravity vector (no frame grab, no depth compute)."""
    import pyzed.sl as sl
    init = sl.InitParameters()
    init.set_from_svo_file(svo)
    init.svo_real_time_mode = False
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # avoid the NEURAL pre-warm
    # SDK 5.4 bug (measured): sdk_gpu_id != 0 makes positional tracking return constant
    # poses (camera never moves, state OK). Tracking/extraction must stay on the default GPU.
    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        return np.zeros(3)
    try:
        sd = sl.SensorsData()
        if zed.get_sensors_data(sd) == sl.ERROR_CODE.SUCCESS:
            return np.array(sd.get_gravity_vector()).astype(float)
        return np.zeros(3)
    finally:
        zed.close()


def frames_complete(frames_dir):
    """True when a previous extraction is complete (pose_state.txt + color + intrinsics)."""
    return (os.path.isfile(os.path.join(frames_dir, "pose_state.txt"))
            and os.path.isdir(os.path.join(frames_dir, "color"))
            and len(os.listdir(os.path.join(frames_dir, "color"))) > 0
            and os.path.isfile(os.path.join(frames_dir, "intrinsic_depth.txt")))


def ensure_frames(svo, work_dir, gpu=None, replace=False):
    """Extract SVO frames once and reuse complete sets; `replace=True` re-extracts.
    Returns the same info dict as extract()."""
    frames = os.path.join(work_dir, "frames")
    if replace and os.path.islink(frames):
        os.remove(frames)
    elif replace and os.path.isdir(frames):
        shutil.rmtree(frames)
    if frames_complete(frames):
        print(f"[extract] reusing existing frames (complete)")
        grav_path = os.path.join(frames, "gravity.npy")
        return {
            "K": np.loadtxt(os.path.join(frames, "intrinsic_depth.txt"))[:3, :3],
            "poses": np.load(os.path.join(frames, "camera_to_world.npy")),
            "states": open(os.path.join(frames, "pose_state.txt")).read().split(),
            "svo_frames": len(open(os.path.join(frames, "pose_state.txt")).read().split()),
            "gravity": np.load(grav_path) if os.path.exists(grav_path) else read_gravity(svo, gpu=gpu),
        }
    info = extract(svo, work_dir, gpu=gpu)
    print(f"[extract] {info['svo_frames']} svo frames, {len(info['poses'])} exported, "
          f"K={info['K'][0, 0]:.2f} f / {info['K'][0, 2]:.1f},{info['K'][1, 2]:.1f} c")
    return info


def extract(svo, work_dir, gpu=None):
    """Play the SVO once, write work_dir/frames/{color,depth,pose} + pose_state.txt +
    intrinsic files. Returns dict(poses=(N,4,4), states, K=(3,3), gravity, svo_frames=N).
    Poses are ZED RIGHT_HANDED_Y_UP, camera OpenCV (x right, y down, z fwd)."""
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
    # SDK 5.4 bug (measured): sdk_gpu_id != 0 makes positional tracking return constant
    # poses (camera never moves, state OK). Tracking/extraction must stay on the default GPU.

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"failed to open {svo}")
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())  # defaults: area memory on

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
    ctw, states, c_ts, d_ts = [], [], [], []
    gravity = None
    i = 0
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"grab failed at frame {i}: {err}")
        if i == total - 1:
            break  # trailing frame is always invalid (verified on 6 release .sens)
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
        c_ts.append(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds())
        d_ts.append(c_ts[-1])
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

    with open(os.path.join(frames, "pose_state.txt"), "w") as f:
        f.write("\n".join(states))

    np.save(os.path.join(frames, "camera_to_world.npy"), np.stack(ctw))
    np.save(os.path.join(frames, "gravity.npy"), gravity)
    return dict(poses=np.stack(ctw), states=states, K=K, gravity=gravity,
                svo_frames=total)
