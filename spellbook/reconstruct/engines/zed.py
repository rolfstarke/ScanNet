"""ZED engine: ZED SDK positional tracking (two-pass with .area relocalization) for the
poses, then the shared Open3D TSDF re-integration of ZED's own depth frames for the
mesh -- the track-then-reintegrate architecture ScanNet itself uses (BundleFusion poses ->
VoxelHashing mesh).

Why not ZED spatial mapping: the SDK's FusedPointCloud is unreliable in this setup --
extract_whole_spatial_map returned 34-38M points with attribute-set resolution (a solid
point block, i.e. resolution ignored) and 0 points with enum-set resolution, both with
MAPPING_STATE.OK the whole run. The documented SDK recommendation for offline mapping is
the .area two-pass (used here).

Poses: get_position during pass 2 with the .area loaded = loop-closure-relocalized in the
same frame as pass 1's map. ScanNet parameters: depth 0.1-6.0 m, voxel 2 cm (deviation,
see open3d engine), truncation 0.06 m.
"""
import os

import numpy as np

from . import open3d as open3d_engine


def _init(svo, gpu):
    import pyzed.sl as sl
    init = sl.InitParameters()
    init.set_from_svo_file(svo)
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init.depth_minimum_distance = 0.1
    init.depth_maximum_distance = 6.0
    # SDK 5.4 bug (measured): sdk_gpu_id != 0 makes positional tracking return constant
    # poses (camera never moves, state OK). Tracking must stay on the default GPU.
    return init


def _grab_all(zed):
    import pyzed.sl as sl
    runtime = sl.RuntimeParameters()
    total = zed.get_svo_number_of_frames()
    i = 0
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"grab failed at frame {i}: {err}")
        i += 1
    return i


def reconstruct(work, root, gpu=None):
    import pyzed.sl as sl

    marker = os.path.join(work, "svo_path.txt")
    if not os.path.exists(marker):
        raise RuntimeError("run.py must write recon/svo_path.txt before engines run")
    svo = open(marker).read().strip()
    area = os.path.join(work, "map.area")

    # ---- pass 1: build the .area file (area memory / relocalization map) ----
    zed = sl.Camera()
    if zed.open(_init(svo, gpu)) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("failed to open svo (pass 1)")
    tr = sl.PositionalTrackingParameters()
    tr.enable_area_memory = True
    zed.enable_positional_tracking(tr)
    _grab_all(zed)
    if not zed.save_area_map(area):
        raise RuntimeError("save_area_map failed")
    zed.disable_positional_tracking()
    zed.close()

    # ---- pass 2: replay with the area loaded, record relocalized poses ----
    zed = sl.Camera()
    if zed.open(_init(svo, gpu)) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("failed to open svo (pass 2)")
    tr = sl.PositionalTrackingParameters()
    tr.enable_area_memory = True
    tr.area_file_path = area
    zed.enable_positional_tracking(tr)
    runtime = sl.RuntimeParameters()
    total = zed.get_svo_number_of_frames()
    pose = sl.Pose()
    poses, keep = [], []
    i = 0
    while True:
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"grab failed at frame {i} (pass 2): {err}")
        state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        if i < total - 1 and state == sl.POSITIONAL_TRACKING_STATE.OK:
            poses.append(pose.pose_data(sl.Transform()).m.copy())
            keep.append(i)
        i += 1
    zed.disable_positional_tracking()
    zed.close()

    if len(keep) < 100:
        raise RuntimeError(f"only {len(keep)} valid poses")

    # ---- shared 4 mm TSDF re-integration (same code path as the open3d engine) ----
    poses = np.stack(poses)
    native, poses = open3d_engine._integrate(work, keep, gpu, poses)
    return native, poses, keep, "zed_opencv"
