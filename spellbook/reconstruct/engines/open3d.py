"""Open3D TSDF integration with an externally provided trajectory (the documented
external-pose path: ScalableTSDFVolume.integrate with per-frame camera extrinsics).

Pose source: the caller's own poses (ZED tracking for the zed engine; Stage A poses for the
open3d engine). ScanNet parameters: voxel 2 cm (DEVIATION, see _integrate), truncation
0.06 m, depth 0.1-6.0 m, depth_scale 1000, integration res 320x240.
"""
import os

import numpy as np

# ZED RIGHT_HANDED_Y_UP camera basis (x right, y up, z BACK) -> OpenCV/ScanNet camera
# basis (x right, y down, z forward). Poses from the ZED SDK must be conjugated with D
# before they can drive projection of the pinhole images (D = D^-1).
_ZED_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _integrate(work, keep, gpu=None, poses=None):
    """TSDF integration via the legacy ScalableTSDFVolume (the documented CPU pipeline
    o3d.pipelines.integration). Voxel 2 cm -- a documented DEVIATION: the installed
    open3d 0.19 build's tensor VoxelBlockGrid is broken (CUDA MemoryCache crash,
    CPU hashmap segfault) and the legacy volume at 4 mm balloons to ~185 GB RSS for 100
    frames. 2 cm keeps the memory bounded; 4 mm re-integration is a tuning follow-up.
    Depth/color integrated at 320x240 with scaled intrinsics (ScanNet
    s_integrationWidth/Height, zParametersScanNet.txt:20-21).
    Returns (native_mesh_path, conjugated_poses)."""
    import cv2
    import open3d as o3d

    frames_dir = os.path.join(work, "frames")
    K = np.loadtxt(os.path.join(frames_dir, "intrinsic_depth.txt"))[:3, :3]
    iw, ih = 320, 240
    valid = [i for i in keep if (os.path.exists(os.path.join(frames_dir, "color", f"{i}.jpg"))
                                 and os.path.exists(os.path.join(frames_dir, "depth", f"{i}.png")))]
    if len(valid) < 100:
        raise RuntimeError(f"only {len(valid)} readable frames, aborting")
    print(f"[integrate] {len(valid)} frames at {iw}x{ih}, voxel 20 mm")

    # intrinsics scaled to the integration resolution (ScanNet does the same);
    # scale factors from the actual frame dimensions (HD1080 vs HD1200 recordings)
    c0 = cv2.imread(os.path.join(frames_dir, "color", f"{valid[0]}.jpg"))
    dw, dh = c0.shape[1], c0.shape[0]
    Ki = K.copy()
    Ki[0] *= iw / float(dw)
    Ki[1] *= ih / float(dh)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02, sdf_trunc=0.06,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    cam = o3d.camera.PinholeCameraIntrinsic(iw, ih, Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2])
    pos_lookup = {fi: j for j, fi in enumerate(keep)}  # keep index -> pose index
    for fi in valid:
        c = cv2.imread(os.path.join(frames_dir, "color", f"{fi}.jpg"))
        d = cv2.imread(os.path.join(frames_dir, "depth", f"{fi}.png"), cv2.IMREAD_UNCHANGED)
        c = cv2.resize(c, (iw, ih), interpolation=cv2.INTER_LINEAR)
        d = cv2.resize(d, (iw, ih), interpolation=cv2.INTER_NEAREST)
        rgb = o3d.geometry.Image(np.ascontiguousarray(c[:, :, ::-1]))
        depth = o3d.geometry.Image(np.ascontiguousarray(d.astype(np.float32) / 1000.0))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_scale=1.0, depth_trunc=6.0, convert_rgb_to_intensity=False)
        # use the ENGINE's poses when given (the zed engine's .area pass-2 trajectory);
        # fall back to the Stage A pose files otherwise (the open3d engine).
        # ScalableTSDFVolume.integrate expects the camera extrinsic = inverse of the
        # camera-to-world pose (Open3D's own integrate_scene.py: `np.linalg.inv(pose)`).
        if poses is not None:
            e = _ZED_TO_OPENCV @ poses[pos_lookup[fi]] @ _ZED_TO_OPENCV
        else:
            e = _ZED_TO_OPENCV @ np.loadtxt(os.path.join(frames_dir, "pose", f"{fi}.txt")) @ _ZED_TO_OPENCV
        vol.integrate(rgbd, cam, np.linalg.inv(e))
    mesh = vol.extract_triangle_mesh()
    native = os.path.join(work, "engine_native.ply")
    o3d.io.write_triangle_mesh(native, mesh, write_vertex_colors=True)
    print(f"[integrate] mesh {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")
    if poses is not None:
        poses = np.stack([_ZED_TO_OPENCV @ p @ _ZED_TO_OPENCV for p in poses])
    return native, poses


def reconstruct(work, root, gpu=None):
    """open3d engine: Stage A ZED tracking poses + shared 4 mm integration."""
    frames = os.path.join(work, "frames")
    states = open(os.path.join(frames, "pose_state.txt")).read().split()
    keep = [i for i, s in enumerate(states) if s == "OK"]
    if len(keep) < 100:
        raise RuntimeError(f"only {len(keep)} valid frames, aborting")
    poses = np.stack([np.loadtxt(os.path.join(frames, "pose", f"{i}.txt")) for i in keep])
    native, poses = _integrate(work, keep, gpu, poses)
    return native, poses, keep, "zed_opencv"
