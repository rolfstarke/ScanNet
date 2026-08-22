"""ZED engine: BLOCKED.

ZED SDK positional tracking (two-pass with .area relocalization) for the poses, then
the shared Open3D TSDF re-integration of ZED's own depth frames for the mesh -- the
track-then-reintegrate architecture ScanNet itself uses (BundleFusion poses ->
VoxelHashing mesh).

Why not ZED spatial mapping: the SDK's FusedPointCloud is unreliable in this setup --
extract_whole_spatial_map returned 34-38M points with attribute-set resolution (a solid
point block, i.e. resolution ignored) and 0 points with enum-set resolution, both with
MAPPING_STATE.OK the whole run. The documented SDK recommendation for offline mapping is
the .area two-pass (used here).

Why blocked: the SDK selects its default CUDA device itself (sdk_gpu_id != 0 returns
constant poses, #18), which lands on physical GPU 0 -- now user-reserved. Managed
CUDA_VISIBLE_DEVICES remapping is not validated yet, so ZED runs are disabled until a
separate pose-validity test approves a remapped path. ScanNet parameters (used by the
historical runs, kept for the future remapped path): depth 0.1-6.0 m, voxel 2 cm
(deviation, see open3d engine), truncation 0.06 m.
"""

GPU_POLICY = "blocked"
SERIAL = True

BLOCK_REASON = ("ZED disabled: its SDK default-device path would use user-reserved "
                "physical GPU 0 (#18); managed remapping not validated")


def preflight():
    return BLOCK_REASON


def gpu_check(gpu=None, lease_fd=None, hold_seconds=5):
    return dict(status="blocked", reason=BLOCK_REASON)


def reconstruct(work, root, gpu=None, lease_fd=None):
    raise RuntimeError(BLOCK_REASON)