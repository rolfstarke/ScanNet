"""Open3D engine: Stage A ZED tracking poses + shared TSDF integration (CPU).

GPU_POLICY "cpu": the legacy ScalableTSDFVolume integration is CPU-only; no GPU lease
is taken and `gpu` is always None here. Shared integration lives in
spellbook/reconstruct/tsdf.py (integrate_tsdf).
"""
import os

import numpy as np

from ..tsdf import integrate_tsdf

GPU_POLICY = "cpu"
SERIAL = False


def preflight():
    try:
        import open3d  # noqa: F401
    except Exception as ex:
        return f"open3d not importable: {ex}"
    return None


def reconstruct(work, root, gpu=None, lease_fd=None):
    """open3d engine: Stage A ZED tracking poses + shared TSDF integration."""
    frames = os.path.join(work, "frames")
    states = open(os.path.join(frames, "pose_state.txt")).read().split()
    keep = [i for i, s in enumerate(states) if s == "OK"]
    if len(keep) < 100:
        raise RuntimeError(f"only {len(keep)} valid frames, aborting")
    poses = np.stack([np.loadtxt(os.path.join(frames, "pose", f"{i}.txt")) for i in keep])
    native, poses = integrate_tsdf(work, keep, poses)
    return native, poses, keep, "zed_opencv"
