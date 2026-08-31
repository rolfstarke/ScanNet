"""Runs Mosaic3D directly from the point cloud alone (no posed RGB-D frames needed): SpUNet/PPT
encoder -> per-point open-vocabulary CLIP-alignment classification -> per-class DBSCAN instance
formation. Writes the result in ScanNet benchmark instance-segmentation format (scan-net.org).

The actual model/algorithm lives in the Mosaic3D repo
(/home/rolf/GIT/Mosaic3D/scripts/run_custom_scene.py:run_inference) -- this script only adapts
its output into ScanNet format, so the algorithm has one home instead of being duplicated.

Up-axis is detected per scene via a 5th-95th-percentile axis-range check (whichever axis has
the tightest range is "up", matching room height).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    _benchmark_spec, add_run_args, decimate, load_overrides,
    scene_id_from_pointcloud, write_scannet_submission,
)

MOSAIC3D_REPO = "/home/rolf/GIT/Mosaic3D"
CHECKPOINT = "/data/mosaic3d/ckpts/sc.ckpt"
CONDITION = "ScanNet"
GRID_SIZE = 0.02
INSTANCE_HEAD = "mask3d"
OPENINS3D_PY = "/data/openins3d/conda/envs/openins3d/bin/python"
MASK3D_CONFIDENCE = 0.001

POINT_LIMIT = 1_500_000  # SpUNet's per-point feature gather OOMs on a 16GB card above ~2M points
MIN_MASK_POINTS = 100
SCRATCH_ROOT = "/data/mosaic3d/scratch"  # transient decimated meshes -- kept off the submission root

STRUCTURAL_CLASS_PROFILES = {
    "wall": dict(eps=0.40, min_points=100, max_extent=8.0),
    "floor": dict(eps=0.40, min_points=100, max_extent=8.0),
    "ceiling": dict(eps=0.40, min_points=100, max_extent=8.0),
}


def _mask3d_masks(pointcloud, run_id, scene_id, confidence):
    out = os.path.join(SCRATCH_ROOT, "mask3d_proposals", scene_id, "mask3d_masks.npz")
    if os.path.isfile(out):
        return np.load(out)["masks"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    script = os.path.join(os.path.dirname(__file__), "_mask3d_proposals.py")
    fd = os.environ.get("SPELLBOOK_GPU_LEASE_FD")
    pass_fds = (int(fd),) if fd else ()
    subprocess.run(
        [OPENINS3D_PY, script, "--pointcloud", pointcloud, "--out", out,
         "--confidence", str(confidence)],
        env=os.environ.copy(), check=True, pass_fds=pass_fds)
    return np.load(out)["masks"]


def _detect_up_axis(pts):
    lo, hi = np.percentile(pts, [5, 95], axis=0)
    ranges = hi - lo
    axis = int(np.argmin(ranges))
    print(f"[INFO] detected up_axis={axis} ({'XYZ'[axis]}) -- 5-95pct ranges: "
          f"X={ranges[0]:.2f}m Y={ranges[1]:.2f}m Z={ranges[2]:.2f}m")
    return axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True, help="ScanNet scene pointcloud (.ply)")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="predictions output dir")
    ap.add_argument("--benchmark", default="ScanNet20",
                    choices=["ScanNet20", "ScanNet200"])
    add_run_args(ap)
    args = ap.parse_args()
    spec = _benchmark_spec(args.benchmark)
    scene_id = scene_id_from_pointcloud(args.pointcloud)
    params = load_overrides(args.parameters_json, {
        "grid_size", "point_limit", "min_mask_points", "condition",
        "instance_head", "mask3d_confidence", "checkpoint"})
    grid_size = float(params.get("grid_size", GRID_SIZE))
    point_limit = int(params.get("point_limit", POINT_LIMIT))
    min_mask_points = int(params.get("min_mask_points", MIN_MASK_POINTS))
    condition = str(params.get("condition", CONDITION))
    instance_head = str(params.get("instance_head", INSTANCE_HEAD))
    mask3d_confidence = float(params.get("mask3d_confidence", MASK3D_CONFIDENCE))
    checkpoint = str(params.get("checkpoint", CHECKPOINT))
    print(f"[INFO] {scene_id} run_id={args.run_id} head={instance_head} ckpt={checkpoint} overrides={params}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out, exist_ok=True)

    pcd = o3d.io.read_point_cloud(args.pointcloud)
    full_pts = np.asarray(pcd.points)
    full_cols = np.asarray(pcd.colors)
    up_axis = _detect_up_axis(full_pts)

    working_pts, nn_idx = decimate(full_pts, point_limit)
    if len(working_pts) < len(full_pts):
        working_cols = full_cols[cKDTree(full_pts).query(working_pts, k=1, workers=-1)[1]]
        working_ply = os.path.join(SCRATCH_ROOT, args.run_id, scene_id, "working_scene.ply")
        os.makedirs(os.path.dirname(working_ply), exist_ok=True)
        working_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(working_pts))
        working_pcd.colors = o3d.utility.Vector3dVector(working_cols)
        o3d.io.write_point_cloud(working_ply, working_pcd)
        scene_ply = working_ply
    else:
        scene_ply = args.pointcloud

    sys.path.insert(0, MOSAIC3D_REPO)
    os.chdir(MOSAIC3D_REPO)
    if instance_head == "mask3d":
        from scripts.run_custom_scene import classify_mask3d_proposals
        masks = _mask3d_masks(args.pointcloud, args.run_id, scene_id, mask3d_confidence)
        if masks.shape[0] != len(full_pts):
            raise RuntimeError(f"Mask3D N={masks.shape[0]} != mesh N={len(full_pts)}")
        objects, _ = classify_mask3d_proposals(
            args.pointcloud, args.classes, checkpoint, device, masks,
            condition=condition, grid_size=grid_size, up_axis=up_axis,
            min_mask_points=min_mask_points)
        nn_idx = np.arange(len(full_pts))
        working_pts = full_pts
    else:
        from scripts.run_custom_scene import run_inference
        objects, _, _ = run_inference(
            scene_ply, args.classes, checkpoint, device, condition=condition,
            grid_size=grid_size, up_axis=up_axis,
            class_profiles=STRUCTURAL_CLASS_PROFILES,
        )

    def _instances():
        for obj in objects:
            sel_working = np.zeros(len(working_pts), dtype=bool)
            sel_working[obj["point_indices"]] = True
            yield sel_working[nn_idx], obj["class_name"], obj["score"]

    n_written = write_scannet_submission(args.out, scene_id, args.classes, _instances(),
                                          min_mask_points, spec)
    print(f"[INFO] Wrote {n_written} instances to {args.out} ({len(working_pts)}/{len(full_pts)} points used)")


if __name__ == "__main__":
    main()
