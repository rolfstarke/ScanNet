"""Runs OpenIns3D's Mask -> Snap -> Lookup pipeline (class-agnostic Mask3D proposals, or the released
precomputed `masks/detected/*.pt` -> synthetic scene renders -> ODISE/YOLO-World open-vocab labelling)
inside the `openins3d` conda env, writing the result in ScanNet benchmark instance-segmentation
format (scan-net.org).

Snap synthesizes 2D views directly from the point cloud, so -- unlike OpenMask3D/OpenYOLO3D --
no posed frames are needed (point-cloud-only, like Mosaic3D). The pipeline itself lives in
OpenIns3D; this script just loads the cloud, calls OpenIns3D's own functions exactly as its
zero_shot_multi_vocs.py demo does, and reformats the output.
"""
import os
import sys

import argparse
import importlib.util

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    _benchmark_spec, add_run_args, decimate, load_overrides,
    scene_id_from_pointcloud, write_scannet_submission,
)

OPENINS3D_REPO = "/data/openins3d/repo-56ccc22"  # worktree of the logged upstream commit 56ccc22
PRECOMPUTED_MASK_DIR = os.path.join(OPENINS3D_REPO, "data", "scannet", "masks", "detected")
SCRATCH_ROOT = "/data/openins3d/scratch"  # bulky, transient synthetic renders -- kept off /home
CKPT = "third_party/scannet200_val.ckpt"

POINT_LIMIT = 1_000_000
IMAGE_SIZE = [800, 800]
ADJUST_CAMERA = [2, 0.1, 1.0]
# 0.5 is the repo demo default; the paper's benchmark pipeline evaluates at ~0.001 recall
# (precomputed masks), so 0.5 kills too many proposals (e.g. scene0304_00: only 4/150 masks
# survived). 0.05 is a pragmatic middle ground for on-the-fly proposals.
MASK_CONFIDENCE_THRESHOLD = 0.05
LOOKUP_THRESHOLD = 0.3
MIN_MASK_POINTS = 20


def _lookup_vocab(spec, classes):
    """Official 200-way ScanNet200 Lookup prompts (wall/floor are dropped only at submission)."""
    if spec.name != "ScanNet200":
        return list(classes)  # [inferred] other benchmarks keep their own prompt list
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    path = os.path.join(repo, "BenchmarkScripts", "ScanNet200", "scannet200_constants.py")
    spec_mod = importlib.util.spec_from_file_location("scannet200_constants", path)
    module = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(module)
    return list(module.CLASS_LABELS_200)


def _load_precomputed_masks(scene_id, n_vertices, mask_dir=PRECOMPUTED_MASK_DIR):
    """Released proposals as OpenIns3D's main.py loads them; fails closed unless vertex-aligned."""
    path = os.path.join(mask_dir, f"{scene_id}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"precomputed masks missing: {path}")
    masks = torch.load(path, map_location="cpu")
    masks = masks.to_dense() if masks.is_sparse else masks
    if masks.ndim != 2 or masks.shape[0] != n_vertices:
        raise ValueError(f"{path}: masks {tuple(masks.shape)} do not match {n_vertices} vertices")
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True, help="scene pointcloud (.ply, xyz+rgb)")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="predictions output dir")
    ap.add_argument("--benchmark", default="ScanNet20",
                    choices=["ScanNet20", "ScanNet200"],
                    help="benchmark backend (default ScanNet20)")
    ap.add_argument("--detector", default="odise", choices=["odise", "yoloworld"],
                    help="2D open-vocab detector for the Lookup stage (default odise -- the "
                          "detector OpenIns3D's paper numbers use; yoloworld is the repo demo default)")
    add_run_args(ap)
    args = ap.parse_args()
    spec = _benchmark_spec(args.benchmark)
    params = load_overrides(args.parameters_json, {
        "detector", "mask_confidence_threshold", "lookup_threshold",
        "point_limit", "min_mask_points", "use_precomputed"})
    use_precomputed = bool(params.get("use_precomputed", False))  # [inferred] default off
    detector = str(params.get("detector", args.detector))
    mask_confidence_threshold = float(params.get(
        "mask_confidence_threshold", MASK_CONFIDENCE_THRESHOLD))
    lookup_threshold = float(params.get("lookup_threshold", LOOKUP_THRESHOLD))
    point_limit = int(params.get("point_limit", POINT_LIMIT))
    min_mask_points = int(params.get("min_mask_points", MIN_MASK_POINTS))

    os.makedirs(args.out, exist_ok=True)
    scene_id = scene_id_from_pointcloud(args.pointcloud)
    print(f"[INFO] {scene_id} run_id={args.run_id} detector={detector} "
          f"use_precomputed={use_precomputed} "
          f"mask_conf={mask_confidence_threshold} lookup={lookup_threshold}")

    pcd = o3d.io.read_point_cloud(args.pointcloud)
    full_pts = np.asarray(pcd.points)  # ScanNet is natively z-up, as Snap's rendered mesh is
    full_cols = np.asarray(pcd.colors)

    if use_precomputed:
        # Masks are per-vertex: no decimation, or Lookup and Snap would see different geometry.
        working_pts, working_cols = full_pts, full_cols
        nn_idx = np.arange(len(full_pts))
        mask_list = _load_precomputed_masks(scene_id, len(full_pts))
    else:
        working_pts, nn_idx = decimate(full_pts, point_limit)
        if len(working_pts) < len(full_pts):
            working_cols = full_cols[cKDTree(full_pts).query(working_pts, k=1, workers=-1)[1]]
        else:
            working_cols = full_cols
    pcd_rgb = np.hstack([working_pts, working_cols * 255.0])

    os.chdir(OPENINS3D_REPO)
    sys.path.insert(0, OPENINS3D_REPO)
    sys.path.insert(0, os.path.join(OPENINS3D_REPO, "openins3d"))  # upstream's bare `utils`/`snap`
    from openins3d.snap import Snap
    from openins3d.lookup import Lookup

    if not use_precomputed:
        from openins3d.mask3d import get_model, prepare_data_pcd, map_output_to_pointcloud
        device = torch.device("cuda")
        model = get_model(CKPT).to(device).eval()
        data, _, _, features, _, inverse_map = prepare_data_pcd(
            working_pts, working_cols * 255.0, device)
        with torch.no_grad():
            mask_list = map_output_to_pointcloud(model(data, raw_coordinates=features), inverse_map,
                                                   confidence_threshold=mask_confidence_threshold)

    if mask_list.shape[1] == 0:  # as OpenIns3D's main.py: no proposals, empty submission
        write_scannet_submission(args.out, scene_id, args.classes, [], min_mask_points, spec)
        print(f"[INFO] Wrote 0 instances to {args.out} (0 raw masks)")
        return

    vocab = _lookup_vocab(spec, args.classes)
    scored = set(args.classes)
    snap_folder = os.path.join(SCRATCH_ROOT, args.run_id, scene_id)
    snap = Snap(IMAGE_SIZE, ADJUST_CAMERA, snap_folder)
    lookup = Lookup(IMAGE_SIZE, ADJUST_CAMERA[2], snap_folder, text_input=vocab,
                    results_folder=os.path.join(SCRATCH_ROOT, args.run_id, f"{scene_id}_results"))
    if detector == "odise":
        lookup.call_ODISE()
    else:
        lookup.call_YOLOWORLD()

    snap.scene_image_rendering(args.pointcloud, scene_id, mode=["global", "wide", "corner"])
    mask_cls, score = lookup.lookup_pipelie(pcd_rgb, mask_list, scene_id, threshold=lookup_threshold)

    masks_np = mask_list.cpu().numpy().astype(bool)

    def _instances():
        for i, cls in enumerate(mask_cls):
            if cls == -1 or vocab[cls] not in scored:  # wall/floor: not ScanNet200-submittable
                continue
            yield masks_np[:, i][nn_idx], vocab[cls], float(score[i])

    n_written = write_scannet_submission(args.out, scene_id, args.classes, _instances(),
                                          min_mask_points, spec)
    print(f"[INFO] Wrote {n_written} instances to {args.out} "
          f"({mask_list.shape[1]} raw masks, {len(working_pts)}/{len(full_pts)} points used)")


if __name__ == "__main__":
    main()
