"""Runs Mosaic3D directly from the point cloud alone (no posed RGB-D frames needed): SpUNet/PPT
encoder -> per-point open-vocabulary CLIP-alignment classification -> instances from frozen Mask3D
proposals (default, paper Table 2 row c: mean-softmax mask voting) or per-class DBSCAN (lower bound,
#10). Writes the result in ScanNet benchmark instance-segmentation format (scan-net.org).

The model side lives in _mosaic3d_scene.py (recreated `run_custom_scene.py`, built only from
Mosaic3D's own hydra configs/modules) -- this script adapts its output into ScanNet format and
holds the run knobs: winner presets, label-blind geometric rescoring, dedup, prompt mode.

Up-axis is detected per scene via a 5th-95th-percentile axis-range check (whichever axis has
the tightest range is "up", matching room height).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    _benchmark_spec, add_run_args, decimate, load_overrides, publish_clip_features,
    scene_id_from_pointcloud, write_scannet_submission_rows,
)

MOSAIC3D_REPO = "/data/Mosaic3D"
CHECKPOINT = "/data/mosaic3d/ckpts/sc.ckpt"
CONDITION = "ScanNet"
GRID_SIZE = 0.02
INSTANCE_HEAD = "mask3d"
OPENINS3D_PY = "/data/openins3d/conda/envs/openins3d/bin/python"
MASK3D_CONFIDENCE = 0.001

POINT_LIMIT = 1_500_000  # SpUNet's per-point feature gather OOMs on a 16GB card above ~2M points
MIN_MASK_POINTS = 100
SCRATCH_ROOT = "/data/mosaic3d/scratch"  # transient decimated meshes -- kept off the submission root
SCENE_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mosaic3d_scene.py")

# Mask3D proposal npz: v1 = {masks}; v2 = {masks[N,M] bool, scores[M] float32, format=2}.
PROPOSAL_NPZ_FORMAT = 2

RESCORE_MODES = ("none", "geom1", "geom2")
PROMPT_MODES = ("single", "ensemble_bg")
INSTANCE_HEADS = ("mask3d", "dbscan")
# #33 promotions: h3 (batch 1, AP 0.10687) and b2-geom2 (batch 2, AP 0.107841, current best).
WINNER_PRESETS = {
    "h3-proxyrescore": {"rescore_mode": "geom1"},
    "b2-geom2": {"rescore_mode": "geom2"},
}
ALLOWED_PARAMS = frozenset({
    "grid_size", "point_limit", "min_mask_points", "condition", "instance_head",
    "mask3d_confidence", "checkpoint", "preset", "rescore_mode", "dedup_iou", "prompt_mode",
})

# [inferred] geometric proxy constants (see #33; local notes: spellbook/archive/recovery/mosaic3d-derivation.md)
GEOM_VOXEL = 0.05        # connectivity grid (m); 26-neighbourhood on occupied voxels
GEOM2_HALF_POINTS = 200  # geom2 size prior n / (n + n0)

STRUCTURAL_CLASS_PROFILES = {
    "wall": dict(eps=0.40, min_points=100, max_extent=8.0),
    "floor": dict(eps=0.40, min_points=100, max_extent=8.0),
    "ceiling": dict(eps=0.40, min_points=100, max_extent=8.0),
}


def resolve_params(params):
    """Expand `preset` (explicit keys win) and validate the Mosaic3D-only enums. `{}` -> `{}`."""
    params = dict(params)
    preset = params.pop("preset", None)
    if preset is not None:
        if preset not in WINNER_PRESETS:
            raise ValueError(f"preset must be one of {sorted(WINNER_PRESETS)}, got {preset!r}")
        params = {**WINNER_PRESETS[preset], **params}
    for key, choices in (("rescore_mode", RESCORE_MODES), ("prompt_mode", PROMPT_MODES),
                         ("instance_head", INSTANCE_HEADS)):
        if key in params and params[key] not in choices:
            raise ValueError(f"{key} must be one of {list(choices)}, got {params[key]!r}")
    if params.get("rescore_mode", "none") != "none" and \
            params.get("instance_head", INSTANCE_HEAD) != "mask3d":
        raise ValueError("rescore_mode applies to the mask3d instance head only")
    return params


def proposal_cache_path(scene_id, confidence):
    """[inferred] Cache keyed on the Mask3D confidence threshold (scene-level, shared across runs)."""
    return os.path.join(SCRATCH_ROOT, "mask3d_proposals", scene_id,
                        f"mask3d_masks.conf{float(confidence)!r}.v{PROPOSAL_NPZ_FORMAT}.npz")


def load_proposals(path, need_scores=False):
    """Returns (masks[N,M] bool, scores[M] or None). Readers that need scores fail fast on v1."""
    with np.load(path) as data:
        fmt = int(data["format"]) if "format" in data.files else 1
        masks = data["masks"].astype(bool)
        scores = data["scores"].astype(np.float32) if fmt >= 2 else None
    if need_scores and scores is None:
        raise ValueError(f"{path}: proposal npz v{fmt} has no scores (need v{PROPOSAL_NPZ_FORMAT})")
    if scores is not None and len(scores) != masks.shape[1]:
        raise ValueError(f"{path}: {len(scores)} scores for {masks.shape[1]} masks")
    return masks, scores


def _mask3d_masks(pointcloud, run_id, scene_id, confidence):
    out = proposal_cache_path(scene_id, confidence)
    if not os.path.isfile(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        script = os.path.join(os.path.dirname(__file__), "_mask3d_proposals.py")
        fd = os.environ.get("SPELLBOOK_GPU_LEASE_FD")
        pass_fds = (int(fd),) if fd else ()
        subprocess.run(
            [OPENINS3D_PY, script, "--pointcloud", pointcloud, "--out", out,
             "--confidence", str(confidence)],
            env=os.environ.copy(), check=True, pass_fds=pass_fds)
    return load_proposals(out)[0]


def connectivity_ratio(points, voxel=GEOM_VOXEL):
    """[inferred] Share of a mask's points in its largest 26-connected voxel component."""
    if len(points) == 0:
        return 0.0
    keys, inverse = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    pairs = cKDTree(keys).query_pairs(r=np.sqrt(3) + 1e-6, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else
                       ([], ([], [])), shape=(len(keys), len(keys)))
    _, comp = connected_components(graph, directed=False)
    return float(np.bincount(comp[inverse]).max() / len(points))


def geometric_factor(mode, points):
    """[inferred] Label-blind mask-quality proxy multiplied onto the mean-softmax score."""
    if mode == "geom1":
        return connectivity_ratio(points)
    if mode == "geom2":
        n = len(points)
        return connectivity_ratio(points) * n / (n + GEOM2_HALF_POINTS)
    raise ValueError(f"unknown rescore_mode {mode!r}")


def rescore(candidates, points, mode):
    if mode == "none":
        return candidates
    return [(mask, name, score * geometric_factor(mode, points[mask]))
            for mask, name, score in candidates]


def dedup(candidates, iou_threshold):
    """[inferred] Greedy same-class mask NMS on the final scores (#33 h1); survivors keep order."""
    def iou(a, b):
        inter = np.count_nonzero(a & b)
        return inter / max(np.count_nonzero(a) + np.count_nonzero(b) - inter, 1)

    kept = []
    for i in sorted(range(len(candidates)), key=lambda i: -candidates[i][2]):
        mask_i, name_i, _ = candidates[i]
        if all(name_i != candidates[j][1] or iou(mask_i, candidates[j][0]) <= iou_threshold
               for j in kept):
            kept.append(i)
    return [candidates[i] for i in sorted(kept)]


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
    params = resolve_params(load_overrides(args.parameters_json, ALLOWED_PARAMS))
    grid_size = float(params.get("grid_size", GRID_SIZE))
    point_limit = int(params.get("point_limit", POINT_LIMIT))
    min_mask_points = int(params.get("min_mask_points", MIN_MASK_POINTS))
    condition = str(params.get("condition", CONDITION))
    instance_head = str(params.get("instance_head", INSTANCE_HEAD))
    mask3d_confidence = float(params.get("mask3d_confidence", MASK3D_CONFIDENCE))
    checkpoint = str(params.get("checkpoint", CHECKPOINT))
    rescore_mode = str(params.get("rescore_mode", "none"))
    dedup_iou = params.get("dedup_iou")
    prompt_mode = str(params.get("prompt_mode", "single"))
    print(f"[INFO] {scene_id} run_id={args.run_id} head={instance_head} ckpt={checkpoint} overrides={params}")

    import torch
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
    os.chdir(MOSAIC3D_REPO)  # Mosaic3D's own imports assume repo root as cwd
    from _mosaic3d_scene import (  # noqa: E402
        CLIP_CFG, build_net, load_backbone_ckpt, ppt_conditions_for, run_inference)
    from src.models.utils.clip_models import build_clip_model  # noqa: E402
    ppt_conditions = ppt_conditions_for(checkpoint)

    class _CaptureNet:
        def __init__(self, net):
            self.net = net
            self.point = None

        def __call__(self, *args, **kwargs):
            self.point = self.net(*args, **kwargs)
            return self.point

        def eval(self):
            self.net.eval()
            return self

        def to(self, device_):
            self.net.to(device_)
            return self

    feature_rows = None
    if instance_head == "mask3d":
        from _mosaic3d_scene import classify_mask3d_proposals
        masks = _mask3d_masks(args.pointcloud, args.run_id, scene_id, mask3d_confidence)
        if masks.shape[0] != len(full_pts):
            raise RuntimeError(f"Mask3D N={masks.shape[0]} != mesh N={len(full_pts)}")
        objects, _ = classify_mask3d_proposals(
            args.pointcloud, args.classes, checkpoint, device, masks,
            condition=condition, grid_size=grid_size, up_axis=up_axis,
            min_mask_points=min_mask_points, ppt_conditions=ppt_conditions,
            prompt_mode=prompt_mode)
        nn_idx = np.arange(len(full_pts))
        working_pts = full_pts
    else:
        if prompt_mode != "single":
            raise ValueError("prompt_mode applies to the mask3d instance head only")
        net = build_net(ppt_conditions)
        load_backbone_ckpt(net, checkpoint)
        net = net.to(device).eval()
        clip_model = build_clip_model(CLIP_CFG, device=device)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        capturer = _CaptureNet(net)
        objects, _, _ = run_inference(
            scene_ply, args.classes, checkpoint, device, condition=condition,
            grid_size=grid_size, up_axis=up_axis,
            class_profiles=STRUCTURAL_CLASS_PROFILES, net=capturer, clip_model=clip_model,
        )
        if capturer.point is None:
            raise RuntimeError("Mosaic3D encoder produced no captured point features")
        point = capturer.point
        point_feat = point.sparse_conv_feat.features[point.v2p_map]
        point_feat = torch.nn.functional.normalize(
            point_feat.float(), dim=-1).detach().cpu().numpy()
        feature_rows = []
        for obj in objects:
            vec = point_feat[obj["point_indices"]].mean(axis=0)
            norm = np.linalg.norm(vec)
            if norm >= 1e-6:
                vec = vec / norm
            feature_rows.append(vec.astype(np.float32))

    candidates = []
    for obj in objects:
        sel_working = np.zeros(len(working_pts), dtype=bool)
        sel_working[obj["point_indices"]] = True
        candidates.append((sel_working[nn_idx], obj["class_name"], obj["score"]))
    candidates = rescore(candidates, full_pts, rescore_mode)
    if dedup_iou is not None:
        candidates = dedup(candidates, float(dedup_iou))

    rows = write_scannet_submission_rows(args.out, scene_id, args.classes, candidates,
                                        min_mask_points, spec)
    if feature_rows is not None:
        publish_clip_features(
            args.features_out, feature_rows, rows, spec, args.run_id, "mosaic3d", scene_id,
            "open_clip", CLIP_CFG["model_id"], 768)
    else:
        print(f"[INFO] mask3d head exposes no point features; skipping clip publish")
    print(f"[INFO] Wrote {rows['n_written']} instances to {args.out} ({len(working_pts)}/{len(full_pts)} points used)")


if __name__ == "__main__":
    main()
