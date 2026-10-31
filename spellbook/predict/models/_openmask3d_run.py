"""Runs OpenMask3D's full pipeline (class-agnostic mask computation -> CLIP mask-feature
computation -> per-mask CLIP classification) inside the `openmask3d` conda env, writing
the result in ScanNet benchmark instance-segmentation format (scan-net.org).

Adapted from the proven ov3dis-comparison wrapper (utils/_openmask3d_run.py) to the
ScanNet harness: official ScanNet checkpoint (scannet200_model.ckpt), the ScanNet frame
layout (intrinsic_color.txt), and the shared spellbook helpers. GPU selection is purely
environment-driven: the runner exposes one physical GPU as visible CUDA device 0, and
OPENMASK3D_FORCE_GPU=0 pins the repo's own get_free_gpu() (which would otherwise scan
ALL physical devices) to that visible device. The lease descriptor is forwarded to both
blocking subprocesses with pass_fds so no child outlives the lease.

Official segment pooling (#46): the published ScanNet200 evaluation snaps class-agnostic
Mask3D masks to ScanNet segments before SAM/CLIP (trainer.py:get_full_res_mask with
eval_on_segments: per-segment scatter_mean -> strict > 0.5 -> broadcast back to points).
This wrapper applies the same operation with <scene>_vh_clean_2.0.010000.segs.json and fails
closed on missing/mismatched segments. SAM prompts are scaled depth->color by the external
openmask3d checkout (branch spellbook-baseline, "scale SAM prompts from depth to color").
"""
import argparse
import json
import os
import subprocess
import sys

import clip
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    _benchmark_spec, add_run_args, decimate, load_overrides, publish_clip_features,
    scene_id_from_pointcloud, write_scannet_submission_rows,
)

OPENMASK3D_REPO = "/data/openmask3d"
CHECKPOINT = "/data/openmask3d/resources/scannet200_model.ckpt"
SAM_CHECKPOINT = "/data/openmask3d/resources/sam_vit_h_4b8939.pth"

POINT_LIMIT = 500_000  # Mask3D mask computation OOMs on a 16GB card above ~900k points (empirical)
MIN_MASK_POINTS = 1
NUM_QUERIES = 150
DBSCAN_MIN_POINTS = 1
FREQUENCY = 10
CLIP_PROMPT = "a {} in a scene"
DEDUP_IOU = 0.5
SCRATCH_ROOT = "/data/openmask3d/scratch"
SEGS_SUFFIX = "_vh_clean_2.0.010000.segs.json"
POOLED_MASKS_NAME = "working_scene_segsnap_masks.pt"  # must end in _masks.pt (see features_path)


def _lease_fd():
    fd = os.environ.get("SPELLBOOK_GPU_LEASE_FD")
    return (int(fd),) if fd else ()


def _run_mask_computation(scene_ply, out_dir, num_queries, dbscan_min_points):
    env = os.environ.copy()
    env["OPENMASK3D_FORCE_GPU"] = "0"
    args = [
        sys.executable, "class_agnostic_mask_computation/get_masks_single_scene.py",
        "general.experiment_name=openmask3d_util",
        f"general.checkpoint={CHECKPOINT}",
        "general.train_mode=false",
        "data.test_mode=test",
        f"model.num_queries={num_queries}",
        "general.use_dbscan=true",
        "general.dbscan_eps=0.95",
        f"general.dbscan_min_points={dbscan_min_points}",
        "general.save_visualizations=false",
        f"general.scene_path={scene_ply}",
        f"general.mask_save_dir={out_dir}",
        f"hydra.run.dir={out_dir}/hydra_outputs/class_agnostic_mask_computation",
    ]
    subprocess.run(args, cwd=os.path.join(OPENMASK3D_REPO, "openmask3d"),
                   env=env, check=True, pass_fds=_lease_fd())
    scene_name = os.path.basename(scene_ply)[:-4]
    return os.path.join(out_dir, f"{scene_name}_masks.pt")


def _run_feature_computation(scene_ply, masks_path, frames, out_dir, frequency):
    env = os.environ.copy()
    # get_free_gpu() scans ALL physical GPUs via nvidia-smi; the visible device is
    # already pinned by the runner, so force its logical index 0.
    env["OPENMASK3D_FORCE_GPU"] = "0"
    with Image.open(os.path.join(frames, "color", "0.jpg")) as im:
        width, height = im.size
    args = [
        sys.executable, "compute_features_single_scene.py",
        f"data.masks.masks_path={masks_path}",
        f"data.camera.poses_path={os.path.join(frames, 'pose')}",
        f"data.camera.intrinsic_path={os.path.join(frames, 'intrinsic_color.txt')}",
        f"data.camera.intrinsic_resolution=[{height},{width}]",
        f"data.depths.depths_path={os.path.join(frames, 'depth')}",
        "data.depths.depth_scale=1000",
        "data.depths.depths_ext=.png",
        f"data.images.images_path={os.path.join(frames, 'color')}",
        "data.images.images_ext=.jpg",
        f"data.point_cloud_path={scene_ply}",
        f"output.output_directory={out_dir}",
        "output.save_crops=False",
        f"openmask3d.frequency={frequency}",
        f"hydra.run.dir={out_dir}/hydra_outputs/mask_features_computation",
        f"external.sam_checkpoint={SAM_CHECKPOINT}",
        "gpu.optimize_gpu_usage=False",
    ]
    subprocess.run(args, cwd=os.path.join(OPENMASK3D_REPO, "openmask3d"),
                   env=env, check=True, pass_fds=_lease_fd())
    return features_path(out_dir, masks_path)


def features_path(out_dir, masks_path):  # [inferred]
    """Upstream compute_features_single_scene.py names its output
    basename(masks_path)[:-9] + "_openmask3d_features.npy" (strips "_masks.pt")."""
    name = os.path.basename(masks_path)
    if not name.endswith("_masks.pt"):
        raise ValueError(f"masks file must end in _masks.pt for the upstream feature "
                         f"naming contract: {masks_path}")
    return os.path.join(out_dir, f"{name[:-9]}_openmask3d_features.npy")


def segments_path(pointcloud):  # [inferred]
    """<scan_dir>/<scene>_vh_clean_2.0.010000.segs.json next to the --pointcloud mesh."""
    scan_dir = os.path.dirname(os.path.abspath(pointcloud))
    return os.path.join(scan_dir, os.path.basename(scan_dir) + SEGS_SUFFIX)


def load_segments(segs_json, n_points):  # [inferred]
    """Per-vertex ScanNet segment ids; fail closed if missing or not one id per point."""
    if not os.path.isfile(segs_json):
        raise FileNotFoundError(f"segment file missing: {segs_json}")
    with open(segs_json) as f:
        seg = np.asarray(json.load(f)["segIndices"], dtype=np.int64)
    if seg.ndim != 1 or len(seg) != n_points:
        raise ValueError(f"{segs_json}: {len(seg)} segment ids for {n_points} points")
    if len(seg) and seg.min() < 0:
        raise ValueError(f"{segs_json}: negative segment id")
    return seg


def segment_pool(masks, seg):  # [inferred]
    """Official get_full_res_mask pooling on (N, K) masks: per-segment mean
    (torch_scatter.scatter_mean over dim 0), strict > 0.5, broadcast back to points."""
    masks = np.asarray(masks, dtype=np.float64)
    if masks.ndim != 2 or masks.shape[0] != len(seg):
        raise ValueError(f"masks {masks.shape} do not match {len(seg)} segment ids")
    _, inverse = np.unique(seg, return_inverse=True)
    sums = np.zeros((inverse.max() + 1 if len(inverse) else 0, masks.shape[1]))
    np.add.at(sums, inverse, masks)
    counts = np.bincount(inverse)[:, None]
    pooled = (sums / counts) > 0.5
    return pooled[inverse].astype(np.float32)


def segment_violations(masks, seg):  # [inferred]
    """Number of (segment, mask) pairs whose points disagree; 0 after segment_pool."""
    masks = np.asarray(masks) > 0.5
    _, inverse = np.unique(seg, return_inverse=True)
    n_seg = inverse.max() + 1 if len(inverse) else 0
    on = np.zeros((n_seg, masks.shape[1]), dtype=np.int64)
    np.add.at(on, inverse, masks.astype(np.int64))
    counts = np.bincount(inverse, minlength=n_seg)[:, None]
    return int(np.sum((on > 0) & (on < counts)))


def _segment_snap(masks_path, seg, out_path):  # [inferred]
    """Pool Mask3D masks onto ScanNet segments and save as <..>_masks.pt for SAM/CLIP."""
    masks = np.asarray(torch.load(masks_path))
    pooled = segment_pool(masks, seg)
    violations = segment_violations(pooled, seg)
    if violations:
        raise RuntimeError(f"segment pooling left {violations} segment violations")
    features_path(os.path.dirname(out_path), out_path)  # enforce the _masks.pt contract
    torch.save(pooled.astype(np.float16), out_path)
    print(f"[INFO] segment pooling: {masks.shape[0]} points / {len(np.unique(seg))} segments / "
          f"{pooled.shape[1]} masks, segment violations 0 -> {out_path}")
    return out_path


def _clip_vocab(spec):
    if spec.name != "ScanNet200":
        return list(spec.class_labels)
    import importlib.util
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    path = os.path.join(repo, "BenchmarkScripts", "ScanNet200", "scannet200_constants.py")
    spec_mod = importlib.util.spec_from_file_location("scannet200_constants", path)
    module = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(module)
    return list(module.CLASS_LABELS_200)


def _classify(masks, feats, classes, device, min_mask_points=MIN_MASK_POINTS,
              clip_prompt=CLIP_PROMPT):
    model, _ = clip.load("ViT-L/14@336px", device=device)
    with torch.no_grad():
        texts = [clip_prompt.format(c) for c in classes]
        text_ft = model.encode_text(clip.tokenize(texts).to(device))
        text_ft = (text_ft / text_ft.norm(dim=-1, keepdim=True)).cpu().numpy()

    instances = []
    skipped = 0
    for mi in tqdm(range(masks.shape[1]), desc="classifying masks", unit="mask"):
        sel = masks[:, mi] > 0.5
        if sel.sum() < min_mask_points:
            continue
        norm = np.linalg.norm(feats[mi])
        if norm < 1e-6:
            continue
        unit = feats[mi] / norm
        sims = unit @ text_ft.T
        best = int(np.argmax(sims))
        name = classes[best]
        if name in ("wall", "floor"):
            skipped += 1
            continue
        instances.append((sel, name, 1.0, unit))
    print(f"[INFO] skipped {skipped} wall/floor CLIP assignments")
    return instances


def _dedup_instances(instances, iou_threshold=DEDUP_IOU):
    """Greedily keeps the highest-confidence mask within each same-class overlapping
    cluster (overlap = intersection over the SMALLER mask, so a small spurious proposal
    nested inside a larger correct one still counts as a duplicate)."""
    by_class = {}
    for i, inst in enumerate(instances):
        by_class.setdefault(inst[1], []).append(i)

    keep = [False] * len(instances)
    for idxs in by_class.values():
        idxs.sort(key=lambda i: -instances[i][2])
        kept_masks = []
        for i in idxs:
            mask = instances[i][0]
            mask_size = mask.sum()
            if any(mask_size and kept.sum() and
                   np.logical_and(mask, kept).sum() / min(mask_size, kept.sum()) > iou_threshold
                   for kept in kept_masks):
                continue
            kept_masks.append(mask)
            keep[i] = True
    return [inst for inst, k in zip(instances, keep) if k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True, help="ScanNet scene pointcloud (.ply)")
    ap.add_argument("--frames", required=True, help="extracted frames dir (frames.py)")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="predictions output dir")
    ap.add_argument("--benchmark", default="ScanNet20",
                    choices=["ScanNet20", "ScanNet200"])
    add_run_args(ap)
    args = ap.parse_args()
    spec = _benchmark_spec(args.benchmark)
    scene_id = scene_id_from_pointcloud(args.pointcloud)
    params = load_overrides(args.parameters_json, {
        "point_limit", "min_mask_points", "num_queries", "dbscan_min_points",
        "frequency", "clip_prompt", "dedup_iou"})
    point_limit = int(params.get("point_limit", POINT_LIMIT))
    min_mask_points = int(params.get("min_mask_points", MIN_MASK_POINTS))
    num_queries = int(params.get("num_queries", NUM_QUERIES))
    dbscan_min_points = int(params.get("dbscan_min_points", DBSCAN_MIN_POINTS))
    frequency = int(params.get("frequency", FREQUENCY))
    clip_prompt = str(params.get("clip_prompt", CLIP_PROMPT))
    dedup_iou = float(params.get("dedup_iou", DEDUP_IOU))
    print(f"[INFO] {scene_id} run_id={args.run_id} overrides={params} "
          f"queries={num_queries} dbscan_min={dbscan_min_points} freq={frequency}")

    os.makedirs(args.out, exist_ok=True)
    scratch = os.path.join(SCRATCH_ROOT, args.run_id, scene_id)
    os.makedirs(scratch, exist_ok=True)

    pcd = o3d.io.read_point_cloud(args.pointcloud)
    full_pts = np.asarray(pcd.points)
    full_cols = np.asarray(pcd.colors)
    working_pts, nn_idx = decimate(full_pts, point_limit)
    if len(working_pts) != len(full_pts):
        raise RuntimeError(f"{scene_id}: segment pooling needs the full mesh, but point_limit "
                           f"{point_limit} decimated {len(full_pts)} -> {len(working_pts)}")
    seg = load_segments(segments_path(args.pointcloud), len(full_pts))
    if len(working_pts) < len(full_pts):
        working_cols = full_cols[cKDTree(full_pts).query(working_pts, k=1, workers=-1)[1]]
    else:
        working_cols = full_cols

    scene_ply = os.path.join(scratch, "working_scene.ply")
    working_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(working_pts))
    working_pcd.colors = o3d.utility.Vector3dVector(working_cols)
    o3d.io.write_point_cloud(scene_ply, working_pcd)

    raw_masks_path = _run_mask_computation(scene_ply, scratch, num_queries, dbscan_min_points)
    masks_path = _segment_snap(raw_masks_path, seg, os.path.join(scratch, POOLED_MASKS_NAME))

    features_path = _run_feature_computation(scene_ply, masks_path, args.frames,
                                             scratch, frequency)

    masks = np.asarray(torch.load(masks_path))
    feats = np.load(features_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_classes = _clip_vocab(spec)
    instances = _classify(masks, feats, clip_classes, device, min_mask_points, clip_prompt)
    deduped = _dedup_instances(instances, iou_threshold=dedup_iou)

    candidates = []
    feature_rows = []
    for item in deduped:
        sel_decimated, class_name, confidence, feat = item
        candidates.append((sel_decimated[nn_idx], class_name, confidence))
        feature_rows.append(feat)

    rows = write_scannet_submission_rows(args.out, scene_id, args.classes, candidates,
                                        min_mask_points, spec)
    publish_clip_features(
        args.features_out, feature_rows, rows, spec, args.run_id, "openmask3d", scene_id,
        "openai_clip", "ViT-L/14@336px", 768)
    print(f"[INFO] {len(instances)} raw instances -> {len(deduped)} after IoU-overlap "
          f"dedup, {rows['n_written']} written to {args.out}")


if __name__ == "__main__":
    main()