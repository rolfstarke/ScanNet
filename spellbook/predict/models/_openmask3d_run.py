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
"""
import argparse
import glob
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

OPENMASK3D_REPO = "/home/rolf/GIT/openmask3d"
CHECKPOINT = "/data/openmask3d/resources/scannet200_model.ckpt"
SAM_CHECKPOINT = "/data/openmask3d/resources/sam_vit_h_4b8939.pth"

POINT_LIMIT = 500_000  # Mask3D mask computation OOMs on a 16GB card above ~900k points (empirical)
MIN_MASK_POINTS = 20
DEDUP_IOU = 0.5
SCRATCH_ROOT = "/data/openmask3d/scratch"


def _lease_fd():
    fd = os.environ.get("SPELLBOOK_GPU_LEASE_FD")
    return (int(fd),) if fd else ()


def _run_mask_computation(scene_ply, out_dir):
    args = [
        sys.executable, "class_agnostic_mask_computation/get_masks_single_scene.py",
        "general.experiment_name=openmask3d_util",
        f"general.checkpoint={CHECKPOINT}",
        "general.train_mode=false",
        "data.test_mode=test",
        "model.num_queries=120",
        "general.use_dbscan=true",
        "general.dbscan_eps=0.95",
        "general.dbscan_min_points=10",
        "general.save_visualizations=false",
        f"general.scene_path={scene_ply}",
        f"general.mask_save_dir={out_dir}",
        f"hydra.run.dir={out_dir}/hydra_outputs/class_agnostic_mask_computation",
    ]
    subprocess.run(args, cwd=os.path.join(OPENMASK3D_REPO, "openmask3d"),
                   env=os.environ.copy(), check=True, pass_fds=_lease_fd())
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
    scene_name = os.path.basename(scene_ply)[:-4]
    return os.path.join(out_dir, f"{scene_name}_openmask3d_features.npy")


def _classify(masks, feats, classes, device, min_mask_points=MIN_MASK_POINTS):
    model, _ = clip.load("ViT-L/14@336px", device=device)
    with torch.no_grad():
        text_ft = model.encode_text(clip.tokenize([f"a photo of a {c}." for c in classes]).to(device))
        text_ft = (text_ft / text_ft.norm(dim=-1, keepdim=True)).cpu().numpy()

    instances = []
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
        instances.append((sel, classes[best], (float(sims[best]) + 1) / 2, unit))
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
        "point_limit", "min_mask_points", "dedup_iou"})
    point_limit = int(params.get("point_limit", POINT_LIMIT))
    min_mask_points = int(params.get("min_mask_points", MIN_MASK_POINTS))
    dedup_iou = float(params.get("dedup_iou", DEDUP_IOU))
    print(f"[INFO] {scene_id} run_id={args.run_id} overrides={params}")

    os.makedirs(args.out, exist_ok=True)
    scratch = os.path.join(SCRATCH_ROOT, args.run_id, scene_id)
    os.makedirs(scratch, exist_ok=True)

    pcd = o3d.io.read_point_cloud(args.pointcloud)
    full_pts = np.asarray(pcd.points)
    full_cols = np.asarray(pcd.colors)
    working_pts, nn_idx = decimate(full_pts, point_limit)
    if len(working_pts) < len(full_pts):
        working_cols = full_cols[cKDTree(full_pts).query(working_pts, k=1, workers=-1)[1]]
    else:
        working_cols = full_cols

    scene_ply = os.path.join(scratch, "working_scene.ply")
    working_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(working_pts))
    working_pcd.colors = o3d.utility.Vector3dVector(working_cols)
    o3d.io.write_point_cloud(scene_ply, working_pcd)

    masks_path = _run_mask_computation(scene_ply, scratch)

    total_frames = len(glob.glob(os.path.join(args.frames, "pose", "*.txt")))
    frequency = max(1, total_frames // 400)
    features_path = _run_feature_computation(scene_ply, masks_path, args.frames,
                                             scratch, frequency)

    masks = np.asarray(torch.load(masks_path))
    feats = np.load(features_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instances = _classify(masks, feats, args.classes, device, min_mask_points)
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