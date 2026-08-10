"""Runs Open3DIS's 3-stage pipeline (2D grounding -> hierarchical-agglomerative clustering ->
CLIP feature refinement) inside the `open3dis` conda env, then does CLIP classification from the
refined per-instance features and writes ScanNet benchmark instance-segmentation format.

Uses Open3DIS's own generic `Ov3disSceneReader` (open3dis/dataset/ov3dis_loader.py, templated
off configs/ov3dis_scene4.yaml) with proposals.p2d:True/p3d:False -- 2D-mask-guided clustering
only, no superpoints/3D proposals needed. This script adapts the proven ov3dis-comparison
wrapper (utils/_open3dis_run.py) to ScanNet's own data layout:
    --frames points at /data/scannet/scans/<scene>/frames/ (see frames.py)
    scene_id comes from the scene directory name (common.scene_id_from_pointcloud)

The vendored Open3DIS checkout at /home/rolf/GIT/Open3DIS already carries the local patches
this code path needs (verified: Ov3disSceneReader present, generate_3d_inst.py's get_final_
instances() gated behind `if False`, tracker_*.txt skip-lists present).
"""
import os
import sys

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

import argparse
import pathlib
import subprocess

import clip
import cv2
import numpy as np
import open3d as o3d
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from common import _benchmark_spec, scene_id_from_pointcloud, write_scannet_submission  # noqa: E402

OPEN3DIS_REPO = "/home/rolf/GIT/Open3DIS"
OPEN3DIS_PY = "/data/open3dis/conda/envs/open3dis/bin/python"
SCRATCH_ROOT = "/data/open3dis/scratch"  # transient split/config files -- kept off the submission root

MIN_MASK_POINTS = 20
DECIMATE_LIMIT = 700_000
FINAL_INSTANCE_TOP_K = 300

GENERIC_TEMPLATE_CONFIG = os.path.join(OPEN3DIS_REPO, "configs", "ov3dis_scene4.yaml")


def _rle_decode(rle):
    """Identical to generate_3d_inst.py's own rle_decode -- inlined since it's the only piece
    of that heavy __main__-oriented module this wrapper needs."""
    length = rle["length"]
    s = rle["counts"]
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def _link_frames(frames_dir, scene_2d_dir):
    """(Re)symlinks this scene's color/depth/pose/intrinsic_depth.txt into Open3DIS's own data
    tree -- Ov3disSceneReader reads them from there, under cfg.data.datapath/<scene_id>."""
    os.makedirs(scene_2d_dir, exist_ok=True)
    links = {
        "color": os.path.join(frames_dir, "color"),
        "depth": os.path.join(frames_dir, "depth"),
        "pose": os.path.join(frames_dir, "pose"),
        "intrinsic.txt": os.path.join(frames_dir, "intrinsic_depth.txt"),
    }
    for name, target in links.items():
        link = os.path.join(scene_2d_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)


def _frame_img_dim(frames_dir):
    """[width, height] straight from this scene's own first frame. ScanNet's color (1296x968) and
    depth (640x480) resolutions differ, so each is read separately: img_dim is the DEPTH resolution
    (projection/depth alignment), rgb_img_dim the COLOR resolution (2D masks) -- matching Open3DIS's
    own configs/scannet200.yaml."""
    color_dir = os.path.join(frames_dir, "color")
    first_color = sorted(os.listdir(color_dir), key=lambda f: int(os.path.splitext(f)[0]))[0]
    h, w = cv2.imread(os.path.join(color_dir, first_color)).shape[:2]
    rgb_img_dim = [w, h]
    depth_dir = os.path.join(frames_dir, "depth")
    first_depth = sorted(os.listdir(depth_dir), key=lambda f: int(os.path.splitext(f)[0]))[0]
    dh, dw = cv2.imread(os.path.join(depth_dir, first_depth), -1).shape[:2]
    img_dim = [dw, dh]
    return img_dim, rgb_img_dim


def _ensure_working_ply(pointcloud_path, working_ply, limit):
    """Symlinks --pointcloud into Open3DIS's data tree if small enough, else builds (and caches)
    a voxel-decimated copy. Returns whether decimation was applied."""
    os.makedirs(os.path.dirname(working_ply), exist_ok=True)
    pcd = o3d.io.read_point_cloud(pointcloud_path)

    if len(pcd.points) <= limit:
        if os.path.islink(working_ply) or os.path.exists(working_ply):
            os.remove(working_ply)
        os.symlink(os.path.abspath(pointcloud_path), working_ply)
        return False

    if (os.path.exists(working_ply) and not os.path.islink(working_ply)
            and os.path.getmtime(working_ply) >= os.path.getmtime(pointcloud_path)):
        return True

    voxel_size = 0.01
    down = pcd
    while len(down.points) > limit:
        voxel_size *= 1.4
        down = pcd.voxel_down_sample(voxel_size)
    if os.path.islink(working_ply):
        os.remove(working_ply)
    o3d.io.write_point_cloud(working_ply, down)
    return True


def _make_run_config(base_config, data_overrides, exp_name, classes, out_dir):
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"].update(data_overrides)
    cfg["data"]["custom_classes"] = list(classes)
    cfg["data"]["num_classes"] = len(classes)
    cfg["exp"]["exp_name"] = exp_name
    run_config = os.path.join(out_dir, "open3dis_run.yaml")
    with open(run_config, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return run_config


def _run_stage(script, config, env):
    subprocess.run([OPEN3DIS_PY, script, "--config", config], cwd=OPEN3DIS_REPO, env=env, check=True)


def _classify(inst_feat, classes, device):
    """CLIP text-vs-instance-feature classification; inst_feat rows are already L2-normalized."""
    model, _ = clip.load("ViT-L/14@336px", device=device)
    with torch.no_grad():
        text_ft = model.encode_text(clip.tokenize([f"a photo of a {c}." for c in classes]).to(device))
        text_ft = (text_ft / text_ft.norm(dim=-1, keepdim=True)).cpu().numpy()

    sims = inst_feat.numpy() @ text_ft.T
    best = sims.argmax(1)
    confidence = (sims[np.arange(len(best)), best] + 1) / 2
    return best, confidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True, help="scene pointcloud (.ply, xyz+rgb)")
    ap.add_argument("--frames", required=True, help="extracted frames dir (frames.py)")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="predictions output dir")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--benchmark", default="ScanNet20",
                    choices=["ScanNet20", "ScanNet200"])
    args = ap.parse_args()
    spec = _benchmark_spec(args.benchmark)

    os.makedirs(args.out, exist_ok=True)
    scene_id = scene_id_from_pointcloud(args.pointcloud)

    scene_2d_dir = os.path.join(OPEN3DIS_REPO, "data", "ov3dis_scene", "ov3dis_scene_2d", scene_id)
    working_ply = os.path.join(OPEN3DIS_REPO, "data", "ov3dis_scene", "original_ply", f"{scene_id}.ply")
    _link_frames(args.frames, scene_2d_dir)
    decimated = _ensure_working_ply(args.pointcloud, working_ply, DECIMATE_LIMIT)

    split_path = os.path.join(SCRATCH_ROOT, scene_id, "open3dis_split.txt")
    with open(split_path, "w") as f:
        f.write(scene_id + "\n")

    img_dim, rgb_img_dim = _frame_img_dim(args.frames)
    exp_name = f"{scene_id}_ov3discomp"
    run_config = _make_run_config(
        GENERIC_TEMPLATE_CONFIG,
        dict(split_path=split_path, img_dim=img_dim, rgb_img_dim=rgb_img_dim),
        exp_name, args.classes, os.path.join(SCRATCH_ROOT, scene_id),
    )
    exp_dir = os.path.join(OPEN3DIS_REPO, "exp", exp_name)

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONPATH"] = OPEN3DIS_REPO + ":" + env.get("PYTHONPATH", "")
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    # fresh run every invocation: each tools/*.py script's own tracker_*.txt skip-list would
    # otherwise silently no-op a scene it's already seen.
    for tracker in ("tracker_2d.txt", "tracker_lifted.txt", "tracker_refine.txt"):
        open(os.path.join(OPEN3DIS_REPO, tracker), "w").close()

    _run_stage("tools/grounding_2d.py", run_config, env)
    _run_stage("tools/generate_3d_inst.py", run_config, env)
    _run_stage("tools/refine_grounding_feat.py", run_config, env)

    clustered = torch.load(os.path.join(exp_dir, "hier_agglo", f"{scene_id}.pth"))
    refined = torch.load(os.path.join(exp_dir, "refined_grounded_feat", f"{scene_id}.pth"))
    masks_rle = clustered["ins"]
    inst_feat = refined["inst_feat"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_idx, confidence = _classify(inst_feat, args.classes, device)

    nn_idx = None
    if decimated:
        full_pts = np.asarray(o3d.io.read_point_cloud(args.pointcloud).points)
        working_pts = np.asarray(o3d.io.read_point_cloud(working_ply).points)
        nn_idx = cKDTree(working_pts).query(full_pts, k=1, workers=-1)[1]

    def _instances():
        for i, rle in enumerate(masks_rle):
            if np.linalg.norm(inst_feat[i].numpy()) < 1e-6:
                continue
            sel = _rle_decode(rle).astype(bool)
            if nn_idx is not None:
                sel = sel[nn_idx]
            yield sel, args.classes[best_idx[i]], confidence[i]

    candidates = sorted(_instances(), key=lambda t: -t[2])[:FINAL_INSTANCE_TOP_K]
    n_written = write_scannet_submission(args.out, scene_id, args.classes, candidates,
                                         MIN_MASK_POINTS, spec)
    print(f"[INFO] Wrote {n_written} instances to {args.out}")


if __name__ == "__main__":
    main()
