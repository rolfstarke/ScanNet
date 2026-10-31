"""Runs the official Open3DIS ScanNet200 2D+3D pipeline (configs/scannet200.yaml, #60) for one
scene inside the `open3dis` conda env and writes ScanNet benchmark instance-segmentation format.

Stages (docs/RUN.md), every tool with cwd = <scratch>/cwd so each run/scene owns its
tracker_{2d,lifted,refine,final}.txt (#15); stale tracker entries are dropped first (0ca2762):
    0. _open3dis_prepare.py      per-scene data tree + ISBNet head_scannetv2_200_val.pth
                                 class-agnostic proposals / dc_feat (lost fork 459aa14)
    1. grounding                 Grounded-SAM (GroundingDINO SwinT_OGC 0.4/0.4 + SAM ViT-H),
                                 CLIP ViT-L/14@336px 768-d point features
    2. generate                  hierarchical agglomerative clustering -> hier_agglo
    3. tools/refine_grounding_feat.py (upstream, top_k 5) -> refined_grounded_feat
    4. generate (final config)   skip_clustering + emit_final, proposals p2d/p3d/refined,
                                 agnostic False -> final_result_hier_agglo (official 2D+3D
                                 fusion with mask-wise CLIP scoring, final_instance defaults)
The submission is final_result_hier_agglo/<scene>.pth as-is ({ins, conf, final_class}); no own
CLIP re-classification. Scratch: <scannet_root>/derived/open3dis/<run-id>/<scene>/ (#24).
"""
import os
import sys

import argparse
import subprocess

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    _benchmark_spec, add_run_args, load_overrides, publish_clip_features,
    scene_id_from_pointcloud, write_scannet_submission_rows,
)
from _open3dis_config import (  # noqa: E402
    BASE_CONFIG, OPEN3DIS_REPO, derive_config, ensure_tracker, load_yaml,
    output_path, reconcile_tracker, scannet_root_from_pointcloud, scene_layout, scratch_dir,
    write_yaml,
)

OPEN3DIS_PY = "/data/open3dis/conda/envs/open3dis/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
PREPARE_SCRIPT = os.path.join(HERE, "_open3dis_prepare.py")
STAGE_SCRIPT = os.path.join(HERE, "_open3dis_stage.py")
REFINE_SCRIPT = os.path.join(OPEN3DIS_REPO, "tools", "refine_grounding_feat.py")

MIN_MASK_POINTS = 20


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


def _frame_img_dim(frames_dir):
    """[width, height] straight from this scene's own first frame. ScanNet's color (1296x968) and
    depth (640x480) resolutions differ, so each is read separately: img_dim is the DEPTH resolution
    (projection/depth alignment), rgb_img_dim the COLOR resolution (2D masks) -- matching Open3DIS's
    own configs/scannet200.yaml (#3)."""
    import cv2
    color = cv2.imread(os.path.join(frames_dir, "color", "0.jpg"))
    depth = cv2.imread(os.path.join(frames_dir, "depth", "0.png"), -1)
    return [depth.shape[1], depth.shape[0]], [color.shape[1], color.shape[0]]


def _child_env():
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONPATH"] = OPEN3DIS_REPO + ":" + env.get("PYTHONPATH", "")
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    return env


def _run(args, cwd, env):
    fd = os.environ.get("SPELLBOOK_GPU_LEASE_FD")
    pass_fds = (int(fd),) if fd else ()
    print(f"[INFO] open3dis: {' '.join(os.path.basename(a) for a in args[:2])}", flush=True)
    subprocess.run([OPEN3DIS_PY, *args], cwd=cwd, env=env, check=True, pass_fds=pass_fds)


def _run_tool(args, cfg, tracker, layout, env):
    """One upstream tool step in the private cwd, after dropping stale tracker entries."""
    path = os.path.join(layout["cwd"], tracker)
    ensure_tracker(path)
    dropped = reconcile_tracker(path, cfg)
    if dropped:
        print(f"[INFO] open3dis: dropped stale {tracker} entries {dropped}")
    _run(args, layout["cwd"], env)


def _instance_features(pc_feat_path, masks):
    """Per-instance CLIP feature for the query sidecar: mean of the L2-normalized point
    features over the mask, renormalized."""  # [inferred] sidecar convention, not scoring
    feat = torch.load(pc_feat_path, map_location="cpu")["feat"].float()
    feat = torch.nn.functional.normalize(feat, dim=1)
    rows = []
    for mask in masks:
        idx = torch.from_numpy(np.flatnonzero(mask))
        v = feat[idx].mean(0) if len(idx) else torch.zeros(feat.shape[1])
        rows.append(torch.nn.functional.normalize(v, dim=0).numpy())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True, help="scene pointcloud (<scene>_vh_clean_2.ply)")
    ap.add_argument("--frames", required=True, help="extracted frames dir (frames.py)")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="predictions output dir")
    ap.add_argument("--benchmark", default="ScanNet200",
                    choices=["ScanNet20", "ScanNet200"])
    add_run_args(ap)
    args = ap.parse_args()
    spec = _benchmark_spec(args.benchmark)
    params = load_overrides(args.parameters_json, {"min_mask_points"})
    min_mask_points = int(params.get("min_mask_points", MIN_MASK_POINTS))

    sys.path.insert(0, OPEN3DIS_REPO)
    from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
    class_names = list(INSTANCE_CAT_SCANNET_200)
    if list(args.classes) != class_names:
        raise ValueError("open3dis scannet200 path scores the official 198 ScanNet200 classes "
                         "(INSTANCE_CAT_SCANNET_200); --benchmark ScanNet200 required")

    os.makedirs(args.out, exist_ok=True)
    scene_id = scene_id_from_pointcloud(args.pointcloud)
    scannet_root = scannet_root_from_pointcloud(args.pointcloud)
    scratch = scratch_dir(scannet_root, args.run_id, scene_id)
    layout = scene_layout(scratch, scene_id)
    os.makedirs(layout["cwd"], exist_ok=True)
    os.makedirs(layout["isbnet_cwd"], exist_ok=True)
    print(f"[INFO] {scene_id} run_id={args.run_id} scratch={scratch} overrides={params}")

    # #24: parent dir exists before the split file is written
    os.makedirs(os.path.dirname(layout["split"]), exist_ok=True)
    with open(layout["split"], "w") as f:
        f.write(scene_id + "\n")

    img_dim, rgb_img_dim = _frame_img_dim(args.frames)
    base = load_yaml(BASE_CONFIG)
    cfg = derive_config(base, layout, scene_id, args.run_id, img_dim, rgb_img_dim,
                        num_classes=len(class_names))
    cfg_final = derive_config(base, layout, scene_id, args.run_id, img_dim, rgb_img_dim,
                              num_classes=len(class_names), final=True)
    write_yaml(cfg, layout["config"])
    write_yaml(cfg_final, layout["config_final"])

    env = _child_env()
    _run([PREPARE_SCRIPT, "--scene-dir", os.path.dirname(os.path.abspath(args.pointcloud)),
          "--frames", args.frames, "--scratch", scratch], layout["isbnet_cwd"], env)
    _run_tool([STAGE_SCRIPT, "grounding", "--config", layout["config"]],
              cfg, "tracker_2d.txt", layout, env)
    _run_tool([STAGE_SCRIPT, "generate", "--config", layout["config"]],
              cfg, "tracker_lifted.txt", layout, env)
    _run_tool([REFINE_SCRIPT, "--config", layout["config"]],
              cfg, "tracker_refine.txt", layout, env)
    _run_tool([STAGE_SCRIPT, "generate", "--config", layout["config_final"]],
              cfg_final, "tracker_final.txt", layout, env)

    final = torch.load(output_path(cfg_final, "final_output", scene_id), map_location="cpu")
    masks = [_rle_decode(rle).astype(bool) for rle in final["ins"]]
    conf = np.asarray(final["conf"], dtype=np.float64).reshape(-1)
    cls = np.asarray(final["final_class"]).reshape(-1).astype(int)
    print(f"[INFO] {scene_id}: {len(masks)} fused instances from final_result_hier_agglo")

    instances = [(m, class_names[c], float(s)) for m, c, s in zip(masks, cls, conf)]
    rows = write_scannet_submission_rows(args.out, scene_id, class_names, instances,
                                        min_mask_points, spec)
    if args.features_out:
        pc_feat = output_path(cfg_final, "refined_grounded_feat_output", scene_id)
        if not os.path.isfile(pc_feat):
            pc_feat = output_path(cfg_final, "grounded_feat_output", scene_id)
        feature_rows = _instance_features(pc_feat, masks)
        publish_clip_features(
            args.features_out, feature_rows, rows, spec, args.run_id, "open3dis", scene_id,
            "openai_clip", "ViT-L/14@336px", 768)
    print(f"[INFO] Wrote {rows['n_written']} instances to {args.out}")


if __name__ == "__main__":
    main()
