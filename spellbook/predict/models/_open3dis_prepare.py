"""Per-scene Open3DIS ScanNet200 data prep, run in the `open3dis` conda env.

Spellbook-side replacement of the lost fork script `scripts/prepare_scannet200_scene.py`
(`459aa14`). It builds the official docs/DATA.md tree for ONE scene inside the per-run scratch
(see _open3dis_config.scene_layout), never inside /data/Open3DIS:

    Scannet200_2D_5interval/val/<scene>/{color,depth,pose}/NNNNN.* + intrinsic.txt
        symlinks to every 5th frame of <scans>/<scene>/frames (frames.py, SensReader export)
    Scannet200_3D/val/original_ply_files/<scene>.ply      symlink to <scene>_vh_clean_2.ply
    Scannet200_3D/val/superpoints/<scene>.pth             ScanNet Segmentator segIndices
    Scannet200_3D/val/groundtruth/<scene>_inst_nostuff.pth ISBNet dataset/scannet200 preprocess
    Scannet200_3D/val/{isbnet_clsagnostic,dc_feat}_scannet200/<scene>.pth
        ISBNet head_scannetv2_200_val.pth inference (ISBNet tools/test.py model path, whose
        forward_test writes both to ../../data/... relative to its cwd)

Usage: python _open3dis_prepare.py --scene-dir <scans>/<scene> --frames <frames> --scratch <dir>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _open3dis_config import (  # noqa: E402
    OPEN3DIS_REPO, derive_isbnet_config, load_yaml, sampled_frames, scene_layout, write_yaml,
)

ISBNET_DIR = os.path.join(OPEN3DIS_REPO, "segmenter3d", "ISBNet")
ISBNET_BASE_CONFIG = os.path.join(ISBNET_DIR, "configs", "scannet200", "isbnet_scannet200.yaml")
ISBNET_CKPT = "/data/pretrains/scannet200/head_scannetv2_200_val.pth"
LABEL_MAP = "/data/scannetdata/v2/scannetv2-labels.combined.tsv"


def _symlink(target, link):
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(target, link)


def link_frames(frames_dir, scene2d):
    """Official 5-interval 2D layout as symlinks; names keep the original frame index."""
    ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(frames_dir, "depth"))]
    for sub in ("color", "depth", "pose"):
        os.makedirs(os.path.join(scene2d, sub), exist_ok=True)
    for src, name in sampled_frames(ids):
        for sub, ext in (("color", ".jpg"), ("depth", ".png"), ("pose", ".txt")):
            _symlink(os.path.join(frames_dir, sub, f"{src}{ext}"),
                     os.path.join(scene2d, sub, name + ext))
    _symlink(os.path.join(frames_dir, "intrinsic_color.txt"),
             os.path.join(scene2d, "intrinsic.txt"))


def write_superpoints(scene_dir, scene_id, out_path):
    """ISBNet dataset/scannetv2/prepare_superpoint.py runs segmentator.segment_mesh on
    <scene>_vh_clean_2.ply with its defaults (kThresh 0.01, segMinVerts 20). ScanNet ships that
    exact Segmentator output as <scene>_vh_clean_2.0.010000.segs.json, so it is reused."""
    # [inferred] the fork vendored segmentator; ScanNet's own segs.json is the same algorithm
    import numpy as np
    import torch
    with open(os.path.join(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json")) as f:
        spp = np.asarray(json.load(f)["segIndices"], dtype=np.int64)
    torch.save(spp, out_path)


def write_groundtruth(scene_dir, scene_id, layout):
    """ISBNet's official preprocess_scannet200.handle_process (axis-aligned xyz, rgb/127.5-1,
    ScanNet200 labels). It writes <output>/val/<scene>_inst_nostuff.pth, so it is pointed at
    the 3D root with `groundtruth` as its val folder name via a staging dir."""
    import pandas as pd
    sys.path.insert(0, os.path.join(ISBNET_DIR, "dataset", "scannet200"))
    import preprocess_scannet200  # noqa: E402  (saves reverse_norm_ids.npy into cwd)
    staging = os.path.join(layout["scratch"], "isbnet", "staging")
    os.makedirs(os.path.join(staging, "val"), exist_ok=True)
    labels_pd = pd.read_csv(LABEL_MAP, sep="\t", header=0)
    preprocess_scannet200.handle_process(scene_dir, staging, labels_pd, [], [scene_id])
    name = f"{scene_id}_inst_nostuff.pth"
    os.replace(os.path.join(staging, "val", name), os.path.join(layout["groundtruth"], name))


def run_isbnet(layout):
    """ISBNet tools/test.py model path (build, load ckpt, forward over the one-scene test set)
    without its evaluation/saving tail; forward_test itself writes dc_feat + class-agnostic
    proposals relative to cwd = layout['isbnet_cwd']."""
    import torch
    from isbnet.data import build_dataloader, build_dataset
    from isbnet.model import ISBNet
    from isbnet.util import get_root_logger, load_checkpoint
    from munch import Munch

    cfg = Munch.fromDict(load_yaml(layout["isbnet_config"]))
    logger = get_root_logger()
    model = ISBNet(**cfg.model, dataset_name=cfg.data.train.type).cuda()
    load_checkpoint(ISBNET_CKPT, logger, model)
    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=False, **cfg.dataloader.test)
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                model(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, help="<scannet_root>/scans/<scene>")
    ap.add_argument("--frames", required=True, help="extracted frames dir (frames.py)")
    ap.add_argument("--scratch", required=True, help="per-run per-scene scratch dir")
    args = ap.parse_args()

    scene_id = os.path.basename(os.path.normpath(args.scene_dir))
    layout = scene_layout(args.scratch, scene_id)
    for key in ("scene2d", "original_ply", "groundtruth", "superpoints", "proposals3d",
                "dc_feat", "isbnet_cwd"):
        os.makedirs(layout[key], exist_ok=True)
    os.chdir(layout["isbnet_cwd"])

    link_frames(args.frames, layout["scene2d"])
    _symlink(os.path.join(os.path.abspath(args.scene_dir), f"{scene_id}_vh_clean_2.ply"),
             os.path.join(layout["original_ply"], f"{scene_id}.ply"))

    spp_path = os.path.join(layout["superpoints"], f"{scene_id}.pth")
    if not os.path.isfile(spp_path):
        write_superpoints(args.scene_dir, scene_id, spp_path)
    if not os.path.isfile(os.path.join(layout["groundtruth"], f"{scene_id}_inst_nostuff.pth")):
        write_groundtruth(args.scene_dir, scene_id, layout)

    done = all(os.path.isfile(os.path.join(layout[k], f"{scene_id}.pth"))
               for k in ("proposals3d", "dc_feat"))
    if not done:
        write_yaml(derive_isbnet_config(load_yaml(ISBNET_BASE_CONFIG), layout),
                   layout["isbnet_config"])
        run_isbnet(layout)
    print(f"[INFO] open3dis data prepared for {scene_id} in {layout['scratch']}")


if __name__ == "__main__":
    main()
