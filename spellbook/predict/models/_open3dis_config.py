"""Pure-Python config/layout helpers for the Open3DIS wrapper (_open3dis_run.py).

No torch/CUDA imports, so CPU unit tests can exercise them from the spellbook env. Restores
what the lost `spellbook-baseline` fork patches did, without touching the upstream checkout:

- Per-run, per-scene scratch tree under <scannet_root>/derived/open3dis/<run-id>/<scene>/
  (#15, #24): Open3DIS data tree for one scene, the exp/ outputs, derived configs, and a
  private tool cwd/ holding the tracker_*.txt skip-lists. Nothing is written into the repo.
- Derived configs from the official configs/scannet200.yaml (2D+3D path, #60) instead of
  editing configs/ in the repo. Relative upstream paths are made absolute, so tools can run
  from the scratch cwd.
- The two upstream generate_3d_inst.py gates (`if True:` clustering, `if False:` final
  instances) become config keys exp.skip_clustering / exp.emit_final (fork `12e05a8`),
  defaulting to the upstream behaviour (False / False). Read by _open3dis_stage.py.
- Tracker reconciliation (fork `0ca2762`): drop tracker lines whose stage output is missing,
  so a killed run never makes a later invocation skip a scene.
"""
import copy
import os

import yaml

OPEN3DIS_REPO = "/data/Open3DIS"
BASE_CONFIG = os.path.join(OPEN3DIS_REPO, "configs", "scannet200.yaml")

# official Scannet200_2D_5interval sampling (docs/DATA.md) -- frames.py extracts every frame
FRAME_INTERVAL = 5  # [inferred] the lost prepare script's 2D sampling is unrecorded
FRAME_NAME_DIGITS = 5  # docs/DATA.md: 00000.jpg -- zero-padded so the loader's sort is numeric

# 2D Grounded-SAM masks; the official default folder name is 'maskyoloworldsam'. Only a folder
# name -- the 2D replay worker reads <exp>/maskGdino/<scene>.pth (utils/replay2d.py).
MASK2D_OUTPUT = "maskGdino"  # [inferred]

# ScanNet depth 640x480 vs color 1296x968 (#3); same values as configs/scannet200.yaml
DEFAULT_IMG_DIM = [640, 480]
DEFAULT_RGB_IMG_DIM = [1296, 968]

# tracker file -> exp output key whose <scene>.pth proves the stage finished (0ca2762)
TRACKER_OUTPUTS = {
    "tracker_2d.txt": "mask2d_output",
    "tracker_lifted.txt": "clustering_3d_output",
    "tracker_refine.txt": "refined_grounded_feat_output",
    "tracker_final.txt": "final_output",  # [inferred] own tracker for the step-4 re-run
}


def scratch_dir(scannet_root, run_id, scene_id):
    """Per-run, per-scene Open3DIS scratch: everything transient lives here (#15, #24)."""
    return os.path.join(scannet_root, "derived", "open3dis", run_id, scene_id)


def scannet_root_from_pointcloud(pointcloud_path):
    """<scannet_root>/scans/<scene>/<scene>_vh_clean_2.ply -> <scannet_root>."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(pointcloud_path))))


def scene_layout(scratch, scene_id):
    """Official Open3DIS ScanNet200 data tree (docs/DATA.md) rooted in the scene scratch.
    ISBNet's forward_test hard-codes `../../data/Scannet200/Scannet200_3D/val/...` relative to
    its cwd, so `isbnet_cwd` sits exactly two levels below `scratch`."""
    data = os.path.join(scratch, "data", "Scannet200")
    val2d = os.path.join(data, "Scannet200_2D_5interval", "val")
    val3d = os.path.join(data, "Scannet200_3D", "val")
    return {
        "scratch": scratch,
        "val2d": val2d,
        "scene2d": os.path.join(val2d, scene_id),
        "val3d": val3d,
        "original_ply": os.path.join(val3d, "original_ply_files"),
        "groundtruth": os.path.join(val3d, "groundtruth"),
        "superpoints": os.path.join(val3d, "superpoints"),
        "proposals3d": os.path.join(val3d, "isbnet_clsagnostic_scannet200"),
        "dc_feat": os.path.join(val3d, "dc_feat_scannet200"),
        "exp": os.path.join(scratch, "exp"),
        "cwd": os.path.join(scratch, "cwd"),
        "isbnet_cwd": os.path.join(scratch, "isbnet", "cwd"),
        "split": os.path.join(scratch, "open3dis_split.txt"),
        "config": os.path.join(scratch, "open3dis_run.yaml"),
        "config_final": os.path.join(scratch, "open3dis_final.yaml"),
        "isbnet_config": os.path.join(scratch, "isbnet_scannet200.yaml"),
    }


def sampled_frames(frame_ids, interval=FRAME_INTERVAL):
    """[(src_id, padded_name)] for every `interval`-th frame of a 0..N-1 sequence."""
    ids = sorted(int(f) for f in frame_ids)
    return [(i, f"{i:0{FRAME_NAME_DIGITS}d}") for i in ids if i % interval == 0]


def _abs_repo_path(value, repo=OPEN3DIS_REPO):
    if isinstance(value, str) and (value.startswith("./") or value.startswith("../")):
        return os.path.normpath(os.path.join(repo, value))
    return value


def derive_config(base, layout, scene_id, exp_name, img_dim=None, rgb_img_dim=None,
                  num_classes=198, final=False, repo=OPEN3DIS_REPO):
    """Run config from the official scannet200.yaml dict. `final=True` gives the step-4
    fusion config (RUN.md: re-run generate_3d_inst after refinement): no re-clustering,
    emit final_result_hier_agglo with official mask-wise CLIP scoring (#60)."""
    cfg = copy.deepcopy(base)
    cfg["foundation_model"] = {k: _abs_repo_path(v, repo)
                               for k, v in cfg["foundation_model"].items()}
    cfg["data"].update(
        dataset_name="scannet200",
        split_path=layout["split"],
        datapath=layout["val2d"],
        gt_pth=layout["groundtruth"],
        original_ply=layout["original_ply"],
        spp_path=layout["superpoints"],
        cls_agnostic_3d_proposals_path=layout["proposals3d"],
        dc_features_path=layout["dc_feat"],
        img_dim=list(img_dim or DEFAULT_IMG_DIM),
        rgb_img_dim=list(rgb_img_dim or DEFAULT_RGB_IMG_DIM),
        num_classes=int(num_classes),
    )
    cfg["exp"].update(
        exp_name=exp_name,
        save_dir=layout["exp"],
        mask2d_output=MASK2D_OUTPUT,
        skip_clustering=bool(final),
        emit_final=bool(final),
    )
    cfg["proposals"] = dict(cfg["proposals"], p2d=True, p3d=True, refined=True)
    if final:
        cfg["proposals"]["agnostic"] = False
    cfg["evaluate"] = dict(evalvocab=False, evalagnostic=False)
    return cfg


def derive_isbnet_config(base, layout):
    """ISBNet configs/scannet200/isbnet_scannet200.yaml with absolute data roots: the per-scene
    3D val dir holding groundtruth/<scene>_inst_nostuff.pth + superpoints/<scene>.pth."""
    cfg = copy.deepcopy(base)
    for split in ("train", "test"):
        cfg["data"][split]["data_root"] = layout["val3d"]
    cfg["data"]["test"]["prefix"] = "groundtruth"
    cfg["dataloader"]["test"] = dict(batch_size=1, num_workers=1)
    return cfg


def output_path(cfg, key, scene_id):
    return os.path.join(cfg["exp"]["save_dir"], cfg["exp"]["exp_name"],
                        cfg["exp"][key], f"{scene_id}.pth")


def reconcile_tracker(tracker_path, cfg):
    """Drop tracker entries whose stage output .pth is missing (0ca2762): upstream tools append
    the scene before writing, so a killed run would otherwise skip it forever (#60 incident).
    Returns the dropped entries."""
    name = os.path.basename(tracker_path)
    key = TRACKER_OUTPUTS[name]
    if not os.path.isfile(tracker_path):
        return []
    with open(tracker_path) as f:
        lines = [line.rstrip("\n") for line in f]
    keep, dropped = [], []
    out_dir = os.path.join(cfg["exp"]["save_dir"], cfg["exp"]["exp_name"], cfg["exp"][key])
    for line in lines:
        entry = line.strip()
        if entry.endswith(".pth") and not os.path.isfile(os.path.join(out_dir, entry)):
            dropped.append(entry)
        else:
            keep.append(line)
    if dropped:
        with open(tracker_path, "w") as f:
            f.write("".join(line + "\n" for line in keep))
    return dropped


def ensure_tracker(tracker_path):
    """Upstream grounding_2d.py opens tracker_2d.txt for reading without creating it."""
    if not os.path.isfile(tracker_path):
        os.makedirs(os.path.dirname(tracker_path), exist_ok=True)
        with open(tracker_path, "w") as f:
            f.write("Processed Scenes .\n")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path
