"""Spellbook-side entry points for two Open3DIS tools, run in the `open3dis` conda env with
cwd = the per-scene scratch cwd/ (private tracker_*.txt, #15) and PYTHONPATH=/data/Open3DIS.
They import the upstream modules and reproduce the tools' __main__ loops, so the upstream
checkout (VinAIResearch bc2a3b3) stays untouched.

    grounding  tools/grounding_2d.py for segmenter2d.model 'Grounded-SAM' only. Upstream's
               __main__ also imports the RAM / YOLO-World segmenters (mmyolo, ram), which the
               official Grounded-SAM path never calls.  # [inferred] fork ran upstream directly
    generate   tools/generate_3d_inst.py with the fork `12e05a8` gates restored as config keys:
                 `if True:`  -> `if not cfg.exp.skip_clustering:` (else load hier_agglo)
                 `if False:` -> `if cfg.exp.emit_final:`
               (official RUN.md step 2, and step 4 "re-run step 2 to finalize").

Usage: python _open3dis_stage.py {grounding,generate} --config <derived yaml>
"""
import argparse
import os
import sys

OPEN3DIS_REPO = "/data/Open3DIS"


def _scene_ids(cfg):
    with open(cfg.data.split_path) as f:
        return sorted(line.rstrip("\n") for line in f if line.strip())


def _tracked(tracker, path):
    """Upstream tracker check-then-append (scene recorded before its output is written)."""
    if not os.path.exists(tracker):
        with open(tracker, "w") as f:
            f.write("Processed Scenes .\n")
    with open(tracker) as f:
        if any(path in line.strip() for line in f):
            print("existed " + path)
            return True
    with open(tracker, "a") as f:
        f.write(path + "\n")
    return False


def grounding(cfg):
    import torch
    from tqdm import tqdm
    from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
    from util2d.grounded_sam import Grounded_Sam

    if cfg.data.dataset_name != "scannet200" or cfg.segmenter2d.model != "Grounded-SAM":
        raise ValueError("spellbook grounding entry supports scannet200 + Grounded-SAM only")
    class_names = INSTANCE_CAT_SCANNET_200
    model = Grounded_Sam(cfg)
    save_dir = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)
    save_dir_feat = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.grounded_feat_output)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_feat, exist_ok=True)
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(_scene_ids(cfg)):
            if _tracked("tracker_2d.txt", scene_id + ".pth"):
                continue
            print("Process", scene_id)
            grounded_data_dict, grounded_features = model.gen_grounded_mask_and_feat(
                scene_id, class_names, cfg=cfg)
            torch.save({"feat": grounded_features}, os.path.join(save_dir_feat, scene_id + ".pth"))
            torch.save(grounded_data_dict, os.path.join(save_dir, scene_id + ".pth"))
            torch.cuda.empty_cache()


def generate(cfg):
    import clip
    import torch
    from tqdm import tqdm
    sys.path.insert(0, os.path.join(OPEN3DIS_REPO, "tools"))
    import generate_3d_inst as g3d
    from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
    from open3dis.src.clustering.clustering import (
        process_hierarchical_agglomerative_nospp, process_hierarchical_agglomerative_spp,
    )

    if cfg.data.dataset_name != "scannet200":
        raise ValueError("spellbook generate entry supports scannet200 only")
    skip_clustering = bool(cfg.exp.get("skip_clustering", False))
    emit_final = bool(cfg.exp.get("emit_final", False))
    # [inferred] step 4 re-runs on a scene already in tracker_lifted.txt -> own tracker
    tracker = "tracker_final.txt" if skip_clustering else "tracker_lifted.txt"

    clip_adapter, _ = clip.load(cfg.foundation_model.clip_model, device="cuda")
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_adapter.encode_text(clip.tokenize(INSTANCE_CAT_SCANNET_200).cuda())
        text_features /= text_features.norm(dim=-1, keepdim=True)
    save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output)
    os.makedirs(save_dir_cluster, exist_ok=True)
    os.makedirs(save_dir_final, exist_ok=True)

    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(_scene_ids(cfg)):
            print("Process", scene_id)
            if _tracked(tracker, scene_id + ".pth"):
                continue
            g3d.scene_id = scene_id  # get_final_instances reads the module-global scene_id
            if not skip_clustering:
                if cfg.final_instance.spp_level:
                    proposals3d, confidence = process_hierarchical_agglomerative_spp(scene_id, cfg)
                else:
                    proposals3d, confidence = process_hierarchical_agglomerative_nospp(scene_id, cfg)
                if proposals3d is None:
                    continue
                cluster_dict = {"ins": g3d.rle_encode_gpu_batch(proposals3d), "conf": confidence}
                torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))
            if emit_final:
                cluster_dict = torch.load(os.path.join(save_dir_cluster, f"{scene_id}.pth"))
                masks_final, cls_final, scores_final = g3d.get_final_instances(
                    cfg, text_features, cluster_dict=cluster_dict,
                    use_2d_proposals=cfg.proposals.p2d, use_3d_proposals=cfg.proposals.p3d,
                    only_instance=cfg.proposals.agnostic)
                if scores_final is None:
                    final_dict = {"ins": g3d.rle_encode_gpu_batch(masks_final),
                                  "conf": None, "final_class": None}
                else:
                    final_dict = {"ins": g3d.rle_encode_gpu_batch(masks_final),
                                  "conf": scores_final.cpu(), "final_class": cls_final.cpu()}
                torch.save(final_dict, os.path.join(save_dir_final, f"{scene_id}.pth"))
            print("Done")
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("grounding", "generate"))
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    sys.path.insert(0, OPEN3DIS_REPO)
    import yaml
    from munch import Munch
    with open(args.config) as f:
        cfg = Munch.fromDict(yaml.safe_load(f))
    {"grounding": grounding, "generate": generate}[args.stage](cfg)


if __name__ == "__main__":
    main()
