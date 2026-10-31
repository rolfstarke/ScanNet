"""Runs OpenYOLO3D's Mask3D-proposals + YOLO-World-2D-voting pipeline inside the `openyolo3d`
conda env, writing ScanNet benchmark instance-segmentation format (scan-net.org).

The actual model/algorithm lives in the OpenYOLO3D repo (/data/OpenYOLO3D) -- this
script only adapts its output into ScanNet format. Uses the model's own single-scene API
(utils.OpenYolo3D().predict()) against a scratch dir that symlinks this scene's extracted
frames/ (see frames.py) plus the mesh, and overrides the text prompts with the requested classes.

Run parameters (all optional; `{}` stays byte-equivalent to the official path):
  proposal_source       live (default) | official_precomputed -- the authors' frozen
                        mesh+superpoint Mask3D proposals via the documented
                        predict(path_to_3d_masks=...) (README: "use pre-computed masks to
                        reproduce exact numbers").
  frequency             frame stride (config openyolo3d.frequency, default 10)
  network2d_th          YOLO-World score threshold (config network2d.th, default 0.08)
  network3d_nms         Mask3D proposal NMS IoU (config network3d.nms, default 0.6)
  vote_weighting        count (default, upstream pixel mode) | detector_confidence
  proposal_score_power  multiply class distributions by mask3d_score**power (default 0 = off)

OpenYOLO3D needs the LD_LIBRARY_PATH to include /data/openyolo3D/cuda-11.3/lib64 (runner.py
sets it before launching this script).
"""
import argparse
import hashlib
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import _benchmark_spec, add_run_args, load_overrides, write_scannet_submission  # noqa: E402

OPENYOLO3D_REPO = "/data/OpenYOLO3D"
SCRATCH_ROOT = "/data/openyolo3D/scratch"
CONFIG_TEMPLATE = os.path.join(OPENYOLO3D_REPO, "pretrained", "config_scannet200.yaml")
MIN_MASK_POINTS = 20
PROPOSAL_SOURCE = "live"
PRECOMPUTED_MASKS_DIR = "/data/openyolo3D/output/scannet200/scannet200_masks"
VOTE_WEIGHTING = "count"
PROPOSAL_SCORE_POWER = 0.0
ALLOWED_PARAMS = frozenset({
    "min_mask_points", "proposal_source", "frequency", "network2d_th", "network3d_nms",
    "vote_weighting", "proposal_score_power",
})
# run parameter -> (config section, config key) in config_scannet200.yaml
CONFIG_KEYS = {
    "frequency": ("openyolo3d", "frequency"),
    "network2d_th": ("network2d", "th"),
    "network3d_nms": ("network3d", "nms"),
}


def _build_scratch_scene(frames_dir, pointcloud, scene_id, run_id, scratch_root=SCRATCH_ROOT):
    """Symlink frames/ + mesh into OpenYOLO3D's expected scene layout: poses/ (plural), a single
    intrinsics.txt (from frames_dir/intrinsic_COLOR.txt -- OpenYOLO3D's adjust_intrinsic expects
    color-resolution intrinsics and rescales to depth internally, see WORLD_2_CAM), color/,
    depth/, and one *.ply at the root."""
    scratch = os.path.join(scratch_root, run_id, scene_id)
    if os.path.islink(scratch) or os.path.exists(scratch):
        shutil.rmtree(scratch)
    os.makedirs(scratch)

    links = {
        "color": os.path.join(frames_dir, "color"),
        "depth": os.path.join(frames_dir, "depth"),
        "poses": os.path.join(frames_dir, "pose"),  # model expects plural "poses"
        "intrinsics.txt": os.path.join(frames_dir, "intrinsic_color.txt"),
        f"{scene_id}.ply": pointcloud,
    }
    for name, target in links.items():
        os.symlink(target, os.path.join(scratch, name))
    return scratch


def apply_config_overrides(cfg, params):  # [inferred]
    """Write only the explicitly given config knobs; absent keys leave the official
    config_scannet200.yaml values untouched."""
    for key, (section, name) in CONFIG_KEYS.items():
        if key in params:
            cfg[section][name] = params[key]
    return cfg


def needs_custom_voting(vote_weighting, proposal_score_power):  # [inferred]
    """True only when a knob departs from upstream voting; defaults run upstream code."""
    return vote_weighting != "count" or float(proposal_score_power) != 0.0


def ply_vertex_count(path):  # [inferred]
    """Vertex count from the PLY header (no mesh parsing)."""
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"no vertex element in PLY header: {path}")


def _sha256_file(path):  # [inferred]
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_precomputed(masks_dir, scene_id, n_vertices, loader=None):  # [inferred]
    """Fail closed unless <masks_dir>/<scene_id>.pt exists and is the (masks (N, K), scores (K,))
    tuple upstream predict() expects, with N == mesh vertex count. Returns (path, sha256)."""
    path = os.path.join(masks_dir, f"{scene_id}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"official precomputed proposals missing: {path}")
    if loader is None:
        import torch
        loader = torch.load
    data = loader(path)
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError(f"{path}: expected (masks, scores) tuple, got {type(data).__name__}")
    masks, scores = data
    if len(masks.shape) != 2 or masks.shape[0] != n_vertices:
        raise ValueError(f"{path}: masks shape {tuple(masks.shape)} does not match "
                         f"{n_vertices} mesh vertices")
    if scores.shape[0] != masks.shape[1]:
        raise ValueError(f"{path}: {scores.shape[0]} scores for {masks.shape[1]} masks")
    return path, _sha256_file(path)


# [inferred] -- vote weighting / score calibration helpers (lost with the batch-1 worktree;
# rewritten from #35/#54: weight point votes by detector confidence, calibrate with Mask3D score)
def vote_distribution(labels, weights, num_classes):
    """Per-mask class distribution over projected point labels (-1 = no box). With unit
    weights this is upstream's count histogram normalized by its max and the mode.
    Returns (distribution, class_label, support of class_label)."""
    labels = np.asarray(labels).astype(np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    covered = labels != -1
    distribution = np.zeros(num_classes, dtype=np.float64)
    if covered.any() and weights[covered].sum() > 0:
        np.add.at(distribution, labels[covered], weights[covered])
        class_label = int(np.argmax(distribution))
        support = float(distribution[class_label])
        distribution = distribution / distribution.max()
        return distribution, class_label, support
    distribution[-1] = 1.0
    return distribution, num_classes - 1, 0.0


def proposal_scale(proposal_scores, power):
    """mask3d_score**power per proposal; power 0 -> all ones (no calibration)."""
    scores = np.clip(np.asarray(proposal_scores, dtype=np.float64), 0.0, None)
    if float(power) == 0.0:
        return np.ones_like(scores)
    return scores ** float(power)


def flatten_topk(distributions, topk):
    """Upstream topk_per_image flattening: every (mask, class) cell is a candidate, keep the
    `topk` highest. Returns (mask_idx, class_idx, scores)."""
    distributions = np.asarray(distributions, dtype=np.float32)
    n_masks, n_classes = distributions.shape if distributions.ndim == 2 else (0, 0)
    flat = distributions.reshape(-1)
    order = np.argsort(-flat, kind="stable")[:min(topk, len(flat))]
    return order // max(n_classes, 1), order % max(n_classes, 1), flat[order]


def _make_model_class(base, vote_weighting, proposal_score_power):  # [inferred]
    """Subclass of upstream OpenYolo3D whose voting step mirrors
    label_3d_masks_from_label_maps but weights votes / calibrates scores."""
    import torch
    from utils import compute_iou, get_visibility_mat

    class SpellbookOpenYolo3D(base):
        def construct_label_maps(self, predictions_2d_bboxes, save_label_map=False):
            # Same stamping order as upstream (largest box first), stamping detector scores.
            score_maps = torch.zeros((len(predictions_2d_bboxes), self.world2cam.height,
                                      self.world2cam.width), dtype=torch.float32)
            for frame_id, pred in enumerate(predictions_2d_bboxes.values()):
                bboxes = pred["bbox"].long().clone()
                scores = pred["scores"].float()
                bboxes[:, 0] = bboxes[:, 0] * self.scaling_params[1]
                bboxes[:, 2] = bboxes[:, 2] * self.scaling_params[1]
                bboxes[:, 1] = bboxes[:, 1] * self.scaling_params[0]
                bboxes[:, 3] = bboxes[:, 3] * self.scaling_params[0]
                bboxes_weights = (bboxes[:, 2] - bboxes[:, 0]) + (bboxes[:, 3] - bboxes[:, 1])
                sorted_indices = bboxes_weights.sort(descending=True).indices
                for bbox, score in zip(bboxes[sorted_indices], scores[sorted_indices]):
                    score_maps[frame_id, bbox[1]:bbox[3], bbox[0]:bbox[2]] = score
            self._score_maps = score_maps
            return super().construct_label_maps(predictions_2d_bboxes, save_label_map)

        def label_3d_masks_from_label_maps(self, prediction_3d_masks, predictions_2d_bboxes,
                                           projections_mesh_to_frame, keep_visible_points,
                                           is_gt):
            cfg = self.openyolo3d_config["openyolo3d"]
            label_maps = self.construct_label_maps(predictions_2d_bboxes)
            visibility_matrix = get_visibility_mat(
                prediction_3d_masks.cuda().permute(1, 0), keep_visible_points.cuda(),
                topk=25 if is_gt else cfg["topk"])
            valid = (visibility_matrix.sum(dim=0) >= 1).cpu()

            prediction_3d_masks = prediction_3d_masks.permute(1, 0).cpu()
            masks_np = prediction_3d_masks.numpy()
            projections = projections_mesh_to_frame[
                valid.to(projections_mesh_to_frame.device)].cpu().numpy()
            visibility = visibility_matrix[:, valid.to(visibility_matrix.device)].cpu().numpy()
            visible = keep_visible_points[valid.to(keep_visible_points.device)].cpu().numpy()
            label_maps = label_maps[valid].numpy()
            score_maps = self._score_maps[valid].numpy()
            valid_np = valid.numpy()
            boxes_valid = [b for bi, b in enumerate(predictions_2d_bboxes.values())
                           if valid_np[bi]]
            scale = proposal_scale(self.preds_3d[1].cpu().numpy(), proposal_score_power)

            class_labels, class_probs, distributions = [], [], []
            for mask_id, mask in enumerate(masks_np):
                prob_normalizer = 0
                labels, weights, iou_vals = [], [], []
                for fid in np.where(visibility[mask_id])[0]:
                    visible_points = (visible[fid].squeeze() * mask).astype(bool)
                    prob_normalizer += visible_points.sum()
                    xy = projections[fid][np.where(visible_points)].astype(np.int64)
                    boxes = boxes_valid[fid]["bbox"].long()
                    if len(boxes) > 0 and len(xy) > 10:
                        x_l, x_r = xy[:, 0].min(), xy[:, 0].max() + 1
                        y_t, y_b = xy[:, 1].min(), xy[:, 1].max() + 1
                        box = torch.tensor([x_l / self.scaling_params[1], y_t / self.scaling_params[0],
                                            x_r / self.scaling_params[1], y_b / self.scaling_params[0]])
                        iou_vals.append(compute_iou(box, boxes).max().item())
                    labels.append(label_maps[fid, xy[:, 1], xy[:, 0]])
                    if vote_weighting == "detector_confidence":
                        weights.append(score_maps[fid, xy[:, 1], xy[:, 0]])
                    else:
                        weights.append(np.ones(len(xy)))
                labels = np.concatenate(labels) if labels else np.array([-1])
                weights = np.concatenate(weights) if weights else np.array([0.0])
                distribution, class_label, support = vote_distribution(
                    labels, weights, self.num_classes)
                class_prob = support / prob_normalizer if prob_normalizer else 0.0
                iou_vals = np.asarray(iou_vals)
                iou_prob = float(iou_vals[iou_vals != 0].mean()) if (iou_vals != 0).any() else 0.0
                class_labels.append(class_label)
                class_probs.append(class_prob * iou_prob * scale[mask_id])
                distributions.append(distribution * scale[mask_id])

            pred_classes = torch.tensor(class_labels)
            pred_scores = torch.tensor(class_probs)
            if cfg["topk_per_image"] != -1 and not is_gt:
                dists = np.stack(distributions) if distributions else np.zeros((0, self.num_classes))
                mask_idx, cls_idx, top_scores = flatten_topk(dists, cfg["topk_per_image"])
                pred_classes = torch.from_numpy(cls_idx)
                pred_scores = torch.from_numpy(top_scores)
                prediction_3d_masks = prediction_3d_masks[torch.from_numpy(mask_idx)]
            return prediction_3d_masks.permute(1, 0), pred_classes, pred_scores

    return SpellbookOpenYolo3D


def retry_after_cudnn_failure(module, release_cache):
    """Retry `module` once, after `release_cache`, when cuDNN fails inside it. Nothing changes
    unless that failure happens.

    Why: with cudnn.benchmark on, the first conv tries an 8.3 GB workspace. If that still fits
    next to the allocator's cached pool, cuDNN has no memory left for itself and reports
    CUDNN_STATUS_INTERNAL_ERROR; if it does not fit, torch empties the cache by itself. Only a
    narrow band of cached-pool sizes hits the failure (scene0046_00 under the default config)."""
    forward = module.forward

    def _forward(*args, **kwargs):
        try:
            return forward(*args, **kwargs)
        except RuntimeError as err:
            if "cuDNN error" not in str(err):
                raise
            print(f"[WARN] {err}; retrying after releasing cached GPU memory", flush=True)
            release_cache()
            return forward(*args, **kwargs)

    module.forward = _forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True, help="scene pointcloud (.ply, xyz+rgb)")
    ap.add_argument("--frames", required=True, help="extracted frames dir (frames.py)")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="predictions output dir")
    ap.add_argument("--benchmark", default="ScanNet20",
                    choices=["ScanNet20", "ScanNet200"],
                    help="benchmark backend (default ScanNet20)")
    add_run_args(ap)
    args = ap.parse_args()
    spec = _benchmark_spec(args.benchmark)
    params = load_overrides(args.parameters_json, ALLOWED_PARAMS)
    min_mask_points = int(params.get("min_mask_points", MIN_MASK_POINTS))
    proposal_source = params.get("proposal_source", PROPOSAL_SOURCE)
    vote_weighting = params.get("vote_weighting", VOTE_WEIGHTING)
    proposal_score_power = float(params.get("proposal_score_power", PROPOSAL_SCORE_POWER))

    os.makedirs(args.out, exist_ok=True)
    scene_id = os.path.basename(os.path.dirname(args.pointcloud))
    print(f"[INFO] {scene_id} run_id={args.run_id} overrides={params}")

    path_to_3d_masks = None
    if proposal_source == "official_precomputed":
        masks_file, masks_sha = validate_precomputed(
            PRECOMPUTED_MASKS_DIR, scene_id, ply_vertex_count(args.pointcloud))
        path_to_3d_masks = PRECOMPUTED_MASKS_DIR
        print(f"[INFO] official precomputed proposals {masks_file} sha256={masks_sha}")

    scratch = _build_scratch_scene(args.frames, args.pointcloud, scene_id, args.run_id)

    # Per-run config: reuse the repo's scannet200 template but with exactly the requested
    # classes as text prompts.
    import yaml
    with open(CONFIG_TEMPLATE) as f:
        cfg = yaml.safe_load(f)
    cfg["network2d"]["text_prompts"] = list(args.classes)
    cfg["network3d"]["is_gt"] = False
    apply_config_overrides(cfg, params)
    cfg_path = os.path.join(scratch, "openyolo3d_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    os.chdir(OPENYOLO3D_REPO)
    sys.path.insert(0, OPENYOLO3D_REPO)
    from utils import OpenYolo3D

    model_cls = OpenYolo3D
    if needs_custom_voting(vote_weighting, proposal_score_power):
        model_cls = _make_model_class(OpenYolo3D, vote_weighting, proposal_score_power)
    openyolo3d = model_cls(cfg_path)
    if path_to_3d_masks is None:
        import torch
        retry_after_cudnn_failure(openyolo3d.network_3d.model.model.query_projection,
                                  torch.cuda.empty_cache)
        prediction = openyolo3d.predict(path_2_scene_data=scratch, depth_scale=1000.0,
                                        text=args.classes)
    else:
        prediction = openyolo3d.predict(path_2_scene_data=scratch, depth_scale=1000.0,
                                        text=args.classes, path_to_3d_masks=path_to_3d_masks)
    masks, classes, scores = prediction[scene_id]

    # classes are indices into the prompt list; the model reserves the last index
    # (num_classes-1 = len(args.classes)) for unassigned/background -- drop those.
    background = len(args.classes)
    masks_np = masks.cpu().numpy().astype(bool)  # (n_points, n_instances)
    classes_np = classes.cpu().numpy()
    scores_np = scores.cpu().numpy()

    def _instances():
        for i in range(masks_np.shape[1]):
            if classes_np[i] == background:
                continue
            yield masks_np[:, i], args.classes[classes_np[i]], float(scores_np[i])

    n_written = write_scannet_submission(args.out, scene_id, args.classes, _instances(),
                                          min_mask_points, spec)
    print(f"[INFO] Wrote {n_written} instances to {args.out} "
          f"({masks_np.shape[1]} raw masks, {masks_np.shape[0]} points)")


if __name__ == "__main__":
    main()
