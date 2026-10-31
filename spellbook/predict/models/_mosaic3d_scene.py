"""Mosaic3D scene inference (recreates the lost local helper `scripts/run_custom_scene.py`).

Runs in the mosaic3d conda env. Everything model-side goes through Mosaic3D's own documented
pieces, unmodified:
- network: hydra `experiment=train_spunet_multidata_ppt data=sc model/clip=recap_clip`
  (`src.models.networks.ppt.model.PPT` around SpUNet34C/PDNorm), `net.*` keys of the checkpoint;
- input: the val transforms of `configs/data/sc.yaml` (CenterShift(apply_z) + NormalizeColor,
  feat = color, `condition`) and `grid_size` sparsify (`src.models.utils.structure.Point`);
- text: `src.models.utils.clip_models.build_clip_model` + `caption_utils.forward_text_encoder`,
  prompt `a {c} in a scene` (`use_prompt: true`, OpenScene setting);
- classification: `CLIPAlignmentEval.predict(..., return_logit=True)` (normalize_input false),
  then the mask voting of `DenseLanguageLitModule._update_instance_segmentation_metrics`:
  per-point softmax, mean over the mask, max -> (score, class) ("mean-softmax").
"""
import os
import sys

import numpy as np

MOSAIC3D_REPO = "/data/Mosaic3D"
if MOSAIC3D_REPO not in sys.path:
    sys.path.insert(0, MOSAIC3D_REPO)

import torch  # noqa: E402

CLIP_CFG = {"model_id": "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP", "text_dim": 768}  # configs/model/clip/recap_clip.yaml
# configs/data/sc.yaml / sc+ar+sc++.yaml train_dataset.conditions, indexed by embedding_table rows
PPT_CONDITIONS = {
    1: ["ScanNet"],
    3: ["ScanNet", "ARKitScenes", "ScanNetPP"],
}
PROMPT_TEMPLATE = "a {} in a scene"
# [inferred] prompt_mode=ensemble_bg (#33 h2): template ensemble + a "background" sink column
ENSEMBLE_TEMPLATES = ("a {} in a scene", "a {}", "a photo of a {}", "there is a {} in the scene")
BACKGROUND_PROMPTS = ("background", "an empty part of a room")
DEFAULT_CLASS_PROFILE = dict(eps=0.30, min_points=50, max_extent=None)  # [inferred] #10 "eps 0.30-0.40m"


def _compose(conditions):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.join(MOSAIC3D_REPO, "configs"), version_base="1.3"):
        return compose(config_name="eval", overrides=[
            "experiment=train_spunet_multidata_ppt", "data=sc", "trainer=cpu",
            "model/clip=recap_clip", "model/loss=caption",
            f"data.train_dataset.conditions=[{','.join(conditions)}]",
        ])


def _state_dict(ckpt_path):
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]


def ppt_conditions_for(ckpt_path):
    """Shape-probed PPT selector: the checkpoint's `net.embedding_table.weight` row count decides
    the condition list (`[1,256]` -> sc.ckpt, `[3,256]` -> spunet34c.ckpt sc+ar+sc++)."""
    rows = int(_state_dict(ckpt_path)["net.embedding_table.weight"].shape[0])
    if rows not in PPT_CONDITIONS:
        raise ValueError(f"{ckpt_path}: unsupported PPT embedding table with {rows} conditions")
    return list(PPT_CONDITIONS[rows])


def build_net(conditions=None):
    from hydra.utils import instantiate
    return instantiate(_compose(conditions or ["ScanNet"]).model.net)()


def load_backbone_ckpt(net, ckpt_path):
    sd = {k[len("net."):]: v for k, v in _state_dict(ckpt_path).items() if k.startswith("net.")}
    net.load_state_dict(sd, strict=True)
    return net


def _load_points(pointcloud, up_axis):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(pointcloud)
    pts = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors, dtype=np.float32)
    if cols.size and cols.max() <= 1.0 + 1e-6:
        cols = cols * 255.0
    if up_axis is not None and int(up_axis) != 2:
        a = int(up_axis)
        pts = pts[:, [(a + 1) % 3, (a + 2) % 3, a]]
    return pts, cols


def _encode(net, pts, cols, condition, grid_size, device):
    from src.data.utils.transform import CenterShift, NormalizeColor
    data = NormalizeColor()(CenterShift(apply_z=True)({"coord": pts.copy(), "color": cols.copy()}))
    batch = {
        "coord": torch.from_numpy(np.ascontiguousarray(data["coord"])).float().to(device),
        "feat": torch.from_numpy(np.ascontiguousarray(data["color"])).float().to(device),
        "offset": torch.tensor([len(pts)], device=device),
        "grid_size": grid_size,
        "condition": [condition],
    }
    with torch.no_grad():
        point = net(batch)
    return point, point.sparse_conv_feat.features[point.v2p_map]


def _text_embedding(classes, clip_model, device, prompt_mode="single"):
    """Returns (T[C(+B), D], n_class). Background rows (ensemble_bg) sit after the classes."""
    from src.utils.caption_utils import forward_text_encoder
    if prompt_mode == "single":
        prompts = [PROMPT_TEMPLATE.format(c) if "other" not in c else "other" for c in classes]
        return forward_text_encoder(prompts, clip_model, normalize=True, device=device), len(classes)
    if prompt_mode != "ensemble_bg":
        raise ValueError(f"unknown prompt_mode {prompt_mode!r}")
    rows = []
    for c in classes:
        emb = forward_text_encoder([t.format(c) for t in ENSEMBLE_TEMPLATES], clip_model,
                                   normalize=True, device=device).mean(0)
        rows.append(torch.nn.functional.normalize(emb, dim=-1))
    bg = forward_text_encoder(list(BACKGROUND_PROMPTS), clip_model, normalize=True, device=device)
    return torch.cat([torch.stack(rows), bg], 0), len(classes)


def _clip(clip_model, device):
    from src.models.utils.clip_models import build_clip_model
    if clip_model is None:
        clip_model = build_clip_model(CLIP_CFG, device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model


def _ready_net(net, checkpoint, device, ppt_conditions):
    if net is None:
        net = load_backbone_ckpt(build_net(ppt_conditions or ppt_conditions_for(checkpoint)), checkpoint)
    return net.to(device).eval()


def classify_mask3d_proposals(pointcloud, classes, checkpoint, device, masks, condition="ScanNet",
                              grid_size=0.02, up_axis=2, min_mask_points=100, ppt_conditions=None,
                              prompt_mode="single", net=None, clip_model=None):
    """Mean-softmax open-vocabulary classification of class-agnostic proposals `masks[N, M]`.
    Returns (objects, mask_probs): objects = [{point_indices, class_name, score, proposal}] in
    proposal order (masks under `min_mask_points` skipped); mask_probs[M_kept, C] mean softmax."""
    masks = np.asarray(masks, dtype=bool)
    pts, cols = _load_points(pointcloud, up_axis)
    if masks.shape[0] != len(pts):
        raise ValueError(f"masks N={masks.shape[0]} != pointcloud N={len(pts)}")
    net = _ready_net(net, checkpoint, device, ppt_conditions)
    clip_model = _clip(clip_model, device)
    _, feat = _encode(net, pts, cols, condition, grid_size, device)
    text, n_class = _text_embedding(classes, clip_model, device, prompt_mode)
    objects, probs_out = [], []
    with torch.no_grad():
        probs = torch.softmax(feat.float() @ text.float().t(), dim=-1)
        for m in range(masks.shape[1]):
            sel = masks[:, m]
            if sel.sum() < min_mask_points:
                continue
            idx = np.flatnonzero(sel)
            mean = probs[torch.from_numpy(idx).to(probs.device)].mean(dim=0)[:n_class]
            score, cls = torch.max(mean, dim=0)
            objects.append({"point_indices": idx, "class_name": classes[int(cls)],
                            "score": float(score), "proposal": m})
            probs_out.append(mean.cpu().numpy())
    mask_probs = np.stack(probs_out) if probs_out else np.zeros((0, n_class), np.float32)
    return objects, mask_probs


def _dbscan(points, eps, min_points):
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1).fit_predict(points)


def run_inference(scene_ply, classes, checkpoint, device, condition="ScanNet", grid_size=0.02,
                  up_axis=2, class_profiles=None, net=None, clip_model=None, ppt_conditions=None):
    """Lower-bound DBSCAN head (#10): per-point CLIP argmax, then per-class DBSCAN; score = mean
    per-point softmax of the cluster's class. Returns (objects, point_class, point_prob)."""
    pts, cols = _load_points(scene_ply, up_axis)
    net = _ready_net(net, checkpoint, device, ppt_conditions)
    clip_model = _clip(clip_model, device)
    _, feat = _encode(net, pts, cols, condition, grid_size, device)
    text, n_class = _text_embedding(classes, clip_model, device)
    with torch.no_grad():
        probs = torch.softmax(feat.float() @ text.float().t(), dim=-1).cpu().numpy()
    point_class = probs.argmax(1)
    objects = []
    for ci, name in enumerate(classes):
        idx = np.flatnonzero(point_class == ci)
        prof = {**DEFAULT_CLASS_PROFILE, **(class_profiles or {}).get(name, {})}
        if len(idx) < prof["min_points"]:
            continue
        labels = _dbscan(pts[idx], prof["eps"], prof["min_points"])
        for lab in np.unique(labels[labels >= 0]):
            members = idx[labels == lab]
            extent = float((pts[members].max(0) - pts[members].min(0)).max())
            if prof["max_extent"] is not None and extent > prof["max_extent"]:
                continue
            objects.append({"point_indices": members, "class_name": name,
                            "score": float(probs[members, ci].mean())})
    return objects, point_class, probs
