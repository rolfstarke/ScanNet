"""Dump class-agnostic Mask3D proposals for one ScanNet mesh (OpenIns3D env).

Output npz v2 (`PROPOSAL_NPZ_FORMAT` in _mosaic3d_run.py): masks[N,M] bool, scores[M] float32
(Mask3D confidence = max class softmax x mean in-mask sigmoid, as in OpenIns3D's
`map_output_to_pointcloud`), format=2. Written atomically.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch

OPENINS3D_REPO = "/data/openins3d/repo-56ccc22"  # worktree of the logged upstream commit 56ccc22
CKPT = "/data/openins3d/checkpoints/scannet200_val.ckpt"


def _lease_fd():
    fd = os.environ.get("SPELLBOOK_GPU_LEASE_FD")
    return (int(fd),) if fd else ()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--confidence", type=float, default=0.001)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    pointcloud = os.path.abspath(args.pointcloud)

    os.chdir(OPENINS3D_REPO)
    sys.path.insert(0, OPENINS3D_REPO)
    from openins3d.mask3d import get_model, prepare_data_pcd

    pcd = o3d.io.read_point_cloud(pointcloud)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if cols.max() <= 1.0 + 1e-6:
        cols = cols * 255.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(CKPT).to(device).eval()
    data, _, _, features, _, inverse_map = prepare_data_pcd(pts, cols, device)
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
    masks_np, scores_np = _proposals(outputs, inverse_map, args.confidence, len(pts))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out[:-len(".npz")] + f".tmp{os.getpid()}.npz" if out.endswith(".npz") else out + ".tmp.npz"
    np.savez_compressed(tmp, masks=masks_np, scores=scores_np, format=np.int64(2))
    os.replace(tmp, out)
    print(f"[INFO] Mask3D proposals {masks_np.shape} -> {out}")


def _proposals(outputs, inverse_map, confidence, n_points):
    """OpenIns3D `map_output_to_pointcloud` thresholding (label < 200, conf > threshold), keeping
    the per-mask confidence that function discards."""
    logits = outputs["pred_logits"][0].detach().cpu()
    masks = outputs["pred_masks"][0].detach().cpu()
    inverse_map = np.asarray(inverse_map)
    kept, scores = [], []
    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        m = p_masks > 0.5
        c = torch.max(p_labels) * (p_masks[m].sum() / (m.sum() + 1e-8))
        if torch.argmax(p_labels, dim=-1) < 200 and c > confidence:
            kept.append(m.numpy()[inverse_map])
            scores.append(float(c))
    if not kept:
        return np.zeros((n_points, 0), dtype=bool), np.zeros(0, dtype=np.float32)
    return np.stack(kept, axis=1).astype(bool), np.asarray(scores, dtype=np.float32)


if __name__ == "__main__":
    main()
