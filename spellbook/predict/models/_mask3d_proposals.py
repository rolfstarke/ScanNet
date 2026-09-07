"""Dump class-agnostic Mask3D proposals for one ScanNet mesh (OpenIns3D env)."""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch

OPENINS3D_REPO = "/home/rolf/GIT/OpenIns3D"
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

    os.chdir(OPENINS3D_REPO)
    sys.path.insert(0, OPENINS3D_REPO)
    from openins3d.mask3d import get_model, map_output_to_pointcloud, prepare_data

    pcd = o3d.io.read_point_cloud(args.pointcloud)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if cols.max() <= 1.0 + 1e-6:
        cols = cols * 255.0
    pcd_rgb = np.hstack([pts, cols])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(CKPT).to(device).eval()
    data, features, _, inverse_map = prepare_data(pcd_rgb, device)
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
        try:
            masks = map_output_to_pointcloud(
                outputs, inverse_map, confidence_threshold=args.confidence)
        except Exception:
            masks = torch.zeros((len(pts), 0), dtype=torch.bool)
    masks_np = np.asarray(masks).astype(bool)
    if masks_np.ndim == 1:
        masks_np = masks_np[:, None]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, masks=masks_np)
    print(f"[INFO] Mask3D proposals {masks_np.shape} -> {args.out}")


if __name__ == "__main__":
    main()
