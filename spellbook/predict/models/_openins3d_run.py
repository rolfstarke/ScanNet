"""Runs OpenIns3D's Mask -> Snap -> Lookup pipeline (class-agnostic Mask3D proposals -> synthetic
scene renders -> YOLO-World open-vocab labelling) inside the `openins3d` conda env, writing the
result in ScanNet benchmark instance-segmentation format (scan-net.org).

Snap synthesizes 2D views directly from the point cloud, so -- unlike OpenMask3D/OpenYOLO3D --
no posed frames are needed (point-cloud-only, like Mosaic3D). The pipeline itself lives in
OpenIns3D; this script just loads the cloud, calls OpenIns3D's own functions exactly as its
zero_shot_multi_vocs.py demo does, and reformats the output.
"""
import os
import sys

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

import argparse

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from common import decimate, scene_id_from_pointcloud, write_scannet_predictions  # noqa: E402

OPENINS3D_REPO = "/home/rolf/GIT/OpenIns3D"
SCRATCH_ROOT = "/data/openins3d/scratch"  # bulky, transient synthetic renders -- kept off /home
CKPT = "third_party/scannet200_val.ckpt"

POINT_LIMIT = 1_000_000
IMAGE_SIZE = [800, 800]
ADJUST_CAMERA = [2, 0.1, 1.0]
# 0.5 is the repo demo default; the paper's benchmark pipeline evaluates at ~0.001 recall
# (precomputed masks), so 0.5 kills too many proposals (e.g. scene0304_00: only 4/150 masks
# survived). 0.05 is a pragmatic middle ground for on-the-fly proposals.
MASK_CONFIDENCE_THRESHOLD = 0.05
LOOKUP_THRESHOLD = 0.3
MIN_MASK_POINTS = 20


def _ensure_z_up(pts):
    """OpenIns3D's Snap module hardcodes the vertical axis as index 2. ScanNet meshes are
    already Z-up, so this normally returns pts unchanged; the check stays for robustness
    against any scene that isn't (detect vertical as the tightest bounding-box extent)."""
    ext = pts.max(0) - pts.min(0)
    up = int(ext.argmin())
    if up == 2:
        return pts
    if up == 1:  # Y-up: (x, y, z) -> (x, -z, y)
        return np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=1)
    print(f"[WARN] unexpected up-axis (index {up}, extents {ext}) -- leaving orientation as-is")
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True, help="scene pointcloud (.ply, xyz+rgb)")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="predictions output dir")
    ap.add_argument("--label_set", default="scannet18",
                    choices=["scannet18", "scannet200"],
                    help="label vocabulary: scannet18 (NYU40, default) or scannet200 (raw id-column ids)")
    ap.add_argument("--detector", default="odise", choices=["odise", "yoloworld"],
                    help="2D open-vocab detector for the Lookup stage (default odise -- the "
                         "detector OpenIns3D's paper numbers use; yoloworld is the repo demo default)")
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scene_id = scene_id_from_pointcloud(args.pointcloud)

    pcd = o3d.io.read_point_cloud(args.pointcloud)
    full_pts = _ensure_z_up(np.asarray(pcd.points))
    full_cols = np.asarray(pcd.colors)

    working_pts, nn_idx = decimate(full_pts, POINT_LIMIT)
    if len(working_pts) < len(full_pts):
        working_cols = full_cols[cKDTree(full_pts).query(working_pts, k=1, workers=-1)[1]]
    else:
        working_cols = full_cols
    pcd_rgb = np.hstack([working_pts, working_cols * 255.0])

    os.chdir(OPENINS3D_REPO)
    sys.path.insert(0, OPENINS3D_REPO)
    from openins3d.mask3d import get_model, prepare_data, map_output_to_pointcloud
    from openins3d.snap import Snap
    from openins3d.lookup import Lookup

    device = torch.device("cuda")
    model = get_model(CKPT).to(device).eval()
    data, features, _, inverse_map = prepare_data(pcd_rgb, device)
    with torch.no_grad():
        mask_list = map_output_to_pointcloud(model(data, raw_coordinates=features), inverse_map,
                                              confidence_threshold=MASK_CONFIDENCE_THRESHOLD)

    snap_folder = os.path.join(SCRATCH_ROOT, scene_id)
    snap = Snap(IMAGE_SIZE, ADJUST_CAMERA, snap_folder)
    lookup = Lookup(IMAGE_SIZE, ADJUST_CAMERA[2], snap_folder, text_input=args.classes,
                    results_folder=os.path.join(SCRATCH_ROOT, f"{scene_id}_results"))
    if args.detector == "odise":
        lookup.call_ODISE()
    else:
        lookup.call_YOLOWORLD()

    snap.scene_image_rendering(args.pointcloud, scene_id, mode=["global", "wide", "corner"])
    mask_cls, score = lookup.lookup_pipelie(pcd_rgb, mask_list, scene_id, threshold=LOOKUP_THRESHOLD)

    masks_np = mask_list.cpu().numpy().astype(bool)

    def _instances():
        for i, cls in enumerate(mask_cls):
            if cls == -1:
                continue
            yield masks_np[:, i][nn_idx], args.classes[cls], float(score[i])

    n_written = write_scannet_predictions(args.out, args.classes, _instances(), MIN_MASK_POINTS,
                                          label_set=args.label_set)
    print(f"[INFO] Wrote {n_written} instances to {args.out} "
          f"({mask_list.shape[1]} raw masks, {len(working_pts)}/{len(full_pts)} points used)")


if __name__ == "__main__":
    main()
