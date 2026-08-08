"""Shared helpers for the model prediction run scripts: voxel decimation, ScanNet-format
prediction writing, and ScanNet scene-id extraction. Each _<model>_run.py runs standalone as a
subprocess in its own conda env (see runner.py), so this is imported the same way the scripts
are -- `sys.path.insert(0, os.path.dirname(__file__))` then `from common import ...`.

The prediction format here is ScanNet's benchmark instance-segmentation format (scan-net.org):
labels.txt + predictions.txt + predicted_masks/<NNN>.txt over the full point cloud, which is the
shape ScanNet's own evaluator (BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py)
reads directly.
"""
import importlib.util
import os
import pathlib

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# The 18 official ScanNet instance-segmentation benchmark classes, with their real NYU40 label
# ids. Source of truth: BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py (CLASS_LABELS /
# VALID_CLASS_IDS). ScanNet's evaluator silently discards any prediction whose label id is not in
# this set, so these are the only class names write_scannet_predictions can score.
BENCHMARK_CLASS_LABELS = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture",
    "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
]
BENCHMARK_VALID_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
BENCHMARK_LABEL_TO_ID = dict(zip(BENCHMARK_CLASS_LABELS, BENCHMARK_VALID_CLASS_IDS))

# ScanNet200's 189 classes present in the official validation split. Source of truth:
# BenchmarkScripts/ScanNet200/scannet200_splits.py (VALID_CLASS_IDS_200_VALIDATION holds the
# class NAMES, CLASS_LABELS_200_VALIDATION the raw `id`-column ids -- note the naming is the
# reverse of the 18-class convention above). 11 of the 200 classes are train-only and never
# scored. Imported from the official file to avoid drift; falls back to a literal copy only if
# the file is not importable.
_SCANNET200_SPLITS_PATH = "/home/rolf/GIT/ScanNet/BenchmarkScripts/ScanNet200/scannet200_splits.py"
_scan200 = None
if os.path.isfile(_SCANNET200_SPLITS_PATH):
    _spec = importlib.util.spec_from_file_location("scannet200_splits", _SCANNET200_SPLITS_PATH)
    _scan200 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_scan200)
if _scan200 is None:
    # Fallback copy (identical to BenchmarkScripts/ScanNet200/scannet200_splits.py, repo head as
    # of 2026-08-07). Only used if the official file is missing.
    _scan200 = type("scannet200_splits", (), {})()
    _scan200.VALID_CLASS_IDS_200_VALIDATION = (
        'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk',
        'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf',
        'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box', 'refrigerator', 'lamp',
        'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool',
        'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag',
        'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard', 'blanket',
        'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe',
        'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine',
        'mirror', 'copier', 'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower',
        'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate',
        'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
        'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light',
        'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat',
        'speaker', 'column', 'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
        'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
        'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace',
        'soap dish', 'kitchen counter', 'doorframe', 'toilet paper dispenser', 'mini fridge',
        'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter',
        'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
        'toilet seat cover dispenser', 'furniture', 'cart', 'scale', 'tissue box',
        'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector',
        'closet door', 'vacuum cleaner', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
        'broom', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'vent',
        'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
        'music stand', 'laundry detergent', 'dumbbell', 'tube', 'closet rod', 'coffee kettle',
        'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'folded chair',
        'fire alarm', 'power strip', 'calendar', 'poster',
    )
    _scan200.CLASS_LABELS_200_VALIDATION = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27,
        28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52,
        54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
        78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
        105, 106, 107, 110, 112, 115, 116, 118, 120, 122, 125, 128, 130, 131, 132, 134, 136,
        138, 139, 140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163, 165, 166, 168, 169,
        170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 229, 230, 232, 233, 242,
        250, 261, 264, 276, 283, 300, 304, 312, 323, 325, 342, 356, 370, 392, 395, 408, 417,
        488, 540, 562, 570, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169,
        1170, 1171, 1172, 1173, 1175, 1176, 1179, 1180, 1181, 1182, 1184, 1185, 1186, 1187,
        1188, 1189, 1191,
    )
BENCHMARK200_CLASS_LABELS = list(_scan200.VALID_CLASS_IDS_200_VALIDATION)
BENCHMARK200_VALID_CLASS_IDS = list(_scan200.CLASS_LABELS_200_VALIDATION)
BENCHMARK200_LABEL_TO_ID = dict(zip(BENCHMARK200_CLASS_LABELS, BENCHMARK200_VALID_CLASS_IDS))


def scene_id_from_pointcloud(pointcloud_path):
    """ScanNet scene id straight from the scene directory name:
    /data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply -> scene0000_00."""
    return pathlib.Path(pointcloud_path).parent.name


def decimate(pts, limit):
    """Voxel-downsample until under `limit` points; returns (kept_points, nn_idx) where
    nn_idx[i] is the index into kept_points nearest to the ORIGINAL points[i] -- used to
    propagate per-instance masks back onto every original point."""
    if len(pts) <= limit:
        return pts, np.arange(len(pts))

    voxel_size = 0.01
    down = pts
    while len(down) > limit:
        voxel_size *= 1.4
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        down = np.asarray(pcd.voxel_down_sample(voxel_size).points)

    nn_idx = cKDTree(down).query(pts, k=1, workers=-1)[1]
    return down, nn_idx


def write_scannet_predictions(out_dir, classes, instances, min_mask_points, label_set="scannet18"):
    """Writes labels.txt + predictions.txt + predicted_masks/<NNN>.txt in ScanNet benchmark
    instance-segmentation format (scan-net.org). `instances` is an iterable of
    (mask_bool_over_full_cloud, class_name, confidence); masks under `min_mask_points` are
    dropped. `label_set` selects the label vocabulary: "scannet18" (default, NYU40) or
    "scannet200" (raw `id`-column ids of the 189 classes present in the validation split).
    Returns the number written."""
    if label_set == "scannet200":
        class_labels = BENCHMARK200_CLASS_LABELS
        label_to_id = BENCHMARK200_LABEL_TO_ID
        bench_desc = "ScanNet200 189-class validation set (BENCHMARK200_CLASS_LABELS in common.py)"
        id_desc = "raw ScanNet200 `id`-column"
    else:
        class_labels = BENCHMARK_CLASS_LABELS
        label_to_id = BENCHMARK_LABEL_TO_ID
        bench_desc = "ScanNet's 18-class instance benchmark set (BENCHMARK_CLASS_LABELS in common.py)"
        id_desc = "NYU40"
    unknown = [c for c in classes if c not in label_to_id]
    if unknown:
        raise ValueError(
            f"write_scannet_predictions: class(es) {unknown} are not in {bench_desc} — cannot "
            f"assign a scoreable {id_desc} label id. Use only the official benchmark class "
            f"names, or write predictions through a different path if this is a "
            f"non-benchmark/custom-scene run."
        )
    label_ids = {c: label_to_id[c] for c in classes}
    pred_mask_dir = os.path.join(out_dir, "predicted_masks")
    os.makedirs(pred_mask_dir, exist_ok=True)
    with open(os.path.join(out_dir, "labels.txt"), "w") as f:
        for c, lid in label_ids.items():
            f.write(f"{lid} {c}\n")

    n_written = 0
    with open(os.path.join(out_dir, "predictions.txt"), "w") as pred_f:
        for mask, class_name, confidence in instances:
            if mask.sum() < min_mask_points:
                continue
            mask_name = f"{n_written:03d}.txt"
            np.savetxt(os.path.join(pred_mask_dir, mask_name), mask.astype(int), fmt="%d")
            pred_f.write(f"predicted_masks/{mask_name} {label_ids[class_name]} {confidence:.4f}\n")
            n_written += 1
    return n_written
