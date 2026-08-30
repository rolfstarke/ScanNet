"""Shared helpers for the model prediction run scripts: voxel decimation, official ScanNet
submission-format prediction writing, and ScanNet scene-id extraction. Each _<model>_run.py runs
standalone as a subprocess in its own conda env (see runner.py), so this is imported the same way
the scripts are -- `sys.path.insert(0, os.path.dirname(__file__))` then `from common import ...`.

Prediction output follows ScanNet's official benchmark submission layout (scan-net.org): one
<scene_id>.txt per scan at the submission root plus predicted_masks/<scene_id>_NNN.txt (one int
per mesh vertex), which is the shape ScanNet's own evaluator
(BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py) reads directly.
"""
import json
import os
import pathlib
import secrets
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


_INT_BOUNDS = {
    "point_limit": (1, None),
    "min_mask_points": (1, None),
    "decimate_limit": (1, None),
    "final_instance_top_k": (1, None),
}
_FLOAT_BOUNDS = {
    "dedup_iou": (0.0, 1.0),
    "grid_size": (0.0, None),
}


def _benchmark_spec(benchmark_name):
    """Resolve a benchmark name to its BenchmarkSpec (spellbook/evaluation/benchmark.py).
    Wrappers run as subprocesses in OTHER conda envs with no spellbook path set, and MUST NOT
    get spellbook dir on sys.path (it shadows e.g. OpenIns3D's `from utils import ...`), so load
    benchmark.py via importlib from its absolute path instead."""
    import importlib.util
    spellbook_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec_path = os.path.join(spellbook_dir, "evaluation", "benchmark.py")
    _spec = importlib.util.spec_from_file_location("spellbook_benchmark", spec_path)
    module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(module)
    return module.resolve_benchmark(benchmark_name)


def scene_id_from_pointcloud(pointcloud_path):
    """ScanNet scene id straight from the scene directory name:
    /data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply -> scene0000_00."""
    return pathlib.Path(pointcloud_path).parent.name


def validate_run_id(run_id):
    if not isinstance(run_id, str) or not run_id or run_id.strip() != run_id:
        raise ValueError(f"invalid run_id {run_id!r}")
    if os.sep in run_id or (os.altsep and os.altsep in run_id) or run_id in (".", ".."):
        raise ValueError(f"invalid run_id {run_id!r}")
    return run_id


def _as_number(value, kind, key):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    if kind is int:
        if int(value) != value:
            raise ValueError(f"{key} must be an integer, got {value!r}")
        return int(value)
    return float(value)


def _check_bounds(key, value, lo, hi):
    if value < lo or (hi is not None and value > hi):
        bound = f">= {lo}" if hi is None else f"in [{lo}, {hi}]"
        raise ValueError(f"{key} must be {bound}, got {value!r}")
    return value


def decimate(pts, limit):
    """Voxel-downsample until under `limit` points; returns (kept_points, nn_idx) where
    nn_idx[i] is the index into kept_points nearest to the ORIGINAL points[i] -- used to
    propagate per-instance masks back onto every original point."""
    if limit <= 0:
        raise ValueError("point_limit must be positive")
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


def add_run_args(parser):
    parser.add_argument("--run-id", default="adhoc", type=validate_run_id)
    parser.add_argument("--parameters-json", default=None)


def load_overrides(path, allowed):
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    if type(data) is not dict:
        raise ValueError(f"parameters JSON must be an object: {path}")
    extra = [key for key in data if key not in allowed]
    if extra:
        raise ValueError(f"unknown parameter(s): {extra}")
    out = {}
    for key, value in data.items():
        if key in _INT_BOUNDS:
            value = _check_bounds(key, _as_number(value, int, key), *_INT_BOUNDS[key])
        elif key in _FLOAT_BOUNDS:
            value = _check_bounds(key, _as_number(value, float, key), *_FLOAT_BOUNDS[key])
        out[key] = value
    return out


def write_scannet_submission(submission_root, scene_id, classes, instances, min_mask_points, spec):
    """Writes <scene_id>.txt + predicted_masks/<scene_id>_NNN.txt in ScanNet's official
    benchmark submission layout (scan-net.org): one line per instance
    "predicted_masks/<scene_id>_NNN.txt <label_id> <confidence>", each mask one int per mesh
    vertex. `instances` is an iterable of (mask_bool_over_full_cloud, class_name, confidence);
    masks with fewer than `min_mask_points` set points are skipped. `spec` is a
    benchmark.BenchmarkSpec; its label_to_id assigns the scoreable label ids, and any class not
    in it raises ValueError. Returns the number of instances written."""
    unknown = [c for c in classes if c not in spec.label_to_id]
    if unknown:
        raise ValueError(
            f"write_scannet_submission: class(es) {unknown} are not in benchmark {spec.name} "
            f"({len(spec.class_labels)} classes) - cannot assign a scoreable label id. Use only "
            f"the official benchmark class names."
        )
    label_ids = {c: spec.label_to_id[c] for c in classes}
    mask_dir = os.path.join(submission_root, "predicted_masks")
    os.makedirs(mask_dir, exist_ok=True)
    index_path = os.path.join(submission_root, f"{scene_id}.txt")
    tmp_index = index_path + ".tmp"
    gen = secrets.token_hex(4)
    written = []
    n_written = 0
    with open(tmp_index, "w") as scene_f:
        for mask, class_name, confidence in instances:
            if mask.sum() < min_mask_points:
                continue
            mask_name = f"{scene_id}_{gen}_{n_written:03d}.txt"
            final_mask = os.path.join(mask_dir, mask_name)
            tmp_mask = final_mask + ".tmp"
            np.savetxt(tmp_mask, mask.astype(int), fmt="%d")
            with open(tmp_mask, "rb") as mask_f:
                os.fsync(mask_f.fileno())
            os.replace(tmp_mask, final_mask)
            written.append(mask_name)
            scene_f.write(f"predicted_masks/{mask_name} {label_ids[class_name]} {confidence:.4f}\n")
            n_written += 1
        scene_f.flush()
        os.fsync(scene_f.fileno())
    os.replace(tmp_index, index_path)
    keep = set(written)
    prefix = scene_id + "_"
    if os.path.isdir(mask_dir):
        for name in os.listdir(mask_dir):
            if name.startswith(prefix) and name.endswith(".txt") and name not in keep:
                os.remove(os.path.join(mask_dir, name))
    return n_written
