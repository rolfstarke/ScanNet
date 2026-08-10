"""Flat per-vertex GT export + benchmark evaluation dispatch (Phase 5).

- export-gt: writes per-vertex GT (label_id*1000 + instance_id, one line per
  _vh_clean_2.ply vertex) filtered to the benchmark's valid class ids.
- evaluate: dispatches a benchmark run's submission root to the official
  ScanNet20 evaluator or the ScanNet200 evaluator; writes a per-class CSV.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "BenchmarkScripts"))

from benchmark import load_settings, resolve_benchmark, artifact_paths, submission_dir  # noqa: E402
import util  # noqa: E402
import util_3d  # noqa: E402

PYTHON = "/home/rolf/anaconda3/envs/3disspellbook/bin/python"
_LABEL_MAP_FALLBACK = "/data/scannet/v2/scannetv2-labels.combined.tsv"

_EVALUATOR_SCRIPTS = {
    "official": os.path.join(
        _REPO_ROOT, "BenchmarkScripts", "3d_evaluation", "evaluate_semantic_instance.py"),
    "scannet200": os.path.join(_REPO_ROOT, "spellbook", "scannet200_evaluator.py"),
}


def _label_map_file(scannet_root):
    path = os.path.join(scannet_root, "v2", "scannetv2-labels.combined.tsv")
    if not os.path.isfile(path):
        path = _LABEL_MAP_FALLBACK
    if not os.path.isfile(path):
        raise FileNotFoundError("label map not found: tried "
                                f"{os.path.join(scannet_root, 'v2', 'scannetv2-labels.combined.tsv')}"
                                f" and {_LABEL_MAP_FALLBACK}")
    return path


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        for group in data['segGroups']:
            object_id = group['objectId'] + 1
            label = group['label']
            segs = group['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        for i, seg_id in enumerate(data['segIndices']):
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, len(data['segIndices'])


def export_gt(scan_path, output_file, spec, scannet_root=None):
    scan_name = os.path.basename(scan_path)
    label_map = util.read_label_mapping(
        _label_map_file(scannet_root), label_from='raw_category', label_to=spec.gt_column)
    mesh_file = os.path.join(scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(scan_path, scan_name + '.aggregation.json')
    seg_file = os.path.join(scan_path, scan_name + '_vh_clean_2.0.010000.segs.json')
    util_3d.read_mesh_vertices(mesh_file)

    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)

    valid_ids = set(spec.valid_ids)
    label_ids = np.zeros(num_verts, dtype=np.uint32)
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            label_ids[seg_to_verts[seg]] = label_id
    if spec.gt_column == "id":  # ScanNet200: zero non-198 classes (old 189-set kept wall/floor)
        label_ids[~np.isin(label_ids, list(valid_ids))] = 0

    instance_ids = np.zeros(num_verts, dtype=np.uint32)
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            instance_ids[seg_to_verts[seg]] = object_id

    with open(output_file, 'w') as f:
        for li, ii in zip(label_ids, instance_ids):
            f.write('%d\n' % (li * 1000 + ii))


def normalize_scene_id(scene):
    scene = scene.strip()
    if not scene:
        raise ValueError("empty scene id in --scenes (trailing comma?)")
    if scene.startswith("scene"):
        scene = scene[5:]
    return "scene" + scene


def _parse_list(values):
    entries = [v for v in values.split(",")]
    for e in entries:
        if not e.strip():
            raise ValueError(f"empty entry in comma-separated list {values!r} (trailing comma?)")
    return [normalize_scene_id(e) for e in entries]


def _ensure_settings(root):
    if root is None:
        return load_settings()["scannet_root"]
    return root


def export_gt_cli(argv=None):
    ap = argparse.ArgumentParser(description="Export flat per-vertex instance GT for a scene")
    ap.add_argument("--scene", required=True, help="scene id, e.g. 0568_00 or scene0568_00")
    ap.add_argument("--benchmark", default=None, help="ScanNet20 | ScanNet200 (default: settings)")
    ap.add_argument("--scannet-root", default=None)
    args = ap.parse_args(argv)
    spec = resolve_benchmark(args.benchmark)
    root = _ensure_settings(args.scannet_root)
    scene = normalize_scene_id(args.scene)
    scan_path = os.path.join(root, "scans", scene)
    out_file = os.path.join(artifact_paths(spec, root)["gt"], scene + ".txt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    export_gt(scan_path, out_file, spec, scannet_root=root)
    print(f"GT -> {out_file}")


def evaluate_cli(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a benchmark run with the official evaluator")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--models", required=True, help="comma-separated model names")
    ap.add_argument("--scenes", required=True,
                    help="comma-separated scene ids (0568_00 or scene0568_00)")
    ap.add_argument("--benchmark", default=None, help="ScanNet20 | ScanNet200 (default: settings)")
    ap.add_argument("--scannet-root", default=None)
    ap.add_argument("--pred-path", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    spec = resolve_benchmark(args.benchmark)
    root = _ensure_settings(args.scannet_root)
    paths = artifact_paths(spec, root)

    models = [m.strip() for m in args.models.split(",")]
    for m in models:
        if not m:
            raise ValueError(f"empty entry in --models {args.models!r} (trailing comma?)")
    scenes = _parse_list(args.scenes)

    for model in models:
        pred_dir = args.pred_path or submission_dir(spec, args.run_id, model, scannet_root=root)
        out_dir = os.path.join(paths["evaluations"], args.run_id)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, model + ".csv")

        missing = []
        if not os.path.isdir(pred_dir):
            raise FileNotFoundError(f"prediction root missing: {pred_dir}")
        for scene in scenes:
            if not os.path.isfile(os.path.join(pred_dir, scene + ".txt")):
                missing.append(f"pred {os.path.join(pred_dir, scene + '.txt')}")
            if not os.path.isfile(os.path.join(paths["gt"], scene + ".txt")):
                missing.append(f"gt   {os.path.join(paths['gt'], scene + '.txt')}")
        if missing:
            raise FileNotFoundError(
                f"[{model}] missing files before evaluation:\n  " + "\n  ".join(missing))

        evaluator = _EVALUATOR_SCRIPTS[spec.evaluator]
        print(f"[{model}] evaluating {len(scenes)} scenes -> {out_file}")
        proc = subprocess.run(
            [PYTHON, evaluator, "--pred_path", pred_dir, "--gt_path", paths["gt"],
             "--output_file", out_file],
            capture_output=True, text=True, check=False)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"evaluator failed for {model} (exit {proc.returncode})")


if __name__ == "__main__":
    sub_cmds = {
        "export-gt": export_gt_cli,
        "evaluate": evaluate_cli,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in sub_cmds:
        print("usage: python evaluate.py {export-gt,evaluate} [options]")
        sys.exit(2)
    sub_cmds[sys.argv[1]](sys.argv[2:])
