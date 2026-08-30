"""Flat per-vertex GT export + benchmark evaluation dispatch (Phase 5).

- export-gt: writes per-vertex GT (label_id*1000 + instance_id, one line per
  _vh_clean_2.ply vertex) filtered to the benchmark's valid class ids.
- evaluate: dispatches a benchmark run's submission root to the official
  ScanNet20 evaluator or the ScanNet200 evaluator; writes a per-class CSV.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_SPELLBOOK = os.path.dirname(_EVAL_DIR)
_REPO_ROOT = os.path.dirname(_SPELLBOOK)
sys.path.insert(0, _SPELLBOOK)
sys.path.insert(0, os.path.join(_REPO_ROOT, "BenchmarkScripts"))

from evaluation.benchmark import (  # noqa: E402
    load_settings, normalize_scene_id, resolve_benchmark, artifact_paths, submission_dir)
import util  # noqa: E402
import util_3d  # noqa: E402

PYTHON = "/home/rolf/anaconda3/envs/3disspellbook/bin/python"
_LABEL_MAP_FALLBACK = "/data/scannet/v2/scannetv2-labels.combined.tsv"

SIDECAR_SCHEMA = 1
SIDECAR_METRIC = "scannet_instance_ap50"
AP50_THRESHOLD = 0.5
MIN_REGION_SIZE = 100

_EVALUATOR_SCRIPTS = {
    "official": os.path.join(
        _REPO_ROOT, "BenchmarkScripts", "3d_evaluation", "evaluate_semantic_instance.py"),
    "scannet200": os.path.join(_EVAL_DIR, "scannet200_evaluator.py"),
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


def score_sidecar_path(spec, run_id, model, scene_id, scannet_root=None):
    return os.path.join(artifact_paths(spec, scannet_root)["evaluations"],
                        run_id, model, scene_id + ".tp50.json")


def _relative_mask(path, submission_root):
    root = os.path.normpath(submission_root) + os.sep
    path = os.path.normpath(path)
    if path.startswith(root):
        return path[len(root):].replace("\\", "/")
    return os.path.basename(path)


def load_pred_instances(submission_root, scene_id):
    scene_file = os.path.join(submission_root, scene_id + ".txt")
    if not os.path.isfile(scene_file):
        return None
    instances = util_3d.read_instance_prediction_file(scene_file, submission_root)
    for mask_file, prediction in list(instances.items()):
        prediction["pred_mask"] = util_3d.load_ids(mask_file) > 0
    return instances


def write_score_sidecar(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_score_sidecar(path, scene_id, label_set, run_id, model):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
        if int(doc.get("schema", -1)) != SIDECAR_SCHEMA:
            return None
        if doc.get("metric") != SIDECAR_METRIC:
            return None
        if float(doc.get("iou_threshold")) != AP50_THRESHOLD:
            return None
        if int(doc.get("min_region_size")) != MIN_REGION_SIZE:
            return None
        if doc.get("scene_id") != scene_id:
            return None
        if doc.get("label_set") != label_set:
            return None
        if doc.get("run_id") != run_id:
            return None
        if doc.get("model") != model:
            return None
        tp = int(doc["tp"])
        gt = int(doc["gt"])
        if tp < 0 or gt < 0 or tp > gt:
            return None
        verdicts = doc.get("verdicts") or {}
        if not isinstance(verdicts, dict):
            return None
        return {"tp": tp, "gt": gt, "verdicts": verdicts}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def score_prediction_scene(submission_root, scene_id, spec, run_id, model, scannet_root=None):
    gt_file = os.path.join(artifact_paths(spec, scannet_root)["gt"], scene_id + ".txt")
    if not os.path.isfile(gt_file):
        return None
    pred_instances = load_pred_instances(submission_root, scene_id)
    if pred_instances is None:
        return None
    from evaluation.scannet200_evaluator import scene_instance_summary
    summary = scene_instance_summary(util_3d.load_ids(gt_file), pred_instances, spec, scene_id)
    verdicts = {}
    for key, value in summary["verdicts"].items():
        verdicts[_relative_mask(key, submission_root)] = value
    return {
        "schema": SIDECAR_SCHEMA,
        "metric": SIDECAR_METRIC,
        "iou_threshold": AP50_THRESHOLD,
        "min_region_size": MIN_REGION_SIZE,
        "scene_id": scene_id,
        "label_set": spec.name,
        "run_id": run_id,
        "model": model,
        "tp": int(summary["tp"]),
        "gt": int(summary["gt"]),
        "verdicts": verdicts,
    }


def score_and_write_sidecar(submission_root, scene_id, spec, run_id, model, scannet_root=None):
    payload = score_prediction_scene(
        submission_root, scene_id, spec, run_id, model, scannet_root=scannet_root)
    if payload is None:
        return None
    write_score_sidecar(
        score_sidecar_path(spec, run_id, model, scene_id, scannet_root=scannet_root), payload)
    return payload


def scene_submission_status(submission_root, scene_id, spec, run_id, model, scannet_root=None):
    scene_file = os.path.join(submission_root, scene_id + ".txt")
    if not os.path.isfile(scene_file):
        return "missing"
    try:
        instances = load_pred_instances(submission_root, scene_id)
    except Exception:
        return "missing"
    if instances is None:
        return "missing"
    for mask_file, prediction in instances.items():
        mask = prediction.get("pred_mask")
        if mask is None or mask.size == 0:
            return "missing"
        label_id = prediction.get("label_id")
        if label_id not in spec.valid_ids and label_id not in spec.id_to_label:
            try:
                if int(label_id) not in spec.valid_ids:
                    return "missing"
            except (TypeError, ValueError):
                return "missing"
        conf = prediction.get("conf")
        try:
            if not np.isfinite(float(conf)):
                return "missing"
        except (TypeError, ValueError):
            return "missing"
    sidecar = load_score_sidecar(
        score_sidecar_path(spec, run_id, model, scene_id, scannet_root=scannet_root),
        scene_id, spec.name, run_id, model)
    if sidecar is None:
        return "needs_score"
    return "complete"


def _stage_eval_dirs(pred_dir, gt_dir, scenes):
    staging = tempfile.mkdtemp(prefix="spellbook-eval-")
    pred_stage = os.path.join(staging, "pred")
    gt_stage = os.path.join(staging, "gt")
    mask_stage = os.path.join(pred_stage, "predicted_masks")
    os.makedirs(mask_stage)
    os.makedirs(gt_stage)
    for scene in scenes:
        src_pred = os.path.join(pred_dir, scene + ".txt")
        src_gt = os.path.join(gt_dir, scene + ".txt")
        os.symlink(os.path.abspath(src_pred), os.path.join(pred_stage, scene + ".txt"))
        os.symlink(os.path.abspath(src_gt), os.path.join(gt_stage, scene + ".txt"))
        instances = util_3d.read_instance_prediction_file(src_pred, pred_dir)
        for mask_file in instances:
            name = os.path.basename(mask_file)
            os.symlink(os.path.abspath(mask_file), os.path.join(mask_stage, name))
    return staging, pred_stage, gt_stage


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
    ap.add_argument("--models", default=None, help="comma-separated model names (default: manifest)")
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scene ids; must match the run manifest")
    ap.add_argument("--benchmark", default=None, help="ScanNet20 | ScanNet200 (default: settings)")
    ap.add_argument("--scannet-root", default=None)
    ap.add_argument("--pred-path", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    spec = resolve_benchmark(args.benchmark)
    root = _ensure_settings(args.scannet_root)
    paths = artifact_paths(spec, root)
    from evaluation.runs import load_manifest, manifest_path, read_evaluator_csv

    man = load_manifest(manifest_path(spec, args.run_id, root), spec=spec, run_id=args.run_id)
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
        for m in models:
            if not m:
                raise ValueError(f"empty entry in --models {args.models!r} (trailing comma?)")
        if models != list(man["methods"]):
            raise ValueError(" --models must match the run manifest")
    else:
        models = list(man["methods"])
    if args.scenes:
        scenes = _parse_list(args.scenes)
        if sorted(scenes) != list(man["scenes"]):
            raise ValueError("--scenes must match the run manifest")
    else:
        scenes = list(man["scenes"])

    for model in models:
        pred_dir = args.pred_path or submission_dir(spec, args.run_id, model, scannet_root=root)
        out_dir = os.path.join(paths["evaluations"], args.run_id)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, model + ".csv")

        missing = []
        if not os.path.isdir(pred_dir):
            raise FileNotFoundError(f"prediction root missing: {pred_dir}")
        for scene in scenes:
            if scene_submission_status(
                    pred_dir, scene, spec, args.run_id, model, scannet_root=root) != "complete":
                missing.append(f"pred/sidecar {scene}")
            if not os.path.isfile(os.path.join(paths["gt"], scene + ".txt")):
                missing.append(f"gt   {os.path.join(paths['gt'], scene + '.txt')}")
        if missing:
            raise FileNotFoundError(
                f"[{model}] missing files before evaluation:\n  " + "\n  ".join(missing))

        evaluator = _EVALUATOR_SCRIPTS[spec.evaluator]
        staging, pred_stage, gt_stage = _stage_eval_dirs(pred_dir, paths["gt"], scenes)
        tmp_out = out_file + ".tmp"
        print(f"[{model}] evaluating {len(scenes)} scenes -> {out_file}")
        try:
            proc = subprocess.run(
                [PYTHON, evaluator, "--pred_path", pred_stage, "--gt_path", gt_stage,
                 "--output_file", tmp_out],
                capture_output=True, text=True, check=False)
            sys.stdout.write(proc.stdout)
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr)
                raise RuntimeError(f"evaluator failed for {model} (exit {proc.returncode})")
            read_evaluator_csv(tmp_out)
            os.replace(tmp_out, out_file)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if os.path.isfile(tmp_out):
                os.remove(tmp_out)


if __name__ == "__main__":
    sub_cmds = {
        "export-gt": export_gt_cli,
        "evaluate": evaluate_cli,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in sub_cmds:
        print("usage: python spellbook/evaluation/evaluate.py {export-gt,evaluate} [options]")
        sys.exit(2)
    sub_cmds[sys.argv[1]](sys.argv[2:])
