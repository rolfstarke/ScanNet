"""Flat per-vertex GT export + benchmark evaluation dispatch (Phase 5).

- export-gt: writes per-vertex GT (label_id*1000 + instance_id, one line per
  _vh_clean_2.ply vertex) filtered to the benchmark's valid class ids.
- evaluate: dispatches a benchmark run's submission root to the official
  ScanNet20 evaluator or the ScanNet200 evaluator; writes a per-class CSV.
- score-sidecars: writes per-scene AP sidecars plus one pooled micro-F1
  global threshold artifact per benchmark/run/model.
"""
import argparse
import hashlib
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
# Imported here (not lazily) so `utils.scan_lock` binds before
# evaluation.scannet200_evaluator appends BenchmarkScripts/ScanNet200
# (which contains its own utils.py) to sys.path.
from evaluation.runs import manifest_path as _run_manifest_path  # noqa: E402

PYTHON = "/home/rolf/anaconda3/envs/3disspellbook/bin/python"
_LABEL_MAP_FALLBACK = "/data/scannet/v2/scannetv2-labels.combined.tsv"

SIDECAR_SCHEMA = 4
SIDECAR_METRIC = "scannet_instance_ap50"
AP50_THRESHOLD = 0.5
MIN_REGION_SIZE = 100
GLOBAL_THRESHOLD_SCHEMA = 1
GLOBAL_THRESHOLD_METRIC = "scannet_instance_micro_f1"
GLOBAL_THRESHOLD_IOU = 0.5

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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    try:
        instances = util_3d.read_instance_prediction_file(scene_file, submission_root)
    except (Exception, SystemExit) as exc:
        raise ValueError(f"malformed instance prediction file: {scene_file}") from exc
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


def _validate_eligible_gt_by_class(doc, label_set):
    from evaluation.benchmark import BENCHMARKS

    spec = BENCHMARKS.get(label_set)
    if spec is None:
        return None
    counts = doc.get("eligible_gt_by_class")
    if not isinstance(counts, dict):
        return None
    valid = set(spec.class_labels)
    out = {}
    for name, count in counts.items():
        if name not in valid:
            return None
        if type(count) is not int or count < 0:
            return None
        out[name] = count
    if set(out) != valid:
        return None
    return out


def _validate_matched_gt(doc, eligible_gt_by_class):
    verdicts = doc.get("verdicts")
    matched = doc.get("matched_gt")
    if not isinstance(verdicts, dict) or not isinstance(matched, dict):
        return None
    out = {}
    seen_gt = set()
    for key, gid in matched.items():
        if verdicts.get(key) != "tp":
            return None
        if type(gid) is not int or gid < 1000:
            return None
        if gid in seen_gt:
            return None
        seen_gt.add(gid)
        out[key] = gid
    if len(out) != sum(1 for v in verdicts.values() if v == "tp"):
        return None
    return out


def load_score_sidecar(path, scene_id, label_set, run_id, model,
                       index_sha256=None, gt_sha256=None):
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
        stored_index = doc.get("index_sha256")
        stored_gt = doc.get("gt_sha256")
        if not isinstance(stored_index, str) or len(stored_index) != 64:
            return None
        if not isinstance(stored_gt, str) or len(stored_gt) != 64:
            return None
        if index_sha256 is not None and stored_index != index_sha256:
            return None
        if gt_sha256 is not None and stored_gt != gt_sha256:
            return None
        tp = int(doc["tp"])
        gt = int(doc["gt"])
        if tp < 0 or gt < 0 or tp > gt:
            return None
        verdicts = doc.get("verdicts") or {}
        if not isinstance(verdicts, dict):
            return None
        raw_ap = doc.get("ap")
        if raw_ap is None:
            scene_ap = None
        else:
            import math as _math

            scene_ap = float(raw_ap)
            if not (_math.isfinite(scene_ap) and 0.0 <= scene_ap <= 1.0):
                return None
        ap_class_count = int(doc.get("ap_class_count", -1))
        if ap_class_count < 0:
            return None
        if scene_ap is None and ap_class_count != 0:
            return None
        eligible_gt_by_class = _validate_eligible_gt_by_class(doc, label_set)
        if eligible_gt_by_class is None:
            return None
        if sum(eligible_gt_by_class.values()) != gt:
            return None
        matched_gt = _validate_matched_gt(doc, eligible_gt_by_class)
        if matched_gt is None:
            return None
        return {
            "tp": tp,
            "gt": gt,
            "verdicts": verdicts,
            "matched_gt": matched_gt,
            "eligible_gt_by_class": eligible_gt_by_class,
            "ap": scene_ap,
            "ap_class_count": int(ap_class_count),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def score_prediction_scene(submission_root, scene_id, spec, run_id, model, scannet_root=None):
    gt_file = os.path.join(artifact_paths(spec, scannet_root)["gt"], scene_id + ".txt")
    if not os.path.isfile(gt_file):
        return None
    pred_instances = load_pred_instances(submission_root, scene_id)
    if pred_instances is None:
        return None
    from evaluation.scannet200_evaluator import scene_evaluation_summary
    summary = scene_evaluation_summary(
        util_3d.load_ids(gt_file), pred_instances, spec, scene_id)
    verdicts = {}
    for key, value in summary["verdicts"].items():
        verdicts[_relative_mask(key, submission_root)] = value
    matched_gt = {}
    for key, gid in summary["matched_gt"].items():
        matched_gt[_relative_mask(key, submission_root)] = int(gid)
    return {
        "schema": SIDECAR_SCHEMA,
        "metric": SIDECAR_METRIC,
        "iou_threshold": AP50_THRESHOLD,
        "min_region_size": MIN_REGION_SIZE,
        "scene_id": scene_id,
        "label_set": spec.name,
        "run_id": run_id,
        "model": model,
        "index_sha256": _sha256_file(os.path.join(submission_root, scene_id + ".txt")),
        "gt_sha256": _sha256_file(gt_file),
        "tp": int(summary["tp"]),
        "gt": int(summary["gt"]),
        "verdicts": verdicts,
        "matched_gt": matched_gt,
        "eligible_gt_by_class": {k: int(v) for k, v in summary["eligible_gt_by_class"].items()},
        "ap": None if summary["ap"] is None else float(summary["ap"]),
        "ap_class_count": int(summary["ap_class_count"]),
    }


def score_and_write_sidecar(submission_root, scene_id, spec, run_id, model, scannet_root=None):
    payload = score_prediction_scene(
        submission_root, scene_id, spec, run_id, model, scannet_root=scannet_root)
    if payload is None:
        return None
    write_score_sidecar(
        score_sidecar_path(spec, run_id, model, scene_id, scannet_root=scannet_root), payload)
    return payload


def global_threshold_path(spec, run_id, model, scannet_root=None):
    return os.path.join(artifact_paths(spec, scannet_root)["evaluations"],
                        run_id, model, "thresholds.f1.json")


def _global_threshold_doc(spec, run_id, model, scenes, manifest_sha256,
                          sidecar_sha256, thresholds):
    return {
        "schema": GLOBAL_THRESHOLD_SCHEMA,
        "metric": GLOBAL_THRESHOLD_METRIC,
        "threshold_objective": "f1",
        "threshold_iou": float(GLOBAL_THRESHOLD_IOU),
        "min_region_size": MIN_REGION_SIZE,
        "benchmark": spec.name,
        "label_set": spec.name,
        "run_id": run_id,
        "model": model,
        "scenes": list(scenes),
        "manifest_sha256": manifest_sha256,
        "scene_sidecar_sha256": dict(sidecar_sha256),
        "thresholds": {
            name: {
                "confidence": float(entry["confidence"]),
                "f1": float(entry["f1"]),
                "tp": int(entry["tp"]),
                "fp": int(entry["fp"]),
                "fn": int(entry["fn"]),
                "gt": int(entry["gt"]),
            }
            for name, entry in thresholds.items()
        },
    }


def write_global_thresholds(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_global_thresholds(path, spec, run_id, model, scenes,
                           manifest_sha256=None, sidecar_sha256=None):
    """Load a pooled micro-F1 threshold artifact, or None when stale.

    Rejects wrong schema/metric/identity, a scene tuple that is not exactly
    ``scenes``, unknown classes, non-finite or out-of-range values, and any
    manifest/sidecar hash mismatch.
    """
    import math as _math

    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
        if int(doc.get("schema", -1)) != GLOBAL_THRESHOLD_SCHEMA:
            return None
        if doc.get("metric") != GLOBAL_THRESHOLD_METRIC:
            return None
        if doc.get("threshold_objective") != "f1":
            return None
        if float(doc.get("threshold_iou")) != float(GLOBAL_THRESHOLD_IOU):
            return None
        if int(doc.get("min_region_size")) != MIN_REGION_SIZE:
            return None
        if doc.get("benchmark") != spec.name or doc.get("label_set") != spec.name:
            return None
        if doc.get("run_id") != run_id or doc.get("model") != model:
            return None
        if list(doc.get("scenes") or []) != list(scenes):
            return None
        if manifest_sha256 is not None and doc.get("manifest_sha256") != manifest_sha256:
            return None
        stored_hashes = doc.get("scene_sidecar_sha256") or {}
        if sidecar_sha256 is not None:
            if dict(stored_hashes) != dict(sidecar_sha256):
                return None
        valid = set(spec.class_labels)
        thresholds = doc.get("thresholds")
        if not isinstance(thresholds, dict):
            return None
        out = {}
        for name, entry in thresholds.items():
            if name not in valid or not isinstance(entry, dict):
                return None
            try:
                conf = float(entry["confidence"])
                f1 = float(entry["f1"])
                tp = int(entry["tp"])
                fp = int(entry["fp"])
                fn = int(entry["fn"])
                gt = int(entry["gt"])
            except (TypeError, ValueError, KeyError):
                return None
            if not (_math.isfinite(conf) and _math.isfinite(f1)):
                return None
            if not (0.0 <= conf <= 1.0 and 0.0 <= f1 <= 1.0):
                return None
            if min(tp, fp, fn, gt) < 0 or tp + fn != gt:
                return None
            out[name] = {"confidence": conf, "f1": f1, "tp": tp,
                         "fp": fp, "fn": fn, "gt": gt}
        return out
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def filter_population(spec, run_id, scannet_root=None):
    """Ordered protocol scenes usable for global threshold fitting.

    The run manifest's scenes intersected with the fixed 16-scene protocol
    tuple (tuple order kept). Runs predate the 16-scene protocol, so the
    population is whatever protocol scenes the run actually covers; the
    artifact records its population and the loader hash-binds it. Empty when
    the run covers no protocol scene.
    """
    from evaluation.benchmark import PREDICTION_EVALUATION_SCENES
    from evaluation.runs import load_manifest

    scenes = list(PREDICTION_EVALUATION_SCENES)
    try:
        man = load_manifest(
            _run_manifest_path(spec, run_id, scannet_root=scannet_root),
            spec=spec, run_id=run_id)
        have = set(man.get("scenes") or [])
        scenes = [s for s in scenes if s in have]
    except (OSError, ValueError):
        pass
    return scenes


def score_global_thresholds(submission_root, spec, run_id, model, scenes,
                            scannet_root=None):
    """Fit pooled micro-F1 thresholds over ``scenes`` and persist them.

    Requires every listed scene complete with a valid schema sidecar.
    Returns the fitted {class: {...}} mapping, or None when the subset is
    incomplete. Callers pass :func:`filter_population` (available protocol
    scenes, at least one); the artifact records its exact population.
    """
    from evaluation.scannet200_evaluator import compute_global_thresholds, scene_sweep_inputs

    scenes = list(scenes)
    gt_dir = artifact_paths(spec, scannet_root)["gt"]
    for scene_id in scenes:
        if scene_submission_status(
                submission_root, scene_id, spec, run_id, model,
                scannet_root=scannet_root) != "complete":
            return None
    scene_inputs = []
    sidecar_sha256 = {}
    for scene_id in scenes:
        gt_file = os.path.join(gt_dir, scene_id + ".txt")
        pred_instances = load_pred_instances(submission_root, scene_id)
        if pred_instances is None:
            return None
        scene_inputs.append(scene_sweep_inputs(
            util_3d.load_ids(gt_file), pred_instances, spec, scene_id))
        sidecar_sha256[scene_id] = _sha256_file(
            score_sidecar_path(spec, run_id, model, scene_id, scannet_root=scannet_root))
    thresholds = compute_global_thresholds(scene_inputs, spec.class_labels)
    manifest_sha256 = None
    manifest = _run_manifest_path(spec, run_id, scannet_root=scannet_root)
    if os.path.isfile(manifest):
        manifest_sha256 = _sha256_file(manifest)
    doc = _global_threshold_doc(spec, run_id, model, scenes, manifest_sha256,
                                sidecar_sha256, thresholds)
    from utils.scan_lock import exclusive_lock
    lock_path = os.path.join(os.path.dirname(
        global_threshold_path(spec, run_id, model, scannet_root=scannet_root)),
        ".thresholds.lock")
    with exclusive_lock(lock_path):
        write_global_thresholds(
            global_threshold_path(spec, run_id, model, scannet_root=scannet_root), doc)
    return thresholds


def _ply_vertex_count(path):
    with open(path, "rb") as f:
        for raw in f:
            if raw.startswith(b"element vertex"):
                return int(raw.split()[2])
            if raw.startswith(b"end_header"):
                break
    raise ValueError(f"no vertex count in {path}")


def scene_submission_status(submission_root, scene_id, spec, run_id, model, scannet_root=None):
    scene_file = os.path.join(submission_root, scene_id + ".txt")
    if not os.path.isfile(scene_file):
        return "missing"
    try:
        instances = load_pred_instances(submission_root, scene_id)
    except (Exception, SystemExit):
        return "missing"
    if instances is None:
        return "missing"
    n_verts = None
    root = scannet_root
    if root is None:
        try:
            root = load_settings()["scannet_root"]
        except (OSError, KeyError, TypeError, ValueError):
            root = None
    if root:
        ply = os.path.join(root, "scans", scene_id, f"{scene_id}_vh_clean_2.ply")
        if not os.path.isfile(ply):
            return "missing"
        try:
            n_verts = _ply_vertex_count(ply)
        except (OSError, ValueError, IndexError):
            return "missing"
    for mask_file, prediction in instances.items():
        mask = prediction.get("pred_mask")
        if mask is None or mask.size == 0:
            return "missing"
        if n_verts is not None and mask.size != n_verts:
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
    gt_file = None
    if root:
        gt_file = os.path.join(artifact_paths(spec, root)["gt"], scene_id + ".txt")
    sidecar = load_score_sidecar(
        score_sidecar_path(spec, run_id, model, scene_id, scannet_root=scannet_root),
        scene_id, spec.name, run_id, model,
        index_sha256=_sha256_file(scene_file),
        gt_sha256=_sha256_file(gt_file) if gt_file and os.path.isfile(gt_file) else None)
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
        instances = load_pred_instances(pred_dir, scene)
        if instances is None:
            raise ValueError(f"malformed or missing prediction index: {src_pred}")
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


def score_sidecars_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Write per-scene AP sidecars plus pooled global F1 thresholds")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--models", required=True, help="comma-separated model names")
    ap.add_argument("--scenes", required=True,
                    help="comma-separated scene ids (0568_00 or scene0568_00)")
    ap.add_argument("--benchmark", default=None, help="ScanNet20 | ScanNet200 (default: settings)")
    ap.add_argument("--scannet-root", default=None)
    ap.add_argument("--missing-only", action="store_true")
    args = ap.parse_args(argv)
    spec = resolve_benchmark(args.benchmark)
    root = _ensure_settings(args.scannet_root)
    models = [m.strip() for m in args.models.split(",")]
    for model in models:
        if not model:
            raise ValueError(f"empty entry in --models {args.models!r} (trailing comma?)")
    scenes = _parse_list(args.scenes)
    for model in models:
        pred_dir = submission_dir(spec, args.run_id, model, scannet_root=root)
        for scene in scenes:
            if args.missing_only and scene_submission_status(
                    pred_dir, scene, spec, args.run_id, model, scannet_root=root) == "complete":
                print(f"[skip] {scene} {model}")
                continue
            payload = score_and_write_sidecar(
                pred_dir, scene, spec, args.run_id, model, scannet_root=root)
            if payload is None:
                print(f"[miss] {scene} {model}")
            else:
                print(f"[ok] {scene} {model} TP/GT {payload['tp']}/{payload['gt']}")
        population = filter_population(spec, args.run_id, scannet_root=root)
        if not population:
            print(f"[thresholds-skip] {model} (run covers no protocol scene)")
            continue
        fitted = score_global_thresholds(
            pred_dir, spec, args.run_id, model, population, scannet_root=root)
        if fitted is None:
            print(f"[thresholds-miss] {model} (need {len(population)} scenes complete: "
                  f"{','.join(population)})")
        else:
            print(f"[thresholds-ok] {model} {len(fitted)} classes "
                  f"over {len(population)} scenes")


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
        if os.path.isfile(out_file):
            raise ValueError(f"run_id {args.run_id!r} already evaluated for {model}: {out_file}")

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
        "score-sidecars": score_sidecars_cli,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in sub_cmds:
        print("usage: python spellbook/evaluation/evaluate.py "
              "{export-gt,evaluate,score-sidecars} [options]")
        sys.exit(2)
    sub_cmds[sys.argv[1]](sys.argv[2:])
