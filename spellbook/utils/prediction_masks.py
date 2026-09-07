import argparse
import os
import sys
import tempfile
import time

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_SPELLBOOK)
sys.path.insert(0, _SPELLBOOK)
sys.path.insert(0, os.path.join(_REPO_ROOT, "BenchmarkScripts"))

from evaluation.benchmark import (  # noqa: E402
    load_settings, normalize_scene_id, resolve_benchmark, submission_dir)
import util_3d  # noqa: E402

SCHEMA = 1
CACHE_REL = ("derived", "visualization", "masks")


def mask_cache_path(spec, run_id, model, scene_id, scannet_root=None):
    root = scannet_root if scannet_root is not None else load_settings()["scannet_root"]
    return os.path.join(root, *CACHE_REL, spec.name, run_id, model, scene_id + ".npz")


def packed_width(vertex_count):
    return (int(vertex_count) + 7) // 8


def pack_mask(mask):
    return np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")


def unpack_mask(packed, vertex_count):
    vertex_count = int(vertex_count)
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), bitorder="little", count=vertex_count).astype(bool)


def parse_mask_text(path, vertex_count=None):
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    if b"\r" in raw:
        raw = raw.replace(b"\r", b"")
    values = None
    if vertex_count is not None and len(raw) == 2 * vertex_count - 1 and (
            vertex_count == 1 or raw[1:2] == b"\n"):
        digits = np.frombuffer(raw[::2], dtype=np.uint8)
        if digits.size == vertex_count and not np.any((digits != 48) & (digits != 49)):
            values = digits - np.uint8(48)
    elif vertex_count is not None and len(raw) == vertex_count:
        digits = np.frombuffer(raw, dtype=np.uint8)
        if not np.any((digits != 48) & (digits != 49)):
            values = digits - np.uint8(48)
    if values is None:
        values = np.fromstring(raw + b"\n", dtype=np.uint8, sep="\n")
    if vertex_count is None:
        vertex_count = int(values.size)
    if int(values.size) != int(vertex_count):
        raise ValueError(f"{path}: expected {vertex_count} mask values, got {values.size}")
    values = np.ascontiguousarray(values, dtype=np.uint8)
    if values.size and int(values.max()) > 1:
        raise ValueError(f"{path}: mask values must be 0 or 1")
    return values


def prediction_rows(submission_root, scene_id, spec=None):
    scene_file = os.path.join(submission_root, f"{scene_id}.txt")
    if not os.path.isfile(scene_file):
        return []
    instances = util_3d.read_instance_prediction_file(scene_file, submission_root)
    root = os.path.normpath(submission_root)
    rows = []
    for mask_file, prediction in instances.items():
        label_id = prediction["label_id"]
        if spec is not None and label_id not in spec.id_to_label:
            continue
        rel = os.path.relpath(os.path.normpath(mask_file), root).replace("\\", "/")
        rows.append({
            "path": mask_file,
            "key": rel,
            "label_id": int(label_id),
            "conf": float(prediction["conf"]),
        })
    return rows


def _scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item() if hasattr(arr.reshape(-1)[0], "item") else arr.reshape(-1)[0]
    raise ValueError("expected scalar cache field")


def _require_array(doc, key, dtype=None, ndim=None):
    if key not in doc.files:
        raise ValueError(f"cache missing {key}")
    arr = doc[key]
    if arr.dtype == object:
        raise ValueError(f"cache field {key} is an object array")
    if dtype is not None and arr.dtype != dtype:
        arr = np.asarray(arr, dtype=dtype)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"cache field {key} has ndim {arr.ndim}, expected {ndim}")
    return arr


def validate_cache(doc, expected):
    if int(_scalar(doc["schema"])) != SCHEMA:
        raise ValueError("unsupported mask cache schema")
    if str(_scalar(doc["scene_id"])) != expected["scene_id"]:
        raise ValueError("cache scene mismatch")
    if str(_scalar(doc["benchmark"])) != expected["benchmark"]:
        raise ValueError("cache benchmark mismatch")
    if str(_scalar(doc["run_id"])) != expected["run_id"]:
        raise ValueError("cache run mismatch")
    if str(_scalar(doc["model"])) != expected["model"]:
        raise ValueError("cache model mismatch")
    vertex_count = int(_scalar(doc["vertex_count"]))
    if vertex_count != int(expected["vertex_count"]):
        raise ValueError("cache vertex_count mismatch")
    stored_mtime = float(_scalar(doc["source_mtime"]))
    if abs(stored_mtime - float(expected["source_mtime"])) > 1e-6:
        raise ValueError("cache source_mtime mismatch")
    keys = _require_array(doc, "keys", ndim=1)
    if keys.dtype.kind not in "SU":
        raise ValueError("cache keys must be strings")
    label_ids = _require_array(doc, "label_ids", dtype=np.int32, ndim=1)
    confs = _require_array(doc, "confs", dtype=np.float64, ndim=1)
    packed = _require_array(doc, "packed", dtype=np.uint8, ndim=2)
    n = keys.shape[0]
    if label_ids.shape[0] != n or confs.shape[0] != n or packed.shape[0] != n:
        raise ValueError("cache row counts disagree")
    if packed.shape[1] != packed_width(vertex_count):
        raise ValueError("cache packed width mismatch")
    return {
        "keys": [str(key) for key in keys.tolist()],
        "label_ids": np.ascontiguousarray(label_ids, dtype=np.int32),
        "confs": np.ascontiguousarray(confs, dtype=np.float64),
        "packed": np.ascontiguousarray(packed, dtype=np.uint8),
        "vertex_count": vertex_count,
        "source_mtime": stored_mtime,
    }


def try_load_cache(path, expected):
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as doc:
            return validate_cache(doc, expected)
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None


def write_cache_atomic(path, payload, expected):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".npz", dir=directory)
    try:
        os.close(fd)
        os.unlink(tmp)
        np.savez(
            tmp,
            schema=np.int32(SCHEMA),
            scene_id=np.asarray(expected["scene_id"]),
            benchmark=np.asarray(expected["benchmark"]),
            run_id=np.asarray(expected["run_id"]),
            model=np.asarray(expected["model"]),
            vertex_count=np.int32(expected["vertex_count"]),
            source_mtime=np.float64(expected["source_mtime"]),
            keys=np.asarray(payload["keys"]),
            label_ids=np.ascontiguousarray(payload["label_ids"], dtype=np.int32),
            confs=np.ascontiguousarray(payload["confs"], dtype=np.float64),
            packed=np.ascontiguousarray(payload["packed"], dtype=np.uint8),
        )
        os.replace(tmp, path)
    except Exception:
        for leftover in (tmp, tmp + ".npz"):
            try:
                os.unlink(leftover)
            except OSError:
                pass
        raise


def build_packed_masks(submission_root, scene_id, vertex_count, spec=None):
    rows = prediction_rows(submission_root, scene_id, spec=spec)
    width = packed_width(vertex_count)
    packed = np.empty((len(rows), width), dtype=np.uint8)
    keys = []
    label_ids = np.empty(len(rows), dtype=np.int32)
    confs = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        mask = parse_mask_text(row["path"], vertex_count)
        packed[index] = pack_mask(mask)
        keys.append(row["key"])
        label_ids[index] = row["label_id"]
        confs[index] = row["conf"]
    return {
        "keys": keys,
        "label_ids": label_ids,
        "confs": confs,
        "packed": packed,
        "vertex_count": int(vertex_count),
    }


def cache_expected(spec, run_id, model, scene_id, vertex_count, source_mtime):
    return {
        "schema": SCHEMA,
        "scene_id": scene_id,
        "benchmark": spec.name,
        "run_id": run_id,
        "model": model,
        "vertex_count": int(vertex_count),
        "source_mtime": float(source_mtime),
    }


def load_or_build_packed_masks(
        submission_root, scene_id, vertex_count, spec, run_id, model,
        source_mtime, scannet_root=None, force_rebuild=False):
    expected = cache_expected(spec, run_id, model, scene_id, vertex_count, source_mtime)
    path = mask_cache_path(spec, run_id, model, scene_id, scannet_root=scannet_root)
    if not force_rebuild:
        loaded = try_load_cache(path, expected)
        if loaded is not None:
            return loaded, path, False
    payload = build_packed_masks(submission_root, scene_id, vertex_count, spec=spec)
    write_cache_atomic(path, payload, expected)
    payload["source_mtime"] = float(source_mtime)
    return payload, path, True


def merge_packed_overlay(points, objects, fp_color, tp_color):
    points = np.asarray(points)
    vertex_count = len(points)
    width = packed_width(vertex_count)
    fp_rows = []
    tp_rows = []
    for obj in objects:
        verdict = obj.get("verdict")
        row = np.asarray(obj["packed"], dtype=np.uint8)
        if row.shape != (width,):
            raise ValueError("packed mask width mismatch")
        if verdict == "fp":
            fp_rows.append(row)
        elif verdict == "tp":
            tp_rows.append(row)
    zeros = np.zeros(width, dtype=np.uint8)
    fp_union = np.bitwise_or.reduce(np.stack(fp_rows), axis=0) if fp_rows else zeros
    tp_union = np.bitwise_or.reduce(np.stack(tp_rows), axis=0) if tp_rows else zeros
    fp_mask = unpack_mask(fp_union, vertex_count)
    tp_mask = unpack_mask(tp_union, vertex_count)
    keep = fp_mask | tp_mask
    colors = np.zeros((vertex_count, 3), dtype=float)
    colors[fp_mask] = np.asarray(fp_color, dtype=float)
    colors[tp_mask] = np.asarray(tp_color, dtype=float)
    return points[keep], colors[keep]


def _parse_list(values):
    entries = [item for item in values.split(",")]
    for entry in entries:
        if not entry.strip():
            raise ValueError(f"empty entry in comma-separated list {values!r} (trailing comma?)")
    return [normalize_scene_id(entry) for entry in entries]


def _root(scannet_root):
    return scannet_root if scannet_root is not None else load_settings()["scannet_root"]


def _models(text):
    models = [item.strip() for item in text.split(",")]
    for model in models:
        if not model:
            raise ValueError(f"empty entry in --models {text!r} (trailing comma?)")
    return models


def _vertex_count_for(submission_root, scene_id, spec):
    rows = prediction_rows(submission_root, scene_id, spec=spec)
    if not rows:
        raise FileNotFoundError(f"no prediction masks for {scene_id} in {submission_root}")
    return int(parse_mask_text(rows[0]["path"]).size)


def build_cli(argv=None):
    from utils.visualize import prediction_artifact_mtime

    parser = argparse.ArgumentParser(description="Build derived packed prediction-mask caches")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", required=True, help="comma-separated model names")
    parser.add_argument("--scenes", required=True,
                        help="comma-separated scene ids (0568_00 or scene0568_00)")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--missing-only", action="store_true")
    args = parser.parse_args(argv)
    spec = resolve_benchmark(args.benchmark)
    root = _root(args.scannet_root)
    models = _models(args.models)
    scenes = _parse_list(args.scenes)
    for model in models:
        pred_dir = submission_dir(spec, args.run_id, model, scannet_root=root)
        for scene in scenes:
            path = mask_cache_path(spec, args.run_id, model, scene, scannet_root=root)
            if args.missing_only and os.path.isfile(path):
                print(f"[skip] {scene} {model}")
                continue
            mtime = prediction_artifact_mtime(pred_dir, scene)
            if mtime is None:
                print(f"[miss] {scene} {model}")
                continue
            vertex_count = _vertex_count_for(pred_dir, scene, spec)
            _, path, built = load_or_build_packed_masks(
                pred_dir, scene, vertex_count, spec, args.run_id, model, mtime,
                scannet_root=root, force_rebuild=not args.missing_only)
            print(f"[{'ok' if built else 'hit'}] {scene} {model} -> {path}")


def benchmark_cli(argv=None):
    from utils.visualize import prediction_artifact_mtime

    parser = argparse.ArgumentParser(description="Time packed mask cache load/build")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--cold", action="store_true")
    args = parser.parse_args(argv)
    spec = resolve_benchmark(args.benchmark)
    root = _root(args.scannet_root)
    scene = normalize_scene_id(args.scene)
    pred_dir = submission_dir(spec, args.run_id, args.model, scannet_root=root)
    mtime = prediction_artifact_mtime(pred_dir, scene)
    if mtime is None:
        raise FileNotFoundError(f"no prediction artifacts for {scene} in {pred_dir}")
    vertex_count = _vertex_count_for(pred_dir, scene, spec)
    path = mask_cache_path(spec, args.run_id, args.model, scene, scannet_root=root)
    if args.cold and os.path.isfile(path):
        os.unlink(path)
    t0 = time.perf_counter()
    payload, path, built = load_or_build_packed_masks(
        pred_dir, scene, vertex_count, spec, args.run_id, args.model, mtime,
        scannet_root=root, force_rebuild=False)
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    positives = 0
    for row in payload["packed"]:
        positives += int(unpack_mask(row, payload["vertex_count"]).sum())
    unpack_s = time.perf_counter() - t1
    source_bytes = 0
    scene_file = os.path.join(pred_dir, scene + ".txt")
    if os.path.isfile(scene_file):
        source_bytes += os.path.getsize(scene_file)
    for rel in payload["keys"]:
        full = os.path.join(pred_dir, rel)
        if os.path.isfile(full):
            source_bytes += os.path.getsize(full)
    cache_bytes = os.path.getsize(path) if os.path.isfile(path) else 0
    print(
        f"scene={scene} model={args.model} masks={len(payload['keys'])} "
        f"vertices={payload['vertex_count']} built={int(built)} "
        f"source_bytes={source_bytes} cache_bytes={cache_bytes} "
        f"load_s={load_s:.4f} unpack_s={unpack_s:.4f} total_s={load_s + unpack_s:.4f} "
        f"positives={positives} path={path}")


if __name__ == "__main__":
    commands = {"build": build_cli, "benchmark": benchmark_cli}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("usage: python spellbook/utils/prediction_masks.py {build,benchmark} [options]")
        sys.exit(2)
    commands[sys.argv[1]](sys.argv[2:])
