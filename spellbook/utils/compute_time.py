import json
import math
import os

TIMING_SCHEMA = 1


def prediction_timing_path(spec, run_id, model, scene_id, scannet_root=None):
    from evaluation.benchmark import artifact_paths
    return os.path.join(artifact_paths(spec, scannet_root)["evaluations"],
                        run_id, model, scene_id + ".timing.json")


def reconstruction_timing_path(scene_dir):
    return os.path.join(scene_dir, "recon", "timing.json")


def format_elapsed(seconds):
    if seconds is None or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return None
    if seconds < 0:
        return None
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def write_timing(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_timing(path, kind, **identity):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            doc = json.load(fh)
        if int(doc.get("schema", -1)) != TIMING_SCHEMA:
            return None
        if doc.get("kind") != kind:
            return None
        for key, value in identity.items():
            if doc.get(key) != value:
                return None
        elapsed = float(doc["elapsed_s"])
        if not math.isfinite(elapsed) or elapsed < 0:
            return None
        return elapsed
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def write_prediction_timing(spec, run_id, model, scene_id, elapsed_s, scannet_root=None):
    write_timing(prediction_timing_path(spec, run_id, model, scene_id, scannet_root), {
        "schema": TIMING_SCHEMA,
        "kind": "prediction",
        "scene_id": scene_id,
        "run_id": run_id,
        "model": model,
        "elapsed_s": float(elapsed_s),
    })


def write_reconstruction_timing(scene_dir, scene_id, engine, elapsed_s):
    write_timing(reconstruction_timing_path(scene_dir), {
        "schema": TIMING_SCHEMA,
        "kind": "reconstruction",
        "scene_id": scene_id,
        "engine": engine,
        "elapsed_s": float(elapsed_s),
    })
