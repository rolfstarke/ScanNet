"""Prediction-run manifests and cross-run ranking from official evaluator CSVs."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys

from evaluation.benchmark import (
    PREDICTION_EVALUATION_SCENES, PREDICTION_METHODS, artifact_paths,
    load_settings, normalize_scene_id, resolve_benchmark, submission_dir,
    validate_prediction_methods, validate_prediction_scenes,
)
from utils.scan_lock import exclusive_lock, prediction_index_lock_path

MANIFEST_SCHEMA = 2
MANIFEST_NAME = "run.json"
RANKING_NAME = "ranking.csv"
METRICS = ("ap", "ap50", "ap25")
CSV_COLUMNS = ("class", "class id", "ap", "ap50", "ap25")
RANKING_COLUMNS = (
    "benchmark", "method", "run_id", "scenes", "scene_count",
    "ap", "ap50", "ap25", "rank_ap", "rank_ap50", "rank_ap25",
    "parameters", "issue",
)
_TOP_KEYS = frozenset(("schema", "benchmark", "run_id", "scenes", "methods", "issue"))
_METHOD_KEYS = frozenset(("parameters", "provenance"))
_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_SPELLBOOK)
_WRAPPERS = {
    "mosaic3d": os.path.join("predict", "models", "_mosaic3d_run.py"),
    "openins3d": os.path.join("predict", "models", "_openins3d_run.py"),
    "openyolo3d": os.path.join("predict", "models", "_openyolo3d_run.py"),
    "open3dis": os.path.join("predict", "models", "_open3dis_run.py"),
    "openmask3d": os.path.join("predict", "models", "_openmask3d_run.py"),
}
_EXTERNAL_REPOS = {
    "mosaic3d": "/home/rolf/GIT/Mosaic3D",
    "openins3d": "/home/rolf/GIT/OpenIns3D",
    "openyolo3d": "/home/rolf/GIT/OpenYOLO3D",
    "open3dis": "/home/rolf/GIT/Open3DIS",
    "openmask3d": "/home/rolf/GIT/openmask3d",
}
_REQUIRED_EXTERNAL = {
    "mosaic3d": (
        "/home/rolf/GIT/Mosaic3D/scripts/run_custom_scene.py",
        "/data/mosaic3d/ckpts/spunet34c.ckpt",
    ),
    "openins3d": ("/home/rolf/GIT/OpenIns3D/third_party/scannet200_val.ckpt",),
    "openyolo3d": (
        "/home/rolf/GIT/OpenYOLO3D/pretrained/config_scannet200.yaml",
        "/home/rolf/GIT/OpenYOLO3D/pretrained/checkpoints/scannet200_val.ckpt",
        "/home/rolf/GIT/OpenYOLO3D/pretrained/checkpoints/"
        "yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth",
    ),
    "open3dis": (
        "/home/rolf/GIT/Open3DIS/configs/ov3dis_scene4.yaml",
        "/home/rolf/GIT/Open3DIS/open3dis/dataset/ov3dis_loader.py",
    ),
    "openmask3d": (
        "/data/openmask3d/resources/scannet200_model.ckpt",
        "/data/openmask3d/resources/sam_vit_h_4b8939.pth",
    ),
}


def manifest_path(spec, run_id, scannet_root=None):
    return os.path.join(artifact_paths(spec, scannet_root)["evaluations"], run_id, MANIFEST_NAME)


def ranking_path(spec, scannet_root=None):
    return os.path.join(artifact_paths(spec, scannet_root)["evaluations"], RANKING_NAME)


def _atomic_write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _dump(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n"


def _compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def normalize_run_id(run_id):
    if not isinstance(run_id, str) or not run_id or run_id.strip() != run_id:
        raise ValueError(f"invalid run_id {run_id!r}")
    if os.sep in run_id or (os.altsep and os.altsep in run_id) or run_id in (".", ".."):
        raise ValueError(f"invalid run_id {run_id!r}")
    return run_id


def normalize_scenes(scenes, require_complete=False):
    out = validate_prediction_scenes(scenes, require_complete=require_complete)
    if require_complete:
        return list(PREDICTION_EVALUATION_SCENES)
    return sorted(out)


def normalize_issue(issue):
    if issue is None:
        return None
    if isinstance(issue, bool) or not isinstance(issue, int) or issue <= 0:
        raise ValueError(f"issue must be a positive integer, got {issue!r}")
    return issue


def _require_object(value, what):
    if type(value) is not dict:
        raise ValueError(f"{what} must be a JSON object")
    return value


def _json_object(value, what):
    _require_object(value, what)
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not JSON-serializable") from exc
    return value


def normalize_run_parameters(data, methods):
    data = _require_object(data, "run parameters")
    methods = list(methods)
    extra = [key for key in data if key not in methods]
    if extra:
        raise ValueError(f"run parameters include unknown method(s): {extra}")
    out = {}
    for method in methods:
        if method not in data:
            out[method] = {}
            continue
        out[method] = _json_object(data[method], f"parameters for {method}")
    return out


def load_run_parameters(path, methods):
    with open(path) as f:
        data = json.load(f)
    return normalize_run_parameters(data, methods)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path):
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"cannot read git HEAD for {path}: {result.stderr.strip()}")
    return result.stdout.strip()


def collect_method_provenance(method):
    method = validate_prediction_methods([method])[0]
    wrapper = os.path.join(_SPELLBOOK, _WRAPPERS[method])
    if not os.path.isfile(wrapper):
        raise ValueError(f"wrapper missing: {wrapper}")
    repo = _EXTERNAL_REPOS[method]
    resources = []
    for path in _REQUIRED_EXTERNAL.get(method, ()):
        if not os.path.isfile(path):
            raise ValueError(f"{method}: required source/checkpoint missing: {path}")
        resources.append({
            "path": path,
            "size": os.path.getsize(path),
            "sha256": _sha256_file(path),
        })
    return {
        "scannet_commit": _git_head(_REPO_ROOT),
        "wrapper": os.path.relpath(wrapper, _REPO_ROOT),
        "wrapper_sha256": _sha256_file(wrapper),
        "external_path": repo,
        "external_commit": _git_head(repo),
        "resources": resources,
    }


def _normalize_methods(methods, run_parameters, provenance=None):
    names = validate_prediction_methods(methods)
    if len(names) != 1:
        raise ValueError("a prediction run must contain exactly one method")
    params = normalize_run_parameters(run_parameters or {}, names)
    out = {}
    for method in names:
        body = {"parameters": params[method]}
        if provenance is not None and method in provenance:
            body["provenance"] = _json_object(provenance[method], f"provenance for {method}")
        out[method] = body
    return out


def build_manifest(benchmark, run_id, scenes, methods, run_parameters=None, issue=None,
                   provenance=None):
    spec = resolve_benchmark(benchmark)
    doc = {
        "schema": MANIFEST_SCHEMA,
        "benchmark": spec.name,
        "run_id": normalize_run_id(run_id),
        "scenes": normalize_scenes(scenes),
        "methods": _normalize_methods(methods, run_parameters, provenance=provenance),
    }
    issue = normalize_issue(issue)
    if issue is not None:
        doc["issue"] = issue
    return doc


def validate_manifest(doc, spec=None, run_id=None):
    doc = _require_object(doc, "manifest")
    extra = [key for key in doc if key not in _TOP_KEYS]
    if extra:
        raise ValueError(f"manifest has unknown key(s): {extra}")
    if type(doc.get("schema")) is not int or doc.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    resolved = resolve_benchmark(doc.get("benchmark"))
    if spec is not None and resolved.name != spec.name:
        raise ValueError(f"manifest benchmark {resolved.name!r} does not match {spec.name!r}")
    if spec is None:
        spec = resolved
    elif doc.get("benchmark") != spec.name:
        raise ValueError(f"manifest benchmark {doc.get('benchmark')!r} does not match {spec.name!r}")
    doc_run = normalize_run_id(doc.get("run_id"))
    if run_id is not None and doc_run != run_id:
        raise ValueError(f"manifest run_id {doc_run!r} does not match directory {run_id!r}")
    scenes = normalize_scenes(doc.get("scenes"))
    methods_doc = _require_object(doc.get("methods"), "methods")
    if not methods_doc:
        raise ValueError("methods must be a non-empty object")
    methods = {}
    for method, body in methods_doc.items():
        body = _require_object(body, f"methods.{method}")
        extra_method = [key for key in body if key not in _METHOD_KEYS]
        if extra_method:
            raise ValueError(f"methods.{method} has unknown key(s): {extra_method}")
        entry = {
            "parameters": _json_object(body.get("parameters"), f"parameters for {method}"),
        }
        if "provenance" in body:
            entry["provenance"] = _json_object(body.get("provenance"), f"provenance for {method}")
        methods[method] = entry
    issue = normalize_issue(doc.get("issue"))
    return build_manifest(
        spec.name, doc_run, scenes, list(methods),
        {name: body["parameters"] for name, body in methods.items()},
        issue,
        provenance={name: body["provenance"] for name, body in methods.items()
                    if "provenance" in body} or None)


def load_manifest(path, spec=None, run_id=None):
    with open(path) as f:
        doc = json.load(f)
    return validate_manifest(doc, spec=spec, run_id=run_id)


def write_run_manifest(spec, run_id, scenes, methods, run_parameters=None, issue=None,
                       scannet_root=None, provenance=None, collect_provenance=False):
    spec = resolve_benchmark(spec.name if hasattr(spec, "name") else spec)
    run_id = normalize_run_id(run_id)
    if provenance is None and collect_provenance:
        provenance = {method: collect_method_provenance(method) for method in methods}
    doc = build_manifest(spec.name, run_id, scenes, methods, run_parameters, issue,
                         provenance=provenance)
    path = manifest_path(spec, run_id, scannet_root)
    text = _dump(doc)
    with exclusive_lock(prediction_index_lock_path(scannet_root)):
        if os.path.isfile(path):
            existing = load_manifest(path, spec=spec, run_id=run_id)
            if _dump(existing) != text:
                raise ValueError(f"run_id {run_id!r} already has a different manifest: {path}")
            return existing
        _atomic_write_text(path, text)
    return doc


def _parse_metric(text):
    text = (text or "").strip()
    if not text or text.lower() == "nan":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _mean(values):
    nums = [value for value in values if value is not None and math.isfinite(value)]
    if not nums:
        return None
    return sum(nums) / len(nums)


def read_evaluator_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"malformed evaluator csv: {path}")
        fields = [name.strip() for name in reader.fieldnames]
        if list(fields) != list(CSV_COLUMNS):
            raise ValueError(f"malformed evaluator csv columns {fields} in {path}")
        rows = []
        for row in reader:
            rows.append({key: (row.get(key) or "").strip() for key in CSV_COLUMNS})
    if not rows:
        raise ValueError(f"empty evaluator csv: {path}")
    metrics = {}
    for metric in METRICS:
        metrics[metric] = _mean([_parse_metric(row[metric]) for row in rows])
        if metrics[metric] is None:
            raise ValueError(f"no finite {metric} values in {path}")
    return metrics


def discover_manifests(spec, scannet_root=None):
    root = artifact_paths(spec, scannet_root)["evaluations"]
    if not os.path.isdir(root):
        return []
    found = []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        path = os.path.join(run_dir, MANIFEST_NAME)
        if not os.path.isdir(run_dir) or not os.path.isfile(path):
            continue
        found.append(load_manifest(path, spec=spec, run_id=name))
    return found


def _join_rows(spec, manifests, scannet_root=None):
    eval_root = artifact_paths(spec, scannet_root)["evaluations"]
    rows = []
    for doc in manifests:
        for method, body in doc["methods"].items():
            csv_path = os.path.join(eval_root, doc["run_id"], f"{method}.csv")
            if not os.path.isfile(csv_path):
                print(f"[WARN] skipping {doc['run_id']}/{method}: missing {csv_path}",
                      file=sys.stderr)
                continue
            try:
                metrics = read_evaluator_csv(csv_path)
            except (OSError, ValueError) as exc:
                print(f"[WARN] skipping {doc['run_id']}/{method}: {exc}", file=sys.stderr)
                continue
            rows.append({
                "benchmark": doc["benchmark"],
                "method": method,
                "run_id": doc["run_id"],
                "scenes": list(doc["scenes"]),
                "scene_count": len(doc["scenes"]),
                "ap": metrics["ap"],
                "ap50": metrics["ap50"],
                "ap25": metrics["ap25"],
                "parameters": body["parameters"],
                "issue": doc.get("issue"),
            })
    return rows


def _assign_ranks(rows):
    groups = {}
    for row in rows:
        key = (row["benchmark"], row["method"], tuple(row["scenes"]))
        groups.setdefault(key, []).append(dict(row))
    ranked = []
    for key in sorted(groups):
        members = groups[key]
        for metric in METRICS:
            if metric == "ap":
                ordered = sorted(members, key=lambda row: (-row["ap"], -row["ap50"], row["run_id"]))
            else:
                ordered = sorted(members, key=lambda row: (-row[metric], row["run_id"]))
            lookup = {row["run_id"]: index for index, row in enumerate(ordered, start=1)}
            for row in members:
                row[f"rank_{metric}"] = lookup[row["run_id"]]
        ranked.extend(members)
    ranked.sort(key=lambda row: (row["method"], ";".join(row["scenes"]),
                                 -row["ap"], -row["ap50"], row["run_id"]))
    return ranked


def _format_row(row):
    return {
        "benchmark": row["benchmark"],
        "method": row["method"],
        "run_id": row["run_id"],
        "scenes": ";".join(row["scenes"]),
        "scene_count": str(row["scene_count"]),
        "ap": format(row["ap"], ".10g"),
        "ap50": format(row["ap50"], ".10g"),
        "ap25": format(row["ap25"], ".10g"),
        "rank_ap": str(row["rank_ap"]),
        "rank_ap50": str(row["rank_ap50"]),
        "rank_ap25": str(row["rank_ap25"]),
        "parameters": _compact(row["parameters"]),
        "issue": "" if row["issue"] is None else str(row["issue"]),
    }


def write_ranking_csv(path, rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=RANKING_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_format_row(row))
    _atomic_write_text(path, buf.getvalue())


def _is_comparable(doc):
    return (doc.get("benchmark") == "ScanNet200"
            and list(doc.get("scenes") or []) == list(PREDICTION_EVALUATION_SCENES)
            and len(doc.get("methods") or {}) == 1
            and next(iter(doc["methods"])) in PREDICTION_METHODS)


def rank_runs(spec, scannet_root=None, comparable_only=False):
    manifests = discover_manifests(spec, scannet_root)
    if comparable_only:
        manifests = [doc for doc in manifests if _is_comparable(doc)]
    rows = _assign_ranks(_join_rows(spec, manifests, scannet_root))
    path = ranking_path(spec, scannet_root)
    write_ranking_csv(path, rows)
    return rows, path


def evaluator_csv_path(spec, run_id, method, scannet_root=None):
    return os.path.join(artifact_paths(spec, scannet_root)["evaluations"],
                        run_id, f"{method}.csv")


def tasks_path(spec, run_id, method, scannet_root=None):
    return os.path.join(artifact_paths(spec, scannet_root)["evaluations"],
                        run_id, f"{method}.tasks")


def run_is_evaluated(spec, run_id, method, scannet_root=None):
    return os.path.isfile(evaluator_csv_path(spec, run_id, method, scannet_root))


def prune_prediction_artifacts(spec, run_id, method, scannet_root=None, apply=False):
    spec = resolve_benchmark(spec.name if hasattr(spec, "name") else spec)
    run_id = normalize_run_id(run_id)
    method = validate_prediction_methods([method])[0]
    root = scannet_root or load_settings()["scannet_root"]
    pred_dir = os.path.realpath(submission_dir(spec, run_id, method, scannet_root=root))
    pred_root = os.path.realpath(artifact_paths(spec, root)["predictions"])
    if pred_dir == pred_root or not pred_dir.startswith(pred_root + os.sep):
        raise ValueError(f"refusing to prune outside prediction root: {pred_dir}")
    man = load_manifest(manifest_path(spec, run_id, root), spec=spec, run_id=run_id)
    if list(man["methods"]) != [method]:
        raise ValueError(f"manifest methods {list(man['methods'])} do not match {method}")
    csv_file = evaluator_csv_path(spec, run_id, method, root)
    sidecar_dir = os.path.join(artifact_paths(spec, root)["evaluations"], run_id, method)
    missing = []
    if not os.path.isfile(csv_file):
        missing.append(csv_file)
    if not os.path.isfile(tasks_path(spec, run_id, method, root)):
        missing.append(tasks_path(spec, run_id, method, root))
    for scene in man["scenes"]:
        sidecar = os.path.join(sidecar_dir, scene + ".tp50.json")
        if not os.path.isfile(sidecar):
            missing.append(sidecar)
    if missing:
        raise ValueError("refusing to prune; incomplete metadata:\n  " + "\n  ".join(missing))
    if not apply:
        return {"apply": False, "target": pred_dir}
    if os.path.isdir(pred_dir):
        shutil.rmtree(pred_dir)
    return {"apply": True, "target": pred_dir}


def _print_table(rows, metric, method=None):
    selected = [row for row in rows if method is None or row["method"] == method]
    if not selected:
        print("no completed runs")
        return
    rank_key = f"rank_{metric}"
    groups = {}
    for row in selected:
        key = (row["method"], tuple(row["scenes"]))
        groups.setdefault(key, []).append(row)
    for (group_method, scenes), members in groups.items():
        members = sorted(members, key=lambda row: (row[rank_key], row["run_id"]))
        print(f"# {members[0]['benchmark']} {group_method} scenes={';'.join(scenes)}")
        print(f"{rank_key:<10} {'run_id':<28} {'ap':>8} {'ap50':>8} {'ap25':>8}")
        for row in members:
            print(f"{row[rank_key]:<10} {row['run_id']:<28} "
                  f"{row['ap']:8.3f} {row['ap50']:8.3f} {row['ap25']:8.3f}")
        print()


def rank_cli(argv=None):
    parser = argparse.ArgumentParser(description="Rank prediction runs from manifests and CSVs")
    parser.add_argument("--benchmark", default=None, help="ScanNet20 | ScanNet200 (default: settings)")
    parser.add_argument("--method", default=None, help="optional terminal filter")
    parser.add_argument("--metric", default="ap", choices=list(METRICS))
    parser.add_argument("--scannet-root", default=None)
    args = parser.parse_args(argv)
    spec = resolve_benchmark(args.benchmark)
    root = args.scannet_root or load_settings()["scannet_root"]
    rows, path = rank_runs(spec, scannet_root=root, comparable_only=spec.name == "ScanNet200")
    _print_table(rows, args.metric, method=args.method)
    print(f"ranking -> {path}")


def prune_cli(argv=None):
    parser = argparse.ArgumentParser(description="Delete heavy prediction artifacts for one run")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--apply", action="store_true", help="actually delete; default is dry-run")
    args = parser.parse_args(argv)
    spec = resolve_benchmark(args.benchmark)
    root = args.scannet_root or load_settings()["scannet_root"]
    result = prune_prediction_artifacts(
        spec, args.run_id, args.method, scannet_root=root, apply=args.apply)
    prefix = "deleted" if result["apply"] else "dry-run"
    print(f"{prefix} -> {result['target']}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("rank", "prune"):
        print("usage: python spellbook/evaluation/runs.py {rank,prune} [options]")
        sys.exit(2)
    if argv[0] == "rank":
        rank_cli(argv[1:])
    else:
        prune_cli(argv[1:])


if __name__ == "__main__":
    main()
