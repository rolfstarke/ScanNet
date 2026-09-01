import json
import math
import os
import re
import sys
import time
from collections import Counter, OrderedDict

import numpy as np
from matplotlib import colormaps

_BENCHMARKSCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "BenchmarkScripts")
sys.path.insert(0, _BENCHMARKSCRIPTS)
_SPELLBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _SPELLBOOK)
import util  # noqa: E402
import util_3d  # noqa: E402
from evaluation.benchmark import BENCHMARKS, artifact_paths, resolve_benchmark, submission_dir  # noqa: E402
from evaluation.evaluate import load_score_sidecar, score_sidecar_path  # noqa: E402
from evaluation.runs import _is_comparable, load_manifest, read_evaluator_csv  # noqa: E402
from utils import hud  # noqa: E402
from utils.compute_time import (  # noqa: E402
    load_timing, prediction_timing_path, reconstruction_timing_path)
from utils.prediction_masks import (  # noqa: E402
    load_or_build_packed_masks, merge_packed_overlay, unpack_mask)


DEFAULT_SCANNET_DIR = "/data/scannet/scans"
LABEL_MAP_FILE = "/data/scannet/v2/scannetv2-labels.combined.tsv"
_SCENE_ID_RE = re.compile(r"^(scene\d{4})_(\d{2})$")
_GEOMETRY_REF = os.path.join(_SPELLBOOK, "reconstruct", "geometry_reference.yaml")

_PALETTE = colormaps["tab20"].colors
TARGET_LABEL_PX = 14.0
TP_COLOR = np.array([0.10, 0.80, 0.20], dtype=float)
FP_COLOR = np.array([0.90, 0.15, 0.15], dtype=float)
MIN_REGION_SIZE = 100
AP50_THRESHOLD = 0.5

SOURCE_GT = "ground_truth"
SOURCE_SCENE = "scene_only"
SOURCE_PRED = "prediction"
COLOR_CLASS = "class"
COLOR_INSTANCE = "instance"
COLOR_TPGT = "tp_gt"
KIND_SCENE = "scene"
KIND_SCAN = "scan"
KIND_METHOD = "method"
KIND_RUN = "run"
FOCUS_TREE = "tree"
FOCUS_SETTINGS = "settings"

KEY_TAB = 258
KEY_ENTER = 257
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265
KEY_ESCAPE = 256
KEY_PRESS = 1
LABEL_UPDATE_INTERVAL = 0.1
LABEL_CAMERA_ATOL = 1e-5

SETTING_ROWS = ("benchmark", "mode", "color", "geometry", "boxes", "ceiling", "cad")
SETTING_NAMES = {
    "benchmark": "Label set",
    "mode": "Mode",
    "color": "Color",
    "geometry": "Geometry",
    "boxes": "Boxes",
    "ceiling": "Ceiling",
    "cad": "CAD report",
}


def _class_color(name, nyu40map, palette, _cache={}):
    if name not in _cache:
        nyu40id = nyu40map.get(name)
        if nyu40id is not None and nyu40id < len(palette):
            _cache[name] = np.array(palette[nyu40id], dtype=float) / 255.0
        else:
            _cache[name] = _PALETTE[len(_cache) % len(_PALETTE)]
    return _cache[name]


def _detect_up_axis(pts):
    lo, hi = np.percentile(pts, [5, 95], axis=0)
    return int(np.argmin(hi - lo))


def _below_height(geom, up_axis, max_val):
    box = geom.get_axis_aligned_bounding_box()
    max_bound = list(box.max_bound)
    max_bound[up_axis] = max_val
    box.max_bound = max_bound
    return geom.crop(box)


def _label_base(text, color):
    import open3d as o3d
    t_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0)
    raw_width = float(t_mesh.get_axis_aligned_bounding_box().max_bound[0].item())
    mesh = t_mesh.to_legacy()
    mesh.paint_uniform_color(color)
    base_vertices = np.asarray(mesh.vertices) - np.array([raw_width / 2, 0, 0])
    return mesh, base_vertices


def _place_label(mesh, base_vertices, position, char_size, x_dir, y_dir):
    import open3d as o3d
    z_dir = np.cross(x_dir, y_dir)
    rotation = np.column_stack([x_dir, y_dir, z_dir])
    mesh.vertices = o3d.utility.Vector3dVector((base_vertices * char_size) @ rotation.T + position)
    return mesh


def load_label_map(label_map_file=LABEL_MAP_FILE):
    return util.read_label_mapping(label_map_file, label_from="raw_category", label_to="nyu40id")


def load_gt_instances(scene_dir, points=None, colors=None):
    scene_id = os.path.basename(scene_dir.rstrip("/"))
    mesh_file = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")
    agg_file = os.path.join(scene_dir, f"{scene_id}.aggregation.json")
    seg_file = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json")
    if not all(os.path.isfile(f) for f in (agg_file, seg_file)):
        return None
    if points is None:
        if not os.path.isfile(mesh_file):
            return None
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        points = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
    with open(agg_file) as f:
        agg = json.load(f)
    with open(seg_file) as f:
        segs = json.load(f)
    seg_indices = np.asarray(segs["segIndices"], dtype=int)
    if len(seg_indices) != len(points):
        return None
    objects = []
    for group in agg["segGroups"]:
        mask = np.isin(seg_indices, group["segments"])
        if not mask.any():
            continue
        objects.append({
            "points": points[mask],
            "colors": colors[mask],
            "class_name": group["label"],
            "sel": mask,
        })
    return {"objects": objects, "scene_points": points, "scene_colors": colors}


def load_predictions(submission_root, scene_id, points, colors, spec,
                     run_id=None, model=None, scannet_root=None, source_mtime=None):
    scene_file = os.path.join(submission_root, f"{scene_id}.txt")
    if not os.path.isfile(scene_file):
        return None
    packed = None
    if (run_id is not None and model is not None and scannet_root is not None
            and source_mtime is not None):
        try:
            packed, _, _ = load_or_build_packed_masks(
                submission_root, scene_id, len(points), spec, run_id, model,
                source_mtime, scannet_root=scannet_root)
        except (OSError, ValueError, KeyError):
            packed = None
    objects = []
    pred_instances = {}
    if packed is not None:
        root = os.path.normpath(submission_root)
        for key, label_id, conf, row in zip(
                packed["keys"], packed["label_ids"], packed["confs"], packed["packed"]):
            if int(label_id) not in spec.id_to_label:
                continue
            mask = unpack_mask(row, packed["vertex_count"])
            abs_path = os.path.normpath(os.path.join(root, key))
            objects.append({
                "points": points[mask],
                "colors": colors[mask],
                "class_name": spec.id_to_label[int(label_id)],
                "score": float(conf),
                "sel": mask,
                "key": key,
                "packed": np.ascontiguousarray(row, dtype=np.uint8),
            })
            pred_instances[abs_path] = {
                "label_id": int(label_id),
                "conf": float(conf),
                "pred_mask": mask,
            }
        return {"objects": objects, "scene_points": points, "scene_colors": colors,
                "pred_instances": pred_instances, "submission_root": submission_root}
    instances = util_3d.read_instance_prediction_file(scene_file, submission_root)
    for mask_file, prediction in instances.items():
        label_id = prediction["label_id"]
        if label_id not in spec.id_to_label:
            continue
        mask = util_3d.load_ids(mask_file) > 0
        rel = os.path.relpath(os.path.normpath(mask_file), os.path.normpath(submission_root))
        objects.append({
            "points": points[mask],
            "colors": colors[mask],
            "class_name": spec.id_to_label[label_id],
            "score": prediction["conf"],
            "sel": mask,
            "key": rel.replace("\\", "/"),
        })
        pred_instances[mask_file] = {
            "label_id": label_id,
            "conf": prediction["conf"],
            "pred_mask": mask,
        }
    return {"objects": objects, "scene_points": points, "scene_colors": colors,
            "pred_instances": pred_instances, "submission_root": submission_root}


def predictions_root(spec, scannet_root=None):
    return artifact_paths(spec, scannet_root)["predictions"]


def prediction_artifact_mtime(model_dir, scene_id):
    if not _SCENE_ID_RE.match(scene_id):
        return None
    scene_file = os.path.join(model_dir, scene_id + ".txt")
    if not os.path.isfile(scene_file):
        return None
    root = os.path.normpath(model_dir) + os.sep
    try:
        with open(scene_file) as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError:
        return None
    newest = os.path.getmtime(scene_file)
    found_mask = False
    for line in lines:
        rel = line.split()[0]
        if rel.endswith(".json"):
            continue
        full = os.path.normpath(os.path.join(model_dir, rel))
        if not full.startswith(root) or not os.path.isfile(full):
            continue
        found_mask = True
        newest = max(newest, os.path.getmtime(full))
    return newest if found_mask else None


def runs_for_scene(spec, scene_id, scannet_root=None):
    root = predictions_root(spec, scannet_root)
    if not os.path.isdir(root):
        return []
    runs = []
    for run_id in os.listdir(root):
        run_dir = os.path.join(root, run_id)
        if not os.path.isdir(run_dir):
            continue
        newest = None
        for model in os.listdir(run_dir):
            mtime = prediction_artifact_mtime(os.path.join(run_dir, model), scene_id)
            if mtime is not None:
                newest = mtime if newest is None else max(newest, mtime)
        if newest is not None:
            runs.append((run_id, newest))
    runs.sort(key=lambda entry: -entry[1])
    return [run_id for run_id, _ in runs]


def available_models(run_root, scene_id=None):
    if not os.path.isdir(run_root):
        return []
    models = []
    for name in sorted(os.listdir(run_root)):
        model_dir = os.path.join(run_root, name)
        if not os.path.isdir(model_dir):
            continue
        if scene_id is not None and prediction_artifact_mtime(model_dir, scene_id) is None:
            continue
        models.append(name)
    return models


def classify_ap50(gt_ids, pred_instances, spec, scene_id, overlap_th=AP50_THRESHOLD):
    from evaluation.scannet200_evaluator import scene_instance_summary
    return scene_instance_summary(gt_ids, pred_instances, spec, scene_id, overlap_th)


def discover_reconstructions(scannet_dir):
    if not os.path.isdir(scannet_dir):
        return []
    ids = []
    for name in os.listdir(scannet_dir):
        if not _SCENE_ID_RE.match(name):
            continue
        ply = os.path.join(scannet_dir, name, f"{name}_vh_clean_2.ply")
        if os.path.isfile(ply):
            ids.append(name)
    ids.sort(key=lambda s: (int(s[5:9]), int(s[10:12])))
    return ids


def require_scene(scene_id, recon_ids, scannet_dir):
    if scene_id not in recon_ids:
        raise FileNotFoundError(f"scene not found: {os.path.join(scannet_dir, scene_id)}")
    return scene_id


def group_scenes(recon_ids):
    groups = OrderedDict()
    for rid in recon_ids:
        groups.setdefault(rid[:9], []).append(rid)
    return groups


def index_predictions(scannet_root, recon_ids):
    out = {rid: [] for rid in recon_ids}
    recon_set = set(recon_ids)
    for spec in BENCHMARKS.values():
        root = artifact_paths(spec, scannet_root)["predictions"]
        if not os.path.isdir(root):
            continue
        for run_id in os.listdir(root):
            run_dir = os.path.join(root, run_id)
            if not os.path.isdir(run_dir):
                continue
            for model in os.listdir(run_dir):
                model_dir = os.path.join(run_dir, model)
                if not os.path.isdir(model_dir):
                    continue
                try:
                    names = os.listdir(model_dir)
                except OSError:
                    continue
                for name in names:
                    if not name.endswith(".txt"):
                        continue
                    rid = name[:-4]
                    if rid not in recon_set:
                        continue
                    mtime = prediction_artifact_mtime(model_dir, rid)
                    if mtime is None:
                        continue
                    out[rid].append({
                        "label_set": spec.name,
                        "run_id": run_id,
                        "model": model,
                        "mtime": mtime,
                    })
    for leaves in out.values():
        leaves.sort(key=lambda p: (-p["mtime"], p["model"], p["label_set"]))
        apply_prediction_labels(leaves)
    return out


def apply_prediction_labels(leaves):
    counts = Counter((p["model"], p["run_id"]) for p in leaves)
    for p in leaves:
        label = f"{p['model']} / {p['run_id']}"
        if counts[(p["model"], p["run_id"])] > 1:
            label += " 200" if p["label_set"] == "ScanNet200" else " 20"
        p["label"] = label
    return leaves


def load_geometry_reference(path=_GEOMETRY_REF):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_cad_accuracy(scene_id, scene_dir, reference):
    path = os.path.join(scene_dir, "recon", "geometry_score.yaml")
    if not os.path.isfile(path) or not reference:
        return None
    try:
        import yaml
        with open(path) as f:
            doc = yaml.safe_load(f)
        sc = (reference.get("scenes") or {}).get(scene_id[:9])
        if sc is None or doc.get("scene") != scene_id:
            return None
        if doc.get("metric") != reference.get("metric"):
            return None
        if doc.get("reference_sha256") != sc.get("reference_sha256"):
            return None
        if doc.get("visible_voxels_sha256") != sc.get("visible_voxels_sha256"):
            return None
        if int(doc.get("voxel_mm", -1)) != int(reference.get("voxel_mm", -2)):
            return None
        acc = float(doc["accuracy_mean_cm"])
        comp = float(doc["completeness_mean_cm"])
        mean = float(doc["mean_bidirectional_distance_cm"])
        if not all(math.isfinite(x) and x >= 0.0 for x in (acc, comp, mean)):
            return None
        return acc
    except (OSError, TypeError, ValueError, KeyError):
        return None


def collect_cad_scores(scannet_dir, recon_ids, reference):
    scores = {}
    for rid in recon_ids:
        acc = load_cad_accuracy(rid, os.path.join(scannet_dir, rid), reference)
        if acc is not None:
            scores[rid] = acc
    return scores


def collect_reconstruction_times(scannet_dir, recon_ids):
    times = {}
    for rid in recon_ids:
        elapsed = load_timing(
            reconstruction_timing_path(os.path.join(scannet_dir, rid)),
            "reconstruction", scene_id=rid)
        if elapsed is not None:
            times[rid] = elapsed
    return times


def collect_prediction_times(scannet_root, pred_index):
    times = {}
    for rid, leaves in pred_index.items():
        for pred in leaves:
            spec = BENCHMARKS[pred["label_set"]]
            elapsed = load_timing(
                prediction_timing_path(
                    spec, pred["run_id"], pred["model"], rid, scannet_root),
                "prediction", scene_id=rid, run_id=pred["run_id"], model=pred["model"])
            if elapsed is not None:
                times[run_node_id(rid, pred)] = elapsed
    return times


def collect_tp_scores(scannet_root, pred_index):
    scores = {}
    for rid, leaves in pred_index.items():
        for p in leaves:
            spec = BENCHMARKS[p["label_set"]]
            path = score_sidecar_path(spec, p["run_id"], p["model"], rid, scannet_root)
            pred_mtime = p.get("mtime")
            if pred_mtime is not None:
                try:
                    if os.path.getmtime(path) < pred_mtime:
                        continue
                except OSError:
                    continue
            loaded = load_score_sidecar(path, rid, p["label_set"], p["run_id"], p["model"])
            if loaded:
                scores[run_node_id(rid, p)] = loaded
    return scores


def collect_run_metrics(scannet_root, pred_index):
    out = {}
    for leaves in pred_index.values():
        for p in leaves:
            key = (p["label_set"], p["run_id"], p["model"])
            if key in out:
                continue
            spec = BENCHMARKS[p["label_set"]]
            eval_root = artifact_paths(spec, scannet_root)["evaluations"]
            csv_path = os.path.join(eval_root, p["run_id"], f"{p['model']}.csv")
            try:
                metrics = read_evaluator_csv(csv_path)
                comparable = False
                try:
                    comparable = _is_comparable(load_manifest(
                        os.path.join(eval_root, p["run_id"], "run.json"),
                        spec=spec, run_id=p["run_id"]))
                except (OSError, ValueError):
                    pass
                out[key] = {
                    "ap": metrics["ap"], "ap50": metrics["ap50"], "ap25": metrics["ap25"],
                    "comparable": comparable,
                }
            except (OSError, ValueError, KeyError):
                out[key] = None
    return out


def collect_ap50_scores(scannet_root, pred_index):
    return {
        key: (value["ap50"] if value else None)
        for key, value in collect_run_metrics(scannet_root, pred_index).items()
    }


def scene_node_id(scene):
    return (KIND_SCENE, scene)


def scan_node_id(recon):
    return (KIND_SCAN, recon)


def method_node_id(recon, model):
    return (KIND_METHOD, recon, model)


def run_node_id(recon, pred):
    return (KIND_RUN, recon, pred["label_set"], pred["run_id"], pred["model"])


def tp_gt_text(score):
    if not score:
        return None
    return f"TP/GT {score['tp']}/{score['gt']}"


def group_methods(leaves):
    grouped = OrderedDict()
    for pred in leaves:
        grouped.setdefault(pred["model"], []).append(pred)
    return grouped


def best_prediction(leaves, metrics=None):
    if not leaves:
        return None
    metrics = metrics or {}

    def key(pred):
        value = metrics.get((pred["label_set"], pred["run_id"], pred["model"]))
        if not isinstance(value, dict):
            return (1, 1, 0.0, 0.0, -pred.get("mtime", 0.0), pred["run_id"], pred["label_set"])
        return (0, 0 if value.get("comparable") else 1, -value["ap"], -value["ap50"],
                -pred.get("mtime", 0.0), pred["run_id"], pred["label_set"])

    return min(leaves, key=key)


def relative_mask_key(path, submission_root):
    path = os.path.normpath(path)
    root = os.path.normpath(submission_root)
    if os.path.isabs(path):
        try:
            return os.path.relpath(path, root).replace("\\", "/")
        except ValueError:
            return os.path.basename(path)
    return path.replace("\\", "/")


def normalize_verdicts(verdicts, submission_root):
    out = {}
    for key, value in (verdicts or {}).items():
        out[relative_mask_key(key, submission_root)] = value
    return out


def apply_verdicts(objects, verdicts):
    bound_tp = 0
    for obj in objects:
        verdict = verdicts.get(obj["key"])
        obj["verdict"] = verdict
        if verdict == "tp":
            bound_tp += 1
    return bound_tp


def merge_tp_gt_overlay(points, objects):
    scored = [obj for obj in objects if obj.get("verdict") in ("tp", "fp")]
    if scored and all(obj.get("packed") is not None for obj in scored):
        return merge_packed_overlay(points, objects, FP_COLOR, TP_COLOR)
    points = np.asarray(points)
    n = len(points)
    colors = np.zeros((n, 3), dtype=float)
    keep = np.zeros(n, dtype=bool)
    for obj in objects:
        if obj.get("verdict") != "fp":
            continue
        sel = np.asarray(obj["sel"], dtype=bool)
        colors[sel] = FP_COLOR
        keep[sel] = True
    for obj in objects:
        if obj.get("verdict") != "tp":
            continue
        sel = np.asarray(obj["sel"], dtype=bool)
        colors[sel] = TP_COLOR
        keep[sel] = True
    return points[keep], colors[keep]


def object_bounds(points, up_axis):
    pts = np.asarray(points)
    if pts.size == 0:
        return None
    mins = pts.min(0)
    maxs = pts.max(0)
    center = (mins + maxs) * 0.5
    anchor = np.array(center, dtype=float)
    anchor[int(up_axis)] = float(maxs[int(up_axis)]) + 0.05
    return {"min": mins, "max": maxs, "center": center, "anchor": anchor}


def render_object_fields(src, key, up_axis):
    return {
        "class_name": src["class_name"],
        "key": key,
        "sel": src.get("sel"),
        "packed": src.get("packed"),
        "verdict": src.get("verdict"),
        "bounds": object_bounds(src["points"], up_axis),
        "box": None,
        "label": None,
    }


def reset_camera_label_state():
    return {"extrinsic": None, "fy": None, "t": 0.0, "pending": False}


def camera_label_update(state, extrinsic, fy, now,
                        interval=LABEL_UPDATE_INTERVAL, atol=LABEL_CAMERA_ATOL):
    extrinsic = np.asarray(extrinsic)
    prev = state.get("extrinsic")
    changed = (
        prev is None
        or abs(float(fy) - float(state.get("fy") or 0.0)) > atol
        or np.max(np.abs(extrinsic - prev)) > atol
    )
    if not changed and not state.get("pending"):
        return False
    last = float(state.get("t") or 0.0)
    if now - last < interval:
        state["pending"] = True
        return False
    state["extrinsic"] = np.array(extrinsic, copy=True)
    state["fy"] = float(fy)
    state["t"] = now
    state["pending"] = False
    return True


def capped_geometry(session, kind, cropper):
    key = "mesh_capped" if kind == "mesh" else "pointcloud_capped"
    geom = session.get(key)
    if geom is None:
        src = session["mesh"] if kind == "mesh" else session["pointcloud"]
        geom = cropper(src, session["up_axis"], session["ceiling_val"])
        session[key] = geom
    return geom


def ensure_object_decorations(obj, color, label_x_dir, label_y_dir, up_axis):
    if obj.get("box") is None:
        box = obj["pcd"].get_axis_aligned_bounding_box()
        box.color = color
        obj["box"] = box
    else:
        obj["box"].color = color
    if obj.get("label") is None:
        bounds = obj.get("bounds")
        if bounds is None:
            label_pos = np.array(obj["box"].get_center())
            label_pos[up_axis] = obj["box"].max_bound[up_axis] + 0.05
        else:
            label_pos = np.array(bounds["anchor"], dtype=float)
        mesh, base = _label_base(obj["class_name"], color)
        _place_label(mesh, base, label_pos, 0.06, label_x_dir, label_y_dir)
        obj["label"] = {"mesh": mesh, "base": base, "anchor": label_pos}
    else:
        obj["label"]["mesh"].paint_uniform_color(color)
    return obj


def fp_count(score):
    if not score:
        return None
    return sum(1 for value in (score.get("verdicts") or {}).values() if value == "fp")


def format_metric(value):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    return f"{value:.3f}"


def information_payload(scene_id, selected_pred, scores, run_metrics,
                        recon_times=None, pred_times=None):
    info = {
        "scene": scene_id,
        "method": None,
        "run": None,
        "ap": None,
        "ap50": None,
        "ap25": None,
        "tp": None,
        "gt": None,
        "fp": None,
        "reconstruction_s": None,
        "prediction_s": None,
    }
    if scene_id:
        info["reconstruction_s"] = (recon_times or {}).get(scene_id)
    if not selected_pred:
        return info
    info["method"] = selected_pred["model"]
    info["run"] = selected_pred["run_id"]
    metrics = run_metrics.get(
        (selected_pred["label_set"], selected_pred["run_id"], selected_pred["model"]))
    if metrics:
        info["ap"] = metrics.get("ap")
        info["ap50"] = metrics.get("ap50")
        info["ap25"] = metrics.get("ap25")
    if scene_id:
        run_key = run_node_id(scene_id, selected_pred)
        score = scores.get(run_key)
        if score:
            info["tp"] = score["tp"]
            info["gt"] = score["gt"]
            info["fp"] = fp_count(score)
        info["prediction_s"] = (pred_times or {}).get(run_key)
    return info


def flatten_tree(groups, predictions, expanded, cad, scores, benchmark=None, ap50=None):
    rows = []
    ap50 = ap50 or {}
    for scene, recons in groups.items():
        sid = scene_node_id(scene)
        scene_open = sid in expanded
        rows.append({
            "id": sid, "kind": KIND_SCENE, "label": f"{scene} [{len(recons)}]",
            "suffix": None, "depth": 0, "expanded": scene_open, "parent": None,
            "folder": True, "recon_id": None, "prediction": None,
        })
        if not scene_open:
            continue
        for recon in recons:
            sid_scan = scan_node_id(recon)
            leaves = [p for p in (predictions.get(recon) or [])
                      if benchmark is None or p["label_set"] == benchmark]
            methods = group_methods(leaves)
            scan_open = sid_scan in expanded
            suffix = f"CAD acc {cad[recon]:.2f} cm" if recon in cad else None
            rows.append({
                "id": sid_scan, "kind": KIND_SCAN, "label": recon,
                "suffix": suffix, "depth": 1, "expanded": scan_open, "parent": sid,
                "folder": True, "recon_id": recon, "prediction": None,
            })
            if not scan_open:
                continue
            for model, runs in methods.items():
                mid = method_node_id(recon, model)
                method_open = mid in expanded
                best = best_prediction(runs, ap50)
                best_score = scores.get(run_node_id(recon, best)) if best else None
                rows.append({
                    "id": mid, "kind": KIND_METHOD, "label": model,
                    "suffix": tp_gt_text(best_score), "depth": 2,
                    "expanded": method_open, "parent": sid_scan,
                    "folder": True, "recon_id": recon, "prediction": best,
                })
                if not method_open:
                    continue
                for pred in sorted(runs, key=lambda p: (p["run_id"], p["label_set"])):
                    pid = run_node_id(recon, pred)
                    rows.append({
                        "id": pid, "kind": KIND_RUN, "label": pred["run_id"],
                        "suffix": tp_gt_text(scores.get(pid)), "depth": 3,
                        "expanded": False, "parent": mid,
                        "folder": False, "recon_id": recon, "prediction": pred,
                    })
    return rows


def tree_payload_rows(rows, cursor_id, active_pred, active_recon, active_method=None):
    active_scene = active_recon[:9] if active_recon else None
    out = []
    for row in rows:
        path_level = 0
        if active_pred and row["id"] == active_pred:
            path_level = 4
        elif active_method and row["id"] == active_method:
            path_level = 3
        elif active_recon and row["kind"] == KIND_SCAN and row["recon_id"] == active_recon:
            path_level = 2
        elif active_scene and row["kind"] == KIND_SCENE and row["id"][1] == active_scene:
            path_level = 1
        item = dict(row)
        item["cursor"] = row["id"] == cursor_id
        item["path_level"] = path_level
        item.pop("prediction", None)
        item.pop("parent", None)
        out.append(item)
    return out


def cursor_ancestors(cursor):
    if not cursor:
        return
    kind = cursor[0]
    if kind == KIND_RUN:
        recon, model = cursor[1], cursor[4]
        yield method_node_id(recon, model)
        yield scan_node_id(recon)
        yield scene_node_id(recon[:9])
    elif kind == KIND_METHOD:
        recon = cursor[1]
        yield scan_node_id(recon)
        yield scene_node_id(recon[:9])
    elif kind == KIND_SCAN:
        yield scene_node_id(cursor[1][:9])


def clamp_cursor(state, rows):
    if not rows:
        state["cursor_id"] = None
        return 0
    ids = [row["id"] for row in rows]
    cursor = state.get("cursor_id")
    if cursor in ids:
        return ids.index(cursor)
    for ancestor in cursor_ancestors(cursor):
        if ancestor in ids:
            state["cursor_id"] = ancestor
            return ids.index(ancestor)
    state["cursor_id"] = rows[0]["id"]
    return 0


def handle_tree(state, key, rows):
    action = {"activate": None}
    if not rows:
        return action
    idx = clamp_cursor(state, rows)
    row = rows[idx]
    expanded = state["expanded"]
    if key == KEY_DOWN:
        if idx + 1 < len(rows):
            state["cursor_id"] = rows[idx + 1]["id"]
    elif key == KEY_UP:
        if idx > 0:
            state["cursor_id"] = rows[idx - 1]["id"]
    elif key == KEY_RIGHT:
        if row["folder"] and row["id"] not in expanded:
            expanded.add(row["id"])
        elif row["folder"]:
            for nxt in rows[idx + 1:]:
                if nxt["parent"] == row["id"]:
                    state["cursor_id"] = nxt["id"]
                    break
    elif key == KEY_LEFT:
        if row["folder"]:
            expanded.discard(row["id"])
    elif key == KEY_ENTER:
        if row["kind"] in (KIND_SCAN, KIND_METHOD, KIND_RUN):
            action["activate"] = row
        elif row["folder"]:
            if row["id"] in expanded:
                expanded.discard(row["id"])
            else:
                expanded.add(row["id"])
    return action


def setting_values(specs):
    return [spec["value"] for spec in specs]


def enabled_values(specs):
    return [spec["value"] for spec in specs if spec.get("enabled", True)]


def clamp_setting_index(state):
    n = len(SETTING_ROWS)
    idx = state.get("setting_index", 0)
    if n <= 0:
        state["setting_index"] = 0
        return 0
    state["setting_index"] = min(max(0, idx), n - 1)
    return state["setting_index"]


def handle_settings(state, key, option_lists):
    action = {"apply": None}
    idx = clamp_setting_index(state)
    n = len(SETTING_ROWS)
    if key in (KEY_UP, KEY_DOWN):
        state["candidate"] = None
        nxt = idx + (-1 if key == KEY_UP else 1)
        if 0 <= nxt < n:
            state["setting_index"] = nxt
        return action
    row = SETTING_ROWS[idx]
    specs = option_lists.get(row) or []
    enabled = enabled_values(specs)
    applied = state_value(state, row)
    cand = state.get("candidate")
    current = cand[1] if cand is not None and cand[0] == row else applied
    if key in (KEY_LEFT, KEY_RIGHT) and enabled:
        if current in enabled:
            pos = enabled.index(current)
        else:
            pos = 0 if key == KEY_RIGHT else len(enabled) - 1
        nxt = pos + (-1 if key == KEY_LEFT else 1)
        if 0 <= nxt < len(enabled):
            state["candidate"] = (row, enabled[nxt])
        return action
    if key == KEY_ENTER:
        value = current
        state["candidate"] = None
        if value != applied and value in enabled:
            action["apply"] = (row, value)
    return action


def handle_navigation(state, key, rows, option_lists):
    if key == KEY_TAB:
        state["candidate"] = None
        state["focus"] = (FOCUS_SETTINGS if state["focus"] == FOCUS_TREE else FOCUS_TREE)
        return {"activate": None, "apply": None}
    if state["focus"] == FOCUS_TREE:
        action = handle_tree(state, key, rows)
        action["apply"] = None
        return action
    action = handle_settings(state, key, option_lists)
    action["activate"] = None
    return action


def state_value(state, row):
    return {
        "benchmark": state["benchmark"],
        "mode": state["source_mode"],
        "color": state["color_mode"],
        "geometry": state["geometry"],
        "boxes": state["boxes"],
        "ceiling": state["ceiling_hidden"],
        "cad": state.get("cad_open", False),
    }[row]


def settings_payload(state, option_lists, display_value):
    pending = state.get("candidate")
    idx = state.get("setting_index", 0)
    out = []
    for i, name in enumerate(SETTING_ROWS):
        specs = option_lists.get(name) or []
        applied = state_value(state, name)
        target = pending[1] if pending is not None and pending[0] == name else applied
        options = []
        for spec in specs:
            value = spec["value"]
            enabled = spec.get("enabled", True)
            if not enabled:
                kind = "disabled"
            elif i == idx and pending is not None and pending[0] == name and value == pending[1] and value != applied:
                kind = "pending"
            elif value == applied:
                kind = "applied"
            else:
                kind = "idle"
            options.append({
                "text": display_value(name, value),
                "kind": kind,
                "sel": False,
                "enabled": enabled,
            })
        if i == idx:
            for option, spec in zip(options, specs):
                if spec["value"] == target and spec.get("enabled", True):
                    option["sel"] = True
                    break
        out.append({
            "key": name,
            "name": SETTING_NAMES[name],
            "focused": i == idx,
            "options": options,
        })
    return out


def cad_report_png(scene_dir):
    path = os.path.join(scene_dir, "recon", "cad_comparison.png")
    return path if os.path.isfile(path) else None


def clear_tracked(vis, displayed, key):
    for geom in list(displayed[key]):
        vis.remove_geometry(geom, reset_bounding_box=False)
    displayed[key] = set()


def _instance_counts(objects):
    return Counter(o["class_name"] for o in objects)


def _display_size():
    import subprocess
    try:
        width, height = subprocess.check_output(
            ["xdotool", "getdisplaygeometry"], text=True).split()
        return int(width), int(height)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 1920, 1080


def load_scene_bundle(scene_id, scannet_dir, ceiling_height, nyu40map, palette):
    import open3d as o3d
    scene_dir = os.path.join(scannet_dir, scene_id)
    mesh_file = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")
    if not os.path.isfile(mesh_file):
        raise FileNotFoundError(f"mesh not found: {mesh_file}")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    scene_pts = np.asarray(mesh.vertices)
    if scene_pts.size == 0:
        raise ValueError(f"empty mesh: {mesh_file}")
    scene_colors = np.asarray(mesh.vertex_colors)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(scene_pts)
    pointcloud.colors = o3d.utility.Vector3dVector(scene_colors)
    up_axis = _detect_up_axis(scene_pts)
    floor_val = np.percentile(scene_pts[:, up_axis], 1)
    ceiling_val = floor_val + ceiling_height
    up_vec = np.zeros(3)
    up_vec[up_axis] = 1.0
    horiz_axes = [a for a in range(3) if a != up_axis]
    read_axis = max(horiz_axes, key=lambda a: (scene_pts.max(0) - scene_pts.min(0))[a])
    label_x_dir, label_y_dir = np.zeros(3), np.zeros(3)
    label_x_dir[read_axis] = 1.0
    label_y_dir[up_axis] = 1.0
    gt = load_gt_instances(scene_dir, scene_pts, scene_colors)
    return {
        "id": scene_id,
        "dir": scene_dir,
        "mesh": mesh,
        "pointcloud": pointcloud,
        "mesh_capped": None,
        "pointcloud_capped": None,
        "scene_pts": scene_pts,
        "scene_colors": scene_colors,
        "up_axis": up_axis,
        "up_vec": up_vec,
        "ceiling_val": ceiling_val,
        "label_x_dir": label_x_dir,
        "label_y_dir": label_y_dir,
        "gt": gt,
        "gt_counts": _instance_counts(gt["objects"]) if gt else Counter(),
        "source_cache": {},
        "nyu40map": nyu40map,
        "palette": palette,
    }


def _start_publisher(conn):
    import threading
    from queue import Empty, Full, Queue
    box = Queue(maxsize=1)
    stop = object()

    def loop():
        while True:
            item = box.get()
            if item is stop:
                break
            while True:
                try:
                    nxt = box.get_nowait()
                except Empty:
                    break
                if nxt is stop:
                    return
                item = nxt
            try:
                conn.send(item)
            except Exception:
                break

    threading.Thread(target=loop, daemon=True).start()

    def publish(payload):
        try:
            box.put_nowait(payload)
        except Full:
            try:
                box.get_nowait()
            except Empty:
                pass
            try:
                box.put_nowait(payload)
            except Full:
                pass

    def stop_pub():
        try:
            box.put(stop)
        except Exception:
            pass

    return publish, stop_pub


def _stop_hud(conn, process, stop_pub=None):
    if stop_pub:
        stop_pub()
    try:
        conn.send(None)
    except Exception:
        pass
    process.join(timeout=2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
    try:
        conn.close()
    except Exception:
        pass


def visualize(scene_id=None, scannet_dir=DEFAULT_SCANNET_DIR, benchmark="ScanNet200", run_id=None,
              ceiling_height=2.0):
    import multiprocessing
    import glfw
    import open3d as o3d

    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "No DISPLAY set -- Open3D needs a real or virtual X display to open a window.")

    recon_ids = discover_reconstructions(scannet_dir)
    scannet_root = os.path.dirname(scannet_dir.rstrip("/"))
    groups = group_scenes(recon_ids)
    pred_index = index_predictions(scannet_root, recon_ids)
    try:
        reference = load_geometry_reference()
    except OSError:
        reference = None
    cad = collect_cad_scores(scannet_dir, recon_ids, reference)
    scores = collect_tp_scores(scannet_root, pred_index)
    run_metrics = collect_run_metrics(scannet_root, pred_index)
    recon_times = collect_reconstruction_times(scannet_dir, recon_ids)
    pred_times = collect_prediction_times(scannet_root, pred_index)

    nyu40map = load_label_map()
    palette = util.create_color_palette()
    session = None
    first_scene = next(iter(groups), None)

    state = {
        "benchmark": benchmark,
        "source_mode": SOURCE_SCENE,
        "color_mode": COLOR_CLASS,
        "selected_pred": None,
        "setting_index": 0,
        "geometry": "mesh",
        "boxes": False,
        "ceiling_hidden": False,
        "status": "Enter a scan",
        "focus": FOCUS_TREE,
        "candidate": None,
        "expanded": set(),
        "cursor_id": scene_node_id(first_scene) if first_scene else None,
        "active_pred": None,
        "active_method": None,
        "cad_open": False,
    }

    display_width, display_height = _display_size()
    view_w = max(640, display_width - hud.LEFT_WIDTH - hud.RIGHT_WIDTH - 10)
    view_h = max(480, display_height - 100)
    view_rect = (hud.LEFT_WIDTH, 0, view_w, view_h)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=hud.VIEWER_TITLE,
                      width=view_w, height=view_h,
                      left=hud.LEFT_WIDTH, top=0)

    current = {"scene": None, "placeholder": None}
    displayed = {"objs": set(), "boxes": set(), "labels": set()}
    objects = []
    overlay = {"pcd": None}
    label_cam = reset_camera_label_state()

    context = multiprocessing.get_context("spawn")
    left_recv, left_q = context.Pipe(duplex=False)
    right_recv, right_q = context.Pipe(duplex=False)
    left_key_recv, left_key_send = context.Pipe(duplex=False)
    right_key_recv, right_key_send = context.Pipe(duplex=False)
    left_p = context.Process(
        target=hud.run, args=(left_recv, os.getpid(), "left", view_rect, left_key_send),
        daemon=True)
    right_p = context.Process(
        target=hud.run, args=(right_recv, os.getpid(), "right", view_rect, right_key_send),
        daemon=True)
    left_p.start()
    right_p.start()
    publish_left, stop_left = _start_publisher(left_q)
    publish_right, stop_right = _start_publisher(right_q)

    def _spec(name=None):
        return resolve_benchmark(name or state["benchmark"])

    def _pred():
        return state.get("selected_pred")

    def _tp_available():
        pred = _pred()
        scene_id = pred["scene_id"] if pred else (session["id"] if session else None)
        if not pred or not scene_id:
            return False
        gt_file = os.path.join(
            artifact_paths(_spec(pred["label_set"]), scannet_root)["gt"], scene_id + ".txt")
        return os.path.isfile(gt_file)

    def _cad_png():
        if session is None:
            return None
        return cad_report_png(session["dir"])

    def _opts(values, enabled=True):
        if isinstance(enabled, dict):
            return [{"value": value, "enabled": bool(enabled.get(value, True))} for value in values]
        return [{"value": value, "enabled": bool(enabled)} for value in values]

    def _option_list(row):
        if row == "benchmark":
            return _opts(list(BENCHMARKS))
        if row == "mode":
            return _opts([SOURCE_GT, SOURCE_SCENE, SOURCE_PRED],
                         {SOURCE_GT: True, SOURCE_SCENE: True, SOURCE_PRED: _pred() is not None})
        if row == "color":
            scene_only = state["source_mode"] == SOURCE_SCENE
            return _opts(
                [COLOR_CLASS, COLOR_INSTANCE, COLOR_TPGT],
                {COLOR_CLASS: not scene_only, COLOR_INSTANCE: not scene_only,
                 COLOR_TPGT: state["source_mode"] == SOURCE_PRED and _tp_available()})
        if row == "geometry":
            return _opts(["mesh", "pointcloud"])
        if row == "boxes":
            return _opts([True, False])
        if row == "ceiling":
            return _opts([False, True])
        if row == "cad":
            return _opts([False, True], bool(_cad_png()))
        return []

    def _option_lists():
        return {row: _option_list(row) for row in SETTING_ROWS}

    def _display_value(row, value=None):
        if value is None:
            value = state_value(state, row)
        if row == "mode":
            return {SOURCE_GT: "Ground truth", SOURCE_SCENE: "Scene only",
                    SOURCE_PRED: "Prediction"}[value]
        if row == "color":
            return {COLOR_CLASS: "Classes", COLOR_INSTANCE: "Instances",
                    COLOR_TPGT: "TP/GT"}[value]
        if row == "geometry":
            return {"mesh": "Mesh", "pointcloud": "Points"}[value]
        if row == "boxes":
            return "On" if value else "Off"
        if row == "ceiling":
            return "Hidden" if value else "Visible"
        if row == "cad":
            return "Open" if value else "Closed"
        return value

    def _tree_rows():
        return flatten_tree(groups, pred_index, state["expanded"], cad, scores,
                            benchmark=state["benchmark"], ap50=run_metrics)

    def _make_objects(data, keys=None):
        import open3d as o3d
        made = []
        up_axis = session["up_axis"]
        for idx, o in enumerate(data["objects"]):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(o["points"])
            pcd.colors = o3d.utility.Vector3dVector(o["colors"])
            fields = render_object_fields(o, keys[idx] if keys else None, up_axis)
            fields.update({
                "pcd": pcd,
                "rgb": o["colors"],
                "verdict": None,
            })
            made.append(fields)
        return made

    def _score_predictions(data, root, pred):
        run_key = run_node_id(session["id"], pred)
        spec = _spec(pred["label_set"])
        sidecar = scores.get(run_key)
        verdicts = None
        status = ""
        if sidecar:
            verdicts = normalize_verdicts(sidecar["verdicts"], root)
            bound = apply_verdicts(objects, verdicts)
            if bound == int(sidecar["tp"]):
                return status
            verdicts = None
        gt_file = os.path.join(artifact_paths(spec, scannet_root)["gt"], session["id"] + ".txt")
        if not os.path.isfile(gt_file):
            apply_verdicts(objects, {})
            return "TP/GT unavailable: no evaluation GT for this label set."
        summary = classify_ap50(
            util_3d.load_ids(gt_file), data["pred_instances"], spec, session["id"])
        verdicts = normalize_verdicts(summary["verdicts"], root)
        bound = apply_verdicts(objects, verdicts)
        scores[run_key] = {
            "tp": int(summary["tp"]), "gt": int(summary["gt"]), "verdicts": verdicts,
        }
        if bound != int(summary["tp"]):
            return "TP/GT coloring mismatch: bound predictions disagree with the evaluator."
        return status

    def _load_source():
        nonlocal objects, label_cam
        label_cam = reset_camera_label_state()
        if overlay["pcd"] is not None:
            if overlay["pcd"] in displayed["objs"]:
                vis.remove_geometry(overlay["pcd"], reset_bounding_box=False)
                displayed["objs"].discard(overlay["pcd"])
            overlay["pcd"] = None
        if session is None:
            objects = []
            return
        pred = _pred()
        pred_key = None if pred is None else (
            pred["label_set"], pred["run_id"], pred["model"], pred["scene_id"])
        key = (state["source_mode"], pred_key)
        cache = session["source_cache"]
        if key in cache:
            objects, state["status"] = cache[key]
            return
        if state["source_mode"] == SOURCE_SCENE:
            objects = []
            state["status"] = ""
        elif state["source_mode"] == SOURCE_GT:
            if session["gt"] is None:
                objects = []
                state["status"] = "No ground truth annotations for this scene."
            else:
                objects = _make_objects(session["gt"])
                state["status"] = ""
        else:
            if pred is None or pred["scene_id"] != session["id"]:
                objects = []
                state["status"] = "No prediction selected."
            else:
                spec = _spec(pred["label_set"])
                root = submission_dir(spec, pred["run_id"], pred["model"], scannet_root)
                data = load_predictions(
                    root, session["id"], session["scene_pts"],
                    session["scene_colors"], spec,
                    run_id=pred["run_id"], model=pred["model"],
                    scannet_root=scannet_root,
                    source_mtime=prediction_artifact_mtime(root, session["id"]))
                if data is None:
                    objects = []
                    state["status"] = f"No prediction for model {pred['model']} in this scene."
                else:
                    objects = _make_objects(data, keys=[o["key"] for o in data["objects"]])
                    for obj, src in zip(objects, data["objects"]):
                        obj["sel"] = src["sel"]
                    state["status"] = _score_predictions(data, root, pred)
        cache[key] = (objects, state["status"])

    def _variant():
        kind = "mesh" if state["geometry"] == "mesh" else "pointcloud"
        if state["ceiling_hidden"]:
            return capped_geometry(session, kind, _below_height)
        return session[kind]

    def _swap_scene(reset_view=False):
        if current.get("placeholder") is not None:
            vis.remove_geometry(current["placeholder"], reset_bounding_box=False)
            current["placeholder"] = None
        if current["scene"] is not None:
            vis.remove_geometry(current["scene"], reset_bounding_box=False)
            current["scene"] = None
        if session is None:
            return
        current["scene"] = _variant()
        vis.add_geometry(current["scene"], reset_bounding_box=reset_view)

    def _ceiling_hidden(obj):
        if session is None or not state["ceiling_hidden"]:
            return False
        bounds = obj.get("bounds")
        if bounds is None:
            return False
        return bounds["max"][session["up_axis"]] > session["ceiling_val"]

    def _rebuild_overlay():
        if overlay["pcd"] is not None:
            if overlay["pcd"] in displayed["objs"]:
                vis.remove_geometry(overlay["pcd"], reset_bounding_box=False)
                displayed["objs"].discard(overlay["pcd"])
            overlay["pcd"] = None
        if (session is None or state["source_mode"] != SOURCE_PRED
                or state["color_mode"] != COLOR_TPGT or not objects):
            return
        kept = [obj for obj in objects if obj.get("sel") is not None and not _ceiling_hidden(obj)]
        pts, cols = merge_tp_gt_overlay(session["scene_pts"], kept)
        if len(pts) == 0:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        overlay["pcd"] = pcd

    def _wanted():
        if state["source_mode"] == SOURCE_SCENE:
            return set(), set(), set()
        keep = []
        for i, obj in enumerate(objects):
            if state["color_mode"] == COLOR_TPGT and obj["verdict"] not in ("tp", "fp"):
                continue
            if _ceiling_hidden(obj):
                continue
            keep.append(i)
        if state["color_mode"] == COLOR_TPGT:
            want_objs = {overlay["pcd"]} if overlay["pcd"] is not None else set()
        else:
            want_objs = {objects[i]["pcd"] for i in keep}
        want_boxes = set()
        want_labels = set()
        if state["boxes"]:
            for i in keep:
                obj = objects[i]
                ensure_object_decorations(
                    obj, _object_color(i, obj),
                    session["label_x_dir"], session["label_y_dir"], session["up_axis"])
                want_boxes.add(obj["box"])
                want_labels.add(obj["label"]["mesh"])
        return want_objs, want_boxes, want_labels

    def _apply_diff(displayed_set, wanted_set):
        for g in displayed_set - wanted_set:
            vis.remove_geometry(g, reset_bounding_box=False)
        for g in wanted_set - displayed_set:
            vis.add_geometry(g, reset_bounding_box=False)
        return wanted_set

    def _update_labels(_vis):
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        extrinsic = np.asarray(params.extrinsic)
        cam_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
        fy = params.intrinsic.get_focal_length()[1]
        if not camera_label_update(label_cam, extrinsic, fy, time.monotonic()):
            return False
        updated = False
        for o in objects:
            lab = o.get("label")
            if not lab or lab["mesh"] not in displayed["labels"]:
                continue
            to_cam = cam_pos - lab["anchor"]
            to_cam[session["up_axis"]] = 0.0
            dist = np.linalg.norm(to_cam)
            if dist < 1e-6:
                continue
            to_cam /= dist
            x_dir = np.cross(session["up_vec"], to_cam)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-6:
                continue
            x_dir /= x_norm
            char_size = TARGET_LABEL_PX * dist / fy
            _place_label(lab["mesh"], lab["base"], lab["anchor"], char_size,
                         x_dir, session["up_vec"])
            vis.update_geometry(lab["mesh"])
            updated = True
        return updated

    def _sync():
        nonlocal label_cam
        prev_labels = displayed["labels"]
        want_objs, want_boxes, want_labels = _wanted()
        displayed["objs"] = _apply_diff(displayed["objs"], want_objs)
        displayed["boxes"] = _apply_diff(displayed["boxes"], want_boxes)
        displayed["labels"] = _apply_diff(displayed["labels"], want_labels)
        if displayed["labels"] - prev_labels:
            label_cam = reset_camera_label_state()

    def _object_color(idx, o):
        if state["color_mode"] == COLOR_TPGT:
            return TP_COLOR if o["verdict"] == "tp" else FP_COLOR
        if state["color_mode"] == COLOR_INSTANCE:
            return colormaps["turbo"](idx / max(len(objects), 1))[:3]
        return _class_color(o["class_name"], nyu40map, palette)

    def _paint():
        clear_tracked(vis, displayed, "boxes")
        _rebuild_overlay()
        for idx, o in enumerate(objects):
            color = _object_color(idx, o)
            if state["color_mode"] != COLOR_TPGT:
                colors = np.tile(color, (len(o["pcd"].points), 1))
                o["pcd"].colors = o3d.utility.Vector3dVector(colors)
                if o["pcd"] in displayed["objs"]:
                    vis.update_geometry(o["pcd"])
            if o.get("box") is not None:
                o["box"].color = color
            lab = o.get("label")
            if lab is not None:
                lab["mesh"].paint_uniform_color(color)
                if lab["mesh"] in displayed["labels"]:
                    vis.update_geometry(lab["mesh"])

    def _detach_all():
        nonlocal label_cam
        label_cam = reset_camera_label_state()
        overlay["pcd"] = None
        clear_tracked(vis, displayed, "objs")
        clear_tracked(vis, displayed, "boxes")
        clear_tracked(vis, displayed, "labels")
        if current.get("placeholder") is not None:
            vis.remove_geometry(current["placeholder"], reset_bounding_box=False)
            current["placeholder"] = None
        if current["scene"] is not None:
            vis.remove_geometry(current["scene"], reset_bounding_box=False)
            current["scene"] = None

    def _hud_payload():
        counts = _instance_counts(objects) if objects else Counter()
        gt_counts = session["gt_counts"] if session else Counter()
        classes = sorted(set(gt_counts) | set(counts))
        png = _cad_png()
        pred = _pred()
        return {
            "status": state["status"],
            "settings": settings_payload(state, _option_lists(), _display_value),
            "info": information_payload(
                session["id"] if session else None, pred, scores, run_metrics,
                recon_times, pred_times),
            "classes": classes,
            "colors": {name: tuple(_class_color(name, nyu40map, palette)) for name in classes},
            "focus": state["focus"],
            "tree": tree_payload_rows(
                _tree_rows(), state["cursor_id"], state["active_pred"],
                session["id"] if session else None, state.get("active_method")),
            "cad_overlay": png if state.get("cad_open") and png else None,
        }

    def _update_hud():
        payload = _hud_payload()
        publish_left({"tree": payload["tree"]})
        publish_right({k: payload[k] for k in (
            "settings", "classes", "colors", "status", "focus", "cad_overlay", "info")})

    def _normalize_color():
        if state["source_mode"] == SOURCE_SCENE:
            return
        if state["color_mode"] == COLOR_TPGT and (
                state["source_mode"] != SOURCE_PRED or not _tp_available()):
            state["color_mode"] = COLOR_CLASS

    def _apply(row, value):
        need_reload = False
        if row == "benchmark":
            if value == state["benchmark"]:
                return
            state["benchmark"] = value
            pred = _pred()
            if pred and pred["label_set"] != value:
                state["selected_pred"] = None
                state["active_pred"] = None
                state["active_method"] = None
                if state["source_mode"] == SOURCE_PRED:
                    state["source_mode"] = SOURCE_SCENE
            clamp_cursor(state, _tree_rows())
            need_reload = True
        elif row == "mode":
            if value == state["source_mode"]:
                return
            if value == SOURCE_PRED and _pred() is None:
                return
            state["source_mode"] = value
            need_reload = True
        elif row == "color":
            if value == state["color_mode"]:
                return
            state["color_mode"] = value
        elif row == "geometry":
            if value == state["geometry"]:
                return
            state["geometry"] = value
            _swap_scene()
        elif row == "boxes":
            state["boxes"] = value
        elif row == "ceiling":
            if value == state["ceiling_hidden"]:
                return
            state["ceiling_hidden"] = value
            if session is not None:
                _swap_scene()
        elif row == "cad":
            state["cad_open"] = bool(value)
            _update_hud()
            return
        _normalize_color()
        if session is None:
            _update_hud()
            return
        if need_reload:
            _load_source()
        _paint()
        _sync()
        _update_hud()

    def _bind_session(recon_id):
        nonlocal session, objects
        if session is not None and session["id"] == recon_id:
            return True
        state["status"] = f"Loading {recon_id}…"
        _update_hud()
        try:
            bundle = load_scene_bundle(
                recon_id, scannet_dir, ceiling_height, nyu40map, palette)
        except Exception as exc:
            state["status"] = f"Failed to load {recon_id}: {exc}"
            _update_hud()
            return False
        _detach_all()
        session = bundle
        objects = []
        state["cad_open"] = False
        state["expanded"].add(scene_node_id(recon_id[:9]))
        state["expanded"].add(scan_node_id(recon_id))
        return True

    def _activate_scan(row):
        recon = row["recon_id"]
        pred = _pred()
        already = (session is not None and session["id"] == recon
                   and state["source_mode"] == SOURCE_SCENE)
        if already:
            return
        if not _bind_session(recon):
            return
        if pred is not None and pred["scene_id"] != recon:
            state["selected_pred"] = None
            state["active_pred"] = None
            state["active_method"] = None
        state["source_mode"] = SOURCE_SCENE
        state["candidate"] = None
        _normalize_color()
        _load_source()
        _paint()
        _sync()
        _swap_scene(reset_view=True)
        _update_hud()

    def _activate_run(row):
        recon = row["recon_id"]
        pred = row["prediction"]
        if pred is None:
            return
        need_view = session is None or session["id"] != recon
        if not _bind_session(recon):
            return
        state["benchmark"] = pred["label_set"]
        state["selected_pred"] = {
            "label_set": pred["label_set"],
            "run_id": pred["run_id"],
            "model": pred["model"],
            "scene_id": recon,
        }
        state["source_mode"] = SOURCE_PRED
        state["candidate"] = None
        state["active_pred"] = run_node_id(recon, pred)
        state["active_method"] = method_node_id(recon, pred["model"])
        _normalize_color()
        _load_source()
        if need_view:
            _swap_scene(reset_view=True)
        _paint()
        _sync()
        _update_hud()

    def _activate_method(row):
        _activate_run(row)

    def _dispatch(key):
        if key in (ord("Q"), KEY_ESCAPE):
            vis.close()
            return
        action = handle_navigation(state, key, _tree_rows(), _option_lists())
        if action.get("activate"):
            row = action["activate"]
            if row["kind"] == KIND_SCAN:
                _activate_scan(row)
            elif row["kind"] == KIND_METHOD:
                _activate_method(row)
            elif row["kind"] == KIND_RUN:
                _activate_run(row)
            return
        if action.get("apply"):
            _apply(*action["apply"])
            return
        clamp_cursor(state, _tree_rows())
        _update_hud()

    def _drain_keys():
        for conn in (left_key_recv, right_key_recv):
            while True:
                try:
                    if not conn.poll():
                        break
                    key = conn.recv()
                except (EOFError, OSError):
                    break
                if isinstance(key, int):
                    _dispatch(key)

    def _tick(_vis):
        _drain_keys()
        if displayed["labels"]:
            return _update_labels(_vis)
        return False

    def _key_action(key, handler):
        def callback(_vis, action, _mods):
            if action != KEY_PRESS:
                return False
            handler()
            return False
        vis.register_key_action_callback(key, callback)

    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        if left_p.is_alive() and right_p.is_alive():
            break
        time.sleep(0.02)
    _update_hud()

    _key_action(KEY_UP, lambda: _dispatch(KEY_UP))
    _key_action(KEY_DOWN, lambda: _dispatch(KEY_DOWN))
    _key_action(KEY_LEFT, lambda: _dispatch(KEY_LEFT))
    _key_action(KEY_RIGHT, lambda: _dispatch(KEY_RIGHT))
    _key_action(KEY_ENTER, lambda: _dispatch(KEY_ENTER))
    _key_action(KEY_TAB, lambda: _dispatch(KEY_TAB))
    _key_action(ord("Q"), lambda: _dispatch(ord("Q")))
    _key_action(KEY_ESCAPE, lambda: _dispatch(KEY_ESCAPE))
    vis.register_animation_callback(_tick)
    print("Keys: Tab HUD, arrows, Enter, mouse view, Esc/Q quit")
    vis.run()
    _stop_hud(left_q, left_p, stop_left)
    _stop_hud(right_q, right_p, stop_right)
    vis.destroy_window()
