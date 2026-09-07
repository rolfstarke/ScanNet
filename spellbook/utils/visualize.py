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
from utils import hud, query_prompt  # noqa: E402
from utils.compute_time import (  # noqa: E402
    load_timing, prediction_timing_path, reconstruction_timing_path)
from utils.camera_replay import (  # noqa: E402
    CAMERA_OFF, CAMERA_PATH, CAMERA_REPLAY, FRUSTUM_LINES, REPLAY_FPS,
    REPLAY_TRANSPORT, advance_playback, discover_frames,
    fill_replay_cache, frustum_corners, is_valid_pose, load_calibration,
    load_color_rgb, load_pose, move_replay_selection, resize_rgb,
    rgb_to_rgba_bytes, step_replay_frame, trajectory_polylines)
from utils import replay2d  # noqa: E402
from utils.detect_replay import draw_detections, draw_masks  # noqa: E402
from utils.prediction_masks import (  # noqa: E402
    load_or_build_packed_masks, merge_packed_overlay, unpack_mask)
from utils.query import (  # noqa: E402
    ENCODERS, INSTANT_MODELS, NATIVE_LOOKUP_MODELS, QUERY_IMAGE, QUERY_OFF,
    QUERY_SEARCH, QueryClient, clip_features_path,
    clip_path_ready, load_clip_features, method_python, openins_snap_complete,
    openins_snap_root, query_modes_for)


DEFAULT_SCANNET_DIR = "/data/scannet/scans"
LABEL_MAP_FILE = "/data/scannet/v2/scannetv2-labels.combined.tsv"
_SCENE_ID_RE = re.compile(r"^(scene\d{4})_(\d{2})$")
_GEOMETRY_REF = os.path.join(_SPELLBOOK, "reconstruct", "geometry_reference.yaml")

_PALETTE = colormaps["tab20"].colors
TP_COLOR = np.array([0.10, 0.80, 0.20], dtype=float)
FP_COLOR = np.array([0.90, 0.15, 0.15], dtype=float)
IGNORED_COLOR = np.array([0.55, 0.55, 0.55], dtype=float)
MIN_REGION_SIZE = 100
AP50_THRESHOLD = 0.5

MODEL_PIPELINE = {
    "mosaic3d": (
        "A single 3D network reads the point cloud directly — no photos needed at run time. "
        "It was trained on millions of automatically captioned 3D regions, aligning every "
        "point's feature with text. A lightweight decoder turns those language-aligned "
        "features straight into object masks. A written query is matched against the "
        "masks in one pass."
    ),
    "openins3d": (
        "A 3D network cuts the point cloud into unnamed masks, using no photos as input. "
        "The method renders its own overview pictures from the 3D scene and lets a 2D "
        "detector list everything it sees in them. Each 3D mask is projected into the "
        "same pictures and matched against those detections by overlap. Unmatched masks "
        "get a second chance on close-up crops; masks that still find no name are dropped."
    ),
    "openyolo3d": (
        "A 3D network cuts the point cloud into unnamed object masks. A 2D detector "
        "labels boxes in the scan photos, and each mask adopts whichever label sits "
        "under most of its projected points."
    ),
    "open3dis": (
        "Object candidates come from two sides: a 3D network proposing masks from the "
        "point cloud, and a 2D segmenter drawing masks in the photos. The photo masks "
        "are projected onto small uniform point regions and grown into coherent 3D "
        "pieces, then merged across views into fuller objects. This catches small and "
        "unusual objects the 3D network alone misses. Each candidate is described with "
        "image-language features from several views, and queries are answered by "
        "comparing text to those descriptions."
    ),
    "openmask3d": (
        "A 3D network cuts the point cloud into unnamed object masks. For each mask, "
        "the method picks the few photos where it is most visible, outlines it there, "
        "and crops the result at several zoom levels. Each crop is embedded by a "
        "vision-language model. The crops are averaged into one feature vector per "
        "mask, and a query returns the masks closest to the query text."
    ),
}

SOURCE_GT = "ground_truth"
SOURCE_SCENE = "scene_only"
SOURCE_PRED = "prediction"
COLOR_CLASS = "class"
COLOR_INSTANCE = "instance"
COLOR_TPGT = "tp_gt"
COLOR_QUERY = "query"
KIND_SCENE = "scene"
KIND_SCAN = "scan"
KIND_METHOD = "method"
KIND_RUN = "run"
FOCUS_TREE = "tree"
FOCUS_SETTINGS = "settings"
FOCUS_REPLAY = "replay"
REPLAY_LABELS = {"back": "Back", "toggle": "Pause", "forward": "Forward"}

KEY_TAB = 258
KEY_ENTER = 257
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265
KEY_ESCAPE = 256
KEY_PRESS = 1

SETTING_ROWS = ("benchmark", "mode", "color", "geometry", "boxes", "labels", "ceiling",
                "cad", "query", "view")
PANEL_CLASSES = "classes"
PANEL_PIPELINE = "pipeline"
PANEL_PREVIEW = "preview"
VIEW_CLASSES = "classes"
VIEW_PIPELINE = "pipeline"
VIEW_PATH = "path"
VIEW_REPLAY = "replay"
LABEL_WORLD = {"off": 0.0, "small": 0.05, "large": 0.09}
LABEL_FONT_UNITS = 10.0
SETTING_NAMES = {
    "benchmark": "Label set",
    "mode": "Mode",
    "color": "Color",
    "geometry": "Geometry",
    "boxes": "Boxes",
    "labels": "Labels",
    "ceiling": "Ceiling",
    "cad": "CAD report",
    "query": "Query",
    "view": "View",
}


def _class_color(name, nyu40map, palette, _cache={}):
    if name not in _cache:
        nyu40id = nyu40map.get(name)
        if nyu40id is not None and nyu40id < len(palette):
            _cache[name] = np.array(palette[nyu40id], dtype=float) / 255.0
        else:
            _cache[name] = _PALETTE[len(_cache) % len(_PALETTE)]
    return _cache[name]


def instance_color(index):
    """Deterministic PASCAL-VOC-style RGB, unique for up to 2^24 - 1 instances."""
    label = int(index) + 1
    if label <= 0 or label >= (1 << 24):
        raise ValueError("instance index must be in [0, 2^24 - 2]")
    red = green = blue = 0
    for bit in range(8):
        red |= (label & 1) << (7 - bit)
        green |= ((label >> 1) & 1) << (7 - bit)
        blue |= ((label >> 2) & 1) << (7 - bit)
        label >>= 3
    return np.array([red, green, blue], dtype=float) / 255.0


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
    t_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=1.0)
    bounds = t_mesh.get_axis_aligned_bounding_box()
    center = np.array(
        ((bounds.min_bound + bounds.max_bound) / 2).numpy(), dtype=float)
    mesh = t_mesh.to_legacy()
    # create_text emits inverted winding: the capped face is culled from its
    # own outward side (verified: hollow/mirrored glyphs on screen). Flip once
    # so the cap survives back-face culling, then derive normals from it.
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.triangles)[:, ::-1])
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.paint_uniform_color(color)
    base_vertices = np.asarray(mesh.vertices, dtype=float) - center
    return mesh, base_vertices


def _place_label(mesh, base_vertices, position, char_size, x_dir, y_dir):
    # Update vertices in place: the legacy visualizer binds the original
    # vertex buffer at add_geometry time, so replacing the attribute object
    # would leave the renderer showing stale geometry.
    # Proper rotation (det +1): local +X is the reading direction on
    # screen-right, local +Y is glyph-up, local +Z faces the camera. The old
    # code negated z (a reflection, det -1), which flipped winding and showed
    # hollow mirrored outlines. Verified on screen: solid correct glyphs.
    z_dir = np.cross(x_dir, y_dir)
    rotation = np.column_stack([x_dir, y_dir, z_dir])
    np.asarray(mesh.vertices)[:, :] = (
        (base_vertices * char_size) @ rotation.T + position)
    # Normals must follow the same rotation or lighting stays glued to the
    # font frame while the glyphs turn. (Placement happens once per label.)
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        np.asarray(mesh.vertex_normals)[:, :] = normals @ rotation.T
    if mesh.has_triangle_normals():
        normals = np.asarray(mesh.triangle_normals)
        np.asarray(mesh.triangle_normals)[:, :] = normals @ rotation.T
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
        "score": src.get("score"),
        "verdict": src.get("verdict"),
        "bounds": object_bounds(src["points"], up_axis),
        "box": None,
        "label": None,
    }


def capped_geometry(session, kind, cropper):
    key = "mesh_capped" if kind == "mesh" else "pointcloud_capped"
    geom = session.get(key)
    if geom is None:
        src = session["mesh"] if kind == "mesh" else session["pointcloud"]
        geom = cropper(src, session["up_axis"], session["ceiling_val"])
        session[key] = geom
    return geom


def ensure_object_decorations(obj, color, label_x_dir, label_y_dir, up_axis,
                              label_height=0.05, want_box=True, want_label=True):
    """Build box + text label once. Labels are static world-size geometry:
    `label_height` is the cap height in meters, facing is fixed at creation,
    and nothing is rewritten per frame — zoom and orbit never rescale them."""
    if want_box:
        if obj.get("box") is None:
            box = obj["pcd"].get_axis_aligned_bounding_box()
            box.color = color
            obj["box"] = box
        else:
            obj["box"].color = color
    if not want_label:
        return obj
    if obj.get("label") is None:
        bounds = obj.get("bounds")
        if bounds is None:
            if obj.get("box") is not None:
                label_pos = np.array(obj["box"].get_center())
                label_pos[up_axis] = obj["box"].max_bound[up_axis] + 0.05
            elif obj.get("pcd") is not None:
                bb = obj["pcd"].get_axis_aligned_bounding_box()
                label_pos = np.array(bb.get_center())
                label_pos[up_axis] = bb.max_bound[up_axis] + 0.05
            else:
                return obj
        else:
            label_pos = np.array(bounds["anchor"], dtype=float)
        mesh, base = _label_base(obj.get("query_label") or obj["class_name"], color)
        _place_label(mesh, base, label_pos,
                     label_height / LABEL_FONT_UNITS,
                     label_x_dir, label_y_dir)
        obj["label"] = {"mesh": mesh, "base": base, "anchor": label_pos}
    else:
        obj["label"]["mesh"].paint_uniform_color(color)
    return obj


def fp_count(score):
    if not score:
        return None
    return sum(1 for value in (score.get("verdicts") or {}).values() if value == "fp")


def colorize_detections(detections, color_of):
    """Attach 0-255 RGB colors to adapter boxes.

    Adapters return 3-tuples (xyxy, name, score); the drawing helper needs
    4-tuples (xyxy, name, score, color). 4-tuples pass through untouched so
    mixed inputs never crash the Replay tick.
    """
    out = []
    for det in detections or []:
        if len(det) == 4:
            out.append(tuple(det))
            continue
        xyxy, name, score = det
        out.append((xyxy, name, score, tuple(color_of(name))))
    return out


def colorize_masks(masks, color_of):
    """Attach 0-255 RGB colors to adapter masks (same contract as boxes)."""
    out = []
    for item in masks or []:
        if len(item) == 4:
            out.append(tuple(item))
            continue
        mask, name, score = item
        out.append((mask, name, score, tuple(color_of(name))))
    return out


def format_metric(value):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    return f"{value:.3f}"


def information_payload(scene_id, selected_pred, scores, run_metrics,
                        recon_times=None, pred_times=None, masks=None):
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
        "ignored": None,
        "masks_pred": None,
        "masks_gt": None,
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
    if masks:
        info["masks_pred"] = masks.get("pred")
        info["masks_gt"] = masks.get("gt")
        if masks.get("tp") is not None:
            info["tp"] = masks["tp"]
            info["fp"] = masks.get("fp")
            info["ignored"] = masks.get("ignored")
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


def visible_setting_rows(option_lists):
    out = []
    for name in SETTING_ROWS:
        specs = option_lists.get(name) or []
        if any(spec.get("enabled", True) for spec in specs):
            out.append(name)
    return out


def clamp_setting_index(state, option_lists=None):
    visible = visible_setting_rows(option_lists) if option_lists is not None else list(SETTING_ROWS)
    if not visible:
        state["setting_index"] = 0
        return 0
    idx = min(max(0, state.get("setting_index", 0)), len(SETTING_ROWS) - 1)
    row = SETTING_ROWS[idx]
    if row not in visible:
        state["setting_index"] = SETTING_ROWS.index(visible[0])
        return state["setting_index"]
    state["setting_index"] = idx
    return idx


def handle_settings(state, key, option_lists):
    action = {"apply": None}
    visible = visible_setting_rows(option_lists)
    if not visible:
        state["setting_index"] = 0
        return action
    idx = clamp_setting_index(state, option_lists)
    row = SETTING_ROWS[idx]
    if key in (KEY_UP, KEY_DOWN):
        state["candidate"] = None
        pos = visible.index(row)
        nxt = pos + (-1 if key == KEY_UP else 1)
        if 0 <= nxt < len(visible):
            state["setting_index"] = SETTING_ROWS.index(visible[nxt])
        return action
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
        reopen_query = row == "query" and value in (QUERY_SEARCH, QUERY_IMAGE)
        if value in enabled and (value != applied or reopen_query):
            action["apply"] = (row, value)
    return action


def handle_replay(state, key):
    """Arrow-select Back / Pause-Play / Forward, Enter confirms.

    Mirrors the settings contract: Left/Right move a selection, Enter
    applies it. Returns {"transport": option or None}.
    """
    action = {"transport": None}
    idx = int(state.get("replay_index", 1)) % len(REPLAY_TRANSPORT)
    if key == KEY_LEFT:
        state["replay_index"] = move_replay_selection(idx, -1)
    elif key == KEY_RIGHT:
        state["replay_index"] = move_replay_selection(idx, 1)
    elif key == KEY_ENTER:
        action["transport"] = REPLAY_TRANSPORT[idx]
    return action


def _focus_order(state):
    order = [FOCUS_TREE, FOCUS_SETTINGS]
    if state.get("camera_mode") == CAMERA_REPLAY:
        order.append(FOCUS_REPLAY)
    return order


def handle_navigation(state, key, rows, option_lists):
    if key == KEY_TAB:
        state["candidate"] = None
        order = _focus_order(state)
        try:
            pos = order.index(state.get("focus"))
        except ValueError:
            pos = 0
        state["focus"] = order[(pos + 1) % len(order)]
        return {"activate": None, "apply": None}
    if state["focus"] == FOCUS_TREE:
        action = handle_tree(state, key, rows)
        action["apply"] = None
        return action
    if state["focus"] == FOCUS_REPLAY:
        action = handle_replay(state, key)
        action["activate"] = None
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
        "labels": state.get("labels", "off"),
        "ceiling": state["ceiling_hidden"],
        "cad": state.get("cad_open", False),
        "query": state.get("query_mode", QUERY_OFF),
        "view": state.get("view", VIEW_CLASSES),
    }[row]


def settings_payload(state, option_lists, display_value):
    pending = state.get("candidate")
    visible = visible_setting_rows(option_lists)
    idx = clamp_setting_index(state, option_lists)
    out = []
    for name in visible:
        specs = [spec for spec in (option_lists.get(name) or []) if spec.get("enabled", True)]
        applied = state_value(state, name)
        target = pending[1] if pending is not None and pending[0] == name else applied
        i = SETTING_ROWS.index(name)
        options = []
        for spec in specs:
            value = spec["value"]
            if i == idx and pending is not None and pending[0] == name and value == pending[1] and value != applied:
                kind = "pending"
            elif value == applied:
                kind = "applied"
            else:
                kind = "idle"
            options.append({
                "text": display_value(name, value),
                "kind": kind,
                "sel": False,
                "enabled": True,
            })
        if i == idx:
            for option, spec in zip(options, specs):
                if spec["value"] == target:
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


def eligible_gt_counts(gt_ids, spec, min_region_size=MIN_REGION_SIZE):
    gt_ids = np.asarray(gt_ids, dtype=np.int64)
    counts = Counter()
    instance_ids, sizes = np.unique(gt_ids, return_counts=True)
    for instance_id, size in zip(instance_ids, sizes):
        label_id = int(instance_id) // 1000
        if (instance_id >= 1000 and int(size) >= int(min_region_size)
                and label_id in spec.id_to_label):
            counts[spec.id_to_label[label_id]] += 1
    return counts


def class_count_stats(objects, classes, gt_counts=None):
    pred = _instance_counts(objects)
    tp = Counter(o["class_name"] for o in objects if o.get("verdict") == "tp")
    fp = Counter(o["class_name"] for o in objects if o.get("verdict") == "fp")
    ignored = Counter(o["class_name"] for o in objects if o.get("verdict") == "ignored")
    rows = {}
    for name in classes:
        rows[name] = {
            "pred": int(pred[name]),
            "gt": None if gt_counts is None else int(gt_counts[name]),
            "tp": int(tp[name]),
            "fp": int(fp[name]),
            "ignored": int(ignored[name]),
        }
    return rows


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
        "gt": gt,
        "gt_counts": _instance_counts(gt["objects"]) if gt else Counter(),
        "eligible_gt_counts": {},
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


def query_result_lines(mode, keys, scores, labels=None):
    lines = []
    for key, score in zip(keys or [], scores or []):
        name = os.path.basename(key)
        lines.append(f"{float(score):.3f}  {name}")
        if len(lines) >= 5:
            break
    return lines


def visualize(scene_id=None, scannet_dir=DEFAULT_SCANNET_DIR, benchmark="ScanNet200", run_id=None,
              ceiling_height=2.0):
    import multiprocessing
    import subprocess
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
        "labels": "off",
        "ceiling_hidden": False,
        "status": "Enter a scan",
        "focus": FOCUS_TREE,
        "candidate": None,
        "expanded": set(),
        "cursor_id": scene_node_id(first_scene) if first_scene else None,
        "active_pred": None,
        "active_method": None,
        "cad_open": False,
        "query_mode": QUERY_OFF,
        "view": VIEW_CLASSES,
        # camera_mode/panel mirror "view" for the render loop and HUD payload;
        # "view" is the single setting the user edits.
        "camera_mode": CAMERA_OFF,
        "panel": PANEL_CLASSES,
        # replay_index selects Back / Pause-Play / Forward in FOCUS_REPLAY.
        "replay_index": 1,
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
    displayed = {"objs": set(), "boxes": set(), "labels": set(), "camera": set()}
    objects = []
    overlay = {"pcd": None}
    query_client = QueryClient()
    query_rt = {
        "request_id": 0, "status": "", "labels": {}, "scores": {}, "ranks": [],
        "reset_id": 0, "pending": None, "features": None, "input": "",
        "openins": None, "native_result": None,
    }
    prompt_rt = {
        "process": None, "recv": None, "mode": None,
    }
    camera_rt = {
        "sequence": None, "calib": None, "poses": {}, "frame": 0, "playing": False,
        "last_tick": None,
        "geoms": {"path": None, "marker": None}, "video": None, "status": "",
        "video_revision": 0, "color_size": None,
        "replay": None, "replay_key": None,
        # RAM-only replay cache: phase is idle/loading/ready/error, ids the
        # ordered frame keys, cache maps fid -> final RGBA frame bytes.
        # Nothing is ever written to disk; everything drops on close.
        "phase": "idle", "ids": [], "cache": {}, "requested": set(),
        "cache_t0": None, "last_report": None,
    }

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

    def _eligible_scene_gt_counts():
        if session is None:
            return None
        pred = _pred()
        spec = _spec(pred["label_set"] if pred else state["benchmark"])
        cached = session["eligible_gt_counts"]
        if spec.name in cached:
            return cached[spec.name]
        gt_file = os.path.join(
            artifact_paths(spec, scannet_root)["gt"], session["id"] + ".txt")
        if not os.path.isfile(gt_file):
            cached[spec.name] = None
            return None
        try:
            cached[spec.name] = eligible_gt_counts(util_3d.load_ids(gt_file), spec)
        except (OSError, ValueError):
            cached[spec.name] = None
        return cached[spec.name]

    def _cad_png():
        if session is None:
            return None
        return cad_report_png(session["dir"])

    def _clear_query(status=""):
        query_rt["request_id"] += 1
        query_rt["status"] = status
        query_rt["labels"] = {}
        query_rt["scores"] = {}
        query_rt["ranks"] = []
        query_rt["pending"] = None
        query_rt["input"] = ""
        query_rt["native_result"] = None
        query_rt["reset_id"] += 1
        proc = query_rt.get("openins")
        query_rt["openins"] = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _close_query_prompt():
        proc = prompt_rt.get("process")
        recv = prompt_rt.get("recv")
        prompt_rt.update({"process": None, "recv": None, "mode": None})
        if recv is not None:
            try:
                recv.close()
            except OSError:
                pass
        if proc is not None:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=1)

    def _launch_query_prompt(mode):
        _close_query_prompt()
        recv, send = context.Pipe(duplex=False)
        proc = context.Process(
            target=query_prompt.run,
            args=(send, os.getpid(), mode, view_rect),
            daemon=True)
        proc.start()
        send.close()
        prompt_rt.update({"process": proc, "recv": recv, "mode": mode})
        state["focus"] = FOCUS_SETTINGS
        query_rt["status"] = "Waiting for prompt"

    def _poll_query_prompt():
        proc = prompt_rt.get("process")
        recv = prompt_rt.get("recv")
        if proc is None or recv is None:
            return False
        answer = None
        ready = False
        try:
            if recv.poll():
                answer = recv.recv()
                ready = True
        except (EOFError, OSError):
            ready = True
        if not ready and proc.is_alive():
            return False
        mode = prompt_rt.get("mode")
        _close_query_prompt()
        state["focus"] = FOCUS_SETTINGS
        query_rt["status"] = ""
        if answer and state["query_mode"] == mode:
            _submit_query(answer)
        _update_hud()
        return True

    def _load_query_features():
        query_rt["features"] = None
        pred = _pred()
        if pred is None or session is None:
            return
        spec = _spec(pred["label_set"])
        root = submission_dir(spec, pred["run_id"], pred["model"], scannet_root)
        index_path = os.path.join(root, session["id"] + ".txt")
        path = clip_features_path(spec, pred["run_id"], pred["model"], session["id"], scannet_root)
        query_rt["features"] = load_clip_features(
            path, spec, pred["run_id"], pred["model"], session["id"], index_path)

    def _load_camera_sequence():
        camera_rt["sequence"] = None
        camera_rt["calib"] = None
        camera_rt["poses"] = {}
        camera_rt["color_size"] = None
        camera_rt["status"] = ""
        if session is None:
            return
        frames_dir = os.path.join(session["dir"], "frames")
        discovered = discover_frames(frames_dir)
        if not discovered["rgb_ids"]:
            return
        calib = load_calibration(frames_dir)
        poses = {fid: load_pose(frames_dir, fid) for fid in discovered["rgb_ids"]}
        camera_rt["sequence"] = discovered
        camera_rt["calib"] = calib
        camera_rt["poses"] = poses
        first = os.path.join(frames_dir, "color", f"{discovered['rgb_ids'][0]}.jpg")
        try:
            rgb = load_color_rgb(first)
            camera_rt["color_size"] = (rgb.shape[1], rgb.shape[0])
        except Exception:
            camera_rt["color_size"] = None

    def _detach_camera():
        clear_tracked(vis, displayed, "camera")
        camera_rt["geoms"]["path"] = None
        camera_rt["geoms"]["marker"] = None
        camera_rt["video"] = None
        camera_rt["video_revision"] = 0
        camera_rt["playing"] = False
        camera_rt["last_tick"] = None
        camera_rt["phase"] = "idle"
        camera_rt["ids"] = []
        camera_rt["cache"] = {}
        camera_rt["requested"] = set()
        camera_rt["cache_t0"] = None
        camera_rt["last_report"] = None

    def _rebuild_camera_path():
        clear_tracked(vis, displayed, "camera")
        camera_rt["geoms"]["path"] = None
        camera_rt["geoms"]["marker"] = None
        if state["camera_mode"] == CAMERA_OFF or camera_rt["sequence"] is None:
            return
        poses = [camera_rt["poses"].get(fid) for fid in camera_rt["sequence"]["rgb_ids"]]
        polylines = trajectory_polylines(poses)
        if not polylines:
            camera_rt["status"] = "tracking lost"
            return
        import open3d as o3d
        points = []
        lines = []
        colors = []
        offset = 0
        total = sum(len(line) for line in polylines)
        seen = 0
        for line in polylines:
            for i, pt in enumerate(line):
                points.append(pt)
                t = seen / max(total - 1, 1)
                colors.append([0.1 + 0.8 * t, 0.4, 0.9 - 0.6 * t])
                if i:
                    lines.append([offset + i - 1, offset + i])
                seen += 1
            offset += len(line)
        path = o3d.geometry.LineSet()
        path.points = o3d.utility.Vector3dVector(np.asarray(points))
        path.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        path.colors = o3d.utility.Vector3dVector(np.asarray(colors[:len(points)]))
        camera_rt["geoms"]["path"] = path
        vis.add_geometry(path, reset_bounding_box=False)
        displayed["camera"].add(path)

    def _update_camera_marker(pose):
        """Classical wireframe camera pyramid at the current replay pose."""
        marker = camera_rt["geoms"].get("marker")
        seq = camera_rt["sequence"]
        calib = camera_rt.get("calib")
        size = camera_rt.get("color_size")
        if (state["camera_mode"] != CAMERA_REPLAY
                or camera_rt.get("phase") != "ready"
                or pose is None or not is_valid_pose(pose)
                or calib is None or calib.get("K_color") is None
                or not size or seq is None):
            if marker in displayed["camera"]:
                vis.remove_geometry(marker, reset_bounding_box=False)
                displayed["camera"].discard(marker)
            return
        try:
            origin, corners = frustum_corners(
                pose, calib["K_color"], size[0], size[1])
        except Exception:
            return
        pts = np.vstack([origin, corners])
        segs = np.asarray(FRUSTUM_LINES, dtype=np.int32)
        if marker is None:
            marker = o3d.geometry.LineSet()
            camera_rt["geoms"]["marker"] = marker
            vis.add_geometry(marker, reset_bounding_box=False)
            displayed["camera"].add(marker)
        marker.points = o3d.utility.Vector3dVector(pts)
        marker.lines = o3d.utility.Vector2iVector(segs)
        marker.paint_uniform_color([1.0, 0.55, 0.1])
        if marker in displayed["camera"]:
            vis.update_geometry(marker)

    def _replay_frame_count():
        if state["camera_mode"] != CAMERA_REPLAY:
            return 0
        return len(camera_rt.get("ids") or [])

    def _replay_caption():
        if state["camera_mode"] != CAMERA_REPLAY:
            return None
        adapter = camera_rt.get("replay")
        if adapter is None:
            return None
        return getattr(adapter, "caption", None) or None

    def _close_replay():
        adapter = camera_rt.get("replay")
        camera_rt["replay"] = None
        camera_rt["replay_key"] = None
        camera_rt["phase"] = "idle"
        camera_rt["ids"] = []
        camera_rt["cache"] = {}
        camera_rt["requested"] = set()
        camera_rt["cache_t0"] = None
        camera_rt["last_report"] = None
        camera_rt["video"] = None
        camera_rt["status"] = ""
        if state.get("focus") == FOCUS_REPLAY:
            state["focus"] = FOCUS_SETTINGS
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass

    def _replay_color(name):
        rgb01 = _class_color(name, nyu40map, palette)
        return tuple(int(round(v * 255)) for v in rgb01)

    def _ensure_replay():
        """Return (adapter, reason). Reused while the selection is unchanged."""
        if state["camera_mode"] != CAMERA_REPLAY:
            return None, ""
        pred = _pred()
        key = (pred["run_id"], pred["model"], pred["label_set"],
               session["id"] if session else None) if pred else None
        adapter = camera_rt.get("replay")
        if adapter is not None and camera_rt.get("replay_key") == key:
            return adapter, ""
        _close_replay()
        try:
            spec = _spec(pred["label_set"] if pred else state["benchmark"])
            adapter = replay2d.build(
                pred, sequence=camera_rt["sequence"], spec=spec,
                scannet_root=scannet_root, color_of=_replay_color)
        except Exception as exc:
            return None, str(exc)
        camera_rt["replay"] = adapter
        camera_rt["replay_key"] = key
        return adapter, ""

    def _composite_replay_entry(entry):
        """Composite one adapter entry into final RGBA frame bytes (RAM)."""
        rgb = np.asarray(entry["image"], dtype=np.uint8)
        small = resize_rgb(rgb)
        sx = small.shape[1] / max(rgb.shape[1], 1)
        small = draw_detections(
            small, colorize_detections(entry.get("boxes"), _replay_color),
            scale=sx)
        if entry.get("masks"):
            small = draw_masks(
                small, colorize_masks(entry["masks"], _replay_color), scale=sx)
        return rgb_to_rgba_bytes(small)

    def _start_replay_cache():
        """Build the adapter and begin RAM-only prefetch on replay-press."""
        camera_rt["phase"] = "loading"
        camera_rt["ids"] = []
        camera_rt["cache"] = {}
        camera_rt["requested"] = set()
        camera_rt["video"] = None
        camera_rt["frame"] = 0
        camera_rt["playing"] = False
        camera_rt["last_tick"] = None
        camera_rt["last_report"] = None
        camera_rt["cache_t0"] = time.monotonic()
        state["replay_index"] = 1
        state["focus"] = FOCUS_REPLAY
        adapter, reason = _ensure_replay()
        if adapter is None:
            camera_rt["phase"] = "error"
            camera_rt["status"] = reason or "replay unavailable"
            return False
        try:
            camera_rt["ids"] = list(adapter.frames() or [])
        except Exception:
            camera_rt["ids"] = []
        camera_rt["status"] = "starting 2D workers…"
        return True

    def _pump_replay_cache():
        """One non-blocking prefetch slice. Returns True when HUD-worthy
        progress happened. Playback starts only once every frame is cached."""
        if (state["camera_mode"] != CAMERA_REPLAY
                or camera_rt.get("phase") != "loading"):
            return False
        adapter = camera_rt.get("replay")
        if adapter is None:
            camera_rt["phase"] = "error"
            camera_rt["status"] = "replay unavailable"
            return True
        try:
            ids = list(adapter.frames() or [])
        except Exception as exc:
            camera_rt["phase"] = "error"
            camera_rt["status"] = f"replay failed: {exc}"
            return True
        if ids:
            camera_rt["ids"] = ids
        else:
            try:
                adapter.poll()
            except Exception:
                pass
            camera_rt["status"] = "loading 2D frames…"
            report = (0, 0, camera_rt["status"])
            if report != camera_rt["last_report"]:
                camera_rt["last_report"] = report
                return True
            return False
        _done, status = fill_replay_cache(
            adapter, camera_rt["ids"], camera_rt["cache"],
            camera_rt["requested"], _composite_replay_entry)
        total = len(camera_rt["ids"])
        have = len(camera_rt["cache"])
        if status:
            camera_rt["status"] = status
        elif have < total:
            camera_rt["status"] = f"caching video {have}/{total}…"
        if have >= total and total:
            camera_rt["phase"] = "ready"
            camera_rt["frame"] = 0
            camera_rt["playing"] = True
            camera_rt["last_tick"] = None
            camera_rt["status"] = ""
            _show_cached_frame(0)
            camera_rt["last_report"] = None
            return True
        report = (have, total, camera_rt["status"])
        if report != camera_rt["last_report"]:
            camera_rt["last_report"] = report
            return True
        return False

    def _show_cached_frame(index):
        """Display one fully-cached frame. No disk, worker, or PIL work."""
        ids = camera_rt.get("ids") or []
        if not ids:
            return False
        fid = ids[int(index) % len(ids)]
        rgba = camera_rt["cache"].get(fid)
        if rgba is None:
            return False
        pose = camera_rt["poses"].get(fid if isinstance(fid, int) else -1)
        _update_camera_marker(pose)
        camera_rt["video_revision"] += 1
        camera_rt["video"] = {
            "frame_id": int(fid),
            "revision": camera_rt["video_revision"],
            "rgba": rgba,
            "ready": True,
        }
        return True

    def _invalidate_replay_cache():
        """Drop composited frames after a recolor; keep the warm worker."""
        if state["camera_mode"] != CAMERA_REPLAY:
            return
        adapter = camera_rt.get("replay")
        if adapter is None:
            _start_replay_cache()
            return
        try:
            adapter.invalidate_colors()
        except Exception:
            pass
        camera_rt["cache"] = {}
        camera_rt["requested"] = set()
        camera_rt["video"] = None
        camera_rt["frame"] = 0
        camera_rt["playing"] = False
        camera_rt["last_tick"] = None
        camera_rt["last_report"] = None
        camera_rt["cache_t0"] = time.monotonic()
        camera_rt["phase"] = "loading"
        try:
            camera_rt["ids"] = list(adapter.frames() or [])
        except Exception:
            pass

    def _ensure_encoder():
        pred = _pred()
        feats = query_rt.get("features")
        if pred is None or feats is None:
            return False
        family, name, _dim = ENCODERS[pred["model"]]
        query_client.ensure(method_python(pred["model"]), family, name, device="cuda:0")
        return True

    def _start_openins_query(text, mode):
        pred = _pred()
        if pred is None or session is None:
            return
        query_rt["request_id"] += 1
        request_id = query_rt["request_id"]
        query_rt["status"] = "Querying snapshots on GPU 0"
        query_rt["pending"] = request_id
        classes = [text.strip()]
        classes = [item for item in classes if item]
        if not classes:
            query_rt["status"] = "empty prompt"
            return
        script = os.path.join(os.path.dirname(__file__), "..", "predict", "models", "_openins3d_query.py")
        out_json = os.path.join("/tmp", f"openins-query-{os.getpid()}-{request_id}.json")
        spec = _spec(pred["label_set"])
        root = submission_dir(spec, pred["run_id"], pred["model"], scannet_root)
        param = os.path.join(
            artifact_paths(spec, scannet_root)["evaluations"], pred["run_id"],
            f"{pred['model']}.parameters.json")
        args = [
            method_python("openins3d"), os.path.abspath(script),
            "--pointcloud", os.path.join(session["dir"], f"{session['id']}_vh_clean_2.ply"),
            "--out-json", out_json, "--scene-id", session["id"],
            "--submission-root", root, "--snap-root", openins_snap_root(pred["run_id"], session["id"]),
            "--classes", *classes, "--request-id", str(request_id),
            "--run-id", pred["run_id"],
        ]
        if os.path.isfile(param):
            args += ["--parameters-json", param]

        def work():
            proc = None
            try:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "0"
                env.pop("SPELLBOOK_GPU_LEASE_FD", None)
                proc = subprocess.Popen(args, env=env)
                query_rt["openins"] = proc
                returncode = proc.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, args)
                if query_rt["pending"] != request_id:
                    return
                with open(out_json) as fh:
                    payload = json.load(fh)
                if int(payload.get("request_id", -1)) != request_id:
                    return
                query_rt["native_result"] = {
                    "id": request_id,
                    "mode": mode,
                    "labels": dict(zip(payload.get("keys") or [], payload.get("labels") or [])),
                    "scores": dict(zip(payload.get("keys") or [], payload.get("scores") or [])),
                }
            except Exception as exc:
                if query_rt["pending"] == request_id:
                    query_rt["native_result"] = {"id": request_id, "error": str(exc)}
            finally:
                if query_rt.get("openins") is proc:
                    query_rt["openins"] = None
                try:
                    os.unlink(out_json)
                except OSError:
                    pass

        import threading
        threading.Thread(target=work, daemon=True).start()

    def _apply_native_query_response(msg):
        if not msg or msg.get("id") != query_rt.get("pending"):
            return False
        error = msg.get("error")
        if error:
            query_rt["status"] = error
            query_rt["pending"] = None
            return True
        labels = msg.get("labels") or {}
        query_rt["labels"] = labels
        query_rt["scores"] = {
            key: float(value) for key, value in (msg.get("scores") or {}).items()
        }
        if msg.get("mode") == QUERY_SEARCH:
            ranked = sorted(
                ((key, query_rt["scores"].get(key, 0.0)) for key in labels if labels[key]),
                key=lambda item: -item[1])
            query_rt["ranks"] = [key for key, _score in ranked[:10]]
        else:
            query_rt["ranks"] = []
        query_rt["status"] = ""
        query_rt["pending"] = None
        return True

    def _submit_query(text):
        mode = state["query_mode"]
        query_rt["input"] = text
        pred = _pred()
        if pred is None or mode == QUERY_OFF:
            return
        if pred["model"] in NATIVE_LOOKUP_MODELS:
            _start_openins_query(text, mode)
            return
        if not _ensure_encoder():
            query_rt["status"] = "CLIP features unavailable"
            return
        query_rt["request_id"] += 1
        request_id = query_rt["request_id"]
        query_rt["pending"] = request_id
        query_rt["status"] = "Encoding"
        spec = _spec(pred["label_set"])
        root = submission_dir(spec, pred["run_id"], pred["model"], scannet_root)
        payload = {
            "op": "image" if mode == QUERY_IMAGE else "search",
            "features_path": clip_features_path(
                spec, pred["run_id"], pred["model"], session["id"], scannet_root),
            "index_path": os.path.join(root, session["id"] + ".txt"),
        }
        if mode == QUERY_IMAGE:
            payload["path"] = text
        else:
            payload["text"] = text
        try:
            sent = query_client.submit(payload)
            query_rt["request_id"] = sent
            query_rt["pending"] = sent
        except Exception as exc:
            query_rt["status"] = str(exc)
            query_rt["pending"] = None

    def _apply_query_response(msg):
        if not msg or msg.get("id") != query_rt.get("pending"):
            if msg and not msg.get("ok") and query_rt.get("pending") is None:
                query_rt["status"] = msg.get("error") or "query failed"
            return False
        if not msg.get("ok"):
            query_rt["status"] = msg.get("error") or "query failed"
            query_rt["pending"] = None
            return True
        keys = msg.get("keys") or []
        query_rt["ranks"] = keys
        query_rt["scores"] = dict(zip(keys, msg.get("scores") or []))
        query_rt["labels"] = {}
        query_rt["status"] = ""
        query_rt["pending"] = None
        return True

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
        if row == "labels":
            return _opts(["off", "small", "large"])
        if row == "ceiling":
            return _opts([False, True])
        if row == "cad":
            return _opts([False, True], bool(_cad_png()))
        if row == "query":
            pred = _pred()
            model = pred["model"] if pred else None
            has_features = query_rt.get("features") is not None
            if not has_features and pred is not None and session is not None:
                has_features = clip_path_ready(clip_features_path(
                    _spec(pred["label_set"]), pred["run_id"], pred["model"],
                    session["id"], scannet_root))
            modes = query_modes_for(
                model, has_features=has_features,
                has_snap=bool(pred) and openins_snap_complete(pred["run_id"], pred["scene_id"]),
                source_pred=state["source_mode"] == SOURCE_PRED)
            enabled = {QUERY_OFF: True, QUERY_SEARCH: QUERY_SEARCH in modes,
                       QUERY_IMAGE: QUERY_IMAGE in modes}
            return _opts([QUERY_OFF, QUERY_SEARCH, QUERY_IMAGE], enabled)
        if row == "view":
            has_frames = camera_rt["sequence"] is not None and bool(camera_rt["sequence"]["rgb_ids"])
            pred = _pred()
            can_replay, _reason = replay2d.available(pred, camera_rt["sequence"])
            return _opts(
                [VIEW_CLASSES, VIEW_PIPELINE, VIEW_PATH, VIEW_REPLAY],
                {VIEW_CLASSES: True,
                 VIEW_PIPELINE: bool(pred and pred["model"] in MODEL_PIPELINE),
                 VIEW_PATH: has_frames,
                 VIEW_REPLAY: bool(has_frames and can_replay)})
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
        if row == "labels":
            return {"off": "Off", "small": "Small", "large": "Large"}[value]
        if row == "ceiling":
            return "Hidden" if value else "Visible"
        if row == "cad":
            return "Open" if value else "Closed"
        if row == "query":
            return {QUERY_OFF: "Off", QUERY_SEARCH: "Search", QUERY_IMAGE: "Image"}[value]
        if row == "view":
            return {VIEW_CLASSES: "Classes", VIEW_PIPELINE: "Pipeline",
                    VIEW_PATH: "Path", VIEW_REPLAY: "Replay"}.get(value, str(value))
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
        nonlocal objects
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
            _load_query_features()
            if state["camera_mode"] == CAMERA_REPLAY:
                _start_replay_cache()
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
        _load_query_features()
        if state["camera_mode"] == CAMERA_REPLAY:
            _render_replay_frame()

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
        rank_set = set(query_rt["ranks"]) if state["query_mode"] in (QUERY_SEARCH, QUERY_IMAGE) and query_rt["ranks"] else None
        keep = []
        for i, obj in enumerate(objects):
            if rank_set is not None and obj.get("key") not in rank_set:
                continue
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
        want_box = bool(state["boxes"])
        label_height = LABEL_WORLD.get(state.get("labels", "off"), 0.0)
        want_text = state.get("labels", "off") != "off"
        # Face new labels toward the live camera once; never touch them again.
        try:
            _params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            _ext = np.asarray(_params.extrinsic)
            _cam = -_ext[:3, :3].T @ _ext[:3, 3]
        except Exception:
            _cam = None
        if _cam is not None and np.linalg.norm(_cam) > 1e-6:
            _to_cam = _cam - session["scene_pts"].mean(0)
            _to_cam[session["up_axis"]] = 0.0
            if np.linalg.norm(_to_cam) < 1e-6:
                _to_cam = np.array([1.0, 0.0, 0.0])
                _to_cam[session["up_axis"]] = 0.0
            _to_cam /= np.linalg.norm(_to_cam)
            _x_dir = np.cross(session["up_vec"], _to_cam)
            if np.linalg.norm(_x_dir) < 1e-6:
                _x_dir = np.array([1.0, 0.0, 0.0])
            _x_dir /= np.linalg.norm(_x_dir)
        else:
            _x_dir = np.array([1.0, 0.0, 0.0])
            _x_dir[session["up_axis"]] = 0.0
            _x_dir /= max(np.linalg.norm(_x_dir), 1e-9)
        if want_box or want_text:
            for i in keep:
                obj = objects[i]
                ensure_object_decorations(
                    obj, _object_color(i, obj),
                    _x_dir, session["up_vec"], session["up_axis"],
                    label_height=label_height, want_box=want_box, want_label=want_text)
                if want_box and obj.get("box") is not None:
                    want_boxes.add(obj["box"])
                if want_text and obj.get("label") is not None:
                    want_labels.add(obj["label"]["mesh"])
        return want_objs, want_boxes, want_labels

    def _apply_diff(displayed_set, wanted_set):
        for g in displayed_set - wanted_set:
            vis.remove_geometry(g, reset_bounding_box=False)
        for g in wanted_set - displayed_set:
            vis.add_geometry(g, reset_bounding_box=False)
        return wanted_set

    def _sync():
        want_objs, want_boxes, want_labels = _wanted()
        displayed["objs"] = _apply_diff(displayed["objs"], want_objs)
        displayed["boxes"] = _apply_diff(displayed["boxes"], want_boxes)
        displayed["labels"] = _apply_diff(displayed["labels"], want_labels)

    def _object_color(idx, o):
        if state["color_mode"] == COLOR_QUERY:
            score = float(query_rt["scores"].get(o.get("key"), 0.0))
            t = max(0.0, min(1.0, (score + 1.0) * 0.5))
            return colormaps["plasma"](t)[:3]
        if state["color_mode"] == COLOR_TPGT:
            if o["verdict"] == "tp":
                return TP_COLOR
            if o["verdict"] == "fp":
                return FP_COLOR
            if o["verdict"] == "ignored":
                return IGNORED_COLOR
            return _class_color(o.get("query_label") or o["class_name"], nyu40map, palette)
        if state["color_mode"] == COLOR_INSTANCE:
            return instance_color(idx)
        name = o.get("query_label") or o["class_name"]
        return _class_color(name, nyu40map, palette)

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
        overlay["pcd"] = None
        clear_tracked(vis, displayed, "objs")
        clear_tracked(vis, displayed, "boxes")
        clear_tracked(vis, displayed, "labels")
        _detach_camera()
        if current.get("placeholder") is not None:
            vis.remove_geometry(current["placeholder"], reset_bounding_box=False)
            current["placeholder"] = None
        if current["scene"] is not None:
            vis.remove_geometry(current["scene"], reset_bounding_box=False)
            current["scene"] = None

    def _hud_payload():
        counts = _instance_counts(objects) if objects else Counter()
        eligible_gt = _eligible_scene_gt_counts()
        if eligible_gt is not None:
            presence = eligible_gt
        elif state["source_mode"] == SOURCE_PRED:
            presence = counts
        elif session and session["gt_counts"]:
            presence = session["gt_counts"]
        else:
            presence = counts
        classes = sorted(presence)
        class_stats = {}
        if state["source_mode"] == SOURCE_PRED and eligible_gt is not None:
            class_stats = class_count_stats(objects, classes, eligible_gt)
        png = _cad_png()
        pred = _pred()
        masks = None
        if state["source_mode"] == SOURCE_PRED and objects:
            masks = {
                "pred": len(objects),
                "gt": sum(eligible_gt.values()) if eligible_gt else None,
                "tp": sum(1 for o in objects if o.get("verdict") == "tp"),
                "fp": sum(1 for o in objects if o.get("verdict") == "fp"),
                "ignored": sum(1 for o in objects if o.get("verdict") == "ignored"),
            }
        return {
            "status": state["status"],
            "settings": settings_payload(state, _option_lists(), _display_value),
            "info": information_payload(
                session["id"] if session else None, pred, scores, run_metrics,
                recon_times, pred_times, masks),
            "pipeline": (
                {"method": pred["model"], "text": MODEL_PIPELINE[pred["model"]]}
                if pred and pred["model"] in MODEL_PIPELINE else None),
            "classes": classes,
            "colors": {name: tuple(_class_color(name, nyu40map, palette)) for name in classes},
            "class_stats": class_stats,
            "focus": state["focus"],
            "tree": tree_payload_rows(
                _tree_rows(), state["cursor_id"], state["active_pred"],
                session["id"] if session else None, state.get("active_method")),
            "cad_overlay": png if state.get("cad_open") and png else None,
            "query": {
                "mode": state["query_mode"],
                "status": query_rt["status"],
                "input": query_rt["input"],
                "reset_id": query_rt["reset_id"],
                "can_search": any(
                    opt["value"] == QUERY_SEARCH and opt.get("enabled", True)
                    for opt in _option_list("query")),
                "can_image": any(
                    opt["value"] == QUERY_IMAGE and opt.get("enabled", True)
                    for opt in _option_list("query")),
                "results": query_result_lines(
                    state["query_mode"], query_rt["ranks"],
                    [query_rt["scores"].get(k, 0.0) for k in query_rt["ranks"]],
                    list(query_rt["labels"].values())),
            },
            "camera": {
                "mode": state["camera_mode"],
                "frame": camera_rt["frame"],
                "count": _replay_frame_count(),
                "playing": camera_rt["playing"],
                "status": camera_rt.get("status") or "",
                "caption": _replay_caption(),
                "phase": camera_rt.get("phase") or "idle",
                "done": len(camera_rt.get("cache") or {}),
                "total": len(camera_rt.get("ids") or []),
                "elapsed": (time.monotonic() - camera_rt["cache_t0"]
                            if camera_rt.get("cache_t0") else 0.0),
                "replay_index": int(state.get("replay_index", 1)) % len(REPLAY_TRANSPORT),
            },
            "video": None if state.get("cad_open") else camera_rt.get("video"),
            "panel": state.get("panel", PANEL_CLASSES),
        }

    def _update_hud():
        payload = _hud_payload()
        publish_left({"tree": payload["tree"]})
        publish_right({k: payload[k] for k in (
            "settings", "classes", "colors", "class_stats", "status", "focus", "cad_overlay", "info",
            "query", "camera", "video", "pipeline", "panel")})

    def _normalize_color():
        if state["source_mode"] == SOURCE_SCENE:
            return
        if state["color_mode"] == COLOR_QUERY and state["query_mode"] not in (
                QUERY_SEARCH, QUERY_IMAGE):
            state["color_mode"] = COLOR_CLASS
        if state["color_mode"] == COLOR_TPGT and (
                state["source_mode"] != SOURCE_PRED or not _tp_available()):
            state["color_mode"] = COLOR_CLASS

    def _apply(row, value):
        need_reload = False
        recolor_replay = False
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
            recolor_replay = True
        elif row == "geometry":
            if value == state["geometry"]:
                return
            state["geometry"] = value
            _swap_scene()
        elif row == "boxes":
            state["boxes"] = value
        elif row == "labels":
            state["labels"] = value
        elif row == "ceiling":
            if value == state["ceiling_hidden"]:
                return
            state["ceiling_hidden"] = value
            if session is not None:
                _swap_scene()
        elif row == "cad":
            state["cad_open"] = bool(value)
            if state["cad_open"]:
                camera_rt["playing"] = False
            _update_hud()
            return
        elif row == "query":
            if value == QUERY_OFF:
                _close_query_prompt()
            state["query_mode"] = value
            _clear_query()
            if value in (QUERY_SEARCH, QUERY_IMAGE):
                state["color_mode"] = COLOR_QUERY
                _launch_query_prompt(value)
            elif state["color_mode"] == COLOR_QUERY:
                state["color_mode"] = COLOR_CLASS
            recolor_replay = True
        elif row == "view":
            if value == state.get("view", VIEW_CLASSES):
                return
            camera_value = {
                VIEW_REPLAY: CAMERA_REPLAY, VIEW_PATH: CAMERA_PATH,
            }.get(value, CAMERA_OFF)
            if state["camera_mode"] == CAMERA_REPLAY:
                _close_replay()
            state["view"] = value
            state["camera_mode"] = camera_value
            state["panel"] = {
                VIEW_REPLAY: PANEL_PREVIEW, VIEW_PIPELINE: PANEL_PIPELINE,
            }.get(value, PANEL_CLASSES)
            camera_rt["frame"] = 0
            camera_rt["last_tick"] = None
            camera_rt["playing"] = False
            _rebuild_camera_path()
            if camera_value == CAMERA_REPLAY:
                _start_replay_cache()
            else:
                camera_rt["video"] = None
        _normalize_color()
        if session is None:
            _update_hud()
            return
        if need_reload:
            _load_source()
        _paint()
        _sync()
        if recolor_replay and state["camera_mode"] == CAMERA_REPLAY:
            _invalidate_replay_cache()
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
        _close_query_prompt()
        state["query_mode"] = QUERY_OFF
        _clear_query()
        _load_camera_sequence()
        if state["camera_mode"] != CAMERA_OFF:
            _rebuild_camera_path()
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

    def _sync_query_objects():
        for obj in objects:
            obj["query_label"] = None
        _paint()
        _sync()
        if state["camera_mode"] == CAMERA_REPLAY:
            _invalidate_replay_cache()

    def _replay_ids():
        if state["camera_mode"] != CAMERA_REPLAY:
            return []
        return list(camera_rt.get("ids") or [])

    def _exec_replay_transport(option):
        """Execute a confirmed Back / Pause-Play / Forward action."""
        if (state["camera_mode"] != CAMERA_REPLAY
                or camera_rt.get("phase") != "ready"):
            return
        ids = camera_rt.get("ids") or []
        if not ids:
            return
        if option == "toggle":
            camera_rt["playing"] = not camera_rt["playing"]
            camera_rt["last_tick"] = None
        elif option in ("back", "forward"):
            delta = -1 if option == "back" else 1
            camera_rt["frame"] = step_replay_frame(
                camera_rt["frame"], len(ids), delta)
            camera_rt["playing"] = False
            camera_rt["last_tick"] = None
            _show_cached_frame(camera_rt["frame"])
        else:
            return
        _update_hud()

    def _handle_hud_action(action):
        kind = action.get("type")
        if kind == "replay_toggle":
            _exec_replay_transport("toggle")
            return
        if kind == "replay_transport":
            _exec_replay_transport(action.get("option"))
            return
        if kind == "replay_frame":
            count = len(_replay_ids())
            if count and camera_rt.get("phase") == "ready":
                camera_rt["frame"] = min(max(0, int(action.get("frame") or 0)), count - 1)
                camera_rt["playing"] = False
                camera_rt["last_tick"] = None
                _show_cached_frame(camera_rt["frame"])
                _update_hud()

    def _dispatch(key):
        if isinstance(key, dict):
            _handle_hud_action(key)
            return
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
        if action.get("transport"):
            _exec_replay_transport(action["transport"])
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
                if isinstance(key, (int, dict)):
                    _dispatch(key)

    def _tick(_vis):
        _drain_keys()
        _poll_query_prompt()
        native = query_rt.get("native_result")
        if native is not None:
            query_rt["native_result"] = None
            if _apply_native_query_response(native):
                _sync_query_objects()
                _update_hud()
        msg = query_client.poll()
        if msg and _apply_query_response(msg):
            _sync_query_objects()
            _update_hud()
        updated = False
        if (state["camera_mode"] == CAMERA_REPLAY and camera_rt["sequence"]
                and not state.get("cad_open")):
            if camera_rt.get("phase") == "loading":
                # Prefetch every frame into RAM first; playback starts only
                # once the whole video is cached.
                if _pump_replay_cache():
                    _update_hud()
                    updated = True
            elif camera_rt.get("phase") == "ready":
                adapter = camera_rt.get("replay")
                fps = getattr(adapter, "fps", REPLAY_FPS) if adapter else REPLAY_FPS
                ids = _replay_ids()
                if ids:
                    # The full video is cached: stepping is a dict lookup,
                    # never disk, worker, or PIL work.
                    nxt, last = advance_playback(
                        camera_rt["playing"], camera_rt["frame"], len(ids),
                        camera_rt["last_tick"], time.monotonic(), fps)
                    if nxt != camera_rt["frame"]:
                        camera_rt["frame"] = nxt
                        camera_rt["last_tick"] = last
                        _show_cached_frame(nxt)
                        _update_hud()
                        updated = True
                    else:
                        camera_rt["last_tick"] = last
        return updated

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
    _close_replay()
    _close_query_prompt()
    query_client.close()
    proc = query_rt.get("openins")
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
    _stop_hud(left_q, left_p, stop_left)
    _stop_hud(right_q, right_p, stop_right)
    vis.destroy_window()
