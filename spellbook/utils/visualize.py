import json
import multiprocessing
import os
import subprocess
import sys
from collections import Counter

import glfw
import numpy as np
import open3d as o3d
from matplotlib import colormaps

_BENCHMARKSCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "BenchmarkScripts")
sys.path.insert(0, _BENCHMARKSCRIPTS)
_SPELLBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _SPELLBOOK)
import util  # noqa: E402
import util_3d  # noqa: E402
from benchmark import BENCHMARKS, artifact_paths, resolve_benchmark, submission_dir  # noqa: E402
from utils import hud  # noqa: E402

DEFAULT_SCANNET_DIR = "/data/scannet/scans"
LABEL_MAP_FILE = "/data/scannet/v2/scannetv2-labels.combined.tsv"

_PALETTE = colormaps["tab20"].colors
TARGET_LABEL_PX = 14.0

TP_COLOR = np.array([0.10, 0.80, 0.20], dtype=float)
FP_COLOR = np.array([0.90, 0.15, 0.15], dtype=float)

MIN_REGION_SIZE = 100
AP50_THRESHOLD = 0.5

SETTING_ROWS = ("benchmark", "run", "model", "mode", "geometry", "boxes", "ceiling")
SETTING_NAMES = {
    "benchmark": "Benchmark",
    "run": "Run",
    "model": "Prediction model",
    "mode": "Mode",
    "geometry": "Geometry",
    "boxes": "Boxes",
    "ceiling": "Ceiling",
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
    t_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0)
    raw_width = float(t_mesh.get_axis_aligned_bounding_box().max_bound[0].item())
    mesh = t_mesh.to_legacy()
    mesh.paint_uniform_color(color)
    base_vertices = np.asarray(mesh.vertices) - np.array([raw_width / 2, 0, 0])
    return mesh, base_vertices


def _place_label(mesh, base_vertices, position, char_size, x_dir, y_dir):
    z_dir = np.cross(x_dir, y_dir)
    rotation = np.column_stack([x_dir, y_dir, z_dir])
    mesh.vertices = o3d.utility.Vector3dVector((base_vertices * char_size) @ rotation.T + position)
    return mesh


def load_label_map(label_map_file=LABEL_MAP_FILE):
    return util.read_label_mapping(label_map_file, label_from="raw_category", label_to="nyu40id")


def load_gt_instances(scene_dir):
    scene_id = os.path.basename(scene_dir.rstrip("/"))
    mesh_file = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")
    agg_file = os.path.join(scene_dir, f"{scene_id}.aggregation.json")
    seg_file = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json")

    if not all(os.path.isfile(f) for f in (mesh_file, agg_file, seg_file)):
        return None

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)

    with open(agg_file) as f:
        agg = json.load(f)
    with open(seg_file) as f:
        segs = json.load(f)
    seg_indices = np.asarray(segs["segIndices"], dtype=int)

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


def load_predictions(submission_root, scene_id, points, colors, spec):
    """Predictions in official ScanNet submission layout: <scene_id>.txt one line per instance
    "predicted_masks/<scene_id>_NNN.txt <label_id> <confidence>", masks under predicted_masks/.
    Returns None if the scene has no prediction file in this submission root."""
    scene_file = os.path.join(submission_root, f"{scene_id}.txt")
    if not os.path.isfile(scene_file):
        return None

    instances = util_3d.read_instance_prediction_file(scene_file, submission_root)
    objects = []
    pred_instances = {}
    for mask_file, prediction in instances.items():
        label_id = prediction["label_id"]
        if label_id not in spec.id_to_label:
            continue
        mask = util_3d.load_ids(mask_file) > 0
        objects.append({
            "points": points[mask],
            "colors": colors[mask],
            "class_name": spec.id_to_label[label_id],
            "score": prediction["conf"],
            "sel": mask,
            "key": mask_file,
        })
        pred_instances[mask_file] = {
            "label_id": label_id,
            "conf": prediction["conf"],
            "pred_mask": mask,
        }
    return {"objects": objects, "scene_points": points, "scene_colors": colors,
            "pred_instances": pred_instances}


def predictions_root(spec, scannet_root=None):
    return artifact_paths(spec, scannet_root)["predictions"]


def runs_for_scene(spec, scene_id, scannet_root=None):
    """Run ids whose submission contains <scene_id>.txt for at least one model, newest first."""
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
            scene_file = os.path.join(run_dir, model, scene_id + ".txt")
            if os.path.isfile(scene_file):
                mtime = os.path.getmtime(scene_file)
                newest = mtime if newest is None else max(newest, mtime)
        if newest is not None:
            runs.append((run_id, newest))
    runs.sort(key=lambda entry: -entry[1])
    return [run_id for run_id, _ in runs]


def available_models(run_root, scene_id=None):
    """Model directories under a run that contain <scene_id>.txt (if given), sorted."""
    if not os.path.isdir(run_root):
        return []
    models = []
    for name in sorted(os.listdir(run_root)):
        model_dir = os.path.join(run_root, name)
        if not os.path.isdir(model_dir):
            continue
        if scene_id is not None and not os.path.isfile(os.path.join(model_dir, scene_id + ".txt")):
            continue
        models.append(name)
    return models


def classify_ap50(gt_ids, pred_instances, spec, scene_id, overlap_th=AP50_THRESHOLD):
    """Replay the evaluator's instance assignment loop at one IoU threshold and tag each
    prediction. Uses the evaluator's own overlap graph (Evaluator.assign_instances_for_scan)
    so verdicts agree with benchmark evaluation by construction.

    Returns (verdicts, events):
    verdicts: {mask_path: "tp" | "fp" | "ignored"} for predictions that entered the
                evaluator (valid class, >= MIN_REGION_SIZE vertices); filtered masks
                (invalid class or too small) are absent.
    events:   [(mask_path, y_true, y_score)] in the evaluator's AP50 append order
                (per class: claimed TPs in GT order, then lower-confidence duplicates,
                then unmatched FPs in prediction-file order).
    """
    from scannet200_evaluator import Evaluator
    evaluator = Evaluator(spec.class_labels, spec.valid_ids)
    evaluator.add_gt(gt_ids, scene_id)
    evaluator.add_prediction(pred_instances, scene_id)
    gt2pred, pred2gt = evaluator.assign_instances_for_scan(scene_id)

    def _key(filename):
        return filename[len(scene_id) + 1:]

    verdicts = {}
    events = []
    pred_visited = set()

    for label in spec.class_labels:
        gt_instances = [g for g in gt2pred[label]
                        if g["instance_id"] >= 1000 and g["vert_count"] >= MIN_REGION_SIZE]
        cur_match = [False] * len(gt_instances)
        cur_score = [-float("inf")] * len(gt_instances)
        tp_keys = [None] * len(gt_instances)
        tp_events = []
        dup_events = []
        for gti, gt in enumerate(gt_instances):
            for pred in gt["matched_pred"]:
                if pred["filename"] in pred_visited:
                    continue
                overlap = float(pred["intersection"]) / (
                    gt["vert_count"] + pred["vert_count"] - pred["intersection"])
                if overlap > overlap_th:
                    if cur_match[gti]:
                        max_score = max(cur_score[gti], pred["confidence"])
                        min_score = min(cur_score[gti], pred["confidence"])
                        cur_score[gti] = max_score
                        key = _key(pred["filename"])
                        dup_events.append((key, 0, min_score))
                        verdicts[key] = "fp"
                    else:
                        cur_match[gti] = True
                        cur_score[gti] = pred["confidence"]
                        tp_keys[gti] = _key(pred["filename"])
                        pred_visited.add(pred["filename"])
        for gti, matched in enumerate(cur_match):
            if matched:
                key = tp_keys[gti]
                tp_events.append((key, 1, cur_score[gti]))
                verdicts[key] = "tp"
        events.extend(tp_events)
        events.extend(dup_events)
        for pred in pred2gt[label]:
            found_gt = any(
                float(gt["intersection"]) / (
                    gt["vert_count"] + pred["vert_count"] - gt["intersection"]) > overlap_th
                for gt in pred["matched_gt"])
            if not found_gt:
                num_ignore = pred["void_intersection"]
                for gt in pred["matched_gt"]:
                    if gt["instance_id"] < 1000 or gt["vert_count"] < MIN_REGION_SIZE:
                        num_ignore += gt["intersection"]
                proportion_ignore = float(num_ignore) / pred["vert_count"]
                key = _key(pred["filename"])
                if proportion_ignore <= overlap_th:
                    events.append((key, 0, pred["confidence"]))
                    verdicts[key] = "fp"
                else:
                    verdicts[key] = "ignored"
    return verdicts, events


def _display_size():
    try:
        width, height = subprocess.check_output(
            ["xdotool", "getdisplaygeometry"], text=True).split()
        return int(width), int(height)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 1920, 1080


def _instance_counts(objects):
    return Counter(o["class_name"] for o in objects)


def visualize(scene_id, scannet_dir=DEFAULT_SCANNET_DIR, benchmark="ScanNet20", run_id=None,
              ceiling_height=2.0):
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "No DISPLAY set -- Open3D needs a real or virtual X display to open a window.")

    scene_dir = os.path.join(scannet_dir, scene_id)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"scene not found: {scene_dir}")

    mesh = o3d.io.read_triangle_mesh(os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply"))
    mesh.compute_vertex_normals()
    scene_pts = np.asarray(mesh.vertices)
    scene_colors = np.asarray(mesh.vertex_colors)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(scene_pts)
    pointcloud.colors = o3d.utility.Vector3dVector(scene_colors)
    up_axis = _detect_up_axis(scene_pts)
    floor_val = np.percentile(scene_pts[:, up_axis], 1)
    ceiling_val = floor_val + ceiling_height
    mesh_capped = _below_height(mesh, up_axis, ceiling_val)
    pointcloud_capped = _below_height(pointcloud, up_axis, ceiling_val)

    nyu40map = load_label_map()
    palette = util.create_color_palette()

    horiz_axes = [a for a in range(3) if a != up_axis]
    read_axis = max(horiz_axes, key=lambda a: (scene_pts.max(0) - scene_pts.min(0))[a])
    label_x_dir, label_y_dir = np.zeros(3), np.zeros(3)
    label_x_dir[read_axis] = 1.0
    label_y_dir[up_axis] = 1.0

    gt = load_gt_instances(scene_dir)
    gt_counts = _instance_counts(gt["objects"]) if gt else Counter()

    state = {
        "benchmark": benchmark,
        "run_id": run_id,
        "model": "ground_truth",
        "setting_index": 0,
        "geometry": "mesh",
        "boxes": bool(gt["objects"]) if gt else False,
        "ceiling_hidden": False,
        "color_mode": "class",
        "status": "",
    }

    display_width, display_height = _display_size()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"ScanNet - {scene_id}",
                      width=max(640, display_width - hud.WIDTH - 10),
                      height=max(480, display_height - 100),
                      left=0, top=0)

    current = {"scene": None}

    context = multiprocessing.get_context("spawn")
    hud_updates = context.Queue(maxsize=1)
    hud_process = context.Process(target=hud.run, args=(hud_updates, os.getpid(), scene_id), daemon=True)
    hud_process.start()

    def _spec():
        return resolve_benchmark(state["benchmark"])

    def _tp_available():
        if state["model"] == "ground_truth":
            return False
        gt_file = os.path.join(artifact_paths(_spec())["gt"], scene_id + ".txt")
        return os.path.isfile(gt_file)

    def _run_root():
        if not state["run_id"]:
            return None
        return os.path.join(predictions_root(_spec()), state["run_id"])

    def _option_list(row):
        if row == "benchmark":
            return list(BENCHMARKS)
        if row == "run":
            return [None] + runs_for_scene(_spec(), scene_id)
        if row == "model":
            root = _run_root()
            return ["ground_truth"] + (available_models(root, scene_id) if root else [])
        if row == "mode":
            return ["class", "instance"] + (["tp_fp"] if _tp_available() else [])
        if row == "geometry":
            return ["mesh", "pointcloud"]
        if row == "boxes":
            return [True, False]
        return [False, True]  # ceiling: hidden / visible

    def _state_value(row):
        return {
            "benchmark": state["benchmark"],
            "run": state["run_id"],
            "model": state["model"],
            "mode": state["color_mode"],
            "geometry": state["geometry"],
            "boxes": state["boxes"],
            "ceiling": state["ceiling_hidden"],
        }[row]

    def _display_value(row):
        if row == "run":
            return state["run_id"] or "none"
        if row == "model":
            return "Ground truth" if state["model"] == "ground_truth" else state["model"]
        if row == "mode":
            return {"class": "Classes", "instance": "Instances", "tp_fp": "TP/FP"}[state["color_mode"]]
        if row == "geometry":
            return {"mesh": "Mesh", "pointcloud": "Points"}[state["geometry"]]
        if row == "boxes":
            return "On" if state["boxes"] else "Off"
        if row == "ceiling":
            return "Hidden" if state["ceiling_hidden"] else "Visible"
        return state["benchmark"]

    def _make_objects(data, keys=None):
        objects = []
        for idx, o in enumerate(data["objects"]):
            color = _class_color(o["class_name"], nyu40map, palette)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(o["points"])
            pcd.colors = o3d.utility.Vector3dVector(o["colors"])
            box = pcd.get_axis_aligned_bounding_box()
            box.color = color
            label_pos = np.array(box.get_center())
            label_pos[up_axis] = box.max_bound[up_axis] + 0.05
            label_mesh, label_base = _label_base(o["class_name"], color)
            _place_label(label_mesh, label_base, label_pos, 0.06, label_x_dir, label_y_dir)
            objects.append({
                "pcd": pcd,
                "rgb": o["colors"],
                "class_name": o["class_name"],
                "key": keys[idx] if keys else None,
                "verdict": None,
                "box": box,
                "label": {"mesh": label_mesh, "base": label_base, "anchor": label_pos},
            })
        return objects

    source_cache = {}
    objects = []

    def _load_source():
        nonlocal objects
        key = (state["benchmark"], state["run_id"], state["model"])
        if key in source_cache:
            objects, state["status"] = source_cache[key]
            return
        if state["model"] == "ground_truth":
            if gt is None:
                objects = []
                state["status"] = "No ground truth annotations for this scene."
            else:
                objects = _make_objects(gt)
                state["status"] = ""
        else:
            data = load_predictions(
                submission_dir(_spec(), state["run_id"], state["model"]),
                scene_id, scene_pts, scene_colors, _spec())
            if data is None:
                objects = []
                state["status"] = f"No prediction for model {state['model']} in this scene."
            else:
                objects = _make_objects(data, keys=[o["key"] for o in data["objects"]])
                gt_file = os.path.join(artifact_paths(_spec())["gt"], scene_id + ".txt")
                if os.path.isfile(gt_file):
                    verdicts, _ = classify_ap50(
                        util_3d.load_ids(gt_file), data["pred_instances"], _spec(), scene_id)
                    state["status"] = ""
                else:
                    verdicts = None
                    state["status"] = "TP/FP unavailable: no evaluation GT for this benchmark."
                for o in objects:
                    o["verdict"] = verdicts.get(o["key"]) if verdicts else None
        source_cache[key] = (objects, state["status"])

    def _variant():
        if state["geometry"] == "mesh":
            return mesh_capped if state["ceiling_hidden"] else mesh
        return pointcloud_capped if state["ceiling_hidden"] else pointcloud

    def _swap_scene(reset_view=False):
        if current["scene"] is not None:
            vis.remove_geometry(current["scene"], reset_bounding_box=False)
            current["scene"] = None
        current["scene"] = _variant()
        vis.add_geometry(current["scene"], reset_bounding_box=reset_view)

    displayed = {"objs": set(), "boxes": set(), "labels": set()}

    def _wanted():
        keep = []
        for i, o in enumerate(objects):
            if state["color_mode"] == "tp_fp" and o["verdict"] not in ("tp", "fp"):
                continue
            if state["ceiling_hidden"] and o["box"].max_bound[up_axis] > ceiling_val:
                continue
            keep.append(i)
        want_objs = {objects[i]["pcd"] for i in keep}
        want_boxes = {objects[i]["box"] for i in keep} if state["boxes"] else set()
        want_labels = {objects[i]["label"]["mesh"] for i in keep}
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
        if state["color_mode"] == "tp_fp":
            return TP_COLOR if o["verdict"] == "tp" else FP_COLOR
        if state["color_mode"] == "instance":
            return colormaps["turbo"](idx / max(len(objects), 1))[:3]
        return _class_color(o["class_name"], nyu40map, palette)

    def _paint():
        for idx, o in enumerate(objects):
            color = _object_color(idx, o)
            colors = np.tile(color, (len(o["pcd"].points), 1))
            o["pcd"].colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(o["pcd"])
            o["box"].color = color
            o["label"]["mesh"].paint_uniform_color(color)
            vis.update_geometry(o["label"]["mesh"])
        for o in objects:
            if o["box"] in displayed["boxes"]:
                vis.remove_geometry(o["box"], reset_bounding_box=False)
        displayed["boxes"] = set()

    def _hud_payload():
        counts = _instance_counts(objects)
        classes = sorted(set(gt_counts) | set(counts))
        tp = sum(1 for o in objects if o["verdict"] == "tp")
        fp = sum(1 for o in objects if o["verdict"] == "fp")
        return {
            "scene": scene_id,
            "source": "Ground truth" if state["model"] == "ground_truth" else state["model"],
            "visible_count": tp + fp if state["color_mode"] == "tp_fp" else len(objects),
            "tp": tp if state["color_mode"] == "tp_fp" else None,
            "fp": fp if state["color_mode"] == "tp_fp" else None,
            "status": state["status"],
            "settings": [{"name": SETTING_NAMES[row],
                          "value": _display_value(row),
                          "selected": i == state["setting_index"]}
                         for i, row in enumerate(SETTING_ROWS)],
            "classes": classes,
            "colors": {name: tuple(_class_color(name, nyu40map, palette)) for name in classes},
            "help": "Up/Down select | Left/Right change | mouse view | Esc quit",
        }

    def _update_hud():
        payload = _hud_payload()
        try:
            hud_updates.get_nowait()
        except Exception:
            pass
        try:
            hud_updates.put_nowait(payload)
        except Exception:
            pass

    def _apply(row, value):
        need_reload = False
        if row == "benchmark":
            if value == state["benchmark"]:
                return
            state["benchmark"] = value
            state["run_id"] = None
            state["model"] = "ground_truth"
            need_reload = True
        elif row == "run":
            if value == state["run_id"]:
                return
            state["run_id"] = value
            state["model"] = "ground_truth"
            need_reload = True
        elif row == "model":
            if value == state["model"]:
                return
            state["model"] = value
            need_reload = True
        elif row == "mode":
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
            _swap_scene()
        if state["color_mode"] == "tp_fp" and (
                state["model"] == "ground_truth" or not _tp_available()):
            state["color_mode"] = "class"
        if need_reload:
            _load_source()
        _paint()
        _sync()
        _update_hud()

    def _cycle_row(delta):
        state["setting_index"] = (state["setting_index"] + delta) % len(SETTING_ROWS)
        _update_hud()

    def _cycle_value(delta):
        row = SETTING_ROWS[state["setting_index"]]
        options = _option_list(row)
        current = _state_value(row)
        idx = options.index(current) if current in options else 0
        _apply(row, options[(idx + delta) % len(options)])

    def _key_action(key, handler):
        def callback(_vis):
            handler()
        vis.register_key_callback(key, callback)

    _load_source()
    state["boxes"] = bool(objects)
    _swap_scene(reset_view=True)
    _sync()
    _update_hud()

    _key_action(glfw.KEY_UP, lambda: _cycle_row(-1))
    _key_action(glfw.KEY_DOWN, lambda: _cycle_row(1))
    _key_action(glfw.KEY_LEFT, lambda: _cycle_value(-1))
    _key_action(glfw.KEY_RIGHT, lambda: _cycle_value(1))
    _key_action(ord("Q"), lambda: vis.close())

    up_vec = np.zeros(3)
    up_vec[up_axis] = 1.0

    def _update_labels(_vis):
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        extrinsic = np.asarray(params.extrinsic)
        cam_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
        fy = params.intrinsic.get_focal_length()[1]
        updated = False
        for o in objects:
            lab = o["label"]
            if lab["mesh"] not in displayed["labels"]:
                continue
            to_cam = cam_pos - lab["anchor"]
            to_cam[up_axis] = 0.0
            dist = np.linalg.norm(to_cam)
            if dist < 1e-6:
                continue
            to_cam /= dist
            x_dir = np.cross(up_vec, to_cam)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-6:
                continue
            x_dir /= x_norm
            char_size = TARGET_LABEL_PX * dist / fy
            _place_label(lab["mesh"], lab["base"], lab["anchor"], char_size, x_dir, up_vec)
            vis.update_geometry(lab["mesh"])
            updated = True
        return updated

    vis.register_animation_callback(_update_labels)

    print("Keys: Up/Down select setting, Left/Right change value, Esc/Q quit")

    vis.run()
    try:
        hud_updates.get_nowait()
    except Exception:
        pass
    try:
        hud_updates.put_nowait(None)
    except Exception:
        pass
    hud_process.join(timeout=2)
    if hud_process.is_alive():
        hud_process.terminate()
        hud_process.join(timeout=2)
    hud_updates.close()
    hud_updates.join_thread()
    vis.destroy_window()
