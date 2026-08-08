import json
import multiprocessing
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import open3d as o3d
from matplotlib import colormaps

_BENCHMARKSCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "BenchmarkScripts")
sys.path.insert(0, _BENCHMARKSCRIPTS)
import util  # noqa: E402
import util_3d  # noqa: E402
from . import hud

DEFAULT_SCANNET_DIR = "/data/scannet/scans"
LABEL_MAP_FILE = "/data/scannet/v2/scannetv2-labels.combined.tsv"
PREDICTIONS_ROOT = "predictions"

_PALETTE = colormaps["tab20"].colors
TARGET_LABEL_PX = 14.0


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


def load_predictions(pred_dir, points, colors):
    labels = {}
    with open(os.path.join(pred_dir, "labels.txt")) as f:
        for line in f:
            lid, name = line.strip().split(" ", 1)
            labels[int(lid)] = name

    objects = []
    predictions_file = os.path.join(pred_dir, "predictions.txt")
    for mask_file, prediction in util_3d.read_instance_prediction_file(
            predictions_file, pred_dir).items():
        mask = util_3d.load_ids(mask_file) > 0
        label_id = prediction["label_id"]
        objects.append({
            "points": points[mask],
            "colors": colors[mask],
            "class_name": labels[label_id],
            "score": prediction["conf"],
            "sel": mask,
        })
    return {"objects": objects, "scene_points": points, "scene_colors": colors}


def available_models(scene_dir):
    pred_root = os.path.join(scene_dir, PREDICTIONS_ROOT)
    if not os.path.isdir(pred_root):
        return []
    return sorted(d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d)))


def _display_size():
    try:
        width, height = subprocess.check_output(
            ["xdotool", "getdisplaygeometry"], text=True).split()
        return int(width), int(height)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 1920, 1080


def _instance_counts(objects):
    return Counter(o["class_name"] for o in objects)


def _hud_payload(scene_id, state, objects, gt_counts, nyu40map, palette):
    counts = _instance_counts(objects)
    classes = sorted(set(gt_counts) | set(counts))
    return {
        "scene": scene_id,
        "model": state["model"] or "scene",
        "geometry": state["geometry"],
        "color_mode": state["color_mode"],
        "prediction_mode": state["model"] not in (None, "ground_truth"),
        "counts": dict(counts),
        "ground_truth": dict(gt_counts),
        "classes": classes,
        "colors": {name: tuple(_class_color(name, nyu40map, palette)) for name in classes},
    }


def visualize(scene_id, scannet_dir=DEFAULT_SCANNET_DIR, ceiling_height=2.0):
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
    pred_models = available_models(scene_dir)
    model_options = [None, "ground_truth"] + pred_models
    model = "ground_truth"

    def _load(name):
        objects, boxes, labels = [], [], []
        if name == "ground_truth":
            if gt is None:
                print("No ground truth annotations for this scene.")
                return objects, boxes, labels
            data = gt
        elif name is not None:
            data = load_predictions(os.path.join(scene_dir, PREDICTIONS_ROOT, name),
                                    scene_pts, scene_colors)
        else:
            return objects, boxes, labels
        for o in data["objects"]:
            color = _class_color(o["class_name"], nyu40map, palette)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(o["points"])
            pcd.colors = o3d.utility.Vector3dVector(o["colors"])
            objects.append({"pcd": pcd, "rgb": o["colors"], "class_name": o["class_name"]})
            box = pcd.get_axis_aligned_bounding_box()
            box.color = color
            boxes.append(box)
            label_pos = np.array(box.get_center())
            label_pos[up_axis] = box.max_bound[up_axis] + 0.05
            label_mesh, label_base = _label_base(o["class_name"], color)
            _place_label(label_mesh, label_base, label_pos, 0.06, label_x_dir, label_y_dir)
            labels.append({"mesh": label_mesh, "base": label_base, "anchor": label_pos})
        classes = sorted({o["class_name"] for o in objects})
        print(f"[{name or 'scene'}] {len(objects)} objects" + (f", classes: {classes}" if objects else ""))
        return objects, boxes, labels

    objects, boxes, labels = _load(model)

    display_width, display_height = _display_size()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"ScanNet - {scene_id} - {model or 'scene'}",
                      width=max(640, display_width - hud.WIDTH - 10),
                      height=max(480, display_height - 100),
                      left=0, top=0)

    state = {"boxes": bool(objects), "model": model, "ceiling_hidden": False,
             "geometry": "mesh", "color_mode": "RGB"}
    current = {"scene": None}

    context = multiprocessing.get_context("spawn")
    hud_updates = context.Queue(maxsize=1)
    hud_process = context.Process(target=hud.run, args=(hud_updates, os.getpid(), scene_id), daemon=True)
    hud_process.start()

    def _update_hud():
        payload = _hud_payload(scene_id, state, objects, gt_counts, nyu40map, palette)
        try:
            hud_updates.get_nowait()
        except Exception:
            pass
        try:
            hud_updates.put_nowait(payload)
        except Exception:
            pass

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
            box = boxes[i]
            if state["ceiling_hidden"] and box.max_bound[up_axis] > ceiling_val:
                continue
            keep.append(i)
        want_objs = {objects[i]["pcd"] for i in keep}
        want_boxes = {boxes[i] for i in keep} if state["boxes"] else set()
        want_labels = {labels[i]["mesh"] for i in keep}
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

    _swap_scene(reset_view=True)
    _sync()
    _update_hud()

    def toggle_geometry(_vis):
        state["geometry"] = "pointcloud" if state["geometry"] == "mesh" else "mesh"
        _swap_scene()
        _update_hud()

    def toggle_ceiling(_vis):
        state["ceiling_hidden"] = not state["ceiling_hidden"]
        _swap_scene()
        _sync()
        _update_hud()

    def toggle_boxes(_vis):
        state["boxes"] = not state["boxes"]
        _sync()
        _update_hud()

    def _paint_objects():
        n = max(len(objects), 1)
        for i, o in enumerate(objects):
            if state["color_mode"] == "class":
                color = _class_color(o["class_name"], nyu40map, palette)
                colors = np.tile(color, (len(o["pcd"].points), 1))
            elif state["color_mode"] == "instance":
                color = colormaps["turbo"](i / n)[:3]
                colors = np.tile(color, (len(o["pcd"].points), 1))
            else:
                colors = o["rgb"]
            o["pcd"].colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(o["pcd"])

    def toggle_class_color(_vis):
        state["color_mode"] = "RGB" if state["color_mode"] == "class" else "class"
        _paint_objects()
        _update_hud()

    def toggle_instance_color(_vis):
        state["color_mode"] = "RGB" if state["color_mode"] == "instance" else "instance"
        _paint_objects()
        _update_hud()

    def next_model(_vis):
        nonlocal objects, boxes, labels
        idx = model_options.index(state["model"]) if state["model"] in model_options else 0
        state["model"] = model_options[(idx + 1) % len(model_options)]
        objects, boxes, labels = _load(state["model"])
        state["boxes"] = bool(objects)
        _paint_objects()
        _sync()
        _update_hud()

    vis.register_key_callback(ord("M"), toggle_geometry)
    vis.register_key_callback(ord("H"), toggle_ceiling)
    vis.register_key_callback(ord("B"), toggle_boxes)
    vis.register_key_callback(ord("C"), toggle_class_color)
    vis.register_key_callback(ord("I"), toggle_instance_color)
    vis.register_key_callback(ord("N"), next_model)

    up_vec = np.zeros(3)
    up_vec[up_axis] = 1.0

    def _update_labels(_vis):
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        extrinsic = np.asarray(params.extrinsic)
        cam_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
        fy = params.intrinsic.get_focal_length()[1]
        updated = False
        for lab in labels:
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

    print(f"Keys: M=mesh/pointcloud, H=hide above {ceiling_height:.1f}m, B=toggle boxes, "
          f"C=class/RGB, I=instance/RGB, "
          f"N=next model ({', '.join(m or 'scene' for m in model_options)}), Q/Esc=quit")

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
