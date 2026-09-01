import math
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from evaluation.evaluate import write_score_sidecar  # noqa: E402
from utils.hud import _draw_right, ellipsize, format_metric  # noqa: E402
from utils.visualize import (  # noqa: E402
    COLOR_CLASS, COLOR_TPGT, FOCUS_SETTINGS, FOCUS_TREE, KEY_DOWN, KEY_ENTER,
    KEY_LEFT, KEY_RIGHT, KEY_TAB, KEY_UP, KIND_METHOD, KIND_RUN, KIND_SCAN,
    SETTING_ROWS, SOURCE_GT, SOURCE_PRED, SOURCE_SCENE, apply_prediction_labels,
    best_prediction, cad_report_png, camera_label_update, capped_geometry,
    clamp_cursor, clear_tracked, collect_tp_scores, discover_reconstructions,
    flatten_tree, group_scenes, handle_navigation, index_predictions,
    collect_prediction_times, collect_reconstruction_times, information_payload,
    load_cad_accuracy, merge_tp_gt_overlay, method_node_id,
    object_bounds, prediction_artifact_mtime, render_object_fields,
    require_scene, reset_camera_label_state, run_node_id, scan_node_id,
    scene_node_id, settings_payload,
)


def _touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _nav_state(**extra):
    state = {
        "focus": FOCUS_TREE,
        "setting_index": 0,
        "candidate": None,
        "expanded": set(),
        "cursor_id": None,
        "benchmark": "ScanNet200",
        "source_mode": SOURCE_SCENE,
        "selected_pred": None,
        "color_mode": COLOR_CLASS,
        "geometry": "mesh",
        "boxes": True,
        "ceiling_hidden": False,
        "cad_open": False,
    }
    state.update(extra)
    return state


class DiscoverTests(unittest.TestCase):
    def test_discovers_mesh_valid_only(self):
        with tempfile.TemporaryDirectory() as root:
            _touch(os.path.join(root, "scene0568_00", "scene0568_00_vh_clean_2.ply"))
            _touch(os.path.join(root, "scene9004_10", "scene9004_10_vh_clean_2.ply"))
            os.makedirs(os.path.join(root, "scene0000_00"))
            _touch(os.path.join(root, "notes.txt"))
            self.assertEqual(
                discover_reconstructions(root), ["scene0568_00", "scene9004_10"])

    def test_rejects_missing_initial_scene(self):
        with tempfile.TemporaryDirectory() as root:
            _touch(os.path.join(root, "scene0568_00", "scene0568_00_vh_clean_2.ply"))
            with self.assertRaises(FileNotFoundError):
                require_scene("scene9999_00", discover_reconstructions(root), root)

    def test_groups_by_scene_prefix(self):
        groups = group_scenes(["scene0568_00", "scene0568_01", "scene9004_10"])
        self.assertEqual(list(groups), ["scene0568", "scene9004"])
        self.assertEqual(groups["scene0568"], ["scene0568_00", "scene0568_01"])

    def test_index_requires_mask_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            pred = os.path.join(root, "predictions", "ScanNet200", "run-a", "mosaic3d")
            idx = os.path.join(pred, "scene0568_00.txt")
            _touch(idx, "predicted_masks/scene0568_00_000.txt 3 0.9\n")
            self.assertIsNone(prediction_artifact_mtime(pred, "scene0568_00"))
            _touch(os.path.join(pred, "run.json"), "{}")
            json_only = os.path.join(root, "predictions", "ScanNet200", "run-json", "mosaic3d")
            _touch(os.path.join(json_only, "run.json"), "{}")
            found = index_predictions(root, ["scene0568_00"])
            self.assertEqual(found["scene0568_00"], [])
            _touch(os.path.join(pred, "predicted_masks", "scene0568_00_000.txt"), "1\n")
            found = index_predictions(root, ["scene0568_00"])
            self.assertEqual([p["run_id"] for p in found["scene0568_00"]], ["run-a"])


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.groups = group_scenes(["scene0568_00", "scene9004_10"])
        self.pred = {
            "scene0568_00": apply_prediction_labels([
                {"model": "mosaic3d", "run_id": "run-a", "label_set": "ScanNet200",
                 "mtime": 2},
                {"model": "open3dis", "run_id": "run-a", "label_set": "ScanNet200",
                 "mtime": 1},
            ]),
            "scene9004_10": [],
        }

    def test_counts_exclude_gt_and_scene_only(self):
        rows = flatten_tree(self.groups, self.pred, set(), {}, {})
        scene = next(r for r in rows if r["kind"] == "scene" and r["id"][1] == "scene0568")
        self.assertTrue(scene["label"].endswith("[1]"))
        self.assertEqual(len(rows), 2)

    def test_four_level_tree_and_labels(self):
        leaves = apply_prediction_labels([
            {"model": "mosaic3d", "run_id": "run-a", "label_set": "ScanNet20"},
            {"model": "mosaic3d", "run_id": "run-a", "label_set": "ScanNet200"},
        ])
        self.assertTrue(leaves[0]["label"].endswith("20"))
        self.assertTrue(leaves[1]["label"].endswith("200"))
        expanded = {
            scene_node_id("scene0568"), scan_node_id("scene0568_00"),
            method_node_id("scene0568_00", "mosaic3d"),
        }
        rows = flatten_tree(self.groups, self.pred, expanded, {}, {}, benchmark="ScanNet200")
        scan = next(r for r in rows if r["id"] == scan_node_id("scene0568_00"))
        self.assertEqual(scan["label"], "scene0568_00")
        methods = [r for r in rows if r["kind"] == KIND_METHOD]
        self.assertEqual({m["label"] for m in methods}, {"mosaic3d", "open3dis"})
        runs = [r for r in rows if r["kind"] == KIND_RUN]
        self.assertEqual([r["label"] for r in runs], ["run-a"])

    def test_cad_and_tp_suffixes(self):
        expanded = {
            scene_node_id("scene9004"), scan_node_id("scene9004_10"),
            method_node_id("scene9004_10", "mosaic3d"),
        }
        pred = self.pred["scene0568_00"][0]
        pid = run_node_id("scene9004_10", pred)
        rows = flatten_tree(
            self.groups, {"scene0568_00": [], "scene9004_10": self.pred["scene0568_00"]},
            expanded, {"scene9004_10": 21.876}, {pid: {"tp": 3, "gt": 57}},
            benchmark="ScanNet200")
        scan = next(r for r in rows if r["id"] == scan_node_id("scene9004_10"))
        self.assertEqual(scan["suffix"], "CAD acc 21.88 cm")
        method = next(r for r in rows if r["kind"] == KIND_METHOD and r["label"] == "mosaic3d")
        self.assertEqual(method["suffix"], "TP/GT 3/57")
        run = next(r for r in rows if r["kind"] == KIND_RUN)
        self.assertEqual(run["suffix"], "TP/GT 3/57")

    def test_best_prediction_prefers_ap_then_ap50(self):
        leaves = [
            {"model": "mosaic3d", "run_id": "old", "label_set": "ScanNet200", "mtime": 9},
            {"model": "mosaic3d", "run_id": "new", "label_set": "ScanNet200", "mtime": 1},
        ]
        metrics = {
            ("ScanNet200", "old", "mosaic3d"): {"ap": 0.20, "ap50": 0.90},
            ("ScanNet200", "new", "mosaic3d"): {"ap": 0.21, "ap50": 0.10},
        }
        self.assertEqual(best_prediction(leaves, metrics)["run_id"], "new")
        tied = {
            ("ScanNet200", "old", "mosaic3d"): {"ap": 0.20, "ap50": 0.10},
            ("ScanNet200", "new", "mosaic3d"): {"ap": 0.20, "ap50": 0.80},
        }
        self.assertEqual(best_prediction(leaves, tied)["run_id"], "new")
        self.assertEqual(best_prediction(leaves, {})["run_id"], "old")

    def test_best_prediction_prefers_comparable(self):
        leaves = [
            {"model": "mosaic3d", "run_id": "smoke", "label_set": "ScanNet200", "mtime": 9},
            {"model": "mosaic3d", "run_id": "full", "label_set": "ScanNet200", "mtime": 1},
        ]
        metrics = {
            ("ScanNet200", "smoke", "mosaic3d"): {
                "ap": 0.90, "ap50": 0.95, "comparable": False,
            },
            ("ScanNet200", "full", "mosaic3d"): {
                "ap": 0.10, "ap50": 0.20, "comparable": True,
            },
        }
        self.assertEqual(best_prediction(leaves, metrics)["run_id"], "full")


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.groups = group_scenes(["scene0568_00"])
        self.pred = {"scene0568_00": apply_prediction_labels([
            {"model": "mosaic3d", "run_id": "run-a", "label_set": "ScanNet200"},
        ])}
        self.expanded = {scene_node_id("scene0568")}
        self.rows = flatten_tree(self.groups, self.pred, self.expanded, {}, {})
        self.state = _nav_state(
            expanded=self.expanded, cursor_id=scene_node_id("scene0568"))
        self.options = {
            "benchmark": [{"value": "ScanNet20"}, {"value": "ScanNet200"}],
            "mode": [
                {"value": SOURCE_GT}, {"value": SOURCE_SCENE},
                {"value": SOURCE_PRED, "enabled": False},
            ],
            "color": [
                {"value": COLOR_CLASS}, {"value": "instance"},
                {"value": COLOR_TPGT, "enabled": False},
            ],
            "geometry": [{"value": "mesh"}, {"value": "pointcloud"}],
            "boxes": [{"value": True}, {"value": False}],
            "ceiling": [{"value": False}, {"value": True}],
            "cad": [
                {"value": False, "enabled": False},
                {"value": True, "enabled": False},
            ],
        }

    def test_enter_toggles_folder_without_activation(self):
        action = handle_navigation(self.state, KEY_ENTER, self.rows, self.options)
        self.assertIsNone(action["activate"])
        self.assertNotIn(scene_node_id("scene0568"), self.state["expanded"])

    def test_enter_scan_activates(self):
        rows = flatten_tree(self.groups, self.pred, self.state["expanded"], {}, {})
        scan = next(r for r in rows if r["kind"] == KIND_SCAN)
        self.state["cursor_id"] = scan["id"]
        action = handle_navigation(self.state, KEY_ENTER, rows, self.options)
        self.assertEqual(action["activate"]["id"], scan["id"])
        self.assertEqual(action["activate"]["kind"], KIND_SCAN)

    def test_enter_method_and_run_activate(self):
        self.state["expanded"].add(scan_node_id("scene0568_00"))
        self.state["expanded"].add(method_node_id("scene0568_00", "mosaic3d"))
        rows = flatten_tree(self.groups, self.pred, self.state["expanded"], {}, {},
                            benchmark="ScanNet200")
        method = next(r for r in rows if r["kind"] == KIND_METHOD)
        self.state["cursor_id"] = method["id"]
        action = handle_navigation(self.state, KEY_ENTER, rows, self.options)
        self.assertEqual(action["activate"]["kind"], KIND_METHOD)
        self.assertEqual(action["activate"]["prediction"]["run_id"], "run-a")
        run = next(r for r in rows if r["kind"] == KIND_RUN)
        self.state["cursor_id"] = run["id"]
        action = handle_navigation(self.state, KEY_ENTER, rows, self.options)
        self.assertEqual(action["activate"]["id"], run["id"])

    def test_arrows_clamp_and_tab_keeps_state(self):
        handle_navigation(self.state, KEY_DOWN, self.rows, self.options)
        self.assertEqual(self.state["cursor_id"], scan_node_id("scene0568_00"))
        handle_navigation(self.state, KEY_DOWN, self.rows, self.options)
        self.assertEqual(self.state["cursor_id"], scan_node_id("scene0568_00"))
        handle_navigation(self.state, KEY_UP, self.rows, self.options)
        handle_navigation(self.state, KEY_UP, self.rows, self.options)
        self.assertEqual(self.state["cursor_id"], scene_node_id("scene0568"))
        handle_navigation(self.state, KEY_TAB, self.rows, self.options)
        self.assertEqual(self.state["focus"], FOCUS_SETTINGS)
        self.assertEqual(self.state["source_mode"], SOURCE_SCENE)

    def test_left_right_only_toggle_expand(self):
        groups = group_scenes(["scene0568_00", "scene0568_01"])
        pred = {"scene0568_00": [], "scene0568_01": []}
        expanded = {scene_node_id("scene0568")}
        rows = flatten_tree(groups, pred, expanded, {}, {})
        state = _nav_state(expanded=set(expanded), cursor_id=scan_node_id("scene0568_00"))
        handle_navigation(state, KEY_LEFT, rows, self.options)
        self.assertEqual(state["cursor_id"], scan_node_id("scene0568_00"))
        handle_navigation(state, KEY_DOWN, rows, self.options)
        self.assertEqual(state["cursor_id"], scan_node_id("scene0568_01"))
        handle_navigation(state, KEY_RIGHT, rows, self.options)
        self.assertEqual(state["cursor_id"], scan_node_id("scene0568_01"))
        self.assertIn(scan_node_id("scene0568_01"), state["expanded"])

    def test_down_after_expand_method_selects_run(self):
        expanded = {scene_node_id("scene0568"), scan_node_id("scene0568_00")}
        rows = flatten_tree(self.groups, self.pred, expanded, {}, {}, benchmark="ScanNet200")
        method = next(r for r in rows if r["kind"] == KIND_METHOD)
        state = _nav_state(expanded=set(expanded), cursor_id=method["id"])
        handle_navigation(state, KEY_RIGHT, rows, self.options)
        rows = flatten_tree(self.groups, self.pred, state["expanded"], {}, {},
                            benchmark="ScanNet200")
        handle_navigation(state, KEY_DOWN, rows, self.options)
        self.assertEqual(state["cursor_id"][0], KIND_RUN)

    def test_clamp_cursor_falls_back_to_scan(self):
        rows = flatten_tree(self.groups, self.pred, self.expanded, {}, {})
        self.state["cursor_id"] = run_node_id("scene0568_00", self.pred["scene0568_00"][0])
        clamp_cursor(self.state, rows)
        self.assertEqual(self.state["cursor_id"], scan_node_id("scene0568_00"))

    def test_settings_candidate_then_enter(self):
        self.state["focus"] = FOCUS_SETTINGS
        handle_navigation(self.state, KEY_RIGHT, self.rows, self.options)
        self.assertIsNone(self.state["candidate"])
        handle_navigation(self.state, KEY_LEFT, self.rows, self.options)
        self.assertEqual(self.state["candidate"], ("benchmark", "ScanNet20"))
        action = handle_navigation(self.state, KEY_ENTER, self.rows, self.options)
        self.assertEqual(action["apply"], ("benchmark", "ScanNet20"))
        self.state["focus"] = FOCUS_SETTINGS
        self.state["candidate"] = ("benchmark", "ScanNet20")
        handle_navigation(self.state, KEY_DOWN, self.rows, self.options)
        self.assertIsNone(self.state["candidate"])
        self.state["candidate"] = ("mode", SOURCE_GT)
        handle_navigation(self.state, KEY_TAB, self.rows, self.options)
        self.assertIsNone(self.state["candidate"])
        self.assertEqual(self.state["focus"], FOCUS_TREE)

    def test_settings_option_kinds(self):
        self.state["focus"] = FOCUS_SETTINGS
        self.state["candidate"] = ("benchmark", "ScanNet20")
        rows = settings_payload(self.state, self.options, lambda _r, v: str(v))
        kinds = {o["text"]: o["kind"] for o in rows[0]["options"]}
        self.assertEqual(kinds["ScanNet200"], "applied")
        self.assertEqual(kinds["ScanNet20"], "pending")
        self.assertTrue(next(o["sel"] for o in rows[0]["options"] if o["text"] == "ScanNet20"))
        self.assertEqual(rows[0]["name"], "Label set")

    def test_cad_candidate_then_enter(self):
        options = dict(self.options)
        options["cad"] = [{"value": False}, {"value": True}]
        self.state["focus"] = FOCUS_SETTINGS
        self.state["setting_index"] = SETTING_ROWS.index("cad")
        handle_navigation(self.state, KEY_RIGHT, self.rows, options)
        action = handle_navigation(self.state, KEY_ENTER, self.rows, options)
        self.assertEqual(action["apply"], ("cad", True))
        self.state["cad_open"] = True
        handle_navigation(self.state, KEY_LEFT, self.rows, options)
        action = handle_navigation(self.state, KEY_ENTER, self.rows, options)
        self.assertEqual(action["apply"], ("cad", False))

    def test_cad_row_disabled_without_png(self):
        rows = settings_payload(self.state, self.options, lambda _r, v: str(v))
        cad = next(row for row in rows if row["key"] == "cad")
        self.assertTrue(cad["options"])
        self.assertTrue(all(option["kind"] == "disabled" for option in cad["options"]))


class CadPngTests(unittest.TestCase):
    def test_cad_report_png(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(cad_report_png(root))
            path = os.path.join(root, "recon", "cad_comparison.png")
            _touch(path, "x")
            self.assertEqual(cad_report_png(root), path)


class GeometryHelperTests(unittest.TestCase):
    def test_box_cleanup_removes_untracked(self):
        class Dummy:
            def __init__(self):
                self.removed = []

            def remove_geometry(self, geom, reset_bounding_box=False):
                self.removed.append(geom)

        old_box, new_box = object(), object()
        displayed = {"boxes": {old_box}}
        vis = Dummy()
        clear_tracked(vis, displayed, "boxes")
        self.assertEqual(vis.removed, [old_box])
        self.assertEqual(displayed["boxes"], set())
        self.assertIsNot(new_box, old_box)


class CadReportTests(unittest.TestCase):
    def _reference(self):
        return {
            "metric": "observed_surface_voxel_mean_bidirectional_distance_v1",
            "voxel_mm": 10,
            "scenes": {
                "scene9004": {
                    "reference_sha256": "aaa",
                    "visible_voxels_sha256": "bbb",
                }
            },
        }

    def _doc(self, **extra):
        doc = {
            "metric": "observed_surface_voxel_mean_bidirectional_distance_v1",
            "mean_bidirectional_distance_cm": 17.3,
            "accuracy_mean_cm": 21.88,
            "completeness_mean_cm": 12.7,
            "scene": "scene9004_10",
            "reference_sha256": "aaa",
            "visible_voxels_sha256": "bbb",
            "voxel_mm": 10,
        }
        doc.update(extra)
        return doc

    def test_accepts_matching_accuracy(self):
        ref = self._reference()
        with tempfile.TemporaryDirectory() as root:
            scene_dir = os.path.join(root, "scene9004_10")
            _touch(os.path.join(scene_dir, "recon", "geometry_score.yaml"),
                   yaml.safe_dump(self._doc()))
            self.assertAlmostEqual(load_cad_accuracy("scene9004_10", scene_dir, ref), 21.88)

    def test_rejects_stale_and_nan(self):
        ref = self._reference()
        with tempfile.TemporaryDirectory() as root:
            scene_dir = os.path.join(root, "scene9004_10")
            _touch(os.path.join(scene_dir, "recon", "geometry_score.yaml"),
                   yaml.safe_dump(self._doc(reference_sha256="nope")))
            self.assertIsNone(load_cad_accuracy("scene9004_10", scene_dir, ref))
            _touch(os.path.join(scene_dir, "recon", "geometry_score.yaml"),
                   yaml.safe_dump(self._doc(accuracy_mean_cm=float("nan"))))
            self.assertTrue(math.isnan(float("nan")))
            self.assertIsNone(load_cad_accuracy("scene9004_10", scene_dir, ref))
            _touch(os.path.join(scene_dir, "recon", "geometry_score.yaml"),
                   yaml.safe_dump(self._doc(scene="scene9004_00")))
            self.assertIsNone(load_cad_accuracy("scene9004_10", scene_dir, ref))


class SidecarTreeTests(unittest.TestCase):
    def test_collects_valid_sidecar_only(self):
        with tempfile.TemporaryDirectory() as root:
            pred = {"model": "mosaic3d", "run_id": "run-a", "label_set": "ScanNet20"}
            path = os.path.join(root, "derived", "evaluations", "ScanNet20",
                                "run-a", "mosaic3d", "scene0568_00.tp50.json")
            write_score_sidecar(path, {
                "schema": 2, "metric": "scannet_instance_ap50", "iou_threshold": 0.5,
                "min_region_size": 100, "scene_id": "scene0568_00", "label_set": "ScanNet20",
                "run_id": "run-a", "model": "mosaic3d", "tp": 3, "gt": 57, "verdicts": {},
                "index_sha256": "a" * 64, "gt_sha256": "b" * 64,
            })
            scores = collect_tp_scores(root, {"scene0568_00": [pred]})
            self.assertEqual(scores[run_node_id("scene0568_00", pred)]["tp"], 3)


class HudHelperTests(unittest.TestCase):
    def test_ellipsize(self):
        self.assertEqual(ellipsize("abcdefghij", 5, len), "ab...")
        self.assertEqual(ellipsize("ok", 10, len), "ok")

    def test_metric_format(self):
        self.assertEqual(format_metric(0.4863713529), "0.486")
        self.assertEqual(format_metric(None), "-")
        self.assertEqual(format_metric(float("nan")), "-")

    def test_right_information_metrics_render(self):
        class FakeImgui:
            WINDOW_NO_TITLE_BAR = 1
            WINDOW_NO_RESIZE = 2
            WINDOW_NO_MOVE = 4
            WINDOW_NO_COLLAPSE = 8
            WINDOW_NO_SAVED_SETTINGS = 16
            WINDOW_NO_INPUTS = 32

            def __init__(self):
                self.texts = []

            def text(self, value):
                self.texts.append(value)

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        imgui = FakeImgui()
        _draw_right(imgui, {"info": {
            "scene": "scene0568_01",
            "method": "openyolo3d",
            "run": "baseline-openyolo3d",
            "ap": 0.3699671793,
            "ap50": 0.4863713529,
            "ap25": 0.5384671658,
            "tp": 15,
            "gt": 25,
            "fp": 293,
            "reconstruction_s": 1575.8,
            "prediction_s": 825.0,
        }}, 1080, 680)
        self.assertIn("openyolo3d / baseline-openyolo3d", imgui.texts)
        self.assertIn("Reconstruction 26m 16s", imgui.texts)
        self.assertIn("Prediction 13m 45s", imgui.texts)
        self.assertIn("AP 0.370 · AP50 0.486 · AP25 0.538", imgui.texts)
        self.assertIn("TP/GT 15/25 · FP 293", imgui.texts)


class InformationTests(unittest.TestCase):
    def test_information_payload_and_tp_overlay(self):
        pred = {
            "label_set": "ScanNet200",
            "run_id": "baseline-openyolo3d",
            "model": "openyolo3d",
        }
        score = {"tp": 1, "gt": 2, "verdicts": {"a": "fp", "b": "tp"}}
        metrics = {("ScanNet200", "baseline-openyolo3d", "openyolo3d"): {
            "ap": 0.3, "ap50": 0.4, "ap25": 0.5,
        }}
        run_key = run_node_id("scene0568_01", pred)
        info = information_payload(
            "scene0568_01", pred, {run_key: score}, metrics,
            recon_times={"scene0568_01": 88.0}, pred_times={run_key: 12.4})
        self.assertEqual((info["tp"], info["gt"], info["fp"]), (1, 2, 1))
        self.assertEqual((info["ap"], info["ap50"], info["ap25"]), (0.3, 0.4, 0.5))
        self.assertEqual(info["reconstruction_s"], 88.0)
        self.assertEqual(info["prediction_s"], 12.4)
        hidden = information_payload("scene0568_01", None, {}, {})
        self.assertIsNone(hidden["reconstruction_s"])
        self.assertIsNone(hidden["prediction_s"])

        points = np.arange(12, dtype=float).reshape(4, 3)
        objects = [
            {"verdict": "fp", "sel": [True, True, False, False]},
            {"verdict": "tp", "sel": [False, True, True, False]},
        ]
        kept, colors = merge_tp_gt_overlay(points, objects)
        np.testing.assert_array_equal(kept, points[:3])
        np.testing.assert_array_equal(colors[0], [0.90, 0.15, 0.15])
        np.testing.assert_array_equal(colors[1:], [[0.10, 0.80, 0.20]] * 2)


class LazyHelperTests(unittest.TestCase):
    def test_object_bounds_and_lazy_fields(self):
        points = np.array([[0.0, 1.0, 2.0], [4.0, 1.0, 8.0]])
        bounds = object_bounds(points, 1)
        np.testing.assert_array_equal(bounds["min"], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(bounds["max"], [4.0, 1.0, 8.0])
        np.testing.assert_array_equal(bounds["anchor"], [2.0, 1.05, 5.0])
        self.assertIsNone(object_bounds(np.zeros((0, 3)), 2))
        stub = render_object_fields(
            {"class_name": "chair", "points": points, "sel": [True, False]},
            "predicted_masks/a.txt", 1)
        self.assertIsNone(stub["box"])
        self.assertIsNone(stub["label"])
        self.assertEqual(stub["key"], "predicted_masks/a.txt")

    def test_camera_label_throttle_and_final(self):
        state = reset_camera_label_state()
        extrinsic = np.eye(4)
        self.assertTrue(camera_label_update(state, extrinsic, 800.0, 10.0, interval=0.1))
        self.assertFalse(camera_label_update(state, extrinsic, 800.0, 10.05, interval=0.1))
        moved = np.eye(4)
        moved[0, 3] = 1.0
        self.assertFalse(camera_label_update(state, moved, 800.0, 10.05, interval=0.1))
        self.assertTrue(state["pending"])
        self.assertTrue(camera_label_update(state, moved, 800.0, 10.2, interval=0.1))
        self.assertFalse(state["pending"])
        self.assertFalse(camera_label_update(state, moved, 800.0, 10.25, interval=0.1))

    def test_capped_geometry_created_once(self):
        session = {
            "mesh": "mesh",
            "pointcloud": "pcd",
            "mesh_capped": None,
            "pointcloud_capped": None,
            "up_axis": 2,
            "ceiling_val": 1.5,
        }
        calls = []

        def cropper(src, up_axis, ceiling_val):
            calls.append((src, up_axis, ceiling_val))
            return f"capped-{src}"

        first = capped_geometry(session, "mesh", cropper)
        second = capped_geometry(session, "mesh", cropper)
        self.assertEqual(first, "capped-mesh")
        self.assertIs(second, first)
        self.assertEqual(calls, [("mesh", 2, 1.5)])
        self.assertIsNone(session["pointcloud_capped"])


if __name__ == "__main__":
    unittest.main()
