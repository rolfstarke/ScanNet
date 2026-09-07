import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import yaml

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from evaluation.evaluate import write_score_sidecar  # noqa: E402
from utils.hud import (  # noqa: E402
    _draw_class_row, _draw_pipeline_block, _draw_replay_block, _draw_right,
    ellipsize, format_metric)
from utils.visualize import (  # noqa: E402
    COLOR_CLASS, COLOR_TPGT, FOCUS_REPLAY, FOCUS_SETTINGS, FOCUS_TREE,
    KEY_DOWN, KEY_ENTER,
    KEY_LEFT, KEY_RIGHT, KEY_TAB, KEY_UP, KIND_METHOD, KIND_RUN, KIND_SCAN,
    MODEL_PIPELINE, SETTING_ROWS, SOURCE_GT, SOURCE_PRED, SOURCE_SCENE,
    _place_label, apply_prediction_labels,
    best_prediction, cad_report_png, capped_geometry,
    class_count_stats, clamp_cursor, clear_tracked, collect_tp_scores,
    colorize_detections, colorize_masks,
    discover_reconstructions, eligible_gt_counts,
    ensure_object_decorations,
    flatten_tree, group_scenes, handle_navigation, handle_replay,
    index_predictions,
    instance_color,
    collect_prediction_times, collect_reconstruction_times, information_payload,
    load_cad_accuracy, merge_tp_gt_overlay, method_node_id,
    object_bounds, prediction_artifact_mtime, render_object_fields,
    require_scene, run_node_id, scan_node_id,
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
            "query": [{"value": "off"}],
            "view": [{"value": "classes"}],
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

    def test_enter_reopens_active_search_prompt(self):
        self.state["focus"] = FOCUS_SETTINGS
        self.state["setting_index"] = SETTING_ROWS.index("query")
        self.state["query_mode"] = "search"
        options = dict(self.options)
        options["query"] = [{"value": "off"}, {"value": "search"}]
        action = handle_navigation(self.state, KEY_ENTER, self.rows, options)
        self.assertEqual(action["apply"], ("query", "search"))

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

    def test_query_search_stays_visible_when_enabled(self):
        options = dict(self.options)
        options["query"] = [
            {"value": "off"},
            {"value": "search"},
            {"value": "image", "enabled": False},
        ]
        rows = settings_payload(self.state, options, lambda _r, v: str(v))
        query = next(row for row in rows if row["key"] == "query")
        self.assertEqual([opt["text"] for opt in query["options"]], ["off", "search"])

    def test_cad_row_hidden_without_png(self):
        rows = settings_payload(self.state, self.options, lambda _r, v: str(v))
        self.assertFalse(any(row["key"] == "cad" for row in rows))
        mode = next(row for row in rows if row["key"] == "mode")
        self.assertEqual([opt["text"] for opt in mode["options"]],
                         [SOURCE_GT, SOURCE_SCENE])
        color = next(row for row in rows if row["key"] == "color")
        self.assertFalse(any(opt["text"] == COLOR_TPGT for opt in color["options"]))


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
    def test_class_row_draws_pred_gt_bar(self):
        class FakeDraw:
            def __init__(self):
                self.texts = []
                self.rects = []
                self.lines = []

            def add_text(self, *args):
                self.texts.append(args[-1])

            def add_rect_filled(self, *args):
                self.rects.append(args)

            def add_line(self, *args):
                self.lines.append(args)

        class FakeImgui:
            def __init__(self):
                self.draw = FakeDraw()

            def get_window_draw_list(self):
                return self.draw

            def get_cursor_screen_pos(self):
                return 0.0, 0.0

            def get_color_u32_rgba(self, *rgba):
                return rgba

            def dummy(self, *_args):
                pass

        imgui = FakeImgui()
        _draw_class_row(imgui, "chair", (0.2, 0.3, 0.4), {
            "pred": 3, "gt": 2, "tp": 1, "fp": 1, "ignored": 1,
        })
        self.assertIn("chair [3/2]", imgui.draw.texts)
        self.assertEqual(len(imgui.draw.lines), 1)
        # swatch + background + single class-color fill (no TP/FP segments)
        self.assertEqual(len(imgui.draw.rects), 3)
        self.assertEqual(imgui.draw.rects[2][4], (0.2, 0.3, 0.4, 1.0))

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

    def test_query_block_shows_when_search_available(self):
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

            def input_text(self, *_a, **_k):
                return False, ""

            def button(self, *_a, **_k):
                return False

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        imgui = FakeImgui()
        _draw_right(imgui, {
            "info": {"scene": "scene0618_00"},
            "query": {"mode": "off", "can_search": True, "reset_id": 1,
                      "input": "", "results": []},
        }, 1080, 680, hud_state={"query_text": "", "reset_id": None})
        self.assertIn("Query", imgui.texts)


class InformationTests(unittest.TestCase):
    def test_scene_present_classes_and_prediction_stats(self):
        spec = SimpleNamespace(id_to_label={3: "chair", 5: "table"})
        gt_ids = np.array([3001] * 100 + [3002] * 99 + [5001] * 120 + [9001] * 200)
        counts = eligible_gt_counts(gt_ids, spec)
        self.assertEqual(counts, {"chair": 1, "table": 1})
        objects = [
            {"class_name": "chair", "verdict": "tp"},
            {"class_name": "chair", "verdict": "fp"},
            {"class_name": "chair", "verdict": "ignored"},
            {"class_name": "chair", "verdict": None},
            {"class_name": "lamp", "verdict": "fp"},
        ]
        stats = class_count_stats(objects, sorted(counts), counts)
        self.assertEqual(stats["chair"], {
            "pred": 4, "gt": 1, "tp": 1, "fp": 1, "ignored": 1,
        })
        self.assertNotIn("lamp", stats)

    def test_instance_colors_are_unique_beyond_colormap_limit(self):
        colors = {
            tuple(np.rint(instance_color(index) * 255).astype(np.uint8))
            for index in range(1000)
        }
        self.assertEqual(len(colors), 1000)

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
            {"class_name": "chair", "points": points, "sel": [True, False], "score": 0.8},
            "predicted_masks/a.txt", 1)
        self.assertIsNone(stub["box"])
        self.assertIsNone(stub["label"])
        self.assertEqual(stub["key"], "predicted_masks/a.txt")
        self.assertEqual(stub["score"], 0.8)

    def test_static_labels_use_world_height(self):
        from utils.visualize import LABEL_FONT_UNITS, LABEL_WORLD
        up = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])

        def _obj():
            return {"class_name": "chair",
                    "bounds": {"anchor": np.array([1.0, 2.0, 3.0])},
                    "box": None, "label": None, "pcd": None}

        small = _obj()
        ensure_object_decorations(
            small, (1.0, 0.0, 0.0), x, up, 2,
            label_height=LABEL_WORLD["small"],
            want_box=False, want_label=True)
        v = np.asarray(small["label"]["mesh"].vertices)
        # cap height is 10 font units -> world height equals label_height
        self.assertAlmostEqual(
            float(v[:, 2].max() - v[:, 2].min()), 0.05, places=6)
        self.assertAlmostEqual(
            float(v[:, 0].max() + v[:, 0].min()) / 2.0, 1.0, places=6)
        self.assertGreater(LABEL_FONT_UNITS, 0.0)
        # cap faces the camera side (a third of the triangles), so glyphs
        # survive back-face culling instead of rendering as hollow outlines
        normals = np.asarray(small["label"]["mesh"].triangle_normals)
        facing = float((normals @ np.array([0.0, -1.0, 0.0]) > 0.9).mean())
        self.assertAlmostEqual(facing, 1.0 / 3.0, places=2)
        # static: re-decoration never rewrites placed vertices (zoom-proof)
        before = v.copy()
        ensure_object_decorations(
            small, (0.0, 1.0, 0.0), -x, up, 2,
            label_height=LABEL_WORLD["large"],
            want_box=False, want_label=True)
        np.testing.assert_array_equal(
            np.asarray(small["label"]["mesh"].vertices), before)
        large = _obj()
        ensure_object_decorations(
            large, (1.0, 0.0, 0.0), x, up, 2,
            label_height=LABEL_WORLD["large"],
            want_box=False, want_label=True)
        w = np.asarray(large["label"]["mesh"].vertices)
        self.assertAlmostEqual(
            float(w[:, 2].max() - w[:, 2].min()), 0.09, places=6)

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


class HudContentTests(unittest.TestCase):
    def test_pipeline_texts_cover_all_methods_in_six_sentences(self):
        self.assertEqual(
            sorted(MODEL_PIPELINE),
            ["mosaic3d", "open3dis", "openins3d", "openmask3d", "openyolo3d"])
        for method, text in MODEL_PIPELINE.items():
            sentences = [s for s in text.split(". ") if s.strip()]
            self.assertGreaterEqual(len(sentences), 1, method)
            self.assertLessEqual(len(sentences), 6, method)
            for banned in ("utils/", ".py", "topk", "ckpt", "subprocess"):
                self.assertNotIn(banned, text, method)

    def test_information_payload_reports_official_counts(self):
        pred = {"label_set": "ScanNet20", "run_id": "run-a", "model": "mosaic3d"}
        score = {"tp": 10, "gt": 22, "verdicts": {"a": "fp"}}
        run_key = run_node_id("scene0568_00", pred)
        info = information_payload(
            "scene0568_00", pred, {run_key: score}, {},
            masks={"pred": 598, "gt": 22, "tp": 10, "fp": 558, "ignored": 30})
        self.assertEqual(
            (info["masks_pred"], info["masks_gt"]), (598, 22))
        self.assertEqual((info["tp"], info["fp"], info["ignored"]), (10, 558, 30))
        self.assertEqual(info["gt"], 22)

    def test_label_base_is_float_finite_and_has_normals(self):
        from utils.visualize import _label_base
        mesh, base = _label_base("chair", (1.0, 0.0, 0.0))
        self.assertEqual(base.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(base)))
        self.assertGreater(len(np.asarray(mesh.triangle_normals)), 0)
        # centred on the full bounding box, not just x
        np.testing.assert_allclose(
            base.max(0) + base.min(0), np.zeros(3), atol=1e-9)

    def test_place_label_math_is_finite_scaled_and_positioned(self):
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        base = np.array([[0.0, 0.0, 0.0], [43.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        mesh.vertices = o3d.utility.Vector3dVector(base)
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        pos = np.array([1.0, 2.0, 3.0])
        out = _place_label(mesh, base, pos, 0.1,
                           np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        self.assertEqual(len(out.vertices), 3)
        v = np.asarray(out.vertices)
        self.assertTrue(np.all(np.isfinite(v)))
        np.testing.assert_allclose(v[0], pos, atol=1e-9)
        # Reading direction maps to +x (screen-right), glyph up maps to +z.
        self.assertAlmostEqual(v[1, 0] - v[0, 0], 4.3, places=6)
        self.assertAlmostEqual(v[2, 2] - v[0, 2], 1.0, places=6)

    def test_labels_setting_row_lists_sizes(self):
        state = _nav_state()
        options = {
            "benchmark": [{"value": "ScanNet200"}],
            "labels": [{"value": v} for v in ("off", "small", "large")],
        }
        rows = settings_payload(state, options, lambda _r, v: str(v))
        labels = next(row for row in rows if row["key"] == "labels")
        self.assertEqual(
            [opt["text"] for opt in labels["options"]], ["off", "small", "large"])


class PanelRenderTests(unittest.TestCase):
    class _Imgui:
        WINDOW_NO_TITLE_BAR = 1
        WINDOW_NO_RESIZE = 2
        WINDOW_NO_MOVE = 4
        WINDOW_NO_COLLAPSE = 8
        WINDOW_NO_SAVED_SETTINGS = 16
        WINDOW_NO_INPUTS = 32

        def __init__(self):
            self.texts = []
            self.wrapped = []

        def text(self, value):
            self.texts.append(value)

        def text_wrapped(self, value):
            self.wrapped.append(value)

        def button(self, *_a, **_k):
            return False

        def same_line(self, *_a, **_k):
            return None

        def slider_int(self, *_a, **_k):
            return False, None

        def separator(self, *_a, **_k):
            return None

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    def test_masks_line_and_official_suffix(self):
        imgui = self._Imgui()
        _draw_right(imgui, {"info": {
            "scene": "scene0618_00",
            "method": "openyolo3d",
            "run": "vis-replay-openyolo3d",
            "ap": 0.1, "ap50": 0.2, "ap25": 0.3,
            "tp": 10, "gt": 22, "fp": 558, "ignored": 30,
            "masks_pred": 598, "masks_gt": 22,
        }}, 1080, 680)
        self.assertIn("AP 0.100 · AP50 0.200 · AP25 0.300", imgui.texts)
        self.assertIn("Masks [598/22] · TP 10 · FP 558 · ignored 30", imgui.texts)

    def test_pipeline_block_renders_method_text(self):
        imgui = self._Imgui()
        _draw_pipeline_block(imgui, {"pipeline": {
            "method": "openyolo3d", "text": MODEL_PIPELINE["openyolo3d"]}})
        self.assertIn("Pipeline: openyolo3d", imgui.texts)
        self.assertEqual(imgui.wrapped, [MODEL_PIPELINE["openyolo3d"]])
        quiet = self._Imgui()
        _draw_pipeline_block(quiet, {})
        self.assertEqual(quiet.texts, [])

    def test_replay_block_shows_adapter_caption(self):
        imgui = self._Imgui()
        _draw_replay_block(imgui, {"camera": {
            "mode": "replay", "phase": "ready", "count": 90, "frame": 3,
            "playing": True, "replay_index": 1,
            "status": "", "caption": "YOLO-World · every 10th frame",
        }}, None)
        self.assertIn("YOLO-World · every 10th frame", imgui.texts)
        self.assertIn("3/89", imgui.texts)
        hidden = self._Imgui()
        _draw_replay_block(hidden, {"camera": {"mode": "off"}}, None)
        self.assertEqual(hidden.texts, [])


class ReplayTransportTests(unittest.TestCase):
    class _Imgui:
        def __init__(self, clicks=()):
            self.texts = []
            self.buttons = []
            self._clicks = list(clicks)

        def text(self, value):
            self.texts.append(value)

        def button(self, label):
            self.buttons.append(label)
            return bool(self._clicks.pop(0)) if self._clicks else False

        def same_line(self, *_a, **_k):
            return None

        def separator(self, *_a, **_k):
            return None

        def slider_int(self, *_a, **_k):
            return False, None

        def progress_bar(self, frac):
            self.texts.append(f"progress:{frac:.2f}")

    class _Keys:
        def __init__(self):
            self.sent = []

        def send(self, payload):
            self.sent.append(payload)

    def test_tab_reaches_replay_only_while_active(self):
        state = _nav_state(focus=FOCUS_SETTINGS)
        handle_navigation(state, KEY_TAB, [], {})
        self.assertEqual(state["focus"], FOCUS_TREE)
        state = _nav_state(focus=FOCUS_SETTINGS, camera_mode="replay")
        handle_navigation(state, KEY_TAB, [], {})
        self.assertEqual(state["focus"], FOCUS_REPLAY)
        handle_navigation(state, KEY_TAB, [], {})
        self.assertEqual(state["focus"], FOCUS_TREE)

    def test_arrows_select_enter_confirms_transport(self):
        state = _nav_state(focus=FOCUS_REPLAY, camera_mode="replay",
                           replay_index=1)
        action = handle_navigation(state, KEY_RIGHT, [], {})
        self.assertEqual(state["replay_index"], 2)
        self.assertIsNone(action.get("transport"))
        action = handle_navigation(state, KEY_ENTER, [], {})
        self.assertEqual(action.get("transport"), "forward")
        handle_navigation(state, KEY_LEFT, [], {})
        handle_navigation(state, KEY_LEFT, [], {})
        self.assertEqual(state["replay_index"], 0)
        action = handle_navigation(state, KEY_ENTER, [], {})
        self.assertEqual(action.get("transport"), "back")

    def test_handle_replay_ignores_other_keys(self):
        state = {"replay_index": 1}
        self.assertIsNone(handle_replay(state, KEY_UP)["transport"])
        self.assertEqual(state["replay_index"], 1)

    def test_loading_block_shows_progress_not_transport(self):
        imgui = self._Imgui()
        _draw_replay_block(imgui, {
            "camera": {
                "mode": "replay", "phase": "loading", "count": 0,
                "frame": 0, "playing": False, "replay_index": 1,
                "caption": "YOLO-World · every 10th frame",
                "done": 3, "total": 9, "elapsed": 12.0,
                "status": "loading YOLO-World (12s)…",
            },
        }, None)
        self.assertIn("YOLO-World · every 10th frame", imgui.texts)
        self.assertIn("Caching video 3/9…", imgui.texts)
        self.assertIn("progress:0.33", imgui.texts)
        self.assertIn("12s elapsed", imgui.texts)
        self.assertIn("loading YOLO-World (12s)…", imgui.texts)
        self.assertEqual(imgui.buttons, [])

    def test_ready_block_selects_and_clicks_transport(self):
        keys = self._Keys()
        imgui = self._Imgui(clicks=(False, False, True))
        _draw_replay_block(imgui, {
            "focus": FOCUS_REPLAY,
            "camera": {
                "mode": "replay", "phase": "ready", "count": 90,
                "frame": 3, "playing": False, "replay_index": 2,
                "status": "", "caption": "YOLO-World · every 10th frame",
            },
        }, keys)
        self.assertEqual(imgui.buttons, ["Back", "Play", "> Forward <"])
        self.assertEqual(keys.sent, [{"type": "replay_transport",
                                      "option": "forward"}])

    def test_error_block_reports_status(self):
        imgui = self._Imgui()
        _draw_replay_block(imgui, {
            "camera": {
                "mode": "replay", "phase": "error", "count": 0,
                "frame": 0, "playing": False, "replay_index": 1,
                "status": "2D output not retained for this run",
            },
        }, None)
        self.assertIn("2D output not retained for this run", imgui.texts)
        self.assertEqual(imgui.buttons, [])


class ReplayColorizeTests(unittest.TestCase):
    def test_boxes_gain_color_from_adapter_triples(self):
        dets = colorize_detections(
            [([10, 10, 30, 30], "chair", 0.9)], lambda n: (255, 0, 0))
        self.assertEqual(
            dets, [([10, 10, 30, 30], "chair", 0.9, (255, 0, 0))])

    def test_boxes_pass_colored_quadruples_through(self):
        dets = [([1, 1, 5, 5], "chair", 0.5, (0, 255, 0))]
        self.assertEqual(
            colorize_detections(dets, lambda n: (0, 0, 0)), dets)

    def test_empty_boxes_never_crash(self):
        self.assertEqual(colorize_detections([], lambda n: (0, 0, 0)), [])
        self.assertEqual(colorize_detections(None, lambda n: (0, 0, 0)), [])

    def test_masks_gain_color_from_adapter_triples(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 2:5] = True
        out = colorize_masks([(mask, "chair", 0.8)], lambda n: (255, 0, 0))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1:], ("chair", 0.8, (255, 0, 0)))
        np.testing.assert_array_equal(out[0][0], mask)

    def test_colorized_boxes_draw_without_unpack_error(self):
        from utils.detect_replay import draw_detections
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        dets = colorize_detections(
            [([10, 10, 30, 30], "chair", 0.9)], lambda n: (255, 0, 0))
        out = draw_detections(rgb, dets)
        self.assertEqual(out.shape, (48, 64, 3))
        self.assertGreater(int(out.sum()), 0)


class BottomPanelTests(unittest.TestCase):
    class _Imgui:
        WINDOW_NO_TITLE_BAR = 1
        WINDOW_NO_RESIZE = 2
        WINDOW_NO_MOVE = 4
        WINDOW_NO_COLLAPSE = 8
        WINDOW_NO_SAVED_SETTINGS = 16
        WINDOW_NO_INPUTS = 32

        def __init__(self):
            self.texts = []
            self.images = []

        def text(self, value):
            self.texts.append(value)

        def image(self, *_a, **_k):
            self.images.append(True)

        def get_cursor_screen_pos(self):
            return (0.0, 0.0)

        def get_window_draw_list(self):
            class _Draw:
                def add_rect_filled(self, *_a, **_k):
                    return None

                def add_rect(self, *_a, **_k):
                    return None

                def add_line(self, *_a, **_k):
                    return None

                def add_text(self, *_a, **_k):
                    return None
            return _Draw()

        def get_color_u32_rgba(self, *_a, **_k):
            return 0

        def calc_text_size(self, text):
            class _Size:
                x = float(len(text))
            return _Size()

        def columns(self, *_a, **_k):
            return None

        def next_column(self, *_a, **_k):
            return None

        def dummy(self, *_a, **_k):
            return None

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    def _payload(self, panel=None):
        payload = {
            "info": {"scene": "scene0618_00"},
            "settings": [],
            "classes": ["chair"],
            "colors": {"chair": (1.0, 0.0, 0.0)},
            "class_stats": {},
            "camera": {"mode": "off"},
            "pipeline": {"method": "openyolo3d",
                         "text": MODEL_PIPELINE["openyolo3d"]},
        }
        if panel is not None:
            payload["panel"] = panel
        return payload

    def _video(self):
        return {"kind": "video", "id": 7, "wh": (64, 48)}

    def test_default_panel_shows_classes_only(self):
        imgui = self._Imgui()
        _draw_right(imgui, self._payload(), 1080, 680, overlay=self._video())
        self.assertIn("Classes", imgui.texts)
        self.assertNotIn("Pipeline: openyolo3d", imgui.texts)
        self.assertEqual(imgui.images, [])

    def test_preview_panel_shows_video_only(self):
        imgui = self._Imgui()
        _draw_right(imgui, self._payload("preview"), 1080, 680,
                    overlay=self._video())
        self.assertEqual(len(imgui.images), 1)
        self.assertNotIn("Classes", imgui.texts)
        self.assertNotIn("Pipeline: openyolo3d", imgui.texts)

    def test_pipeline_panel_shows_explainer_only(self):
        imgui = self._Imgui()
        _draw_right(imgui, self._payload("pipeline"), 1080, 680,
                    overlay=self._video())
        self.assertIn("Pipeline: openyolo3d", imgui.texts)
        self.assertNotIn("Classes", imgui.texts)
        self.assertEqual(imgui.images, [])

    def test_view_setting_row_lists_options(self):
        state = _nav_state()
        options = {"view": [{"value": v} for v in
                            ("classes", "pipeline", "path", "replay")]}
        rows = settings_payload(state, options, lambda _r, v: str(v))
        self.assertEqual([row["key"] for row in rows], ["view"])
        view = rows[0]
        self.assertEqual(
            [opt["text"] for opt in view["options"]],
            ["classes", "pipeline", "path", "replay"])


if __name__ == "__main__":
    unittest.main()
