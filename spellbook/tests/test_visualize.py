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
    COLOR_CLASS, COLOR_TPFN, FILTER_OFF, FILTER_ON, FOCUS_REPLAY, FOCUS_SETTINGS, FOCUS_TREE,
    KEY_DOWN, KEY_ENTER,
    KEY_LEFT, KEY_RIGHT, KEY_TAB, KEY_UP, KIND_METHOD, KIND_RUN, KIND_SCAN,
    LABEL_SCALE, MODEL_PIPELINE, SETTING_ROWS, SOURCE_GT, SOURCE_PRED, SOURCE_SCENE,
    THRESHOLDS_OFF, THRESHOLDS_ON,
    aabb_edges, apply_prediction_labels,
    best_prediction, box_lineset, build_threshold_rows, cad_report_png, capped_geometry,
    class_count_stats, clamp_cursor, clear_tracked, collect_scene_scores,
    collect_tp_scores,
    colorize_detections, colorize_masks,
    discover_reconstructions, effective_threshold, eligible_gt_counts,
    ensure_object_box, ensure_object_decorations, fitted_thresholds,
    filter_survives, derive_keep_indices, flatten_tree, group_scenes,
    handle_navigation, handle_replay, handle_settings, apply_query_result,
    normalize_query_scores,
    index_predictions,
    instance_color,
    collect_prediction_times, collect_reconstruction_times, information_payload,
    label_scale, label_text,
    load_cad_accuracy, merge_tp_fn_overlay, merge_tp_gt_overlay, method_node_id,
    object_bounds, prediction_artifact_mtime, remove_label_handle, render_object_fields,
    require_scene, run_node_id, scan_node_id, scene_ap_text, eligible_gt_id_masks,
    unmatched_gt_ids,
    scene_node_id, settings_payload, sync_label, threshold_survives,
    union_prediction_classes,
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
            expanded, {"scene9004_10": 21.876},
            {pid: {"tp": 3, "gt": 57, "ap": 0.271, "ap_class_count": 5}},
            benchmark="ScanNet200")
        scan = next(r for r in rows if r["id"] == scan_node_id("scene9004_10"))
        self.assertEqual(scan["suffix"], "CAD acc 21.88 cm")
        method = next(r for r in rows if r["kind"] == KIND_METHOD and r["label"] == "mosaic3d")
        self.assertEqual(method["suffix"], "AP 0.271 (5 classes)")
        run = next(r for r in rows if r["kind"] == KIND_RUN)
        self.assertEqual(run["suffix"], "AP 0.271 (5 classes)")

    def test_best_prediction_prefers_ap_only(self):
        leaves = [
            {"model": "mosaic3d", "run_id": "old", "label_set": "ScanNet200", "mtime": 9},
            {"model": "mosaic3d", "run_id": "new", "label_set": "ScanNet200", "mtime": 1},
        ]
        metrics = {
            ("ScanNet200", "old", "mosaic3d"): {"ap": 0.20},
            ("ScanNet200", "new", "mosaic3d"): {"ap": 0.21},
        }
        self.assertEqual(best_prediction(leaves, metrics)["run_id"], "new")
        tied = {
            ("ScanNet200", "old", "mosaic3d"): {"ap": 0.20},
            ("ScanNet200", "new", "mosaic3d"): {"ap": 0.20},
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
                {"value": COLOR_TPFN, "enabled": False},
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
        self.assertEqual([opt["text"] for opt in query["options"]],
                         ["off", "search", "image"])
        self.assertFalse(query["options"][2]["enabled"])

    def test_cad_row_hidden_without_png(self):
        rows = settings_payload(self.state, self.options, lambda _r, v: str(v))
        self.assertFalse(any(row["key"] == "cad" for row in rows))
        mode = next(row for row in rows if row["key"] == "mode")
        self.assertEqual([opt["text"] for opt in mode["options"]],
                         [SOURCE_GT, SOURCE_SCENE, SOURCE_PRED])
        self.assertFalse(mode["options"][2]["enabled"])
        color = next(row for row in rows if row["key"] == "color")
        tpfn = [opt for opt in color["options"] if opt["text"] == COLOR_TPFN]
        self.assertEqual(len(tpfn), 1)
        self.assertFalse(tpfn[0]["enabled"])


class CadPngTests(unittest.TestCase):
    def test_cad_report_png(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(cad_report_png(root))
            path = os.path.join(root, "recon", "cad_comparison.png")
            _touch(path, "x")
            self.assertEqual(cad_report_png(root), path)


class GeometryHelperTests(unittest.TestCase):
    def test_name_cleanup_removes_registered(self):
        class FakeBackend:
            def __init__(self):
                self.names = {"object/1/0": object(), "box/1/0": object()}
                self.removed = []

            def remove(self, name):
                self.removed.append(name)
                self.names.pop(name, None)

        backend = FakeBackend()
        names = {"object/1/0", "box/1/0"}
        clear_tracked(backend, names)
        self.assertEqual(names, set())
        self.assertEqual(sorted(backend.removed),
                         ["box/1/0", "object/1/0"])
        # idempotent: clearing an empty set makes no backend calls
        clear_tracked(backend, names)
        self.assertEqual(len(backend.removed), 2)


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
                "schema": 4, "metric": "scannet_instance_ap50", "iou_threshold": 0.5,
                "min_region_size": 100, "scene_id": "scene0568_00", "label_set": "ScanNet20",
                "run_id": "run-a", "model": "mosaic3d", "tp": 3, "gt": 3, "verdicts": {
                    "a.txt": "tp", "b.txt": "tp", "c.txt": "tp"},
                "matched_gt": {"a.txt": 3001, "b.txt": 3002, "c.txt": 3003},
                "eligible_gt_by_class": {
                    "cabinet": 0, "bed": 0, "chair": 0, "sofa": 3, "table": 0,
                    "door": 0, "window": 0, "bookshelf": 0, "picture": 0,
                    "counter": 0, "desk": 0, "curtain": 0, "refrigerator": 0,
                    "shower curtain": 0, "toilet": 0, "sink": 0, "bathtub": 0,
                    "otherfurniture": 0},
                "index_sha256": "a" * 64, "gt_sha256": "b" * 64,
                "ap": 0.271, "ap_class_count": 5,
            })
            scores = collect_scene_scores(root, {"scene0568_00": [pred]})
            key = run_node_id("scene0568_00", pred)
            self.assertEqual(scores[key]["tp"], 3)
            self.assertEqual(scores[key]["ap"], 0.271)
            self.assertEqual(scene_ap_text(scores[key]), "AP 0.271 (5 classes)")


class HudHelperTests(unittest.TestCase):
    def test_class_row_confusion_suffix_colors(self):
        class FakeDraw:
            def __init__(self):
                self.texts = []

            def add_text(self, *args):
                self.texts.append((args[-2], args[-1]))

            def add_rect_filled(self, *args):
                pass

            def add_line(self, *args):
                pass

        class FakeImgui:
            def __init__(self):
                self.draw = FakeDraw()

            def get_window_draw_list(self):
                return self.draw

            def get_cursor_screen_pos(self):
                return 0.0, 0.0

            def get_color_u32_rgba(self, *rgba):
                return rgba

            def calc_text_size(self, text):
                class _Size:
                    x = 8.0 * len(text)
                return _Size()

            def dummy(self, *_args):
                pass

        imgui = FakeImgui()
        _draw_class_row(imgui, "chair", (0.2, 0.3, 0.4), {
            "pred": 9, "gt": 12, "tp": 7, "fp": 5, "fn": 5, "ignored": 0,
        }, scale=5)
        texts = [t for _c, t in imgui.draw.texts]
        colors = {t: c for c, t in imgui.draw.texts}
        self.assertIn("TP 7", texts)
        self.assertIn("FP 5", texts)
        self.assertIn("FN 5", texts)
        # TP green; max-scale errors fully dark red.
        self.assertEqual(colors["TP 7"][:3], (0.10, 0.55, 0.25))
        self.assertEqual(colors["FP 5"][:3], (0.62, 0.08, 0.08))
        self.assertTrue(any(t.startswith("chair  P 9  G 12  [") for t in texts))

    def test_class_row_light_red_for_few_errors(self):
        from utils.hud import _row_segments
        segs = dict(_row_segments("h", {"pred": 1, "gt": 4, "tp": 0,
                                        "fp": 1, "fn": 4}, scale=4))
        self.assertEqual(segs["TP 0"], (0.08, 0.08, 0.08))
        fp = segs["FP 1"]
        fn = segs["FN 4"]
        # Few errors: closer to light red than dark red.
        self.assertGreater(fp[0], 0.8)
        self.assertEqual(fn, (0.62, 0.08, 0.08))

    def test_class_row_zero_errors_no_red(self):
        from utils.hud import _row_segments
        segs = dict(_row_segments("h", {"pred": 2, "gt": 2, "tp": 2,
                                        "fp": 0, "fn": 0}, scale=4))
        self.assertNotIn("FN -", segs)
        self.assertEqual(segs["FP 0"], (0.08, 0.08, 0.08))
        self.assertEqual(segs["FN 0"], (0.08, 0.08, 0.08))
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
            "scene_ap": 0.271,
            "scene_ap_classes": 5,
            "run_ap": 0.3699671793,
            "run_ap_classes": 53,
            "run_scenes": 10,
            "visible_pred": 42,
            "eligible_gt": 22,
            "reconstruction_s": 1575.8,
            "prediction_s": 825.0,
        }}, 1080, 680)
        self.assertIn("openyolo3d / baseline-openyolo3d", imgui.texts)
        self.assertIn("Reconstruction 26m 16s", imgui.texts)
        self.assertIn("Prediction 13m 45s", imgui.texts)
        self.assertIn("Scene AP 0.271 (5 evaluated classes)", imgui.texts)
        self.assertIn("Run AP 0.370 (10 scenes, 53 evaluated classes)", imgui.texts)
        self.assertIn("Visible predictions 42 | Eligible GT 22", imgui.texts)
        for banned in ("AP50", "AP25", "ignored", "Masks", "TP/GT", "22/29"):
            self.assertFalse(any(banned in text for text in imgui.texts))

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
            "pred": 4, "gt": 1, "tp": 1, "fp": 1, "fn": 0, "ignored": 1,
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
        score = {"tp": 1, "gt": 2, "verdicts": {"a": "fp", "b": "tp"},
                 "ap": 0.271, "ap_class_count": 2}
        metrics = {("ScanNet200", "baseline-openyolo3d", "openyolo3d"): {
            "ap": 0.3, "ap_count": 53, "scene_count": 10,
        }}
        run_key = run_node_id("scene0568_01", pred)
        info = information_payload(
            "scene0568_01", pred, {run_key: score}, metrics,
            recon_times={"scene0568_01": 88.0}, pred_times={run_key: 12.4},
            visible_pred=2, eligible_gt=2)
        self.assertEqual((info["scene_ap"], info["scene_ap_classes"]), (0.271, 2))
        self.assertEqual((info["run_ap"], info["run_ap_classes"], info["run_scenes"]),
                         (0.3, 53, 10))
        self.assertEqual((info["visible_pred"], info["eligible_gt"]), (2, 2))
        self.assertNotIn("ap50", info)
        self.assertNotIn("ap25", info)
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
        np.testing.assert_array_equal(colors[0], [0.55, 0.08, 0.08])
        np.testing.assert_array_equal(colors[1:], [[0.10, 0.80, 0.20]] * 2)

    def test_tp_fn_overlay_precedence_tp_fn_fp(self):
        points = np.arange(12, dtype=float).reshape(4, 3)
        objects = [
            {"verdict": "fp", "sel": [True, True, True, True]},
            {"verdict": "tp", "sel": [False, True, False, False]},
            {"verdict": "ignored", "sel": [False, False, False, True]},
        ]
        fn = [np.array([False, False, True, False])]
        kept, colors = merge_tp_fn_overlay(points, objects, fn)
        np.testing.assert_array_equal(kept, points)
        # index 0: FP only; index 1: TP wins over FP; index 2: FN wins over
        # FP; index 3: FP only (ignored selection paints nothing).
        np.testing.assert_array_equal(colors[0], [0.55, 0.08, 0.08])
        np.testing.assert_array_equal(colors[1], [0.10, 0.80, 0.20])
        np.testing.assert_array_equal(colors[2], [0.95, 0.55, 0.50])
        np.testing.assert_array_equal(colors[3], [0.55, 0.08, 0.08])

    def test_unmatched_gt_ids(self):
        eligible = {"chair": [3001, 3002], "table": [5001]}
        out = unmatched_gt_ids(eligible, [3001], ["chair", "table"])
        self.assertEqual(out, {"chair": [3002], "table": [5001]})
        self.assertEqual(unmatched_gt_ids(eligible, [3001, 3002, 5001], ["chair"]),
                         {})

    def test_eligible_gt_id_masks(self):
        from evaluation.benchmark import BENCHMARKS
        spec = BENCHMARKS["ScanNet20"]
        label = spec.valid_ids[0]
        gt = np.zeros(250, dtype=np.int32)
        gt[:100] = label * 1000 + 1
        gt[100:200] = label * 1000 + 2
        gt[200:250] = 7
        name = spec.class_labels[0]
        masks = eligible_gt_id_masks(gt, spec)
        self.assertEqual(sorted(masks[name]), [label * 1000 + 1, label * 1000 + 2])
        self.assertTrue(masks[name][label * 1000 + 1][:100].all())
        self.assertFalse(masks[name][label * 1000 + 1][100:].any())


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

    def test_box_lineset_cached_from_bounds(self):
        pts, lines = aabb_edges([0.0, 1.0, 2.0], [4.0, 3.0, 8.0])
        self.assertEqual(pts.shape, (8, 3))
        self.assertEqual(lines.shape, (12, 2))
        np.testing.assert_array_equal(pts.min(0), [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(pts.max(0), [4.0, 3.0, 8.0])
        obj = {"bounds": {"min": np.array([0.0, 1.0, 2.0]),
                          "max": np.array([4.0, 3.0, 8.0])},
               "box": None}
        first = ensure_object_box(obj)
        self.assertIsNotNone(first)
        # cached: second call returns the same LineSet, no rebuild
        self.assertIs(ensure_object_box(obj), first)
        via_decor = ensure_object_decorations(
            {"bounds": obj["bounds"], "box": None}, want_box=True)
        self.assertIsNotNone(via_decor["box"])

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


class LabelHandleTests(unittest.TestCase):
    class _Backend:
        def __init__(self):
            self.adds = []
            self.updates = []
            self.removes = []

        def add_label(self, pos, text, color=(1, 1, 1, 1), scale=1.0):
            handle = SimpleNamespace(
                position=np.asarray(pos, dtype=float), text=text,
                color=tuple(color), scale=float(scale))
            self.adds.append(handle)
            return handle

        def update_label(self, handle, pos=None, text=None, color=None,
                         scale=None):
            self.updates.append(handle)
            if pos is not None:
                handle.position = np.asarray(pos, dtype=float)
            if text is not None:
                handle.text = text
            if color is not None:
                handle.color = tuple(color)
            if scale is not None:
                handle.scale = float(scale)

        def remove_label(self, handle):
            self.removes.append(handle)

    def _obj(self, **extra):
        obj = {"class_name": "chair", "query_label": None,
               "bounds": {"anchor": np.array([1.0, 2.0, 3.0])},
               "label_handle": None, "label_state": None}
        obj.update(extra)
        return obj

    def test_create_once_then_idempotent(self):
        backend = self._Backend()
        obj = self._obj()
        first = sync_label(backend, obj, True, "small",
                           color=(1.0, 0.0, 0.0))
        self.assertIsNotNone(first)
        self.assertEqual(len(backend.adds), 1)
        # repeated sync with unchanged state makes no backend calls
        second = sync_label(backend, obj, True, "small",
                            color=(1.0, 0.0, 0.0))
        self.assertIs(second, first)
        self.assertEqual(len(backend.adds), 1)
        self.assertEqual(backend.updates, [])

    def test_scale_color_text_mutation(self):
        backend = self._Backend()
        obj = self._obj()
        handle = sync_label(backend, obj, True, "small",
                            color=(1.0, 0.0, 0.0))
        sync_label(backend, obj, True, "large", color=(1.0, 0.0, 0.0))
        self.assertAlmostEqual(handle.scale, 1.4)
        sync_label(backend, obj, True, "large", color=(0.0, 1.0, 0.0))
        self.assertEqual(handle.color, (0.0, 1.0, 0.0))
        # query relabel mutates text on the same handle
        obj["query_label"] = "armchair"
        sync_label(backend, obj, True, "large", color=(0.0, 1.0, 0.0))
        self.assertEqual(handle.text, "armchair")
        self.assertEqual(len(backend.adds), 1)
        self.assertEqual(len(backend.updates), 3)

    def test_hidden_or_off_removes_handle(self):
        backend = self._Backend()
        obj = self._obj()
        handle = sync_label(backend, obj, True, "small")
        sync_label(backend, obj, False, "small")
        self.assertIsNone(obj["label_handle"])
        self.assertEqual(backend.removes, [handle])
        # off setting removes as well; double removal is a no-op
        handle2 = sync_label(backend, obj, True, "small")
        sync_label(backend, obj, True, "off")
        self.assertIsNone(obj["label_handle"])
        self.assertEqual(backend.removes, [handle, handle2])
        remove_label_handle(backend, obj)
        self.assertEqual(backend.removes, [handle, handle2])

    def test_per_instance_handles(self):
        backend = self._Backend()
        objs = [self._obj() for _ in range(3)]
        handles = [sync_label(backend, obj, True, "small") for obj in objs]
        self.assertEqual(len({id(h) for h in handles}), 3)
        self.assertEqual(len(backend.adds), 3)

    def test_tick_performs_no_label_camera_work(self):
        import inspect
        import utils.visualize as viz
        source = inspect.getsource(viz)
        start = source.index("def _tick()")
        end = source.index("def _shutdown_cleanup")
        tick_source = source[start:end]
        for banned in ("sync_label", "add_label", "update_label",
                       "remove_label", "label_handle"):
            self.assertNotIn(banned, tick_source)


class SceneWidgetKeyTests(unittest.TestCase):
    def test_keyname_mapping_and_handler(self):
        from open3d.visualization import gui

        from utils.scene_widget import (
            KEY_TO_ACTION, make_key_handler, map_key_event)
        self.assertEqual(set(KEY_TO_ACTION),
                         {gui.KeyName.UP, gui.KeyName.DOWN, gui.KeyName.LEFT,
                          gui.KeyName.RIGHT, gui.KeyName.ENTER,
                          gui.KeyName.TAB, gui.KeyName.Q,
                          gui.KeyName.ESCAPE})

        class Event:
            def __init__(self, key, etype):
                self.key = key
                self.type = etype

        cases = [(gui.KeyName.UP, "up"), (gui.KeyName.DOWN, "down"),
                 (gui.KeyName.LEFT, "left"), (gui.KeyName.RIGHT, "right"),
                 (gui.KeyName.ENTER, "enter"), (gui.KeyName.TAB, "tab"),
                 (gui.KeyName.Q, "q"), (gui.KeyName.ESCAPE, "escape")]
        for key, action in cases:
            self.assertEqual(
                map_key_event(Event(key, gui.KeyEvent.Type.DOWN)), action)
            # key-up is ignored
            self.assertIsNone(
                map_key_event(Event(key, gui.KeyEvent.Type.UP)))
        # unhandled keys map to None and are not consumed
        self.assertIsNone(
            map_key_event(Event(gui.KeyName.F1, gui.KeyEvent.Type.DOWN)))
        seen = []
        handler = make_key_handler(seen.append)
        self.assertTrue(handler(Event(gui.KeyName.ENTER,
                                      gui.KeyEvent.Type.DOWN)))
        self.assertEqual(seen, ["enter"])
        self.assertFalse(handler(Event(gui.KeyName.ENTER,
                                       gui.KeyEvent.Type.UP)))
        self.assertFalse(handler(Event(gui.KeyName.F1,
                                       gui.KeyEvent.Type.DOWN)))


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

    def test_information_payload_reports_explicit_counts(self):
        pred = {"label_set": "ScanNet20", "run_id": "run-a", "model": "mosaic3d"}
        score = {"tp": 10, "gt": 22, "verdicts": {"a": "fp"},
                 "ap": 0.271, "ap_class_count": 5}
        run_key = run_node_id("scene0568_00", pred)
        info = information_payload(
            "scene0568_00", pred, {run_key: score}, {},
            visible_pred=42, eligible_gt=22)
        self.assertEqual((info["visible_pred"], info["eligible_gt"]), (42, 22))
        self.assertEqual(info["scene_ap"], 0.271)
        self.assertNotIn("masks_pred", info)
        self.assertNotIn("ap50", info)

    def test_label_scale_map_is_screen_space(self):
        self.assertIsNone(LABEL_SCALE["off"])
        self.assertIsNone(label_scale("off"))
        self.assertEqual(label_scale("small"), 1.0)
        self.assertEqual(label_scale("large"), 1.4)
        self.assertIsNone(label_scale("bogus"))
        for value in LABEL_SCALE.values():
            if value is not None:
                self.assertLess(value, 2.0)

    def test_label_text_prefers_query(self):
        self.assertEqual(
            label_text({"class_name": "chair", "query_label": None}), "chair")
        self.assertEqual(
            label_text({"class_name": "chair", "query_label": "armchair"}),
            "armchair")

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

        def slider_float(self, *_a, **_k):
            return False, None

        def separator(self, *_a, **_k):
            return None

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    def test_ap_lines_and_explicit_counts(self):
        imgui = self._Imgui()
        _draw_right(imgui, {"info": {
            "scene": "scene0618_00",
            "method": "openyolo3d",
            "run": "vis-replay-openyolo3d",
            "scene_ap": 0.1,
            "scene_ap_classes": 5,
            "run_ap": 0.37,
            "run_ap_classes": 53,
            "run_scenes": 10,
            "visible_pred": 42,
            "eligible_gt": 22,
        }}, 1080, 680)
        self.assertIn("Scene AP 0.100 (5 evaluated classes)", imgui.texts)
        self.assertIn("Run AP 0.370 (10 scenes, 53 evaluated classes)", imgui.texts)
        self.assertIn("Visible predictions 42 | Eligible GT 22", imgui.texts)

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
        self.assertIn("Classes   P prediction   G eligible GT   [TP green / errors red]", imgui.texts)
        self.assertNotIn("Pipeline: openyolo3d", imgui.texts)
        self.assertEqual(imgui.images, [])

    def test_preview_panel_shows_video_only(self):
        imgui = self._Imgui()
        _draw_right(imgui, self._payload("preview"), 1080, 680,
                    overlay=self._video())
        self.assertEqual(len(imgui.images), 1)
        self.assertNotIn("Classes   P prediction   G eligible GT   [TP green / errors red]", imgui.texts)
        self.assertNotIn("Pipeline: openyolo3d", imgui.texts)

    def test_pipeline_panel_shows_explainer_only(self):
        imgui = self._Imgui()
        _draw_right(imgui, self._payload("pipeline"), 1080, 680,
                    overlay=self._video())
        self.assertIn("Pipeline: openyolo3d", imgui.texts)
        self.assertNotIn("Classes   P prediction   G eligible GT   [TP green / errors red]", imgui.texts)
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


class ThresholdFilterTests(unittest.TestCase):
    def test_union_includes_pred_without_gt(self):
        self.assertEqual(
            union_prediction_classes({"chair": 1}, {"chair": 2, "lamp": 3}),
            ["chair", "lamp"])

    def test_off_shows_all_on_filters(self):
        fitted = {"chair": 0.5}
        manual = {}
        gt_counts = {"chair": 2}
        low = {"class_name": "chair", "score": 0.4}
        high = {"class_name": "chair", "score": 0.6}
        self.assertTrue(filter_survives(low, fitted, manual, FILTER_OFF, gt_counts))
        self.assertFalse(filter_survives(low, fitted, manual, FILTER_ON, gt_counts))
        self.assertTrue(filter_survives(high, fitted, manual, FILTER_ON, gt_counts))

    def test_filter_hides_class_without_scene_gt(self):
        fitted = {"lamp": 0.0}
        obj = {"class_name": "lamp", "score": 0.99}
        self.assertFalse(filter_survives(obj, fitted, {}, FILTER_ON, {"chair": 1}))
        self.assertTrue(filter_survives(obj, fitted, {}, FILTER_ON, {"lamp": 1}))
        self.assertTrue(filter_survives(obj, fitted, {}, FILTER_OFF, {"chair": 1}))

    def test_filter_hides_unfitted_class(self):
        obj = {"class_name": "lamp", "score": 0.99}
        self.assertFalse(filter_survives(obj, {}, {}, FILTER_ON, {"lamp": 1}))

    def test_manual_overrides_fitted(self):
        fitted = {"chair": 0.5}
        manual = {"chair": 0.9}
        self.assertEqual(effective_threshold(fitted, manual, "chair"), 0.9)
        self.assertIsNone(effective_threshold(fitted, {}, "lamp"))

    def test_build_rows_marks_unfitted(self):
        rows = build_threshold_rows(["chair", "lamp"], {"chair": 0.73}, {},
                                    {"chair": 2, "lamp": 3}, {"chair": 1})
        by_name = {row["name"]: row for row in rows}
        self.assertTrue(by_name["chair"]["is_fitted"])
        self.assertFalse(by_name["lamp"]["is_fitted"])
        self.assertEqual(by_name["lamp"]["effective"], 0.0)
        self.assertEqual((by_name["lamp"]["pred"], by_name["lamp"]["gt"]), (3, 0))

    def test_keep_indices_single_source_for_backend_and_hud(self):
        objs = [
            {"key": "a", "class_name": "chair", "score": 0.9, "verdict": "tp"},
            {"key": "b", "class_name": "chair", "score": 0.1, "verdict": "fp"},
            {"key": "c", "class_name": "lamp", "score": 0.9, "verdict": "tp"},
        ]
        fitted = {"chair": 0.5}
        gt_counts = {"chair": 1}
        keep = derive_keep_indices(objs, None, COLOR_CLASS, True, fitted, {},
                                   gt_counts)
        self.assertEqual(keep, [0])
        # Rank filtering composes; empty ranks select nothing (empty query).
        self.assertEqual(
            derive_keep_indices(objs, set(), COLOR_CLASS, False, {}, {}, gt_counts),
            [])
        self.assertEqual(
            derive_keep_indices(objs, {"a", "b"}, COLOR_CLASS, False, {}, {},
                                gt_counts),
            [0, 1])
        # TP/FP verdict gate drops ignored and verdict-less objects.
        objs2 = [dict(o, verdict=v) for o, v in
                 zip(objs, ("tp", "ignored", None))]
        self.assertEqual(
            derive_keep_indices(objs2, None, COLOR_TPFN, False, {}, {}, gt_counts),
            [0])
        # Ceiling predicate composes last.
        self.assertEqual(
            derive_keep_indices(objs, None, COLOR_CLASS, False, {}, {}, gt_counts,
                                hidden=lambda o: o["key"] == "a"),
            [1, 2])

    def test_build_rows_marks_unfitted(self):
        rows = build_threshold_rows(["chair", "lamp"], {"chair": 0.73}, {},
                                    {"chair": 2, "lamp": 3}, {"chair": 1})
        by_name = {row["name"]: row for row in rows}
        self.assertTrue(by_name["chair"]["is_fitted"])
        self.assertFalse(by_name["lamp"]["is_fitted"])
        self.assertEqual(by_name["lamp"]["effective"], 0.0)
        self.assertEqual((by_name["lamp"]["pred"], by_name["lamp"]["gt"]), (3, 0))

    def test_ap_text_and_navigator(self):
        self.assertEqual(scene_ap_text({"ap": 0.271, "ap_class_count": 5}),
                         "AP 0.271 (5 classes)")
        self.assertEqual(scene_ap_text({}), "AP unavailable")
        self.assertEqual(scene_ap_text(None), "AP unavailable")

    def test_filter_delta_rows_no_sliders(self):
        from utils.hud import _draw_right

        class FakeImgui:
            WINDOW_NO_TITLE_BAR = 1
            WINDOW_NO_RESIZE = 2
            WINDOW_NO_MOVE = 4
            WINDOW_NO_COLLAPSE = 8
            WINDOW_NO_SAVED_SETTINGS = 16
            WINDOW_NO_INPUTS = 32

            def __init__(self):
                self.texts = []
                self.sliders = []
                self.buttons = []

            def text(self, value):
                self.texts.append(value)

            def slider_float(self, name, value, _lo, _hi):
                self.sliders.append((name, value))
                return False, None

            def button(self, label):
                self.buttons.append(label)
                return False

            def columns(self, *_a, **_k):
                return None

            def next_column(self, *_a, **_k):
                return None

            def get_cursor_screen_pos(self):
                return (0.0, 0.0)

            def get_window_draw_list(self):
                class _Draw:
                    def add_rect_filled(self, *_a, **_k):
                        return None

                    def add_line(self, *_a, **_k):
                        return None

                    def add_text(self, *args):
                        return None
                return _Draw()

            def get_color_u32_rgba(self, *_a, **_k):
                return 0

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        class Keys:
            def __init__(self):
                self.sent = []

            def send(self, payload):
                self.sent.append(payload)

        imgui = FakeImgui()
        keys = Keys()
        _draw_right(imgui, {
            "info": {"scene": "scene0618_00", "method": "m", "run": "r",
                     "scene_ap": 0.2, "scene_ap_classes": 2,
                     "run_ap": 0.3, "run_ap_classes": 5, "run_scenes": 2,
                     "visible_pred": 2, "eligible_gt": 1},
            "classes": ["chair", "lamp"],
            "colors": {"chair": (1.0, 0.0, 0.0), "lamp": (0.0, 1.0, 0.0)},
            "class_stats": {"chair": {"pred": 1, "gt": 1},
                            "lamp": {"pred": 1, "gt": 0}},
            "filter": {"mode": "on", "active": True, "paused_by_query": False,
                       "reason": "", "selected": "chair", "rows": [
                {"name": "chair", "fitted": 0.73, "manual": None,
                 "effective": 0.73, "is_fitted": True, "is_manual": False,
                 "pred": 1, "gt": 1},
                {"name": "lamp", "fitted": None, "manual": None,
                 "effective": 0.0, "is_fitted": False, "is_manual": False,
                 "pred": 1, "gt": 0},
            ]},
        }, 1080, 680, keys_out=keys)
        self.assertIn("Filter: On", imgui.texts)
        self.assertIn("Classes   P prediction   G eligible GT   [TP green / errors red]",
                      imgui.texts)
        self.assertIn("Up/Down select, Left/Right adjust, Enter reset", imgui.texts)
        self.assertFalse(any(name.startswith("##thr_") for name, _v in imgui.sliders))
        self.assertFalse(any("Reset" in label for label in imgui.buttons))
        for banned in ("AP50", "AP25", "ignored", "Masks", "22/29", "[Pred/GT]"):
            self.assertFalse(any(banned in text for text in imgui.texts))


class FilterKeyboardTests(unittest.TestCase):
    def _ctx(self):
        return {"classes": ["chair", "table"], "fitted": {"chair": 0.5, "table": 0.2},
                "manual": {}, "editable": True}

    def _options(self):
        return {"filter": [{"value": "off"}, {"value": "on"}]}

    def test_down_past_last_row_selects_first_class(self):
        state = _nav_state(focus=FOCUS_SETTINGS,
                           setting_index=SETTING_ROWS.index("filter"),
                           filter_selected=None)
        action = handle_settings(state, KEY_DOWN, self._options(), self._ctx())
        self.assertEqual(state["filter_selected"], "chair")
        self.assertIsNone(action["apply"])

    def test_up_down_moves_between_classes_up_returns_to_settings(self):
        state = _nav_state(focus=FOCUS_SETTINGS,
                           setting_index=SETTING_ROWS.index("filter"),
                           filter_selected="chair")
        handle_settings(state, KEY_DOWN, self._options(), self._ctx())
        self.assertEqual(state["filter_selected"], "table")
        handle_settings(state, KEY_UP, self._options(), self._ctx())
        self.assertEqual(state["filter_selected"], "chair")
        handle_settings(state, KEY_UP, self._options(), self._ctx())
        self.assertIsNone(state["filter_selected"])

    def test_left_right_adjusts_and_enter_resets(self):
        state = _nav_state(focus=FOCUS_SETTINGS,
                           setting_index=SETTING_ROWS.index("filter"),
                           filter_selected="chair")
        action = handle_settings(state, KEY_RIGHT, self._options(), self._ctx())
        self.assertEqual(action["filter_adjust"], ("chair", 0.51))
        action = handle_settings(state, KEY_LEFT, self._options(), self._ctx())
        self.assertEqual(action["filter_adjust"], ("chair", 0.49))
        ctx = self._ctx()
        ctx["manual"] = {"chair": 0.9}
        action = handle_settings(state, KEY_ENTER, self._options(), ctx)
        self.assertEqual(action["filter_reset"], "chair")

    def test_selection_survives_reorder_by_identity(self):
        state = _nav_state(focus=FOCUS_SETTINGS,
                           setting_index=SETTING_ROWS.index("filter"),
                           filter_selected="chair")
        ctx = self._ctx()
        ctx["classes"] = ["table", "chair"]
        handle_settings(state, KEY_DOWN, self._options(), ctx)
        self.assertEqual(state["filter_selected"], "chair")

    def test_no_ctx_leaves_settings_navigation_unchanged(self):
        state = _nav_state(focus=FOCUS_SETTINGS,
                           setting_index=SETTING_ROWS.index("filter"),
                           filter_selected=None)
        action = handle_settings(state, KEY_DOWN, self._options())
        self.assertIsNone(state["filter_selected"])
        self.assertIsNone(action["apply"])


class QueryScoreNormTests(unittest.TestCase):
    def test_flat_cosine_span_maps_to_full_gradient(self):
        scores = {f"m{i}": -0.056 - 0.0007 * i for i in range(10)}
        norm = normalize_query_scores(scores)
        self.assertAlmostEqual(norm["m0"], 1.0)
        self.assertAlmostEqual(norm["m9"], 0.0)
        self.assertGreater(norm["m0"] - norm["m9"], 0.9)

    def test_constant_scores_read_hot(self):
        norm = normalize_query_scores({"a": 0.5, "b": 0.5})
        self.assertEqual(norm, {"a": 1.0, "b": 1.0})

    def test_empty_scores(self):
        self.assertEqual(normalize_query_scores({}), {})


class QueryResultTests(unittest.TestCase):
    def test_clip_prompt_labels_ranked_masks(self):
        objs = [{"key": "a", "query_label": None}, {"key": "b", "query_label": None}]
        kept, status = apply_query_result(objs, ["a", "b"], {}, "chair", True)
        self.assertEqual(kept, ["a", "b"])
        self.assertEqual(status, "")
        self.assertEqual([o["query_label"] for o in objs], ["chair", "chair"])

    def test_native_labels_preferred_over_prompt(self):
        objs = [{"key": "a", "query_label": None}]
        kept, _ = apply_query_result(objs, ["a"], {"a": "seat"}, "chair", True)
        self.assertEqual(objs[0]["query_label"], "seat")

    def test_empty_ranks_select_nothing_with_status(self):
        objs = [{"key": "a", "query_label": "old"}]
        kept, status = apply_query_result(objs, [], {}, "zzz", True)
        self.assertEqual(kept, [])
        self.assertEqual(status, "No matches for 'zzz'")
        self.assertIsNone(objs[0]["query_label"])

    def test_stale_keys_dropped_with_status(self):
        objs = [{"key": "a", "query_label": None}]
        kept, status = apply_query_result(objs, ["a", "ghost"], {}, "chair", True)
        self.assertEqual(kept, ["a"])
        self.assertEqual(status, "1 stale result(s) ignored")

    def test_inactive_query_clears_labels(self):
        objs = [{"key": "a", "query_label": "chair"}]
        kept, status = apply_query_result(objs, ["a"], {}, "chair", False)
        self.assertIsNone(objs[0]["query_label"])
        self.assertEqual(status, "")

    def test_hud_tpfn_legend(self):
        from utils.hud import _draw_right

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

            def get_cursor_screen_pos(self):
                return (0.0, 0.0)

            def get_window_draw_list(self):
                class _Draw:
                    def add_rect_filled(self, *_a, **_k):
                        return None

                    def add_line(self, *_a, **_k):
                        return None

                    def add_text(self, *args):
                        return None
                return _Draw()

            def get_color_u32_rgba(self, *_a, **_k):
                return 0

            def dummy(self, *_a, **_k):
                return None

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        imgui = FakeImgui()
        _draw_right(imgui, {
            "info": {"scene": "scene0618_00"},
            "classes": ["chair"],
            "colors": {"chair": (1.0, 0.0, 0.0)},
            "class_stats": {"chair": {"pred": 1, "gt": 1}},
            "color_mode": "tp_fn",
            "filter": {"mode": "off", "rows": []},
        }, 1080, 680)
        self.assertIn("TP green | FP dark red | FN light red", imgui.texts)

    def test_hud_query_row_replaces_delta_rows(self):
        from utils.hud import _draw_right

        class FakeImgui:
            WINDOW_NO_TITLE_BAR = 1
            WINDOW_NO_RESIZE = 2
            WINDOW_NO_MOVE = 4
            WINDOW_NO_COLLAPSE = 8
            WINDOW_NO_SAVED_SETTINGS = 16
            WINDOW_NO_INPUTS = 32

            def __init__(self):
                self.texts = []
                self.ended = False

            def text(self, value):
                self.texts.append(value)

            def end(self):
                self.ended = True

            def __getattr__(self, _name):
                return lambda *args, **kwargs: None

        imgui = FakeImgui()
        _draw_right(imgui, {
            "info": {"scene": "scene0618_00"},
            "classes": ["chair"],
            "colors": {"chair": (1.0, 0.0, 0.0)},
            "class_stats": {"chair": {"pred": 2, "gt": 1}},
            "filter": {"mode": "on", "active": False, "paused_by_query": True,
                       "reason": "", "rows": []},
            "query": {"mode": "search", "status": "", "input": "chair",
                      "count": 2, "results": []},
        }, 1080, 680)
        self.assertTrue(imgui.ended)
        self.assertIn("Query 'chair'  P 2  G -", imgui.texts)
        self.assertFalse(any("+/- difference" in t for t in imgui.texts))


if __name__ == "__main__":
    unittest.main()
