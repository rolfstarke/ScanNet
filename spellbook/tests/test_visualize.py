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
from utils.hud import ellipsize  # noqa: E402
from utils.visualize import (  # noqa: E402
    FOCUS_SETTINGS, FOCUS_TREE, KEY_DOWN, KEY_ENTER, KEY_LEFT, KEY_RIGHT, KEY_TAB,
    KEY_UP, KIND_METHOD, KIND_RUN, KIND_SCAN, SOURCE_GT, apply_prediction_labels,
    best_prediction,     cad_report_png, clamp_cursor, clear_tracked, collect_tp_scores,
    discover_reconstructions, flatten_tree, group_scenes, handle_navigation,
    index_predictions, load_cad_accuracy, method_node_id, prediction_artifact_mtime,
    require_scene, run_node_id, scan_node_id, scene_node_id, settings_payload,
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
        "run_id": None,
        "model": SOURCE_GT,
        "color_mode": "class",
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
            "benchmark": ["ScanNet20", "ScanNet200"],
            "mode": ["scene", "class", "instance", "tp_fp"],
            "geometry": ["mesh", "pointcloud"],
            "boxes": [True, False],
            "ceiling": [False, True],
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
        self.assertEqual(self.state["model"], SOURCE_GT)

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
        self.assertEqual(self.state["candidate"], "ScanNet20")
        action = handle_navigation(self.state, KEY_ENTER, self.rows, self.options)
        self.assertEqual(action["apply"], ("benchmark", "ScanNet20"))
        self.state["focus"] = FOCUS_SETTINGS
        self.state["candidate"] = "ScanNet20"
        handle_navigation(self.state, KEY_DOWN, self.rows, self.options)
        self.assertIsNone(self.state["candidate"])
        self.state["candidate"] = "ScanNet200"
        handle_navigation(self.state, KEY_TAB, self.rows, self.options)
        self.assertIsNone(self.state["candidate"])
        self.assertEqual(self.state["focus"], FOCUS_TREE)

    def test_settings_option_kinds(self):
        self.state["focus"] = FOCUS_SETTINGS
        self.state["candidate"] = "ScanNet20"
        rows = settings_payload(self.state, self.options, lambda _r, v: str(v))
        kinds = {o["text"]: o["kind"] for o in rows[0]["options"]}
        self.assertEqual(kinds["ScanNet200"], "applied")
        self.assertEqual(kinds["ScanNet20"], "pending")
        self.assertTrue(next(o["sel"] for o in rows[0]["options"] if o["text"] == "ScanNet20"))
        self.assertEqual(rows[0]["name"], "Label set")

    def test_cad_enter_toggles(self):
        options = dict(self.options)
        options["cad"] = [False, True]
        self.state["focus"] = FOCUS_SETTINGS
        self.state["setting_index"] = 5
        action = handle_navigation(self.state, KEY_ENTER, self.rows, options)
        self.assertEqual(action["apply"], ("cad", True))
        self.state["cad_open"] = True
        self.state["candidate"] = None
        action = handle_navigation(self.state, KEY_ENTER, self.rows, options)
        self.assertEqual(action["apply"], ("cad", False))

    def test_cad_row_absent_without_png(self):
        rows = settings_payload(self.state, self.options, lambda _r, v: str(v))
        self.assertFalse(any(r["key"] == "cad" for r in rows))


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
                "schema": 1, "metric": "scannet_instance_ap50", "iou_threshold": 0.5,
                "min_region_size": 100, "scene_id": "scene0568_00", "label_set": "ScanNet20",
                "run_id": "run-a", "model": "mosaic3d", "tp": 3, "gt": 57, "verdicts": {},
            })
            scores = collect_tp_scores(root, {"scene0568_00": [pred]})
            self.assertEqual(scores[run_node_id("scene0568_00", pred)]["tp"], 3)


class HudHelperTests(unittest.TestCase):
    def test_ellipsize(self):
        self.assertEqual(ellipsize("abcdefghij", 5, len), "ab...")
        self.assertEqual(ellipsize("ok", 10, len), "ok")


if __name__ == "__main__":
    unittest.main()
