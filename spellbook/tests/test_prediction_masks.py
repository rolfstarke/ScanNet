import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_SPELLBOOK)
sys.path.insert(0, _SPELLBOOK)
sys.path.insert(0, os.path.join(_REPO, "BenchmarkScripts"))

import util_3d  # noqa: E402
from evaluation.benchmark import BENCHMARKS  # noqa: E402
from utils.prediction_masks import (  # noqa: E402
    SCHEMA, build_cli, build_packed_masks, cache_expected, load_or_build_packed_masks,
    mask_cache_path, merge_packed_overlay, pack_mask, parse_mask_text, prediction_rows,
    try_load_cache, unpack_mask, write_cache_atomic)
from utils.visualize import (  # noqa: E402
    FP_COLOR, TP_COLOR, load_predictions, merge_tp_gt_overlay, prediction_artifact_mtime)


def _spec():
    return BENCHMARKS["ScanNet20"]


def _write_mask(path, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("".join(f"{int(v)}\n" for v in values))


def _layout(root, scene="scene0568_00", masks=None, label=None, confs=None):
    spec = _spec()
    label = spec.valid_ids[0] if label is None else label
    pred = os.path.join(root, "predictions", spec.name, "run-a", "mosaic3d")
    os.makedirs(os.path.join(pred, "predicted_masks"), exist_ok=True)
    if masks is None:
        masks = [np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1], dtype=np.uint8)]
    confs = [0.9] * len(masks) if confs is None else confs
    lines = []
    for index, mask in enumerate(masks):
        rel = f"predicted_masks/{scene}_{index:03d}.txt"
        _write_mask(os.path.join(pred, rel), mask)
        lines.append(f"{rel} {label} {confs[index]}\n")
    with open(os.path.join(pred, scene + ".txt"), "w") as fh:
        fh.writelines(lines)
    return spec, pred, scene


class ParsePackTests(unittest.TestCase):
    def test_round_trip_shapes(self):
        cases = {
            "empty": np.zeros(10, dtype=np.uint8),
            "full": np.ones(10, dtype=np.uint8),
            "sparse": np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.uint8),
            "unaligned": np.array([1, 0, 1, 1, 0], dtype=np.uint8),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, mask in cases.items():
                path = os.path.join(tmp, name + ".txt")
                _write_mask(path, mask)
                parsed = parse_mask_text(path, len(mask))
                np.testing.assert_array_equal(parsed, mask)
                official = util_3d.load_ids(path)
                np.testing.assert_array_equal(parsed, official.astype(np.uint8))
                packed = pack_mask(parsed)
                np.testing.assert_array_equal(unpack_mask(packed, len(mask)), mask.astype(bool))

    def test_rejects_length_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.txt")
            _write_mask(path, [0, 1, 0])
            with self.assertRaises(ValueError):
                parse_mask_text(path, 4)
            _write_mask(path, [0, 2, 0])
            with self.assertRaises(ValueError):
                parse_mask_text(path, 3)


class CacheTests(unittest.TestCase):
    def test_order_ids_and_hit_skips_parser(self):
        spec = _spec()
        masks = [
            np.array([1, 0, 1, 0, 0], dtype=np.uint8),
            np.array([0, 1, 1, 1, 0], dtype=np.uint8),
        ]
        with tempfile.TemporaryDirectory() as root:
            spec, pred, scene = _layout(root, masks=masks, confs=[0.4, 0.8])
            mtime = prediction_artifact_mtime(pred, scene)
            payload, path, built = load_or_build_packed_masks(
                pred, scene, 5, spec, "run-a", "mosaic3d", mtime, scannet_root=root)
            self.assertTrue(built)
            self.assertEqual(payload["keys"], [
                "predicted_masks/scene0568_00_000.txt",
                "predicted_masks/scene0568_00_001.txt",
            ])
            np.testing.assert_array_equal(payload["label_ids"], [spec.valid_ids[0]] * 2)
            np.testing.assert_allclose(payload["confs"], [0.4, 0.8])
            rows = prediction_rows(pred, scene, spec=spec)
            self.assertEqual([row["key"] for row in rows], payload["keys"])
            with mock.patch("utils.prediction_masks.parse_mask_text") as parser:
                loaded, _, built_again = load_or_build_packed_masks(
                    pred, scene, 5, spec, "run-a", "mosaic3d", mtime, scannet_root=root)
                parser.assert_not_called()
            self.assertFalse(built_again)
            self.assertEqual(loaded["keys"], payload["keys"])
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(
                path, mask_cache_path(spec, "run-a", "mosaic3d", scene, scannet_root=root))

    def test_stale_wrong_and_malformed_fall_back(self):
        with tempfile.TemporaryDirectory() as root:
            spec, pred, scene = _layout(root)
            mtime = prediction_artifact_mtime(pred, scene)
            expected = cache_expected(spec, "run-a", "mosaic3d", scene, 10, mtime)
            payload = build_packed_masks(pred, scene, 10, spec=spec)
            path = mask_cache_path(spec, "run-a", "mosaic3d", scene, scannet_root=root)
            write_cache_atomic(path, payload, expected)
            self.assertIsNone(try_load_cache(path, cache_expected(
                spec, "run-a", "mosaic3d", scene, 10, mtime + 10)))
            self.assertIsNone(try_load_cache(path, cache_expected(
                spec, "run-b", "mosaic3d", scene, 10, mtime)))
            self.assertIsNone(try_load_cache(path, cache_expected(
                spec, "run-a", "mosaic3d", scene, 11, mtime)))
            with open(path, "wb") as fh:
                fh.write(b"not-an-npz")
            self.assertIsNone(try_load_cache(path, expected))

    def test_interrupted_write_leaves_no_cache(self):
        with tempfile.TemporaryDirectory() as root:
            spec, pred, scene = _layout(root)
            mtime = prediction_artifact_mtime(pred, scene)
            expected = cache_expected(spec, "run-a", "mosaic3d", scene, 10, mtime)
            payload = build_packed_masks(pred, scene, 10, spec=spec)
            path = mask_cache_path(spec, "run-a", "mosaic3d", scene, scannet_root=root)

            def boom(*_args, **_kwargs):
                raise OSError("disk")

            with mock.patch("numpy.savez", side_effect=boom):
                with self.assertRaises(OSError):
                    write_cache_atomic(path, payload, expected)
            self.assertFalse(os.path.isfile(path))
            leftover = [
                name for name in os.listdir(os.path.dirname(path))
                if name.endswith(".npz") or name.endswith(".tmp")
            ] if os.path.isdir(os.path.dirname(path)) else []
            self.assertEqual(leftover, [])

    def test_unknown_label_skipped(self):
        spec = _spec()
        with tempfile.TemporaryDirectory() as root:
            pred = os.path.join(root, "predictions", spec.name, "run-a", "mosaic3d")
            os.makedirs(os.path.join(pred, "predicted_masks"), exist_ok=True)
            rel = "predicted_masks/scene0568_00_000.txt"
            _write_mask(os.path.join(pred, rel), [1, 0, 1, 0])
            with open(os.path.join(pred, "scene0568_00.txt"), "w") as fh:
                fh.write(f"{rel} 1 0.9\n")
            rows = prediction_rows(pred, "scene0568_00", spec=spec)
            self.assertEqual(rows, [])


class OverlayAndLoadTests(unittest.TestCase):
    def test_packed_matches_reference_and_tp_wins(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(32, 3))
        packed_objs = []
        ref_objs = []
        for verdict, sel in (
                ("fp", rng.random(32) > 0.6),
                ("tp", rng.random(32) > 0.6),
                ("ignored", rng.random(32) > 0.5),
                ("fp", rng.random(32) > 0.7),
        ):
            mask = sel.astype(np.uint8)
            obj = {
                "verdict": verdict,
                "sel": sel,
                "packed": pack_mask(mask),
            }
            packed_objs.append(obj)
            ref_objs.append({"verdict": verdict, "sel": sel})
        packed_pts, packed_cols = merge_packed_overlay(
            points, packed_objs, FP_COLOR, TP_COLOR)
        ref_pts, ref_cols = merge_tp_gt_overlay(points, [
            {"verdict": obj["verdict"], "sel": obj["sel"]} for obj in ref_objs])
        np.testing.assert_array_equal(packed_pts, ref_pts)
        np.testing.assert_array_equal(packed_cols, ref_cols)
        overlap = packed_objs[0]["sel"] & packed_objs[1]["sel"]
        if overlap.any():
            keep = packed_objs[0]["sel"] | packed_objs[1]["sel"] | packed_objs[3]["sel"]
            kept_overlap = overlap[keep]
            self.assertTrue(kept_overlap.any())
            np.testing.assert_array_equal(
                packed_cols[kept_overlap], np.broadcast_to(TP_COLOR, (int(kept_overlap.sum()), 3)))

    def test_load_predictions_cache_matches_text(self):
        spec = _spec()
        masks = [
            np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8),
            np.array([0, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8),
        ]
        points = np.arange(24, dtype=float).reshape(8, 3)
        colors = np.ones((8, 3), dtype=float)
        with tempfile.TemporaryDirectory() as root:
            spec, pred, scene = _layout(root, masks=masks, confs=[0.2, 0.7])
            text = load_predictions(pred, scene, points, colors, spec)
            mtime = prediction_artifact_mtime(pred, scene)
            cached = load_predictions(
                pred, scene, points, colors, spec,
                run_id="run-a", model="mosaic3d", scannet_root=root, source_mtime=mtime)
            self.assertEqual([o["key"] for o in text["objects"]],
                             [o["key"] for o in cached["objects"]])
            for left, right in zip(text["objects"], cached["objects"]):
                np.testing.assert_array_equal(left["sel"], right["sel"])
                np.testing.assert_array_equal(left["points"], right["points"])
                self.assertEqual(left["class_name"], right["class_name"])
                self.assertEqual(left["score"], right["score"])
                self.assertIsNotNone(right["packed"])
            dispatch = merge_tp_gt_overlay(points, [
                {**cached["objects"][0], "verdict": "fp"},
                {**cached["objects"][1], "verdict": "tp"},
            ])
            reference = merge_tp_gt_overlay(points, [
                {"verdict": "fp", "sel": text["objects"][0]["sel"]},
                {"verdict": "tp", "sel": text["objects"][1]["sel"]},
            ])
            np.testing.assert_array_equal(dispatch[0], reference[0])
            np.testing.assert_array_equal(dispatch[1], reference[1])


class DuplicateSubmissionTests(unittest.TestCase):
    def _dup_layout(self, root):
        spec = _spec()
        lab_a, lab_b = spec.valid_ids[0], spec.valid_ids[1]
        scene = "scene0568_00"
        pred = os.path.join(root, "predictions", spec.name, "run-a", "mosaic3d")
        os.makedirs(os.path.join(pred, "predicted_masks"), exist_ok=True)
        same = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        other = np.array([0, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
        rows = [(same, lab_a, 0.2), (same, lab_b, 1.0), (other, lab_a, 0.5)]
        lines = []
        for index, (mask, label, conf) in enumerate(rows):
            rel = f"predicted_masks/{scene}_{index:03d}.txt"
            _write_mask(os.path.join(pred, rel), mask)
            lines.append(f"{rel} {label} {conf}\n")
        with open(os.path.join(pred, scene + ".txt"), "w") as fh:
            fh.writelines(lines)
        return spec, pred, scene, lab_b

    def test_text_path_keeps_all_submitted_rows(self):
        # Same mask submitted under two labels: the official submission keeps
        # both rows, so load_predictions must keep all three objects.
        points = np.arange(24, dtype=float).reshape(8, 3)
        colors = np.ones((8, 3), dtype=float)
        with tempfile.TemporaryDirectory() as root:
            spec, pred, scene, lab_b = self._dup_layout(root)
            data = load_predictions(pred, scene, points, colors, spec)
            self.assertEqual(len(data["objects"]), 3)
            self.assertEqual(len(data["pred_instances"]), 3)
            by_key = {o["key"]: o for o in data["objects"]}
            self.assertEqual(by_key["predicted_masks/scene0568_00_001.txt"]["score"], 1.0)
            self.assertEqual(
                by_key["predicted_masks/scene0568_00_001.txt"]["class_name"],
                spec.id_to_label[lab_b])

    def test_packed_path_matches_text(self):
        points = np.arange(24, dtype=float).reshape(8, 3)
        colors = np.ones((8, 3), dtype=float)
        with tempfile.TemporaryDirectory() as root:
            spec, pred, scene, _ = self._dup_layout(root)
            text = load_predictions(pred, scene, points, colors, spec)
            mtime = prediction_artifact_mtime(pred, scene)
            cached = load_predictions(
                pred, scene, points, colors, spec,
                run_id="run-a", model="mosaic3d", scannet_root=root,
                source_mtime=mtime)
            self.assertEqual(
                sorted(o["key"] for o in cached["objects"]),
                sorted(o["key"] for o in text["objects"]))


class CliTests(unittest.TestCase):
    def test_build_missing_only(self):
        with tempfile.TemporaryDirectory() as root:
            spec, pred, scene = _layout(root)
            build_cli([
                "--benchmark", spec.name, "--run-id", "run-a", "--models", "mosaic3d",
                "--scenes", "0568_00", "--scannet-root", root,
            ])
            path = mask_cache_path(spec, "run-a", "mosaic3d", scene, scannet_root=root)
            self.assertTrue(os.path.isfile(path))
            stamp = os.path.getmtime(path)
            build_cli([
                "--benchmark", spec.name, "--run-id", "run-a", "--models", "mosaic3d",
                "--scenes", "0568_00", "--scannet-root", root, "--missing-only",
            ])
            self.assertEqual(os.path.getmtime(path), stamp)


if __name__ == "__main__":
    unittest.main()
