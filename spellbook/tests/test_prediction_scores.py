import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from evaluation.benchmark import BENCHMARKS  # noqa: E402
from evaluation.evaluate import (  # noqa: E402
    load_score_sidecar, scene_submission_status, score_and_write_sidecar, score_sidecar_path,
    score_sidecars_cli, write_score_sidecar)
from evaluation.scannet200_evaluator import Evaluator, scene_instance_summary  # noqa: E402

sys.path.insert(0, _SPELLBOOK)


def _spec20():
    return BENCHMARKS["ScanNet20"]


def _spec200():
    return BENCHMARKS["ScanNet200"]


def _gt_ids(n, instance_id):
    ids = np.zeros(n, dtype=np.int32)
    ids[:n] = instance_id
    return ids


def _pred(mask, label_id, conf=0.9, key="mask"):
    return {key: {"label_id": label_id, "conf": conf, "pred_mask": np.asarray(mask, dtype=np.int32)}}


class Ap50SummaryTests(unittest.TestCase):
    def setUp(self):
        self.spec = _spec20()
        self.label = self.spec.valid_ids[0]
        self.gt_id = self.label * 1000 + 1

    def test_perfect_100_vertex_match(self):
        gt = _gt_ids(100, self.gt_id)
        pred = _pred(np.ones(100, dtype=np.int32), self.label)
        out = scene_instance_summary(gt, pred, self.spec, "scene0000_00")
        self.assertEqual(out["tp"], 1)
        self.assertEqual(out["gt"], 1)
        self.assertEqual(out["verdicts"]["mask"], "tp")

    def test_99_vertex_gt_excluded_100_kept(self):
        gt = np.zeros(200, dtype=np.int32)
        gt[:99] = self.gt_id
        pred = _pred((np.arange(200) < 100).astype(np.int32), self.label)
        out = scene_instance_summary(gt, pred, self.spec, "scene0000_00")
        self.assertEqual(out["gt"], 0)
        gt[99] = self.gt_id
        out = scene_instance_summary(gt, pred, self.spec, "scene0000_00")
        self.assertEqual(out["gt"], 1)
        self.assertEqual(out["tp"], 1)

    def test_99_vertex_prediction_absent(self):
        gt = _gt_ids(100, self.gt_id)
        mask = np.zeros(100, dtype=np.int32)
        mask[:99] = 1
        out = scene_instance_summary(gt, _pred(mask, self.label), self.spec, "scene0000_00")
        self.assertEqual(out["verdicts"], {})
        self.assertEqual(out["tp"], 0)
        self.assertEqual(out["gt"], 1)

    def test_iou_boundary(self):
        n = 300
        gt = np.zeros(n, dtype=np.int32)
        gt[:100] = self.gt_id
        pred_eq = np.zeros(n, dtype=np.int32)
        pred_eq[:200] = 1
        out = scene_instance_summary(gt, _pred(pred_eq, self.label), self.spec, "scene0000_00")
        self.assertEqual(out["verdicts"]["mask"], "fp")
        pred_over = np.zeros(n, dtype=np.int32)
        pred_over[:199] = 1
        out = scene_instance_summary(gt, _pred(pred_over, self.label), self.spec, "scene0000_00")
        self.assertEqual(out["verdicts"]["mask"], "tp")

    def test_invalid_class_excluded(self):
        gt = _gt_ids(100, 1 * 1000 + 1)
        pred = _pred(np.ones(100, dtype=np.int32), 1)
        out = scene_instance_summary(gt, pred, self.spec, "scene0000_00")
        self.assertEqual(out["gt"], 0)
        self.assertEqual(out["tp"], 0)
        self.assertEqual(out["verdicts"], {})

    def test_duplicate_one_tp_one_fp(self):
        gt = _gt_ids(100, self.gt_id)
        mask = np.ones(100, dtype=np.int32)
        pred = {
            "a": {"label_id": self.label, "conf": 0.9, "pred_mask": mask},
            "b": {"label_id": self.label, "conf": 0.8, "pred_mask": mask},
        }
        out = scene_instance_summary(gt, pred, self.spec, "scene0000_00")
        self.assertEqual(out["tp"], 1)
        self.assertEqual(sorted(out["verdicts"].values()), ["fp", "tp"])

    def test_void_ignored_matches_evaluate_matches(self):
        n = 200
        gt = np.zeros(n, dtype=np.int32)
        gt[:80] = 1 * 1000 + 1
        pred_mask = np.ones(n, dtype=np.int32)
        pred = _pred(pred_mask, self.label)
        summary = scene_instance_summary(gt, pred, self.spec, "scene0000_00")
        evaluator = Evaluator(self.spec.class_labels, self.spec.valid_ids)
        evaluator.add_gt(gt, "scene0000_00")
        evaluator.add_prediction(pred, "scene0000_00")
        gt2pred, pred2gt = evaluator.assign_instances_for_scan("scene0000_00")
        ap = evaluator.evaluate_matches({"scene0000_00": {"gt": gt2pred, "pred": pred2gt}})
        o50 = int(np.where(np.isclose(evaluator.overlaps, 0.5))[0][0])
        self.assertTrue(np.isnan(ap[0, 0, o50]) or ap[0, 0, o50] == 0.0)
        self.assertIn(summary["verdicts"]["mask"], ("ignored", "fp"))

    def test_scannet200_uses_own_ids(self):
        spec = _spec200()
        label = spec.valid_ids[0]
        gt = _gt_ids(100, label * 1000 + 1)
        pred = _pred(np.ones(100, dtype=np.int32), label)
        out = scene_instance_summary(gt, pred, spec, "scene0000_00")
        self.assertEqual(out["tp"], 1)
        self.assertEqual(out["gt"], 1)


class SidecarTests(unittest.TestCase):
    def test_atomic_write_and_validation(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            path = score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_00", scannet_root=root)
            payload = {
                "schema": 1,
                "metric": "scannet_instance_ap50",
                "iou_threshold": 0.5,
                "min_region_size": 100,
                "scene_id": "scene0568_00",
                "label_set": "ScanNet20",
                "run_id": "run-a",
                "model": "mosaic3d",
                "tp": 3,
                "gt": 57,
                "verdicts": {"predicted_masks/scene0568_00_000.txt": "tp"},
            }
            write_score_sidecar(path, payload)
            self.assertFalse(os.path.isfile(path + ".tmp"))
            loaded = load_score_sidecar(path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d")
            self.assertEqual(loaded["tp"], 3)
            self.assertEqual(loaded["gt"], 57)
            payload["tp"] = 4
            write_score_sidecar(path, payload)
            loaded = load_score_sidecar(path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d")
            self.assertEqual(loaded["tp"], 4)
            self.assertIsNone(load_score_sidecar(path, "scene0568_00", "ScanNet200", "run-a", "mosaic3d"))
            with open(path) as f:
                doc = json.load(f)
            doc["tp"] = 99
            with open(path, "w") as f:
                json.dump(doc, f)
            self.assertIsNone(load_score_sidecar(path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d"))

    def test_missing_gt_writes_nothing(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            pred_dir = os.path.join(root, "predictions", "ScanNet20", "run-a", "mosaic3d")
            os.makedirs(pred_dir)
            with open(os.path.join(pred_dir, "scene0568_00.txt"), "w") as f:
                f.write("predicted_masks/scene0568_00_000.txt 3 0.9\n")
            os.makedirs(os.path.join(pred_dir, "predicted_masks"))
            with open(os.path.join(pred_dir, "predicted_masks", "scene0568_00_000.txt"), "w") as f:
                f.write("1\n" * 4)
            out = score_and_write_sidecar(
                pred_dir, "scene0568_00", spec, "run-a", "mosaic3d", scannet_root=root)
            self.assertIsNone(out)
            self.assertFalse(os.path.isfile(
                score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_00", scannet_root=root)))

    def test_score_sidecars_missing_only_skips_existing(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            path = score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_00", scannet_root=root)
            write_score_sidecar(path, {
                "schema": 1, "metric": "scannet_instance_ap50", "iou_threshold": 0.5,
                "min_region_size": 100, "scene_id": "scene0568_00", "label_set": "ScanNet20",
                "run_id": "run-a", "model": "mosaic3d", "tp": 1, "gt": 1, "verdicts": {},
            })
            with mock.patch("evaluation.evaluate.score_and_write_sidecar") as scored:
                score_sidecars_cli([
                    "--benchmark", "ScanNet20", "--run-id", "run-a", "--models", "mosaic3d",
                    "--scenes", "0568_00", "--scannet-root", root, "--missing-only",
                ])
            scored.assert_not_called()


class RunnerFinalizeTests(unittest.TestCase):
    def test_success_appends_tasks_after_scoring(self):
        from predict import runner

        with tempfile.TemporaryDirectory() as tmp:
            tasks_log = os.path.join(tmp, "mosaic3d.tasks")
            with mock.patch("evaluation.evaluate.score_and_write_sidecar", return_value={"tp": 1}):
                runner._finalize_prediction(
                    "mosaic3d", "scene0568_01", tmp, "ScanNet20", "run-a", tasks_log)
            with open(tasks_log) as f:
                self.assertEqual(f.read(), "scene0568_01\n")

    def test_scoring_failure_does_not_append_tasks(self):
        from predict import runner

        with tempfile.TemporaryDirectory() as tmp:
            tasks_log = os.path.join(tmp, "mosaic3d.tasks")
            with mock.patch("evaluation.evaluate.score_and_write_sidecar",
                            side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    runner._finalize_prediction(
                        "mosaic3d", "scene0568_01", tmp, "ScanNet20", "run-a", tasks_log)
            self.assertFalse(os.path.isfile(tasks_log))

    def test_child_failure_skips_finalize(self):
        from predict import runner

        class Lease:
            def fileno(self):
                return 3

            index = 1

        with mock.patch("subprocess.run", side_effect=runner.subprocess.CalledProcessError(1, "x")):
            result = runner._run_one(
                "mosaic3d", "scene0568_01", None, ["chair"], 1, "/tmp", "ScanNet20",
                "/tmp/x.tasks", Lease(), "run-a", None, "/tmp")
        self.assertFalse(result[-1])
        self.assertIsNone(result[2])


class CleanupTests(unittest.TestCase):
    def test_purge_removes_sidecar(self):
        from reconstruct.cleanup import purge_scan_predictions

        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            pred = os.path.join(root, "predictions", "ScanNet20", "run-a", "mosaic3d")
            os.makedirs(os.path.join(pred, "predicted_masks"))
            with open(os.path.join(pred, "scene0568_00.txt"), "w") as f:
                f.write("predicted_masks/scene0568_00_000.txt 3 0.9\n")
            with open(os.path.join(pred, "predicted_masks", "scene0568_00_000.txt"), "w") as f:
                f.write("1\n")
            sidecar = score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_00", scannet_root=root)
            os.makedirs(os.path.dirname(sidecar))
            write_score_sidecar(sidecar, {
                "schema": 1, "metric": "scannet_instance_ap50", "iou_threshold": 0.5,
                "min_region_size": 100, "scene_id": "scene0568_00", "label_set": "ScanNet20",
                "run_id": "run-a", "model": "mosaic3d", "tp": 1, "gt": 1, "verdicts": {},
            })
            eval_dir = os.path.join(root, "derived", "evaluations", "ScanNet20", "run-a")
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "mosaic3d.tasks"), "w") as f:
                f.write("scene0568_00\n")
            with mock.patch("reconstruct.cleanup.load_settings", return_value={"scannet_root": root}):
                purge_scan_predictions("scene0568_00", scannet_root=root)
            self.assertFalse(os.path.isfile(sidecar))
            self.assertFalse(os.path.isfile(os.path.join(pred, "scene0568_00.txt")))


def _load_common():
    path = os.path.join(_SPELLBOOK, "predict", "models", "common.py")
    spec = importlib.util.spec_from_file_location("spellbook_predict_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_ply(path, n):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement vertex %d\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n" % n)
        f.write("0 0 0\n" * n)


class CommonContractTests(unittest.TestCase):
    def test_decimate_rejects_nonpositive(self):
        common = _load_common()
        with self.assertRaises(ValueError):
            common.decimate(np.zeros((4, 3)), 0)

    def test_load_overrides_rejects_point_limit(self):
        common = _load_common()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.json")
            with open(path, "w") as f:
                json.dump({"point_limit": 0}, f)
            with self.assertRaises(ValueError):
                common.load_overrides(path, {"point_limit"})

    def test_validate_run_id_rejects_paths(self):
        common = _load_common()
        with self.assertRaises(ValueError):
            common.validate_run_id("../x")
        with self.assertRaises(ValueError):
            common.validate_run_id("a/b")
        self.assertEqual(common.validate_run_id("adhoc"), "adhoc")


class SubmissionTests(unittest.TestCase):
    def test_rerun_does_not_mix_generations(self):
        sys.path.insert(0, os.path.join(os.path.dirname(_SPELLBOOK), "BenchmarkScripts"))
        import util_3d
        common = _load_common()
        spec = _spec20()
        label = spec.class_labels[0]
        with tempfile.TemporaryDirectory() as root:
            mask1 = np.zeros(4, dtype=bool)
            mask1[:2] = True
            common.write_scannet_submission(
                root, "scene0568_01", [label], [(mask1, label, 0.9)], 1, spec)
            idx = os.path.join(root, "scene0568_01.txt")
            with open(idx) as f:
                gen1_refs = {line.split()[0] for line in f if line.strip()}
            mask_dir = os.path.join(root, "predicted_masks")
            np.savetxt(os.path.join(mask_dir, "scene0568_01_deadbeef_000.txt"),
                       np.ones(4, dtype=int), fmt="%d")
            instances = util_3d.read_instance_prediction_file(idx, root)
            data = util_3d.load_ids(list(instances.keys())[0])
            self.assertEqual(int(data[:2].sum()), 2)
            self.assertEqual(int(data[2:].sum()), 0)
            mask2 = np.ones(4, dtype=bool)
            common.write_scannet_submission(
                root, "scene0568_01", [label], [(mask2, label, 0.95)], 1, spec)
            with open(idx) as f:
                gen2_refs = {line.split()[0] for line in f if line.strip()}
            self.assertTrue(gen1_refs.isdisjoint(gen2_refs))
            kept = [name for name in os.listdir(mask_dir) if name.startswith("scene0568_01_")]
            self.assertEqual(len(kept), 1)


class CompletionTests(unittest.TestCase):
    def test_malformed_index_is_missing(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            _write_ply(os.path.join(root, "scans", "scene0568_01",
                                    "scene0568_01_vh_clean_2.ply"), 4)
            pred = os.path.join(root, "pred")
            os.makedirs(pred)
            with open(os.path.join(pred, "scene0568_01.txt"), "w") as f:
                f.write("predicted_masks/scene0568_01_000.txt 3\n")
            status = scene_submission_status(
                pred, "scene0568_01", spec, "run-a", "mosaic3d", scannet_root=root)
            self.assertEqual(status, "missing")

    def test_mask_length_mismatch_is_missing(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            _write_ply(os.path.join(root, "scans", "scene0568_01",
                                    "scene0568_01_vh_clean_2.ply"), 4)
            pred = os.path.join(root, "pred")
            os.makedirs(os.path.join(pred, "predicted_masks"))
            with open(os.path.join(pred, "scene0568_01.txt"), "w") as f:
                f.write("predicted_masks/scene0568_01_000.txt 3 0.9\n")
            with open(os.path.join(pred, "predicted_masks", "scene0568_01_000.txt"), "w") as f:
                f.write("1\n1\n1\n")
            status = scene_submission_status(
                pred, "scene0568_01", spec, "run-a", "mosaic3d", scannet_root=root)
            self.assertEqual(status, "missing")


class OpenMaskOverrideTests(unittest.TestCase):
    def test_classify_uses_min_mask_points_argument(self):
        path = os.path.join(_SPELLBOOK, "predict", "models", "_openmask3d_run.py")
        with open(path) as f:
            text = f.read()
        self.assertIn(
            "def _classify(masks, feats, classes, device, min_mask_points=MIN_MASK_POINTS):",
            text)
        self.assertIn("_classify(masks, feats, args.classes, device, min_mask_points)", text)
        self.assertNotIn("if sel.sum() < MIN_MASK_POINTS:", text)


if __name__ == "__main__":
    unittest.main()
