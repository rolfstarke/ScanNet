import hashlib
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

from evaluation.benchmark import BENCHMARKS, submission_dir  # noqa: E402
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


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        digest.update(f.read())
    return digest.hexdigest()


def _sidecar_doc(**overrides):
    spec = BENCHMARKS["ScanNet20"]
    key = "predicted_masks/scene0568_00_000.txt"
    counts = {name: 0 for name in spec.class_labels}
    counts[spec.class_labels[0]] = 1
    doc = {
        "schema": 4,
        "metric": "scannet_instance_ap50",
        "iou_threshold": 0.5,
        "min_region_size": 100,
        "scene_id": "scene0568_00",
        "label_set": "ScanNet20",
        "run_id": "run-a",
        "model": "mosaic3d",
        "index_sha256": "a" * 64,
        "gt_sha256": "b" * 64,
        "tp": 1,
        "gt": 1,
        "verdicts": {key: "tp"},
        "matched_gt": {key: spec.valid_ids[0] * 1000 + 1},
        "eligible_gt_by_class": counts,
        "ap": 0.5,
        "ap_class_count": 1,
    }
    doc.update(overrides)
    return doc


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

    def test_matched_gt_explains_tp(self):
        gt = _gt_ids(100, self.gt_id)
        pred = _pred(np.ones(100, dtype=np.int32), self.label)
        out = scene_instance_summary(gt, pred, self.spec, "scene0000_00")
        self.assertEqual(out["matched_gt"], {"mask": self.gt_id})
        self.assertEqual(len(out["matched_gt"]), out["tp"])

    def test_duplicate_fp_has_no_matched_gt(self):
        gt = _gt_ids(100, self.gt_id)
        mask = np.ones(100, dtype=np.int32)
        pred = {
            "a": {"label_id": self.label, "conf": 0.9, "pred_mask": mask},
            "b": {"label_id": self.label, "conf": 0.8, "pred_mask": mask},
        }
        out = scene_instance_summary(gt, pred, self.spec, "scene0000_00")
        self.assertEqual(out["matched_gt"], {"a": self.gt_id})
        self.assertEqual(out["verdicts"]["b"], "fp")

    def test_eligible_gt_by_class_counts(self):
        gt = _gt_ids(100, self.gt_id)
        pred = _pred(np.ones(100, dtype=np.int32), self.label)
        out = scene_instance_summary(gt, pred, self.spec, "scene0000_00")
        label_name = self.spec.id_to_label[self.label]
        self.assertEqual(out["eligible_gt_by_class"][label_name], 1)
        self.assertEqual(sum(out["eligible_gt_by_class"].values()), out["gt"])


class SidecarTests(unittest.TestCase):
    def test_atomic_write_and_validation(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            path = score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_00", scannet_root=root)
            payload = _sidecar_doc()
            write_score_sidecar(path, payload)
            self.assertFalse(os.path.isfile(path + ".tmp"))
            loaded = load_score_sidecar(path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d")
            self.assertEqual(loaded["tp"], 1)
            self.assertEqual(loaded["gt"], 1)
            self.assertEqual(loaded["matched_gt"],
                             {"predicted_masks/scene0568_00_000.txt": spec.valid_ids[0] * 1000 + 1})
            self.assertEqual(loaded["eligible_gt_by_class"][spec.class_labels[0]], 1)
            self.assertIsNone(load_score_sidecar(
                path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d",
                index_sha256="c" * 64))
            keys = [f"predicted_masks/scene0568_00_{i:03d}.txt" for i in range(4)]
            payload["tp"] = 4
            payload["gt"] = 4
            payload["verdicts"] = {k: "tp" for k in keys}
            payload["matched_gt"] = {k: spec.valid_ids[0] * 1000 + 1 + i
                                     for i, k in enumerate(keys)}
            payload["eligible_gt_by_class"] = {
                name: (4 if name == spec.class_labels[0] else 0)
                for name in spec.class_labels}
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
            pred, index = _write_valid_pred(root, spec, "scene0568_00")
            path = score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_00", scannet_root=root)
            write_score_sidecar(path, _sidecar_doc(
                index_sha256=_file_sha256(index)))
            with mock.patch("evaluation.evaluate.score_and_write_sidecar") as scored:
                score_sidecars_cli([
                    "--benchmark", "ScanNet20", "--run-id", "run-a", "--models", "mosaic3d",
                    "--scenes", "0568_00", "--scannet-root", root, "--missing-only",
                ])
            scored.assert_not_called()
            self.assertEqual(pred, os.path.join(
                root, "predictions", "ScanNet20", "run-a", "mosaic3d"))

    def test_score_sidecars_missing_only_rescores_stale(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            _write_valid_pred(root, spec, "scene0568_00")
            path = score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_00", scannet_root=root)
            write_score_sidecar(path, _sidecar_doc())
            with mock.patch("evaluation.evaluate.score_and_write_sidecar") as scored:
                scored.return_value = {"tp": 1, "gt": 1}
                score_sidecars_cli([
                    "--benchmark", "ScanNet20", "--run-id", "run-a", "--models", "mosaic3d",
                    "--scenes", "0568_00", "--scannet-root", root, "--missing-only",
                ])
            scored.assert_called_once()


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

    def test_ensure_task_repairs_missing_marker(self):
        from predict import runner

        with tempfile.TemporaryDirectory() as tmp:
            tasks_log = os.path.join(tmp, "mosaic3d.tasks")
            runner._ensure_task(tasks_log, "scene0568_01", scannet_root=tmp)
            runner._ensure_task(tasks_log, "scene0568_01", scannet_root=tmp)
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
            timing = os.path.join(os.path.dirname(sidecar), "scene0568_00.timing.json")
            clip = os.path.join(os.path.dirname(sidecar), "scene0568_00.clip.npz")
            with open(timing, "w") as f:
                f.write("{}\n")
            with open(clip, "w") as f:
                f.write("x\n")
            write_score_sidecar(sidecar, _sidecar_doc(tp=1, gt=1, verdicts={}))
            eval_dir = os.path.join(root, "derived", "evaluations", "ScanNet20", "run-a")
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "mosaic3d.tasks"), "w") as f:
                f.write("scene0568_00\n")
            with mock.patch("reconstruct.cleanup.load_settings", return_value={"scannet_root": root}):
                purge_scan_predictions("scene0568_00", scannet_root=root)
            self.assertFalse(os.path.isfile(sidecar))
            self.assertFalse(os.path.isfile(timing))
            self.assertFalse(os.path.isfile(clip))
            self.assertFalse(os.path.isfile(os.path.join(pred, "scene0568_00.txt")))


def _load_common():
    path = os.path.join(_SPELLBOOK, "predict", "models", "common.py")
    spec = importlib.util.spec_from_file_location("spellbook_predict_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_valid_pred(root, spec, scene_id, run_id="run-a", model="mosaic3d", n=4):
    _write_ply(os.path.join(root, "scans", scene_id, f"{scene_id}_vh_clean_2.ply"), n)
    pred = submission_dir(spec, run_id, model, scannet_root=root)
    os.makedirs(os.path.join(pred, "predicted_masks"))
    with open(os.path.join(pred, "predicted_masks", f"{scene_id}_000.txt"), "w") as f:
        f.write("1\n" * 2 + "0\n" * (n - 2))
    index = os.path.join(pred, scene_id + ".txt")
    with open(index, "w") as f:
        f.write(f"predicted_masks/{scene_id}_000.txt {spec.valid_ids[0]} 0.9\n")
    return pred, index


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

    def test_load_overrides_rejects_nan_inf_zero_grid_and_enums(self):
        common = _load_common()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.json")
            cases = (
                ({"grid_size": 0}, {"grid_size"}),
                ({"grid_size": float("nan")}, {"grid_size"}),
                ({"grid_size": float("inf")}, {"grid_size"}),
                ({"point_limit": float("inf")}, {"point_limit"}),
                ({"detector": "nope"}, {"detector"}),
                ({"condition": "S3DIS"}, {"condition"}),
                ({"mask_confidence_threshold": 1.5}, {"mask_confidence_threshold"}),
            )
            for payload, allowed in cases:
                with open(path, "w") as f:
                    json.dump(payload, f)
                with self.assertRaises(ValueError):
                    common.load_overrides(path, allowed)

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

    def test_stale_sidecar_needs_score(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            pred, index = _write_valid_pred(root, spec, "scene0568_01")
            path = score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_01", scannet_root=root)
            counts = {name: (1 if name == spec.class_labels[0] else 0)
                      for name in spec.class_labels}
            write_score_sidecar(path, _sidecar_doc(
                scene_id="scene0568_01", index_sha256=_file_sha256(index),
                tp=0, gt=1, verdicts={}, matched_gt={},
                eligible_gt_by_class=counts))
            self.assertEqual(
                scene_submission_status(
                    pred, "scene0568_01", spec, "run-a", "mosaic3d", scannet_root=root),
                "complete")
            with open(index, "w") as f:
                f.write(f"predicted_masks/scene0568_01_000.txt {spec.valid_ids[0]} 0.8\n")
            self.assertEqual(
                scene_submission_status(
                    pred, "scene0568_01", spec, "run-a", "mosaic3d", scannet_root=root),
                "needs_score")


class EvaluateCliTests(unittest.TestCase):
    def test_refuses_existing_csv(self):
        from evaluation.evaluate import evaluate_cli
        from evaluation.runs import write_run_manifest

        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            write_run_manifest(spec, "run-a", ["scene0568_01"], ["mosaic3d"], scannet_root=root)
            csv_path = os.path.join(
                root, "derived", "evaluations", "ScanNet20", "run-a", "mosaic3d.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w") as f:
                f.write("class,class id,ap,ap50,ap25\nchair,5,0.1,0.2,0.3\n")
            with self.assertRaises(ValueError):
                evaluate_cli([
                    "--benchmark", "ScanNet20", "--run-id", "run-a",
                    "--models", "mosaic3d", "--scannet-root", root,
                ])


class SceneApThresholdTests(unittest.TestCase):
    def setUp(self):
        self.spec = _spec20()
        self.label = self.spec.valid_ids[0]
        self.gt_id = self.label * 1000 + 1

    def test_perfect_match_ap_one(self):
        from evaluation.scannet200_evaluator import scene_evaluation_summary
        gt = _gt_ids(100, self.gt_id)
        pred = _pred(np.ones(100, dtype=np.int32), self.label, conf=0.9)
        out = scene_evaluation_summary(gt, pred, self.spec, "scene0000_00")
        self.assertEqual(out["ap"], 1.0)
        self.assertEqual(out["ap_class_count"], 1)
        self.assertEqual(out["matched_gt"], {"mask": self.gt_id})

    def test_gt_no_pred_ap_zero(self):
        from evaluation.scannet200_evaluator import scene_evaluation_summary
        gt = _gt_ids(100, self.gt_id)
        out = scene_evaluation_summary(gt, {}, self.spec, "scene0000_00")
        self.assertEqual(out["ap"], 0.0)
        self.assertEqual(out["matched_gt"], {})

    def test_no_gt_ap_none(self):
        from evaluation.scannet200_evaluator import scene_evaluation_summary
        gt = np.zeros(100, dtype=np.int32)
        pred = _pred(np.ones(100, dtype=np.int32), self.label, conf=0.7)
        out = scene_evaluation_summary(gt, pred, self.spec, "scene0000_00")
        self.assertIsNone(out["ap"])
        self.assertEqual(out["ap_class_count"], 0)

    def test_schema3_rejected_schema4_roundtrips(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            path = score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_00", scannet_root=root)
            old = _sidecar_doc(schema=3)
            old.pop("matched_gt", None)
            old.pop("eligible_gt_by_class", None)
            write_score_sidecar(path, old)
            self.assertIsNone(load_score_sidecar(
                path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d"))
            good = _sidecar_doc()
            write_score_sidecar(path, good)
            loaded = load_score_sidecar(path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["ap"], 0.5)

    def test_invalid_matched_gt_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            spec = _spec20()
            path = score_sidecar_path(spec, "run-a", "mosaic3d", "scene0568_00", scannet_root=root)
            bad = _sidecar_doc(matched_gt={"predicted_masks/scene0568_00_000.txt": 42})
            write_score_sidecar(path, bad)
            self.assertIsNone(load_score_sidecar(
                path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d"))
            bad2 = _sidecar_doc(ap=2.0)
            write_score_sidecar(path, bad2)
            self.assertIsNone(load_score_sidecar(
                path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d"))
            bad3 = _sidecar_doc(eligible_gt_by_class={"nope": 1})
            write_score_sidecar(path, bad3)
            self.assertIsNone(load_score_sidecar(
                path, "scene0568_00", "ScanNet20", "run-a", "mosaic3d"))


class GlobalThresholdTests(unittest.TestCase):
    def setUp(self):
        self.spec = _spec20()
        self.label = self.spec.valid_ids[0]
        self.gt_id = self.label * 1000 + 1
        self.name = self.spec.class_labels[0]

    def _sweep(self, scene_id, confs, dup=False):
        from evaluation.scannet200_evaluator import scene_sweep_inputs
        gt = _gt_ids(100, self.gt_id)
        mask = np.ones(100, dtype=np.int32)
        if dup:
            pred = {f"{scene_id}_{c}": {"label_id": self.label, "conf": c,
                                        "pred_mask": mask} for c in confs}
        else:
            pred = {f"{scene_id}_{c}": {"label_id": self.label, "conf": c,
                                        "pred_mask": mask} for c in confs}
        return scene_sweep_inputs(gt, pred, self.spec, scene_id)

    def test_pooled_perfect_match_selects_lowest_conf(self):
        from evaluation.scannet200_evaluator import compute_global_thresholds
        scenes = [self._sweep("scene0001_00", [0.9]), self._sweep("scene0002_00", [0.7])]
        out = compute_global_thresholds(scenes, self.spec.class_labels)
        self.assertEqual(out[self.name]["confidence"], 0.7)
        self.assertEqual(out[self.name]["f1"], 1.0)
        self.assertEqual(out[self.name]["tp"], 2)
        self.assertEqual(out[self.name]["fn"], 0)
        self.assertEqual(out[self.name]["gt"], 2)

    def test_duplicate_chooses_higher_threshold(self):
        from evaluation.scannet200_evaluator import compute_global_thresholds
        scenes = [self._sweep("scene0001_00", [0.9, 0.8], dup=True)]
        out = compute_global_thresholds(scenes, self.spec.class_labels)
        self.assertEqual(out[self.name]["confidence"], 0.9)
        self.assertEqual(out[self.name]["tp"], 1)
        self.assertEqual(out[self.name]["fp"], 0)

    def test_pooled_differs_from_scene_average(self):
        from evaluation.scannet200_evaluator import compute_global_thresholds
        scenes = [self._sweep("scene0001_00", [0.9]), self._sweep("scene0002_00", [0.3])]
        out = compute_global_thresholds(scenes, self.spec.class_labels)
        self.assertEqual(out[self.name]["confidence"], 0.3)
        self.assertNotEqual(out[self.name]["confidence"], (0.9 + 0.3) / 2)

    def test_scene_order_does_not_matter(self):
        from evaluation.scannet200_evaluator import compute_global_thresholds
        a = self._sweep("scene0001_00", [0.9])
        b = self._sweep("scene0002_00", [0.3])
        out1 = compute_global_thresholds([a, b], self.spec.class_labels)
        out2 = compute_global_thresholds([b, a], self.spec.class_labels)
        self.assertEqual(out1, out2)

    def test_absent_class_omitted(self):
        from evaluation.scannet200_evaluator import compute_global_thresholds
        scenes = [self._sweep("scene0001_00", [0.9])]
        out = compute_global_thresholds(scenes, self.spec.class_labels)
        self.assertIn(self.name, out)
        self.assertEqual(len(out), 1)

    def test_no_gt_class_omitted(self):
        from evaluation.scannet200_evaluator import compute_global_thresholds, scene_sweep_inputs
        gt = np.zeros(100, dtype=np.int32)
        pred = _pred(np.ones(100, dtype=np.int32), self.label, conf=0.7)
        scenes = [scene_sweep_inputs(gt, pred, self.spec, "scene0001_00")]
        out = compute_global_thresholds(scenes, self.spec.class_labels)
        self.assertEqual(out, {})

    def test_global_artifact_roundtrip_and_hash_rejection(self):
        from evaluation.evaluate import (
            global_threshold_path, load_global_thresholds, write_global_thresholds,
            _global_threshold_doc)
        from evaluation.scannet200_evaluator import compute_global_thresholds
        spec = _spec20()
        scenes = [self._sweep("scene0001_00", [0.9])]
        thresholds = compute_global_thresholds(scenes, spec.class_labels)
        with tempfile.TemporaryDirectory() as root:
            path = global_threshold_path(spec, "run-a", "mosaic3d", scannet_root=root)
            manifest_sha = "m" * 64
            sidecar_sha = {"scene0001_00": "s" * 64}
            doc = _global_threshold_doc(
                spec, "run-a", "mosaic3d", ["scene0001_00"], manifest_sha,
                sidecar_sha, thresholds)
            write_global_thresholds(path, doc)
            self.assertFalse(os.path.isfile(path + ".tmp"))
            loaded = load_global_thresholds(
                path, spec, "run-a", "mosaic3d", ["scene0001_00"],
                manifest_sha256=manifest_sha, sidecar_sha256=sidecar_sha)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded[self.name]["confidence"], 0.9)
            self.assertIsNone(load_global_thresholds(
                path, spec, "run-a", "mosaic3d", ["scene0001_00"],
                manifest_sha256="x" * 64, sidecar_sha256=sidecar_sha))
            self.assertIsNone(load_global_thresholds(
                path, spec, "run-a", "mosaic3d", ["scene0001_00", "scene0002_00"],
                manifest_sha256=manifest_sha, sidecar_sha256=sidecar_sha))
            self.assertIsNone(load_global_thresholds(
                path, spec, "run-a", "other", ["scene0001_00"]))

    def test_filter_population_intersects_manifest_with_protocol_tuple(self):
        from evaluation.benchmark import PREDICTION_EVALUATION_SCENES
        from evaluation.evaluate import filter_population
        from evaluation.runs import write_run_manifest
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            subset = [PREDICTION_EVALUATION_SCENES[0], "scene0019_01"]
            write_run_manifest(spec, "run-a", subset, ["mosaic3d"], scannet_root=root)
            self.assertEqual(
                filter_population(spec, "run-a", scannet_root=root),
                [PREDICTION_EVALUATION_SCENES[0]])
            self.assertEqual(
                filter_population(spec, "nope", scannet_root=root),
                list(PREDICTION_EVALUATION_SCENES))


class OpenMaskOverrideTests(unittest.TestCase):
    def test_classify_uses_min_mask_points_argument(self):
        path = os.path.join(_SPELLBOOK, "predict", "models", "_openmask3d_run.py")
        with open(path) as f:
            text = f.read()
        self.assertIn("def _classify(masks, feats, classes, device,", text)
        self.assertIn("min_mask_points=MIN_MASK_POINTS", text)
        self.assertIn(
            "_classify(masks, feats, clip_classes, device, min_mask_points, clip_prompt)",
            text)
        self.assertNotIn("if sel.sum() < MIN_MASK_POINTS:", text)


if __name__ == "__main__":
    unittest.main()
