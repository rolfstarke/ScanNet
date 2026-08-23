import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(_SPELLBOOK)
sys.path.insert(0, _SPELLBOOK)
sys.path.insert(0, os.path.join(_REPO_ROOT, "BenchmarkScripts"))

from benchmark import resolve_benchmark  # noqa: E402
from utils import visualize  # noqa: E402
from scannet200_evaluator import Evaluator  # noqa: E402


def _evaluator_ap50_events(spec, gt2pred, pred2gt, scene_id):
    """Independent replica of the evaluator's per-scene AP50 loop
    (spellbook/scannet200_evaluator.py evaluate_matches inner loop)."""
    overlap_th = visualize.AP50_THRESHOLD
    min_region = visualize.MIN_REGION_SIZE
    y_true = np.empty(0)
    y_score = np.empty(0)
    pred_visited = {}
    for label in spec.class_labels:
        for p in pred2gt[label]:
            pred_visited[p["filename"]] = False
    for label in spec.class_labels:
        pred_instances = pred2gt[label]
        gt_instances = [gt for gt in gt2pred[label]
                        if gt["instance_id"] >= 1000 and gt["vert_count"] >= min_region]
        cur_true = np.ones(len(gt_instances))
        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
        cur_match = np.zeros(len(gt_instances), dtype=np.bool_)
        for (gti, gt) in enumerate(gt_instances):
            found_match = False
            for pred in gt["matched_pred"]:
                if pred_visited[pred["filename"]]:
                    continue
                overlap = float(pred["intersection"]) / (
                    gt["vert_count"] + pred["vert_count"] - pred["intersection"])
                if overlap > overlap_th:
                    confidence = pred["confidence"]
                    if cur_match[gti]:
                        max_score = max(cur_score[gti], confidence)
                        min_score = min(cur_score[gti], confidence)
                        cur_score[gti] = max_score
                        cur_true = np.append(cur_true, 0)
                        cur_score = np.append(cur_score, min_score)
                        cur_match = np.append(cur_match, True)
                    else:
                        found_match = True
                        cur_match[gti] = True
                        cur_score[gti] = confidence
                        pred_visited[pred["filename"]] = True
        cur_true = cur_true[cur_match == True]  # noqa: E712
        cur_score = cur_score[cur_match == True]  # noqa: E712
        for pred in pred_instances:
            found_gt = False
            for gt in pred["matched_gt"]:
                overlap = float(gt["intersection"]) / (
                    gt["vert_count"] + pred["vert_count"] - gt["intersection"])
                if overlap > overlap_th:
                    found_gt = True
                    break
            if not found_gt:
                num_ignore = pred["void_intersection"]
                for gt in pred["matched_gt"]:
                    if gt["instance_id"] < 1000 or gt["vert_count"] < min_region:
                        num_ignore += gt["intersection"]
                proportion_ignore = float(num_ignore) / pred["vert_count"]
                if proportion_ignore <= overlap_th:
                    cur_true = np.append(cur_true, 0)
                    cur_score = np.append(cur_score, pred["confidence"])
        y_true = np.append(y_true, cur_true)
        y_score = np.append(y_score, cur_score)
    return y_true, y_score


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.spec = resolve_benchmark("ScanNet20")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _touch(self, path, mtime):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("0\n")
        os.utime(path, (mtime, mtime))

    def test_runs_for_scene_missing_root(self):
        self.assertEqual(visualize.runs_for_scene(self.spec, "scene0568_00", self.tmp), [])

    def test_runs_for_scene_filters_other_scenes_and_sorts_newest_first(self):
        root = os.path.join(self.tmp, "predictions", "ScanNet20")
        self._touch(f"{root}/old/mosaic3d/scene0568_00.txt", 100.0)
        self._touch(f"{root}/new/open3dis/scene0568_00.txt", 200.0)
        self._touch(f"{root}/new/open3dis/scene9999_00.txt", 250.0)
        self._touch(f"{root}/other/mosaic3d/scene0217_00.txt", 300.0)
        self.assertEqual(visualize.runs_for_scene(self.spec, "scene0568_00", self.tmp),
                         ["new", "old"])

    def test_available_models_filters_scene_and_sorts(self):
        root = os.path.join(self.tmp, "predictions", "ScanNet20", "run")
        self._touch(f"{root}/b_model/scene0568_00.txt", 1.0)
        self._touch(f"{root}/a_model/scene0568_00.txt", 1.0)
        self._touch(f"{root}/c_model/scene9999_00.txt", 1.0)
        self.assertEqual(visualize.available_models(root, "scene0568_00"),
                         ["a_model", "b_model"])
        self.assertEqual(visualize.available_models(os.path.join(self.tmp, "missing")), [])
        self.assertEqual(visualize.available_models(root), ["a_model", "b_model", "c_model"])


class ClassifyTests(unittest.TestCase):
    N = 1000

    def setUp(self):
        self.spec = resolve_benchmark("ScanNet20")
        self.label_a = self.spec.valid_ids[0]
        self.label_b = self.spec.valid_ids[1]
        self.scene_id = "sceneX"
        self.gt_ids = np.zeros(self.N, dtype=np.int64)
        self.gt_ids[0:100] = self.label_a * 1000 + 1      # GT A
        self.gt_ids[100:200] = self.label_b * 1000 + 2    # GT B
        self.gt_ids[200:300] = self.label_a * 1000 + 3    # GT C
        self.gt_ids[300:400] = self.label_a * 1000 + 4    # GT D

    def _pred(self, label_id, vertices, conf):
        mask = np.zeros(self.N, dtype=bool)
        mask[vertices] = True
        return {"label_id": label_id, "conf": conf, "pred_mask": mask}

    def _instances(self, entries):
        instances = {}
        for i, (label_id, vertices, conf) in enumerate(entries):
            instances[f"predicted_masks/sceneX_{i}.txt"] = self._pred(label_id, vertices, conf)
        return instances

    def test_ap50_verdicts(self):
        entries = [
            (self.label_a, range(0, 100), 0.9),                        # perfect match -> TP
            (self.label_a, list(range(200, 300)) + list(range(500, 590)), 0.8),  # IoU ~0.526 -> TP
            (self.label_a, list(range(300, 400)) + list(range(590, 690)), 0.7),  # IoU 0.5 -> FP
            (self.label_a, range(100, 200), 0.6),                      # wrong-class overlap -> FP
            (self.label_a, range(0, 100), 0.95),                       # duplicate, higher conf -> FP
            (self.label_a, range(0, 100), 0.4),                        # duplicate, lower conf -> FP
            (self.label_a, range(600, 801), 0.5),                      # void-only -> ignored
            (self.label_a, range(801, 900), 0.5),                      # 99 verts -> filtered
            (99999, range(0, 100), 0.5),                               # invalid class -> filtered
        ]
        verdicts, events = visualize.classify_ap50(
            self.gt_ids, self._instances(entries), self.spec, self.scene_id)

        keys = [k for k, _, _ in events]
        self.assertEqual(verdicts["predicted_masks/sceneX_0.txt"], "tp")   # perfect
        self.assertEqual(verdicts["predicted_masks/sceneX_1.txt"], "tp")   # IoU 0.526
        self.assertEqual(verdicts["predicted_masks/sceneX_2.txt"], "fp")   # IoU exactly 0.5
        self.assertEqual(verdicts["predicted_masks/sceneX_3.txt"], "fp")   # wrong class
        self.assertEqual(verdicts["predicted_masks/sceneX_4.txt"], "fp")   # duplicate
        self.assertEqual(verdicts["predicted_masks/sceneX_5.txt"], "fp")   # duplicate
        self.assertEqual(verdicts["predicted_masks/sceneX_6.txt"], "ignored")
        self.assertNotIn("predicted_masks/sceneX_7.txt", verdicts)        # < 100 verts
        self.assertNotIn("predicted_masks/sceneX_8.txt", verdicts)        # invalid class
        self.assertEqual(len(keys), len(set(keys)))

    def test_events_match_evaluator_ap50(self):
        entries = [
            (self.label_a, range(0, 100), 0.9),
            (self.label_a, list(range(200, 300)) + list(range(500, 590)), 0.8),
            (self.label_a, list(range(300, 400)) + list(range(590, 690)), 0.7),
            (self.label_a, range(100, 200), 0.6),
            (self.label_a, range(0, 100), 0.95),
            (self.label_a, range(0, 100), 0.4),
            (self.label_a, range(600, 801), 0.5),
        ]
        instances = self._instances(entries)
        verdicts, events = visualize.classify_ap50(
            self.gt_ids, instances, self.spec, self.scene_id)

        evaluator = Evaluator(self.spec.class_labels, self.spec.valid_ids)
        evaluator.add_gt(self.gt_ids, self.scene_id)
        evaluator.add_prediction(instances, self.scene_id)
        gt2pred, pred2gt = evaluator.assign_instances_for_scan(self.scene_id)
        expected_true, expected_score = _evaluator_ap50_events(
            self.spec, gt2pred, pred2gt, self.scene_id)

        event_true = np.array([y for _, y, _ in events])
        event_score = np.array([s for _, _, s in events])
        np.testing.assert_array_equal(event_true, expected_true)
        np.testing.assert_allclose(event_score, expected_score)

        claimed = sum(1 for v in verdicts.values() if v == "tp")
        self.assertEqual(claimed, int(expected_true.sum()))

    def test_empty_prediction_has_no_events(self):
        verdicts, events = visualize.classify_ap50(
            self.gt_ids, {}, self.spec, self.scene_id)
        self.assertEqual(verdicts, {})
        self.assertEqual(events, [])

    def test_perfect_instances_on_both_labels(self):
        entries = [
            (self.label_a, range(0, 100), 0.9),
            (self.label_b, range(100, 200), 0.9),
        ]
        verdicts, _ = visualize.classify_ap50(
            self.gt_ids, self._instances(entries), self.spec, self.scene_id)
        self.assertEqual(verdicts["predicted_masks/sceneX_0.txt"], "tp")
        self.assertEqual(verdicts["predicted_masks/sceneX_1.txt"], "tp")


if __name__ == "__main__":
    unittest.main()
