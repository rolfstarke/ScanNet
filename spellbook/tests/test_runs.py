import csv
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from evaluation.benchmark import BENCHMARKS  # noqa: E402
from evaluation.runs import (  # noqa: E402
    build_manifest, load_run_parameters, manifest_path, rank_cli, rank_runs,
    ranking_path, read_evaluator_csv, write_run_manifest,
)


def _spec20():
    return BENCHMARKS["ScanNet20"]


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "class id", "ap", "ap50", "ap25"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _csv_rows(ap, ap50, ap25, extra=None):
    rows = [
        {"class": "chair", "class id": "5", "ap": ap, "ap50": ap50, "ap25": ap25},
        {"class": "table", "class id": "7", "ap": "nan", "ap50": "nan", "ap25": "nan"},
    ]
    if extra:
        rows.append(extra)
    return rows


class ManifestTests(unittest.TestCase):
    def test_path_and_deterministic_document(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            doc = write_run_manifest(
                spec, "run-a", ["scene0568_01", "0575_00"], ["openyolo3d"],
                {"openyolo3d": {"network2d.nms": 0.25}}, issue=40, scannet_root=root)
            path = manifest_path(spec, "run-a", root)
            self.assertEqual(path, os.path.join(
                root, "derived", "evaluations", "ScanNet20", "run-a", "run.json"))
            with open(path) as f:
                text = f.read()
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(text, json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
            self.assertEqual(json.loads(text)["scenes"], ["scene0568_01", "scene0575_00"])
            self.assertFalse(os.path.isfile(path + ".tmp"))

    def test_duplicate_scenes_rejected(self):
        with self.assertRaises(ValueError):
            build_manifest("ScanNet20", "run-a", ["scene0568_01", "0568_01"], ["mosaic3d"])

    def test_identical_retry_succeeds(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            args = (spec, "run-a", ["scene0568_01"], ["mosaic3d"])
            first = write_run_manifest(*args, scannet_root=root)
            second = write_run_manifest(*args, scannet_root=root)
            self.assertEqual(first, second)

    def test_conflict_on_changed_fields(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            write_run_manifest(spec, "run-a", ["scene0568_01"], ["mosaic3d"],
                               issue=1, scannet_root=root)
            with self.assertRaises(ValueError):
                write_run_manifest(spec, "run-a", ["scene0568_01"], ["mosaic3d"],
                                   issue=2, scannet_root=root)
            with self.assertRaises(ValueError):
                write_run_manifest(spec, "run-a", ["scene0575_00"], ["mosaic3d"],
                                   issue=1, scannet_root=root)
            with self.assertRaises(ValueError):
                write_run_manifest(spec, "run-a", ["scene0568_01"], ["openyolo3d"],
                                   issue=1, scannet_root=root)
            with self.assertRaises(ValueError):
                write_run_manifest(spec, "run-a", ["scene0568_01"], ["mosaic3d"],
                                   {"mosaic3d": {"x": 1}}, issue=1, scannet_root=root)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            build_manifest("ScanNet20", "run/a", ["scene0568_01"], ["mosaic3d"])
        with self.assertRaises(ValueError):
            build_manifest("ScanNet20", "", ["scene0568_01"], ["mosaic3d"])
        with self.assertRaises(ValueError):
            build_manifest("ScanNet20", "run-a", ["scene0568_01"], ["mosaic3d"], issue=0)
        with self.assertRaises(ValueError):
            build_manifest("ScanNet20", "run-a", ["scene0568_01"], ["mosaic3d"], issue=True)
        with self.assertRaises(ValueError):
            build_manifest("ScanNet20", "run-a", ["not-a-scene"], ["mosaic3d"])
        with self.assertRaises(ValueError):
            build_manifest("ScanNet20", "run-a", ["scene0568_01"], ["mosaic3d"],
                           {"openyolo3d": {}})
        with self.assertRaises(ValueError):
            build_manifest("ScanNet20", "run-a", ["scene0568_01"], ["mosaic3d"],
                           {"mosaic3d": []})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "params.json")
            with open(path, "w") as f:
                json.dump({"openyolo3d": {"n": 1}}, f)
            with self.assertRaises(ValueError):
                load_run_parameters(path, ["mosaic3d"])


class RankingTests(unittest.TestCase):
    def _layout(self, root, run_id, method, scenes, rows, params=None, issue=None):
        spec = _spec20()
        write_run_manifest(spec, run_id, scenes, [method],
                           {method: params or {}}, issue=issue, scannet_root=root)
        _write_csv(os.path.join(root, "derived", "evaluations", "ScanNet20",
                                run_id, f"{method}.csv"), rows)
        return spec

    def test_comparable_runs_rank_and_nan_mean(self):
        with tempfile.TemporaryDirectory() as root:
            spec = self._layout(root, "run-b", "mosaic3d", ["scene0568_01"],
                                _csv_rows("0.6", "0.7", "0.9"), issue=2)
            self._layout(root, "run-a", "mosaic3d", ["scene0568_01"],
                         _csv_rows("0.5", "0.8", "0.2"), issue=1)
            rows, path = rank_runs(spec, scannet_root=root)
            by_id = {row["run_id"]: row for row in rows}
            self.assertEqual(by_id["run-a"]["ap50"], 0.8)
            self.assertEqual(by_id["run-a"]["ap"], 0.5)
            self.assertEqual(by_id["run-a"]["rank_ap50"], 1)
            self.assertEqual(by_id["run-b"]["rank_ap50"], 2)
            self.assertEqual(by_id["run-b"]["rank_ap"], 1)
            self.assertEqual(by_id["run-a"]["rank_ap"], 2)
            self.assertEqual(by_id["run-b"]["rank_ap25"], 1)
            self.assertEqual(by_id["run-a"]["rank_ap25"], 2)
            self.assertEqual(path, ranking_path(spec, root))
            self.assertFalse(os.path.isfile(path + ".tmp"))
            with open(path, newline="") as f:
                dumped = list(csv.DictReader(f))
            self.assertEqual([row["run_id"] for row in dumped], ["run-b", "run-a"])

    def test_equal_ap_orders_by_ap50(self):
        with tempfile.TemporaryDirectory() as root:
            spec = self._layout(root, "run-b", "mosaic3d", ["scene0568_01"],
                                _csv_rows("0.5", "0.9", "0.1"))
            self._layout(root, "run-a", "mosaic3d", ["scene0568_01"],
                         _csv_rows("0.5", "0.2", "0.9"))
            rows, path = rank_runs(spec, scannet_root=root)
            with open(path, newline="") as f:
                dumped = list(csv.DictReader(f))
            self.assertEqual([row["run_id"] for row in dumped], ["run-b", "run-a"])
            by_id = {row["run_id"]: row for row in rows}
            self.assertGreater(by_id["run-b"]["ap50"], by_id["run-a"]["ap50"])
            self.assertEqual(by_id["run-b"]["rank_ap"], 1)
            self.assertEqual(by_id["run-a"]["rank_ap"], 2)

    def test_groups_do_not_cross_method_or_scenes(self):
        with tempfile.TemporaryDirectory() as root:
            spec = self._layout(root, "run-a", "mosaic3d", ["scene0568_01"],
                                _csv_rows("0.1", "0.1", "0.1"))
            self._layout(root, "run-b", "openyolo3d", ["scene0568_01"],
                         _csv_rows("0.9", "0.9", "0.9"))
            self._layout(root, "run-c", "mosaic3d", ["scene0575_00"],
                         _csv_rows("0.2", "0.2", "0.2"))
            rows, _ = rank_runs(spec, scannet_root=root)
            ranks = {(row["run_id"], row["method"]): row["rank_ap50"] for row in rows}
            self.assertEqual(ranks[("run-a", "mosaic3d")], 1)
            self.assertEqual(ranks[("run-b", "openyolo3d")], 1)
            self.assertEqual(ranks[("run-c", "mosaic3d")], 1)

    def test_missing_and_malformed_csv_skipped(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            write_run_manifest(spec, "run-ok", ["scene0568_01"], ["mosaic3d"],
                               scannet_root=root)
            _write_csv(os.path.join(root, "derived", "evaluations", "ScanNet20",
                                    "run-ok", "mosaic3d.csv"),
                       _csv_rows("0.1", "0.2", "0.3"))
            write_run_manifest(spec, "run-missing", ["scene0568_01"], ["mosaic3d"],
                               scannet_root=root)
            write_run_manifest(spec, "run-bad", ["scene0568_01"], ["mosaic3d"],
                               scannet_root=root)
            bad = os.path.join(root, "derived", "evaluations", "ScanNet20",
                               "run-bad", "mosaic3d.csv")
            os.makedirs(os.path.dirname(bad), exist_ok=True)
            with open(bad, "w") as f:
                f.write("nope\n")
            rows, path = rank_runs(spec, scannet_root=root)
            self.assertEqual([row["run_id"] for row in rows], ["run-ok"])
            with open(path, newline="") as f:
                dumped = list(csv.DictReader(f))
            self.assertEqual([row["run_id"] for row in dumped], ["run-ok"])

    def test_method_filter_does_not_shrink_csv(self):
        with tempfile.TemporaryDirectory() as root:
            spec = self._layout(root, "run-a", "mosaic3d", ["scene0568_01"],
                                _csv_rows("0.1", "0.1", "0.1"))
            self._layout(root, "run-b", "openyolo3d", ["scene0568_01"],
                         _csv_rows("0.2", "0.2", "0.2"))
            rank_cli(["--benchmark", "ScanNet20", "--method", "mosaic3d",
                      "--scannet-root", root])
            with open(ranking_path(spec, root), newline="") as f:
                dumped = list(csv.DictReader(f))
            self.assertEqual(sorted(row["method"] for row in dumped),
                             ["mosaic3d", "openyolo3d"])

    def test_read_evaluator_csv_rejects_bad_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.csv")
            with open(path, "w") as f:
                f.write("a,b\n1,2\n")
            with self.assertRaises(ValueError):
                read_evaluator_csv(path)


class DummyLock:
    def __init__(self, *args, **kwargs):
        pass

    def acquire(self):
        return self

    def downgrade_to_shared(self):
        pass

    def release(self):
        pass


@contextmanager
def _lease(*args, **kwargs):
    yield mock.Mock(index=1, fileno=lambda: 3)


class RunnerTests(unittest.TestCase):
    def _scene_root(self, root):
        scans = os.path.join(root, "scans")
        ply = os.path.join(scans, "scene0568_01", "scene0568_01_vh_clean_2.ply")
        os.makedirs(os.path.dirname(ply))
        with open(ply, "w") as f:
            f.write("ply\n")
        return scans

    def _run(self, root, scans, **predict_kwargs):
        from predict import runner
        settings = {"gpu_pool": [1], "scannet_root": root, "default": "ScanNet20"}
        patches = [
            mock.patch("predict.runner.preflight_model", return_value=None),
            mock.patch("predict.runner.SCANS_DIR", scans),
            mock.patch("predict.runner.load_settings", return_value=settings),
            mock.patch("utils.scan_lock.ScanLock", DummyLock),
            mock.patch("utils.gpu.gpu_lease", _lease),
            mock.patch("predict.runner._finalize_prediction"),
            mock.patch("evaluation.runs.collect_method_provenance", return_value={}),
            mock.patch("evaluation.evaluate.scene_submission_status", return_value="missing"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return runner.predict(["scene0568_01"], ["mosaic3d"], None, "ScanNet20",
                              "run-a", **predict_kwargs)

    def test_manifest_written_once_before_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            scans = self._scene_root(root)
            calls = []

            def write(*args, **kwargs):
                calls.append(("write", args, kwargs))
                return {}

            def run_one(*args, **kwargs):
                calls.append(("run", args, kwargs))
                return ("mosaic3d", "scene0568_01", "/tmp", 0.1, True)

            with mock.patch("evaluation.runs.write_run_manifest", side_effect=write), \
                    mock.patch("predict.runner._run_one", side_effect=run_one):
                self._run(root, scans, run_parameters={"mosaic3d": {"x": 1}}, issue=7)
            self.assertEqual([item[0] for item in calls], ["write", "run"])
            _spec, run_id, scenes, methods = calls[0][1]
            kwargs = calls[0][2]
            self.assertEqual(run_id, "run-a")
            self.assertEqual(scenes, ["scene0568_01"])
            self.assertEqual(methods, ["mosaic3d"])
            self.assertEqual(kwargs["run_parameters"], {"mosaic3d": {"x": 1}})
            self.assertEqual(kwargs["issue"], 7)
            self.assertEqual(kwargs["scannet_root"], root)
            timing = os.path.join(
                root, "derived", "evaluations", "ScanNet20", "run-a", "mosaic3d",
                "scene0568_01.timing.json")
            self.assertTrue(os.path.isfile(timing))
            run_kwargs = calls[1][2]
            self.assertTrue(run_kwargs["features_out"].endswith(
                "scene0568_01.clip.npz"))
            self.assertIn("/derived/evaluations/", run_kwargs["features_out"].replace("\\", "/"))

    def test_manifest_conflict_skips_models(self):
        spec = _spec20()
        with tempfile.TemporaryDirectory() as root:
            scans = self._scene_root(root)
            write_run_manifest(spec, "run-a", ["scene0568_01"], ["mosaic3d"],
                               issue=1, scannet_root=root)
            with mock.patch("predict.runner._run_one") as run_one:
                with self.assertRaises(ValueError):
                    self._run(root, scans, issue=2)
            run_one.assert_not_called()


if __name__ == "__main__":
    unittest.main()
