import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from evaluation.benchmark import (  # noqa: E402
    PREDICTION_EVALUATION_SCENES, PREDICTION_METHODS, official_val_scenes,
    validate_prediction_methods, validate_prediction_scenes,
)
from evaluation.runs import (  # noqa: E402
    _is_comparable, load_manifest, prune_prediction_artifacts, write_run_manifest,
)
from predict.runner import (  # noqa: E402
    MODEL_NEEDS_FRAMES, MODEL_PYTHON, MODEL_RESOURCES, MODEL_RUN_SCRIPT, SUPPORTED_MODELS,
)


LOCKED = (
    "scene0046_00", "scene0084_01", "scene0086_01", "scene0100_02",
    "scene0164_00", "scene0207_02", "scene0221_00", "scene0251_00",
    "scene0307_00", "scene0334_00", "scene0357_00", "scene0535_00",
    "scene0618_00", "scene0644_00", "scene0678_01", "scene0699_00",
)
LEGACY_10 = (
    "scene0019_01", "scene0217_00", "scene0304_00", "scene0412_00",
    "scene0414_00", "scene0426_02", "scene0488_00", "scene0549_00",
    "scene0568_01", "scene0575_00",
)


class ProtocolTests(unittest.TestCase):
    def test_fixed_tuple_is_official_and_unique(self):
        self.assertEqual(PREDICTION_EVALUATION_SCENES, LOCKED)
        official = official_val_scenes()
        self.assertEqual(len(PREDICTION_EVALUATION_SCENES), 16)
        self.assertTrue(set(PREDICTION_EVALUATION_SCENES) <= official)
        bases = [scene.rsplit("_", 1)[0] for scene in PREDICTION_EVALUATION_SCENES]
        self.assertEqual(len(set(bases)), 16)

    def test_rejects_custom_and_nonselected_and_duplicates(self):
        with self.assertRaises(ValueError):
            validate_prediction_scenes(["scene9004_40"])
        with self.assertRaises(ValueError):
            validate_prediction_scenes(["scene0568_00"])
        with self.assertRaises(ValueError):
            validate_prediction_scenes(["scene0568_01"])
        with self.assertRaises(ValueError):
            validate_prediction_scenes(["nope"])
        with self.assertRaises(ValueError):
            validate_prediction_scenes(["scene0046_00", "0046_00"])
        self.assertEqual(
            validate_prediction_scenes(["0046_00", "scene0084_01"]),
            ["scene0046_00", "scene0084_01"])
        self.assertEqual(
            validate_prediction_scenes(LOCKED, require_complete=True),
            list(LOCKED))
        with self.assertRaises(ValueError):
            validate_prediction_scenes(["scene0046_00"], require_complete=True)

    def test_legacy_manifest_loads_but_is_not_comparable(self):
        from evaluation.benchmark import BENCHMARKS
        spec = BENCHMARKS["ScanNet200"]
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "run.json")
            doc = {
                "schema": 2,
                "benchmark": "ScanNet200",
                "run_id": "legacy-run",
                "scenes": list(LEGACY_10),
                "methods": {"mosaic3d": {"parameters": {}}},
            }
            with open(path, "w") as f:
                json.dump(doc, f)
            loaded = load_manifest(path, spec=spec, run_id="legacy-run")
            self.assertEqual(loaded["scenes"], list(LEGACY_10))
            self.assertFalse(_is_comparable(loaded))

    def test_current_manifest_is_comparable(self):
        from evaluation.benchmark import BENCHMARKS
        spec = BENCHMARKS["ScanNet200"]
        with tempfile.TemporaryDirectory() as root:
            current = write_run_manifest(
                spec, "current-run", list(LOCKED), ["mosaic3d"], scannet_root=root)
            self.assertTrue(_is_comparable(current))

    def test_methods_and_registries_match(self):
        self.assertEqual(tuple(SUPPORTED_MODELS), PREDICTION_METHODS)
        self.assertEqual(set(MODEL_PYTHON), set(SUPPORTED_MODELS))
        self.assertEqual(set(MODEL_RUN_SCRIPT), set(SUPPORTED_MODELS))
        self.assertEqual(set(MODEL_NEEDS_FRAMES), set(SUPPORTED_MODELS))
        self.assertEqual(set(MODEL_RESOURCES), set(SUPPORTED_MODELS))
        with self.assertRaises(ValueError):
            validate_prediction_methods(["mosaic3d", "mosaic3d"])
        with self.assertRaises(ValueError):
            validate_prediction_methods(["not-a-model"])


class WrapperCliTests(unittest.TestCase):
    def test_wrappers_expose_run_id_and_parameters(self):
        models = os.path.join(_SPELLBOOK, "predict", "models")
        for name in PREDICTION_METHODS:
            with open(os.path.join(models, f"_{name}_run.py")) as f:
                text = f.read()
            self.assertIn("add_run_args(ap)", text)
            self.assertIn("load_overrides(", text)


class PruneTests(unittest.TestCase):
    def test_dry_run_refuses_incomplete_and_stays_in_root(self):
        from evaluation.benchmark import BENCHMARKS
        spec = BENCHMARKS["ScanNet200"]
        with tempfile.TemporaryDirectory() as root:
            write_run_manifest(spec, "run-a", list(LOCKED), ["mosaic3d"], scannet_root=root)
            with self.assertRaises(ValueError):
                prune_prediction_artifacts(spec, "run-a", "mosaic3d", scannet_root=root)
            result = None
            eval_root = os.path.join(root, "derived", "evaluations", "ScanNet200", "run-a")
            os.makedirs(os.path.join(eval_root, "mosaic3d"), exist_ok=True)
            with open(os.path.join(eval_root, "mosaic3d.csv"), "w") as f:
                f.write("class,class id,ap,ap50,ap25\nchair,5,0.1,0.2,0.3\n")
            with open(os.path.join(eval_root, "mosaic3d.tasks"), "w") as f:
                f.write("\n".join(LOCKED) + "\n")
            for scene in LOCKED:
                with open(os.path.join(eval_root, "mosaic3d", scene + ".tp50.json"), "w") as f:
                    f.write("{}\n")
            result = prune_prediction_artifacts(
                spec, "run-a", "mosaic3d", scannet_root=root, apply=False)
            self.assertFalse(result["apply"])
            self.assertTrue(result["target"].startswith(os.path.realpath(root)))


class MainPredictTests(unittest.TestCase):
    def test_omitted_scene_uses_full_tuple(self):
        import main as spellbook_main
        from contextlib import contextmanager
        captured = {}

        def fake_predict(scenes, models, classes, benchmark, run_id, **kwargs):
            captured["scenes"] = scenes
            captured["classes"] = classes
            return [("mosaic3d", scenes[0], "/tmp", 0.1, True)]

        @contextmanager
        def _noop_job(*a, **k):
            yield "job-test"

        argv = ["prog", "--foreground", "--predict", "--models", "mosaic3d",
                "--benchmark", "ScanNet200"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("predict.runner.predict", side_effect=fake_predict), \
                mock.patch("jobs.job_context", _noop_job):
            spellbook_main.main()
        self.assertEqual(captured["scenes"], list(LOCKED))
        self.assertIsNone(captured["classes"])

    def test_classes_rejected(self):
        import main as spellbook_main
        from contextlib import contextmanager

        @contextmanager
        def _noop_job(*a, **k):
            yield "job-test"

        argv = ["prog", "--foreground", "--predict", "--models", "mosaic3d",
                "--classes", "chair"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("jobs.job_context", _noop_job):
            with self.assertRaises(SystemExit):
                spellbook_main.main()


if __name__ == "__main__":
    unittest.main()
