import json
import os
import sys
import tempfile
import unittest

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from evaluation.benchmark import BENCHMARKS  # noqa: E402
from utils.compute_time import (  # noqa: E402
    format_elapsed, load_timing, prediction_timing_path, reconstruction_timing_path,
    write_prediction_timing, write_reconstruction_timing)


class FormatTests(unittest.TestCase):
    def test_format_elapsed(self):
        self.assertEqual(format_elapsed(12), "12s")
        self.assertEqual(format_elapsed(825), "13m 45s")
        self.assertEqual(format_elapsed(3654), "1h 00m 54s")
        self.assertIsNone(format_elapsed(None))
        self.assertIsNone(format_elapsed(-1))
        self.assertIsNone(format_elapsed(float("nan")))


class SidecarTests(unittest.TestCase):
    def test_prediction_round_trip(self):
        spec = BENCHMARKS["ScanNet20"]
        with tempfile.TemporaryDirectory() as root:
            write_prediction_timing(spec, "run-a", "mosaic3d", "scene0568_01", 12.4,
                                    scannet_root=root)
            path = prediction_timing_path(spec, "run-a", "mosaic3d", "scene0568_01", root)
            self.assertEqual(
                load_timing(path, "prediction", scene_id="scene0568_01",
                            run_id="run-a", model="mosaic3d"),
                12.4)
            self.assertIsNone(load_timing(path, "prediction", scene_id="scene0568_01",
                                          run_id="run-b", model="mosaic3d"))

    def test_reconstruction_round_trip_and_reject(self):
        with tempfile.TemporaryDirectory() as root:
            scene_dir = os.path.join(root, "scene9004_40")
            write_reconstruction_timing(scene_dir, "scene9004_40", "open3d", 88.0)
            path = reconstruction_timing_path(scene_dir)
            self.assertEqual(
                load_timing(path, "reconstruction", scene_id="scene9004_40"), 88.0)
            with open(path, "w") as fh:
                json.dump({"schema": 1, "kind": "reconstruction",
                           "scene_id": "scene9004_40", "elapsed_s": -2}, fh)
            self.assertIsNone(load_timing(path, "reconstruction", scene_id="scene9004_40"))


if __name__ == "__main__":
    unittest.main()
