import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from evaluation.benchmark import BENCHMARKS  # noqa: E402
from utils.query import (  # noqa: E402
    QUERY_IMAGE, QUERY_OFF, QUERY_SEARCH, QueryClient,
    clip_features_path, cosine_search, l2_normalize_rows, load_clip_features,
    query_modes_for, write_aligned_features,
    write_clip_features,
)


def _index(path, keys):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for i, key in enumerate(keys):
            fh.write(f"{key} 3 0.9{i}\n")


class ArtifactTests(unittest.TestCase):
    def test_write_load_and_stale_hash(self):
        spec = BENCHMARKS["ScanNet200"]
        keys = ["predicted_masks/scene0568_01_000.txt", "predicted_masks/scene0568_01_001.txt"]
        feats = l2_normalize_rows(np.eye(2, 768, dtype=np.float32))
        with tempfile.TemporaryDirectory() as root:
            index = os.path.join(root, "scene0568_01.txt")
            _index(index, keys)
            path = clip_features_path(spec, "run-a", "openmask3d", "scene0568_01", scannet_root=root)
            write_clip_features(
                path, feats, keys, index, "scene0568_01", spec.name, "run-a", "openmask3d",
                "openai_clip", "ViT-L/14@336px", 768)
            self.assertFalse(os.path.isfile(path + ".tmp"))
            loaded = load_clip_features(path, spec, "run-a", "openmask3d", "scene0568_01", index)
            self.assertEqual(loaded["keys"], keys)
            self.assertEqual(loaded["features"].shape, (2, 768))
            with open(index, "a") as fh:
                fh.write("predicted_masks/scene0568_01_002.txt 3 0.1\n")
            self.assertIsNone(load_clip_features(
                path, spec, "run-a", "openmask3d", "scene0568_01", index))

    def test_aligned_rows_and_reject_bad(self):
        spec = BENCHMARKS["ScanNet200"]
        keys = ["predicted_masks/a.txt"]
        rows = [np.ones(768, dtype=np.float32), np.full(768, 2.0, dtype=np.float32)]
        with tempfile.TemporaryDirectory() as root:
            index = os.path.join(root, "scene0568_01.txt")
            _index(index, keys)
            path = os.path.join(root, "scene0568_01.clip.npz")
            write_aligned_features(
                path, rows, [1], keys, index, scene_id="scene0568_01",
                benchmark=spec.name, run_id="run-a", model="openmask3d",
                encoder_family="openai_clip", encoder_name="ViT-L/14@336px", feature_dim=768)
            loaded = load_clip_features(path, spec, "run-a", "openmask3d", "scene0568_01", index)
            np.testing.assert_allclose(loaded["features"][0, 0], loaded["features"][0, 1])
            with self.assertRaises(ValueError):
                write_clip_features(
                    path, np.zeros((1, 768), dtype=np.float32), keys, index, "scene0568_01",
                    spec.name, "run-a", "openmask3d", "openai_clip", "ViT-L/14@336px", 768)


class SimilarityTests(unittest.TestCase):
    def test_search_and_modes(self):
        feats = l2_normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32))
        ranks, scores = cosine_search(feats, [1.0, 0.0], top_k=2)
        self.assertEqual(list(ranks), [0, 2])
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(
            query_modes_for("openmask3d", has_features=True, source_pred=True),
            [QUERY_OFF, QUERY_SEARCH, QUERY_IMAGE])
        self.assertEqual(
            query_modes_for("openins3d", has_snap=True, source_pred=True),
            [QUERY_OFF, QUERY_SEARCH])
        self.assertEqual(query_modes_for("openyolo3d", source_pred=True), [QUERY_OFF])
        from utils.query import clip_path_ready
        self.assertFalse(clip_path_ready(None))
        self.assertFalse(clip_path_ready("/no/such/clip.npz"))


class ClientTests(unittest.TestCase):
    def test_latest_request_and_stale(self):
        client = QueryClient()
        class Dummy:
            def __init__(self):
                self.stdin = self
                self.stdout = self
                self.lines = []
            def write(self, line):
                self.lines.append(line)
            def flush(self):
                pass
            def poll(self):
                return None
            def close(self):
                pass
            def terminate(self):
                pass
            def wait(self, timeout=None):
                return 0
            def kill(self):
                pass
        dummy = Dummy()
        client.proc = dummy
        first = client.submit({"op": "search"})
        second = client.submit({"op": "search"})
        self.assertEqual(second, first + 1)
        self.assertEqual(len(dummy.lines), 2)
        client.close()

    def test_worker_is_pinned_to_gpu_zero_without_lease(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        with mock.patch.dict(os.environ, {"SPELLBOOK_GPU_LEASE_FD": "123"}):
            with mock.patch("utils.query.subprocess.Popen", return_value=proc) as popen:
                client = QueryClient()
                client.ensure("/method/python", "open_clip", "encoder", device="cuda:0")
        args, kwargs = popen.call_args
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("SPELLBOOK_GPU_LEASE_FD", kwargs["env"])
        self.assertEqual(args[0][-2:], ["--device", "cuda:0"])


class WriterRowTests(unittest.TestCase):
    def test_source_indices_skip_small_masks(self):
        import importlib.util
        path = os.path.join(_SPELLBOOK, "predict", "models", "common.py")
        spec = importlib.util.spec_from_file_location("spellbook_predict_common", path)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        bench = BENCHMARKS["ScanNet20"]
        with tempfile.TemporaryDirectory() as tmp:
            instances = [
                (np.array([1, 0, 0, 0], dtype=bool), "chair", 0.9),
                (np.array([1, 1, 1, 0], dtype=bool), "chair", 0.8),
            ]
            rows = common.write_scannet_submission_rows(
                tmp, "scene0568_01", ["chair"], instances, 2, bench)
            self.assertEqual(rows["n_written"], 1)
            self.assertEqual(rows["source_indices"], [1])
            self.assertTrue(rows["keys"][0].startswith("predicted_masks/"))


if __name__ == "__main__":
    unittest.main()
