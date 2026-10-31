"""CPU-only tests for the OpenMask3D wrapper: official segment pooling, fail-closed segment
checks, the upstream `_masks.pt` feature-naming contract, and official defaults (#46).
[inferred] -- the original tests were lost; rewritten from the #46 attempt-6 description."""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_SPELLBOOK, "predict", "models")


def _wrapper():
    # The spellbook env has no torch/clip; the tested helpers are pure numpy.
    stubs = [name for name in ("torch", "clip")
             if name not in sys.modules and importlib.util.find_spec(name) is None]
    for name in stubs:
        sys.modules[name] = types.ModuleType(name)
    try:
        spec = importlib.util.spec_from_file_location(
            "spellbook_openmask3d_run", os.path.join(_MODELS, "_openmask3d_run.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name in stubs:  # do not leak fake modules into other test files
            sys.modules.pop(name, None)
    return module


M = _wrapper()


def _scatter_mean_reference(masks, seg):
    """Literal per-segment loop of trainer.py:get_full_res_mask."""
    out = np.zeros_like(masks, dtype=np.float32)
    for s in np.unique(seg):
        sel = seg == s
        out[sel] = (masks[sel].mean(axis=0) > 0.5)
    return out


class DefaultsTests(unittest.TestCase):
    def test_official_defaults(self):
        self.assertEqual((M.NUM_QUERIES, M.DBSCAN_MIN_POINTS, M.FREQUENCY), (150, 1, 10))
        self.assertEqual(M.CLIP_PROMPT, "a {} in a scene")
        self.assertEqual((M.DEDUP_IOU, M.MIN_MASK_POINTS, M.POINT_LIMIT), (0.5, 1, 500_000))

    def test_pooled_name_keeps_masks_suffix(self):
        self.assertTrue(M.POOLED_MASKS_NAME.endswith("_masks.pt"))


class FeaturePathTests(unittest.TestCase):
    def test_upstream_strip_last_nine(self):
        self.assertEqual(M.features_path("/s", "/s/working_scene_segsnap_masks.pt"),
                         "/s/working_scene_segsnap_openmask3d_features.npy")

    def test_derived_from_masks_not_scene(self):
        self.assertNotEqual(M.features_path("/s", "/s/working_scene_segsnap_masks.pt"),
                            "/s/working_scene_openmask3d_features.npy")

    def test_bad_suffix_rejected(self):
        with self.assertRaises(ValueError):
            M.features_path("/s", "/s/working_scene_segsnap.pt")


class SegmentFileTests(unittest.TestCase):
    def _segs(self, tmp, ids):
        scan = os.path.join(tmp, "scene0535_00")
        os.makedirs(scan)
        with open(os.path.join(scan, "scene0535_00_vh_clean_2.0.010000.segs.json"), "w") as f:
            json.dump({"segIndices": ids}, f)
        return os.path.join(scan, "scene0535_00_vh_clean_2.ply")

    def test_segments_path_is_sibling(self):
        self.assertEqual(M.segments_path("/scans/scene0535_00/scene0535_00_vh_clean_2.ply"),
                         "/scans/scene0535_00/scene0535_00_vh_clean_2.0.010000.segs.json")

    def test_load_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            ply = self._segs(tmp, [7, 7, 3])
            np.testing.assert_array_equal(M.load_segments(M.segments_path(ply), 3), [7, 7, 3])

    def test_missing_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            M.load_segments("/nonexistent/x.segs.json", 3)

    def test_length_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ply = self._segs(tmp, [7, 7, 3])
            with self.assertRaises(ValueError):
                M.load_segments(M.segments_path(ply), 4)

    def test_negative_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ply = self._segs(tmp, [0, -1])
            with self.assertRaises(ValueError):
                M.load_segments(M.segments_path(ply), 2)


class SegmentPoolTests(unittest.TestCase):
    def test_majority_snaps(self):
        seg = np.array([0, 0, 0, 1, 1])
        masks = np.array([[1], [1], [0], [1], [0]], dtype=np.float16)
        np.testing.assert_array_equal(M.segment_pool(masks, seg)[:, 0], [1, 1, 1, 0, 0])

    def test_exact_half_is_dropped(self):
        seg = np.array([5, 5])
        masks = np.array([[1.0], [0.0]])
        np.testing.assert_array_equal(M.segment_pool(masks, seg), [[0.0], [0.0]])

    def test_matches_scatter_mean_reference(self):
        rng = np.random.default_rng(0)
        seg = rng.integers(0, 40, size=500) * 3  # sparse, unordered ids
        masks = (rng.random((500, 12)) > 0.6).astype(np.float16)
        np.testing.assert_array_equal(M.segment_pool(masks, seg),
                                      _scatter_mean_reference(masks.astype(np.float32), seg))

    def test_pooled_is_segment_constant(self):
        rng = np.random.default_rng(1)
        seg = rng.integers(0, 30, size=300)
        masks = (rng.random((300, 8)) > 0.5).astype(np.float32)
        self.assertGreater(M.segment_violations(masks, seg), 0)
        self.assertEqual(M.segment_violations(M.segment_pool(masks, seg), seg), 0)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            M.segment_pool(np.zeros((4, 2)), np.zeros(5, dtype=np.int64))

    def test_idempotent(self):
        rng = np.random.default_rng(2)
        seg = rng.integers(0, 20, size=200)
        once = M.segment_pool((rng.random((200, 5)) > 0.5).astype(np.float32), seg)
        np.testing.assert_array_equal(M.segment_pool(once, seg), once)


if __name__ == "__main__":
    unittest.main()
