"""CPU-only tests for the OpenYOLO3D wrapper run parameters (#35) and intrinsics (#2).
[inferred] -- the original 23 tests were lost; rewritten from the #35 batch-1 description."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_SPELLBOOK, "predict", "models")


def _wrapper():
    spec = importlib.util.spec_from_file_location(
        "spellbook_openyolo3d_run", os.path.join(_MODELS, "_openyolo3d_run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _wrapper()

OFFICIAL_CFG = {
    "openyolo3d": {"frequency": 10, "vis_depth_threshold": 0.05, "depth_scale": 1000.0,
                   "topk": 40, "topk_per_image": 600},
    "network2d": {"text_prompts": ["chair"], "topk": 100, "th": 0.08, "nms": 0.3},
    "network3d": {"th": 0.04, "nms": 0.6, "is_gt": False},
}


def _params(raw):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.json")
        with open(path, "w") as f:
            json.dump(raw, f)
        return M.load_overrides(path, M.ALLOWED_PARAMS)


class _Arr:
    def __init__(self, shape):
        self.shape = shape


class ParameterTests(unittest.TestCase):
    def test_defaults_are_official(self):
        self.assertEqual((M.PROPOSAL_SOURCE, M.VOTE_WEIGHTING, M.PROPOSAL_SCORE_POWER),
                         ("live", "count", 0.0))
        self.assertEqual(M.MIN_MASK_POINTS, 20)
        self.assertEqual(M.PRECOMPUTED_MASKS_DIR,
                         "/data/openyolo3D/output/scannet200/scannet200_masks")

    def test_allowed_params(self):
        self.assertEqual(M.ALLOWED_PARAMS, {
            "min_mask_points", "proposal_source", "frequency", "network2d_th",
            "network3d_nms", "vote_weighting", "proposal_score_power"})

    def test_empty_params(self):
        self.assertEqual(M.load_overrides(None, M.ALLOWED_PARAMS), {})
        self.assertEqual(_params({}), {})

    def test_winner_params_accepted(self):
        self.assertEqual(_params({"proposal_source": "official_precomputed"}),
                         {"proposal_source": "official_precomputed"})

    def test_unknown_param_rejected(self):
        with self.assertRaises(ValueError):
            _params({"use_precomputed": True})

    def test_bad_enums_rejected(self):
        for raw in ({"proposal_source": "precomputed"}, {"vote_weighting": "score"}):
            with self.assertRaises(ValueError):
                _params(raw)

    def test_frequency_bounds(self):
        self.assertEqual(_params({"frequency": 5}), {"frequency": 5})
        for bad in (0, 5.5, True):
            with self.assertRaises(ValueError):
                _params({"frequency": bad})

    def test_threshold_bounds(self):
        self.assertEqual(_params({"network2d_th": 0.15, "network3d_nms": 0.45}),
                         {"network2d_th": 0.15, "network3d_nms": 0.45})
        for raw in ({"network2d_th": 1.5}, {"network3d_nms": -0.1},
                    {"proposal_score_power": -1.0}):
            with self.assertRaises(ValueError):
                _params(raw)


class ConfigTests(unittest.TestCase):
    def test_empty_overrides_leave_config_identical(self):
        cfg = copy.deepcopy(OFFICIAL_CFG)
        M.apply_config_overrides(cfg, {})
        self.assertEqual(cfg, OFFICIAL_CFG)

    def test_winner_leaves_config_identical(self):
        cfg = copy.deepcopy(OFFICIAL_CFG)
        M.apply_config_overrides(cfg, {"proposal_source": "official_precomputed"})
        self.assertEqual(cfg, OFFICIAL_CFG)

    def test_config_knobs_written(self):
        cfg = M.apply_config_overrides(copy.deepcopy(OFFICIAL_CFG), {
            "frequency": 5, "network2d_th": 0.15, "network3d_nms": 0.45})
        self.assertEqual(cfg["openyolo3d"]["frequency"], 5)
        self.assertEqual(cfg["network2d"]["th"], 0.15)
        self.assertEqual(cfg["network3d"]["nms"], 0.45)
        self.assertEqual(cfg["network2d"]["nms"], 0.3)

    def test_custom_voting_only_off_default(self):
        self.assertFalse(M.needs_custom_voting("count", 0.0))
        self.assertTrue(M.needs_custom_voting("detector_confidence", 0.0))
        self.assertTrue(M.needs_custom_voting("count", 1.0))


class PrecomputedTests(unittest.TestCase):
    def _ply(self, tmp, n):
        path = os.path.join(tmp, "scene0046_00_vh_clean_2.ply")
        with open(path, "wb") as f:
            f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                    "property float x\nelement face 0\nend_header\n".encode())
        return path

    def test_ply_vertex_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(M.ply_vertex_count(self._ply(tmp, 186857)), 186857)

    def test_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                M.validate_precomputed(tmp, "scene0046_00", 10, loader=lambda p: None)

    def test_row_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "scene0046_00.pt"), "wb").close()
            with self.assertRaises(ValueError):
                M.validate_precomputed(tmp, "scene0046_00", 10,
                                       loader=lambda p: (_Arr((9, 4)), _Arr((4,))))

    def test_non_tuple_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "scene0046_00.pt"), "wb").close()
            with self.assertRaises(ValueError):
                M.validate_precomputed(tmp, "scene0046_00", 10, loader=lambda p: _Arr((10, 4)))

    def test_valid_file_returns_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "scene0046_00.pt"), "wb") as f:
                f.write(b"x")
            path, sha = M.validate_precomputed(tmp, "scene0046_00", 10,
                                               loader=lambda p: (_Arr((10, 4)), _Arr((4,))))
            self.assertEqual(path, os.path.join(tmp, "scene0046_00.pt"))
            self.assertEqual(sha, "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881")


class VotingTests(unittest.TestCase):
    def test_count_matches_upstream_mode(self):
        labels = np.array([2, 2, 1, -1, 2, 1])
        dist, label, support = M.vote_distribution(labels, np.ones(len(labels)), 4)
        self.assertEqual(label, 2)
        self.assertEqual(support, 3.0)
        np.testing.assert_allclose(dist, [0.0, 2 / 3, 1.0, 0.0])

    def test_detector_confidence_reweights(self):
        labels = np.array([2, 2, 2, 1])
        weights = np.array([0.1, 0.1, 0.1, 0.9])
        _, label, _ = M.vote_distribution(labels, weights, 4)
        self.assertEqual(label, 1)

    def test_no_votes_is_background(self):
        dist, label, support = M.vote_distribution(np.array([-1, -1]), np.ones(2), 4)
        self.assertEqual((label, support), (3, 0.0))
        np.testing.assert_allclose(dist, [0, 0, 0, 1])

    def test_proposal_scale(self):
        scores = np.array([0.5, 0.25, -0.1])
        np.testing.assert_allclose(M.proposal_scale(scores, 0.0), [1, 1, 1])
        np.testing.assert_allclose(M.proposal_scale(scores, 1.0), [0.5, 0.25, 0.0])

    def test_flatten_topk(self):
        dists = np.array([[0.2, 1.0, 0.0], [1.0, 0.5, 0.0]])
        mask_idx, cls_idx, scores = M.flatten_topk(dists, 3)
        self.assertEqual(list(mask_idx), [0, 1, 1])
        self.assertEqual(list(cls_idx), [1, 0, 1])
        np.testing.assert_allclose(scores, [1.0, 1.0, 0.5])


class ScratchSceneTests(unittest.TestCase):
    def test_intrinsics_link_is_color(self):  # #2
        with tempfile.TemporaryDirectory() as tmp:
            scratch = M._build_scratch_scene("/frames", "/scan/s.ply", "scene0046_00", "r",
                                             scratch_root=tmp)
            self.assertEqual(os.readlink(os.path.join(scratch, "intrinsics.txt")),
                             "/frames/intrinsic_color.txt")
            self.assertEqual(os.readlink(os.path.join(scratch, "poses")), "/frames/pose")
            self.assertEqual(os.readlink(os.path.join(scratch, "scene0046_00.ply")), "/scan/s.ply")


class CudnnRetryTests(unittest.TestCase):
    class _Module:
        def __init__(self, errors):
            self.errors, self.calls = list(errors), 0

        def forward(self, x):
            self.calls += 1
            if self.errors:
                raise self.errors.pop(0)
            return x + 1

    def _wrap(self, errors):
        module, released = self._Module(errors), []
        _wrapper().retry_after_cudnn_failure(module, lambda: released.append(True))
        return module, released

    def test_no_failure_runs_once_without_release(self):
        module, released = self._wrap([])
        self.assertEqual(module.forward(1), 2)
        self.assertEqual((module.calls, released), (1, []))

    def test_cudnn_failure_retries_after_release(self):
        module, released = self._wrap([RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")])
        self.assertEqual(module.forward(1), 2)
        self.assertEqual((module.calls, released), (2, [True]))

    def test_other_errors_propagate(self):
        module, released = self._wrap([RuntimeError("shape mismatch")])
        with self.assertRaises(RuntimeError):
            module.forward(1)
        self.assertEqual((module.calls, released), (1, []))

    def test_second_cudnn_failure_propagates(self):
        err = RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")
        module, _ = self._wrap([err, err])
        with self.assertRaises(RuntimeError):
            module.forward(1)
        self.assertEqual(module.calls, 2)


if __name__ == "__main__":
    unittest.main()
