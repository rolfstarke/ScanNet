"""CPU-only tests for the Mosaic3D wrapper knobs (presets, params parsing, npz v2, rescoring)."""
import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_SPELLBOOK, "predict", "models")
sys.path.insert(0, _SPELLBOOK)


def _wrapper():
    spec = importlib.util.spec_from_file_location(
        "spellbook_mosaic3d_run", os.path.join(_MODELS, "_mosaic3d_run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _wrapper()


def _params(raw):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.json")
        with open(path, "w") as f:
            json.dump(raw, f)
        return M.resolve_params(M.load_overrides(path, M.ALLOWED_PARAMS))


class MosaicWinnerPresetTests(unittest.TestCase):
    def test_empty_is_baseline(self):
        self.assertEqual(M.resolve_params(M.load_overrides(None, M.ALLOWED_PARAMS)), {})
        self.assertEqual(_params({}), {})
        cands = [(np.array([True, False]), "chair", 0.3)]
        self.assertIs(M.rescore(cands, np.zeros((2, 3)), "none"), cands)

    def test_baseline_defaults(self):
        self.assertEqual(M.CHECKPOINT, "/data/mosaic3d/ckpts/sc.ckpt")
        self.assertEqual((M.CONDITION, M.GRID_SIZE, M.INSTANCE_HEAD), ("ScanNet", 0.02, "mask3d"))
        self.assertEqual((M.MASK3D_CONFIDENCE, M.MIN_MASK_POINTS), (0.001, 100))

    def test_presets_expand(self):
        self.assertEqual(_params({"preset": "h3-proxyrescore"}), {"rescore_mode": "geom1"})
        self.assertEqual(_params({"preset": "b2-geom2"}), {"rescore_mode": "geom2"})

    def test_explicit_params_win(self):
        self.assertEqual(
            _params({"preset": "h3-proxyrescore", "rescore_mode": "geom2",
                     "mask3d_confidence": 0.1}),
            {"rescore_mode": "geom2", "mask3d_confidence": 0.1})

    def test_rejections(self):
        for raw in ({"preset": "nope"}, {"rescore_mode": "genuine"},
                    {"prompt_mode": "fancy"}, {"instance_head": "decoder"},
                    {"rescore_mode": "geom1", "instance_head": "dbscan"},
                    {"dedup_iou": 1.5}, {"unknown_key": 1}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _params(raw)

    def test_batch_knobs_accepted(self):
        self.assertEqual(_params({"dedup_iou": 0.5, "prompt_mode": "ensemble_bg"}),
                         {"dedup_iou": 0.5, "prompt_mode": "ensemble_bg"})


class MosaicProposalCacheTests(unittest.TestCase):
    def test_cache_keyed_on_confidence(self):
        a = M.proposal_cache_path("scene0000_00", 0.001)
        b = M.proposal_cache_path("scene0000_00", 0.2)
        self.assertNotEqual(a, b)
        self.assertIn("conf0.001", a)

    def test_npz_v1_and_v2(self):
        masks = np.array([[True, False], [True, True]])
        with tempfile.TemporaryDirectory() as tmp:
            v1 = os.path.join(tmp, "v1.npz")
            np.savez_compressed(v1, masks=masks)
            got, scores = M.load_proposals(v1)
            self.assertTrue((got == masks).all())
            self.assertIsNone(scores)
            with self.assertRaises(ValueError):
                M.load_proposals(v1, need_scores=True)
            v2 = os.path.join(tmp, "v2.npz")
            np.savez_compressed(v2, masks=masks, scores=np.array([0.5, 0.1], np.float32),
                                format=np.int64(M.PROPOSAL_NPZ_FORMAT))
            got, scores = M.load_proposals(v2, need_scores=True)
            self.assertEqual(scores.tolist(), [0.5, np.float32(0.1)])


class MosaicRescoreTests(unittest.TestCase):
    def test_connectivity_ratio(self):
        blob = np.random.default_rng(0).uniform(0, 0.2, (300, 3))
        self.assertEqual(M.connectivity_ratio(blob), 1.0)
        split = np.vstack([blob, blob[:100] + 5.0])
        self.assertAlmostEqual(M.connectivity_ratio(split), 0.75)

    def test_geom_modes_label_blind(self):
        pts = np.random.default_rng(1).uniform(0, 0.2, (400, 3))
        mask = np.ones(400, bool)
        a = M.rescore([(mask, "chair", 0.5)], pts, "geom1")[0][2]
        b = M.rescore([(mask, "lamp", 0.5)], pts, "geom1")[0][2]
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, 0.5)
        self.assertAlmostEqual(M.rescore([(mask, "chair", 0.5)], pts, "geom2")[0][2],
                               0.5 * 400 / (400 + M.GEOM2_HALF_POINTS))

    def test_dedup_same_class_only(self):
        m = np.array([1, 1, 1, 0], bool)
        cands = [(m, "chair", 0.2), (m, "chair", 0.9), (m, "table", 0.1)]
        self.assertEqual([c[2] for c in M.dedup(cands, 0.5)], [0.9, 0.1])


class MosaicIntegrationTests(unittest.TestCase):
    def test_helper_and_checkpoint_paths(self):
        helper = os.path.join(_MODELS, "_mosaic3d_scene.py")
        self.assertTrue(os.path.isfile(helper))
        for rel in (os.path.join("predict", "runner.py"), os.path.join("evaluation", "runs.py")):
            with open(os.path.join(_SPELLBOOK, rel)) as f:
                text = f.read()
            self.assertIn('"_mosaic3d_scene.py"', text)
            self.assertIn('"/data/mosaic3d/ckpts/sc.ckpt"', text)
            self.assertNotIn("spunet34c", text)
            self.assertNotIn("run_custom_scene", text)

    def test_ppt_condition_table(self):
        with open(os.path.join(_MODELS, "_mosaic3d_scene.py")) as f:
            tree = ast.parse(f.read())
        table = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and getattr(n.targets[0], "id", None) == "PPT_CONDITIONS")
        self.assertEqual(table, {1: ["ScanNet"], 3: ["ScanNet", "ARKitScenes", "ScanNetPP"]})


if __name__ == "__main__":
    unittest.main()
