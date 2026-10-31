"""CPU-only tests for the OpenIns3D wrapper: precomputed-mask fail-closed loading, the 200-way
Lookup vocabulary, parameter validation, and the removed up-axis heuristic (#45, #34).
[inferred] -- the original tests were lost; rewritten from the #45 16-scene baseline comment."""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_SPELLBOOK, "predict", "models")


def _wrapper():
    # The spellbook env has no torch/open3d; the tested helpers do not touch them at import.
    stubs = [name for name in ("torch", "open3d")
             if name not in sys.modules and importlib.util.find_spec(name) is None]
    for name in stubs:
        sys.modules[name] = types.ModuleType(name)
    try:
        spec = importlib.util.spec_from_file_location(
            "spellbook_openins3d_run", os.path.join(_MODELS, "_openins3d_run.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name in stubs:  # do not leak fake modules into other test files
            sys.modules.pop(name, None)
    return module


M = _wrapper()


class _Masks:
    def __init__(self, shape, sparse=False):
        self.shape = shape
        self.ndim = len(shape)
        self.is_sparse = sparse

    def to_dense(self):
        return _Masks(self.shape)


class PrecomputedMasksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with open(os.path.join(self.tmp.name, "scene0000_00.pt"), "w"):
            pass

    def _load(self, masks, n_vertices):
        saved = M.torch
        M.torch = types.SimpleNamespace(load=lambda *a, **k: masks)
        self.addCleanup(setattr, M, "torch", saved)
        return M._load_precomputed_masks("scene0000_00", n_vertices, self.tmp.name)

    def test_sparse_masks_are_densified(self):
        self.assertFalse(self._load(_Masks((5, 2), sparse=True), 5).is_sparse)

    def test_vertex_count_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            self._load(_Masks((5, 2)), 6)

    def test_missing_file_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            M._load_precomputed_masks("scene9999_00", 5, self.tmp.name)


class VocabTest(unittest.TestCase):
    def test_scannet200_uses_official_200_labels(self):
        vocab = M._lookup_vocab(types.SimpleNamespace(name="ScanNet200"), ["chair"])
        self.assertEqual(len(vocab), 200)
        self.assertEqual(vocab[:3], ["wall", "chair", "floor"])

    def test_other_benchmark_keeps_its_classes(self):
        vocab = M._lookup_vocab(types.SimpleNamespace(name="ScanNet20"), ["chair", "bed"])
        self.assertEqual(vocab, ["chair", "bed"])


class ParametersTest(unittest.TestCase):
    ALLOWED = {"use_precomputed", "detector"}

    def _overrides(self, data):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
            json.dump(data, fh)
            fh.flush()
            return M.load_overrides(fh.name, self.ALLOWED)

    def test_best_run_parameters_accepted(self):
        params = {"detector": "yoloworld", "use_precomputed": True}
        self.assertEqual(self._overrides(params), params)

    def test_use_precomputed_must_be_boolean(self):
        with self.assertRaises(ValueError):
            self._overrides({"use_precomputed": "true"})

    def test_losing_run_parameters_are_not_wired(self):
        source = open(os.path.join(_MODELS, "_openins3d_run.py")).read()
        for name in ("\"lookup_vocab\"", "single_detection", "_ensure_z_up"):
            self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
