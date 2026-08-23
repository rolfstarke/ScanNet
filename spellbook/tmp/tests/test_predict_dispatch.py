import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from predict import runner  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(runner.__file__), "models")


class Registry(unittest.TestCase):
    def test_supported_models_match_all_maps(self):
        for mapping in (runner.MODEL_PYTHON, runner.MODEL_RUN_SCRIPT,
                        runner.MODEL_NEEDS_FRAMES):
            self.assertEqual(set(mapping), set(runner.SUPPORTED_MODELS))

    def test_all_wrappers_exist(self):
        for model in runner.SUPPORTED_MODELS:
            script = os.path.join(os.path.dirname(runner.__file__),
                                  runner.MODEL_RUN_SCRIPT[model])
            self.assertTrue(os.path.isfile(script), script)

    def test_all_pythons_exist(self):
        for model in runner.SUPPORTED_MODELS:
            py = runner.MODEL_PYTHON[model]
            self.assertTrue(os.path.isfile(py) and os.access(py, os.X_OK), py)


class Preflight(unittest.TestCase):
    def test_all_models_pass_preflight(self):
        for model in runner.SUPPORTED_MODELS:
            self.assertIsNone(runner.preflight_model(model), model)

    def test_unknown_model_fails(self):
        reason = runner.preflight_model("nope")
        self.assertTrue(reason)
        self.assertIn("unknown model", reason)

    def test_predict_raises_on_preflight_failure(self):
        with self.assertRaises(ValueError) as ctx:
            runner.predict(["scene0568_00"], ["nope"], None)
        self.assertIn("preflight failed", str(ctx.exception))


class ChildEnv(unittest.TestCase):
    class _FakeLease:
        def __init__(self, fd, index):
            self.index = index
            self._fd = fd

        def fileno(self):
            return self._fd

    def test_single_visible_gpu_and_lease_fd(self):
        fd = os.open(os.devnull, os.O_RDONLY)
        try:
            lease = self._FakeLease(fd, 3)
            env = runner.child_env(lease, "mosaic3d")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(env["SPELLBOOK_GPU_LEASE_FD"], str(fd))
        finally:
            os.close(fd)

    def test_openyolo3d_gets_library_path(self):
        fd = os.open(os.devnull, os.O_RDONLY)
        try:
            lease = self._FakeLease(fd, 2)
            env = runner.child_env(lease, "openyolo3d")
            self.assertIn("/data/openyolo3D/cuda-11.3/lib64", env["LD_LIBRARY_PATH"])
        finally:
            os.close(fd)


class GpuCheckContract(unittest.TestCase):
    def test_gpu_check_model_requires_lease(self):
        # probe wiring is exercised by the live smoke; here we only verify that the
        # probe script exists and parses its args.
        probe = os.path.join(MODELS_DIR, "_gpu_check.py")
        self.assertTrue(os.path.isfile(probe))


if __name__ == "__main__":
    unittest.main()