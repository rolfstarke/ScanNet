import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SPELLBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(SPELLBOOK)

EXPECTED = {
    "zed": ("zed-default", True),
    "open3d": ("cpu", False),
    "metashape": ("managed", True),
    "rtabmap": ("managed", False),
    "isaac": ("managed", True),
    "bundlefusion": ("managed", False),
}


class EngineMetadata(unittest.TestCase):
    def test_policy_and_serial_table(self):
        import importlib
        for engine, (policy, serial) in EXPECTED.items():
            mod = importlib.import_module(f"reconstruct.engines.{engine}")
            self.assertEqual(mod.GPU_POLICY, policy, engine)
            self.assertEqual(mod.SERIAL, serial, engine)
            self.assertTrue(callable(mod.preflight), engine)
            self.assertTrue(callable(mod.reconstruct), engine)

    def test_no_engine_imports_another_engine(self):
        import importlib
        for engine in EXPECTED:
            mod = importlib.import_module(f"reconstruct.engines.{engine}")
            src = open(mod.__file__).read()
            for other in EXPECTED:
                if other == engine:
                    continue
                self.assertNotIn(f"from . import {other}", src,
                                 f"{engine} -> {other} (relative engine import)")

    def test_managed_engines_forward_lease_fd(self):
        import inspect
        import importlib
        for engine in ("metashape", "rtabmap", "isaac", "bundlefusion"):
            mod = importlib.import_module(f"reconstruct.engines.{engine}")
            sig = inspect.signature(mod.reconstruct)
            self.assertIn("lease_fd", sig.parameters, engine)
            src = open(mod.__file__).read()
            self.assertIn("pass_fds", src, engine)


class CliRejectsManualGpu(unittest.TestCase):
    def test_main_cli_rejects_gpu(self):
        r = subprocess.run([sys.executable, os.path.join(SPELLBOOK, "main.py"),
                            "--scene", "9004", "--gpu", "1"],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0, "main.py must not accept --gpu")
        self.assertIn("unrecognized arguments", r.stderr)

    def test_run_cli_rejects_gpu(self):
        r = subprocess.run([sys.executable, "-m", "spellbook.reconstruct.run",
                            "--scene", "9004", "--engine", "zed", "--gpu", "1"],
                           capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertNotEqual(r.returncode, 0, "reconstruct.run must not accept --gpu")
        self.assertIn("unrecognized arguments", r.stderr)


if __name__ == "__main__":
    unittest.main()
