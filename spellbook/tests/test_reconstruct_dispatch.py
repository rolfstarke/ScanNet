import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SPELLBOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(SPELLBOOK)

VALID_POLICIES = ("managed", "cpu", "blocked")

EXPECTED = {
    "zed": ("blocked", True),
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
            self.assertTrue(callable(mod.gpu_check), engine)

    def test_all_policies_valid(self):
        import importlib
        for engine in EXPECTED:
            mod = importlib.import_module(f"reconstruct.engines.{engine}")
            self.assertIn(mod.GPU_POLICY, VALID_POLICIES, engine)

    def test_no_gpu0_fallback_in_managed_adapters(self):
        import importlib
        for engine in ("metashape", "rtabmap", "isaac", "bundlefusion"):
            mod = importlib.import_module(f"reconstruct.engines.{engine}")
            src = open(mod.__file__).read()
            self.assertNotIn("gpu or 0", src, engine)
            self.assertNotIn("if gpu is not None else 0", src, engine)

    def test_zed_blocked_before_pyzed(self):
        import importlib
        mod = importlib.import_module("reconstruct.engines.zed")
        reason = mod.preflight()
        self.assertTrue(reason)
        self.assertIn("GPU 0", reason)
        src = open(mod.__file__).read()
        self.assertNotIn("import pyzed", src)

    def test_managed_checks_require_gpu_and_fd(self):
        import importlib
        for engine in ("metashape", "rtabmap", "isaac", "bundlefusion"):
            mod = importlib.import_module(f"reconstruct.engines.{engine}")
            row = mod.gpu_check(None, None, 0)
            self.assertEqual(row.get("status"), "fail", engine)
            self.assertIn("gpu + lease_fd", row.get("reason", ""), engine)

    def test_open3d_cpu_check_takes_no_lease(self):
        import importlib
        mod = importlib.import_module("reconstruct.engines.open3d")
        row = mod.gpu_check(None, None, 0)
        self.assertEqual(row.get("status"), "pass")
        self.assertIsNone(row.get("physical_gpu"))
        self.assertIsNone(row.get("lease_fd"))

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


class ZedBlockedAtCli(unittest.TestCase):
    def test_run_cli_blocks_zed_before_any_work(self):
        r = subprocess.run([sys.executable, "-m", "spellbook.reconstruct.run",
                            "--scene", "9999", "--engine", "zed"],
                           capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("GPU 0", r.stderr)

    def test_gpu_check_cli_needs_no_scene(self):
        r = subprocess.run([sys.executable, os.path.join(SPELLBOOK, "main.py"),
                            "--gpu-check", "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)


class ExtractBlocked(unittest.TestCase):
    def test_replace_raises(self):
        import tempfile
        from reconstruct import extract
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                extract.ensure_frames("/tmp/nope.svo2", d, replace=True)

    def test_incomplete_frames_raise(self):
        import tempfile
        from reconstruct import extract
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                extract.ensure_frames("/tmp/nope.svo2", d)

    def test_complete_frames_reuse_without_pyzed(self):
        import tempfile
        import numpy as np
        from reconstruct import extract
        with tempfile.TemporaryDirectory() as d:
            frames = os.path.join(d, "frames")
            os.makedirs(os.path.join(frames, "color"))
            open(os.path.join(frames, "color", "0.jpg"), "w").close()
            np.savetxt(os.path.join(frames, "intrinsic_depth.txt"), np.eye(4))
            np.save(os.path.join(frames, "camera_to_world.npy"), np.eye(4)[None])
            np.save(os.path.join(frames, "gravity.npy"), np.zeros(3))
            open(os.path.join(frames, "pose_state.txt"), "w").write("OK\n")
            info = extract.ensure_frames("/tmp/nope.svo2", d)
            self.assertEqual(info["svo_frames"], 1)


class OrchestratorValidation(unittest.TestCase):
    def test_validate_rejects_gpu0_and_overlap(self):
        from gpu_check import _validate
        rows = [dict(kind="predict", method="m1", status="pass", policy="managed",
                     physical_gpu=1),
                dict(kind="predict", method="m2", status="pass", policy="managed",
                     physical_gpu=1)]
        problems = _validate(rows, [1, 2, 3, 4], [(1, 0, 10, "m1"), (1, 5, 20, "m2")],
                             [(123, "evil")])
        joined = " | ".join(problems)
        self.assertIn("GPU 0", joined)
        self.assertIn("overlap", joined)

    def test_validate_accepts_clean_report(self):
        from gpu_check import _validate
        rows = [dict(kind="predict", method="m1", status="pass", policy="managed",
                     physical_gpu=1),
                dict(kind="predict", method="m2", status="pass", policy="managed",
                     physical_gpu=2),
                dict(kind="predict", method="m3", status="pass", policy="managed",
                     physical_gpu=3),
                dict(kind="predict", method="m4", status="pass", policy="managed",
                     physical_gpu=4)]
        problems = _validate(rows, [1, 2, 3, 4], [], [])
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()