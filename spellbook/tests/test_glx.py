import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import glx  # noqa: E402

SOFTWARE = "direct rendering: Yes\nOpenGL renderer string: llvmpipe (LLVM 20, 256 bits)\n"
NVIDIA = "direct rendering: Yes\nOpenGL renderer string: NVIDIA RTX A4000/PCIe/SSE2\n"
INDIRECT = "direct rendering: No\nOpenGL renderer string: NVIDIA RTX A4000/PCIe/SSE2\n"


def _fake(default, nvidia):
    def run(cmd, env, **kwargs):
        out = nvidia if env.get(glx._VENDOR_ENV) == "nvidia" else default
        if isinstance(out, Exception):
            raise out
        return subprocess.CompletedProcess(cmd, 0, stdout=out)
    return run


class EnsureNvidiaGlxTests(unittest.TestCase):
    def _call(self, default, nvidia, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=False):
            if not env:
                os.environ.pop(glx._VENDOR_ENV, None)
            with mock.patch.object(glx.subprocess, "run", _fake(default, nvidia)):
                return glx.ensure_nvidia_glx(), os.environ.get(glx._VENDOR_ENV)

    def test_switches_from_software_when_trial_proves_nvidia(self):
        self.assertEqual(self._call(SOFTWARE, NVIDIA), (True, "nvidia"))

    def test_keeps_hardware_default(self):
        self.assertEqual(self._call(NVIDIA, NVIDIA), (False, None))

    def test_keeps_software_when_trial_is_indirect(self):
        self.assertEqual(self._call(SOFTWARE, INDIRECT), (False, None))

    def test_keeps_software_when_trial_times_out(self):
        self.assertEqual(self._call(SOFTWARE, subprocess.TimeoutExpired("glxinfo", 3)), (False, None))

    def test_no_glxinfo_changes_nothing(self):
        self.assertEqual(self._call(FileNotFoundError(), NVIDIA), (False, None))

    def test_explicit_vendor_is_respected(self):
        self.assertEqual(self._call(SOFTWARE, NVIDIA, {glx._VENDOR_ENV: "mesa"}), (False, "mesa"))


if __name__ == "__main__":
    unittest.main()
