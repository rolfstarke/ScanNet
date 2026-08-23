import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from benchmark import load_settings, validate_gpu_pool  # noqa: E402
from utils import gpu as gpu_mod  # noqa: E402

CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gpu_lease_child.py")


class ValidatePool(unittest.TestCase):
    def test_settings_default_pool(self):
        self.assertEqual(load_settings()["gpu_pool"], [1, 2, 3, 4])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            validate_gpu_pool([])

    def test_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            validate_gpu_pool([1, 2, 2])

    def test_rejects_zero(self):
        with self.assertRaises(ValueError):
            validate_gpu_pool([0, 1, 2, 3, 4])

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            validate_gpu_pool([-1, 2])

    def test_rejects_non_integer(self):
        with self.assertRaises(ValueError):
            validate_gpu_pool([1, 2.5])


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="gpu-test-")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        gpu_mod._present_gpus = lambda: {1, 2, 3, 4}

    def _spawn(self, pool, hold, detach=False):
        args = [sys.executable, CHILD, self.root, ",".join(map(str, pool)), str(hold)]
        if detach:
            args.append("--detach")
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)

    def test_no_gpu0_lock_created(self):
        with gpu_mod.gpu_lease([1], self.root) as lease:
            self.assertEqual(lease.index, 1)
            self.assertTrue(os.path.exists(os.path.join(
                self.root, "derived", "locks", "gpus", "gpu-1.lock")))
            self.assertFalse(os.path.exists(os.path.join(
                self.root, "derived", "locks", "gpus", "gpu-0.lock")))

    def test_same_gpu_exclusive(self):
        with gpu_mod.gpu_lease([1], self.root):
            child = self._spawn([1], 20)
            time.sleep(1.5)
            self.assertIsNone(child.poll(), "second holder must not acquire the same GPU")
            child.kill()
            child.wait()

    def _last_line(self, text):
        return text.strip().splitlines()[-1]

    def test_different_gpus_concurrent(self):
        with gpu_mod.gpu_lease([1], self.root):
            child = self._spawn([2], 1)
            out, _ = child.communicate(timeout=10)
            self.assertEqual(self._last_line(out), "2")

    def test_killed_holder_releases(self):
        holder = self._spawn([1], 60)
        time.sleep(1.5)
        waiter = self._spawn([1], 2)
        time.sleep(1.5)
        self.assertIsNone(waiter.poll())
        holder.kill()
        holder.wait()
        out, _ = waiter.communicate(timeout=15)
        self.assertEqual(self._last_line(out), "1")

    def test_parent_kill_child_holds_blocks_waiter(self):
        holder = self._spawn([1], 25, detach=True)
        out, _ = holder.communicate(timeout=10)
        self.assertEqual(self._last_line(out), "1")
        waiter = self._spawn([1], 2)
        time.sleep(1.5)
        self.assertIsNone(waiter.poll(), "lease must outlive the dead parent via the fd")
        subprocess.run(["pkill", "-f", f"sleep {int(25)}"], capture_output=True)
        out, _ = waiter.communicate(timeout=15)
        self.assertEqual(self._last_line(out), "1")

    def test_five_holders_cover_pool_and_fifth_waits(self):
        """4 concurrent holders must take distinct GPUs 1-4; the 5th must wait and
        reuse one after a holder exits (pool [1,2,3,4], never GPU 0)."""
        pool = [1, 2, 3, 4]
        children = [self._spawn(pool, 3) for _ in range(4)]
        time.sleep(1.5)
        waiter = self._spawn(pool, 1)
        time.sleep(1.5)
        self.assertIsNone(waiter.poll(), "5th holder must wait while the pool is full")
        outs = [c.communicate(timeout=20)[0] for c in children]
        assigned = {self._last_line(o) for o in outs}
        self.assertEqual(assigned, {str(g) for g in pool})
        out, _ = waiter.communicate(timeout=20)
        self.assertIn(self._last_line(out), {str(g) for g in pool})


if __name__ == "__main__":
    unittest.main()
