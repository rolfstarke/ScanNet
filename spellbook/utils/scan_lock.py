"""Cross-process flock locks for reconstruction frames, scan artifacts, and prediction indexes.

Lock files persist under /data/scannet/derived/locks/; the kernel releases ownership on process
death. Global acquisition order:

1. canonical reconstruction-frame lock
2. scene-engine allocation lock
3. per-scan artifact lock
4. global prediction-index lock (always exclusive)
5. GPU lease (never while holding the prediction-index lock)

Scan locks must be acquired before GPU leases.
"""
import fcntl
import os
from contextlib import contextmanager

from evaluation.benchmark import load_settings

LOCK_ROOT_REL = os.path.join("derived", "locks")


def _root(scannet_root=None):
    return scannet_root or load_settings()["scannet_root"]


def frames_lock_path(scene_num, scannet_root=None):
    return os.path.join(_root(scannet_root), LOCK_ROOT_REL, "reconstruction", "frames",
                        f"scene{int(scene_num):04d}.lock")


def scene_engine_lock_path(scene_num, engine, scannet_root=None):
    return os.path.join(_root(scannet_root), LOCK_ROOT_REL, "reconstruction",
                        f"scene{int(scene_num):04d}_{engine}.lock")


def scan_lock_path(scene_id, scannet_root=None):
    return os.path.join(_root(scannet_root), LOCK_ROOT_REL, "scans", f"{scene_id}.lock")


def prediction_index_lock_path(scannet_root=None):
    return os.path.join(_root(scannet_root), LOCK_ROOT_REL, "prediction-artifacts.lock")


class ScanLock:
    """Held flock. Supports exclusive->shared downgrade on the same descriptor."""

    def __init__(self, path, exclusive=True):
        self.path = path
        self.exclusive = exclusive
        self._fd = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        flag = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        fcntl.flock(self._fd, flag)
        return self

    def downgrade_to_shared(self):
        if self._fd is None:
            raise RuntimeError("lock not held")
        fcntl.flock(self._fd, fcntl.LOCK_SH)
        self.exclusive = False

    def release(self):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def fileno(self):
        return self._fd

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


@contextmanager
def exclusive_lock(path):
    lock = ScanLock(path, exclusive=True)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


@contextmanager
def shared_lock(path):
    lock = ScanLock(path, exclusive=False)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
