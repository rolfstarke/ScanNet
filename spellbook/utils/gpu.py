"""Cross-process GPU leasing for managed workloads.

One persistent lock file per configured physical GPU (gpu-<index>.lock) under
<scannet_root>/derived/locks/gpus/. GPU 0 is NEVER managed here: ZED SDK
extraction/tracking stays on the default device (SDK 5.4 bug #18) and no gpu-0.lock
file is ever created. Locks are advisory fcntl.flock leases; the kernel releases them
automatically when the last holder descriptor closes, including on crash/SIGKILL.

The lease descriptor can be forwarded to GPU children via subprocess pass_fds so a
GPU job survives its orchestration parent; the child must keep the descriptor open
for the whole workload.
"""
import fcntl
import os
import subprocess
import time
from contextlib import contextmanager

LOCK_ROOT = os.path.join("derived", "locks", "gpus")
POLL_SECONDS = 10


def _present_gpus():
    """Physical GPU indices from nvidia-smi -L; empty set on failure.

    Runs with CUDA_VISIBLE_DEVICES removed so discovery sees all physical devices.
    Tests patch this function instead of touching real hardware.
    """
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                             env=env, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if out.returncode != 0:
        return set()
    return set(range(len([l for l in out.stdout.splitlines() if l.strip()])))


class Lease:
    """A held GPU reservation. `.index` is the physical GPU index; `fileno()` is the
    lock descriptor for subprocess pass_fds forwarding."""

    def __init__(self, fd, index):
        self._fd = fd
        self.index = index

    def fileno(self):
        return self._fd

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


@contextmanager
def gpu_lease(gpu_pool, scannet_root):
    """Context manager acquiring the first free GPU in `gpu_pool` (settings order).

    Waits (polling) until one of the configured indices is free. Yields a Lease;
    the lock is released when the context exits. Raises RuntimeError before waiting
    if a configured index is not present on this host.
    """
    present = _present_gpus()
    missing = [g for g in gpu_pool if g not in present]
    if missing:
        raise RuntimeError(
            f"configured gpu_pool {gpu_pool} missing physical GPU(s) {missing} "
            f"(found {sorted(present)})")
    lock_dir = os.path.join(scannet_root, LOCK_ROOT)
    os.makedirs(lock_dir, exist_ok=True)
    waited = False
    while True:
        for g in gpu_pool:
            fd = os.open(os.path.join(lock_dir, f"gpu-{g}.lock"),
                         os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            if waited:
                print(f"[gpu] acquired lease for GPU {g} (after waiting)")
            else:
                print(f"[gpu] acquired lease for GPU {g}")
            lease = Lease(fd, g)
            try:
                yield lease
            finally:
                lease.close()
            return
        else:
            if not waited:
                print(f"[gpu] pool {gpu_pool} busy; waiting for a free GPU")
                waited = True
            time.sleep(POLL_SECONDS)
