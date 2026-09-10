"""Read-only host resource snapshots for the status dashboard."""
import os
import re
import subprocess


def cpu_times():
    """Total/idle jiffies from /proc/stat; None on failure."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        if not parts or parts[0] != "cpu":
            return None
        nums = [int(x) for x in parts[1:]]
        return {"total": sum(nums), "idle": nums[3] + nums[4]}
    except (OSError, ValueError, IndexError):
        return None


def cpu_util(prev, cur):
    """Fraction 0..1 of non-idle time between two cpu_times() snapshots."""
    if not prev or not cur:
        return None
    d_total = cur["total"] - prev["total"]
    d_idle = cur["idle"] - prev["idle"]
    if d_total <= 0:
        return None
    return max(0.0, min(1.0, (d_total - d_idle) / d_total))


def load_average():
    """(1, 5, 15) minute load averages; Nones on failure."""
    try:
        return os.getloadavg()
    except OSError:
        return (None, None, None)


def _nvidia_env():
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def gpu_stats():
    """{index: {util, mem_used, mem_total, temp}}; empty on failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, env=_nvidia_env(), timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if out.returncode != 0:
        return {}
    stats = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        try:
            stats[int(parts[0])] = {
                "util": float(parts[1]),
                "mem_used": float(parts[2]),
                "mem_total": float(parts[3]),
                "temp": float(parts[4]),
            }
        except ValueError:
            continue
    return stats


def _lock_owner_pids(paths):
    """Return GPU lock owners from `/proc/locks`, verified through their fds."""
    by_inode = {}
    for gpu, path in paths.items():
        try:
            by_inode.setdefault(os.stat(path).st_ino, []).append((gpu, path))
        except OSError:
            pass
    owners = {}
    try:
        with open("/proc/locks") as f:
            records = f.readlines()
    except OSError:
        return owners
    for record in records:
        fields = record.split()
        if len(fields) < 6 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        try:
            pid = int(fields[4])
            inode = int(fields[5].rsplit(":", 1)[1])
        except (ValueError, IndexError):
            continue
        candidates = by_inode.get(inode, ())
        if not candidates:
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            for gpu, _ in candidates:
                owners.setdefault(gpu, None)
            continue
        for gpu, path in candidates:
            for fd in fds:
                try:
                    if os.path.samefile(path, f"/proc/{pid}/fd/{fd}"):
                        owners[gpu] = pid
                        break
                except OSError:
                    continue
    return owners


def gpu_leases(lock_dir, pool):
    """Held managed leases as `{gpu: owner_pid_or_None}`.

    Missing and unlocked files are omitted. No lock file is created or changed.
    """
    paths = {}
    for g in pool:
        path = os.path.join(lock_dir, f"gpu-{g}.lock")
        if os.path.isfile(path):
            paths[g] = path
    return _lock_owner_pids(paths)


def process_info(pid):
    """Command, session location, job id, and start time for a live process."""
    if not pid:
        return {}
    proc = f"/proc/{pid}"
    try:
        with open(os.path.join(proc, "cmdline"), "rb") as f:
            argv = [part.decode(errors="replace")
                    for part in f.read().split(b"\0") if part]
    except OSError:
        return {}
    try:
        cwd = os.readlink(os.path.join(proc, "cwd"))
    except OSError:
        cwd = None
    try:
        started = os.stat(proc).st_ctime
    except OSError:
        started = None
    try:
        with open(os.path.join(proc, "cgroup")) as f:
            cgroup = f.read()
    except OSError:
        cgroup = ""
    match = re.search(r"spellbook-job-([A-Za-z0-9_-]+)\.service", cgroup)
    return {
        "pid": pid,
        "argv": argv,
        "cwd": cwd,
        "started": started,
        "job_id": match.group(1) if match else None,
    }
