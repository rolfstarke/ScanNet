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


def job_cgroup_path(pid):
    """Unified cgroup-v2 directory for a live process; None when unusable.

    Read-only: parses the ``0::/...`` entry of ``/proc/<pid>/cgroup`` and
    joins it beneath ``/sys/fs/cgroup``. Rejects v1 layouts, deleted
    cgroups, escapes, and missing processes without raising.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return None
    try:
        with open(f"/proc/{pid_int}/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3 or parts[0] != "0" or parts[1] != "":
            continue
        rel = parts[2]
        if not rel.startswith("/") or ".." in rel.split("/") \
                or "(deleted)" in rel:
            return None
        path = os.path.join("/sys/fs/cgroup", rel.lstrip("/"))
        try:
            if not os.path.isdir(path):
                return None
        except OSError:
            return None
        return path
    return None


def read_cpu_usec(cgroup_path):
    """Cumulative ``cpu.stat:usage_usec`` for a cgroup; None on any failure."""
    if not cgroup_path or not isinstance(cgroup_path, str):
        return None
    try:
        if ".." in cgroup_path.split(os.sep):
            return None
        with open(os.path.join(cgroup_path, "cpu.stat")) as f:
            for line in f:
                if line.startswith("usage_usec"):
                    value = int(line.split()[1])
                    return value if value >= 0 else None
    except (OSError, ValueError, IndexError):
        return None
    return None


def cgroup_pids(cgroup_path, limit=64):
    """Live PIDs in a cgroup, bounded; empty list when unavailable."""
    if not cgroup_path or not isinstance(cgroup_path, str):
        return []
    try:
        with open(os.path.join(cgroup_path, "cgroup.procs")) as f:
            out = []
            for line in f:
                line = line.strip()
                if not line.isdigit():
                    continue
                out.append(int(line))
                if len(out) >= limit:
                    break
            return out
    except (OSError, ValueError):
        return []


def proc_cmdline(pid):
    """Argv of a live process; empty list on any failure."""
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            return [part.decode(errors="replace")
                    for part in f.read().split(b"\0") if part]
    except (OSError, ValueError):
        return []


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
