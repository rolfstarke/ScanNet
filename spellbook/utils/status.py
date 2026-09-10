"""Minimal live resource dashboard used by `main.py --status`."""
import os
import sys
import time

ACTIVE_JOB_STATES = ("queued", "starting", "running", "cancelling")
TERMINAL_JOB_STATES = ("succeeded", "failed", "cancelled", "lost")


def dur(seconds):
    """Compact duration: 45s, 12m, 3h04m."""
    s = max(0, int(seconds or 0))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m"
    return f"{m // 60}h{m % 60:02d}m"


def short_branch(branch):
    """debug/prediction-openins3d -> openins3d."""
    if not branch:
        return "-"
    for prefix in ("debug/prediction-", "debug/reconstruction-"):
        if branch.startswith(prefix):
            return branch[len(prefix):]
    return branch[-11:]


def _arg(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _session(owner, row=None):
    if row and row.get("branch"):
        return short_branch(row["branch"])
    name = os.path.basename((owner or {}).get("cwd") or "")
    for prefix in ("prediction-", "reconstruction-"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name or "-"


def _run_name(owner, row=None):
    if row and row.get("run_id"):
        return row["run_id"]
    argv = (owner or {}).get("argv") or []
    run_id = _arg(argv, "--run-id")
    if run_id:
        return run_id
    engine = _arg(argv, "--engine")
    scene = _arg(argv, "--scene")
    if engine:
        return f"{engine}:scene{scene}" if scene else engine
    return _arg(argv, "--models") or "unknown"


def render_snapshot(res, pool, prev_cpu, rows, scannet_root):
    """One dashboard frame as plain text lines."""
    now = time.time()
    lines = []
    cur = res.cpu_times()
    util = res.cpu_util(prev_cpu, cur)
    try:
        ncpu = os.cpu_count() or 0
    except OSError:
        ncpu = 0
    la1, la5, la15 = res.load_average()
    fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else "?"
    upct = f"{util * 100:.0f}%" if util is not None else "?%"
    lines.append(f"cpu {upct} load {fmt(la1)} {fmt(la5)} {fmt(la15)} x{ncpu}")

    stats = res.gpu_stats()
    lock_dir = os.path.join(scannet_root, "derived", "locks", "gpus")
    leases = res.gpu_leases(lock_dir, pool)
    rows_by_id = {row["job_id"]: row for row in rows}
    owner_jobs = set()

    lines.append("GPU UTIL      MEMORY TEMP    TIME SESSION      RUN")
    for i in sorted(set(stats) | set(pool) | {0}):
        st = stats.get(i)
        if st is None:
            lines.append(f"{i:>3}    -           -    -       - -            unavailable")
            continue
        mem = f"{st['mem_used'] / 1024:.1f}/{st['mem_total'] / 1024:.1f}G"
        if i == 0:
            elapsed, session, run = "-", "-", "reserved"
        elif i not in pool:
            elapsed, session, run = "-", "-", "unmanaged"
        elif i in leases:
            owner = res.process_info(leases[i])
            row = rows_by_id.get(owner.get("job_id"))
            if row:
                owner_jobs.add(row["job_id"])
            started = owner.get("started")
            elapsed = dur(now - started) if started else "-"
            session = _session(owner, row)
            run = _run_name(owner, row)
        else:
            elapsed, session = "-", "-"
            active = st["util"] > 5 or st["mem_used"] > 512
            run = "unleased" if active else "free"
        lines.append(f"{i:>3} {st['util']:>3.0f}% {mem:>11} "
                     f"{st['temp']:>3.0f}C {elapsed:>7} "
                     f"{session[:12]:<12} {run}")

    waiting = [row for row in rows
               if row["state"] in ACTIVE_JOB_STATES
               and row["job_id"] not in owner_jobs]
    if waiting:
        lines.append("")
        lines.append("STATE    TIME SESSION      RUN")
    labels = {"queued": "QUEUE", "starting": "START", "running": "WAIT",
              "cancelling": "STOP"}
    for row in waiting:
        started = row.get("started")
        elapsed = dur(now - started) if started else "-"
        session = short_branch(row.get("branch"))
        lines.append(f"{labels[row['state']]:<5} {elapsed:>7} "
                     f"{session[:12]:<12} {row.get('run_id') or row['kind']}")

    recent = sorted(
        (row for row in rows if row["state"] in TERMINAL_JOB_STATES),
        key=lambda row: row.get("ended") or row.get("started")
        or row.get("submitted") or 0,
        reverse=True,
    )[:20]
    if recent:
        lines.append("")
        lines.append("RECENT JOBS")
        lines.append("STATE    TOTAL SESSION      RUN")
    past_labels = {"succeeded": "DONE", "failed": "FAIL",
                   "cancelled": "CANCEL", "lost": "LOST"}
    for row in recent:
        started, ended = row.get("started"), row.get("ended")
        wait = row.get("gpu_wait")
        if isinstance(started, (int, float)) \
                and isinstance(ended, (int, float)) \
                and isinstance(wait, (int, float)):
            total = dur(max(0.0, ended - started - wait))
        else:
            total = "-"
        state = past_labels[row["state"]]
        if state == "FAIL" and row.get("exit_code") not in (None, 0):
            state += str(row["exit_code"])
        session = short_branch(row.get("branch"))
        lines.append(f"{state:<6} {total:>7} {session[:12]:<12} "
                     f"{row.get('run_id') or row['kind']}")
    return lines


def _pool():
    from evaluation.benchmark import load_settings
    try:
        return load_settings()["gpu_pool"]
    except Exception:
        return [1, 2, 3, 4]


def show_once(scannet_root=None):
    """Print one dashboard frame (plain text when redirected)."""
    from utils import resources as res
    from evaluation.benchmark import load_settings
    root = scannet_root or load_settings()["scannet_root"]
    pool = _pool()
    prev = res.cpu_times()
    time.sleep(1.0)
    from jobs import list_jobs
    rows = list_jobs(root)
    sys.stdout.write("\n".join(render_snapshot(res, pool, prev, rows, root)) + "\n")
    sys.stdout.flush()
