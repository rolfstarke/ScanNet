"""--gpu-check orchestrator: distribution smoke for every prediction method and
reconstruction engine, producing no results.

All runnable methods launch their native environment/container under the real
settings-backed GPU lease (pool 1-4, physical GPU 0 user-reserved and never touched),
prove the assigned GPU mapping, hold briefly to expose contention, then exit. ZED
reports BLOCKED (SDK default-device path would use GPU 0, #18) and Isaac reports
BLOCKED while its image is missing (#22). See spellbook/main.py --gpu-check.
"""
import importlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark import load_settings
from utils.gpu import gpu_lease

ENGINE_ORDER = ["zed", "open3d", "metashape", "rtabmap", "isaac", "bundlefusion"]
BLOCKED_ENGINES = {"zed", "isaac"}

_GPU0_UUID = None


def _gpu_uuids():
    """{index: uuid} from nvidia-smi -L."""
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                         timeout=30).stdout
    return {int(m.group(1)): m.group(2)
            for m in re.finditer(r"GPU (\d+): .* \(UUID: ([^)]+)\)", out)}


def _compute_apps():
    """{pid: (uuid, name)} from nvidia-smi compute-apps; empty on failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    apps = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit():
            apps[int(parts[1])] = (parts[0], parts[2])
    return apps


def _engine_module(engine):
    return importlib.import_module(f"spellbook.reconstruct.engines.{engine}")


def _row(kind, method, status, policy, gpu, visible, lease_fd, runtime, seconds, reason):
    return dict(kind=kind, method=method, status=status, policy=policy,
                physical_gpu=gpu, visible_gpu=visible, lease_fd=lease_fd,
                runtime=runtime, seconds=seconds, reason=reason)


def _validate(rows, pool, intervals, gpu0_violations, require_full_pool=True):
    """Report-wide invariant checks; returns a list of problem strings."""
    problems = []
    if gpu0_violations:
        problems.append(f"new compute PIDs on GPU 0: {gpu0_violations}")

    for row in rows:
        if row.get("status") == "fail":
            problems.append(f"{row['method']} FAIL: {row.get('reason')}")
        elif row.get("status") not in ("pass", "blocked"):
            problems.append(f"{row['method']} unexpected status {row.get('status')}")

    managed_gpus = [row.get("physical_gpu") for row in rows
                    if row.get("status") == "pass" and row.get("policy") == "managed"]
    if any(g not in pool for g in managed_gpus):
        problems.append(f"managed assignment outside pool {pool}: {managed_gpus}")
    if require_full_pool and set(managed_gpus) != set(pool):
        problems.append(f"pool coverage incomplete: assigned {sorted(set(managed_gpus))}, "
                        f"expected {sorted(pool)}")

    seen = set()
    for row in rows:
        key = (row.get("kind"), row.get("method"))
        if key in seen:
            problems.append(f"duplicate row {key}")
        seen.add(key)

    for gpu, a, r, name in intervals:
        for other_gpu, oa, orr, other_name in intervals:
            if gpu == other_gpu and name != other_name and not (r <= oa or orr <= a):
                problems.append(f"overlap on GPU {gpu}: {name} and {other_name}")
                break
    return problems


def _check_managed(task, hold):
    """One managed check: acquire the real lease, run the native check, release.
    Returns (row, acquire_ts, release_ts)."""
    from predict.runner import gpu_check_model

    kind, name, settings = task
    pool, root = settings["gpu_pool"], settings["scannet_root"]
    request_ts = time.time()
    with gpu_lease(pool, root) as lease:
        acquire_ts = time.time()
        if kind == "predict":
            row = gpu_check_model(name, lease, hold)
        else:
            row = _engine_module(name).gpu_check(lease.index, lease.fileno(), hold)
        if row.get("status") != "pass":
            row.setdefault("physical_gpu", lease.index)
        row.setdefault("kind", kind)
        row.setdefault("method", name)
        row.setdefault("policy", "managed")
        row.setdefault("waited", acquire_ts - request_ts > 2.0)
        release_ts = time.time()
        return row, acquire_ts, release_ts


def _run_engine_check(name, hold):
    row = _engine_module(name).gpu_check(None, None, hold)
    row.setdefault("kind", "reconstruct")
    row.setdefault("method", name)
    return row


def run_checks(models=None, engines=None, hold_seconds=5):
    """Run the distribution smoke; returns (rows, ok). `rows` is the full report in
    fixed order (predictors, then engines). `ok` is False on any FAIL, GPU-0 use,
    missing GPU-0 lock check, leaked lock, or unexpected block."""
    from predict import runner

    settings = load_settings()
    pool, root = settings["gpu_pool"], settings["scannet_root"]
    models = list(runner.SUPPORTED_MODELS if models is None else models)
    engines = list(ENGINE_ORDER if engines is None else engines)

    assert 0 not in pool, f"gpu_pool must never contain 0: {pool}"

    uuids = _gpu_uuids()
    gpu0_uuid = uuids.get(0)
    baseline = _compute_apps()
    baseline_pids = set(baseline)
    gpu0_violations = []

    rows = []
    runnable_managed, runnable_cpu, blocked = [], [], []

    for m in models:
        reason = runner.preflight_model(m)
        if reason:
            rows.append(_row("predict", m, "fail", "managed", None, None, None, None,
                             None, reason))
        else:
            runnable_managed.append(("predict", m))

    for e in engines:
        reason = _engine_module(e).preflight()
        if reason:
            if e in BLOCKED_ENGINES:
                rows.append(_row("reconstruct", e, "blocked",
                                 "managed" if e == "isaac" else "blocked",
                                 None, None, None, None, None, reason))
                blocked.append(e)
            else:
                rows.append(_row("reconstruct", e, "fail", "managed", None, None, None,
                                 None, None, reason))
        elif _engine_module(e).GPU_POLICY == "cpu":
            runnable_cpu.append(e)
        else:
            runnable_managed.append(("reconstruct", e))

    managed_tasks = [(k, n, settings) for k, n in runnable_managed]
    assignments = {}
    intervals = []

    def _monitor(stop_event):
        while not stop_event.is_set():
            for pid, (uuid, name) in _compute_apps().items():
                if pid in baseline_pids:
                    continue
                if uuid == gpu0_uuid:
                    gpu0_violations.append((pid, name))
            stop_event.wait(1.0)

    stop = threading.Event()
    monitor = threading.Thread(target=_monitor, args=(stop,), daemon=True)
    monitor.start()

    try:
        with ThreadPoolExecutor(max_workers=max(1, len(managed_tasks))) as pool_ex:
            managed_futures = [pool_ex.submit(_check_managed, t, hold_seconds)
                               for t in managed_tasks]
            managed_results = [f.result() for f in managed_futures]

        cpu_rows = [_run_engine_check(e, hold_seconds) for e in runnable_cpu]

        for row, acquire_ts, release_ts in managed_results:
            assignments.setdefault(row.get("method"), row.get("physical_gpu"))
            if row.get("physical_gpu") is not None:
                intervals.append((row.get("physical_gpu"), acquire_ts, release_ts,
                                  row.get("method")))
            rows.append(row)
        for r in cpu_rows:
            rows.append(r)
    finally:
        stop.set()
        monitor.join(timeout=5)

    problems = _validate(rows, pool, intervals, gpu0_violations)

    managed_gpus = [row.get("physical_gpu") for row in rows
                    if row.get("status") == "pass" and row.get("policy") == "managed"]
    n_waited = sum(1 for row in rows if row.get("waited"))
    ok = not problems
    return rows, ok, dict(assigned=sorted(set(managed_gpus)), waited=n_waited,
                          pool=pool, problems=problems)


def _print_table(rows, summary):
    header = ("kind method status policy physical_gpu visible_gpu lease_fd "
              "runtime seconds reason")
    print(header)
    for row in rows:
        print(" ".join(str(row.get(k, "")) for k in
                       ("kind", "method", "status", "policy", "physical_gpu",
                        "visible_gpu", "lease_fd", "runtime", "seconds", "reason")))
    n_pass = sum(1 for r in rows if r.get("status") == "pass")
    n_blocked = sum(1 for r in rows if r.get("status") == "blocked")
    n_fail = sum(1 for r in rows if r.get("status") == "fail")
    print(f"summary: {n_pass} PASS, {n_blocked} BLOCKED, {n_fail} FAIL; "
          f"pool={summary['pool']} assigned={summary['assigned']} "
          f"waiters={summary['waited']}")
    for p in summary["problems"]:
        print(f"problem: {p}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="GPU distribution smoke (no results)")
    ap.add_argument("--models", default=None,
                    help="comma-separated prediction methods (default: all)")
    ap.add_argument("--engine", nargs="+", default=None,
                    choices=ENGINE_ORDER,
                    help="reconstruction engines (default: all)")
    ap.add_argument("--hold", type=float, default=5.0,
                    help="seconds each native check stays alive (default 5)")
    a = ap.parse_args()

    models = None if a.models is None else a.models.split(",")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "tmp", "logs", f"gpu-check-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    start = time.time()
    rows, ok, summary = run_checks(models=models, engines=a.engine,
                                   hold_seconds=a.hold)
    _print_table(rows, summary)

    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump(dict(rows=rows, summary=summary,
                       elapsed_s=round(time.time() - start, 1)), f, indent=2)
    print(f"logs: {log_dir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()