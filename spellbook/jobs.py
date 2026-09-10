"""Detached Spellbook job queue: submit once, process independent of OpenCode.

Each submission writes a durable record under
/data/scannet/derived/jobs/<job-id>/ and launches one transient
`systemd --user` unit running `jobs.py _run <job-id>`. The submitting
process exits immediately; closing OpenCode, tmux, or the shell does not
stop the job. There is no queue daemon: waiting jobs block in the existing
`gpu_lease()` flock wait, and each detached job is capped to one concurrent
GPU worker so a fifth job starts automatically when any active job releases
its lease through success, failure, cancellation, or death.

Physical GPU 0 is never managed (see spellbook/utils/gpu.py).
"""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

_SPELLBOOK = os.path.dirname(os.path.abspath(__file__))
if _SPELLBOOK not in sys.path:
    sys.path.insert(0, _SPELLBOOK)

SCHEMA_VERSION = 1
ACTIVE_STATES = ("queued", "starting", "running", "cancelling")
TERMINAL_STATES = ("succeeded", "failed", "cancelled", "lost")

ENV_ALLOWLIST = ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH")
ENV_PREFIX_ALLOWLIST = ("SPELLBOOK_",)
JOB_ID_ENV = "SPELLBOOK_JOB_ID"
JOB_MAX_WORKERS_ENV = "SPELLBOOK_JOB_MAX_GPU_WORKERS"


def _scannet_root(scannet_root=None):
    if scannet_root:
        return scannet_root
    from evaluation.benchmark import load_settings
    return load_settings()["scannet_root"]


def jobs_root(scannet_root=None):
    return os.path.join(_scannet_root(scannet_root), "derived", "jobs")


def submit_lock_path(scannet_root=None):
    return os.path.join(_scannet_root(scannet_root), "derived", "locks", "jobs",
                        "submit.lock")


def job_dir(job_id, scannet_root=None):
    return os.path.join(jobs_root(scannet_root), job_id)


def _atomic_write_json(path, doc):
    """Atomically replace `path`. The temp file carries the writer PID so two
    processes updating the same record (e.g. submitter marking "starting"
    while the unit marks "running") never steal each other's temp file."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _new_job_id():
    import random
    return "job-%s-%04x" % (time.strftime("%Y%m%d-%H%M%S"), random.getrandbits(16))


def _git_info(cwd):
    info = {"root": None, "branch": None, "commit": None, "dirty": None}
    try:
        info["root"] = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=15).stdout.strip() or None
        info["branch"] = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=15).stdout.strip() or None
        info["commit"] = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15).stdout.strip() or None
        dirty = subprocess.run(
            ["git", "-C", cwd, "status", "--porcelain"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        info["dirty"] = bool(dirty)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


def _canonical_spellbook(cwd):
    """spellbook/ of the main checkout; linked worktrees share its git dir."""
    try:
        r = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--path-format=absolute",
             "--git-common-dir"], capture_output=True, text=True, timeout=15)
        common = r.stdout.strip()
        if r.returncode == 0 and common:
            cand = os.path.join(os.path.dirname(common), "spellbook")
            if os.path.isfile(os.path.join(cand, "jobs.py")):
                return cand
    except (OSError, subprocess.TimeoutExpired):
        pass
    return _SPELLBOOK


_IDENTITY_FLAGS = ("--engine", "--scene", "--models", "--benchmark", "--classes")


def _flag_values(argv, flag):
    out = []
    for i, tok in enumerate(argv):
        if tok != flag:
            continue
        for val in argv[i + 1:]:
            if val.startswith("-"):
                break
            out.append(val)
    return out


def _run_key(argv, kind, run_id):
    """Logical identity of a run, independent of checkout and commit."""
    if run_id:
        return f"{kind}:{run_id}"
    parts = [f"{f}={','.join(sorted(_flag_values(argv, f)))}"
             for f in _IDENTITY_FLAGS if _flag_values(argv, f)]
    return f"{kind}:" + "|".join(parts)


def _iter_jobs(scannet_root=None):
    """[(job_id, job_doc_or_None, status_doc_or_None)] sorted by submitted."""
    root = jobs_root(scannet_root)
    out = []
    if not os.path.isdir(root):
        return out
    for job_id in sorted(os.listdir(root)):
        d = os.path.join(root, job_id)
        if not os.path.isdir(d):
            continue
        try:
            job = _read_json(os.path.join(d, "job.json"))
        except (OSError, ValueError):
            job = None
        try:
            status = _read_json(os.path.join(d, "status.json"))
        except (OSError, ValueError):
            status = None
        out.append((job_id, job, status))
    out.sort(key=lambda row: (row[2] or {}).get("submitted", 0))
    return out


def _unit_active(unit):
    """True/False from the user manager; None when it cannot be reached."""
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", unit],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.returncode == 0


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def derive_state(job_id, job, status, scannet_root=None):
    """Reconciled state; a dead worker's record is written down as 'lost'."""
    state = (status or {}).get("state", "lost")
    if state not in ACTIVE_STATES or not job:
        return state
    if job.get("unit"):
        if _unit_active(job["unit"]) is not False:
            return state
    elif (status or {}).get("wrapper_pid"):
        if _pid_alive(status["wrapper_pid"]):
            return state
    else:
        return state
    if scannet_root:
        try:
            _update_status(job_id, scannet_root, state="lost",
                           ended=(status or {}).get("ended") or time.time())
        except OSError:
            pass
    return "lost"


def _update_status(job_id, scannet_root, **fields):
    path = os.path.join(job_dir(job_id, scannet_root), "status.json")
    try:
        status = _read_json(path)
    except (OSError, ValueError):
        status = {"job_id": job_id}
    status.update(fields)
    _atomic_write_json(path, status)
    return status


def _launch_unit(job_id, job):
    cmd = [
        "systemd-run", "--user", "--collect",
        f"--unit=spellbook-job-{job_id}.service",
        f"--working-directory={job['cwd']}",
        "--property=Type=exec",
        "--property=KillMode=control-group",
        "--property=TimeoutStopSec=30s",
        job["python"], job.get("runner") or os.path.join(_SPELLBOOK, "jobs.py"),
        "--scannet-root", job["scannet_root"], "_run", job_id,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"systemd-run failed: {(r.stderr or r.stdout).strip()[-500:]}")
    return f"spellbook-job-{job_id}.service"


def submit(argv, cwd=None, kind="command", run_id=None, scannet_root=None,
           launch=True, retry_of=None):
    """Submit a detached job; raises ValueError when a job already owns the run."""
    if not argv or not all(isinstance(a, str) and a for a in argv):
        raise ValueError("submit requires a non-empty argv list of strings")
    if kind not in ("predict", "reconstruct", "extract", "command"):
        raise ValueError(f"unknown job kind {kind!r}")
    cwd = os.path.abspath(cwd or os.getcwd())
    if not os.path.isdir(cwd):
        raise ValueError(f"working directory not found: {cwd}")
    if "--visualize" in argv or "--compare" in argv:
        raise ValueError("interactive modes (--visualize/--compare) cannot be detached")
    exe = argv[0] if os.path.isabs(argv[0]) else shutil.which(argv[0])
    if not exe or not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
        raise ValueError(f"executable not found: {argv[0]}")
    argv = [exe] + list(argv[1:])
    root = _scannet_root(scannet_root)

    git = _git_info(cwd)
    run_key = _run_key(argv, kind, run_id)
    env = {k: v for k, v in os.environ.items()
           if k in ENV_ALLOWLIST or k.startswith(ENV_PREFIX_ALLOWLIST)}
    now = time.time()

    lock_path = submit_lock_path(root)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            for jid, j, st in _iter_jobs(root):
                if not j or not st:
                    continue
                if j.get("run_key") == run_key and \
                        derive_state(jid, j, st, root) in ACTIVE_STATES:
                    raise ValueError(
                        f"active job {jid} already owns {run_key}; "
                        f"use a different --run-id")
            job_id = _new_job_id()
            job = {
                "schema": SCHEMA_VERSION,
                "job_id": job_id,
                "kind": kind,
                "argv": argv,
                "cwd": cwd,
                "python": sys.executable,
                "runner": os.path.join(_canonical_spellbook(cwd), "jobs.py"),
                "scannet_root": root,
                "submitted": now,
                "run_key": run_key,
                "run_id": run_id,
                "git": git,
                "env": env,
                "retry_of": retry_of,
                "unit": f"spellbook-job-{job_id}.service" if launch else None,
            }
            d = job_dir(job_id, root)
            os.makedirs(d, exist_ok=False)
            _atomic_write_json(os.path.join(d, "job.json"), job)
            _atomic_write_json(os.path.join(d, "status.json"), {
                "job_id": job_id, "state": "queued", "submitted": now, "attempt": 1,
            })
            if launch:
                try:
                    _launch_unit(job_id, job)
                except Exception as exc:
                    _update_status(job_id, root, state="failed", ended=now,
                                   error=f"launch failed: {exc}")
                    raise
                _update_status(job_id, root, state="starting")
            print(f"[jobs] {job_id} -> {os.path.join(d, 'job.log')}")
            return job_id
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _run(job_id, scannet_root=None):
    """Internal wrapper executed inside the systemd unit. Returns exit code."""
    root = _scannet_root(scannet_root)
    d = job_dir(job_id, root)
    try:
        job = _read_json(os.path.join(d, "job.json"))
    except (OSError, ValueError) as exc:
        print(f"[jobs] {job_id}: missing job.json ({exc})", file=sys.stderr)
        return 2
    log_path = os.path.join(d, "job.log")
    start = time.time()
    _update_status(job_id, root, state="running", started=start,
                   wrapper_pid=os.getpid(), gpu_wait=0.0)

    env = os.environ.copy()
    env.update(job.get("env") or {})
    env["PYTHONUNBUFFERED"] = "1"
    env[JOB_ID_ENV] = job_id
    env[JOB_MAX_WORKERS_ENV] = "1"

    child = None
    cancelled = []

    def _forward(signum, frame):
        cancelled.append(signum)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except (OSError, ProcessLookupError):
                pass

    old_int = signal.signal(signal.SIGINT, _forward)
    old_term = signal.signal(signal.SIGTERM, _forward)
    try:
        with open(log_path, "a", buffering=1) as logf:
            logf.write(f"[jobs] {job_id} start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            logf.write(f"[jobs] cwd={job['cwd']}\n")
            logf.write(f"[jobs] argv={json.dumps(job['argv'])}\n")
            child = subprocess.Popen(
                job["argv"], cwd=job["cwd"], env=env, start_new_session=True,
                stdout=logf, stderr=subprocess.STDOUT)
            _update_status(job_id, root, child_pid=child.pid)
            rc = child.wait()
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
    end = time.time()
    with open(log_path, "a") as logf:
        logf.write(f"[jobs] {job_id} end rc={rc} elapsed={end - start:.1f}s\n")
    try:
        status = _read_json(os.path.join(d, "status.json"))
    except (OSError, ValueError):
        status = {}
    if status.get("state") == "cancelling" or signal.SIGTERM in cancelled:
        _update_status(job_id, root, state="cancelled", ended=end,
                       exit_code=rc if isinstance(rc, int) else None)
        return 0 if rc == 0 else 1
    if rc == 0:
        _update_status(job_id, root, state="succeeded", ended=end, exit_code=0)
        return 0
    _update_status(job_id, root, state="failed", ended=end, exit_code=rc,
                   error=f"exit code {rc}")
    return rc if isinstance(rc, int) and 0 < rc < 256 else 1


@contextmanager
def job_context(argv, cwd=None, kind="command", run_id=None, scannet_root=None):
    """Run inline under a durable job record.

    No-op when already inside a job: the detached wrapper sets JOB_ID_ENV
    before spawning the child, so the child adopts the existing record
    instead of creating a second one.
    """
    if os.environ.get(JOB_ID_ENV):
        yield os.environ[JOB_ID_ENV]
        return
    root = _scannet_root(scannet_root)
    job_id = submit(argv, cwd=cwd, kind=kind, run_id=run_id,
                    scannet_root=root, launch=False)
    _update_status(job_id, root, state="running", started=time.time(),
                   wrapper_pid=os.getpid(), gpu_wait=0.0)
    os.environ[JOB_ID_ENV] = job_id
    try:
        yield job_id
    except KeyboardInterrupt:
        _update_status(job_id, root, state="cancelled", ended=time.time())
        raise
    except BaseException as exc:
        _update_status(job_id, root, state="failed", ended=time.time(),
                       error=str(exc)[:500])
        raise
    else:
        _update_status(job_id, root, state="succeeded", ended=time.time(),
                       exit_code=0)
    finally:
        os.environ.pop(JOB_ID_ENV, None)


def list_jobs(scannet_root=None):
    rows = []
    for job_id, job, status in _iter_jobs(scannet_root):
        if not job or not status:
            continue
        rows.append({
            "job_id": job_id,
            "kind": job.get("kind"),
            "run_id": job.get("run_id"),
            "state": derive_state(job_id, job, status, scannet_root),
            "submitted": status.get("submitted"),
            "started": status.get("started"),
            "ended": status.get("ended"),
            "gpu_wait": status.get("gpu_wait"),
            "exit_code": status.get("exit_code"),
            "branch": (job.get("git") or {}).get("branch"),
            "log": os.path.join(job_dir(job_id, _scannet_root(scannet_root)), "job.log"),
        })
    return rows


def show_job(job_id, scannet_root=None):
    root = _scannet_root(scannet_root)
    job = _read_json(os.path.join(job_dir(job_id, root), "job.json"))
    try:
        status = _read_json(os.path.join(job_dir(job_id, root), "status.json"))
    except (OSError, ValueError):
        status = {}
    print(json.dumps({"job": job, "status": status,
                      "derived_state": derive_state(job_id, job, status)},
                     indent=2, sort_keys=True))


def job_logs(job_id, scannet_root=None, follow=False, last=60):
    path = os.path.join(job_dir(job_id, _scannet_root(scannet_root)), "job.log")
    if follow:
        os.execvp("tail", ["tail", "-F", path])
    if not os.path.isfile(path):
        print(f"no log yet: {path}")
        return
    with open(path, errors="replace") as f:
        lines = f.readlines()
    sys.stdout.write("".join(lines[-last:]))


def cancel_job(job_id, scannet_root=None, timeout=30):
    root = _scannet_root(scannet_root)
    job = _read_json(os.path.join(job_dir(job_id, root), "job.json"))
    try:
        status = _read_json(os.path.join(job_dir(job_id, root), "status.json"))
    except (OSError, ValueError):
        status = {}
    state = derive_state(job_id, job, status, root)
    if state in TERMINAL_STATES:
        print(f"[jobs] {job_id} already {state}")
        return state
    _update_status(job_id, root, state="cancelling")
    if job.get("unit"):
        subprocess.run(["systemctl", "--user", "stop", job["unit"]],
                       capture_output=True, timeout=timeout + 10)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _unit_active(job["unit"]) is False:
                break
            time.sleep(0.5)
    else:
        pid = status.get("wrapper_pid")
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (OSError, ValueError):
                pass
    _update_status(job_id, root, state="cancelled", ended=time.time())
    print(f"[jobs] {job_id} cancelled")
    return "cancelled"


def retry_job(job_id, scannet_root=None):
    root = _scannet_root(scannet_root)
    job = _read_json(os.path.join(job_dir(job_id, root), "job.json"))
    if not os.path.isdir(job["cwd"]):
        raise ValueError(f"{job_id} cwd is gone: {job['cwd']}")
    try:
        status = _read_json(os.path.join(job_dir(job_id, root), "status.json"))
    except (OSError, ValueError):
        status = {}
    if derive_state(job_id, job, status, root) in ACTIVE_STATES:
        raise ValueError(f"{job_id} is still active; cancel it before retry")
    new_id = _new_job_id()
    now = time.time()
    new_job = dict(job)
    new_job.update({"job_id": new_id, "submitted": now, "retry_of": job_id,
                    "unit": f"spellbook-job-{new_id}.service"})
    d = job_dir(new_id, root)
    os.makedirs(d, exist_ok=False)
    _atomic_write_json(os.path.join(d, "job.json"), new_job)
    _atomic_write_json(os.path.join(d, "status.json"), {
        "job_id": new_id, "state": "queued", "submitted": now, "attempt": 1,
    })
    _launch_unit(new_id, new_job)
    _update_status(new_id, root, state="starting")
    print(f"[jobs] {new_id} (retry of {job_id})")
    return new_id


def _cmd_list(args):
    import time as _time
    from utils.status import dur as _dur
    rows = list_jobs(args.scannet_root)
    if not rows:
        print("no jobs")
        return
    now = _time.time()
    for r in rows:
        age = _dur(now - r["submitted"]) if r.get("submitted") else "-"
        print(f"{r['job_id']} {r['kind']} {r['run_id']} {r['state']} {age}")


def _cmd_submit(args):
    if not args.command:
        raise SystemExit("submit requires a command after --")
    job_id = submit(list(args.command), cwd=args.cwd, kind=args.kind,
                    run_id=args.run_id, scannet_root=args.scannet_root)
    print(job_id)


def render_snapshot(res, pool, prev_cpu, scannet_root=None):
    """One dashboard frame; kept for compatibility, delegates to utils.status."""
    from utils.status import render_snapshot as _render
    return _render(res, pool, prev_cpu, list_jobs(scannet_root),
                   _scannet_root(scannet_root))


def main(argv=None):
    ap = argparse.ArgumentParser(description="detached Spellbook job queue")
    ap.add_argument("--scannet-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("submit", help="submit a detached job")
    p.add_argument("--kind", default="command",
                   choices=["predict", "reconstruct", "extract", "command"])
    p.add_argument("--cwd", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=_cmd_submit)

    p = sub.add_parser("list", help="one-shot job listing")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("show", help="dump one job record")
    p.add_argument("job_id")
    p.set_defaults(func=lambda a: show_job(a.job_id, a.scannet_root))

    p = sub.add_parser("logs", help="print a job log")
    p.add_argument("job_id")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--last", type=int, default=60)
    p.set_defaults(func=lambda a: job_logs(a.job_id, a.scannet_root, a.follow, a.last))

    p = sub.add_parser("cancel", help="stop a job's unit")
    p.add_argument("job_id")
    p.set_defaults(func=lambda a: cancel_job(a.job_id, a.scannet_root))

    p = sub.add_parser("retry", help="resubmit a terminal job")
    p.add_argument("job_id")
    p.set_defaults(func=lambda a: print(retry_job(a.job_id, a.scannet_root)))

    p = sub.add_parser("_run", help=argparse.SUPPRESS)
    p.add_argument("job_id")
    p.set_defaults(func=lambda a: sys.exit(_run(a.job_id, a.scannet_root)))

    args = ap.parse_args(argv)
    if args.cmd == "submit" and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    args.func(args)


if __name__ == "__main__":
    main()
