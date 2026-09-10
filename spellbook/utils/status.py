"""Read-only live dashboard used by `main.py --status`.

Sections: CURRENT JOBS (one row per processing job plus device-only rows
for GPUs without a job), FUTURE JOBS (observable queue, oldest first),
PAST JOBS (latest 20 terminal jobs). No writes, no samplers, no sidecars.
"""
import os
import re
import sys
import time

ACTIVE_JOB_STATES = ("queued", "starting", "running", "cancelling")
TERMINAL_JOB_STATES = ("succeeded", "failed", "cancelled", "lost")

CPU_CURRENT_THRESHOLD = 0.05
LOG_TAIL_BYTES = 16384


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


def short_job(job_id):
    """Compact registry id suffix, e.g. job-20260910-181605-4db8 -> 181605-4db8."""
    if not job_id:
        return "-"
    text = str(job_id)
    if text.startswith("job-"):
        text = text[len("job-"):]
    return text[-11:] if len(text) > 11 else text


def _arg(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _multi_arg(argv, name):
    """All values following each occurrence of a repeatable flag."""
    out = []
    argv = list(argv or [])
    for i, tok in enumerate(argv):
        if tok != name:
            continue
        for val in argv[i + 1:]:
            if val.startswith("-"):
                break
            out.append(val)
    return out


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


def _effective_runtime(row, now):
    started, wait = row.get("started"), row.get("gpu_wait")
    if isinstance(started, (int, float)) and isinstance(wait, (int, float)):
        return dur(max(0.0, now - started - wait))
    if isinstance(started, (int, float)) and wait is None:
        return dur(max(0.0, now - started))
    return "-"


def _job_pid(row, res):
    """Verified live PID for CPU accounting; None on race or PID reuse."""
    job_id = row.get("job_id")
    for key in ("wrapper_pid", "child_pid"):
        pid = row.get(key)
        if not pid:
            continue
        try:
            with open(f"/proc/{int(pid)}/cgroup") as f:
                cgroup = f.read()
        except (OSError, ValueError):
            continue
        if f"spellbook-job-{job_id}.service" not in cgroup:
            continue
        return int(pid)
    return None


def _cpu_cores_for(row, res, cpu_base, cpu_interval, cpu_cores):
    if cpu_cores is not None and row.get("job_id") in cpu_cores:
        value = cpu_cores[row["job_id"]]
        return float(value) if isinstance(value, (int, float)) else None
    if not cpu_base or not cpu_interval:
        return None
    base = cpu_base.get(row.get("job_id"))
    if base is None:
        return None
    pid = _job_pid(row, res)
    if pid is None:
        return None
    try:
        from utils import resources as _res
        cur = _res.read_cpu_usec(_res.job_cgroup_path(pid))
    except Exception:
        return None
    if cur is None or cur < base or cpu_interval <= 0:
        return None
    return (cur - base) / (cpu_interval * 1_000_000)


def _fmt_cpu(cores):
    return f"{cores:.1f}c" if isinstance(cores, (int, float)) else "-"


def _read_json_quiet(path):
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _log_tail(row, limit=LOG_TAIL_BYTES):
    path = row.get("log")
    if not path or not isinstance(path, str):
        return ""
    try:
        with open(path, "rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - limit))
            except OSError:
                pass
            return f.read().decode(errors="replace").replace("\r", "\n")
    except OSError:
        return ""


def _live_cmdlines(row, res):
    """Bounded live descendant commands for UNIT detection (read-only)."""
    seen, out = set(), []
    pids = []
    for key in ("wrapper_pid", "child_pid"):
        pid = row.get(key)
        if isinstance(pid, int) and pid not in seen:
            seen.add(pid)
            pids.append(pid)
    try:
        anchor = pids[0] if pids else None
        path = res.job_cgroup_path(anchor) if anchor else None
        if path:
            for pid in res.cgroup_pids(path):
                if pid not in seen:
                    seen.add(pid)
                    pids.append(pid)
                if len(pids) >= 24:
                    break
    except Exception:
        pass
    for pid in pids[:24]:
        try:
            argv = res.proc_cmdline(pid)
        except Exception:
            continue
        if argv:
            out.append(argv)
    return out


def _predict_plan(row, scannet_root, memo):
    """(model, expected_scenes, spec, run_id) or None when unsupported."""
    argv = row.get("argv") or []
    models_raw = _arg(argv, "--models")
    models = [m.strip() for m in (models_raw or "").split(",") if m.strip()]
    if len(models) != 1:
        return None
    model = models[0]
    run_id = row.get("run_id") or _arg(argv, "--run-id")
    if not run_id:
        return None
    bench = _arg(argv, "--benchmark")
    try:
        from evaluation.benchmark import (
            PREDICTION_EVALUATION_SCENES, resolve_benchmark)
        spec = resolve_benchmark(bench)
    except Exception:
        return None
    key = ("predict-plan", spec.name, run_id, model)
    if key in memo:
        return memo[key]
    scenes = None
    try:
        from evaluation.runs import load_manifest, manifest_path
        man = load_manifest(manifest_path(spec, run_id, scannet_root),
                            spec=spec, run_id=run_id)
        if list(man.get("methods") or []) == [model]:
            scenes = list(man.get("scenes") or [])
    except Exception:
        scenes = None
    if not scenes:
        only = _multi_arg(argv, "--scene")
        scenes = only or list(PREDICTION_EVALUATION_SCENES)
    plan = (model, scenes, spec, run_id)
    memo[key] = plan
    return plan


def _predict_progress(row, scannet_root, memo):
    plan = _predict_plan(row, scannet_root, memo)
    if plan is None:
        return {"done": None, "total": None, "unit": "-",
                "avg_s": None, "eta_s": None, "plan": None}
    model, expected, spec, run_id = plan
    key = ("predict-progress", spec.name, run_id, model, tuple(expected))
    if key in memo:
        prog = dict(memo[key])
    else:
        done_scenes = []
        try:
            from evaluation.runs import tasks_path
            path = tasks_path(spec, run_id, model, scannet_root)
            with open(path) as f:
                seen = set()
                for line in f:
                    scene = line.strip()
                    if scene and scene not in seen:
                        seen.add(scene)
                        done_scenes.append(scene)
        except OSError:
            done_scenes = []
        want = set(expected)
        done = [s for s in done_scenes if s in want]
        timings = []
        try:
            from utils.compute_time import load_timing, prediction_timing_path
            for scene in done:
                try:
                    value = load_timing(
                        prediction_timing_path(spec, run_id, model, scene,
                                               scannet_root),
                        "prediction", model=model, run_id=run_id,
                        scene_id=scene)
                except Exception:
                    value = None
                if value is not None:
                    timings.append(float(value))
        except Exception:
            timings = []
        avg = sum(timings) / len(timings) if timings else None
        eta = avg * (len(expected) - len(done)) \
            if avg is not None and len(timings) >= 2 \
            and len(expected) > len(done) else None
        prog = {"done": len(done), "total": len(expected),
                "avg_s": avg, "eta_s": eta}
        memo[key] = dict(prog)
    unit = _predict_unit(row, model, set(expected))
    prog = dict(prog)
    prog["unit"] = unit
    prog["plan"] = f"{len(expected)} scenes x {model}"
    return prog


def _predict_unit(row, model, expected):
    for argv in _live_argvs(row):
        scenes = [t for t in argv if t in expected]
        if scenes:
            return f"{model}:{scenes[-1]}"
    tail = _log_tail(row)
    if tail:
        matches = re.findall(r"running (\S+) on (\S+)", tail)
        for found_model, scene in reversed(matches):
            if found_model == model and scene in expected:
                return f"{model}:{scene}"
    return "-"


def _live_argvs(row):
    try:
        from utils import resources as _res
        argvs = []
        for argv in _live_cmdlines(row, _res):
            argvs.append(argv)
        return argvs
    except Exception:
        return []


def _scene_bases(argv):
    bases = []
    for tok in _multi_arg(argv, "--scene"):
        m = re.match(r"^scene(\d{4})(?:_\d{2})?$", tok)
        if m:
            bases.append(int(m.group(1)))
            continue
        if re.match(r"^\d{3,5}$", tok):
            bases.append(int(tok))
    seen, out = set(), []
    for base in bases:
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


def _reconstruct_progress(row, scannet_root, memo):
    argv = row.get("argv") or []
    engines = _multi_arg(argv, "--engine")
    bases = _scene_bases(argv)
    if not engines or not bases:
        return {"done": None, "total": None, "unit": "-",
                "avg_s": None, "eta_s": None, "plan": None}
    try:
        from reconstruct import engine_run_range, scan_dir, scan_id
        from utils.compute_time import load_timing, reconstruction_timing_path
    except Exception:
        return {"done": None, "total": None, "unit": "-",
                "avg_s": None, "eta_s": None, "plan": None}
    pairs = [(b, e) for b in bases for e in engines]
    done, timings, by_engine = 0, [], {}
    for base, engine in pairs:
        try:
            slots = engine_run_range(engine)
        except (ValueError, KeyError):
            continue
        hit, elapsed = False, None
        for run_num in slots[-10:]:
            try:
                suffix = run_num - ({"zed": 0, "metashape": 1, "rtabmap": 2,
                                     "isaac": 3, "open3d": 4,
                                     "bundlefusion": 5}[engine] * 10)
                sid = scan_id(base, engine, suffix)
                sdir = scan_dir(base, engine, suffix)
            except (ValueError, KeyError):
                continue
            key = ("recon-timing", sdir)
            if key in memo:
                value = memo[key]
            else:
                try:
                    value = load_timing(reconstruction_timing_path(sdir),
                                        "reconstruction", scene_id=sid,
                                        engine=engine)
                except Exception:
                    value = None
                memo[key] = value
            if value is not None:
                hit, elapsed = True, float(value)
                break
            if _recon_complete(sdir):
                hit = True
                break
        if hit:
            done += 1
            if elapsed is not None:
                timings.append(elapsed)
                by_engine.setdefault(engine, []).append(elapsed)
    avg = sum(timings) / len(timings) if timings else None
    remaining = len(pairs) - done
    eta = avg * remaining if avg is not None and len(timings) >= 2 \
        and remaining > 0 else None
    engines_txt = ",".join(engines)
    return {"done": done, "total": len(pairs),
            "unit": _recon_unit(row),
            "avg_s": avg, "eta_s": eta,
            "plan": f"{len(pairs)} tasks x {engines_txt}"}


def _recon_complete(sdir):
    # Direct isfile checks for canonical names derived from directory name.
    name = os.path.basename(sdir.rstrip("/"))
    candidates = [os.path.join(sdir, f"{name}.sens"),
                  os.path.join(sdir, f"{name}.txt"),
                  os.path.join(sdir, f"{name}_vh_clean.ply"),
                  os.path.join(sdir, f"{name}_vh_clean_2.ply"),
                  os.path.join(sdir, "recon", "final_poses.npy")]
    if not all(os.path.isfile(p) for p in candidates):
        return False
    try:
        for f in os.listdir(sdir):
            if f.endswith(".segs.json"):
                return True
    except OSError:
        return False
    return False


def _recon_unit(row):
    for argv in _live_argvs(row):
        joined = " ".join(argv)
        m = re.search(r"scene\d{4}_\d{2}", joined)
        if m:
            engine = _arg(argv, "--engine") or ""
            return f"{engine}:{m.group(0)}" if engine else m.group(0)
    tail = _log_tail(row)
    if tail:
        matches = re.findall(r"scene\d{4}_\d{2}", tail)
        if matches:
            return matches[-1]
    scene = _arg(row.get("argv") or [], "--scene")
    engine = _arg(row.get("argv") or [], "--engine")
    if scene and engine:
        return f"{engine}:scene{scene}"
    return "-"


def _extract_progress(row, scannet_root, memo):
    argv = row.get("argv") or []
    bases = _scene_bases(argv)
    if not bases:
        return {"done": None, "total": None, "unit": "-",
                "avg_s": None, "eta_s": None, "plan": None}
    done = 0
    for base in bases:
        key = ("extract", base)
        if key in memo:
            ok = memo[key]
        else:
            man = os.path.join(str(scannet_root), "derived", "reconstruction",
                               "frames", f"scene{base:04d}",
                               "frames_manifest.json")
            ok = False
            try:
                doc = _read_json_quiet(man)
                if isinstance(doc, dict) and int(doc.get("color_n", 0)) > 0:
                    ok = True
            except (TypeError, ValueError):
                ok = False
            memo[key] = ok
        if ok:
            done += 1
    return {"done": done, "total": len(bases),
            "unit": _recon_unit(row),
            "avg_s": None, "eta_s": None,
            "plan": f"extract scene{bases[0]:04d}" if len(bases) == 1
            else f"{len(bases)} scenes extract"}


def _progress(row, scannet_root, memo):
    kind = row.get("kind")
    try:
        if kind == "predict":
            return _predict_progress(row, scannet_root, memo)
        if kind == "reconstruct":
            return _reconstruct_progress(row, scannet_root, memo)
        if kind == "extract":
            return _extract_progress(row, scannet_root, memo)
    except Exception:
        pass
    return {"done": None, "total": None, "unit": "-",
            "avg_s": None, "eta_s": None, "plan": None}


def _plan_string(row):
    argv = row.get("argv") or []
    kind = row.get("kind")
    if kind == "predict":
        models_raw = _arg(argv, "--models") or ""
        models = [m for m in models_raw.split(",") if m]
        scenes = _multi_arg(argv, "--scene")
        n = len(scenes) if scenes else 16
        model = models[0] if len(models) == 1 else (models_raw or "?")
        return f"{n} scenes x {model}"
    if kind == "reconstruct":
        engines = _multi_arg(argv, "--engine") or ["?"]
        bases = _scene_bases(argv) or ["?"]
        return f"{len(bases)} scenes x {','.join(engines)}"
    if kind == "extract":
        bases = _scene_bases(argv)
        if len(bases) == 1:
            return f"extract scene{bases[0]:04d}"
        if bases:
            return f"extract {len(bases)} scenes"
        return "extract"
    cmd = (argv[-1] if argv else None) or row.get("kind") or "command"
    return os.path.basename(str(cmd))[:40]


def _fmt_progress(prog):
    done, total = prog.get("done"), prog.get("total")
    if isinstance(done, int) and isinstance(total, int) and total > 0:
        return f"{done}/{total}"
    return "-"


def _fmt_avg_eta(prog):
    avg = prog.get("avg_s")
    eta = prog.get("eta_s")
    avg_txt = dur(avg) if isinstance(avg, (int, float)) else "-"
    eta_txt = dur(eta) if isinstance(eta, (int, float)) else "-"
    return avg_txt, eta_txt


CURRENT_COLS = ("GPU", "GPU%", "MEMORY", "RUNTIME", "CPUc", "PROGRESS", "ETA",
                "SESSION", "RUN")
CURRENT_FMT = "{:>7} {:>7} {:>13} {:>7} {:>7} {:>8} {:>7} {:<12} {}"
FUTURE_COLS = ("STATUS", "SESSION", "RUN")
FUTURE_FMT = "{:<6} {:<12} {}"
PAST_COLS = ("STATE", "RUNTIME", "SESSION", "RUN")
PAST_FMT = "{:<8} {:>7} {:<12} {}"


def _header(fmt, cols):
    """Header padded by the same format its rows use, so columns line up."""
    return fmt.format(*cols).rstrip()


def render_snapshot(res, pool, prev_cpu, rows, scannet_root,
                    cpu_base=None, cpu_interval=None, cpu_cores=None,
                    now=None):
    """One dashboard frame as plain text lines."""
    now = now if isinstance(now, (int, float)) else time.time()
    lines = []
    cur = res.cpu_times()
    util = res.cpu_util(prev_cpu, cur)
    try:
        ncpu = os.cpu_count() or 0
    except OSError:
        ncpu = 0
    fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else "?"
    la1, la5, la15 = res.load_average()
    upct = f"{util * 100:.0f}%" if util is not None else "?%"
    lines.append(f"cpu {upct} load {fmt(la1)} {fmt(la5)} {fmt(la15)} x{ncpu}")

    stats = res.gpu_stats()
    lock_dir = os.path.join(scannet_root, "derived", "locks", "gpus")
    leases = res.gpu_leases(lock_dir, pool)
    rows_by_id = {row["job_id"]: row for row in rows}
    owned = {}
    owners = {}
    for gpu, pid in leases.items():
        try:
            owner = res.process_info(pid)
        except Exception:
            owner = {}
        owners[gpu] = owner
        job_id = (owner or {}).get("job_id")
        if job_id in rows_by_id:
            owned.setdefault(job_id, []).append(gpu)

    if cpu_cores is None:
        cpu_cores = {}
        if cpu_base and cpu_interval:
            for row in rows:
                if row.get("state") not in ACTIVE_JOB_STATES:
                    continue
                value = _cpu_cores_for(row, res, cpu_base, cpu_interval, None)
                if value is not None:
                    cpu_cores[row["job_id"]] = value

    current_ids = set()
    for row in rows:
        if row.get("state") not in ACTIVE_JOB_STATES:
            continue
        if row["job_id"] in owned:
            current_ids.add(row["job_id"])
            continue
        cores = cpu_cores.get(row["job_id"])
        if isinstance(cores, (int, float)) and cores >= CPU_CURRENT_THRESHOLD:
            current_ids.add(row["job_id"])

    memo = {}
    current_rows = sorted(
        (r for r in rows if r["job_id"] in current_ids),
        key=lambda r: r.get("submitted") or 0)

    lines.append("")
    lines.append("CURRENT JOBS")
    lines.append(_header(CURRENT_FMT, CURRENT_COLS))
    by_gpu_job = {}
    for row in current_rows:
        for gpu in owned.get(row["job_id"], []):
            by_gpu_job[gpu] = row
    for gpu in sorted(set(stats) | set(pool) | set(leases) | {0}):
        if gpu in by_gpu_job:
            row = by_gpu_job[gpu]
            lines.append(_current_line(row, gpu, stats, cpu_cores, memo,
                                       scannet_root, now))
        elif gpu in leases:
            lines.append(_holder_line(gpu, stats, owners.get(gpu), memo,
                                      scannet_root, now))
        else:
            lines.append(_device_line(gpu, stats.get(gpu), pool))
    for row in current_rows:
        if row["job_id"] in owned:
            continue
        lines.append(_current_line(row, None, stats, cpu_cores, memo,
                                   scannet_root, now))

    future = sorted(
        (r for r in rows if r.get("state") in ACTIVE_JOB_STATES
         and r["job_id"] not in current_ids),
        key=lambda r: r.get("submitted") or 0)
    lines.append("")
    lines.append("FUTURE JOBS")
    lines.append(_header(FUTURE_FMT, FUTURE_COLS))
    labels = {"queued": "QUEUE", "starting": "START", "running": "WAIT",
              "cancelling": "STOP"}
    for row in future:
        lines.append(FUTURE_FMT.format(
            labels.get(row["state"], row["state"]),
            short_branch(row.get("branch"))[:12],
            row.get("run_id") or row["kind"]))

    recent = sorted(
        (row for row in rows if row["state"] in TERMINAL_JOB_STATES),
        key=lambda row: row.get("ended") or row.get("started")
        or row.get("submitted") or 0,
        reverse=True,
    )[:20]
    lines.append("")
    lines.append("PAST JOBS")
    lines.append(_header(PAST_FMT, PAST_COLS))
    past_labels = {"succeeded": "DONE", "failed": "FAIL",
                   "cancelled": "CANCEL", "lost": "LOST"}
    for row in recent:
        started, ended = row.get("started"), row.get("ended")
        wait = row.get("gpu_wait")
        if isinstance(started, (int, float)) \
                and isinstance(ended, (int, float)) \
                and isinstance(wait, (int, float)):
            runtime = dur(max(0.0, ended - started - wait))
        else:
            runtime = "-"
        state = past_labels[row["state"]]
        if state == "FAIL" and row.get("exit_code") not in (None, 0):
            state += f":{row['exit_code']}"
        lines.append(PAST_FMT.format(
            state, runtime, short_branch(row.get("branch"))[:12],
            row.get("run_id") or row["kind"]))
    return lines


def _gpu_cells(gpu, stats):
    """GPU/util/memory cells for one device, or dashes for CPU-only work."""
    if gpu is None:
        return "-", "-", "-"
    st = stats.get(gpu)
    if st is None:
        return str(gpu), "-", "-"
    return str(gpu), f"{st['util']:.0f}%", \
        f"{st['mem_used'] / 1024:.1f}/{st['mem_total'] / 1024:.1f}G"


def _synthetic_row(owner):
    """Registry-free job row from a live lease holder (read-only).

    Foreground runs without a job record still carry kind/benchmark/
    model/run-id/cwd in argv, which is all the progress resolvers need.
    """
    owner = owner or {}
    argv = list(owner.get("argv") or [])
    kind = "command"
    if "--predict" in argv:
        kind = "predict"
    elif "--engine" in argv:
        kind = "reconstruct"
    elif "--extract-frames" in argv:
        kind = "extract"
    return {"job_id": f"pid-{owner.get('pid') or '?'}", "kind": kind,
            "argv": argv, "cwd": owner.get("cwd"),
            "run_id": _arg(argv, "--run-id"),
            "started": owner.get("started"), "gpu_wait": 0.0,
            "state": "running"}


def _job_line(gpu, stats, row, cpu, session, run, memo, scannet_root, now):
    """Shared row body for registry jobs and record-free lease holders."""
    gpu_txt, util_txt, mem_txt = _gpu_cells(gpu, stats)
    prog = _progress(row, scannet_root, memo)
    _, eta_txt = _fmt_avg_eta(prog)
    return CURRENT_FMT.format(
        gpu_txt, util_txt, mem_txt, _effective_runtime(row, now),
        cpu, _fmt_progress(prog), eta_txt, session[:12], run)


def _current_line(row, gpu, stats, cpu_cores, memo, scannet_root, now):
    return _job_line(gpu, stats, row, _fmt_cpu(cpu_cores.get(row["job_id"])),
                     short_branch(row.get("branch")),
                     row.get("run_id") or row["kind"], memo, scannet_root, now)


def _holder_line(gpu, stats, owner, memo, scannet_root, now):
    """One row for a leased GPU whose holder has no job record."""
    owner = owner or {}
    row = _synthetic_row(owner)
    return _job_line(gpu, stats, row, "-", _session(owner),
                     row.get("run_id") or _run_name(owner), memo,
                     scannet_root, now)


def _device_line(gpu, st, pool):
    """Row for a GPU that no current job holds."""
    if st is None:
        return CURRENT_FMT.format(gpu, "-", "-", "-", "-", "-", "-", "-",
                                  "unavailable")
    mem = f"{st['mem_used'] / 1024:.1f}/{st['mem_total'] / 1024:.1f}G"
    util = f"{st['util']:.0f}%"
    if gpu == 0:
        run = "reserved"
    elif gpu not in pool:
        run = "unmanaged"
    elif st["util"] > 5 or st["mem_used"] > 512:
        run = "unleased"
    else:
        run = "free"
    return CURRENT_FMT.format(gpu, util, mem, "-", "-", "-", "-", "-", run)


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
    from jobs import list_jobs
    rows = list_jobs(root)
    base = {}
    for row in rows:
        if row.get("state") not in ACTIVE_JOB_STATES:
            continue
        pid = _job_pid(row, res)
        if pid is None:
            continue
        try:
            value = res.read_cpu_usec(res.job_cgroup_path(pid))
        except Exception:
            value = None
        if value is not None:
            base[row["job_id"]] = value
    start = time.monotonic()
    time.sleep(1.0)
    interval = time.monotonic() - start
    sys.stdout.write("\n".join(render_snapshot(
        res, pool, prev, rows, root, cpu_base=base,
        cpu_interval=interval)) + "\n")
    sys.stdout.flush()
