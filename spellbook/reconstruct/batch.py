"""Batch runner: many scenes x engines over a GPU pool with a quiet terminal.

Phase 1 (sequential): one shared frame set per scene under
/data/scannet/derived/reconstruction/frames/<scene>, extracted at most once
(--replace regenerates). Phase 2 (parallel): one subprocess per (scene, engine)
task, each pinned to one GPU slot, stdout/stderr piped to
spellbook/tmp/logs/reconstruct-<run-id>/<scan_id>.log; the terminal only shows
tqdm bars (one per GPU slot). Metashape and isaac are serialized (license /
container constraints); engines whose runtime is missing are skipped with a
clear reason instead of failing the whole batch.
"""
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from . import ENGINE_INDEX, svo_path
from . import extract as extract_mod

FRAMES_ROOT = "/data/scannet/derived/reconstruction/frames"
LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tmp", "logs")
SERIAL_ENGINES = {"metashape", "isaac"}


def _preflight_engines(engines):
    """Return {engine: None|reason}. None = runtime present."""
    reasons = {}
    if "metashape" in engines:
        py = "/data/zed-metashape/conda/env/bin/python"
        if not os.path.exists(py):
            reasons["metashape"] = f"Metashape env python missing: {py}"
    if "bundlefusion" in engines:
        if subprocess.run(["docker", "image", "inspect", "bundlefusion:latest"],
                          capture_output=True).returncode != 0:
            reasons["bundlefusion"] = "docker image bundlefusion:latest missing"
    if "rtabmap" in engines:
        if subprocess.run(["podman", "image", "exists", "localhost/zed-rtabmap:jazzy"],
                          capture_output=True).returncode != 0:
            reasons["rtabmap"] = "podman image localhost/zed-rtabmap:jazzy missing"
    if "isaac" in engines:
        if subprocess.run(["podman", "image", "exists", "zed-isaac-nvblox:spellbook"],
                          capture_output=True).returncode != 0:
            reasons["isaac"] = ("podman image zed-isaac-nvblox:spellbook missing "
                                "(NGC pull + zed layer build required)")
    return reasons


def _gpu_count():
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return len([l for l in out.stdout.splitlines() if l.strip()]) if out.returncode == 0 else 0


def _prepare_frames(scenes, gpus, replace):
    """Sequential per-scene extraction into FRAMES_ROOT; returns {scene: frames_dir}."""
    os.makedirs(FRAMES_ROOT, exist_ok=True)
    gpu = gpus[0] if gpus else None
    by_scene = {}
    with tqdm(total=len(scenes), desc="frames", unit="scene", leave=False) as pbar:
        for scene in scenes:
            svo = svo_path(scene)
            if not os.path.exists(svo):
                raise SystemExit(f"svo not found: {svo}")
            work_dir = os.path.join(FRAMES_ROOT, f"scene{scene:04d}")
            os.makedirs(work_dir, exist_ok=True)
            by_scene[scene] = os.path.join(work_dir, "frames")
            extract_mod.ensure_frames(svo, work_dir, gpu=gpu, replace=replace)
            pbar.update(1)
            pbar.set_postfix_str(f"scene{scene:04d}")
    return by_scene


def _run_task(scene, engine, gpu, frames_dir, log_path, bar, proc_registry):
    sid = f"scene{scene:04d}_{ENGINE_INDEX[engine]:02d}"
    cmd = [sys.executable, "-m", "spellbook.reconstruct.run",
           "--scene", str(scene), "--engine", engine, "--gpu", str(gpu),
           "--frames", frames_dir]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    with proc_registry["lock"]:
        proc_registry["procs"].add(proc)
    bar.set_description(f"gpu{gpu} {sid}")
    stage, tail = "running", []
    with open(log_path, "w") as logf:
        logf.write("cmd: " + " ".join(cmd) + "\n")
        for line in proc.stdout:
            logf.write(line)
            tail.append(line.rstrip())
            if len(tail) > 20:
                tail.pop(0)
            m = re.match(r"^\[([a-z0-9]+)", line.strip())
            if m and m.group(1) not in ("run", "extract", "qc"):
                stage = m.group(1)
                bar.set_description(f"gpu{gpu} {sid}: {stage}")
    rc = proc.wait()
    with proc_registry["lock"]:
        proc_registry["procs"].discard(proc)
    ok = rc == 0
    bar.set_description(f"gpu{gpu} {sid}: {'done' if ok else f'FAIL(rc={rc})'}")
    return sid, ok, tail


def run_batch(scenes, engines, gpus=None, replace=False):
    """Returns (n_ok, n_failed). Never raises on task failures."""
    engines = list(engines)
    scenes = [int(s) for s in scenes]
    unknown = [e for e in engines if e not in ENGINE_INDEX]
    if unknown:
        raise SystemExit(f"unknown engine(s): {unknown}")
    if not scenes:
        raise SystemExit("no scenes given")

    reasons = _preflight_engines(engines)
    for e, r in reasons.items():
        print(f"[preflight] SKIP {e}: {r}")
    engines = [e for e in engines if e not in reasons]
    if not engines:
        raise SystemExit("no engines left after preflight")

    if gpus:
        n_gpu = _gpu_count()
        bad = [g for g in gpus if g < 0 or g >= n_gpu]
        if bad:
            raise SystemExit(f"gpu index(es) {bad} out of range (0..{n_gpu - 1})")
        gpus = list(gpus)
    else:
        n_gpu = _gpu_count()
        gpus = list(range(n_gpu)) if n_gpu else [0]

    run_id = time.strftime("run-%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_ROOT, f"reconstruct-{run_id}")
    os.makedirs(log_dir, exist_ok=True)

    frames_by_scene = _prepare_frames(scenes, gpus, replace)
    tasks = [(s, e) for s in scenes for e in engines]

    free_gpus = queue.Queue()
    for g in gpus:
        free_gpus.put(g)
    slots = list(range(min(len(gpus), len(tasks))))
    bars = [tqdm(total=1, position=i, bar_format="{desc}", leave=False) for i in slots]
    sem = {e: threading.Semaphore(1) for e in SERIAL_ENGINES}
    registry = {"lock": threading.Lock(), "procs": set()}

    def work(task):
        scene, engine = task
        gpu = free_gpus.get()
        s = sem.get(engine)
        if s is not None:
            s.acquire()
        try:
            log_path = os.path.join(log_dir,
                                    f"scene{scene:04d}_{ENGINE_INDEX[engine]:02d}.log")
            return _run_task(scene, engine, gpu, frames_by_scene[scene],
                             log_path, bars[gpus.index(gpu)], registry)
        finally:
            if s is not None:
                s.release()
            free_gpus.put(gpu)

    results, failed = [], []
    try:
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = [pool.submit(work, t) for t in tasks]
            for fut in as_completed(futures):
                sid, ok, tail = fut.result()
                results.append((sid, ok))
                if not ok:
                    failed.append((sid, tail))
    except KeyboardInterrupt:
        with registry["lock"]:
            for p in registry["procs"]:
                p.terminate()
        for b in bars:
            b.close()
        print(f"[batch] interrupted; {len(results)}/{len(tasks)} tasks completed")
        raise SystemExit(130)

    n_ok = sum(1 for _, ok in results if ok)
    print(f"[batch] done {n_ok}/{len(tasks)} tasks, logs: {log_dir}")
    for sid, tail in failed:
        last = next((l for l in reversed(tail) if l.strip()), "")
        print(f"[FAIL] {sid} ({last[:200]}) -> {os.path.join(log_dir, sid + '.log')}")
    return n_ok, len(failed)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="batch reconstruction")
    ap.add_argument("--scene", nargs="+", type=int, required=True)
    ap.add_argument("--engine", nargs="+", required=True)
    ap.add_argument("--gpu", nargs="*", type=int, default=None)
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args()
    ok, failed = run_batch(a.scene, a.engine, a.gpu, a.replace)
    sys.exit(1 if failed else 0)
