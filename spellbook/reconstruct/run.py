"""CLI: python -m spellbook.reconstruct.run --scene 9004 --engine open3d

Pipeline: shared frames -> allocate slot -> reconstruct -> align/meshes/sens -> score (scene9004).

Timed region = engine.reconstruct + _pipeline (align, meshes, .sens/.txt, segmentator); it is
written to recon/timing.json and capped at 3x the scene length (runtime rule 2026-09-15,
SCENE_DURATION_S in reconstruct/__init__.py). Lease wait and CAD scoring are outside it;
`--offline-score` skips the inline score so it can run later
(`python -m spellbook.reconstruct.geometry_score --scan-id <sid>`). Adapters with
RUNTIME_CAP_EXEMPT = True (Metashape) are unbounded. Overrun -> rc 124, slot left incomplete
(the allocator reuses it).

Provenance: recon/run.json (schema 1) with engine, scan, issue, variant/preset + effective
config (adapter provenance() hook), commit/dirty, adapter sha256, duration/limit/elapsed.
"""
import argparse
import hashlib
import importlib
import json
import os
import shutil
import signal
import sys
import time

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_SPELLBOOK)
sys.path.insert(0, _SPELLBOOK)
sys.path.insert(0, _REPO_ROOT)

import numpy as np
import yaml

from evaluation.benchmark import load_settings
from utils.gpu import gpu_lease
from utils.scan_lock import exclusive_lock, frames_lock_path, scan_lock_path, scene_engine_lock_path
from . import (SCANS_DIR, SCENE_DURATION_S, engine_run_range, frames_pool_dir, parse_scan_id,
               runtime_cap_s, scan_dir, scan_id, svo_path)
from . import cleanup, extract as extract_mod
from . import finalize, scannet
from .geometry_score import DEFAULT_CONFIG, load_config, score_scan

VALID_POLICIES = ("managed", "cpu", "blocked")
BENCHMARK_SCENE = 9004
RUN_SCHEMA = 1
CAP_EXIT_CODE = 124
# Run-0 replacement on the ten-slot benchmark scene (#53 2026-09-08, verbatim name): purge +
# remove slot 0 under locks and reuse it instead of the normal slot policy.
TARGET_RUN0_ENV = "SPELLBOOK_TARGET_RUN0"


class RuntimeCapExceeded(BaseException):
    """Raised in the main thread by SIGALRM when the timed region overruns the cap.
    BaseException so adapter `except Exception` blocks cannot swallow it; subprocess.run
    kills its child on the way out."""


class runtime_cap:
    """Context manager arming a one-shot SIGALRM after `cap_s` seconds (None = no cap)."""

    def __init__(self, cap_s):
        self.cap_s = cap_s
        self._old = None

    def _fire(self, signum, frame):
        raise RuntimeCapExceeded(f"runtime cap {self.cap_s:.6f}s exceeded")

    def __enter__(self):
        if self.cap_s:
            self._old = signal.signal(signal.SIGALRM, self._fire)
            signal.setitimer(signal.ITIMER_REAL, float(self.cap_s))
        return self

    def __exit__(self, *exc):
        if self.cap_s:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old or signal.SIG_DFL)
        return False


def cap_for(engine, scene):
    """Cap seconds for (adapter module, scene), or None when exempt/unknown."""
    if getattr(engine, "RUNTIME_CAP_EXEMPT", False):
        return None
    return runtime_cap_s(scene)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, doc):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def _run_record(args, engine, sid, svo, shared, cap):
    """Initial recon/run.json document (updated in place when the run ends)."""
    from jobs import _git_info

    prov = {}
    if hasattr(engine, "provenance"):
        prov = dict(engine.provenance() or {})
    adapter = os.path.abspath(engine.__file__)
    git = _git_info(_REPO_ROOT)
    duration = SCENE_DURATION_S.get(args.scene)
    return {
        "schema": RUN_SCHEMA,
        "engine": args.engine,
        "scene": f"scene{args.scene:04d}",
        "scan_id": sid,
        "issue": args.issue,
        "variant": prov.pop("variant", None),
        "config": prov.pop("config", None),
        **prov,
        "commit": git.get("commit"),
        "branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "adapter": os.path.relpath(adapter, _REPO_ROOT),
        "adapter_sha256": _sha256(adapter),
        "svo": svo,
        "frames": shared,
        "job_id": os.environ.get("SPELLBOOK_JOB_ID"),
        "duration_s": duration,
        "limit_s": cap,
        "cap_exempt": bool(getattr(engine, "RUNTIME_CAP_EXEMPT", False)),
        "started": time.time(),
        "status": "running",
    }


def _complete_scan(root, sid, require_score=False):
    need = [
        os.path.join(root, f"{sid}.sens"),
        os.path.join(root, f"{sid}.txt"),
        os.path.join(root, f"{sid}_vh_clean.ply"),
        os.path.join(root, f"{sid}_vh_clean_2.ply"),
        os.path.join(root, "recon", "final_poses.npy"),
    ]
    if not all(os.path.isfile(p) for p in need):
        return False
    segs = [f for f in os.listdir(root) if f.endswith(".segs.json")]
    if not segs:
        return False
    if require_score:
        return _score_comparable(os.path.join(root, "recon", "geometry_score.yaml"))
    return True


def _score_comparable(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)
        cfg = load_config()
        sc = cfg["scenes"]["scene9004"]
        if doc.get("metric") != cfg.get("metric"):
            return False
        if doc.get("reference_sha256") != sc.get("reference_sha256"):
            return False
        if doc.get("visible_voxels_sha256") != sc.get("visible_voxels_sha256"):
            return False
        if int(doc.get("voxel_mm", -1)) != int(cfg.get("voxel_mm", -2)):
            return False
        s = float(doc["mean_bidirectional_distance_cm"])
        acc = float(doc["accuracy_mean_cm"])
        comp = float(doc["completeness_mean_cm"])
        return all(np.isfinite(x) and x >= 0.0 for x in (s, acc, comp))
    except Exception:
        return False


def _read_score(path):
    """Mean bidirectional distance cm; lower is better."""
    with open(path) as f:
        return float(yaml.safe_load(f)["mean_bidirectional_distance_cm"])


def _pipeline(work, root, sid, info, engine_name, mesh_native, poses, keep, convention,
              frames_dir=None):
    """`frames_dir` overrides the shared pool as .sens source (engines that package their own
    per-frame depth, e.g. Metashape native depth maps); same K and frame indices."""
    print(f"[engine:{engine_name}] mesh={mesh_native} frames={len(keep)} conv={convention}")
    if not keep or len(poses) != len(keep):
        sys.exit("engine returned inconsistent poses/keep")

    aligned, aligned_poses = scannet.align_mesh(mesh_native, poses, info["gravity"],
                                                work, convention)
    clean, clean2 = scannet.make_meshes(aligned, root, sid)
    axis = scannet.axis_alignment_matrix(clean2)
    print(f"[scannet] _vh_clean={os.path.basename(clean)} _vh_clean_2={os.path.basename(clean2)}")

    sens_path, width, height = finalize.finalize(root, sid,
                                                 frames_dir or os.path.join(work, "frames"),
                                                 info["K"], aligned_poses, keep)
    finalize.write_txt(os.path.join(root, f"{sid}.txt"), info["K"], len(keep), axis,
                       os.environ.get("SCENE_TYPE", "Custom"), width, height)
    print(f"[finalize] {os.path.basename(sens_path)} + {sid}.txt ({len(keep)} frames)")

    segs = scannet.run_segmentator(clean2)
    print(f"[segmentator] {os.path.basename(segs)}")


def _prepare_shared_frames(scene, replace):
    pool = frames_pool_dir(scene)
    with exclusive_lock(frames_lock_path(scene)):
        if scene == BENCHMARK_SCENE and not extract_mod.frames_complete(pool) \
                and os.path.isdir(extract_mod.LEGACY_9004_FRAMES):
            extract_mod.promote_scene9004_frames()
        if not extract_mod.frames_complete(pool):
            if replace or scene != BENCHMARK_SCENE:
                extract_mod.extract_scenes([scene], replace=replace)
            else:
                sys.exit(f"shared frames missing for scene{scene:04d}; "
                         f"run --extract-frames or promote first")
        if not extract_mod.frames_complete(pool):
            sys.exit(f"shared frames incomplete: {pool}")
        return os.path.abspath(pool)


def _allocate(scene, engine):
    """Return (run_num, sid, root) under scene-engine lock already held."""
    cfg = load_config()
    retained = 10 if scene == BENCHMARK_SCENE else 1
    sc = cfg.get("scenes", {}).get(f"scene{scene:04d}", {})
    retained = int(sc.get("retained_runs", retained))

    if retained == 1 or scene != BENCHMARK_SCENE or os.environ.get(TARGET_RUN0_ENV) == "1":
        run_num = 0
        sid = scan_id(scene, engine, run_num)
        root = scan_dir(scene, engine, run_num)
        with exclusive_lock(scan_lock_path(sid)):
            if os.path.exists(root):
                print(f"[replace] {sid}")
                cleanup.purge_scan_predictions(sid)
                cleanup.remove_scan_dir(root)
        return run_num, sid, root

    # scene9004 ten-slot policy
    candidates = []
    for run_num in range(10):
        sid = scan_id(scene, engine, run_num)
        root = scan_dir(scene, engine, run_num)
        score_path = os.path.join(root, "recon", "geometry_score.yaml")
        if not os.path.exists(root):
            candidates.append(("absent", run_num, sid, root, None, None))
            continue
        if not _complete_scan(root, sid, require_score=False):
            candidates.append(("incomplete", run_num, sid, root, None, None))
            continue
        if not _score_comparable(score_path):
            # try rescore
            try:
                score_scan(sid)
            except Exception as ex:
                print(f"[geometry] rescore failed {sid}: {ex}")
                candidates.append(("failed", run_num, sid, root, None, None))
                continue
        if not _score_comparable(score_path):
            candidates.append(("failed", run_num, sid, root, None, None))
            continue
        scv = _read_score(score_path)
        mtime = os.path.getmtime(os.path.join(root, "recon", "cmdline.txt")) \
            if os.path.isfile(os.path.join(root, "recon", "cmdline.txt")) else time.time()
        candidates.append(("scored", run_num, sid, root, scv, mtime))

    for kind in ("incomplete", "failed", "absent"):
        pool = [c for c in candidates if c[0] == kind]
        if pool:
            pool.sort(key=lambda c: c[1])
            _, run_num, sid, root, _, _ = pool[0]
            if kind != "absent" and os.path.exists(root):
                print(f"[reuse] {sid} ({kind})")
                with exclusive_lock(scan_lock_path(sid)):
                    cleanup.purge_scan_predictions(sid)
                    cleanup.remove_scan_dir(root)
            return run_num, sid, root

    scored = [c for c in candidates if c[0] == "scored"]
    if len(scored) < 10:
        # should not happen
        run_num = 0
        return run_num, scan_id(scene, engine, run_num), scan_dir(scene, engine, run_num)

    # evict worst = highest mean_bidirectional_distance_cm
    scored.sort(key=lambda c: (-c[4], c[5], c[1]))
    _, run_num, sid, root, scv, _ = scored[0]
    print(f"[evict] {sid} mean_bidirectional_distance_cm={scv:.2f}")
    with exclusive_lock(scan_lock_path(sid)):
        cleanup.purge_scan_predictions(sid)
        cleanup.remove_scan_dir(root)
    return run_num, sid, root


def _reconstruct_and_pipeline(engine, args, work, root, sid, info, gpu=None, lease_fd=None):
    """The timed region. Returns elapsed seconds."""
    started = time.monotonic()
    if gpu is None:
        result = engine.reconstruct(work, root)
    else:
        result = engine.reconstruct(work, root, gpu=gpu, lease_fd=lease_fd)
    mesh_native, poses, keep, convention = result[:4]
    frames_dir = result[4] if len(result) > 4 else None
    _pipeline(work, root, sid, info, args.engine, mesh_native, poses, keep, convention,
              frames_dir)
    return time.monotonic() - started


def main():
    ap = argparse.ArgumentParser(description="SVO2 -> ScanNet-native scan")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--engine",
                    choices=["zed", "metashape", "rtabmap", "isaac", "open3d", "bundlefusion"],
                    required=True)
    ap.add_argument("--svo", default=None)
    ap.add_argument("--frames", default=None, help="shared frames dir (batch)")
    ap.add_argument("--replace", action="store_true",
                    help="re-extract shared frames (main extract path only)")
    ap.add_argument("--issue", type=int, default=None,
                    help="GitHub issue number recorded in recon/run.json")
    ap.add_argument("--offline-score", action="store_true",
                    help="skip the inline CAD score (score later with geometry_score)")
    args = ap.parse_args()

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    engine = importlib.import_module(f"spellbook.reconstruct.engines.{args.engine}")
    policy = engine.GPU_POLICY
    if policy not in VALID_POLICIES:
        sys.exit(f"[{args.engine}] invalid GPU_POLICY {policy!r}")
    reason = engine.preflight()
    if reason:
        sys.exit(f"[preflight] {args.engine} blocked: {reason}")
    if policy == "blocked":
        sys.exit(f"[{args.engine}] not runnable: {reason}")

    svo = args.svo or svo_path(args.scene)
    if not os.path.exists(svo):
        sys.exit(f"svo not found: {svo}")

    # frames first
    if args.frames:
        shared = os.path.abspath(args.frames)
        if not extract_mod.frames_complete(shared):
            sys.exit(f"shared frames incomplete: {shared}")
    else:
        shared = _prepare_shared_frames(args.scene, args.replace)

    cap = cap_for(engine, args.scene)
    with exclusive_lock(scene_engine_lock_path(args.scene, args.engine)):
        run_num, sid, root = _allocate(args.scene, args.engine)
        print(f"[run] {sid} <- {svo}")
        with exclusive_lock(scan_lock_path(sid)):
            work = os.path.join(root, "recon")
            os.makedirs(work, exist_ok=True)
            with open(os.path.join(work, "cmdline.txt"), "w") as f:
                f.write(" ".join(sys.argv) + "\n")
            with open(os.path.join(work, "svo_path.txt"), "w") as f:
                f.write(svo + "\n")

            local = os.path.join(work, "frames")
            if os.path.islink(local):
                os.remove(local)
            elif os.path.isdir(local):
                shutil.rmtree(local)
            os.symlink(shared, local)
            info = extract_mod.load_info(shared)

            record = _run_record(args, engine, sid, svo, shared, cap)
            run_json = os.path.join(work, "run.json")
            _write_json(run_json, record)
            shutil.copy2(engine.__file__, os.path.join(work, f"adapter_{args.engine}.py"))
            print(f"[cap] {sid} limit="
                  + (f"{cap:.6f}s" if cap else "none (exempt/unknown scene length)")
                  + f" variant={record.get('variant')}")

            from utils.compute_time import write_reconstruction_timing

            # The cap arms only once the lease is held: lease wait is not charged.
            try:
                if policy == "managed":
                    settings = load_settings()
                    # GPU_POOL: an adapter may pin its lease (bundlefusion's ordered pair).
                    pool = getattr(engine, "GPU_POOL", settings["gpu_pool"])
                    with gpu_lease(pool, settings["scannet_root"]) as lease:
                        record["gpu"] = lease.index
                        with runtime_cap(cap):
                            elapsed = _reconstruct_and_pipeline(
                                engine, args, work, root, sid, info,
                                gpu=lease.index, lease_fd=lease.fileno())
                else:
                    with runtime_cap(cap):
                        elapsed = _reconstruct_and_pipeline(engine, args, work, root, sid,
                                                            info)
            except (RuntimeCapExceeded, *getattr(engine, "CAP_ERRORS", ())) as ex:
                record.update(status="timeout", ended=time.time(), cap_pass=False)
                _write_json(run_json, record)
                print(f"[cap] FAIL {sid}: {ex}; slot left incomplete")
                sys.exit(CAP_EXIT_CODE)

            write_reconstruction_timing(root, sid, args.engine, elapsed)
            cap_pass = cap is None or elapsed <= cap
            record.update(status="complete" if cap_pass else "timeout", ended=time.time(),
                          elapsed_s=elapsed, cap_pass=cap_pass)
            if hasattr(engine, "provenance_after"):
                record.update(engine.provenance_after(work) or {})
            _write_json(run_json, record)
            print(f"[cap] {sid} elapsed={elapsed:.2f}s "
                  + (f"<= {cap:.6f}s PASS" if cap and cap_pass else
                     f"> {cap:.6f}s FAIL" if cap else "(no cap)"))
            if not cap_pass:
                sys.exit(CAP_EXIT_CODE)

            if args.scene == BENCHMARK_SCENE:
                if args.offline_score:
                    print(f"[geometry] offline: python -m spellbook.reconstruct.geometry_score "
                          f"--scan-id {sid}")
                else:
                    try:
                        score_scan(sid)
                    except Exception as ex:
                        print(f"[geometry] FAILED {sid}: {ex}")
                        sys.exit(1)

            print(f"[done] {sid}")


if __name__ == "__main__":
    main()
