"""CLI: python -m spellbook.reconstruct.run --scene 9004 --engine open3d

Pipeline: shared frames -> allocate slot -> reconstruct -> align/meshes/sens -> score (scene9004).
"""
import argparse
import importlib
import os
import shutil
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
from . import (SCANS_DIR, engine_run_range, frames_pool_dir, parse_scan_id, scan_dir, scan_id,
               svo_path)
from . import cleanup, extract as extract_mod
from . import finalize, scannet
from .geometry_score import DEFAULT_CONFIG, load_config, score_scan

VALID_POLICIES = ("managed", "cpu", "blocked")
BENCHMARK_SCENE = 9004


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
        if int(doc.get("distance_threshold_mm", -1)) != int(cfg.get("distance_threshold_mm", -2)):
            return False
        s = float(doc["geometry_score"])
        return np.isfinite(s) and 0.0 <= s <= 100.0
    except Exception:
        return False


def _read_score(path):
    with open(path) as f:
        return float(yaml.safe_load(f)["geometry_score"])


def _pipeline(work, root, sid, info, engine_name, mesh_native, poses, keep, convention):
    print(f"[engine:{engine_name}] mesh={mesh_native} frames={len(keep)} conv={convention}")
    if not keep or len(poses) != len(keep):
        sys.exit("engine returned inconsistent poses/keep")

    aligned, aligned_poses = scannet.align_mesh(mesh_native, poses, info["gravity"],
                                                work, convention)
    clean, clean2 = scannet.make_meshes(aligned, root, sid)
    axis = scannet.axis_alignment_matrix(clean2)
    print(f"[scannet] _vh_clean={os.path.basename(clean)} _vh_clean_2={os.path.basename(clean2)}")

    sens_path, width, height = finalize.finalize(root, sid, os.path.join(work, "frames"),
                                                 info["K"], aligned_poses, keep)
    finalize.write_txt(os.path.join(root, f"{sid}.txt"), info["K"], len(keep), axis,
                       os.environ.get("SCENE_TYPE", "Custom"), width, height)
    print(f"[finalize] {os.path.basename(sens_path)} + {sid}.txt ({len(keep)} frames)")

    segs = scannet.run_segmentator(clean2)
    print(f"[segmentator] {os.path.basename(segs)}")


def _prepare_shared_frames(scene, replace):
    pool = frames_pool_dir(scene)
    with exclusive_lock(frames_lock_path(scene)):
        if scene == BENCHMARK_SCENE and not extract_mod.frames_complete(pool):
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

    if retained == 1 or scene != BENCHMARK_SCENE:
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

    # evict worst
    scored.sort(key=lambda c: (c[4], c[5], c[1]))
    _, run_num, sid, root, scv, _ = scored[0]
    print(f"[evict] {sid} score={scv:.2f}")
    with exclusive_lock(scan_lock_path(sid)):
        cleanup.purge_scan_predictions(sid)
        cleanup.remove_scan_dir(root)
    return run_num, sid, root


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

            if policy == "managed":
                settings = load_settings()
                with gpu_lease(settings["gpu_pool"], settings["scannet_root"]) as lease:
                    mesh_native, poses, keep, convention = engine.reconstruct(
                        work, root, gpu=lease.index, lease_fd=lease.fileno())
                    _pipeline(work, root, sid, info, args.engine, mesh_native, poses,
                              keep, convention)
            else:
                mesh_native, poses, keep, convention = engine.reconstruct(work, root)
                _pipeline(work, root, sid, info, args.engine, mesh_native, poses,
                          keep, convention)

            if args.scene == BENCHMARK_SCENE:
                try:
                    score_scan(sid)
                except Exception as ex:
                    print(f"[geometry] FAILED {sid}: {ex}")
                    sys.exit(1)

            print(f"[done] {sid}")


if __name__ == "__main__":
    main()
