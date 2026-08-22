"""CLI: python -m spellbook.reconstruct.run --scene 9009 --engine zed

Full pipeline: extract -> engine -> align/meshes/segs -> sens/txt -> qc.
Engines always recompute; frames are extracted once and reused (--replace re-extracts,
--frames points at a shared per-scene frame set prepared by the batch runner).

GPU policy comes from the engine adapter's GPU_POLICY:
  - "managed": acquire a settings-backed cross-process lease (gpu_pool, never GPU 0)
    and forward the physical index + lease descriptor to the adapter;
  - "cpu": no lease, gpu=None (Open3D legacy TSDF is CPU-only);
  - "blocked": engine preflight returns a reason and run exits before any work
    (ZED: SDK default-device path would use user-reserved GPU 0, #18).
Adapter preflight runs before SVO checks, directory creation, or frame setup.
The lease is held through engine, alignment, finalize, segmentator, and QC so the
engine cannot outlive its reservation. CUDA_VISIBLE_DEVICES is cleared before any
ZED SDK work and before GPU discovery so all physical devices are visible.
"""
import argparse
import importlib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from benchmark import load_settings
from . import scan_dir, scan_id, svo_path
from . import extract as extract_mod
from . import finalize, qc, scannet
from utils.gpu import gpu_lease

VALID_POLICIES = ("managed", "cpu", "blocked")


def _pipeline(work, root, sid, svo, info, engine_name, mesh_native, poses, keep,
              convention, no_qc):
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

    if not no_qc:
        fails = qc.run_qc(sid, root)
        if fails:
            print(f"[qc] FAILED metrics: {fails}")
    print(f"[done] {sid}")


def main():
    ap = argparse.ArgumentParser(description="SVO2 -> ScanNet-native scan")
    ap.add_argument("--scene", type=int, required=True, help="9000-range scene number")
    ap.add_argument("--engine", choices=["zed", "metashape", "rtabmap", "isaac", "open3d",
                                         "bundlefusion"], required=True)
    ap.add_argument("--svo", default=None, help="override /data/scannet/custom/raw/<scene>.svo2")
    ap.add_argument("--frames", default=None,
                    help="shared per-scene frames dir (batch mode); frames are never "
                         "re-extracted here")
    ap.add_argument("--replace", action="store_true",
                    help="re-extract frames even if a complete set exists")
    ap.add_argument("--no-qc", action="store_true")
    args = ap.parse_args()

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    engine = importlib.import_module(f"spellbook.reconstruct.engines.{args.engine}")
    policy = engine.GPU_POLICY
    if policy not in VALID_POLICIES:
        sys.exit(f"[{args.engine}] invalid GPU_POLICY {policy!r} (expected one of "
                 f"{VALID_POLICIES})")
    reason = engine.preflight()
    if reason:
        sys.exit(f"[preflight] {args.engine} blocked: {reason}")
    if policy == "blocked":
        sys.exit(f"[{args.engine}] not runnable: {reason}")

    sid = scan_id(args.scene, args.engine)
    root = scan_dir(args.scene, args.engine)
    svo = args.svo or svo_path(args.scene)
    if not os.path.exists(svo):
        sys.exit(f"svo not found: {svo}")
    work = os.path.join(root, "recon")
    os.makedirs(os.path.join(work, "frames"), exist_ok=True)
    with open(os.path.join(work, "cmdline.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")
    with open(os.path.join(work, "svo_path.txt"), "w") as f:
        f.write(svo + "\n")

    print(f"[run] {sid} <- {svo}")
    if args.frames:
        shared = os.path.abspath(args.frames)
        if not extract_mod.frames_complete(shared):
            sys.exit(f"shared frames incomplete: {shared}")
        local = os.path.join(work, "frames")
        if os.path.islink(local):
            os.remove(local)
        elif os.path.isdir(local):
            shutil.rmtree(local)
        os.symlink(shared, local)
        info = extract_mod.ensure_frames(svo, work)
    else:
        info = extract_mod.ensure_frames(svo, work, replace=args.replace)

    if policy == "managed":
        settings = load_settings()
        with gpu_lease(settings["gpu_pool"], settings["scannet_root"]) as lease:
            mesh_native, poses, keep, convention = engine.reconstruct(
                work, root, gpu=lease.index, lease_fd=lease.fileno())
            _pipeline(work, root, sid, svo, info, args.engine, mesh_native, poses,
                      keep, convention, args.no_qc)
    else:
        mesh_native, poses, keep, convention = engine.reconstruct(work, root)
        _pipeline(work, root, sid, svo, info, args.engine, mesh_native, poses,
                  keep, convention, args.no_qc)


if __name__ == "__main__":
    main()
