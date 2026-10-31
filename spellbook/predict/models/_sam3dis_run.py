"""Runs sam3dis (SAM 3 text-prompted 2D masks lifted onto ScanNet superpoints, training-free)
inside the `sam3dis` conda env, writing ScanNet benchmark instance-segmentation format
(scan-net.org).

The method lives in the sam3dis repo (/data/sam3dis, rebuilt from issue #69); this script only
adapts it: parameter plumbing (`{}` = R4 baseline `sam3dis-r4`), the lift cache location
<scannet_root>/derived/cache/sam3dis/<scene>_i<interval>_tg<tau_geom>.pt, and the submission.
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    _benchmark_spec, add_run_args, load_overrides, scene_id_from_pointcloud,
    write_scannet_submission,
)

SAM3DIS_REPO = "/data/sam3dis"
HF_HOME = "/data/sam3dis/hf_cache"


def cache_dir_for(pointcloud):
    """<scannet_root>/derived/cache/sam3dis, with scannet_root taken from the scene path
    <scannet_root>/scans/<scene>/<scene>_vh_clean_2.ply."""
    root = pathlib.Path(pointcloud).resolve().parents[2]
    return os.path.join(root, "derived", "cache", "sam3dis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True, help="scene mesh (<scene>_vh_clean_2.ply)")
    ap.add_argument("--frames", default=None,
                    help="extracted frames dir (frames.py); only needed on a lift-cache miss")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="predictions output dir")
    ap.add_argument("--benchmark", default="ScanNet200", choices=["ScanNet20", "ScanNet200"])
    add_run_args(ap)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, SAM3DIS_REPO)
    from sam3dis.params import ALLOWED
    from sam3dis.pipeline import run_scene

    spec = _benchmark_spec(args.benchmark)
    overrides = load_overrides(args.parameters_json, ALLOWED)
    scene_id = scene_id_from_pointcloud(args.pointcloud)
    os.makedirs(args.out, exist_ok=True)
    print(f"[INFO] {scene_id} run_id={args.run_id} overrides={overrides}", flush=True)

    p, instances = run_scene(args.pointcloud, args.frames, list(args.classes), overrides,
                             cache_dir_for(args.pointcloud),
                             log=lambda msg: print(msg, flush=True))
    n_written = write_scannet_submission(args.out, scene_id, args.classes, instances,
                                         p["min_mask_points"], spec)
    print(f"[INFO] Wrote {n_written} instances to {args.out} ({len(instances)} rows, "
          f"engine={p['proposal_engine']})")


if __name__ == "__main__":
    main()
