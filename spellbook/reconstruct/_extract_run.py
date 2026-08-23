"""Child process: extract one SVO under CUDA_VISIBLE_DEVICES=<leased GPU>.

Must never set sdk_gpu_id. Parent acquires the GPU lease and frame lock.
"""
import argparse
import os
import sys

import numpy as np

# CUDA device must be pinned before any ZED import.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    sys.exit("CUDA_VISIBLE_DEVICES must be set by the parent before extraction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svo", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--lease-fd", type=int, default=None)
    args = ap.parse_args()

    lease_file = None
    if args.lease_fd is not None:
        lease_file = os.fdopen(args.lease_fd, "r")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reconstruct import extract as extract_mod
    info = extract_mod.extract(args.svo, args.work_dir)
    poses = info["poses"]
    states = info["states"]
    ok_idx = [i for i, s in enumerate(states) if s == "OK"]
    probe = ok_idx[:60] if len(ok_idx) >= 10 else list(range(min(60, len(poses))))
    if len(probe) < 10:
        sys.exit(f"pose gate failed: only {len(probe)} probe frames")
    P = poses[probe]
    span = float(np.linalg.norm(P[-1, :3, 3] - P[0, :3, 3]))
    path = float(np.sum(np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1)))
    print(f"[extract-gate] n={len(poses)} probe={len(probe)} span={span:.3f}m path={path:.3f}m "
          f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if span < 0.01 and path < 0.05:
        sys.exit(f"pose gate failed: constant poses span={span:.4f} path={path:.4f}")
    if lease_file is not None:
        lease_file.close()


if __name__ == "__main__":
    main()
