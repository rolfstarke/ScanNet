"""GPU distribution probe for prediction methods (--gpu-check).

Runs inside each model's own conda python with the exact child environment the normal
prediction runner provides (CUDA_VISIBLE_DEVICES=<physical lease index>,
SPELLBOOK_GPU_LEASE_FD=<lease fd> via pass_fds). It verifies the wiring end-to-end and
creates a real CUDA context on the assigned GPU, then holds briefly -- it never imports
model code, reads scenes, or writes outputs.
"""
import argparse
import json
import os
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--hold", type=float, default=5.0)
    args = ap.parse_args()

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parts = [p.strip() for p in cvd.split(",") if p.strip()]
    if len(parts) != 1:
        sys.exit(f"CUDA_VISIBLE_DEVICES must name exactly one GPU, got {cvd!r}")
    physical = int(parts[0])
    if physical <= 0:
        sys.exit(f"physical GPU must be in the managed pool (> 0), got {physical}")

    fd_env = os.environ.get("SPELLBOOK_GPU_LEASE_FD")
    if not fd_env:
        sys.exit("SPELLBOOK_GPU_LEASE_FD not set")
    try:
        fd = int(fd_env)
        os.fstat(fd)
    except (ValueError, OSError) as ex:
        sys.exit(f"lease fd {fd_env!r} not open: {ex}")

    if not torch.cuda.is_available():
        sys.exit("torch.cuda not available")
    if torch.cuda.device_count() != 1:
        sys.exit(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")

    device = torch.device("cuda", 0)
    x = torch.zeros(1, device=device)
    torch.cuda.synchronize()
    info = dict(method=args.method, pid=os.getpid(), physical=physical, visible=0,
                device=torch.cuda.get_device_name(0), lease_fd=fd)
    json.dump(info, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()

    time.sleep(args.hold)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()