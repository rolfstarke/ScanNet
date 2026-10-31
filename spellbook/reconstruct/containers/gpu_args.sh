#!/usr/bin/env bash
# Emit `docker run` args (one per line) exposing NVIDIA GPUs to a container.
# Usage: gpu_args.sh <physical_gpu_index>
# Docker here is the snap package: its daemon sees the host only under
# /var/lib/snapd/hostfs, and the NVIDIA runtime hook and --gpus do not work, so every
# GPU node and the driver userspace are passed explicitly. All GPU nodes are exposed;
# the leased GPU is pinned via CUDA_VISIBLE_DEVICES=<index>.
set -euo pipefail
GPU="${1:?usage: gpu_args.sh <physical_gpu_index>}"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "gpu_args.sh: bad gpu index '$GPU'" >&2; exit 2; }
[ -e "/dev/nvidia$GPU" ] || { echo "gpu_args.sh: /dev/nvidia$GPU missing" >&2; exit 3; }
HOSTFS=/var/lib/snapd/hostfs

for dev in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
  [ -e "$dev" ] && printf '%s\n' --device "$dev"
done

ldconfig -p | awk '/=> \//{print $1, $NF}' | grep -E '^lib(cuda|nvidia|nvoptix|nvcuvid)' |
  grep -v 'libnvidia-container' | sort -u |
while read -r soname path; do
  real=$(readlink -f "$path"); [ -f "$real" ] || continue
  printf '%s\n' -v "$HOSTFS$real:/usr/lib/x86_64-linux-gnu/$soname:ro"
done
for bin in /usr/bin/nvidia-smi /usr/bin/nvidia-debugdump; do
  [ -x "$bin" ] && printf '%s\n' -v "$HOSTFS$bin:$bin:ro"
done

printf '%s\n' -e "CUDA_VISIBLE_DEVICES=$GPU" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
