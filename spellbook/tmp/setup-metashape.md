# Metashape 2.3.1 env setup (exact working commands)

Adapter imports: Metashape, numpy (top level); open3d (lazy, inside function). No cv2.

## 1. conda env (python 3.11, numpy only; conda-forge)
mkdir -p /data/zed-metashape/conda/pkgs
source /usr/lib/miniconda3/etc/profile.d/conda.sh
CONDA_PKGS_DIRS=/data/zed-metashape/conda/pkgs conda create -y -p /data/zed-metashape/conda/env python=3.11 numpy -c conda-forge --override-channels

Notes: the shared ~/.conda/pkgs cache stalled under concurrent conda jobs, so a private pkgs dir was used.
A first attempt with opencv pulled CUDA packages and was dropped (adapter does not import cv2).

## 2. Metashape wheel (official URL is lowercase; capitalized 404s)
mkdir -p /data/zed-metashape/wheel/dl && cd /data/zed-metashape/wheel/dl
curl -sL --fail -o metashape-2.3.1-cp39.cp310.cp311.cp312.cp313-abi3-linux_x86_64.whl \
  https://download.agisoft.com/metashape-2.3.1-cp39.cp310.cp311.cp312.cp313-abi3-linux_x86_64.whl

## 3. Install into env
cd /data/zed-metashape
/data/zed-metashape/conda/env/bin/python -m pip install -q --no-cache-dir ./wheel/dl/metashape-2.3.1-cp39.cp310.cp311.cp312.cp313-abi3-linux_x86_64.whl open3d
# open3d resolved to 0.20.0 via pip (conda-forge open3d not used)

## 4. License activation (key read from .secrets.env, never printed)
cd /data/zed-metashape
set -a && . /data/ScanNet/spellbook/.secrets.env && set +a
/data/zed-metashape/conda/env/bin/python -I -c '
import os, Metashape
Metashape.License().activate(os.environ["METASHAPE_LICENSE"])
print("activated:", Metashape.app.activated)
print("version:", Metashape.app.version)'
# result: "License activated" / activated: True / version: 2.3.1

## Verified versions
python 3.11.18, numpy 2.4.6, open3d 0.20.0, Metashape 2.3.1
