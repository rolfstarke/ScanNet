# setup-sam3dis: exact commands that worked

Env: /data/sam3dis/conda/envs/sam3dis (py3.12). HF cache: /data/sam3dis/hf_cache. Repo: /data/sam3 (untouched except ignored sam3.egg-info; excluded in .git/info/exclude).

```
source /usr/lib/miniconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=/data/sam3dis/conda/pkgs   # default ~/.conda/pkgs cache was corrupt
conda create -y -p /data/sam3dis/conda/envs/sam3dis python=3.12
conda activate /data/sam3dis/conda/envs/sam3dis
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e /data/sam3
pip install "numpy>=1.26,<2"            # pip -e pulled numpy 2.x; sam3 needs <2
pip install "setuptools<81"             # sam3/model_builder imports pkg_resources
pip install einops ninja
pip install pycocotools opencv-python-headless scikit-image scikit-learn scipy hydra-core omegaconf open_clip_torch openai psutil pandas matplotlib torchmetrics submitit
# a concurrent install left torch 2.15 files mixed in; fixed with:
pip uninstall -y torch torchvision
pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip check   # No broken requirements found.

# weights (token loaded from secrets file, never printed)
set -a; . /data/ScanNet/spellbook/.secrets.env; set +a
export HF_HOME=/data/sam3dis/hf_cache
hf download facebook/sam3 sam3.pt       # -> snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt (3.45 GB)
hf download facebook/sam3 config.json

# smoke (GPU 4 only)
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4
python -I -c "import sam3; from sam3.model_builder import build_sam3_image_model; m=build_sam3_image_model(device='cuda'); print(type(m).__name__)"
# -> Sam3Image, 840.5M params, RTX A4000, torch 2.10.0+cu128, numpy 1.26.4
```

# adapter/test deps (2026-10-30): spellbook common.py needs open3d; tests need pytest
pip install pytest "open3d==0.19.0" "plyfile<1.1" "numpy>=1.26,<2"   # plyfile>=1.1.5 needs numpy 2
pip check   # No broken requirements found.
