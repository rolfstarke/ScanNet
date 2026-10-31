# Mosaic3D setup log

## Weights (Hugging Face junhalee/Mosaic3D, public)
    mkdir -p /data/mosaic3d/ckpts
    curl -L -C - -o /data/mosaic3d/ckpts/sc.ckpt        https://huggingface.co/junhalee/Mosaic3D/resolve/main/sc.ckpt
    curl -L -C - -o /data/mosaic3d/ckpts/spunet34c.ckpt https://huggingface.co/junhalee/Mosaic3D/resolve/main/spunet34c.ckpt
Filename mapping: HF names already equal the expected names (sc.ckpt, spunet34c.ckpt); no symlinks needed.
Sizes: sc.ckpt 2735243146 B, spunet34c.ckpt 2735727008 B.

## Env
    source /usr/lib/miniconda3/etc/profile.d/conda.sh
    conda create -y -p /data/mosaic3d/conda/envs/mosaic3d -c pytorch -c nvidia -c conda-forge python=3.10 "pytorch=2.4.*" "torchvision=0.19.*" "pytorch-cuda=12.1"
Retry (first attempt failed: shared ~/.conda/pkgs cache cleaned by another agent). Working command:
    CONDA_PKGS_DIRS=/data/mosaic3d/conda/pkgs conda create -y -p /data/mosaic3d/conda/envs/mosaic3d -c pytorch -c nvidia -c conda-forge python=3.10 "pytorch=2.4.*" "torchvision=0.19.*" "pytorch-cuda=12.1"
    -> torch 2.4.1, CUDA 12.1 build

## pip (env python)
    P=/data/mosaic3d/conda/envs/mosaic3d/bin
    $P/pip install "numpy<2" "lightning==2.2" "torchmetrics>=0.11.4" "hydra-core==1.3.2" hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 rootutils rich omegaconf natsort einops tqdm scipy plyfile easydict pyyaml imageio matplotlib scikit-image transformers open_clip_torch timm addict overrides fire gdown requests jaxtyping "open3d==0.18.0" spconv-cu120 torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
    $P/pip install "setuptools==69.5.1"          # pkg_resources needed by lightning 2.2
    $P/pip install "git+https://github.com/openai/CLIP.git"   # `from clip import clip` in src/models/utils/clip_models.py

Not installed (not needed for sc.ckpt load): cuml-cu12 (DBSCAN in mask_language_module), warpconvnet, pointops.

## Smoke (CPU, no GPU used)
    cd /data/ScanNet/spellbook/tmp/setup-mosaic3d-smoke && CUDA_VISIBLE_DEVICES= HYDRA_FULL_ERROR=1 $P/python -I load_sc.py
    -> TYPE DenseLanguageLitModule; LOADED keys 1064 missing 0 unexpected 0 (strict match)
    spunet34c.ckpt: 1654 keys, net.backbone.* present (not load-tested into a model)
