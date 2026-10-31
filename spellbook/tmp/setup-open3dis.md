# Open3DIS setup log (CUDA exts built, smoke ok)

Repo: /data/Open3DIS (upstream VinAIResearch/Open3DIS @ bc2a3b3; fork rolfstarke/Open3DIS not reachable, no spellbook-baseline patches applied)
Env: /data/open3dis/conda/envs/open3dis (python 3.8)
Weights: /data/pretrains/foundation_models/{sam_vit_h_4b8939.pth, groundingdino_swint_ogc.pth}

## Commands that worked
- conda create -y -p /data/open3dis/conda/envs/open3dis python=3.8
- CONDA_PKGS_DIRS=/data/open3dis/conda/pkgs conda install -y -c nvidia cuda-nvcc=11.3.58 cuda-libraries-dev=11.3.0  (nvcc 11.3.58)
- CONDA_PKGS_DIRS=/data/open3dis/conda/pkgs conda install -y -c conda-forge "gxx_linux-64=9"  (host g++ 9.5, CUDA 11.3 max is 10)
- pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
- pip install spconv-cu113==2.1.25
- pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
- pip install -e segmenter2d/segment_anything  (ok)
- GroundingDINO: pip install --no-build-isolation -e segmenter2d/GroundingDINO with PATH=/usr/bin:/bin (CPU ext only; CUDA ext not built)
- pip install -r segmenter3d/ISBNet/requirements.txt scikit-image opencv-python imageio plyfile supervision==0.11.1 (ok)
- Weights: curl SAM ViT-H and GroundingDINO swint_ogc from the URLs in docs/INSTALL.md (ok)

## Fixes applied (later)
- conda install -p /data/open3dis/conda/envs/open3dis -c conda-forge sysroot_linux-64=2.17 gcc_linux-64=9 gxx_linux-64=9 (sysroot 2.17 avoids Ubuntu 24.04 linux/types.h __s128 error)
- Env vars for builds: CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=8.6 CC/CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-{gcc,g++} NVCC_PREPEND_FLAGS="--compiler-bindir=$CXX" CPATH=<sparsehash + cuda header dirs>
- cd segmenter3d/ISBNet && python setup.py build_ext develop (ok, BUILD_RC=0)
- cd segmenter2d/GroundingDINO: pip install --no-build-isolation --no-deps -e . (CUDA ext ok)
- gdown --id 1ZEZgQeT6dIakljSTx4s5YZM0n2rwC3Kw -> /data/pretrains/scannet200/head_scannetv2_200_val.pth (123 MB)
- pip install git+https://github.com/facebookresearch/ov-seg@36f49d496714998058d115ffb6172d9d84c59065#egg=clip&subdirectory=third_party/CLIP (ok)
- clip.load('ViT-L/14@336px') -> ~/.cache/clip/ViT-L-14-336px.pt (cached)

## Old blockers (resolved above except where noted)
- ISBNet CUDA ext (isbnet.ops.ops) fails in nvcc 11.3: linux/types.h "expected initializer before '__s128'" (system and conda sysroot headers both).
- GroundingDINO CUDA ext fails for the same reason; falls back to CPU build.
- ISBNet build needs sparsehash (no sudo): worked around with headers from github.com/sparsehash/sparsehash plus a hand-written sparseconfig.h under /data/open3dis/sparsehash-include (not a clean install).
- CUDA headers (cuda_runtime.h, cusparse.h, cublas_v2.h, curand.h, cusolver) taken from pip wheels nvidia-*-cu11 under /data/open3dis/cudart117 and /data/open3dis/cudalibs_hdr (conda cudatoolkit-dev install stalled).
- CLIP (ov-seg third_party/CLIP) not installed yet. ISBNet head_scannetv2_200_val.pth not downloaded (gdrive id 1ZEZgQeT6dIakljSTx4s5YZM0n2rwC3Kw).

## Smoke (CUDA_VISIBLE_DEVICES=4, from /data/Open3DIS)
- torch 1.12.1+cu113, cuda True, RTX A4000
- spconv, torch_scatter, segment_anything, groundingdino, open3dis: import OK
- isbnet, isbnet.ops, groundingdino._C, clip: OK after fixes (torch 1.12.1+cu113, RTX A4000)

Upstream bc2a3b3 tracks build artifacts; restored via git checkout after builds. Fork rolfstarke/Open3DIS confirmed absent.
