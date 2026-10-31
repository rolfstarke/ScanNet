# OpenYOLO3D setup log

Env: /data/openyolo3D/conda/envs/openyolo3d (CPython 3.10.9, torch 1.12.1+cu113, numpy 1.24.2)
CUDA 11.3: /data/openyolo3D/cuda-11.3 (conda cuda-toolkit, lib64 -> lib symlink)
Host compiler: /data/openyolo3D/toolchain-gcc9-s217 (gxx/gcc 9, sysroot 2.17, openblas 0.3.21), pkgs cache /data/openyolo3D/conda/pkgs
Weights: /data/OpenYOLO3D/pretrained/checkpoints/ (scannet200_val.ckpt, yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth)
Masks: /data/openYOLO3D/output/scannet200/{scannet200_masks,scannet200_ground_truth_masks}; /data/OpenYOLO3D/output -> /data/openYOLO3D/output (symlink, excluded in .git/info/exclude)

## Downloads
    PYTHONPATH=/data/openyolo3D/tools python3 -m gdown 1FneLaYaClWDO51L9lIvlTQbheh5SfOFD -O OpenYOLO3D.zip   # gdown in /data/openyolo3D/tools
    unzip OpenYOLO3D.zip (into an empty dir), then mv checkpoints/* -> pretrained/checkpoints/, mv output/scannet200 -> /data/openYOLO3D/output/

## Toolchain
    conda create -y -p /data/openyolo3D/cuda-11.3 -c nvidia/label/cuda-11.3.1 cuda-toolkit
    conda create -y -p /data/openyolo3D/toolchain-gcc9-s217 -c conda-forge --override-channels "gxx_linux-64=9" "gcc_linux-64=9" "sysroot_linux-64=2.17" "openblas=0.3.21"
    (Do NOT install gxx into the python env: conda-forge swapped its CPython for GraalPy.)

## Env
    conda create -y -p /data/openyolo3D/conda/envs/openyolo3d python=3.10.9 pip
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html   # resolves to 2.1.0+pt112cu113
    pip install pytorch-lightning==1.7.2 black==21.4b2 cloudpickle==3.0.0 future hydra-core==1.0.5 "pycocotools>=2.0.2" pydot iopath==0.1.7 loguru albumentations
    pip install --no-deps numpy==1.24.2 ; re-pin torch/torchvision with --no-deps and albumentations==1.2.1 opencv-python-headless==4.7.0.72 (constraints file)
    pip uninstall albucore ; setuptools==65.6.3 (torch cpp_extension needs pkg_resources)

## Builds (CUDA_HOME=/data/openyolo3D/cuda-11.3, CC/CXX=gcc9, NVCC_PREPEND_FLAGS="-ccbin <gxx9>", TORCH_CUDA_ARCH_LIST=8.6)
    MinkowskiEngine @02fc608 (copy of gitlink dir): python setup.py install --force_cuda --blas=openblas --blas_include_dirs=$TC/include --blas_library_dirs=$TC/lib
    pointnet2 (copy): python setup.py install
    ScanNet Segmentator @3e57265 (copy): make
    detectron2 @710e779: pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf
    Mask3D (copy): pip install -c constraints --no-build-isolation <copy>
    YOLO-World (copy): pip install --no-deps <copy>; then supervision==0.19.0 transformers==4.30.2 tokenizers==0.13.3 (pyproject needs torch>=2.1, incompatible with the torch 1.12 pin)
    mmyolo==0.6.0 mmdet==3.0.0 plyfile openmim ; MMCV_WITH_OPS=1 pip install --no-build-isolation mmcv==2.0.0
    open3d==0.19.0 (--no-deps) + plotly==5.14.1 dash==2.9.3 addict pyquaternion pyviz3d==0.2.28 plyfile trimesh==3.14.0 imageio==2.21.1 pandas==2.0.0 scikit-learn==1.1.2 scipy==1.9.0 matplotlib==3.7.1 scikit-image==0.20.0
    pillow==9.1.0 (--no-deps)

## Smoke (GPU 4)
    cd /data/OpenYOLO3D && CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=/data/openyolo3D/cuda-11.3/lib:/data/openyolo3D/toolchain-gcc9-s217/lib PYTHONPATH=/data/OpenYOLO3D python -c "import torch, MinkowskiEngine, open3d; from utils import OpenYolo3D"
    Result: torch 1.12.1+cu113 cuda True (RTX A4000); ME 0.5.4; open3d 0.19.0; OpenYolo3D import OK; CLIPTextModel import OK.

## Open problems
- Runtime needs LD_LIBRARY_PATH for CUDA 11.3 libs (libcusparse.so.11) and openblas.
- YOLO-World installed with --no-deps (pyproject torch>=2.1 conflicts with the required torch 1.12.1 pin).
- Full inference on a scene not yet run (smoke only).
- Pinned build of mmcv/detectron2 compiled from source; first import took long.
