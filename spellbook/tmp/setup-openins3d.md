# OpenIns3D setup (env /data/openins3d/conda/envs/openins3d)

Env vars for every step: `source /usr/lib/miniconda3/etc/profile.d/conda.sh; export CONDA_PKGS_DIRS=/data/openins3d/conda/pkgs`
Build env: `TORCH_CUDA_ARCH_LIST=8.6 FORCE_CUDA=1 CUDA_HOME=$E CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++`
Pip constraints file (scratchpad): torch==1.13.1, torchvision==0.14.1, torchaudio==0.13.1, numpy==1.23.5

1. `conda create -p $E python=3.9 -y`
2. `conda install -y -c pytorch -c nvidia/label/cuda-11.6.1 -c nvidia pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6`
3. `pip install omegaconf==2.1.1 iopath==0.1.8 gdown==4.6.3 loguru plyfile python-dotenv pyviz3d open3d pytorch-lightning==1.9.5 hydra-core==1.1.2 nltk albumentations scipy matplotlib pillow tqdm imageio` (with constraints)
4. `conda install -c conda-forge gxx_linux-64=9 gcc_linux-64=9 sysroot_linux-64=2.17`
5. `conda install -c nvidia/label/cuda-11.6.1 -c nvidia libcusolver-dev libcusparse-dev libcurand-dev libcublas-dev`
6. `pip install --no-build-isolation torch_scatter`
7. detectron2 v0.6: `git clone --branch v0.6 --depth 1 https://github.com/facebookresearch/detectron2.git /data/openins3d/detectron2`; `pip install --no-build-isolation --no-deps -e .`; then `pip install fvcore==0.1.5.post20221221 pycocotools yacs termcolor tabulate cloudpickle tensorboard black==21.4b2 future pydot`
8. ODISE: `git clone https://github.com/NVlabs/ODISE.git /data/OpenIns3D/third_party/ODISE`; `pip install --no-build-isolation --no-deps -e .`
9. pytorch3d v0.7.2 (no cu116 conda build exists): `git clone --branch v0.7.2 --depth 1 https://github.com/facebookresearch/pytorch3d.git /data/openins3d/pytorch3d`; `pip install --no-build-isolation --no-deps .`
10. MinkowskiEngine at pinned commit 02fc608: `git clone --recursive https://github.com/NVIDIA/MinkowskiEngine /data/OpenIns3D/third_party/MinkowskiEngine && git -C ... checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`; `conda install -c conda-forge openblas`; `python setup.py install --force_cuda --blas=openblas --blas_include_dirs=$E/include --blas_library_dirs=$E/lib`
11. pointnet2: `cd /data/OpenIns3D/third_party/pointnet2 && python setup.py install`
12. Checkpoint: gdown id 1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B -> /data/OpenIns3D/third_party/scannet200_val.ckpt (sha256 6a0c4347..., 158873495 B); `ln -s` to /data/openins3d/checkpoints/scannet200_val.ckpt
13. Git excludes added to /data/OpenIns3D/.git/info/exclude: third_party/ODISE/, third_party/scannet200_val.ckpt, masks/

Smoke (CUDA_VISIBLE_DEVICES=4, repo dir): detectron2, odise, pytorch3d, MinkowskiEngine, pointnet2, openins3d.mask3d (get_model, prepare_data, map_output_to_pointcloud), openins3d.snap, openins3d.lookup all import.
Upstream 56ccc22 exported to /data/openins3d/src-56ccc22 (git archive, checkout untouched): Snap, Lookup, mask3d names import.

Not done: precomputed masks/detected/*.pt source not found (no link in README, none on disk).
