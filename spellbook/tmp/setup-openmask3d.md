# OpenMask3D setup log

Env: `/data/openmask3d/conda/envs/openmask3d` (py3.8.5 CPython, CUDA 11.3, gcc/g++ 9, torch 1.12.1+cu113)
Repo: `/data/openmask3d`, branch `spellbook-baseline` (commit `fb5df6b`, depth-to-color SAM prompt patch, issue 46)
Weights: `/data/openmask3d/resources/scannet200_model.ckpt`, `/data/openmask3d/resources/sam_vit_h_4b8939.pth`

## Commands that worked
```
printf 'conda/\nresources/\n' >> /data/openmask3d/.git/info/exclude
conda create -y -p /data/openmask3d/conda/envs/openmask3d -c nvidia/label/cuda-11.3.1 -c conda-forge --override-channels 'python=3.8.5=*_cpython' cuda-toolkit=11.3.1 gxx_linux-64=10 gcc_linux-64=10 pip
conda install -y -p $E -c conda-forge --override-channels "python=3.8.5=*_cpython" gxx_linux-64=9 gcc_linux-64=9 sysroot_linux-64=2.17 libxcrypt
conda install -y -p $E -c anaconda --override-channels "python=3.8.5=*_cpython" openblas-devel
export CONDA_PKGS_DIRS=/data/openmask3d/conda/pkgs
python -m pip install --force-reinstall --no-deps "pip==23.3.2"
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install ninja==1.10.2.3 ; pip install pytorch-lightning==1.7.2 fire==0.5.0 imageio==2.23.0 tqdm==4.64.1 wandb==0.13.2
pip install python-dotenv==0.21.0 pyviz3d==0.2.32 scipy==1.9.3 plyfile==0.7.4 scikit-learn==1.2.0 trimesh==3.17.1 loguru==0.6.0 albumentations==1.3.0 volumentations==0.1.8
pip install antlr4-python3-runtime==4.8 black==21.4b2 omegaconf==2.0.6 hydra-core==1.0.5 --no-deps
CPATH=$E/include CUDA_HOME=$E TORCH_CUDA_ARCH_LIST=8.6 FORCE_CUDA=1 CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=$E/bin/x86_64-conda-linux-gnu-g++ \
  pip install --no-build-isolation --no-deps 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf'
SETUPTOOLS_USE_DISTUTILS=stdlib pip install --no-build-isolation --no-deps git+https://github.com/NVIDIA/MinkowskiEngine -v --config-settings="--blas_include_dirs=$E/include" --config-settings="--blas=openblas"
pip install pynvml==11.4.1 gpustat==1.0.0 tabulate==0.9.0 pytest==7.2.0 tensorboardx==2.5.1 yapf==0.32.0 termcolor==2.1.1 addict==2.4.0 blessed==1.19.1
pip install gorilla-core==0.2.7.8 matplotlib==3.7.2 cython
pip install --only-binary=pycocotools pycocotools==2.0.7 h5py==3.7.0 transforms3d==0.4.1 open3d==0.16.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torchmetrics==0.11.0 setuptools==68.0.0 ; pip install fvcore==0.1.5.post20221221 cloudpickle==2.1.0 Pillow==9.3.0
cd openmask3d/class_agnostic_mask_computation/third_party/pointnet2 && pip install --no-build-isolation .
pip install --no-deps 'git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16'
pip install --no-deps 'git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588'
pip install ftfy==6.1.1 regex==2023.10.3
cd /data/openmask3d && pip install -e .
curl -L -C - -o resources/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
curl -L -C - -o resources/scannet200_model.ckpt "https://drive.usercontent.google.com/download?id=1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B&export=download&confirm=t"
python -c "import clip; clip.load('ViT-L/14@336px', device='cpu')"   # -> ~/.cache/clip/ViT-L-14-336px.pt
```

## Env fixes (needed on this host, env-local only)
- `conda create` with `cuda-toolkit=11.3` pulled CUDA 13.3 components; fixed by the `nvidia/label/cuda-11.3.1` channel.
- Unpinned python in conda installs resolved to GraalPy; always pin `python=3.8.5=*_cpython`.
- pip 24.x rejects `pytorch-lightning==1.7.2` metadata; pinned pip 23.3.2 (conda re-solves reset it, so pin after conda steps).
- detectron2 needs gcc <= 10 for CUDA 11.3; used gcc/g++ 9 with `sysroot_linux-64=2.17` and `libxcrypt`.
- `$E/compiler_compat/ld` (old conda ld) removed: could not find `/lib64/libpthread.so.0`; system ld used instead.
- `$E/x86_64-conda-linux-gnu/sysroot/usr/lib64/{libc.so,libpthread.so}` linker scripts rewritten to bare names (`GROUP ( libpthread.so.0 libpthread_nonshared.a )`); originals kept as `*.orig`.
- `CPATH=$E/include` so `crypt.h` is found for lmdb/detectron2. `LIBRARY_PATH` not used (it pulled env libgfortran, which needs glibc >= 2.26).
- pycocotools 2.0.6 has no cp38 wheel and its sdist link hits the same libgfortran issue; installed pycocotools 2.0.7 wheel (deviation).
- The conda-installed setuptools/numpy path needs `SETUPTOOLS_USE_DISTUTILS=stdlib` for MinkowskiEngine's setup.py.

## Smoke (GPU 4 only)
- `import torch, MinkowskiEngine, segment_anything, clip, openmask3d`: OK. torch 1.12.1+cu113, cuda True, RTX A4000.
- MinkowskiEngine 0.5.4 sparse conv forward on GPU: OK (output tensor on cuda).
- Checksums: scannet200_model.ckpt `6a0c4347616a27348bed32ff1ae225cab0087e0a0148f36d1023ca2d46f82534` (158873495 B), sam_vit_h_4b8939.pth `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`.
- CLIP ViT-L/14@336px cached at `~/.cache/clip/ViT-L-14-336px.pt` (934088680 B), not inside the env prefix (clip's default location).

## Open
- Depth-to-color SAM patch (issue 46) is not in this fork's history; reapplied as commit `fb5df6b` on `spellbook-baseline`. Upstream commit `0b4a7ed` is not reachable from any local clone.
- `scannet200_model.ckpt` from the README link is the 158 MB file used by the canonical baseline; issue 63 mentions a different 952 MB Drive file (`da4b68cb`), not used here.
- pycocotools 2.0.7 instead of the pinned 2.0.6.
