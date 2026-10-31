# Spellbook setup

How to rebuild the data, method environments and container images on a host without root. Layout
and goals are in `spellbook/PROJECT_STATUS.md`; problems and results are in the GitHub issues.

## Common rules
- Conda: `source /usr/lib/miniconda3/etc/profile.d/conda.sh`. Use prefix envs and a private package
  cache per method (`export CONDA_PKGS_DIRS=/data/<method>/conda/pkgs`): concurrent installs corrupt
  the shared `~/.conda/pkgs`.
- CUDA comes from conda (`nvidia` channel), because the host has no system nvcc.
- Older CUDA builds use `gcc/g++ 9` with `sysroot_linux-64=2.17`, which avoids the Ubuntu 24.04
  `linux/types.h` `__s128` error.
- Conda-forge solves without a pinned CPython may swap Python for GraalPy: pin `python=X=*_cpython`,
  or keep the compilers in a separate env.
- Secrets live only in `spellbook/.secrets.env` (mode 600, gitignored). Load it with
  `set -a; . /data/ScanNet/spellbook/.secrets.env; set +a`, and never print it.
- GPU smoke tests use a pool GPU only (`CUDA_VISIBLE_DEVICES=4`); GPU 0 is reserved.

## Data
```bash
cd /data/ScanNet
PY=~/.conda/envs/3disspellbook/bin/python

# 16 protocol scenes, label map (accepting the ScanNet ToS prompt is required)
$PY spellbook/download_scans.py --scene $(PYTHONPATH=spellbook $PY -c \
  'from evaluation.benchmark import PREDICTION_EVALUATION_SCENES; print(*PREDICTION_EVALUATION_SCENES)') \
  --workers 4 --segments 8
# place v2/scannetv2-labels.combined.tsv under /data/scannetdata/v2/

# ZED recordings
ln -s HD1200_SN46307300_18-27-20_raum312_v2.svo2 /data/scannetdata/custom/raw/scene9004.svo2
ln -s HD1080_SN46307300_19-33-28.svo2            /data/scannetdata/custom/raw/scene9009.svo2

# CAD reference: put LG2C_Raum312_v2.3dm in custom/reference/ first
$PY spellbook/tmp/export_scene9004_cad.py        # needs rhino3dm; writes scene9004_cad.ply
$PY spellbook/main.py --extract-frames --scene 9004 9009   # expect 3612 and 2356 frames
$PY spellbook/tmp/build_scene9004_reference.py   # 20 mm visible mask; needs scene9004_40 (open3d B1) for alignment

# Ground truth
$PY spellbook/evaluation/evaluate.py export-gt --scene 0046_00 --benchmark ScanNet200   # per scene
```
Expected hashes are in `spellbook/reconstruct/geometry_reference.yaml`.

Build ScanNet's Segmentator in `/data/ScanNet/Segmentator` with `make`.

## Spellbook env and ZED SDK 5.4.1 (`~/.conda/envs/3disspellbook`)
```bash
mkdir -p /data/zed-sdk-installer ~/opt/zed-sdk/install/settings
cd /data/zed-sdk-installer
curl -sSfL -o ZED_SDK_Ubuntu24_cuda12.8_tensorrt10.9_v5.4.1.zstd.run \
  https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/5.4/ZED_SDK_Ubuntu24_cuda12.8_tensorrt10.9_v5.4.1.zstd.run
chmod +x ZED_SDK_*.run
./ZED_SDK_Ubuntu24_cuda12.8_tensorrt10.9_v5.4.1.zstd.run --noexec --target ~/opt/zed-sdk/install
curl -sSfL -o ~/opt/zed-sdk/install/settings/SN46307300.conf "https://calib.stereolabs.com/?SN=46307300"

# The installer's get_python_api.py needs root, so fetch the wheel it would use.
curl -sSfL -o ~/opt/zed-sdk/pyzed-5.4-cp310-cp310-linux_x86_64.whl \
  https://download.stereolabs.com/zedsdk/5.4/whl/linux_x86_64/pyzed-5.4-cp310-cp310-linux_x86_64.whl

CONDA_PKGS_DIRS=/data/spellbook/conda/pkgs conda env create -y -p ~/.conda/envs/3disspellbook \
  -f spellbook/environment.yaml
```
- Create `etc/conda/{activate.d,deactivate.d}/zed_sdk_env.sh` in the env. Activation exports
  `ZED_DIR=$HOME/opt/zed-sdk/install` and
  `LD_LIBRARY_PATH=$ZED_DIR/lib:$ZED_DIR/dependencies/ai_dependencies/lib:$CONDA_PREFIX/lib`;
  deactivation restores both.
- Activate the env for ZED work: `LD_LIBRARY_PATH` must come from the shell, and setting it inside
  Python does not reach `dlopen`.
- Check: open `custom/raw/scene9009.svo2` with pyzed (expect S/N 46307300, 1920x1080, 2357 frames).
- Open3D 0.19.0 in this env must report `open3d.core.cuda.is_available()` True for the Open3D engine.

Run:
```bash
SPELLBOOK_ZED_PRESET=native-default  SPELLBOOK_TARGET_RUN0=1 python spellbook/main.py --engine zed --scene 9004 --issue 47 --offline-score --foreground
SPELLBOOK_ZED_PRESET=s2b8-fixedcalib python spellbook/main.py --engine zed --scene 9004 --issue 39 --offline-score --foreground
```

## Open3D
No extra setup, uses the spellbook env.
```bash
SPELLBOOK_OPEN3D_PRESET=tensor SPELLBOOK_TARGET_RUN0=1 python spellbook/main.py --engine open3d --scene 9004 --issue 51 --offline-score --foreground
```

## Metashape 2.3.1 (`/data/zed-metashape/conda/env`)
```bash
mkdir -p /data/zed-metashape/conda/pkgs /data/zed-metashape/wheel/dl
CONDA_PKGS_DIRS=/data/zed-metashape/conda/pkgs conda create -y -p /data/zed-metashape/conda/env \
  python=3.11 numpy -c conda-forge --override-channels
W=metashape-2.3.1-cp39.cp310.cp311.cp312.cp313-abi3-linux_x86_64.whl   # the URL is lowercase
curl -sL --fail -o /data/zed-metashape/wheel/dl/$W https://download.agisoft.com/$W
cd /data/zed-metashape
conda/env/bin/python -m pip install -q --no-cache-dir ./wheel/dl/$W open3d    # open3d 0.20.0

set -a; . /data/ScanNet/spellbook/.secrets.env; set +a
conda/env/bin/python -I -c 'import os, Metashape; Metashape.License().activate(os.environ["METASHAPE_LICENSE"]); print(Metashape.app.activated)'
```
Run:
```bash
SPELLBOOK_METASHAPE_PRESET=native-default SPELLBOOK_TARGET_RUN0=1 python spellbook/main.py --engine metashape --scene 9004 --issue 53 --offline-score --foreground
SPELLBOOK_METASHAPE_PRESET=b5-kf350 python spellbook/main.py --engine metashape --scene 9004 --issue 40 --offline-score --foreground
```

## Docker engines (RTAB-Map, Isaac, BundleFusion)
- Docker is the snap package. `reconstruct/containers/__init__.py` and `gpu_args.sh` handle two of
  its limits: host paths are mounted through `/var/lib/snapd/hostfs` (plain `/data` paths mount as
  empty dirs), and GPU nodes and driver libraries are passed explicitly (`--gpus` does not work).
- `--foreground` is required, from a shell that has the docker group. Detached jobs run under the
  systemd user manager, which lacks that group until it restarts (full logout or admin restart).

### RTAB-Map (`localhost/zed-rtabmap:jazzy`)
```bash
cd /data/ScanNet/spellbook/reconstruct/containers/rtabmap
docker build --network host -t localhost/zed-rtabmap:jazzy - < Dockerfile
cd /data/ScanNet
SPELLBOOK_RTABMAP_PROFILE=assisted-hybrid SPELLBOOK_RTABMAP_STRIDE=90 SPELLBOOK_TARGET_RUN0=1 \
  python spellbook/main.py --engine rtabmap --scene 9004 --issue 49 --offline-score --foreground
```

### Isaac (`zed-isaac-nvblox:spellbook`)
Built from public debs (Isaac ROS release-4.5, no NGC login); takes about 15 min uncached.
```bash
cd /data/ScanNet/spellbook/reconstruct/containers/isaac
docker build -t zed-isaac-nvblox:spellbook .
cd /data/ScanNet
python -m spellbook.reconstruct.engines.isaac --prime --scene 9004 9009   # ZED model cache, once per SDK/driver
SPELLBOOK_ISAAC_PRESET=official_default SPELLBOOK_TARGET_RUN0=1 python spellbook/main.py --engine isaac --scene 9004 --issue 48 --offline-score --foreground
SPELLBOOK_ISAAC_PRESET=conf100_best python spellbook/main.py --engine isaac --scene 9004 --issue 42 --offline-score --foreground
```
Keep the GPU free of other polling processes while it runs (see #42).

### BundleFusion (`bundlefusion:noflip`, the adapter default)
Source: public fork FangGet/BundleFusion_Ubuntu_Pangolin @ f350ab06, not `/data/bundlefusion`
(upstream, Windows). `noflip.patch` removes the fork's input mirroring.
```bash
cd /data/ScanNet/spellbook/reconstruct/containers/bundlefusion
docker build -t bundlefusion:latest .
docker build -f Dockerfile.noflip -t bundlefusion:noflip .
docker build -f Dockerfile.siftfix -t bundlefusion:spellbook-siftfix .    # optional, historical presets
cd /data/ScanNet
python -m spellbook.reconstruct.engines.bundlefusion --prepare 9004 9009  # cached input, outside the solver clock
SPELLBOOK_TARGET_RUN0=1 python spellbook/main.py --engine bundlefusion --scene 9004 --issue 52 --offline-score --foreground
```
The solver uses a GPU pair, and only the solver counts against the 3x runtime cap.

## Shared detectron2 (`/data/detectron2`, v0.6)
```bash
git clone --branch v0.6 --depth 1 https://github.com/facebookresearch/detectron2.git /data/detectron2
```
Open3DIS and OpenIns3D each install it from a throwaway copy, so the clone stays clean, using that
method's build variables and `pip install --no-build-isolation --no-deps <copy>`.

## Prediction methods
Run any method with
`python spellbook/main.py --predict --benchmark ScanNet200 --models <method> --run-id <id> --issue <N> --foreground`,
adding `--run-parameters spellbook/tmp/overview/params/<file>.json` for the runs listed below.

### OpenYOLO3D (`/data/openyolo3D/conda/envs/openyolo3d`, repo `/data/OpenYOLO3D`)
Weights and official masks come from one archive:
```bash
python3 -m gdown 1FneLaYaClWDO51L9lIvlTQbheh5SfOFD -O OpenYOLO3D.zip   # unzip into an empty dir
# checkpoints/* -> /data/OpenYOLO3D/pretrained/checkpoints/
# output/scannet200 -> /data/openYOLO3D/output/  (/data/OpenYOLO3D/output is a symlink to it)
```
Toolchain in separate envs (gxx in the Python env swaps CPython for GraalPy):
```bash
conda create -y -p /data/openyolo3D/cuda-11.3 -c nvidia/label/cuda-11.3.1 cuda-toolkit    # add a lib64 -> lib symlink
conda create -y -p /data/openyolo3D/toolchain-gcc9-s217 -c conda-forge --override-channels \
  "gxx_linux-64=9" "gcc_linux-64=9" "sysroot_linux-64=2.17" "openblas=0.3.21"
```
Env:
```bash
conda create -y -p /data/openyolo3D/conda/envs/openyolo3d python=3.10.9 pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install pytorch-lightning==1.7.2 black==21.4b2 cloudpickle==3.0.0 future hydra-core==1.0.5 "pycocotools>=2.0.2" pydot iopath==0.1.7 loguru albumentations
pip install --no-deps numpy==1.24.2 albumentations==1.2.1 opencv-python-headless==4.7.0.72 torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip uninstall -y albucore; pip install setuptools==65.6.3
```
Source builds, with `CUDA_HOME=/data/openyolo3D/cuda-11.3`, gcc 9 from the toolchain env,
`NVCC_PREPEND_FLAGS="-ccbin <gxx9>"`, `TORCH_CUDA_ARCH_LIST=8.6`. Build from copies of the submodules:
```bash
python setup.py install --force_cuda --blas=openblas --blas_include_dirs=$TC/include --blas_library_dirs=$TC/lib   # MinkowskiEngine @02fc608
python setup.py install    # pointnet2
make                       # ScanNet Segmentator @3e57265
pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf
pip install --no-build-isolation <Mask3D copy>
pip install --no-deps <YOLO-World copy>; pip install supervision==0.19.0 transformers==4.30.2 tokenizers==0.13.3
pip install mmyolo==0.6.0 mmdet==3.0.0 plyfile openmim; MMCV_WITH_OPS=1 pip install --no-build-isolation mmcv==2.0.0
pip install --no-deps open3d==0.19.0 pillow==9.1.0
pip install plotly==5.14.1 dash==2.9.3 addict pyquaternion pyviz3d==0.2.28 trimesh==3.14.0 imageio==2.21.1 pandas==2.0.0 scikit-learn==1.1.2 scipy==1.9.0 matplotlib==3.7.1 scikit-image==0.20.0
```
Best run parameters: `openyolo3d-h1.json`.

### Mosaic3D (`/data/mosaic3d/conda/envs/mosaic3d`, repo `/data/Mosaic3D`)
Needs the OpenIns3D env for its Mask3D proposals. `spellbook/predict/models/_mosaic3d_scene.py`
replaces the lost `scripts/run_custom_scene.py`.
```bash
mkdir -p /data/mosaic3d/ckpts
curl -L -C - -o /data/mosaic3d/ckpts/sc.ckpt        https://huggingface.co/junhalee/Mosaic3D/resolve/main/sc.ckpt
curl -L -C - -o /data/mosaic3d/ckpts/spunet34c.ckpt https://huggingface.co/junhalee/Mosaic3D/resolve/main/spunet34c.ckpt

CONDA_PKGS_DIRS=/data/mosaic3d/conda/pkgs conda create -y -p /data/mosaic3d/conda/envs/mosaic3d \
  -c pytorch -c nvidia -c conda-forge python=3.10 "pytorch=2.4.*" "torchvision=0.19.*" "pytorch-cuda=12.1"
P=/data/mosaic3d/conda/envs/mosaic3d/bin
$P/pip install "numpy<2" "lightning==2.2" "torchmetrics>=0.11.4" "hydra-core==1.3.2" hydra-colorlog==1.2.0 hydra-optuna-sweeper==1.2.0 \
  rootutils rich omegaconf natsort einops tqdm scipy plyfile easydict pyyaml imageio matplotlib scikit-image \
  open_clip_torch timm addict overrides fire gdown requests jaxtyping "open3d==0.18.0" spconv-cu120 torch_scatter \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
$P/pip install "transformers==4.46.3"       # transformers 5 does not see torch 2.4
$P/pip install "setuptools==69.5.1"         # lightning 2.2 needs pkg_resources
$P/pip install "git+https://github.com/openai/CLIP.git"
```
Best run parameters: `mosaic3d-best.json`.

### OpenIns3D (`/data/openins3d/conda/envs/openins3d`, code at worktree `/data/openins3d/repo-56ccc22`)
`/data/OpenIns3D` holds a fork at another commit, so the wrapper uses a worktree of the logged commit:
```bash
git -C /data/OpenIns3D fetch https://github.com/Pointcept/OpenIns3D 56ccc220dc003e38c644116d7d2078057b6136ad
git -C /data/OpenIns3D worktree add --detach /data/openins3d/repo-56ccc22 56ccc22
```
Env, with `E=/data/openins3d/conda/envs/openins3d`, `CONDA_PKGS_DIRS=/data/openins3d/conda/pkgs`, and a
pip constraints file pinning torch==1.13.1, torchvision==0.14.1, torchaudio==0.13.1, numpy==1.23.5:
```bash
conda create -p $E python=3.9 -y
conda install -y -c pytorch -c nvidia/label/cuda-11.6.1 -c nvidia pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6
pip install omegaconf==2.1.1 iopath==0.1.8 gdown==4.6.3 loguru plyfile python-dotenv pyviz3d open3d pytorch-lightning==1.9.5 hydra-core==1.1.2 nltk albumentations scipy matplotlib pillow tqdm imageio
conda install -c conda-forge gxx_linux-64=9 gcc_linux-64=9 sysroot_linux-64=2.17 openblas
conda install -c nvidia/label/cuda-11.6.1 -c nvidia libcusolver-dev libcusparse-dev libcurand-dev libcublas-dev
```
Build, with `TORCH_CUDA_ARCH_LIST=8.6 FORCE_CUDA=1 CUDA_HOME=$E CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++`:
```bash
pip install --no-build-isolation torch_scatter
# detectron2: see "Shared detectron2", then:
pip install fvcore==0.1.5.post20221221 pycocotools yacs termcolor tabulate cloudpickle tensorboard black==21.4b2 future pydot
git clone https://github.com/NVlabs/ODISE.git /data/OpenIns3D/third_party/ODISE
pip install --no-build-isolation --no-deps -e /data/OpenIns3D/third_party/ODISE
git clone --branch v0.7.2 --depth 1 https://github.com/facebookresearch/pytorch3d.git /data/openins3d/pytorch3d
pip install --no-build-isolation --no-deps /data/openins3d/pytorch3d    # no cu116 conda build exists
git clone --recursive https://github.com/NVIDIA/MinkowskiEngine /data/OpenIns3D/third_party/MinkowskiEngine
git -C /data/OpenIns3D/third_party/MinkowskiEngine checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
(cd /data/OpenIns3D/third_party/MinkowskiEngine && python setup.py install --force_cuda --blas=openblas --blas_include_dirs=$E/include --blas_library_dirs=$E/lib)
(cd /data/OpenIns3D/third_party/pointnet2 && python setup.py install)
```
Data and weights:
```bash
gdown 1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B -O /data/OpenIns3D/third_party/scannet200_val.ckpt   # sha256 6a0c4347...
ln -s /data/OpenIns3D/third_party/scannet200_val.ckpt /data/openins3d/checkpoints/scannet200_val.ckpt
gdown 1NfnNBTvapKHi_ZrIrBiKrW8VSKzYc-V_ -O /data/openins3d/downloads/masks_zip/scannet.zip   # into repo-56ccc22/data/scannet (masks/detected/*.pt)
gdown 1UF1vi19pwTvp8CT9sJ7Bg26aErvL-OgZ -O /data/openins3d/downloads/yoloworld_zip/yoloworld.zip   # YOLO-World third_party + /data/openins3d/pretrained
R=/data/openins3d/repo-56ccc22/third_party
ln -s /data/OpenIns3D/third_party/ODISE $R/ODISE
ln -s /data/openins3d/pretrained $R/pretrained
ln -s /data/OpenIns3D/third_party/scannet200_val.ckpt $R/scannet200_val.ckpt
```
Run parameters: `openins3d-baseline.json`, `openins3d-best.json`.

### Open3DIS (`/data/open3dis/conda/envs/open3dis`, repo `/data/Open3DIS`)
Upstream VinAIResearch/Open3DIS @ bc2a3b3 without patches. The original fork is gone, and the
spellbook side replaces it (`_open3dis_prepare.py`, `_open3dis_config.py`, `_open3dis_stage.py`).
```bash
E=/data/open3dis/conda/envs/open3dis; export CONDA_PKGS_DIRS=/data/open3dis/conda/pkgs
conda create -y -p $E python=3.8
conda install -y -p $E -c nvidia cuda-nvcc=11.3.58 cuda-libraries-dev=11.3.0
conda install -y -p $E -c conda-forge sysroot_linux-64=2.17 gcc_linux-64=9 gxx_linux-64=9
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install spconv-cu113==2.1.25
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install -e segmenter2d/segment_anything
pip install -r segmenter3d/ISBNet/requirements.txt scikit-image opencv-python imageio plyfile supervision==0.11.1
pip install "git+https://github.com/facebookresearch/ov-seg@36f49d496714998058d115ffb6172d9d84c59065#egg=clip&subdirectory=third_party/CLIP"
python -c "import clip; clip.load('ViT-L/14@336px')"
```
CUDA extensions build with `CUDA_HOME=$E TORCH_CUDA_ARCH_LIST=8.6`,
`CC/CXX=$E/bin/x86_64-conda-linux-gnu-{gcc,g++}`, `NVCC_PREPEND_FLAGS="--compiler-bindir=$CXX"` and
`CPATH` pointing at sparsehash headers (`/data/open3dis/sparsehash-include`, from
github.com/sparsehash/sparsehash plus a hand-written `sparseconfig.h`) and the CUDA headers:
```bash
(cd segmenter3d/ISBNet && python setup.py build_ext develop)
(cd segmenter2d/GroundingDINO && pip install --no-build-isolation --no-deps -e .)
# detectron2: see "Shared detectron2"
git checkout -- .     # upstream tracks build artifacts; restore them after building
```
Weights (SAM ViT-H and GroundingDINO URLs from `docs/INSTALL.md`):
```bash
# /data/pretrains/foundation_models/{sam_vit_h_4b8939.pth, groundingdino_swint_ogc.pth}
gdown --id 1ZEZgQeT6dIakljSTx4s5YZM0n2rwC3Kw -O /data/pretrains/scannet200/head_scannetv2_200_val.pth
```

### OpenMask3D (`/data/openmask3d/conda/envs/openmask3d`, repo `/data/openmask3d`, branch `spellbook-baseline`)
Archived method; the env stays in place. The branch carries the depth-to-color SAM prompt patch and the `OPENMASK3D_FORCE_GPU` fix for
`get_free_gpu`. Always pin Python, and re-pin pip after conda steps.
```bash
E=/data/openmask3d/conda/envs/openmask3d; export CONDA_PKGS_DIRS=/data/openmask3d/conda/pkgs
printf 'conda/\nresources/\n' >> /data/openmask3d/.git/info/exclude
conda create -y -p $E -c nvidia/label/cuda-11.3.1 -c conda-forge --override-channels 'python=3.8.5=*_cpython' cuda-toolkit=11.3.1 gxx_linux-64=10 gcc_linux-64=10 pip
conda install -y -p $E -c conda-forge --override-channels "python=3.8.5=*_cpython" gxx_linux-64=9 gcc_linux-64=9 sysroot_linux-64=2.17 libxcrypt
conda install -y -p $E -c anaconda --override-channels "python=3.8.5=*_cpython" openblas-devel
python -m pip install --force-reinstall --no-deps "pip==23.3.2"
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install ninja==1.10.2.3 pytorch-lightning==1.7.2 fire==0.5.0 imageio==2.23.0 tqdm==4.64.1 wandb==0.13.2
pip install python-dotenv==0.21.0 pyviz3d==0.2.32 scipy==1.9.3 plyfile==0.7.4 scikit-learn==1.2.0 trimesh==3.17.1 loguru==0.6.0 albumentations==1.3.0 volumentations==0.1.8
pip install antlr4-python3-runtime==4.8 black==21.4b2 omegaconf==2.0.6 hydra-core==1.0.5 --no-deps
CPATH=$E/include CUDA_HOME=$E TORCH_CUDA_ARCH_LIST=8.6 FORCE_CUDA=1 CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=$E/bin/x86_64-conda-linux-gnu-g++ \
  pip install --no-build-isolation --no-deps 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf'
SETUPTOOLS_USE_DISTUTILS=stdlib pip install --no-build-isolation --no-deps git+https://github.com/NVIDIA/MinkowskiEngine -v --config-settings="--blas_include_dirs=$E/include" --config-settings="--blas=openblas"
pip install pynvml==11.4.1 gpustat==1.0.0 tabulate==0.9.0 pytest==7.2.0 tensorboardx==2.5.1 yapf==0.32.0 termcolor==2.1.1 addict==2.4.0 blessed==1.19.1
pip install gorilla-core==0.2.7.8 matplotlib==3.7.2 cython
pip install --only-binary=pycocotools pycocotools==2.0.7 h5py==3.7.0 transforms3d==0.4.1 open3d==0.16.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torchmetrics==0.11.0 setuptools==68.0.0 fvcore==0.1.5.post20221221 cloudpickle==2.1.0 Pillow==9.3.0
(cd openmask3d/class_agnostic_mask_computation/third_party/pointnet2 && pip install --no-build-isolation .)
pip install --no-deps 'git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16'
pip install --no-deps 'git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588'
pip install ftfy==6.1.1 regex==2023.10.3
(cd /data/openmask3d && pip install -e .)
curl -L -C - -o resources/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
curl -L -C - -o resources/scannet200_model.ckpt "https://drive.usercontent.google.com/download?id=1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B&export=download&confirm=t"
python -c "import clip; clip.load('ViT-L/14@336px', device='cpu')"
```
Env-local fixes this host needed:
- Remove `$E/compiler_compat/ld`, because the old conda ld cannot find `/lib64/libpthread.so.0`.
- Rewrite `$E/x86_64-conda-linux-gnu/sysroot/usr/lib64/{libc.so,libpthread.so}` linker scripts to bare
  names, for example `GROUP ( libpthread.so.0 libpthread_nonshared.a )`.
- Leave `LIBRARY_PATH` unset: it pulls an env libgfortran that needs glibc >= 2.26.

### sam3dis (`/data/sam3dis/conda/envs/sam3dis`, repo `/data/sam3dis`, SAM 3 code `/data/sam3`)
`/data/sam3dis` is a local repo without a remote, rewritten from #69 (see its `DERIVATION.md`).
```bash
export CONDA_PKGS_DIRS=/data/sam3dis/conda/pkgs
conda create -y -p /data/sam3dis/conda/envs/sam3dis python=3.12
conda activate /data/sam3dis/conda/envs/sam3dis
pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e /data/sam3
pip install "numpy>=1.26,<2" "setuptools<81" einops ninja pytest "open3d==0.19.0" "plyfile<1.1"
pip install pycocotools opencv-python-headless scikit-image scikit-learn scipy hydra-core omegaconf open_clip_torch openai psutil pandas matplotlib torchmetrics submitit
pip check

set -a; . /data/ScanNet/spellbook/.secrets.env; set +a     # gated weights
export HF_HOME=/data/sam3dis/hf_cache
hf download facebook/sam3 sam3.pt
hf download facebook/sam3 config.json
```
Best run parameters: `sam3dis-best.json`.
