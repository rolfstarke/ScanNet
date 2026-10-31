# ZED SDK 5.4.1 user-local setup log

## SDK download and extract (no root)
    mkdir -p /data/zed-sdk-installer ~/opt/zed-sdk/install/settings
    cd /data/zed-sdk-installer
    curl -sSfL -o ZED_SDK_Ubuntu24_cuda12.8_tensorrt10.9_v5.4.1.zstd.run "https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/5.4/ZED_SDK_Ubuntu24_cuda12.8_tensorrt10.9_v5.4.1.zstd.run"
    chmod +x ZED_SDK_*.run
    ./ZED_SDK_Ubuntu24_cuda12.8_tensorrt10.9_v5.4.1.zstd.run --noexec --target ~/opt/zed-sdk/install
Size 1419030975 B, sha256 bbed0c5fc563cdf1b611d1ea5fdbecd8ae5d059f85d0cf56ca93d8a0d6706877, MD5 check OK.
Note: download.stereolabs.com/zedsdk/5.4/... redirects to 5.4.1; the CDN URL above is the direct one.

## Calibration (SN 46307300)
    curl -sSfL -o /data/zed-sdk-installer/SN46307300.conf "https://calib.stereolabs.com/?SN=46307300"
    cp /data/zed-sdk-installer/SN46307300.conf ~/opt/zed-sdk/install/settings/SN46307300.conf

## Conda env (prefix, environment.yaml unchanged)
Pyzed wheel line removed in a scratch copy (wheel path /home/rolf/... does not exist here).
Private package cache used: concurrent conda processes on this host corrupted/removed the shared ~/.conda/pkgs entries
(CondaVerificationError for sdl3/libglib/libpq; FileNotFoundError for libtasn1 mid-solve).
    source /usr/lib/miniconda3/etc/profile.d/conda.sh
    CONDA_PKGS_DIRS=/data/spellbook/conda/pkgs conda env create -y -p ~/.conda/envs/3disspellbook -f <scratch>/env-nowheel.yaml

## Wheel (pyzed 5.4 cp310)
get_python_api.py refuses to run: it hard-codes /usr/local/zed (not ZED_DIR) and needs root to create it. Fetched the same wheel URL it uses instead:
    curl -sSfL -o ~/opt/zed-sdk/pyzed-5.4-cp310-cp310-linux_x86_64.whl "https://download.stereolabs.com/zedsdk/5.4/whl/linux_x86_64/pyzed-5.4-cp310-cp310-linux_x86_64.whl"
    ~/.conda/envs/3disspellbook/bin/python -m pip install --no-deps ~/opt/zed-sdk/pyzed-5.4-cp310-cp310-linux_x86_64.whl
Result: Successfully installed pyzed-5.4

## Env hooks
Files: ~/.conda/envs/3disspellbook/etc/conda/{activate.d,deactivate.d}/zed_sdk_env.sh
activate exports ZED_DIR=/home/starke/opt/zed-sdk/install and LD_LIBRARY_PATH=$ZED_DIR/lib:$ZED_DIR/dependencies/ai_dependencies/lib:$CONDA_PREFIX/lib(:previous); deactivate restores both.
Verified: conda activate / conda deactivate round-trip.

## Smoke test
    source /usr/lib/miniconda3/etc/profile.d/conda.sh && conda activate ~/.conda/envs/3disspellbook
    CUDA_VISIBLE_DEVICES=4 python -I <scratch>/zed_smoke.py
Output: open SUCCESS, S/N 46307300, depth NONE, resolution 1920x1080 @ 15 fps, SVO total 2357 frames, 10 frames grabbed.
Note: runtime ZED_DIR set via os.environ inside python does not work for dlopen; LD_LIBRARY_PATH must come from the activated shell.

## Open items
- environment.yaml still has /home/rolf/opt/zed-sdk/pyzed-5.4-...whl; this host user is starke. Not edited per instructions.
- /usr/local/zed not created; not needed by the runtime when ZED_DIR is set (smoke used ZED_DIR path and loaded calibration).
