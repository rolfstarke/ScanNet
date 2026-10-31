# Container setup log (rootless podman 4.9.3, single uid mapping)

## 1. Rootless storage (WORKS)
```
cat > ~/.config/containers/storage.conf <<'EOF'
[storage]
driver = "overlay"
runroot = "/run/user/1066005817/containers"
graphroot = "/home/starke/.local/share/containers/storage"

[storage.options.overlay]
ignore_chown_errors = "true"
EOF
podman pull docker.io/library/ubuntu:24.04   # ok; without the file: lchown /etc/gshadow EINVAL
```

## 2. GPU access
- CDI (`--device nvidia.com/gpu=N`): BROKEN. Host spec /var/run/cdi/nvidia.yaml is cdiVersion 0.7.0 with
  `additionalGids`; podman 4.9.3 cannot parse it ("unresolvable CDI devices"). podman 4.9.3 ignores
  `cdi_spec_dirs` and reads only /etc/cdi and /var/run/cdi (root-owned). Fix needs admin (podman >= 4.9.x CDI
  or regenerate the spec in an older format). Adapters isaac.py `--device nvidia.com/gpu=all` will fail.
- Working: helper `spellbook/reconstruct/containers/podman_gpu.sh <gpu>`:
```
ARGS=(); mapfile -t ARGS < <(/data/ScanNet/spellbook/reconstruct/containers/podman_gpu.sh 4)
podman run --rm "${ARGS[@]}" docker.io/library/ubuntu:24.04 nvidia-smi -L
# -> CUDA_VISIBLE_DEVICES=4, GPU 0..4 listed
```

## RTAB-Map image (BUILT, docker) `localhost/zed-rtabmap:jazzy` sha256:61d896f324fe, 4.68 GB
- Baseline replays the shared pool, so no ZED SDK / zed-ros2-wrapper; base ubuntu:24.04. No build context needed:
```
cd /data/ScanNet/spellbook/reconstruct/containers/rtabmap
docker build --network host -t localhost/zed-rtabmap:jazzy - < Dockerfile
```
- Verified in image: rtabmap-export, `With GTSAM: true`, Optimizer/Strategy default 2, cv2/cv_bridge/rtabmap_msgs/tf2_ros import.
  `--opt 2` here means "poses already optimized in the db" (GTSAM comes from Optimizer/Strategy 2).
- Baseline, from /data/ScanNet with the conda env active (--foreground: see blocker):
```
SPELLBOOK_RTABMAP_PROFILE=assisted-hybrid SPELLBOOK_RTABMAP_STRIDE=90 python spellbook/main.py --engine rtabmap --scene 9004 --foreground
```
- Smoke `SPELLBOOK_TARGET_RUN0=1 ... --scene 9009 --issue 49 --offline-score --foreground`: 27/27, span 99.4%, slam+export 83.0 s, fuse 7.5 s, total 113.5 s <= 471.1 s.
- BLOCKER, detached jobs: the systemd --user manager lacks the docker group (`id` in `systemd-run --user` has no 990), so docker gets
  permission denied and jobs fail preflight; `sg docker` asks for a password. Needs a fresh user manager (all sessions logged out, or admin restart of user@1066005817).

## 4. Isaac image (BUILT, docker) `zed-isaac-nvblox:spellbook` sha256:1cfac621979a, 19.3 GB
Public debs only, no NGC login: Isaac ROS release-4.5, Jetson r38.4 (libnvvpi4, nvsci), CUDA 13 apt repos.
ZED SDK 5.4.1 + zed-ros2-wrapper v5.4.1 from source (`--packages-skip zed_debug`: it links against driver libs absent at build).
```
cd /data/ScanNet/spellbook/reconstruct/containers/isaac
docker build -t zed-isaac-nvblox:spellbook .     # /data build context works; about 15 min uncached
```
ZED models: the SDK optimizes per depth mode on first open (about 6 min), which would eat the runtime cap. Cache is
`/data/spellbook/cache/zed_resources`, mounted over /usr/local/zed/resources. Prime once, or after an SDK/driver change:
```
cd /data/ScanNet && python -m spellbook.reconstruct.engines.isaac --prime --scene 9004 9009
```
Run from a shell with docker access: detached jobs have no docker group, so use `--foreground`.
```
cd /data/ScanNet
SPELLBOOK_ISAAC_PRESET=official_default SPELLBOOK_TARGET_RUN0=1 python spellbook/main.py --engine isaac --scene 9009 --issue 48 --offline-score --foreground
SPELLBOOK_ISAAC_PRESET=official_default SPELLBOOK_TARGET_RUN0=1 python spellbook/main.py --engine isaac --scene 9004 --issue 48 --foreground
python spellbook/main.py --engine isaac --scene 9004 --issue 42 --foreground   # default conf100_best
```
Verified: scene9009_30 official_default, GPU 2, domain 102: 214.75 s <= 471.13 s, 2332/2356 poses, VO state 1 on all
2333 messages, traj 67.849 m, mesh 113030v/172228f, SVO end 2356/2357, .sens 2332 frames. Not run yet: 9004, NEURAL presets
beyond the first frame. Gotchas: wrapper needs `video.publish_left_right` + `video.publish_gray` true for cuVSLAM;
zed_components needs isaac_ros_gxf `gxf/lib/serialization` on LD_LIBRARY_PATH; set -u must follow the ROS setup sourcing.

## 5. BundleFusion (BUILT, docker) default `bundlefusion:noflip` sha256:f937c0cb7ee1, 6.97 GB
- Source: public fork FangGet/BundleFusion_Ubuntu_Pangolin @ f350ab06 (the fork ce25997 names), not `/data/bundlefusion`
  (upstream niessner, Windows). Base nvidia/cuda:11.4.3-devel-ubuntu20.04, VISUALIZATION=OFF, no Pangolin.
- `spellbook/reconstruct/containers/bundlefusion/`: `Dockerfile`, `headless.patch` (ce25997 patch list +
  `bundle_fusion_headless`), `Dockerfile.siftfix` + `siftfix.patch` (niessner PR #28, #31 ported to fork paths).
- A /data build context works with the snap CLI; no tar or $HOME context needed.
```
cd /data/ScanNet/spellbook/reconstruct/containers/bundlefusion
docker build -t bundlefusion:latest .                                      # ~2 min after the base pull
docker build -f Dockerfile.siftfix -t bundlefusion:spellbook-siftfix .     # sha256:7a77943a2b6b, optional
cd /data/ScanNet; export PATH=~/.conda/envs/3disspellbook/bin:$PATH
python -m spellbook.reconstruct.engines.bundlefusion --prepare 9004 9009   # cached input; 9009 44 s
SPELLBOOK_TARGET_RUN0=1 python spellbook/main.py --engine bundlefusion --scene 9009 --issue 52 --offline-score --foreground
```
- **`--foreground` is required.** Detached jobs start through `systemd-run --user`, whose manager has no docker group:
  "permission denied ... /var/run/docker.sock". Same for every docker engine; the user manager needs a restart by the admin
  or a full logout to pick up the group.
- Smoke 2026-10-30 scene9009_50 (GPUs 1,2, rounds 0): solver 59.05/471.13 s PASS, delivery 147.53 s, validate 1.37 s,
  2355/2356 poses (190 opt + 2165 int), 222 lost. Logged #52: 60.32 s, 216+2140, 220 lost, 2356/2356. Bundling sha d8c9e284.
- 3x rule: `RUNTIME_CAP_EXEMPT` for the whole pipeline, the adapter caps the solver (#52 2026-10-01); over cap = rc 124.
- x-flip settled 2026-10-30, 9004 foreground, one run each (evidence + scripts: `spellbook/tmp/bundlefusion-noflip/`):
  | image | mean/acc/comp cm | solver s | poses kept | submaps/lost | depth on mesh | CAD as-is vs reflected |
  |---|---|---|---|---|---|---|
  | `bundlefusion:noflip` sha256:f937c0cb7ee1 | 53.246/31.970/74.522 | 176.26 | 3579 (2313+1266) | 2/184 | plain 2-3.5 cm, 68-92% (mesh_aligned, tracked frames) | 53.25 vs 57.94 |
  | `bundlefusion:latest` sha256:c73bfb5b793d | 47.885/30.914/64.856 | 162.37 | 3388 (1916+1472) | 18/173 | mirrored 3.14 cm, plain 12.65 (vh_clean, first 200) | 47.89 vs 45.33 |
  noflip reproduces #52's signature (2315+1262, 184 lost, 1210/1220) and is the adapter default; flipped is reflected relative to the CAD.
  Logged score 50.745 is 2.5 cm away, outside the +/-0.5 cm band. `Dockerfile.siftfix` still builds on the flipped `latest`.
- Build noflip: `docker build -f Dockerfile.noflip -t bundlefusion:noflip .` (needs `latest`). Flipped: `SPELLBOOK_BUNDLEFUSION_IMAGE=bundlefusion:latest`.
- Queue: `python spellbook/main.py --engine bundlefusion --scene 9009 9004 --issue 52 --foreground`

## Notes
- Adapter GPU_SCRIPT points to /home/rolf/GIT/zed-rtabmap/scripts/podman_gpu.sh (missing for this user). Repoint to
  /data/ScanNet/spellbook/reconstruct/containers/podman_gpu.sh (adapters are read-only for this task).
