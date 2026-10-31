# Recovery report B: open3d, bundlefusion, rtabmap reconstruction adapters

Sources: `spellbook/tmp/issues/{019,020,021,023,041,049,050,051,052,061,070}.md`, `git log --all [-S]` on master (224 commits, no branches/stashes; `a0f31b7`, `6aad3c1`, `af8d5a7` = lost worktree commits, not in the object store), `git show ce25997:spellbook/tmp/{reconstruction_plan,rtabmap_zed_research}.md` (original design notes).
**Key finding: none of the post-2026-08-27 adapter work landed in master.** `git log --all -S` for `_BOUNDED`, `OPEN3D_BOUNDED`, `_recsys_loopfix`, `siftfix`, `SPELLBOOK_BUNDLEFUSION_PRESET`, `rtabmap_replay`, `frame_matches`, `EXTERNAL_ODOM`, `effective_config`, `frame_indices`, `keep_rule`, `stage_timings` returns **nothing**. Master files: `open3d.py`@3a01665 (2026-08-22), `tsdf.py`@29b9796 (08-13), `rtabmap.py`@3a01665 (08-22), `bundlefusion.py`@f939018 (08-27; merges 73ee110/a846c81 on 09-08 leave it byte-identical). Working-tree diffs on these files only relocate paths (`/data/scannetdata`, repo-relative `Server/tools/recons`, `containers/podman_gpu.sh`).
Untracked and recreated today (2026-10-29 20:51/20:53): `reconstruct/containers/podman_gpu.sh` and `containers/rtabmap/Dockerfile`. Both are checked in §3.4/§4.
Inferred (not quoted) items are marked **[inf]**. Runtime cap: 3x scene length, **9004 361.2 s, 9009 471.1 s**. Scoring is outside the cap. For BundleFusion only the solver is timed (§2).

---------------------------------------------------------------------------------------------------
## 1. Open3D (issues 19, 20, 21, 51, 61)

### 1.1 Timeline
| date | source | event |
|---|---|---|
| 2026-08-10 | 5de6671; #19/#20/#21 | `integrate(..., inv(e))`, single floor transform, K scaled from frame dims. Residual 1.9 mm |
| 2026-08-13 | 29b9796 | shared `tsdf.py` (legacy ScalableTSDFVolume, 2 cm, CPU) |
| 2026-08-22 | 3a01665 | `gpu_check()`, `GPU_POLICY="cpu"`. **Master open3d.py stops here** (Stage-A OK poses + 2 cm TSDF) |
| 2026-08-31 | #51 c.08-31 | classic `reconstruction_system` adapter: `write_pinhole_camera_intrinsic`, column-major pose-graph decode, loop-gate persistence (worktree; committed later as `a0f31b7`, lost) |
| 2026-09-01 | #51 | `scene9004_40` classic 26.403 cm (make/register/refine 1620/11597/1866 s) |
| 2026-09-08 | #51 c.09-08 11:28 | rerun on master `818cd57`: **9004_40 26.145** (acc 26.517 / comp 25.773), 9009_40 0 loops |
| 2026-09-09 | #61 | batch-1 R1 voxel 0.03 interrupted; `_recsys_loopfix/` vendored (R4: RANSAC (4000000,0.9999), info gate 0.12, prune 0.10). **R4 never run, excluded** |
| 2026-09-15 | #61 c.20:59 | **runtime-rule purge** `_40` (15275 s = 126.8x) and `_41` (82265 s). Classic path dead under the cap |
| 2026-09-15 | #61 c.21:34 | S2 fast A–L on Metashape `_17` poses. No promotion (scratch candidates, not adapter code) |
| 2026-09-16 | #51 c.06:16 | reset plan: official fusion defaults + "deterministic source-pose integration (stride-9 pool poses, D-conjugation)" |
| 2026-09-17 01:00 | #51 | B1 stride-9 (403/3612) tensor: `vbg.extract_triangle_mesh` CUDA illegal memory access (3 jobs, GPU1 and GPU2) |
| 2026-09-17 01:25 | #51 | **B1 max-80 PASS: `scene9004_40` 29.676032** (23.181353/36.170710), timing 201.55 s. 9009 max-80 fails the same way |
| 2026-09-17 01:43 | #51 | **9009_40 max-60 PASS** (134.22 s). `_BOUNDED_MAX_KEEP_BY_SCENE={9004:80,9009:60}`. #51 closed |
| 2026-09-26 | #61 | S2 3x batch 41-46 (199/320 frames, masks, weight 2.0, prune), best 18.43. No promotion. **"unsuccessful S2 candidate profile/CLI machinery rolled back; successful bounded B1 baseline remains (`_BOUNDED=True`, `_BOUNDED_PROFILES=("b1",)`, run-0-only guard)"** |
| 2026-10-04 | #61 | batch-2 (t1_depth3m … h3_trajobb) PROMOTE NONE, slots purged. **Excluded** |
| now | HEAD | no B1, no tensor backend, no provenance files. No scans on disk |

### 1.2 Target state (B1, as of 2026-09-26)
- `GPU_POLICY="managed"`. Tensor CUDA VBG on the leased GPU (`CUDA_VISIBLE_DEVICES=<gpu>` -> `CUDA:0`). Needs open3d 0.19.0 with CUDA (`cuda_available True`) in the `3disspellbook` env.
- Backend selector `OPEN3D_BOUNDED_BACKEND=tensor`. Any other value falls back to legacy and logs a fallback string. A valid baseline log must not contain that string.
- Poses: "pool-D" = the shared Stage-A pool pose `P` (`frames/pose/<i>.txt`, ZED Y-up), used as `D@P@D` with `D=diag(1,-1,-1,1)`; extrinsic `inv(D@P@D)`. Convention `zed_opencv`, as in master.
- Keep rule `full-span-max-K`: `idx = np.rint(np.linspace(0, N-1, K)).astype(int)`, K = `_BOUNDED_MAX_KEEP_BY_SCENE[scene]`. This reproduces the published 9009 list exactly and the 9004 list `0,46,91,…,3565,3611` (checked).
- Fusion, quoted: "ScanNet fusion 320x240 voxel 0.010 trunc 0.060 depth 0.1-4.0 scale 1000, Tensor VBG block 16/50000 trunc_mult 6.0". Color LINEAR and depth NEAREST resize (as tsdf.py). `extract_triangle_mesh()` uses its default weight_threshold 3.0 (#61: "wt-thresh 3.0"; 2.0 was the H2 loser). Integrate color too **[inf]**, because vh_clean must be colored.
- Profiles: `_BOUNDED=True`, `_BOUNDED_PROFILES=("b1",)`. Run-0-only guard: bounded B1 applies only to slot `_40`.
- Provenance written by the adapter or run.py:
  - `recon/run.json`: backend tensor, profile b1, keep `80/3612`, keep_rule `full-span-max-80`, max_keep 80, poses `zed-pool-D`, issue 51, duration 120.400110, limit 361.200330, fps 15.
  - `recon/effective_config.json`, `recon/open3d/reconstruction.log` (sid/backend/profile/keep/config/timings), `recon/stage_timings.json`, `recon/frame_indices.npy`, an adapter snapshot, `cmdline.txt`, `svo_path.txt`.
- Reference hashes (#61 09-26): 9004 run `caa6f245…`, config `b1f837b2…`, score `b4792497…`. 9009 run `9b3043ad…`, config `b2c6bc3b…`.

### 1.3 Gaps vs master HEAD
1. `open3d.py`: CPU legacy 2 cm on all OK frames with `depth_trunc 6.0` -> B1 tensor VBG bounded (whole body missing).
2. No `OPEN3D_BOUNDED_BACKEND` handling. `jobs.py` `ENV_ALLOWLIST`/`ENV_PREFIX_ALLOWLIST=("SPELLBOOK_",)` **drops it in detached jobs**, while #51 says "job.json env preserved". Add `"OPEN3D_BOUNDED_BACKEND"` to `ENV_ALLOWLIST`, or read `SPELLBOOK_OPEN3D_BACKEND` with a fallback **[inf]**.
3. No run.json, effective_config, stage_timings, frame_indices or adapter snapshot (shared run.py gap; see recon-a §ZED item 3).
4. Not restorable from git: the classic path (`a0f31b7`) and `_recsys_loopfix/`. Neither is needed for the target.
5. `tsdf.py` stays as is (the zed engine uses it). B1 code lives in `open3d.py`; adapters must not import each other.

### 1.4 Restore spec
- `engines/open3d.py`: keep `preflight` (add a `o3d.core.cuda.is_available()` check) and `gpu_check` (switch to managed: require gpu+lease_fd, probe `o3d.core.Device("CUDA:0")`). The new `reconstruct(work, root, gpu, lease_fd)`:
  1. read pool `pose_state.txt`/`pose/*.txt` and `intrinsic_depth.txt`.
  2. compute `idx` with the keep rule. Abort if any selected frame is not `OK` **[inf]**.
  3. `vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'), attr_dtypes=(f32,f32,f32), attr_channels=(1,1,3), voxel_size=0.010, block_resolution=16, block_count=50000, device=CUDA:0)`.
  4. per frame: resize to 320x240, K scaled per axis from the frame dims, `ext=inv(D@P@D)`, `compute_unique_block_coordinates(depth, K, ext, 1000.0, 4.0, trunc_voxel_multiplier=6.0)`, then `integrate(...)` with depth_min 0.1 **[inf: official integrate.py API]**.
  5. `vbg.extract_triangle_mesh().to_legacy()` -> `engine_native.ply`. Return `(native, D-conjugated poses, idx, "zed_opencv")`.
- Write provenance (§1.2) and record fuse and total in `stage_timings.json`.
- Use `tensor` as the code default as well, because the env does not reach detached jobs (gap 2).

### 1.5 Reproduce
`OPEN3D_BOUNDED_BACKEND=tensor /data/…/3disspellbook/bin/python spellbook/main.py --engine open3d --scene 9004` (then `--scene 9009`, one at a time). Expected: `scene9004_40` 29.676 ±0.5 cm, `timing.json` ≈201.6 s ≤361.2 (reconstruct+finalize; job wall 433 s incl. scoring). `scene9009_40` 60 frames, ≈134 s ≤471.1, engine_native 1,897,967 v.
Risk: 9004 at 403 frames and 9009 at 80 frames crash deterministically in VBG extract with 50000 blocks. Keep the K caps.

---------------------------------------------------------------------------------------------------
## 2. BundleFusion (issues 50, 52)

### 2.1 Timeline
| date | source | event |
|---|---|---|
| 2026-08-09 | ce25997 plan | fork = FangGet/BundleFusion_Ubuntu_Pangolin at `/data/zed-bundlefusion` (lost). Images `bundlefusion:latest` / `:latest-hostdriver` (9.15 GB). Fork patch list (§2.5) |
| 2026-08-10/13/22 | 5de6671, 29b9796, 3a01665 | docker adapter, managed GPU lease, gpu_check |
| 2026-08-27 | f939018 | **canonical 10 mm** (drop 4 mm), effective configs persisted. **= master HEAD** (rounds 10, timeout 5400, stretch-resize staging, 1 GPU) |
| 2026-08-31 | #52, #50 b1 | official 9004 80.265 (restored value). b1 winner siftfix 640x480 1.5M/2.0M/10M `s_minKeyScale 1.0` 36.223 (`_51`) |
| 2026-09-08 | #52 | attempts 2-6: official 56-92 cm, teleports. Attempt 6 = byte-copy staging + cv2 preload. Timeout 10,800 s. Adapter "restored to `bundlefusion:latest`" |
| 2026-09-08 | #50 | presets `SPELLBOOK_BUNDLEFUSION_PRESET` (official-default default, `_51`, `_52` 35.946 `s_minKeyScale 0.5`) |
| 2026-09-09 | #50 b3 | no winner. **Env overlays `SPELLBOOK_BUNDLEFUSION_APP/BUNDLING/IMAGE/NAME/HYPOTHESIS` kept**. verifyfix/globaldense images lose |
| 2026-09-10 | #50 b4 | `_54` 35.710 (`s_surfAreaPcaThresh 0.04f` on `batch2-scene9004_52`) |
| 2026-09-14 | #52 | strict original run `_57` 55.99 cm, 1051 s ("realtime FAIL") |
| 2026-09-15 | #50 c.20:59 | **runtime-rule purge** `_50,_51,_52,_54`, 9009 `_50` (11.67x) |
| 2026-09-17 | #52 v2/v3 | crop-correct K, pair GPUs, opt0. Staging 357-616 s blows the cap -> prepared-volume (staging-free). Solver over cap (578/861 s wall scope) |
| 2026-09-26 | #52 v4 | **corrected scope: solver = Docker BF process only.** C1 9009 60.32 s, C2 9004 156.26 s PASS. Retained |
| 2026-10-01 | #52/#50 | **baseline confirmed `scene9004_50` 50.745 cm (28.282/73.207), `scene9009_50`**. 6 candidates, no winner |
| 2026-10-06 | #50 b5 | 0/6 (one overlay each). Baseline immutable. **= current baseline** |

### 2.2 Target state (2026-10-01 baseline)
- Image `bundlefusion:latest@sha256:528315a1324aa6bedeaf7f474bcef9c21f44038affed4647822133a1019c7b51`. No source patches, no siftfix.
- Configs, quoted: "canonical ScanNet configs with only the fork-required sensor/K adaptation". Source shas match the repo files today: app `347dbdf6…` and bundling `d8c9e284…` (`Server/tools/recons`, verified).
  - App patch = master `_patch_app_config` (`s_sensorIdx 8->1`, append `s_cameraIntrinsicFx/Fy/Cx/Cy`). 320x240, 600k/800k/5M, `s_minKeyScale=3.0`.
- Input, quoted: "validated prepared 640×480 sensor stream", "crop-correct". This is a center crop to 4:3, then resize to 640x480 (color LINEAR, depth NEAREST), with `fx'=fx·s`, `cx'=(cx−x0)·s`, `cy'=(cy−y0)·s`.
  - Expected K, quoted: 9004 `294.12/294.12/312.71/239.79`, 9009 `329.41/329.41/311.96/239.67`. Both are consistent with crops 1600x1200 (s=0.4) and 1440x1080 (s=0.444).
  - Prepared once per pool, outside the solver clock. Preflight `PREPARED_HIT+VOLUME_HIT+PAYLOAD_OK` with source/cache/volume hashes. Manifests: 9009 `aad0e2cb`, 9004 `ab6c902f`.
- Run: "dual GPUs", "ordered pair [1,2]", "rounds 0", i.e. `CUDA_VISIBLE_DEVICES=1,2` and headless arg `0` (master passes 10). Safety timeout 10,800 s.
- Timing: solver = container input-read -> pose/mesh write, ≤3x SVO elapsed.
  - Stages `prepared_validate/solver/output_copy/align/clean/finalize/segmentator/scoring` are timed separately. 9004: 1.23/156.26/0.55/3.88/45.38/683.07/2.51/115.50 s.
  - Delivery 892.97 s is "info-only". Note that this is over the 3x cap if the whole pipeline is measured.
- Presets, kept but not the default: `official-default` (default), `batch1-scene9004_51`, `batch2-scene9004_52`, plus the b4 `_54` profile. All their artifacts were purged as noncompliant.
- Provenance: `recon/run.json` (issue 52, variant `official-default`, commit, image digest, physical GPUs, timeout, `git_dirty`) and an experiment json, plus effective configs (in master already).

### 2.3 Gaps vs master HEAD (`bundlefusion.py`)
1. `_OPT_ROUNDS = 10` -> `0`. `_TIMEOUT = 5400` -> `10800`.
2. `_stage_input` stretches 1920x1200 -> 640x480, re-encodes JPEG, stages inside the timed run, with `K·(640/dw, 480/dh)` -> crop-correct prepared cache + volume, outside the clock. Also preload `cv2` before mesh libs (GLib order, #52 09-08).
3. Single lease, `CUDA_VISIBLE_DEVICES={gpu}` -> ordered **two-GPU** lease. `utils/gpu.gpu_lease` leases one GPU, so add a pair lease **[inf]**.
4. Timing: run.py times reconstruct+pipeline -> engine-declared solver scope plus `stage_timings.json`.
5. Missing `SPELLBOOK_BUNDLEFUSION_PRESET` and the APP/BUNDLING/IMAGE/NAME/HYPOTHESIS overlays (merged over the frozen preset), digest pin check, run.json/experiment json.
6. **Host: `docker` socket permission denied for this user**, and only podman images exist. Either the user is added to the docker group, or the adapter is ported to rootless podman with `podman_gpu.sh 1,2` **[inf]**.

### 2.4 Restore spec (bundlefusion.py)
- Constants: `_IMAGE="bundlefusion:latest"`, `_IMAGE_DIGEST="sha256:528315a1…1019c7b51"` (verify), `_OPT_ROUNDS=0`, `_TIMEOUT=10800`, `GPU_COUNT=2`.
- `PRESETS`, quoted from #50/#52:
  - `official-default`: image latest, 320x240, `s_hashNumSDFBlocks 600000`, `s_hashNumBuckets 800000`, `s_marchingCubesMaxNumTriangles 5000000`, `s_minKeyScale 3.0f`, `s_surfAreaPcaThresh 0.032f`.
  - `batch1-scene9004_51`: image `bundlefusion:spellbook-siftfix`, 640x480 integration/raycast, 1500000 blocks, 2000000 buckets, 10M triangles, `s_minKeyScale 1.0f`.
  - `batch2-scene9004_52`: `_51` + `s_minKeyScale 0.5f`.
  - `s2b4-surfarea-004`: `_52` + `s_surfAreaPcaThresh 0.04f`.
- Overlays: `SPELLBOOK_BUNDLEFUSION_APP` / `_BUNDLING` (`key=value;…` merged into the configs **[inf format]**), `_IMAGE`, `_NAME`, `_HYPOTHESIS` -> run.json.
- `prepare(scene)`, a separate CLI step:
  - center-crop, resize, K as in §2.2, written to `<scannet_root>/derived/reconstruction/bundlefusion/prepared/scene900N/{color/<i>.jpg,depth/<i>.png,manifest.json}` **[inf path]**.
  - The manifest holds the pool manifest sha, crop, K, and per-file sha.
- `reconstruct()`: validate the prepared manifest, write the patched configs, run the container with `CUDA_VISIBLE_DEVICES=g1,g2` (docker `--privileged` or podman+gpu args) and rounds 0, timing the process only.
  - Keep master's log checks (`[info] processed N/M`, `[warn] skipping frame`), `_parse_poses` (drop non-finite poses, never identity) and convention `bundlefusion_world`.
- Expected 9004: 3612 staged / 3577 kept, 2315 opt + 1262 int, 184 lost / 186 invalid, invalid submaps 1210/1220, teleport 8.64 m @2211, **50.745 ±0.5 cm** (high nondeterminism, 0.3–0.96 cm band).
- Expected 9009: 2356/2356, solver ≈60 s.

### 2.5 Container build spec (no Dockerfile in any issue; nothing in git)
- `bundlefusion:latest`:
  - Source: FangGet/BundleFusion_Ubuntu_Pangolin. `/data/bundlefusion` is upstream niessner (Windows), **not** the fork.
  - Base: CUDA devel image + Eigen, OpenCV, Pangolin.
  - Patches (ce25997 plan): CUDA 11+ `__shfl_down_sync` in `cudaUtil.h`/`ProgramCU.cu`/`SolverBundlingUtil.h`; `FDIV`->`/` (`ProgramCU.cu:50`); `sm_86` in `CMakeLists.txt:14` (RTX A4000); `RGBDSensor.h:87-88` start/stop of `m_bIsReceivingFrames`; bundling-thread exit check (`BundleFusion.cpp:519-529`); `deinitBundleFusion` cleanup (`:443-460`); **save poses before the reintegration drain** (`:889-895`).
  - Headless binary `/app/build/bundle_fusion_headless <app.txt> <bundling.txt> <in_dir> <out_dir> <rounds>`:
    - reads `in/color/<i>.jpg` and `in/depth/<i>.png` (PrimeSense dir, `s_sensorIdx=1`).
    - writes `out/pose/<i>.txt` (row-major c2w, finite only, filename = frame index) and `out/mesh.ply`.
    - prints `[info] processed N/M frames`.
  - Host driver libs baked in (`latest-hostdriver`) or mounted. Dual-GPU via `CUDA_VISIBLE_DEVICES` pair **[inf]**.
  - The digest cannot be rebuilt bit-exactly, so the new image is a new provenance. Re-baseline and accept ±0.5 cm.
- `bundlefusion:spellbook-siftfix` (digest `sha256:9f9651c8…04ccb`): `latest` + FangGet PR #28 and PR #31 (SIFT feature-count and match-sort). **Not needed for the current baseline**, only for historical presets.

---------------------------------------------------------------------------------------------------
## 3. RTAB-Map (issues 23, 41, 49, 70)

### 3.1 Timeline
| date | source | event |
|---|---|---|
| 2026-08-09 | ce25997 research/plan | design: `use_zed_odometry`, reprocess (rtabmap #1605), detectMoreLoopClosures, drop `--opt 0` |
| 2026-08-10 | 5de6671; #23 | `CONTAINER_NAME` per pid. live SVO zed_wrapper adapter |
| 2026-08-13/22 | 29b9796, 3a01665 | managed lease, gpu_check. **Master rtabmap.py stops here** |
| 2026-08-30 | #41 | first sanity on `debug/reconstruction-rtabmap`: 9004_20 55.135 cm. Adapter corrections listed (§3.3 gap 2) |
| 2026-09-08 | #49 | forwarded s3: shared-pool replay, DetectionRate 0. **9004_20 30.832** (1492 poses, 9839.7 s), 9009_20 1087 poses (6838.9 s) |
| 2026-09-09 | #41 | batch-1 T1–T3/H1–H3 planned, stalled 16.5 h, no results. **Excluded** |
| 2026-09-15 | #41 c.20:59 | **runtime-rule purge** 9004_20 (81.67x), _21 (no timing), 9009_20 (43.53x). Verbatim params kept (§3.2) |
| 2026-10-01 | #70 (+#49 ptr) | restart under 2x cap 240.80 s: c1 pure VO starved; c1 assisted timeout; c3 s90 eval over budget. **c4 s90: recon 91.401 s, `_20` assisted-hybrid 20.515862** (21.5595/19.472223), 41/41, span 99.7% = **current baseline** |
| 2026-10-04 | #41 | restart-s2 `_21`.. `_26` (s72, s60, replicate, sdf.04, depth3 17.967, ICP). No promotion, all deleted. **Excluded** |

### 3.2 Target state (`_20` assisted-hybrid s90, 2026-10-01)
Not written down verbatim anywhere. It is reconstructed from the `_21` `live91-hybrid-s45-v02` deltas (#41 09-15) plus #70. "Explicitly not pure SLAM".
- Replay from the shared pool (base: #49 `rtabmap_replay.py` "legacy 15 Hz rgb8/32FC1 identical-stamp path").
  - Frames are sent with a per-frame ack handshake ("81/81 frames acked"). `FAST=1` **[inf: no realtime pacing]**.
- `MAP_ONLY=1`, map stride **90**: only pool frames `0,90,…,3600` (41 frames, 3600/3611 = 99.7% span, matches "41/41 span 99.7%") go to `/map_rgbd/*`.
- `EXTERNAL_ODOM=1` ("assisted"): odometry is the Stage-A pool ZED pose, published as odom/TF **[inf]**. There is no `rgbd_odometry` node.
- Nodes: `rgbd_sync_map` on `/map_rgbd/*` -> `rtabmap` on `/map_rgbd/rgbd_image` + odom.
- Effective params, quoted from `inside.sh` deltas on the #49 baseline:
  - rgbd_sync `subscribe_rgbd:true`, queue 30/30.
  - odom `frame_id:camera_link odom_frame_id:odom publish_tf:true +wait_for_transform_duration:1.0`.
  - rtabmap `Mem/IncrementalMemory:true Rtabmap/DetectionRate:0 RGBD/LinearUpdate:0 RGBD/AngularUpdate:0`, `wait_for_transform:0.5 tf_tolerance:0.5`, plus `approx_sync:true Kp/MaxFeatures:400 Optimizer/Strategy:2 Rtabmap/ImagesAlreadyRectified:true Rtabmap/ImageBufferSize:0`.
- No reprocess and no detectMoreLoopClosures. Export is poses only: `rtabmap-export --poses_camera --poses_format 11 --opt 2` (GTSAM).
- Mesh: TSDF fusion of the 41 node frames at the exported poses **[inf]**.
  - Evidence: the export has no mesh flags, and the losers were named `sdf.04`/`depth3`.
  - Defaults **[inf]**: voxel 0.010, trunc 0.06, depth 0.1–4.0, 320x240 (= open3d B1 fusion).
- `.sens` "written only after pass", "sparse-direct" (41 frames). "validate_scan PASS".
- Frame mapping via `frame_matches.tsv` (node -> pool index; #41 RGB-content match). With the stride replay the indices are known directly.
- Visual gate B: walls present, no spray, no double layer.

### 3.3 Gaps vs master HEAD (`rtabmap.py`)
1. The whole architecture: master is a live SVO zed_wrapper -> rtabmap -> reprocess -> mesh export. The target is pool replay + external odom + stride-90 map-only + GTSAM pose export + external TSDF.
2. #41 fixes missing from master even for the legacy path:
   - YAML trailing comma `"Mem/IncrementalMemory": "true",` (invalid).
   - `grab_resolution 'NATIVE'` -> `SVGA`.
   - `approx_sync_max_interval 0.5` -> `2.0`.
   - export `--voxel 0` -> `0.01`.
   - reprocess `-odom` -> drop it.
   - SVO `-inf` preflight and fail-fast node death.
   - contiguous-id `keep` -> RGB-content `keep` (accept adjacent/clearly-better matches, drop the trailing unmatched SVO-last frame).
   - Open3D textured-OBJ abort -> cv2 OBJ+UV bake.
3. Missing pieces: `rtabmap_replay.py`, `inside.sh` profile switches (FAST/MAP_ONLY/EXTERNAL_ODOM/stride), the poses-only path, TSDF fusion, sparse `.sens`, run.json, `TIMEOUT` (7200) vs cap watchdog.
4. Cmdlines ran with `--frames <pool>`. `batch.py` already passes it.

### 3.4 Restore spec
- Keep `rtabmap.py` `gpu_check`, `_podman_gpu_args`, `_parse_poses` (format 11), `CONTAINER_NAME`.
- Add the profile `assisted-hybrid`, defaults `stride=90`, `map_only=1`, `external_odom=1`, `fast=1`. Env `SPELLBOOK_RTABMAP_PROFILE`/`SPELLBOOK_RTABMAP_STRIDE` **[inf names; SPELLBOOK_ prefix survives jobs.py]**.
- `rtabmap_replay.py`, mounted into the container:
  - rclpy publisher of pool `color/<i>.jpg` -> `rgb8`, `depth/<i>.png`/1000 -> `32FC1`, and camera_info from `intrinsic_depth.txt` (color resized to depth res).
  - One shared stamp per frame. Odom/TF `odom->camera_link` from `D·P·D` converted to ROS (x fwd, z up) **[inf]**.
  - Publishes the map stride on `/map_rgbd/*` and waits for the rtabmap ack (node count) before the next frame.
- `inside.sh`: the params quoted in §3.2, then `rtabmap-export --poses_camera --poses_format 11 --opt 2 <db>`.
- Adapter: parse poses, then map node -> pool index (`range(0,N,stride)`). Fuse the TSDF (reuse the B1 VBG routine in a shared module, e.g. `tsdf.py`, since adapters must not import each other). Write `engine_native.ply`. Return keep = node pool indices.
  - Convention `rtabmap_map` if the odom is ROS z-up **[inf, verify backprojection ≤20 mm]**.
- `containers/podman_gpu.sh` (recreated) matches every call site:
  - call: `bash podman_gpu.sh <idx>`, 60 s timeout.
  - output: one podman arg per line, rc≠0 on error.
  - it mounts all `/dev/nvidia*`, the host `libcuda/libnvidia*` and `nvidia-smi`, and sets `CUDA_VISIBLE_DEVICES=<physical idx>`, which `gpu_check` asserts.
  - For BF over podman: allow `^[0-9]+(,[0-9]+)*$` **[inf]**.

### 3.5 Container spec `localhost/zed-rtabmap:jazzy`
- Known facts (#41/#49): ROS 2 Jazzy; ZED SDK 5.4.1; `zed_wrapper`; rmw_cyclonedds_cpp; `/opt/ros/jazzy` + `/root/ros2_ws/install`; tools `rtabmap-reprocess`, `rtabmap-detectMoreLoopClosures`, `rtabmap-export`; "image defaults (GTSAM optimizer, GFTT/ORB, no SIFT)".
- Mounts used by the adapter: `/input` (SVO dir ro), `/output`, `/usr/local/zed/{resources,settings}` from `$ZED_RTABMAP_DATA` (default `/data/zed-rtabmap`), `/inside.sh`.
- The recreated `containers/rtabmap/Dockerfile` (cuda 13.0.2 runtime noble + ros-jazzy-rtabmap-ros + ZED SDK 5.4 cu13 + zed-ros2-wrapper `v5.4.1`) covers this. **Add** `ros-jazzy-cv-bridge python3-opencv python3-numpy` for the replay. Verify that the wrapper tag exists and that `rtabmap --params | grep Optimizer/Strategy` reports GTSAM available.

### 3.6 Reproduce
`SPELLBOOK_RTABMAP_PROFILE=assisted-hybrid SPELLBOOK_RTABMAP_STRIDE=90 python spellbook/main.py --engine rtabmap --scene 9004`. Expected `scene9004_20` 20.516 ±0.5 cm, 41 poses, recon ≈91.4 s ≤361.2 (eval 141.7 s outside). 9009 had no assisted-hybrid baseline: same CLI, stride 90 -> 27 nodes; no score target.

---------------------------------------------------------------------------------------------------
## 4. Shared run.py / batch.py / extract.py items (these engines)
1. `run.py`: run.json, `stage_timings.json`, per-engine timing scope (BF = solver only; open3d/rtabmap = reconstruct+finalize), scoring outside the cap, multi-GPU lease for BF, run-0 guard. See recon-a item 3 for the shared `--offline-score`/watchdog design.
2. `extract.py`: `promote_scene9004_frames` needs the lost `scene9004_04/recon/frames`. Extract 9004 directly from the SVO and check pool counts **3612 / 2356** (both baselines assume these exact pools; the BF prepared manifests depend on them).
3. `jobs.py`: env allowlist (open3d gap 2).
4. `batch.py`: per-task single lease. BF needs 2 GPUs: semaphore or slot accounting **[inf]**.
