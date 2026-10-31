# Recovery report A: zed, isaac, metashape reconstruction adapters

Sources: `spellbook/tmp/issues/{018,029,039,047,057,022,042,048,025,040,053}.md` (+041/050/061 for the 2026-09-15 runtime rule), `git log`/`git log --all -S` on master (224 commits, no branches/stashes).
**Key finding: none of the working adapter code (native ZED mapping, Isaac public-deb graph, Metashape presets/scale bars/native depth, run.json/offline-score/threaded finalize) ever landed in master.** `git log --all -S` for `b5-gen1-neural`, `gen3-stability10`, `native-default`, `ZED_VARIANT`, `optimized_cad_best`, `conf100_best`, `map_clearing_radius`, `SCALE_BARS`, `_verify_rendered`, `SPELLBOOK_TARGET_RUN0`, `score_scan_at`, `engine_timing`, `0206fa01` returns **nothing**. Only restorable-from-git pieces:
- `git show 5de6671:spellbook/reconstruct/engines/zed.py` - two-pass `.area` tracking skeleton (pass1 `enable_area_memory`+`save_area_map`, pass2 `area_file_path`+`get_position` loop, `_init()` with NEURAL_PLUS / 0.1-6.0 m / METER / RIGHT_HANDED_Y_UP). TSDF mesh route (superseded), but the grab/pose loop is the "proven grab()/get_position() loop".
- `spellbook/jobs.py` HEAD already has `ENV_PREFIX_ALLOWLIST = ("SPELLBOOK_",)` (6523c26, 2026-09-10) = the #40 env-passthrough fix. Nothing to restore; see ZED caveat below.
- `git show ce25997:spellbook/tmp/scout_zed_official.md`, `cuvslam_nvblox_research.md`, `scout_metashape_official.md` - official-API research notes (reference only).
Everything else must be re-written from the issue text below. Inferred (not verbatim) items are marked **[inf]**.

Runtime cap (all non-Metashape, from 2026-09-15 rule): recon wall < 3x scene length: **9004 361.2 s, 9009 471.1 s**; queue/lease-wait and official CAD scoring excluded. Metashape exempt (unbounded since 2026-09-10).

---------------------------------------------------------------------------------------------------
## 1. ZED (issues 18, 29, 39, 47, 57; #25 context)

### 1.1 Timeline
| date | source | event |
|---|---|---|
| 2026-08-10 | 5de6671; #18 | TSDF-route zed.py; `sdk_gpu_id!=0` -> constant poses; never set `sdk_gpu_id` |
| 2026-08-22 | 3a01665; #29 | zed.py `GPU_POLICY="blocked"` (GPU0 reserved) |
| 2026-08-23 | 53290ca | master zed.py final state = blocked stub (37 lines). **Master stops here** |
| 2026-08-24 | #29 c.08-24 | remap validated on worktree: `GPU_POLICY=managed`, `CUDA_VISIBLE_DEVICES` before `import pyzed`; 9004 TSDF 26.92 cm |
| 2026-08-27 | #29 c.08-27 | native ZEDfu: GEN_1 pass1 + async `save_area_map`; pass2 `.area` reload + stock MESH mapping; 9004 31.06 cm |
| 2026-08-28 | #29/#39 b1 | SHORT 3.5 m + conf 50 = 24.02; GEN_3+stability 10 -> **winner `_03` 23.621 (gen3-stability10)** |
| 2026-08-31 | #47 att.2 | **official baseline `native-default`** `_00` 28.936 cm (att.1 31.06; mislabel corrected) |
| 2026-09-01 | #39 b2 | no promotion (stab4, conf40, range5, loc-only, IMU off, MEDIUM filter) |
| 2026-09-08 | #57 | baseline refresh @17d28d0: 9009 OK (2171.7 s), 9004 29.852 cm (2895.6 s) |
| 2026-09-09 | #39 b3 | **winner `b3-neural` `_02` 23.485877** (NEURAL_PLUS->NEURAL); seek600/threepass rejected |
| 2026-09-10 | #39 b4 + correction | invalid batch (confounded base); b4 entries rolled back |
| 2026-09-10 | #39 b5 | **winner `b5-gen1-neural` `_06` 22.601959**; adapter default flipped to it |
| 2026-09-14 | #39 b6 | 1-pass vs 2-pass A/B: no promotion, `b6-gen1-neural-1pass` removed; *retained generic deferred-.sens support (`score_scan_at`, `--scan-dir`, `keep.npy`, `engine_timing.json`)*; historical anchors void |
| 2026-09-15 | rule (#41/#42/#50/#61) | **runtime-rule purge**: all zed slots gone (9-16 log: "no scene9004_0*/scene9009_0* remain") |
| 2026-09-16 | #47 att.1-5 | s3 rebaseline @6523c26: att.2/3 timeout (serial finalize); **threaded finalize 0206fa01** -> 9004 `_00` 302.0 s, offline score **30.662521**/40.11/21.21; 9009 `_00` 262.8 s PASS |
| 2026-09-16 | #39 b7 | **winner `b5-gen1-neural` `_04` 21.395865** (247.3 s); fastest `s2b7-native-onepass` `_05` 201.0 s 30.54 (named, non-default); `gen3-stability10` TIMEOUT (non-compliant) |
| 2026-10-04 | #39 b8 | **current best `s2b8-fixedcalib` `_03` 21.306363**/27.520/15.092 (231.26 s); retained, **default stays `b5-gen1-neural`** (no flip approved) |

### 1.2 Target state (zed.py as at b8 closeout = b7 adapter + one named preset)
- `GPU_POLICY="managed"`, `SERIAL=True`; no pyzed import at module level; `CUDA_VISIBLE_DEVICES=<leased>` set before `import pyzed.sl`; never set `sdk_gpu_id`; never lease GPU 0.
- Variant selected by env `ZED_VARIANT` (verbatim), default `b5-gen1-neural`.
- Pipeline (two-pass, official offline ZEDfu): open SVO (`svo_real_time_mode=False`, METER, RIGHT_HANDED_Y_UP), pass 1 tracking (`enable_area_memory=True`, mode per preset) over whole SVO, async `save_area_map(area)` + poll `get_area_export_state()` until done ("area_export" split); close. Pass 2: reopen, `area_file_path=area`, same tracking mode, `enable_spatial_mapping(SpatialMappingParameters(map_type=MESH, ...))` in the **same pass** as pose capture via `grab()`/`get_position(WORLD)` only (**no per-frame `get_positional_tracking_status()` -> SIGSEGV**); `extract_whole_spatial_map(mesh)`, `mesh.filter(MeshFilterParameters LOW)`, colored OBJ export (`engine_native`). Keep frames with state OK and `i < total-1`; require >=100 poses. Log mapping state (INITIALIZING->OK; fail on FPS_TOO_LOW/NOT_ENOUGH_MEMORY).
- `passes` key: 2 normally; 1 for `s2b7-native-onepass` (no `.area`, single open).
- Write native split to `recon/engine_timing.json`: `open / tracking / area_export / mapping / mesh / native_total` (#47 9-16).
- Write `recon/engine_config.yaml` + `recon/run.json` with `variant`, `issue`, `commit`, schema 1.
- Presets (verbatim values; base for all = native-default unless stated):

```python
_NATIVE_DEFAULT = dict(passes=2, tracking="GEN_1", depth_mode="NEURAL_PLUS",
    depth_min=0.1, depth_max=6.0, resolution_meter=0.05, range_meter=0.0,  # AUTO
    confidence_threshold=95, texture_confidence_threshold=100, stability_counter=0,
    max_memory_usage=2048, mesh_filter="LOW", depth_stabilization=30,
    camera_disable_self_calib=False, use_chunk_only=False, pose_smoothing=False,
    enable_imu_fusion=True, save_texture=False)
PRESETS = {}  # built in order (later entries reference earlier ones)
PRESETS_ORDERED = {
  "native-default": _NATIVE_DEFAULT,                                   # #47/#57 baseline
  "gen3-stability10": dict(_NATIVE_DEFAULT, tracking="GEN_3", range_meter=3.5,
                        confidence_threshold=50, stability_counter=10),  # #39 b1 (now >cap)
  "b3-neural":       dict(PRESETS["gen3-stability10"], depth_mode="NEURAL"),         # #39 b3
  "b5-gen1-neural":  dict(PRESETS["b3-neural"], tracking="GEN_1"),                   # DEFAULT (#39 b5/b7)
  "s2b7-native-onepass": dict(_NATIVE_DEFAULT, passes=1),                # fastest, never default
  "s2b8-fixedcalib": dict(PRESETS["b5-gen1-neural"], camera_disable_self_calib=True),  # current best (#39 b8)
}
DEFAULT_VARIANT = "b5-gen1-neural"
```
SDK placement (b8 verbatim): `texture_confidence_threshold`/`confidence_threshold` -> RuntimeParameters; `depth_stabilization`, `camera_disable_self_calib`, `depth_mode`, depth min/max -> InitParameters; `use_chunk_only`, `resolution_meter`, `range_meter`, `stability_counter`, `max_memory_usage` -> SpatialMappingParameters; tracking mode / `enable_imu_fusion` / `enable_pose_smoothing` -> PositionalTrackingParameters. pyzed 5.4 has no `decay`/`enable_forget_past`/`disparity_std`; use `set_svo_position` (not `set_svoposition`).
- Excluded (rolled back): all b2/b4/b6 entries, seek600/delayed-mapping/threepass/localization_only dispatch, temporary slot dispatch.

### 1.3 Gaps vs master HEAD
1. `engines/zed.py`: entire native adapter missing (blocked stub). Restore skeleton from `git show 5de6671:...zed.py` (lines 23-110), drop the `open3d_engine._integrate` call, add mapping/presets/managed GPU/`gpu_check` per 1.2.
2. `engines/__init__.py`/`gpu_check.py`: stale "blocked" refs removed on worktree (#29 c.08-27) - check `git grep -n blocked spellbook/`.
3. `reconstruct/run.py`: no `run.json` (issue/variant/commit), no `--offline-score` (score outside cap), no `--s3-baseline`, no `--scan-dir`, no cap watchdog; scoring is inline -> 9004 baseline att.4 was killed during inline `score_scan` (~216 s). Add `--offline-score` and wall-cap enforcement (rc 124 on overrun) per `zed_watchdog.py` behaviour (worker elapsed vs cap). Also add `SPELLBOOK_TARGET_RUN0=1` guard (metashape #53, verbatim) for run-0 replacement.
4. `reconstruct/geometry_score.py`: no `score_scan_at(scan_dir)` (#39 b6).
5. `reconstruct/finalize.py` + `extract.write_sens`: serial `zlib level=6`. Needs **threaded finalize 0206fa01** = fast valid .sens (`zlib` level 1, ordered parallel compression, atomic tmp+fsync+replace; #22 c.09-26; `_pipeline` 94.4 s on 9004, 65.5 s on 9009). Extract writer pin `acc26df1` (content hash, not a commit). Also persist `recon/keep.npy`.
6. `jobs.py`: passes only `PATH/PYTHONPATH/LD_LIBRARY_PATH` + `SPELLBOOK_*`. `ZED_VARIANT` and `ZED_DIR` are **dropped in detached jobs** (b7 used inline `job_context`). Fix: read `SPELLBOOK_ZED_VARIANT` with fallback `ZED_VARIANT`, and derive ZED_DIR from `settings.yaml: zed_dir` **[inf]**.

### 1.4 Reproduce
Env (b5/b7 verbatim):
```bash
export ZED_DIR=<settings zed_dir, e.g. ~/opt/zed-sdk/install>   # must contain settings/SN46307300.conf
export LD_LIBRARY_PATH=$ZED_DIR/lib:$ZED_DIR/dependencies/ai_dependencies/lib:$CONDA_PREFIX/lib
# missing ai_dependencies/lib -> "NEURAL TRT NOT FOUND" rc-11 at Camera::open
```
Baseline (expect 9004 ~29-31 cm, 9004 recon <=361.2 s [302.0], 9009 <=471.1 s [262.8]):
`ZED_VARIANT=native-default python spellbook/main.py --engine zed --scene 9004` (then `--scene 9009`). Score offline after the cap.
Default/b7 winner: `ZED_VARIANT=b5-gen1-neural ... --scene 9004` (21.395865, 247.3 s; 9009 170.2 s).
Current best: `ZED_VARIANT=s2b8-fixedcalib python spellbook/main.py --engine zed --scene 9004` (21.306363, 231.26 s). Replay noise ~0.6-0.96 cm.
Fastest: `ZED_VARIANT=s2b7-native-onepass` (201.0 s, 30.54 cm). Do **not** run `gen3-stability10` (timeout 361.2 s).
Requires frame pool `derived/reconstruction/frames/scene9004` (3612) / `scene9009` (2356); never `--replace`.

---------------------------------------------------------------------------------------------------
## 2. Isaac (issues 22, 42, 48)

### 2.1 Timeline
| date | source | event |
|---|---|---|
| 2026-08-10 | 5de6671; #22 | NGC-route isaac.py (4 mm, save_ply 60 s, rosbags poses) - never runnable |
| 2026-08-22 | 3a01665; #22 c.08-22 | preflight BLOCKED (image missing); **master stops here** (+ local unstaged GPU_SCRIPT path edit) |
| 2026-08-27 | #22 c.08-27 | public-deb image + composable graph; **baseline `_30` 39.21 cm** |
| 2026-08-28 | #22/#42 b1 | clearing 0 -> 21.47; weight 10 -> 21.33; **winner `_33` NEURAL 20.987116 (`neural-noclear-w10` = `optimized_cad_best`)** |
| 2026-08-31 | #48 | official-default baseline `_30` 39.140 |
| 2026-09-01 | #42 b2 | no promotion |
| 2026-09-08 | #48 | baseline refresh @17d28d0 39.231037 (1223.2 s) |
| 2026-09-09 | #42 b3 | no promotion; two-pass cuVSLAM rolled back to single-pass |
| 2026-09-15 | #42 b5 | no promotion (wmax20 20.80/20.93 at-threshold) |
| 2026-09-15 | #42 c.20:59 | **runtime-rule purge** `_30` (10.15x), `_33` (no timing), 9009 `_30` (5.53x) |
| 2026-09-26 | #48 | s3 recovery: 9009 `_30` 224.28 s PASS; 9004 `_30` **38.914404**, 209.05 s PASS |
| 2026-09-26 | #42 c.19:47 | incumbent restored `_31` optimized_cad_best 21.080823, 221.69 s |
| 2026-09-27 | #42 c.23:16; #22 | **current best `conf100_best` `_34` 20.505585**/23.044/17.967, recon 173.5 s (+57.4 finalize = 230.9 s); new default |

### 2.2 Target state (isaac.py at `_34`)
- Image: public-deb Isaac ROS 4.5 image `zed-isaac-nvblox:spellbook` (no NGC creds); master's `_ensure_image` NGC Dockerfile is wrong. ZED SDK 5.4 + zed-ros2-wrapper inside.
- Graph: one `component_container_mt` with ZedCamera + VisualSlamNode + NvbloxNode, plus `robot_state_publisher`. ZED: SVO playback, `use_svo_timestamps`, half-res publish (`pub_downscale` 0.5 **[inf: b2 "pub_downscale 1.0" was the delta]**), `depth_stabilization 0`, ZED positional tracking off, grab fps 15. cuVSLAM stereo VO `tracking_mode 0` on rectified gray. nvblox `static_tsdf`, fed by cuVSLAM **TF** (topic-transform route fails).
- Pose->frame: rclpy line-buffered recorder (not rosbag) on `/zed/zed_node/status/svo` (`SvoStatus.frame_id`); pose = `P_world_center @ T_center_optical`; branched static-TF graph search; strict final-frame/total SVO gate; all waits bounded; `save_ply` bounded 600 s; per-lease `ROS_DOMAIN_ID = 100 + gpu` (domains 101/102/104 in logs); `recon/isaac_export/track_stats.yaml`, effective-params echo; fixed 300k-face postprocess; mesh gray (bgr8 colour rejected by nvblox).
- Presets (7-key order NEURAL/clearing/decay/voxel/max_dist/trunc/weight, verbatim):
```python
PRESETS = {
 "official_default":  dict(depth_mode="NEURAL_LIGHT", map_clearing_radius_m=5.0,
     decay_tsdf_rate_hz=5.0, voxel_size=0.05, max_integration_distance_m=7.0,
     truncation_distance_vox=4.0, max_weight=5.0),          # + tracking_mode 0; weighting kInverseSquareWeight; ZED conf 95
 "optimized_cad_best": dict(depth_mode="NEURAL", map_clearing_radius_m=0.0,
     decay_tsdf_rate_hz=5.0, voxel_size=0.05, max_integration_distance_m=7.0,
     truncation_distance_vox=4.0, max_weight=10.0),         # run.json variant "neural-noclear-w10"; adapter hash 383834ab
 "conf100_best": dict(PRESETS["optimized_cad_best"], depth_confidence=100),  # DEFAULT (echo "Integer 100")
}
```
- Excluded: VIO (`tracking_mode 1`), 2.5 cm, decay 0/2, workspace bbox, two-pass save_map/localize, wmax 15/20/50, trunc 5/6, weighting variants, replay_rate, `mesh_integrator_min_weight`.
- Fast .sens (zlib-1 ordered parallel atomic) shared with ZED (gap 1.3-5).

### 2.3 Gaps vs master HEAD
1. `engines/isaac.py`: whole public-deb composable graph, recorder, TF search, presets, `ROS_DOMAIN_ID`, 600 s save_ply, export gates missing; master still 4 mm NGC route with `TIMEOUT=7200`. No git source exists; rewrite from 2.2 (+ `cuvslam_nvblox_research.md` in ce25997).
2. Image build recipe (public debs) not in repo: add `spellbook/reconstruct/containers/isaac/Dockerfile` **[inf]** next to the recreated `podman_gpu.sh`.
3. `batch.py` "worker-limit" change (#22 staged trio) - absent; content unknown beyond name.
4. run.py/finalize/run.json/offline-score/cap gaps as ZED 1.3-3..5.
5. Preset selector env: not named in issues; use `SPELLBOOK_ISAAC_PRESET` **[inf]** so jobs.py passes it.

### 2.4 Reproduce
Baseline: `SPELLBOOK_ISAAC_PRESET=official_default python spellbook/main.py --engine isaac --scene 9009` (224.28 s <= 471.1) then `--scene 9004` (209.05 s <= 361.2; 38.914404 cm).
Best: `python spellbook/main.py --engine isaac --scene 9004` (default `conf100_best`; 20.505585, 173.5 s + finalize). Run variance ~0.09-0.14 cm.

---------------------------------------------------------------------------------------------------
## 3. Metashape (issues 25, 40, 53) - exempt from the 3x cap

### 3.1 Timeline
| date | source | event |
|---|---|---|
| 2026-08-22 | 3a01665, 246333f | master metashape.py = pre-#53 defaults, Stage-A depth, `TIMEOUT_S=5400`. **Master stops here** |
| 2026-08-30 | #40 b1 | winner `_13` 15.990 (Ultra/Mild/DisabledInterp + Estimated-guided rematch); baseline 17.332 |
| 2026-08-31 | #40, #53 | slot shuffles; **official baseline `native-default` `_10` 17.915** |
| 2026-09-01 | #40 b2 | **winner `_17` 15.378787 (b2-h3 sparse cleanup RE 1 px / IC 3)** |
| 2026-09-08 | #53 att.2 | rewritten adapter (native depth, metric validation, provenance); baseline refresh `_10` **18.096334** (9004 2740.3 s, 9009 3427.9 s) |
| 2026-09-10 | #40 b3 | no promotion; jobs.py `SPELLBOOK_` passthrough + `_verify_rendered()` + stale-output purge; 10800 s timeout then **removed (unbounded)** |
| 2026-09-15 | #40 b4 | **winner `_11` b4-scalebars3 15.324562** (3644 s) |
| 2026-09-15 | rule | runtime purge: Metashape **exempt** (nothing purged) |
| 2026-10-04 | #40 b5 | **current best `_12` b5-kf350 14.627121**/19.275/9.980 (stage 2602.3 s, total 2742.9 s) |

### 3.2 Target state (metashape.py at `_12`)
Shared fixed: Metashape 2.3.1, fixed ZED pinhole (`f=fx, b1=fy-fx`, centred cx/cy), location priors 0.15 m, rotation priors off, keyframes 0.08 m / 8 deg / gap 45, `cpu_enable=True`, GPU mask from lease, metric-as-is (divide by chunk scale), native Metashape F32 depth exported per keyframe -> uint16 mm (0-5999) depth + color + pose packaged into the scan (sens frames == keep == n_depth); `_verify_rendered()` fails if slot artifacts don't match current env; stale-output purge on slot reuse; **no timeout**.
Profiles (verbatim knobs):
```python
NATIVE_DEFAULT = dict(  # issue 53 "native-default"
  match_downscale=1, keypoint_limit=40000, keypoint_limit_per_mpx=1000, tiepoint_limit=4000,
  generic_preselection=True, reference_preselection=True,      # ReferencePreselectionSource
  align_adaptive_fitting=False, depth_downscale=2, filter_mode="MildFiltering",
  max_neighbors=16, face_count="HighFaceCount", interpolation="EnabledInterpolation",
  source="DepthMapsData", rematch=False, sparse_cleanup=False, scale_bars=0,
  tiepoint_accuracy_px=1.0, location_accuracy_m=0.15, max_keyframes=400)
B2_H3 = dict(NATIVE_DEFAULT, depth_downscale=1,                 # Ultra High
  interpolation="DisabledInterpolation",
  rematch=True,  # 2nd matchPhotos ReferencePreselectionEstimated, guided, reset_matches=False, re-align
  sparse_cleanup=True)  # TiePoints.Filter ReprojectionError 1 px, ImageCount 3, optimizeCameras
B4_SCALEBARS3 = dict(B2_H3, scale_bars=3)  # tracker-distance bars, accuracy 0.02 m, pairs auto >=70% dmax
#   9004 pairs: [2036,853,9.5283], [862,1908,9.7315], [872,1902,9.7193]
B5_KF350 = dict(B4_SCALEBARS3, max_keyframes=350)   # DEFAULT (bare defaults == this)
```
Pins (verbatim, #40 b5): `_10` = native-default/53 + `MAX_KEYFRAMES=400` + depth/interp/rematch/cleanup pins; `_17` = b2-h3 + `SCALE_BARS=0` + 400; `_11` = b4-scalebars3 + 400. Run-0/native controls pinned to zero bars (leak test).
Excluded: tie 0.3/0.5/0.75, kp60k, loc 0.5, nb 8/12/100, Sequential, rotation priors, GS descent, sa-greedy, OBB trim, b5 gpu-only/wi40 (not promoted; P3 gpu-only+wi40 fastest 1931 s noted for future).

### 3.3 Gaps vs master HEAD
1. `engines/metashape.py`: no profiles/env knobs, no Ultra/DisabledInterp/rematch/sparse cleanup/scale bars, `MAX_KEYFRAMES=400`, `keypoint_limit_per_mpx`/`max_neighbors` unset, Stage-A depth instead of native depth packaging, `TIMEOUT_S=5400` (remove), no `_verify_rendered`, no run.json provenance (issue/variant/settings sha). 
2. Env knob names: issues quote `SCALE_BARS`, `MAX_KEYFRAMES`, `SPELLBOOK_TARGET_RUN0`; given the jobs.py fix they were `SPELLBOOK_`-prefixed -> use `SPELLBOOK_METASHAPE_PROFILE`, `SPELLBOOK_METASHAPE_SCALE_BARS`, `SPELLBOOK_METASHAPE_MAX_KEYFRAMES` **[inf]**.
3. Tests (`test_metashape`, 42 tests; `b5_launcher` 26) gone - regenerate minimal: profile render, run-0 zero-bar pin, env delivery, poison-slot.
4. run.py `SPELLBOOK_TARGET_RUN0=1` purge+remove-under-locks path for run-0 replacement on 9004 (ten-slot scene).

### 3.4 Reproduce
Baseline: `SPELLBOOK_METASHAPE_PROFILE=native-default SPELLBOOK_TARGET_RUN0=1 python spellbook/main.py --engine metashape --scene 9004` (18.096334 cm, ~2740 s; 9009 ~3428 s, 400/400 aligned).
Best: `python spellbook/main.py --engine metashape --scene 9004` (default b5-kf350; 14.627121, ~2743 s, 350/349/346). Noise +/-0.65 cm.

---------------------------------------------------------------------------------------------------
## 4. Common checks for every reproduce run
GPU pool 1-4 only (never GPU 0); serial per engine; scorer `observed_surface_voxel_mean_bidirectional_distance_v1`, 10 mm, CAD sha `76613112...`, visible-voxels sha `9e47eb86...`; 9009 is unscored (coverage only); complete artifacts = `.sens` v4 + `.txt` (17 keys) + clean meshes + segs + `final_poses` + `timing.json` + `run.json` + `geometry_score.yaml` + `cad_comparison.png`.
