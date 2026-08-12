# Scout report: ZED X + ZED SDK official documentation (defaults, mapping quality, SVO offline, coordinate systems)

Local install verified: **ZED SDK 5.4.0** (`/home/rolf/opt/zed-sdk`, `ZED_SDK_Ubuntu24_cuda12.8_v5.4.0.zstd.run`, pyzed 5.4). C++ headers at `/home/rolf/opt/zed-sdk/install/include/sl/Camera.hpp` were cross-checked against the online API reference for enum names/defaults.

---

## 1. ZED X hardware specs (indoor room scanning)

Official product page: https://www.stereolabs.com/zed-x/ (specs also at https://docs.stereolabs.com/docs/products/cameras/zedx.md)

| Spec | ZED X | ZED X Mini | ZED X Nano |
|---|---|---|---|
| Stereo baseline | **120 mm** | 50 mm | 18 mm |
| Sensor | 2x (1920x1200) color **global shutter**, 3 µm pixel | same | same |
| Framerate | 60 fps @1200p; **120 fps @960x600** | 60 fps @1200p | 60 fps @1200p |
| IMU | 16-bit triaxial accel + gyro, vibration-resistant | same | same |
| Interface | GMSL2 (needs Jetson + capture card), hw frame sync <100 µs | same | same |
| Enclosure | IP67 | IP67 | none (indoor) |
| Lens options | **2 mm wide** or **4 mm narrow** (9-element glass) | 2.2 mm / 4 mm | 2.8 mm |
| Depth range (docs) | **Wide: 0.3–20 m (ideal 0.3–12 m); Narrow: 1.0–35 m (ideal 1.0–20 m)** | 0.1–8 m (ideal 0.1–4 m) | 0.03–2.0 m |
| FOV | max 110° H x 80° V x 120° D (2 mm lens) | — | — |

- Source: https://www.stereolabs.com/zed-x/ (baseline/fps/IMU/lenses), https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-settings.md (per-camera depth range table), https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-modes.md (notes "baseline 120 mm" for ZED X GS tests).
- **There is no 20 cm variant.** The only baseline variants are ZED X (120 mm) vs ZED X Mini (50 mm) vs ZED X Nano (18 mm). Lens variants (2 mm vs 4 mm) change range/FOV, not baseline.
- **Min-depth for indoor use (official FAQ)**: "If depth sensing is necessary below 50 cm, you should select the ZED X Mini" (https://www.stereolabs.com/zed-x/ FAQ). ZED X Wide official min depth = **0.3 m** (`depth_minimum_distance` default -1 → hardware default; docs say "cannot be greater than 3 m"). For room scanning (doorways/walls typically ≥0.5 m) ZED X 120 mm baseline is fine and better than ZED 2i (0.3 m min, rolling shutter). ZED X global shutter removes rolling-shutter motion artifacts during handheld traversal — the main hardware advantage over ZED 2i for walking capture.

## 2. Official SDK defaults (exact values, SDK 5.x API reference)

API reference: https://www.stereolabs.com/docs/api/structsl_1_1InitParameters.html, https://www.stereolabs.com/docs/api/structsl_1_1RuntimeParameters.html, https://www.stereolabs.com/docs/api/structsl_1_1SpatialMappingParameters.html, https://www.stereolabs.com/docs/api/structsl_1_1PositionalTrackingParameters.html

### InitParameters (constructor defaults, SDK 5.x)
| Param | Default | Note |
|---|---|---|
| `camera_resolution` | `AUTO` → **HD1200** for ZED X/X Mini, HD720 for others | |
| `camera_fps` | 0 (highest of resolution) | |
| `svo_real_time_mode` | **false** | play SVO as fast as possible (see §6) |
| `depth_mode` | **`NEURAL`** (on Jetson: `NEURAL_LIGHT`) | flags below |
| `coordinate_units` | **`MILLIMETER`** | set METER for ScanNet |
| `coordinate_system` | **`IMAGE`** | flags below (NOT z-up) |
| `depth_minimum_distance` | **-1** (hardware default; ZED X Wide = 0.3 m) | |
| `depth_maximum_distance` | **-1** (hardware default; ZED X = 20 m wide / 35 m narrow) | |
| `depth_stabilization` | **30** (0 = off, range [0–100]) | temporal filter; auto-enables positional tracking in background; too high → ghosting on fast motion |
| `enable_image_enhancement` | true | |
| `open_timeout_sec` | 5.0 | |
| `camera_disable_self_calib` | false | leave on |

### RuntimeParameters (per-grab, constructor defaults)
| Param | Default |
|---|---|
| `enable_depth` | true |
| `enable_fill_mode` | **false** (overrides both confidence thresholds if true) |
| `confidence_threshold` | **95** (range [1,100]; lower = stricter) |
| `texture_confidence_threshold` | **100** (= no filtering; range [1,100]) |
| `measure3D_reference_frame` | `CAMERA` |
| `remove_saturated_areas` | false |

### SpatialMappingParameters (defaults)
| Param | Default |
|---|---|
| `resolution_meter` | **0.05 m** (= `MAPPING_RESOLUTION::MEDIUM`) |
| `range_meter` | **0** (= `MAPPING_RANGE::AUTO`; computed from resolution + camera) |
| `max_memory_usage` | **2048 MB** (CPU memory for meshing) |
| `save_texture` | false |
| `use_chunk_only` | **false** |
| `reverse_vertex_order` | **false** (winding order; keep false unless face culling wrong) |
| `map_type` | `MESH` |
| `stability_counter` | 0 (auto per resolution) |
| `disparity_std` | 0.3 (smaller if depth accurate) |
| `decay` | 1.0 |
| `enable_forget_past` | **false** (true = keep only 1.5x range around camera; discards history!) |

Presets per docs (https://docs.stereolabs.com/docs/development/zed-sdk/modules/spatial-mapping/using-the-api.md): `HIGH`=2 cm, `MEDIUM`=5 cm, `LOW`=8 cm; range `NEAR`=3.5 m, `MEDIUM`=5 m, `FAR`=10 m. Allowed bounds: resolution 1–12 cm, range 2–20 m (overview page).
- **Version flag (important)**: the docs' using-the-api page still names the range presets `NEAR/MEDIUM/FAR`, but **local SDK 5.4 headers define `MAPPING_RANGE { SHORT, MEDIUM, LONG, AUTO }`** (Camera.hpp:11382) and `MAPPING_RESOLUTION { HIGH, MEDIUM, LOW }` (Camera.hpp:11368). Older SDK 3.x docs used 1/2/4 cm resolutions. **Do not hardcode — call `SpatialMappingParameters::get(preset)` at runtime** to read values from the linked SDK.

### PositionalTrackingParameters (constructor defaults)
| Param | Default |
|---|---|
| `mode` | **`GEN_3`** (GEN_1 = dense/depth-based; GEN_2 deprecated; GEN_3 = feature-based, works without depth, has loop closure) |
| `enable_area_memory` | **true** (loop closure + relocalization; "recommend leaving it on") |
| `enable_pose_smoothing` | **false** |
| `set_floor_as_origin` | **false** |
| `enable_imu_fusion` | **true** |
| `set_as_static` | false |
| `set_gravity_as_origin` | **true** (world z-axis = anti-gravity) |
| `depth_min_range` | **-1** (no min; only affects GEN_1) |
| `area_file_path` | "" (empty) |
| `enable_localization_only` | false |
| `enable_2d_ground_mode` | false |
| `initial_world_transform` | identity |

Docs: https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking/settings.md (same values; depth_min_range noted GEN_1-only), modes: https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking/modes.md

### Version-change flags (defaults shifted across SDK generations)
- Default depth mode: `PERFORMANCE` (SDK ≤3.9) → `NEURAL` (SDK 4.0+, default on desktop; NEURAL_LIGHT on Jetson). Old QUALITY/ULTRA/PERFORMANCE are deprecated (still in 5.4 headers).
- `SENSING_MODE` (STANDARD/FILL) removed in 4.x → replaced by `enable_fill_mode` + confidence thresholds.
- 4.0.7: NEURAL confidence distribution re-calibrated ("more accurately estimate low-quality pixels such as flying pixels") — thread by staff Myzhar: https://community.stereolabs.com/t/4-0-7-confidence-ranges-changed/3580/4. Thresholds tuned against 3.x docs may not transfer.
- `depth_stabilization` (default 30) added ~4.2/5.0; doesn't exist in old SDKs.
- Positional tracking `GEN_3` became default in 5.x; GEN_1 was default in 4.x. `GEN_2` deprecated.

## 3. Official spatial-mapping quality guidance

- Overview: https://docs.stereolabs.com/docs/development/zed-sdk/modules/spatial-mapping.md — resolution 1–12 cm, range 2–20 m; "use the lowest density possible for your application"; lower range improves accuracy; mesh filtering presets HIGH/MEDIUM/LOW + texturing.
- Using the API: https://docs.stereolabs.com/docs/development/zed-sdk/modules/spatial-mapping/using-the-api.md
  - **"we recommend using the HD720 video mode at 60fps for optimal results"** (mapping uses tracking poses).
  - Mapping range is **limited to 10 m** internally.
  - Mapping state `FPS_TOO_LOW` / `NOT_ENOUGH_MEMORY` → SDK **stops integrating new data** (map still extractable). Memory limit = `max_memory_usage` (2048 MB default).
  - `enable_forget_past` (default false) exists specifically "to limit memory and drift issues" for very large areas — it discards everything >1.5x range behind the camera. **Do not use for full multi-room capture.**
- Known limitations: no official "max scene size" number for the mesh, but `NOT_ENOUGH_MEMORY` state + `max_memory_usage` cap are the documented ceiling. Fused point cloud takes more memory than mesh (color).
- Mesh export: `.obj` via `map.save()`, texture via `applyTexture()` after `filter()`.

## 4. Drift, loop closure, area memory, large multi-room trajectories

Docs: https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking/vslam-mapping-tutorial.md and https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking/settings.md

- **Yes, loop closure exists**: `enable_area_memory=true` builds an Area map → loop closure corrects accumulated drift; status `LOOP_CLOSED` briefly after each detection.
- **Documented limit for multi-room/long trajectories (key quote)**: *"To build the most accurate map graph possible, single-loop trajectories should not exceed roughly 20 m. As a result, large environments should be divided into several mapping sections."* Each room / aisle = one section; for each section: clockwise loop + anti-clockwise loop + multi-angle exploration; double the loops in featureless/repetitive areas; then move to next section. This is the official mapping procedure and matches multi-room apartment scanning well (per-room closed loops, loop through doorways back into earlier rooms to close loops).
- **Offline area-map creation is officially supported**: record SVO → `ZED_Positional_Tracking --svo recording.svo2 --map -o map.area`. Lifelong mapping (load `.area`, extend, re-save) and relocalization (`-i map.area`) are documented workflows. `.area` is tied to the depth mode used at recording (GEN_1) — recommendation: build area maps with GEN_3.
- **GNSS**: only for outdoor/global localization (VIO+GNSS fusion, https://docs.stereolabs.com/docs/development/zed-sdk/modules/global-localization.md) — not applicable for indoor; indoor you rely on area memory loop closure.
- **Fusion module** (https://docs.stereolabs.com/docs/development/zed-sdk/modules/fusion.md): multi-camera publish/subscribe fusion (positional tracking, spatial mapping, OD/BT, GNSS). Not needed for single-camera SVO offline reconstruction; it's for multi-cam arrays.
- GEN_3 accuracy numbers (official, ZED X, ~400 m sequences): indoor warehouse w/ reflective lights mean APE 0.56 m / max 1.2 m (VIO). GEN_1: 0.8/1.8 m. Source: modes.md. So expect ≥10 cm-scale trajectory error over multi-room loops unless loop closures bind the trajectory — relevant for ScanNet-quality (loop-closing over reused views is your friend).

## 5. Glass, reflections, low-texture walls — official guidance

- Official statement (https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing.md): depth accuracy reduced "by outlier measurements on homogeneous or textureless surfaces, such as white walls, green screens, or reflective areas... leading to temporal instability and less reliable depth measurements."
- Official filtering knobs (https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-settings.md): `confidence_threshold` (edges/noisy pixels), `texture_confidence_threshold` (uniform regions — exactly the white-wall case), `depth_stabilization` (temporal smoothing vs ghosting), `depth_maximum_distance` clamp to cut far jitter, `remove_saturated_areas` for blown highlights (glare on windows).
- Depth modes comparison (https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-modes.md): **NEURAL_PLUS "most robust to environmental changes and reflections"**, highest ideal range 0.3–12 m, <1% error up to 9 m, <2% up to 12 m (ZED X). NEURAL ideal 0.3–9 m. NEURAL_LIGHT worst (up to 8% error 5–12 m).
- Forum threads (Stereolabs staff involved / relevant): reflective depth issues https://community.stereolabs.com/t/depth-issues-on-reflective-parts-for-zed2i/1957; specular lighting bias/drift + mitigations (reduce exposure, depth mode change, polarization filter, confidence masking) https://community.stereolabs.com/t/specular-lighting-causing-systematic-stereo-depth-bias-drift/10371; confidence semantics https://community.stereolabs.com/t/interpreting-depth-confidence/6996.
- Hardware note: ZED X ships with a **built-in polarizing filter** on the lens (per official page + Mouser) — officially marketed to mitigate specular/reflection issues vs other ZEDs.
- Practical take: for glass-heavy scenes use NEURAL_PLUS + `texture_confidence_threshold` (e.g. ~50–70) + moderate `confidence_threshold`, and mask glass holes in meshing rather than trusting depth on glass panes.

## 6. SVO2 offline re-processing at higher quality

- **Yes, officially supported.** SVO playback behaves like a live camera; every module (depth, tracking, spatial mapping) works from SVO. Setting: `init_parameters.svo_real_time_mode = false` (default) → `grab()` reads frames as fast as possible, ignoring capture timestamps; set true to replay in real time. (https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording.md, API ref InitParameters::svo_real_time_mode)
- No official "batch/headless re-render at NEURAL_PLUS" tool, but since depth is computed at `grab()`-time, simply open the SVO with `depth_mode = NEURAL_PLUS`, `svo_real_time_mode = false` and re-run tracking+mapping — this is the officially sanctioned offline path (the Area-memory tutorial itself documents computing maps offline from SVO: `--svo recording.svo2 --map`).
- SVO2 format: default since SDK 4.1; records sensors at native frequency (needed for GEN_2/3 high-freq IMU fusion) and supports custom data (e.g. GNSS). Compression: LOSSLESS ~42% of RAW, H.264/H.265 (NVENC) ~1%. **For offline quality reconstruction use LOSSLESS** (PNG/ZSTD) so neural depth runs on untouched pixels.
- **Official export sample: `recording/export/svo`** ("ZED_SVO_Export"): AVI or PNG sequences of LEFT+RIGHT, LEFT+DEPTH_VIEW, LEFT+DEPTH_16Bit. https://github.com/stereolabs/zed-sdk/tree/master/recording/export/svo (+ sensors export sample, encrypted samples). No pose/PLY export in that sample — a custom exporter (retrieveMeasure XYZRGBA in WORLD frame + getPosition per frame) is the standard way to get posed RGB-D frames; the "spatial mapping" sample exports mesh OBJ/FPC PLY.

## 7. Dataset formats + coordinate systems (ScanNet z-up mapping)

- **No officially documented export to ScanNet/ARkit/standard SLAM dataset formats.** Official exports are: SVO→images/depth (PNG/AVI), mesh→OBJ/PLY, fused point cloud→PLY, sensors→file. A "posed RGB-D" dump is only possible via your own code (SDK gives per-frame `getPosition()` + `retrieveMeasure(XYZRGBA)`), or via ROS 2 wrapper (rosbag: rgb/depth/camera_info/pose topics) — https://docs.stereolabs.com/docs/integrations/ros-2/record-and-replay-data.md.
- **`COORDINATE_SYSTEM` enum** (6 values, API ref group Core: https://www.stereolabs.com/docs/api/group__Core__group.html; same in local 5.4 headers Camera.hpp:517):
  - `IMAGE` — camera frame, Z forward (default!)
  - `LEFT_HANDED_Y_UP` — Unity/DX
  - `RIGHT_HANDED_Y_UP` — OpenGL
  - `RIGHT_HANDED_Z_UP` — **Z up, Y forward** (3DSMax)
  - `LEFT_HANDED_Z_UP` — Unreal
  - `RIGHT_HANDED_Z_UP_X_FWD` — Z up, X forward (ROS REP-103)
- **ScanNet mapping**: ScanNet meshes are "Binary PLY format mesh with +Z axis in upright orientation" (this repo's README, /home/rolf/GIT/ScanNet/README.md:46) — i.e. **right-handed, Z up**. The two direct z-up matches are `RIGHT_HANDED_Z_UP` (Y forward) and `RIGHT_HANDED_Z_UP_X_FWD` (X forward); they differ by a 90° rotation about Z. Either maps onto ScanNet's z-up with a fixed in-plane axis permutation (or set `set_gravity_as_origin=true` + `set_floor_as_origin=true` for gravity/floor-aligned z-up world, then rotate about Z to ScanNet's x/y orientation). **Default `IMAGE` is NOT z-up — you must set the coordinate system explicitly.**
- `coordinate_units` default is MILLIMETER — set `UNIT::METER` for ScanNet-scale meshes (ScanNet meshes are in meters).

---

## Bottom line for our pipeline
1. Record: ZED X (120 mm), HD1200@30/60 fps, global shutter, LOSSLESS SVO2.
2. Reconstruct offline from SVO2 with `svo_real_time_mode=false`, `DEPTH_MODE::NEURAL_PLUS`, `UNIT::METER`, `COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP`, defaults for confidence (95/100) then tighten texture_confidence for white walls; mapping resolution ≤2 cm (HIGH) with auto range or ~5–8 m for rooms.
3. Tracking: GEN_3 (default), `enable_area_memory=true`, `set_floor_as_origin=true` (floor-plane z=0); walk per-room clockwise+counter-clockwise loops ≤20 m, double loops in featureless areas, close loops through doorways; optionally pre-build a `.area` map offline from the SVO and replay tracking against it (relocalization) to kill residual drift.
4. Export posed RGB-D + trajectory with a custom exporter in WORLD frame (Z-up) or the SDK mesh path (OBJ, filter+texture); convert with the ScanNet utils (SensReader) conventions.

## Open question / suggestion
- Verify actual `MAPPING_RESOLUTION::HIGH` value at runtime (`get()` returns meters) since docs (2 cm) and old SDK docs (1 cm) conflict — cheap to log once. Suggestion: keep mapping at SDK defaults (MEDIUM 5 cm / AUTO range) for a first pass to stay within the 2048 MB memory cap, then a second pass at 1–2 cm for the rooms that matter, since 3000 verts/m² needs ~1.8 cm voxels.
