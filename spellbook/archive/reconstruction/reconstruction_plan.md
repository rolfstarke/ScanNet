# Action Plan: SVO2 -> ScanNet-native reconstructions (6 engines)

Target audience: build agent. Follow literally. Do not improvise parameter values.
Every number below was measured or read from a named source file. If a value is missing,
STOP and report -- do not guess.

## Goal

Turn ZED X `.svo2` recordings into scans that are byte-format-identical to real ScanNet v2
scans, at ScanNet's measured geometric quality, so ScanNet's annotation tooling can be used
on them. Six reconstruction engines, one scan each. No tuning pass -- correct official
implementations only.

## Hard constraints

- Do NOT modify `/data/scannet/scans/scene0*` or any ScanNet repo file outside `spellbook/`.
- All new code goes in `spellbook/reconstruct/`.
- No caching. Every invocation recomputes from scratch. No `--force` flag exists.
- No `paths.yaml` or equivalent central config. SVO discovery is by filename convention.
- Use official/documented code. Priority: (1) ScanNet's own scripts/binaries/configs,
  (2) the engine's own official docs/launch files, (3) write glue only where nothing exists.
  Do NOT copy `/home/rolf/GIT/ov3dis-comparison` logic -- it is known-faulty (bug list below).
- Camera is a **ZED X** (1920x1200 global shutter, 120mm baseline). Anywhere a camera model
  is named, it is `zedx`. `configs/settings.yaml:48`'s `zed2i` was wrong.

## Naming and layout

```
/data/scannet/custom/raw/scene9004.svo2          symlink to the real .svo2
/data/scannet/scans/scene9004_00/
    scene9004_00.sens
    scene9004_00.txt
    scene9004_00_vh_clean.ply
    scene9004_00_vh_clean_2.ply
    scene9004_00_vh_clean_2.0.010000.segs.json
    recon/                                        non-ScanNet extras live ONLY here
        qc.yaml
        cmdline.txt
        engine_native.ply                         engine's raw output, provenance
        frames/  color/ depth/ pose/              intermediates
```

Scene number = 9000 + test-environment index (04_BTU_LG2C_R312_chaos -> 9004,
09_BTU_Lab -> 9009). Scan index = engine:

| index | engine |
|---|---|
| `_00` | zed |
| `_01` | metashape |
| `_02` | rtabmap |
| `_03` | isaac |
| `_04` | open3d |
| `_05` | bundlefusion |

`scene9004_00` is a valid `sceneNNNN_NN` id, so every ScanNet tool parses it. The 9000 range
is unused by ScanNet v2 (0000-0806), so the numbering alone marks a custom scan. No prefix.

CLI: `python -m spellbook.reconstruct.run --scene 9004 --engine zed`
Reads `/data/scannet/custom/raw/scene9004.svo2`. Optional `--svo <path>` override.

SVO source files (create the symlinks in `/data/scannet/custom/raw/` first):
`/home/rolf/GIT/ov3dis-comparison/configs/paths.yaml` lists the current `.svo2` paths for
scenes 4-9. Read it ONCE to create the symlinks, then never depend on it again.

## Code layout

```
spellbook/reconstruct/
  __init__.py
  run.py          CLI
  extract.py      Stage A: svo2 -> recon/frames/ + <id>.sens + <id>.txt
  scannet.py      Stage C: align -> _vh_clean.ply -> _vh_clean_2.ply -> .segs.json
  qc.py           Stage D: acceptance metrics -> recon/qc.yaml
  engines/
    __init__.py
    zed.py  metashape.py  rtabmap.py  isaac.py  open3d.py  bundlefusion.py
```

Each engine module exposes exactly one function:
`reconstruct(frames_dir, sens_path, out_mesh_path, work_dir) -> trajectory`
It must produce a triangle mesh at `out_mesh_path` and return per-frame 4x4 camera-to-world
poses in the engine's own frame, plus a string naming that frame convention.

---

# BUILD ORDER

Phase 0 must be complete and verified before Phase 1. Do not reorder.

| phase | content | dev scene |
|---|---|---|
| 0 | qc.py + extract.py + scannet.py, proven end-to-end | 9009 |
| 1 | engines/zed.py | 9009 |
| 2 | engines/open3d.py | 9009 |
| 3 | engines/bundlefusion.py | 9009 |
| 4 | engines/metashape.py | 9009 |
| 5 | engines/rtabmap.py | 9009 |
| 6 | engines/isaac.py | 9009 |

After each phase passes on 9009, run it on 9008 (smoke, 67s) then 9005, 9006, 9007. Skip 9004
for loop-closure-dependent engines (rtabmap, isaac, bundlefusion) -- it has only 39.7% revisit.

Measured scene properties (source: trajectory audit of existing pose files):

| scene | frames | dur s | traj m | motion duty % | revisit % | extent m |
|---|---|---|---|---|---|---|
| 9004 | 3613 | 120 | 43.0 | 92.4 | 39.7 | 9.64 |
| 9005 | 10969 | 366 | 109.1 | 85.6 | 57.0 | 9.23 |
| 9006 | 9731 | 324 | 40.2 | 78.3 | 93.6 | 9.57 |
| 9007 | 10969 | 366 | 146.1 | 88.3 | 60.0 | 10.13 |
| 9008 | 2000 | 67 | 24.7 | 100.0 | 53.5 | 3.09 |
| 9009 | 2357 | 79 | 67.8 | 98.7 | 93.8 | 8.28 |

---

# PHASE 0a -- qc.py (BUILD THIS FIRST)

~120 lines. Writes `recon/qc.yaml`. Every engine calls it at the end.

Metrics to compute:

| key | how |
|---|---|
| `frames_svo` | `zed.get_svo_number_of_frames()` |
| `frames_exported` | count of `recon/frames/color/*.jpg` |
| `pose_valid_pct` | fraction of frames with `POSITIONAL_TRACKING_STATE == OK` |
| `pose_identity_count` | count of poses equal to identity within 1e-6 (MUST be 0) |
| `traj_length_m` | sum of consecutive camera-center distances |
| `revisit_pct` | fraction of poses within 0.5 m of another pose >300 frames away |
| `vh_clean_verts`, `vh_clean_faces` | from the ply header |
| `vh_clean_2_verts`, `vh_clean_2_faces` | from the ply header |
| `floor_area_m2` | 2D convex hull area of vertices projected to xy |
| `verts_per_m2_floor` | `vh_clean_2_verts / floor_area_m2` |
| `surface_area_m2` | sum of triangle areas of `_vh_clean_2` |
| `floor_pct1_z` | 1st percentile of vertex z |
| `floor_normal_dot_z` | RANSAC plane on vertices with z < floor_pct1+0.2, `abs(n[2])` |
| `wall_azimuth_dev_deg` | after applying axisAlignment: weighted histogram (2 deg bins) of horizontal wall normals; max distance of the dominant mode to the nearest x/y axis |
| `axis_alignment_det` | `det(axisAlignment)` |
| `backproj_residual_mm` | median nearest-neighbour distance from backprojected depth points to mesh vertices, over 5 evenly spaced frames |

Acceptance bar (measured from 6 real ScanNet scenes: 0568_00, 0019_00, 0426_00, 0217_00,
0575_02, 0304_00). Write PASS/FAIL per metric into qc.yaml; do NOT abort on FAIL, report it.

| metric | bar | ScanNet measured |
|---|---|---|
| `pose_valid_pct` | >= 95 | n/a |
| `pose_identity_count` | == 0 | 0 |
| `vh_clean_2_verts` | 120000-250000 | 126385-232453 |
| `vh_clean_2_faces` | 230000-460000 | 239493-444515 |
| `verts_per_m2_floor` | >= 4600 | 4652-9719 |
| `surface_area_m2` | 30-80 | ~55 |
| `floor_pct1_z` | abs <= 0.10 | 0.014-0.083 (0568_00 is an outlier at 0.733) |
| `floor_normal_dot_z` | >= 0.999 | 0.9992-1.0000 |
| `wall_azimuth_dev_deg` | <= 2.0 | 1.0 after axisAlignment |
| `axis_alignment_det` | 1.0 +- 1e-5 | 1.000000 |
| `backproj_residual_mm` | <= 20 | 9-15 |
| `frames_exported` | == frames_svo - 1 | trailing frame is always invalid |

Also: on first run, compute the same metrics for the 6 real ScanNet scenes above and write
`spellbook/reconstruct/scannet_reference.yaml`. The bar then comes from measurement, and any
future disagreement about "ScanNet quality" is a diff, not an argument.

# PHASE 0b -- extract.py

`svo2 -> recon/frames/{color,depth,pose}/ + intrinsic_color.txt + extrinsic_color.txt +
intrinsic_depth.txt + extrinsic_depth.txt + <id>.sens + <id>.txt`

## InitParameters

```python
init.set_from_svo_file(svo)
init.svo_real_time_mode = False
init.coordinate_units = sl.UNIT.METER
init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP   # ZED native; converted in Stage C
init.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
init.depth_minimum_distance = 0.1     # ScanNet s_sensorDepthMin, zParametersScanNet.txt:34
init.depth_maximum_distance = 6.0     # ScanNet s_sensorDepthMax, zParametersScanNet.txt:35
```
Runtime params: leave at SDK defaults (`confidence_threshold=95`,
`texture_confidence_threshold=100`, `fill_mode=False`).
Positional tracking: `enable_area_memory=True`, `gravity_as_origin=True` (default).

## Grab loop -- MANDATORY corrections

These are confirmed bugs in `ov3dis-comparison/utils/zed_reconstruct.py`. Do not reproduce them.

1. **Termination.** `zed_reconstruct.py:111,164` uses `while zed.grab(rt) == SUCCESS`, which
   exits on the FIRST non-SUCCESS code and silently truncates on any transient error. Correct:
   ```python
   while True:
       err = zed.grab(runtime)
       if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
           break
       if err != sl.ERROR_CODE.SUCCESS:
           raise RuntimeError(f"grab failed at frame {i}: {err}")
       ...
   ```
2. **Tracking state.** `zed_reconstruct.py:114` discards `get_position`'s return code, so poses
   get written for SEARCHING/OFF frames. Capture the state per frame, write it to
   `recon/frames/pose_state.txt` (one token per line), and count it in qc.
3. **Invalid depth is 0, never 100.** `bundlefusion_reconstruct.py:80-83` does
   `nan_to_num(...)*1000` then `clip(100, 15000)`, turning every invalid pixel into 100 mm.
   Verified damage: the shipped `09_BTU_Lab/.../bundlefusion/ref` depth PNGs contain zero
   0-pixels and ~24k px/frame in the 100-200mm bin -- a phantom shell fused into the TSDF.
   Correct: NaN/Inf/<=0 -> 0. Clip only the top at 65535.
4. **No cross-aspect intrinsic rescaling.** `zed-metashape/zed_metashape/svo_extract.py:54-60`
   scales fx by `width/src_w` and fy by `height/src_h` independently; on ZED X those differ
   (16:10 -> 16:9) and manufacture a 10% anisotropy on a camera with genuinely square pixels
   (native fx=fy=735.308, verified by reading the SVO). Retrieve at NATIVE resolution
   (1920x1200) and do not rescale at all.
5. **Skip the trailing frame.** Every released ScanNet `.sens` has a NaN pose / zero valid
   depth in its last frame. Drop the last SVO frame.

## Output files

- `color/<i>.jpg` BGR, `depth/<i>.png` uint16 millimetres, `pose/<i>.txt` 4x4 camera-to-world
  `%.6f`, contiguous 0-based indices.
- `intrinsic_color.txt` = `intrinsic_depth.txt` = 4x4 ScanNet form
  (`fx 0 cx 0 / 0 fy cy 0 / 0 0 1 0 / 0 0 0 1`). `extrinsic_color.txt` =
  `extrinsic_depth.txt` = identity (ZED depth is registered to left color; verified all 6
  release scenes have both extrinsics identity).

## .sens writer

Mirror `/home/rolf/GIT/ScanNet/SensReader/c++/src/sensorData.h:1057-1109`. Verified against
`SensReader/python/SensorData.py:52-74` and 6 real release files. Little-endian throughout.

```
uint32   version = 4
uint64   len(sensor_name); bytes sensor_name = b"ZED X"
float32[16] intrinsic_color   (row-major 4x4)
float32[16] extrinsic_color   (identity)
float32[16] intrinsic_depth
float32[16] extrinsic_depth   (identity)
int32    color_compression = 2   # jpeg
int32    depth_compression = 1   # zlib_ushort
uint32   color_width  = 1920
uint32   color_height = 1200
uint32   depth_width  = 1920
uint32   depth_height = 1200
float32  depth_shift = 1000.0
uint64   num_frames
per frame:
    float32[16] camera_to_world
    uint64   timestamp_color
    uint64   timestamp_depth
    uint64   color_size_bytes ; bytes (jpeg-encoded)
    uint64   depth_size_bytes ; bytes (zlib.compress of raw uint16 buffer)
uint64   num_IMU = 0
```
Note: `SensorData.py` never reads the trailing IMU block, but the C++ reader does -- always
write the `num_IMU` field. Write it as 0 unless IMU export is added later.

Verify by round-tripping through `SensReader/python/SensorData.py`:
`export_depth_images` / `export_color_images` / `export_poses` / `export_intrinsics` must all
succeed and the exported frame count must match.

## `<id>.txt`

Exactly these keys, this order (verified across 20 real scenes; only
`colorToDepthExtrinsics` varies in presence -- omit it, our extrinsics are identity):

```
axisAlignment = <16 floats, row-major, space-separated>
colorHeight = 1200
colorWidth = 1920
depthHeight = 1200
depthWidth = 1920
fx_color = <float>
fx_depth = <float>
fy_color = <float>
fy_depth = <float>
mx_color = <cx>
mx_depth = <cx>
my_color = <cy>
my_depth = <cy>
numColorFrames = <int>
numDepthFrames = <int>
numIMUmeasurements = 0
sceneType = <string>
```
`axisAlignment` is filled in by Stage C, not here -- write the file after Stage C, or write a
placeholder and rewrite. Note: for calibrated ScanNet scenes the txt intrinsics disagree with
the `.sens` header; keep ours consistent (write the same values to both).

# PHASE 0c -- scannet.py

## C1. Frame conversion to the ScanNet output frame

The output frame is fixed and identical for all engines. Per-engine input frames differ; each
engine module declares its own convention string and `scannet.py` converts once, here.
Nothing downstream may guess a frame. Delete any use of
`ov3dis-comparison/utils/_common.py:52-91 resolve_pointcloud_frame` -- trying both conventions
and keeping the smaller error is a symptom, not a solution.

Known engine-native frames:

| engine | native frame | conversion |
|---|---|---|
| zed | RIGHT_HANDED_Y_UP (cloud AND mesh) | Y-up -> Z-up |
| metashape | Y-up (from ZED priors), poses+cloud share one frame | Y-up -> Z-up |
| rtabmap | ROS map frame, Z-up already | verify, likely identity |
| isaac | cuVSLAM output frame -- MEASURE IT, do not assume | see below |
| open3d | camera frame of the input trajectory (= our pose frame) | same as pose source |
| bundlefusion | first camera pose = identity | gravity-align from ZED gravity |

For isaac: the old code had a definite bug -- `zed-isaac/scripts/fuse_open3d.py:88-92` flips
the mesh once and `:110-114` flips the pcd twice, so mesh and pointcloud end up in DIFFERENT
frames. The shipped scene9 cloud is mirrored against its own poses (backprojection residual
0.621 m raw vs 0.085 m flipped). Measure the residual both ways, pick the correct one, and
assert it is < 20 mm. Never ship a guess.

## C2. Gravity z-up alignment (ScanNet's `alignment.h:154-308`)

Reimplement ScanNet's own algorithm; the original is Windows+CGAL and cannot be built here.

1. Up vector: use ZED's gravity (available from `sl.SensorsData` / `gravity_as_origin`).
   ScanNet's fallback is the averaged camera up vector from poses
   (`Alignment/src/alignment.h:82-107`) -- implement that as a fallback and log which was used.
2. Rotate so up -> +z. Apply to mesh AND all poses (they must stay in one frame).
3. Floor plane: `PlaneExtract` equivalent (`Alignment/src/planeExtract.h:73-143`) --
   normal threshold 0.90, distance threshold 0.05, min cluster 500 points, bounding cluster
   10 cm / 100 points. Take the first cluster with `normal . (0,0,1) > 0.8`.
4. Translate so floor min-z = 0, and center x/y.
5. Do NOT do the CGAL OBB rotation. Verified: released ScanNet meshes have walls at arbitrary
   azimuth (up to 39 deg off-axis), so that step did not survive into the release.

## C3. axisAlignment (computed, NOT applied)

This is the single most-misunderstood part. Verified by measuring 6 released scenes:

- Released `_vh_clean_2.ply` is the RAW scan frame: z-up (floor normal z 0.9992-1.0000,
  residual tilt <= 2.3 deg), floor at ARBITRARY z (0.014-0.083 m for 5/6 scenes, 0.733 for
  scene0568_00), walls at ARBITRARY azimuth (up to 39 deg off-axis).
- `.sens` poses are in that SAME raw frame: backprojection residual 9-15 mm. Against the
  axis-aligned frame it is 0.1-4.8 m, i.e. gross mismatch.
- `axisAlignment` is a PURE z-rotation plus translation: `R[2][2] == 1` exactly, all z
  cross-terms exactly 0, `det == 1.000000`, never identity, angles -50.5 to -178.5 deg.
  Applying it puts walls within 1 deg of x/y and floor at -0.075..-0.027.

So: compute it, write it to `<id>.txt`, and DO NOT transform the mesh or poses with it.

Computation: build a weighted histogram (2 deg bins) of horizontal wall-normal azimuths
(vertices whose normal has `abs(n_z) < 0.2`, weighted by triangle area). Take the dominant
mode, and choose the z-rotation that maps it to the nearest x/y axis. Translation: after that
rotation, put floor min-z at exactly 0 and shift x/y so the scan sits in the positive quadrant
(matches the observed released translations, e.g. scene0568_00 t = [2.822, 3.740, -0.807],
whose z component exactly cancels its 0.733 m floor).

Assert `det == 1` and `R[2][2] == 1` before writing.

## C4. Meshes

pymeshlab (`pip install pymeshlab`, no sudo needed; PLY vertex colors are preserved by
default, no `-m vc` equivalent required).

Step 1 -- `_vh_clean.ply`: run ScanNet's `Server/tools/meshclean/clean.mlx` UNMODIFIED via
`ms.load_filter_script(...)` + `ms.apply_filter_script()`. That script is:
Merge Close Vertices threshold `0.0010689` -> Remove Duplicate Faces -> Remove Isolated
pieces MinComponentSize `7500` -> Remove Unreferenced Vertex.
Save WITH vertex normals (real ScanNet `_vh_clean.ply` has normals, comment "MLIB generated").

Step 2 -- `_vh_clean_2.ply`: apply `meshing_decimation_quadric_edge_collapse` with ScanNet's
exact quality parameters from `simplify.mlx`, changing ONLY the target:
```
targetfacenum   = 300000      # ScanNet uses targetfacenum=0 / targetperc=0.2
targetperc      = 0
qualitythr      = 0.3
preserveboundary= False
boundaryweight  = 1
preservenormal  = False
preservetopology= False
optimalplacement= True
planarquadric   = False
qualityweight   = False
autoclean       = True
selected        = False
```
then the same 4-step cleaning block as `simplify.mlx` (Merge Close Vertices 0.0010689,
Remove Duplicate Faces, Remove Isolated pieces MinComponentSize 1000, Remove Unreferenced).
Save WITHOUT normals (real `_vh_clean_2.ply` has none, comment "VCGLIB generated").

WHY the one deviation: ScanNet's `simplify.mlx` is a RELATIVE 20% reduction applied twice
(= 4%). Its output density is therefore a function of its input density. ScanNet's input
(`_vh.ply`, from the unreleased VoxelHashing "improve" stage) is 3.0-5.7M verts at ~3.1 mm
spacing, so 4% lands at 126-232k verts / 239-444k faces. Our engines have different input
densities, so a fixed 4% would land anywhere. Using the same filter's absolute
`targetfacenum` form pins the OUTPUT to ScanNet's measured band (mean 317k faces) for every
engine uniformly, including ZED, whose SDK caps mapping resolution at 1 cm. Record this
deviation in `recon/qc.yaml` under `deviations:`.

`_vh_clean_1.ply` is NOT produced (ScanNet never releases it).

PLY format assertions before writing: `binary_little_endian`, float x/y/z, uchar r/g/b/a.

## C5. Segments

Build ScanNet's own Segmentator once: `make` in `/home/rolf/GIT/ScanNet/Segmentator`
(g++ and make are present, no external deps). Run with DEFAULTS:
```
./segmentator <id>_vh_clean_2.ply
```
Defaults are kThresh=0.01, segMinVerts=20 (`Segmentator/segmentator.cpp:274-275`); the output
filename is `basename + "." + to_string(kThresh) + ".segs.json"`
(`segmentator.cpp:285`) -> `<id>_vh_clean_2.0.010000.segs.json`. Correct by construction.

Verify: `len(segIndices) == vertex count of _vh_clean_2.ply` (holds for all 6 real scenes).

Known ScanNet inconsistency, do not try to reproduce it: the released JSON's `params` block
says `kThresh: "0.0001"` while its filename says `0.010000`, and it contains extra keys
(`minPoints`, `maxPoints`, `thinThresh`, `flatThresh`, `minLength`, `maxLength`) that
`segmentator.cpp:257` never writes. The release files were post-edited by tooling not in the
repo. Our output follows the binary.

---

# PHASE 1 -- engines/zed.py

Official ZED SDK spatial mapping. Docs: SpatialMappingParameters in
`/home/rolf/opt/zed-sdk/install/include/sl/Camera.hpp:11332-11500`.

```python
mp = sl.SpatialMappingParameters(map_type=sl.SPATIAL_MAP_TYPE.MESH)
mp.resolution_meter = 0.01     # SDK minimum: allowed_resolution.first
mp.range_meter = 6.0           # ScanNet s_sensorDepthMax
mp.max_memory_usage = 8192     # default 2048 is too small; see below
mp.save_texture = False
mp.use_chunk_only = False
```

Mandatory corrections:

1. **One pass, not two.** The old code runs `_extract_frames` and `reconstruct` as two
   SEPARATE SVO passes. `get_position` returns ONLINE poses; spatial mapping fuses using
   loop-closure-corrected ones. So poses and mesh diverge exactly where loop closure fires,
   and `trim_scene` then trims the cloud using divergent poses. Do the mapping and the frame
   export in ONE pass, or re-read the corrected trajectory after mapping.
2. **Check `MAPPING_STATE` every frame.** `get_spatial_mapping_state()` returning
   `FPS_TOO_LOW` or `NOT_ENOUGH_MEMORY` means the SDK has SILENTLY stopped integrating.
   `max_memory_usage` defaults to 2048 MB, which is the most likely cause of the fragmented
   `ref` variant in the old runs. Log every state transition; fail loudly if integration stops.
3. **Two-pass `.area`.** The documented offline workflow: first pass saves an `.area` file,
   second pass loads it for relocalization. The old code never used it. Use it.
4. `range_meter` note: the SDK caps mapping range at 10 m internally, so the old
   `range_meter: 15.0` was recorded in `settings_used.yaml` but never applied. 6.0 is within
   range and is ScanNet's value.
5. Do NOT apply statistical outlier removal. That was tuning, and it was documented as
   ineffective against the real artifact anyway.
6. Declare frame convention `"zed_y_up"` to scannet.py.

Expected: `_vh_clean` ~550k verts (1 cm spacing). `_vh_clean_2` at targetfacenum 300000 will
require only a modest reduction. This is ZED's ceiling; qc.yaml will show it.

# PHASE 2 -- engines/open3d.py

Open3D tensor reconstruction, the ONLY documented external-pose integration path.

- Needs Open3D built with `-DBUILD_CUDA_MODULE=ON` (official PyPI wheels are CPU-only).
  Build it into the `3disspellbook` env, or use CPU and accept the runtime.
- Entry point: the official `t_reconstruction_system/integrate.py` with `--path_trajectory`
  (accepts TUM-style `.log` or a posegraph `.json`). Convert our `recon/frames/pose/*.txt`
  to that format.
- Parameters: `voxel_size = 0.004`, `trunc_voxel_multiplier` such that truncation = 0.06 m
  (ScanNet `s_SDFTruncation`; with 4 mm voxels that is multiplier 15), `depth_min = 0.1`,
  `depth_max = 6.0`, `depth_scale = 1000`, `block_count` raised (default 40000 blocks ~
  3.3 GB; a 4 mm room needs more -- measure and raise until it stops evicting).
- ScanNet's `s_SDFTruncationScale` (0.02 per metre of depth) has NO Open3D equivalent.
  Record that under `deviations:` in qc.yaml.
- Frame convention: same as the input pose frame.
- Layout note: the legacy pose-graph system expects `image/` + `depth/` +
  `camera_intrinsic.json` with keys `width`/`height`/`intrinsic_matrix` (9 floats row-major).
  The tensor path is the one we want -- it takes the trajectory.

# PHASE 3 -- engines/bundlefusion.py

This is the ScanNet reference engine. Fidelity matters most here.

Source: `/data/zed-bundlefusion` (a checkout of FangGet/BundleFusion_Ubuntu_Pangolin, NOT
niessner/BundleFusion, and NOT ScanNet's own build). Docker images
`bundlefusion:latest` / `bundlefusion:latest-hostdriver` (9.15 GB each) already exist and
already reach the GPU.

## Reset the configs to ScanNet's, byte-for-byte

Use `/home/rolf/GIT/ScanNet/Server/tools/recons/zParametersScanNet.txt` and
`zParametersBundlingScanNet.txt` as the authoritative files. The fork deviates on these
(fork value -> ScanNet value):

| param | fork | ScanNet |
|---|---|---|
| `s_sensorIdx` | 1 (PrimeSense dir emulation) | **8 (SensorDataReader / .sens)** |
| `s_sensorDepthMax` / `s_renderDepthMax` | 15.0 | **6.0** |
| `s_overwriteOrigSensTrajectory` | absent (-> 0) | **true** |
| `s_minKeyScale` | 2.0 | **3.0** |
| `s_siftMatchThresh` | 0.5 | **0.7** |
| `s_maxNumKeysPerImage` | 2048 | **512** |
| `s_maxNumCorrPerImage` | 256 | **2000** |
| `s_maxNumResidualsRemoved` | 3 | **10** |
| `s_numLocalNonLin/LinIterations` | 3/150 | **2/100** |
| `s_numGlobalNonLin/LinIterations` | 5/200 | **3/150** |
| `s_verifySiftErrThresh` | 0.2 | **0.12** |
| `s_timingsDetailledEnabled`, `s_verbose` | false | true |

Change from ScanNet only: `s_SDFVoxelSize = 0.004`. Reason: ScanNet's 1 cm value is for the
TRACKING mesh (`<id>.ply`, never released); the released mesh comes from the separate
VoxelHashing `improve` stage at ~4 mm. Since we produce one mesh per engine, 4 mm is the value
that matches ScanNet's released density. Record under `deviations:`.

`s_maxNumCorrPerImage` is DEAD config in this fork -- its only consumers are in `SBA_CPU.cpp`,
which is gated by `USE_CPU_SOLVE`, which is commented out (`SBA_param.h:5`); the built object
file contains only static-init code. Set it to ScanNet's 2000 anyway for fidelity, and note it
has no runtime effect.

## Feed it a real `.sens`

`s_sensorIdx = 8` routes through `SensorDataReader`, which reads intrinsics from the `.sens`
header and divides depth by `m_depthShift`. This makes three of the fork's patches unnecessary
and removes the 640x480 buffer-overrun path entirely, so native 1920x1200 needs no resize
patch and no per-scene intrinsics regex injection. Delete
`bundlefusion_reconstruct.py`'s `_prepare_bf_input` intrinsics patching from the new code.

## Keep these fork patches (genuine fixes any headless build needs)

- CUDA 11+ `__shfl_down_sync` fixes in `cudaUtil.h`, `ProgramCU.cu`, `SolverBundlingUtil.h`;
  `FDIV` -> `/` in `ProgramCU.cu:50`; `sm_86` arch in `CMakeLists.txt:14` (matches the A4000s).
- `RGBDSensor.h:87-88` start/stop toggling `m_bIsReceivingFrames`.
- Bundling-thread exit check (`BundleFusion.cpp:519-529`) and `deinitBundleFusion` cleanup
  (`:443-460`).
- **Save poses BEFORE the reintegration drain** (`BundleFusion.cpp:889-895`). The drain
  overwrites `integratedTransform` with optimized ones, which are `-inf` for invalidated
  frames (`TrajectoryManager.cpp:58,144`). This ordering is a real upstream bug fix.
- The Docker headless build with baked host driver libs (snap Docker cannot use the nvidia
  container toolkit) and `--privileged` launch.

## NEVER backfill identity poses

`BundleFusion.cpp:944` skips frames with no finite transform, and
`bundlefusion_reconstruct.py:360-365` then writes identity for every missing `pose/{i}.txt`.
Verified damage: 345 of 2357 shipped poses (14.6%) in
`09_BTU_Lab/.../bundlefusion/ref/frames/pose` are exact identity matrices. An identity pose is
indistinguishable from a real one downstream, so those frames unproject into a wall at the
origin. Correct: DROP unposed frames and report the count. `qc.pose_identity_count` must be 0.

## Add a subprocess timeout

`bundlefusion_reconstruct.py:191` has no `timeout=`; a hung container hangs the run forever.
Use a generous but finite timeout and kill the container on expiry.

# PHASE 4 -- engines/metashape.py

Agisoft Metashape 2.3.1 at `/data/zed-metashape/conda/env/bin/python`. Headless:
`metashape.sh -r script.py -platform offscreen` (Pro license required, Python API is Pro-only).

Official API defaults (verified against Manual 2.3 / Python API 2.3.1):
```
matchPhotos:   downscale=1, generic_preselection=True, reference_preselection=True,
               keypoint_limit=40000, tiepoint_limit=4000
alignCameras:  adaptive_fitting=False, reset_alignment=False
buildDepthMaps: downscale=2, filter_mode=MildFiltering
buildPointCloud: point_colors=True
buildModel:    face_count=HighFaceCount, source_data=DepthMapsData, interpolation=Enabled
```
`downscale=2` (High) is a deliberate deviation from the API default 4 -- it is the documented
quality lever and 16 GB VRAM handles 2K frames at High. `tiepoint_limit` goes back to the API
default 4000 (the old code used 10000).

ZED poses as weak priors, per the documented pattern: `camera.reference.location`,
`camera.reference.location_accuracy = (0.15, 0.15, 0.15)`, `rotation_enabled = False`.
Unset accuracy silently defaults to 10 m, so set it per camera.

Corrections:

1. **fy == fx is NOT a bug.** Agisoft's camera model (Manual Appendix D) has ONE focal length:
   `u = cx + x'*f + x'*B1 + y'*B2`, `v = cy + y'*f`. Non-square pixels live in `B1`. Do not
   "calibrate both focal lengths". Instead: on import set `user.f = fx` and
   `user.b1 = fy - fx`; on export write `fx = f + b1`, `fy = f`. The old code drops `b1`
   entirely (`metashape_pipeline.py:271`, `export_scannet.py:18`) while
   `optimizeCameras(adaptive_fitting=True)` at `:420` can estimate it -- so the exported K was
   wrong by b1.
2. **Fail loudly on depth-render failure.** `metashape_pipeline.py:298-300` does
   `if depth is None: depth = np.zeros(...)` and writes a valid black PNG. Four paths reach
   that: depth-maps access failure (`:194-195`), model/transform None (`:200-201`),
   `renderDepth` raising twice (`:204-209`), `_image_to_depth_m` parse failure (`:166-167`).
   Also `:210` returns the render result with NO `max>0` check. And
   `export_scannet.py:158-170` claims to validate "nonzero content" but only checks
   `depth[0]` is readable. Measured: 15 of 400 frames in the shipped scene4 output have <5%
   nonzero coverage (min 0.3%). Correct: raise on any all-zero or <1%-coverage depth.
3. **Count unaligned cameras.** `metashape_pipeline.py:252-254` silently drops cameras with no
   `transform`, and `n_aligned_skipped` at `:281-286` does not count them (it only counts
   colour-label mismatches). Logs showed 244-249 of 250 aligned -- 1-6 frames vanished
   untracked. Report the count; fail if > 10%.
4. **Do not swallow `optimizeCameras` failure** (`:422-423` prints and continues, leaving
   unoptimized alignCameras poses in the output).
5. **Do not substitute mesh sampling for a missing dense cloud** (`export_scannet.py:232-247`
   silently replaces an empty dense cloud with 50k-500k uniform mesh samples).
6. Poses: `chunk.transform.matrix @ camera.transform` is CORRECT (verified: 1.5 cm
   backprojection, all camera centres inside the cloud bbox). Note the result is a similarity
   transform -- chunk scale (~0.3-0.5) is baked into the rotation columns. Orthonormalize
   for the ScanNet output and put the scale where it belongs.
7. `renderDepth(camera.transform, sensor.calibration)` returns metres, camera-frame z
   (perspective depth), already in the aligned metric frame. This matches ScanNet's
   backprojection convention. Verified empirically.
8. Frame convention: `"y_up"` (world frame comes from the ZED priors).

# PHASE 5 -- engines/rtabmap.py

Two stages, both official. Research: `spellbook/tmp/rtabmap_zed_research.md`.

## Stage 1 -- capture

Use the OFFICIAL launch file: `rtabmap_examples/zed.launch.py` with
`use_zed_odometry:=true`. That remaps `odom` to `/zed/zed_node/odom`, i.e. ZED SDK positional
tracking as the odometry source, with RTAB-Map still doing its own loop-closure detection and
pose-graph optimization. Supports `camera_model:=zedx`.

The current `zed-rtabmap/scripts/run_reconstruct.sh:189-223` hand-rolls the graph with
`rgbd_sync` + `rgbd_odometry` and EXPLICITLY DISABLES ZED tracking (`:107-108`,
`pos_tracking_enabled: false`), so RTAB-Map's own visual odometry runs. Its in-file comment
claiming "Official pattern from zed.launch.py" (`:94`) is false. This is a graph rewrite, not
a flag flip.

Root cause of the 17-frame runs, with runtime evidence: RTAB-Map's own VO died ~20 s in
("Not enough inliers 0/20" from `OdometryF2M.cpp`, `Vis/MinInliers` default 20), stopped
publishing the `odom` TF, and rtabmap then could not localize a single later frame (WM stuck
at 18, "extrapolation into the future" spam). The same SVO yielded 2357 frames via the ZED
engine, so the recording is fine.

Keep RTAB-Map's SLAM parameters at defaults for this pass EXCEPT:
- `wait_for_transform` and `tf_tolerance`: raise from the defaults 0.2 / 0.1. The odom TF is
  stamped at frame time but published ~2.3 s late, so frames get dropped even when odom is
  healthy. Set both to >= 3.0.
- `approx_sync_max_interval`: keep 0.5 (already fixed and documented).
- `svo_realtime: true`, `use_svo_timestamps: false` (both already fixed and documented;
  SVO stamps caused a 937502 s TF-extrapolation gap).
- Pre-warm the NEURAL depth model before starting (first run spends ~6 min optimizing and
  blocks publishing).
- `enable_ipc: false`.

Do NOT lower `Vis/MinInliers` in this pass -- with ZED odometry the VO failure path is gone.
Threshold tuning is a later pass.

## Stage 2 -- offline refinement (maintainer's own recipe, rtabmap issue 1605)

```
rtabmap-reprocess -default --Rtabmap/DetectionRate 0 -odom rtabmap.db output.db
# then detect_more_loop_closures (rtabmap_slam service, or databaseViewer Post-processing)
rtabmap-export --cloud --mesh --texture --images_id --poses_camera output.db
```

**Drop `--opt 0`.** `run_reconstruct.sh:330` passes it, which exports the RAW graph poses --
the pose-graph optimization result is computed and then thrown away. This is why the export
showed 4.37 m centroid drift and a 26x22x6 m bbox against ZED's 15x5x13 m.

Other notes: `Mem/DepthAsMask` / `Vis/DepthAsMask` are already true by default and drop
features on invalid-depth glass pixels -- no extra glass handling. IMU is unusable in SVO mode
(the wrapper stamps IMU at publish time, so RTAB-Map's interpolation fails with "IMU won't be
added to graph") -- do not try to fix that here, just record it. Container shutdown: SIGTERM
5 s after SIGINT can truncate the DB save on larger maps; raise the grace period.

`normalize_export.py`'s depth/pose/intrinsics handling was verified correct (uint16 mm,
camera-to-world optical, same frame as the cloud) -- reuse its parsing logic, not its
pipeline.

Frame convention: `"rtabmap_map"` (Z-up already) -- verify with the backprojection residual.

# PHASE 6 -- engines/isaac.py

Real cuVSLAM + Nvblox. Research: `spellbook/tmp/cuvslam_nvblox_research.md`.

The current `zed-isaac` is NOT cuVSLAM/Nvblox: `src/` and `launch/` are empty directories,
`reconstruct_svo.py:99-103` hardcodes `auto -> open3d`, and `:121-122` raises
`NotImplementedError` for any other backend. What ran was pyzed extract + ZED SDK poses +
Open3D TSDF. The 34 cuVSLAM/Nvblox debs in `/data/zed-isaac/debs/` are on disk but unreachable
by any code path.

## Resolve this FIRST, before writing anything

`libvpi` is absent from both the debs and the `zed-isaac:jazzy` image (its `manifest.txt` has
no vpi entry). cuVSLAM's GXF graph init may need it. Verify before investing in the ROS graph:

```
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable \
  nvcr.io/nvidia/isaac/ros:release-4.5 nvidia-smi
```
Host prerequisites are all confirmed present: podman 4.9.3, driver 580.159.03 (>= 580),
RTX A4000 (Ampere), CDI specs in both `/var/run/cdi/nvidia.yaml` and
`/home/rolf/.config/cdi/nvidia.yaml`. `libcuvslam.so` in the existing image links cleanly with
the CUDA-13 mount. Nothing needs sudo.

If the NGC image is unusable, the cheaper validation path is `cuvslam_api_launcher` (the C
API, no ROS) against the already-extracted stereo grey frames + timestamps -- it validates
libcuvslam and VPI in one smoke test, and Nvblox can be appended once poses exist.

## Official pipeline

1. Pull `nvcr.io/nvidia/isaac/ros:<release-tag>`; build the ZED layer via the official
   `Dockerfile.zed` image key (ZED SDK + zed-ros2-wrapper baked into the image, nothing on
   host). Use rootless podman/buildah.
2. SVO2 replay: zed_wrapper `camera_model:=virtual` + `svo_file`. Record to a rosbag; the
   official Nvblox offline path is rosbag replay (`rosbag:=<path>`, `use_sim_time`).
3. cuVSLAM: `isaac_ros_examples.launch.py launch_fragments:=zed_stereo_rect,visual_slam`
   with the zedx quickstart interface specs (`base_frame`, `camera_optical_frames` per the
   official quickstart).
4. Nvblox: `nvblox_examples_bringup zed_example.launch.py camera:=zedx rosbag:=<path>`,
   with **cuVSLAM odometry** wired as the pose source. The official example defaults to
   `/zed/zed_node/pose` (ZED SDK pose) -- that is the old fake path; use cuVSLAM.
5. Nvblox params: `voxel_size = 0.004`,
   `static_mapper.projective_integrator_max_integration_distance_m = 4.0` (ScanNet
   `s_SDFMaxIntegrationDistance`; the default is 7.0),
   `projective_integrator_truncation_distance_vox = 15` (-> 0.06 m truncation at 4 mm voxels,
   = ScanNet `s_SDFTruncation`; the default is 4). Raise `block_count` for the finer voxel.
6. Export the mesh via `after_shutdown_map_save_path`.
7. `/data` must be mounted into the container -- the old `docker/run.sh:23-26` mounts only
   host libs and device nodes, so SVO files were unreachable in-container.
8. Add subprocess timeouts (there are none anywhere in the old code).

Frame convention: MEASURE IT. Do not assume. See C1's note on the double-flip bug.

---

# Cross-cutting rules

1. **No caching.** Delete every existence-based cache. All four engines had one, none keyed on
   settings: `metashape_reconstruct.py:62-63`, `isaac_ros_reconstruct.py:42-44`,
   `rtabmap_reconstruct.py:38-42`, `bundlefusion_reconstruct.py:311-313`. A partial 17-frame
   run was permanently cached as complete.
2. **No swallowed exceptions.** Never `except Exception: pass`. Never write a plausible-looking
   fallback artifact (zero depth, identity pose, mesh-sampled cloud) in place of a failure.
3. **Never `stderr=subprocess.DEVNULL`** (`zed_reconstruct.py:235` discards every worker's
   diagnostics). Capture to `recon/logs/<engine>.log`.
4. **Every subprocess gets a `timeout=`.**
5. **`settings_used.yaml` after the run, not before.** `ov3dis-comparison/main.py:112-116`
   writes it first, so a failed run leaves provenance claiming settings that never applied.
   Write the full command line into `recon/cmdline.txt`.
6. **Do not port `trim_scene.py`.** Two bugs: `margin` is added to the HEIGHT as well
   (`:81`), so `trim.margin: 3.0` grows the vertical box by 6 m and defeats the
   0.5/99.5-percentile height clipping the module exists for; and `np.linalg.eigh` plus column
   reversal (`:66-67`) can produce `det(R) = -1`, feeding a reflection to
   `OrientedBoundingBox`. If cropping is needed later, write it fresh.

# Verification before declaring a phase done

1. `recon/qc.yaml` exists, every metric PASS or an explicitly reported FAIL with a reason.
2. `python -c "from SensorData import SensorData; ..."` round-trips the `.sens`: frame count
   matches, `export_intrinsics` writes 4 files, a spot-checked depth PNG is uint16 with
   0 for invalid.
3. `len(segIndices) == vh_clean_2_verts`.
4. `det(axisAlignment) == 1`, `R[2][2] == 1`, z cross-terms == 0.
5. Backprojection residual <= 20 mm between `_vh_clean_2` and the `.sens` poses.
6. `/data/scannet/scans/scene0*` unchanged (`git status` on the repo, plus a spot-check that
   no ScanNet scan dir mtime moved).

# Known open risks -- report, do not paper over

- **R1** ScanNet's `improve`-stage (VoxelHashing) config is NOT in the repo. The 4 mm figure
  comes from the paper only and cannot be verified locally; even the local BundleFusion fork's
  default config uses 0.01. Our 4 mm is matched by MEASURED OUTPUT DENSITY, not by a
  documented parameter. This is the one place byte-fidelity cannot be claimed.
- **R2** Anything globbing `/data/scannet/scans/scene*` will now see scans with no GT. Check
  `/data/scannet/visualize_gt.py` before the first write. spellbook takes explicit scene ids
  and is unaffected.
- **R3** `libvpi` absence may block cuVSLAM entirely (Phase 6).
- **R4** ScanNet's `.aggregation.json` / `_vh_clean_2.labels.ply` / 2D label zips come from
  annotation, which is out of scope here. The interactive annotator is closed-source (SSTK,
  embedded by iframe in `WebUI/views/annotations.jade:14`); only Segmentator and the
  post-annotation 2D projection (`AnnotationTools/`) are public. Our job ends at
  `.segs.json`.

# After the pass

Delete this file and the `scout_*.md` / `*_research.md` files in `spellbook/tmp/`.
Update `spellbook/PROJECT_STATUS.md` with: the six scan ids produced, their qc.yaml numbers,
and the `deviations:` list per engine.
