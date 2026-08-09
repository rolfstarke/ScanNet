# Scout report: operational definition of "ScanNet quality" for custom multi-room reconstructions

Date: 2026-08-08. Sources: ScanNet paper (arXiv:1702.04405, Dai et al., CVPR 2017), BundleFusion paper (arXiv:1604.01093, Dai et al., ACM TOG 2017), scan-net.org changelog, ScanNet GitHub repo (local clone at /home/rolf/GIT/ScanNet), and direct measurement of the 20 released scenes at /data/scannet/scans (read-only).

## 1. ScanNet reconstruction pipeline (paper + production config)

Sensor (paper §3.1): **Structure Sensor (Occipital, PrimeSense/Kinect-v1-like) attached to an iPad Air 2**; depth and color hardware-synchronized at **30 Hz**; depth **640×480 (16-bit)**, color **1296×968 (H.264, 15 Mbps)**; auto white-balance/exposure on; per-device calibration from a printed checkerboard sequence (intrinsics + colorToDepth extrinsics + depth unwarping).

Reconstruction (paper §3.2): uploaded scans run through the data server (pipeline code: `Server/scan_processor.py`):
1. `convert` + `calibrate` → `.sens`
2. `recons`: **BundleFusion** ("FriedLiver") — global pose estimation **at 1 cm³ voxels** (paper: "we first run BundleFusion at a voxel resolution of 1 cm³"; production file `Server/tools/recons/zParametersScanNet.txt`).
3. `clean`: MeshLab `cleanLoRes.mlx` + `alignment.exe`
4. `improve`: VoxelHashing ("DepthSensing") re-integration with BundleFusion poses → `_vh.ply` mesh ("MLIB generated" header). Paper: hi-res mesh extracted by **Marching Cubes on the TSDF at 4 mm³ voxels** (§3.2).
5. `decimate` (MeshLab, `Server/tools/meshclean/`): `clean.mlx` → `_vh_clean.ply`; then **two passes of Quadric Edge Collapse decimation at 20 % faces per pass** (`simplify.mlx`) → `_vh_clean_1.ply` → `_vh_clean_2.ply`.

**Exact production TSDF settings** (`Server/tools/recons/zParametersScanNet.txt`, the actual file used for the dataset):
- `s_SDFVoxelSize = 0.010f` (1 cm voxels)
- `s_SDFTruncation = 0.06f` (6 cm) + `s_SDFTruncationScale = 0.02f` (+2 cm per metre distance)
- `s_sensorDepthMin = 0.1f`, `s_sensorDepthMax = 6.0f` (depth range 10 cm – 6 m)
- `s_SDFMaxIntegrationDistance = 4.0f` (only voxels ≤ 4 m from camera are integrated)
- integration re-sampled to **320×240**
- MC threshold = 10 × voxel size; `s_maxFrameFixes = 30` (max 30 frames re-integrated per frame)

BundleFusion pose-optimization config (`zParametersBundlingScanNet.txt`): SIFT keypoints at 640×480; local submaps of `s_submapSize = 10` frames; `s_maxNumImages = 2000` cap on the global pose graph; dense ICP/color residuals between 0.5–4.0 m; hierarchical local-to-global optimization with **implicit loop closure** (global SIFT matching, no explicit loop detector) and **on-the-fly de-/re-integration** of frames to fix drift when poses improve (BundleFusion §5). Demonstrated up to 14,785 frames (≈ 8.2 min @ 30 Hz, SUN3D sequence) in the paper.

## 2. Quantitative characteristics of released `_vh_clean_2.ply` meshes (measured on all 20 local scenes)

Paper Table 2 (official): **1513 scenes**, avg **floor area 22.6 m²**, avg **surface area 51.6 m²**, avg 1648 frames/scan ≈ **55 s @ 30 Hz**.

Measured on the 20 scenes at /data/scannet/scans (all-faces-valid parser, cross-validated: our 55.2 m² mean matches the paper's 51.6 m²):

| metric | mean | median | min–max |
|---|---|---|---|
| `_vh_clean.ply` vertices | 4.05 M | ~3.4 M | 2.66 M – 5.82 M |
| `_vh_clean_2.ply` vertices | 165 k | 163 k | 108 k – 239 k |
| `_vh_clean_2.ply` faces | 314 k | 310 k | 210 k – 457 k |
| bbox extent (m) | 5.8 × 5.9 × 2.5 | 6.0 × 6.4 × 2.5 | ~4–8 m horizontal, 1.4–3.1 m tall |
| surface area (m²) | 55.2 | 54.6 | 37 – 80 |
| vertex density (verts/m² of floor bbox) | 5 225 | 4 542 | 3 000 – 9 500 |
| scan duration | — | ~46 s | 19 – 102 s (579–3052 depth frames) |

Vertex/frame counts in the paper text: 2,492,518 RGB-D frames over 1513 scans; 36,213 labeled instances.

Note: `_vh_clean_2` ≈ 4 % of `_vh_clean` face count (two 20 % decimation passes). Also note (v2 changelog 2018-06-11): v2 re-released annotation files; segments labeled `remove` **were removed from the meshes**, so v2 `_vh_clean(_2).ply` can have holes — they remain in the release (annotation/mask files still reference them).

## 3. `axisAlignment` (4×4 matrix in `sceneXXXX_XX.txt`)

- Semantics (scan-net.org changelog, v2, 2018-06-11): "a 4x4 matrix encoding **the rigid transform to axis alignment** for the scan, as 16 values in row-major order". Paper §3.2: canonical frame = **z-axis up**, xy-plane parallel to the floor, scan **translated into the positive octant**; computed automatically: planar region extraction (normal threshold 25°, planar offset 5 cm), floor selection via **IMU gravity prior** (projected into first-frame coords), **PCA on mesh vertices** for the rotation about z, then translation.
- Empirically (scene0019_00, scene0304_00, scene0575_02): the released `_vh_clean_2.ply` is already **z-up (floor ≈ z = 0)** but is **NOT xy-axis-aligned**: applying `axisAlignment` (pure rotation about z + translation, det = 1) shrinks the horizontal bbox (e.g. 6.32 × 6.04 m → 5.33 × 6.35 m). The **positive-octant translation is NOT in the released matrix** (min is negative after application) — a v2-vs-paper discrepancy; irrelevant for networks.
- Consumers: the **official ScanNet200 preprocessing applies it** (`BenchmarkScripts/ScanNet200/preprocess_scannet200.py:49-59` rotates the `_vh_clean_2.ply` vertices by `axisAlignment`). **Every ScanNet200-trained model — Mosaic3D, Open3DIS, OpenYOLO3D, OpenMask3D — is trained/evaluated on this axis-aligned point cloud** (z-up, walls parallel to axes). None of these repos apply `axisAlignment` themselves; they consume the preprocessed data (verified: no `axisAlignment` references in /data/open3dis, /data/mosaic3d, /data/openmask3d, /data/openyolo3D code).
- Must consumers apply it? **Yes for the canonical ScanNet200 frame.** v1 meshes were pre-aligned; v2 added the matrix so consumers can align raw `.sens` data themselves.
- Caveat (empirical): in this release, `.sens` camera poses (camera-to-world) are **not in the same frame as the mesh** (camera-up ≈ −z, path does not match mesh bbox after applying `axisAlignment`). Don't mix raw poses with aligned geometry without transforming.

## 4. Single-room vs multi-room, drift handling, limits

- **ScanNet scenes are single spaces.** The 20 local scenes are all single-room types (bedroom×5, kitchen×4, conference×4, living×3, hallway×2, lobby, copy-room). Paper §4: spaces range from small (bathroom, closet) to large (apartments, classrooms, libraries) — a few multi-room apartments/offices exist, but the dataset norm is one room of ≈ 23 m² floor / ≈ 52 m² surface.
- Drift is handled by **BundleFusion's global pose optimization**: hierarchical local-to-global (submaps of 10 frames), implicit loop closure via global SIFT matching, and continuous de-/re-integration (up to 30 frames/frame) so the TSDF is corrected as poses improve. There is **no documented limit on room count**, but the production pose-graph caps at `s_maxNumImages = 2000` frames (≈ 66 s @ 30 Hz), and validation (paper §3.2) automatically discards scans that are **short, have high residual reconstruction error, or low % of aligned frames**, plus manual rejection of visible misalignment.
- Practical conclusion: ScanNet quality was achieved on **~1-minute, single-room, handheld** recordings with strong visual texture; multi-room needs equivalent global-optimization + loop-closure machinery, and drift tolerance is implicit in the validation criteria.

## 5. Capture protocol (paper §3.1, ScannerApp)

- Intended for **untrained users**: iOS app with live RGB-D view and a **"featurefulness" bar** (log-scale RGB feature detector metric) giving live feedback on tracking robustness — the only quality guidance in the UI.
- Recommended behavior (implicit): continuous slow handheld motion covering surfaces from multiple angles; loop closure = revisiting areas (BundleFusion explicitly benefits from revisits); depth 0.1–6 m, effective integration 0.5–4 m.
- No explicit duration guidance; the production default `s_numSolveFramesBeforeExit = 60` (60 extra pose-optimization frames after capture) and the 2000-frame bundling cap imply a practical target of **≤ 1–2 min per scene**; released scans are 19–102 s.
- Storage/format: 16-bit zLib-compressed depth, H.264 15 Mbps color, IMU at ~53 Hz, timestamps for all streams.
- Relevance to the reported "~6 s motion, ~150 s static" recordings: ScanNet contains no such pattern (mean scan is ~55 s of continuous motion). 150 s static is wasted data for a 55 s-scan target, adds zero coverage, and inflates pose-graph/fusion time; worse, a static sensor with moving people/objects can poison re-integration. Target continuous coverage with revisits instead.

## 6. `_vh_clean.ply` vs `_vh_clean_2.ply` vs `.labels.ply`

From `Server/scan_processor.py` + `Server/tools/meshclean/*.mlx` (the actual production code):
- `_vh_clean.ply`: MeshLab `clean.mlx` on the VoxelHashing output: Merge Close Vertices (1.07 mm), Remove Duplicate Faces, Remove Isolated Pieces (< 1000 faces), Remove Unreferenced Vertices. "vh" = VoxelHashing ("MLIB generated" header).
- `_vh_clean_2.ply`: `simplify.mlx` × 2 (Quadric Edge Collapse, TargetPerc 0.2, QualityThr 0.3, OptimalPlacement, AutoClean) + the same cleanup — the **annotation/benchmark mesh** (evaluation is per-vertex in this exact vertex order).
- `.labels.ply`: identical geometry to `_vh_clean_2.ply`, plus per-vertex `label` (ScanNet label id) and instance color; produced from the aggregation JSONs.
- README: `_vh_clean.ply` = "high quality reconstructed mesh"; `_vh_clean_2.ply` = "cleaned and decimated mesh for semantic annotations".

## 7. Operational acceptance thresholds — what a custom scan must satisfy to be "ScanNet-like"

Target the distribution below (from §2; use median as the acceptance bar):

| check | acceptance threshold (ScanNet median) |
|---|---|
| surface area | 30–80 m² (median ≈ 55 m²); ± anything, but ≥ ~25 m² to resemble a room |
| bbox extent | 4–8 m horizontal, 1.5–3 m height; z-up, floor flat at z = 0 |
| vertex density (vh_clean_2-like mesh) | ≥ ~3 000 verts/m² floor (median 4 500) — i.e. ≈ 1–2 cm spacing on surfaces |
| mesh resolution | faces ≈ 200 k–460 k for a 50 m² room; hi-res ≥ 2 M verts before decimation |
| frame coverage | ≥ ~500 valid depth frames, ≥ 95 % frames with valid poses |
| geometry quality | no visible double-wall/ghosting (drift), floor planar and horizontal, closed enough for instance segmentation (walls/floors present, no huge holes) |
| frame/pose consistency | poses consistent with mesh (project depths into mesh within ~2 cm) |
| alignment | **axis-aligned z-up canonical frame** (apply your equivalent of `axisAlignment` before feeding the models — all four target models expect the ScanNet200-aligned frame) |

Feeding the pretrained models: produce per-scene: aligned point cloud or mesh with vertex colors (like `_vh_clean_2.ply` at ~160 k verts / 300 k faces), optional `_vh_clean_2.labels.ply`-style label channel, and (for OpenMask3D-style pipelines) fused depth+rgb+poses consistent with the aligned geometry.

## Sources

- ScanNet paper: https://arxiv.org/abs/1702.04405 (§3.1, §3.2, §4, Table 2)
- BundleFusion paper: https://arxiv.org/abs/1604.01093 (§5: hierarchy, integration/de-integration, voxel 4 mm default, 14,785-frame SUN3D demo)
- Production config: /home/rolf/GIT/ScanNet/Server/tools/recons/zParametersScanNet.txt, zParametersBundlingScanNet.txt; pipeline: Server/scan_processor.py; meshlab filters: Server/tools/meshclean/{clean,cleanLoRes,simplify}.mlx
- axisAlignment: scan-net.org/changelog (v2, 2018-06-11); BenchmarkScripts/ScanNet200/preprocess_scannet200.py
- Measured data: /data/scannet/scans/scene{0019,0217,0304,0412,0414,0426,0488,0549,0568,0575}* (20 scenes)
