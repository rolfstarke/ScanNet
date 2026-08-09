# Scout report: Agisoft Metashape official docs — indoor reconstruction, defaults, priors, rigs, calibration

Date: 2026-08-08. Sources: Metashape Pro User Manual 2.3 (PDF), Python API Reference 2.3.1 (PDF, Apr 2026), Agisoft Helpdesk KB, Agisoft forum (staff: Alexey Pasumansky).
All defaults below are quoted from the **2.3.1 API reference** unless noted; the manual chapters are quoted from the **2.3 manual**. Version caveats flagged where defaults changed historically (changelog included in the API PDF).

---

## 1. Official indoor/interior workflow

There is **no dedicated "interior processing" tutorial**, but there IS an official KB article for interior capture and a close-range processing tutorial:

- **KB "Suggested scenario for photo shooting an interior"** — https://agisoft.freshdesk.com/support/solutions/articles/31000163324 (the only official indoor-specific guidance; capture-side, not processing-side). Key points:
  - Multi-room: *"If the scenario includes digital reconstruction of several adjacent rooms, we recommend performing test shooting to work out the best scenario and also to adjust the overlap for rooms and for transitions between rooms."*
  - Shoot by walking with your back to a wall, aiming at the opposite wall; **never shoot from one point in the room center**.
  - **Overlap: at least 60–70%, each point visible on ≥3 photos.**
  - Adjacent rooms: open doors, photograph from **each side of the door**, adjacent room walls must be visible in the images.
  - Difficult surfaces (official list): transparent windows and doors, smooth/plain walls, lamps, reflecting surfaces → use a **circular polarizing filter**.
  - Plain/one-color walls are explicitly called "difficult for reconstruction".
  - Ceiling: shoot vertically up with ≥60% side / ≥80% forward overlap (low ceilings: lower the camera, tilt slightly).
  - Use a tripod/gimbal; lowest ISO; max resolution; fixed wide-angle lens; don't crop/resize images.
- **KB "3D model reconstruction" (close-range, updated Apr 2026)** — https://agisoft.freshdesk.com/support/solutions/articles/31000152092 — official processing recommendations for close-range:
  - Align Photos: enable **Generic preselection** for >100 images; **Key point limit 20 000–100 000; Tie point limit 2 000–40 000**.
  - **Build Model from Source data = Depth maps** (GPU-accelerated, "mostly provides better results ... for objects and scenes with a big number of minor details").
  - Texture atlas 4096–16384 px; refine the bounding box after alignment (reconstruction only uses data inside it).
- **KB "Terrestrial shooting scenario"** — https://agisoft.freshdesk.com/support/solutions/articles/31000149339 — avoid camera inclination >45°, tracks parallel to the surface, mask irrelevant foreground/background, add markers for referencing.
- Aerial-vs-interior settings difference: Agisoft does **not** prescribe different alignment/depth-map accuracy presets per scene type; the differences are capture-side (overlap, camera path) and model-side (surface_type Arbitrary vs HeightField, see §2).
- Staff-recommended workflow for difficult near-range data (Alexey Pasumansky, forum topic 13412, https://www.agisoft.com/forum/index.php?topic=13412.0): *"limit the key points to 40 000 (for tie point limit you can use 4000, but I've used 10 000). adjust the bounding box size after camera alignment to include all the parts of the scene of interest... build mesh using the depth maps source option skipping the dense cloud generation."*

## 2. Exact API defaults (Python API Reference 2.3.1, PDF: https://www.agisoft.com/pdf/metashape_python_api_2_3_1.pdf)

### Chunk.matchPhotos (p.66; Task: `Metashape.Tasks.MatchPhotos`, p.~283)
```
matchPhotos(downscale=1, downscale_3d=1, generic_preselection=True, reference_preselection=True,
  reference_preselection_mode=ReferencePreselectionSource, filter_mask=False, mask_tiepoints=True,
  filter_stationary_points=True, keypoint_limit=40000, keypoint_limit_3d=100000,
  keypoint_limit_depth_maps=10000, keypoint_limit_per_mpx=1000, tiepoint_limit=4000,
  keep_keypoints=False, guided_matching=False, reset_matches=False, subdivide_task=True,
  workitem_size_cameras=20, workitem_size_pairs=80, max_workgroup_size=100, ...)
```
- downscale mapping (exact quote): **"(0 - Highest, 1 - High, 2 - Medium, 4 - Low, 8 - Lowest)"**; default **1 (High)**.
- `reference_preselection_mode: Metashape.ReferencePreselectionMode` = `[ReferencePreselectionSource, ReferencePreselectionEstimated, ReferencePreselectionSequential]` (p.147).

### Chunk.alignCameras (p.35)
```
alignCameras([cameras][, point_clouds], min_image=2, adaptive_fitting=False, reset_alignment=False,
  subdivide_task=True, align_laser_scans=False)
```
- `adaptive_fitting` = "Enable adaptive fitting of distortion coefficients" (default **False** in API; GUI checkbox is on by default — manual p.46).
- `reset_alignment` default False (changelog: changed to False in 1.6.2).

### Chunk.buildDepthMaps (p.40)
```
buildDepthMaps(downscale=4, filter_mode=MildFiltering, reuse_depth=False, max_neighbors=16,
  subdivide_task=True, workitem_size_cameras=20, max_workgroup_size=100)
```
- downscale mapping (exact quote): **"(1 - Ultra high, 2 - High, 4 - Medium, 8 - Low, 16 - Lowest)"**; default **4 (Medium)**. Note: unlike matchPhotos there is no "Highest/0" level for depth maps.
- `filter_mode: Metashape.FilterMode` = `[NoFiltering, MildFiltering, ModerateFiltering, AggressiveFiltering]`; default **MildFiltering**.

### Chunk.buildPointCloud (p.40; GUI "Build Dense Cloud" in ≤1.x, "Build Point Cloud" in 2.x)
```
buildPointCloud(source_data=DepthMapsData, point_colors=True, point_confidence=False, keep_depth=True,
  max_neighbors=100, uniform_sampling=True, points_spacing=0.1, ...)
```

### Chunk.buildModel (p.40–41)
```
buildModel(surface_type=Arbitrary, interpolation=EnabledInterpolation, face_count=HighFaceCount,
  face_count_custom=200000, source_data=DepthMapsData, vertex_colors=True, vertex_confidence=True,
  volumetric_masks=False, keep_depth=True, split_in_blocks=False, build_texture=True, ...)
```
- `surface_type` = `[Arbitrary, HeightField]` — manual p.51: **Arbitrary for closed objects (statues, buildings)**; **HeightField only for planar scenes/aerial**.
- `source_data` = `[PointCloudData, DenseCloudData (alias), DepthMapsData, TiePointsData, ModelData, ...]` — manual p.50: Depth Maps source is recommended for Arbitrary reconstruction, skip the dense cloud.
- `face_count: Metashape.FaceCount` = `[LowFaceCount, MediumFaceCount, HighFaceCount, CustomFaceCount]`; 2.3.1 API default is **HighFaceCount** (GUI default "High"). Manual p.51: for point-cloud source the ratios are 1/5 (High), 1/15 (Medium), 1/45 (Low) of the point count.
- `interpolation` default **EnabledInterpolation**; disabled → accurate, only areas present in the source.

### Chunk.buildUV / buildTexture (p.45 / p.43)
```
buildUV(mapping_mode=GenericMapping, page_count=1, texture_size=8192, pixel_size=0)
buildTexture(blending_mode=NaturalBlending, texture_size=8192, downscale=2, sharpening=1,
  fill_holes=True, ghosting_filter=True, out_of_focus_filter=False, color_enhancement=False,
  texture_type=DiffuseMap, source_data=ImagesData, transfer_texture=True, anti_aliasing=1)
```
- buildTexture default blending **NaturalBlending** (changelog: changed from Mosaic to Natural in 2.0.x). Manual describes Mosaic vs Natural (Natural auto-selects best image per triangle).

### Downscale ⇔ GUI preset table (official)
| GUI preset | matchPhotos downscale | buildDepthMaps downscale | manual description (Align Photos, p.44 / Build Model, p.56) |
|---|---|---|---|
| Highest / Ultra High | 0 | 1 | Align: image upscaled ×4; Depth: original photos |
| High | 1 | 2 | Align: original size |
| Medium | 2 | 4 | ×4 area (2× per side) |
| Low | 4 | 8 | ×16 area (4× per side) |
| Lowest | 8 | 16 | ×64 area (8× per side) |

Manual quotes: *"High accuracy setting the software works with the photos of the original size, Medium setting causes image downscaling by factor of 4 (2 times by each side)..."* (Align Photos, p.44); *"Ultra High quality setting means processing of original photos, while each following step implies preliminary image size downscaling by factor of 4 (2 times by each side)"* (Depth maps, p.56).

## 3. Camera-position/rotation priors (reference data)

Documented mechanism (manual ch. "Reference and calibration", p.100–107; API "Camera.Reference" p.28, "Sensor.Reference" p.149, "Chunk" p.46):
- `camera.reference.location`, `camera.reference.rotation` (rotation as Euler angles; configurable convention via `chunk.euler_angles`, `[YPR, OPK, POK, sANK]`), `camera.reference.location_accuracy` (m), `camera.reference.rotation_accuracy` (deg), plus per-item `enabled` flags (`location_enabled`, `rotation_enabled`). Chunk-wide: `chunk.camera_location_accuracy`, `chunk.camera_rotation_accuracy`.
- Import: `chunk.importReference(path, format=ReferenceFormatCSV, columns=..., delimiter=..., skip_rows=..., group_delimiters=..., crs=..., load_location_accuracy=False, load_rotation_accuracy=False, load_enabled=False, ...)` (p.60). CSV columns include X,Y,Z and yaw/pitch/roll or omega/phi/kappa; accuracy columns with "/" delimiters per-axis (manual p.101).
- Default accuracy if unset: **10 m** (manual p.107, RTK/PPK paragraph: *"Otherwise default accuracy value (10 m) will be assumed for all camera coordinates"*). **You must set accuracy explicitly** — otherwise your 0.15 m prior is effectively ignored and weighting is flat.
- Disabling rotation: manual p.107: *"It is important to enable checkboxes for coordinates and angles on the Reference pane to take these information into account during alignment."* → a config with only location enabled (rotation disabled) is the documented, supported mode. **0.15 m location + rotation disabled is a perfectly sensible, documented configuration** — accuracy values are just weights in the bundle adjustment; nothing official prescribes a value, but 0.15 m is small enough to dominate the fit while leaving room for drift correction. Keep `alignCameras(adaptive_fitting=False)` to avoid free-floating distortion params on weak indoor geometry.
- `Metashape.Tasks.AlignCameras`, `Metashape.Tasks.MatchPhotos`, `Metashape.Tasks.OptimizeCameras` (Task API, p.283+).
- `reference_preselection` role (manual p.45): in **Source** mode, pairs are chosen by measured camera locations (requires `Capture distance` for oblique imagery; ignored if rotation angles absent); **Estimated** mode uses previously estimated camera poses (for re-runs); **Sequential** mode uses photo sequence order (first and last are also compared). For video sequences this is the documented way to massively speed up matching — with our ZED priors, Source mode (or default in 2.3.1 = Source) is right; without priors, Sequential.
- Optimization with priors: `chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True, fit_b1=False, fit_b2=False, fit_k1=True, fit_k2=True, fit_k3=True, fit_k4=False, fit_p1=True, fit_p2=True, fit_corrections=False, adaptive_fitting=False, ...)` (p.66) — manual: always run after editing reference data.

## 4. Video / frame-sequence guidance

- Official: `File > Import Video...` (manual p.34–35). Frame extraction rate: automatic step **Small / Medium / Large ≈ 3% / 7% / 14% of image width shift** (`Metashape.FrameStep = [CustomFrameStep, SmallFrameStep, MediumFrameStep, LargeFrameStep]`, API p.146); `Start from` / `End at` to trim; extracted frames are added to the active chunk automatically. No official minimum-overlap percentage for video per se — the interior KB's **60–70% overlap / point visible in ≥3 photos** is the official rule of thumb (≈3–7% shift per frame matches it comfortably).
- **Analyze quality**: `chunk.analyzeImages(filter_mask=False)` → camera.meta["Image/Quality"]; manual: *"Cameras with quality less than 0.5 are considered blurred and we recommend to disable them"* (API p.36). This is the documented answer to motion blur — disable blurry frames.
- Incremental alignment (manual p.47): add frames, re-run Align Photos **with "Reset current alignment" unchecked** (API: `reset_alignment=False`) so new frames match against the existing model.
- `chunk.reduceOverlap(...)` (Tools > Reduce Overlap, manual p.24) — removes redundant frames from video dumps after a first alignment + rough mesh.

## 5. Indoor-specific problems — official mitigations

- **Low texture walls**: official acknowledgment in the interior KB ("Surfaces like walls without texture/pattern... are difficult for reconstruction") — mitigations: shoot close to the wall so details are visible; markers; use **Mild depth filtering** (manual p.61: *"Mild depth filtering mode is also required for the depth maps based mesh reconstruction"* and recommended *"for aerial projects in case the area contains poorly textured roofs"*). Staff (topic 13412): raise tie point limit to 10 000, build mesh from depth maps directly.
- **Glass/windows, reflections**: polarizing filter (KB interior); masks (manual "Using masks", p.130: masked areas excluded from matching, depth maps, model, texture).
- **Repetitive/identical surfaces**: KB terrestrial: increase unique elements / markers + high overlap.
- **Doorways/adjacent rooms**: interior KB — open doors, shoot both sides (see §1). If the dataset splits into components (manual p.46–47): align components with **≥3 markers** not on a line, then `Merge Components`. KB: https://agisoft.freshdesk.com/support/solutions/articles/31000158896 (connected components).
- **Coded targets & scale bars**: manual p.107+ (`Tools > Print Markers` generates Circular or AprilTag targets; automatic detection; used as true matches + scale/georeferencing). KB: https://agisoft.freshdesk.com/support/solutions/articles/31000148855 ("Coded targets and Scale bars"). Scale bars fix metric scale for interiors without GCPs — relevant for us since ZED poses give scale from stereo.
- **Masking**: `chunk.generateMasks` (model-based, background-photo based); `filter_mask`/`mask_tiepoints` in matchPhotos ("Apply mask to key points/tie points" in GUI). KB: https://agisoft.freshdesk.com/support/solutions/articles/31000153479, .../31000163388 (automatic masking from model).
- **Multi-room via multi-chunk**: manual ch. "Working with chunks" (p.168): process rooms in separate chunks, then `Document.alignChunks([chunks], reference, method=0|1|2, fit_scale=True, downscale=1, generic_preselection=False, filter_mask=False, mask_tiepoints=False, keypoint_limit=40000)` (API p.84) — methods: 0 tie-point based, 1 marker based, 2 camera based. KB: https://agisoft.freshdesk.com/support/solutions/articles/31000167860 (batch processing for multiple chunks).

## 6. Stereo pair / camera rigs — official support

**Yes, officially supported**, documented feature name: **"Rigid camera rig data" / "Multi-camera system"** (manual p.33–35):
- Organize each rig camera's images in its own subfolder; `Workflow > Add Folder`, in Add Photos select **"Multi-camera system layout"** and **"Create sensor from each subfolder"**.
- One camera becomes **master**, others **slave**; default assumption: *"synchronized cameras have the same position in space"*. Relative offsets can be **input manually with accuracy** or auto-estimated (*Adjust location* in the **Slave offset** tab of Tools > Camera Calibration); offset variance shown in the GNSS/INS Offset tab.
- This gives the ZED use case exactly: left+right at each pose as a calibrated rigid rig with a **known 12 cm baseline** → relative orientation of the two cameras fixed/constrained → stronger geometry and scale; absolute metric scale still needs scale bars or a known-distance reference (or your ZED trajectories, which are already metric).
- API surface (2.3.1): `camera.master` ("Master camera", p.30), `camera.group` → `Metashape.CameraGroup` with `Type = [Folder, Station]` (p.31–32). There is **no dedicated Python class for rigs** — rig setup is a GUI import-layout feature; the API exposes it via master/slave references. Caveat: with both eyes of one stereo pair the two views share pose baselines of only 6 cm each side — fine for matching, but the two images of a pair are nearly redundant for triangulation; the gains are texture coverage, redundancy, and constraint on scale.

## 7. Calibration model — and the fx/fy question

**Model** (manual Appendix D "Camera models", p.231–233; API `Metashape.Calibration`, p.19–20):
- Parameters: `f` (focal length, px), `cx, cy` (principal point, px), `k1..k4` (radial), `p1, p2` (tangential/decentering), `b1` (Affinity), `b2` (Non-orthogonality/skew).
- Frame camera projection (exact equations from the manual):
  - `u = w*0.5 + cx + x'*f + x'*B1 + y'*B2`
  - `v = h*0.5 + cy + y'*f`
  - with `x', y'` the normalized+distorted coordinates (Brown's model).
- **ANSWER: there is no separate fy — Metashape's model has a single focal length `f`, and `v = cy + y'*f` exactly.** Effective horizontal focal is `f + B1` (via the `x'*B1` term), so `fx≠fy` is only representable as an affine correction `B1` (in pixels; expected magnitude "not more than a few units", KB camera calibration article). **"Forcing fy=fx" is NOT a bug and NOT a workaround — it is simply the structure of the model.** Importing a ZED calibration: set `f = fy` (and optionally `b1 = fx - f`), `cx, cy, k1..k4, p1, p2`. API confirms: `optimizeCameras(fit_b1=...)` is documented as *"Enable optimization of aspect ratio"*, `fit_b2` as *"skew coefficient"*; `Calibration.b1` = "Affinity", `b2` = "Non-orthogonality" (API p.19–20).
- Fixing calibration: **`sensor.fixed_calibration`** ("Fix calibration flag", API p.151); GUI: Camera Calibration dialog → **"Fixed parameters"** (manual p.103: fixed parameters are not adjusted during alignment/optimization; also recommended for precalibrated cameras, p.114). Import: `Calibration.load(path, format=CalibrationFormatXML)` or set fields directly: `sensor.calibration.f = ...; sensor.width/height` (width/height in px must match the images).
- KB "What does camera calibration results mean in Metashape?" — https://agisoft.freshdesk.com/support/solutions/articles/31000158119: cx/cy should be within a few dozen px of center, b1/b2 "usually not more than a few units"; if hundreds/thousands → calibration failed; fix params and re-align with **adaptive camera model fitting disabled**.

## 8. Depth map export / rendering

- **Option A (recommended by us): render depth from the model per camera** — `Metashape.Model.renderDepth(transform, calibration, cull_faces=True, add_alpha=True) → Metashape.Image` (API p.110): call with `camera.transform` and `sensor.calibration` → full-resolution depth image. Batch variant: `Metashape.Tasks.RenderDepthMaps` (API p.288–289; `cameras`, `path_depth`, `save_depth/save_diffuse/save_normals`; renamed from `ExportDepth` in 2.0) = GUI Tools > Render Depth Maps. Depth values are in **chunk coordinate-system units (meters if the chunk is scaled/referenced)** — no explicit units statement in the API doc; verify empirically once (staff confirm chunk units).
- **Option B: use built depth maps** — `chunk.buildDepthMaps(...)`, then `chunk.depth_maps` (`Metashape.DepthMaps`: `keys()/items()/values()/meta`), per camera `Metashape.DepthMap.image(level=0)`, calibration at each downscale level via `getCalibration(level)` (API p.82–83). The stored pixel values are metric distances in chunk units (per level downscale); `level` selects the pyramid level.
- `renderDepth` also exists on `PointCloud`/`TiledModel` (with `resolution` screen-pixel parameter, API p.135/p.313). There is **no official `Chunk.renderDepth`** — the official per-camera paths are the two above.

## 9. Headless / CLI / licensing

- Documented (manual ch. "Python scripting", p.172; KB "How to run the script in headless mode from the command-line", https://agisoft.freshdesk.com/support/solutions/articles/31000133141):
  - `./metashape.sh -r script.py [args]` (Linux), `metashape.exe -r script.py` (Win), `MetashapePro -r script.py` (mac); add **`-platform offscreen`** on headless/GUI-less Linux.
  - Autorun folders: `~/.local/share/Agisoft/Metashape Pro/scripts/` etc. (manual p.172).
- Licensing: no extra license for headless; the normal node-locked or floating Metashape Pro license is used. Python API is **Professional-only** (API overview, p.7: "Python scripting is supported only in Metashape Professional edition"). Floating license borrowing for offline batch runs: KB .../31000157620. Trial: KB "How to try full Metashape functionality before buying" (.../31000135259).

---

## Practical takeaways for ScanNet pipeline

1. **Reset to 2.3.1 official defaults**: `matchPhotos(downscale=1, generic_preselection=True, reference_preselection=True, keypoint_limit=40000, tiepoint_limit=4000, keep_keypoints=False)`, `alignCameras(adaptive_fitting=False, reset_alignment=False)`, `buildDepthMaps(downscale=4, filter_mode=MildFiltering)`, `buildModel(surface_type=Arbitrary, source_data=DepthMapsData, face_count=HighFaceCount, interpolation=EnabledInterpolation)`, `buildUV(texture_size=8192)`, `buildTexture(blending_mode=NaturalBlending, texture_size=8192)`.
2. For ScanNet density (≈3000 verts/m²) on ZED 2K frames + A4000 16GB: depth maps at **downscale=2 (High)** first, try **1 (Ultra High)** for wall-detail-critical regions; mesh from depth maps (skip dense cloud — also the staff recommendation); face_count Medium/High.
3. Priors: `camera.reference.location` + `location_accuracy=(0.15,0.15,0.15)` + `rotation_enabled=False` is a documented, sensible config; set `reference_preselection_mode=ReferencePreselectionSource` (default) with imported orientation, or `Sequential` without it. Set accuracy on every camera — unset means 10 m.
4. fx/fy: not a bug; single-`f` model, use `b1 = fx − f` (pixels) if desired. Fix calibration with `sensor.fixed_calibration = True`.
5. Depth per camera for the model: `model.renderDepth(cam.transform, cam.sensor.calibration)` → depth in chunk units (meters if scaled); keep `chunk.scale = 1` via `chunk.transform`/reference for metric output.
6. Multi-room: process per-room chunks + `doc.alignChunks(...)`; or one chunk with door-frames bridging, per the official KB shooting guidance.

## Source URLs
- Manual 2.3: https://www.agisoft.com/pdf/metashape-pro_2_3_en.pdf (240 pp, chapters General workflow / Reference and calibration / Appendix D)
- API 2.3.1: https://www.agisoft.com/pdf/metashape_python_api_2_3_1.pdf (367 pp incl. changelog)
- KB interior: https://agisoft.freshdesk.com/support/solutions/articles/31000163324
- KB close-range 3D reconstruction: https://agisoft.freshdesk.com/support/solutions/articles/31000152092
- KB terrestrial: https://agisoft.freshdesk.com/support/solutions/articles/31000149339
- KB general capture: https://agisoft.freshdesk.com/support/solutions/articles/31000149337
- KB calibration results: https://agisoft.freshdesk.com/support/solutions/articles/31000158119
- KB headless CLI: https://agisoft.freshdesk.com/support/solutions/articles/31000133141
- KB smart cameras w/ depth: https://agisoft.freshdesk.com/support/solutions/articles/31000162212
- Forum staff recommendations: https://www.agisoft.com/forum/index.php?topic=13412.0
- Tutorials hub: https://www.agisoft.com/support/tutorials/
