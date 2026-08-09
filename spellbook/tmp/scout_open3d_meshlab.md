# Scout report: Open3D Reconstruction System + pymeshlab as meshlabserver replacement

Date: 2026-08-08. Sources verified: Open3D 0.19.0 (main, github.com/isl-org/Open3D), pymeshlab 2025.7.post1 (PyPI, based on MeshLab 2025.07), MeshLab source main.

---

## TOPIC A — Open3D Reconstruction System (official)

### A1. Stage sequence (run_system.py)

```
python run_system.py config.json --make --register --refine --integrate   [--slac --slac_integrate]
```
4 (+2 optional SLAC) stages, each a module in `examples/python/reconstruction_system/`:

| Stage | Consumes | Produces |
|---|---|---|
| **make_fragments** | `image/` (jpg/png), `depth/` (png), `path_intrinsic` | `fragments/fragment_%03d.ply` (colored pointcloud), `fragments/fragment_%03d.json` (posegraph), `fragments/fragment_optimized_%03d.json` |
| **register_fragments** | `fragments/*.ply` + `fragment_optimized_*.json` | `scene/global_registration.json`, `scene/global_registration_optimized.json` |
| **refine_registration** | `fragment_*.ply`, `global_registration_optimized.json` | `scene/refined_registration.json`, `scene/refined_registration_optimized.json`, `scene/trajectory.log` (per-frame 4x4 poses) |
| **integrate_scene** | `refined_registration_optimized.json` + `fragment_optimized_*.json` | `scene/integrated.ply` (colored triangle mesh), `scene/trajectory.log` |
| slac (opt.) | fragments | `slac/` control grid + `optimized_posegraph_slac.json` |
| slac_integrate (opt.) | slac output | final pointcloud/mesh in `slac/` |

Details (from source):
- make_fragments: consecutive frames → `compute_rgbd_odometry` (identity init, `RGBDOdometryJacobianFromHybridTerm`, `OdometryOption.depth_diff_max`); non-adjacent keyframe pairs (`n_keyframes_per_n_frame` step) → OpenCV ORB + 5-point RANSAC init, then odometry; per-fragment pose graph optimized (`GlobalOptimizationLM`, `edge_prune_threshold=0.25`, `preference_loop_closure=preference_loop_closure_odometry`); fragment TSDF integrated (`ScalableTSDFVolume`, voxel `tsdf_cubic_size/512`, `sdf_trunc=0.04` hardcoded, `RGB8`) → mesh → pointcloud ply. Multiprocessing via `multiprocessing.Pool(spawn)`, `OMP_NUM_THREADS=1`.
- register_fragments: **loop closure = every non-consecutive fragment pair**; FPFH + RANSAC (checkers: edge-length 0.9, distance; `RANSACConvergenceCriteria(1000000, 0.999)`) or FGR; accepted if information ratio > 0.3; consecutive pairs seeded by aggregated fragment odometry + multiscale ICP `[voxel_size],[50]`; scene pose graph optimized with `preference_loop_closure_registration`.
- refine_registration: re-registers every posegraph edge with multiscale ICP `[voxel_size, voxel_size/2, voxel_size/4]`, iterations `[50, 30, 14]`, distance `voxel_size*1.4`, method per `icp_method`; then posegraph optimization + writes `scene/trajectory.log`.
- integrate_scene: global TSDF of all frames (`pose = posegraph_fragment[fragment] @ posegraph_rgbd[frame]`), `ScalableTSDFVolume(voxel_length=tsdf_cubic_size/512, sdf_trunc=0.04, RGB8)`, `extract_triangle_mesh()` → `scene/integrated.ply` (binary, compressed PLY), plus `trajectory.log`.

Official docs: https://www.open3d.org/docs/release/tutorial/reconstruction_system/system_overview.html (and /make_fragments.html, /register_fragments.html, /refine_registration.html, /integrate_scene.html, /capture_your_own_dataset.html). Sources: examples/python/reconstruction_system/{run_system,initialize_config,make_fragments,register_fragments,refine_registration,integrate_scene,optimize_posegraph,data_loader}.py, open3d_example.py.

### A2. Config JSON schema — EVERY default (initialize_config.py, authoritative)

```json
{
  "path_dataset": "/abs/path",            // required
  "path_intrinsic": "",                   // optional; default = PrimeSense 640x480 fx=fy=525, cx=319.5, cy=239.5
  "depth_map_type": "redwood",            // default
  "n_frames_per_fragment": 100,
  "n_keyframes_per_n_frame": 5,
  "depth_min": 0.3,                       // NOTE: set but UNUSED by legacy stages (used by slac + tensor engine)
  "depth_max": 3.0,
  "voxel_size": 0.05,                     // FPFH/ICP downsampling voxel (m)
  "depth_diff_max": 0.07,                 // odometry depth correspondence threshold
  "depth_scale": 1000,                    // depth png -> meters (ScanNet mm depth works as-is)
  "preference_loop_closure_odometry": 0.1,   // GlobalOptimizationOption.preference_loop_closure (fragment pg)
  "preference_loop_closure_registration": 5.0, // scene pg
  "tsdf_cubic_size": 3.0,                 // -> voxel_length = tsdf_cubic_size / 512.0
  "icp_method": "color",                  // point_to_point | point_to_plane | color | generalized
  "global_registration": "ransac",        // ransac | fgr
  "python_multi_threading": true
}
```
SLAC block (set defaults): `max_iterations=5`, `sdf_trunc=0.04`, `block_count=40000`, `distance_threshold=0.07`, `fitness_threshold=0.3`, `regularizer_weight=1`, `method="slac"`, `device="CPU:0"`, `save_output_as="pointcloud"`, `folder_slac="slac/"`, `template_optimized_posegraph_slac="optimized_posegraph_slac.json"`, `subfolder_slac="slac/%0.3f/"` (=`slac/0.050/`).
Path templates: `folder_fragment="fragments/"`, `template_fragment_posegraph="fragments/fragment_%03d.json"`, `template_fragment_posegraph_optimized="fragments/fragment_optimized_%03d.json"`, `template_fragment_pointcloud="fragments/fragment_%03d.ply"`, `folder_scene="scene/"`, `template_global_posegraph="scene/global_registration.json"`, `template_global_posegraph_optimized="scene/global_registration_optimized.json"`, `template_refined_posegraph="scene/refined_registration.json"`, `template_refined_posegraph_optimized="scene/refined_registration_optimized.json"`, `template_global_mesh="scene/integrated.ply"`, `template_global_traj="scene/trajectory.log"`.

**tsdf_cubic_size ↔ voxel size**: voxel_length = tsdf_cubic_size/512, documented convention ("3.0/512 = 0.0059 m ≡ 3 m room at 512³"). For 1 cm voxels → `tsdf_cubic_size: 5.12`. `sdf_trunc` is hardcoded 0.04 m in the legacy stages (NOT configurable there; configurable in tensor engine via `trunc_voxel_multiplier`).
Official examples: `config/tutorial.json` (above values, Lounge), `config/realsense.json`. Docs note on `python_multi_threading` (docs claim joblib; source uses multiprocessing).

### A3. Consuming external poses (ZED trajectory)

**No documented way to seed poses into the pose-graph pipeline.** make_fragments always computes its own odometry; register/refine compute their own registration. Open3D never reads external poses except posegraphs it wrote itself.
Documented alternatives:
- `o3d.pipelines.odometry.compute_rgbd_odometry(src_rgbd, tgt_rgbd, intrinsic, init_transformation, Jacobian, option)` — accepts an initial guess (tutorial https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html). You could fork the system and pass ZED relative poses as `odo_init` (small patch in `register_one_rgbd_pair`).
- **Tensor engine integration path (documented)**: `examples/python/t_reconstruction_system/integrate.py --path_trajectory traj.log` (or posegraph `.json`) consumes externally computed poses and does TSDF integration only. Trajectory format = TUM-style `.log`: 5 lines per frame: `"<i> <i> <i+1>"` then 4x4 matrix rows (see `open3d_example.write_poses_to_log`; read via `o3d.io.read_pinhole_camera_trajectory`, extrinsics = `param.extrinsic`). This is the clean documented way to use ZED poses for reconstruction; the pose-graph/loop-closure backend is not externally seedable.
- Hack (undocumented): replace node poses in `fragments/fragment_optimized_%03d.json` before `--integrate`; integrate_scene would then use your poses.

### A4. Input dataset layout

```
path_dataset/
  image/  (or rgb/ or color/)   *.jpg,*.png  (alphanumeric sort, must count-match depth)
  depth/                        *.png (16-bit)
camera_intrinsic.json           (set via "path_intrinsic")
```
`camera_intrinsic.json` exact keys (from cpp/open3d/camera/PinholeCameraIntrinsic.cpp):
```json
{ "width": 640, "height": 480,
  "intrinsic_matrix": [fx, 0, 0, 0, fy, 0, cx, cy, 1] }
```
(9 floats, row-major). Color folder auto-detected among `image/`, `rgb/`, `color/` (`get_rgbd_folders`).
**ScanNet-style support: NO.** Verified current main and v0.8.0 tree: Open3D ships no SUN3D/ScanNet dataloader; the official `data_loader.py` has only lounge/bedroom/jack_jack (via `o3d.data.*`). Docs mention SUN3D/TUM/SceneNN/ICL-NUIM only as "any RGB-D data works". ScanNet `.sens` is not readable; must convert to the redwood layout above (color jpg + uint16 mm depth png + intrinsic json). ScanNet's per-frame 4x4 `pose/*.txt` can be converted to the `.log` format (A3) for the tensor integration path. ScanNet conventions fit: `depth_scale: 1000`, `depth_max` per scene.

### A5. Tensor / GPU variant (open3d.t.pipelines, "Reconstruction system (Tensor)")

- Documented: https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/ (voxel_block_grid, integration, customized_integration, ray_casting, dense_slam) + examples/python/t_reconstruction_system/{integrate.py, dense_slam.py, ...} + API open3d.t.pipelines.{odometry,registration,slac,slam}.
- **CUDA: yes** — `o3d.core.Device("CUDA:0")`; kernel integration ~100 Hz on GTX 1070 (docs). **BUT: official PyPI `open3d` wheels (0.17.0–0.19.0) are CPU-only** (verified wheel filenames, no cuda). CUDA needs build-from-source `-DBUILD_CUDA_MODULE=ON` (https://www.open3d.org/docs/release/compilation.html). RTX A4000 16 GB is ample: VoxelBlockGrid with `block_count=40000` (default) = 40000×16³×20 B ≈ 3.3 GB.
- Tensor default_config.yml (documented parameters): `fragment_size=100, device=CUDA:0, engine=tensor, multiprocessing=false, depth_folder=depth, color_folder=color, path_intrinsic='', path_color_intrinsic='', depth_min=0.1, depth_max=3.0, depth_scale=1000.0, odometry_method=hybrid (point2plane|intensity|hybrid|frame2model), odometry_loop_interval=10, odometry_loop_weight=0.1, odometry_distance_thr=0.07, icp_method=colored, icp_voxelsize=0.05, icp_distance_thr=0.07, global_registration_method=ransac, registration_loop_weight=0.1, integrate_color=true, voxel_size=0.0058, trunc_voxel_multiplier=8.0, block_count=40000, est_point_count=6000000, surface_weight_thr=3.0`.
- VoxelBlockGrid: `attr_names=('tsdf','weight','color')`, block_resolution=16, voxel_size (0.0058≈3/512), save/load `.npz`, `extract_triangle_mesh(weight_threshold)` (marching cubes), `extract_point_cloud`.
- Dense SLAM (dense_slam.py): frame-to-model tracking, Model(voxel_size, 16, block_count, ...), `track_frame_to_model/integrate/synthesize_model_frame`. **Explicit docs caveat**: "not fully optimized for accuracy, no relocalization, no loop closure, room-scale with moderate motion, may fail on challenging sequences" — not suitable as our backend; use `integrate.py` + external poses instead.

### A6. Multi-room / large-scene / 10000 frames guidance

- No explicit multi-room documentation. Legacy system is fragment-based and is the documented large-scene path (`n_fragments = ceil(n_frames/100)`); all-pairs fragment registration is O(n_fragments²) — 11000 frames ≈ 110 fragments ≈ 6k pairs (parallelized).
- SLAC (optional steps 5-6) is the documented large-scene drift correction (rigid+deformable fragment optimization, params in A2).
- VoxelBlockGrid docs: "reserve 50000 blocks for a living-room-scale scene"; "3.0/512" convention.
- Memory scale for our target (10 m extent @ 1 cm): voxel_size 0.0059 → ~1700 voxels/axis capacity; blocks allocated only near surface; block_count 40000 default ≈ 3.3 GB fits A4000. Legacy ScalableTSDFVolume is CPU hashed volume (memory ~GBs, slower).

### A7. Output formats

- `scene/integrated.ply`: binary (write_ascii=False, compressed=True) triangle mesh PLY with **RGB8 per-vertex colors** (TSDF color integration), resolution = `tsdf_cubic_size/512` m (1 cm at 5.12), marching cubes via `extract_triangle_mesh()`, normals computed.
- `scene/trajectory.log`: per-frame 4x4 poses (camera-to-world), 5 lines/frame.
- Fragments: `fragments/fragment_%03d.ply` colored pointclouds (compressed PLY).
- Optional `color_map_optimization_for_reconstruction_system.py` improves vertex colors/geometry.
- Tensor engine: `.npz` voxel grid, `output.ply` mesh or pointcloud via `extract_triangle_mesh(weight_threshold=surface_weight_thr)`.

---

## TOPIC B — pymeshlab as meshlabserver replacement

### B1. Install (no sudo)

- `pip3 install pymeshlab` — official wheels on PyPI: latest **2025.7.post1** (MeshLab 2025.07), `cp310–cp314`, `manylinux_2_35_x86_64` + `aarch64`, macOS, Windows. Docs requirement: 64-bit Python ≥ 3.7 (current wheels 3.10–3.14). Install into a venv or `--user` → no sudo.
- conda-forge: `pymeshlab` 2025.7.post1 available (install into user env, no sudo).
- Docs: https://pymeshlab.readthedocs.io/en/latest/installation.html. Note: pymeshlab is **double precision** (MeshLab builds with "d" suffix are equivalent).

### B2. Exact filter names (current canonical, pymeshlab 2025.7)

Filters were **renamed in pymeshlab 2022.2**; old names deprecated, removed in later releases. Current canonical names (verified in docs filter_list + MeshLab main source `cleanfilter.cpp`/`meshfilter.cpp` `pythonFilterName()`):

| ScanNet .mlx (human) filter | pymeshlab name (current) | Old alias (≤2021.x, removed) | Parameters (current names / values used by ScanNet) |
|---|---|---|---|
| "Merge Close Vertices" | **`meshing_merge_close_vertices`** | `merge_close_vertices` | `threshold` = 0.0010689 (PercentageValue; **used as absolute distance in mesh units**, `getAbsPerc` returns stored float → 1.07 mm) |
| "Remove Duplicate Faces" | **`meshing_remove_duplicate_faces`** | `remove_duplicate_faces` | none |
| "Remove Isolated pieces (wrt Face Num.)" | **`meshing_remove_connected_component_by_face_number`** | `remove_isolated_pieces` | `mincomponentsize` = 1000 (or 7500); `removeunref` = True (default) |
| "Remove Unreferenced Vertices" | **`meshing_remove_unreferenced_vertices`** | `remove_unreferenced_vertices` | none |
| "Simplification: Quadric Edge Collapse Decimation" | **`meshing_decimation_quadric_edge_collapse`** | `simplification_quadric_edge_collapse_decimation` | `targetfacenum`=0, `targetperc`=0.2, `qualitythr`=0.3, `preserveboundary`=False, `boundaryweight`=1, `preservenormal`=False, `preservetopology`=False, `optimalplacement`=True, `planarquadric`=False, `planarweight`=0.001 (unused), `qualityweight`=False, `autoclean`=True, `selected`=False |

Migration helper: `pymeshlab.replace_pymeshlab_filter_names('script.py' or dir)`.
URLs: https://pymeshlab.readthedocs.io/en/latest/filter_list.html (#meshing_merge_close_vertices, #meshing_remove_duplicate_faces, #meshing_remove_connected_component_by_face_number, #meshing_remove_unreferenced_vertices, #meshing_decimation_quadric_edge_collapse); changelog: .../changelog.html (2022.2 entry).

### B3. .mlx script support — YES, unmodified ScanNet scripts work

MeshSet API (https://pymeshlab.readthedocs.io/en/latest/classes/meshset.html):
- `ms.load_filter_script('clean.mlx')` → "Loads from a .mlx file the current filter script."
- `ms.apply_filter_script()` → "Applies all the filters currently present in the filter script."
- also `save_filter_script`, `print_filter_script`, `clear_filter_script`, and low-level `apply_filter(name, **params)`.
Caveats:
1. .mlx contains only filters — mesh I/O stays in pymeshlab: `ms.load_new_mesh('in.ply')` … `ms.save_current_mesh('out.ply')` (the meshlabserver `-m vc` is pymeshlab's default save behavior).
2. ScanNet's mlx uses human filter names ("Merge Close Vertices", …) and camelCase param names ("Threshold", "MinComponentSize", "TargetFaceNum", "TargetPerc", "QualityThr", "PreserveBoundary", "BoundaryWeight", "PreserveNormal", "PreserveTopology", "OptimalPlacement", "PlanarQuadric", "QualityWeight", "AutoClean", "Selected") — all unchanged in current MeshLab (verified in source), so scripts run unmodified. `threshold` semantics (absolute, getAbsPerc) identical to ScanNet-era meshlabserver.
3. Double-precision build → numerically marginally different results than single-precision meshlabserver 2016 (irrelevant at these tolerances).
4. mlx support existed since pymeshlab 0.1.3 (2020-08); pre-2016 XML script formats unsupported (not our case).

### B4. Vertex colors

Preserved end-to-end: PLY vertex colors loaded by default; `meshing_merge_close_vertices` keeps the surviving vertex's color; `meshing_decimation_quadric_edge_collapse` transfers/interpolates vertex color to collapsed vertices (VCG AttributeSeam, `COLOR_PER_VERTEX`, in MeshLab `meshfilter.cpp` quadric path); the three removal filters do not touch colors; `save_current_mesh` writes vertex colors by default. No `-m vc` equivalent needed.

---

## Key URLs
- https://www.open3d.org/docs/release/tutorial/reconstruction_system/system_overview.html (+ make_fragments, register_fragments, refine_registration, integrate_scene, capture_your_own_dataset)
- https://github.com/isl-org/Open3D/tree/main/examples/python/reconstruction_system (run_system.py, initialize_config.py, make_fragments.py, register_fragments.py, refine_registration.py, integrate_scene.py, optimize_posegraph.py, open3d_example.py, config/tutorial.json)
- https://www.open3d.org/docs/release/tutorial/t_reconstruction_system/ (voxel_block_grid, integration, dense_slam) + examples/python/t_reconstruction_system/ (integrate.py, dense_slam.py, default_config.yml, common.py)
- https://www.open3d.org/docs/release/compilation.html (CUDA build)
- https://pymeshlab.readthedocs.io/en/latest/installation.html | index.html (renaming note) | filter_list.html | classes/meshset.html | changelog.html
- https://pypi.org/project/pymeshlab/ (2025.7.post1, cp310-cp314 wheels) | conda-forge pymeshlab
