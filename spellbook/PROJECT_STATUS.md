# ScanNet Spellbook - Project Status

## Project Goal

Integrate open-vocabulary 3D instance segmentation models (Mosaic3D, OpenIns3D, OpenYOLO3D, Open3DIS) into the ScanNet repository structure, run them on official ScanNet val scenes, and evaluate them with ScanNet's own benchmark protocol — 18-class NYU40 and ScanNet200 (198-class instance protocol). ScanNet itself must remain fully intact and operable.

Second goal: turn ZED X `.svo2` recordings into byte-format-identical ScanNet v2 scans (`.sens`, `<id>.txt`, `_vh_clean*.ply`, `.segs.json`) under custom ids `scene90NN_MM` (9000-range = custom; tens digit = engine, units digit = run), so ScanNet's own tooling works on them unchanged. Scene9004 geometry quality is scored against CAD with mean bidirectional nearest-neighbour distance cm (`geometry_score.py`; lower better). CAD stays fixed; a temporary rigid recon→CAD transform is applied in memory only for scoring/views.

Bugs, problems, and their attempted fixes live in GitHub Issues (`rolfstarke/ScanNet`, referenced as #N) — never in this file. This file is the repository structure explanation, goal, and project plan for new sessions.

---

## Repository Structure

### ScanNet (upstream, first priority)
- `BenchmarkScripts/` — official evaluators, ScanNet200 constants/splits. Python-3 compatibility patches plus the empty-match edge-case fix from #8; metric logic otherwise unchanged.
- `SensReader/python/SensorData.py` — Python-3 ported (.sens reader); logic untouched (see #5).
- `Tasks/Benchmark/scannetv2_val.txt` — official val split list.

### spellbook/ (all project code lives here)
```
spellbook/
├── main.py                      # CLI: --visualize, --predict, --engine, --extract-frames, --gpu-check
├── settings.yaml                # default benchmark + scannet_root + gpu_pool ([1,2,3,4]; 0 user-reserved)
├── gpu_check.py                 # --gpu-check: per-model/engine native distribution probe over real pool leases
├── environment.yaml             # 3disspellbook conda env (ZED SDK activation sets ZED_DIR/LD_LIBRARY_PATH)
├── PROJECT_STATUS.md            # this file
├── tmp/                          # active scripts only (README); executed plans → archive/
├── evaluation/
│   ├── benchmark.py             # BenchmarkSpec (ScanNet20 18 / ScanNet200 198), paths, gpu_pool validation
│   ├── evaluate.py              # GT export + evaluation dispatch (official / scannet200)
│   └── scannet200_evaluator.py  # Python-3 port of Rozenberszki ScanNet200 evaluator (198-class)
├── utils/
│   ├── gpu.py                   # cross-process GPU leases (flock per pool index, pass_fds forwarding)
│   ├── scan_lock.py             # frames / scene-engine / per-scan / prediction-index flock locks
│   ├── visualize.py             # Open3D viewer + ImGui legend (GT + official submission predictions)
│   └── hud.py
└── predict/
    ├── runner.py                # model→env dispatch over automatic gpu_pool leases; sequential up-front frame extraction
    ├── frames.py                # .sens extraction via ScanNet's SensorData exporters (0..N-1), idempotent
    └── models/
        ├── common.py            # decimate(), write_scannet_submission (official submission layout)
        ├── _mosaic3d_run.py     # point-cloud only
        ├── _openins3d_run.py    # point-cloud only; --detector {odise,yoloworld}
        ├── _openyolo3d_run.py   # needs frames
        └── _open3dis_run.py     # needs frames; forwards the lease fd to its nested subprocess
```

### spellbook/reconstruct/ (SVO2 -> ScanNet-native scans; invoked via `main.py --engine`)
```
spellbook/reconstruct/
├── batch.py                  # multi-scene x multi-engine: shared frames then parallel run.py children
├── run.py                    # allocate run slot -> reconstruct -> score scene9004 (post-GPU)
├── extract.py                # main-checkout multi-GPU SVO extraction into shared frame pools
├── _extract_run.py           # leased-GPU child: CUDA_VISIBLE_DEVICES set before pyzed import
├── geometry_score.py         # CAD mean bidirectional distance cm + density overlay
├── geometry_reference.yaml   # scene9004 CAD/visibility hashes, grid, retention
├── cleanup.py                # exact prediction/task-line purge on scan eviction/replace
├── tsdf.py                   # shared Open3D TSDF integration
├── scannet.py                # gravity z-up align, pymeshlab clean, axisAlignment, Segmentator
├── finalize.py               # .sens v4 + <id>.txt writers
└── engines/
    ├── __init__.py           # adapter contract: GPU_POLICY (managed/cpu/blocked), SERIAL, preflight(), gpu_check(), reconstruct(work, root, gpu, lease_fd)
    ├── zed.py                # BLOCKED (#29): SDK default-device path would use user-reserved GPU 0
    ├── open3d.py             # Stage-A poses + shared TSDF (GPU_POLICY cpu)
    ├── metashape.py          # Metashape Pro (managed GPU, serial, license)
    ├── rtabmap.py            # RTAB-Map in podman (managed GPU, reprocess + loop closures)
    ├── isaac.py              # cuVSLAM + Nvblox in podman (managed GPU, serial) — see #22
    └── bundlefusion.py       # ScanNet's reference engine in docker (managed GPU, canonical 10 mm)
```

### Worktrees (parallel engine debugging + prediction layout)

Eight git worktrees total: the main checkout on `master`, plus seven plugin-owned linked
worktrees under `~/.local/share/opencode/worktree/<projectId>/debug/`:

- `debug/reconstruction-{open3d,zed,bundlefusion,metashape,rtabmap,isaac}` — engine debugging;
  all fast-forwarded to current `master` (same SHA as main checkout), one opencode session per
  engine in tmux windows `hoenecker` 2-7.
- `debug/prediction` — prediction work; same `master` tip, session `prediction` in tmux window 8
  (fork of the archived `main-prediction` session, which stays in the main checkout).

Worktree sessions keep bare titles and their stored directory is the worktree (relocated via
opencode's control-plane move API after the plugin fork). Before debugging, confirm
`git rev-parse --short HEAD` matches main and that `spellbook/evaluation/` exists. Restart the
TUI after a fast-forward so tools see new paths.

### Data Layout (`/data/scannet/` — scans/ stays official, artifacts outside)
```
/data/scannet/
├── scans/<scene_id>/            # official ScanNet data, unchanged
│   ├── sceneXXXX_YY_vh_clean_2.ply          # mesh used as model input + GT vertex basis
│   ├── sceneXXXX_YY.aggregation.json        # GT instance annotations
│   ├── sceneXXXX_YY_vh_clean_2.0.010000.segs.json
│   ├── sceneXXXX_YY.sens                    # RGB-D sensor data
│   ├── frames/{color,depth,pose}/{0..N-1}.{jpg,png,txt}
│   └── frames/intrinsic_{color,depth}.txt + extrinsic_{color,depth}.txt
├── predictions/<Benchmark>/<run-id>/<model>/   # official submission root (zippable as-is)
│   ├── sceneXXXX_YY.txt                       # "predicted_masks/<scene>_NNN.txt <label> <conf>"
│   └── predicted_masks/<scene>_NNN.txt
├── custom/
│   ├── raw/scene900X.svo2                   # SVO symlinks (9004-9007, 9009)
│   └── reference/                           # scene9004 CAD PLY + visible-voxel NPZ
├── derived/
│   ├── ground_truth/<Benchmark>/<scene>.txt
│   ├── evaluations/<Benchmark>/<run-id>/    # result CSVs + <model>.tasks
│   ├── reconstruction/frames/sceneNNNN/frames/   # shared SVO frame pools
│   └── locks/{gpus,reconstruction,scans}/  # GPU + scan/frame flock leases
└── v2/scannetv2-labels.combined.tsv
```

### Custom scans (9000-range)
```
/data/scannet/scans/scene9004_40/            # tens=engine (0 zed … 4 open3d … 5 bundlefusion),
│   *.sens/.txt/_vh_clean*.ply/.segs.json    # units=run 0-9; scene9004 keeps ten slots/engine
└── recon/
    ├── frames -> derived/.../scene9004/frames   # symlink to shared pool
    ├── geometry_score.yaml + cad_comparison.png   # score report: plan/elev/isometric
    ├── engine_native.ply mesh_aligned.ply final_poses.npy
    └── cmdline.txt svo_path.txt
```

### Scenes (20 official val scenes, ~2.5GB each)
`0568_00/01/02, 0304_00, 0488_00/01, 0412_00/01, 0217_00, 0019_00/01, 0414_00, 0575_00/01/02, 0426_00/01/02/03, 0549_00`

---

## Model Integration

| Model | Input | Repo | Env | Notes |
|---|---|---|---|---|
| Mosaic3D | point cloud | `/home/rolf/GIT/Mosaic3D` | `/data/mosaic3d/conda/envs/mosaic3d` | lower-bound reference, see #10 |
| OpenIns3D | point cloud | `/home/rolf/GIT/OpenIns3D` | `/data/openins3d/conda/envs/openins3d` | ODISE detector (paper's), see #4 |
| OpenYOLO3D | frames | `/home/rolf/GIT/OpenYOLO3D` | `/data/openyolo3D/conda/envs/openyolo3d` | needs LD_LIBRARY_PATH hook (runner.py) |
| Open3DIS | frames | `/home/rolf/GIT/Open3DIS` (patched checkout) | `/data/open3dis/conda/envs/open3dis` | img_dim=depth res, rgb_img_dim=color res |
| OpenMask3D | frames | `/home/rolf/GIT/openmask3d` | `/home/rolf/anaconda3/envs/openmask3d` | scannet200_model.ckpt; OPENMASK3D_FORCE_GPU=0 pins the repo's get_free_gpu to the visible device |

Per-model integration issues: #2 (openyolo3d intrinsics), #3 (open3dis rescale), #4 (openins3d demo path), #10 (mosaic3d), #13 (openins3d recall); all five models pass `--gpu-check` distribution probes.

---

## Reconstruction Engines

| Engine | Pose source | Mesh | Runtime |
|---|---|---|---|
| zed | ZED SDK `.area` two-pass tracking | shared Open3D TSDF, 2 cm voxel (deviation) | **blocked** — SDK default-device path would use user-reserved GPU 0 (#29) |
| open3d | Stage A ZED tracking poses | shared Open3D TSDF, 2 cm voxel (deviation) | pyzed 5.4 |
| bundlefusion | BundleFusion global BA (docker) | BundleFusion 10 mm TSDF (canonical public stage; proprietary 4 mm improve stage unavailable) | docker `bundlefusion:latest` |
| metashape | Metashape SfM + BA (keyframe priors) | Metashape depth maps | `/data/zed-metashape` (node-locked, serialized) |
| rtabmap | RTAB-Map graph-optimized | RTAB-Map textured mesh | podman `localhost/zed-rtabmap:jazzy` |
| isaac | cuVSLAM | Nvblox 4 mm TSDF | podman `zed-isaac-nvblox:spellbook` (see #22) |

Zed/open3d are local-pose baselines; multiroom drift and density remain tracked in #25. BundleFusion, Metashape, RTAB-Map, and Isaac provide the global-optimization comparisons. Scene9004 scores live in `recon/geometry_score.yaml` (`mean_bidirectional_distance_cm`, lower better) + density overlay `recon/cad_comparison.png`.

---

## Key Technical Decisions

1. **Native model pipelines**: call each model's own ScanNet-capable API, not ov3dis-comparison wrappers.
2. **Canonical frame pool**: full-density extraction to `frames/` (sequential 0..N-1 names, 4 native intrinsic files); models subsample via their own configs. ScanNet's SensReader `export_*` would keep gapped indices — rejected (#5).
3. **Prediction output**: official ScanNet submission layout, one directory per (benchmark, run, model): `/data/scannet/predictions/<Benchmark>/<run-id>/<model>/` with `<scene>.txt` + `predicted_masks/<scene>_NNN.txt`. Directly zippable as a benchmark submission; runs never overwrite each other (#16).
4. **Label ids**: real NYU40 ids (ScanNet20) resp. raw `id`-column ids (ScanNet200, 198 classes = 200 minus wall/floor) from ScanNet's own constants, derived in `benchmark.py` as the single source of truth; unknown class names raise (see #1, #12).
5. **Benchmark protocol**: `settings.yaml` selects the default backend (ScanNet20 = official evaluator ported in place to Python 3; ScanNet200 = port of the benchmark author's evaluator, since ScanNet/ScanNet publishes no ScanNet200 instance evaluator). `--benchmark` overrides; `--classes` is for custom (non-benchmark) prediction only.
6. **GPU scheduling (automatic)**: `settings.yaml gpu_pool` (`[1,2,3,4]`) is the only managed GPU list; `utils/gpu.py` leases a pool GPU via persistent `flock` lock files under `<scannet_root>/derived/locks/gpus/`, held for the whole task (descriptor forwarded to GPU children with `pass_fds`, kernel-released on crash). **Physical GPU 0 is user-reserved — Spellbook never locks, selects, or initializes it.** No manual `--gpu` exists anywhere; Open3D's legacy TSDF is CPU-only and takes no lease. Frame extraction uses the same pool via `--extract-frames`. The ZED reconstruction engine remains blocked until remapping is validated (#29).
7. **Adapter metadata**: each engine adapter declares `GPU_POLICY` (`managed`/`cpu`/`blocked`), `SERIAL`, `preflight()` (runtime presence check) and `gpu_check()` (native distribution probe); `batch.py` and `--gpu-check` consume these instead of duplicated constants.
8. **Evaluation**: flat per-vertex GT encoding via `evaluate.py export-gt` (ScanNet's own export tool is inconsistent with its evaluator, see #8); evaluators ported to Python 3 with edge-case fix (#8); `evaluate.py evaluate` dispatches per benchmark with pre-flight validation (#11).
9. **Custom scan ids**: `scene90NN_MM` (9000-range unused by ScanNet v2); tens digit = engine, units digit = run 0-9 (scene9004 keeps ten slots per engine with highest-`mean_bidirectional_distance_cm` eviction; other scenes replace run 0). SVO discovery by filename (`custom/raw/scene<NNNN>.svo2`).
10. **Byte fidelity**: `.sens` v4 per `SensReader/c++/src/sensorData.h` (jpeg + zlib_ushort, depth_shift 1000); `<id>.txt` with the ScanNet 17 keys; `.segs.json` from ScanNet's own built Segmentator (defaults kThresh 0.01, segMinVerts 20).
11. **Meshes**: ScanNet's `clean.mlx` filter chain via pymeshlab (equivalent calls — this pymeshlab cannot load `.mlx`), decimation `targetfacenum=300000` (absolute, not ScanNet's relative 20% — deviation); PLY written float32 binary (Segmentator's tinyply rejects double).
12. **axisAlignment**: computed (pure z-rotation + translation, det == 1) and written to `<id>.txt`, NEVER applied — released ScanNet meshes and `.sens` poses share the raw frame (consumers apply it).
13. **Depth/pose conventions**: ZED X native resolution (1920x1080 / 1920x1200 per recording); depth 0.1-6.0 m, invalid = 0, trailing SVO frame dropped; camera-to-world poses conjugated `diag(1,-1,-1,1)` from ZED's z-backward basis; gravity z-up alignment per `alignment.h`; the batch extracts each SVO once (shared frames, `--replace` to regenerate) and runs engine tasks in parallel subprocesses.
14. **Geometry score (scene9004)**: mean bidirectional nearest-neighbour distance on 10 mm surface voxels vs visible CAD (`mean_bidirectional_distance_cm` = average of recon→CAD and CAD→recon means, cm, lower better). CAD original orientation is fixed; temporary rigid ICP moves an in-memory recon copy into CAD frame only (never saved). Report PNG `cad_comparison.png`: plan | elev occupancy (magenta=CAD, cyan=recon) plus full-width isometric recon coloured by distance to CAD (magenta close → cyan far); ceiling height-hide floor+2.5 m for display only. No PASS/FAIL gate.
15. **Frame extraction**: main checkout only via `--extract-frames`; multi-GPU leases on pool 1-4; `sdk_gpu_id` never set; scene9004 frames are promoted from the existing complete set, not re-extracted. Worktrees consume the shared pool read-only.
16. **Foreign-environment imports**: model subprocesses load `spellbook/evaluation/benchmark.py` by absolute `importlib` spec. Adding `spellbook/` to `sys.path` shadows model repositories' top-level packages such as OpenIns3D's `utils`.
17. **Worktrees (convention)**: engine debugging happens one `debug/reconstruction-<engine>` branch+worktree per engine; prediction work uses `debug/prediction`. Seven linked worktrees under `~/.local/share/opencode/worktree/<projectId>/debug/` stay fast-forwarded to `master` before each debug session (`git merge --ff-only master` in each clean tree). Each branch may touch only its own scope; shared core files stay frozen on master. The worktree plugin forks a session (recorded against main) and launches a TUI from the tree; the fork must be relocated via OpenCode's control-plane move API (`moveChanges=false`) and renamed, then relaunched, so its tools and footer bind to the tree. `worktree_delete` works only from the session that created the tree and always adds a `chore(worktree): session snapshot` commit (tree must be clean first); the plugin holds one project-wide pending delete, so trees close strictly one at a time with `git worktree list` + plugin DB verification. Debugging is manual; integration merges `--no-ff` per branch.

---

## Commands

```bash
# Prediction (default benchmark from settings.yaml; --benchmark overrides)
python spellbook/main.py --predict --scene 0568_00 0304_00 --models mosaic3d,openins3d,openyolo3d,open3dis \
    --benchmark ScanNet20 --run-id myrun        # classes default to the benchmark's official list

# Ground truth export (all 20 scenes done; re-run after adding scenes)
python spellbook/evaluation/evaluate.py export-gt --scene 0568_00 --benchmark ScanNet20|ScanNet200

# Evaluation (predictions must exist under predictions/<Benchmark>/<run-id>/<model>/)
python spellbook/evaluation/evaluate.py evaluate --run-id myrun --models mosaic3d,open3dis \
    --scenes 0568_00,0304_00,... --benchmark ScanNet20|ScanNet200

# Visualization (opens GT; arrow keys select benchmark, compatible run/model, and mode)
python spellbook/main.py --visualize --scene 0568_00

# Shared frame extraction (main checkout only; multi-GPU; never from engine worktrees)
python spellbook/main.py --extract-frames --scene 9004 9009

# Reconstruction batch (shared frames, then parallel engines; logs under spellbook/tmp/logs/)
python spellbook/main.py --scene 9004 9009 --engine open3d metashape rtabmap

# Single (scene, engine); scene9004 auto-allocates run slot and writes geometry_score
python -m spellbook.reconstruct.run --scene 9004 --engine open3d

# Standalone geometry score
python -m spellbook.reconstruct.geometry_score --scan-id scene9004_40

# GPU distribution smoke
python spellbook/main.py --gpu-check
```

Class lists: derived in `spellbook/evaluation/benchmark.py` from `BenchmarkScripts/ScanNet200/scannet200_constants.py` — ScanNet20: 20 minus wall/floor = 18 NYU40 ids; ScanNet200: 200 minus ids {1,3} = 198 raw ids.

---

## Dependencies

- `3disspellbook` conda env (visualization, orchestration, eval) — `spellbook/environment.yaml` + pypng; **activate it before any ZED SDK work** (its `activate.d/zed_sdk_env.sh` sets `ZED_DIR` and `LD_LIBRARY_PATH`; calling the env's python directly makes `zed.open` fail with CALIBRATION FILE NOT AVAILABLE)
- Per-model conda envs (see table); OpenIns3D env additionally has detectron2 0.6 + ODISE (see #14)
- Label mapping: `/data/scannet/v2/scannetv2-labels.combined.tsv`
- GitHub CLI `gh` (issue workflow), `~/.local/bin/gh`

---

## Current Plan / Next Steps

1. Investigate OpenIns3D's ScanNet200 collapse / anomaly scenes — #13.
2. Hardening: atomic/resumable prediction outputs #16, batch supervision #17, Open3DIS tracker race #15, env reproducibility #14.
3. Optional: extend from 20 to the full 312-scene val split once hardening is in place.
4. Engine debugging (manual, per engine): worktrees consume shared frames from main; score scene9004 via `geometry_score`. Verify remaining engines (metashape, rtabmap, isaac #22) against #25; bundlefusion verified 2026-08-27 — untuned canonical 10 mm baseline is mechanically valid, quality gaps tracked in #25. Frames for 9004/9009 are in the shared pool.
5. ZED engine still blocked (#29); frame extraction multi-GPU path is live via `--extract-frames`. Validate remapping for the zed reconstruction engine itself.
6. Isaac: build `zed-isaac-nvblox:spellbook` from public Isaac debs (no NGC credentials) and switch off the broken CDI flag — #22.
7. Optional: 4 mm re-integration needs a working CUDA Open3D build (tensor VoxelBlockGrid broken in the installed 0.19; legacy volume at 4 mm hits ~185 GB RSS) — #31.
8. Integration: after each engine tree is closed (plugin `worktree_delete`, one at a time), merge `--no-ff debug/reconstruction-<engine>` into master and run the full batch without GPU arguments.
