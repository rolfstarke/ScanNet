# ScanNet Spellbook - Project Status

## Project Goal

Integrate open-vocabulary 3D instance segmentation models (Mosaic3D, OpenIns3D, OpenYOLO3D, Open3DIS, OpenMask3D) into the ScanNet repository structure, run them on a fixed 10-scene official ScanNet validation set, and evaluate them with ScanNet's own benchmark protocol — 18-class NYU40 and ScanNet200 (198-class instance protocol). Comparable ranking uses ScanNet200 mean AP (IoU 0.50:0.95), with AP50 as tie-break. ScanNet itself must remain fully intact and operable.

Second goal: turn ZED X `.svo2` recordings into byte-format-identical ScanNet v2 scans (`.sens`, `<id>.txt`, `_vh_clean*.ply`, `.segs.json`) under custom ids `scene90NN_MM` (9000-range = custom; tens digit = engine, units digit = run), so ScanNet's own tooling works on them unchanged. Scene9004 geometry quality is scored against CAD with mean bidirectional nearest-neighbour distance cm (`geometry_score.py`; lower better). CAD stays fixed; a temporary rigid recon→CAD transform is applied in memory only for scoring/views.

Bugs, problems, and their attempted fixes live in GitHub Issues (`rolfstarke/ScanNet`, referenced as #N) — never in this file. This file is the repository structure explanation, goal, and project plan for new sessions.

---

## Repository Structure

### ScanNet (upstream, first priority)
- `BenchmarkScripts/` — official evaluators, ScanNet200 constants/splits. Python-3 compatibility patches plus the empty-match edge-case fix from #8; metric logic otherwise unchanged.
- `SensReader/python/SensorData.py` — Python-3 ported (.sens reader); logic untouched (see #5).
- `Tasks/Benchmark/scannetv2_val.txt` — official val split list.

### OpenCode skills
- `.opencode/skills/s1-document/` — after-build memory: structure/goals in this file, attempts in GitHub Issues, executed plans in `spellbook/archive/`
- `.opencode/skills/s2-improve/` — six-run reconstruction/prediction improvement; prediction batches use `run.json` + `runs.py rank`
- `.opencode/skills/s3-baseline/` — one untuned official-default reconstruction or prediction baseline for the current worktree method

### spellbook/ (all project code lives here)
```
spellbook/
├── main.py                      # CLI: --visualize, --predict, --engine, --extract-frames, --gpu-check
├── settings.yaml                # default ScanNet200 + scannet_root + gpu_pool ([1,2,3,4]; 0 user-reserved)
├── gpu_check.py                 # --gpu-check: per-model/engine native distribution probe over real pool leases
├── environment.yaml             # 3disspellbook conda env (ZED SDK activation sets ZED_DIR/LD_LIBRARY_PATH)
├── PROJECT_STATUS.md            # this file
├── tests/                        # headless visualizer + mask-cache + prediction-score + run-registry helpers
├── tmp/                          # active scripts only (README); executed plans → archive/
├── evaluation/
│   ├── benchmark.py             # BenchmarkSpec, PREDICTION_EVALUATION_SCENES (10 official val scans), PREDICTION_METHODS
│   ├── evaluate.py              # GT export, run.json-scoped staging evaluator, score-sidecars CLI, TP/GT sidecars
│   ├── runs.py                  # run.json manifests, AP-first ranking, dry-run prediction prune
│   └── scannet200_evaluator.py  # Python-3 port of Rozenberszki ScanNet200 evaluator (198-class)
├── utils/
│   ├── gpu.py                   # cross-process GPU leases (flock per pool index, pass_fds forwarding)
│   ├── scan_lock.py             # frames / scene-engine / per-scan / prediction-index / per-method flock locks
│   ├── visualize.py             # Open3D viewer + scene/scan/method/run tree (best comparable run = AP then AP50)
│   ├── prediction_masks.py      # packed uint8 mask cache under derived/visualization/masks
│   ├── compute_time.py          # reconstruction/prediction elapsed sidecars for the right HUD
│   └── hud.py                   # left tree + right Information/Settings/Classes (CAD overlay)
└── predict/
    ├── runner.py                # one-method dispatch over gpu_pool; official classes only; Open3DIS serialized
    ├── frames.py                # .sens extraction via ScanNet's SensorData exporters (0..N-1), idempotent
    └── models/
        ├── common.py            # decimate(), generation-safe write_scannet_submission, --run-id/--parameters-json
        ├── _mosaic3d_run.py     # point-cloud only
        ├── _openins3d_run.py    # point-cloud only
        ├── _openyolo3d_run.py   # needs frames
        ├── _open3dis_run.py     # needs frames; method lock before GPU lease
        └── _openmask3d_run.py   # needs frames; scratch isolated from submission root
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
├── cleanup.py                # exact prediction/task-line/TP50-sidecar purge on scan eviction/replace
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

### Worktrees

Main checkout on `master`, plus plugin-owned linked worktrees under
`~/.local/share/opencode/worktree/<projectId>/debug/`:

- `debug/reconstruction-{open3d,zed,bundlefusion,metashape,rtabmap,isaac}` — engine debugging;
  tmux `hoenecker` windows 2–7, session titles matching the engine name.
- `debug/prediction-{mosaic3d,openins3d,openyolo3d,open3dis,openmask3d}` — one method each,
  forked from the shared prediction base on `master`; tmux windows 8–12, session titles
  `mosaic3d` / `openins3d` / `openyolo3d` / `open3dis` / `openmask3d`.
- `debug/visualizer` — visualizer feature work; tmux `hoenecker` window 14, session title `visualizer`.

Shared prediction/evaluation core lands on `master` first. Method branches own only wrapper
and method-specific changes. Reconstruction branches stay independent and are not reset by
prediction work.

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
│   ├── evaluations/<Benchmark>/
│   │   ├── ranking.csv                      # derived AP/AP50/AP25 table from run.json + CSVs
│   │   └── <run-id>/                        # run.json, result CSVs, <model>.tasks, <model>/<scene>.tp50.json + .timing.json
│   ├── reconstruction/frames/sceneNNNN/frames/   # shared SVO frame pools
│   ├── visualization/masks/<Benchmark>/<run-id>/<model>/<scene>.npz  # disposable packed 0/1 caches
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
     └── cmdline.txt svo_path.txt timing.json   # wall time of reconstruct+finalize, no CAD
```

### Official prediction evaluation scenes (fixed in `benchmark.py`)
`0019_01, 0217_00, 0304_00, 0412_00, 0414_00, 0426_02, 0488_00, 0549_00, 0568_01, 0575_00`
One scan per physical scene; all members of `Tasks/Benchmark/scannetv2_val.txt`. Custom `scene90xx` scans stay on disk for reconstruction and are rejected by managed prediction/evaluation.

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
5. **Benchmark protocol**: `settings.yaml` selects the default backend (ScanNet20 = official evaluator ported in place to Python 3; ScanNet200 = port of the benchmark author's evaluator, since ScanNet/ScanNet publishes no ScanNet200 instance evaluator). Comparable prediction always passes `--benchmark ScanNet200`, omits `--scene` (full 10-scene tuple) and rejects `--classes`.
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
17. **Worktrees (convention)**: one `debug/reconstruction-<engine>` tree per engine and one `debug/prediction-<method>` tree per prediction method. Shared core stays on `master`. Each reconstruction branch may touch only its engine; each prediction branch may touch only its wrapper. The worktree plugin forks a session and launches a TUI from the tree; relocate via the control-plane move API (`moveChanges=false`), rename to the bare method/engine title, then relaunch. `worktree_delete` works only from the creating session, one tree at a time.
18. **Prediction run registry**: each run writes `derived/evaluations/<Benchmark>/<run-id>/run.json` before GPU work (one method, protocol scenes, provenance). Ranking default is AP then AP50, and comparable ScanNet200 ranks require the exact 10-scene tuple. Run IDs are permanent; evaluated runs are immutable (`evaluate` refuses to replace an existing CSV). TP/GT sidecars bind to prediction-index and GT hashes. Method-improvement losers keep `run.json`, CSVs, tasks, and 10 TP/GT sidecars and drop only heavy prediction artifacts (`runs.py prune`, dry-run default). Open3DIS scene tasks take a method lock before the GPU lease (#15).
19. **Visualizer**: four-level tree `scene → scan → method → run`. Method default prefers a comparable 10-scene ScanNet200 run, then AP, then AP50; unlabeled fallback if none is comparable. Mode is Ground truth / Scene only / Prediction; Color is Classes / Instances / TP/GT. Scan Enter is scene-only; method/run Enter loads that prediction. Tree TP/GT and the right Information block read `.tp50.json` sidecars plus evaluator CSVs; TP/GT coloring is one overlay with TP green over FP red (#38). Official prediction text stays immutable; the viewer may write disposable packed mask caches under `derived/visualization/masks/` and builds boxes/labels only when Boxes is On. Reconstruction and prediction elapsed times come from `recon/timing.json` and `<scene>.timing.json`; missing files hide the HUD lines.

---

## Commands

```bash
# Comparable prediction (omitted --scene = fixed 10-scene tuple; official 198 classes)
python spellbook/main.py --predict --benchmark ScanNet200 --models mosaic3d \
    --run-id baseline-mosaic3d --issue 33

# Ground truth export
python spellbook/evaluation/evaluate.py export-gt --scene 0568_01 --benchmark ScanNet200

# Evaluation (scenes/methods from run.json)
python spellbook/evaluation/evaluate.py evaluate --benchmark ScanNet200 \
    --run-id baseline-mosaic3d --models mosaic3d

# TP/GT sidecars for existing predictions (visualizer tree counts)
python spellbook/evaluation/evaluate.py score-sidecars --benchmark ScanNet200 \
    --run-id baseline-mosaic3d --models mosaic3d --scenes 0568_01 --missing-only

# Rank comparable ScanNet200 runs (AP primary)
python spellbook/evaluation/runs.py rank --benchmark ScanNet200 --method mosaic3d --metric ap

# Visualization (empty 3D until Enter on a scan; left scene/scan/method/run tree)
python spellbook/main.py --visualize

# Optional packed mask caches for the visualizer (derived only; official .txt untouched)
python spellbook/utils/prediction_masks.py build --benchmark ScanNet200 \
    --run-id baseline-openyolo3d --models openyolo3d --scenes 0568_01 --missing-only

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

1. Finish restoring official `scene0019_01` and `scene0414_00` (mesh, segs, `.sens`, GT, frames), then run fresh `baseline-<method>` predictions (#33–#37).
2. One six-run improvement batch per method in its prediction worktree (#33 Mosaic3D / #10, #34 OpenIns3D / #13, #35 OpenYOLO3D / #2, #36 Open3DIS / #3 #15 #24, #37 OpenMask3D).
3. Open3DIS tracker files remain globally stateful; scene tasks stay serialized until a run-scoped fix lands (#15).
4. Engine debugging stays in reconstruction worktrees. Score scene9004 via `geometry_score`. ZED blocked (#29). Isaac image/CDI (#22). Multiroom quality (#25). Optional 4 mm TSDF (#31).
5. Do not expand prediction evaluation to the full 312-scene val split; the comparable set is the fixed 10.
