# ScanNet Spellbook - Project Status

## Project Goal

Integrate open-vocabulary 3D instance segmentation models (Mosaic3D, OpenIns3D, OpenYOLO3D, Open3DIS) into the ScanNet repository structure, run them on official ScanNet val scenes, and evaluate them with ScanNet's own benchmark protocol — 18-class NYU40 and ScanNet200 (198-class instance protocol). ScanNet itself must remain fully intact and operable.

Second goal: turn ZED X `.svo2` recordings into byte-format-identical ScanNet v2 scans (`.sens`, `<id>.txt`, `_vh_clean*.ply`, `.segs.json`) under custom ids `scene90NN_MM` (9000-range = custom; `_MM` = engine index), so ScanNet's own tooling (SensReader, Segmentator, benchmark, annotation) works on them unchanged. Quality is gated by QC metrics (`qc.py`) with bars measured from real ScanNet scenes.

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
├── main.py                      # CLI: --visualize, --predict (--benchmark, --run-id, multi-scene), --engine
├── settings.yaml                # default benchmark + scannet_root + gpu_pool ([1,2,3,4]; 0 = ZED, never managed)
├── benchmark.py                 # BenchmarkSpec (ScanNet20: 18 classes / ScanNet200: 198 = 200 - wall/floor), paths, gpu_pool validation
├── evaluate.py                  # GT export + evaluation dispatch (official / scannet200 evaluator)
├── scannet200_evaluator.py      # Python-3 port of Rozenberszki's ScanNet200 evaluator (198-class)
├── gpu_check.py                 # --gpu-check: per-model/engine native distribution probe over real pool leases
├── environment.yaml             # 3disspellbook conda env (ZED SDK activation sets ZED_DIR/LD_LIBRARY_PATH)
├── PROJECT_STATUS.md            # this file
├── tmp/                          # active plans/research and transient logs
├── utils/
│   ├── gpu.py                   # cross-process GPU leases (flock per pool index, pass_fds forwarding)
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
├── batch.py                  # multi-scene x multi-engine runner: sequential shared frame extraction, then
│                             #   parallel per-task subprocesses (each run.py child leases its own GPU), tqdm bars
├── run.py                    # single (scene, engine) pipeline: frames -> engine -> align -> meshes
│                             #   -> sens/txt -> segmentator -> qc; acquires the GPU lease per GPU_POLICY
├── extract.py                # SVO2 -> frames/ (uint16 mm depth, pose_state.txt, gravity, intrinsics);
│                             #   ensure_frames(): reuse complete sets, --replace re-extracts
├── tsdf.py                   # shared Open3D TSDF integration (integrate_tsdf), ZED-basis conjugation
├── scannet.py                # gravity z-up align (alignment.h), pymeshlab clean chain + decimation,
│                             #   float32 binary PLY writer, axisAlignment, Segmentator wrapper
├── finalize.py               # .sens v4 + <id>.txt writers
├── qc.py                     # 13 metrics vs measured ScanNet bars -> recon/qc.yaml
├── scannet_reference.yaml    # QC bars measured from real ScanNet scenes
└── engines/
    ├── __init__.py           # adapter contract: GPU_POLICY (managed/cpu/blocked), SERIAL, preflight(), gpu_check(), reconstruct(work, root, gpu, lease_fd)
    ├── zed.py                # BLOCKED (#29): SDK default-device path would use user-reserved GPU 0
    ├── open3d.py             # Stage-A poses + shared TSDF (GPU_POLICY cpu)
    ├── metashape.py          # Metashape Pro (managed GPU, serial, license)
    ├── rtabmap.py            # RTAB-Map in podman (managed GPU, reprocess + loop closures)
    ├── isaac.py              # cuVSLAM + Nvblox in podman (managed GPU, serial) — see #22
    └── bundlefusion.py       # ScanNet's reference engine in docker (managed GPU, 4 mm)
```

### Engine worktrees (parallel engine debugging layout)

Seven git worktrees total: the main checkout on `master`, plus six linked worktrees under
`~/.local/share/opencode/worktree/<projectId>/debug/reconstruction-<engine>` for
`open3d, zed, bundlefusion, metashape, rtabmap, isaac` (branches `debug/reconstruction-*`,
all created from `master` `5a3fa9e`). Each engine runs as its own opencode session in its own
tmux window (`hoenecker` 2-7); sessions keep bare engine names and their stored directory is
the worktree (relocated via opencode's control-plane move API after the plugin fork).

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
├── derived/
│   ├── ground_truth/<Benchmark>/<scene>.txt   # flat per-vertex label*1000+instance
│   ├── evaluations/<Benchmark>/<run-id>/      # result CSVs + <model>.tasks completion markers
│   └── legacy/                                # pre-migration results (18-class valid, 189-class invalid per #12)
└── v2/scannetv2-labels.combined.tsv           # label map (raw_category -> nyu40id | id)
```

### Custom scans (9000-range; SVO symlinks in `/data/scannet/custom/raw/<scene>.svo2`)
```
/data/scannet/scans/scene9009_00/             # _00 zed, _01 metashape, _02 rtabmap, _03 isaac,
│   scene9009_00.sens/.txt/_vh_clean*.ply/    #   _04 open3d, _05 bundlefusion
│   _vh_clean_2.0.010000.segs.json
└── recon/                                    # non-ScanNet extras: frames (symlink), qc.yaml,
    ├── engine_native.ply mesh_aligned.ply    #   cmdline.txt, svo_path.txt, final_poses.npy
    └── logs/                                 #   per-task logs
/data/scannet/derived/reconstruction/frames/<scene>/   # shared per-scene extraction (once per SVO)
/data/scannet/custom/raw/scene900X.svo2                # symlinks to recordings (9004-9007, 9009)
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
| bundlefusion | BundleFusion global BA (docker) | BundleFusion 4 mm TSDF | docker `bundlefusion:latest` |
| metashape | Metashape SfM + BA (keyframe priors) | Metashape depth maps | `/data/zed-metashape` (node-locked, serialized) |
| rtabmap | RTAB-Map graph-optimized | RTAB-Map textured mesh | podman `localhost/zed-rtabmap:jazzy` |
| isaac | cuVSLAM | Nvblox 4 mm TSDF | podman `zed-isaac-nvblox:spellbook` (see #22) |

Zed/open3d are local-pose baselines; multiroom drift and density remain tracked in #25. BundleFusion, Metashape, RTAB-Map, and Isaac provide the global-optimization comparisons; per-scan measurements live only in `recon/qc.yaml` and issues.

---

## Key Technical Decisions

1. **Native model pipelines**: call each model's own ScanNet-capable API, not ov3dis-comparison wrappers.
2. **Canonical frame pool**: full-density extraction to `frames/` (sequential 0..N-1 names, 4 native intrinsic files); models subsample via their own configs. ScanNet's SensReader `export_*` would keep gapped indices — rejected (#5).
3. **Prediction output**: official ScanNet submission layout, one directory per (benchmark, run, model): `/data/scannet/predictions/<Benchmark>/<run-id>/<model>/` with `<scene>.txt` + `predicted_masks/<scene>_NNN.txt`. Directly zippable as a benchmark submission; runs never overwrite each other (#16).
4. **Label ids**: real NYU40 ids (ScanNet20) resp. raw `id`-column ids (ScanNet200, 198 classes = 200 minus wall/floor) from ScanNet's own constants, derived in `benchmark.py` as the single source of truth; unknown class names raise (see #1, #12).
5. **Benchmark protocol**: `settings.yaml` selects the default backend (ScanNet20 = official evaluator ported in place to Python 3; ScanNet200 = port of the benchmark author's evaluator, since ScanNet/ScanNet publishes no ScanNet200 instance evaluator). `--benchmark` overrides; `--classes` is for custom (non-benchmark) prediction only.
6. **GPU scheduling (automatic)**: `settings.yaml gpu_pool` (`[1,2,3,4]`) is the only managed GPU list; `utils/gpu.py` leases a pool GPU via persistent `flock` lock files under `<scannet_root>/derived/locks/gpus/`, held for the whole task (descriptor forwarded to GPU children with `pass_fds`, kernel-released on crash). **Physical GPU 0 is user-reserved — Spellbook never locks, selects, or initializes it.** No manual `--gpu` exists anywhere; Open3D's legacy TSDF is CPU-only and takes no lease. ZED is disabled because its SDK default-device path would use GPU 0 (#18, #29).
7. **Adapter metadata**: each engine adapter declares `GPU_POLICY` (`managed`/`cpu`/`blocked`), `SERIAL`, `preflight()` (runtime presence check) and `gpu_check()` (native distribution probe); `batch.py` and `--gpu-check` consume these instead of duplicated constants.
8. **Evaluation**: flat per-vertex GT encoding via `evaluate.py export-gt` (ScanNet's own export tool is inconsistent with its evaluator, see #8); evaluators ported to Python 3 with edge-case fix (#8); `evaluate.py evaluate` dispatches per benchmark with pre-flight validation (#11).
9. **Custom scan ids**: `scene90NN_MM` (9000-range unused by ScanNet v2); engine index per the table above; SVO discovery by filename convention (`custom/raw/scene<NNNN>.svo2`), `--svo` override.
10. **Byte fidelity**: `.sens` v4 per `SensReader/c++/src/sensorData.h` (jpeg + zlib_ushort, depth_shift 1000); `<id>.txt` with the ScanNet 17 keys; `.segs.json` from ScanNet's own built Segmentator (defaults kThresh 0.01, segMinVerts 20).
11. **Meshes**: ScanNet's `clean.mlx` filter chain via pymeshlab (equivalent calls — this pymeshlab cannot load `.mlx`), decimation `targetfacenum=300000` (absolute, not ScanNet's relative 20% — deviation); PLY written float32 binary (Segmentator's tinyply rejects double).
12. **axisAlignment**: computed (pure z-rotation + translation, det == 1) and written to `<id>.txt`, NEVER applied — released ScanNet meshes and `.sens` poses share the raw frame (consumers apply it).
13. **Depth/pose conventions**: ZED X native resolution (1920x1080 / 1920x1200 per recording); depth 0.1-6.0 m, invalid = 0, trailing SVO frame dropped; camera-to-world poses conjugated `diag(1,-1,-1,1)` from ZED's z-backward basis; gravity z-up alignment per `alignment.h`; the batch extracts each SVO once (shared frames, `--replace` to regenerate) and runs engine tasks in parallel subprocesses.
14. **QC**: 13 metrics vs bars measured from real ScanNet scenes (`scannet_reference.yaml`); PASS/FAIL written to `recon/qc.yaml`, FAIL reported, never aborts.
15. **Foreign-environment imports**: model subprocesses load `spellbook/benchmark.py` by absolute `importlib` spec. Adding `spellbook/` to `sys.path` shadows model repositories' top-level packages such as OpenIns3D's `utils`.
16. **Engine worktrees (convention)**: engine debugging happens one `debug/reconstruction-<engine>` branch+worktree per engine. Six linked worktrees exist from master `5a3fa9e` (2026-08-22) for open3d, zed, bundlefusion, metashape, rtabmap, isaac under `~/.local/share/opencode/worktree/<projectId>/debug/reconstruction-<engine>`; each branch may touch only its adapter; shared core files stay frozen. The worktree plugin forks a session (recorded against main) and launches a TUI from the tree; the fork must be relocated via OpenCode's control-plane move API (`moveChanges=false`) and renamed, then relaunched, so its tools and footer bind to the tree. `worktree_delete` works only from the session that created the tree and always adds a `chore(worktree): session snapshot` commit (tree must be clean first); the plugin holds one project-wide pending delete, so trees close strictly one at a time with `git worktree list` + plugin DB verification. Engine debugging is manual; integration merges `--no-ff` per engine branch.

---

## Commands

```bash
# Prediction (default benchmark from settings.yaml; --benchmark overrides)
python spellbook/main.py --predict --scene 0568_00 0304_00 --models mosaic3d,openins3d,openyolo3d,open3dis \
    --benchmark ScanNet20 --run-id myrun        # classes default to the benchmark's official list

# Ground truth export (all 20 scenes done; re-run after adding scenes)
python spellbook/evaluate.py export-gt --scene 0568_00 --benchmark ScanNet20|ScanNet200

# Evaluation (predictions must exist under predictions/<Benchmark>/<run-id>/<model>/)
python spellbook/evaluate.py evaluate --run-id myrun --models mosaic3d,open3dis \
    --scenes 0568_00,0304_00,... --benchmark ScanNet20|ScanNet200

# Visualization (opens GT; arrow keys select benchmark, compatible run/model, and mode)
python spellbook/main.py --visualize --scene 0568_00

# Reconstruction batch (frames extracted once per scene, then tasks automatically share
# the settings gpu_pool GPUs 1-4 via cross-process leases; GPU 0 stays unmanaged for ZED
# SDK extraction/tracking. Terminal shows only tqdm bars; logs under spellbook/tmp/logs/reconstruct-<run-id>/)
python spellbook/main.py --scene 9004 9009 --engine metashape isaac bundlefusion open3d zed rtabmap \
                                        # add --replace to re-extract frames

# Single (scene, engine); GPU is leased automatically for managed engines
python -m spellbook.reconstruct.run --scene 9009 --engine bundlefusion

# QC re-run for one scan
python -m spellbook.reconstruct.qc scene9009_04

# GPU distribution smoke (no results): launches every prediction method + reconstruction
# engine under the real pool (1-4; GPU 0 user-reserved), proves assignment, holds briefly.
# Expected on this host: 9 PASS (mosaic3d, openins3d, openyolo3d, open3dis, openmask3d,
# open3d, metashape, rtabmap, bundlefusion), 2 BLOCKED (zed #29, isaac #22), 0 FAIL.
python spellbook/main.py --gpu-check          # --models mosaic3d,... / --engine ... to filter
```

Class lists: derived in `spellbook/benchmark.py` from `BenchmarkScripts/ScanNet200/scannet200_constants.py` — ScanNet20: 20 minus wall/floor = 18 NYU40 ids; ScanNet200: 200 minus ids {1,3} = 198 raw ids.

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
4. Engine debugging (manual, per engine): six `debug/reconstruction-<engine>` worktrees exist from master `5a3fa9e`; verify the never-run engines (bundlefusion, metashape, rtabmap, isaac #22) end-to-end and compare global methods against the local-pose baseline (#25). Per-engine QC results stay in `recon/qc.yaml` + issues. Frames for 9009 are complete and reusable; new extraction is blocked (ZED disabled, #29) — 9004 and any new SVO need the remapped-ZED work of #29 first.
5. ZED: validate managed remapping (pool lease + CUDA_VISIBLE_DEVICES before pyzed import, sdk_gpu_id unset, 60-frame nonconstant-pose gate, full trajectory regression) to re-enable extraction and the zed engine — #29.
6. Isaac: build `zed-isaac-nvblox:spellbook` from public Isaac debs (no NGC credentials) and switch off the broken CDI flag — #22.
7. Optional: 4 mm re-integration needs a working CUDA Open3D build (tensor VoxelBlockGrid broken in the installed 0.19; legacy volume at 4 mm hits ~185 GB RSS) — #31.
8. Integration: after each engine tree is closed (plugin `worktree_delete`, one at a time), merge `--no-ff debug/reconstruction-<engine>` into master and run the full batch without GPU arguments.
