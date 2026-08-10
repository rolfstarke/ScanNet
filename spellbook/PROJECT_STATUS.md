# ScanNet Spellbook - Project Status

## Project Goal

Integrate open-vocabulary 3D instance segmentation models (Mosaic3D, OpenIns3D, OpenYOLO3D, Open3DIS) into the ScanNet repository structure, run them on official ScanNet val scenes, and evaluate them with ScanNet's own benchmark protocol — 18-class NYU40 and ScanNet200 (198-class instance protocol). ScanNet itself must remain fully intact and operable.

Bugs, problems, and their attempted fixes live in GitHub Issues (`rolfstarke/ScanNet`, referenced as #N) — never in this file. This file is the repository structure explanation, goal, and project plan for new sessions.

---

## Repository Structure

### ScanNet (upstream, first priority)
- `BenchmarkScripts/` — official evaluators, ScanNet200 constants/splits. Only minimal Python-3 patches applied (see #5); logic untouched.
- `SensReader/python/SensorData.py` — Python-3 ported (.sens reader); logic untouched (see #5).
- `Tasks/Benchmark/scannetv2_val.txt` — official val split list.

### spellbook/ (all project code lives here)
```
spellbook/
├── main.py                      # CLI: --visualize, --predict (--benchmark, --run-id, multi-scene, --gpu)
├── settings.yaml                # default benchmark + scannet_root
├── benchmark.py                 # BenchmarkSpec (ScanNet20: 18 classes / ScanNet200: 198 = 200 - wall/floor), paths
├── evaluate.py                  # GT export + evaluation dispatch (official / scannet200 evaluator)
├── scannet200_evaluator.py      # Python-3 port of Rozenberszki's ScanNet200 evaluator (198-class)
├── environment.yaml             # 3disspellbook conda env
├── PROJECT_STATUS.md            # this file
├── utils/
│   ├── visualize.py             # Open3D viewer + ImGui legend (GT + official submission predictions)
│   └── hud.py
└── predict/
    ├── runner.py                # model→env dispatch, free-GPU queue, sequential up-front frame extraction
    ├── frames.py                # .sens extraction via ScanNet's SensorData exporters (0..N-1), idempotent
    └── models/
        ├── common.py            # decimate(), write_scannet_submission (official submission layout)
        ├── _mosaic3d_run.py     # point-cloud only
        ├── _openins3d_run.py    # point-cloud only; --detector {odise,yoloworld}
        ├── _openyolo3d_run.py   # needs frames
        └── _open3dis_run.py     # needs frames
```

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
├── frames/{color,depth,pose}/{0..N-1}.{jpg,png,txt}
└── frames/intrinsic_{color,depth}.txt + extrinsic_{color,depth}.txt
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
| OpenMask3D | frames | excluded per user decision | — | — |

Per-model integration issues: #2 (openyolo3d intrinsics), #3 (open3dis rescale), #4 (openins3d demo path), #10 (mosaic3d), #13 (openins3d recall).

---

## Key Technical Decisions

1. **Native model pipelines**: call each model's own ScanNet-capable API, not ov3dis-comparison wrappers.
2. **Canonical frame pool**: full-density extraction to `frames/` (sequential 0..N-1 names, 4 native intrinsic files); models subsample via their own configs. ScanNet's SensReader `export_*` would keep gapped indices — rejected (#5).
3. **Prediction output**: official ScanNet submission layout, one directory per (benchmark, run, model): `/data/scannet/predictions/<Benchmark>/<run-id>/<model>/` with `<scene>.txt` + `predicted_masks/<scene>_NNN.txt`. Directly zippable as a benchmark submission; runs never overwrite each other (#16).
4. **Label ids**: real NYU40 ids (ScanNet20) resp. raw `id`-column ids (ScanNet200, 198 classes = 200 minus wall/floor) from ScanNet's own constants, derived in `benchmark.py` as the single source of truth; unknown class names raise (see #1, #12).
5. **Benchmark protocol**: `settings.yaml` selects the default backend (ScanNet20 = official evaluator ported in place to Python 3; ScanNet200 = port of the benchmark author's evaluator, since ScanNet/ScanNet publishes no ScanNet200 instance evaluator). `--benchmark` overrides; `--classes` is for custom (non-benchmark) prediction only.
6. **GPU dispatch**: free-GPU queue over `--gpu` list (default all GPUs); frames extracted sequentially up front (see #9).
7. **Evaluation**: flat per-vertex GT encoding via `evaluate.py export-gt` (ScanNet's own export tool is inconsistent with its evaluator, see #8); evaluators ported to Python 3 with edge-case fix (#8); `evaluate.py evaluate` dispatches per benchmark with pre-flight validation (#11).

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

# Visualization (predictions need --run-id; without it: GT only)
python spellbook/main.py --visualize --scene 0568_00 --benchmark ScanNet20 --run-id myrun
```

Class lists: derived in `spellbook/benchmark.py` from `BenchmarkScripts/ScanNet200/scannet200_constants.py` — ScanNet20: 20 minus wall/floor = 18 NYU40 ids; ScanNet200: 200 minus ids {1,3} = 198 raw ids.

---

## Dependencies

- `3disspellbook` conda env (visualization, orchestration, eval) — `spellbook/environment.yaml` + pypng
- Per-model conda envs (see table); OpenIns3D env additionally has detectron2 0.6 + ODISE (see #14)
- Label mapping: `/data/scannet/v2/scannetv2-labels.combined.tsv`
- GitHub CLI `gh` (issue workflow), `~/.local/bin/gh`

---

## Current Plan / Next Steps

1. Fix ScanNet200 protocol to the official 198-class instance set (prompts, GT, evaluator) and re-run — #12.
2. Investigate OpenIns3D's 189-class collapse / anomaly scenes — #13.
3. Hardening: eval CLI validation #11, atomic/isolated prediction outputs #16, batch supervision #17, Open3DIS tracker race #15, env reproducibility #14.
4. Optional: extend from 20 to the full 312-scene val split once hardening is in place.
5. Optional: re-evaluate after protocol fix to obtain the defensible ScanNet200 numbers.

Status of prior benchmark results is recorded in the issues (#1–#4, #12) and in `spellbook/eval/results_*.txt`; this file deliberately keeps no result chronology.
