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
├── main.py                      # CLI: --visualize, --predict (multi-scene, --gpu, --classes, --label_set)
├── environment.yaml             # 3disspellbook conda env
├── PROJECT_STATUS.md            # this file
├── utils/
│   ├── visualize.py             # Open3D viewer + ImGui legend (GT + predictions)
│   └── hud.py
├── predict/
│   ├── runner.py                # model→env dispatch, free-GPU queue, sequential up-front frame extraction
│   ├── frames.py                # .sens extraction via ScanNet's SensorData exporters (0..N-1), idempotent
│   └── models/
│       ├── common.py            # decimate(), write_scannet_predictions (label_set: scannet18 | scannet200)
│       ├── _mosaic3d_run.py     # point-cloud only
│       ├── _openins3d_run.py    # point-cloud only; --detector {odise,yoloworld}
│       ├── _openyolo3d_run.py   # needs frames
│       └── _open3dis_run.py     # needs frames
└── eval/
    ├── export_gt.py             # flat per-vertex label*1000+instance GT (18 + 200 label sets)
    ├── evaluate_semantic_instance.py  # official evaluator, Python-3 ported, --label_set
    ├── run_eval.py              # collects per-scene preds + runs evaluator per model
    ├── gt/, gt200/              # generated GT files
    └── results_*.txt            # per-model result CSVs (+ results_prefix/ = pre-fix reference)
```

### Data Layout (`/data/scannet/scans/<scene_id>/` — ScanNet native structure)
```
scene0568_00/
├── scene0568_00_vh_clean_2.ply          # mesh used as model input + GT vertex basis
├── scene0568_00.aggregation.json        # GT instance annotations
├── scene0568_00_vh_clean_2.0.010000.segs.json
├── scene0568_00.sens                    # RGB-D sensor data
├── frames/{color,depth,pose}/{0..N-1}.{jpg,png,txt}
├── frames/intrinsic_{color,depth}.txt + extrinsic_{color,depth}.txt
└── predictions/<model>/{labels.txt, predictions.txt, predicted_masks/NNN.txt}
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
3. **Prediction output**: `/data/scannet/scans/<scene_id>/predictions/<model>/` in ScanNet benchmark format (`labels.txt`, `predictions.txt`, `predicted_masks/NNN.txt`, one mask line per mesh vertex).
4. **Label ids**: real NYU40 ids (18-class) resp. raw `id`-column ids (ScanNet200) from ScanNet's own constants; unknown class names raise (see #1).
5. **Benchmark protocol**: official ScanNet18 (18 classes) and ScanNet200 instance (198 classes = 200 − wall/floor; 187 effective on val) — see #12.
6. **GPU dispatch**: free-GPU queue over `--gpu` list (default all GPUs); frames extracted sequentially up front (see #9).
7. **Evaluation**: flat per-vertex GT encoding via `export_gt.py` (ScanNet's own export tool is inconsistent with its evaluator, see #8); evaluator ported to Python 3 with edge-case fix (#8).

---

## Commands

```bash
# Prediction (multi-scene, multi-GPU; label_set: scannet18 | scannet200)
python spellbook/main.py --predict --scene 0568_00 0304_00 --models mosaic3d,openins3d,openyolo3d,open3dis \
    --classes cabinet,bed,chair,... --label_set scannet18

# Ground truth export
python spellbook/eval/export_gt.py --scan_path /data/scannet/scans/<scene> --output_file <out>.txt \
    --label_map_file /data/scannet/v2/scannetv2-labels.combined.tsv --label_set scannet18|scannet200

# Evaluation
python spellbook/eval/run_eval.py --models m1,m2 --scenes scene0568_00,... --label_set scannet18|scannet200

# Visualization
python spellbook/main.py --visualize --scene 0568_00
```

Class lists: 18-class names in `common.py` (`BENCHMARK_CLASS_LABELS`); ScanNet200 198-instance list from `BenchmarkScripts/ScanNet200/scannet200_constants.py` (200) minus ids {1,3}.

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
