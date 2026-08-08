# ScanNet Spellbook - Project Status

## Project Overview
Integration of 3D instance segmentation prediction models into ScanNet repository structure, maintaining ScanNet's native data organization and benchmarking capabilities.

---

## Current Status

### Completed Features
1. **Visualization System** ✅
   - Side-by-side Open3D viewer + ImGui legend
   - Displays ground truth annotations from `.aggregation.json`
   - Supports prediction overlay (ScanNet benchmark format)
   - Hotkeys: M=mesh/point, H=ceiling, C=class colors, I=instance colors, N=cycle models
   - Location: `/spellbook/main.py --visualize --scene <scene_id>`

2. **Environment Setup** ✅
   - Conda env: `3disspellbook` with Open3D, ImGui, numpy, plyfile, etc.
   - Location: `/spellbook/environment.yaml`

3. **Data Structure** ✅
   - Downloaded scenes: val split, scene0568_00/01/02, scene0304_00, scene0488_00/01, scene0412_00/01 (8 scenes)
   - Location: `/data/scannet/scans/<scene_id>/`
   - Labels: `/data/scannet/v2/scannetv2-labels.combined.tsv` (NYU40)

4. **Investigation & Research** ✅
   - Analyzed all 5 model repositories
   - Confirmed all have native ScanNet support
   - Documented ScanNet benchmark format
   - Location: `/spellbook/SCANNET_BENCHMARK_GUIDE.md`, `/spellbook/INVESTIGATION_SUMMARY.md`

### Completed Features (continued)

5. **Prediction Integration (all models except OpenMask3D)** ✅
   - `python spellbook/main.py --predict --scene <NNNN>_NN --models mosaic3d,openins3d,openyolo3d,open3dis --classes chair,table`
   - Frame extraction: `spellbook/predict/frames.py` — delegates to ScanNet's own `SensorData` exporter (`export_color/depth/poses/intrinsics`, frame_skip=1 → contiguous 0..N-1 names), no custom .sens parsing. Canonical output: `frames/{color,depth,pose}/{0..N-1}.{jpg,png,txt}` + 4 native intrinsic files (`intrinsic_color/extrinsic_color/intrinsic_depth/extrinsic_depth.txt`). Idempotent — cache check uses a header-only `_sens_num_frames()` (~0s, avoids SensorData's full-file decode). Deps: pypng in 3disspellbook.
   - Format mapping solved ON IMPORT per model (never touch frames.py for new models):
     - OpenYOLO3D: symlink `frames/intrinsic_depth.txt` → scratch `intrinsics.txt`, `frames/pose/` → scratch `poses/` (model expects plural).
     - Open3DIS: symlink `frames/intrinsic_depth.txt` → its `intrinsic.txt`.
   - OpenYOLO3D: `models/_openyolo3d_run.py` — uses repo's `OpenYolo3D().predict()`, scratch dir with `poses/` symlink (model expects plural), background class = len(classes) filtered out.
   - Open3DIS: `models/_open3dis_run.py` — adapted ov3dis wrapper to ScanNet layout; **img_dim must be depth res (640x480), rgb_img_dim color res (1296x968)** — ScanNet resolutions differ, ov3dis's single-dim pattern only worked because ZED color==depth. Patched checkout at /home/rolf/GIT/Open3DIS verified (Ov3disSceneReader, tracker files, if-False gate).
   - Results on scene0000_00 (train, deleted): mosaic3d 10 (48s), openins3d 2-4 (~100s), openyolo3d 136 (140s), open3dis 65 (352s)
   - Results on scene0568_00 (val, 1651 frames, chair/table): mosaic3d 11 (46s), openins3d 12 (92s), openyolo3d 88 (108s), open3dis 24 (222s)
   - All validated with ScanNet's own `util_3d.read_instance_prediction_file`
   - Files: `/spellbook/predict/` (runner.py, frames.py, models/common.py, models/_mosaic3d_run.py, models/_openins3d_run.py, models/_openyolo3d_run.py, models/_open3dis_run.py)
   - SensReader `SensorData.py` patched to Python 3 (also fixed: sensor_name bytes join, lazy png import, direct byte reads instead of per-byte struct.unpack)

### Scenes
- Training scenes scene0000_00..scene0009_00: downloaded, used for validation, then **deleted** (2026-08-07)
- Current scenes (val split, first 8 lines of scannetv2_val.txt): scene0568_00/01/02, scene0304_00, scene0488_00/01, scene0412_00/01 — all full 13-file sets; frames/ + predictions/ extracted for all 8 scenes

### Batch Run (2026-08-07, 18-class benchmark set, all 5 GPUs, ~33 min)
- New CLI: `--scene` takes multiple IDs; `--gpu` takes a list, defaults to ALL GPUs (nvidia-smi auto-detect); (model×scene) tasks dispatched via free-GPU queue (pattern adapted from ov3dis-comparison `pipeline.predict_parallel`); frames extracted sequentially up front (parallel would race on the cache)
- Instance counts (all validated via `read_instance_prediction_file`, masks match predictions.txt):

| scene | mosaic3d | openins3d | openyolo3d | open3dis |
|---|---|---|---|---|
| 0568_00 | 43 (62s) | 18 (168s) | 569 (287s) | 69 (361s) |
| 0568_01 | 50 (72s) | 15 (163s) | 568 (290s) | 65 (348s) |
| 0568_02 | 47 (64s) | 12 (168s) | 564 (260s) | 61 (337s) |
| 0304_00 | 51 (55s) | **1** (146s) | 324 (130s) | 66 (349s) |
| 0488_00 | 22 (47s) | 5 (145s) | 342 (106s) | 53 (242s) |
| 0488_01 | 19 (51s) | 7 (131s) | 378 (115s) | 71 (271s) |
| 0412_00 | 81 (79s) | **1** (154s) | 574 (202s) | 111 (442s) |
| 0412_01 | 56 (69s) | 5 (129s) | 522 (152s) | 72 (304s) |

- 32/32 tasks OK; no failures. openins3d anomalously low (1 instance) on 0304_00/0412_00 — detection-threshold suspicion, spot-check needed. open3dis = throughput bottleneck (2.5× mosaic3d). Log: /tmp/opencode/predict-batch-20260807.log

### Official ScanNet Benchmark Evaluation (2026-08-07)
- **NYU40 label fix**: `spellbook/predict/models/common.py` `write_scannet_predictions` previously wrote sequential label ids (class_index+1) — the official evaluator silently discards any label id not in its 18 VALID_CLASS_IDS, so ALL prior predictions scored zero. Now maps class names → real NYU40 ids from ScanNet's own evaluator constants (BENCHMARK_CLASS_LABELS/BENCHMARK_VALID_CLASS_IDS), and hard-errors on non-benchmark class names (custom/scene-specific vocab runs are unscored by design).
- **Eval harness** (`spellbook/eval/`):
  - `evaluate_semantic_instance.py` — official evaluator ported to Python 3 (print/except-as/np.bool/np.float fixes). Import path hardcoded to BenchmarkScripts/ util+util_3d (untouched originals).
  - `export_train_mesh_for_evaluation.py` — ported (print/iteritems/bare-raise). NOTE: ScanNet's own GT-export pair is internally inconsistent: `util_3d.export_instance_ids_for_eval` writes a per-instance-mask layout that the evaluator's `load_ids()` cannot parse (and masks go to CWD-relative paths). So `export_gt.py` was written instead — reuses read_aggregation/read_segmentation logic, emits the flat per-vertex `nyu40id*1000+inst_id` encoding the evaluator itself defines (get_label_id: `//1000`).
  - `run_eval.py` — per model: collects per-scene predictions into `pred_path/<scene>.txt` + scene-unique mask names (avoids 000.txt collisions), runs the ported evaluator against `spellbook/eval/gt/`.
  - Fixed upstream evaluator edge-case bug: crash when a label has GT+predictions but zero matches (y_true_sorted_cumsum[-1] on empty array) → AP=0.0.
- **AP results (8 val scenes, 18-class NYU40, official protocol)**:

| Model | AP | AP50 | AP25 |
|---|---|---|---|
| mosaic3d | 0.067 | 0.251 | 0.549 |
| openins3d | 0.097 | 0.145 | 0.147 |
| openyolo3d | **0.242** | **0.334** | **0.399** |
| open3dis | 0.000 | 0.000 | 0.149 |

- Caveats: (a) published numbers are mostly on ScanNet200 (200 classes) with their own preprocessed caches — not directly comparable to this 18-class run; (b) classes absent from the 8 scenes (bed, toilet, desk, shower curtain, bathtub) report nan and drag averages; (c) open3dis AP≈0 despite 50-110 instances/scene and AP25=0.149 → masks/confidences likely misaligned for strict overlap (investigate mask-mesh vertex alignment); (d) openins3d 1-instance scenes (0304_00/0412_00) hurt its scores. Results CSV: `spellbook/eval/results_<model>.txt`, log: /tmp/opencode/eval-run-20260807.log
- Predictions re-run 2026-08-07 with corrected NYU40 labels (log /tmp/opencode/predict-batch-eval-20260807.log, ~23 min)

---

## System Architecture

### Data Organization
```
/data/scannet/scans/
├── scene0568_00/
│   ├── scene0568_00_vh_clean_2.ply           # Cleaned mesh
│   ├── scene0568_00.aggregation.json         # Ground truth annotations
│   ├── scene0568_00_vh_clean_2.0.010000.segs.json  # Segmentation
│   ├── scene0568_00.sens                     # RGB-D sensor data
│   ├── frames/                               # ✅ full-density extraction (color/depth/pose + 4 intrinsic files)
│   └── predictions/                          # Model predictions
│       ├── mosaic3d/                         # ✅
│       ├── openins3d/                        # ✅
│       ├── openyolo3d/                       # ✅
│       ├── open3dis/                         # ✅
│       └── openmask3d/                       # excluded (user decision)
```

### Spellbook Structure
```
/spellbook/
├── main.py                          # CLI entry point (--visualize, --predict)
├── environment.yaml                 # Conda environment
├── utils/
│   ├── visualize.py                 # Open3D visualization
│   └── hud.py                       # ImGui legend
├── predict/                         # ✅ implemented (4 of 5 models)
│   ├── runner.py                    # model→env dispatch, gpu handling, shared frame extraction
│   ├── frames.py                    # ✅ full-density .sens extraction, sequential 0..N-1
│   └── models/
│       ├── common.py                # decimate, write_scannet_predictions
│       ├── _mosaic3d_run.py         # ✅
│       ├── _openins3d_run.py        # ✅
│       ├── _openyolo3d_run.py       # ✅
│       ├── _open3dis_run.py         # ✅
│       └── _openmask3d_run.py       # not created (excluded)
└── PROJECT_STATUS.md / PROJECT_PLAN.md
```

---

## Model Integration Status

### 5 Models Identified

1. **Mosaic3D** - Point-cloud only ✅ IMPLEMENTED
   - Repo: `/home/rolf/GIT/Mosaic3D`
   - Env: `/data/mosaic3d/conda/envs/mosaic3d`
   - Run script: `_mosaic3d_run.py` (uses `scripts/run_custom_scene.py:run_inference`)
   - scene0568_00: 11 instances, ~46s

2. **OpenIns3D** - Point-cloud only ✅ IMPLEMENTED
   - Repo: `/home/rolf/GIT/OpenIns3D`
   - Env: `/data/openins3d/conda/envs/openins3d`
   - Run script: `_openins3d_run.py` (Mask3D→Snap→YOLO-World Lookup)
   - scene0568_00: 12 instances, ~92s
   - NOTE: instance count varies run-to-run (YOLO-World confidence on synthetic renders); ov3dis never filled its openins3d column either
   - IMPORTANT: must render the MESH path (not raw point cloud tensor) — matches OpenIns3D's own ScanNet flow, ~2x better detection

3. **OpenMask3D** - Requires RGB-D frames, EXCLUDED per user decision
   - Repo: `/home/rolf/GIT/openmask3d`
   - Env: `/home/rolf/anaconda3/envs/openmask3d`

4. **OpenYOLO3D** - RGB-D frames ✅ IMPLEMENTED
   - Repo: `/home/rolf/GIT/OpenYOLO3D`
   - Env: `/data/openyolo3D/conda/envs/openyolo3d`
   - Run script: `_openyolo3d_run.py` (Mask3D proposals + YOLO-World 2D voting)
   - scene0568_00: 88 instances, ~108s
   - Needs `LD_LIBRARY_PATH=/data/openyolo3D/cuda-11.3/lib64` (hook in runner.py)
   - Background class = len(classes) (last index) filtered from output

5. **Open3DIS** - RGB-D frames, multi-stage ✅ IMPLEMENTED
   - Repo: `/home/rolf/GIT/Open3DIS` (patched checkout, verified: Ov3disSceneReader present, if-False gate, tracker files)
   - Env: `/data/open3dis/conda/envs/open3dis`
   - Run script: `_open3dis_run.py` (2D grounding → hier-agglo clustering → CLIP refine → classify)
   - scene0568_00: 24 instances, ~222s
   - CRITICAL: `img_dim` = depth res (640x480), `rgb_img_dim` = color res (1296x968) — ScanNet resolutions differ, unlike ov3dis's ZED scenes

---

## Key Technical Decisions

### 1. Use Native Model Pipelines
**Decision**: Call each model's native ScanNet prediction scripts directly, NOT ov3dis-comparison wrappers.

**Rationale**: 
- All models already have ScanNet support
- Avoids unnecessary abstraction layer
- Simpler maintenance
- Better compatibility with model updates

### 2. Frame Extraction
**Decision**: Full-density extraction to `/data/scannet/scans/<scene_id>/frames/` via a self-contained `.sens` parser (`frames.py`) with sequential 0..N-1 filenames; models subsample via their own configs (OpenYOLO3D `frequency`, Open3DIS `img_interval`).

**Rationale**:
- One canonical frame pool per scene, extracted once, idempotent
- SensReader's own `export_*` keeps gapped original indices when subsampling — neither model accepts that
- Per-model sampling density is decoupled from extraction density

### 3. Prediction Output Location
**Decision**: Store predictions at `/data/scannet/scans/<scene_id>/predictions/<model>/`

**Rationale**:
- Follows ScanNet structure
- Easy to find
- Compatible with visualization tool
- Can run ScanNet's evaluator directly

### 4. ScanNet Benchmark Format
**Decision**: All predictions must match ScanNet benchmark instance segmentation format.

**Format**:
```
predictions.txt:
  predicted_masks/000.txt <label_id> <confidence>
  predicted_masks/001.txt <label_id> <confidence>

predicted_masks/NNN.txt:
  0
  1
  1
  ... (one line per vertex, matching mesh vertex count)

labels.txt:
  <label_id> <class_name>
```

### 5. Implementation Priority
**Phase 1**: Point-cloud-only models (OpenIns3D, Mosaic3D)  
**Phase 2**: Frame extraction (SensReader integration)  
**Phase 3**: Frame-dependent models (OpenMask3D, OpenYOLO3D, Open3DIS)

---

## Known Issues & Constraints

### 1. Python 2 vs 3 (FIXED)
- SensReader `SensorData.py` print statements (lines 81, 96, 113, 121), bytes join, np.fromstring — all patched to Python 3

### 2. OpenIns3D Instance Variance
- YOLO-World detection confidence on synthetic Snap renders is low (0.1-0.35); instance count varies 0-4 per run
- `multiview_aggregation` requires normalized score ≥ 0.5; `assign_label_with_bbox` IoU ≥ 0.3
- Rendering the MESH path (not point-cloud tensor) substantially improves detection

### 3. Model CLI Differences
- Each model runs via its own env python + standalone `_<model>_run.py` (argparse: --pointcloud, --classes, --out, --gpu)
- runner.py holds the env→python mapping

### 4. Frame-Dependent Models (pending)
- openmask3d/openyolo3d/open3dis need RGB-D frames from `.sens`
- `frames.py` not yet written; SensReader python tool is patched and ready

### 5. GPU Management
- runner.py supports `--gpu` (passed as CUDA_VISIBLE_DEVICES in run scripts)
- OpenYOLO3D needs LD_LIBRARY_PATH to `/data/openyolo3D/cuda-11.3/lib64`

---

## Dependencies

### Python Packages (3disspellbook env)
- open3d, imgui, glfw, PyOpenGL (visualization)
- numpy, plyfile, imageio, opencv (data handling)
- Additional per-model requirements in model repos

### External Tools
- ScanNet SensReader (frame extraction)
- ScanNet BenchmarkScripts (evaluation)
- 5 model repositories with separate conda envs

### Data Requirements
- ScanNet scans in `/data/scannet/scans/`
- .sens files for frame-dependent models
- Label mapping: `scannetv2-labels.combined.tsv`

---

## Next Steps (Immediate)

1. Port ScanNet's `evaluate_semantic_instance.py` to Python 3 for official local benchmarking
2. Export ground truth with `BenchmarkScripts/3d_helpers/export_train_mesh_for_evaluation.py`
3. For official benchmark: scenes on disk are already the val split (scene0568_00 etc.) — just need the Python 3 port of the evaluator
4. Optionally integrate OpenMask3D later (needs frames, was excluded)

---

## Long-Term Roadmap

### Future Enhancements
- Batch scene processing
- Parallel model execution on multiple GPUs
- Auto-evaluation after prediction
- Prediction comparison visualization
- Model performance benchmarking
- Support for ScanNet200 (200 classes vs NYU40's 20)

### Maintenance
- Keep model wrappers updated with upstream changes
- Monitor ScanNet benchmark format updates
- Optimize frame extraction (caching, compression)
- Add error handling and recovery

---

## References

### Documentation
- `/spellbook/SCANNET_BENCHMARK_GUIDE.md` - Format specification
- `/spellbook/INVESTIGATION_SUMMARY.md` - Model analysis
- `/spellbook/PROJECT_PLAN.md` - Implementation plan
- `PREDICT_PLAN.md` (root) - Legacy ov3dis-comparison plan (superseded)

### External Resources
- ScanNet benchmark: http://kaldir.vc.in.tum.de/scannet_benchmark/
- ScanNet repository: https://github.com/ScanNet/ScanNet
- Model repositories: See individual model sections above

---

**Last Updated**: 2026-08-08  
**Current Phase**: wrapper bugs fixed + ScanNet200 evaluation DONE (20 scenes, 189 classes)  
**Next**: verify openins3d's 200-class collapse (works at 18 classes, near-zero at 189 — ODISE vocab/aggregation suspicion), optional full-312-scene run

---

## 2026-08-08: Wrapper bug fixes + ScanNet200 expansion

### Root-cause analysis (4 parallel explore agents, 2026-08-07)
Low 18-class AP was NOT model weakness — three integration bugs in our glue code + one benchmark-mismatch:
1. **openyolo3d**: `_openyolo3d_run.py` symlinked `intrinsics.txt` → `intrinsic_depth.txt`. OpenYOLO3D's
   `WORLD_2_CAM.adjust_intrinsic` expects COLOR-resolution intrinsics (1296x968) and rescales to depth
   internally → projections ~2x displaced, ~86% of instances got zero 2D votes → mislabels.
   Confirmed via maintainer GitHub issue #9 + issue #14 + OpenMask3D's explicit config. FIX: 1-line →
   `intrinsic_color.txt`.
2. **open3dis**: `ov3dis_scene` branch (our glue, copied from the ZED/Replica path) in
   `Open3DIS/open3dis/src/clustering/clustering.py` + `tools/refine_grounding_feat.py` omitted the
   depth→color rescale the official `scannet200` branch does → masks indexed at wrong pixels
   (median bestIoU vs GT 0.016, AP 0.000). FIX: added the same `scaling_mapping(...)` call.
3. **openins3d**: we ran the repo's DEMO path (`MASK_CONFIDENCE_THRESHOLD=0.5`, YOLO-World) not the
   paper's eval path (precomputed masks at ~0.001 recall + ODISE detector). FIX: threshold 0.5→0.05
   + installed ODISE (cloned `third_party/ODISE`, detectron2 0.6 built for torch 1.13.1+cu116) and
   switched the wrapper to `lookup.call_ODISE()` (paper's detector) via new `--detector` flag.
   NOTE: scene0304_00 still yields only ~2-6 instances — Mask3D's own on-the-fly proposal scores
   collapse there (scene-content issue, not the wrapper).
4. **mosaic3d**: not fixable via wrapper — paper's instance numbers need a trained mask-decoder
   checkpoint that is NOT released; our wrapper (per-point CLIP argmax + per-class DBSCAN) is a
   lower-bound reference. Also: paper reports ScanNet200 only, never ScanNet20.

### ScanNet200 harness (all additive — 18-class path byte-identical, pre-fix results preserved in `spellbook/eval/results_prefix/`)
- Official protocol confirmed via kaldir.vc.in.tum.de docs + `BenchmarkScripts/ScanNet200/scannet200_splits.py`:
  evaluate the **189 val-present classes** (11 train-only excluded), raw `id`-column ids (not nyu40id),
  identical thresholds (IoU 0.25/0.5/[0.5:0.95:0.05], min 100 verts), wall/floor ignored in instance tasks.
- `common.py`: `BENCHMARK200_LABEL_TO_ID` imported from official splits (with fallback literal copy);
  `write_scannet_predictions(..., label_set=)` picks scannet18/scannet200.
- `export_gt.py --label_set scannet200` → `label_to='id'`, filtering to the 189 valid ids → `gt200/` (20 scenes).
- `evaluate_semantic_instance.py --label_set scannet200`: rebinds 18→189 class tables (FIXED a bug where
  the unconditional 18-class block clobbered the 200-class rebind — 200 mode silently used 18 tables) +
  head/common/tail sub-averages per official leaderboard.
- `run_eval.py --label_set`: collects into `collected200/`, writes `results_<model>200.txt` (never clobbers 18-class).
- All 4 wrappers accept `--label_set` (added to openyolo3d/openins3d which were missing it).

### Scenes: 20 val scenes now (2.5GB each)
Original 8 (0568_00/01/02, 0304_00, 0488_00/01, 0412_00/01) + 12 new (0217_00, 0019_00/01, 0414_00,
0575_00/01/02, 0426_00/01/02/03, 0549_00). All 13-file sets validated, frames extracted.

### Results — 18-class benchmark, 8 scenes (official protocol, before vs after fixes)

| Model | AP pre | AP50 pre | AP25 pre | AP post | AP50 post | AP25 post |
|---|---|---|---|---|---|---|
| mosaic3d | 0.067 | 0.251 | 0.549 | 0.067 | 0.251 | 0.549 |
| openins3d | 0.097 | 0.145 | 0.147 | **0.299** | **0.474** | **0.510** |
| openyolo3d | 0.242 | 0.334 | 0.399 | **0.381** | **0.535** | **0.587** |
| open3dis | 0.000 | 0.000 | 0.149 | **0.205** | **0.346** | **0.391** |

### Results — ScanNet200 benchmark, 20 scenes, 189 classes (post-fix)

| Model | AP | AP50 | AP25 | head AP50 | common AP50 | tail AP50 |
|---|---|---|---|---|---|---|
| mosaic3d | 0.049 | 0.104 | 0.254 | 0.140 | 0.000 | 0.000 |
| openins3d | 0.049 | 0.080 | 0.089 | 0.108 | 0.000 | 0.000 |
| openyolo3d | **0.325** | **0.444** | **0.501** | 0.470 | 0.418 | 0.231 |
| open3dis | 0.092 | 0.195 | 0.256 | 0.153 | 0.270 | 0.442 |

Notes: not directly comparable to papers' 312-scene ScanNet200 numbers; openins3d collapses at 189
classes (18-class fine) — investigate vocab/ODISE aggregation; classes absent from the 20 scenes' GT
report nan and are excluded from means.

### Operations notes
- Two background runs were killed externally (once via `/tmp/opencode/` cleanup, once mid-shutdown) —
  the second kill happened after all outputs were written (buffered log lines lost, no data loss).
  Use `spellbook/tmp/logs/` (persistent) for long jobs, count `Wrote` lines in the log as the
  reliable progress signal (`spellbook/tmp/progress.sh`).
- Predictions for the original 8 scenes currently hold the 18-class post-fix outputs (the 200-class
  preds there were overwritten — results files retained).

