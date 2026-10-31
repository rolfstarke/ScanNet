# ScanNet Spellbook - Project Status

## Project Goal

Integrate open-vocabulary 3D instance segmentation models (Mosaic3D, OpenIns3D, OpenYOLO3D, Open3DIS, sam3dis; OpenMask3D is archived) into the ScanNet repository structure, run them on a fixed 16-scene audit-focused official ScanNet validation set, and evaluate them with ScanNet's own benchmark protocol — 18-class NYU40 and ScanNet200 (198-class instance protocol). Comparable ranking uses ScanNet200 mean AP (IoU 0.50:0.95), AP only. ScanNet itself must remain fully intact and operable.

Second goal: turn ZED X `.svo2` recordings into byte-format-identical ScanNet v2 scans (`.sens`, `<id>.txt`, `_vh_clean*.ply`, `.segs.json`) under custom ids `scene90NN_MM` (9000-range = custom; tens digit = engine, units digit = run), so ScanNet's own tooling works on them unchanged. Scene9004 geometry quality is scored against CAD with a mean bidirectional distance in cm (`geometry_score.py`, decision 14; lower better). CAD stays fixed; a temporary rigid recon→CAD transform is applied in memory only for scoring/views.

Bugs, problems, results and their attempted fixes live in GitHub Issues (`rolfstarke/ScanNet`, referenced as #N) — never in this file. This file explains the repository structure and goals for new sessions.

---

## Repository Structure

### ScanNet (upstream, first priority)
- `BenchmarkScripts/` — official evaluators, ScanNet200 constants/splits. Python-3 compatibility patches plus the empty-match edge-case fix from #8; metric logic otherwise unchanged.
- `SensReader/python/SensorData.py` — Python-3 ported (.sens reader); logic untouched (see #5).
- `Tasks/Benchmark/scannetv2_val.txt` — official val split list.

### Agent skills (`.opencode/skills/`, linked from `.claude/skills/`)
- `s1-document` — after-build memory: structure/goals in this file, attempts in GitHub Issues, executed plans in `spellbook/archive/`
- `s2-improve` — six-run reconstruction/prediction improvement; prediction batches use `run.json` + `runs.py rank`
- `s3-baseline` — one untuned official-default reconstruction or prediction baseline for the current worktree method

### spellbook/ (all project code lives here)
```
spellbook/
├── main.py                      # CLI: --visualize, --predict, --engine, --extract-frames, --gpu-check
├── settings.yaml                # default ScanNet200 + scannet_root + gpu_pool ([1,2,3,4]; 0 user-reserved; `SPELLBOOK_GPU_POOL` overrides per run)
├── gpu_check.py                 # --gpu-check: per-model/engine native distribution probe over real pool leases
├── download_scans.py            # official v1/v2 restore: bounded parallelism, resume, size verification
├── environment.yaml             # 3disspellbook conda env (ZED SDK activation sets ZED_DIR/LD_LIBRARY_PATH)
├── PROJECT_STATUS.md            # this file
├── docs/setup.md                # reproducible per-method environment, data and image setup
├── tests/                       # unit tests: `python -m unittest discover -s spellbook/tests`
├── archive/                     # executed plans and reports (gitignored)
├── tmp/                         # active scripts only: overview/ (run queue + params), build_scene9004_reference.py, export_scene9004_cad.py
├── evaluation/
│   ├── benchmark.py             # BenchmarkSpec, PREDICTION_EVALUATION_SCENES (16 official val scans), PREDICTION_METHODS
│   ├── evaluate.py              # GT export, run.json-scoped staging evaluator, score-sidecars CLI, AP + F1 sidecars
│   ├── runs.py                  # run.json manifests, AP-only ranking, dry-run prediction prune
│   ├── comparison.py            # offline comparison dashboard data + HTML renderer (`main.py --compare`)
│   └── scannet200_evaluator.py  # Python-3 port of Rozenberszki ScanNet200 evaluator (198-class)
├── utils/
│   ├── gpu.py                   # cross-process GPU leases (flock per pool index, pass_fds forwarding)
│   ├── scan_lock.py             # frames / scene-engine / per-scan / prediction-index / per-method flock locks
│   ├── visualize.py             # Open3D SceneWidget viewer + scene/scan/method/run tree (best comparable run = AP only)
│   ├── scene_widget.py          # SceneWidget backend: window/registry/materials/Label3D handles (open3d==0.19.0)
│   ├── prediction_masks.py      # packed uint8 mask cache under derived/visualization/masks
│   ├── compute_time.py          # reconstruction/prediction elapsed sidecars for the right HUD
│   ├── query.py                 # hash-bound .clip.npz + physical-GPU-0 CLIP worker/client
│   ├── camera_replay.py         # frame discovery, frustum pyramid, RAM cache pump, transport, playback
│   ├── replay2d.py              # per-method 2D-stage Replay adapters (frames/fetch/close)
│   ├── detect_replay.py         # warm detector worker client + box/mask drawing
│   ├── hud.py                   # left tree + right Information/Settings/Query/replay video (CAD overlay)
│   ├── glx.py                   # selects the NVIDIA GLX provider for the Open3D viewer on remote displays (#55)
│   ├── resources.py             # read-only host resource snapshots for the status dashboard
│   ├── status.py                # `main.py --status` tables
│   └── query_prompt.py          # keyboard prompt for Query
└── predict/
    ├── runner.py                # one-method dispatch over gpu_pool; official classes only; method env/repo/resource tables
    ├── frames.py                # .sens extraction via ScanNet's SensorData exporters (0..N-1), idempotent
    └── models/
        ├── common.py            # decimate(), generation-safe write_scannet_submission_rows, --run-id/--parameters-json/--features-out
        ├── _mosaic3d_run.py     # point-cloud only; Recap-CLIP instance features
        ├── _mosaic3d_scene.py   # Mosaic3D scene pipeline (net, inference, proposal classification)
        ├── _mask3d_proposals.py # Mask3D proposals, run in the OpenIns3D env for Mosaic3D
        ├── _openins3d_run.py    # point-cloud only
        ├── _openins3d_query.py  # Lookup-only child over cached Snap images + final masks
        ├── _openyolo3d_run.py   # needs frames
        ├── _openyolo3d_detect_worker.py # warm detector for visualizer Replay
        ├── _open3dis_run.py     # needs frames; official 2D+3D path
        ├── _open3dis_{config,prepare,stage}.py # per-run config, per-scene data prep, stage child
        ├── _open3dis_replay_worker.py # cached-mask worker for visualizer Replay
        ├── _openmask3d_run.py   # archived method; needs frames; scratch isolated from submission root
        ├── _sam3dis_run.py      # needs frames; adapter for the external /data/sam3dis
        └── _gpu_check.py        # --gpu-check child
```

### spellbook/reconstruct/ (SVO2 -> ScanNet-native scans; invoked via `main.py --engine`)
```
spellbook/reconstruct/
├── batch.py                  # multi-scene x multi-engine: shared frames then parallel run.py children
├── run.py                    # allocate run slot -> reconstruct -> score scene9004 (post-GPU)
├── extract.py                # main-checkout multi-GPU SVO extraction into shared frame pools
├── _extract_run.py           # leased-GPU child: CUDA_VISIBLE_DEVICES set before pyzed import
├── geometry_score.py         # score v2: 2 cm gap-free occupancy, two-way mean distance to visible CAD, report PNG
├── geometry_reference.yaml   # metric version, scene9004 CAD/visibility hashes, grid, retention, plot scale
├── cleanup.py                # exact prediction/task-line/TP50-sidecar purge on scan eviction/replace
├── tsdf.py                   # shared Open3D TSDF integration
├── scannet.py                # gravity z-up align, pymeshlab clean, axisAlignment, Segmentator
├── finalize.py               # .sens v4 + <id>.txt writers
├── containers/               # docker helper (__init__.py, gpu_args.sh) + build contexts rtabmap/, isaac/, bundlefusion/
└── engines/
    ├── __init__.py           # adapter contract: GPU_POLICY (managed/cpu/blocked), SERIAL, preflight(), gpu_check(), reconstruct(work, root, gpu, lease_fd)
    ├── zed.py                # ZED SDK native mapping (managed GPU, serial)
    ├── open3d.py             # Stage A poses + bounded tensor TSDF (managed GPU)
    ├── metashape.py          # Metashape Pro (managed GPU, serial, license)
    ├── rtabmap.py            # RTAB-Map replay in docker (managed GPU)
    ├── isaac.py              # cuVSLAM + Nvblox in docker (managed GPU, serial)
    └── bundlefusion.py       # ScanNet's reference engine in docker (managed GPU pair, canonical 10 mm)
```

### Worktrees and external repositories

Main checkout `/data/ScanNet`. Method repositories and their conda envs live under `/data/<X>`, see the Model Integration and Reconstruction Engines tables. `spellbook/` wrappers call them; upstream code stays untouched apart from documented patches on `spellbook-baseline` branches.

### Data Layout (`/data/scannetdata/` — scans/ stays official, artifacts outside)
```
/data/scannetdata/
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
│   ├── raw/scene900X.svo2                   # SVO symlinks (9004, 9009)
│   └── reference/                           # scene9004 CAD (.3dm, PLY) + visible-cell NPZ (20 mm)
├── derived/
│   ├── ground_truth/<Benchmark>/<scene>.txt
│   ├── evaluations/<Benchmark>/
│   │   ├── ranking.csv                      # derived AP/AP50/AP25 table from run.json + CSVs
│   │   └── <run-id>/                        # run.json, result CSVs, <model>.tasks, <model>/<scene>.tp50.json + .timing.json + .clip.npz
│   ├── evaluations/comparison.html          # offline dashboard via `main.py --compare` (predictions + reconstructions + custom scans)
│   ├── reconstruction/frames/sceneNNNN/frames/   # shared SVO frame pools
│   ├── visualization/masks/<Benchmark>/<run-id>/<model>/<scene>.npz  # disposable packed 0/1 caches
│   ├── reconstruction/bundlefusion/prepared/ # prepared BundleFusion input stream
│   ├── cache/ open3dis/                      # method caches
│   ├── jobs/<job-id>/                        # job records
│   └── locks/{gpus,reconstruction,scans}/  # GPU + scan/frame flock leases
└── v2/scannetv2-labels.combined.tsv
```

### Custom scans (9000-range)
```
/data/scannetdata/scans/scene9004_40/            # tens=engine (0 zed, 1 metashape, 2 rtabmap, 3 isaac, 4 open3d, 5 bundlefusion),
│   *.sens/.txt/_vh_clean*.ply/.segs.json    # units=run 0-9; scene9004 keeps ten slots/engine
└── recon/
    ├── frames -> derived/.../scene9004/frames   # symlink to shared pool
    ├── geometry_score.yaml + cad_comparison.png   # score report: plan/elev/isometric
    ├── engine_native.ply mesh_aligned.ply final_poses.npy
     └── cmdline.txt svo_path.txt timing.json   # wall time of reconstruct+finalize, no CAD
```

### Official prediction evaluation scenes (fixed in `benchmark.py`)
`0046_00, 0084_01, 0086_01, 0100_02, 0164_00, 0207_02, 0221_00, 0251_00, 0307_00, 0334_00, 0357_00, 0535_00, 0618_00, 0644_00, 0678_01, 0699_00`
One scan per physical scene; all members of `Tasks/Benchmark/scannetv2_val.txt`. Custom `scene90xx` scans stay on disk for reconstruction and are rejected by managed prediction/evaluation.

---

## Model Integration

| Model | Input | Repo | Env | Notes |
|---|---|---|---|---|
| Mosaic3D | point cloud | `/data/Mosaic3D` | `/data/mosaic3d/conda/envs/mosaic3d` | Mask3D proposals via the OpenIns3D env; `_mosaic3d_scene.py` holds the scene pipeline; geom rescoring formulas inferred (#33) |
| OpenIns3D | point cloud | `/data/openins3d/repo-56ccc22` (worktree of `/data/OpenIns3D`) | `/data/openins3d/conda/envs/openins3d` | released precomputed masks; ODISE or YOLO-World labelling |
| OpenYOLO3D | frames | `/data/OpenYOLO3D` | `/data/openyolo3D/conda/envs/openyolo3d` | CUDA 11.3 at `/data/openyolo3D/cuda-11.3` via LD_LIBRARY_PATH hook (runner.py) |
| Open3DIS | frames | `/data/Open3DIS` (upstream, unpatched) | `/data/open3dis/conda/envs/open3dis` | official 2D+3D path driven by `_open3dis_{config,prepare,stage,run}.py`; shared weights in `/data/pretrains` |
| sam3dis | frames | `/data/sam3dis` (external, SAM 3 code in `/data/sam3`) | `/data/sam3dis/conda/envs/sam3dis` | lifts SAM 3 masks to superpoints; registered like the other methods (#69) |

Archived method (not an active comparison method; wrapper code and env stay in place, #46, #37):

| Model | Input | Repo | Env | Notes |
|---|---|---|---|---|
| OpenMask3D | frames | `/data/openmask3d` (branch `spellbook-baseline`) | `/data/openmask3d/conda/envs/openmask3d` | scannet200_model.ckpt; OPENMASK3D_FORCE_GPU=0 pins the repo's get_free_gpu to the visible device |

Shared build input: `/data/detectron2` (v0.6) is built into the Open3DIS and OpenIns3D envs.

Per-model integration issues: #2 (openyolo3d intrinsics), #3 (open3dis rescale), #4 (openins3d demo path), #10 (mosaic3d), #13 (openins3d recall), #69 (sam3dis).

---

## Reconstruction Engines

| Engine (scan tens digit) | Pose source | Mesh | Runtime | Presets (default in bold) |
|---|---|---|---|---|
| zed (0) | ZED SDK two-pass `.area` tracking | ZED native spatial mapping | pyzed 5.4, SDK at `~/opt/zed-sdk/install` | native-default, **b5-gen1-neural**, s2b8-fixedcalib |
| metashape (1) | Metashape SfM + BA (keyframe priors) | Metashape depth maps | `/data/zed-metashape/conda/env` (licensed, serial, exempt from the 3x runtime cap) | native-default, **b5-kf350** |
| rtabmap (2) | Frame-pool replay with Stage A odometry, GTSAM | TSDF of the map frames (`tsdf.py`, 2 cm, no weight threshold) | docker `localhost/zed-rtabmap:jazzy` | **assisted-hybrid** stride 90 |
| isaac (3) | cuVSLAM | Nvblox TSDF | docker `zed-isaac-nvblox:spellbook` (serial) | official_default, **conf100_best** |
| open3d (4) | Stage A pool poses | Tensor CUDA VoxelBlockGrid, 1 cm, bounded frame count | `3disspellbook` env, open3d 0.19 CUDA | **bounded B1** |
| bundlefusion (5) | BundleFusion global BA on a prepared 640x480 stream | BundleFusion 10 mm TSDF | docker `bundlefusion:noflip` (GPU pair) | **official-default** |

Scene9004 scores live in `recon/geometry_score.yaml` (`mean_bidirectional_distance_cm`, lower better) + report `recon/cad_comparison.png`. Non-Metashape engines must finish within 3x the scene length.

`reconstruct/containers/` holds the docker build contexts and helper: docker here is the snap package, so host paths mount through `/var/lib/snapd/hostfs` and GPU nodes plus driver libraries are passed explicitly (`gpu_args.sh`).

---

## Key Technical Decisions

1. **Native model pipelines**: call each model's own ScanNet-capable API, not ov3dis-comparison wrappers.
2. **Canonical frame pool**: full-density extraction to `frames/` (sequential 0..N-1 names, 4 native intrinsic files); models subsample via their own configs. ScanNet's SensReader `export_*` would keep gapped indices — rejected (#5).
3. **Prediction output**: official ScanNet submission layout, one directory per (benchmark, run, model): `/data/scannetdata/predictions/<Benchmark>/<run-id>/<model>/` with `<scene>.txt` + `predicted_masks/<scene>_NNN.txt`. Directly zippable as a benchmark submission; runs never overwrite each other (#16).
4. **Label ids**: real NYU40 ids (ScanNet20) resp. raw `id`-column ids (ScanNet200, 198 classes = 200 minus wall/floor) from ScanNet's own constants, derived in `benchmark.py` as the single source of truth; unknown class names raise (see #1, #12).
5. **Benchmark protocol**: `settings.yaml` selects the default backend (ScanNet20 = official evaluator ported in place to Python 3; ScanNet200 = port of the benchmark author's evaluator, since ScanNet/ScanNet publishes no ScanNet200 instance evaluator). Comparable prediction always passes `--benchmark ScanNet200`, omits `--scene` (full 16-scene tuple) and rejects `--classes`. The audit-focused set covers 52/54 target classes across 11 room types and 16 physical scenes; all 25,734 `.sens` frames, canonical files, and both GT variants are locally verified.

   Active tuple: `scene0046_00`, `scene0084_01`, `scene0086_01`, `scene0100_02`, `scene0164_00`, `scene0207_02`, `scene0221_00`, `scene0251_00`, `scene0307_00`, `scene0334_00`, `scene0357_00`, `scene0535_00`, `scene0618_00`, `scene0644_00`, `scene0678_01`, `scene0699_00`.
6. **GPU scheduling (automatic)**: `settings.yaml gpu_pool` (`[1,2,3,4]`) is the managed GPU list; `utils/gpu.py` leases a pool GPU via persistent `flock` lock files under `<scannet_root>/derived/locks/gpus/`, held for the whole task (descriptor forwarded to GPU children with `pass_fds`, kernel-released on crash). **Physical GPU 0 is excluded from managed prediction/reconstruction workloads.** The visualizer's explicitly requested interactive workers are the sole exception: the Query encoder worker and the Replay 2D workers each select physical GPU 0 directly and take no lease. No manual `--gpu` exists. Custom SVO frame extraction uses the same pool via `--extract-frames`. The env var `SPELLBOOK_GPU_POOL` (comma-separated) overrides the pool per run, for a queue that hands out only free GPUs.
7. **Adapter metadata**: each engine adapter declares `GPU_POLICY` (`managed`/`cpu`/`blocked`), `SERIAL`, `preflight()` (runtime presence check) and `gpu_check()` (native distribution probe); `batch.py` and `--gpu-check` consume these instead of duplicated constants.
8. **Evaluation**: flat per-vertex GT encoding via `evaluate.py export-gt` (ScanNet's own export tool is inconsistent with its evaluator, see #8); evaluators ported to Python 3 with edge-case fix (#8); `evaluate.py evaluate` dispatches per benchmark with pre-flight validation (#11).
9. **Custom scan ids**: `scene90NN_MM` (9000-range unused by ScanNet v2); tens digit = engine, units digit = run 0-9 (scene9004 keeps ten slots per engine with highest-`mean_bidirectional_distance_cm` eviction; other scenes replace run 0). SVO discovery by filename (`custom/raw/scene<NNNN>.svo2`).
10. **Byte fidelity**: `.sens` v4 per `SensReader/c++/src/sensorData.h` (jpeg + zlib_ushort, depth_shift 1000); `<id>.txt` with the ScanNet 17 keys; `.segs.json` from ScanNet's own built Segmentator (defaults kThresh 0.01, segMinVerts 20).
11. **Meshes**: ScanNet's `clean.mlx` filter chain via pymeshlab (equivalent calls — this pymeshlab cannot load `.mlx`), decimation `targetfacenum=300000` (absolute, not ScanNet's relative 20% — deviation); PLY written float32 binary (Segmentator's tinyply rejects double).
12. **axisAlignment**: computed (pure z-rotation + translation, det == 1) and written to `<id>.txt`, NEVER applied — released ScanNet meshes and `.sens` poses share the raw frame (consumers apply it).
13. **Depth/pose conventions**: ZED X native resolution (1920x1080 / 1920x1200 per recording); depth 0.1-6.0 m, invalid = 0, trailing SVO frame dropped; camera-to-world poses conjugated `diag(1,-1,-1,1)` from ZED's z-backward basis; gravity z-up alignment per `alignment.h`; the batch extracts each SVO once (shared frames, `--replace` to regenerate) and runs engine tasks in parallel subprocesses.
14. **Geometry score v2 (scene9004)**: metric `observed_surface_voxel_mean_bidirectional_distance_v2`, configured in `reconstruct/geometry_reference.yaml`, on one 2 cm lattice. Occupancy is gap-free: triangles are bisected until they sit inside one cell or their longest edge is at most half a cell. The reference is the visible CAD mask `custom/reference/scene9004_visible_voxels_20mm.npz`, built by `tmp/build_scene9004_reference.py` with an inverse ray test: every CAD cell centre inside a camera's image gets one ray from that camera against the CAD, and cells seen from at least two of 120 poses are kept. `mean_bidirectional_distance_cm` is the plain mean of recon→CAD (accuracy) and CAD→recon (completeness), lower better. CAD original orientation is fixed; a temporary rigid ICP moves an in-memory recon copy into the CAD frame only (never saved). Report PNG `cad_comparison.png`: plan | elev occupancy (magenta=CAD, cyan=recon) plus full-width isometric recon coloured by distance to CAD; all plot scales are global (distance colour 0-100 cm) so reports compare across scans; ceiling height-hide floor+2.5 m for display only. No PASS/FAIL gate. Scores from metric v1 are not comparable.
15. **Custom SVO frame extraction**: main checkout only via `--extract-frames`; multi-GPU leases on the pool; `sdk_gpu_id` never set; scenes are extracted from `custom/raw/scene<NNNN>.svo2` (scene9004 promotes a legacy `scene9004_04` frame set only when one exists). Worktrees consume the shared pool read-only. Official `.sens` prediction frames instead use the idempotent CPU `predict.frames.extract_frames()` path and are extracted automatically by frame-dependent prediction models.
16. **Foreign-environment imports**: model subprocesses load `spellbook/evaluation/benchmark.py` by absolute `importlib` spec. Adding `spellbook/` to `sys.path` shadows model repositories' top-level packages such as OpenIns3D's `utils`.
17. **Worktrees (convention)**: the main checkout is `/data/ScanNet`; method debugging happens in one linked worktree per engine or prediction method, created on demand and removed afterwards. Shared core stays on `master`; each method branch touches only its own engine or wrapper.
18. **Prediction run registry**: each run writes `derived/evaluations/<Benchmark>/<run-id>/run.json` before GPU work (one method, protocol scenes, provenance). Ranking default is AP only, and comparable ScanNet200 ranks require the exact current 16-scene tuple. Historical official-val manifests remain loadable but are automatically non-comparable. Run IDs are permanent; evaluated runs are immutable (`evaluate` refuses to replace an existing CSV). AP sidecars bind to prediction-index and GT hashes. Method-improvement losers keep `run.json`, CSVs, tasks, and per-scene AP sidecars and drop only heavy prediction artifacts (`runs.py prune`, dry-run default). Open3DIS scene tasks run in parallel with per-scene working directories and trackers (#15).
20. **Detached jobs**: prediction/reconstruction/frame extraction detach by default: an immutable spec lands under `/data/scannetdata/derived/jobs/<job-id>/` and one transient `systemd --user` unit runs the job (the child re-enters `main.py` with an injected `--foreground` plus a one-worker cap); the submitter exits and OpenCode/tmux may close. A managed GPU lease requires a job record (`SPELLBOOK_JOB_ID`); `--foreground` runs inline with full pool-sized workers (may occupy all 4 managed GPUs) but still under a durable record via `job_context`, so no GPU work exists outside the registry. Submit refuses a run identity already owned by an active job (run id, else kind+engine+scene; cross-worktree duplicates rejected by name). The unit execs the main checkout's `jobs.py` (canonical runner) while the payload keeps the submitting worktree's code and cwd. `main.py --status` shows CURRENT JOBS (one row per processing job with leased GPU, effective runtime `now - started - gpu_wait`, live CPU cores, unit-level PROGRESS from existing manifests/timings, plus device-only rows for jobless GPUs), FUTURE JOBS (observable queue, oldest first, plan only), then the latest 20 PAST JOBS with total runtime (`ended - started - accumulated GPU-lease wait`; records without wait data show `-`); dead workers' records are persisted as `lost` on read; full history remains in `jobs.py list`. Complete-frame prediction runs take shared scan locks and overlap (#58); exclusive is held only while extracting frames. Jobs survive OpenCode/tmux closure but not full logout or reboot (`Linger=no`); unfinished records resume via explicit `retry` (refused when the submitting worktree's cwd is gone). Docker engines need `--foreground` from a shell with the docker group until the systemd user manager has that group, because detached units run under that manager.
19. **Visualizer**: four-level tree `scene → scan → method → run`. Method default prefers a comparable 16-scene ScanNet200 run, then AP; unlabeled fallback if none is comparable. Mode is Ground truth / Scene only / Prediction; Color is Classes / Instances / TP/FP/FN (TP green, FP red, FN amber point overlay). Settings also include Boxes, Labels (`Off/Small/Large`, default `Large`, screen-space Label3D via SceneWidget, independent of Boxes; anchors snap to representative mask points), Query (`Off/Search/Image`, restores prior color on exit), Filter (`Off/On`, paused while Query is active; prediction entry defaults to TP/FP/FN plus Filter On when GT/thresholds allow, otherwise safe fallback), and a single View setting (`Classes/Pipeline/Path/Replay`, exclusive bottom-right content; entering Replay prefetches the whole video into RAM behind a progress bar and starts playback only when every frame is cached; transport is Back / Pause-Play / Forward, arrow-selected and Enter-confirmed like the rest of navigation); unavailable options are hidden. FN ground-truth masks render in the overlay and carry class-only amber labels. Camera Replay is the method's own 2D stage: OpenYOLO3D replays live YOLO-World boxes on every 10th frame via a warm GPU-0 worker, Open3DIS replays cached Grounded-SAM masks instantly with no GPU, OpenIns3D/OpenMask3D adapters are deferred, and Mosaic3D has Replay disabled (no 2D stage). Search/Image opens a focused keyboard prompt, submits on Enter, closes, and returns focus to Settings. Scan Enter is scene-only; method/run Enter loads that prediction. Navigator and Information read schema-4 `.tp50.json` sidecars plus evaluator CSVs; navigator shows strict scene AP with evaluated-class count, Information shows Scene AP, Run AP, Visible predictions, and Eligible GT. Filter On keeps only prediction classes with eligible scene GT above the run/model/class micro-F1 threshold fitted over the run's available protocol scenes (`thresholds.f1.json`, population + manifest + sidecar hash-bound; missing/stale disables On with a visible reason); manual edits are process-local, selected with Up/Down and adjusted with Left/Right (±0.01, Enter restores fitted). Classes show `P G [TP green / FP red / FN red]` with error counts shaded light-to-dark red by the shared per-scene error scale. Query ranks existing proposal masks, applies the prompt as label, colors results with a min-max normalized plasma heatmap, and shows `P N  G -`; empty results render nothing with an explicit No-matches status. Geometry, counts, class rows, and panels derive from one view snapshot per state revision. The viewer keeps the official submission unchanged; AP stays the official unthresholded score. Every successful prediction run now auto-exports missing ground truth and evaluates itself (warn-only on failure). Center renderer is `gui.SceneWidget` backend + per-instance `Label3D` labels (`spellbook/utils/scene_widget.py`, `open3d==0.19.0`), replacing frozen mesh text.

---

## Commands

```bash
# Restore/verify all canonical files for the active protocol
python spellbook/download_scans.py --scene $(PYTHONPATH=spellbook python -c \
    'from evaluation.benchmark import PREDICTION_EVALUATION_SCENES; print(*PREDICTION_EVALUATION_SCENES)') \
    --workers 4 --segments 8

# Comparable prediction (omitted --scene = fixed 16-scene tuple; official 198 classes)
python spellbook/main.py --predict --benchmark ScanNet200 --models mosaic3d \
    --run-id baseline16-mosaic3d --issue 33

# Ground truth export
python spellbook/evaluation/evaluate.py export-gt --scene 0046_00 --benchmark ScanNet200

# Evaluation (scenes/methods from run.json)
python spellbook/evaluation/evaluate.py evaluate --benchmark ScanNet200 \
    --run-id baseline16-mosaic3d --models mosaic3d

# AP sidecars + pooled global F1 thresholds (visualizer navigator + Filter)
python spellbook/evaluation/evaluate.py score-sidecars --benchmark ScanNet200 \
    --run-id baseline16-mosaic3d --models mosaic3d --scenes 0046_00 --missing-only

# Rank comparable ScanNet200 runs (AP primary)
python spellbook/evaluation/runs.py rank --benchmark ScanNet200 --method mosaic3d --metric ap

# Visualization (empty 3D until Enter on a scan; left scene/scan/method/run tree)
python spellbook/main.py --visualize

# Optional packed mask caches for the visualizer (derived only; official .txt untouched)
python spellbook/utils/prediction_masks.py build --benchmark ScanNet200 \
    --run-id baseline16-openyolo3d --models openyolo3d --scenes 0046_00 --missing-only

# Shared frame extraction (main checkout only; multi-GPU; never from engine worktrees)
python spellbook/main.py --extract-frames --scene 9004 9009

# Reconstruction batch (shared frames, then parallel engines; logs under spellbook/tmp/logs/)
python spellbook/main.py --scene 9004 9009 --engine open3d metashape
# Docker engines (rtabmap, isaac, bundlefusion) add --foreground, see decision 20

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

- Per-method environments, repositories and images: tables above; reproducible commands in `spellbook/docs/setup.md`.
- `3disspellbook` conda env (visualization, orchestration, eval, ZED, Open3D CUDA) — `spellbook/environment.yaml`; **activate it before any ZED SDK work** (its `activate.d/zed_sdk_env.sh` sets `ZED_DIR` and `LD_LIBRARY_PATH`; calling the env's python directly makes `zed.open` fail with CALIBRATION FILE NOT AVAILABLE)
- Label mapping: `/data/scannetdata/v2/scannetv2-labels.combined.tsv`
- GitHub CLI `gh` (issue workflow), `~/.local/bin/gh`
- Secrets: `spellbook/.secrets.env` (mode 600, gitignored)
