# Recovery report B: Mosaic3D, Open3DIS, sam3dis

Read-only research, 2026-10-29. Sources: master git history, `spellbook/tmp/issues/*.md`, and the upstream clones in `/data`.

Also checked, with nothing found: the `origin` refs (master only), `git worktree`, stash, the second clone `/localhomes/starke/pcpool/ScanNet` (same master, no extra commits), the opencode db and snapshot (they hold only issue text), the Claude transcripts, the public GitHub `rolfstarke/Mosaic3D` (`main` = `39c9d9d` only), and `rolfstarke/{Open3DIS,sam3dis}` (private or missing, needs auth). No `run.json` survives on disk, because `/data/scannetdata` has no `derived/`.

Wrapper SHA check: `sha256(git show <c>:_mosaic3d_run.py)` at `4e5dda0` = `ad2698e3…`. This equals the wrapper SHA in the 10-scene `baseline-mosaic3d` run.json quoted in #33, so the method of matching a commit to a run is valid. The Open3DIS baseline wrapper SHA `d2c6027e…` matches no master commit.

---------------------------------------------------------------------------------------------------
## 1. Mosaic3D (#10, #33, #43)

### 1.1 Timeline
| date | source | event |
|---|---|---|
| 2026-08-08 | `01a8c7d` | First wrapper: DBSCAN path via `run_custom_scene.run_inference`, `spunet34c.ckpt`, `CONDITION="ARKitScenes"`, `MIN_MASK_POINTS=20` |
| 08-09 | #10 | Closed as a lower bound (AP 0.067/0.251/0.549, ScanNet20). Fixes identified but not adopted: ScanNet condition, mean-softmax scores bunched, DBSCAN eps |
| 08-30 | `96af588` | `load_overrides` allowlist `{grid_size,point_limit,min_mask_points,condition}`, run-scoped scratch |
| 08-31 12:19 | `5a3f483` | `CONDITION="ScanNet"` |
| 08-31 15:10 | `683904a` | Mask3D head: `_mask3d_proposals.py` (OpenIns3D env), `INSTANCE_HEAD="mask3d"`, `MASK3D_CONFIDENCE=0.001`, `MIN_MASK_POINTS=100`, `classify_mask3d_proposals` |
| 08-31 10:26 / 13:52 | #43 | Attempt 1, DBSCAN: 0.099/0.189/0.305. Attempt 2, `spunet34c` + Mask3D: 0.256/0.359/0.378 (10 scenes) |
| 08-31 19:33 | `5b16db5` | `CHECKPOINT=sc.ckpt`, `checkpoint` override, proposal cache moved to `scratch/mask3d_proposals/<scene>/` |
| 08-31 19:37 | `4e5dda0` | `ppt_conditions=["ScanNet"] if ckpt endswith sc.ckpt`. **Wrapper sha `ad2698e3…` = 10-scene baseline** (#33 21:16, 09-01) |
| 08-31 22:53 | #43 | Canonical 10-scene `baseline-mosaic3d` 0.228/0.324/0.334. run.json: ScanNet `4e5dda04`, Mosaic3D `6fe85c8c…` (lost local commit), `run_custom_scene.py` sha `05980909…` |
| 09-01 | `affaf49`, `70cc205` | CLIP feature publish (DBSCAN path only). 16-scene protocol |
| 09-08 00:34 | `8ae05b8` | Worktree merge (wrapper sha `260b9f2e…` = master HEAD). Mask3D path unchanged vs `4e5dda0` |
| 09-08 10:30 | #43 | **BASELINE** `baseline-mosaic3d` (16 scenes, `{}`): AP 0.09687 / AP50 0.1597 / AP25 0.2313, 621 masks. run.json `scannet_commit=8ae05b8`. Provenance fix `0d1864d` (runs.py `spunet34c`→`sc.ckpt` + Mask3D ckpt), **not in master** |
| 09-10 16:30 | #33 | Batch 1: t1 maskconf0.2 0.09465, t2 minmask30 0.09687 (tie), h1 dedup0.5 0.09585, h2 `ensemble_bg` 0.09674, **h3 `rescore_mode=geom1` 0.10687 PROMOTED**, t3 spunet34c failed 0/16 (`[3,256]` vs `[1,256]`). Preset `h3-proxyrescore`, `WINNER_PRESETS`, `MosaicWinnerPresetTests` |
| 09-12 04:45 | #33 | Batch 2: **b2-geom2 `{"rescore_mode":"geom2"}` 0.107841 PROMOTED (CURRENT BEST)**. Losers: grid0015 0.08548, maskconf005 0.10173, maskconf01 0.10465, genuine 0.09807, spunet34c 0.10636 (now 16/16 via shape-probed selector). **Rollback:** `genuine` enum/branch, product helper, tests removed. **Kept:** npz v2 (`PROPOSAL_NPZ_FORMAT=2`) and shape-probed `ppt_conditions` |

### 1.2 Target state
- **Baseline:** master-HEAD `_mosaic3d_run.py` (= `8ae05b8`, Mask3D path byte-equal to `4e5dda0`), plus the `0d1864d` provenance fix, plus the lost `run_custom_scene.py` (sha `05980909…`).
- **Best:** the wrapper state after the batch-2 close-out, which includes:
  - `rescore_mode ∈ {none, geom1, geom2}` (no `genuine`)
  - `WINNER_PRESETS = {"h3-proxyrescore": {"rescore_mode":"geom1"}, "b2-geom2": {"rescore_mode":"geom2"}}`
  - `preset` allowlist key, with explicit params winning the merge
  - the knobs `dedup_iou`, `prompt_mode` (`ensemble_bg`) and `mask3d_confidence` from batch 1, all default-off
  - npz v2
  - the shape-probed PPT selector

  With `{}` the output stays byte-identical to the baseline.

### 1.3 Gaps vs master HEAD (`git log -S` finds none of these except `ppt_conditions`→`4e5dda0`)
1. `/data/Mosaic3D/scripts/run_custom_scene.py` is missing from the public fork (the scripts folder holds only `mask_mapping.py` and `visualize.py`). The wrapper imports it in two places, and `MODEL_RESOURCES`/`_REQUIRED_EXTERNAL` require it, so preflight fails today.
2. `runs.py:50-53` still lists `spunet34c.ckpt` (the `0d1864d` fix is lost).
3. Missing: `rescore_mode` (geom1/geom2), `WINNER_PRESETS`, `preset`, `dedup_iou`, `prompt_mode`, `PROPOSAL_NPZ_FORMAT=2` scored npz, the shape-probed `ppt_conditions`, and `MosaicWinnerPresetTests`.
4. **Formulas not recorded anywhere.** For geom1/geom2 the only evidence is "frozen label-blind geom1 rescore, mask3d head only", "geom2 REPLACES geom1", and "proxy rescore". `genuine` = `mask_conf * p_lang, no geometry` (rolled back). These must be re-derived, with the AP numbers as the acceptance test.
5. Cache-key defect: HEAD caches proposals at `scratch/mask3d_proposals/<scene>/mask3d_masks.npz` regardless of `--confidence`. The t1 and b2-maskconf runs scored differently from baseline, so the lost code either keyed by confidence or stored scores (npz v2) and filtered on read. Restore one of these.

### 1.4 Restore spec
**`run_custom_scene.py`.** Recreate it inside spellbook, e.g. `spellbook/predict/models/_mosaic3d_scene.py`, following the recovery plan. Then point `MODEL_RESOURCES` and `_REQUIRED_EXTERNAL` at it. Signatures, from the wrapper call sites at `01a8c7d`, `affaf49` and `4e5dda0`:
- `CLIP_CFG: dict`: open_clip, `CLIP_CFG["model_id"]`, 768-d. Built with `src.models.utils.clip_models.build_clip_model(CLIP_CFG, device=)`.
- `build_net() -> net`. SpUNet34C + PPT (`src/models/networks/ppt/model.py`, `embedding_table = nn.Embedding(len(conditions),256)`). Called with a condition, and returns a `point` with `.sparse_conv_feat.features` and `.v2p_map`.
- `load_backbone_ckpt(net, ckpt_path)`.
- `run_inference(scene_ply, classes, checkpoint, device, condition=, grid_size=, up_axis=, class_profiles=, net=None, clip_model=None) -> (objects, _, _)`.
  - Per-point CLIP argmax, then per-class DBSCAN. Eps is 0.30-0.40 m: `STRUCTURAL_CLASS_PROFILES` gives wall/floor/ceiling `eps=0.40, min_points=100, max_extent=8.0`.
  - Score = mean per-point softmax (#10).
  - `objects` = list of `{"point_indices","class_name","score"}`.
- `classify_mask3d_proposals(pointcloud, classes, checkpoint, device, masks[N,M] bool, condition=, grid_size=, up_axis=, min_mask_points=, ppt_conditions=None) -> (objects, _)`. Per #43, "mean-softmax class scores, prompt `a {c} in a scene`".
  - `ppt_conditions=["ScanNet"]` builds the 1-condition table for `sc.ckpt`.
  - Batch-2 shape-probed selector: read `embedding_table.weight` shape from the checkpoint. `[1,256]` gives `["ScanNet"]`. `[3,256]` gives the 3-condition list for `spunet34c` (`sc+ar+sc++`). The order is unrecorded; check against `configs/model/*ppt*.yaml` / `train_spunet_multidata_ppt.yaml`.
- Score-ordering caveat (#10): per-point softmax means bunch at 0.06-0.28 (near 1/22).

**`_mosaic3d_run.py`.** Keep HEAD defaults:
- `CHECKPOINT=/data/mosaic3d/ckpts/sc.ckpt`, `CONDITION="ScanNet"`, `GRID_SIZE=0.02`, `INSTANCE_HEAD="mask3d"`
- `MASK3D_CONFIDENCE=0.001`, `MIN_MASK_POINTS=100`, `POINT_LIMIT=1_500_000`
- Mask3D ckpt `/data/openins3d/checkpoints/scannet200_val.ckpt`

Add:
- Allowlist keys `preset, rescore_mode, dedup_iou, prompt_mode`.
- `WINNER_PRESETS` as in 1.2. Merge as `{**WINNER_PRESETS[preset], **explicit}`. `{}` means no rescoring.
- `rescore_mode` default `none`, choices `geom1|geom2`, mask3d head only. Label-blind geometric proxy applied to the per-mask score (re-derive).
- `PROPOSAL_NPZ_FORMAT=2`: versioned keys (`masks`, `scores`, `format`). Readers that need scores fail fast on v1.
- Cache key includes confidence.

**`_mask3d_proposals.py`** also writes the Mask3D scores (v2).

**`runs.py` `_REQUIRED_EXTERNAL["mosaic3d"]`** = (`run_custom_scene` path, `/data/mosaic3d/ckpts/sc.ckpt`, `/data/openins3d/checkpoints/scannet200_val.ckpt`).

**Tests:** `MosaicWinnerPresetTests`:
- `{}` is byte-identical to the baseline
- the preset expands
- explicit params win
- an unknown preset or `genuine` is rejected

**CLI**
```
python spellbook/main.py --predict --benchmark ScanNet200 --models mosaic3d --run-id repro-baseline-mosaic3d --issue 43 --foreground          # target AP 0.09687 / 0.15970 / 0.23133
echo '{"mosaic3d":{"rescore_mode":"geom2"}}' > /data/tmp/m3d_best.json   # == preset b2-geom2
python spellbook/main.py --predict --benchmark ScanNet200 --models mosaic3d --run-id repro-best-mosaic3d --run-parameters /data/tmp/m3d_best.json --issue 33 --foreground   # target 0.107841 / 0.180116 / 0.256019
```
Intermediate check: `{"mosaic3d":{"rescore_mode":"geom1"}}` should give 0.106867 / 0.17745 / 0.24786.

---------------------------------------------------------------------------------------------------
## 2. Open3DIS (#3, #15, #24, #36, #60)

### 2.1 Timeline
| date | source | event |
|---|---|---|
| 08-08 | `01a8c7d` | ov3dis_scene 2D-only wrapper (`ov3dis_scene4.yaml`, p2d True / p3d False) |
| 08-09 | #3 | `img_dim=640x480`, `rgb_img_dim=1296x968`. `scaling_mapping` added to `ov3dis_scene` branches of `clustering.py` and `tools/refine_grounding_feat.py` (v×968/480, u×1296/640). 18-class AP 0→0.205 |
| 08-10 | `edb0fef` (#24) | `os.makedirs(dirname(split_path))` before writing |
| 08-30 | `96af588` (#15) | Cross-process method lock serializes open3dis scenes (runner.py:380) |
| 08-30 | fork `5548ab5` (lost) | Runtime patches: clustering/grounding tools, ov3dis_scene mapping, data symlink |
| 09-01 | #36 | Legacy 2D-only diagnostics 0.0994/0.2019/0.2664 (no manifest) |
| 09-10 | fork `459aa14`,`55df438`,`12e05a8` (lost) | `scripts/prepare_scannet200_scene.py` (vendored segmentator superpoints, ISBNet class-agnostic proposals + dc_feat via `head_scannetv2_200_val.pth`), isbnet yaml, absolute ckpt paths, 16-scene protocol, config-driven `skip_clustering`/`emit_final` gates |
| 09-10 | #60 | Smoke 1 AP 0 (masks/labels misaligned with p3d). Fix: step-4 fusion `emit_final`, `proposals.agnostic=False`, read `final_result_hier_agglo`. Smoke 2: 14/19 TP |
| 09-10 21:21 | #60 / `290a87f` (lost) | Integration "official Open3DIS 2D+3D path: ISBNet, step-4 fusion, parallel scenes", wrapper sha `d2c6027e…`, method lock removed, per-scene cwd trackers |
| 09-11 (posted 09-12 00:20) | #60 | **BASELINE = BEST** `baseline-open3dis`: AP 0.1465 / AP50 0.2198 / AP25 0.2847 (scannet `290a87f`, external `spellbook-baseline@12e05a8`) |
| 09-12 | #15 / `0ca2762` (lost) | Drop tracker entries whose stage outputs are missing (post-baseline robustness fix, output-neutral; include it) |
No optimization batch was ever run (#36 has no batch), so best = baseline.

### 2.2 Target state
The wrapper at `290a87f`, plus `0ca2762`, plus the fork branch `spellbook-baseline@12e05a8`.

Master HEAD still has the old 2D-only path: `ov3dis_scene4`, `hier_agglo` zipped with `refined` features, own `a photo of a {c}.` CLIP argmax, global tracker truncation, and the method lock. Under the 2D+3D config this is exactly the misaligned "previously-broken custom path".

### 2.3 Gaps (`emit_final`, `skip_clustering`, `scannet_loader`, `final_result_hier_agglo` never in master)
1. The wrapper needs a full rewrite to the official `scannet200` path.
2. `runner.py:380-381` still holds the open3dis method lock. Remove it.
3. `runner.py:73-77` and `runs.py:61-64` point at `ov3dis_scene4.yaml` + `ov3dis_loader.py`. Change them to `configs/scannet200.yaml` + `open3dis/dataset/scannet_loader.py` + the ISBNet `head_scannetv2_200_val.pth` (path not recorded; the fork used absolute paths).
4. The Open3DIS fork patches are lost and the `/data/Open3DIS` clone is upstream `bc2a3b3`. Upstream `generate_3d_inst.py:279 if True:` / `:296 if False:` are the two lines that need gating.
5. `prepare_scannet200_scene.py` is lost. Rebuild it from `docs/DATA.md:73-90`: ISBNet `tools/test.py configs/scannet200/isbnet_scannet200.yaml …/head_scannetv2_200_val.pth` creates `isbnet_clsagnostic_scannet200` + `dc_feat_scannet200`. Superpoints come from ISBNet `dataset/scannet200`.

### 2.4 Restore spec
**Fork branch `spellbook-baseline`, on `/data/Open3DIS`, never on main.**
- (a) `#3` `scaling_mapping` in the `ov3dis_scene` branches. Only needed for the legacy path.
- (b) `tools/generate_3d_inst.py`:
  - `if True:` becomes `if not cfg.<gate>.skip_clustering:`. In that case load `cluster_dict` from `hier_agglo`.
  - `if False:` becomes `if cfg.<gate>.emit_final:`.
  - The key location is unrecorded. Pick `exp:` and default both to the upstream behaviour (False/False).
- (c) Data under `data/Scannet200/{Scannet200_2D_5interval/val, Scannet200_3D/val/{original_ply_files,superpoints,isbnet_clsagnostic_scannet200,dc_feat_scannet200,groundtruth}}` for the 16 scenes. Old scenes are retired.
- (d) `prepare_scannet200_scene.py` and an isbnet yaml with absolute checkpoint paths.

**`_open3dis_run.py`.** Settings from #60 (official defaults, no retune):
- Base config `configs/scannet200.yaml`, `scannet_loader.py`. Per run: `data.split_path`, `num_classes=198`, `exp.exp_name`, and `exp.save_dir = Open3DIS/exp/<run-id>/<scene>`.
- Every tool runs with `cwd=Open3DIS/exp/<run-id>/<scene>/cwd` (#15), so trackers are per scene. Before each stage, drop tracker lines whose `.pth` is missing (`0ca2762`).
- Stages:
  1. `tools/grounding_2d.py` (Grounded-SAM: GDINO SwinT_OGC box/text 0.4, SAM ViT-H `sam_vit_h_4b8939.pth`, CLIP ViT-L/14@336px 768-d)
  2. `generate_3d_inst.py` (cluster)
  3. `refine_grounding_feat.py` (`top_k 5`)
  4. `generate_3d_inst.py` with `skip_clustering=True, emit_final=True, proposals: {p2d:True, p3d:True, agnostic:False, refined:True}`
- Read `final_result_hier_agglo/<scene>.pth` → `ins` (RLE), `conf`, `final_class`, i.e. official mask-wise CLIP scoring (`final_instance: duplicate True, iou_overlap 0.9, top_k 300, scale_semantic_score 300`). Write the submission from those. No own CLIP re-classify.
- Keep `edb0fef` makedirs and the `#3` dims `img_dim [640,480]`, `rgb_img_dim [1296,968]`.

**`runner.py`:** remove the open3dis `lock_ctx`. **Tests:** cover the tracker reconcile and run-scoped cwd.

**CLI (baseline = best)**
```
python spellbook/main.py --predict --benchmark ScanNet200 --models open3dis --run-id repro-baseline-open3dis --issue 60 --foreground   # target 0.1465 / 0.2198 / 0.2847 (h/c/t 0.125/0.136/0.225)
```
Shortcut for comparison only: upstream ships `Scannet200_OpenVocab_ISBNet-GSAM` results (docs/DATA.md:11).

---------------------------------------------------------------------------------------------------
## 3. sam3dis (#69)

### 3.1 How it was wired (inferred; nothing is in master, `git log -S sam3dis` is empty)
- A separate worktree added `sam3dis` to:
  - `evaluation/benchmark.py` `PREDICTION_METHODS`, plus its hard self-check assert at `:178`
  - `runner.py` `MODEL_PYTHON`, `MODEL_RUN_SCRIPT`, `MODEL_NEEDS_FRAMES` (True: SAM 3 runs on posed frames), and `MODEL_RESOURCES` (the SAM 3 checkpoint)
  - `runs.py` `_WRAPPERS`, `_EXTERNAL_REPOS` and `_REQUIRED_EXTERNAL`
- Files, per the #69 provenance:
  - `spellbook/predict/models/_sam3dis_run.py`: adapter seam plus parameter plumbing
  - `_sam3dis_pool.py`: the batch-1 engine
  - `_sam3dis_query_build.py`, `_sam3dis_query_scan.py`, `_sam3dis_query_lift.py`: batch 2, 32 tests
  - `spellbook/significance_ap50.py`
- The R4 merge is `sam3dis/vendor/clustering_core.py:170-222` (connected components) in the external `/home/rolf/GIT/sam3dis`, which is lost. Its knobs (`visi`, `recall`) follow Open3DIS clustering, so it is likely vendored from `Open3DIS/open3dis/src/clustering`.
- Baseline R4 was produced standalone. `sam3dis-adapter-r1` reproduced it bit-identically through the spellbook adapter.
- Env: `/data/sam3dis/conda/envs/sam3dis` (py3.12, install in progress). SAM 3 code: `/data/sam3` (`967fdd6`).
- Lift cache naming: `_i10_tg0.2.pt` (interval, tau_geom). Query cache: `<root>/derived/cache/sam3dis_query/`.

### 3.2 Timeline
| date | source | event |
|---|---|---|
| ≤09-15 | #69 body | **BASELINE** `sam3dis-r4` AP 0.170720 / 0.306429 / 0.451297, params below. `sam3dis-adapter-r1` bit-identical |
| 09-16 09:11 | #69 | Batch 1, pool engine behind `proposal_engine`: t1 0.25817, t2 0.25154, t3 0.24960, h1 0.26964, h2 0.27213, **h3 `improve1-sam3dis-h3-miou20-mv2` 0.27518716 / 0.40559053 / 0.50842816 = BEST**. 414 tests |
| 09-18 14:39 | #69 | Batch 2: 48 variants, **no promotion**. Rescoring knobs + query path added default-off (h3 bit-identical). 563 tests |

### 3.3 Target state
The adapter (R4 path), plus the batch-1 pool engine with its new knobs, all defaulting to legacy. Batch-2 knobs and the query path are excluded: they are not needed for the best run and none were promoted (optional, default-off).

### 3.4 Gaps
Everything:
- the registry entries in 3 files
- the 2 core modules plus the vendored `clustering_core`
- the tests
- the external `sam3dis` repo (lift: SAM 3 frame masks → ScanNet superpoints `*.segs.json`)

No formulas are recorded beyond the stage list and knob names, so the implementation is a re-derivation. The bit-exact numbers below are the acceptance test.

### 3.5 Restore spec
- **Constraints (#69):** no ScanNet-trained 3D nets, features or checkpoints. ScanNet Felzenszwalb superpoints are allowed. The external SAM 3 checkpoint stays untouched.
- **R4 defaults, verbatim:**
  ```
  {"chunk":16,"device":"cpu","interval":10,"labels_per_mask":10,"min_mask_points":237,
   "presence_prune":0.0,"recall":0.979,"score_threshold":0.05,"tau_geom":0.2,
   "tau_label":0.1,"visi":0.6}
  ```
- **Lift stage (GPU, cached):** SAM 3 on every 10th frame, `score_threshold 0.05`, prompts in chunks of 16. Masks are lifted to superpoints with `tau_geom 0.2`, cached per `_i{interval}_tg{tau_geom}.pt`.
- **R4 merge:** CC merge with `visi 0.6` / `recall 0.979`, then `tau_label 0.1`, then up to 10 labels per mask as independent rows (no argmax), with `min_mask_points 237`.
- **Pool engine (`proposal_engine="pool"`):** dedup → multi-view scoring → NMS → top-N → top-K labels.
  - Knobs: `score_mode` (`maxjoint_median` | `noisyor_median_frames`), `nms_iou`, `nms_mode` (`hard`|`linear`|`gaussian`), `nms_sigma`, `nms_score_floor`, `top_n`, `tau_support`, `view_power`, `frame_term_mode` (`power`), `label_sharpen`, `label_margin`, `class_cap`, `match_iou`, `min_views`, `pool_tau_geom` (split from lift `tau_geom`, must not touch the cache key), `geom_power` (=1.0), `min_verts`.
  - All default to the legacy/no-op value.
- **Best (verbatim):**
  ```
  {"sam3dis": {"proposal_engine": "pool", "score_mode": "maxjoint_median", "nms_iou": 0.5, "nms_mode": "linear", "top_n": 400, "tau_support": 0.40, "view_power": 0.25, "label_sharpen": 0.5, "match_iou": 0.20, "min_views": 2, "min_mask_points": 100}}
  ```
- **Intermediate checks, all 16 scenes:**

  | run | params | AP |
  |---|---|---|
  | t1 | `proposal_engine=pool, score_mode=maxjoint_median, nms_iou=0.5, top_n=400, tau_support=0.35, view_power=0.25, min_mask_points=100` | 0.25817123 |
  | h1 | t1 + `label_sharpen=0.5, tau_support=0.40` | 0.26963680 |
  | h2 | h1 + `nms_mode=linear` | 0.27212910 |

**CLI**
```
echo '{"sam3dis":{"chunk":16,"device":"cpu","interval":10,"labels_per_mask":10,"min_mask_points":237,"presence_prune":0.0,"recall":0.979,"score_threshold":0.05,"tau_geom":0.2,"tau_label":0.1,"visi":0.6}}' > /data/tmp/s3_base.json
python spellbook/main.py --predict --benchmark ScanNet200 --models sam3dis --run-id repro-baseline-sam3dis --run-parameters /data/tmp/s3_base.json --issue 69 --foreground   # 0.170720 / 0.306429 / 0.451297
python spellbook/main.py --predict --benchmark ScanNet200 --models sam3dis --run-id repro-best-sam3dis --run-parameters /data/tmp/s3_best.json --issue 69 --foreground    # 0.27518716 / 0.40559053 / 0.50842816
```
`s3_best.json` holds the best params above.

---------------------------------------------------------------------------------------------------
## 4. Cross-cutting
- `--run-parameters` JSON maps model → params. `runner.py:339` writes `<model>.parameters.json`.
- Bit-exact reproduction is impossible for Mosaic3D geom1/geom2 and sam3dis (formulas lost). Use the ±0.005 AP tolerance from the recovery plan. Open3DIS should land near 0.1465 once the official 2D+3D path is restored.
- `runs.py rank` fails on manifests of methods not in `PREDICTION_METHODS` (#33). Register sam3dis before ranking.
