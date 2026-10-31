# Mosaic3D recovery: what is restored, what is inferred

Sources: `prediction-b.md` §1, issues #10, #33, #43 (the `genuine` rescore in #33 batch 2 was rolled back and is left out).

## Restored from the record (verbatim values)
| item | value | source |
|---|---|---|
| encoder | `/data/mosaic3d/ckpts/sc.ckpt`, PPT condition `ScanNet`, grid 0.02 | #43 2026-09-08 |
| proposals | frozen Mask3D `/data/openins3d/checkpoints/scannet200_val.ckpt`, conf 0.001 | #43 2026-09-08 |
| classification | mean-softmax, prompt `a {c} in a scene` | #43 2026-09-08 |
| defaults | `MIN_MASK_POINTS=100`, `POINT_LIMIT=1_500_000`, `INSTANCE_HEAD=mask3d` | master `_mosaic3d_run.py` |
| presets | `h3-proxyrescore={"rescore_mode":"geom1"}`, `b2-geom2={"rescore_mode":"geom2"}`, explicit params win | #33 2026-09-10 and 09-12 |
| kept repairs | npz v2 (`PROPOSAL_NPZ_FORMAT=2`, versioned keys, scored reads fail fast on v1), shape-probed `ppt_conditions` | #33 2026-09-12 |
| t3 cause | `spunet34c.ckpt` table is `[3,256]`, `sc.ckpt` is `[1,256]` | #33 2026-09-10 (checked on disk) |

## Rebuilt from Mosaic3D's own code (`spellbook/predict/models/_mosaic3d_scene.py`)
The lost `scripts/run_custom_scene.py` (sha `05980909…`) cannot be recovered. The rebuild uses only Mosaic3D modules:
- **Net:** hydra `eval` + `experiment=train_spunet_multidata_ppt data=sc model/clip=recap_clip`, `net.*` keys, strict load. Both checkpoints load with all keys matched.
- **PPT order for 3 conditions:** `["ScanNet","ARKitScenes","ScanNetPP"]`, from `configs/data/sc+ar+sc++.yaml`. The checkpoints store no hyperparameters.
- **Input:** the val transforms of `configs/data/sc.yaml`: `CenterShift(apply_z)`, `NormalizeColor`, feat = color, plus `grid_size` sparsify.
- **Scoring:** `DenseLanguageLitModule._update_instance_segmentation_metrics` logic: logits = feat · textᵀ (`normalize_input: false`), softmax per point, mean over the mask, then max gives (score, class). Classes whose name contains "other" use the prompt `other` (upstream `use_prompt` rule).
- **DBSCAN head (not the default):** default profile `eps 0.30, min_points 50` [inferred, from #10 "eps 0.30-0.40 m"]. Structural classes keep `eps 0.40, min_points 100, max_extent 8.0`.
- **Mask3D proposals:** the OpenIns3D fork exposes `prepare_data_pcd(points, colors, device)` and `map_output_to_pointcloud(mesh, outputs, inverse_map, …)`. The old call site `prepare_data(pcd_rgb, device)` no longer matches that API, and a bare `except` hid the error by writing empty proposals. `_mask3d_proposals.py` now calls `prepare_data_pcd` and copies the threshold rule of `map_output_to_pointcloud` (label < 200, conf = max softmax × mean in-mask sigmoid > threshold). It also keeps the per-mask conf, written as v2 `scores`.

## Inferred (marked `# [inferred]` in code)
**Evidence for geom1/geom2:**
- "frozen label-blind geom1 rescore, mask3d head only", "proxy rescore" (h3)
- tail AP rose from 0.057 to 0.086
- geom2 is "standalone, REPLACES geom1" and adds +0.00097 AP
- `genuine` = "mask_conf * p_lang, no geometry" scored worse (0.0981)
- geom1 was built in batch 1, before npz v2 existed, so it can only have used the binary masks and points

**Formulas chosen:**
- **geom1:** `score = p_lang × c`. `c` is the share of the mask's points that lie in its largest 26-connected component on a 5 cm voxel grid. Rationale: it is label-blind and geometric. At conf 0.001, Mask3D returns many scattered, fragmented masks, and demoting them helps ranking across all classes, including tail classes.
- **geom2:** `score = p_lang × c × n/(n+200)`, where n is the mask's point count. Rationale: geom1 plus a saturating size prior, which demotes the tiny fragments that conf 0.001 lets through. Still label-blind, and it does not use the Mask3D score (that would make it `genuine`).
- **`prompt_mode=ensemble_bg`** (h2): templates `a {c} in a scene | a {c} | a photo of a {c} | there is a {c} in the scene`, averaged per class. Two background prompts are added to the softmax and dropped before the argmax.
- **`dedup_iou`** (h1): greedy NMS within each class on the final scores. Survivors keep proposal order.
- **Cache key:** `scratch/mask3d_proposals/<scene>/mask3d_masks.conf<conf>.v2.npz`, one file per confidence.

**Acceptance (±0.005 AP, recovery-plan tolerance; bit-exact is impossible):**
- `{}`: 0.09687 / 0.15970 / 0.23133
- geom1: 0.106867 / 0.17745 / 0.24786
- geom2: 0.107841 / 0.180116 / 0.256019

If geom1 or geom2 misses, tune `GEOM_VOXEL` / `GEOM2_HALF_POINTS` first.

## Environment blocker (not changed by this restore)
The mosaic3d env has `transformers 5.20.2` (installed 2026-10-29 21:10) with torch 2.4.1. As a result, `is_torch_available()` is False, and the Recap-CLIP HF tokenizer fails with "PyTorch is not installed". Pin `transformers<5` (for example 4.46.x) before any GPU run. spconv's implicit GEMM is CUDA-only, so the encoder cannot be smoke-tested on CPU.
