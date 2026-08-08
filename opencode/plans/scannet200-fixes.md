# Plan: Fix 3 wrapper bugs + expand to ScanNet200 (20 scenes, 189 classes)

Target executor: build agent. Repo root: `/home/rolf/GIT/ScanNet`. All custom code lives in
`spellbook/`. Never touch `BenchmarkScripts/` or `SensReader/` except the already-applied Python 3
ports. Do not modify the official ScanNet evaluator except the already-patched copy in
`spellbook/eval/`.

Context: a full 18-class NYU40 ScanNet benchmark run (8 val scenes, 4 models: mosaic3d, openins3d,
openyolo3d, open3dis) produced low AP scores. Deep investigation found the low scores are caused
by bugs in OUR integration/wrapper code (not model limitations), except mosaic3d whose gap is
mostly a fundamentally different/unreleased algorithm (documented, not fixed here). User approved:
fix openyolo3d intrinsics, fix open3dis rescale, fix openins3d threshold + install ODISE, and
expand the benchmark to ScanNet200 (189 val-present classes) on 20 scenes.

## Phase 1 — openyolo3d intrinsics fix (trivial, confirmed correct via maintainer GitHub issues)

File: `spellbook/predict/models/_openyolo3d_run.py`

Current (bug): symlinks `intrinsics.txt -> frames_dir/intrinsic_depth.txt`
(`_build_scratch_scene`, `links` dict, key `"intrinsics.txt"`).

Fix: symlink to `frames_dir/intrinsic_color.txt` instead. OpenYOLO3D's `WORLD_2_CAM` /
`adjust_intrinsic` expects `intrinsics.txt` to hold the COLOR-resolution intrinsic matrix
(1296x968) and rescales it internally to depth resolution (640x480) — confirmed by the
maintainer's own example data in GitHub issue #9, a community fix in issue #14, and the parent
OpenMask3D project's config (`intrinsic_path: intrinsic_color.txt`).

After the fix, sanity-check: `fx≈1170, cx≈648` in the linked file (not `fx≈578, cx≈319`).

Verify: re-run openyolo3d on scene0568_00 only, confirm instance count/labels look reasonable
(previously 489/568 instances had confidence 0.0000 — that fraction should drop sharply).

## Phase 2 — open3dis rescale fix (2 files, mirror the existing working branch)

Repo: `/home/rolf/GIT/Open3DIS` (outside ScanNet repo — this is a separate checkout, editing it is
fine, it's a model repo not the ScanNet repo).

File 1: `open3dis/src/clustering/clustering.py`, function `process_hierarchical_agglomerative_nospp`,
branch `elif "ov3dis_scene" in cfg.data.dataset_name:` (~line 668-673).

Current (bug):
```python
elif "ov3dis_scene" in cfg.data.dataset_name:
    mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
    mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)
```

Fix — add the same rescale the working `scannet200` branch does (~line 648-655), adapted for this
branch's depth-resolution mapping:
```python
elif "ov3dis_scene" in cfg.data.dataset_name:
    mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
    mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)
    new_mapping = scaling_mapping(
        torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[1], rgb_img_dim[0]
    )
    mapping[:, 1:4] = torch.cat((new_mapping, mapping[:, 3].unsqueeze(1)), dim=1)
```
Note the arg order deliberately differs from the `scannet200` branch's call (which was found to be
dimensionally transposed for this repo's `compute_mapping_torch` output columns `[v, u, valid]`) —
use `img_dim[1], img_dim[0], rgb_img_dim[1], rgb_img_dim[0]` here so scale_x maps depth-height to
color-height and scale_y maps depth-width to color-width correctly. Verify empirically after the
change (see below) rather than trusting this blindly — if masks still look warped, try the
non-transposed arg order `img_dim[0], img_dim[1], rgb_img_dim[0], rgb_img_dim[1]` instead and
compare.

File 2: `tools/refine_grounding_feat.py`, same `ov3dis_scene` branch (~line 203) — apply the
identical rescale so CLIP instance features are pulled from correctly-aligned image crops.

Verify: re-run open3dis on scene0568_00 only. Pick 2-3 predicted masks, compute best IoU against
GT instances (via `spellbook/eval/gt/scene0568_00.txt`) — should see IoU > 0.3 for reasonable masks
(previously median best-IoU was 0.016, essentially zero overlap).

## Phase 3 — openins3d fix (two parts; ODISE install is the risky one)

File: `spellbook/predict/models/_openins3d_run.py`

Part A (safe, do first): lower `MASK_CONFIDENCE_THRESHOLD` from 0.5 — this is a demo-mode default,
not what the paper's benchmark pipeline uses (paper uses precomputed masks at ~0.001 recall
threshold, which we don't have). Tune empirically starting around 0.05-0.1 on scene0304_00 (the
scene that produced only 4 raw masks at 0.5) — pick a value that yields a reasonable instance count
without exploding runtime/memory.

Part B (risky, isolate from Part A): install ODISE into the `openins3d` conda env
(`/data/openins3d/conda/envs/openins3d`, python 3.9.25, torch 1.13.1+cu116) using the repo's own
`openins3d/build_lookup_odise.py` and `third_party/ODISE` (already vendored in
`/home/rolf/GIT/OpenIns3D/third_party/`). Requires building `detectron2` + Mask2Former CUDA ops
against torch 1.13.1+cu116 — expect this to be the most fragile step. Attempt it isolated in a
scratch/test invocation first, NOT inside the main wrapper, so a build failure doesn't block
everything else.

If ODISE build succeeds: switch `_openins3d_run.py` from `lookup.call_YOLOWORLD()` to
`lookup.call_ODISE()` (already an existing method on `Lookup`, `openins3d/lookup.py:30`) — this is
what the paper's reported ScanNet numbers actually used (README explicitly: "ODISE works better on
pcd-rendered images").

If ODISE build fails: fall back to Part A only (threshold fix) and clearly note ODISE was skipped
due to a build failure, do not silently degrade.

Verify: re-run openins3d on scene0304_00 and scene0412_00 (the two scenes with the 1-instance
anomaly), confirm instance counts are no longer near-zero.

## Phase 4 — download 12 more val scenes (independent, safe, do anytime)

Currently downloaded (8): scene0568_00/01/02, scene0304_00, scene0488_00/01, scene0412_00/01.

Download using the existing tool `/tmp/download-scannet.py` (same one used for the current 8
scenes — check `spellbook/PROJECT_STATUS.md` for the exact invocation pattern used previously).

Add these 12 scenes (next entries from `Tasks/Benchmark/scannetv2_val.txt` after the current 8),
giving 20 scenes total (~50GB, well within the ~191GB free budget at ~2.5GB/scene):
```
scene0217_00, scene0019_00, scene0019_01, scene0414_00, scene0575_00, scene0575_01,
scene0575_02, scene0426_00, scene0426_01, scene0426_02, scene0426_03, scene0549_00
```

After download, extract frames for all 12 new scenes the same way the existing 8 were done
(`spellbook/predict/frames.py`, via `main.py --predict` normal flow — frame extraction happens
automatically up front). Verify each new scene has the full 13-file set like the existing 8.

## Phase 5 — ScanNet200 plumbing (additive only — do not remove/break the 18-class path)

Confirmed via official docs (kaldir.vc.in.tum.de/scannet_benchmark) + local `scannet200_splits.py`:
- Use **189 classes** = `VALID_CLASS_IDS_200_VALIDATION` / `CLASS_LABELS_200_VALIDATION` from
  `BenchmarkScripts/ScanNet200/scannet200_splits.py` (NOT the full 200 — 11 categories are
  train-only and absent from val; official benchmark explicitly excludes them from val scoring).
- Label id column is the **raw ScanNet `id`** (not `nyu40id`) — read from the `id` column of
  `scannetv2-labels.combined.tsv` via `raw_category` lookup, same file already used for GT export,
  just a different target column.
- Instance evaluation ignores wall/floor (already true of the class list — they're id 1/2, not
  in the 189).
- Evaluation parameters are IDENTICAL to the 18-class benchmark: AP@0.25, AP@0.5, AP over
  [0.5:0.95:0.05], min_region_size=100 verts, distance_thresh=inf — only the class/id list changes.
- Also report head/common/tail sub-averages (66/68/66 classes by frequency) — the official
  leaderboard shows this breakdown; grouping data already exists in `scannet200_constants.py`
  (`HEAD_CATS_SCANNET_200`, `COMMON_CATS_SCANNET_200`, `TAIL_CATS_SCANNET_200`).

Required changes (all additive — keep existing 18-class behavior unchanged, add a mode/param):

1. `spellbook/predict/models/common.py`: `write_scannet_predictions` currently hardcodes
   `BENCHMARK_CLASS_LABELS`/`BENCHMARK_VALID_CLASS_IDS` (18-class NYU40) and raises ValueError on
   any unknown class name (`common.py:62-69`) — this is a hard blocker for ScanNet200 runs. Add a
   parallel 189-class label→id map sourced from `BenchmarkScripts/ScanNet200/scannet200_splits.py`
   `VALID_CLASS_IDS_200_VALIDATION`/`CLASS_LABELS_200_VALIDATION` (import or copy verbatim — do not
   hand-retype 189 entries, read them programmatically from that file to avoid transcription
   errors). Make `write_scannet_predictions` accept a `label_map`/`benchmark` parameter so callers
   choose which of the two class sets to use; default stays 18-class NYU40 (no behavior change to
   existing callers).

2. `spellbook/eval/export_gt.py`: currently maps `raw_category -> nyu40id`. Add a mode/flag to map
   `raw_category -> id` (raw ScanNet200 id) instead, gated by the same benchmark selector. Keep the
   flat `label*1000+instance_id` encoding — same format, just a different id.

3. `spellbook/eval/evaluate_semantic_instance.py` (ported evaluator): currently hardcodes the
   18-class `CLASS_LABELS`/`VALID_CLASS_IDS` at module level (mirrors official
   `BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py`). Add a ScanNet200 variant (new
   module-level constants sourced from `VALID_CLASS_IDS_200_VALIDATION`/
   `CLASS_LABELS_200_VALIDATION`) — either a second constants block selectable by param, or a
   sibling script `evaluate_semantic_instance_200.py` if easier to keep isolated. Reuse the same
   evaluation logic (already patched for the empty-array crash and `np.float` — do not re-introduce
   either bug). Add head/common/tail sub-average computation using
   `HEAD_CATS_SCANNET_200`/`COMMON_CATS_SCANNET_200`/`TAIL_CATS_SCANNET_200`.

4. `spellbook/eval/run_eval.py`: add a `--benchmark {18class,scannet200}` (or similar) switch that
   picks the right class list, GT files, and evaluator variant. Keep default = 18-class so existing
   results/behavior are reproducible.

5. `spellbook/main.py` / `spellbook/predict/runner.py`: allow passing the 189-class list as
   `--classes` for a `--predict` run (already supports arbitrary `--classes` via nargs+ — just need
   `common.py`'s writer to accept them, per item 1). No structural change expected here beyond what
   item 1 requires.

Verify: run a smoke test on scene0568_00 with a handful of ScanNet200-only classes (e.g. a class
present in the 189 but not the 18, like "office chair" or "pillow") through the full
predict→GT-export→eval chain, confirm no crash and a sane (non-nan) AP entry for that class if GT
contains it.

## Phase 6 — full run + re-evaluate

1. Predict all 20 scenes x 4 models with the 189-class ScanNet200 val list (`--classes` = the full
   list from `VALID_CLASS_IDS_200_VALIDATION`). Use `--gpu` multi-GPU dispatch as before (5 GPUs
   available). Expect longer runtime than the 8-scene/18-class run — size accordingly, consider
   running overnight or in background with logging to `/tmp/opencode/`.
2. Regenerate GT for all 20 scenes in ScanNet200 raw-id encoding (Phase 5 item 2).
3. Re-run `run_eval.py --benchmark scannet200` for all 4 models — produces
   `spellbook/eval/results_<model>_200.txt` (do not overwrite the existing 18-class results files).
4. Also re-run the ORIGINAL 18-class benchmark on the 20 scenes (not just 8) with the three fixed
   wrappers (openyolo3d, open3dis, openins3d) + unfixed mosaic3d, to isolate: how much did each fix
   improve the comparable-format 18-class numbers, before also looking at the new 189-class numbers.
5. Report both: (a) 18-class before/after per model (isolates fix impact), (b) new 189-class
   head/common/tail/avg AP/AP50/AP25 per model (closer to, but still not equal to, each paper's
   312-scene protocol — note that explicitly in the report, do not claim direct comparability).

## Phase 7 — cleanup

1. Update `spellbook/PROJECT_STATUS.md` with: the three wrapper fixes (what/why/file:line), the 12
   new scenes, the ScanNet200 189-class eval harness location and how it differs from the 18-class
   one, and the final before/after results tables.
2. Delete this plan file (`opencode/plans/scannet200-fixes.md`) once every phase above is executed
   and verified — per project convention, plans are deleted after execution, not stubbed out.
3. Update the todo list to reflect completion.

## Order of execution

Phases 1, 2, 4, 5 are independent and low-risk — can proceed in any order, ideally in parallel with
Phase 3's ODISE build attempt (Part B) which is isolated and may fail without blocking anything
else. Spot-check each of Phase 1/2/3 on 1-2 scenes individually BEFORE Phase 6's full 20-scene run,
so a broken fix doesn't waste a full multi-hour batch run. Phase 6 depends on all of 1-5 being done
and verified. Phase 7 is last.
