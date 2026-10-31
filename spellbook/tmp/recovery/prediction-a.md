# Recovery A — openyolo3d / openins3d / openmask3d prediction wrappers

Sources: `spellbook/tmp/issues/{002,035,044,054,068,004,014,034,045,037,046}.md`, `git log` on master,
sha256 of every historical wrapper blob vs SHAs quoted in run.json excerpts. Researched 2026-10-29.

## 0. Cross-cutting findings
- Working-tree diff of all three wrappers vs HEAD = repo path only (`/home/rolf/GIT/* -> /data/*`). Ignore for SHA matching.
- Wrapper SHA matches (blob sha256 == quoted run.json sha):
  | quoted sha | run | master blob |
  |---|---|---|
  | `6b9b6e25…` | openyolo3d `baseline-openyolo3d` (10-scene) | `96af588:_openyolo3d_run.py` == HEAD |
  | `cf3fb952…` | openins3d `baseline-openins3d` r01 (10-scene) | `96af588:_openins3d_run.py` == HEAD |
  | `9d177e77…` | openmask3d `baseline-openmask3d` attempt 4 (10-scene) | `22f7bd1:_openmask3d_run.py` |
  | `3bdfafe6` r02, `fd557d31` r03, `e7276251` r04/r05, `833f3abc` r06, `d5b0ee93` oi-base16, `4e8ffcda` oi-b01 | openins3d | **never in git** |
- `git log --all -S` finds NO trace of: `proposal_source`, `official_precomputed`, `path_to_3d_masks`, `vote_weighting`,
  `proposal_score_power`, `network2d_th`, `network3d_nms`, `use_precomputed`, `lookup_vocab`, `benchmark198`,
  `single_detection`, `scatter_mean`, `get_full_res_mask`, `test_unparseable_manifest_skipped_with_warning`.
  -> every post-baseline improvement must be re-written; nothing to `git show`-restore except what HEAD already has.
- External repos drifted: `/data/OpenYOLO3D` HEAD `24f344d` (quoted `f457dc3` absent); `/data/OpenIns3D` is an old
  2024 fork `9eca6f7` (quoted `56ccc22` absent; no `call_ODISE`/`lookup_pipelie`/`scene_image_rendering` -> HEAD
  openins3d wrapper cannot even run); `/data/openmask3d` HEAD `fb5df6b` (re-applied depth->color SAM prompt patch,
  quoted `0b4a7ed` absent).
- Data: only 5/16 protocol scans in `/data/scannetdata/scans` (0046_00, 0084_01*, 0086_01*, 0100_02, 0164_00; *no ply yet).
  No run.json / predictions / evaluations survive on disk.
- run-parameters file format (HEAD `runs.normalize_run_parameters`): `{"<method>": {<params>}}`.

## 1. OpenYOLO3D

### Timeline
| date | source | event |
|---|---|---|
| 2026-08-08 | `01a8c7d` | wrapper created |
| 2026-08-09 | #2 c.22:43:47 | intrinsics symlink -> `intrinsic_color.txt` (AP 0.242->0.381, 18-cls 8 scenes) |
| 2026-08-30 19:23 | `96af588` | last wrapper commit (sha `6b9b6e25…`) = HEAD |
| 2026-08-31 10:50 | #44 | `baseline-openyolo3d` 10-scene: AP .370/.486/.539, params `{}`, ScanNet `a8c137c`, ext `f457dc3`, wrapper `6b9b6e25` |
| 2026-09-01 | #35 | 10-scene baseline archived; 16-scene protocol `70cc205` |
| 2026-09-06 | #54 | label-stamping analysis (directions only, never run) |
| 2026-09-08 10:30 | #44 | **baseline** re-run 16-scene, `baseline-openyolo3d`, params `{}`, ScanNet `17d28d0`, cfg sha `a8fd4b96`: AP .2129/.3248/.4243 (wrapper at `17d28d0` = `96af588` blob) |
| ~2026-09-10 | #35 c.09-11 | batch 1 (6 runs), worktree `818cd57`->`6523c26` mid-batch |
| 2026-09-10 | #35 | **winner** `improve1-openyolo3d-h1-precomputed` `{"proposal_source":"official_precomputed"}` AP .3022/.4000/.4732 |
| 2026-09-10 | #35 | losers (pruned): t1 `network2d_th 0.15` .1995; t2 `network3d_nms 0.45` .1862; h2 `vote_weighting detector_confidence` .1969; h3 `proposal_score_power 1.0` .1844; t3 `frequency 5` stranded 15/16 (#68) |
| 2026-09-11 | #68 | resume guard bug (open, not part of target) |

### Target state
- Baseline: HEAD wrapper exactly (blob `6b9b6e25`), official `config_scannet200.yaml`, `is_gt=False`, freq 10, `min_mask_points` 20.
- Best: HEAD + `proposal_source` param. Issue: "`{}` stays byte-equivalent to the official path". Loser knobs
  (`frequency`, `network2d_th`, `network3d_nms`, `vote_weighting`, `proposal_score_power`) were inert at defaults in the
  winner run; excluded (optional re-add only if batch 2 wants them).

### Gaps (master HEAD vs winner)
1. `proposal_source` param (`live` | `official_precomputed`) missing in wrapper + `common._ENUMS`.
2. Upstream API call missing: `OpenYolo3D.predict(..., path_to_3d_masks=<dir>)` (upstream `utils/__init__.py:103`, loads `<dir>/<scene>.pt`).
3. Pre-validation missing: "masks N == `_vh_clean_2.ply` vertex count"; per-file sha256 logged at run time.
4. Proposal cache missing: `/data/openyolo3D/output/scannet200/scannet200_masks/` does not exist. Source exists:
   `/data/openyolo3D/downloads/openyolo3d_zip/OpenYOLO3D.zip` -> `OpenYOLO3D/output/scannet200/scannet200_masks/` (312 `.pt`, all 16 protocol scenes present; zip-internal sha256 e.g. scene0046_00 `415110591cb3…`). Issue combined digest `1269640d…` (combination method not recorded; not reproduced).
5. `spellbook/tests/test_openyolo3d.py` (23 CPU tests) lost.
6. External `/data/OpenYOLO3D` at `24f344d`, not `f457dc3` — check whether `f457dc3` is upstream (aminebdj/OpenYOLO3D) and pin it.

### Restore spec
`spellbook/predict/models/_openyolo3d_run.py`
- `PROPOSAL_SOURCE = "live"`; `PRECOMPUTED_MASKS_DIR = "/data/openyolo3D/output/scannet200/scannet200_masks"`.
- `load_overrides(args.parameters_json, {"min_mask_points", "proposal_source"})`.
- if `official_precomputed`: assert `<dir>/<scene>.pt` exists, `torch.load` -> masks.shape[0] == vertex count of
  `--pointcloud`, else fail closed; print sha256; call `predict(path_2_scene_data=scratch, depth_scale=1000.0,
  text=args.classes, path_to_3d_masks=PRECOMPUTED_MASKS_DIR)`. `live` = current call unchanged.
`spellbook/predict/models/common.py`: `_ENUMS["proposal_source"] = ("live", "official_precomputed")`.
Data: `unzip -j OpenYOLO3D.zip 'OpenYOLO3D/output/scannet200/scannet200_masks/*' -d /data/openyolo3D/output/scannet200/scannet200_masks/`.
CLI:
```
# baseline (params {})
python spellbook/main.py --predict --benchmark ScanNet200 --models openyolo3d --run-id baseline-openyolo3d --issue 44
# best: oy_h1.json = {"openyolo3d": {"proposal_source": "official_precomputed"}}
python spellbook/main.py --predict --benchmark ScanNet200 --models openyolo3d \
  --run-id improve1-openyolo3d-h1-precomputed --run-parameters oy_h1.json --issue 35
```

## 2. OpenIns3D

### Timeline
| date | source | event |
|---|---|---|
| 2026-08-09 | #4 | demo path -> ODISE + `--detector {odise,yoloworld}` (default odise), mask conf 0.5->0.05 |
| 2026-08-30 19:23 | `96af588` | last wrapper commit (sha `cf3fb952…`) = HEAD; params detector/mask_confidence_threshold/lookup_threshold/point_limit/min_mask_points |
| 2026-08-31 11:14 | #45 | r01 `baseline-openins3d` 10-scene AP .0769, `{}`, wrapper `cf3fb952` |
| 2026-08-31 13:07–15:01 | #45 | r02 paper masks (`3bdfafe6`) .0691; r03 +CLASS_LABELS_200 prompts (`fd557d31`) .0711; r04 `{"mask_confidence_threshold":0.001,"use_precomputed":false}` (`e7276251`) .0731; r05 yoloworld+precomputed .0395; "Wrapper defaults restored to [r01]" |
| 2026-08-31 22:41 | #45 | r06 ScanNet20 `{"detector":"odise","use_precomputed":true}` (`833f3abc`) AP .2969 |
| 2026-09-08 21:20 | #45 | **baseline** 16-scene `baseline-openins3d` `{"use_precomputed": true}`, wrapper `d5b0ee93…`, ScanNet `17d28d0`, ext `56ccc22`: AP .04959/.07409/.09538 (z-up fix; released `masks/detected/*.pt`; ODISE, lookup 0.3) |
| 2026-09-10 15:25 | #34 | **winner** `openins3d-b01-t1-yolopre` `{"detector":"yoloworld","use_precomputed":true}` AP .05245/.06659/.07865, wrapper `4e8ffcda…` |
| 2026-09-10 | #34 | losers (pruned): T2 `{"use_precomputed":false}` .0322; T3 `+mask_confidence_threshold 0.001` .0333; H1 `lookup_threshold 0.1` .0389; H2 `{"lookup_vocab":"benchmark198"}` .0507; H3 `single_detection` .0003 |

### Target state
- = baseline wrapper `d5b0ee93` behaviour (winner T1 needs only `detector` + `use_precomputed`, both present at baseline).
  `lookup_vocab`/`single_detection` (added in `4e8ffcda` for H2/H3, losers) excluded.
- Newer info wins: 16-scene comment (2026-09-08) supersedes the 10-scene "defaults restored to r01": native z-up, released masks.
- Default vocab = 200-way (#34 H2: "lookup vocab 200→198"), i.e. r03's "official ScanNet200 CLASS_LABELS_200 ODISE prompts
  (wall/floor dropped only at ScanNet submission)". Default `use_precomputed` unknown (both runs passed it explicitly) -> keep `false`.

### Gaps
1. `use_precomputed` param + loader for released "vertex-aligned `masks/detected/*.pt`" proposals (replaces Mask3D step).
2. z-up: HEAD calls `_ensure_z_up` (smallest-extent heuristic). Issue: "The wrapper now preserves ScanNet's native z-up coordinates" -> remove the rotation.
3. 200-way Lookup vocab (CLASS_LABELS_200 incl. wall/floor), mapped back to `args.classes`, wall/floor dropped at submission. HEAD uses `args.classes` (198).
4. External: `/data/OpenIns3D` is wrong repo state (fork `9eca6f7`). Need Pointcept/OpenIns3D `56ccc220dc003e38c644116d7d2078057b6136ad`, ODISE + detectron2 wheels (#4, #14), `third_party/scannet200_val.ckpt` sha `6a0c4347…` (158873495 B), and the "official scannet.zip" masks — none on disk.
5. `runs.py discover_manifests` WARN-skip fix + test `test_unparseable_manifest_skipped_with_warning` (#34) not in master.
6. prune gap: `<scene>_results` dirs not pruned (#34 addendum) — optional.

### Restore spec
`spellbook/predict/models/_openins3d_run.py`
- `PRECOMPUTED_MASK_DIR = "<OpenIns3D>/data/scannet/masks/detected"` (exact subpath from 56ccc22 README; issue only says `masks/detected/*.pt`).
- `load_overrides(..., {"detector","mask_confidence_threshold","lookup_threshold","point_limit","min_mask_points","use_precomputed"})`;
  `use_precomputed = bool(params.get("use_precomputed", False))`.
- if precomputed: `mask_list = torch.load(f"{dir}/{scene_id}.pt")` (dense, bool, (N_vertices, K)); fail closed if N != vertex count; skip decimation (`nn_idx = arange`).
- replace `_ensure_z_up(...)` by plain `np.asarray(pcd.points)`.
- `lookup = Lookup(..., text_input=CLASS_LABELS_200)`; map `cls` index -> name; skip names not in `args.classes` (wall/floor).
- keep defaults: `detector "odise"`, `LOOKUP_THRESHOLD 0.3`, `IMAGE_SIZE [800,800]`, `ADJUST_CAMERA [2,0.1,1.0]`, `MASK_CONFIDENCE_THRESHOLD 0.05`, `MIN_MASK_POINTS 20`.
`common.py`: add `use_precomputed` boolean validation.
CLI:
```
# baseline: oi_base.json = {"openins3d": {"use_precomputed": true}}
python spellbook/main.py --predict --benchmark ScanNet200 --models openins3d --run-id baseline-openins3d --run-parameters oi_base.json --issue 45
# best: oi_t1.json = {"openins3d": {"detector": "yoloworld", "use_precomputed": true}}
python spellbook/main.py --predict --benchmark ScanNet200 --models openins3d --run-id openins3d-b01-t1-yolopre --run-parameters oi_t1.json --issue 34
```
Expect ODISE nondeterminism ±0.02 AP points (#45).

## 3. OpenMask3D

### Timeline
| date | source | event |
|---|---|---|
| 2026-08-22..30 | `3a01665`,`96af588`,`1e42620` | wrapper registered / protocol |
| 2026-08-31 13:14 | #46 / `050ca11` 15:14 | r2 official defaults (queries 150, dbscan_min 1, freq 10, `a {} in a scene`, conf 1.0) AP .073 |
| 2026-08-31 23:24 | `22f7bd1` | 200-way CLIP names (blob `9d177e77`) |
| 2026-09-01 08:58 | #46 | attempt 4 10-scene `baseline-openmask3d` AP .166/.229/.267, ScanNet `22f7bd1`, ext `0b4a7ed` (SAM depth->color prompt patch), wrapper `9d177e77` |
| 2026-09-01 22:55 / 09-08 00:33 | `affaf49`, `9968d4a` | rows API, `publish_clip_features`, `DEDUP_IOU=0.5`, `MIN_MASK_POINTS 20->1`, `clip_prompt` in `_classify` (= HEAD blob `2c1ac3a3`) |
| 2026-09-08 10:46 | #46 | attempt 5 16-scene `{}` AP .1039/.1996/.2887 (HEAD-equivalent wrapper, provenance ScanNet `f870ca6` — not in git) |
| 2026-09-08 22:53 | #46 | CPU proof segment pooling AP .165199; scene0535 smoke OK |
| 2026-09-09 19:55 | #46 | **baseline = best** `baseline-openmask3d-segsnap-r1` promoted to `baseline-openmask3d`, `{}`: AP .1809/.2444/.3056 |
| — | #37 | no improvement batch run |

### Target state
HEAD wrapper (`9968d4a` blob) + official segment pooling + mask-basename/feature-output contract fix; external openmask3d
with depth->color SAM prompt scaling (now `fb5df6b`) + 200-way CLIP (already in HEAD).

### Gaps
1. Segment pooling missing (no `scatter_mean`/segs in git ever).
2. Fail-closed checks missing ("fail-closed on missing/mismatched segs"; "segment violations 0").
3. Feature path contract: upstream `compute_features_single_scene.py:77` names output `basename(masks_path)[:-9] + "_openmask3d_features.npy"` -> pooled file must end in `_masks.pt` and wrapper must derive features path from the pooled masks basename (HEAD derives from `scene_ply`).
4. 64 unit tests lost (#46); no wrapper sha quoted for segsnap.

### Restore spec
`spellbook/predict/models/_openmask3d_run.py`
- new `_segment_snap(masks_path, segs_json, out_path)`: `seg = json.load(..)["segIndices"]`; masks `(N,K)`;
  fail if `len(seg) != N`; per-segment mean (`torch_scatter.scatter_mean(mask, seg, dim=0)` as in upstream
  `trainer.py:get_full_res_mask`), `> 0.5`, broadcast `mask[seg]`; `torch.save` to `<scratch>/working_scene_segsnap_masks.pt`
  (suffix `_masks.pt`); verify every mask is segment-constant.
- `segs_json = <scan_dir>/<scene>_vh_clean_2.0.010000.segs.json` (sibling of `--pointcloud`); fail if missing,
  or if `len(working_pts) != len(full_pts)` (500k cap must not decimate; protocol scenes are 50k–190k verts).
- order in `main`: `_run_mask_computation` -> `_segment_snap` -> `_run_feature_computation(pooled)` -> classify on pooled masks.
- `_run_feature_computation` returns `os.path.join(out_dir, basename(masks_path)[:-9] + "_openmask3d_features.npy")`.
- defaults unchanged: `NUM_QUERIES 150`, `DBSCAN_MIN_POINTS 1`, `dbscan_eps 0.95`, `FREQUENCY 10`, `CLIP_PROMPT "a {} in a scene"`,
  `DEDUP_IOU 0.5`, `MIN_MASK_POINTS 1`, `POINT_LIMIT 500_000`, confidence 1.0.
CLI:
```
python spellbook/main.py --predict --benchmark ScanNet200 --models openmask3d --run-id baseline-openmask3d --issue 46
```

## 4. Restore order / prerequisites
1. Download remaining 11 protocol scans (`download_scans.py`) into `/data/scannetdata/scans`.
2. Pin externals: OpenYOLO3D `f457dc3`, OpenIns3D `56ccc22` (+ODISE), openmask3d `0b4a7ed`-equivalent (`fb5df6b`).
3. openmask3d (smallest code delta, highest certainty) -> openyolo3d (one param + unzip) -> openins3d (repo rebuild).
4. Accept reproduction if within ODISE/YOLO noise (~±0.5 AP pts); record new wrapper sha in run.json.
