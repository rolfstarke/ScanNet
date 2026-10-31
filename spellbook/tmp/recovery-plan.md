# Spellbook recovery plan

## Context
Spellbook (`/data/ScanNet/spellbook`) compares 3D reconstruction engines and open-vocabulary 3D instance-segmentation models on ScanNet. The host was rebuilt:
- The method repos are cloned but have no envs or weights.
- The worktrees holding the newest wrapper code are gone.
- The data root `/data/scannetdata` holds only the two `.svo2` files and the `.3dm` file.
- The code still points at `/data/scannet` and `/home/rolf/...`.

The goal is to rebuild the setup, restore the wrappers from the issue logs, re-download the 16 val scenes, and reproduce the logged baselines and current best runs.

Scope, as chosen by the user:
- Prediction: OpenYOLO3D, Mosaic3D, OpenIns3D, OpenMask3D, Open3DIS, sam3dis.
- Reconstruction: zed, metashape, rtabmap, isaac, open3d, bundlefusion.
- Skipped: search3d, SpaCeFormer, OV3D-CG, openscene.

## Ground rules
- All new code goes in `spellbook/`. Upstream repos stay untouched apart from the patches the issues document, which go on a fork branch (`spellbook-baseline`) and never on master.
- Data follows the layout in `PROJECT_STATUS.md` under `/data/scannetdata`.
- Envs are prefix envs at `/data/<method>/conda/envs/<name>`, matching `predict/runner.py:32-38`.
- The CUDA toolkit comes from conda (`nvidia` channel), because the host has no nvcc and no sudo.
- GPU 0 is reserved; the pool is 1–4. All GPU work goes through `main.py`/`jobs.py`.
- Time cap: non-Metashape reconstructions must finish in under 3× the scene length (9004: 361.2 s, 9009: 471.1 s). Metashape is exempt.
- Secrets (Metashape license, HF token) go in `spellbook/.secrets.env` with mode 600, added to `.gitignore`, and are written before anything else runs. They never go in the plan, commits, or logs.

## Step 0: Prep (main context)
1. Give a time estimate: about 1 day of setup plus 1–2 days of GPU runs.
2. Write `.secrets.env` and add it to `.gitignore`. Copy this plan to `spellbook/tmp/recovery-plan.md`, as AGENTS.md requires.
3. Data root:
   - Set `settings.yaml: scannet_root: /data/scannetdata`.
   - Make the 22 hard-coded `/data/scannet` sites read from `settings.yaml` (one helper in `spellbook/utils`) instead of swapping one string for another.
4. Replace the `/home/rolf/GIT/<X>` and `/home/rolf/anaconda3/...` paths with the `/data/<X>` clones and prefix envs, using the same settings helper: keys `repos:`, `envs:` and `zed_dir:`.
5. Install `gh` user-local into `~/.local/bin`. The user runs `! gh auth login` so results can be posted to the issues.

## Step 1: Set up methods (parallel Haiku subagents, one per repo, each following its own README)
Each agent:
- builds the env at the agreed prefix;
- installs CUDA through conda when needed;
- downloads the checkpoints to the paths the wrappers expect;
- applies the issue-documented fixes;
- reports a smoke import.

The agents:
| Agent | Repo | Notes from the issues |
|---|---|---|
| A | `/data/OpenYOLO3D` | py3.10, torch 1.12.1+cu113, MinkowskiEngine `--force_cuda`. Also needs the YOLO-World ckpt, the `scannet200_val.ckpt` checkpoint and the official precomputed masks (for the winner). |
| B | `/data/Mosaic3D` | Weights from HF `junhalee/Mosaic3D` → `/data/mosaic3d/ckpts/sc.ckpt` (+ `spunet34c.ckpt`) |
| C | `/data/OpenIns3D` | py3.9, torch 1.13.1+cu116, detectron2 0.6 + ODISE built from source with `--no-deps`, `third_party/ODISE`, `np.bool` patch, ckpt `scannet200_val.ckpt` (also used by Mosaic3D proposals) |
| D | `/data/openmask3d` (+ `/data/Mask3D`) | py3.8.5, `resources/scannet200_model.ckpt` + `sam_vit_h_4b8939.pth`, SAM prompt-scaling patch on branch |
| E | Open3DIS | Clone `rolfstarke/Open3DIS` branch `spellbook-baseline` (fallback: upstream + fixes from #3/#15/#24), Grounded-SAM, ISBNet `head_scannetv2_200_val.pth` |
| F | `/data/sam3` + sam3dis | py3.12, torch 2.10+cu128, HF login from `.secrets.env`. Recover the sam3dis code from the `rolfstarke` GitHub repos, falling back to rebuilding it from #69. |
| G | ZED SDK 5.x user-local | Extract the official installer into `~/opt/zed-sdk/install`, add calibration `SN46307300.conf`, build the `3disspellbook` env from `spellbook/environment.yaml` with the local pyzed wheel and the activate.d hook |
| H | Recon containers | podman `localhost/zed-rtabmap:jazzy` + `podman_gpu.sh` (recreate under `spellbook/reconstruct/containers/`). Isaac image via the adapter's built-in build. Docker `bundlefusion:latest` + `:spellbook-siftfix` (recreate the Dockerfiles from #50/#52). |
| I | Metashape | Metashape Pro Python module in `/data/zed-metashape/conda/env`, licence activated from `.secrets.env` |

## Step 2: Data (main context, using existing tools)
1. **Scans:** fetch the 16 val scenes from `evaluation/benchmark.py:25-42` into `/data/scannetdata/scans/` with `spellbook/download_scans.py`, base URL switched to `kaldir.vc.cit.tum.de` (the upstream `download-scannet.py` is the fallback). Also fetch `v2/scannetv2-labels.combined.tsv`. Accepting the ScanNet ToS prompt counts as the user's go-ahead.
2. **SVO files:** move both into `custom/raw/` under their original names and add the symlinks `scene9004.svo2` (HD1200) and `scene9009.svo2` (HD1080).
3. **CAD file:**
   - Move `.3dm` to `custom/reference/LG2C_Raum312_v2.3dm`, point `geometry_reference.yaml:cad_source` at it, and check sha `ff00c139…`.
   - Regenerate `scene9004_cad.ply` and the visible-voxel npz. Search git history and the issues for the original conversion (likely rhino3dm); otherwise write it with `rhino3dm`.
   - The output must match the shas `76613112…` and `9e47eb86…`. If it doesn't, report it, because scores would not be comparable.
4. **Frame pools:** run `main.py --extract-frames` for 9004 and 9009 and check the counts are 3612 and 2356.
5. **Ground truth:** run `evaluation/evaluate.py export-gt`.

## Step 3: Reconnect the wrappers (main context and subagents per method, one git branch)
Recovery sources:
1. git objects (`git fsck --lost-found`, reflog);
2. the local `spellbook/archive` and `spellbook/tmp`;
3. issue comments (exact params, function names, presets).

Timestamp check, per method, before restoring anything:
- Build a timeline from four sources: commit dates for master and dangling objects, file mtimes in `archive/` and `tmp/`, issue comment dates, and the `created`/`ended` times in surviving `run.json` files and job records.
- Restore from whichever source is newest and still matches the winning run, i.e. the latest version that was there when the logged best result was produced. Code changes made after that run (later losing batches, rollbacks like Mosaic3D's `genuine` mode) get dropped.
- When a newer issue comment contradicts older code (for example the `sc.ckpt` vs `spunet34c` fix, or the default preset), the newer comment wins.
- Record the chosen source and timestamp for each restored piece in `spellbook/tmp/recovery-timeline.md`.

Restore per method:
- **OpenYOLO3D:** the `proposal_source` / `vote_weighting` / … params and their tests.
- **Mosaic3D:**
  - `WINNER_PRESETS` (`h3-proxyrescore`, `b2-geom2`) and npz v2.
  - `scripts/run_custom_scene.py`, moved out of the fork into the spellbook wrapper.
  - Fix the checkpoint name clash between `runs.py:52` and `runner.py:62`.
- **OpenIns3D:** z-up fix, `detector` param, `runs.py` manifest skip.
- **OpenMask3D:** segment pooling.
- **Open3DIS:** `emit_final` and the derived config.
- **sam3dis:** `_sam3dis_pool.py`, `_sam3dis_run.py`, the query path.
- **Reconstruction engine presets:**
  - zed: `b5-gen1-neural` (default) and `s2b8-fixedcalib`.
  - isaac: `official_default` and `conf100_best`.
  - metashape: `b5-kf350`.
  - open3d: bounded B1.
  - bundlefusion: presets and image.
  - rtabmap: assisted-hybrid s90.
- **Other fixes:**
  - Add the missing `utils/glx.py` (or drop the import).
  - Build the `/data/ScanNet/Segmentator` and restore the `Server/tools/recons` and meshclean params.
  - Make `extract.py` extract 9004 frames directly from the SVO.

Run `spellbook/tests` and the preflight for every method.

## Step 4: Reproduce (jobs system, run ids `repro-baseline-<m>` / `repro-best-<m>`)
| Method | Baseline target | Best target |
|---|---|---|
| OpenYOLO3D | AP 0.2129 | 0.3022 (`official_precomputed`) |
| Mosaic3D | 0.0969 | 0.10784 (`b2-geom2`) |
| OpenIns3D | 0.0496 | 0.05245 (`yoloworld`) |
| OpenMask3D | 0.1809 | (no batches) |
| Open3DIS | 0.1465 | (no batches) |
| sam3dis | 0.1707 | 0.27519 (pool engine) |
| zed | 30.66 cm (`_00`) | 21.31 cm (`s2b8`; default stays `b5`) |
| metashape | 18.10 cm | 14.63 cm (`b5-kf350`) |
| isaac | 38.91 cm | 20.51 cm (`conf100_best`) |
| open3d | 29.68 cm | (baseline only) |
| bundlefusion | 50.75 cm | (baseline only) |
| rtabmap | 20.52 cm | (baseline only) |

- **Tolerance:** AP within ±0.005 (inside the 0.025 CI); geometry within ±0.5 cm. Reconstructions also have to stay under the 3× cap.
- **Mismatches:** where a run doesn't hit its target, run one bounded diagnosis and log it.
- **Slots:** reconstructions use `_X0` for the baseline and the next free slot for the best run. Prediction artifacts follow the s2-improve retention rules.

## Step 5: Close
1. Post a short repro comment on each method issue.
2. Update `PROJECT_STATUS.md` and the setup notes in `spellbook/docs/setup.md`.
3. Run the s1-document skill.
4. Commit and push after the user approves.

## Verification
- `main.py --gpu-check` and the per-method preflight pass.
- `spellbook/tests` passes.
- `main.py --status` shows no lost jobs.
- Every repro run has a `run.json`, the evaluator CSV / `geometry_score.yaml`, and a runtime under the cap. The ranking is in `derived/evaluations/ScanNet200/ranking.csv`.

## Known risks
- The rtabmap image, the bundlefusion Dockerfiles, the sam3dis code and the `.3dm` conversion may only be partly recoverable, so the results may not be bit-identical.
- The Open3DIS fork branch may not exist on GitHub.
- The ZED SDK may fail to install without root. In that case the user installs it system-wide.
