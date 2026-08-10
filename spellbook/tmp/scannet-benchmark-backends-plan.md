# ScanNet20 + ScanNet200 Benchmark Backends

Target executor: DeepSeek V4 Flash build agent. Execute phases in order and stop when an acceptance gate fails. Do not revert unrelated user/session changes, do not touch `.opencode/`, do not commit or push unless explicitly requested, and preserve all existing predictions/results until the replacement pipeline is verified.

## Goal

Use `spellbook/settings.yaml` to select one coherent benchmark backend for prediction, GT export, local evaluation, visualization, and submission packaging:

- `ScanNet20`: official ScanNet 18-instance-class protocol and the official evaluator ported in place to Python 3.
- `ScanNet200`: official 198-instance-class protocol (200 minus wall/floor) and a Python-3 evaluator based on ScanNet200 author David Rozenberszki's published `LanguageGroundedSemseg` implementation.

Keep `/data/scannet/scans/` unchanged. Write each model/run directly in ScanNet's official submission layout under the existing `/data/scannet/predictions/` root; put GT and local evaluation results under the existing `/data/scannet/derived/` root. Remove `spellbook/eval/` only after parity and migration checks pass.

## Current State / Guardrails

- `spellbook/settings.yaml` is user-created and currently contains `default: ScanNet20`; preserve this choice.
- GitHub issues #1-#17 contain the bug history. Relevant open issues: #11, #12, #13, #14, #15, #16, #17. Add reasoning/results as comments; close only after verified resolution.
- Git `master` is ahead of `origin/master` by one commit; `AGENTS.md`, `.opencode/`, and `settings.yaml` may contain concurrent user changes. Never revert them.
- Existing 18-class results are valid. Existing 189-class ScanNet200 results are diagnostic and protocol-invalid (#12); preserve them as legacy evidence.
- Progress reporting needs only a durable log and rough completed/total count. Do not build an advanced ETA system.

## Official Sources of Truth

### ScanNet20

- Evaluator: `BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py`.
- Utilities: `BenchmarkScripts/util.py`, `BenchmarkScripts/util_3d.py`.
- Classes: official ScanNet 20-class labels minus wall/floor = 18 instance classes; NYU40 ids.
- Submission: one `<scene>.txt` per scan at submission root plus `predicted_masks/<scene>_NNN.txt`.

### ScanNet200

- Constants/splits: `BenchmarkScripts/ScanNet200/scannet200_constants.py`, `scannet200_splits.py`.
- Instance classes: `VALID_CLASS_IDS_200`/`CLASS_LABELS_200` minus raw ids 1 (`wall`) and 3 (`floor`) = exactly 198.
- Evaluator reference: `RozDavid/LanguageGroundedSemseg/downstream/insseg/datasets/evaluation/scannet_benchmark_utils/scripts/evaluate_semantic_instance.py` (ScanNet200 benchmark author). ScanNet/ScanNet itself publishes no ScanNet200 instance evaluator; exhaustive branch/history/org search confirmed this.
- Label map column: raw `id`, not `nyu40id`.

Never use `VALID_CLASS_IDS_200_VALIDATION` as the prompt/evaluator class list; it is the 189-class semantic validation subset and includes wall/floor. Delete all hardcoded fallback class copies and fail loudly if official constants cannot be imported.

## Final Settings

Keep `settings.yaml` intentionally small:

```yaml
default: ScanNet20
scannet_root: /data/scannet
```

Protocol details are fixed in code and cannot be overridden in YAML. This prevents invalid combinations such as ScanNet200 prompts with the ScanNet20 evaluator.

Add runtime CLI options:

- `--benchmark {ScanNet20,ScanNet200}`: optional override; defaults to `settings.yaml:default`.
- `--run-id <name>`: required for prediction/evaluation output isolation; if omitted for prediction, generate one UTC run id once in `main.py` and pass it to every task.
- Benchmark predictions derive their class list automatically. If `--classes` is supplied, treat it as custom prediction only and reject benchmark evaluation for that run.

## Final Data Layout

`/data/scannet/scans/` remains official data and contains no new benchmark predictions.

```text
/data/scannet/
├── scans/                                  # official scans, unchanged
├── predictions/
│   ├── ScanNet20/<run-id>/<model>/         # official submission root
│   │   ├── sceneXXXX_YY.txt
│   │   └── predicted_masks/
│   │       └── sceneXXXX_YY_NNN.txt
│   └── ScanNet200/<run-id>/<model>/        # same official layout
└── derived/
    ├── ground_truth/
    │   ├── ScanNet20/<scene>.txt
    │   └── ScanNet200/<scene>.txt
    ├── evaluations/
    │   ├── ScanNet20/<run-id>/<model>.txt
    │   └── ScanNet200/<run-id>/<model>.txt
    └── legacy/
        └── ScanNet200-189/                  # archived invalid diagnostic results
```

Each `<model>/` directory under `predictions/` is directly zippable as an official submission. Never place result files or metadata inside it; ScanNet forbids extra submission files.

## Code Layout After Migration

```text
spellbook/
├── settings.yaml
├── benchmark.py                  # settings/spec resolution, paths, GT export, evaluator dispatch
├── scannet200_evaluator.py       # faithful attributed port of Rozenberszki evaluator
├── main.py
├── predict/
└── utils/
```

The official ScanNet20 evaluator remains in `BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py`, ported in place to Python 3. `spellbook/eval/` no longer exists after successful migration.

## Phase 1 - Benchmark Specification and Settings

Create `spellbook/benchmark.py` with a small immutable `BenchmarkSpec` for each backend:

- name (`ScanNet20` / `ScanNet200`)
- class names
- valid ids
- label-map target column (`nyu40id` / `id`)
- evaluator kind (`official` / `scannet200`)
- prediction root, GT root, evaluation root derived from `scannet_root`

Implementation rules:

1. Load YAML once in the orchestration process.
2. Validate `default` and `scannet_root`.
3. Derive ScanNet20's 18 names/ids from official ScanNet constants by excluding wall/floor.
4. Derive ScanNet200's 198 names/ids from official ScanNet200 constants by excluding raw ids 1 and 3.
5. Assert name/id lengths 18 and 198, no duplicates, no wall/floor, and one-to-one mapping.
6. Expose `resolve_benchmark(name)` to `main.py`, runner, writer, GT exporter, and evaluator dispatch.
7. Do not add fallback literals.

Acceptance gate:

- `resolve_benchmark("ScanNet20")` => 18 classes, NYU40 ids.
- `resolve_benchmark("ScanNet200")` => 198 classes, raw ids, ids 1/3 absent.
- Invalid default/root/class override fails before GPU work.

## Phase 2 - Port Official ScanNet20 Evaluator In Place

Port `BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py` to Python 3 with a minimal reviewed diff:

1. Python-2 print/exception syntax only.
2. `np.float` -> `float`; `np.bool` -> `np.bool_`.
3. Retain official ScanNet20 class constants, CLI arguments, IoU thresholds, min-region rules, output format, relative-mask-path validation, and duplicate-prediction behavior.
4. Retain the verified zero-match guard from issue #8 (AP=0 rather than IndexError).
5. Do not add ScanNet200 conditionals or imports to this official file.
6. Add a short comment identifying the upstream commit and Python-3-only changes.

Parity test before deleting the old port:

- Run the current `spellbook/eval/evaluate_semantic_instance.py` and the in-place port against the same existing 8-scene ScanNet20 predictions/GT.
- Diff every per-class AP/AP50/AP25 and overall average; require exact or float-format-only equality.
- Add a synthetic zero-match case and verify AP=0 instead of crash.
- `python -m py_compile` must pass.

Stop if parity fails.

## Phase 3 - Add ScanNet200 Evaluator

Create `spellbook/scannet200_evaluator.py` as a faithful, attributed Python-3 port of the evaluator published by ScanNet200 author David Rozenberszki.

Adaptations allowed:

- import official ScanNet constants from this repo instead of duplicating large class tables;
- use the shared `BenchmarkSpec` 198-class names/ids;
- preserve official ScanNet submission-path handling and metric math;
- output per-class AP/AP50/AP25 plus average and head/common/tail;
- add a CLI/API accepting `pred_path`, `gt_path`, `output_file`.

Do not copy model-specific output remapping or training code from OpenYOLO3D/Open3DIS.

Parity tests:

1. Synthetic one-TP / one-FP / one-duplicate / void / min-region cases.
2. Compare output with the Rozenberszki evaluator on the same synthetic data.
3. Compare with OpenYOLO3D's 198-class evaluator on the same flat submission.
4. Assert wall/floor predictions are ignored/void and absent from reported classes.
5. Assert all 11 train-only classes are valid but yield `nan` when absent from val GT.

Stop if metric parity fails.

## Phase 4 - Direct Official-Format Prediction Output

Update `main.py`, `predict/runner.py`, `predict/models/common.py`, and all four wrappers:

1. Resolve benchmark once in `main.py`; pass benchmark name, official class list, output root, and run id to the runner.
2. Replace `--label_set` with `--benchmark`; update all internal callers/docs. Do not maintain two protocol switches.
3. Write one `<scene>.txt` directly into the model's official submission root.
4. Write masks as `predicted_masks/<scene>_NNN.txt`; scene prefix prevents cross-scene collisions during parallel tasks.
5. Use benchmark label mapping from `BenchmarkSpec`; writer hard-errors on any unknown class.
6. Write each scene/masks into a temporary task directory, validate mask lengths/ids, then atomically move files into the submission root.
7. Add per-scene completion metadata outside the submission root under `derived/evaluations/...` so interrupted runs can resume without contaminating the archive.
8. Never leave stale masks from a previous run id.

Visualization:

- update prediction loading to use `benchmark + run-id + model + scene` from the official submission root;
- if no run id is supplied, show GT only rather than guessing the latest run.

Acceptance gate:

- Predict the same scene once as ScanNet20 and once as ScanNet200; both roots coexist and neither file changes after the other run.
- Submission path checker confirms only `<scene>.txt` + `predicted_masks/` and all mask paths remain inside root.
- Existing visualization loads both runs explicitly.

## Phase 5 - GT and Evaluation Dispatch

Move the custom flat-GT logic from `spellbook/eval/export_gt.py` into `spellbook/benchmark.py` (or a private helper imported by it):

- ScanNet20: `raw_category -> nyu40id`, filter to official 18 ids.
- ScanNet200: `raw_category -> id`, filter to official 198 ids.
- Preserve `label_id * 1000 + instance_id`, one line per `_vh_clean_2.ply` vertex.

Evaluation command:

1. Resolve benchmark/settings/run/model.
2. Pass the model submission root directly as `pred_path`; no collection/copy stage.
3. Dispatch ScanNet20 to the in-place official Python-3 evaluator.
4. Dispatch ScanNet200 to `spellbook/scannet200_evaluator.py`.
5. Write result file only under `/data/scannet/derived/evaluations/...`.
6. Normalize/reject invalid scene inputs (issue #11): accept `0568_00` or `scene0568_00`, reject empty entries/trailing commas, fail before evaluation if any requested scene file is absent.

Acceptance gate:

- Existing ScanNet20 results reproduce exactly.
- Two intentionally invalid scene lists reproduce issue #11 failures before the fix, then pass/fail correctly after it.
- No `collected/` directory is created.

## Phase 6 - Migrate Artifacts and Remove spellbook/eval

Do not delete anything until Phases 1-5 pass.

1. Copy existing valid ScanNet20 GT/results into `/data/scannet/derived/`.
2. Copy existing 189-class ScanNet200 result files into `derived/legacy/ScanNet200-189/` with a README saying protocol-invalid (#12).
3. Preserve `results_prefix/` evidence already documented in issues #1-#4.
4. Existing per-scene prediction directories are legacy; do not delete them until direct official-format output is verified for all four models.
5. After parity/migration checks, delete `spellbook/eval/` and remove the broad `spellbook/eval/` `.gitignore` rule.
6. Update `PROJECT_STATUS.md` structure/commands/plan and issue comments #8, #11, #12, #16.

Acceptance gate:

- No unique result/GT evidence is lost.
- `spellbook/eval/` has no remaining importers before deletion.
- Git contains all required evaluator/benchmark code (nothing essential hidden by `.gitignore`).

## Phase 7 - Pre-Batch Regression

Run before any full ScanNet200 rerun:

1. Unit/static checks for 18/198 class specs.
2. ScanNet20 evaluator parity test.
3. ScanNet200 evaluator parity test.
4. One complete frames -> predict -> evaluate run per model on scene0568_00 for both benchmarks.
5. OpenIns3D 198-prompt smoke on scene0304_00 and scene0412_00; define an instance-count gate and stop if collapse persists (#13).
6. Two concurrent Open3DIS scenes after private tracker-state fix (#15); require both outputs and no shared tracker mutation.
7. Run ScanNet20 after ScanNet200 on one scene and verify no overwrite/stale-mask contamination (#16).
8. Rough progress only: completed/total and durable log.

Only after all gates pass: run all four models on the 20 val scenes with 198 prompts, then evaluate. Do not expand to 312 scenes yet.

## Phase 8 - Issue and Documentation Closure

- #11: close after scene-id/trailing-comma tests pass.
- #12: close only after 198-prompt prediction + evaluator run succeeds; record corrected results.
- #13: keep open unless OpenIns3D recall is explained and verified.
- #14: out of this plan unless env rebuild becomes necessary for evaluator parity.
- #15: close after concurrent Open3DIS test.
- #16: close after cross-benchmark no-overwrite + atomic/resume tests.
- #17: do not implement advanced ETA; add a comment that rough completed/total is accepted. Keep open only for process-survival/resume work not covered by #16.

Before `/compact`, update relevant issues with theoretical reasoning, attempted fixes, evidence, and final decisions. `PROJECT_STATUS.md` remains structure, goal, and plan only.

## Final Deliverables

- `spellbook/settings.yaml` with selected default/root.
- `spellbook/benchmark.py`.
- `spellbook/scannet200_evaluator.py`.
- Official ScanNet20 evaluator ported in place to Python 3 with parity tests.
- Direct official-format prediction roots under `/data/scannet/predictions/`.
- GT/results under `/data/scannet/derived/`.
- `spellbook/eval/` removed only after verified migration.
- Updated PROJECT_STATUS and GitHub issues.
