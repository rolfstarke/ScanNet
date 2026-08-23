# Action plan: migrate eval modules into `spellbook/evaluation/`

**Status.** Executed (package name `evaluation/`, not `eval/`). Smoke: `python spellbook/tmp/smoke_eval_package.py` → 17 OK.

**Goal.** Group evaluation domain code under one package. Do not put CLIs or benchmark config in `utils/`.

**Out of scope.** Behavior changes; merging scannet200 into evaluate; moving `gpu_check.py` / `visualize.py`; installable package rename to `import spellbook.*`.

---

## Target layout

```
spellbook/eval/
├── __init__.py                 # thin public re-exports (optional, keep small)
├── benchmark.py                # from spellbook/benchmark.py (unchanged logic)
├── evaluate.py                 # from spellbook/evaluate.py (path fixes only)
└── scannet200_evaluator.py     # from spellbook/scannet200_evaluator.py (import fix only)
```

Delete after move:
- `spellbook/benchmark.py`
- `spellbook/evaluate.py`
- `spellbook/scannet200_evaluator.py`

`utils/` stays helpers only: `gpu.py`, `scan_lock.py`, `visualize.py`, `hud.py`.

---

## Constraints

1. **Foreign-env absolute load** — `predict/models/common.py` loads benchmark by file path via `importlib`. Model envs must **never** get `spellbook/` on `sys.path` (shadows OpenIns3D `utils`). After move path is:
   ```
   <spellbook_dir>/eval/benchmark.py
   ```
2. **Entry-point path convention** — `main.py` does `sys.path.insert(0, spellbook_dir)`. Keep that. Imports become `from eval.benchmark import …` (preferred), not flat `from benchmark import …`.
3. **`evaluate.py` `_REPO_ROOT`** — today `dirname(dirname(__file__))` = repo root when file is `spellbook/evaluate.py`. After move file is `spellbook/eval/evaluate.py`, so need **one more** `dirname`:
   ```python
   _SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # spellbook/
   _REPO_ROOT = os.path.dirname(_SPELLBOOK)                                  # repo/
   ```
   Same fix for `scannet200_evaluator.py` if it derives repo/spellbook roots from `__file__`.
4. **Subprocess evaluator path** — update:
   ```python
   "scannet200": os.path.join(_REPO_ROOT, "spellbook", "eval", "scannet200_evaluator.py"),
   ```
5. **No ScanNet upstream edits.** All code under `spellbook/`.
6. **No logic changes** in metric math, GT encoding, label ids, CLI flags.
7. **Docs** — update `PROJECT_STATUS.md` structure + Commands only (no activity log).

---

## Import strategy (locked)

| Context | How to load benchmark |
|---|---|
| Main checkout scripts (`main.py`, `gpu_check.py`, `predict/runner.py`, `reconstruct/*`, `utils/*`, `eval/*`) | `sys.path` has `spellbook/`; use `from eval.benchmark import …` |
| Model subprocesses (`predict/models/*_run.py` → `common.py`) | `importlib.util.spec_from_file_location` on `spellbook/eval/benchmark.py` only |
| Visualize lazy AP50 | `from eval.scannet200_evaluator import Evaluator` |
| Evaluate CLI subprocess | absolute path to `spellbook/eval/scannet200_evaluator.py` |

Optional thin `eval/__init__.py`:
```python
from eval.benchmark import (
    BENCHMARKS, BenchmarkSpec, load_settings, resolve_benchmark,
    artifact_paths, submission_dir, validate_gpu_pool,
)
```
Do **not** import Evaluator at package import time (heavy numpy + BenchmarkScripts path).

---

## File-by-file edits

### Create
1. `spellbook/eval/__init__.py` — re-exports above (or empty if preferred minimal).
2. Move three modules into `spellbook/eval/` (git mv preferred to keep history).

### Patch inside moved files

**`eval/benchmark.py`**
- Logic unchanged.
- Confirm no `__file__`-based parent that assumed top-level location. Today it loads `BenchmarkScripts` via repo-relative paths — verify after one extra directory depth.

**`eval/evaluate.py`**
- Fix `_REPO_ROOT` (three dirname levels from `__file__` → wait: `__file__` = `…/spellbook/eval/evaluate.py` → dirname = eval, dirname = spellbook, dirname = repo). So three dirnames from abspath file.
- `sys.path.insert(0, spellbook_dir)` before `from eval.benchmark import …` **or** insert spellbook and use package import; also keep BenchmarkScripts on path.
- Recommended header:
  ```python
  _EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
  _SPELLBOOK = os.path.dirname(_EVAL_DIR)
  _REPO_ROOT = os.path.dirname(_SPELLBOOK)
  sys.path.insert(0, _SPELLBOOK)
  sys.path.insert(0, os.path.join(_REPO_ROOT, "BenchmarkScripts"))
  from eval.benchmark import load_settings, resolve_benchmark, artifact_paths, submission_dir
  ```
- `_EVALUATOR_SCRIPTS["scannet200"]` → `os.path.join(_SPELLBOOK, "eval", "scannet200_evaluator.py")`
- Docstring usage line: `python spellbook/eval/evaluate.py …`

**`eval/scannet200_evaluator.py`**
- Same root/path fix if it uses `__file__` for BenchmarkScripts / `from benchmark import`.
- Change to `from eval.benchmark import resolve_benchmark` after inserting `_SPELLBOOK` on path.
- Usage line in module docstring.

### Call sites (flat → package)

| File | Change |
|---|---|
| `spellbook/main.py` | `from eval.benchmark import load_settings, resolve_benchmark` |
| `spellbook/gpu_check.py` | `from eval.benchmark import load_settings` |
| `spellbook/predict/runner.py` | both import sites → `eval.benchmark` |
| `spellbook/utils/scan_lock.py` | `from eval.benchmark import load_settings` |
| `spellbook/utils/visualize.py` | `from eval.benchmark import …`; lazy `from eval.scannet200_evaluator import Evaluator` |
| `spellbook/reconstruct/batch.py` | `from eval.benchmark import load_settings` |
| `spellbook/reconstruct/run.py` | same |
| `spellbook/reconstruct/extract.py` | same |
| `spellbook/reconstruct/cleanup.py` | `from eval.benchmark import BENCHMARKS, artifact_paths, load_settings` |
| `spellbook/predict/models/common.py` | `spec_path = join(spellbook_dir, "eval", "benchmark.py")`; docstring path |

### Docs
`spellbook/PROJECT_STATUS.md`:
- Structure tree: remove top-level `benchmark.py` / `evaluate.py` / `scannet200_evaluator.py`; add `eval/` block.
- Commands: `python spellbook/eval/evaluate.py export-gt|evaluate …`
- Decision 4 / class-list line: `spellbook/eval/benchmark.py` if mentioned.
- Foreign-env decision: still absolute importlib, path now under `eval/`.

### Grep gate (must be empty after)
```bash
rg -n "from benchmark import|import benchmark|scannet200_evaluator|spellbook/benchmark\.py|spellbook/evaluate\.py|spellbook/scannet200_evaluator" \
  spellbook --glob '!archive/**' --glob '!tmp/**' --glob '!**/__pycache__/**'
```
Allowed leftovers: comments inside `eval/` docstring provenance that name the *upstream* Rozenberszki path (not local file path).

---

## Smoke test (add + run)

### New file: `spellbook/tmp/smoke_eval_package.py`

Self-contained, **delete after green** (or keep under `spellbook/tmp/` until next archive pass). No pytest dependency. Exit nonzero on failure.

**Checks (in order):**

1. **Layout**
   - `spellbook/eval/{__init__,benchmark,evaluate,scannet200_evaluator}.py` exist
   - Top-level `spellbook/{benchmark,evaluate,scannet200_evaluator}.py` do **not** exist

2. **Package import (main path)**
   ```python
   sys.path.insert(0, spellbook_dir)
   from eval.benchmark import resolve_benchmark, BENCHMARKS, load_settings, artifact_paths
   s20 = resolve_benchmark("ScanNet20")
   s200 = resolve_benchmark("ScanNet200")
   assert s20.name == "ScanNet20" and len(s20.classes) == 18
   assert s200.name == "ScanNet200" and len(s200.classes) == 198
   assert set(BENCHMARKS) >= {"ScanNet20", "ScanNet200"}
   cfg = load_settings()
   assert "scannet_root" in cfg and "gpu_pool" in cfg
   ```

3. **Foreign-env path (common.py contract)**
   ```python
   # Simulate model-env rule: do NOT put spellbook on sys.path
   from importlib.util import spec_from_file_location, module_from_spec
   path = os.path.join(spellbook_dir, "eval", "benchmark.py")
   spec = spec_from_file_location("spellbook_benchmark", path)
   mod = module_from_spec(spec); spec.loader.exec_module(mod)
   assert mod.resolve_benchmark("ScanNet20").label_to_id  # non-empty
   # And the real helper:
   sys.path.insert(0, os.path.join(spellbook_dir, "predict", "models"))
   from common import _benchmark_spec
   assert _benchmark_spec("ScanNet200").name == "ScanNet200"
   ```

4. **Evaluator script path resolvable**
   - Import/evaluate module constants or recompute path; `os.path.isfile` on scannet200 script under `eval/`.

5. **CLI help (no data required)**
   ```bash
   python spellbook/eval/evaluate.py  # prints usage, exit != 0 ok
   python spellbook/eval/evaluate.py export-gt --help
   python spellbook/eval/evaluate.py evaluate --help
   ```

6. **GT export smoke (data required)** — one official scene if present:
   ```bash
   # only if /data/scannet/scans/scene0568_00 exists
   python spellbook/eval/evaluate.py export-gt --scene 0568_00 --benchmark ScanNet20
   # assert derived GT file exists and is non-empty
   ```
   Skip with clear `[SKIP] no scene0568_00` if missing (do not fail CI-less laptop).

7. **Lazy visualize import path**
   ```python
   from eval.scannet200_evaluator import Evaluator
   assert callable(Evaluator) or hasattr(Evaluator, "evaluate")
   ```

8. **Stale import grep** — subprocess `rg` as above; fail if any hit outside eval provenance comments.

Print one line per check: `OK …` / `FAIL …` / `SKIP …`. Final summary counts. Exit 1 if any FAIL.

### Run command
```bash
/home/rolf/anaconda3/envs/3disspellbook/bin/python spellbook/tmp/smoke_eval_package.py
```

### Additional manual smoke (not in script; operator)
```bash
python -m compileall -q spellbook
python spellbook/main.py --gpu-check   # optional; proves load_settings via eval
# if predictions exist:
python spellbook/eval/evaluate.py evaluate --run-id <id> --models mosaic3d \
  --scenes 0568_00 --benchmark ScanNet20
```

---

## Execution order

1. Create `spellbook/eval/`; `git mv` the three modules; add `__init__.py`.
2. Patch roots/imports inside the three moved files.
3. Patch all call sites + `common.py` path.
4. Write `spellbook/tmp/smoke_eval_package.py`.
5. Update `PROJECT_STATUS.md`.
6. `compileall` + run smoke script.
7. Grep gate.
8. Commit (when asked): message like  
   `refactor: move benchmark/evaluate/scannet200 into spellbook/eval/`
9. Cleanup: delete smoke script after green **or** leave until document skill archives `tmp/`; do not leave broken top-level shims.

---

## Rollback

```bash
git checkout HEAD -- spellbook/
# or reverse the single commit
```
No data migrations. GT/eval artifacts under `/data/scannet/derived/` unchanged.

---

## Risk checklist

| Risk | Mitigation |
|---|---|
| `common.py` still points at old path | Smoke check 3 + grep |
| `_REPO_ROOT` off-by-one after move | Smoke check 4–6; fix dirname chain first |
| Double package name `eval` vs Python builtin | Package is `eval` under spellbook on path as top-level name — **shadows stdlib `eval` builtin only as module name when imported as `import eval`**. Prefer always `from eval.benchmark import …`. If conflict appears, rename package to `evaluation` before merge. **Pre-flight:** `python -c "import eval; print(eval)"` with spellbook on path — if stdlib confusion, use `spellbook_eval` or `evaluation` instead. |
| Visualize AP50 breaks | Smoke check 7 |
| Docs still show old CLI | PROJECT_STATUS Commands block |

**Name collision note (decide before coding):** Python’s builtin is `eval()` the function; `import eval` loads a module and can confuse. Safer package name if any doubt: **`spellbook/evaluation/`** with `from evaluation.benchmark import …`. Prefer `evaluation/` if a 5-second probe shows footguns; otherwise `eval/` is fine for this repo’s insert-spellbook-on-path pattern.

**Decision for implementer:** run
```bash
cd /home/rolf/GIT/ScanNet && python -c "import sys; sys.path.insert(0,'spellbook'); import eval; print(eval, getattr(eval,'__file__',None))"
```
before creating the package. If anything odd, name the package `evaluation` and substitute throughout this plan.

---

## Success criteria

- [ ] Three modules live only under `spellbook/eval/` (or `evaluation/`)
- [ ] All call sites + foreign-env path updated
- [ ] `smoke_eval_package.py` all non-skipped checks OK
- [ ] `compileall` clean
- [ ] Grep gate clean
- [ ] PROJECT_STATUS structure + commands match
- [ ] No behavior change to scores/GT encoding

---

## Suggested improvement (optional follow-up, not this PR)

Single `SPELLBOOK_ROOT` constant module used by `common.py` and evaluate subprocess paths so the next move is one line.
