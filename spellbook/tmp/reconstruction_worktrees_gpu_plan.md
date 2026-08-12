# Reconstruction worktrees and automatic GPU scheduling

For: build agents and the integration operator

Goal: land one shared reconstruction/GPU core, then debug ZED, Open3D, Metashape,
RTAB-Map, Isaac, and BundleFusion in six non-overlapping Git worktrees. Prediction and
managed reconstruction jobs automatically share physical GPUs 1-4; physical GPU 0 is
never represented by a lock file and remains reserved for ZED SDK extraction/tracking.

This plan is intentionally split into a core phase and an engine phase. Do not create an
engine worktree until the core phase has been merged into `master` and accepted.

## Fixed decisions

- Use the installed OCX worktree plugin for creation and session forking.
- Close the existing `feature/reconstruction` worktree before creating the core worktree.
- Use one core worktree, merge it, then create six sibling engine worktrees from the exact
  same merged core commit.
- Keep one editing session per worktree. The main checkout is integration-only while
  engine worktrees exist.
- Use `gpu_pool: [1, 2, 3, 4]` in `spellbook/settings.yaml` as the sole managed GPU list.
- Never create or acquire `gpu-0.lock`. ZED SDK code leaves `sdk_gpu_id` unset and uses
  physical GPU 0 as its default device.
- Remove all public/manual `--gpu` options and `gpus` function arguments. Physical GPU
  indices may still be passed internally from the allocator to an adapter that needs one.
- Prediction, Metashape, RTAB-Map, Isaac, and BundleFusion wait for an automatic lease on
  GPUs 1-4. Open3D's legacy TSDF is CPU-only and takes no lease.
- Prepare shared SVO frames once before parallel engine debugging; never use `--replace`
  while engine worktrees exist.
- Clean up plugin worktrees sequentially, one pending deletion at a time, with verification
  after every deletion. Merge engine branches only after all six worktrees are closed.

## Verified starting state

As of 2026-08-12:

- Main checkout: `/home/rolf/GIT/ScanNet`, branch `master`, HEAD `992f24d`.
- Existing plugin worktree:
  `/home/rolf/.local/share/opencode/worktree/af8d5a7a5325e9bc2684882359e0eea504f05d5d/feature/reconstruction`.
- `feature/reconstruction` is clean and currently points to the same commit as `master`.
- Plugin session: `ses_018891f24ffexksXMTVELsvw63`.
- Plugin state DB:
  `/home/rolf/.local/share/opencode/plugins/worktree/af8d5a7a5325e9bc2684882359e0eea504f05d5d.sqlite`.
- `.opencode/worktree.jsonc` has no copy, symlink, or hook actions.
- Main is dirty with staged reconstruction-document renames, modified `AGENTS.md`,
  `spellbook/PROJECT_STATUS.md`, `spellbook/utils/hud.py`,
  `spellbook/utils/visualize.py`, and untracked `spellbook/tests/`. Treat these as existing
  user work; inspect and commit by intent, never discard or combine them blindly.

## Plugin facts that constrain the workflow

The installed plugin uses ordinary `git worktree add`, so source checkout isolation is
sound. Its session/lifecycle integration has the following verified limitations:

1. The forked reconstruction session retained the main checkout as its recorded cwd even
   though the new terminal opened for the worktree. Every session must therefore verify
   its actual shell cwd and every agent tool call must use the explicit worktree path.
2. Forked session history transfers, but plan/delegation copying did not transfer in the
   observed session. Every engine session must read this tracked plan from its own checkout.
3. `worktree_delete` queues one project-wide pending operation. Any session-idle event may
   process it, so two deletions must never be requested concurrently.
4. Deletion executes `git add -A`, creates an always-present
   `chore(worktree): session snapshot` commit, force-removes the checkout, and preserves the
   branch. A worktree must be clean before deletion so the snapshot commit is tree-empty.
5. Only the original plugin-created session is registered for deletion. Do not replace it
   with `/new` or resume that session from the main checkout.

## Safety invariants

Stop immediately if any invariant fails.

1. Before editing in a new worktree, all four commands below must identify that worktree:

   ```bash
   pwd
   git rev-parse --show-toplevel
   git branch --show-current
   git status --short
   ```

2. Before any `worktree_create`, `master` must be clean and no branch creation may occur
   while `pending_operations` contains a row.
3. All six engine branches start at one recorded `CORE_SHA`; no engine branch is based on
   another engine branch.
4. Engine branches do not edit core files, shared documentation, or another engine adapter.
5. Runtime writes for one `(scene, engine)` are issued by only one process.
6. Shared extracted frames are read-only while engine worktrees exist.
7. No worktree deletion starts until all jobs in that worktree have stopped and Git status
   is clean, including untracked files. Verify with
   `test -z "$(git status --porcelain)"` immediately before requesting deletion.
8. A second deletion is not requested until the previous worktree path is absent and the
   plugin pending-operation table is empty.

## Phase 0: establish a clean baseline

Work only in `/home/rolf/GIT/ScanNet`.

1. Inspect all current changes before staging anything else:

   ```bash
   git status --short --branch
   git diff
   git diff --cached
   git log --oneline -10
   ```

2. Split existing changes into intent-specific commits. At minimum, keep the staged
   reconstruction archive moves separate from the visualizer/HUD/tests changes; include
   this plan in an appropriate planning/documentation commit.
3. Do not commit generated logs, evaluation artifacts, model outputs, cache files, secrets,
   or unrelated user work.
4. Verify the main checkout is clean:

   ```bash
   git status --porcelain
   ```

   Acceptance: no output.

## Phase 1: close the old reconstruction worktree

This phase must finish before a new plugin worktree is created.

1. In the existing `feature/reconstruction` checkout, fast-forward to the now-clean
   `master` baseline:

   ```bash
   git merge --ff-only master
   git status --porcelain
   git log --oneline master..HEAD
   ```

   Acceptance: clean status and no implementation divergence before deletion.
2. Record the branch tip:

   ```bash
   git rev-parse HEAD
   ```

3. Confirm registered session `ses_018891f24ffexksXMTVELsvw63` is still live in OpenCode.
   If it is live, invoke `worktree_delete` once from that session with reason
   `replace monolithic reconstruction tree with isolated core and engine trees`. If it is
   not live, skip the plugin call and use the clean-worktree fallback below; a different
   session cannot safely impersonate the registered owner.
4. Do not request any other deletion. From the main checkout, wait until all checks pass:

   ```bash
   git worktree list --porcelain
   sqlite3 /home/rolf/.local/share/opencode/plugins/worktree/af8d5a7a5325e9bc2684882359e0eea504f05d5d.sqlite \
     'select * from pending_operations; select * from sessions;'
   git branch --list feature/reconstruction
   ```

   Acceptance: old path absent, pending table empty, old session row absent, branch still
   present.
5. Inspect the plugin snapshot before integration:

   ```bash
   git show --stat --oneline feature/reconstruction
   git diff --exit-code master feature/reconstruction
   ```

   Acceptance: the snapshot has no tree changes.
6. Fast-forward `master` through the empty snapshot, then delete the merged branch normally:

   ```bash
   git merge --ff-only feature/reconstruction
   git branch -d feature/reconstruction
   ```

### Deletion fallback

If the registered session is dead, or the plugin leaves a pending row/worktree after its
deletion session becomes idle, do not call `worktree_delete` again. Inspect status and the
pending row, then use ordinary `git worktree remove <path>` only when
`test -z "$(git -C <path> status --porcelain)"` succeeds. Use `--force` only after explicit
user approval and a recorded diff. Clear stale plugin state only after the Git worktree and
preserved branch have been verified.

## Phase 2: create the shared core worktree

1. Start a fresh OpenCode session from the clean main checkout.
2. Invoke:

   ```text
   worktree_create(
     branch: "feature/reconstruction-core",
     baseBranch: "master"
   )
   ```

3. In the newly opened terminal, run the four-command worktree identity gate from the
   safety invariants. The expected branch is `feature/reconstruction-core`; the two paths
   must equal the plugin-created core path.
4. Read this plan from the core checkout before editing. Do not rely on copied session plan
   state.

## Phase 3: implement the shared core

All changes in this phase precede the engine split.

### 3.1 Settings

Edit `spellbook/settings.yaml`:

```yaml
default: ScanNet20
scannet_root: /data/scannet
gpu_pool: [1, 2, 3, 4]
```

Edit `spellbook/benchmark.py` so `load_settings()` also returns `gpu_pool` and validates:

- value is a non-empty list;
- every member is an integer greater than zero;
- members are unique;
- zero is rejected explicitly.

Do not run `nvidia-smi` from `load_settings()`, because visualization/evaluation must remain
usable on CPU-only sessions. Validate configured indices against live hardware only when a
GPU workload requests a lease.

### 3.2 Minimal shared allocator

Add `spellbook/utils/gpu.py` with one small public context manager:

```python
with gpu_lease(gpu_pool, scannet_root) as lease:
    physical_gpu = lease.index
```

Required behavior:

- derive the lock directory from settings as
  `<scannet_root>/derived/locks/gpus/`;
- use persistent index-named files `gpu-1.lock` through `gpu-4.lock` and
  `fcntl.flock(LOCK_EX | LOCK_NB)`;
- try configured indices in settings order, sleep briefly, and retry when all are held;
- print one wait line and one acquisition line, not continuous polling output;
- hold the open descriptor for the entire child/job lifetime;
- release by closing the descriptor in `finally`; kernel cleanup must cover exceptions,
  SIGTERM, and SIGKILL;
- expose the descriptor for `subprocess(..., pass_fds=(lease.fileno(),))` so the actual GPU
  process tree retains the lease if an orchestration parent exits;
- factor live-device discovery into a small `_present_gpus()` helper that runs
  `nvidia-smi -L` with `CUDA_VISIBLE_DEVICES` removed from its child environment; tests
  patch this helper rather than probing real hardware;
- fail before waiting if a configured GPU is not present according to that helper;
- never create, inspect, or lock `gpu-0.lock`;
- do not select GPUs by utilization or free-memory polling.

Do not add a daemon, queue database, status service, UUID mapping, manual allocation CLI,
or another settings section.

### 3.3 Prediction integration

Edit `spellbook/main.py`:

- remove public `--gpu` parsing;
- stop forwarding a GPU list to prediction or reconstruction;
- update help text to say tasks use the automatic pool from `settings.yaml`.

Edit `spellbook/predict/runner.py`:

- remove `gpu_count()`, `queue.Queue`, and the `gpus` argument to `predict()`;
- load `gpu_pool` from settings;
- keep at most `len(gpu_pool)` local worker threads;
- acquire a global lease inside each task immediately before launching its model;
- set the child-only `CUDA_VISIBLE_DEVICES` to the acquired physical index;
- pass the lease descriptor to the child and hold the parent context until the child exits;
- put the descriptor number in a private child environment variable so
  `_open3dis_run.py` can forward the same descriptor with `pass_fds` to its nested
  Open3DIS subprocess; no nested GPU process may outlive the lease;
- log the physical index selected automatically;
- keep extraction of `.sens` frames CPU-only and sequential as it is now.

Edit these wrappers:

```text
spellbook/predict/models/_mosaic3d_run.py
spellbook/predict/models/_openins3d_run.py
spellbook/predict/models/_openyolo3d_run.py
spellbook/predict/models/_open3dis_run.py
```

For every wrapper:

- remove `--gpu` parsing and the pre-import argv scan that sets
  `CUDA_VISIBLE_DEVICES`;
- rely on the runner-provided environment;
- address the selected visible GPU only as `cuda` or `cuda:0`;
- make Mosaic3D stop constructing `cuda:<physical index>`.

Do not change prediction algorithms, thresholds, prompts, output layouts, or external model
repositories in this task. OpenMask3D's missing wrapper is unrelated and remains out of
scope.

### 3.4 Reconstruction integration

Edit `spellbook/reconstruct/run.py`:

- remove public `--gpu` parsing;
- load the adapter before execution and read its core-established `GPU_POLICY`;
- for `managed`, acquire a settings-backed lease inside `run.py` for this task and pass the
  automatic physical index and lease descriptor to the adapter;
- for `zed-default` and `cpu`, take no lease and pass `gpu=None`;
- hold the lease through engine execution, ScanNet conversion, finalization, and QC so the
  engine cannot outlive its reservation;
- remove inherited `CUDA_VISIBLE_DEVICES` before any ZED SDK frame extraction/import so
  the SDK default remains physical GPU 0;
- do not set host `CUDA_VISIBLE_DEVICES` for managed reconstruction adapters, because
  RTAB-Map's helper and the container adapters need the physical index. They already map
  that internal index into their own runtime.
- clear inherited `CUDA_VISIBLE_DEVICES` at `run.py` startup before GPU discovery or adapter
  import. In particular, Metashape must enumerate all physical devices before applying its
  physical-index `gpu_mask`.

Edit `spellbook/reconstruct/batch.py`:

- remove `_gpu_count()`, the `gpus` argument, `--gpu`, and the local free-GPU queue;
- stop passing `--gpu` to `spellbook.reconstruct.run`;
- derive worker count from `len(settings["gpu_pool"])`, capped by task count;
- let each `run.py` child contend for the global lease; `batch.py` only limits local
  concurrency and never chooses a physical GPU;
- describe progress rows as worker slots until a child's `[gpu] acquired N` line identifies
  its automatic GPU;
- keep frame extraction sequential and clear inherited `CUDA_VISIBLE_DEVICES` before it;
- keep ZED tasks serial within the one ZED worktree;
- consume adapter preflight/serialization metadata rather than duplicating engine image and
  environment paths in `batch.py`.

Edit `spellbook/reconstruct/extract.py`:

- remove unused `gpu` parameters from `read_gravity`, `ensure_frames`, and `extract`;
- retain the documented SDK 5.4 rule: never set `sdk_gpu_id`;
- do not add a GPU 0 lock.

Edit `spellbook/reconstruct/engines/__init__.py` to document this adapter contract:

```text
GPU_POLICY = "zed-default" | "cpu" | "managed"
SERIAL = bool
preflight() -> None | reason string
reconstruct(work, root, gpu=None, lease_fd=None) -> (mesh, poses, keep, convention)
```

During this core phase, add the minimal metadata/preflight function to each adapter:

| Adapter | `GPU_POLICY` | `SERIAL` |
|---|---|---|
| `zed.py` | `zed-default` | `True` |
| `open3d.py` | `cpu` | `False` |
| `metashape.py` | `managed` | `True` |
| `rtabmap.py` | `managed` | `False` |
| `isaac.py` | `managed` | `True` |
| `bundlefusion.py` | `managed` | `False` |

Move the existing centralized preflight checks into the corresponding adapters so later
engine debugging can change an image/path and its check in the same owned file.

Every managed adapter must pass `lease_fd` to its blocking GPU child with `pass_fds`:

- Metashape Python in `metashape.py`;
- the RTAB-Map `podman run` process in `rtabmap.py`;
- the Isaac `podman run` process in `isaac.py`;
- the BundleFusion `docker run` process in `bundlefusion.py`.

Use an empty `pass_fds` tuple when `lease_fd is None`. Build/preflight subprocesses that run
before the managed workload stay inside the lease context. Clear inherited
`CUDA_VISIBLE_DEVICES` from Metashape's child environment before it computes `gpu_mask`.
Keep container clients attached and blocking; on interruption, stop the named container
before allowing the owning process to exit. The inherited descriptor protects a surviving
host client; a stale container whose owner/client was forcibly killed must be stopped before
another task is started.

### 3.5 ZED/Open3D decoupling

Add `spellbook/reconstruct/tsdf.py` by moving the shared implementation currently stored as
`engines/open3d.py::_integrate` and `_ZED_TO_OPENCV` into a neutral public function such as
`integrate_tsdf(work, keep, poses=None)`.

Edit both adapters:

- `engines/zed.py` imports `integrate_tsdf` from `..tsdf`;
- `engines/open3d.py` becomes a thin Stage-A pose-selection adapter that imports the same
  function;
- no engine imports another engine;
- preserve the corrected inverse extrinsic, actual frame-dimension intrinsic scaling,
  20 mm legacy TSDF deviation, and pose conjugation exactly.

Do not tune reconstruction quality in the core refactor. Compare outputs against the
existing scene9009 Open3D baseline before accepting the move.

### 3.6 Worktree-safe paths and documentation

Edit `spellbook/reconstruct/qc.py` so the default reference path is derived from
`Path(__file__)` or `os.path.dirname(__file__)`; remove the hardcoded
`feature/reconstruction` worktree path.

Edit `spellbook/PROJECT_STATUS.md` commands once in core:

- remove every reconstruction/prediction `--gpu` example;
- state that managed tasks automatically share settings GPUs 1-4;
- state that GPU 0 is unmanaged and reserved for ZED SDK work.

Engine branches must not edit `PROJECT_STATUS.md` later. Record engine debugging details in
their GitHub issues and per-scan `/data/scannet/scans/<scan>/recon/qc.yaml`.

Sweep affected module docstrings in the same core commit so none still describe manual
`--gpu` selection or a process-local GPU queue.

## Phase 4: core tests and acceptance

Add `spellbook/tests/test_gpu.py` and focused reconstruction dispatch tests. Tests must use a
temporary lock directory or temporary `scannet_root`; they must not reserve a real GPU.

Required tests:

1. Settings accepts `[1, 2, 3, 4]` and rejects empty, duplicate, non-integer, negative, and
   zero-containing pools.
2. Two processes cannot hold the same configured lock simultaneously.
3. Different configured GPU locks can be held concurrently.
4. A waiter acquires after the holder exits normally.
5. A waiter acquires after a bare holder with no children is killed.
6. No operation creates `gpu-0.lock`.
7. With `_present_gpus()` and subprocess launch mocked, prediction child environment exposes
   one configured physical GPU and the wrapper uses visible `cuda:0`; this unit test must
   not initialize CUDA or reserve a real GPU.
8. ZED and Open3D dispatch without managed leases; the other four adapters dispatch with a
   lease.
9. Public CLIs reject `--gpu` as an unknown argument.
10. If an orchestration parent is killed while its child inherited the descriptor, a waiter
    remains blocked until that child exits.
11. Each managed adapter forwards `lease_fd` to its blocking Metashape/Docker/Podman child.

Run at minimum:

```bash
python -m unittest discover -s spellbook/tests -p 'test_*.py'
python -m compileall -q spellbook
python spellbook/main.py --help
python -m spellbook.reconstruct.run --help
git diff --check
```

Live acceptance on this five-A4000 host:

1. Start five harmless test holders through the Python allocator. Four acquire GPUs 1-4;
   the fifth waits and starts after one exits.
2. Observe that GPU 0 remains absent from the lock directory.
3. Run one prediction smoke task without any GPU option and verify it executes on one of
   physical GPUs 1-4.
4. Run Open3D on scene9009 and verify it creates no managed lease while preserving the
   established trajectory/backprojection behavior.
5. Run ZED tracking serially and verify GPU 0 plus the established checks from issue #18:
   approximately 67.65 m trajectory, 100% valid poses, and no constant-pose regression.
6. Verify a managed reconstruction adapter logs an automatic GPU in 1-4 and never uses 0.

Do not start the six engine worktrees if core tests fail.

## Phase 5: close and merge the core worktree

1. Commit only the core files listed above and its tests. Inspect status, diff, and recent
   history before committing.
2. Verify the core worktree is clean and record `PRE_DELETE_CORE`.
3. Invoke `worktree_delete` once from the original core session.
4. Verify path removal, empty `pending_operations`, removal of the core session row, and
   preservation of `feature/reconstruction-core`.
5. Verify the snapshot is tree-empty relative to `PRE_DELETE_CORE`.
6. From the clean main checkout:

   ```bash
   git merge --ff-only feature/reconstruction-core
   git rev-parse HEAD
   ```

7. Record the result as `CORE_SHA`. Keep `feature/reconstruction-core` until all six engine
   branches have integrated.
8. Run the core unit tests once from `master`.
9. Populate and validate the shared frame sets for every debugging scene serially before
   creating engine worktrees. Using the automatic Open3D command is acceptable because it
   performs the one-time ZED extraction and then CPU TSDF integration:

   ```bash
   python spellbook/main.py --scene 9004 9009 --engine open3d
   ```

   Verify `frames_complete()` for each
   `/data/scannet/derived/reconstruction/frames/sceneNNNN/frames` directory. Do not use
   `--replace` after this gate.

## Phase 6: create six sibling engine worktrees

Freeze `master` at `CORE_SHA`; do not edit or commit in main while creating or debugging
the siblings.

Invoke these tool calls sequentially from the main session, waiting for each creation to
finish before the next:

```text
worktree_create(branch: "debug/reconstruction-zed",         baseBranch: "master")
worktree_create(branch: "debug/reconstruction-open3d",      baseBranch: "master")
worktree_create(branch: "debug/reconstruction-metashape",   baseBranch: "master")
worktree_create(branch: "debug/reconstruction-rtabmap",     baseBranch: "master")
worktree_create(branch: "debug/reconstruction-isaac",       baseBranch: "master")
worktree_create(branch: "debug/reconstruction-bundlefusion", baseBranch: "master")
```

For every new terminal:

1. Run the worktree identity gate.
2. Verify `git rev-parse HEAD` equals `CORE_SHA` before the first edit.
3. Read this plan from that checkout.
4. Record the plugin-created session ID and keep that original session for cleanup.
5. Do not start a second editing session in that worktree.

After all six creations, verify:

```bash
git worktree list --porcelain
sqlite3 /home/rolf/.local/share/opencode/plugins/worktree/af8d5a7a5325e9bc2684882359e0eea504f05d5d.sqlite \
  'select id, branch, path from sessions order by branch; select * from pending_operations;'
```

Acceptance: main plus exactly six engine worktrees, six corresponding session rows, and no
pending operation.

## Phase 7: strict engine ownership

Each branch may edit only its adapter and uniquely named tests/research:

| Branch | Production file | Optional test |
|---|---|---|
| `debug/reconstruction-zed` | `spellbook/reconstruct/engines/zed.py` | `spellbook/tests/test_reconstruct_zed.py` |
| `debug/reconstruction-open3d` | `spellbook/reconstruct/engines/open3d.py` | `spellbook/tests/test_reconstruct_open3d.py` |
| `debug/reconstruction-metashape` | `spellbook/reconstruct/engines/metashape.py` | `spellbook/tests/test_reconstruct_metashape.py` |
| `debug/reconstruction-rtabmap` | `spellbook/reconstruct/engines/rtabmap.py` | `spellbook/tests/test_reconstruct_rtabmap.py` |
| `debug/reconstruction-isaac` | `spellbook/reconstruct/engines/isaac.py` | `spellbook/tests/test_reconstruct_isaac.py` |
| `debug/reconstruction-bundlefusion` | `spellbook/reconstruct/engines/bundlefusion.py` | `spellbook/tests/test_reconstruct_bundlefusion.py` |

If research files are required, use unique names such as
`spellbook/tmp/debug-reconstruction-<engine>.md`. Never edit the shared plan from an engine
branch.

Forbidden engine-branch edits include:

```text
spellbook/settings.yaml
spellbook/benchmark.py
spellbook/main.py
spellbook/utils/gpu.py
spellbook/reconstruct/__init__.py
spellbook/reconstruct/batch.py
spellbook/reconstruct/run.py
spellbook/reconstruct/extract.py
spellbook/reconstruct/tsdf.py
spellbook/reconstruct/scannet.py
spellbook/reconstruct/finalize.py
spellbook/reconstruct/qc.py
spellbook/PROJECT_STATUS.md
another engine adapter
```

Before declaring an engine complete, run this gate in its worktree, substituting its two
allowed paths:

```bash
git diff --exit-code "$CORE_SHA"...HEAD -- . \
  ':(exclude)spellbook/reconstruct/engines/<engine>.py' \
  ':(exclude)spellbook/tests/test_reconstruct_<engine>.py' \
  ':(exclude)spellbook/tmp/debug-reconstruction-<engine>.md'
```

Acceptance: exit code 0. Also inspect `git diff --name-only "$CORE_SHA"...HEAD` manually.

### Shared-core defect protocol

If an engine uncovers a defect in a forbidden core file:

1. Stop all six debugging sessions and all runtime jobs.
2. Do not patch the defect in an engine branch.
3. Create one dedicated core-fix worktree from current `master`, implement/test one shared
   commit, close it sequentially, and fast-forward/merge it into `master`.
4. In every still-live engine worktree, merge the updated `master` before resuming.
5. Verify all six branches contain the identical core fix and re-record the common base.

Do not cherry-pick six independently authored versions of the same fix and do not rebase
published/live worktree branches.

## Phase 8: runtime isolation while debugging

Source isolation does not isolate `/data`, GPUs, containers, model repositories, or conda
environments. Apply these rules:

1. Shared frames are read-only. No engine session uses `--replace` or invokes extraction
   against an incomplete frame directory.
2. Engine scan IDs remain fixed and disjoint:
   ZED `_00`, Metashape `_01`, RTAB-Map `_02`, Isaac `_03`, Open3D `_04`, BundleFusion `_05`.
3. Do not launch the same `(scene, engine)` twice. One engine worktree has one active batch
   command at a time.
4. Only the ZED worktree performs ZED tracking; its scenes run serially on unmanaged GPU 0.
5. Managed reconstruction and all prediction jobs may run from independent worktrees or
   main; the shared locks serialize them across GPUs 1-4 automatically.
6. Open3D CPU work may overlap managed GPU work, subject to host RAM limits.
7. Metashape license and Isaac serial restrictions remain one-task-at-a-time in their sole
   engine worktrees.
8. Do not edit `/home/rolf/GIT/Open3DIS`, `/home/rolf/GIT/OpenIns3D`,
   `/home/rolf/GIT/OpenYOLO3D`, `/home/rolf/GIT/Mosaic3D`,
   `/home/rolf/GIT/zed-rtabmap`, or other external repositories from these worktrees. If an
   external change is unavoidable, stop and create a separate worktree in that repository.
9. Use unique prediction `--run-id` values. GPU leases do not solve the known shared scratch
   and atomic-output issues tracked in GitHub issues #15-#17.
10. Keep logs and bulky artifacts under `/data/scannet`; do not add generated output to Git.

Per-engine acceptance requires:

- its adapter-specific unit/smoke test;
- one complete scene output under the fixed ScanNet suffix;
- consistent mesh/pose/keep lengths;
- `segs == verts`;
- a written `recon/qc.yaml`;
- automatic managed GPU in 1-4, or the documented ZED/Open3D no-lease policy;
- issue update for unresolved quality/runtime defects.

## Phase 9: freeze and sequentially close all engine worktrees

Do not merge an engine branch while any engine worktree is still being edited. Finish and
close all six first so `master` remains the frozen common base.

For each engine, one at a time:

1. Stop its reconstruction process and containers; verify no process still uses its output.
2. Run tests, `git diff --check`, and the ownership gate.
3. Commit intended adapter/test changes. Confirm `git status --porcelain` is empty.
4. Record `PRE_DELETE_<ENGINE>=$(git rev-parse HEAD)` outside the session notes.
5. Pause the other five OpenCode sessions.
6. Invoke `worktree_delete` once from this engine's original plugin-created session.
7. Wait for all of the following before touching the next engine:

   - exact worktree path absent from `git worktree list --porcelain`;
   - exact session row absent from the plugin DB;
   - `pending_operations` empty;
   - preserved branch still exists;
   - `git diff --exit-code PRE_DELETE_<ENGINE> debug/reconstruction-<engine>` succeeds.

8. If any check fails, stop. Do not queue another deletion and do not delete the branch.

Recommended closure order: Open3D, ZED, BundleFusion, Metashape, RTAB-Map, Isaac. Closure
order does not affect Git integration because all branches share `CORE_SHA` and own disjoint
files.

After the sixth deletion, acceptance is:

```bash
git worktree list --porcelain
sqlite3 /home/rolf/.local/share/opencode/plugins/worktree/af8d5a7a5325e9bc2684882359e0eea504f05d5d.sqlite \
  'select * from sessions; select * from pending_operations;'
```

Only the main worktree remains; both queries return no rows.

## Phase 10: integrate preserved branches

Work only in the clean main checkout. Before merging, inspect every complete branch, not
just its last commit:

```bash
git status --short --branch
git log --oneline --decorate --graph --all -30
git diff --stat "$CORE_SHA"...debug/reconstruction-<engine>
git diff "$CORE_SHA"...debug/reconstruction-<engine>
```

Merge one branch at a time in this order:

1. `debug/reconstruction-open3d`
2. `debug/reconstruction-zed`
3. `debug/reconstruction-bundlefusion`
4. `debug/reconstruction-metashape`
5. `debug/reconstruction-rtabmap`
6. `debug/reconstruction-isaac`

Use a normal non-fast-forward feature merge so each debugging effort remains identifiable:

```bash
git merge --no-ff debug/reconstruction-<engine>
```

After each merge:

- rerun that engine's focused tests and the shared GPU tests;
- inspect `git status` and the merge diff;
- abort and investigate any conflict in a core/shared file, because the ownership gates
  should make such a conflict impossible;
- do not proceed to the next branch until tests pass.

After all six merges:

```bash
python -m unittest discover -s spellbook/tests -p 'test_*.py'
python -m compileall -q spellbook
git diff --check feature/reconstruction-core..master
git status --short --branch
```

Run the six-engine smoke matrix with no GPU argument and verify automatic waiting across
GPUs 1-4 plus serial ZED use of GPU 0. Do not use `--replace`.

Delete only branches Git reports merged:

```bash
git branch --merged master
git branch -d debug/reconstruction-open3d
git branch -d debug/reconstruction-zed
git branch -d debug/reconstruction-bundlefusion
git branch -d debug/reconstruction-metashape
git branch -d debug/reconstruction-rtabmap
git branch -d debug/reconstruction-isaac
git branch -d feature/reconstruction-core
git worktree prune --dry-run
```

Run `git worktree prune` only if the dry run reports expected stale metadata.

## Phase 11: documentation and cleanup

1. Use the `document` skill after the build and integration tests complete.
2. Keep durable architecture and commands in `spellbook/PROJECT_STATUS.md`.
3. Put per-engine failures and measured debugging evidence in GitHub issues and each scan's
   `recon/qc.yaml`, not in project status.
4. Move this completed plan and any uniquely named research reports to
   `spellbook/archive/reconstruction/` according to the document skill.
5. Verify no generated logs, lock files, datasets, model artifacts, or worktree paths are
   tracked.
6. Do not push or create a PR unless explicitly requested.

## Final acceptance checklist

- Existing monolithic reconstruction worktree is gone before core creation.
- Core was merged before six siblings were created.
- Six engine branches all started at one recorded `CORE_SHA`.
- No engine branch changed another engine or a shared core file.
- ZED and Open3D no longer import one another.
- `settings.yaml` is the sole source for managed GPUs `[1, 2, 3, 4]`.
- No public or model-wrapper `--gpu` option remains.
- Prediction and four managed reconstruction engines acquire global leases automatically.
- GPU 0 has no lock file and is used only for serial ZED SDK extraction/tracking.
- A killed holder releases its managed GPU lease without stale state.
- Shared frames were prepared once and never replaced during parallel debugging.
- Plugin deletions occurred sequentially and every pending operation was verified clear.
- Only the main worktree remains after integration.
- All six branches merged without shared-file conflicts and all tests/QC gates completed.
