# Complete GPU distribution checks and smoke test

Status: implementation plan only; do not execute from this document until approved.

For: a build agent working from `/home/rolf/GIT/ScanNet` on `master`.

Goal: every registered prediction and reconstruction method participates in one repeatable
GPU-distribution check. The check launches native environments or containers but stops before
prediction, reconstruction, frame extraction, or result generation. Physical GPU 0 belongs to
the user and must never be locked, assigned, selected, initialized, stopped, or used by
Spellbook. Read-only device enumeration may observe it, but managed workloads automatically share
only physical GPUs 1-4.

## 1. Fixed decisions

1. Keep the existing settings-backed `fcntl.flock` allocator in
   `spellbook/utils/gpu.py`; do not add a scheduler, daemon, database, utilization polling, or
   manual GPU selection.
2. Keep `spellbook/settings.yaml` as the sole pool configuration:

   ```yaml
   gpu_pool: [1, 2, 3, 4]
   ```

3. GPU 0 is user-reserved. Replace every comment or behavior that says GPU 0 is available to
   ZED.
4. Do not add `--gpu`, `--device`, or any equivalent public/manual selector.
5. Run native runtime checks, not inference or reconstruction. A prediction check initializes
   CUDA through that model's own conda Python; a reconstruction check initializes/imports the
   engine's actual Python/container runtime and validates its device mapping.
6. Include all eleven registered methods in one report.
7. Runtime-check nine methods: five predictors, Metashape, RTAB-Map, BundleFusion, and Open3D.
8. Report two explicit expected blocks:
   - ZED: blocked because its current default-device behavior selects physical GPU 0, and
     managed `CUDA_VISIBLE_DEVICES` remapping was not approved for runtime validation. Reference
     closed issue #18.
   - Isaac: blocked because `zed-isaac-nvblox:spellbook` does not exist. Keep issue #22 open.
9. A complete smoke is accepted as `9 PASS + 2 BLOCKED + 0 FAIL`, not as eleven runtime starts.
10. Do not edit `/home/rolf/GIT/openmask3d`, `/home/rolf/GIT/Open3DIS`, or any other external
    model/reconstruction repository. All changes belong under `spellbook/`.
11. Do not use `sudo`, do not build the Isaac image, and do not access NGC credentials.
12. Do not run `--replace`, create frames, write ScanNet scans, or write benchmark predictions.
13. Do not kill or inspect-control user processes on GPU 0. Observation through `nvidia-smi` is
    allowed only to prove that no newly launched Spellbook process appears there.

## 2. Baseline and known constraints

### 2.1 Current shared allocator

- `spellbook/utils/gpu.py::gpu_lease()` already validates live devices, tries the configured pool
  in order, locks `<scannet_root>/derived/locks/gpus/gpu-<index>.lock`, waits when all are held,
  and releases on descriptor close/crash.
- The lease fd can be passed to subprocesses with `pass_fds` and is already forwarded by the
  prediction runner, Open3DIS nested stages, and all managed reconstruction adapters.
- `spellbook/benchmark.py::validate_gpu_pool()` already rejects zero. Change its wording from
  "GPU 0 is reserved for ZED" to "GPU 0 is user-reserved and unavailable to Spellbook."
- Existing tests cover settings validation, same-GPU exclusion, different-GPU concurrency,
  crash release, nested-child fd lifetime, and absence of `gpu-0.lock`.

### 2.2 Current prediction coverage

`spellbook/predict/runner.py` registers five methods, but only four wrappers exist:

| Method | Python | Wrapper | Current state |
|---|---|---|---|
| Mosaic3D | `/data/mosaic3d/conda/envs/mosaic3d/bin/python` | `_mosaic3d_run.py` | present |
| OpenIns3D | `/data/openins3d/conda/envs/openins3d/bin/python` | `_openins3d_run.py` | present |
| OpenYOLO3D | `/data/openyolo3D/conda/envs/openyolo3d/bin/python` | `_openyolo3d_run.py` | present |
| Open3DIS | `/data/open3dis/conda/envs/open3dis/bin/python` | `_open3dis_run.py` | present |
| OpenMask3D | `/home/rolf/anaconda3/envs/openmask3d/bin/python` | `_openmask3d_run.py` | missing |

The working OpenMask3D implementation to adapt is
`/home/rolf/GIT/ov3dis-comparison/utils/_openmask3d_run.py`. The local OpenMask3D repository is
`/home/rolf/GIT/openmask3d`; checkpoints are under `/data/openmask3d/resources/`.

### 2.3 Current reconstruction policies

| Engine | Current policy | Planned check |
|---|---|---|
| ZED | `zed-default` | explicit BLOCKED, no pyzed import/open |
| Open3D | `cpu` | native Open3D import, no lease/CUDA |
| Metashape | `managed` | Metashape Python/API/device enumeration under lease |
| RTAB-Map | `managed` | rootless Podman image/device mapping under lease |
| Isaac | `managed` | explicit BLOCKED from preflight (#22), no lease/container |
| BundleFusion | `managed` | Docker image/device mapping under lease |

`run.py` currently performs frame setup before adapter policy dispatch and does not invoke adapter
preflight for direct runs. `batch.py` preflights engines before shared frame preparation, but its
frame preparation can still invoke ZED SDK on GPU 0 when frames are incomplete or `--replace` is
used. Both paths must be made safe.

### 2.4 Relevant previous failures

- #18: setting `sdk_gpu_id=1` returned constant ZED translations; the old fix was to leave the SDK
  on its default GPU, which currently means GPU 0. The new user policy supersedes that operating
  choice, so ZED must be blocked rather than silently use GPU 0.
- #22: Isaac image/runtime is absent; do not hide this as success.
- #23: parallel RTAB-Map containers require unique names; retain the PID/unique-name pattern.
- #15-#17: prediction scratch/output and supervision have known races. GPU-check mode must bypass
  all scratch/output code and use unbuffered, bounded subprocesses.

## 3. Safety invariants

These are acceptance gates, not suggestions:

1. No code path in the smoke imports pyzed, opens an SVO, extracts frames, or starts ZED tracking.
2. `gpu-0.lock` is never created or opened.
3. Every managed check receives a lease index in `{1,2,3,4}` and a valid inherited lease fd.
4. No two overlapping checks hold the same physical GPU.
5. Prediction children see exactly one CUDA device and use logical `cuda:0`; the report records
   the physical index from the parent lease separately.
6. Metashape may enumerate physical GPUs, but its mask must select only the leased physical index.
7. Containers must receive only the leased assignment through their existing runtime-specific
   mapping. They must be uniquely named, attached, bounded by timeout, and removed in `finally`.
8. Open3D takes no lease and creates no CUDA context.
9. ZED and Isaac take no lease because they are blocked before dispatch.
10. No smoke task writes under `/data/scannet/predictions/`, `/data/scannet/scans/`,
    `/data/scannet/derived/reconstruction/frames/`, or model scratch directories.
11. Cleanup targets only PIDs and container names created by the smoke; never use broad `pkill` or
    delete shared directories.
12. The smoke has one global timeout and a shorter per-check timeout; timeout is a FAIL followed
    by exact-process/container cleanup.

## 4. Target design

### 4.1 Public command

Add one opt-in command through `spellbook/main.py`:

```bash
python spellbook/main.py --gpu-check
```

Behavior:

- `--scene` is not required in GPU-check mode.
- With no filters, all five prediction methods and all six reconstruction engines are checked.
- Existing `--models` and `--engine` may filter checks for debugging, but they never select GPUs.
- `--gpu-check` is mutually exclusive with `--predict`, `--visualize`, and normal reconstruction.
- Normal modes retain their current scene requirements.
- The command prints one final deterministic table with columns:

  ```text
  kind  method  status  policy  physical_gpu  visible_gpu  lease_fd  runtime  seconds  reason
  ```

- Status values are only `PASS`, `BLOCKED`, and `FAIL`.
- Overall success for the complete current-host check is exactly nine PASS, two expected BLOCKED
  (`zed`, `isaac`), and zero FAIL.

### 4.2 Orchestrator

Add `spellbook/gpu_check.py` as the only coordinator. It must reuse, not duplicate, the production
settings loader, allocator, prediction registry/env construction, and reconstruction adapter
metadata.

Responsibilities:

1. Load and validate `gpu_pool`/`scannet_root` through `load_settings()`.
2. Confirm zero is absent before launching anything.
3. Record a read-only baseline mapping of physical GPU index -> UUID -> compute-process PIDs using
   `nvidia-smi`. Do not require GPU 0 to be idle; the user's processes are allowed.
4. Preflight all methods before dispatch and classify the two approved blocks.
5. Submit all runnable checks concurrently:
   - eight managed checks: five prediction methods + Metashape + RTAB-Map + BundleFusion;
   - one CPU check: Open3D.
6. Use enough worker threads for all nine runnable checks. Each managed task acquires its own
   global `gpu_lease`; therefore four acquire immediately and at least four wait/reuse.
7. Hold each native runtime for a small fixed interval (target 5 seconds) after GPU initialization
   so overlap and `nvidia-smi` observation are deterministic.
8. Capture request/acquire/release timestamps per task. Derive waiting and overlap from those
   timestamps instead of parsing human log strings.
9. Sample `nvidia-smi` while the first four managed checks are held. Any new Spellbook child on
   physical GPU 0 is an immediate FAIL; pre-existing user PIDs are ignored.
10. Validate report-wide invariants after all futures complete.
11. Run exact cleanup in `finally`, then verify every pool lock is immediately acquirable.
12. Write human-readable logs only under `spellbook/tmp/logs/gpu-check-<timestamp>/`; do not write
    model/reconstruction results. The final table also prints to stdout.

Keep the orchestrator small. Do not add YAML schemas, plugin systems, generic task frameworks, or a
second model/engine registry.

### 4.3 Native prediction CUDA probe

Add `spellbook/predict/models/_gpu_check.py`, intentionally independent of model dependencies
except PyTorch. The prediction runner launches it with each method's registered conda Python and
the exact same child environment/lease forwarding used by normal prediction.

Probe behavior:

1. Read `CUDA_VISIBLE_DEVICES` and require exactly one decimal physical index in `{1,2,3,4}`.
2. Read `SPELLBOOK_GPU_LEASE_FD`, verify it parses to an open fd (`os.fstat`), and keep it open.
3. Import that environment's `torch`.
4. Require `torch.cuda.is_available()` and `torch.cuda.device_count() == 1`.
5. Allocate and synchronize one tiny tensor on logical `cuda:0` to create a real CUDA context.
6. Print one JSON line containing method, PID, physical index, logical index, device name, and fd.
7. Sleep for the orchestrator-provided bounded hold interval, synchronize, and exit zero.
8. Never import the model, read scene data, create output/scratch directories, or download files.

This is deliberately a distribution smoke, not an inference test. The production runner and the
probe must share one helper for constructing `env` and `pass_fds`; otherwise the probe could pass
while normal prediction remains wired differently.

### 4.4 Reconstruction adapter check contract

Extend `spellbook/reconstruct/engines/__init__.py` documentation with:

```text
gpu_check(gpu=None, lease_fd=None, hold_seconds=5) -> dict
```

Rules:

- Managed adapters require non-`None` `gpu` and `lease_fd`, validate the fd with `os.fstat`, and
  forward it to their blocking native child using `pass_fds`.
- CPU adapters require both arguments to be `None`.
- Blocked adapters return a reason from `preflight()` and are never called.
- The returned dictionary uses the same fields as the final report.
- Check implementations may share tiny subprocess helpers but must remain in their owning adapter
  so runtime-specific mapping cannot drift from normal execution.

## 5. File-specific implementation phases

### Phase 0: baseline gate

Before edits, from `/home/rolf/GIT/ScanNet`:

```bash
git status --short --branch
git log --oneline -10
python -m unittest discover -s spellbook/tests -p 'test_*.py'
python -m compileall -q spellbook
git diff --check
```

Also record, without changing state:

```bash
nvidia-smi -L
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
test ! -e /data/scannet/derived/locks/gpus/gpu-0.lock
podman ps --format '{{.Names}}'
docker ps --format '{{.Names}}'
```

Stop if the worktree has unrelated changes in files this plan owns. Do not revert unrelated user
changes.

### Phase 1: make GPU 0 user-reserved everywhere

Edit:

- `spellbook/benchmark.py`
- `spellbook/utils/gpu.py`
- `spellbook/settings.yaml` comments if any
- `spellbook/reconstruct/run.py`
- `spellbook/reconstruct/batch.py`
- `spellbook/reconstruct/extract.py`
- `spellbook/reconstruct/engines/__init__.py`
- `spellbook/reconstruct/engines/zed.py`
- affected docstrings in prediction/reconstruction runners

Required changes:

1. Replace "GPU 0 reserved for ZED" wording with "GPU 0 user-reserved; Spellbook never uses it."
2. Keep pool validation `g > 0`; do not special-case host topology or add GPU 0 fallback.
3. Change ZED's policy from `zed-default` to an explicit non-runnable policy such as `blocked`.
   Keep its engine index and adapter file intact.
4. Make `zed.preflight()` return a stable reason referencing the user reservation and #18 before
   importing pyzed.
5. Validate `GPU_POLICY` against exactly `managed`, `cpu`, and `blocked`; remove
   `getattr(..., "managed")` fallback so typos cannot silently run unleased.
6. Invoke adapter preflight in direct `reconstruct/run.py` before SVO checks, directory creation,
   frame setup, or any pyzed import.
7. In normal batch mode, retain preflight-before-frames ordering.
8. Prevent implicit ZED extraction:
   - complete existing shared frame sets may be reused;
   - incomplete/missing frame sets must fail before pyzed import;
   - `--replace` must fail with the same policy reason;
   - direct reconstruction without complete `--frames` must fail rather than extract.
9. GPU-check mode bypasses scene/SVO/frame handling entirely.

Open a new GitHub issue during implementation for "ZED disabled while GPU 0 is user-reserved",
reference #18, and leave it open until managed remapping receives a separate pose-validity test.
Do not reopen #18 because its original `sdk_gpu_id` diagnosis remains correct.

### Phase 2: complete OpenMask3D registration

Add:

- `spellbook/predict/models/_openmask3d_run.py`

Adapt only the established implementation from:

```text
/home/rolf/GIT/ov3dis-comparison/utils/_openmask3d_run.py
```

Required Spellbook adaptations:

1. Use the same CLI contract as the other wrappers:

   ```text
   --pointcloud --frames --classes ... --out --benchmark
   ```

2. Remove `--gpu`; rely exclusively on runner-provided `CUDA_VISIBLE_DEVICES`.
3. Use `/data/openmask3d/resources/scannet200_model.ckpt` and
   `/data/openmask3d/resources/sam_vit_h_4b8939.pth`.
4. Use the existing ScanNet frame layout, especially `intrinsic_color.txt`.
5. Replace ov3dis helpers with `models/common.py`:
   - `_benchmark_spec()`;
   - `scene_id_from_pointcloud()`;
   - `write_scannet_submission()`;
   - existing decimation helper where behavior matches.
6. Preserve the established point-limit and mask-dedup behavior; do not tune model quality.
7. Because the runner exposes one physical GPU as logical CUDA device 0, set the local
   OpenMask3D feature-stage override to logical `0`, never the physical index.
8. Forward `SPELLBOOK_GPU_LEASE_FD` with `pass_fds` to both blocking OpenMask3D subprocesses.
9. Keep parent calls blocking so no native child outlives the lease.
10. Do not write or patch the external OpenMask3D checkout.

The live smoke does not run OpenMask3D inference. Acceptance in this task is wrapper existence,
compile/help/preflight, nested-fd contract tests, and successful CUDA runtime probing through its
registered conda environment.

### Phase 3: add prediction preflight and shared child wiring

Edit:

- `spellbook/predict/runner.py`

Add:

- `spellbook/predict/models/_gpu_check.py`
- `spellbook/tests/test_predict_dispatch.py`

Required runner behavior:

1. Define one supported-model ordering containing all five methods; require all existing model
   maps to have exactly that key set.
2. Add a small `preflight_model(model) -> None | reason` checking:
   - registered conda Python exists and is executable;
   - registered wrapper exists;
   - OpenMask3D repository/checkpoints/SAM checkpoint exist;
   - no CUDA context is initialized during preflight.
3. Unknown requested models become a clear input error rather than warning-and-skip.
4. Preflight requested models before frame extraction, output directory creation, or lease
   acquisition.
5. Factor the normal child environment/lease wiring into one helper used by both `_run_one()` and
   GPU-check dispatch:
   - `CUDA_VISIBLE_DEVICES=<physical lease index>`;
   - `SPELLBOOK_GPU_LEASE_FD=<fd>`;
   - `pass_fds=(fd,)`;
   - OpenYOLO3D's existing `LD_LIBRARY_PATH` hook.
6. GPU-check dispatch launches `_gpu_check.py` with the model's own registered Python. It must not
   call `_run_one`, create an output directory, append a `.tasks` marker, or extract frames.
7. Use `Popen` with captured unbuffered output and a timeout, not an unbounded `subprocess.run`.
8. On timeout/failure, terminate then kill only that exact child and wait for it.

Do not change normal prediction algorithms, prompts, class lists, benchmark paths, or output
formats.

### Phase 4: add reconstruction native checks

Edit:

- `spellbook/reconstruct/engines/open3d.py`
- `spellbook/reconstruct/engines/metashape.py`
- `spellbook/reconstruct/engines/rtabmap.py`
- `spellbook/reconstruct/engines/isaac.py`
- `spellbook/reconstruct/engines/bundlefusion.py`
- `spellbook/reconstruct/engines/zed.py`
- `spellbook/reconstruct/engines/__init__.py`

Per-engine behavior:

#### ZED

- `GPU_POLICY = "blocked"`, `SERIAL = True`.
- `preflight()` returns the stable policy reason before pyzed import.
- No `gpu_check()` runtime launch occurs.
- The smoke reports BLOCKED without a lease.

#### Open3D

- Keep `GPU_POLICY = "cpu"`.
- `gpu_check(None, None, hold_seconds)` imports the installed Open3D runtime, reports its version,
  and remains alive for the hold interval.
- It must not import pyzed, read frames, instantiate CUDA APIs, or acquire a lease.

#### Metashape

- Keep `GPU_POLICY = "managed"`, `SERIAL = True`.
- Run `/data/zed-metashape/conda/env/bin/python` with a tiny temporary script that imports
  Metashape, enumerates GPU devices, validates the leased physical index, computes the same
  `gpu_mask = 1 << gpu`, checks license visibility, prints JSON, and sleeps.
- Clear inherited `CUDA_VISIBLE_DEVICES`, because Metashape uses physical-index masks.
- Forward the lease fd and remove the temporary script in `finally`.
- Do not open a project, photos, depth maps, or output paths.

#### RTAB-Map

- Keep `GPU_POLICY = "managed"` and the existing unique container-name rule.
- Reuse `/home/rolf/GIT/zed-rtabmap/scripts/podman_gpu.sh <physical-gpu>` to build the exact same
  rootless device/library arguments as normal execution.
- Start `localhost/zed-rtabmap:jazzy` with `--rm`, a unique `spellbook-gpu-check-rtabmap-<pid>` name,
  the leased mapping, and a bounded shell command that validates the in-container environment,
  invokes the available NVIDIA runtime probe, prints JSON, and sleeps.
- Forward the lease fd to the attached Podman client.
- In `finally`, remove only that exact container if still present.

#### Isaac

- Keep `GPU_POLICY = "managed"`, `SERIAL = True`, and issue #22 open.
- Keep/strengthen preflight so missing `zed-isaac-nvblox:spellbook` returns a reason before lease
  acquisition.
- Add a `gpu_check()` implementation compatible with the RTAB-Map `podman_gpu.sh` device mapping
  for future use; do not use the currently broken CDI `--device nvidia.com/gpu=all` path.
- Do not build or pull an image in this task. On the current host the orchestrator records BLOCKED
  and never invokes `gpu_check()`.

#### BundleFusion

- Keep `GPU_POLICY = "managed"`.
- Start `bundlefusion:latest` through Docker with the same GPU mapping strategy as normal
  execution, `--rm`, and a unique `spellbook-gpu-check-bundlefusion-<pid>` name.
- The bounded command validates the assigned environment/runtime, prints JSON, and sleeps; it must
  not launch BundleFusion reconstruction or create `~/.bf_tmp` data.
- Forward the lease fd to the attached Docker client.
- In `finally`, remove only that exact container if still present.

For all three runnable managed adapters, reject `gpu is None` or `lease_fd is None`. Apply the same
guard to normal `reconstruct()` entry points so direct adapter calls cannot silently fall back to
GPU 0/all GPUs. Remove BundleFusion/Isaac/RTAB-Map expressions such as `gpu or 0` and
`gpu if gpu is not None else 0`.

### Phase 5: implement the complete orchestrator and CLI

Add:

- `spellbook/gpu_check.py`

Edit:

- `spellbook/main.py`

Implementation sequence:

1. Parse `--gpu-check` before requiring `--scene`.
2. Resolve method filters; defaults are all five prediction methods and all six engines.
3. Preflight everything and create BLOCKED rows for ZED and unavailable Isaac.
4. Submit the eight managed tasks and Open3D CPU task concurrently.
5. Each managed task:
   - records request time;
   - enters `gpu_lease(settings["gpu_pool"], settings["scannet_root"])`;
   - records acquisition time and physical index;
   - calls prediction or adapter native check with the lease index/fd;
   - records release time after the context exits.
6. Use a start barrier so all runnable tasks are ready before contention begins.
7. Keep four initial holders alive long enough for the remaining four to enter the allocator's
   wait path.
8. While holders are active, map new compute PIDs from `nvidia-smi` to physical indices. Ignore
   baseline PIDs and fail if any new PID is on GPU 0.
9. Validate interval overlap: tasks assigned the same GPU must have non-overlapping
   acquire/release intervals.
10. Require all managed assignments to be in the configured pool and all four pool indices to be
    exercised at least once.
11. Require at least one managed task to acquire after waiting; expected is four.
12. Require Open3D to overlap managed checks without owning a lock.
13. Sort the final report by fixed method order, not completion order.
14. Exit nonzero on any FAIL, unexpected block, missing row, duplicate row, GPU-0 use, leaked lock,
    or cleanup failure.
15. Treat the two approved current-host blocked rows as non-failures, but print them prominently.

Do not build a result dashboard. One stdout table, one summary JSON, and per-task text logs are
enough.

### Phase 6: unit and contract tests (no real GPU)

Extend:

- `spellbook/tests/test_gpu.py`
- `spellbook/tests/test_reconstruct_dispatch.py`

Add:

- `spellbook/tests/test_predict_dispatch.py`
- optional small fake child under `spellbook/tests/` only if the existing
  `_gpu_lease_child.py` cannot express the case cleanly

Required tests:

1. The configured pool remains `[1,2,3,4]`; zero, duplicates, negatives, empty, booleans, and
   non-integers are rejected.
2. All five prediction registry maps have exactly the supported key set.
3. All five registered wrappers exist; OpenMask3D is no longer a dangling registration.
4. Prediction preflight runs before output/frame/lease work.
5. The shared prediction child-env helper sets one physical `CUDA_VISIBLE_DEVICES`, forwards the
   same lease fd through env and `pass_fds`, and retains OpenYOLO3D's library path.
6. Prediction GPU-check creates no output directory or task marker.
7. Open3DIS and OpenMask3D nested subprocesses forward the lease fd.
8. Every engine declares one valid policy and a callable `preflight`/`gpu_check` as appropriate.
9. Expected engine policies are:

   ```text
   zed=blocked, open3d=cpu,
   metashape=managed, rtabmap=managed, isaac=managed, bundlefusion=managed
   ```

10. Direct reconstruction checks preflight before scene/SVO/frame handling.
11. ZED preflight returns before importing pyzed.
12. Missing/incomplete frames and `--replace` cannot invoke ZED extraction under the user-reserved
    GPU-0 policy; complete frames remain reusable.
13. Managed adapter `reconstruct()` and `gpu_check()` reject missing gpu/fd rather than falling
    back to zero/all devices.
14. Open3D check takes no lease; ZED/Isaac blocked checks take no lease.
15. Four fake managed tasks acquire GPUs 1-4 concurrently and a fifth waits, then reuses a released
    GPU.
16. No test creates `gpu-0.lock`.
17. Public CLIs still reject manual `--gpu`.
18. `--gpu-check --help` needs no scene, while normal predict/reconstruct still require scenes.
19. Orchestrator report validation rejects GPU 0, duplicate simultaneous assignments, missing
    methods, unexpected blocks, and leaked locks.
20. Signal/timeout cleanup addresses only registered child PIDs/container names.

Patch hardware/runtime calls in unit tests. Unit discovery must not import pyzed, initialize CUDA,
run Docker/Podman, consume a Metashape license, or touch `/data/scannet` lock files.

### Phase 7: static verification

Run in the activated `3disspellbook` environment:

```bash
python -m unittest discover -s spellbook/tests -p 'test_*.py'
python -m compileall -q spellbook
python spellbook/main.py --help
python spellbook/main.py --gpu-check --help
python -m spellbook.reconstruct.run --help
git diff --check
```

Then search for forbidden/fallback behavior:

```bash
rg --glob '*.py' --glob '*.yaml' --glob '*.md' \
  'reserved for (the )?ZED|sdk_gpu_id|gpu or 0|else 0|--gpu' spellbook
```

Expected exceptions must be reviewed manually:

- historical issue references may mention #18;
- CLI rejection tests intentionally contain `--gpu`;
- comments may say `sdk_gpu_id` remains unset, but no code may set it;
- no runtime fallback may resolve a missing managed assignment to GPU 0.

### Phase 8: live preflight without launch

Before the complete live smoke:

1. Activate `3disspellbook` so runtime environment variables are correct, even though ZED is
   blocked.
2. Verify all physical GPUs 1-4 exist.
3. Record GPU 0's existing user PIDs; do not stop them.
4. Verify no stale Spellbook checks/containers are present.
5. Verify lock files are currently acquirable; persistent empty files are normal.
6. Run the command's preflight-only internal stage (or a model/engine-filtered check with no native
   launch if implemented) and require exactly:

   ```text
   predictors: 5 runnable
   open3d/metashape/rtabmap/bundlefusion: runnable
   zed: BLOCKED (GPU 0 user-reserved / #18)
   isaac: BLOCKED (missing zed-isaac-nvblox:spellbook / #22)
   ```

If another method is unavailable, stop and fix its preflight/runtime before the complete smoke;
do not silently add it to the allowed-block list.

### Phase 9: complete live smoke

Run exactly once:

```bash
python spellbook/main.py --gpu-check
```

Expected execution:

1. Open3D starts its CPU-only hold without a lease.
2. Eight managed checks request leases together.
3. Four acquire GPUs 1, 2, 3, and 4.
4. Four print one wait event each (the allocator itself prints one shared-style wait line per
   process/task, not polling spam).
5. The first four native checks initialize and hold their runtime.
6. `nvidia-smi` evidence shows their new contexts only on GPUs 1-4. GPU 0 may contain user
   processes from the baseline, but no new smoke PID.
7. After release, the waiting checks acquire/reuse GPUs 1-4 and complete.
8. ZED and Isaac never acquire locks or start children.
9. Final table has exactly eleven unique rows:

   ```text
   PASS:    mosaic3d openins3d openyolo3d open3dis openmask3d
   PASS:    open3d metashape rtabmap bundlefusion
   BLOCKED: zed isaac
   FAIL:    none
   ```

10. Total model/reconstruction result files created: zero.

The smoke tests distribution only. Do not wait for model predictions, meshes, poses, QC, or
benchmark results.

### Phase 10: cleanup and post-smoke audit

The command must perform cleanup automatically. Verify it independently afterward:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
podman ps -a --format '{{.Names}}'
docker ps -a --format '{{.Names}}'
test ! -e /data/scannet/derived/locks/gpus/gpu-0.lock
git status --short
```

Additional checks:

1. No `spellbook-gpu-check-*` process/container remains.
2. Every `gpu-1..4.lock` can be acquired nonblocking; keep the persistent files.
3. No new directory exists under:

   ```text
   /data/scannet/predictions/
   /data/scannet/scans/
   /data/scannet/derived/reconstruction/frames/
   /data/mosaic3d/scratch/
   /data/openins3d/scratch/
   /data/openyolo3D/scratch/
   /data/open3dis/scratch/
   /data/openmask3d/output/
   ~/.bf_tmp/
   ```

4. No `.tasks` marker timestamp changed.
5. No user's GPU-0 process was signaled, killed, or renamed.
6. Retain only the smoke's small logs/summary under `spellbook/tmp/logs/` until documentation is
   complete.

If cleanup fails, treat the smoke as FAIL even if device assignments were correct.

### Phase 11: final verification and documentation

After the live smoke passes:

```bash
python -m unittest discover -s spellbook/tests -p 'test_*.py'
python -m compileall -q spellbook
git diff --check
git status --short --branch
git diff --stat
git diff
```

Documentation workflow:

1. Use the `document` skill.
2. Update `spellbook/PROJECT_STATUS.md` only with durable structure and policy:
   - GPU 0 is user-reserved and never used by Spellbook;
   - managed pool is 1-4;
   - `--gpu-check` command and method matrix;
   - ZED disabled pending validated managed remapping;
   - Isaac blocked by #22.
3. Put ZED reasoning/attempts in the new GitHub issue, not PROJECT_STATUS.
4. Update #22 only with factual Isaac check status; do not close it.
5. Move this completed plan to `spellbook/archive/` only after implementation and verification.
6. Remove transient smoke logs after recording the concise durable outcome, unless a failed check
   needs issue evidence.

## 6. Required pass/fail summary

The implementation is accepted only when all of these hold:

| Gate | Required result |
|---|---|
| Unit tests | all pass without real GPU/container/model work |
| Registry | 5 predictors + 6 engines, no missing wrapper |
| Runtime smoke | 9 PASS, 2 approved BLOCKED, 0 FAIL |
| Managed assignments | all in `{1,2,3,4}` |
| Contention | 4 immediate holders, at least 1 waiter, safe reuse |
| Simultaneous exclusivity | no overlapping owner on one GPU |
| GPU 0 | no new Spellbook PID/context; no lock file |
| Lease fd | valid and forwarded to every managed native child |
| Open3D | CPU-only, no lease/CUDA context |
| ZED | blocked before pyzed/frame/SVO work |
| Isaac | blocked before lease/container, #22 remains open |
| Outputs | no prediction/reconstruction/frame/task artifacts |
| Cleanup | no children/containers/held locks remain |
| Manual GPU controls | absent |

## 7. Rollback and failure handling

1. If static/unit tests fail, do not run the live smoke.
2. If any runnable method fails preflight, stop; do not broaden the expected-block list.
3. If any Spellbook PID appears on GPU 0, terminate only the exact smoke-owned process/container,
   mark FAIL, and investigate mapping before retrying.
4. If a child/container survives timeout, remove only its recorded unique name/PID and verify its
   lease release before another attempt.
5. If OpenMask3D integration requires an external repository edit, stop and revise the plan; do
   not patch `/home/rolf/GIT/openmask3d` in this task.
6. If ZED must become runnable later, create a separate plan: managed lease +
   `CUDA_VISIBLE_DEVICES=<leased GPU>` before pyzed import, `sdk_gpu_id` unset, 60-frame
   nonconstant-pose gate, then full trajectory validation. That work is explicitly excluded here.
7. Keep only one implemented solution. Remove experimental flags/helpers that are not part of the
   passing design before final verification.

## 8. Expected changed files

```text
spellbook/main.py
spellbook/gpu_check.py                                  (new)
spellbook/settings.yaml                                (comment only, if needed)
spellbook/benchmark.py
spellbook/utils/gpu.py
spellbook/predict/runner.py
spellbook/predict/models/_gpu_check.py                  (new)
spellbook/predict/models/_openmask3d_run.py             (new)
spellbook/reconstruct/run.py
spellbook/reconstruct/batch.py
spellbook/reconstruct/extract.py
spellbook/reconstruct/engines/__init__.py
spellbook/reconstruct/engines/zed.py
spellbook/reconstruct/engines/open3d.py
spellbook/reconstruct/engines/metashape.py
spellbook/reconstruct/engines/rtabmap.py
spellbook/reconstruct/engines/isaac.py
spellbook/reconstruct/engines/bundlefusion.py
spellbook/tests/test_gpu.py
spellbook/tests/test_predict_dispatch.py                (new)
spellbook/tests/test_reconstruct_dispatch.py
spellbook/PROJECT_STATUS.md                             (documentation phase only)
spellbook/tmp/gpu_distribution_complete_smoke_plan.md   (this plan; archive after completion)
```

Do not modify ScanNet upstream directories, external model repositories, generated data, or
environment installations as part of this task.
