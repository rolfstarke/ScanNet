# Scene9004 CAD Geometry Benchmark Plan

Status: detailed implementation plan only. Nothing in this plan has been executed.

## 1. Goal

Build a lean, deterministic CAD-referenced geometry benchmark for custom ScanNet-style
reconstructions. For every configured scene9004 benchmark run, produce exactly one comparable
quality number and one diagnostic image:

- `recon/geometry_score.yaml`: one performance value, `geometry_score` in `[0, 100]`.
- `recon/cad_comparison.png`: plan and elevation views of matched, missing, and extra surface
  voxels.

Use the score to retain at most ten scene9004 runs per reconstruction engine. When all ten slots
are occupied, delete the run with the lowest comparable geometry score and remove its exact
prediction/evaluation artifacts before reusing the slot; every non-benchmark scene keeps only
engine run 0 and replaces it on the next reconstruction.

## 2. Locked Decisions

- Metric: observed surface-voxel F1, not filled-volume IoU.
- Surface grid: fixed 10 mm cells in one shared CAD coordinate frame.
- Match tolerance: 50 mm by default, configurable for standalone rescoring.
- Evaluation value: only the harmonic-mean F1 is published as a performance metric.
- No PASS/FAIL verdict and no minimum score gate.
- Reference domain: first-hit CAD surface cells seen from at least two scene9004 capture poses, not
  invisible backs of closed CAD solids. Captured depth is not compared or scored.
- Reconstruction precision domain: all reconstructed surface cells inside the broad configured
  evaluation bounds. Unsupported geometry therefore counts as extra geometry.
- CAD source:
  `/data/08_TestEnvironments/06_BTU_LG2C_R312_order/02_3D-Model/LG2C_Raum312_v2.3dm`.
- The user has confirmed that this v2 CAD and scene9004 both contain the matching chaos furniture
  arrangement, despite the CAD parent directory name.
- CAD export is a one-time Ubuntu operation using cached Rhino render meshes. The temporary
  exporter is deleted after validated output exists.
- The CAD's modeled architecture and full furniture are included, except transparent window glass.
  Curves are omitted because they are not surfaces; the one unmeshed SubD is documented as omitted;
  a floor surface is synthesized from the CAD room footprint.
- Run suffix: tens digit is the engine, units digit is the run number.
- Engine ranges: ZED `_00.._09`, Metashape `_10.._19`, RTAB-Map `_20.._29`, Isaac `_30.._39`,
  Open3D `_40.._49`, BundleFusion `_50.._59`.
- The ten run slots and geometry-score eviction policy apply only to scene9004. Every other scene
  uses engine run 0 and automatically replaces that scan and its predictions on rerun.
- Existing custom reconstruction directories are deleted, not migrated to the new naming scheme.
- Raw SVO symlinks and canonical shared frame pools are retained.
- The only complete scene9004 frame set is promoted before any scan deletion.
- Legacy reconstruction QC is removed completely: no `qc.py`, `scannet_reference.yaml`,
  `recon/qc.yaml`, or `--no-qc` path remains for new runs.
- `spellbook/benchmark.py` and `spellbook/evaluate.py` remain unchanged in purpose. Prediction AP
  evaluation is separate from reconstruction geometry scoring.
- No `bench/` package or new benchmark hierarchy is introduced. Permanent benchmark code remains
  under `spellbook/reconstruct/`.
- No new permanent benchmark tests are added. Synthetic metric, allocation, cleanup, and locking
  tests are temporary and removed after verification.
- GPU 0 remains user-reserved. Geometry scoring is CPU-only and starts only after any managed GPU
  lease has been released.
- No engine debugging is included. ZED extraction remains blocked by #29 and Isaac remains tracked
  by #22.

## 3. Verified Baseline

### Repository

- Repository: `/home/rolf/GIT/ScanNet`.
- Branch at planning time: `master`, ahead of `origin/master` by 13 commits.
- Existing user change: `spellbook/PROJECT_STATUS.md` is modified with engine-worktree topology.
  Preserve those edits while updating reconstruction documentation.
- Existing unrelated untracked data: `spellbook/tmp/session-backups/`. Do not modify, delete, or
  commit it as part of this work.
- The current 45-test suite passes in the `3disspellbook` environment.
- No relevant closed GitHub issue contains an earlier CAD/voxel benchmark implementation. Closed
  issues #9 and #26 do establish that frame extraction must be serialized and that reconstruction
  consumes the canonical `<work_root>/frames` directory; preserve those fixes.
- Relevant open issues: #25 reconstruction drift/density, #29 blocked ZED extraction, #22 Isaac,
  #16 prediction atomicity, and #17 batch supervision.

### Current custom data

- Existing scan directories:
  - `/data/scannet/scans/scene9004_04`
  - `/data/scannet/scans/scene9009_00`
  - `/data/scannet/scans/scene9009_04`
- No current custom-scene prediction index was found under `/data/scannet/predictions`.
- Shared frame root currently contains only
  `/data/scannet/derived/reconstruction/frames/scene9009`.
- `scene9004_04/recon/frames` is a real 4.7 GB directory, not a symlink.
- It contains 3,612 pose-state entries and the complete color/depth/pose/intrinsic/gravity data
  needed for reuse.
- Fresh extraction cannot replace these frames while #29 blocks ZED access.
- Raw SVOs exist as symlinks for scenes 9004, 9005, 9006, 9007, and 9009.
- `scene9004.svo2` resolves to the chaos recording
  `/data/08_TestEnvironments/04_BTU_LG2C_R312_chaos/01_SVO/18-27-20_raum312_v2/HD1200_SN46307300_18-27-20_raum312_v2.svo2`.

### CAD source

- Source SHA-256 at planning time:
  `ff00c139adededf46f94f69c7872a566f2da2617224305b6dd8583d5dd04994a`.
- Rhino units are metres; model tolerance is 1 mm.
- 1,283 objects on 18 layers: 1,025 Breps, 156 Extrusions, 37 PolylineCurves, 6 ArcCurves,
  58 PolyCurves, and 1 SubD.
- Every one of the 9,700 Brep faces and all 156 Extrusions has a cached render mesh accessible to
  `rhino3dm 8.17.0` in the `3discomp` environment.
- Cached surface total before triangulating quads: 729,062 vertices and 696,367 faces.
- Triangulated total: 1,055,622 triangles.
- CAD bounds from cached surfaces are approximately 12.305 x 7.633 x 3.310 m.
- Furniture layers are populated and must not be dropped.
- Layer `Fensterglas / Griff` has no material metadata, but geometry cleanly separates 18 panes
  from 50 hardware objects: panes have minimum extent at most 5 mm and maximum extent at least
  0.5 m. Eight curves on that layer are omitted with the other non-surface curves.
- No explicit floor layer exists.

## 4. Target Files and Artifacts

### Permanent repository changes

| Action | Path | Purpose |
|---|---|---|
| Add | `spellbook/reconstruct/geometry_score.py` | Reference loading, alignment, surface voxelization, F1, atomic YAML/PNG, standalone CLI |
| Add | `spellbook/reconstruct/geometry_reference.yaml` | Scene-specific immutable reference and metric configuration |
| Add | `spellbook/utils/scan_lock.py` | Shared/exclusive scan artifact locks used by reconstruction and prediction |
| Modify | `spellbook/reconstruct/__init__.py` | New engine-digit/run-digit ID helpers and validation |
| Modify | `spellbook/reconstruct/run.py` | Scene9004 slot selection/eviction, default run-0 replacement elsewhere, post-GPU score hook, QC removal |
| Modify | `spellbook/reconstruct/batch.py` | Stop duplicating IDs; discover the child-selected run ID; remove QC stage text |
| Modify | `spellbook/predict/runner.py` | Serialize frame extraction with an exclusive scan lock, then downgrade to shared while models read |
| Modify | `spellbook/PROJECT_STATUS.md` | New naming, score, data layout, commands, test count, and QC removal while preserving user edits |
| Delete | `spellbook/reconstruct/qc.py` | Remove legacy ScanNet-reference PASS/FAIL QC |
| Delete | `spellbook/reconstruct/scannet_reference.yaml` | Remove legacy QC bars/reference |

Do not modify `spellbook/benchmark.py`, `spellbook/evaluate.py`, engine adapters, archived plans,
`.gitignore`, or the unrelated session backup directory.

### Temporary implementation files

- `spellbook/tmp/scene9004_geometry_benchmark_audit.md`: planning audit evidence; retain with this
  plan until final documentation is externalized.
- `spellbook/tmp/export_scene9004_cad.py`: one-time cached-render-mesh export and floor creation.
- `spellbook/tmp/build_scene9004_reference.py`: one-time capture alignment and visible-cell build.
- `spellbook/tmp/test_geometry_score.py`: synthetic metric, allocator, cleanup, and locking smoke
  tests.
- `spellbook/tmp/reset_custom_runs.py`: guarded dry-run/apply cleanup of old custom scans.

Delete these four files after their corresponding outputs and smoke tests have passed. They are not
part of the runtime API.

Retain this plan in `spellbook/tmp/` through implementation and verification. After the build,
externalize the final durable decisions with the repository's `document` skill before deciding
whether the plan itself can be removed.

### Persistent data artifacts

Create `/data/scannet/custom/reference/` with:

- `scene9004_cad.ply`: merged, triangulated, metre-scale v2 CAD without transparent panes, plus
  synthetic floor.
- `scene9004_visible_voxels.npz`: sorted unique visible CAD voxel indices as signed `int32`.

Read capture poses in place from the old scan during the one-time build; do not create a persistent
pose copy. Keep the temporary reference-alignment QA image under `spellbook/tmp` only until the
visible-cell file has been validated.

Promote the shared frame pool to:

- `/data/scannet/derived/reconstruction/frames/scene9004/frames`

Every new scored scene9004 scan writes:

- `/data/scannet/scans/<scene_id>/recon/geometry_score.yaml`
- `/data/scannet/scans/<scene_id>/recon/cad_comparison.png`

## 5. Metric Contract

### 5.1 Sets and score

Let `C` be sorted visible CAD surface voxel centres and `R` be sorted reconstruction surface voxel
centres after rigid gauge alignment. Both use the same 10 mm grid and fixed origin.

```text
precision = count(r in R where nearest_distance(r, C) <= tau) / count(R)
recall    = count(c in C where nearest_distance(c, R) <= tau) / count(C)
F1        = 2 * precision * recall / (precision + recall)
geometry_score = 100 * F1
```

Use `tau = 0.05 m` by default. Convert occupied indices to fixed-grid voxel-centre coordinates and
use `scipy.spatial.cKDTree` on those metre-valued coordinates. This remains deterministic, avoids
dense 3D arrays, and remains valid after rigid float-coordinate transforms.

Require both `C` and `R` to be non-empty. If both precision and recall are zero, define `F1 = 0`
rather than evaluating `0/0`; require the published `geometry_score` to be finite and in `[0, 100]`.

Precision and recall remain internal diagnostics. The YAML exposes only `geometry_score` as a
performance result; reference/metric/alignment fields are metadata, not additional scores.

### 5.2 Why surface voxels

- Do not fill CAD or reconstruction interiors.
- Filled occupancy would compare wall/furniture thickness and watertightness rather than observed
  reconstruction geometry.
- A fixed 10 mm surface grid normalizes mesh tessellation and removes random surface sampling.
- A 50 mm Euclidean neighborhood supplies the physical tolerance; exact cell overlap is not used.
- The fixed grid, reference hashes, voxel size, and tolerance make runs directly comparable.

### 5.3 Reconstruction input

- Score `<scene_id>_vh_clean_2.ply`, the canonical ScanNet mesh consumed by prediction and
  visualization. The 10 mm surface grid makes the denser `_vh_clean.ply` unnecessary.
- Reject missing, empty, non-finite, or non-triangularizable meshes with a clear runtime error.
- Apply the `axisAlignment` matrix from `<scene_id>.txt` exactly once before gauge normalization.
- Treat `axisAlignment` as a world transform: transform mesh vertices by `A`; transform any
  camera-to-world poses by left multiplication `A @ pose`.
- Validate the alignment matrix determinant near 1 and reject scale/shear.

### 5.4 Rigid gauge normalization

Different engines have arbitrary global world origins and 90-degree Manhattan ambiguity. Remove
only that gauge freedom; do not scale or deform a reconstruction.

1. Apply ScanNet `axisAlignment` to float mesh vertices.
2. Make a preliminary 10 mm surface voxelization for robust floor/centre statistics only.
3. Estimate floor from the 1st percentile of preliminary voxel-centre z and translate it to CAD
   z=0.
4. For horizontal-centre estimation only, take the preliminary x/y median, retain cells whose x/y
   offset from that median is inside the configured CAD half-extents plus 1 m, then use the midpoint
   of their 1st/99th percentile x/y bounds. Cells excluded from alignment estimation are restored
   for final scoring and therefore still count as extra geometry.
5. Translate that robust centre to the CAD room centre.
6. Apply each of the four yaw candidates `0, 90, 180, 270` degrees about the z axis through the CAD
   room centre/origin to the float mesh vertices and voxelize each candidate on the final fixed CAD
   grid.
7. Select the yaw with the largest F1 against the full exported CAD surface-cell set at the
   configured 50 mm tolerance; break exact ties by the smallest yaw. This resolves only the four-way
   Manhattan gauge and does not use the visibility-limited benchmark score as a registration target.
8. Apply no scale, roll/pitch optimization, unconstrained ICP, local warp, or per-region alignment.
9. Store the chosen rigid 4x4 transform in score metadata.

The discrete yaw choice is permitted because global orientation is an arbitrary reconstruction
gauge. Local drift, wrong scale, bent walls, missing surfaces, and duplicated geometry remain in
the score.

### 5.5 Fixed observed CAD reference

The reference is built once and never recomputed per engine:

1. Exclude transparent panes during CAD export. On layer `Fensterglas / Griff`, classify an object
   as glass exactly when its minimum bounding-box extent is at most 0.005 m and its maximum extent
   is at least 0.5 m; assert 18 panes excluded, 50 hardware objects retained, and 8 curves omitted.
2. Normalize the exported CAD to metres, floor z=0, and architecture-room x/y centre at zero.
3. Estimate the z/floor/centre/yaw gauge transform from old `scene9004_04_vh_clean_2.ply`, then
   apply that exact world transform to both the old mesh and its camera-to-world capture poses.
4. If this fixed floor/centre/four-yaw procedure does not visibly align the four room walls, stop.
   Do not add ICP, scale fitting, or a score-optimizing registration fallback.
5. Select exactly `min(180, frame_count)` unique pose indices with integer-rounded `numpy.linspace`
   from frame 0 through the final frame, including both endpoints. Record `max_poses: 180`, the
   endpoint-inclusive selection rule, ray resolution, and `min_views: 2` in
   `geometry_reference.yaml`.
6. Cast 480 x 300 pinhole rays from each selected pose into the CAD with Open3D
   `RaycastingScene`. Scale intrinsic row 0 by `480/source_width` and row 1 by
   `300/source_height`; do not load or compare captured depth.
7. Map every finite first-hit CAD point to the fixed 10 mm grid and require the cell to be hit from
   at least two distinct source poses.
8. Intersect those cells with the full exported CAD surface-cell set, sort, deduplicate, and store
   them in `scene9004_visible_voxels.npz`.

This CAD-only visibility mask removes wall backs, solid interiors, occluded furniture backs, and
CAD geometry outside the captured camera views. It remains identical for every reconstruction run.

### 5.6 Grid and bounds

- Store one CAD-normalized grid origin in `geometry_reference.yaml` as exact decimal metres.
- Store broad fixed bounds equal to the normalized CAD bounds expanded by 25 m horizontally and
  5 m vertically. Open3D's voxel grid is sparse, so empty space does not allocate dense cells.
- Voxelize through Open3D's documented triangle-mesh voxelization with explicit bounds; never use
  each mesh's own minimum as the final comparable grid origin. The discarded preliminary pass in
  Section 5.4 may use deterministic snapped local bounds because only relative robust percentiles
  are consumed from it; it never supplies final score cells.
- Detect vertices outside configured bounds. Before publishing the initial reference version,
  expand bounds rather than silently discarding geometry; after publication, treat out-of-bounds
  geometry as an invalid/failed run rather than changing the bounds.
- Set `max_surface_voxels: 12000000` in `geometry_reference.yaml`. Fail above that cap and do not
  substitute random sampling.

### 5.7 Score compatibility

`geometry_score.yaml` records:

```yaml
metric: observed_surface_voxel_f1_v1
geometry_score: 87.3
scene: scene9004_40
reference_sha256: "..."
visible_voxels_sha256: "..."
voxel_mm: 10
distance_threshold_mm: 50
alignment:
  axis_alignment_applied: true
  yaw_deg: 0
  transform: [...16 row-major values...]
comparison: recon/cad_comparison.png
```

Do not add PASS/FAIL, a quality threshold, precision, recall, Chamfer, or Hausdorff values.
Reject a non-finite or out-of-range `geometry_score` instead of publishing it.

Scores are comparable only when metric version, CAD hash, visible-cell hash, voxel size, and
tolerance all match the active reference configuration. Before selecting an eviction victim,
rescore complete stale scene9004 runs with the current configuration. If rescoring fails, classify
that slot as failed/unscored and reuse it before consuming a free slot.

### 5.8 Diagnostic PNG

- Use matplotlib's `Agg` backend.
- Left panel: x/y plan; right panel: long-axis/z elevation.
- Green: matched CAD/reconstruction cells.
- Red: visible CAD cells with no reconstruction match.
- Blue: reconstruction cells with no visible CAD match.
- Use deterministic stride decimation for plotting only; never decimate metric cells.
- Equal aspect ratio, CAD bounds, scene ID, metric name, tolerance, and the one geometry score are
  shown in the title.
- Write PNG and YAML to same-directory temporary files and publish with `os.replace`.
- A failure leaves any prior complete outputs untouched and removes temporary files.

## 6. Run-ID and Retention Contract

### 6.1 ID helpers

In `spellbook/reconstruct/__init__.py`:

- Keep `ENGINE_INDEX` values `0..5`, but document that they are method digits.
- Change `scan_id(scene_num, engine, run_num)` to require `run_num in 0..9` and compute
  `ENGINE_INDEX[engine] * 10 + run_num`.
- Change `scan_dir` to require the run number.
- Add strict parsing helpers for a 12-character `sceneNNNN_MM` ID.
- Keep the official ScanNet filename shape unchanged.
- Do not provide a silent old-style default run argument; missing call-site updates should fail
  during testing rather than recreate `_04` for Open3D.

Examples:

```text
scan_id(9004, "zed", 0)          -> scene9004_00
scan_id(9004, "metashape", 3)    -> scene9004_13
scan_id(9004, "open3d", 0)       -> scene9004_40
scan_id(9004, "bundlefusion", 9) -> scene9004_59
```

### 6.2 Slot selection

`run.py` automatically allocates a slot; do not add a user-facing manual run-number flag.

1. Run engine preflight before taking locks or creating output paths.
2. Prepare or validate the canonical shared reconstruction frames under the source-scene frame lock
   described in Section 6.3, then release that lock before allocation.
3. Acquire an exclusive scene-engine lock at
   `/data/scannet/derived/locks/reconstruction/sceneNNNN_<engine>.lock`.
4. Hold this lock until reconstruction, finalization, segmentation, and any configured scoring
   finish.
5. Read the retention policy from `geometry_reference.yaml`: scene9004 has `retained_runs: 10`;
   every scene absent from the file has `retained_runs: 1`.
6. For a non-benchmark scene, always select engine run 0. Under its exclusive scan lock, purge the
   previous run's exact prediction/evaluation artifacts, delete the previous complete or partial
   scan directory, and build its replacement in the same ID.
7. Do not score non-benchmark scenes and do not retain additional unit-digit runs for them.
8. For scene9004, inspect exactly the ten IDs in the engine's range.
9. Rescore complete scene9004 runs whose score metadata is missing or stale.
10. Reuse the lowest-numbered incomplete, failed, or still-unscored scene9004 slot first. Purge its
    exact prediction/evaluation artifacts and remove the partial scan directory before rebuilding
    that same ID.
11. If none exists, use the lowest-numbered absent slot.
12. If all ten contain comparable scores, choose the lowest `geometry_score`; break exact ties by
    the oldest `recon/cmdline.txt` mtime, then the lowest run number. That victim ID is the selected
    slot: acquire its exclusive per-scan lock before Section 7 cleanup, delete it, and rebuild the
    same ID. There is no separate unlocked destination slot.
13. Keep `--replace` semantics limited to canonical shared-frame re-extraction. Default
    non-benchmark scan
    replacement is automatic and does not overload that flag.

A complete run requires `.sens`, `.txt`, both clean meshes, segmentation JSON, and
`recon/final_poses.npy`. A completed scene9004 benchmark run additionally requires a valid current
`geometry_score.yaml`.

### 6.3 Lock scopes and ordering

Add `spellbook/utils/scan_lock.py` with a small `fcntl.flock` context manager. Lock files persist;
the kernel releases ownership on process death. Define every caller's path in this module:

- Canonical reconstruction-frame lock:
  `/data/scannet/derived/locks/reconstruction/frames/sceneNNNN.lock`.
- Scene-engine allocation lock:
  `/data/scannet/derived/locks/reconstruction/sceneNNNN_<engine>.lock`.
- Per-scan artifact lock: `/data/scannet/derived/locks/scans/<scene_id>.lock`.
- Global prediction-index lock: `/data/scannet/derived/locks/prediction-artifacts.lock`; every
  acquisition of this lock is exclusive.

`batch._prepare_frames()` and direct `run.py` frame setup acquire the source-scene frame lock
exclusively while checking or replacing
`/data/scannet/derived/reconstruction/frames/sceneNNNN/frames`. They release it before acquiring a
scene-engine lock. This serializes concurrent batch/direct `--replace` operations without holding a
GPU or a scan slot.

Reconstruction then acquires the scene-engine lock followed by the selected per-scan lock and holds
both through cleanup, build, and any configured score. In the full-pool case the selected slot is
the eviction victim, so this same lock protects the victim from prediction while it is deleted.
Exact old prediction cleanup briefly acquires the global prediction-index lock exclusively while
the scan lock is held, releases it before requesting a GPU lease, and never removes shared parent
directories.

`predict()` acquires exclusive per-scan locks for all unique requested scene IDs in sorted order before
input validation and scan-local `frames/` extraction. After all extraction completes, atomically
downgrade those same descriptors to shared mode and hold them through every model subprocess.
Acquire each GPU lease only after the scan locks; release the lease before appending the successful
task marker under the global prediction-index lock in exclusive mode. This lock also serializes
eviction/reset task-log rewrites, preventing append/rewrite and rewrite/rewrite lost updates.

The guarded reset acquires scene-engine locks in sorted path order, then per-scan locks in sorted ID
order, then the global prediction-index lock exclusively. No code may hold the prediction-index
lock while waiting for a GPU. Lock waits are intentionally blocking and interruptible; do not add silent lock timeouts.
Taking scan locks before GPU leases prevents a reconstructor from holding a GPU while waiting for a
predictor that still has the same scan open.

### 6.4 GPU lease boundary

Refactor `run.py` so `_pipeline()` returns after finalization and segmentation. For a managed
engine on scene9004:

```text
source-scene frame lock
  prepare/validate canonical frames
source-scene frame lock released
scene-engine lock
  scan exclusive lock
    prediction-index lock for exact stale-artifact cleanup
    prediction-index lock released
    GPU lease
      reconstruct + align + ScanNet artifacts + segmentator
    GPU lease released
    geometry score + YAML + PNG
```

Open3D remains CPU-only. Scene9004 scoring never runs inside a managed GPU lease; non-benchmark
scenes stop after reconstruction/finalization/segmentation.

## 7. Exact Eviction and Prediction Cleanup

Before deleting any stale scene9004 slot, worst scene9004 run, or default non-benchmark run 0:

1. Hold the scene-engine and exclusive per-scan locks.
2. Acquire the global prediction-index lock exclusively.
3. Enumerate both prediction and evaluation roots through `benchmark.BENCHMARKS`/`artifact_paths`,
   not hard-coded ScanNet20-only paths.
4. For every `<benchmark>/<prediction-run>/<model>` directory, look only for exact
   `<scene_id>.txt`.
5. Parse that index and delete only referenced mask paths that resolve inside the model directory.
6. Also delete exact stale masks matching `predicted_masks/<scene_id>_*.txt`.
7. Delete the exact scene index.
8. In every evaluation run/model task file, remove every line exactly equal to the scene ID while
   preserving all other lines/order; publish rewrites atomically.
9. Delete an evaluation run's corresponding `<model>.csv` whenever its task file or the matching
   prediction run/model was changed, because the aggregate is stale.
10. Release the prediction-index lock. Leave shared parent directories in place even when empty.
11. Remove the scan directory directly while retaining the exclusive scan lock. If interruption
    leaves a partial directory, the next allocation classifies it as incomplete, repeats exact
    artifact cleanup, and removes it before reuse.

Never delete ground truth, raw SVOs, shared frames, other scene IDs, entire prediction runs that
still contain another scene, shared prediction/evaluation parent directories, or outputs outside
`/data/scannet`.

## 8. Implementation Phases

### Phase 0: Safety preflight

1. Capture `git status --short --branch`, `git diff -- spellbook/PROJECT_STATUS.md`, and recent log.
2. Confirm the known `PROJECT_STATUS.md` edit, this plan/audit material, and the session-backup
   directory are the only expected unrelated or planning-only changes.
3. Run the existing test suite before changing code:

   ```bash
   /home/rolf/anaconda3/envs/3disspellbook/bin/python -m unittest discover -s spellbook/tests -v
   ```

4. Re-list all custom scan directories using the strict regex `^scene9[0-9]{3}_[0-9]{2}$`.
5. Re-list exact custom prediction indexes, task-log entries, and evaluator CSVs.
6. Verify `scene9004_04/recon/frames` is still a real directory and all 3,612 color, depth, and
   pose files exist with 3,612 `OK` states.
7. Verify `extract.frames_complete()` and load every metadata array/text file.
8. Verify the v2 CAD SHA-256 and cached mesh counts listed above.
9. Verify at least 15 GB free on `/data`.
10. Verify no active reconstruction, prediction, or evaluation process is using a custom scan.
11. Stop before edits if the frame set, CAD hash, or data inventory differs unexpectedly.

### Phase 1: Implement and test the geometry core

1. Add `geometry_reference.yaml` with the scene9004 artifact paths and fixed reference/metric
   constants. Leave generated artifact hashes/transform fields explicitly pending until Phases 2-3.
   Record build provenance hashes but no runtime dependency on the old scan or a copied pose file.
2. Add `geometry_score.py` with small functions for config loading, hash validation,
   axisAlignment parsing, surface voxelization, gauge normalization, nearest-cell matching,
   atomic output, and PNG rendering.
3. Expose a standalone command:

   ```bash
   python -m spellbook.reconstruct.geometry_score \
     --scan-id scene9004_04 \
     [--distance-mm 50] \
     [--reference-config spellbook/reconstruct/geometry_reference.yaml]
   ```

4. Keep `--distance-mm` on this standalone scorer only. Automated run scoring always uses the
   active configured default so eviction scores remain comparable.
5. Create temporary synthetic tests for identical surfaces, zero-match surfaces, 40/60 mm offsets,
   half-missing surfaces, extra surfaces, four yaw cases, empty meshes, stale reference metadata,
   atomic output, and repeat-byte determinism.
6. Require identical synthetic inputs to score exactly 100 within floating-point rounding.
7. Delete the temporary synthetic test after it passes. Do not add or modify a permanent benchmark
   test file.

### Phase 2: Export and validate the CAD reference

1. Create `/data/scannet/custom/reference` only after checking its parent.
2. Write `spellbook/tmp/export_scene9004_cad.py` and run it with:

   ```bash
   /home/rolf/anaconda3/envs/3discomp/bin/python \
     spellbook/tmp/export_scene9004_cad.py
   ```

3. Before reading cached meshes on `Fensterglas / Griff`, exclude an object exactly when
   `min(bbox_extent) <= 0.005 m` and `max(bbox_extent) >= 0.5 m`. Assert 18 panes excluded,
   50 surface hardware objects retained, and 8 curves omitted so a changed CAD fails loudly.
4. Read every retained Brep face cached `MeshType.Any` mesh and every retained Extrusion cached
   mesh.
5. Merge objects in world coordinates and retain per-layer diagnostic colors.
6. Convert triangles directly; split each quad deterministically along `(A,C)`; reject invalid
   indices and zero-area faces.
7. Omit curves and the one unmeshed SubD, recording those omissions in the YAML comments/metadata.
8. Derive the scene-specific floor from the four dominant inner planes of vertical `Wand StB`
   faces: estimate the room centroid from the wall-layer bounds, select each wall solid's face
   nearest that centroid, intersect the resulting inward half-spaces in x/y, and validate every
   corner against `Bodenleiste` geometry below z=0.15 m. If four consistent planes are not found,
   stop for explicit scene-specific coordinates rather than falling back to reconstruction bounds.
   Add a consistently oriented triangulated floor at z=0.
9. Normalize CAD floor and x/y room centre without changing scale or axis orientation.
10. Write binary PLY to a temporary data path, reload it with Open3D, validate finite vertices,
   positive triangle area, metre-scale bounds, expected layers, and furniture presence, then
   atomically rename it to `scene9004_cad.ply`.
11. Hash the output and update `geometry_reference.yaml`.

### Phase 3: Build the fixed visibility reference

1. Load old scene9004 final poses and validate finite values, rigid rotations, frame count, and a
   plausible trajectory relative to old `scene9004_04_vh_clean_2.ply`. Do not load or compare
   captured depth.
2. Write and run `spellbook/tmp/build_scene9004_reference.py`.
3. Apply the old scan's `axisAlignment`, robust centre/floor transform, and selected Manhattan yaw
   to both `_vh_clean_2.ply` and its capture poses.
4. Generate a temporary plan/elevation overlay of normalized CAD, old `_vh_clean_2`, room bounds,
   and camera path. Require visibly correct wall, door, window, and furniture orientation; stop
   rather than introducing ICP, scale fitting, or another registration method.
5. Select source views with the exact endpoint-inclusive `linspace` rule, scale intrinsics from the
   actual source dimensions to 480 x 300, and cast pinhole rays into the CAD. Do not read depth.
6. Map finite first-hit points to the fixed 10 mm grid and keep cells hit from at least two poses.
7. Save sorted unique visible indices as compressed `int32` NPZ.
8. Validate that every major room side and representative furniture cluster contributes visible
   cells and that excluded glass does not appear.
9. Hash the visible-cell file, finalize the YAML hashes/transforms/bounds, and rerun the standalone
   scorer twice on old `scene9004_04`.
10. Require identical YAML score values and identical comparison PNG bytes across reruns. Do not
    impose a minimum quality score on the old reconstruction.
11. Delete the temporary reference-build script and reference-alignment overlay after validation;
    retain only `scene9004_cad.ply` and `scene9004_visible_voxels.npz` in the data reference
    directory. Do not copy the old capture poses into that directory.

### Phase 4: Promote irreplaceable scene9004 frames

1. Create `/data/scannet/derived/reconstruction/frames/scene9004` after confirming it is absent.
2. Atomically move the real directory from `scene9004_04/recon/frames` to
   `derived/reconstruction/frames/scene9004/frames` on the same filesystem.
3. Put a temporary symlink at the old `recon/frames` path so the old scan remains operable until
   cleanup.
4. Re-run strict color/depth/pose counts, pose-state validation, metadata hashes, and
   `extract.frames_complete()` through both paths.
5. Verify the symlink resolves exactly to the canonical shared frame pool.
6. Rollback before old scan deletion is: remove the symlink and move the canonical directory back.

### Phase 5: Guarded reset of old custom runs

This is the destructive phase. It starts only after Phases 1-4 pass and deliberately precedes
activation of the new run-ID semantics: legacy `scene9004_04` must never be mistaken for new ZED
run 4.

1. Re-inventory custom scans/predictions at execution time; do not rely on the three planning-time
   paths if concurrent work added another custom run.
2. Confirm canonical scene9004 and scene9009 shared frame pools are complete and outside every scan
   directory. For each pool, run strict color/depth/pose counts, pose-state `OK` counts,
   `extract.frames_complete()`, and metadata hashes. For both old scene9009 scans, verify every
   `recon/frames` file is represented identically in the canonical scene9009 pool; stop if either
   old scan contains unique or divergent frame data.
3. Confirm CAD/reference artifacts and the old-run score smoke are complete, and confirm no active
   reconstruction, prediction, or evaluation process is using a custom scan.
4. Add the path constructors and basic `fcntl.flock` context manager in `utils/scan_lock.py` without
   changing run-ID semantics or wiring production call sites yet. Verify path generation, sorted
   acquisition, blocking, and process-exit release under temporary roots; delete the temporary test.
5. Write `reset_custom_runs.py` with dry-run default, explicit `--apply`, a strict
   `/data/scannet` root assertion, and strict custom-ID regex. Import all lock paths/acquisition from
   `utils/scan_lock.py`; acquire relevant scene-engine locks in sorted path order, then per-scan
   locks in sorted ID order, then the global prediction-index lock exclusively.
6. Implement the exact Section 7 cleanup in the script. Dry-run and print every scan, prediction
   index, mask, task line, and CSV that would be removed; never select shared parent directories.
7. Review the complete dry-run inventory before `--apply`.
8. Under those locks, purge exact prediction/evaluation artifacts and directly delete every
   existing custom scan directory. Do not migrate any old `_04` run into the new naming scheme.
9. Recheck that no `scene9???_??` scan remains, no deleted scene ID remains in predictions/tasks,
   and both shared frame pools/raw SVO symlinks still exist.
10. Delete the reset script after verification.

After old scan deletion there is intentionally no reconstruction backup. Raw SVOs and shared
frames are the rebuild source. This point is not reversible while ZED extraction is blocked, so
all reference and frame checks above are mandatory.

### Phase 6: Implement IDs, locks, allocation, scoring, and eviction

1. Complete `utils/scan_lock.py` with the four lock scopes in Section 6.3 and explicit
   exclusive-to-shared downgrade support. Use temporary lock roots to test shared/shared
   compatibility, exclusive blocking, atomic downgrade, sorted multi-lock acquisition, and kernel
   release after process exit; delete these lock tests after verification.
2. Update the ID helpers and all call sites; search the entire repository for duplicated
   `scene{scene...}_{ENGINE_INDEX...}` formatting. Verify all six ten-ID engine ranges are disjoint.
3. Make `batch._prepare_frames()` and direct `run.py` frame setup use the canonical source-scene
   frame lock. Keep `--replace` limited to regenerating that shared frame set.
4. Add scene9004 slot scanning, score refresh, and eviction helpers to `run.py`. Purge and remove an
   incomplete/failed reused slot before rebuilding it.
5. Add exact prediction purge under the global prediction-index lock in exclusive mode; leave shared parent
   directories in place.
6. Refactor managed execution so scoring happens after the GPU lease closes but while the
   scene-engine/per-scan locks remain held.
7. For scene9004, make score-generation failure return a non-zero run result and leave the slot
   reusable on the next attempt.
8. For every scene except scene9004, select engine run 0, purge its exact old predictions, delete
   the prior scan under lock, rebuild it in place, and skip score generation.
9. Update batch labels/log names to start as `sceneNNNN_<engine>` and replace them with the exact
   child-selected ID after parsing the `[run] <scene_id>` line.
10. For scored runs, print one concise completion line
    (`[geometry] <scene_id> score=<value>`). When applicable, print one eviction/replacement line
    naming the exact removed scan ID.
11. Do not add `--runs`, `--run`, a manual GPU option, or a replacement-score gate.
12. In `predict()`, deduplicate requested scene IDs, acquire their exclusive locks in sorted order before scan
    validation and `predict/frames.py` extraction; after extraction, downgrade every descriptor to
    shared mode and retain it through all model subprocesses. Revalidate each point cloud after
    acquiring the locks, release each GPU lease before appending its success marker, and serialize
    that append with exact cleanup through the same exclusive global prediction-index lock.
13. Add temporary filesystem tests for scene9004 first-stale/first-free/all-ten-worst selection,
    exact evicted-ID reuse, score tie-break, stale-score rescore, non-benchmark run-0 replacement,
    exact prediction cleanup, task-log rewrite/append serialization, CSV invalidation,
    shared-frame preservation, concurrent canonical-frame setup, concurrent prediction extraction,
    concurrent cleanup, and concurrent allocation. Delete all temporary tests after verification.

### Phase 7: Remove legacy QC and update documentation

1. Delete `qc.py` and `scannet_reference.yaml` only after the geometry scorer has passed synthetic
   and old-scene smoke tests.
2. Search for `qc`, `qc.yaml`, `scannet_reference`, `no-qc`, and `run_qc`; remove only live
   references, not historical archived plans or repo-root session logs.
3. Update `PROJECT_STATUS.md` tree, data layout, engine table notes, key decisions, commands,
   current next steps, and current test count.
4. Preserve the user's uncommitted engine-worktree sections line-for-line unless a nearby
   reconstruction statement must be merged carefully.
5. Update `run.py`, `batch.py`, and `reconstruct/__init__.py` module documentation.

### Phase 8: End-to-end validation

1. Run the full repository unit suite:

   ```bash
   /home/rolf/anaconda3/envs/3disspellbook/bin/python -m unittest discover -s spellbook/tests -v
   ```

2. Run syntax/CLI checks:

   ```bash
   /home/rolf/anaconda3/envs/3disspellbook/bin/python -m spellbook.reconstruct.geometry_score --help
   /home/rolf/anaconda3/envs/3disspellbook/bin/python -m spellbook.reconstruct.run --help
   /home/rolf/anaconda3/envs/3disspellbook/bin/python spellbook/main.py --help
   ```

3. Verify ZED still blocks before creating output or opening GPU 0:

   ```bash
   /home/rolf/anaconda3/envs/3disspellbook/bin/python -m spellbook.reconstruct.run \
     --scene 9999 --engine zed
   ```

4. Run the existing GPU-distribution smoke for Open3D to confirm scheduling code remains intact:

   ```bash
   /home/rolf/anaconda3/envs/3disspellbook/bin/python spellbook/main.py \
     --gpu-check --engine open3d
   ```

5. Run one fresh scene9004 Open3D reconstruction using the canonical frames. It must allocate
    `scene9004_40`, reuse the frame pool through a symlink, generate all ScanNet artifacts, release
    any managed lease before scoring (Open3D itself remains CPU-only), and write the two geometry
    outputs.
6. Rerun standalone scoring twice on the fresh `scene9004_40`; require the same `geometry_score`,
   reference metadata, and image.
7. Run prediction-cleanup tests only on temporary directory trees, not production predictions.
8. Exercise eleven temporary fake scene9004 runs to prove the eleventh evicts exactly the lowest
   score and preserves all other scan/shared-frame/prediction artifacts.
9. Verify no process initialized, locked, or selected physical GPU 0.
10. Inspect `git diff`, `git status`, and all changed files. Ensure only intended repository files
    plus the pre-existing user changes remain.
11. Do not commit, push, or open a GitHub issue unless separately requested.

## 9. Stop Conditions

Stop without destructive cleanup if any of these occur:

- The v2 CAD hash or cached mesh inventory differs unexpectedly.
- The complete scene9004 frame pool cannot be promoted and verified.
- Capture poses are non-rigid, non-finite, mismatched in count, or visibly implausible relative to
  the old `_vh_clean_2` mesh.
- No plausible no-scale rigid CAD/capture alignment can be validated visually and numerically.
- Visible reference cells omit a major room side or most furniture despite source coverage.
- Synthetic metric outputs are nondeterministic, non-finite, or outside `[0, 100]`.
- A score changes when only mesh tessellation changes at fixed geometry.
- A prediction cleanup dry run includes a different scene ID or an entire non-empty run directory.
- A lock-order test can deadlock, a concurrent allocator selects the same slot, or concurrent task
  marker append/rewrite loses an entry.
- The baseline test suite regresses outside intentional QC/ID expectations.
- Any operation would modify the user's engine worktrees, archive, session backups, raw SVOs, or
  official ScanNet scenes.

## 10. Acceptance Criteria

- `scene9004_cad.ply` contains the validated v2 architecture, full furniture, and synthetic floor,
  with transparent panes excluded and window hardware retained.
- The visible CAD cell set is immutable, hashed, capture-supported in at least two views, and
  independent of reconstruction engine/run.
- Identical geometry scores 100; deterministic shifted/missing/extra cases behave as specified.
- New Open3D scene9004 run IDs start at `_40`; all six engine ranges map correctly.
- Direct and batch runs use the same automatic allocator and never collide under concurrency.
- Concurrent direct/batch frame setup cannot replace the same canonical frame pool simultaneously.
- Concurrent prediction completion and eviction preserve every unrelated task marker.
- A low numeric score never aborts a run; only failure to compute a valid score does.
- All completed scene9004 benchmark runs have exactly one comparable `geometry_score` and one PNG.
- The eleventh comparable run evicts only the lowest score and cleans only its exact prediction
  artifacts; the following allocation reuses that exact absent run ID.
- Every non-benchmark scene remains reconstructable in engine run 0 and replaces that run and its
  exact predictions by default; it is never scored or retained in the scene9004 ten-slot pool.
- Scoring occurs outside managed GPU leases; GPU 0 remains untouched.
- All old custom scans are deleted only after scene9004 frames/reference are safe; raw SVOs and
  shared frames remain intact.
- Legacy QC is absent from live code/docs; `benchmark.py` and `evaluate.py` remain operable.
- Full tests and end-to-end scene9004 Open3D smoke pass.

## 11. Improvement After the Minimal Build

After several engines have produced real scores, run a one-time convergence check at 5, 10, and
20 mm surface-grid resolution while keeping the 50 mm tolerance fixed. Keep 10 mm unless ranking
or score changes are material; this validates the discretization without adding another permanent
metric.
