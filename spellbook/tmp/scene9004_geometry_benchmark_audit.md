# Scene9004 Geometry Benchmark Plan Audit

Date: 2026-08-23

Scope: read-only cross-check of `scene9004_geometry_benchmark_plan.md` against the final decision
ledger and current repository/data behavior. No production code or `/data/scannet` artifact was
changed by this audit.

## Sources

- Metric/reference audit: explore session `ses_fd41df2eeffeM0kAK4ra7zattD`.
- Allocation/locking audit: explore session `ses_fd41df2b7ffebai19d0nVyMcjf`.
- Cleanup/scope audit: explore session `ses_fd41df299ffeFFiLvMt7IxME6z`.
- Closed GitHub issues: no previous CAD/voxel benchmark; #9 and #26 confirm extraction
  serialization and canonical frame-directory behavior.

## Corrections Applied to the Plan

- Restricted the goal and output contract explicitly to scene9004 benchmark runs.
- Defined zero-match F1 as zero and required a finite score in `[0, 100]`.
- Clarified robust-centre bounds, yaw rotation centre, and four-yaw selection against full CAD.
- Recorded the 180-pose cap, selection rule, ray resolution, and two-view threshold in config.
- Moved the legacy custom-run reset before activation of the new ID scheme, preventing old
  `scene9004_04` from being interpreted as new ZED run 4.
- Defined canonical-frame, scene-engine, per-scan, and prediction-index lock paths and lifetimes.
- Serialized canonical frame replacement and prediction task-log append/rewrite operations.
- Removed shared-directory pruning and tombstones; cleanup now deletes only exact scene artifacts
  and directly removes the locked scan directory.
- Required partial scene9004 slots to be purged before same-ID reuse.
- Made temporary lock/concurrency tests explicit and required their deletion after verification.
- Excluded archived plans and repo-root session logs from live QC-reference cleanup.
- Required the selected full-pool slot to be the eviction victim and to remain exclusively locked
  through deletion/rebuild.
- Made the global prediction-index lock explicitly exclusive for task-marker append and rewrite.
- Added strict scene9009 old-scan/canonical-frame equivalence checks before deleting old scans.
- Made the reset import canonical lock paths from `scan_lock.py` rather than duplicate them.

## Final Review

- Decision audit: explore session `ses_fd4150ec2ffeC28HudjMLP4xQ7`.
- Feasibility/safety audit: explore session `ses_fd4150e96ffeLKFd01ywh6lqH7`.
- Both reviews found the authoritative benchmark decisions intact after the corrections above.
- Whitespace checks passed for tracked changes, the untracked plan, and this report.

## Deliberately Not Added

- No scale fitting, ICP, depth support, filled occupancy, extra score, pass/fail threshold, or score
  gate.
- No permanent benchmark tests, pose copy, alignment image, benchmark package, or shared-directory
  cleanup.
- No lock timeout or compatibility migration for legacy scan IDs; the guarded reset removes the
  concrete legacy state before the new scheme becomes live.
