---
name: s2-improve
description: Use when improving a ScanNet reconstruction or prediction method.
---

## Method improvement

- First, give a rough time estimate in the chat.
- Improve one method at a time in six-run batches: three parameter-tuning runs and three runs testing independent high-risk, high-upside improvements.
- Before each batch, exhaustively analyze the current best run's shortcomings and read all previous attempts in the `<method>-optimization` issue. Use multiple parallel @explore subagents.
- Based on the analysis and a deep web search, formulate improvement hypotheses. Use multiple parallel @scout subagents.
- Big changes are allowed, but always stay inside the worktree and roll them back if unsuccessful.
- Treat the baseline and all prior winners as immutable. Add the current batch's best candidate only if it improves the global best; otherwise add none.
- **Reconstruction:** use existing scan slots and `geometry_score.yaml`. Retain the new winner's scan and `<scan>/recon/run.json`, if any; delete all other current-batch candidate scans. Deleted candidate scan slots may be reassigned.
- **Prediction:** use the existing prediction/evaluation hierarchy and official evaluator. Retain full prediction artifacts for the new winner, if any; delete them for all other current-batch candidates. Permanently retain each candidate's run ID, `run.json`, evaluator CSVs, and per-scene TP/GT sidecars under `/data/scannet/derived/evaluations/<Benchmark>/<run-id>/`.
- Store each new winner's settings in the existing engine adapter or prediction wrapper without overwriting baseline or prior-winner settings.
- Retry infrastructure failures once; deterministic method failures count toward the six runs.
- Record all six runs, settings, results, failures, and the promotion decision concisely in the `<method>-optimization` issue.
- Once execution starts, make safe assumptions without questions or interruptions and report only after completion.
