---
name: s2-improve
description: Use when improving a ScanNet reconstruction or prediction method.
---

## Method improvement

- First, give a rough time estimate in the chat.
- Improve one method at a time in batches of six runs.
- Before each batch, deeply analyze the current best run's shortcomings and read all previous attempts in the `<method>-optimization` issue.
- Based on the analysis and a deep web search formulate improvement hypotheses.
- Of the six runs, three should be parameter-tuning runs and three should test independent high-risk, high-upside improvements.
- Big changes are allowed, but always stay inside the worktree.
- **Reconstruction:** use existing scan slots and `geometry_score.yaml`; retain only the best scan.
- **Prediction:** use the existing prediction/evaluation hierarchy and official evaluator. Retain only the best run's prediction artifacts; retain `run.json`, evaluator CSVs, and per-scene TP/GT sidecars for every candidate.
- Store winning settings in the existing engine adapter or prediction wrapper. Store reconstruction metadata in `<scan>/recon/run.json` and prediction metadata in `/data/scannet/derived/evaluations/<Benchmark>/<run-id>/run.json`.
- deleted reconstruction scan slots can be reassigned. prediction run IDs are permanent.
- If no candidate improves the primary metric, retain the incumbent.
- Retry infrastructure failures once; deterministic method failures count toward the six runs.
- Record all six runs, settings, results, failures, and the winner concisely in the `<method>-optimization` issue.
- Once execution starts, make safe assumptions without questions or interruptions and report only after completion.
