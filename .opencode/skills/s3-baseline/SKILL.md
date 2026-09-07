---
name: s3-baseline
description: Use to reproduce official reconstruction and prediction quality locally.
---

## Method baseline

- For reconstruction, the goal is to achieve ScanNet-like quality; for prediction, the goal is to achieve officially reported prediction AP on ScanNet.
- Before execution, exhaustively research the current repository, relevant chats, open and closed issues, previous attempts, reports, artifacts, the method's repository, and official online solutions. launch multiple parallel @explore and @scout subagents.
- Fix only integration defects required to run the official-default pipeline, and integrate them into the existing data structure and repository.
- Debug and rerun up to 5 times or until the baseline target is met; record every attempt shortly in the `<method>-baseline` issue, and close it after baseline is achieved.
- Retain only one baseline run per method and store its settings in the existing adapter or wrapper; retain the existing naming: reconstruction baselines use the engine-specific scan ID with run number 0, while prediction baselines use baseline-<method>.
- **Reconstruction:** run this worktree's reconstruction method for the custom scenes 9009 and 9004.
- Require complete ScanNet-native artifacts, valid poses, logs, segmentation, coherent room-scale geometry, and no catastrophic tracking, alignment, or coverage failure.
- Aim for ScanNet-like geometry quality; use screenshots and scene9004's `geometry_score.yaml` for orientation.
- **Prediction:** run this worktree's prediction method on the locally available ScanNet scenes.
- Use the managed prediction pipeline, official classes and evaluator, and the method's official defaults and checkpoint.
- Aim for officially reported prediction quality locally. Report the published source and metric beside local AP, AP50, AP25, scene set, and difference.
