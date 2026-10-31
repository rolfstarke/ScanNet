# Recovery status (2026-10-30, paused for re-login to pick up docker group)

Plan: spellbook/tmp/recovery-plan.md. Branch: `recovery` (uncommitted). Reports: spellbook/tmp/recovery/.

## Done
- Data: 16 val scenes + labels + ScanNet200 GT; SVO/3dm in custom/; CAD PLY sha 76613112 (identical).
- Frame pools 9004 = 3612, 9009 = 2356 (extracted from SVO).
- Visible-voxel mask rebuilt from all pool poses (aligned via the open3d B1 mesh): sha d4d7fbf0, 231360 cells
  (orig 156689). Geometry scores are not comparable with the logged ones; compare rankings only.
- Envs ready: 3disspellbook (+ ZED 5.4.1, pymeshlab), metashape 2.3.1, mosaic3d, sam3dis, open3dis, openyolo3d,
  openmask3d, openins3d. detectron2 v0.6 shared clone /data/detectron2 (built into open3dis + openins3d).
- ZED neural depth models pre-optimized (NEURAL + NEURAL_PLUS) in ~/opt/zed-sdk/install/resources.
- Wrappers restored: recon shared infra, zed, metashape, open3d, mosaic3d, openyolo3d, openmask3d, open3dis
  (smoke pending), sam3dis (/data/sam3dis repo, 10 commits).
- Recon results (cap 361.2 s for 9004):
  - open3d 9004_40: 192.5 s, 15.75 cm (new mask)
  - open3d 9009_40: 165.5 s
  - zed 9009_00 (native-default): 235.4 s
  - zed 9004_00 (native-default): 274.7 s, not scored yet
  - zed 9004_01 (s2b8): 177.1 s, not scored yet

## Interrupted (resume)
- Prediction baselines (cancelled): openmask3d, openyolo3d, sam3dis. Resubmit with run ids repro-baseline-<m>.
- Metashape baseline + b5-kf350 never started (export kwarg fixed: save_colors).
- Open3DIS smoke (detectron2 now installed) and replay2d fix: cancelled mid-way.
- OpenIns3D wrapper: precomputed masks for all 16 scenes found and vertex-aligned; worktree at 56ccc22
  (/data/openins3d/repo-56ccc22) and wrapper wiring were in progress.

## Next
1. After re-login, check `id -nG` includes docker; build bundlefusion image(s) with docker.
2. Score ZED 9004_00 and 9004_01 offline: `python -m spellbook.reconstruct.geometry_score --scan-dir <scan>`.
3. Run mosaic3d (needs the openins3d proposals) and openins3d once the wrapper is done; run all best runs.
4. rtabmap/isaac: need subuid/subgid (podman) or switch the adapters to docker; isaac also needs NGC.
5. Post results to the issues; update PROJECT_STATUS.md; s1-document; commit.
