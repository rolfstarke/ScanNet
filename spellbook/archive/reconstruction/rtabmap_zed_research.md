# Scout report: RTAB-Map official guidance for ZED X stereo, low-texture/glass indoor SLAM

Sources: introlab/rtabmap wiki (cloned), introlab/rtabmap_ros master (cloned), introlab/rtabmap master (cloned), ROS wiki via Wayback, GitHub issues. All URLs below.

## P1 — Odometry/loop-closure parameters (low-texture, stereo)

Defaults (authoritative: `corelib/include/rtabmap/core/Parameters.h` in introlab/rtabmap):
- `Vis/MinInliers=20` (loop-closure acceptance), `Vis/InlierDistance=0.1 m`
- `Odom/Strategy=0` = Frame-to-Map (F2M) — this IS the default and the recommended mode; F2M keeps a local map of up to `OdomF2M/MaxSize=2000` features (maintainer: "local history" approach, helps re-acquire tracking in feature-poor areas; see Change-parameters wiki #Odometry section). F2F (`Odom/Strategy=1`) is only for low-latency cases.
- `Vis/FeatureType=8` = GFTT/ORB default; maintainer's documented preference: GFTT+BRIEF (`=6`) — "GFTT+BRIEF gives the highest reliability across all available binary feature detectors" (Change-parameters wiki).
- `Vis/CorType=0` (feature matching), `Vis/CorNNType` should be BruteForce (`=3`) for binary descriptors, `Vis/CorNNDR=0.8`.
- `Vis/MaxFeatures=1000`, `Vis/MinDepth`/`Vis/MaxDepth=0` (no limit; set `Vis/MaxDepth` ~4 m for stereo — see P3).

Official documented values for stereo/low-texture environments (from maintainer-authored configs/tutorials):
- wiki.ros.org/rtabmap_ros/Tutorials/SetupOnYourRobot "Stereo A/B" sections (maintainer): `Vis/MinInliers=12`, `Odom/MinInliers=12`, `Odom/RoiRatios=0.03 0.03 0.04 0.04`; Kinect config with `Vis/MinInliers=12`.
- Official stereo indoor tutorial config `stereo.ini` (Stereo-mapping wiki, maintainer): `Odom/MinInliers=10`, `Odom/InlierDistance=0.1`, `Odom/FeatureType=6`, `LccBow/MinInliers=10`, `LccBow/InlierDistance=0.1`, `LccReextract/Activated=true`, `LccReextract/FeatureType=4`, `LccReextract/MaxWords=600`, `LccReextract/NNDR=0.9`, `Odom/Strategy=0` (F2M), `OdomF2M/LocalHistorySize=1000` (legacy name).
- Change-parameters wiki: "increasing InlierDistance to 2 cm (default 1 cm) doesn't decrease a lot the quality" — for stereo use ~0.1 m (matches inlier distance in stereo.ini).
- Your symptom "Not enough inliers 0/20": matches `Vis/MinInliers=20` default for loop-closure refinement; PnP-RANSAC is non-deterministic at low inlier counts (maintainer, issue 1530). LCC "Refine" always recomputes correspondences (issue 1530, matlabbe comment 2025-06-29).

URLs:
- https://github.com/introlab/rtabmap/wiki/Change-parameters (odometry + loop-closure sections)
- https://github.com/introlab/rtabmap/wiki/Stereo-mapping (stereo.ini at github.com/introlab/rtabmap.wiki/blob/master/doc/Tutorials/Stereo_indoor/stereo.ini)
- https://web.archive.org/web/20231122153210/http://wiki.ros.org/rtabmap_ros/Tutorials/SetupOnYourRobot
- https://github.com/introlab/rtabmap/issues/1530 (maintainer comments)

## P2 — External odometry (ZED SDK pose) as prior + RTAB-Map loop closure: DOCUMENTED

Three documented mechanisms, all supported:
1. **`odom` topic**: feed `nav_msgs/Odometry` (e.g. `/zed/zed_node/odom`) to rtabmap_slam via `<remap from="odom" ...>`; rtabmap then does its own loop-closure detection + graph optimization on top. Documented in SetupOnYourRobot (all robot configs remap odom to wheel/viso2 odometry; Stereo A/B examples).
2. **`odom_frame_id` (TF)**: rtabmap.launch.py arg `odom_frame_id` — "If set, TF is used to get odometry instead of the topic." (rtabmap_ros/rtabmap_launch/launch/rtabmap.launch.py:447). ROS1 name of same param.
3. **Official ZED example**: rtabmap_examples `zed.launch.py` / `zed_composition.launch.py` — argument `use_zed_odometry:=true` remaps `odom` to `/zed/zed_node/odom` (ZED SDK positional tracking), else rtabmap computes VO and `subscribe_odom_info` is set. Supports `zedx` model. This is the exact workflow you want: ZED pose as odometry prior, RTAB-Map does loop closure + graph optimization (RGBD-SLAM mode).

Note: with external odometry, RTAB-Map does NOT run its own VO; loop closures are still detected/refined by RTAB-Map (visual + optional ICP), and pose graph is optimized by RTAB-Map. That's the standard SLAM pipeline, fully documented.

URLs:
- https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_examples/launch/zed.launch.py
- https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_examples/launch/zed_composition.launch.py
- https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_launch/README.md (ROS1+ROS2 ZED snippets)
- https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_launch/launch/rtabmap.launch.py (odom_frame_id, line ~447)

## P3 — Glass/reflective surfaces, stereo depth quality

No dedicated "glass mask" feature in RTAB-Map; documented mitigations:
- Stereo data: "I would not try to generate point cloud with range farther than 4 meters" (matlabbe, issue 1605 — ZED2i context). Set `Grid/RangeMax` / cloud export `--max_range 4`; `Vis/MaxDepth`, `LccBow/MaxDepth` limit features to reliable depths (stereo.ini uses MaxDepth=10; maintainer suggests 4 m for accuracy).
- `Mem/DepthAsMask=true` / `Vis/DepthAsMask=true` (defaults): features only extracted where depth is valid — automatically skips glass windows with invalid ZED depth; features on reflective/transparent surfaces have NaN depth and are excluded from 3D pose estimation.
- ICP refinement of loop closures (`Reg/Strategy=1` or `RGBD/NeighborLinkRefining=true`) is documented to help geometry-poor environments, but maintainer warns 3D ICP can add undetectable errors (ICP wiki page); ICP is deactivated by default for RGB-D for this reason. For stereo, `LccIcp3/MaxDepth=4` etc. exist.
- ZED-side confidence is handled in zed_wrapper (its own `depth_confidence` / filtering params), not in RTAB-Map.
- If tracking is lost, `Odom/ResetCountdown=10`+ auto-resets odometry after consecutive lost frames (Parameters.h); F2M local map (OdomF2M/MaxSize) helps re-acquire tracking in feature-poor regions.

URLs:
- https://github.com/introlab/rtabmap/issues/1605 (matlabbe guidance, Nov 2025)
- https://github.com/introlab/rtabmap/wiki/ICP
- Parameters.h: https://github.com/introlab/rtabmap/blob/master/corelib/include/rtabmap/core/Parameters.h

## P4 — Offline reprocessing: YES, this is the recommended workflow

RTAB-Map is explicitly designed for offline reprocessing of a recorded database; this is the documented preferred path for your finite SVO recording:
- **`rtabmap-reprocess`** (tools/Reprocess, exists in repo): recomputes odometry from saved images and/or detects more loop closures on a .db. Maintainer's own recipe for ZED2i (issue 1605, Nov 2025): `rtabmap-reprocess -default --Rtabmap/DetectionRate 0 -odom -cam 1 rtabmap.db output.db` then `rtabmap-export --cloud --voxel 0 --max_range 10 --decimation 1 output.db`. Can rerun with different params (e.g. lower MinInliers, different FeatureType) and keep only the best output.
- **Detect More Loop Closures**: GUI Tools->"Post-processing..." dialog (documented in Export-Raster-Layers-to-MeshLab wiki: "Detecting more loop closures give generally good results. If the environment has a lot of geometry, you can refine the links with ICP", + optional Sparse Bundle Adjustment). Service `detect_more_loop_closures` exists in rtabmap_ros (rtabmap_msgs/srv/DetectMoreLoopClosures.srv, implemented in rtabmap_slam CoreWrapper) with cluster radius/angle/iterations args; rtabmap-databaseViewer has the same with "use current optimized graph as guess" option (issue 1530 screenshot). IROS-2014 wiki page documents reprocessing a finished database to recover missed loop closures.
- **RTAB-Map desktop / rtabmap-console**: process a .db as source with "Ignore odometry" and different parameters (Stereo-mapping wiki "Process a database of stereo images"; Benchmark wiki uses `rtabmap-console`).
- For ROS: replay the SVO as a rosbag (zed_wrapper svo_replay) and run rtabmap.launch.py with `use_sim_time:=true` — works, but the .db reprocessing above is the documented higher-quality alternative since it lets you iterate parameters and add loop closures post-hoc.

URLs:
- https://github.com/introlab/rtabmap/issues/1605 (maintainer's rtabmap-reprocess recipe)
- https://github.com/introlab/rtabmap/wiki/Export-Raster-Layers-to-MeshLab (post-processing dialog)
- https://github.com/introlab/rtabmap/wiki/Stereo-mapping (reprocess a .db)
- https://github.com/introlab/rtabmap/wiki/IROS-2014-Kinect-Challenge (offline LC recovery)
- https://github.com/introlab/rtabmap/wiki/Benchmark (rtabmap-console)

## P5 — Stereo/ZED official guidance

- RTAB-Map side: rtabmap_launch README (ROS1/ROS2 ZED snippets, `approx_sync:=false`, `wait_imu_to_init`, imu remap) and rtabmap_examples zed.launch.py/zed_composition.launch.py (zed2i/zedx; use_zed_odometry; VGA grab resolution override; pos_tracking disabled when RTAB-Map owns odometry).
- Stereolabs side: no RTAB-Map examples in zed-ros2-wrapper/zed-ros2-examples repos (checked trees). Stereolabs docs use their own VSLAM; integration with RTAB-Map is via rtabmap's own examples.
- General stereo: SetupOnYourRobot "Stereo B" (recommended) — feed left/right rectified + camera_info to rtabmap_slam with `subscribe_stereo` (RTAB-Map does its own disparity/stereo matching) or as depth. Note: ZED's `depth_registered` already has the stereo done by ZED SDK; feeding ZED depth+RGB (RGB-D mode) is also fine.
- Maintainer expectations for ZED: "relatively good visual odometry, a too noisy point cloud for 3D reconstruction but accurate enough for obstacle detection" (issue 1605) — i.e., expect to compensate with ZED pose as odometry + offline reprocessing + 4 m range limit.

URLs:
- https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_launch/README.md
- https://github.com/introlab/rtabmap_ros/tree/ros2/rtabmap_examples/launch/zed.launch.py
- https://web.archive.org/web/20231122153210/http://wiki.ros.org/rtabmap_ros/Tutorials/SetupOnYourRobot (Stereo B)
- https://github.com/introlab/rtabmap/issues/1605

## Bottom line

1. Switch to offline pipeline: record SVO replay once → produce .db via rtabmap → run `rtabmap-reprocess`/databaseViewer "Detect more loop closures" + optional SBA; iterate parameters per replay. This is the documented higher-quality route for finite recordings.
2. Use ZED's own pose as odometry prior (`use_zed_odometry:=true` / remap odom / odom_frame_id) — documented, keeps RTAB-Map loop closure + graph optimization.
3. Lower `Vis/MinInliers` to 10-12 and `Odom/MinInliers` to 10-12 (official stereo tutorial values), keep F2M, GFTT/BRIEF or GFTT/ORB, BruteForce NN, NNDR 0.8-0.9; increase `Vis/CorGuessWinSize` only if external odometry guess is good.
4. Cap feature/cloud depth ~4 m (stereo), rely on DepthAsMask to drop glass pixels; do not expect crisp point clouds from ZED stereo.
