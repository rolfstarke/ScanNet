# Research: Isaac ROS cuVSLAM + Nvblox via rootless Podman (ZED X, .svo2 replay) — 2026-08-08

Verdict: **ACHIEVABLE rootless/no-sudo.** Host stack already satisfies every prerequisite (see Host check).

## 1. Official container setup (Isaac ROS 4.5 / ROS 2 Jazzy)
- x86_64 requirements: Ampere+ GPU, Ubuntu 24.04, **CUDA 13.0+**, **NVIDIA Driver 580+**, 32 GB disk.
  https://nvidia-isaac-ros.github.io/getting_started/index.html#system-requirements
- Recommended = Docker mode: only host prereqs are Docker engine + nvidia-container-toolkit + driver; user added to docker group. All Debian/pip deps **installed inside the container (host unaffected)**; CUDA/VPI/ROS userspace is baked into the NGC prebuilt image `nvidia/isaac/ros` (image key `isaac_ros`).
  https://nvidia-isaac-ros.github.io/concepts/dev_env/index.html#docker-mode-configuration
  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/containers/ros
- ZED layer = additional Dockerfile.zed image key (`additional_image_keys: [zed]`); ZED SDK + zed-ros2-wrapper installed INTO the image at build. **ZED SDK not needed on host.**
  https://nvidia-isaac-ros.github.io/getting_started/sensors/zed_setup.html
- Official `isaac-ros-cli` is apt/sudo-based convenience only; same result via `podman run --device nvidia.com/gpu=all nvcr.io/nvidia/isaac/ros:<tag>` + local `buildah` build for the `zed` layer.

## 2. ZED integration with cuVSLAM
- Official vslam ZED quickstart (release 4.5):
  `ros2 launch isaac_ros_examples isaac_ros_examples.launch.py launch_fragments:=zed_stereo_rect,visual_slam pub_frame_rate:=30.0 base_frame:=zed2_camera_center camera_optical_frames:="['zed2_left_camera_frame_optical','zed2_right_camera_frame_optical']" interface_specs_file:=.../zed2_quickstart_interface_specs.json`
  Packages: `ros-jazzy-isaac-ros-examples ros-jazzy-isaac-ros-stereo-image-proc ros-jazzy-isaac-ros-zed`; ZED X => replace zed2 with zedx.
  https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html#run-launch-file
- Nvblox ZED example: `ros2 launch nvblox_examples_bringup zed_example.launch.py camera:=zedx [rosbag:=<path>]`; remaps `camera_0/depth/image -> /zed/zed_node/depth/depth_registered`, `pose -> /zed/zed_node/pose` (uses ZED SDK pose in this example; feeding cuVSLAM odometry instead is the documented general graph).
  https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/tutorials/tutorial_zed.html
  Launch: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/blob/main/nvblox_examples/nvblox_examples_bringup/launch/zed_example.launch.py
- .svo2 replay without camera: zed_wrapper `camera_model:=virtual` (+ `svo_file`/`svo_offline_loop` params). ZED X host driver only needed for live USB camera, not replay.
  https://docs.stereolabs.com/integrations/ros-2/stereo-node/
- Note: ZED X not in SQA test list (ZED/2/2i are); works but unsupported.

## 3. Nvblox offline / quality settings
- No dedicated batch-offline mode; official offline path = **rosbag replay** into the node (`rosbag:=<path>` arg, `use_sim_time`), optionally `--rate > 1` for faster-than-realtime. Map persistence: `after_shutdown_map_save_path` param; `load_map`/`save_map` services.
- Indoor room-scale (5-15 m), ZED stereo depth: `voxel_size` default 0.05 m (1 cm supported); raise `static_mapper.projective_integrator_max_integration_distance_m` above default 7.0 m for 15 m rooms; truncation `projective_integrator_truncation_distance_vox` default 4 voxels; `projective_integrator_max_weight` 5.0.
  https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/isaac_ros_nvblox/api/parameters.html

## 4. Rootless Podman + NVIDIA
- Officially supported: Podman uses **CDI** (`--device nvidia.com/gpu=all`, Podman >= 4.1); CDI explicitly "improves compatibility ... with rootless containers".
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html
- Rootless runtime config documented (per-user daemon.json + `no-cgroups`).
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#rootless-mode

## Host check (this machine)
- podman 4.9.3 (rootless, CDI capable) ✔
- nvidia-ctk + nvidia-smi present, driver **580.159.03** (>= 580 required) ✔
- CDI specs already generated: `/var/run/cdi/nvidia.yaml` AND `/home/rolf/.config/cdi/nvidia.yaml` ✔
- RTX A4000 = Ampere, 16 GB ✔
- No host CUDA/VPI install needed (userspace lives in container image).

## Action plan (build agent)
1. Smoke test: `podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable nvcr.io/nvidia/isaac/ros:release-4.5 nvidia-smi`
2. Build zed layer locally (buildah/podman, Dockerfile.zed + ZED SDK download inside image; ~GB download, needs network).
3. In-container: launch `zed_stereo_rect,visual_slam` with `zedx` spec; then `nvblox_examples_bringup zed_example.launch.py camera:=zedx rosbag:=...` (bag recorded from SVO via zed_wrapper `virtual` model).
4. Tune: voxel 0.02-0.05, integration distance 10-15 m, truncation 4-8 voxels.
