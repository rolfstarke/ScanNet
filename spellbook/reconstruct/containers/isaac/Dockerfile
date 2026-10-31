# zed-isaac-nvblox:spellbook
# Isaac ROS 4.5 (cuVSLAM + nvblox) from the PUBLIC NVIDIA apt repos, no NGC login (#22, #42).
# ZED SDK 5.4.1 (CUDA 13) + zed-ros2-wrapper v5.4.1 from source (no jazzy apt package).
# Consumed by spellbook/reconstruct/engines/isaac.py, run as the invoking user:
# everything is outside /root and /usr/local/zed is world-writable. Driver and GPU come from
# containers/gpu_args.sh at `docker run`.
FROM docker.io/library/ubuntu:24.04

ARG ISAAC_ROS_RELEASE=release-4.5
ARG JETSON_REPO=r38.4
ARG ZED_SDK_URL=https://download.stereolabs.com/zedsdk/5.4.1/cu13/ubuntu24
ARG ZED_WRAPPER_TAG=v5.4.1
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git zstd lsb-release libusb-1.0-0 libturbojpeg udev \
 && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
        | gpg --dearmor -o /usr/share/keyrings/isaac-ros.gpg \
 && curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
        | gpg --dearmor -o /usr/share/keyrings/jetson-ota.gpg \
 && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-archive-keyring.gpg \
        -o /usr/share/keyrings/cuda-archive-keyring.gpg \
 && { echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main"; \
      echo "deb [signed-by=/usr/share/keyrings/isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/${ISAAC_ROS_RELEASE} noble main"; \
      echo "deb [signed-by=/usr/share/keyrings/jetson-ota.gpg] https://repo.download.nvidia.com/jetson/x86_64/noble ${JETSON_REPO} main"; \
      echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"; \
    } > /etc/apt/sources.list.d/spellbook-isaac.list

# nvblox_examples_bringup is left out: it pulls detectnet/triton model installers we do not use.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-robot-state-publisher \
        ros-jazzy-xacro \
        ros-jazzy-isaac-ros-visual-slam \
        ros-jazzy-nvblox-ros \
        ros-jazzy-nvblox-msgs \
        python3-colcon-common-extensions python3-rosdep \
 && rm -rf /var/lib/apt/lists/*

# ZED SDK: runtime only. AI models are cached by a bind mount at run time (engines/isaac.py).
RUN curl -fsSL -o /tmp/zed_sdk.run "${ZED_SDK_URL}" \
 && chmod +x /tmp/zed_sdk.run \
 && /tmp/zed_sdk.run -- silent skip_tools skip_drivers skip_python \
 && rm -f /tmp/zed_sdk.run \
 && chmod -R a+rwX /usr/local/zed

RUN apt-get update && rosdep init 2>/dev/null; rosdep update --rosdistro jazzy \
 && mkdir -p /opt/ros2_ws/src && cd /opt/ros2_ws/src \
 && git clone --depth 1 --branch "${ZED_WRAPPER_TAG}" https://github.com/stereolabs/zed-ros2-wrapper.git \
 && cd /opt/ros2_ws && . /opt/ros/jazzy/setup.sh \
 && rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy \
 && colcon build --packages-skip zed_debug --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers 8 \
 && rm -rf build log /var/lib/apt/lists/*

WORKDIR /opt/ros2_ws
CMD ["bash"]
