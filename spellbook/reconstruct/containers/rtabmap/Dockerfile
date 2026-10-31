# localhost/zed-rtabmap:jazzy
# ROS 2 Jazzy + rtabmap_ros for the assisted-hybrid baseline (engines/rtabmap.py): replay of the
# shared frame pool, poses exported from the database. The baseline never opens an SVO, so
# neither the ZED SDK nor zed-ros2-wrapper is installed (the tag name is historical, #41).
# No build context needed (snap docker): docker build --network host -t localhost/zed-rtabmap:jazzy - < Dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    LANG=C.UTF-8

# ROS 2 apt repository (official method, Ubuntu 24.04 noble)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list \
 && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        ros-jazzy-rtabmap-ros \
        ros-jazzy-rtabmap \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-cv-bridge \
        python3-opencv python3-numpy \
 && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
