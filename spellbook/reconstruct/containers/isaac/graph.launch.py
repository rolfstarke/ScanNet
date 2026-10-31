"""Composable cuVSLAM + nvblox + ZED SVO replay graph (run inside zed-isaac-nvblox:spellbook).

One component_container_mt holding VisualSlamNode, NvbloxNode and ZedCamera, plus
robot_state_publisher for the static camera frames (#22 2026-08-27). Topic-transform routing
fails ("Lookup transform failed", empty mesh), so nvblox reads cuVSLAM's TF.

Loaded in that order so both consumers exist before the SVO starts playing. Parameters come
from /output/run_params.json and /output/zed_override.yaml, written by engines/isaac.py.
"""
import json
import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

OUTPUT = os.environ.get("ISAAC_OUTPUT", "/output")
ZED_TOPIC = "/zed/zed_node"


def generate_launch_description():
    with open(os.path.join(OUTPUT, "run_params.json")) as f:
        params = json.load(f)
    wrapper = get_package_share_directory("zed_wrapper")
    urdf = subprocess.run(
        ["xacro", os.path.join(wrapper, "urdf", "zed_descr.urdf.xacro"),
         "camera_name:=zed", f"camera_model:={params['camera_model']}"],
        check=True, capture_output=True, text=True).stdout

    zed_files = [os.path.join(wrapper, "config", "common_stereo.yaml"),
                 os.path.join(wrapper, "config", f"{params['camera_model']}.yaml"),
                 os.path.join(wrapper, "config", "object_detection.yaml"),
                 os.path.join(wrapper, "config", "custom_object_detection.yaml"),
                 os.path.join(OUTPUT, "zed_override.yaml")]
    vslam = ComposableNode(
        package="isaac_ros_visual_slam", name="visual_slam_node",
        plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode",
        parameters=[params["cuvslam"]],
        remappings=[("/visual_slam/image_0", f"{ZED_TOPIC}/left/gray/rect/image"),
                    ("/visual_slam/camera_info_0", f"{ZED_TOPIC}/left/gray/rect/camera_info"),
                    ("/visual_slam/image_1", f"{ZED_TOPIC}/right/gray/rect/image"),
                    ("/visual_slam/camera_info_1", f"{ZED_TOPIC}/right/gray/rect/camera_info")])
    nvblox = ComposableNode(
        package="nvblox_ros", name="nvblox_node", plugin="nvblox::NvbloxNode",
        parameters=[params["nvblox"]],
        remappings=[("camera_0/depth/image", f"{ZED_TOPIC}/depth/depth_registered"),
                    ("camera_0/depth/camera_info", f"{ZED_TOPIC}/depth/camera_info")])
    zed = ComposableNode(
        package="zed_components", namespace="zed", name="zed_node",
        plugin="stereolabs::ZedCamera", parameters=zed_files)

    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             namespace="zed", name="zed_state_publisher", output="screen",
             parameters=[{"robot_description": urdf}]),
        ComposableNodeContainer(
            name="spellbook_isaac_container", namespace="", package="rclcpp_components",
            executable="component_container_mt", output="screen",
            composable_node_descriptions=[vslam, nvblox, zed]),
    ])
