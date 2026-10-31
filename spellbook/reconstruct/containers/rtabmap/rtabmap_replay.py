#!/usr/bin/env python3
"""Feed every `stride`-th shared-pool frame to RTAB-Map, one at a time (runs in the container).

Each frame goes out on /map_rgbd/* (rgb8, 32FC1 metres, camera_info) and /odom + TF with one
shared stamp, then the replay waits for rtabmap's /info before sending the next one (#41 "81/81
frames acked"). Odometry is the Stage-A pool ZED pose ("assisted", #70) [inferred conversion].
"""
import argparse
import json
import os
import sys
import time

import numpy as np

D = np.diag([1.0, -1.0, -1.0, 1.0])  # ZED Y-up camera <-> OpenCV camera
OPTICAL_IN_LINK = np.array([[0.0, 0.0, 1.0, 0.0],  # x forward = optical z
                            [-1.0, 0.0, 0.0, 0.0],  # y left = -optical x
                            [0.0, -1.0, 0.0, 0.0],  # z up = -optical y
                            [0.0, 0.0, 0.0, 1.0]])
ODOM_VARIANCE = 1e-4  # [inferred] rtabmap needs a non-null covariance; value not logged
ACK_TIMEOUT_S = 120.0  # [inferred]


def frame_indices(n_frames, stride):
    return list(range(0, int(n_frames), int(stride)))


def stamp_ns(index, origin_s, fps):
    return int(origin_s) * 10**9 + int(round(index * 1e9 / fps))


def link_pose(pool_pose):
    """Pool c2w (ZED Y-up) -> odom->camera_link in a ROS z-up world, so that
    link_pose @ OPTICAL_IN_LINK = OPTICAL_IN_LINK @ D @ pool_pose @ D."""
    return OPTICAL_IN_LINK @ D @ pool_pose @ D @ np.linalg.inv(OPTICAL_IN_LINK)


def quaternion_xyzw(R):
    """Rotation matrix -> unit quaternion (x, y, z, w); no scipy in the image."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(1.0 + t)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i], q[j], q[k] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return np.asarray(q) / np.linalg.norm(q)


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _transform(parent, child, T, stamp):
    from geometry_msgs.msg import TransformStamped
    msg = TransformStamped()
    msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, parent, child
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = \
        (float(v) for v in T[:3, 3])
    r = msg.transform.rotation
    r.x, r.y, r.z, r.w = (float(v) for v in quaternion_xyzw(T[:3, :3]))
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--stamp-origin", type=int, required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--watch-pids", type=int, nargs="*", default=[])
    args = ap.parse_args()

    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from rtabmap_msgs.msg import Info
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

    n_pool = sum(1 for _ in open(os.path.join(args.frames, "pose_state.txt")))
    indices = frame_indices(n_pool, args.stride)
    K = np.loadtxt(os.path.join(args.frames, "intrinsic_depth.txt"))[:3, :3]

    rclpy.init()
    node = rclpy.create_node("rtabmap_replay")
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    pub_rgb = node.create_publisher(Image, "/map_rgbd/rgb/image", qos)
    pub_depth = node.create_publisher(Image, "/map_rgbd/depth/image", qos)
    pub_info = node.create_publisher(CameraInfo, "/map_rgbd/rgb/camera_info", qos)
    pub_odom = node.create_publisher(Odometry, "/odom", qos)
    tf = TransformBroadcaster(node)
    static_tf = StaticTransformBroadcaster(node)
    acks = []
    node.create_subscription(Info, "/info", lambda m: acks.append(int(m.ref_id)), qos)
    bridge = CvBridge()

    def check_nodes():
        dead = [p for p in args.watch_pids if not _alive(p)]
        if dead:
            sys.exit(f"[replay] node process died: {dead}")

    deadline = time.monotonic() + ACK_TIMEOUT_S
    while not (pub_rgb.get_subscription_count() and pub_depth.get_subscription_count()
               and pub_info.get_subscription_count() and pub_odom.get_subscription_count()
               and node.count_publishers("/info")
               and node.count_subscribers("/map_rgbd/rgbd_image")):
        check_nodes()
        if time.monotonic() > deadline:
            sys.exit("[replay] rgbd_sync/rtabmap never subscribed")
        rclpy.spin_once(node, timeout_sec=0.2)
    settle_until = time.monotonic() + 2.0  # [inferred] both ends of /info must match
    while time.monotonic() < settle_until:
        rclpy.spin_once(node, timeout_sec=0.1)

    static_tf.sendTransform(
        _transform("camera_link", "camera_color_optical_frame", OPTICAL_IN_LINK,
                   Time(nanoseconds=stamp_ns(0, args.stamp_origin, args.fps)).to_msg()))

    frames = []
    for i in indices:
        color = cv2.imread(os.path.join(args.frames, "color", f"{i}.jpg"))
        depth = cv2.imread(os.path.join(args.frames, "depth", f"{i}.png"), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            sys.exit(f"[replay] unreadable pool frame {i}")
        h, w = depth.shape[:2]
        if color.shape[:2] != (h, w):
            color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
        stamp = Time(nanoseconds=stamp_ns(i, args.stamp_origin, args.fps)).to_msg()
        frame_id = "camera_color_optical_frame"

        rgb_msg = bridge.cv2_to_imgmsg(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), "rgb8")
        depth_msg = bridge.cv2_to_imgmsg(depth.astype(np.float32) / 1000.0, "32FC1")
        rgb_msg.header.stamp = depth_msg.header.stamp = stamp
        rgb_msg.header.frame_id = depth_msg.header.frame_id = frame_id

        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, frame_id
        info.width, info.height, info.distortion_model = w, h, "plumb_bob"
        info.d = [0.0] * 5
        info.k = [float(v) for v in K.ravel()]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(K[0, 0]), 0.0, float(K[0, 2]), 0.0,
                  0.0, float(K[1, 1]), float(K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]

        T = link_pose(np.loadtxt(os.path.join(args.frames, "pose", f"{i}.txt")))
        tf_msg = _transform("odom", "camera_link", T, stamp)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "odom", "camera_link"
        odom.pose.pose.position.x = tf_msg.transform.translation.x
        odom.pose.pose.position.y = tf_msg.transform.translation.y
        odom.pose.pose.position.z = tf_msg.transform.translation.z
        odom.pose.pose.orientation = tf_msg.transform.rotation
        cov = [0.0] * 36
        cov[0] = cov[7] = cov[14] = cov[21] = cov[28] = cov[35] = ODOM_VARIANCE
        odom.pose.covariance = cov
        odom.twist.covariance = cov

        before = len(acks)
        tf.sendTransform(tf_msg)
        pub_odom.publish(odom)
        pub_rgb.publish(rgb_msg)
        pub_depth.publish(depth_msg)
        pub_info.publish(info)
        deadline = time.monotonic() + ACK_TIMEOUT_S
        while len(acks) == before:
            check_nodes()
            if time.monotonic() > deadline:
                sys.exit(f"[replay] no rtabmap ack for pool frame {i} after {ACK_TIMEOUT_S}s")
            rclpy.spin_once(node, timeout_sec=0.05)
        frames.append(i)
        print(f"[replay] acked {len(frames)}/{len(indices)} pool frame {i} node {acks[-1]}",
              flush=True)

    span = (indices[-1] - indices[0]) / max(1, n_pool - 1)
    result = dict(pool_frames=n_pool, stride=args.stride, sent=indices, acked=len(frames),
                  span=span)
    with open(args.result, "w") as f:
        json.dump(result, f)
    print(f"[replay] {len(frames)}/{len(indices)} frames acked, span {span:.1%}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
