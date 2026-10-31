"""Line-buffered pose/frame recorder (replaces rosbag, which hung on SIGINT, #22 2026-08-27).

Writes into --out, flushed per line so a killed run keeps what it saw:
  svo_status.txt  frame_ts_ns frame_id total_frames status
  poses.txt       stamp_ns x y z qx qy qz qw        (cuVSLAM odom -> base frame)
  vo_status.txt   stamp_ns vo_state
  tf_static.txt   parent child x y z qx qy qz qw
  svo_end.txt     final_frame_id total_frames       (only once the SVO really ended)
Exit: 0 SVO ended (or first frame with --stop-at-first-frame, used to prime the ZED model cache),
3 no first frame within --startup-s, 4 playback stalled for --stall-s.
"""
import argparse
import os
import sys
import time

import rclpy
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from zed_msgs.msg import SvoStatus

STATUS_END = SvoStatus.STATUS_END


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class Recorder(Node):
    def __init__(self, out, drain_s, startup_s, stall_s, stop_at_first_frame=False):
        super().__init__("spellbook_pose_recorder")
        self.files = {name: open(os.path.join(out, f"{name}.txt"), "w", buffering=1)
                      for name in ("svo_status", "poses", "vo_status", "tf_static")}
        self.out, self.drain_s, self.startup_s, self.stall_s = out, drain_s, startup_s, stall_s
        self.stop_at_first_frame = stop_at_first_frame
        self.started = time.monotonic()
        self.last_frame = None
        self.last_progress = None
        self.end = None
        self.end_time = None
        self.exit_code = None
        self.static_seen = set()
        reliable = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(SvoStatus, "/zed/zed_node/status/svo", self.on_status, reliable)
        self.create_subscription(Odometry, "/visual_slam/tracking/odometry", self.on_odom, reliable)
        self.create_subscription(VisualSlamStatus, "/visual_slam/status", self.on_vo, reliable)
        self.create_subscription(TFMessage, "/tf_static", self.on_static, latched)
        self.create_timer(0.5, self.on_tick)

    def on_status(self, msg):
        self.files["svo_status"].write(
            f"{msg.frame_ts} {msg.frame_id} {msg.total_frames} {msg.status}\n")
        if msg.frame_id != self.last_frame:
            self.last_frame, self.last_progress = msg.frame_id, time.monotonic()
        if msg.status == STATUS_END and self.end is None:
            self.end, self.end_time = (msg.frame_id, msg.total_frames), time.monotonic()

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.files["poses"].write(f"{stamp_ns(msg.header.stamp)} {p.x:.9f} {p.y:.9f} {p.z:.9f} "
                                  f"{q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n")

    def on_vo(self, msg):
        self.files["vo_status"].write(f"{stamp_ns(msg.header.stamp)} {msg.vo_state}\n")

    def on_static(self, msg):
        for t in msg.transforms:
            key = (t.header.frame_id, t.child_frame_id)
            if key in self.static_seen:
                continue
            self.static_seen.add(key)
            v, q = t.transform.translation, t.transform.rotation
            self.files["tf_static"].write(f"{key[0]} {key[1]} {v.x:.9f} {v.y:.9f} {v.z:.9f} "
                                          f"{q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n")

    def on_tick(self):
        now = time.monotonic()
        if self.stop_at_first_frame and self.last_progress is not None:
            self.exit_code = 0
        elif self.end is not None:
            # cuVSLAM may still be publishing the last frames' poses
            if now - self.end_time >= self.drain_s:
                with open(os.path.join(self.out, "svo_end.txt"), "w") as f:
                    f.write(f"{self.end[0]} {self.end[1]}\n")
                self.exit_code = 0
        elif self.last_progress is None:
            if now - self.started > self.startup_s:
                self.exit_code = 3
        elif now - self.last_progress > self.stall_s:
            self.exit_code = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--drain-s", type=float, default=3.0)
    ap.add_argument("--startup-s", type=float, default=240.0)
    ap.add_argument("--stall-s", type=float, default=60.0)
    ap.add_argument("--stop-at-first-frame", action="store_true")
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args.out, args.drain_s, args.startup_s, args.stall_s,
                    args.stop_at_first_frame)
    while rclpy.ok() and node.exit_code is None:
        rclpy.spin_once(node, timeout_sec=0.5)
    code = node.exit_code if node.exit_code is not None else 1
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
