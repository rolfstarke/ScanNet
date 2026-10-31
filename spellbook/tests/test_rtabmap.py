"""RTAB-Map assisted-hybrid baseline: frame selection, pose chain, YAML, export parsing, docker
args. CPU only; no container or GPU is touched."""
import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from reconstruct.engines import rtabmap  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rtabmap_replay",
    os.path.join(_SPELLBOOK, "reconstruct", "containers", "rtabmap", "rtabmap_replay.py"))
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


def _pose(angle=0.4, t=(1.0, 2.0, 3.0)):
    c, s = np.cos(angle), np.sin(angle)
    P = np.eye(4)
    P[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    P[:3, 3] = t
    return P


class FrameSelectionTests(unittest.TestCase):
    def test_baseline_stride_matches_the_log(self):
        idx = replay.frame_indices(3612, 90)
        self.assertEqual((len(idx), idx[0], idx[-1]), (41, 0, 3600))
        self.assertAlmostEqual(idx[-1] / 3611, 0.997, places=3)

    def test_scene9009_node_count(self):
        self.assertEqual(len(replay.frame_indices(2356, 90)), 27)

    def test_stamp_round_trip(self):
        for i in (0, 90, 3600):
            ns = replay.stamp_ns(i, rtabmap.STAMP_ORIGIN, rtabmap.SOURCE_FPS)
            self.assertEqual(rtabmap._pool_index(ns / 1e9), i)


class PoseChainTests(unittest.TestCase):
    def test_exported_camera_is_opencv_camera_in_ros_world(self):
        P = _pose()
        camera = replay.link_pose(P) @ replay.OPTICAL_IN_LINK
        expected = replay.OPTICAL_IN_LINK @ replay.D @ P @ replay.D
        np.testing.assert_allclose(camera, expected)

    def test_identity_pool_pose_is_identity_odom(self):
        np.testing.assert_allclose(replay.link_pose(np.eye(4)), np.eye(4), atol=1e-12)

    def test_quaternion_round_trip(self):
        for angle in (0.0, 0.4, 3.0):
            R = _pose(angle)[:3, :3]
            x, y, z, w = replay.quaternion_xyzw(R)
            np.testing.assert_allclose(rtabmap._quat_to_R(x, y, z, w), R, atol=1e-9)


class ConfigTests(unittest.TestCase):
    def test_yaml_is_valid_and_keeps_logged_values(self):
        params = yaml.safe_load(rtabmap._RTABMAP_YAML)["/**"]["ros__parameters"]
        self.assertEqual(params["Mem/IncrementalMemory"], "true")
        self.assertEqual(params["Rtabmap/DetectionRate"], "0")
        self.assertEqual(params["Optimizer/Strategy"], "2")
        self.assertEqual(params["Kp/MaxFeatures"], "400")
        self.assertEqual(params["approx_sync_max_interval"], 2.0)
        self.assertEqual((params["wait_for_transform"], params["tf_tolerance"]), (0.5, 0.5))
        sync = yaml.safe_load(rtabmap._RGBD_SYNC_YAML)["/**"]["ros__parameters"]
        self.assertEqual((sync["topic_queue_size"], sync["sync_queue_size"]), (30, 30))

    def test_export_is_poses_only_with_gtsam(self):
        self.assertIn("--poses_camera --poses_format 11 --opt 2 --voxel 0.01", rtabmap._INSIDE_SH)
        for flag in ("--cloud", "--mesh", "rtabmap-reprocess", "detectMoreLoopClosures"):
            self.assertNotIn(flag, rtabmap._INSIDE_SH)

    def test_mesh_uses_the_shared_unthresholded_tsdf(self):
        self.assertEqual((rtabmap.FUSION["voxel_size"], rtabmap.FUSION["sdf_trunc"],
                          rtabmap.FUSION["depth_max"]), (0.02, 0.06, 6.0))
        self.assertNotIn("weight_threshold", rtabmap.FUSION)

    def test_container_name_is_per_pid(self):
        self.assertTrue(rtabmap.CONTAINER_NAME.endswith(f"-{os.getpid()}"))

    def test_defaults_survive_empty_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(rtabmap.provenance()["variant"], "assisted-hybrid-s90")
        with mock.patch.dict(os.environ, {"SPELLBOOK_RTABMAP_STRIDE": "45"}, clear=True):
            self.assertEqual(rtabmap.selected_stride(), 45)


class ExportParsingTests(unittest.TestCase):
    def _export(self, d, ids_and_frames):
        with open(os.path.join(d, "scene_camera_poses.txt"), "w") as f:
            f.write("#format\n")
            for node, frame in ids_and_frames:
                stamp = rtabmap.STAMP_ORIGIN + frame / rtabmap.SOURCE_FPS
                f.write(f"{stamp:.6f} 1 2 3 0 0 0 1 {node}\n")

    def test_nodes_map_to_pool_indices(self):
        with tempfile.TemporaryDirectory() as d:
            self._export(d, [(1, 0), (2, 90), (3, 180)])
            poses, keep = rtabmap._parse_export(d, [0, 90, 180])
        self.assertEqual(keep, [0, 90, 180])
        self.assertEqual(poses.shape, (3, 4, 4))
        np.testing.assert_allclose(poses[1, :3, 3], (1, 2, 3))

    def test_missing_node_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self._export(d, [(1, 0), (2, 180)])
            with self.assertRaises(RuntimeError):
                rtabmap._parse_export(d, [0, 90, 180])


class ContainerTests(unittest.TestCase):
    def test_cap_kills_the_container(self):
        class Cap(BaseException):
            pass
        proc = mock.Mock()
        proc.wait.side_effect = [Cap(), 0]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(rtabmap.subprocess, "Popen", return_value=proc), \
                mock.patch.object(rtabmap.subprocess, "run") as run:
            with self.assertRaises(Cap):
                rtabmap._run_container(["docker"], "name", os.path.join(d, "log"), 3)
        self.assertEqual(run.call_args[0][0], ["docker", "kill", "name"])

    def test_gpu_check_requires_lease(self):
        self.assertEqual(rtabmap.gpu_check()["status"], "fail")


if __name__ == "__main__":
    unittest.main()
