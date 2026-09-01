import os
import sys
import tempfile
import unittest

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from utils.camera_replay import (  # noqa: E402
    advance_playback, discover_frames, is_valid_pose, owner_colors,
    packed_visible, project_points, trajectory_polylines, update_seen_tp,
    zbuffer_colors,
)
from utils.prediction_masks import pack_mask  # noqa: E402


class FrameTests(unittest.TestCase):
    def test_numeric_order_and_invalid_poses(self):
        with tempfile.TemporaryDirectory() as root:
            for folder, ext, ids in (
                    ("color", ".jpg", [0, 2, 10]),
                    ("depth", ".png", [0, 2]),
                    ("pose", ".txt", [0, 2, 10]),
            ):
                os.makedirs(os.path.join(root, folder))
                for i in ids:
                    open(os.path.join(root, folder, f"{i}{ext}"), "w").close()
            found = discover_frames(root)
            self.assertEqual(found["rgb_ids"], [0, 2, 10])
            self.assertEqual(found["proj_ids"], [0, 2])
        self.assertFalse(is_valid_pose(None))
        inf = np.full((4, 4), -np.inf)
        self.assertFalse(is_valid_pose(inf))
        self.assertTrue(is_valid_pose(np.eye(4)))
        poses = [np.eye(4), None, np.eye(4), np.eye(4)]
        poses[2][:3, 3] = [1, 0, 0]
        poses[3][:3, 3] = [2, 0, 0]
        lines = trajectory_polylines(poses)
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(lines[0]), 2)


class ProjectionTests(unittest.TestCase):
    def test_identity_and_occlusion(self):
        k = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
        calib = {"K_color": k, "K_depth": k, "E_depth": np.eye(4)}
        depth = np.full((48, 64), 2.0)
        pose = np.eye(4)
        pts = np.array([[0.0, 0.0, 2.0], [10.0, 10.0, 2.0], [0.0, 0.0, -1.0]])
        uv, z, valid = project_points(pts, pose, calib, depth, (64, 48), vis_thresh=0.1, cut_bound=0)
        self.assertTrue(valid[0])
        np.testing.assert_array_equal(uv[0], [32, 24])
        self.assertFalse(valid[1])
        self.assertFalse(valid[2])
        depth[24, 32] = 1.0
        _uv, _z, valid2 = project_points(pts[:1], pose, calib, depth, (64, 48), vis_thresh=0.1, cut_bound=0)
        self.assertFalse(valid2[0])

    def test_nearest_z_wins(self):
        uv = np.array([[1, 1], [1, 1], [2, 2]])
        z = np.array([3.0, 1.0, 2.0])
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        valid = np.array([True, True, True])
        overlay, mask = zbuffer_colors(uv, z, colors, valid, 4, 4)
        np.testing.assert_array_equal(overlay[1, 1], [0.0, 1.0, 0.0])
        self.assertTrue(mask[1, 1])


class PlaybackTpTests(unittest.TestCase):
    def test_playback_and_monotonic_tp(self):
        frame, last = advance_playback(True, 0, 5, 0.0, 0.25, fps=10)
        self.assertEqual(frame, 2)
        frame, _last = advance_playback(False, 2, 5, 0.0, 9.0, fps=10)
        self.assertEqual(frame, 2)
        visible = np.array([True, True, False, False])
        packed = packed_visible(visible)
        objects = [
            {"key": "a", "verdict": "tp", "packed": pack_mask([True, False, False, False])},
            {"key": "b", "verdict": "tp", "packed": pack_mask([False, False, True, True])},
            {"key": "c", "verdict": "fp", "packed": pack_mask([True, True, True, True])},
        ]
        seen, added = update_seen_tp(set(), packed, objects)
        self.assertEqual(seen, {"a"})
        self.assertEqual(added, 1)
        seen, added = update_seen_tp(seen, packed, objects)
        self.assertEqual(added, 0)
        owners, colors = owner_colors(3, [
            {"sel": np.array([True, True, False]), "score": 0.2},
            {"sel": np.array([False, True, True]), "score": 0.9},
        ], lambda i, o: [i + 1.0, 0.0, 0.0])
        self.assertEqual(list(owners), [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
