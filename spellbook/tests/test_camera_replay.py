import os
import sys
import tempfile
import unittest

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from utils.camera_replay import (  # noqa: E402
    FRUSTUM_DEPTH, REPLAY_TRANSPORT, advance_playback, camera_center,
    discover_frames, fill_replay_cache, frustum_corners, is_valid_pose,
    load_calibration, load_color_rgb, load_matrix, load_pose,
    move_replay_selection, resize_rgb, rgb_to_rgba_bytes, step_replay_frame,
    thumbnail_size, trajectory_polylines,
)


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

    def test_camera_center_tracks_pose_translation(self):
        pose = np.eye(4)
        pose[:3, 3] = [1.5, -2.0, 0.25]
        np.testing.assert_array_equal(camera_center(pose), [1.5, -2.0, 0.25])

    def test_matrix_and_calibration_loading(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(load_matrix(os.path.join(root, "missing.txt")))
            good = os.path.join(root, "pose.txt")
            np.savetxt(good, np.eye(4))
            np.testing.assert_array_equal(load_matrix(good), np.eye(4))
            bad = os.path.join(root, "bad.txt")
            with open(bad, "w") as f:
                f.write("not a matrix\n")
            self.assertIsNone(load_matrix(bad))
            self.assertIsNone(load_calibration(root))
            self.assertIsNone(load_pose(root, 0))

    def test_color_round_trip(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "3.jpg")
            Image.fromarray(
                np.full((24, 32, 3), 200, dtype=np.uint8)).save(path)
            rgb = load_color_rgb(path)
            self.assertEqual(rgb.shape, (24, 32, 3))
            self.assertTrue((rgb == 200).all())


class ThumbnailTests(unittest.TestCase):
    def test_thumbnail_size_matches_replay_limit(self):
        self.assertEqual(thumbnail_size((1296, 968)), (664, 496))
        self.assertEqual(thumbnail_size((320, 240)), (320, 240))
        rgb = np.zeros((700, 1000, 3), dtype=np.uint8)
        resized = resize_rgb(rgb)
        self.assertEqual((resized.shape[1], resized.shape[0]), thumbnail_size((1000, 700)))

    def test_rgba_bytes_add_opaque_alpha(self):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        rgba = rgb_to_rgba_bytes(rgb)
        self.assertEqual(rgba.shape, (4, 5, 4))
        self.assertTrue((rgba[..., 3] == 255).all())


class PlaybackTests(unittest.TestCase):
    def test_advance_playback(self):
        frame, last = advance_playback(True, 0, 5, 0.0, 0.25, fps=10)
        self.assertEqual(frame, 2)
        frame, _last = advance_playback(False, 2, 5, 0.0, 9.0, fps=10)
        self.assertEqual(frame, 2)


class FrustumTests(unittest.TestCase):
    def test_identity_pose_faces_positive_z(self):
        k = np.diag([500.0, 500.0, 1.0])
        origin, corners = frustum_corners(np.eye(4), k, 640, 480)
        np.testing.assert_allclose(origin, [0.0, 0.0, 0.0])
        self.assertEqual(corners.shape, (4, 3))
        # image-plane corners sit at the display depth, forward is +Z
        np.testing.assert_allclose(corners[:, 2], FRUSTUM_DEPTH)
        np.testing.assert_allclose(corners[0], [0.0, 0.0, FRUSTUM_DEPTH])
        np.testing.assert_allclose(
            corners[2], [(640 - 1) / 500.0 * FRUSTUM_DEPTH,
                         (480 - 1) / 500.0 * FRUSTUM_DEPTH, FRUSTUM_DEPTH])

    def test_translation_shifts_pyramid(self):
        k = np.diag([500.0, 500.0, 1.0])
        pose = np.eye(4)
        pose[:3, 3] = [1.5, -2.0, 0.25]
        origin, corners = frustum_corners(pose, k, 640, 480)
        np.testing.assert_allclose(origin, [1.5, -2.0, 0.25])
        _, flat = frustum_corners(np.eye(4), k, 640, 480)
        np.testing.assert_allclose(corners - origin, flat)

    def test_yaw_turns_pyramid_sideways(self):
        k = np.diag([500.0, 500.0, 1.0])
        pose = np.eye(4)
        pose[:3, :3] = [[0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0]]
        origin, corners = frustum_corners(pose, k, 640, 480)
        np.testing.assert_allclose(origin, [0.0, 0.0, 0.0])
        # camera +Z now points along world +X
        np.testing.assert_allclose(corners[:, 0], FRUSTUM_DEPTH)


class TransportTests(unittest.TestCase):
    def test_selection_wraps(self):
        self.assertEqual(len(REPLAY_TRANSPORT), 3)
        self.assertEqual(move_replay_selection(0, -1), 2)
        self.assertEqual(move_replay_selection(2, 1), 0)
        self.assertEqual(move_replay_selection(1, 1), 2)

    def test_frame_step_wraps(self):
        self.assertEqual(step_replay_frame(0, 5, -1), 4)
        self.assertEqual(step_replay_frame(4, 5, 1), 0)
        self.assertEqual(step_replay_frame(2, 0, 1), 2)


class CachePumpTests(unittest.TestCase):
    class _Stub:
        def __init__(self, answer=True):
            self._ids = [0, 10, 20]
            self.requests = []
            self._ready = set()
            self._answer = answer

        def frames(self):
            return list(self._ids)

        def fetch(self, fid):
            if fid not in self.requests:
                self.requests.append(fid)
            if self._answer:
                self._ready.add(fid)
            return {"status": f"frame {fid}"}

        def ready(self, fid):
            return fid in self._ready

        def poll(self):
            return False

    def test_fills_everything_and_releases_requests(self):
        adapter = self._Stub(answer=True)
        cached, requested = {}, set()
        done, _status = fill_replay_cache(
            adapter, [0, 10, 20], cached, requested,
            lambda entry: ("rgba", entry["status"]))
        self.assertEqual(sorted(cached), [0, 10, 20])
        self.assertEqual(requested, set())
        self.assertEqual(sorted(done), [0, 10, 20])

    def test_bounded_window_and_idempotent_resubmit(self):
        adapter = self._Stub(answer=False)
        cached, requested = {}, set()
        fill_replay_cache(adapter, list(range(10)), cached, requested,
                          lambda entry: ("rgba", entry["status"]), window=3)
        self.assertEqual(adapter.requests, [0, 1, 2])
        self.assertEqual(cached, {})
        # repeated pumps never duplicate worker requests
        fill_replay_cache(adapter, list(range(10)), cached, requested,
                          lambda entry: ("rgba", entry["status"]), window=3)
        self.assertEqual(adapter.requests, [0, 1, 2])

    def test_fetch_failure_never_raises(self):
        class _Broken(self._Stub):
            def fetch(self, fid):
                raise RuntimeError("worker gone")

        cached, requested = {}, set()
        done, status = fill_replay_cache(
            _Broken(), [0], cached, requested, lambda entry: entry)
        self.assertEqual(done, [])
        self.assertEqual(cached, {})
        self.assertIn("replay failed", status)


if __name__ == "__main__":
    unittest.main()
