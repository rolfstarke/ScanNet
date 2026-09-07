import os
import sys
import tempfile
import unittest
from unittest import mock


_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

import download_scans  # noqa: E402


class DownloadScansTests(unittest.TestCase):
    def test_file_url_uses_v1_only_for_sens(self):
        self.assertEqual(
            download_scans.file_url("scene0458_01", ".sens"),
            "https://kaldir.vc.in.tum.de/scannet/v1/scans/scene0458_01/scene0458_01.sens")
        self.assertEqual(
            download_scans.file_url("scene0458_01", "_vh_clean_2.ply"),
            "https://kaldir.vc.in.tum.de/scannet/v2/scans/scene0458_01/"
            "scene0458_01_vh_clean_2.ply")

    def test_validate_scenes_requires_official_unique_val_ids(self):
        with mock.patch.object(download_scans, "official_val_scenes",
                               return_value=frozenset({"scene0458_01"})):
            self.assertEqual(download_scans.validate_scenes(["0458_01"]), ["scene0458_01"])
            with self.assertRaises(ValueError):
                download_scans.validate_scenes(["scene9004_40"])
            with self.assertRaises(ValueError):
                download_scans.validate_scenes(["scene0458_01", "0458_01"])

    def test_existing_complete_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            scene = "scene0458_01"
            path = os.path.join(root, "scans", scene, scene + ".txt")
            os.makedirs(os.path.dirname(path))
            with open(path, "wb") as handle:
                handle.write(b"data")
            with mock.patch.object(download_scans, "_remote_size", return_value=4), \
                    mock.patch.object(download_scans.subprocess, "run") as run:
                result = download_scans._download_one(scene, ".txt", root)
            self.assertEqual(result["status"], "skipped")
            run.assert_not_called()

    def test_existing_complete_file_cleans_stale_partials(self):
        with tempfile.TemporaryDirectory() as root:
            scene = "scene0458_01"
            path = os.path.join(root, "scans", scene, scene + ".txt")
            os.makedirs(os.path.dirname(path))
            for candidate, data in ((path, b"data"), (path + ".part", b"old"),
                                    (path + ".part.0-3", b"data")):
                with open(candidate, "wb") as handle:
                    handle.write(data)
            with mock.patch.object(download_scans, "_remote_size", return_value=4):
                result = download_scans._download_one(scene, ".txt", root)
            self.assertEqual(result["status"], "skipped")
            self.assertFalse(os.path.exists(path + ".part"))
            self.assertFalse(os.path.exists(path + ".part.0-3"))

    def test_byte_ranges_cover_file_without_overlap(self):
        self.assertEqual(download_scans._byte_ranges(10, 3), [(0, 3), (4, 7), (8, 9)])

    def test_segmented_download_merges_verified_ranges(self):
        with tempfile.TemporaryDirectory() as root:
            destination = os.path.join(root, "scene.sens")
            partial = destination + ".part"

            def write_ranges(command, check):
                self.assertFalse(check)
                for index, value in enumerate(command):
                    if value != "--range":
                        continue
                    start, end = (int(part) for part in command[index + 1].split("-"))
                    output_index = command.index("--output", index)
                    with open(command[output_index + 1], "wb") as handle:
                        handle.write(bytes([start]) * (end - start + 1))
                return mock.Mock(returncode=0)

            with mock.patch.object(download_scans.subprocess, "run", side_effect=write_ranges):
                download_scans._download_segmented(
                    "https://example.test/scene.sens", destination, partial, 10, 3)
            with open(partial, "rb") as handle:
                self.assertEqual(handle.read(), b"\x00" * 4 + b"\x04" * 4 + b"\x08" * 2)
            self.assertEqual(list(glob for glob in os.listdir(root)), ["scene.sens.part"])

    def test_partial_file_resumes_and_is_atomically_promoted(self):
        with tempfile.TemporaryDirectory() as root:
            scene = "scene0458_01"
            path = os.path.join(root, "scans", scene, scene + ".txt")
            partial = path + ".part"
            os.makedirs(os.path.dirname(path))
            with open(partial, "wb") as handle:
                handle.write(b"ab")

            def finish(command, check):
                self.assertFalse(check)
                self.assertIn("--continue-at", command)
                with open(partial, "ab") as handle:
                    handle.write(b"cd")
                return mock.Mock(returncode=0)

            with mock.patch.object(download_scans, "_remote_size", return_value=4), \
                    mock.patch.object(download_scans.subprocess, "run", side_effect=finish):
                result = download_scans._download_one(scene, ".txt", root)
            self.assertEqual(result["status"], "downloaded")
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.exists(partial))
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"abcd")

    def test_complete_partial_is_promoted_and_orphan_segments_removed(self):
        with tempfile.TemporaryDirectory() as root:
            scene = "scene0458_01"
            path = os.path.join(root, "scans", scene, scene + ".txt")
            partial = path + ".part"
            segment = partial + ".0-3"
            os.makedirs(os.path.dirname(path))
            for candidate in (partial, segment):
                with open(candidate, "wb") as handle:
                    handle.write(b"data")
            with mock.patch.object(download_scans, "_remote_size", return_value=4), \
                    mock.patch.object(download_scans.subprocess, "run") as run:
                result = download_scans._download_one(scene, ".txt", root)
            self.assertEqual(result["status"], "downloaded")
            run.assert_not_called()
            self.assertTrue(os.path.isfile(path))
            self.assertFalse(os.path.exists(partial))
            self.assertFalse(os.path.exists(segment))

    def test_replace_discards_final_and_partial_copies(self):
        with tempfile.TemporaryDirectory() as root:
            scene = "scene0458_01"
            path = os.path.join(root, "scans", scene, scene + ".txt")
            partial = path + ".part"
            segment = partial + ".0-2"
            os.makedirs(os.path.dirname(path))
            for candidate in (path, partial, segment):
                with open(candidate, "wb") as handle:
                    handle.write(b"old")

            def download(command, check):
                self.assertFalse(check)
                self.assertFalse(os.path.exists(path))
                self.assertFalse(os.path.exists(segment))
                with open(partial, "wb") as handle:
                    handle.write(b"new!")
                return mock.Mock(returncode=0)

            with mock.patch.object(download_scans, "_remote_size", return_value=4), \
                    mock.patch.object(download_scans.subprocess, "run", side_effect=download):
                result = download_scans._download_one(scene, ".txt", root, replace=True)
            self.assertEqual(result["status"], "downloaded")
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"new!")

    def test_oversized_partial_is_rejected_without_curl(self):
        with tempfile.TemporaryDirectory() as root:
            scene = "scene0458_01"
            path = os.path.join(root, "scans", scene, scene + ".txt")
            partial = path + ".part"
            os.makedirs(os.path.dirname(path))
            with open(partial, "wb") as handle:
                handle.write(b"abcde")
            with mock.patch.object(download_scans, "_remote_size", return_value=4), \
                    mock.patch.object(download_scans.subprocess, "run") as run:
                with self.assertRaises(RuntimeError):
                    download_scans._download_one(scene, ".txt", root)
            run.assert_not_called()
            self.assertFalse(os.path.exists(path))

    def test_curl_failure_keeps_resumable_partial(self):
        with tempfile.TemporaryDirectory() as root:
            scene = "scene0458_01"
            path = os.path.join(root, "scans", scene, scene + ".txt")
            partial = path + ".part"
            os.makedirs(os.path.dirname(path))
            with open(partial, "wb") as handle:
                handle.write(b"ab")
            with mock.patch.object(download_scans, "_remote_size", return_value=4), \
                    mock.patch.object(download_scans.subprocess, "run",
                                      return_value=mock.Mock(returncode=7)):
                with self.assertRaisesRegex(RuntimeError, "partial=2/4"):
                    download_scans._download_one(scene, ".txt", root)
            self.assertFalse(os.path.exists(path))
            self.assertTrue(os.path.isfile(partial))

    def test_remote_size_retries_transient_failure(self):
        response = mock.MagicMock()
        response.__enter__.return_value.headers = {"Content-Length": "4"}
        with mock.patch.object(download_scans.urllib.request, "urlopen",
                               side_effect=[OSError("temporary"), response]) as urlopen, \
                mock.patch.object(download_scans.time, "sleep") as sleep:
            self.assertEqual(download_scans._remote_size("https://example.test/file"), 4)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_incomplete_download_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as root:
            scene = "scene0458_01"
            path = os.path.join(root, "scans", scene, scene + ".txt")
            partial = path + ".part"
            os.makedirs(os.path.dirname(path))
            with open(partial, "wb") as handle:
                handle.write(b"ab")
            with mock.patch.object(download_scans, "_remote_size", return_value=4), \
                    mock.patch.object(download_scans.subprocess, "run",
                                      return_value=mock.Mock(returncode=0)):
                with self.assertRaises(RuntimeError):
                    download_scans._download_one(scene, ".txt", root)
            self.assertFalse(os.path.exists(path))
            self.assertTrue(os.path.isfile(partial))


if __name__ == "__main__":
    unittest.main()
