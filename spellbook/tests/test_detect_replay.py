import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from utils.detect_replay import (  # noqa: E402
    DetectClient, _label_font, detection_stride, draw_detections, draw_masks)


class StrideTests(unittest.TestCase):
    def test_missing_config_falls_back_to_ten(self):
        self.assertEqual(
            detection_stride("/nonexistent/config_scannet200.yaml"), 10)

    def test_reads_frequency_from_config(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "cfg.yaml")
            with open(path, "w") as f:
                f.write("openyolo3d:\n  frequency: 5\n")
            self.assertEqual(detection_stride(path), 5)


class DrawTests(unittest.TestCase):
    def test_draws_boxes_and_preserves_shape(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        out = draw_detections(rgb, [([10, 10, 30, 30], "chair", 0.9, (255, 0, 0))])
        self.assertEqual(out.shape, (48, 64, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(int(out.sum()), 0)
        # area far from the box stays black
        self.assertEqual(int(out[45, 60].sum()), 0)

    def test_empty_detections_leave_image_untouched(self):
        rgb = np.full((16, 16, 3), 7, dtype=np.uint8)
        np.testing.assert_array_equal(draw_detections(rgb, []), rgb)

    def test_scale_maps_fullres_coords_onto_thumbnail(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        out = draw_detections(
            rgb, [([0, 0, 64, 48], "chair", 0.9, (255, 0, 0))], scale=0.5)
        self.assertEqual(out.shape, (48, 64, 3))
        # box corner lands at the scaled position, far corner stays black
        self.assertGreater(int(out[0, 0].sum()), 0)
        self.assertEqual(int(out[47, 63].sum()), 0)

    def test_draw_masks_tints_and_outlines(self):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:8, 4:8] = True
        out = draw_masks(rgb, [(mask, "chair", 0.8, (255, 0, 0))])
        self.assertEqual(out.shape, (16, 16, 3))
        self.assertGreater(int(out[5, 5, 0]), 0)
        self.assertEqual(int(out[0, 0].sum()), 0)

    def test_overlay_font_grows_with_image_height(self):
        _small_font, small_px = _label_font(120)
        _big_font, big_px = _label_font(498)
        self.assertEqual(small_px, 18)
        self.assertEqual(big_px, 31)
        self.assertGreater(big_px, small_px)
        # bigger text paints strictly more label pixels on the same box
        rgb = np.zeros((498, 664, 3), dtype=np.uint8)
        det = [([100, 200, 300, 350], "chair", 0.9, (255, 0, 0))]
        out = draw_detections(rgb, det)
        self.assertGreater(int(out.sum()), 0)
        row = out[200 - big_px - 4:200, 100:320]
        self.assertGreater(int(row.sum()), 0)


class ClientProtocolTests(unittest.TestCase):
    _STUB = (
        "import json,sys\n"
        "while True:\n"
        "    line = sys.stdin.readline()\n"
        "    if not line:\n"
        "        break\n"
        "    req = json.loads(line)\n"
        "    if req.get('op') == 'ping':\n"
        "        sys.stdout.write(json.dumps({'id': req.get('id'), 'ok': True}) + chr(10))\n"
        "    else:\n"
        "        sys.stdout.write(json.dumps({'id': req.get('id'),\n"
        "            'boxes': [[1, 2, 3, 4]], 'labels': [0], 'scores': [0.5]}) + chr(10))\n"
        "    sys.stdout.flush()\n"
    )

    def test_submit_poll_close_against_stub(self):
        import subprocess
        with tempfile.TemporaryDirectory() as root:
            stub = os.path.join(root, "stub_worker.py")
            with open(stub, "w") as f:
                f.write(self._STUB)
            client = DetectClient()
            client.classes_file = None
            client.proc = subprocess.Popen(
                [sys.executable, stub], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1)
            try:
                self.assertTrue(client.alive())
                req_id = client.submit("/tmp/frame.jpg")
                resp = None
                for _ in range(200):
                    resp = client.poll()
                    if resp is not None:
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(resp)
                self.assertEqual(resp["id"], req_id)
                self.assertEqual(resp["boxes"], [[1, 2, 3, 4]])
            finally:
                client.close()
            self.assertFalse(client.alive())
            self.assertIsNone(client.classes_file)

    def test_partial_line_never_blocks(self):
        import subprocess
        client = DetectClient()
        client.proc = subprocess.Popen(
            [sys.executable, "-c",
             "import sys,time; sys.stdout.buffer.write(b'{\"id\": 1'); "
             "sys.stdout.buffer.flush(); time.sleep(30)"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        try:
            t0 = time.monotonic()
            for _ in range(20):
                self.assertIsNone(client.poll())
            self.assertLess(time.monotonic() - t0, 5.0)
        finally:
            client.close()

    def test_generic_request_round_trip(self):
        import subprocess
        with tempfile.TemporaryDirectory() as root:
            stub = os.path.join(root, "stub_worker.py")
            with open(stub, "w") as f:
                f.write(
                    "import json,sys\n"
                    "while True:\n"
                    "    line = sys.stdin.readline()\n"
                    "    if not line:\n"
                    "        break\n"
                    "    req = json.loads(line)\n"
                    "    sys.stdout.write(json.dumps({'id': req.get('id'),\n"
                    "        'op': req.get('op'), 'ok': True}) + chr(10))\n"
                    "    sys.stdout.flush()\n")
            client = DetectClient()
            client.proc = subprocess.Popen(
                [sys.executable, stub], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            try:
                req_id = client.request("frames", {"extra": 1})
                resp = None
                for _ in range(200):
                    resp = client.poll()
                    if resp is not None:
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(resp)
                self.assertEqual((resp["id"], resp["op"]), (req_id, "frames"))
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
