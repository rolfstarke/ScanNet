import os
import sys
import unittest
from types import SimpleNamespace

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from utils import replay2d  # noqa: E402
from utils.replay2d import (  # noqa: E402
    MOSAIC3D_NO_REPLAY, Open3DisReplay, OpenYoloReplay, available,
    open3dis_exp_pth, yolo_frame_ids,
)


def _seq(*ids):
    return {"dir": "/tmp/frames", "rgb_ids": list(ids)}


class AvailableTests(unittest.TestCase):
    def test_needs_frames(self):
        ok, reason = available({"model": "openyolo3d"}, None)
        self.assertFalse(ok)
        self.assertIn("no frames", reason)

    def test_yolo_and_open3dis_available(self):
        for model in ("openyolo3d", "open3dis"):
            ok, reason = available({"model": model}, _seq(0, 1, 2))
            self.assertTrue(ok, model)
            self.assertEqual(reason, "")

    def test_mosaic3d_disabled_with_reason(self):
        ok, reason = available({"model": "mosaic3d"}, _seq(0, 1))
        self.assertFalse(ok)
        self.assertEqual(reason, MOSAIC3D_NO_REPLAY)

    def test_unknown_model_disabled(self):
        ok, reason = available({"model": "wat"}, _seq(0, 1))
        self.assertFalse(ok)
        self.assertIn("wat", reason)

    def test_stage_c_methods_deferred(self):
        for model in ("openins3d", "openmask3d"):
            ok, reason = available({"model": model}, _seq(0, 1))
            self.assertFalse(ok, model)
            self.assertIn("Stage C", reason)


class BuildTests(unittest.TestCase):
    def test_build_rejects_with_reason(self):
        with self.assertRaises(RuntimeError):
            replay2d.build({"model": "mosaic3d", "run_id": "r",
                            "label_set": "ScanNet200"},
                           sequence=_seq(0), spec=None, scannet_root="/tmp",
                           color_of=lambda n: (0, 0, 0))

    def test_open3dis_missing_cache_raises_helpful_error(self):
        spec = SimpleNamespace(class_labels=["chair"])
        pred = {"model": "open3dis", "run_id": "no-such-run",
                "label_set": "ScanNet200"}
        seq = {"dir": "/tmp/frames", "rgb_ids": [0, 15]}
        with self.assertRaises(RuntimeError) as ctx:
            Open3DisReplay(pred, sequence=seq, spec=spec,
                           scannet_root="/tmp", color_of=lambda n: (0, 0, 0))
        self.assertIn("not retained", str(ctx.exception))
        self.assertIn("no-such-run", str(ctx.exception))


class FrameIdTests(unittest.TestCase):
    def test_yolo_stride_ids(self):
        self.assertEqual(yolo_frame_ids(_seq(*range(25)), 10), [0, 10, 20])
        self.assertEqual(yolo_frame_ids(_seq(7), 10), [7])

    def test_exp_path_layout(self):
        path = open3dis_exp_pth("vis-replay-open3dis", "scene0618_00")
        self.assertTrue(path.endswith(
            "vis-replay-open3dis_scene0618_00_ov3discomp/maskGdino/scene0618_00.pth"))


class ContractTests(unittest.TestCase):
    def test_stub_adapter_shapes(self):
        class Stub(replay2d.Replay2D):
            def frames(self):
                return [0, 15]

            def fetch(self, key):
                import numpy as np
                return {"image": np.zeros((8, 8, 3), dtype=np.uint8),
                        "boxes": [([0, 0, 4, 4], "chair", 0.5, (255, 0, 0))],
                        "masks": [], "status": "ok"}

            def ready(self, key):
                return True

        stub = Stub()
        self.assertEqual(stub.frames(), [0, 15])
        entry = stub.fetch(15)
        self.assertEqual(entry["image"].shape, (8, 8, 3))
        self.assertEqual(len(entry["boxes"]), 1)
        self.assertTrue(stub.ready(0))
        stub.close()


class IdempotencyTests(unittest.TestCase):
    class _FakeClient:
        def __init__(self):
            self.requests = []
            self._next = 0

        def request(self, op, payload=None):
            self._next += 1
            self.requests.append((op, payload))
            return self._next

        def poll(self):
            return None

        def alive(self):
            return True

    def _yolo(self):
        adapter = OpenYoloReplay.__new__(OpenYoloReplay)
        adapter._names = ["chair"]
        adapter._ids = [0, 10, 20]
        adapter._dir = "/tmp/frames"
        adapter._pending = {}
        adapter._cache = {}
        adapter._client = self._FakeClient()
        adapter._t0 = 0.0
        adapter._ever_answered = False
        return adapter

    def test_yolo_resubmit_never_duplicates_detects(self):
        adapter = self._yolo()
        adapter.fetch(0)
        adapter.fetch(0)
        adapter.fetch(10)
        zeros = [req for op, req in adapter._client.requests
                 if op == "detect" and (req or {}).get("image", "").endswith("/0.jpg")]
        self.assertEqual(len(zeros), 1)

    def test_yolo_recolor_keeps_raw_boxes(self):
        adapter = self._yolo()
        adapter._cache[0] = {"boxes": [], "labels": [], "scores": []}
        adapter.invalidate_colors()
        self.assertIn(0, adapter._cache)

    def _open3dis(self):
        adapter = Open3DisReplay.__new__(Open3DisReplay)
        adapter._dir = "/tmp/frames"
        adapter._names = ["chair"]
        adapter._color_of = lambda n: (255, 0, 0)
        adapter._ids = [5]
        adapter._pending_frames = True
        adapter._requested = set()
        adapter._cache = {}
        adapter._client = self._FakeClient()
        return adapter

    def test_open3dis_resubmit_never_duplicates_frames(self):
        adapter = self._open3dis()
        adapter.fetch(5)
        adapter.fetch(5)
        frames = [req for op, req in adapter._client.requests if op == "frame"]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["fid"], 5)

    def test_open3dis_recolor_drops_baked_images(self):
        adapter = self._open3dis()
        import numpy as np
        adapter._cache[5] = {"image": np.zeros((8, 8, 3), dtype=np.uint8)}
        adapter._requested.add(5)
        adapter.invalidate_colors()
        self.assertEqual(adapter._cache, {})
        self.assertEqual(adapter._requested, set())


if __name__ == "__main__":
    unittest.main()
