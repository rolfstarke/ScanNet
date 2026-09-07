"""Per-method 2D-stage replay adapters for the visualizer.

Replay shows what a method's own 2D stage saw, frame by frame. Each adapter
answers two questions and owns all of its state (workers, caches, pending
requests):

    frames() -> ordered list of opaque step keys
    fetch(key) -> {"image", "boxes", "masks", "status"} (image is full-res RGB)
    ready(key) -> True once the overlay for key is final
    poll()     -> True if background work completed (call every tick)
    close()    -> release workers / temp files

The viewer resizes, rescales overlays into thumbnail space, and draws.
`build()` constructs the adapter for the selected prediction; `available()`
is the cheap, side-effect-free gate for the Camera menu.
"""
import base64
import io
import os
import time

from utils.camera_replay import load_color_rgb
from utils.detect_replay import (
    DETECT_FPS, OPENYOLO3D_CONFIG, DetectClient, detection_stride,
)

OPEN3DIS_REPO = "/home/rolf/GIT/Open3DIS"
OPEN3DIS_PYTHON = "/data/open3dis/conda/envs/open3dis/bin/python"
OPEN3DIS_CONFIG = "/home/rolf/GIT/Open3DIS/configs/ov3dis_scene4.yaml"
_OPEN3DIS_WORKER = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "predict", "models",
    "_open3dis_replay_worker.py"))

MOSAIC3D_NO_REPLAY = "mosaic3d has no 2D stage — 3D encoder and text only"
STAGE_C_PENDING = ("{model} 2D replay is staged after the openyolo3d/open3dis "
                   "adapters (Stage C)")


class Replay2D:
    caption = ""
    fps = DETECT_FPS

    def frames(self):
        raise NotImplementedError

    def fetch(self, key):
        raise NotImplementedError

    def ready(self, key):
        raise NotImplementedError

    def poll(self):
        return False

    def invalidate_colors(self):
        """Drop color-dependent state after a recolor (default: nothing).

        Adapters whose worker bakes colors into returned images must
        override this to force re-requests; adapters caching raw boxes
        keep them since colors apply at composite time.
        """

    def close(self):
        pass


def available(pred, sequence):
    """Side-effect-free gate for the Camera menu. Returns (ok, reason)."""
    model = (pred or {}).get("model")
    if sequence is None or not sequence.get("rgb_ids"):
        return False, "no frames extracted for this scene"
    if model == "openyolo3d":
        return True, ""
    if model == "open3dis":
        return True, ""
    if model == "mosaic3d":
        return False, MOSAIC3D_NO_REPLAY
    if model in ("openins3d", "openmask3d"):
        return False, STAGE_C_PENDING.format(model=model)
    return False, f"{model or 'no prediction'} has no 2D replay"


def build(pred, *, sequence, spec, scannet_root, color_of):
    """Construct the adapter for the selected prediction. Raises RuntimeError
    with a human reason instead of returning None."""
    model = (pred or {}).get("model")
    if model == "openyolo3d":
        return OpenYoloReplay(pred, sequence=sequence, spec=spec)
    if model == "open3dis":
        return Open3DisReplay(pred, sequence=sequence, spec=spec,
                              scannet_root=scannet_root, color_of=color_of)
    ok, reason = available(pred, sequence)
    raise RuntimeError(reason or f"no 2D replay for {model}")


def yolo_frame_ids(sequence, stride):
    return list(sequence["rgb_ids"])[::max(1, stride)]


class OpenYoloReplay(Replay2D):
    caption = "YOLO-World · every {stride}th frame"
    fps = DETECT_FPS
    PREFETCH = 4

    def __init__(self, pred, *, sequence, spec):
        self._names = list(spec.class_labels)
        self._stride = detection_stride(OPENYOLO3D_CONFIG)
        self._ids = yolo_frame_ids(sequence, self._stride)
        self._dir = sequence["dir"]
        self.caption = f"YOLO-World · every {self._stride}th frame"
        self._client = DetectClient()
        self._client.ensure(
            self._names,
            key=(pred["run_id"], pred["model"], pred["label_set"]))
        self._pending = {}
        self._cache = {}
        self._t0 = time.monotonic()
        self._ever_answered = False

    def frames(self):
        return list(self._ids)

    def _is_requested(self, fid):
        return fid in self._cache or fid in self._pending.values()

    def _submit(self, fid):
        if self._is_requested(fid):
            return
        path = os.path.join(self._dir, "color", f"{fid}.jpg")
        req_id = self._client.request("detect", {"image": path})
        self._pending[req_id] = fid

    def poll(self):
        client = self._client
        if client is None:
            return False
        if not client.alive():
            if self._pending:
                self._pending = {}
                return True
            return False
        changed = False
        while True:
            try:
                msg = client.poll()
            except Exception:
                break
            if msg is None:
                break
            changed = True
            if msg.get("boxes") is None:
                self._pending = {r: f for r, f in self._pending.items()
                                 if r != msg.get("id")}
                continue
            fid = self._pending.pop(msg.get("id"), None)
            if fid is None:
                continue
            self._ever_answered = True
            self._cache[fid] = msg
        return changed

    def fetch(self, fid):
        self.poll()
        if not self._is_requested(fid):
            try:
                self._submit(fid)
                ids = self._ids
                if fid in ids:
                    pos = ids.index(fid)
                    for nxt in ids[pos + 1:pos + 1 + self.PREFETCH]:
                        if not self._is_requested(nxt):
                            self._submit(nxt)
            except Exception as exc:
                return {"image": self._image(fid), "boxes": [], "masks": [],
                        "status": f"2D detect submit failed: {exc}"}
        entry = self._cache.get(fid)
        if entry is None:
            if not self._ever_answered:
                wait = time.monotonic() - self._t0
                return {"image": self._image(fid), "boxes": [], "masks": [],
                        "status": f"loading YOLO-World ({wait:.0f}s)…"}
            return {"image": self._image(fid), "boxes": [], "masks": [],
                    "status": f"Frame {fid}: detecting…"}
        names = self._names
        dets = []
        for box, label, score in zip(entry.get("boxes") or [],
                                     entry.get("labels") or [],
                                     entry.get("scores") or []):
            idx = int(label)
            if not 0 <= idx < len(names):
                continue
            dets.append((box, names[idx], float(score)))
        return {"image": self._image(fid), "boxes": dets, "masks": [],
                "status": (f"2D model frame {fid} · {len(dets)} boxes · "
                           f"{float(entry.get('latency_ms') or 0):.0f} ms")}

    def _image(self, fid):
        try:
            return load_color_rgb(os.path.join(self._dir, "color", f"{fid}.jpg"))
        except Exception:
            import numpy as np
            return np.zeros((8, 8, 3), dtype=np.uint8)

    def ready(self, fid):
        return fid in self._cache

    def close(self):
        client, self._client = self._client, None
        self._pending = {}
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def open3dis_exp_pth(run_id, scene_id):
    return os.path.join(
        OPEN3DIS_REPO, "exp", f"{run_id}_{scene_id}_ov3discomp",
        "maskGdino", f"{scene_id}.pth")


class Open3DisReplay(Replay2D):
    caption = "Grounded-SAM cached · every {stride}th frame"
    fps = 10.0

    def __init__(self, pred, *, sequence, spec, scannet_root, color_of):
        import yaml
        self._dir = sequence["dir"]
        self._scene = os.path.basename(os.path.normpath(self._dir))
        parent = os.path.basename(os.path.dirname(os.path.normpath(self._dir)))
        if parent.startswith("scene"):
            self._scene = parent
        self._names = list(spec.class_labels)
        self._color_of = color_of
        self._stride = 15
        try:
            with open(OPEN3DIS_CONFIG) as f:
                cfg = yaml.safe_load(f) or {}
            self._stride = max(1, int(
                cfg.get("data", {}).get("img_interval", 15)))
        except (OSError, ValueError, TypeError):
            pass
        self.caption = f"Grounded-SAM cached · every {self._stride}th frame"
        pth = open3dis_exp_pth(pred["run_id"], self._scene)
        if not os.path.isfile(pth):
            raise RuntimeError(
                f"2D output not retained for this run (missing {pth})")
        self._client = DetectClient()
        self._client.ensure(
            self._names, key=(pred["run_id"], pred["model"], pred["label_set"],
                              self._scene),
            config_path=OPEN3DIS_CONFIG,
            python=OPEN3DIS_PYTHON, cuda_lib=None,
            script=_OPEN3DIS_WORKER,
            extra_args=("--exp-pth", pth),
            env_extra={"PYTHONPATH": OPEN3DIS_REPO,
                       "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"},
            classes_prefix="open3dis-replay-classes-")
        self._ids = []
        self._pending_frames = False
        self._requested = set()
        self._cache = {}
        try:
            self._client.request("frames", {})
            self._pending_frames = True
        except Exception:
            pass

    def frames(self):
        return list(self._ids)

    def poll(self):
        client = self._client
        if client is None:
            return False
        if not client.alive():
            if self._requested:
                self._requested = set()
                return True
            return False
        changed = False
        while True:
            try:
                msg = client.poll()
            except Exception:
                break
            if msg is None:
                break
            changed = True
            if msg.get("op") == "frames":
                self._ids = [int(v) for v in msg.get("ids", [])]
            elif msg.get("png_b64") is not None and msg.get("fid") is not None:
                try:
                    raw = base64.b64decode(msg["png_b64"])
                    import numpy as np
                    from PIL import Image
                    img = np.asarray(
                        Image.open(io.BytesIO(raw)).convert("RGB"),
                        dtype=np.uint8)
                except Exception:
                    continue
                fid = int(msg["fid"])
                self._requested.discard(fid)
                self._cache[fid] = {
                    "image": img, "n": msg.get("n", 0),
                    "latency_ms": msg.get("latency_ms", 0)}
        return changed

    def fetch(self, fid):
        self.poll()
        if not self._ids and not self._pending_frames:
            try:
                self._client.request("frames", {})
                self._pending_frames = True
            except Exception as exc:
                return {"image": self._blank(), "boxes": [], "masks": [],
                        "status": f"2D replay unavailable: {exc}"}
            return {"image": self._blank(), "boxes": [], "masks": [],
                    "status": "loading cached detections…"}
        entry = self._cache.get(fid)
        if entry is None:
            if fid not in self._requested:
                try:
                    colors = {name: [int(v) for v in self._color_of(name)]
                              for name in self._names}
                    self._client.request(
                        "frame", {"fid": int(fid),
                                  "image": os.path.join(
                                      self._dir, "color", f"{fid}.jpg"),
                                  "colors": colors})
                    self._requested.add(fid)
                except Exception as exc:
                    return {"image": self._blank(), "boxes": [], "masks": [],
                            "status": f"2D replay request failed: {exc}"}
            return {"image": self._blank(), "boxes": [], "masks": [],
                    "status": f"Frame {fid}: decoding…"}
        return {"image": entry["image"], "boxes": [], "masks": [],
                "status": (f"Grounded-SAM frame {fid} · {entry['n']} masks · "
                           f"{float(entry.get('latency_ms') or 0):.0f} ms")}

    def _blank(self):
        import numpy as np
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def ready(self, fid):
        return fid in self._cache

    def invalidate_colors(self):
        # The worker bakes class colors into the decoded PNGs, so both the
        # images and the in-flight requests are stale after a recolor.
        self._cache = {}
        self._requested = set()

    def close(self):
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        self._requested = set()
