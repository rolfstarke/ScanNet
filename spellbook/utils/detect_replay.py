"""Client for the warm OpenYOLO3D 2D detection worker.

Mirrors the QueryClient protocol (JSON lines over a subprocess pipe) but
spawns the OpenYOLO3D interpreter with its CUDA shim. Interactive use only:
physical GPU 0, no lease — the documented visualizer exception.
"""
import json
import os
import select
import subprocess
import tempfile

# Source of truth: predict.runner MODEL_PYTHON / OPENYOLO3D_CUDA_HOME.
OPENYOLO3D_PYTHON = "/data/openyolo3D/conda/envs/openyolo3d/bin/python"
OPENYOLO3D_CUDA_LIB = "/data/openyolo3D/cuda-11.3/lib64"
OPENYOLO3D_CONFIG = "/home/rolf/GIT/OpenYOLO3D/pretrained/config_scannet200.yaml"
_WORKER = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "predict", "models",
    "_openyolo3d_detect_worker.py"))
DETECT_FPS = 3.0


def detection_stride(config_path=OPENYOLO3D_CONFIG):
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return max(1, int(cfg.get("openyolo3d", {}).get("frequency", 10)))
    except (OSError, ValueError, TypeError):
        return 10


def _label_font(height_px):
    """Readable overlay font scaled to the image height.

    Pillow's bitmap default is ~11 px everywhere; on a ~500 px thumbnail
    that is illegible. Scale generously and fall back gracefully headless.
    """
    from PIL import ImageFont
    size = max(18, int(round(max(1, height_px) / 16)))
    try:
        return ImageFont.load_default(size=size), size
    except Exception:
        pass
    for family in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(family, size), size
        except Exception:
            continue
    return ImageFont.load_default(), 11


def draw_detections(rgb, detections, scale=1.0):
    """Draw boxes + labels on an RGB uint8 image. Pure function.

    detections: list of (xyxy, name, score, (r, g, b)) with 0-255 colors.
    scale: multiply coordinates by this (detector full-res coords onto a
    downscaled thumbnail).
    """
    import numpy as np
    from PIL import Image, ImageDraw
    out = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)
    font, px = _label_font(out.height)
    width = 3 if out.height >= 400 else 2
    for xyxy, name, score, color in detections:
        x1, y1, x2, y2 = (int(round(v * scale)) for v in xyxy)
        draw.rectangle([x1, y1, x2, y2], outline=tuple(int(v) for v in color), width=width)
        draw.text((x1 + 3, max(0, y1 - px - 4)), f"{name} {float(score):.2f}",
                  fill=tuple(int(v) for v in color), font=font)
    return np.asarray(out, dtype=np.uint8)


def draw_masks(rgb, masks, scale=1.0, alpha=0.35):
    """Overlay full-resolution bool masks (with outlines + labels). Pure function.

    masks: list of (mask_bool_HxW, name, score, (r, g, b)).
    """
    import numpy as np
    from PIL import Image, ImageDraw
    base = np.ascontiguousarray(rgb, dtype=np.uint8)
    out = Image.fromarray(base, mode="RGB")
    if scale != 1.0:
        out = out.resize(
            (max(1, int(round(out.width * scale))),
             max(1, int(round(out.height * scale)))), Image.BILINEAR)
    canvas = np.asarray(out).astype(np.float32)
    small_masks = []
    for mask, _name, _score, color in masks:
        m = np.asarray(mask, dtype=bool)
        im = Image.fromarray(m)
        im = im.resize((canvas.shape[1], canvas.shape[0]), Image.NEAREST)
        small_masks.append((np.asarray(im, dtype=bool), _name, _score, color))
    for m, _name, _score, color in small_masks:
        if not m.any():
            continue
        tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        canvas[m] = (1.0 - alpha) * canvas[m] + alpha * tint
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)
    font, px = _label_font(out.height)
    for m, name, score, color in small_masks:
        if not m.any():
            continue
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0)
                     & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        ys, xs = np.nonzero(edge)
        rgb = tuple(int(v) for v in color)
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.point((x, y), fill=rgb)
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                       outline=rgb, width=2)
        draw.text((int(xs.min()) + 3, max(0, int(ys.min()) - px - 4)),
                  f"{name} {float(score):.2f}", fill=rgb, font=font)
    return np.asarray(out, dtype=np.uint8)


class DetectClient:
    """JSON-lines worker client. stdout is read through a private byte buffer,
    so a partial line can never block the UI thread."""

    def __init__(self):
        self.proc = None
        self.classes_file = None
        self.key = None
        self._next_id = 0
        self._buf = bytearray()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, classes, key=None, config_path=OPENYOLO3D_CONFIG,
               python=OPENYOLO3D_PYTHON, cuda_lib=OPENYOLO3D_CUDA_LIB,
               script=_WORKER, extra_args=(), env_extra=None,
               classes_prefix="openyolo3d-detect-classes-"):
        if self.alive() and self.key == key:
            return
        self.close()
        fd, path = tempfile.mkstemp(prefix=classes_prefix, suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(classes) + "\n")
        self.classes_file = path
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env.pop("SPELLBOOK_GPU_LEASE_FD", None)
        if cuda_lib:
            lib = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = cuda_lib + (":" + lib if lib else "")
        env.update(env_extra or {})
        # Binary unbuffered stdout: os.read in poll() must see every byte.
        # stdin stays buffered; each request is flushed explicitly.
        self.proc = subprocess.Popen(
            [python, script, "--config", config_path,
             "--classes-file", path, *extra_args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, bufsize=0)
        self.key = key

    def request(self, op, payload=None):
        if not self.alive():
            raise RuntimeError("detect worker is not running")
        self._next_id += 1
        msg = {"id": self._next_id, "op": op}
        msg.update(payload or {})
        data = (json.dumps(msg) + "\n").encode("utf-8")
        view = memoryview(data)
        stdin_fd = self.proc.stdin.fileno()
        while view:
            view = view[os.write(stdin_fd, view):]
        self.proc.stdin.flush()
        return self._next_id

    def submit(self, image_path):
        return self.request("detect", {"image": image_path})

    def poll(self):
        if not self.alive():
            self._buf.clear()
            return {"id": None, "ok": False, "error": "worker exited"}
        try:
            ready, _, _ = select.select([self.proc.stdout], [], [], 0)
        except (OSError, ValueError):
            return {"id": None, "ok": False, "error": "worker exited"}
        if ready:
            try:
                chunk = os.read(self.proc.stdout.fileno(), 65536)
            except OSError:
                chunk = b""
            if not chunk:
                return {"id": None, "ok": False, "error": "worker exited"}
            self._buf.extend(chunk)
        nl = self._buf.find(b"\n")
        if nl < 0:
            return None
        line = bytes(self._buf[:nl]).decode("utf-8", "replace")
        del self._buf[:nl + 1]
        try:
            return json.loads(line)
        except ValueError:
            return None

    def close(self):
        proc, self.proc = self.proc, None
        self.key = None
        self._buf.clear()
        if proc is not None:
            stdin = getattr(proc, "stdin", None)
            if stdin is not None:
                try:
                    stdin.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass
            stdout = getattr(proc, "stdout", None)
            if stdout is not None:
                try:
                    stdout.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self.classes_file is not None:
            try:
                os.unlink(self.classes_file)
            except OSError:
                pass
            self.classes_file = None
