"""Cached Grounded-SAM replay worker for the visualizer's 2D replay panel.

Runs under the Open3DIS conda interpreter. Loads one scene's cached 2D output
(maskGdino/<scene>.pth: per-frame RLE masks + confidence) plus CLIP text
embeddings once, then answers per-frame overlay requests over JSON lines:

    request  {"id": N, "op": "frames"} -> {"id": N, "op": "frames", "ids": [...]}
    request  {"id": N, "op": "frame", "fid": F, "image": path, "colors": {...}}
    response {"id": N, "fid": F, "png_b64": "...", "n": <masks>, "latency_ms": ms}

The overlay (mask outlines + boxes + `name conf` labels) is drawn server-side
because RLE decoding (pycocotools) and the CLIP text encoder only exist here.
Mask names are argmax cosine(img_feat, text) with the wrapper's convention
(`clip.tokenize` over raw class names, L2-normalised).

Usage:
    python _open3dis_replay_worker.py --config <ov3dis_scene4.yaml>
        --classes-file <txt> --exp-pth <maskGdino/scene.pth>
"""
import argparse
import base64
import io
import json
import os
import sys
import time

OPEN3DIS_REPO = "/home/rolf/GIT/Open3DIS"

_PROTO_OUT = sys.stdout


class _LogSink:
    """Keep the stdout protocol stream pure JSON lines."""

    def write(self, text):
        try:
            sys.stderr.write(text)
        except (OSError, ValueError):
            pass

    def flush(self):
        try:
            sys.stderr.flush()
        except (OSError, ValueError):
            pass


def _load_scene(exp_pth, classes, config_path):
    os.chdir(OPEN3DIS_REPO)
    sys.path.insert(0, OPEN3DIS_REPO)
    import torch
    import clip
    import yaml
    clip_model = "ViT-L/14@336px"
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        clip_model = cfg.get("foundation_model", {}).get("clip_model", clip_model)
    except (OSError, ValueError):
        pass
    data = torch.load(exp_pth, map_location="cpu")
    model, _ = clip.load(clip_model, device="cuda")
    with torch.no_grad():
        text = model.encode_text(clip.tokenize(classes).cuda())
        text = (text / text.norm(dim=-1, keepdim=True)).float().cpu().numpy()
    return data, text


def _frame_overlay(data, text, names, fid, image_path, colors):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pycocotools import mask as mask_util
    t0 = time.monotonic()
    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    entry = data.get(str(fid), data.get(int(fid)))
    dets = []
    if entry is not None:
        feats = entry["img_feat"]
        feats = feats.detach().cpu().float().numpy() \
            if hasattr(feats, "detach") else np.asarray(feats, dtype=np.float32)
        conf = entry["conf"]
        conf = conf.detach().cpu().float().numpy() \
            if hasattr(conf, "detach") else np.asarray(conf, dtype=np.float32)
        order = np.argsort(-conf)
        for i in order:
            rle = entry["masks"][int(i)]
            if isinstance(rle, dict) and isinstance(rle.get("counts"), str):
                rle = {"size": list(rle["size"]),
                       "counts": rle["counts"].encode("utf-8")}
            m = mask_util.decode(rle).astype(bool)
            sims = feats[int(i)] @ text.T
            label = int(np.argmax(sims))
            name = names[label] if 0 <= label < len(names) else str(label)
            dets.append((m, name, float(conf[int(i)])))
    out = Image.fromarray(rgb)
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for m, name, score in dets:
        col = tuple(int(v) for v in colors.get(name, (255, 255, 255)))
        if m.shape != rgb.shape[:2]:
            canvas = Image.fromarray(m)
            canvas = canvas.resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
            m = np.asarray(canvas, dtype=bool)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0)
                     & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        ey, ex = np.nonzero(edge)
        for x, y in zip(ex.tolist(), ey.tolist()):
            draw.point((x, y), fill=col)
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                       outline=col, width=1)
        draw.text((int(xs.min()) + 2, max(0, int(ys.min()) - 11)),
                  f"{name} {score:.2f}", fill=col, font=font)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {"png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "n": len(dets),
            "latency_ms": round((time.monotonic() - t0) * 1000.0, 1)}


def _serve(data, text, names):
    stdin = sys.stdin
    while True:
        line = stdin.readline()
        if not line:
            return 0
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid = req.get("id")
        try:
            if req.get("op") == "frames":
                ids = sorted(data.keys(), key=lambda v: int(v))
                resp = {"id": rid, "op": "frames",
                        "ids": [int(v) for v in ids]}
            elif req.get("op") == "frame":
                resp = {"id": rid}
                resp.update(_frame_overlay(
                    data, text, names, req.get("fid"), req.get("image"),
                    req.get("colors") or {}))
                resp["fid"] = req.get("fid")
            else:
                resp = {"id": rid, "ok": False, "error": "unknown op"}
        except Exception as exc:
            resp = {"id": rid, "ok": False, "error": str(exc)}
        _PROTO_OUT.write(json.dumps(resp) + "\n")
        _PROTO_OUT.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--classes-file", required=True)
    ap.add_argument("--exp-pth", required=True)
    args = ap.parse_args()
    with open(args.classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError("empty classes file")
    sys.stdout = _LogSink()
    t0 = time.monotonic()
    data, text = _load_scene(args.exp_pth, classes, args.config)
    print(f"[open3dis-replay] {len(data)} frames, {len(classes)} classes, "
          f"init {time.monotonic() - t0:.1f}s", flush=True)
    return _serve(data, text, classes)


if __name__ == "__main__":
    sys.exit(main())
