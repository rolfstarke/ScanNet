"""Warm YOLO-World 2D detection worker for live replay debugging.

Runs under the OpenYOLO3D conda interpreter. Loads ONLY the 2D detector
(Network_2D) — never the 3D proposal network — and answers single-image
detection requests over JSON lines, mirroring the query worker protocol:

    request  {"id": N, "op": "detect", "image": "/path/to/frame.jpg"}
    response {"id": N, "boxes": [[x1,y1,x2,y2], ...], "labels": [...],
              "scores": [...], "latency_ms": ...}
    request  {"id": N, "op": "ping"}  ->  {"id": N, "ok": true}

Errors are reported per request: {"id": N, "ok": false, "error": "..."}.
The detector applies its own NMS / score threshold / top-k / fullscreen-box
rejection, so output is exactly the pipeline's 2D-stage output for the frame.

Usage:
    python _openyolo3d_detect_worker.py --config <template.yaml> \
        --classes-file <one class per line>
    python _openyolo3d_detect_worker.py --config ... --classes-file ... \
        --self-test <image> <n>
"""
import argparse
import json
import os
import sys
import time

OPENYOLO3D_REPO = "/home/rolf/GIT/OpenYOLO3D"

_PROTO_OUT = sys.stdout


class _LogSink:
    """Divert third-party prints/logs (mmengine dumps its config to stdout)
    to stderr so the stdout protocol stream stays pure JSON lines."""

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


def _load_detector(config_path, classes):
    os.chdir(OPENYOLO3D_REPO)
    sys.path.insert(0, OPENYOLO3D_REPO)
    import yaml  # noqa: E402
    from utils.utils_2d import Network_2D  # noqa: E402
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["network2d"]["text_prompts"] = list(classes)
    net = Network_2D(config)
    return net, list(classes)


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    try:
        return list(value)
    except TypeError:
        return []


def _detect(net, texts, image_path):
    t0 = time.monotonic()
    out = net.get_bounding_boxes([image_path], text=texts) or {}
    pred = out.get(0)
    if pred is None and out:
        pred = next(iter(out.values()))
    pred = pred or {}
    boxes = _as_list(pred.get("bbox"))
    labels = _as_list(pred.get("labels"))
    scores = _as_list(pred.get("scores"))
    return {
        "boxes": [[float(v) for v in b] for b in boxes],
        "labels": [int(v) for v in labels],
        "scores": [float(v) for v in scores],
        "latency_ms": round((time.monotonic() - t0) * 1000.0, 1),
    }


def _serve(net, texts):
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
            if req.get("op") == "ping":
                resp = {"id": rid, "ok": True}
            elif req.get("op") == "detect":
                resp = {"id": rid}
                resp.update(_detect(net, texts, req["image"]))
            else:
                resp = {"id": rid, "ok": False, "error": "unknown op"}
        except Exception as exc:  # per-request failure must not kill the worker
            resp = {"id": rid, "ok": False, "error": str(exc)}
        _PROTO_OUT.write(json.dumps(resp) + "\n")
        _PROTO_OUT.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--classes-file", required=True)
    ap.add_argument("--self-test", default=None,
                    help="image path; run N detections and print timings")
    ap.add_argument("count", nargs="?", default="5")
    args = ap.parse_args()
    with open(args.classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError("empty classes file")
    sys.stdout = _LogSink()
    t0 = time.monotonic()
    net, texts = _load_detector(args.config, classes)
    print(f"[detect-worker] init {time.monotonic() - t0:.1f}s, "
          f"{len(texts)} prompts", flush=True)
    if args.self_test:
        n = max(1, int(args.count))
        for i in range(n):
            res = _detect(net, texts, args.self_test)
            print(f"[detect-worker] frame {i}: {len(res['boxes'])} boxes, "
                  f"{res['latency_ms']} ms", flush=True)
        return 0
    return _serve(net, texts)


if __name__ == "__main__":
    sys.exit(main())
