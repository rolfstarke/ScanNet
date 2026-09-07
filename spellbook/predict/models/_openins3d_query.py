"""Native OpenIns3D Lookup-only query over cached Snap images and immutable final masks."""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import add_run_args, load_overrides  # noqa: E402


OPENINS3D_REPO = "/home/rolf/GIT/OpenIns3D"
IMAGE_SIZE = [800, 800]
ADJUST_CAMERA = [2, 0.1, 1.0]
LOOKUP_THRESHOLD = 0.3


def _load_final_masks(submission_root, scene_id):
    import importlib.util
    spellbook_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    benchmark_dir = os.path.join(os.path.dirname(spellbook_dir), "BenchmarkScripts")
    spec = importlib.util.spec_from_file_location(
        "spellbook_util_3d", os.path.join(benchmark_dir, "util_3d.py"))
    util_3d = importlib.util.module_from_spec(spec)
    sys.path.insert(0, benchmark_dir)
    try:
        spec.loader.exec_module(util_3d)
    finally:
        sys.path.remove(benchmark_dir)
    index_path = os.path.join(submission_root, scene_id + ".txt")
    instances = util_3d.read_instance_prediction_file(index_path, submission_root)
    keys = []
    masks = []
    root = os.path.normpath(submission_root)
    for mask_file, prediction in instances.items():
        rel = os.path.relpath(os.path.normpath(mask_file), root).replace("\\", "/")
        keys.append(rel)
        masks.append((util_3d.load_ids(mask_file) > 0).astype(bool))
    if not masks:
        raise ValueError(f"no prediction masks for {scene_id}")
    return keys, np.stack(masks, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--submission-root", required=True)
    ap.add_argument("--snap-root", required=True)
    ap.add_argument("--detector", default="odise", choices=["odise", "yoloworld"])
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--request-id", type=int, default=0)
    add_run_args(ap)
    args = ap.parse_args()
    params = load_overrides(args.parameters_json, {
        "detector", "lookup_threshold", "mask_confidence_threshold",
        "point_limit", "min_mask_points"})
    detector = str(params.get("detector", args.detector))
    lookup_threshold = float(params.get("lookup_threshold", LOOKUP_THRESHOLD))

    keys, masks_np = _load_final_masks(args.submission_root, args.scene_id)
    pcd = o3d.io.read_point_cloud(args.pointcloud)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if masks_np.shape[0] != len(pts):
        raise ValueError("final mask vertex count does not match point cloud")
    pcd_rgb = np.hstack([pts, cols * 255.0])
    mask_list = torch.from_numpy(masks_np.astype(np.float32))

    os.chdir(OPENINS3D_REPO)
    sys.path.insert(0, OPENINS3D_REPO)
    sys.path.insert(0, os.path.join(OPENINS3D_REPO, "openins3d"))
    from openins3d.lookup import Lookup

    results_folder = os.path.join(args.snap_root, f"{args.scene_id}_query")
    lookup = Lookup(IMAGE_SIZE, ADJUST_CAMERA[2], args.snap_root, text_input=args.classes,
                    results_folder=results_folder)
    if detector == "odise":
        lookup.call_ODISE()
    else:
        lookup.call_YOLOWORLD()
    mask_cls, score = lookup.lookup_pipelie(
        pcd_rgb, mask_list, args.scene_id, threshold=lookup_threshold)

    labels = []
    scores = []
    for i, cls in enumerate(mask_cls):
        if int(cls) < 0 or int(cls) >= len(args.classes):
            labels.append("")
            scores.append(0.0)
        else:
            labels.append(args.classes[int(cls)])
            scores.append(float(score[i]))
    payload = {
        "ok": True,
        "request_id": int(args.request_id),
        "keys": keys,
        "labels": labels,
        "scores": scores,
        "detector": detector,
    }
    directory = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(args.out_json) + ".", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, args.out_json)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()
