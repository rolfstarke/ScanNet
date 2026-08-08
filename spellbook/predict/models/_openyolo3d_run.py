"""Runs OpenYOLO3D's Mask3D-proposals + YOLO-World-2D-voting pipeline inside the `openyolo3d`
conda env, writing ScanNet benchmark instance-segmentation format (scan-net.org).

The actual model/algorithm lives in the OpenYOLO3D repo (/home/rolf/GIT/OpenYOLO3D) -- this
script only adapts its output into ScanNet format. Uses the model's own single-scene API
(utils.OpenYolo3D().predict()) against a scratch dir that symlinks this scene's extracted
frames/ (see frames.py) plus the mesh, and overrides the text prompts with the requested classes.

OpenYOLO3D needs the LD_LIBRARY_PATH to include /data/openyolo3D/cuda-11.3/lib64 (runner.py
sets it before launching this script).
"""
import argparse
import os
import shutil
import sys

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import write_scannet_predictions  # noqa: E402

OPENYOLO3D_REPO = "/home/rolf/GIT/OpenYOLO3D"
SCRATCH_ROOT = "/data/openyolo3D/scratch"
CONFIG_TEMPLATE = os.path.join(OPENYOLO3D_REPO, "pretrained", "config_scannet200.yaml")
MIN_MASK_POINTS = 20


def _build_scratch_scene(frames_dir, pointcloud, scene_id):
    """Symlink frames/ + mesh into OpenYOLO3D's expected scene layout: poses/ (plural), a single
    intrinsics.txt (from frames_dir/intrinsic_COLOR.txt -- OpenYOLO3D's adjust_intrinsic expects
    color-resolution intrinsics and rescales to depth internally, see WORLD_2_CAM), color/,
    depth/, and one *.ply at the root."""
    scratch = os.path.join(SCRATCH_ROOT, scene_id)
    if os.path.islink(scratch) or os.path.exists(scratch):
        shutil.rmtree(scratch)
    os.makedirs(scratch)

    links = {
        "color": os.path.join(frames_dir, "color"),
        "depth": os.path.join(frames_dir, "depth"),
        "poses": os.path.join(frames_dir, "pose"),  # model expects plural "poses"
        "intrinsics.txt": os.path.join(frames_dir, "intrinsic_color.txt"),
        f"{scene_id}.ply": pointcloud,
    }
    for name, target in links.items():
        os.symlink(target, os.path.join(scratch, name))
    return scratch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointcloud", required=True, help="scene pointcloud (.ply, xyz+rgb)")
    ap.add_argument("--frames", required=True, help="extracted frames dir (frames.py)")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="predictions output dir")
    ap.add_argument("--label_set", default="scannet18",
                    choices=["scannet18", "scannet200"],
                    help="label vocabulary: scannet18 (NYU40, default) or scannet200 (raw id-column ids)")
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scene_id = os.path.basename(os.path.dirname(args.pointcloud))

    scratch = _build_scratch_scene(args.frames, args.pointcloud, scene_id)

    # Per-run config: reuse the repo's scannet200 template but with exactly the requested
    # classes as text prompts.
    import yaml
    with open(CONFIG_TEMPLATE) as f:
        cfg = yaml.safe_load(f)
    cfg["network2d"]["text_prompts"] = list(args.classes)
    cfg["network3d"]["is_gt"] = False
    cfg_path = os.path.join(args.out, "openyolo3d_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    os.chdir(OPENYOLO3D_REPO)
    sys.path.insert(0, OPENYOLO3D_REPO)
    from utils import OpenYolo3D

    openyolo3d = OpenYolo3D(cfg_path)
    prediction = openyolo3d.predict(path_2_scene_data=scratch, depth_scale=1000.0, text=args.classes)
    masks, classes, scores = prediction[scene_id]

    # classes are indices into the prompt list; the model reserves the last index
    # (num_classes-1 = len(args.classes)) for unassigned/background -- drop those.
    background = len(args.classes)
    masks_np = masks.cpu().numpy().astype(bool)  # (n_points, n_instances)
    classes_np = classes.cpu().numpy()
    scores_np = scores.cpu().numpy()

    def _instances():
        for i in range(masks_np.shape[1]):
            if classes_np[i] == background:
                continue
            yield masks_np[:, i], args.classes[classes_np[i]], float(scores_np[i])

    n_written = write_scannet_predictions(args.out, args.classes, _instances(), MIN_MASK_POINTS,
                                          label_set=args.label_set)
    print(f"[INFO] Wrote {n_written} instances to {args.out} "
          f"({masks_np.shape[1]} raw masks, {masks_np.shape[0]} points)")


if __name__ == "__main__":
    main()
