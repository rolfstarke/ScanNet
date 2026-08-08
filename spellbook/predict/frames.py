"""Full-density RGB-D frame extraction from ScanNet .sens files, shared by the frame-dependent
prediction models (OpenYOLO3D, Open3DIS).

Extraction delegates to ScanNet's own SensReader SensorData exporter (frame_skip=1 keeps the
original frame indices as contiguous 0..N-1 filenames -- both models require contiguous names
and subsample via their own config, OpenYOLO3D `frequency`, Open3DIS `img_interval`).

Canonical output (per scene):
    /data/scannet/scans/<scene_id>/frames/
    ├── color/{0..N-1}.jpg       # RGB frames (original JPEG bytes)
    ├── depth/{0..N-1}.png       # 16-bit depth (mm; meters = px / 1000)
    ├── pose/{0..N-1}.txt        # 4x4 camera-to-world
    ├── intrinsic_color.txt      # 4x4 color intrinsics
    ├── extrinsic_color.txt      # 4x4 color extrinsics
    ├── intrinsic_depth.txt      # 4x4 depth intrinsics
    └── extrinsic_depth.txt      # 4x4 depth extrinsics
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "SensReader", "python")))
from SensorData import SensorData

SCANS_DIR = "/data/scannet/scans"

INTRINSIC_FILES = ("intrinsic_color.txt", "extrinsic_color.txt",
                   "intrinsic_depth.txt", "extrinsic_depth.txt")


def _count_dir(path):
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))


def _sens_num_frames(sens_path):
    """Read only the .sens header (num_frames field) without decoding frame data.

    ScanNet's own SensorData.load() reads the whole file into memory; the header layout
    comes from SensReader (SensorData.py:54-69). Frame data starts at offset 304+strlen."""
    import struct

    with open(sens_path, "rb") as f:
        f.seek(4)
        strlen = struct.unpack("<Q", f.read(8))[0]
        f.seek(296 + strlen)
        return struct.unpack("<Q", f.read(8))[0]


def extract_frames(scene_id):
    """Extract ALL frames from <scene>.sens to <scene>/frames/ with sequential 0..N-1 names.
    Idempotent: returns early if frames/ already holds the full expected count."""
    scene_dir = os.path.join(SCANS_DIR, scene_id)
    sens_path = os.path.join(scene_dir, f"{scene_id}.sens")
    frames_dir = os.path.join(scene_dir, "frames")
    color_dir, depth_dir, pose_dir = (os.path.join(frames_dir, d) for d in ("color", "depth", "pose"))

    if not os.path.isfile(sens_path):
        raise FileNotFoundError(f"no .sens file at {sens_path}")

    num_frames = _sens_num_frames(sens_path)

    if (_count_dir(color_dir) == num_frames and _count_dir(depth_dir) == num_frames
            and _count_dir(pose_dir) == num_frames
            and all(os.path.isfile(os.path.join(frames_dir, f)) for f in INTRINSIC_FILES)):
        print(f"[INFO] frames already extracted for {scene_id} ({num_frames} frames)")
        return frames_dir

    sd = SensorData(sens_path)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    print(f"[INFO] extracting {num_frames} frames from {sens_path} ...")
    sd.export_color_images(color_dir, frame_skip=1)
    sd.export_depth_images(depth_dir, frame_skip=1)
    sd.export_poses(pose_dir, frame_skip=1)
    sd.export_intrinsics(frames_dir)

    print(f"[INFO] extracted {num_frames} frames -> {frames_dir}")
    return frames_dir


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "scene0000_00"
    extract_frames(scene)
