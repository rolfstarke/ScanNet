"""SVO2 -> ScanNet-native reconstructions.

Six engines (zed, metashape, rtabmap, isaac, open3d, bundlefusion). Scene IDs use a two-digit
suffix: tens digit = engine method, units digit = run number (0-9).
"""
import os
import re

SCANS_DIR = "/data/scannet/scans"
CUSTOM_RAW = "/data/scannet/custom/raw"
FRAMES_ROOT = "/data/scannet/derived/reconstruction/frames"
SEGMENTATOR_DIR = "/home/rolf/GIT/ScanNet/Segmentator"
MESH_CLEAN_MLX = "/home/rolf/GIT/ScanNet/Server/tools/meshclean/clean.mlx"
SIMPLIFY_MLX = "/home/rolf/GIT/ScanNet/Server/tools/meshclean/simplify.mlx"

ENGINE_INDEX = {"zed": 0, "metashape": 1, "rtabmap": 2, "isaac": 3, "open3d": 4, "bundlefusion": 5}
INDEX_ENGINE = {v: k for k, v in ENGINE_INDEX.items()}
_SCENE_ID_RE = re.compile(r"^scene(\d{4})_(\d{2})$")


def scan_id(scene_num: int, engine: str, run_num: int) -> str:
    """scene9004_40 -- tens digit is engine method, units digit is run 0-9."""
    if engine not in ENGINE_INDEX:
        raise ValueError(f"unknown engine {engine!r}")
    run_num = int(run_num)
    if run_num < 0 or run_num > 9:
        raise ValueError(f"run_num must be 0..9, got {run_num}")
    return f"scene{int(scene_num):04d}_{ENGINE_INDEX[engine] * 10 + run_num:02d}"


def scan_dir(scene_num: int, engine: str, run_num: int) -> str:
    return f"{SCANS_DIR}/{scan_id(scene_num, engine, run_num)}"


def svo_path(scene_num: int) -> str:
    return f"{CUSTOM_RAW}/scene{int(scene_num):04d}.svo2"


def frames_pool_dir(scene_num: int) -> str:
    return os.path.join(FRAMES_ROOT, f"scene{int(scene_num):04d}", "frames")


def parse_scan_id(scene_id: str):
    """Return (scene_num, engine, run_num) for a 12-char sceneNNNN_MM id."""
    m = _SCENE_ID_RE.match(scene_id)
    if not m:
        raise ValueError(f"invalid scan id {scene_id!r}")
    scene_num = int(m.group(1))
    mm = int(m.group(2))
    engine_idx, run_num = divmod(mm, 10)
    if engine_idx not in INDEX_ENGINE:
        raise ValueError(f"unknown engine index in {scene_id!r}")
    return scene_num, INDEX_ENGINE[engine_idx], run_num


def engine_run_range(engine: str):
    """Ten suffix values for one engine (e.g. open3d -> 40..49)."""
    base = ENGINE_INDEX[engine] * 10
    return list(range(base, base + 10))
