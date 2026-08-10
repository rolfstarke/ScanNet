"""SVO2 -> ScanNet-native reconstructions.

Six engines (zed, metashape, rtabmap, isaac, open3d, bundlefusion), one scan each under
/data/scannet/scans/scene90NN_NN. See spellbook/tmp/reconstruction_plan.md.
"""

SCANS_DIR = "/data/scannet/scans"
CUSTOM_RAW = "/data/scannet/custom/raw"
SEGMENTATOR_DIR = "/home/rolf/GIT/ScanNet/Segmentator"
MESH_CLEAN_MLX = "/home/rolf/GIT/ScanNet/Server/tools/meshclean/clean.mlx"
SIMPLIFY_MLX = "/home/rolf/GIT/ScanNet/Server/tools/meshclean/simplify.mlx"

# scan index per engine, fixed
ENGINE_INDEX = {"zed": 0, "metashape": 1, "rtabmap": 2, "isaac": 3, "open3d": 4, "bundlefusion": 5}
INDEX_ENGINE = {v: k for k, v in ENGINE_INDEX.items()}


def scan_id(scene_num: int, engine: str) -> str:
    """scene9004_00 -- 9000-range marks custom scans, scan index = engine."""
    return f"scene{scene_num:04d}_{ENGINE_INDEX[engine]:02d}"


def scan_dir(scene_num: int, engine: str) -> str:
    return f"{SCANS_DIR}/{scan_id(scene_num, engine)}"


def svo_path(scene_num: int) -> str:
    return f"{CUSTOM_RAW}/scene{scene_num:04d}.svo2"
