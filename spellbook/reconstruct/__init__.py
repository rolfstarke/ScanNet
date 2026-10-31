"""SVO2 -> ScanNet-native reconstructions.

Six engines (zed, metashape, rtabmap, isaac, open3d, bundlefusion). Scene IDs use a two-digit
suffix: tens digit = engine method, units digit = run number (0-9).
"""
import os
import re

import yaml

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_SPELLBOOK)
# Read settings.yaml directly: this package init runs before callers put spellbook/ on sys.path.
with open(os.path.join(_SPELLBOOK, "settings.yaml")) as _f:
    SCANNET_ROOT = (yaml.safe_load(_f) or {}).get("scannet_root", "/data/scannetdata")

SCANS_DIR = os.path.join(SCANNET_ROOT, "scans")
CUSTOM_RAW = os.path.join(SCANNET_ROOT, "custom", "raw")
FRAMES_ROOT = os.path.join(SCANNET_ROOT, "derived", "reconstruction", "frames")
SEGMENTATOR_DIR = os.path.join(_REPO_ROOT, "Segmentator")
MESH_CLEAN_MLX = os.path.join(_REPO_ROOT, "Server", "tools", "meshclean", "clean.mlx")
SIMPLIFY_MLX = os.path.join(_REPO_ROOT, "Server", "tools", "meshclean", "simplify.mlx")

ENGINE_INDEX = {"zed": 0, "metashape": 1, "rtabmap": 2, "isaac": 3, "open3d": 4, "bundlefusion": 5}
INDEX_ENGINE = {v: k for k, v in ENGINE_INDEX.items()}
_SCENE_ID_RE = re.compile(r"^scene(\d{4})_(\d{2})$")

# Runtime rule (2026-09-15, #41/#42/#50/#61): non-Metashape reconstructions must finish in
# less than three times the scene length. Authoritative lengths = MCAP side_by_side log_time
# span (#51 2026-09-16): 9004 120.400110 s -> 361.200330 s, 9009 157.043787 s -> 471.131361 s.
# Queue/lease wait and CAD scoring are outside the timed region.
SCENE_DURATION_S = {9004: 120.400110, 9009: 157.043787}
RUNTIME_CAP_FACTOR = 3.0


def runtime_cap_s(scene_num):
    """Wall-clock cap in seconds for one reconstruction of `scene_num`, or None if the
    scene length is unknown (no cap enforced)."""
    d = SCENE_DURATION_S.get(int(scene_num))
    return None if d is None else round(RUNTIME_CAP_FACTOR * d, 6)


def preset_env(engine, *legacy_names):
    """Preset/profile name requested via env for `engine`, or None.

    SPELLBOOK_<ENGINE>_PRESET first (survives detached jobs through the jobs.py
    SPELLBOOK_ prefix allowlist), then the historical names in order (e.g. ZED_VARIANT)."""
    for name in (f"SPELLBOOK_{engine.upper()}_PRESET",) + tuple(legacy_names):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


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
