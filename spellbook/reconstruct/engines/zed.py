"""ZED engine: native ZED SDK 5.4 spatial mapping (official offline ZEDfu workflow).

Two-pass `.area` pipeline (#29 2026-08-27, #47): pass 1 = positional tracking with
area memory over the whole SVO, async save_area_map + get_area_export_state() poll;
pass 2 = reopen with area_file_path, same tracking mode, enable_spatial_mapping(MESH) in
the same pass as pose capture. Poses come only from the proven grab()/get_position(WORLD)
loop: a per-frame get_positional_tracking_status() SIGSEGVs the SDK after frame 0 (#39 b1).
Mesh: extract_whole_spatial_map -> MeshFilterParameters LOW -> colored OBJ
(recon/engine_native.obj). Kept frames: tracking state OK and i < total-1 (the pool
extraction drops the last SVO frame too); >= 100 required.

GPU (#29 remap): GPU_POLICY "managed". The SDK runs in a child process
(`python -m spellbook.reconstruct.engines.zed --child ...`) whose CUDA_VISIBLE_DEVICES is the
leased physical GPU before `import pyzed`; sdk_gpu_id is never set (sdk_gpu_id != 0 returns
constant poses, #18); GPU 0 is never leased. No pyzed import at module level.

Runtime env (#39 b5/b7): ZED_DIR must hold settings/SN<serial>.conf (else CALIBRATION FILE
NOT AVAILABLE) and LD_LIBRARY_PATH must contain $ZED_DIR/lib and
$ZED_DIR/dependencies/ai_dependencies/lib (else NEURAL TRT NOT FOUND, rc -11 at open). The
3disspellbook conda activate hook sets both; jobs.py passes ZED_DIR/LD_LIBRARY_PATH.

Preset: SPELLBOOK_ZED_PRESET (falls back to SPELLBOOK_ZED_VARIANT, then ZED_VARIANT, the
#39/#47 name), default `b5-gen1-neural`.
  native-default      official baseline (#47/#57): GEN_1, NEURAL_PLUS, 5 cm, AUTO range,
                      confidence 95, stability 0, 2048 MB, LOW filter.
  b5-gen1-neural      DEFAULT (#39 b5 2026-09-10, b7 2026-09-16 winner 21.395865 cm, 247.3 s).
  s2b8-fixedcalib     current best (#39 b8 2026-10-04, 21.306363 cm, 231.26 s); not default
                      (no flip approved).
  s2b7-native-onepass fastest named benchmark (#39 b7, 201.0 s, 30.54 cm); never default.
  gen3-stability10 / b3-neural  ancestors kept for the derivation chain; gen3-stability10
                      overruns the 361.2 s cap on 9004 (#39 b7 R5).
SDK placement (#39 b8): confidence/texture_confidence -> RuntimeParameters; depth mode,
depth min/max, depth_stabilization, camera_disable_self_calib -> InitParameters;
resolution/range/stability_counter/max_memory/use_chunk_only/save_texture ->
SpatialMappingParameters; mode/imu fusion/pose smoothing -> PositionalTrackingParameters.
pyzed 5.4 has no decay / enable_forget_past / disparity_std.

Outputs in recon/: engine_native.obj, map.area (2-pass), zed_poses.npz, engine_config.yaml,
engine_timing.json (open / tracking / area_export / mapping / mesh / native_total, #47).
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

from .. import preset_env

GPU_POLICY = "managed"
SERIAL = True

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# ZED RIGHT_HANDED_Y_UP camera basis (x right, y up, z back) -> OpenCV/ScanNet camera basis
# (x right, y down, z forward).
_D = np.diag([1.0, -1.0, -1.0, 1.0])

_NATIVE_DEFAULT = dict(
    passes=2, tracking="GEN_1", depth_mode="NEURAL_PLUS",
    depth_min=0.1, depth_max=6.0, resolution_meter=0.05, range_meter=0.0,  # 0.0 = AUTO
    confidence_threshold=95, texture_confidence_threshold=100, stability_counter=0,
    max_memory_usage=2048, mesh_filter="LOW", depth_stabilization=30,
    camera_disable_self_calib=False, use_chunk_only=False, pose_smoothing=False,
    enable_imu_fusion=True, save_texture=False)

PRESETS = {"native-default": _NATIVE_DEFAULT}                                 # #47/#57
PRESETS["gen3-stability10"] = dict(_NATIVE_DEFAULT, tracking="GEN_3", range_meter=3.5,
                                   confidence_threshold=50, stability_counter=10)  # #39 b1
PRESETS["b3-neural"] = dict(PRESETS["gen3-stability10"], depth_mode="NEURAL")      # #39 b3
PRESETS["b5-gen1-neural"] = dict(PRESETS["b3-neural"], tracking="GEN_1")           # #39 b5/b7
PRESETS["s2b7-native-onepass"] = dict(_NATIVE_DEFAULT, passes=1)                   # #39 b7
PRESETS["s2b8-fixedcalib"] = dict(PRESETS["b5-gen1-neural"],
                                  camera_disable_self_calib=True)                  # #39 b8
DEFAULT_VARIANT = "b5-gen1-neural"
BASELINE_VARIANT = "native-default"
BEST_VARIANT = "s2b8-fixedcalib"

MIN_POSES = 100
AREA_EXPORT_TIMEOUT_S = 600  # [inferred] bound on the async .area export poll


def selected_variant():
    name = preset_env("zed", "SPELLBOOK_ZED_VARIANT", "ZED_VARIANT") or DEFAULT_VARIANT
    if name not in PRESETS:
        raise ValueError(f"unknown ZED preset {name!r}; known: {sorted(PRESETS)}")
    return name


def provenance():
    name = selected_variant()
    return {"variant": name, "config": dict(PRESETS[name])}


def provenance_after(work):
    path = os.path.join(work, "engine_timing.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {"engine_timing": json.load(f)}


def preflight():
    if importlib.util.find_spec("pyzed") is None:
        return "pyzed not importable in this environment"
    zed_dir = os.environ.get("ZED_DIR", "")
    if not zed_dir or not os.path.isdir(os.path.join(zed_dir, "settings")):
        return "ZED_DIR unset or without settings/ (calibration .conf); activate 3disspellbook"
    ai = os.path.join(zed_dir, "dependencies", "ai_dependencies", "lib")
    if os.path.isdir(ai) and ai not in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        return f"LD_LIBRARY_PATH lacks {ai} (NEURAL TRT NOT FOUND at Camera::open)"
    try:
        selected_variant()
    except ValueError as ex:
        return str(ex)
    return None


def _child_env(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _REPO_ROOT + (os.pathsep + pp if pp else "")
    return env


def gpu_check(gpu=None, lease_fd=None, hold_seconds=5):
    """Managed distribution check: pyzed imports under CUDA_VISIBLE_DEVICES=<leased GPU> and
    stays alive for the hold interval. No SVO is opened."""
    if gpu is None or lease_fd is None:
        return dict(status="fail", reason="managed check requires gpu + lease_fd")
    reason = preflight()
    if reason:
        return dict(status="fail", reason=reason)
    code = ("import os, sys, time\n"
            "import pyzed.sl as sl\n"
            "print(sl.Camera.get_sdk_version(), os.environ['CUDA_VISIBLE_DEVICES'])\n"
            "sys.stdout.flush(); time.sleep(%d)\n") % hold_seconds
    start = time.time()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=_child_env(gpu), pass_fds=(lease_fd,),
                       timeout=hold_seconds + 120)
    if r.returncode != 0:
        return dict(status="fail", reason=f"pyzed child rc={r.returncode}: "
                                          f"{r.stderr.strip()[-400:]}")
    sdk, visible = (r.stdout.strip().splitlines() or ["? ?"])[-1].split()[:2]
    if visible != str(gpu):
        return dict(status="fail", reason=f"CUDA_VISIBLE_DEVICES {visible} != {gpu}")
    return dict(status="pass", policy="managed", physical_gpu=gpu, visible_gpu=visible,
                lease_fd=lease_fd, runtime=f"ZED SDK {sdk}",
                seconds=round(time.time() - start, 1), reason=None)


def reconstruct(work, root, gpu=None, lease_fd=None):
    if gpu is None or lease_fd is None:
        raise RuntimeError("zed is GPU_POLICY managed: gpu + lease_fd required")
    work = os.path.abspath(work)
    marker = os.path.join(work, "svo_path.txt")
    if not os.path.exists(marker):
        raise RuntimeError("run.py must write recon/svo_path.txt before engines run")
    svo = open(marker).read().strip()
    variant = selected_variant()
    for stale in ("engine_native.obj", "engine_native.mtl", "map.area", "zed_poses.npz",
                  "engine_timing.json", "engine_config.yaml"):
        p = os.path.join(work, stale)
        if os.path.exists(p):
            os.remove(p)
    cmd = [sys.executable, "-m", "spellbook.reconstruct.engines.zed", "--child",
           "--svo", svo, "--work", work, "--variant", variant,
           "--lease-fd", str(lease_fd)]
    print(f"[zed] variant={variant} gpu={gpu} passes={PRESETS[variant]['passes']}")
    r = subprocess.run(cmd, env=_child_env(gpu), pass_fds=(lease_fd,), cwd=_REPO_ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"zed child failed rc={r.returncode}")

    res = np.load(os.path.join(work, "zed_poses.npz"))
    poses, keep = res["poses"], [int(k) for k in res["keep"]]
    if len(keep) < MIN_POSES or len(poses) != len(keep):
        raise RuntimeError(f"zed: {len(poses)} poses / {len(keep)} keep")
    if not np.isfinite(poses).all():
        raise RuntimeError("zed: non-finite poses")
    # [inferred] world stays ZED Y-up (the native mesh lives there); only the camera basis
    # is converted to OpenCV for the .sens poses.
    poses = np.stack([p @ _D for p in poses])
    return os.path.join(work, "engine_native.obj"), poses, keep, "zed_y_up"


# ------------------------------------------------------------------------------------------
# child process (CUDA_VISIBLE_DEVICES pinned by the parent before this import of pyzed)
# ------------------------------------------------------------------------------------------

def _init(sl, svo, cfg):
    init = sl.InitParameters()
    init.set_from_svo_file(svo)
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_mode = getattr(sl.DEPTH_MODE, cfg["depth_mode"])
    init.depth_minimum_distance = cfg["depth_min"]
    init.depth_maximum_distance = cfg["depth_max"]
    init.depth_stabilization = int(cfg["depth_stabilization"])
    init.camera_disable_self_calib = bool(cfg["camera_disable_self_calib"])
    # Never set sdk_gpu_id; device comes from CUDA_VISIBLE_DEVICES only.
    return init


def _tracking(sl, cfg, area=None):
    tr = sl.PositionalTrackingParameters()
    tr.mode = getattr(sl.POSITIONAL_TRACKING_MODE, cfg["tracking"])
    tr.enable_area_memory = True
    tr.enable_imu_fusion = bool(cfg["enable_imu_fusion"])
    tr.enable_pose_smoothing = bool(cfg["pose_smoothing"])
    if area:
        tr.area_file_path = area
    return tr


def _runtime(sl, cfg):
    rt = sl.RuntimeParameters()
    rt.confidence_threshold = int(cfg["confidence_threshold"])
    rt.texture_confidence_threshold = int(cfg["texture_confidence_threshold"])
    return rt


def _mapping(sl, cfg):
    spm = sl.SpatialMappingParameters(map_type=sl.SPATIAL_MAP_TYPE.MESH)
    spm.resolution_meter = float(cfg["resolution_meter"])
    if cfg["range_meter"] > 0:
        spm.range_meter = float(cfg["range_meter"])
    else:
        spm.set_range(sl.MAPPING_RANGE.AUTO)
    spm.max_memory_usage = int(cfg["max_memory_usage"])
    spm.stability_counter = int(cfg["stability_counter"])
    spm.use_chunk_only = bool(cfg["use_chunk_only"])
    spm.save_texture = bool(cfg["save_texture"])
    return spm


def _open(sl, svo, cfg, label):
    zed = sl.Camera()
    err = zed.open(_init(sl, svo, cfg))
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"failed to open svo ({label}): {err}")
    return zed


def _check(sl, err, what):
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")


def _child_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--svo", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--lease-fd", type=int, default=None)
    args = ap.parse_args(argv)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        sys.exit("CUDA_VISIBLE_DEVICES must be set by the parent before importing pyzed")
    lease_file = os.fdopen(args.lease_fd, "r") if args.lease_fd is not None else None

    import pyzed.sl as sl

    cfg = PRESETS[args.variant]
    work = args.work
    area = os.path.join(work, "map.area")
    t = dict(open=0.0, tracking=0.0, area_export=0.0, mapping=0.0, mesh=0.0)
    t_all = time.monotonic()

    if cfg["passes"] == 2:
        # ---- pass 1: tracking with area memory over the whole SVO -> map.area ----
        s = time.monotonic()
        zed = _open(sl, args.svo, cfg, "pass 1")
        _check(sl, zed.enable_positional_tracking(_tracking(sl, cfg)), "enable tracking (1)")
        t["open"] += time.monotonic() - s
        s = time.monotonic()
        rt = _runtime(sl, cfg)
        n1 = 0
        while True:
            err = zed.grab(rt)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            _check(sl, err, f"grab {n1} (pass 1)")
            n1 += 1
        t["tracking"] = time.monotonic() - s
        s = time.monotonic()
        _check(sl, zed.save_area_map(area), "save_area_map")
        while zed.get_area_export_state() == sl.AREA_EXPORTING_STATE.RUNNING:
            if time.monotonic() - s > AREA_EXPORT_TIMEOUT_S:
                raise RuntimeError("area export timed out")
            time.sleep(0.1)
        state = zed.get_area_export_state()
        if state != sl.AREA_EXPORTING_STATE.SUCCESS:
            raise RuntimeError(f"area export failed: {state}")
        zed.disable_positional_tracking()
        zed.close()
        t["area_export"] = time.monotonic() - s
        print(f"[zed] pass 1 {n1} frames, area {os.path.getsize(area)} B "
              f"({t['area_export']:.1f}s export)", flush=True)

    # ---- mapping pass (pass 2, or the single pass): poses + MESH in the same replay ----
    s = time.monotonic()
    zed = _open(sl, args.svo, cfg, "mapping pass")
    _check(sl, zed.enable_positional_tracking(
        _tracking(sl, cfg, area if cfg["passes"] == 2 else None)), "enable tracking (2)")
    _check(sl, zed.enable_spatial_mapping(_mapping(sl, cfg)), "enable_spatial_mapping")
    t["open"] += time.monotonic() - s
    s = time.monotonic()
    rt = _runtime(sl, cfg)
    total = zed.get_svo_number_of_frames()
    pose = sl.Pose()
    poses, keep = [], []
    last_state = None
    i = 0
    while True:
        err = zed.grab(rt)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        _check(sl, err, f"grab {i} (mapping pass)")
        state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        if i < total - 1 and state == sl.POSITIONAL_TRACKING_STATE.OK:
            poses.append(pose.pose_data(sl.Transform()).m.copy())
            keep.append(i)
        # [inferred] mapping-state transitions only (INITIALIZING -> OK expected)
        mstate = zed.get_spatial_mapping_state()
        if mstate != last_state:
            print(f"[zed-mapping] frame {i}: {mstate}", flush=True)
            last_state = mstate
            if mstate in (sl.SPATIAL_MAPPING_STATE.FPS_TOO_LOW,
                          sl.SPATIAL_MAPPING_STATE.NOT_ENOUGH_MEMORY):
                raise RuntimeError(f"spatial mapping failed at frame {i}: {mstate}")
        i += 1
    t["mapping"] = time.monotonic() - s
    s = time.monotonic()
    mesh = sl.Mesh()
    _check(sl, zed.extract_whole_spatial_map(mesh), "extract_whole_spatial_map")
    fp = sl.MeshFilterParameters()
    fp.set(getattr(sl.MESH_FILTER, cfg["mesh_filter"]))
    mesh.filter(fp)
    native = os.path.join(work, "engine_native.obj")
    if not mesh.save(native, sl.MESH_FILE_FORMAT.OBJ):
        raise RuntimeError(f"mesh save failed: {native}")
    t["mesh"] = time.monotonic() - s
    n_verts, n_tris = len(mesh.vertices), len(mesh.triangles)
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    t["native_total"] = time.monotonic() - t_all

    if len(keep) < MIN_POSES:
        raise RuntimeError(f"only {len(keep)} valid poses")
    P = np.stack(poses)
    traj = float(np.sum(np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1)))
    np.savez(os.path.join(work, "zed_poses.npz"), poses=P, keep=np.asarray(keep, np.int64))
    with open(os.path.join(work, "engine_timing.json"), "w") as f:
        json.dump({k: round(v, 3) for k, v in t.items()}, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(work, "engine_config.yaml"), "w") as f:
        yaml.safe_dump({"engine": "zed", "variant": args.variant, "sdk": sl.Camera.get_sdk_version(),
                        "svo_frames": total, "poses": len(keep), "traj_m": round(traj, 3),
                        "mesh_vertices": n_verts, "mesh_triangles": n_tris,
                        "settings": dict(cfg)}, f, sort_keys=False)
    print(f"[zed] {args.variant}: n={len(keep)}/{total} traj={traj:.3f}m "
          f"mesh {n_verts}v/{n_tris}t native_total={t['native_total']:.1f}s "
          + " ".join(f"{k}={v:.1f}" for k, v in t.items() if k != "native_total"), flush=True)
    if lease_file is not None:
        lease_file.close()


if __name__ == "__main__":
    _child_main(sys.argv[1:])
