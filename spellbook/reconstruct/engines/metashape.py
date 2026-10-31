"""Metashape 2.3.1 engine. Weak ZED position priors (0.15 m, rotation disabled, fixed
calibration f=fx, b1=fy-fx, center-relative cx/cy), keyframes 0.08 m / 8 deg / gap 45, then a
named profile of official API knobs (#53 baseline, #40 optimization).

The Metashape script is generated here and executed headless by
/data/zed-metashape/conda/env/bin/python (licensed Pro Python API, exactly the old repo's
invocation pattern: the env's python runs the pipeline code; no metashape.sh, no ZED SDK
involved, so no LD_PRELOAD redirect needed). Contract:
reconstruct(work, root, gpu=None, lease_fd=None)
    -> (engine_native.ply, poses (M,4,4), keep, "y_up", recon/metashape_frames).
GPU_POLICY "managed": run.py leases a settings-pool GPU (never 0) and passes the physical
index + lease fd, which is forwarded to the Metashape python child with pass_fds.
CUDA_VISIBLE_DEVICES is cleared from the child env so enumGPUDevices sees all physical
GPUs and the physical-index gpu_mask stays correct; cpu_enable=True.

Runtime: exempt from the 3x scene-length cap (RUNTIME_CAP_EXEMPT) and unbounded -- the fixed
10800 s engine timeout was removed by maintainer decision (#40 2026-09-10).

Frames ("metric-as-is", #53 2026-09-08): the 0.15 m location priors reference the chunk, so
the world frame of chunk.transform is the metric tracker frame. Poses = chunk.transform.matrix
@ camera.transform with the similarity scale s removed from the rotation only (SVD), and
exportModel writes the same world frame; translations and vertices are NOT divided by s
(pre-#53 did; s is an arbitrary internal-to-world factor, 0.09-0.76 on 9004/9009).
Native depth: Metashape's own F32 depth map per aligned keyframe, internal units -> s * d in
metres -> uint16 mm (0-5999) resized to the color resolution, packaged with the pool color
JPEG + pose into recon/metashape_frames; the .sens holds exactly those frames
(sens frames == keep == n_depth).

Profile: SPELLBOOK_METASHAPE_PRESET (falls back to SPELLBOOK_METASHAPE_PROFILE [inferred
name]), default b5-kf350. Knob overrides SPELLBOOK_METASHAPE_SCALE_BARS /
SPELLBOOK_METASHAPE_MAX_KEYFRAMES [inferred names; #40 quotes SCALE_BARS / MAX_KEYFRAMES]
apply on top of optimized profiles only; native-default is pinned (zero bars, 400
keyframes) so a run-0 rerun can never inherit optimized physics (#40 b4 leak fix).
  native-default  #53 official defaults; baseline scene9004_10 18.096334 cm (2026-09-08)
  b2-h3           #40 b1 Ultra/Mild/DisabledInterpolation + Estimated-guided rematch
                  (2026-08-30) + b2 sparse cleanup RE 1 px / IC 3 (2026-09-01) 15.378787
  b4-scalebars3   + 3 tracker-distance scale bars, 0.02 m (#40 b4 2026-09-15) 15.324562
  b5-kf350        + max keyframes 350 (#40 b5 2026-10-04) 14.627121 = DEFAULT / best
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np

from .. import preset_env

METASHAPE_PY = "/data/zed-metashape/conda/env/bin/python"
MIN_TRANSLATION_M = 0.08
MIN_ROTATION_DEG = 8.0
MAX_GAP_FRAMES = 45
DEPTH_MAX_MM = 5999

GPU_POLICY = "managed"
SERIAL = True
RUNTIME_CAP_EXEMPT = True

NATIVE_DEFAULT = dict(  # issue 53 "native-default"
    match_downscale=1, keypoint_limit=40000, keypoint_limit_per_mpx=1000, tiepoint_limit=4000,
    generic_preselection=True, reference_preselection=True,      # ReferencePreselectionSource
    align_adaptive_fitting=False, depth_downscale=2, filter_mode="MildFiltering",
    max_neighbors=16, face_count="HighFaceCount", interpolation="EnabledInterpolation",
    source="DepthMapsData", rematch=False, sparse_cleanup=False, scale_bars=0,
    tiepoint_accuracy_px=1.0, location_accuracy_m=0.15, max_keyframes=400)
B2_H3 = dict(NATIVE_DEFAULT, depth_downscale=1,                 # Ultra High
             interpolation="DisabledInterpolation",
             rematch=True,         # 2nd matchPhotos ReferencePreselectionEstimated, guided
             sparse_cleanup=True)  # TiePoints.Filter ReprojectionError 1 px, ImageCount 3
B4_SCALEBARS3 = dict(B2_H3, scale_bars=3)  # tracker-distance bars, 0.02 m, pairs >=70% dmax
B5_KF350 = dict(B4_SCALEBARS3, max_keyframes=350)

PROFILES = {
    "native-default": NATIVE_DEFAULT,
    "b2-h3": B2_H3,
    "b4-scalebars3": B4_SCALEBARS3,
    "b5-kf350": B5_KF350,
}
DEFAULT_PROFILE = "b5-kf350"
BASELINE_PROFILE = "native-default"
PINNED_PROFILES = ("native-default",)
KNOB_ENV = {"scale_bars": "SPELLBOOK_METASHAPE_SCALE_BARS",
            "max_keyframes": "SPELLBOOK_METASHAPE_MAX_KEYFRAMES"}
SCALE_BAR_ACCURACY_M = 0.02
SCALE_BAR_MIN_FRACTION = 0.70
# [inferred] guard bound: #53 "enforces metric-frame/pose validation", thresholds not recorded.
# Measured similarity pose->prior scale is 1.000 on scene9004.
METRIC_SCALE_TOLERANCE = 0.25

# recon/ entries this adapter owns; purged before every run so a reused slot can never
# serve stale artifacts from another profile (#40 b3 poison-slot fix).
_OWNED = ("keyframes", "metashape_depth", "metashape_frames", "logs/metashape.log",
          "engine_native.ply", "metashape_result.npz", "metashape_run.py",
          "metashape_settings.json")


def selected_profile():
    """(profile name, effective settings dict, applied overrides)."""
    name = preset_env("metashape", "SPELLBOOK_METASHAPE_PROFILE") or DEFAULT_PROFILE
    if name not in PROFILES:
        raise ValueError(f"unknown Metashape profile {name!r}; known: {sorted(PROFILES)}")
    cfg = dict(PROFILES[name])
    overrides = {}
    for key, env in KNOB_ENV.items():
        raw = os.environ.get(env, "").strip()
        if not raw:
            continue
        if name in PINNED_PROFILES:
            print(f"[metashape] {env}={raw} ignored: profile {name} is pinned")
            continue
        overrides[key] = int(raw)
    cfg.update(overrides)
    if cfg["max_keyframes"] < 2 or cfg["scale_bars"] < 0:
        raise ValueError(f"invalid Metashape knobs: {overrides}")
    return name, cfg, overrides


def _settings_doc(name, cfg, overrides):
    doc = {"profile": name, "settings": cfg, "overrides": overrides}
    blob = json.dumps(doc, sort_keys=True).encode()
    doc["settings_sha256"] = hashlib.sha256(blob).hexdigest()
    return doc


def provenance():
    name, cfg, overrides = selected_profile()
    doc = _settings_doc(name, cfg, overrides)
    return {"variant": name, "config": cfg, "overrides": overrides,
            "settings_sha256": doc["settings_sha256"]}


def provenance_after(work):
    path = os.path.join(work, "metashape_result.npz")
    if not os.path.isfile(path):
        return {}
    r = np.load(path)
    return {"metashape": {"input_keyframes": int(r["n_input"][0]),
                          "aligned": int(len(r["keep"])),
                          "native_depth": int(len(r["depth_keep"])),
                          "chunk_scale": float(r["scale"][0]),
                          "version": str(r["version"][0])}}


def preflight():
    if not os.path.exists(METASHAPE_PY):
        return f"Metashape env python missing: {METASHAPE_PY}"
    try:
        selected_profile()
    except ValueError as ex:
        return str(ex)
    return None


def gpu_check(gpu=None, lease_fd=None, hold_seconds=5):
    """Native Metashape distribution check under the lease: license validity, physical
    GPU enumeration, and the exact gpu_mask math the pipeline uses. No project work."""
    if gpu is None or lease_fd is None:
        return dict(status="fail", reason="managed check requires gpu + lease_fd")
    script = (
        "import json, sys, time\n"
        "import Metashape\n"
        "devices = Metashape.app.enumGPUDevices()\n"
        "mask = (1 << %d) if 0 <= %d < len(devices) else None\n"
        "json.dump(dict(license=Metashape.License().valid, n_devices=len(devices),\n"
        "               gpu=%d, mask=mask), sys.stdout)\n"
        "sys.stdout.write('\\n'); sys.stdout.flush()\n"
        "time.sleep(%d)\n"
    ) % (gpu, gpu, gpu, hold_seconds)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(script)
        script_path = tf.name
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)  # Metashape enumerates all physical GPUs
    start = time.time()
    try:
        r = subprocess.run([METASHAPE_PY, script_path], capture_output=True, text=True,
                           timeout=hold_seconds + 120, env=env,
                           pass_fds=(lease_fd,))
    except subprocess.TimeoutExpired:
        return dict(status="fail", reason="metashape gpu_check timed out")
    finally:
        os.unlink(script_path)
    if r.returncode != 0:
        return dict(status="fail", reason=f"metashape gpu_check rc={r.returncode}: "
                                          f"{r.stderr.strip()[-400:]}")
    import json as _json
    try:
        info = _json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return dict(status="fail", reason=f"unparseable metashape output: {r.stdout[-400:]}")
    if not info.get("license"):
        return dict(status="fail", reason="Metashape license invalid")
    if info.get("mask") != (1 << gpu):
        return dict(status="fail", reason=f"gpu_mask mismatch: {info}")
    return dict(status="pass", policy="managed", physical_gpu=gpu,
                visible_gpu=f"mask-1<<{gpu}", lease_fd=lease_fd,
                runtime=f"Metashape ({info['n_devices']} devices)",
                seconds=round(time.time() - start, 1), reason=None)

_SCRIPT = '''\
import glob
import json
import os

import numpy as np

import Metashape

WORK = "@WORK@"
GPU = @GPU@
CFG = json.loads(r"""@CFG@""")
SETTINGS_SHA256 = "@SHA@"
SCALE_BARS = json.loads(r"""@BARS@""")  # [[frame_a, frame_b, tracker distance m], ...]
FRAMES = os.path.join(WORK, "frames")
KEYFRAMES = os.path.join(WORK, "keyframes")
DEPTH_OUT = os.path.join(WORK, "metashape_depth")


def to_np(m):
    return np.array([[float(m[i, j]) for j in range(4)] for i in range(4)])


def frame_of(cam):
    return int(os.path.splitext(os.path.basename(cam.label))[0])


def aligned_cameras(chunk):
    return [c for c in chunk.cameras
            if c.type == Metashape.Camera.Type.Regular and c.transform is not None]


def n_tiepoints(chunk):
    return len(chunk.tie_points.points) if chunk.tie_points is not None else 0


def main():
    if not Metashape.License().valid:
        raise RuntimeError("Metashape license invalid")
    devices = Metashape.app.enumGPUDevices()
    if devices:
        mask = (1 << GPU) if 0 <= GPU < len(devices) else (1 << len(devices)) - 1
        Metashape.app.gpu_mask = mask
    Metashape.app.cpu_enable = True
    print("[metashape] profile settings sha256 %s" % SETTINGS_SHA256, flush=True)

    photos = sorted(glob.glob(os.path.join(KEYFRAMES, "color", "*.jpg")),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not photos:
        raise RuntimeError("no keyframes in " + KEYFRAMES)
    n = len(photos)

    doc = Metashape.Document()
    chunk = doc.addChunk()
    chunk.addPhotos(photos)
    if not chunk.sensors:
        raise RuntimeError("Metashape created no sensor")
    sensor = chunk.sensors[0]

    K = np.loadtxt(os.path.join(FRAMES, "intrinsic_depth.txt"))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    calib = Metashape.Calibration()
    calib.width = int(sensor.width)
    calib.height = int(sensor.height)
    calib.f = fx
    calib.b1 = fy - fx
    calib.cx = cx - sensor.width / 2.0
    calib.cy = cy - sensor.height / 2.0
    sensor.user_calib = calib
    sensor.fixed_calibration = True

    loc = float(CFG["location_accuracy_m"])
    chunk.camera_location_accuracy = Metashape.Vector([loc, loc, loc])
    chunk.tiepoint_accuracy = float(CFG["tiepoint_accuracy_px"])
    for cam in chunk.cameras:
        pose = np.loadtxt(os.path.join(KEYFRAMES, "pose", "%d.txt" % frame_of(cam)))
        cam.reference.location = Metashape.Vector(pose[:3, 3].tolist())
        cam.reference.location_accuracy = Metashape.Vector([loc, loc, loc])
        cam.reference.location_enabled = True
        cam.reference.rotation_enabled = False

    match = dict(downscale=CFG["match_downscale"],
                 generic_preselection=CFG["generic_preselection"],
                 reference_preselection=CFG["reference_preselection"],
                 keypoint_limit=CFG["keypoint_limit"],
                 keypoint_limit_per_mpx=CFG["keypoint_limit_per_mpx"],
                 tiepoint_limit=CFG["tiepoint_limit"],
                 keep_keypoints=bool(CFG["rematch"]))  # [inferred] guided rematch reuses them
    chunk.matchPhotos(reference_preselection_mode=Metashape.ReferencePreselectionSource,
                      **match)
    chunk.alignCameras(adaptive_fitting=CFG["align_adaptive_fitting"])
    print("[metashape] pass-1 aligned %d/%d, %d tie points"
          % (len(aligned_cameras(chunk)), n, n_tiepoints(chunk)), flush=True)
    if CFG["rematch"]:
        # #40 b1: Estimated-guided re-match on the first alignment. reset_matches=False
        # (True left 318/400 native depths).
        chunk.matchPhotos(
            reference_preselection_mode=Metashape.ReferencePreselectionEstimated,
            guided_matching=True, reset_matches=False, **match)
        chunk.alignCameras(adaptive_fitting=CFG["align_adaptive_fitting"],
                           reset_alignment=False)  # [inferred] incremental re-align
        print("[metashape] rematch aligned %d/%d, %d tie points"
              % (len(aligned_cameras(chunk)), n, n_tiepoints(chunk)), flush=True)

    aligned = aligned_cameras(chunk)
    if len(aligned) < 0.9 * n:
        raise RuntimeError("only %d/%d cameras aligned" % (len(aligned), n))

    chunk.optimizeCameras(adaptive_fitting=True)

    if CFG["sparse_cleanup"]:
        # #40 b2-h3: gradual selection ReprojectionError 1 px, ImageCount 3, re-optimize.
        before = n_tiepoints(chunk)
        for crit, thr in ((Metashape.TiePoints.Filter.ReprojectionError, 1.0),
                          (Metashape.TiePoints.Filter.ImageCount, 3)):
            f = Metashape.TiePoints.Filter()
            f.init(chunk, criterion=crit)
            f.removePoints(thr)
        chunk.optimizeCameras(adaptive_fitting=True)
        print("[metashape] sparse cleanup %d -> %d tie points" % (before, n_tiepoints(chunk)),
              flush=True)

    if SCALE_BARS:
        by_frame = {frame_of(c): c for c in aligned}
        added = 0
        for a, b, dist in SCALE_BARS:
            if a not in by_frame or b not in by_frame:
                continue
            bar = chunk.addScalebar(by_frame[a], by_frame[b])
            bar.reference.distance = float(dist)
            bar.reference.accuracy = float(CFG.get("scale_bar_accuracy_m", 0.02))
            bar.reference.enabled = True
            added += 1
            print("[metashape] scale bar [%d,%d,%.4f]" % (a, b, dist), flush=True)
        if added != len(SCALE_BARS):
            raise RuntimeError("only %d/%d scale bars on aligned cameras"
                               % (added, len(SCALE_BARS)))
        chunk.updateTransform()
        chunk.optimizeCameras(adaptive_fitting=True)  # [inferred] bars enter the adjustment

    chunk.buildDepthMaps(downscale=CFG["depth_downscale"],
                         filter_mode=getattr(Metashape, CFG["filter_mode"]),
                         max_neighbors=CFG["max_neighbors"])
    chunk.buildModel(source_data=getattr(Metashape, CFG["source"]),
                     face_count=getattr(Metashape, CFG["face_count"]),
                     interpolation=getattr(Metashape, CFG["interpolation"]))
    if chunk.model is None:
        raise RuntimeError("buildModel produced no model")

    mesh_path = os.path.join(WORK, "engine_native.ply")
    chunk.exportModel(mesh_path, format=Metashape.ModelFormatPLY, binary=True,
                      save_texture=False, save_colors=True)

    rows = []
    scales = []
    T = chunk.transform.matrix
    for cam in aligned:
        M = to_np(T * cam.transform)
        R = M[:3, :3]
        U, S, Vt = np.linalg.svd(R)
        R_o = U @ Vt
        if np.linalg.det(R_o) < 0:
            Vt[-1] *= -1
            R_o = U @ Vt
        P = np.eye(4)
        P[:3, :3] = R_o
        P[:3, 3] = M[:3, 3]
        rows.append((frame_of(cam), P, cam))
        scales.append(float(S.mean()))
    if not rows:
        raise RuntimeError("no aligned cameras to export")
    rows.sort(key=lambda r: r[0])
    scale = float(np.median(scales))

    # Native depth: depth-map values are chunk-internal units; s * d is metres in the
    # world frame of the poses and mesh -> mm, 0 outside (0, 6 m). [inferred] #53 records only
    # metric uint16 0-5999 depth; measured: unscaled values only back-project onto a mesh
    # inflated by 1/s, which is what fixes the factor.
    os.makedirs(DEPTH_OUT, exist_ok=True)
    depth_keep = []
    have_depth = list(chunk.depth_maps.keys()) if chunk.depth_maps is not None else []
    for idx, _, cam in rows:
        if cam not in have_depth:
            continue
        img = chunk.depth_maps[cam].image()
        d = np.frombuffer(img.tostring(), dtype=np.float32).reshape(img.height, img.width)
        mm = np.where(np.isfinite(d) & (d > 0), np.round(d * scale * 1000.0), 0)
        mm = np.where(mm <= @DMAX@, mm, 0).astype(np.uint16)
        np.save(os.path.join(DEPTH_OUT, "%d.npy" % idx), mm)
        depth_keep.append(idx)

    poses = np.stack([r[1] for r in rows])
    keep = np.array([r[0] for r in rows], dtype=np.int64)
    np.savez(os.path.join(WORK, "metashape_result.npz"),
             poses=poses, keep=keep, scale=np.array([scale]),
             depth_keep=np.array(depth_keep, dtype=np.int64), n_input=np.array([n]),
             settings_sha256=np.array([SETTINGS_SHA256]),
             version=np.array([Metashape.app.version]))
    print("[metashape] aligned %d/%d cameras, native depth %d, scale %.6f"
          % (len(rows), n, len(depth_keep), scale), flush=True)


main()
'''


def _select_keyframes(pose_dir, max_keyframes):
    paths = sorted(glob.glob(os.path.join(pose_dir, "*.txt")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not paths:
        raise RuntimeError(f"no pose files in {pose_dir}")
    poses = np.stack([np.loadtxt(p) for p in paths])
    return select_keyframes(poses, max_keyframes)


def select_keyframes(poses, max_keyframes):
    """Greedy 0.08 m / 8 deg / gap-45 selection on tracker poses, then an even subsample to
    at most `max_keyframes` (first and last frame always kept)."""
    n = len(poses)
    selected = [0]
    last = 0
    for i in range(1, n):
        t = np.linalg.norm(poses[i][:3, 3] - poses[last][:3, 3])
        r = poses[last][:3, :3].T @ poses[i][:3, :3]
        a = np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)))
        if t >= MIN_TRANSLATION_M or a >= MIN_ROTATION_DEG or i - last >= MAX_GAP_FRAMES:
            selected.append(i)
            last = i
    if selected[-1] != n - 1:
        selected.append(n - 1)
    if len(selected) > max_keyframes:
        idxs = sorted({int(round(x)) for x in np.linspace(0, len(selected) - 1,
                                                          num=max_keyframes)})
        if 0 not in idxs:
            idxs = [0] + idxs
        if idxs[-1] != len(selected) - 1:
            idxs.append(len(selected) - 1)
        selected = [selected[i] for i in idxs]
    return selected


def select_scale_bars(positions, count, min_fraction=SCALE_BAR_MIN_FRACTION):
    """Tracker-distance scale bars between keyframes (#40 b4): candidate pairs whose tracker
    distance is >= min_fraction * dmax. [inferred] pick order: longest first, each keyframe
    used at most once. `positions` = {frame: xyz}. Returns [[a, b, d], ...] (a > b)."""
    if count <= 0:
        return []
    frames = sorted(positions)
    P = np.stack([np.asarray(positions[f], float) for f in frames])
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    dmax = float(D.max())
    ii, jj = np.where(np.triu(D, 1) >= min_fraction * dmax)
    cand = sorted(zip(ii.tolist(), jj.tolist()), key=lambda p: -D[p[0], p[1]])
    used, bars = set(), []
    for i, j in cand:
        if i in used or j in used:
            continue
        used.update((i, j))
        a, b = max(frames[i], frames[j]), min(frames[i], frames[j])
        bars.append([int(a), int(b), round(float(D[i, j]), 4)])
        if len(bars) == count:
            break
    if len(bars) < count:
        raise RuntimeError(f"only {len(bars)}/{count} scale bars >= {min_fraction:.0%} dmax")
    return bars


def check_metric_frame(positions, priors, tolerance=METRIC_SCALE_TOLERANCE):
    """Fail unless engine camera positions have the metric scale of their tracker priors
    (#53 metric-frame validation); returns the position-spread ratio. [inferred] spread ratio
    about the centroids, so it is independent of rotation and translation."""
    positions, priors = np.asarray(positions, float), np.asarray(priors, float)
    spread = np.linalg.norm(priors - priors.mean(0), axis=1).sum()
    if spread <= 0:
        raise RuntimeError("metric frame check: tracker priors have no spread")
    ratio = float(np.linalg.norm(positions - positions.mean(0), axis=1).sum() / spread)
    if abs(ratio - 1.0) > tolerance:
        raise RuntimeError(f"metric frame check: poses are {ratio:.3f}x the tracker priors")
    return ratio


def _render(work, gpu, name, cfg, overrides, bars):
    doc = _settings_doc(name, cfg, overrides)
    cfg_json = json.dumps(dict(cfg, scale_bar_accuracy_m=SCALE_BAR_ACCURACY_M),
                          sort_keys=True)
    script = (_SCRIPT
              .replace("@WORK@", work)
              .replace("@GPU@", str(int(gpu)))
              .replace("@CFG@", cfg_json)
              .replace("@SHA@", doc["settings_sha256"])
              .replace("@BARS@", json.dumps(bars))
              .replace("@DMAX@", str(DEPTH_MAX_MM)))
    return script, dict(doc, scale_bars_pairs=bars)


def _verify_rendered(work, gpu, name, cfg, overrides, bars, result=None):
    """Fail loudly unless the slot's rendered script/settings (and result, when given) were
    produced from the current env profile (#40 b3: detached jobs silently re-ran defaults)."""
    script, doc = _render(work, gpu, name, cfg, overrides, bars)
    with open(os.path.join(work, "metashape_run.py")) as f:
        if f.read() != script:
            raise RuntimeError("metashape_run.py does not match the current env profile")
    with open(os.path.join(work, "metashape_settings.json")) as f:
        on_disk = json.load(f)
    if on_disk != json.loads(json.dumps(doc)):
        raise RuntimeError("metashape_settings.json does not match the current env profile")
    if result is not None:
        sha = str(result["settings_sha256"][0])
        if sha != doc["settings_sha256"]:
            raise RuntimeError(f"metashape result settings {sha[:8]} != current "
                               f"{doc['settings_sha256'][:8]}")
    return doc


def _purge_owned(work):
    for rel in _OWNED:
        p = os.path.join(work, rel)
        if os.path.islink(p) or os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)
    for p in glob.glob(os.path.join(work, "*.psx")) + glob.glob(os.path.join(work, "*.files")):
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


def _package_frames(work, frames, keep, depth_dir):
    """recon/metashape_frames/{color,depth}/<i>: pool JPEG (symlink) + native depth PNG at
    the color resolution (nearest)."""
    import cv2
    out = os.path.join(work, "metashape_frames")
    for d in ("color", "depth"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    for i in keep:
        color = os.path.join(frames, "color", f"{i}.jpg")
        c = cv2.imread(color, cv2.IMREAD_UNCHANGED)
        if c is None:
            raise RuntimeError(f"cannot read {color}")
        mm = np.load(os.path.join(depth_dir, f"{i}.npy"))
        if mm.shape[:2] != c.shape[:2]:
            mm = cv2.resize(mm, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST)
        os.symlink(os.path.realpath(color), os.path.join(out, "color", f"{i}.jpg"))
        if not cv2.imwrite(os.path.join(out, "depth", f"{i}.png"), mm.astype(np.uint16)):
            raise RuntimeError(f"cannot write depth {i}")
    return out


def reconstruct(work, root, gpu=None, lease_fd=None):
    if gpu is None or lease_fd is None:
        raise RuntimeError("metashape is GPU_POLICY managed: gpu + lease_fd required "
                           "(run.py acquires the settings-pool lease)")
    name, cfg, overrides = selected_profile()
    work = os.path.abspath(work)
    frames = os.path.join(work, "frames")
    _purge_owned(work)
    keyframes = os.path.join(work, "keyframes")
    logs = os.path.join(work, "logs")
    for d in (keyframes, os.path.join(keyframes, "color"),
              os.path.join(keyframes, "pose"), logs):
        os.makedirs(d, exist_ok=True)

    keep_in = _select_keyframes(os.path.join(frames, "pose"), cfg["max_keyframes"])
    for i in keep_in:
        shutil.copy2(os.path.join(frames, "color", f"{i}.jpg"),
                     os.path.join(keyframes, "color", f"{i}.jpg"))
        shutil.copy2(os.path.join(frames, "pose", f"{i}.txt"),
                     os.path.join(keyframes, "pose", f"{i}.txt"))
    bars = select_scale_bars(
        {i: np.loadtxt(os.path.join(frames, "pose", f"{i}.txt"))[:3, 3] for i in keep_in},
        cfg["scale_bars"])
    print(f"[metashape] profile={name} overrides={overrides or '{}'} "
          f"keyframes={len(keep_in)} scale_bars={bars}")

    if not os.path.exists(METASHAPE_PY):
        raise RuntimeError(f"Metashape env python not found: {METASHAPE_PY}")

    script, doc = _render(work, gpu, name, cfg, overrides, bars)
    script_path = os.path.join(work, "metashape_run.py")
    with open(script_path, "w") as f:
        f.write(script)
    with open(os.path.join(work, "metashape_settings.json"), "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    _verify_rendered(work, gpu, name, cfg, overrides, bars)

    log_path = os.path.join(logs, "metashape.log")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)  # Metashape must enumerate all physical GPUs
    pass_fds = (lease_fd,)
    with open(log_path, "wb") as logf:
        try:
            # No timeout: Metashape runs are unbounded (#40 2026-09-10).
            subprocess.run([METASHAPE_PY, script_path], check=True,
                           stdout=logf, stderr=subprocess.STDOUT, env=env,
                           cwd=work, pass_fds=pass_fds)
        except subprocess.CalledProcessError as e:
            tail = open(log_path, errors="replace").read().splitlines()[-20:]
            raise RuntimeError(f"metashape run failed rc={e.returncode}, see {log_path}\n"
                               + "\n".join(tail))

    result_path = os.path.join(work, "metashape_result.npz")
    if not os.path.exists(result_path):
        raise RuntimeError(f"metashape finished without {result_path}, see {log_path}")
    result = np.load(result_path)
    _verify_rendered(work, gpu, name, cfg, overrides, bars, result)
    poses = result["poses"]
    keep = [int(k) for k in result["keep"]]
    depth_keep = set(int(k) for k in result["depth_keep"])
    scale = float(result["scale"][0])
    if not keep or len(poses) != len(keep):
        raise RuntimeError(f"metashape result inconsistent: {len(poses)} poses, "
                           f"{len(keep)} keep")
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"invalid chunk scale {scale}")
    check_metric_frame(poses[:, :3, 3],
                       [np.loadtxt(os.path.join(keyframes, "pose", f"{k}.txt"))[:3, 3]
                        for k in keep])

    # .sens frames == keep == n_depth: drop aligned keyframes without a native depth map.
    sel = [j for j, k in enumerate(keep) if k in depth_keep]
    if not sel:
        raise RuntimeError("metashape produced no native depth maps")
    poses = poses[sel]
    keep = [keep[j] for j in sel]
    print(f"[metashape] aligned {len(result['keep'])}/{len(keep_in)}, "
          f"native depth {len(keep)}, scale {scale:.6f}")

    mesh_path = os.path.join(work, "engine_native.ply")
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(np.asarray(mesh.vertices)) == 0:
        raise RuntimeError(f"empty mesh at {mesh_path}")
    if not mesh.has_vertex_colors():
        raise RuntimeError(f"mesh at {mesh_path} has no vertex colors")

    out = _package_frames(work, frames, keep, os.path.join(work, "metashape_depth"))
    shutil.rmtree(os.path.join(work, "metashape_depth"), ignore_errors=True)
    return mesh_path, poses, keep, "y_up", out
