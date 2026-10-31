"""BundleFusion (FangGet/Ubuntu fork) in Docker -- the ScanNet reference engine.

Runs the fork's headless binary on a prepared 640x480 sensor stream via the PrimeSense dir
emulation (color/ + depth/). The fork instantiates ONLY that sensor (s_sensorIdx 1), and it
hardcodes 640x480 buffers.

Baseline scene9004_50 (#52 2026-10-01, #50 2026-10-06): canonical ScanNet configs with only the
fork-required sensor/K adaptation, dual GPUs in order [1,2], rounds 0, crop-correct input.

Input: frames are center-cropped to 4:3, then resized to 640x480 (color LINEAR, depth NEAREST),
fx'=fx*s, cx'=(cx-x0)*s, cy'=(cy-y0)*s. Prepared once per pool, cached under
<scannet_root>/derived/reconstruction/bundlefusion/prepared/ and validated by manifest, all
outside the solver clock. Prebuild: python -m spellbook.reconstruct.engines.bundlefusion --prepare 9004

Poses: <out>/pose/<i>.txt, row-major camera-to-world in BundleFusion's world frame (first camera
= identity), written BEFORE the reintegration drain. Filename = original frame index. Frames
without a finite pose are DROPPED (never identity-backfilled). A skipped frame would shift all
indices, so the log is checked and the run fails loudly.

Mesh: <out>/mesh.ply (MLIB binary PLY, xyz + rgba), copied verbatim to engine_native.ply.

Mirror: the public fork's PrimeSenseSensor x-flips depth and color, so its mesh and poses are
mirrored relative to the frames stored in .sens. The default image removes that flip
(containers/bundlefusion/noflip.patch). Measured on scene9004 (2026-10-30): plain depth lands on
mesh_aligned at 2-3 cm on tracked frames, the CAD fit is better as-is than reflected, and the run
reproduces the #52 signature (2313+1266 poses, 184 lost, submaps 1210/1220). The flipped
bundlefusion:latest stays selectable through SPELLBOOK_BUNDLEFUSION_IMAGE; it scores lower
(47.88 cm) only because a reflected mesh happens to fit this CAD better, and is not usable.

Presets and overrides: SPELLBOOK_BUNDLEFUSION_PRESET selects PRESETS (default official-default);
SPELLBOOK_BUNDLEFUSION_APP / _BUNDLING ("key=value;key=value") are merged over it,
_IMAGE replaces the image, _NAME / _HYPOTHESIS go to run.json.

Image: rebuilt from spellbook/reconstruct/containers/bundlefusion/, so it is new provenance: the
historic digest is kept only as a reference and run.json records the actual image id.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import SCANNET_ROOT, frames_pool_dir, parse_scan_id, preset_env, runtime_cap_s
from ..containers import gpu_args, mount, user_args

_IMAGE = "bundlefusion:noflip"  # see Mirror above
_HISTORIC_IMAGE_DIGEST = (  # the lost #52 image; a rebuild cannot match it bit-exactly
    "sha256:528315a1324aa6bedeaf7f474bcef9c21f44038affed4647822133a1019c7b51")
_SIFTFIX_IMAGE = "bundlefusion:spellbook-siftfix"
_SCANNET_RECONS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Server", "tools", "recons")
_APP_CFG_SRC = os.path.join(_SCANNET_RECONS, "zParametersScanNet.txt")
_BUNDLE_CFG_SRC = os.path.join(_SCANNET_RECONS, "zParametersBundlingScanNet.txt")
_OPT_ROUNDS = 0
_TIMEOUT = 10800  # safety only; the rule is the solver cap below
_BF_WIDTH, _BF_HEIGHT = 640, 480
_JPEG_QUALITY = 95  # [inferred] #52 records only "validated prepared 640x480 sensor stream"
_PREPARE_WORKERS = 8
# [inferred] location: #52 records hashes and a "prepared volume", no path.
PREPARED_ROOT = os.path.join(SCANNET_ROOT, "derived", "reconstruction", "bundlefusion", "prepared")

GPU_POLICY = "managed"
SERIAL = False
GPU_COUNT = 2
GPU_PAIR = (1, 2)  # ordered: reconstruction device, bundling device (#52 "ordered pair [1,2]")
# [inferred] pair lease: the historic mechanism is not logged. run.py leases GPU_POOL;
# reconstruct() adds the second, always in this order, so two runs cannot deadlock.
GPU_POOL = [GPU_PAIR[0]]
# #52 2026-10-01: the 3x rule covers the BundleFusion Docker process only; ScanNet delivery
# (finalize alone took 683 s on 9004) is timed separately and is info-only. So the whole-pipeline
# cap in run.py is off for this engine, and reconstruct() enforces the 3x cap on the solver.
RUNTIME_CAP_EXEMPT = True


class SolverCapExceeded(Exception):
    """The solver overran 3x scene length; run.py records status timeout and exits 124."""


CAP_ERRORS = (SolverCapExceeded,)

_FULLRES_APP = {  # #50 batch 1 "full-resolution fusion + capacity"
    "s_integrationWidth": "640", "s_integrationHeight": "480",
    "s_rayCastWidth": "640", "s_rayCastHeight": "480",
    "s_hashNumSDFBlocks": "1500000", "s_hashNumBuckets": "2000000",
    "s_marchingCubesMaxNumTriangles": "10000000",
}
# Frozen #50/#52 profiles. Only official-default is a current baseline: the others' artifacts
# were purged under the runtime rule (#50 2026-09-15) and they need the siftfix image.
PRESETS = {
    "official-default": dict(image=_IMAGE, app={}, bundling={}),
    "batch1-scene9004_51": dict(image=_SIFTFIX_IMAGE, app=_FULLRES_APP,
                                bundling={"s_minKeyScale": "1.0f"}),
    "batch2-scene9004_52": dict(image=_SIFTFIX_IMAGE, app=_FULLRES_APP,
                                bundling={"s_minKeyScale": "0.5f"}),
    "s2b4-surfarea-004": dict(image=_SIFTFIX_IMAGE, app=_FULLRES_APP,
                              bundling={"s_minKeyScale": "0.5f",
                                        "s_surfAreaPcaThresh": "0.04f"}),
}
DEFAULT_PRESET = "official-default"


def _parse_overlay(raw):
    """`key=value;key=value` -> dict. [inferred] format: the issues name the variables only."""
    out = {}
    for item in (raw or "").split(";"):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise RuntimeError(f"bad BundleFusion overlay item {item!r} (want key=value)")
        out[key.strip()] = value.strip()
    return out


def effective_preset():
    """The frozen preset with the SPELLBOOK_BUNDLEFUSION_* overlays merged over it."""
    name = preset_env("bundlefusion") or DEFAULT_PRESET
    if name not in PRESETS:
        raise RuntimeError(f"unknown BundleFusion preset {name!r}; known: {sorted(PRESETS)}")
    base = PRESETS[name]
    return dict(
        preset=name,
        name=os.environ.get("SPELLBOOK_BUNDLEFUSION_NAME", "").strip() or name,
        hypothesis=os.environ.get("SPELLBOOK_BUNDLEFUSION_HYPOTHESIS", "").strip() or None,
        image=os.environ.get("SPELLBOOK_BUNDLEFUSION_IMAGE", "").strip() or base["image"],
        app={**base["app"], **_parse_overlay(os.environ.get("SPELLBOOK_BUNDLEFUSION_APP"))},
        bundling={**base["bundling"],
                  **_parse_overlay(os.environ.get("SPELLBOOK_BUNDLEFUSION_BUNDLING"))})


def provenance():
    eff = effective_preset()
    return {"variant": eff["name"], "preset": eff["preset"], "hypothesis": eff["hypothesis"],
            "config": dict(image=eff["image"], app=eff["app"], bundling=eff["bundling"],
                           opt_rounds=_OPT_ROUNDS, timeout_s=_TIMEOUT, gpus=list(GPU_PAIR),
                           input=f"center-crop 4:3 -> {_BF_WIDTH}x{_BF_HEIGHT}")}


def provenance_after(work):
    path = os.path.join(work, "bundlefusion_run.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def preflight():
    image = effective_preset()["image"]
    r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    if r.returncode:
        return f"docker image {image} unavailable: {r.stderr.strip()[-200:]}"
    return None


def gpu_check(gpu=None, lease_fd=None, hold_seconds=5):
    """Native BundleFusion distribution check under the lease: container start with the
    exact CUDA_VISIBLE_DEVICES mapping, in-container runtime probe, sleep, removal."""
    if gpu is None or lease_fd is None:
        return dict(status="fail", reason="managed check requires gpu + lease_fd")
    start = time.time()
    container = f"spellbook-gpu-check-bundlefusion-{os.getpid()}"
    cmd = ["docker", "run", "--rm", "--name", container, *gpu_args(gpu), *user_args(), _IMAGE,
           "bash", "-lc",
           f"echo CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}; "
           "nvidia-smi -L 2>/dev/null || echo no-nvidia-smi; sleep %d" % hold_seconds]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=hold_seconds + 120, pass_fds=(lease_fd,))
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", container], capture_output=True)
        return dict(status="fail", reason="bundlefusion gpu_check timed out")
    if r.returncode != 0:
        return dict(status="fail", reason=f"bundlefusion gpu_check rc={r.returncode}: "
                                          f"{(r.stderr or r.stdout)[-400:]}")
    if f"CUDA_VISIBLE_DEVICES={gpu}" not in (r.stdout or ""):
        return dict(status="fail",
                    reason=f"assigned env missing in container output: {r.stdout[-300:]}")
    return dict(status="pass", policy="managed", physical_gpu=gpu, visible_gpu=gpu,
                lease_fd=lease_fd, runtime=f"docker {_IMAGE}",
                seconds=round(time.time() - start, 1), reason=None)


def _set_param(text, key, value):
    """Set `key = value;` on its own config line. An unknown key fails: a typo must not run."""
    patched, n = re.subn(rf"(?m)^(\s*{re.escape(key)}\s*=\s*)[^;\n]*;",
                         lambda m: f"{m.group(1)}{value};", text)
    if n != 1:
        raise RuntimeError(f"config key {key!r} found {n} times; expected exactly 1")
    return patched


def _patch_app_config(text, fx, fy, cx, cy, overrides=None):
    if "s_SDFVoxelSize = 0.010f;" not in text:
        raise RuntimeError("canonical config lost s_SDFVoxelSize = 0.010f")
    patched = re.sub(r"s_sensorIdx\s*=\s*8;", "s_sensorIdx = 1;", text)
    if "s_sensorIdx = 1;" not in patched:
        raise RuntimeError("s_sensorIdx patch did not apply")
    for key, value in (("s_sensorDepthMax", "6.0f"), ("s_renderDepthMax", "6.0f")):
        if not re.search(key + r"\s*=\s*6\.0f;", patched):
            raise RuntimeError(f"{key} is not 6.0f in the patched config")
    for key, value in (overrides or {}).items():
        patched = _set_param(patched, key, value)
    patched += (
        f"\ns_cameraIntrinsicFx = {fx:.6f};\n"
        f"s_cameraIntrinsicFy = {fy:.6f};\n"
        f"s_cameraIntrinsicCx = {cx:.6f};\n"
        f"s_cameraIntrinsicCy = {cy:.6f};\n"
    )
    return patched


def _patch_bundling_config(text, overrides=None):
    for key, value in (overrides or {}).items():
        text = _set_param(text, key, value)
    return text


def crop_and_scale(width, height, K):
    """Center crop to 4:3, scale to 640x480. Returns (x0, y0, cw, ch, s, (fx, fy, cx, cy))."""
    cw, ch = width, height
    if width * _BF_HEIGHT > height * _BF_WIDTH:
        cw = int(round(height * _BF_WIDTH / _BF_HEIGHT))
    else:
        ch = int(round(width * _BF_HEIGHT / _BF_WIDTH))
    x0, y0 = (width - cw) // 2, (height - ch) // 2
    s = _BF_WIDTH / cw
    return x0, y0, cw, ch, s, (float(K[0, 0] * s), float(K[1, 1] * s),
                               float((K[0, 2] - x0) * s), float((K[1, 2] - y0) * s))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pool_sha(pool):
    return _sha256(os.path.join(os.path.dirname(pool.rstrip(os.sep)), "frames_manifest.json"))


def _frame_ids(pool):
    ids = sorted(int(f[:-4]) for f in os.listdir(os.path.join(pool, "color"))
                 if f.endswith(".jpg"))
    if not ids:
        raise RuntimeError(f"no frames in {os.path.join(pool, 'color')}")
    if ids != list(range(len(ids))):
        raise RuntimeError(f"frame ids in {pool} are not contiguous; pose indices would shift")
    return ids


def prepared_dir(scene):
    return os.path.join(PREPARED_ROOT, f"scene{int(scene):04d}")


def validate_prepared(path, pool, check_files=True):
    """Return the manifest of a usable prepared stream, else raise RuntimeError."""
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"no prepared manifest at {manifest_path}")
    with open(manifest_path) as f:
        doc = json.load(f)
    if doc.get("pool_manifest_sha256") != _pool_sha(pool):
        raise RuntimeError(f"prepared stream {path} is from a different frame pool")
    if check_files:
        for rel, sha in doc["files"].items():
            target = os.path.join(path, rel)
            if not os.path.isfile(target) or _sha256(target) != sha:
                raise RuntimeError(f"prepared file failed validation: {target}")
    return doc


def prepare(scene, pool=None, out=None):
    """Build the cropped 640x480 stream once, atomically; reuse a valid cache."""
    import cv2

    pool = pool or frames_pool_dir(scene)
    out = out or prepared_dir(scene)
    try:
        return validate_prepared(out, pool, check_files=False)
    except RuntimeError:
        pass
    ids = _frame_ids(pool)
    K = np.loadtxt(os.path.join(pool, "intrinsic_depth.txt"))
    first = cv2.imread(os.path.join(pool, "depth", f"{ids[0]}.png"), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"unreadable depth frame {ids[0]} in {pool}")
    height, width = first.shape[:2]
    x0, y0, cw, ch, s, (fx, fy, cx, cy) = crop_and_scale(width, height, K)

    tmp = f"{out}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "color"))
    os.makedirs(os.path.join(tmp, "depth"))
    size = (_BF_WIDTH, _BF_HEIGHT)

    def one(i):
        rgb = cv2.imread(os.path.join(pool, "color", f"{i}.jpg"))
        dep = cv2.imread(os.path.join(pool, "depth", f"{i}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or dep is None:
            raise RuntimeError(f"unreadable frame {i} in {pool}")
        if rgb.shape[:2] != (height, width) or dep.shape[:2] != (height, width):
            raise RuntimeError(f"frame {i} is not {width}x{height} in {pool}")
        rgb = cv2.resize(rgb[y0:y0 + ch, x0:x0 + cw], size, interpolation=cv2.INTER_LINEAR)
        dep = cv2.resize(dep[y0:y0 + ch, x0:x0 + cw], size, interpolation=cv2.INTER_NEAREST)
        color_rel, depth_rel = f"color/{i}.jpg", f"depth/{i}.png"
        cv2.imwrite(os.path.join(tmp, color_rel), rgb, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        cv2.imwrite(os.path.join(tmp, depth_rel), dep)
        return ((color_rel, _sha256(os.path.join(tmp, color_rel))),
                (depth_rel, _sha256(os.path.join(tmp, depth_rel))))

    try:
        with ThreadPoolExecutor(_PREPARE_WORKERS) as ex:
            files = dict(pair for frame in ex.map(one, ids) for pair in frame)
        doc = dict(schema=1, scene=int(scene), pool=os.path.abspath(pool),
                   pool_manifest_sha256=_pool_sha(pool), n_frames=len(ids),
                   crop=[x0, y0, cw, ch], scale=s, K=[fx, fy, cx, cy],
                   size=[_BF_WIDTH, _BF_HEIGHT], jpeg_quality=_JPEG_QUALITY, files=files)
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.replace(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return doc


def _gpu_env_args(gpus):
    """gpu_args() pins one GPU; widen its pin to the ordered pair, in physical bus order."""
    pin = f"CUDA_VISIBLE_DEVICES={gpus[0]}"
    args = [f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))}" if a == pin else a
            for a in gpu_args(gpus[0])]
    return args + ["-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID"]


def _run_docker(image, prepared, cfg_dir, out_dir, gpus, lease_fds):
    container = f"bundlefusion_{os.getpid()}"
    cmd = [
        "docker", "run", "--rm", "--name", container,
        *_gpu_env_args(gpus), *user_args(),
        *mount(prepared, "/input", read_only=True),
        *mount(cfg_dir, "/config", read_only=True),
        *mount(out_dir, "/output"),
        image,
        "/app/build/bundle_fusion_headless",
        "/config/zParametersScanNet.effective.txt",
        "/config/zParametersBundlingScanNet.effective.txt",
        "/input", "/output",
        str(_OPT_ROUNDS),
    ]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT,
                              pass_fds=tuple(lease_fds))
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", container], capture_output=True)
        raise RuntimeError(f"BundleFusion docker timed out after {_TIMEOUT}s; container "
                           f"{container} killed")


def _parse_poses(pose_dir, n_frames, log):
    """(poses, keep) for finite poses only: never identity-backfilled."""
    if not os.path.isdir(pose_dir):
        raise RuntimeError(f"no pose/ dir in docker output; see {log}")
    ids = sorted(int(f[:-4]) for f in os.listdir(pose_dir) if f.endswith(".txt"))
    if any(i >= n_frames for i in ids):
        raise RuntimeError("pose index exceeds frame count; trajectory mismatch")
    poses, keep = [], []
    for i in ids:
        pose = np.loadtxt(os.path.join(pose_dir, f"{i}.txt")).reshape(4, 4)
        if np.all(np.isfinite(pose)):
            poses.append(pose)
            keep.append(i)
    if not keep:
        raise RuntimeError("no finite poses in docker output; reconstruction failed")
    return np.stack(poses), keep


def _image_id(image):
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _write_json(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")


def reconstruct(work, root, gpu=None, lease_fd=None):
    if gpu is None or lease_fd is None:
        raise RuntimeError("bundlefusion is GPU_POLICY managed: gpu + lease_fd required "
                           "(run.py acquires the first lease of GPU_POOL)")
    if gpu != GPU_PAIR[0]:
        raise RuntimeError(f"leased GPU {gpu}; the ordered pair starts at {GPU_PAIR[0]}")
    import cv2  # noqa: F401  # before the mesh libs: conda GLib load order (#52 2026-09-08)

    from utils.gpu import gpu_lease
    from utils.scan_lock import exclusive_lock, frames_lock_path

    scene = parse_scan_id(os.path.basename(os.path.normpath(root)))[0]
    eff = effective_preset()
    pool = os.path.realpath(os.path.join(work, "frames"))
    prepared = prepared_dir(scene)
    timings = {}

    started = time.monotonic()
    with exclusive_lock(frames_lock_path(scene)):
        prepare(scene, pool, prepared)
    timings["prepare"] = time.monotonic() - started
    started = time.monotonic()
    doc = validate_prepared(prepared, pool)
    timings["prepared_validate"] = time.monotonic() - started
    fx, fy, cx, cy = doc["K"]

    log_dir = os.path.join(work, "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(_APP_CFG_SRC) as f:
        app_cfg = _patch_app_config(f.read(), fx, fy, cx, cy, eff["app"])
    with open(_BUNDLE_CFG_SRC) as f:
        bundling_cfg = _patch_bundling_config(f.read(), eff["bundling"])
    app_path = os.path.join(log_dir, "zParametersScanNet.effective.txt")
    bundling_path = os.path.join(log_dir, "zParametersBundlingScanNet.effective.txt")
    with open(app_path, "w") as f:
        f.write(app_cfg)
    with open(bundling_path, "w") as f:
        f.write(bundling_cfg)

    bf_out = os.path.join(work, "bf_out")
    shutil.rmtree(bf_out, ignore_errors=True)
    os.makedirs(bf_out)
    print(f"[bundlefusion] {doc['n_frames']} frames, K@640x480 = {fx:.2f}/{fy:.2f} f, "
          f"{cx:.2f}/{cy:.2f} c, crop={doc['crop']}, gpus={list(GPU_PAIR)}, "
          f"preset={eff['preset']}, image={eff['image']}")

    with gpu_lease(GPU_PAIR[1:], SCANNET_ROOT) as second:
        started = time.monotonic()
        res = _run_docker(eff["image"], prepared, log_dir, bf_out, GPU_PAIR,
                          (lease_fd, second.fileno()))
        solver_s = time.monotonic() - started
    timings["solver"] = solver_s

    log_path = os.path.join(log_dir, "bundlefusion.log")
    with open(log_path, "w") as f:
        f.write(res.stdout or "")
        f.write(res.stderr or "")
    limit = runtime_cap_s(scene)
    solver_pass = limit is None or solver_s <= limit
    summary = dict(solver_s=solver_s, solver_limit_s=limit, solver_pass=solver_pass,
                   physical_gpus=list(GPU_PAIR), opt_rounds=_OPT_ROUNDS, timeout_s=_TIMEOUT,
                   image=eff["image"], image_id=_image_id(eff["image"]),
                   historic_image_digest=_HISTORIC_IMAGE_DIGEST,
                   prepared=prepared, prepared_K=doc["K"], prepared_crop=doc["crop"],
                   pool_manifest_sha256=doc["pool_manifest_sha256"],
                   app_cfg_sha256=_sha256(app_path), bundling_cfg_sha256=_sha256(bundling_path))
    _write_json(os.path.join(work, "bundlefusion_run.json"), summary)

    if res.returncode != 0:
        raise RuntimeError(f"BundleFusion docker failed (exit {res.returncode}); see {log_path}")
    if not solver_pass:
        raise SolverCapExceeded(f"solver {solver_s:.2f}s > {limit:.6f}s (3x scene length)")
    log = (res.stdout or "") + (res.stderr or "")
    if "[warn] skipping frame" in log:
        raise RuntimeError(f"BundleFusion skipped frames (read error); pose indices would "
                           f"shift; see {log_path}")
    m = re.search(r"\[info\] processed (\d+)/(\d+) frames", log)
    if not m or int(m.group(1)) != int(m.group(2)):
        raise RuntimeError(f"frame count mismatch in run log ({log_path}); cannot map poses")

    started = time.monotonic()
    mesh_src = os.path.join(bf_out, "mesh.ply")
    if not os.path.exists(mesh_src):
        raise RuntimeError(f"no mesh.ply in docker output; see {log_path}")
    native = os.path.join(work, "engine_native.ply")
    shutil.move(mesh_src, native)
    poses, keep = _parse_poses(os.path.join(bf_out, "pose"), doc["n_frames"], log_path)
    shutil.rmtree(bf_out, ignore_errors=True)
    timings["output_copy"] = time.monotonic() - started
    _write_json(os.path.join(work, "stage_timings.json"), timings)

    summary.update(frames_staged=doc["n_frames"], frames_kept=len(keep))
    _write_json(os.path.join(work, "bundlefusion_run.json"), summary)
    print(f"[bundlefusion] solver={solver_s:.2f}s/{limit}s mesh={native} "
          f"poses={len(keep)}/{doc['n_frames']} conv=bundlefusion_world")
    return native, poses, keep, "bundlefusion_world"


def main():
    ap = argparse.ArgumentParser(description="Prepare the cropped 640x480 BundleFusion input")
    ap.add_argument("--prepare", type=int, nargs="+", required=True, metavar="SCENE")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from utils.scan_lock import exclusive_lock, frames_lock_path

    for scene in args.prepare:
        with exclusive_lock(frames_lock_path(scene)):
            doc = prepare(scene)
        print(f"[prepare] scene{scene:04d} {doc['n_frames']} frames K={doc['K']} "
              f"crop={doc['crop']} -> {prepared_dir(scene)}")


if __name__ == "__main__":
    main()
