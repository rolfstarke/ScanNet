"""Open3D engine: bounded B1 baseline (#51 2026-09-17, retained after #61 2026-09-26).

Official ScanNet/Open3D fusion defaults with the multi-hour classic registration replaced by
deterministic source-pose integration ("pool-D"): the shared Stage-A ZED pool pose P
(frames/pose/<i>.txt, ZED Y-up) is conjugated D@P@D with D = diag(1,-1,-1,1) and integrated
with extrinsic inv(D@P@D) (convention zed_opencv, as before).

Keep rule full-span-max-K: idx = rint(linspace(0, N-1, K)), K = _BOUNDED_MAX_KEEP_BY_SCENE
(9004: 80 -> 0,46,91,...,3565,3611; 9009: 60). 403 (stride-9) and 9009 at 80 frames crash
deterministically in vbg.extract_triangle_mesh (CUDA illegal memory access) with 50000
blocks -- keep the caps.

Fusion (verbatim #51): "ScanNet fusion 320x240 voxel 0.010 trunc 0.060 depth 0.1-4.0 scale
1000, Tensor VBG block 16/50000 trunc_mult 6.0"; color LINEAR / depth NEAREST resize (as
tsdf.py); extract_triangle_mesh default weight_threshold 3.0 (#61 "wt-thresh 3.0").

GPU_POLICY "managed": tensor CUDA VoxelBlockGrid on the leased GPU, run in a child process
(`python -m spellbook.reconstruct.engines.open3d --child ...`) whose CUDA_VISIBLE_DEVICES is
the leased physical GPU before Open3D starts -> CUDA:0 (the parent has already imported
open3d, so it never initializes CUDA itself; GPU 0 stays untouched). Backend selector
SPELLBOOK_OPEN3D_PRESET / OPEN3D_BOUNDED_BACKEND (#51 name) = "tensor" (also the code
default, so a detached job without the env still runs B1). Any other value falls back to the
legacy CPU ScalableTSDFVolume (tsdf.py) and logs FALLBACK_LOG; a valid baseline log must not
contain it.

Profiles: _BOUNDED=True, _BOUNDED_PROFILES=("b1",); run-0-only guard: bounded B1 applies to
slot _40 only (other slots refuse -- the S2 candidate machinery was rolled back in #61).
Provenance: recon/run.json fields via provenance()/provenance_after(), plus
recon/effective_config.json, recon/stage_timings.json, recon/frame_indices.npy,
recon/open3d/reconstruction.log.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from .. import parse_scan_id, preset_env, runtime_cap_s, SCENE_DURATION_S

GPU_POLICY = "managed"
SERIAL = False

_BOUNDED = True
_BOUNDED_PROFILES = ("b1",)
_BOUNDED_MAX_KEEP_BY_SCENE = {9004: 80, 9009: 60}
DEFAULT_BACKEND = "tensor"
FALLBACK_LOG = "[open3d] FALLBACK legacy CPU ScalableTSDFVolume"
ISSUE = 51
SOURCE_FPS = 15  # #51 run.json "fps 15" (SVO header)

_D = np.diag([1.0, -1.0, -1.0, 1.0])
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

B1 = dict(width=320, height=240, voxel_size=0.010, sdf_trunc=0.060, depth_min=0.1,
          depth_max=4.0, depth_scale=1000.0, block_resolution=16, block_count=50000,
          trunc_voxel_multiplier=6.0, weight_threshold=3.0)


def selected_backend():
    return preset_env("open3d", "OPEN3D_BOUNDED_BACKEND") or DEFAULT_BACKEND


def keep_indices(n_frames, max_keep):
    """full-span-max-K keep rule (first and last pool frame always included)."""
    k = min(int(max_keep), int(n_frames))
    return np.unique(np.rint(np.linspace(0, n_frames - 1, k)).astype(int))


def _scene_and_run(root):
    scene, engine, run = parse_scan_id(os.path.basename(os.path.normpath(root)))
    if engine != "open3d":
        raise RuntimeError(f"open3d adapter on non-open3d slot {root}")
    return scene, run


def provenance():
    backend = selected_backend()
    return {"variant": "b1" if backend == "tensor" else f"b1-{backend}-fallback",
            "config": dict(B1, backend=backend, poses="zed-pool-D",
                           max_keep_by_scene=_BOUNDED_MAX_KEEP_BY_SCENE),
            "fps": SOURCE_FPS}


def provenance_after(work):
    path = os.path.join(work, "effective_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        eff = json.load(f)
    return {k: eff[k] for k in ("backend", "profile", "keep", "keep_rule", "max_keep", "poses")
            if k in eff}


def preflight():
    try:
        import open3d as o3d
    except Exception as ex:
        return f"open3d not importable: {ex}"
    if selected_backend() == "tensor" and not o3d._build_config.get("BUILD_CUDA_MODULE"):
        return "open3d build without CUDA (tensor B1 needs open3d 0.19 CUDA)"
    return None


def gpu_check(gpu=None, lease_fd=None, hold_seconds=5):
    """Managed check: CUDA:0 (= leased physical GPU) allocates a tensor and holds it."""
    if gpu is None or lease_fd is None:
        return dict(status="fail", reason="managed check requires gpu + lease_fd")
    code = ("import os, time\n"
            "import open3d as o3d\n"
            "assert o3d.core.cuda.is_available()\n"
            "t = o3d.core.Tensor.ones((1024,), device=o3d.core.Device('CUDA:0'))\n"
            "print(o3d.__version__, os.environ['CUDA_VISIBLE_DEVICES'], flush=True)\n"
            "time.sleep(%d)\n") % hold_seconds
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    start = time.time()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                       pass_fds=(lease_fd,), timeout=hold_seconds + 120)
    if r.returncode != 0:
        return dict(status="fail", reason=f"open3d CUDA child rc={r.returncode}: "
                                          f"{r.stderr.strip()[-400:]}")
    ver, visible = (r.stdout.strip().splitlines() or ["? ?"])[-1].split()[:2]
    return dict(status="pass", policy="managed", physical_gpu=gpu, visible_gpu=visible,
                lease_fd=lease_fd, runtime=f"open3d {ver} CUDA",
                seconds=round(time.time() - start, 1), reason=None)


def _log(work, line):
    d = os.path.join(work, "open3d")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "reconstruction.log"), "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _integrate_tensor(frames, idx, poses, K, dw, dh, device):
    import cv2
    import open3d as o3d
    c = B1
    Ki = K.copy()
    Ki[0] *= c["width"] / float(dw)
    Ki[1] *= c["height"] / float(dh)
    intr = o3d.core.Tensor(Ki, o3d.core.Dtype.Float64)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=c["voxel_size"], block_resolution=c["block_resolution"],
        block_count=c["block_count"], device=device)
    min_mm = int(round(c["depth_min"] * c["depth_scale"]))
    for fi, P in zip(idx, poses):
        col = cv2.imread(os.path.join(frames, "color", f"{fi}.jpg"))
        dep = cv2.imread(os.path.join(frames, "depth", f"{fi}.png"), cv2.IMREAD_UNCHANGED)
        if col is None or dep is None:
            raise RuntimeError(f"unreadable frame {fi}")
        col = cv2.resize(col, (c["width"], c["height"]), interpolation=cv2.INTER_LINEAR)
        dep = cv2.resize(dep, (c["width"], c["height"]), interpolation=cv2.INTER_NEAREST)
        dep = np.where(dep >= min_mm, dep, 0).astype(np.uint16)  # [inferred] depth_min 0.1 m
        depth = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(dep))).to(device)
        color = o3d.t.geometry.Image(
            o3d.core.Tensor(np.ascontiguousarray(col[:, :, ::-1]))).to(device)
        ext = o3d.core.Tensor(np.linalg.inv(P), o3d.core.Dtype.Float64)
        coords = vbg.compute_unique_block_coordinates(
            depth, intr, ext, c["depth_scale"], c["depth_max"],
            trunc_voxel_multiplier=c["trunc_voxel_multiplier"])
        vbg.integrate(coords, depth, color, intr, intr, ext, c["depth_scale"],
                      c["depth_max"], trunc_voxel_multiplier=c["trunc_voxel_multiplier"])
    return vbg.extract_triangle_mesh(weight_threshold=c["weight_threshold"]).to_legacy()


def reconstruct(work, root, gpu=None, lease_fd=None):
    """B1: pool-D poses on the full-span-max-K keep set, tensor VBG on the leased GPU."""
    import cv2
    import open3d as o3d

    if gpu is None or lease_fd is None:
        raise RuntimeError("open3d is GPU_POLICY managed: gpu + lease_fd required")
    t0 = time.monotonic()
    scene, run = _scene_and_run(root)
    sid = os.path.basename(os.path.normpath(root))
    if _BOUNDED and run != 0:
        raise RuntimeError(f"bounded B1 is run-0 only (slot _40); refusing {sid}")
    if scene not in _BOUNDED_MAX_KEEP_BY_SCENE:
        raise RuntimeError(f"no bounded keep cap for scene{scene:04d} (VBG extract limit)")
    profile = _BOUNDED_PROFILES[0]
    backend = selected_backend()
    max_keep = _BOUNDED_MAX_KEEP_BY_SCENE[scene]

    frames = os.path.join(work, "frames")
    states = open(os.path.join(frames, "pose_state.txt")).read().split()
    n = len(states)
    idx = keep_indices(n, max_keep)
    bad = [int(i) for i in idx if states[i] != "OK"]
    if bad:  # [inferred] pool-D needs valid Stage-A poses on every selected frame
        raise RuntimeError(f"selected pool frames without OK pose: {bad[:10]}")
    P = np.stack([np.loadtxt(os.path.join(frames, "pose", f"{i}.txt")) for i in idx])
    poses = np.stack([_D @ p @ _D for p in P])
    K = np.loadtxt(os.path.join(frames, "intrinsic_depth.txt"))[:3, :3]
    c0 = cv2.imread(os.path.join(frames, "color", f"{idx[0]}.jpg"))
    dh, dw = c0.shape[:2]

    keep_rule = f"full-span-max-{max_keep}"
    eff = dict(sid=sid, issue=ISSUE, backend=backend, profile=profile,
               keep=f"{len(idx)}/{n}", keep_rule=keep_rule, max_keep=max_keep,
               poses="zed-pool-D", convention="zed_opencv", fps=SOURCE_FPS,
               duration=SCENE_DURATION_S.get(scene), limit=runtime_cap_s(scene),
               fusion=dict(B1), frame_dims=[dw, dh])
    with open(os.path.join(work, "effective_config.json"), "w") as f:
        json.dump(eff, f, indent=2, sort_keys=True)
        f.write("\n")
    np.save(os.path.join(work, "frame_indices.npy"), idx)
    _log(work, f"[open3d] sid={sid} backend={backend} profile={profile} "
               f"keep={len(idx)}/{n} rule={keep_rule} config={json.dumps(B1, sort_keys=True)}")

    native = os.path.join(work, "engine_native.ply")
    if os.path.exists(native):
        os.remove(native)
    t_fuse = time.monotonic()
    if backend == "tensor":
        pose_path = os.path.join(work, "open3d", "b1_poses.npy")
        np.save(pose_path, poses)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = _REPO_ROOT + (os.pathsep + pp if pp else "")
        r = subprocess.run([sys.executable, "-m", "spellbook.reconstruct.engines.open3d",
                            "--child", "--work", work, "--lease-fd", str(lease_fd)],
                           env=env, pass_fds=(lease_fd,), cwd=_REPO_ROOT)
        if r.returncode != 0:
            raise RuntimeError(f"open3d tensor child failed rc={r.returncode}")
        mesh = o3d.io.read_triangle_mesh(native)
        _log(work, f"[open3d] tensor mesh {len(mesh.vertices)} verts / "
                   f"{len(mesh.triangles)} faces, bounded tensor keep={len(idx)}/{n}")
    else:
        from ..tsdf import integrate_tsdf
        _log(work, f"{FALLBACK_LOG} (backend={backend!r})")
        integrate_tsdf(work, [int(i) for i in idx], P)  # writes engine_native.ply
        mesh = o3d.io.read_triangle_mesh(native)
    fuse_s = time.monotonic() - t_fuse
    if len(mesh.vertices) == 0:
        raise RuntimeError("open3d produced an empty mesh")

    timings = dict(fuse=round(fuse_s, 3), reconstruct=round(time.monotonic() - t0, 3))
    with open(os.path.join(work, "stage_timings.json"), "w") as f:
        json.dump(timings, f, indent=2, sort_keys=True)
        f.write("\n")
    _log(work, f"[open3d] timings {json.dumps(timings, sort_keys=True)}")
    return native, poses, [int(i) for i in idx], "zed_opencv"


def _child_main(argv):
    """Tensor VBG fusion under CUDA_VISIBLE_DEVICES=<leased GPU> (set by the parent)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--work", required=True)
    ap.add_argument("--lease-fd", type=int, default=None)
    args = ap.parse_args(argv)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        sys.exit("CUDA_VISIBLE_DEVICES must be set by the parent")
    lease_file = os.fdopen(args.lease_fd, "r") if args.lease_fd is not None else None
    import cv2
    import open3d as o3d
    if not o3d.core.cuda.is_available():
        sys.exit(f"open3d CUDA unavailable (CUDA_VISIBLE_DEVICES="
                 f"{os.environ['CUDA_VISIBLE_DEVICES']})")
    work = args.work
    frames = os.path.join(work, "frames")
    idx = np.load(os.path.join(work, "frame_indices.npy"))
    poses = np.load(os.path.join(work, "open3d", "b1_poses.npy"))
    K = np.loadtxt(os.path.join(frames, "intrinsic_depth.txt"))[:3, :3]
    dh, dw = cv2.imread(os.path.join(frames, "color", f"{idx[0]}.jpg")).shape[:2]
    mesh = _integrate_tensor(frames, idx, poses, K, dw, dh, o3d.core.Device("CUDA:0"))
    if not o3d.io.write_triangle_mesh(os.path.join(work, "engine_native.ply"), mesh,
                                      write_vertex_colors=True):
        sys.exit("cannot write engine_native.ply")
    if lease_file is not None:
        lease_file.close()


if __name__ == "__main__":
    _child_main(sys.argv[1:])
