"""Metashape 2.3.1 engine. Weak ZED position priors (0.15 m, rotation disabled, fixed
calibration f=fx, b1=fy-fx, center-relative cx/cy) + official API defaults (matchPhotos
downscale=1 / keypoint_limit=40000 / tiepoint_limit=4000, alignCameras
adaptive_fitting=False, buildDepthMaps downscale=2 MildFiltering, buildModel HighFaceCount
from DepthMapsData, optimizeCameras adaptive_fitting=True).

The Metashape script is generated here and executed headless by
/data/zed-metashape/conda/env/bin/python (licensed Pro Python API, exactly the old repo's
invocation pattern: the env's python runs the pipeline code; no metashape.sh, no ZED SDK
involved, so no LD_PRELOAD redirect needed). Contract:
reconstruct(work, root, gpu=None) -> (engine_native.ply, poses (M,4,4), keep, "y_up").

Poses = chunk.transform.matrix @ camera.transform, rotation columns orthonormalized (SVD);
the chunk similarity scale s is a global property of the mesh, so vertices and pose
translations are divided by s: rigid exported poses and the mesh share one real metric frame
(the old pipeline shipped the scale baked into the rotation columns). No depth is produced
here -- Stage A depth is used downstream. No caching, no swallowed exceptions.
"""
import glob
import os
import shutil
import subprocess

import numpy as np

METASHAPE_PY = "/data/zed-metashape/conda/env/bin/python"
MIN_TRANSLATION_M = 0.08
MIN_ROTATION_DEG = 8.0
MAX_GAP_FRAMES = 45
MAX_KEYFRAMES = 400
TIMEOUT_S = 5400

_SCRIPT = '''\
import glob
import os

import numpy as np

import Metashape

WORK = "@WORK@"
GPU = @GPU@
LOC = @LOC@
FRAMES = os.path.join(WORK, "frames")
KEYFRAMES = os.path.join(WORK, "keyframes")


def to_np(m):
    return np.array([[float(m[i, j]) for j in range(4)] for i in range(4)])


def main():
    if not Metashape.License().valid:
        raise RuntimeError("Metashape license invalid")
    devices = Metashape.app.enumGPUDevices()
    if devices:
        mask = (1 << GPU) if 0 <= GPU < len(devices) else (1 << len(devices)) - 1
        Metashape.app.gpu_mask = mask
    Metashape.app.cpu_enable = True

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

    chunk.camera_location_accuracy = Metashape.Vector([LOC[0], LOC[1], LOC[2]])
    for cam in chunk.cameras:
        idx = int(os.path.splitext(os.path.basename(cam.label))[0])
        pose = np.loadtxt(os.path.join(KEYFRAMES, "pose", "%d.txt" % idx))
        cam.reference.location = Metashape.Vector(pose[:3, 3].tolist())
        cam.reference.location_accuracy = Metashape.Vector([LOC[0], LOC[1], LOC[2]])
        cam.reference.location_enabled = True
        cam.reference.rotation_enabled = False

    match_kwargs = dict(downscale=1, generic_preselection=True,
                        reference_preselection=True, keypoint_limit=40000,
                        tiepoint_limit=4000)
    if hasattr(Metashape, "ReferencePreselectionSource"):
        try:
            chunk.matchPhotos(
                reference_preselection_mode=Metashape.ReferencePreselectionSource,
                **match_kwargs)
        except TypeError:
            chunk.matchPhotos(**match_kwargs)
    else:
        chunk.matchPhotos(**match_kwargs)
    chunk.alignCameras(adaptive_fitting=False)

    aligned = [c for c in chunk.cameras
               if c.type == Metashape.Camera.Type.Regular and c.transform is not None]
    if len(aligned) < 0.9 * n:
        raise RuntimeError("only %d/%d cameras aligned" % (len(aligned), n))

    chunk.optimizeCameras(adaptive_fitting=True)

    chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.MildFiltering)
    chunk.buildModel(source_data=Metashape.DepthMapsData,
                     face_count=Metashape.HighFaceCount,
                     interpolation=Metashape.EnabledInterpolation)
    if chunk.model is None:
        raise RuntimeError("buildModel produced no model")

    mesh_path = os.path.join(WORK, "engine_native.ply")
    try:
        chunk.exportModel(mesh_path, format=Metashape.ModelFormatPLY, binary=True,
                          save_texture=False, save_vertex_colors=True)
    except TypeError:
        chunk.exportModel(mesh_path, format=Metashape.ModelFormatPLY, binary=True)

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
        rows.append((int(os.path.splitext(os.path.basename(cam.label))[0]), P))
        scales.append(float(S.mean()))
    if not rows:
        raise RuntimeError("no aligned cameras to export")
    rows.sort(key=lambda r: r[0])
    poses = np.stack([r[1] for r in rows])
    keep = np.array([r[0] for r in rows], dtype=np.int64)
    scale = float(np.median(scales))
    np.savez(os.path.join(WORK, "metashape_result.npz"),
             poses=poses, keep=keep, scale=np.array([scale]))
    print("[metashape] aligned %d/%d cameras, scale %.6f" % (len(rows), n, scale),
          flush=True)


main()
'''


def _select_keyframes(pose_dir):
    paths = sorted(glob.glob(os.path.join(pose_dir, "*.txt")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not paths:
        raise RuntimeError(f"no pose files in {pose_dir}")
    poses = np.stack([np.loadtxt(p) for p in paths])
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
    if len(selected) > MAX_KEYFRAMES:
        idxs = sorted({int(round(x)) for x in np.linspace(0, len(selected) - 1,
                                                          num=MAX_KEYFRAMES)})
        if 0 not in idxs:
            idxs = [0] + idxs
        if idxs[-1] != len(selected) - 1:
            idxs.append(len(selected) - 1)
        selected = [selected[i] for i in idxs]
    return selected


def reconstruct(work, root, gpu=None):
    work = os.path.abspath(work)
    frames = os.path.join(work, "frames")
    keyframes = os.path.join(work, "keyframes")
    logs = os.path.join(work, "logs")
    for d in (keyframes, os.path.join(keyframes, "color"),
              os.path.join(keyframes, "pose"), logs):
        os.makedirs(d, exist_ok=True)

    keep_in = _select_keyframes(os.path.join(frames, "pose"))
    for i in keep_in:
        shutil.copy2(os.path.join(frames, "color", f"{i}.jpg"),
                     os.path.join(keyframes, "color", f"{i}.jpg"))
        shutil.copy2(os.path.join(frames, "pose", f"{i}.txt"),
                     os.path.join(keyframes, "pose", f"{i}.txt"))

    if not os.path.exists(METASHAPE_PY):
        raise RuntimeError(f"Metashape env python not found: {METASHAPE_PY}")

    script = (_SCRIPT
              .replace("@WORK@", work)
              .replace("@GPU@", str(gpu) if gpu is not None else "-1")
              .replace("@LOC@", "(0.15, 0.15, 0.15)"))
    script_path = os.path.join(work, "metashape_run.py")
    with open(script_path, "w") as f:
        f.write(script)

    log_path = os.path.join(logs, "metashape.log")
    with open(log_path, "wb") as logf:
        try:
            subprocess.run([METASHAPE_PY, script_path], check=True, timeout=TIMEOUT_S,
                           stdout=logf, stderr=subprocess.STDOUT, env=os.environ.copy(),
                           cwd=work)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"metashape run timed out after {TIMEOUT_S}s, see {log_path}")
        except subprocess.CalledProcessError as e:
            tail = open(log_path).read().splitlines()[-20:]
            raise RuntimeError(f"metashape run failed rc={e.returncode}, see {log_path}\n"
                               + "\n".join(tail))

    result_path = os.path.join(work, "metashape_result.npz")
    if not os.path.exists(result_path):
        raise RuntimeError(f"metashape finished without {result_path}, see {log_path}")
    result = np.load(result_path)
    poses = result["poses"]
    keep = [int(k) for k in result["keep"]]
    scale = float(result["scale"][0])
    if not keep or len(poses) != len(keep):
        raise RuntimeError(f"metashape result inconsistent: {len(poses)} poses, "
                           f"{len(keep)} keep")
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"invalid chunk scale {scale}")
    poses[:, :3, 3] /= scale

    mesh_path = os.path.join(work, "engine_native.ply")
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(np.asarray(mesh.vertices)) == 0:
        raise RuntimeError(f"empty mesh at {mesh_path}")
    if not mesh.has_vertex_colors():
        raise RuntimeError(f"mesh at {mesh_path} has no vertex colors")
    mesh.scale(1.0 / scale, center=(0.0, 0.0, 0.0))
    o3d.io.write_triangle_mesh(mesh_path, mesh, write_vertex_colors=True)
    return mesh_path, poses, keep, "y_up"
