"""Stage C -- ScanNet artifacts: gravity z-up alignment, _vh_clean / _vh_clean_2 meshes via
pymeshlab running ScanNet's own .mlx scripts, axisAlignment into <id>.txt, Segmentator.

Frames: mesh + poses are transformed ONCE from the engine-native frame into the ScanNet
output frame (gravity z-up, floor ~0). axisAlignment is computed and written, NOT applied
(verified: released ScanNet meshes are raw-frame z-up with arbitrary floor/wall azimuth;
consumers apply axisAlignment).
"""
import os
import subprocess

import numpy as np

from . import MESH_CLEAN_MLX, SEGMENTATOR_DIR


def zup_transform(poses, gravity=None):
    """4x4 mapping engine frame -> ScanNet frame (z up, floor min-z = 0, xy centered).
    Up from ZED gravity if given, else averaged camera up (ScanNet alignment.h:82-107)."""
    if gravity is not None and np.linalg.norm(gravity) > 1e-6:
        up = gravity / np.linalg.norm(gravity)
    else:
        ups = np.stack([p[:3, 1] for p in poses])  # camera up in world
        up = -np.mean(ups, axis=0)
        up /= np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    s = np.linalg.norm(v)
    c = float(np.dot(up, z))
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def apply_transform(mesh, poses, T):
    """Apply 4x4 T to mesh vertices in place and to each pose (conjugation)."""
    if mesh is not None and len(np.asarray(mesh.vertices)):
        mesh.transform(T)
    return np.stack([T @ p for p in poses])


def floor_zero_and_center(mesh, poses):
    """Translate mesh+poses so floor min-z = 0 and x/y centered (ScanNet alignment.h:258-286).
    One combined transform, applied identically to mesh and poses."""
    pts = np.asarray(mesh.vertices)
    dz = -float(np.percentile(pts[:, 2], 1))
    cx = -(pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = -(pts[:, 1].min() + pts[:, 1].max()) / 2
    T = np.eye(4)
    T[:3, 3] = [cx, cy, dz]
    mesh.transform(T)
    return np.stack([T @ p for p in poses])


def axis_alignment_matrix(mesh_path):
    """Pure z-rotation + translation mapping walls to x/y axes and floor to z=0."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    horiz = np.abs(N[:, 2]) < 0.2
    if horiz.sum() < 100:
        raise RuntimeError("too few wall-normal vertices for axisAlignment")
    az = np.degrees(np.arctan2(N[horiz, 1], N[horiz, 0]))
    hist, edges = np.histogram(az, bins=360, range=(-180, 180))
    dom = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    # rotate dominant wall normal to +x
    theta = np.radians(-dom)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    t = np.zeros(3)
    t[2] = -float(np.percentile(V[:, 2], 1))
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = t
    assert abs(np.linalg.det(A) - 1.0) < 1e-9 and abs(A[2, 2] - 1.0) < 1e-9
    return A


def _clean_block(meshset, min_component):
    """ScanNet's clean filter chain (clean.mlx / simplify.mlx clean block) via the
    pymeshlab 2025.7 filter names. .mlx loading crashes in this pymeshlab version
    (load_filter_script "something really bad happened"), so the documented-equivalent
    chain is applied directly. Merge Close Vertices' threshold is a PercentageValue
    (% of the mesh bbox diagonal), so it is computed per mesh to reproduce ScanNet's
    absolute 1.0689 mm exactly."""
    import pymeshlab as ml
    diag = float(meshset.current_mesh().bounding_box().diagonal())
    merge_pct = 0.0010689 / diag * 100.0
    meshset.meshing_merge_close_vertices(threshold=ml.PercentageValue(merge_pct))
    meshset.meshing_remove_duplicate_faces()
    meshset.meshing_remove_connected_component_by_face_number(mincomponentsize=min_component)
    meshset.meshing_remove_unreferenced_vertices()


def _decimate(meshset, target_faces=300000):
    import pymeshlab as ml
    meshset.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces, targetperc=0, qualitythr=0.3, preserveboundary=False,
        boundaryweight=1, preservenormal=False, preservetopology=False,
        optimalplacement=True, planarquadric=False, qualityweight=False, autoclean=True,
        selected=False)
    _clean_block(meshset, 1000)


def align_mesh(mesh_src, poses, gravity, work_dir, convention="zed_y_up"):
    """Transform engine-native mesh+poses into the ScanNet output frame once.
    ZED gravity is only meaningful in ZED-frame conventions; others fall back to the
    averaged-camera-up estimate (ScanNet alignment.h:82-107).
    Returns (aligned_mesh_path, aligned_poses)."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(mesh_src)
    g = gravity
    if convention == "zed_opencv":  # world was conjugated diag(1,-1,-1,1)
        g = np.diag([1.0, -1.0, -1.0]) @ gravity
    elif convention not in ("zed_y_up", "y_up"):
        g = None
    T = zup_transform(poses, g)
    poses = apply_transform(mesh, poses, T)
    poses = floor_zero_and_center(mesh, poses)
    aligned = os.path.join(work_dir, "mesh_aligned.ply")
    o3d.io.write_triangle_mesh(aligned, mesh)
    return aligned, poses


def _write_ply_float(mesh, path, with_normals):
    """binary_little_endian PLY with float xyz (ScanNet format; pymeshlab/open3d write
    double, which the Segmentator's tinyply cannot read)."""
    import struct
    V = np.asarray(mesh.vertices, dtype=np.float32)
    T = np.asarray(mesh.triangles, dtype=np.uint32)
    C = np.asarray(mesh.vertex_colors)
    N = np.asarray(mesh.vertex_normals, dtype=np.float32) if with_normals else None
    if N is not None and N.shape[0] != len(V):
        N = None
    elif N is not None and not np.any(N):
        N = None
    if C is not None and C.shape and C.shape[1] >= 3:
        rgba = (np.clip(C[:, :3], 0, 1) * 255).astype(np.uint8)
        rgba = np.hstack([rgba, np.full((len(rgba), 1), 255, np.uint8)])
    else:
        rgba = np.full((len(V), 4), 255, np.uint8)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n"
                b"comment VCGLIB generated\n")
        f.write(f"element vertex {len(V)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        if N is not None:
            f.write(b"property float nx\nproperty float ny\nproperty float nz\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                b"property uchar alpha\n")
        f.write(f"element face {len(T)}\n".encode())
        f.write(b"property list uchar int vertex_indices\nend_header\n")
        for i in range(len(V)):
            f.write(struct.pack("<3f", *V[i]))
            if N is not None:
                f.write(struct.pack("<3f", *N[i]))
            f.write(struct.pack("<4B", *rgba[i]))
        for tri in T:
            f.write(struct.pack("<B3I", 3, *tri))


def make_meshes(mesh_aligned, scan_root, scan_id, out_faces=300000):
    """Aligned engine mesh (PLY, colors) -> <id>_vh_clean.ply (normals) + <id>_vh_clean_2.ply
    (no normals, out_faces faces). ScanNet's clean.mlx + simplify params, only the target
    changed from relative 20% to absolute face count (documented deviation)."""
    import pymeshlab as ml
    import open3d as o3d
    clean_path = os.path.join(scan_root, f"{scan_id}_vh_clean.ply")
    clean2_path = os.path.join(scan_root, f"{scan_id}_vh_clean_2.ply")
    pml_path1 = os.path.join(scan_root, "recon", "_pml_clean.ply")
    pml_path2 = os.path.join(scan_root, "recon", "_pml_clean2.ply")
    os.makedirs(os.path.dirname(pml_path1), exist_ok=True)

    ms = ml.MeshSet()
    ms.load_new_mesh(mesh_aligned)
    _clean_block(ms, 7500)
    ms.compute_normal_per_vertex()
    ms.save_current_mesh(pml_path1, save_vertex_color=True, save_vertex_normal=True,
                         binary=True)
    m = o3d.io.read_triangle_mesh(pml_path1)
    _write_ply_float(m, clean_path, with_normals=True)

    ms.load_new_mesh(mesh_aligned)
    _clean_block(ms, 7500)
    _decimate(ms, out_faces)
    ms.save_current_mesh(pml_path2, save_vertex_color=True, binary=True)
    m2 = o3d.io.read_triangle_mesh(pml_path2)
    _write_ply_float(m2, clean2_path, with_normals=False)
    return clean_path, clean2_path


def run_segmentator(mesh_path):
    """Segmentator defaults (kThresh 0.01, segMinVerts 20) -> <base>.0.010000.segs.json."""
    binary = os.path.join(SEGMENTATOR_DIR, "segmentator")
    if not os.path.exists(binary):
        subprocess.run(["make", "-C", SEGMENTATOR_DIR], check=True,
                       capture_output=True, timeout=300)
    subprocess.run([binary, mesh_path], check=True, capture_output=True, timeout=3600)
    return mesh_path.rsplit(".ply", 1)[0] + ".0.010000.segs.json"
