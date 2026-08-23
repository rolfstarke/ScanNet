"""One-time export of scene9004 CAD to /data/scannet/custom/reference/scene9004_cad.ply."""
import hashlib
import os
import sys
import tempfile

import numpy as np
import open3d as o3d
import rhino3dm
import yaml

CAD = "/data/08_TestEnvironments/06_BTU_LG2C_R312_order/02_3D-Model/LG2C_Raum312_v2.3dm"
OUT_DIR = "/data/scannet/custom/reference"
OUT_PLY = os.path.join(OUT_DIR, "scene9004_cad.ply")
CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "reconstruct", "geometry_reference.yaml")
GLASS_LAYER = "Fensterglas / Griff"


def bbox_extent(obj):
    bb = obj.GetBoundingBox()
    if bb is None:
        return None
    mn = np.array([bb.Min.X, bb.Min.Y, bb.Min.Z], dtype=np.float64)
    mx = np.array([bb.Max.X, bb.Max.Y, bb.Max.Z], dtype=np.float64)
    return mx - mn


def is_glass(obj):
    e = bbox_extent(obj)
    if e is None:
        return False
    return float(e.min()) <= 0.005 and float(e.max()) >= 0.5


def mesh_from_rhino(rmesh):
    V = np.array([[v.X, v.Y, v.Z] for v in rmesh.Vertices], dtype=np.float64)
    F = []
    for f in rmesh.Faces:
        if len(f) == 3:
            F.append([f[0], f[1], f[2]])
        elif len(f) == 4:
            F.append([f[0], f[1], f[2]])
            F.append([f[0], f[2], f[3]])
    if not F:
        return None
    F = np.array(F, dtype=np.int32)
    # drop zero-area
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    F = F[area > 1e-12]
    if len(F) == 0:
        return None
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    return m


def main():
    assert os.path.isfile(CAD), CAD
    h = hashlib.sha256(open(CAD, "rb").read()).hexdigest()
    expected = "ff00c139adededf46f94f69c7872a566f2da2617224305b6dd8583d5dd04994a"
    if h != expected:
        sys.exit(f"CAD hash mismatch: {h}")

    model = rhino3dm.File3dm.Read(CAD)
    layers = {l.Index: l.Name for l in model.Layers}
    meshes = []
    glass_n = hard_n = curve_n = subd_n = 0
    wall_faces = []

    for obj in model.Objects:
        geom = obj.Geometry
        attr = obj.Attributes
        lname = layers.get(attr.LayerIndex, "")
        t = geom.ObjectType

        if t in (rhino3dm.ObjectType.Curve,):
            if lname == GLASS_LAYER:
                curve_n += 1
            continue
        if t == rhino3dm.ObjectType.SubD:
            subd_n += 1
            continue

        if lname == GLASS_LAYER:
            if is_glass(geom):
                glass_n += 1
                continue
            hard_n += 1

        rmeshes = []
        if t == rhino3dm.ObjectType.Brep:
            for face in geom.Faces:
                rm = face.GetMesh(rhino3dm.MeshType.Any)
                if rm is not None:
                    rmeshes.append(rm)
        elif t == rhino3dm.ObjectType.Extrusion:
            rm = geom.GetMesh(rhino3dm.MeshType.Any)
            if rm is not None:
                rmeshes.append(rm)
        elif t == rhino3dm.ObjectType.Mesh:
            rmeshes.append(geom)
        else:
            continue

        for rm in rmeshes:
            m = mesh_from_rhino(rm)
            if m is None or m.is_empty():
                continue
            meshes.append(m)
            if "Wand StB" in lname:
                wall_faces.append(m)

    if glass_n != 18 or hard_n != 50 or curve_n != 8:
        sys.exit(f"glass/hardware assert failed: glass={glass_n} hard={hard_n} curves={curve_n}")
    if subd_n != 1:
        print(f"[warn] SubD count={subd_n} (expected 1)")
    if not meshes:
        sys.exit("no meshes exported")

    merged = meshes[0]
    for m in meshes[1:]:
        merged += m
    merged.remove_duplicated_vertices()
    merged.remove_degenerate_triangles()
    merged.remove_unreferenced_vertices()
    V = np.asarray(merged.vertices)

    # Synthetic floor from wall bounds
    mn, mx = V.min(axis=0), V.max(axis=0)
    # Prefer wall-layer footprint if available
    if wall_faces:
        WV = np.vstack([np.asarray(w.vertices) for w in wall_faces])
        wmn, wmx = WV.min(axis=0), WV.max(axis=0)
        x0, x1 = wmn[0], wmx[0]
        y0, y1 = wmn[1], wmx[1]
    else:
        x0, x1, y0, y1 = mn[0], mx[0], mn[1], mx[1]
    z0 = 0.0
    floor_V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], dtype=np.float64)
    floor_F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    floor = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(floor_V), o3d.utility.Vector3iVector(floor_F))
    merged += floor

    # Normalize: floor z=0 (already), x/y centre at 0
    V = np.asarray(merged.vertices)
    cx = 0.5 * (V[:, 0].min() + V[:, 0].max())
    cy = 0.5 * (V[:, 1].min() + V[:, 1].max())
    T = np.eye(4)
    T[0, 3] = -cx
    T[1, 3] = -cy
    T[2, 3] = -float(V[:, 2].min())
    merged.transform(T)
    V = np.asarray(merged.vertices)
    if not np.isfinite(V).all():
        sys.exit("non-finite CAD vertices")

    os.makedirs(OUT_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".cad_", suffix=".ply", dir=OUT_DIR)
    os.close(fd)
    o3d.io.write_triangle_mesh(tmp, merged, write_ascii=False, write_vertex_colors=False)
    check = o3d.io.read_triangle_mesh(tmp)
    if check.is_empty():
        os.unlink(tmp)
        sys.exit("CAD reload empty")
    os.replace(tmp, OUT_PLY)
    sha = hashlib.sha256(open(OUT_PLY, "rb").read()).hexdigest()

    mn, mx = np.asarray(check.vertices).min(0), np.asarray(check.vertices).max(0)
    half = 0.5 * (mx - mn)
    origin = np.floor(mn / 0.01) * 0.01 - 1.0
    bmin = mn - np.array([25.0, 25.0, 5.0])
    bmax = mx + np.array([25.0, 25.0, 5.0])

    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    sc = cfg["scenes"]["scene9004"]
    sc["reference_sha256"] = sha
    sc["grid_origin_m"] = [float(x) for x in origin]
    sc["bounds_min_m"] = [float(x) for x in bmin]
    sc["bounds_max_m"] = [float(x) for x in bmax]
    sc["cad_half_extents_m"] = [float(half[0]), float(half[1])]
    sc["room_centre_m"] = [0.0, 0.0, 0.0]
    sc["glass_excluded"] = glass_n
    sc["hardware_retained"] = hard_n
    sc["curves_omitted"] = curve_n
    with open(CFG, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[cad] wrote {OUT_PLY} tris={len(check.triangles)} sha={sha[:12]}…")
    print(f"[cad] bounds={mn}..{mx} glass={glass_n} hardware={hard_n}")


if __name__ == "__main__":
    main()
