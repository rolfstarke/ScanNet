"""CAD-referenced Chamfer-L1 scorer for scene9004 reconstructions.

Scores <scan_id>_vh_clean_2.ply against a fixed CAD visibility mask using symmetric
nearest-neighbour mean distance on 10 mm surface voxels (cm, lower better). CPU-only;
never runs inside a managed GPU lease. Standalone:

    python -m spellbook.reconstruct.geometry_score --scan-id scene9004_40
"""
import argparse
import hashlib
import os
import tempfile

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

from . import SCANS_DIR

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "geometry_reference.yaml")
METRIC = "observed_surface_voxel_chamfer_l1_v1"


def load_config(path=None):
    path = path or DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = path
    return cfg


def scene_config(cfg, scene_num=9004):
    key = f"scene{int(scene_num):04d}"
    if key not in cfg.get("scenes", {}):
        raise KeyError(f"no geometry reference for {key}")
    return cfg["scenes"][key]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_axis_alignment(txt_path):
    with open(txt_path) as f:
        for line in f:
            if line.startswith("axisAlignment"):
                vals = [float(x) for x in line.split("=")[1].split()]
                if len(vals) != 16:
                    raise ValueError(f"bad axisAlignment in {txt_path}")
                A = np.array(vals, dtype=np.float64).reshape(4, 4)
                det = np.linalg.det(A[:3, :3])
                if abs(det - 1.0) > 0.05:
                    raise ValueError(f"axisAlignment det={det:.4f} (expected ~1)")
                return A
    return np.eye(4)


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise RuntimeError(f"empty mesh: {path}")
    mesh.compute_vertex_normals()
    if not mesh.has_triangles():
        raise RuntimeError(f"non-triangular mesh: {path}")
    V = np.asarray(mesh.vertices)
    if not np.isfinite(V).all():
        raise RuntimeError(f"non-finite vertices: {path}")
    return mesh


def transform_mesh(mesh, T):
    if hasattr(mesh, "clone"):
        out = mesh.clone()
    else:
        out = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy()),
            o3d.utility.Vector3iVector(np.asarray(mesh.triangles).copy()))
    out.transform(np.asarray(T, dtype=np.float64))
    return out


def _voxel_size_m(cfg):
    return float(cfg["voxel_mm"]) / 1000.0


def surface_voxels(mesh, origin, voxel_m, bounds_min=None, bounds_max=None, max_voxels=None):
    """Return sorted (N,3) int32 voxel indices on a fixed grid origin.

    Triangle-surface sampling (deterministic barycentric) rather than Open3D's slower
    mesh voxelizer; density is at least one sample per voxel along each edge.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    if len(F) == 0:
        raise RuntimeError("surface voxelization produced zero cells")
    origin = np.asarray(origin, dtype=np.float64)
    if bounds_min is not None and bounds_max is not None:
        lo = np.asarray(bounds_min, dtype=np.float64)
        hi = np.asarray(bounds_max, dtype=np.float64)
        if (V < lo - 1e-9).any() or (V > hi + 1e-9).any():
            raise RuntimeError("mesh vertices outside configured evaluation bounds")
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    edge = np.maximum.reduce([
        np.linalg.norm(b - a, axis=1),
        np.linalg.norm(c - b, axis=1),
        np.linalg.norm(a - c, axis=1),
    ])
    # Group triangles by required subdivision level for vectorized sampling.
    levels = np.clip(np.ceil(edge / max(voxel_m, 1e-9)).astype(np.int32), 1, 64)
    chunks = [V]
    for k in np.unique(levels):
        sel = F[levels == k]
        aa, bb, cc = V[sel[:, 0]], V[sel[:, 1]], V[sel[:, 2]]
        if k == 1:
            chunks.extend([aa, bb, cc, (aa + bb + cc) / 3.0])
            continue
        u = (np.arange(k, dtype=np.float64) + 0.5) / k
        uu, vv = np.meshgrid(u, u, indexing="xy")
        mask = (uu + vv) <= 1.0 + 1e-12
        wu, wv = uu[mask], vv[mask]
        ww = 1.0 - wu - wv
        # (T,S,3)
        p = (aa[:, None, :] * ww[None, :, None]
             + bb[:, None, :] * wu[None, :, None]
             + cc[:, None, :] * wv[None, :, None])
        chunks.append(p.reshape(-1, 3))
    pts = np.concatenate(chunks, axis=0)
    idx = np.floor((pts - origin) / voxel_m).astype(np.int32)
    idx = np.unique(idx, axis=0)
    if idx.size == 0:
        raise RuntimeError("surface voxelization produced zero cells")
    if max_voxels is not None and len(idx) > max_voxels:
        raise RuntimeError(f"surface voxels {len(idx)} exceed cap {max_voxels}")
    order = np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))
    return idx[order]


def voxel_centres(idx, origin, voxel_m):
    origin = np.asarray(origin, dtype=np.float64)
    return origin + (np.asarray(idx, dtype=np.float64) + 0.5) * voxel_m


def preliminary_voxels(mesh, voxel_m):
    """Local-origin preliminary pass for robust floor/centre only."""
    V = np.asarray(mesh.vertices)
    lo = np.floor(V.min(axis=0) / voxel_m) * voxel_m - voxel_m
    return surface_voxels(mesh, lo, voxel_m)


def yaw_matrix(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    return T


def chamfer_l1_cm(ref_centres, recon_centres):
    """Symmetric Chamfer-L1 in cm: mean of recon→CAD and CAD→recon nearest means."""
    if len(ref_centres) == 0 or len(recon_centres) == 0:
        raise RuntimeError("empty cell set in Chamfer-L1")
    t_ref = cKDTree(ref_centres)
    t_rec = cKDTree(recon_centres)
    d_rec, _ = t_ref.query(recon_centres, k=1, workers=-1)
    d_ref, _ = t_rec.query(ref_centres, k=1, workers=-1)
    accuracy_mean_m = float(np.mean(d_rec))
    completeness_mean_m = float(np.mean(d_ref))
    chamfer_l1_m = 0.5 * (accuracy_mean_m + completeness_mean_m)
    chamfer_l1 = 100.0 * chamfer_l1_m
    accuracy_mean = 100.0 * accuracy_mean_m
    completeness_mean = 100.0 * completeness_mean_m
    if not np.isfinite(chamfer_l1) or chamfer_l1 < 0:
        raise RuntimeError(f"invalid chamfer_l1_cm {chamfer_l1}")
    return chamfer_l1, accuracy_mean, completeness_mean


def gauge_normalize(mesh, cfg, scene_cfg, cad_centres):
    """Floor/centre + four-yaw selection minimising Chamfer-L1 vs CAD surface cells."""
    voxel_m = _voxel_size_m(cfg)
    V = np.asarray(mesh.vertices)
    floor_z = float(np.percentile(V[:, 2], 1))
    T_floor = np.eye(4)
    T_floor[2, 3] = -floor_z
    mesh_f = transform_mesh(mesh, T_floor)
    Vf = np.asarray(mesh_f.vertices)
    med = np.median(Vf[:, :2], axis=0)
    half = np.asarray(scene_cfg.get("cad_half_extents_m") or [7.0, 5.0], dtype=np.float64)
    if half.size < 2:
        half = np.array([7.0, 5.0])
    keep = (np.abs(Vf[:, 0] - med[0]) <= half[0] + 1.0) & (np.abs(Vf[:, 1] - med[1]) <= half[1] + 1.0)
    if keep.sum() < 100:
        keep = np.ones(len(Vf), dtype=bool)
    sel = Vf[keep]
    cx = 0.5 * (np.percentile(sel[:, 0], 1) + np.percentile(sel[:, 0], 99))
    cy = 0.5 * (np.percentile(sel[:, 1], 1) + np.percentile(sel[:, 1], 99))
    room = np.asarray(scene_cfg.get("room_centre_m") or [0.0, 0.0, 0.0], dtype=np.float64)
    T_c = np.eye(4)
    T_c[0, 3] = room[0] - cx
    T_c[1, 3] = room[1] - cy
    mesh_c = transform_mesh(mesh_f, T_c)
    origin = np.asarray(scene_cfg["grid_origin_m"], dtype=np.float64)
    bounds_min = scene_cfg.get("bounds_min_m")
    bounds_max = scene_cfg.get("bounds_max_m")
    step = max(1, len(cad_centres) // 80000)
    cad_yaw = cad_centres[::step]
    best = None
    for yaw in (0, 90, 180, 270):
        Ty = yaw_matrix(yaw)
        m = transform_mesh(mesh_c, Ty)
        try:
            idx = surface_voxels(m, origin, voxel_m, bounds_min, bounds_max,
                                 cfg.get("max_surface_voxels"))
        except RuntimeError:
            continue
        centres_r = voxel_centres(idx, origin, voxel_m)
        rec_yaw = centres_r[::max(1, len(centres_r) // 80000)]
        ch, _, _ = chamfer_l1_cm(cad_yaw, rec_yaw)
        if best is None or ch < best[0] - 1e-15 or (abs(ch - best[0]) <= 1e-15 and yaw < best[1]):
            best = (ch, yaw, m, idx, centres_r, Ty @ T_c @ T_floor)
    if best is None:
        raise RuntimeError("gauge normalization failed for all yaw candidates")
    return best[2], best[3], best[4], best[1], best[5]


def _density_rgba(H, rgb, scale, alpha_cap=0.85):
    """Map 2D counts to RGBA; empty bins fully transparent."""
    a = np.clip(np.sqrt(np.maximum(H, 0.0)) / scale, 0.0, 1.0) * alpha_cap
    out = np.zeros(H.shape + (4,), dtype=np.float64)
    out[..., 0] = rgb[0]
    out[..., 1] = rgb[1]
    out[..., 2] = rgb[2]
    out[..., 3] = a
    return out


def _composite_over(base_rgb, layer_rgba):
    """Porter-Duff over: layer onto opaque RGB base."""
    a = layer_rgba[..., 3:4]
    return base_rgb * (1.0 - a) + layer_rgba[..., :3] * a


def render_comparison(path, ref_centres, recon_centres, scene_id, chamfer_l1,
                      accuracy_mean, completeness_mean):
    """Two-cloud projected density overlay (CAD orange, recon blue)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cad_rgb = (1.0, 0.45, 0.0)
    rec_rgb = (0.15, 0.35, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dims, label in ((axes[0], (0, 1), "plan x/y"), (axes[1], (0, 2), "elevation x/z")):
        pts = np.vstack([ref_centres[:, list(dims)], recon_centres[:, list(dims)]])
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        pad = np.maximum(0.1, 0.02 * (hi - lo))
        lo = lo - pad
        hi = hi + pad
        extent_xy = hi - lo
        long = float(max(extent_xy[0], extent_xy[1], 1e-6))
        bin_m = max(0.02, long / 400.0)
        nx = max(8, int(np.ceil(extent_xy[0] / bin_m)))
        ny = max(8, int(np.ceil(extent_xy[1] / bin_m)))
        rng = [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]]
        H_cad, xe, ye = np.histogram2d(
            ref_centres[:, dims[0]], ref_centres[:, dims[1]], bins=[nx, ny], range=rng)
        H_rec, _, _ = np.histogram2d(
            recon_centres[:, dims[0]], recon_centres[:, dims[1]], bins=[xe, ye])
        scale = float(np.sqrt(max(H_cad.max(), H_rec.max(), 1.0)))
        white = np.ones((H_cad.shape[0], H_cad.shape[1], 3), dtype=np.float64)
        img = _composite_over(white, _density_rgba(H_cad, cad_rgb, scale))
        img = _composite_over(img, _density_rgba(H_rec, rec_rgb, scale))
        # histogram2d first dim = x rows; imshow expects rows = y → transpose
        ax.imshow(np.clip(img.transpose(1, 0, 2), 0, 1), origin="lower",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest",
                  aspect="equal")
        ax.set_title(label)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
    fig.legend(
        [Patch(facecolor=cad_rgb, edgecolor="none", label="CAD"),
         Patch(facecolor=rec_rgb, edgecolor="none", label="recon")],
        ["CAD", "recon"], loc="lower center", ncol=2)
    fig.suptitle(
        f"{scene_id}  chamfer_l1_cm={chamfer_l1:.2f}  "
        f"(acc={accuracy_mean:.2f}  comp={completeness_mean:.2f})")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def atomic_write_bytes(path, data):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def score_scan(scan_id, config_path=None):
    cfg = load_config(config_path)
    if cfg.get("metric") != METRIC:
        raise RuntimeError(f"config metric {cfg.get('metric')!r} != {METRIC!r}")
    scene_num = int(scan_id[5:9])
    sc = scene_config(cfg, scene_num)
    if sc.get("reference_sha256") is None or sc.get("visible_voxels_sha256") is None:
        raise RuntimeError("geometry reference not finalized; build CAD/visibility first")
    cad_path = sc["cad_ply"]
    vis_path = sc["visible_voxels"]
    if file_sha256(cad_path) != sc["reference_sha256"]:
        raise RuntimeError("CAD hash mismatch against geometry_reference.yaml")
    if file_sha256(vis_path) != sc["visible_voxels_sha256"]:
        raise RuntimeError("visible voxels hash mismatch against geometry_reference.yaml")

    root = os.path.join(SCANS_DIR, scan_id)
    mesh_path = os.path.join(root, f"{scan_id}_vh_clean_2.ply")
    txt_path = os.path.join(root, f"{scan_id}.txt")
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(mesh_path)
    mesh = load_mesh(mesh_path)
    A = parse_axis_alignment(txt_path) if os.path.isfile(txt_path) else np.eye(4)
    mesh = transform_mesh(mesh, A)

    voxel_m = _voxel_size_m(cfg)
    origin = np.asarray(sc["grid_origin_m"], dtype=np.float64)
    cad_mesh = load_mesh(cad_path)
    cad_idx = surface_voxels(cad_mesh, origin, voxel_m, sc.get("bounds_min_m"), sc.get("bounds_max_m"),
                             cfg.get("max_surface_voxels"))
    cad_centres_full = voxel_centres(cad_idx, origin, voxel_m)

    _, _, recon_centres, yaw, T = gauge_normalize(mesh, cfg, sc, cad_centres_full)

    vis = np.load(vis_path)
    vis_idx = vis["indices"].astype(np.int32)
    cad_set = set(map(tuple, cad_idx.tolist()))
    vis_idx = np.array([i for i in vis_idx if tuple(i) in cad_set], dtype=np.int32)
    if len(vis_idx) == 0:
        raise RuntimeError("empty visible CAD cell set after intersection")
    ref_centres = voxel_centres(vis_idx, origin, voxel_m)

    chamfer_l1, accuracy_mean, completeness_mean = chamfer_l1_cm(ref_centres, recon_centres)

    recon_dir = os.path.join(root, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    png_rel = "recon/cad_comparison.png"
    png_path = os.path.join(root, png_rel)
    yaml_path = os.path.join(recon_dir, "geometry_score.yaml")

    fd, tmp_png = tempfile.mkstemp(prefix=".tmp_cmp_", suffix=".png", dir=recon_dir)
    os.close(fd)
    try:
        render_comparison(tmp_png, ref_centres, recon_centres, scan_id,
                          chamfer_l1, accuracy_mean, completeness_mean)
        doc = {
            "metric": METRIC,
            "chamfer_l1_cm": float(round(chamfer_l1, 6)),
            "accuracy_mean_cm": float(round(accuracy_mean, 6)),
            "completeness_mean_cm": float(round(completeness_mean, 6)),
            "scene": scan_id,
            "reference_sha256": sc["reference_sha256"],
            "visible_voxels_sha256": sc["visible_voxels_sha256"],
            "voxel_mm": int(cfg["voxel_mm"]),
            "alignment": {
                "axis_alignment_applied": True,
                "yaw_deg": int(yaw),
                "transform": T.reshape(-1).astype(float).tolist(),
            },
            "comparison": png_rel,
        }
        text = yaml.safe_dump(doc, sort_keys=False)
        atomic_write_text(yaml_path, text)
        os.replace(tmp_png, png_path)
    except Exception:
        try:
            os.unlink(tmp_png)
        except OSError:
            pass
        raise

    print(f"[geometry] {scan_id} chamfer_l1_cm={chamfer_l1:.2f} "
          f"(acc={accuracy_mean:.2f} comp={completeness_mean:.2f})")
    return doc


def main():
    ap = argparse.ArgumentParser(description="CAD Chamfer-L1 geometry score for a ScanNet-style scan")
    ap.add_argument("--scan-id", required=True)
    ap.add_argument("--reference-config", default=DEFAULT_CONFIG)
    args = ap.parse_args()
    score_scan(args.scan_id, args.reference_config)


if __name__ == "__main__":
    main()
