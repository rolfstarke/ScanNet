"""CAD-referenced mean bidirectional distance scorer for scene9004.

Scores <scan_id>_vh_clean_2.ply against a fixed CAD visibility mask. Reconstruction
stays in scanworld; only an in-memory CAD copy is rigidly oriented for the score.
CPU-only; never runs inside a managed GPU lease.

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
METRIC = "observed_surface_voxel_mean_bidirectional_distance_v1"
SCORE_KEY = "mean_bidirectional_distance_cm"


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
    """Return sorted (N,3) int32 voxel indices on a fixed grid origin."""
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


def yaw_matrix(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    return T


def apply_T(pts, T):
    T = np.asarray(T, dtype=np.float64)
    return (T[:3, :3] @ np.asarray(pts, dtype=np.float64).T).T + T[:3, 3]


def mean_bidirectional_distance_cm(ref_centres, recon_centres):
    """Mean of recon→CAD and CAD→recon nearest-neighbour means, in cm."""
    if len(ref_centres) == 0 or len(recon_centres) == 0:
        raise RuntimeError("empty cell set in mean bidirectional distance")
    t_ref = cKDTree(ref_centres)
    t_rec = cKDTree(recon_centres)
    d_rec, _ = t_ref.query(recon_centres, k=1, workers=-1)
    d_ref, _ = t_rec.query(ref_centres, k=1, workers=-1)
    accuracy_mean_m = float(np.mean(d_rec))
    completeness_mean_m = float(np.mean(d_ref))
    score_m = 0.5 * (accuracy_mean_m + completeness_mean_m)
    score = 100.0 * score_m
    accuracy_mean = 100.0 * accuracy_mean_m
    completeness_mean = 100.0 * completeness_mean_m
    if not np.isfinite(score) or score < 0:
        raise RuntimeError(f"invalid {SCORE_KEY} {score}")
    return score, accuracy_mean, completeness_mean


def recon_surface_centres(mesh, voxel_m, max_voxels=None):
    """10 mm surface centres of recon in its own frame (local grid origin)."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    origin = np.floor(V.min(axis=0) / voxel_m) * voxel_m - voxel_m
    idx = surface_voxels(mesh, origin, voxel_m, max_voxels=max_voxels)
    return voxel_centres(idx, origin, voxel_m)


def _pcd(pts):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return p


def _downsample(pts, voxel=0.05, max_n=25000):
    a = np.asarray(_pcd(pts).voxel_down_sample(voxel).points, dtype=np.float64)
    if len(a) == 0:
        raise RuntimeError("downsample produced empty cloud")
    if len(a) > max_n:
        a = a[::int(np.ceil(len(a) / max_n))]
    return a


def _icp(source, target, init, max_corr, max_iter):
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    reg = o3d.pipelines.registration.registration_icp(
        source, target, max_corr, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        crit)
    return np.asarray(reg.transformation, dtype=np.float64)


def align_cad_to_recon(cad_pts, recon_pts):
    """Quick temporary rigid CAD→recon (debug gauge only). Recon is never moved."""
    cad_ds = _downsample(cad_pts)
    rec_ds = _downsample(recon_pts)
    cad_c = cad_ds.mean(axis=0)
    rec_c = rec_ds.mean(axis=0)
    src = _pcd(cad_ds)
    tgt = _pcd(rec_ds)
    best = None
    for yaw in (0, 90, 180, 270):
        T_to_c = np.eye(4)
        T_to_c[:3, 3] = -cad_c
        T_from_c = np.eye(4)
        T_from_c[:3, 3] = cad_c
        T_yaw = T_from_c @ yaw_matrix(yaw) @ T_to_c
        cad_yaw = apply_T(cad_ds, T_yaw)
        T_init = np.eye(4)
        T_init[:3, :3] = T_yaw[:3, :3]
        T_init[:3, 3] = T_yaw[:3, 3] + (rec_c - cad_yaw.mean(axis=0))
        T1 = _icp(src, tgt, T_init, 1.0, 20)
        T2 = _icp(src, tgt, T1, 0.25, 10)
        cad_al = apply_T(cad_ds, T2)
        score, _, _ = mean_bidirectional_distance_cm(cad_al, rec_ds)
        if best is None or score < best[0] - 1e-9 or (
                abs(score - best[0]) <= 1e-9 and yaw < best[1]):
            best = (score, yaw, T2)
    if best is None:
        raise RuntimeError("CAD→recon alignment failed for all yaw seeds")
    return best[2], int(best[1])


def _coarse_occupancy(pts, cell_m):
    """One centre per occupied cell on a rough grid."""
    qi = np.unique(np.floor(np.asarray(pts, dtype=np.float64) / cell_m).astype(np.int64), axis=0)
    return (qi.astype(np.float64) + 0.5) * cell_m


def _overlay_cell_m(ref_centres, recon_centres, target_cells=64, lo=0.08, hi=0.25):
    """cell ≈ longest plan extent / target_cells (clamped). Square side = cell."""
    extent = float(np.ptp(np.vstack([ref_centres[:, :2], recon_centres[:, :2]]), axis=0).max())
    return float(np.clip(extent / target_cells, lo, hi))


def _stamp_squares(pts2, lo, hi, half, pixel):
    """Coverage image: each point adds 1 over an axis-aligned square."""
    w = int(np.ceil((hi[0] - lo[0]) / pixel)) + 1
    h = int(np.ceil((hi[1] - lo[1]) / pixel)) + 1
    cov = np.zeros((h, w), dtype=np.float32)
    hs = half / pixel
    for x, y in pts2:
        c = (x - lo[0]) / pixel
        r = (y - lo[1]) / pixel
        r0, r1 = max(0, int(np.floor(r - hs))), min(h, int(np.ceil(r + hs)))
        c0, c1 = max(0, int(np.floor(c - hs))), min(w, int(np.ceil(c + hs)))
        if r0 < r1 and c0 < c1:
            cov[r0:r1, c0:c1] += 1.0
    return cov


def render_comparison(path, ref_centres, recon_centres, scene_id, score,
                      accuracy_mean, completeness_mean):
    """CAD yellow + recon cyan. Rough occupancy → square stamps, α=1/N_max."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cad_rgb = np.array([1.0, 0.92, 0.05])
    rec_rgb = np.array([0.05, 0.85, 0.90])
    cell = _overlay_cell_m(ref_centres, recon_centres)
    half = 0.5 * cell
    pixel = max(cell / 8.0, 0.015)
    cad = _coarse_occupancy(ref_centres, cell)
    rec = _coarse_occupancy(recon_centres, cell)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    for ax, dims, label in ((axes[0], (0, 1), "plan x/y"),
                            (axes[1], (0, 2), "elevation x/z")):
        c2, r2 = cad[:, list(dims)], rec[:, list(dims)]
        all2 = np.vstack([c2, r2])
        lo, hi = all2.min(0) - half, all2.max(0) + half
        cov_c = _stamp_squares(c2, lo, hi, half, pixel)
        cov_r = _stamp_squares(r2, lo, hi, half, pixel)
        n_max = max(float(cov_c.max()), float(cov_r.max()), 1.0)
        cover_c = np.clip(cov_c / n_max, 0.0, 1.0)
        cover_r = np.clip(cov_r / n_max, 0.0, 1.0)
        rgb = np.ones(cov_c.shape + (3,), dtype=np.float64)
        for i in range(3):
            rgb[..., i] = 1.0 - cover_c * (1.0 - cad_rgb[i])
        for i in range(3):
            rgb[..., i] = rgb[..., i] * (1.0 - cover_r) + rec_rgb[i] * cover_r
        ax.imshow(np.clip(rgb, 0, 1), origin="lower",
                  extent=[lo[0], hi[0], lo[1], hi[1]],
                  interpolation="nearest", aspect="equal")
        ax.set_title(label)
        ax.set_facecolor("white")

    fig.legend([Patch(color=cad_rgb), Patch(color=rec_rgb)],
               ["CAD", "recon"], loc="lower center", ncol=2)
    fig.suptitle(
        f"{scene_id}  {SCORE_KEY}={score:.2f}  "
        f"(acc={accuracy_mean:.2f}  comp={completeness_mean:.2f})  "
        f"cell={cell:.2f}m")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(path, dpi=140, facecolor="white")
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

    # Recon stays in scanworld (only ScanNet axisAlignment from its own .txt).
    mesh = load_mesh(mesh_path)
    A = parse_axis_alignment(txt_path) if os.path.isfile(txt_path) else np.eye(4)
    mesh = transform_mesh(mesh, A)
    voxel_m = _voxel_size_m(cfg)
    recon_centres = recon_surface_centres(mesh, voxel_m, cfg.get("max_surface_voxels"))

    # CAD + visibility in CAD frame, then temporary rigid CAD→recon.
    origin = np.asarray(sc["grid_origin_m"], dtype=np.float64)
    cad_mesh = load_mesh(cad_path)
    cad_idx = surface_voxels(cad_mesh, origin, voxel_m, sc.get("bounds_min_m"),
                             sc.get("bounds_max_m"), cfg.get("max_surface_voxels"))
    cad_set = set(map(tuple, cad_idx.tolist()))
    vis = np.load(vis_path)
    vis_idx = np.array([i for i in vis["indices"].astype(np.int32)
                        if tuple(i) in cad_set], dtype=np.int32)
    if len(vis_idx) == 0:
        raise RuntimeError("empty visible CAD cell set after intersection")
    cad_centres = voxel_centres(vis_idx, origin, voxel_m)

    T_cad, yaw0 = align_cad_to_recon(cad_centres, recon_centres)
    ref_centres = apply_T(cad_centres, T_cad)

    score, accuracy_mean, completeness_mean = mean_bidirectional_distance_cm(
        ref_centres, recon_centres)

    recon_dir = os.path.join(root, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    png_rel = "recon/cad_comparison.png"
    png_path = os.path.join(root, png_rel)
    yaml_path = os.path.join(recon_dir, "geometry_score.yaml")

    fd, tmp_png = tempfile.mkstemp(prefix=".tmp_cmp_", suffix=".png", dir=recon_dir)
    os.close(fd)
    try:
        render_comparison(tmp_png, ref_centres, recon_centres, scan_id,
                          score, accuracy_mean, completeness_mean)
        doc = {
            "metric": METRIC,
            SCORE_KEY: float(round(score, 6)),
            "accuracy_mean_cm": float(round(accuracy_mean, 6)),
            "completeness_mean_cm": float(round(completeness_mean, 6)),
            "scene": scan_id,
            "reference_sha256": sc["reference_sha256"],
            "visible_voxels_sha256": sc["visible_voxels_sha256"],
            "voxel_mm": int(cfg["voxel_mm"]),
            "alignment": {
                "method": "temporary_cad_rigid_icp",
                "moved": "cad",
                "reconstruction_transformed": False,
                "axis_alignment_applied_to_recon": True,
                "initial_yaw_deg": int(yaw0),
                "cad_to_scanworld": T_cad.reshape(-1).astype(float).tolist(),
            },
            "comparison": png_rel,
        }
        atomic_write_text(yaml_path, yaml.safe_dump(doc, sort_keys=False))
        os.replace(tmp_png, png_path)
    except Exception:
        try:
            os.unlink(tmp_png)
        except OSError:
            pass
        raise

    print(f"[geometry] {scan_id} {SCORE_KEY}={score:.2f} "
          f"(acc={accuracy_mean:.2f} comp={completeness_mean:.2f})")
    return doc


def main():
    ap = argparse.ArgumentParser(
        description="CAD mean bidirectional distance score for a ScanNet-style scan")
    ap.add_argument("--scan-id", required=True)
    ap.add_argument("--reference-config", default=DEFAULT_CONFIG)
    args = ap.parse_args()
    score_scan(args.scan_id, args.reference_config)


if __name__ == "__main__":
    main()
