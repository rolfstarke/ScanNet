"""CAD mean bidirectional distance scorer for scene9004.

CAD stays fixed; an in-memory recon copy is rigidly aligned for scoring/views only.
On-disk meshes are never modified.

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
CAD_RGB = np.array([0.92, 0.12, 0.72])   # magenta
RECON_RGB = np.array([0.08, 0.78, 0.90])  # cyan
CEILING_HEIGHT_M = 2.5


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
                if abs(np.linalg.det(A[:3, :3]) - 1.0) > 0.05:
                    raise ValueError(f"axisAlignment det bad in {txt_path}")
                return A
    return np.eye(4)


def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty() or not mesh.has_triangles():
        raise RuntimeError(f"empty/non-triangular mesh: {path}")
    mesh.compute_vertex_normals()
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise RuntimeError(f"non-finite vertices: {path}")
    return mesh


def transform_mesh(mesh, T):
    out = mesh.clone() if hasattr(mesh, "clone") else o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy()),
        o3d.utility.Vector3iVector(np.asarray(mesh.triangles).copy()))
    out.transform(np.asarray(T, dtype=np.float64))
    return out


def apply_T(pts, T):
    T = np.asarray(T, dtype=np.float64)
    return (T[:3, :3] @ np.asarray(pts, dtype=np.float64).T).T + T[:3, 3]


def surface_voxels(mesh, origin, voxel_m, bounds_min=None, bounds_max=None, max_voxels=None):
    """Sorted (N,3) int32 surface voxel indices via barycentric triangle sampling."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    if len(F) == 0:
        raise RuntimeError("surface voxelization produced zero cells")
    origin = np.asarray(origin, dtype=np.float64)
    if bounds_min is not None and bounds_max is not None:
        lo, hi = np.asarray(bounds_min, float), np.asarray(bounds_max, float)
        if (V < lo - 1e-9).any() or (V > hi + 1e-9).any():
            raise RuntimeError("mesh vertices outside configured evaluation bounds")
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    edge = np.maximum.reduce([np.linalg.norm(b - a, axis=1),
                              np.linalg.norm(c - b, axis=1),
                              np.linalg.norm(a - c, axis=1)])
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
        p = (aa[:, None] * ww[None, :, None] + bb[:, None] * wu[None, :, None]
             + cc[:, None] * wv[None, :, None])
        chunks.append(p.reshape(-1, 3))
    idx = np.unique(np.floor((np.concatenate(chunks) - origin) / voxel_m).astype(np.int32), axis=0)
    if idx.size == 0:
        raise RuntimeError("surface voxelization produced zero cells")
    if max_voxels is not None and len(idx) > max_voxels:
        raise RuntimeError(f"surface voxels {len(idx)} exceed cap {max_voxels}")
    return idx[np.lexsort((idx[:, 2], idx[:, 1], idx[:, 0]))]


def voxel_centres(idx, origin, voxel_m):
    return np.asarray(origin, float) + (np.asarray(idx, float) + 0.5) * voxel_m


def mean_bidirectional_distance_cm(ref, recon):
    if len(ref) == 0 or len(recon) == 0:
        raise RuntimeError("empty cell set in mean bidirectional distance")
    d_rec = cKDTree(ref).query(recon, k=1, workers=-1)[0]
    d_ref = cKDTree(recon).query(ref, k=1, workers=-1)[0]
    acc, comp = float(np.mean(d_rec)) * 100.0, float(np.mean(d_ref)) * 100.0
    score = 0.5 * (acc + comp)
    if not np.isfinite(score) or score < 0:
        raise RuntimeError(f"invalid {SCORE_KEY} {score}")
    return score, acc, comp


def recon_surface_centres(mesh, voxel_m, max_voxels=None):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    origin = np.floor(V.min(0) / voxel_m) * voxel_m - voxel_m
    return voxel_centres(surface_voxels(mesh, origin, voxel_m, max_voxels=max_voxels),
                         origin, voxel_m)


def align_cad_to_recon(cad_pts, recon_pts):
    """Rigid T: CAD→recon (four-yaw ICP). Caller inverts so recon moves into CAD frame."""
    def pcd(p):
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(np.asarray(p, float))
        return c

    def down(p, v=0.05, n=25000):
        a = np.asarray(pcd(p).voxel_down_sample(v).points, float)
        if len(a) == 0:
            raise RuntimeError("downsample empty")
        return a[::int(np.ceil(len(a) / n))] if len(a) > n else a

    def icp(src, tgt, init, corr, it):
        crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it)
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, corr, init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), crit)
        return np.asarray(reg.transformation, float)

    cad_ds, rec_ds = down(cad_pts), down(recon_pts)
    cad_c, rec_c = cad_ds.mean(0), rec_ds.mean(0)
    src, tgt = pcd(cad_ds), pcd(rec_ds)
    best = None
    for yaw in (0, 90, 180, 270):
        a = np.deg2rad(yaw)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        T0 = np.eye(4)
        T0[:3, :3] = R
        T0[:3, 3] = cad_c - R @ cad_c
        cad_yaw = apply_T(cad_ds, T0)
        T0[:3, 3] += rec_c - cad_yaw.mean(0)
        T = icp(src, tgt, icp(src, tgt, T0, 1.0, 20), 0.25, 10)
        sc, _, _ = mean_bidirectional_distance_cm(apply_T(cad_ds, T), rec_ds)
        if best is None or sc < best[0] - 1e-9 or (abs(sc - best[0]) <= 1e-9 and yaw < best[1]):
            best = (sc, yaw, T)
    if best is None:
        raise RuntimeError("CAD↔recon alignment failed")
    if abs(np.linalg.det(best[2][:3, :3]) - 1.0) > 1e-3:
        raise RuntimeError("non-rigid ICP rotation")
    return best[2], int(best[1])


def _occupancy_panel(ax, cad, rec, dims, cell):
    """Integer-grid occupancy; cover-weighted magenta/cyan mix; α shared via Nmax."""
    c2, r2 = cad[:, list(dims)], rec[:, list(dims)]
    all2 = np.vstack([c2, r2])
    lo = np.floor(all2.min(0) / cell) * cell
    hi = np.ceil(all2.max(0) / cell) * cell
    n0 = max(1, int(np.round((hi[0] - lo[0]) / cell)))
    n1 = max(1, int(np.round((hi[1] - lo[1]) / cell)))

    def cov(pts):
        ij = np.floor((pts - lo) / cell).astype(np.int64)
        m = (ij[:, 0] >= 0) & (ij[:, 0] < n0) & (ij[:, 1] >= 0) & (ij[:, 1] < n1)
        ij = ij[m]
        out = np.zeros((n1, n0), np.float32)
        if len(ij):
            out.ravel()[:] = np.bincount(ij[:, 1] * n0 + ij[:, 0], minlength=n0 * n1)
        return out

    a_c, a_r = cov(c2), cov(r2)
    n_max = max(float(a_c.max()), float(a_r.max()), 1.0)
    a_c, a_r = a_c / n_max, a_r / n_max
    w = a_c + a_r
    alpha = np.clip(w, 0, 1)
    rgb = np.ones((n1, n0, 3))
    m = w > 0
    for i in range(3):
        mix = np.where(m, (CAD_RGB[i] * a_c + RECON_RGB[i] * a_r) / np.maximum(w, 1e-12), 1.0)
        rgb[..., i] = (1.0 - alpha) + mix * alpha
    ax.imshow(np.clip(rgb, 0, 1), origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]],
              interpolation="nearest", aspect="equal")
    ax.set_facecolor("white")


def render_geometry_report(path, ref_centres, recon_centres, mesh, scene_id, score,
                           accuracy_mean, completeness_mean):
    """Plan | elev on top; full-width isometric recon (distance-coloured) below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    extent = float(np.ptp(np.vstack([ref_centres[:, :2], recon_centres[:, :2]]), 0).max())
    cell = float(np.clip(extent / 100.0, 0.05, 0.12))

    def coarse(p):
        qi = np.unique(np.floor(np.asarray(p, float) / cell).astype(np.int64), axis=0)
        return (qi.astype(float) + 0.5) * cell

    cad, rec = coarse(ref_centres), coarse(recon_centres)

    # Recon mesh only: ceiling height-hide, colour by distance to CAD
    m = mesh
    if len(m.triangles) > 150000:
        m = m.simplify_quadric_decimation(120000)
    V, F = np.asarray(m.vertices, float), np.asarray(m.triangles, np.int64)
    floor = float(np.percentile(V[:, 2], 1))
    kv = V[:, 2] <= floor + CEILING_HEIGHT_M
    F2 = F[kv[F[:, 0]] & kv[F[:, 1]] & kv[F[:, 2]]]
    if len(F2) == 0:
        raise RuntimeError("no faces left after ceiling hide")
    d_cm = cKDTree(ref_centres).query(V, k=1, workers=-1)[0] * 100.0
    d_face = d_cm[F2].mean(1)
    p95 = max(float(np.percentile(d_face, 95)), 1e-3)
    t = np.clip(d_face / p95, 0, 1)
    fcol = CAD_RGB * (1 - t)[:, None] + RECON_RGB * t[:, None]

    # Zoom to recon faces inside CAD room
    fc = V[F2].mean(1)
    clo, chi = ref_centres.min(0), ref_centres.max(0)
    pad = 0.08 * np.maximum(chi - clo, 0.3)
    keep = np.all((fc >= clo - pad) & (fc <= chi + pad), 1)
    if keep.sum() < 80:
        keep = d_face <= np.percentile(d_face, 80)
    if keep.sum() < 30:
        keep = np.ones(len(F2), bool)
    F_iso, fcol_iso = F2[keep], fcol[keep]
    pts = V[np.unique(F_iso.ravel())]
    plo, phi = pts.min(0), pts.max(0)
    span = np.maximum(phi - plo, 0.2)
    lo, hi = plo - 0.04 * span, phi + 0.04 * span

    S, iso_h, gap = 5.0, 2.1 * 5.0, 0.28
    fig_w, fig_h = 2 * S + gap + 1.3, S + gap + iso_h + 1.3
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ml, mb = 0.55 / fig_w, 0.72 / fig_h
    sx, sy, gx, gy = S / fig_w, S / fig_h, gap / fig_w, gap / fig_h
    ih = iso_h / fig_h
    x0, y_iso = ml, mb
    y_top = mb + ih + gy
    ax_p = fig.add_axes([x0, y_top, sx, sy])
    ax_e = fig.add_axes([x0 + sx + gx, y_top, sx, sy])
    ax_i = fig.add_axes([x0, y_iso, 2 * sx + gx, ih], projection="3d", computed_zorder=False)

    _occupancy_panel(ax_p, cad, rec, (0, 1), cell)
    ax_p.set_title("plan x/y"); ax_p.set_xlabel("x"); ax_p.set_ylabel("y")
    _occupancy_panel(ax_e, cad, rec, (0, 2), cell)
    ax_e.set_title("elevation x/z"); ax_e.set_xlabel("x"); ax_e.set_ylabel("z")

    coll = Poly3DCollection(V[F_iso], linewidths=0, edgecolors="none")
    coll.set_facecolor(fcol_iso)
    ax_i.add_collection3d(coll)
    ax_i.set_xlim(lo[0], hi[0]); ax_i.set_ylim(lo[1], hi[1]); ax_i.set_zlim(lo[2], hi[2])
    try:
        ax_i.set_box_aspect((hi - lo) / max(float((hi - lo).max()), 1e-6))
    except Exception:
        pass
    ax_i.view_init(30, -48)
    try:
        ax_i.set_proj_type("ortho")
    except Exception:
        pass
    ax_i.set_axis_off()
    ax_i.grid(False)
    ax_i.set_facecolor("white")
    ax_i.set_title(
        f"isometric recon · distance to CAD  (ceiling floor+{CEILING_HEIGHT_M:.1f}m) · "
        f"p95={p95:.1f} cm", fontsize=10, pad=4)

    cax = fig.add_axes([x0 + 2 * sx + gx + 0.012, y_iso + 0.06 * ih, 0.014, 0.88 * ih])
    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list("nf", [CAD_RGB, RECON_RGB]),
        norm=plt.Normalize(0, p95))
    sm.set_array([])
    fig.colorbar(sm, cax=cax).set_label("distance to CAD (cm)", fontsize=9)
    fig.legend([Patch(color=CAD_RGB), Patch(color=RECON_RGB)],
               ["CAD (magenta)", "recon (cyan)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.suptitle(
        f"{scene_id}  {SCORE_KEY}={score:.2f}  "
        f"(acc={accuracy_mean:.2f}  comp={completeness_mean:.2f})  cell={cell:.2f}m",
        y=0.97)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def atomic_write_text(path, text):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def score_scan(scan_id, config_path=None):
    cfg = load_config(config_path)
    if cfg.get("metric") != METRIC:
        raise RuntimeError(f"config metric {cfg.get('metric')!r} != {METRIC!r}")
    sc = scene_config(cfg, int(scan_id[5:9]))
    for key, path, label in (
        ("reference_sha256", sc["cad_ply"], "CAD"),
        ("visible_voxels_sha256", sc["visible_voxels"], "visible voxels"),
    ):
        if sc.get(key) is None:
            raise RuntimeError("geometry reference not finalized")
        if file_sha256(path) != sc[key]:
            raise RuntimeError(f"{label} hash mismatch against geometry_reference.yaml")

    root = os.path.join(SCANS_DIR, scan_id)
    mesh_path = os.path.join(root, f"{scan_id}_vh_clean_2.ply")
    txt_path = os.path.join(root, f"{scan_id}.txt")
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(mesh_path)

    mesh = load_mesh(mesh_path)
    A = parse_axis_alignment(txt_path) if os.path.isfile(txt_path) else np.eye(4)
    mesh_a = transform_mesh(mesh, A)
    voxel_m = float(cfg["voxel_mm"]) / 1000.0
    recon_centres = recon_surface_centres(mesh_a, voxel_m, cfg.get("max_surface_voxels"))

    origin = np.asarray(sc["grid_origin_m"], float)
    cad_mesh = load_mesh(sc["cad_ply"])
    cad_idx = surface_voxels(cad_mesh, origin, voxel_m, sc.get("bounds_min_m"),
                             sc.get("bounds_max_m"), cfg.get("max_surface_voxels"))
    cad_set = set(map(tuple, cad_idx.tolist()))
    vis_idx = np.array([i for i in np.load(sc["visible_voxels"])["indices"].astype(np.int32)
                        if tuple(i) in cad_set], dtype=np.int32)
    if len(vis_idx) == 0:
        raise RuntimeError("empty visible CAD cell set after intersection")
    cad_centres = voxel_centres(vis_idx, origin, voxel_m)

    T_c2r, yaw0 = align_cad_to_recon(cad_centres, recon_centres)
    T_r2c = np.linalg.inv(T_c2r)
    recon_cad = apply_T(recon_centres, T_r2c)
    mesh_cad = transform_mesh(mesh_a, T_r2c)
    score, acc, comp = mean_bidirectional_distance_cm(cad_centres, recon_cad)

    recon_dir = os.path.join(root, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    png_rel, png_path = "recon/cad_comparison.png", os.path.join(root, "recon/cad_comparison.png")
    yaml_path = os.path.join(recon_dir, "geometry_score.yaml")
    fd, tmp_png = tempfile.mkstemp(prefix=".tmp_cmp_", suffix=".png", dir=recon_dir)
    os.close(fd)
    try:
        render_geometry_report(tmp_png, cad_centres, recon_cad, mesh_cad, scan_id,
                               score, acc, comp)
        doc = {
            "metric": METRIC,
            SCORE_KEY: float(round(score, 6)),
            "accuracy_mean_cm": float(round(acc, 6)),
            "completeness_mean_cm": float(round(comp, 6)),
            "scene": scan_id,
            "reference_sha256": sc["reference_sha256"],
            "visible_voxels_sha256": sc["visible_voxels_sha256"],
            "voxel_mm": int(cfg["voxel_mm"]),
            "alignment": {
                "method": "temporary_recon_rigid_icp",
                "moved": "recon_in_memory_only",
                "reconstruction_saved": False,
                "axis_alignment_applied_to_recon": True,
                "initial_yaw_deg": int(yaw0),
                "recon_to_cad": T_r2c.reshape(-1).astype(float).tolist(),
            },
            "comparison": png_rel,
        }
        atomic_write_text(yaml_path, yaml.safe_dump(doc, sort_keys=False))
        os.replace(tmp_png, png_path)
        legacy = os.path.join(recon_dir, "cad_distance_isometric.png")
        if os.path.isfile(legacy):
            try:
                os.unlink(legacy)
            except OSError:
                pass
    except Exception:
        try:
            os.unlink(tmp_png)
        except OSError:
            pass
        raise

    print(f"[geometry] {scan_id} {SCORE_KEY}={score:.2f} (acc={acc:.2f} comp={comp:.2f})")
    return doc


def main():
    ap = argparse.ArgumentParser(description="CAD mean bidirectional distance score")
    ap.add_argument("--scan-id", required=True)
    ap.add_argument("--reference-config", default=DEFAULT_CONFIG)
    args = ap.parse_args()
    score_scan(args.scan_id, args.reference_config)


if __name__ == "__main__":
    main()
