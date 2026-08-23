"""CAD-referenced mean bidirectional distance scorer for scene9004.

Scores <scan_id>_vh_clean_2.ply against a fixed CAD visibility mask. CAD stays in
its original frame; only an in-memory recon copy is rigidly oriented for scoring
and debug views. On-disk meshes/poses are never modified. CPU-only; never runs
inside a managed GPU lease.

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
# Magenta + cyan: equal mix → blue/purple; near=magenta, far=cyan
CAD_RGB = np.array([0.92, 0.12, 0.72])
RECON_RGB = np.array([0.08, 0.78, 0.90])
CEILING_HEIGHT_M = 2.0  # hide verts above floor_p1 + this (display only)


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
    """Estimate rigid T mapping CAD→recon (four-yaw ICP). Used inverted so recon
    moves into CAD frame; CAD and on-disk recon stay fixed."""
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
        raise RuntimeError("CAD↔recon alignment failed for all yaw seeds")
    T_cad_to_recon = best[2]
    det = float(np.linalg.det(T_cad_to_recon[:3, :3]))
    if abs(det - 1.0) > 1e-3:
        raise RuntimeError(f"non-rigid ICP rotation det={det}")
    return T_cad_to_recon, int(best[1])


def _coarse_occupancy(pts, cell_m):
    """One centre per occupied cell on a rough grid."""
    qi = np.unique(np.floor(np.asarray(pts, dtype=np.float64) / cell_m).astype(np.int64), axis=0)
    return (qi.astype(np.float64) + 0.5) * cell_m


def _overlay_cell_m(ref_centres, recon_centres, target_cells=100, lo=0.05, hi=0.12):
    """Occupancy cell from longest plan extent / target_cells (clamped)."""
    extent = float(np.ptp(np.vstack([ref_centres[:, :2], recon_centres[:, :2]]), axis=0).max())
    return float(np.clip(extent / target_cells, lo, hi))


def _stamp_squares(pts2, lo, cell, n0, n1):
    """Integer occupancy grid: each coarse cell → exactly one grid square (no gaps)."""
    cov = np.zeros((n1, n0), dtype=np.float32)
    # pts are cell centres; map to integer cell indices
    ij = np.floor((pts2 - lo) / cell).astype(np.int64)
    m = (ij[:, 0] >= 0) & (ij[:, 0] < n0) & (ij[:, 1] >= 0) & (ij[:, 1] < n1)
    ij = ij[m]
    if len(ij) == 0:
        return cov
    flat = ij[:, 1] * n0 + ij[:, 0]
    cov.ravel()[:] = np.bincount(flat, minlength=n0 * n1).astype(np.float32)
    return cov


def _panel_overlay(ax, cad, rec, dims, cell, cad_rgb, rec_rgb):
    """Plan/elevation: shared α=1/Nmax; colors mix by relative cover (overlap → green)."""
    c2, r2 = cad[:, list(dims)], rec[:, list(dims)]
    all2 = np.vstack([c2, r2])
    lo = np.floor(all2.min(0) / cell) * cell
    hi = np.ceil(all2.max(0) / cell) * cell
    n0 = max(1, int(np.round((hi[0] - lo[0]) / cell)))
    n1 = max(1, int(np.round((hi[1] - lo[1]) / cell)))
    cov_c = _stamp_squares(c2, lo, cell, n0, n1)
    cov_r = _stamp_squares(r2, lo, cell, n0, n1)
    n_max = max(float(cov_c.max()), float(cov_r.max()), 1.0)
    a_c = np.clip(cov_c / n_max, 0.0, 1.0)
    a_r = np.clip(cov_r / n_max, 0.0, 1.0)
    w = a_c + a_r
    # alpha from either cloud; color = cover-weighted mix (equal → green)
    alpha = np.clip(w, 0.0, 1.0)
    mix = np.zeros((n1, n0, 3), dtype=np.float64)
    m = w > 0
    for i in range(3):
        mix[..., i] = np.where(m, (cad_rgb[i] * a_c + rec_rgb[i] * a_r) / np.maximum(w, 1e-12), 1.0)
    rgb = np.ones((n1, n0, 3), dtype=np.float64)
    for i in range(3):
        rgb[..., i] = (1.0 - alpha) + mix[..., i] * alpha
    ax.imshow(np.clip(rgb, 0, 1), origin="lower",
              extent=[lo[0], hi[0], lo[1], hi[1]],
              interpolation="nearest", aspect="equal")
    ax.set_facecolor("white")


def _hide_ceiling_faces(mesh, ceiling_height=CEILING_HEIGHT_M):
    """Hide faces above floor_p1 + ceiling_height (display only; no mesh save)."""
    m = mesh
    if len(m.triangles) > 150000:
        m = m.simplify_quadric_decimation(target_number_of_triangles=120000)
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.triangles, dtype=np.int64)
    up = 2
    floor = float(np.percentile(V[:, up], 1))
    keep_v = V[:, up] <= floor + ceiling_height
    keep_f = keep_v[F[:, 0]] & keep_v[F[:, 1]] & keep_v[F[:, 2]]
    F2 = F[keep_f]
    if len(F2) == 0:
        raise RuntimeError("no faces left after ceiling height hide")
    return V, F2, floor


def _iso_axis_off(ax):
    """Bare 3D axes: no panes, ticks, or grid."""
    ax.set_axis_off()
    ax.grid(False)
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
    except Exception:
        pass


def render_geometry_report(path, ref_centres, recon_centres, mesh, scene_id, score,
                           accuracy_mean, completeness_mean):
    """One PNG as four equal squares: plan | elev on top; iso = bottom two squares."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cad_rgb, rec_rgb = CAD_RGB, RECON_RGB
    cell = _overlay_cell_m(ref_centres, recon_centres)
    cad = _coarse_occupancy(ref_centres, cell)
    rec = _coarse_occupancy(recon_centres, cell)

    V, F2, floor = _hide_ceiling_faces(mesh)
    d_cm = cKDTree(ref_centres).query(V, k=1, workers=-1)[0] * 100.0
    d_face = d_cm[F2].mean(axis=1)
    p95 = float(np.percentile(d_face, 95)) if len(d_face) else 1.0
    p95 = max(p95, 1e-3)
    t = np.clip(d_face / p95, 0.0, 1.0)
    fcol = cad_rgb[None, :] * (1.0 - t)[:, None] + rec_rgb[None, :] * t[:, None]

    # Four equal unit squares: (0,0) plan (0,1) elev (1,:) iso = 2 squares
    fig = plt.figure(figsize=(12, 12), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
                  hspace=0.14, wspace=0.14,
                  left=0.06, right=0.98, top=0.92, bottom=0.07)

    ax_plan = fig.add_subplot(gs[0, 0])
    _panel_overlay(ax_plan, cad, rec, (0, 1), cell, cad_rgb, rec_rgb)
    ax_plan.set_title("1  plan x/y")
    ax_plan.set_xlabel("x")
    ax_plan.set_ylabel("y")

    ax_elev = fig.add_subplot(gs[0, 1])
    _panel_overlay(ax_elev, cad, rec, (0, 2), cell, cad_rgb, rec_rgb)
    ax_elev.set_title("2  elevation x/z")
    ax_elev.set_xlabel("x")
    ax_elev.set_ylabel("z")

    ax_iso = fig.add_subplot(gs[1, :], projection="3d", computed_zorder=False)
    coll = Poly3DCollection(V[F2], linewidths=0, edgecolors="none")
    coll.set_facecolor(fcol)
    ax_iso.add_collection3d(coll)
    clo, chi = ref_centres.min(0), ref_centres.max(0)
    c = 0.5 * (clo + chi)
    r = 0.55 * float(np.max(chi - clo)) + 0.3
    ax_iso.set_xlim(c[0] - r, c[0] + r)
    ax_iso.set_ylim(c[1] - r, c[1] + r)
    ax_iso.set_zlim(c[2] - r, c[2] + r)
    try:
        ax_iso.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax_iso.view_init(elev=28, azim=-50)
    try:
        ax_iso.set_proj_type("ortho")
    except Exception:
        pass
    _iso_axis_off(ax_iso)
    ax_iso.set_facecolor("white")
    ax_iso.set_title(
        f"3–4  isometric recon  (ceiling hide floor+{CEILING_HEIGHT_M:.1f}m)  "
        f"p95={p95:.1f} cm",
        fontsize=11, pad=2)

    cmap = LinearSegmentedColormap.from_list("cad_recon", [CAD_RGB, RECON_RGB])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, p95))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_iso, fraction=0.03, pad=0.01, shrink=0.9)
    cb.set_label("distance to CAD (cm)")

    fig.legend([Patch(color=cad_rgb), Patch(color=rec_rgb),
                Patch(color=0.5 * (cad_rgb + rec_rgb))],
               ["CAD / near (magenta)", "recon / far (cyan)", "overlap"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        f"{scene_id}  {SCORE_KEY}={score:.2f}  "
        f"(acc={accuracy_mean:.2f}  comp={completeness_mean:.2f})  "
        f"cell={cell:.2f}m",
        y=0.97)
    fig.savefig(path, dpi=160, facecolor="white")
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

    # Recon: ScanNet axisAlignment only (part of scan metadata), then voxelize.
    mesh = load_mesh(mesh_path)
    A = parse_axis_alignment(txt_path) if os.path.isfile(txt_path) else np.eye(4)
    mesh_a = transform_mesh(mesh, A)
    voxel_m = _voxel_size_m(cfg)
    recon_centres = recon_surface_centres(mesh_a, voxel_m, cfg.get("max_surface_voxels"))

    # CAD stays in original frame (visible surface voxels).
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

    # Temporary rigid recon→CAD (invert CAD→recon ICP). In-memory only.
    T_cad_to_recon, yaw0 = align_cad_to_recon(cad_centres, recon_centres)
    T_recon_to_cad = np.linalg.inv(T_cad_to_recon)
    recon_centres_cad = apply_T(recon_centres, T_recon_to_cad)
    mesh_cad = transform_mesh(mesh_a, T_recon_to_cad)

    score, accuracy_mean, completeness_mean = mean_bidirectional_distance_cm(
        cad_centres, recon_centres_cad)

    recon_dir = os.path.join(root, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    png_rel = "recon/cad_comparison.png"
    png_path = os.path.join(root, png_rel)
    yaml_path = os.path.join(recon_dir, "geometry_score.yaml")
    # drop legacy separate isometric if present
    legacy_iso = os.path.join(recon_dir, "cad_distance_isometric.png")

    fd1, tmp_png = tempfile.mkstemp(prefix=".tmp_cmp_", suffix=".png", dir=recon_dir)
    os.close(fd1)
    try:
        render_geometry_report(
            tmp_png, cad_centres, recon_centres_cad, mesh_cad, scan_id,
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
                "method": "temporary_recon_rigid_icp",
                "moved": "recon_in_memory_only",
                "reconstruction_saved": False,
                "axis_alignment_applied_to_recon": True,
                "initial_yaw_deg": int(yaw0),
                "recon_to_cad": T_recon_to_cad.reshape(-1).astype(float).tolist(),
            },
            "comparison": png_rel,
        }
        atomic_write_text(yaml_path, yaml.safe_dump(doc, sort_keys=False))
        os.replace(tmp_png, png_path)
        if os.path.isfile(legacy_iso):
            try:
                os.unlink(legacy_iso)
            except OSError:
                pass
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
