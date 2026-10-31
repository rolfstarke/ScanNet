"""CAD visibility mask for scene9004 from the shared frame-pool poses.

The original mask was raycast from the lost scene9004_04 ZED scan. It is rebuilt here from all
pool poses; the open3d B1 baseline scan (pool poses, fixed floor transform) only supplies the
mesh used to gauge-align that frame to the CAD.
"""
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reconstruct.geometry_score import (  # noqa: E402
    align_cad_to_recon, file_sha256, load_config, load_mesh, parse_axis_alignment,
    recon_surface_centres, surface_voxels, transform_mesh, voxel_centres)

OLD = "/data/scannetdata/scans/scene9004_40"
POOL = "/data/scannetdata/derived/reconstruction/frames/scene9004/frames"
OUT = "/data/scannetdata/custom/reference/scene9004_visible_voxels.npz"
CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "reconstruct", "geometry_reference.yaml")


def main():
    cfg = load_config(CFG)
    sc = cfg["scenes"]["scene9004"]
    if not sc.get("reference_sha256"):
        sys.exit("run export_scene9004_cad.py first")
    cad = load_mesh(sc["cad_ply"])
    if file_sha256(sc["cad_ply"]) != sc["reference_sha256"]:
        sys.exit("CAD hash mismatch")

    mesh = load_mesh(os.path.join(OLD, "scene9004_40_vh_clean_2.ply"))
    A = parse_axis_alignment(os.path.join(OLD, "scene9004_40.txt"))
    mesh = transform_mesh(mesh, A)

    # Scan poses are M @ P @ D for the kept pool poses P (D flips camera y/z, M is the fixed floor
    # transform); recover M from the scan and map every pool pose into the scan frame.
    final = np.load(os.path.join(OLD, "recon", "final_poses.npy"))
    keep = np.load(os.path.join(OLD, "recon", "keep.npy"))
    pool = np.load(os.path.join(POOL, "camera_to_world.npy"))
    D = np.diag([1.0, -1.0, -1.0, 1.0])
    M = final[0] @ np.linalg.inv(pool[keep[0]] @ D)
    err = np.abs(M @ pool[keep] @ D - final).max()
    if err > 1e-4:
        sys.exit(f"scan poses are not a fixed transform of the pool poses (err={err:.2e})")
    poses = M @ pool @ D
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        sys.exit(f"bad poses shape {poses.shape}")
    if not np.isfinite(poses).all():
        sys.exit("non-finite poses")
    poses = np.einsum("ij,njk->nik", A, poses)

    voxel_m = float(cfg["voxel_mm"]) / 1000.0
    origin = np.asarray(sc["grid_origin_m"], dtype=np.float64)
    print("[ref] voxelizing CAD…", flush=True)
    cad_idx = surface_voxels(cad, origin, voxel_m, sc["bounds_min_m"], sc["bounds_max_m"],
                             cfg["max_surface_voxels"])
    cad_centres = voxel_centres(cad_idx, origin, voxel_m)
    print(f"[ref] CAD surface cells={len(cad_idx)}", flush=True)

    print("[ref] gauge-aligning old mesh…", flush=True)
    # Same rigid four-yaw ICP the scorer uses (geometry_score.score_scan_at), full CAD surface.
    recon_centres = recon_surface_centres(mesh, voxel_m, cfg.get("max_surface_voxels"))
    T_c2r, yaw = align_cad_to_recon(cad_centres, recon_centres)
    T = np.linalg.inv(T_c2r)
    print(f"[ref] selected yaw={yaw}", flush=True)
    aligned_poses = np.einsum("ij,njk->nik", T, poses)

    K = np.loadtxt(os.path.join(POOL, "intrinsic_depth.txt"))[:3, :3]
    d0 = cv2.imread(os.path.join(POOL, "depth", "0.png"), cv2.IMREAD_UNCHANGED)
    src_h, src_w = d0.shape[:2]
    rw, rh = int(cfg["ray_width"]), int(cfg["ray_height"])
    Ks = K.copy().astype(np.float64)
    Ks[0, :] *= rw / src_w
    Ks[1, :] *= rh / src_h

    n = len(aligned_poses)
    n_sel = min(int(cfg["max_poses"]), n)
    sel = np.unique(np.round(np.linspace(0, n - 1, n_sel)).astype(int))
    print(f"[ref] casting {len(sel)} poses at {rw}x{rh}", flush=True)

    # CPU only: never touch user-reserved GPU 0 during reference build.
    device = o3d.core.Device("CPU:0")
    print(f"[ref] raycasting device={device}", flush=True)
    cad_t = o3d.t.geometry.TriangleMesh.from_legacy(cad)
    cad_t = cad_t.to(device)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cad_t)

    hit_counts = {}
    for k, pi in enumerate(sel):
        P = aligned_poses[int(pi)]
        # extrinsic = world_to_camera = inv(camera_to_world)
        ext = np.linalg.inv(P).astype(np.float64)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(Ks),
            extrinsic_matrix=o3d.core.Tensor(ext),
            width_px=rw, height_px=rh)
        rays = rays.to(device)
        ans = scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy()
        ok = np.isfinite(t_hit.reshape(-1))
        if not ok.any():
            continue
        rnp = rays.numpy().reshape(-1, 6)
        origins = rnp[ok, :3]
        dirs = rnp[ok, 3:]
        pts = origins + dirs * t_hit.reshape(-1)[ok, None]
        idx = np.floor((pts - origin) / voxel_m).astype(np.int32)
        for cell in map(tuple, np.unique(idx, axis=0)):
            hit_counts[cell] = hit_counts.get(cell, 0) + 1
        if (k + 1) % 20 == 0 or k + 1 == len(sel):
            print(f"[ref] {k + 1}/{len(sel)} poses, unique cells so far={len(hit_counts)}", flush=True)

    min_views = int(cfg["min_views"])
    cad_set = set(map(tuple, cad_idx.tolist()))
    kept = sorted(c for c, nviews in hit_counts.items()
                  if nviews >= min_views and c in cad_set)
    if len(kept) < 1000:
        sys.exit(f"too few visible cells: {len(kept)}")
    arr = np.array(kept, dtype=np.int32)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, indices=arr)
    sha = file_sha256(OUT)

    with open(CFG) as f:
        cfgw = yaml.safe_load(f)
    cfgw["scenes"]["scene9004"]["visible_voxels_sha256"] = sha
    cfgw["scenes"]["scene9004"]["visibility_yaw_deg"] = int(yaw)
    cfgw["scenes"]["scene9004"]["visibility_n_poses"] = int(len(sel))
    cfgw["scenes"]["scene9004"]["visibility_n_cells"] = int(len(arr))
    with open(CFG, "w") as f:
        yaml.safe_dump(cfgw, f, sort_keys=False)
    print(f"[ref] wrote {OUT} cells={len(arr)} sha={sha[:12]}…", flush=True)


if __name__ == "__main__":
    main()
