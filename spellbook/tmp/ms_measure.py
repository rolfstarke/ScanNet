"""Diagnostic: measure a Metashape scan slot (mesh extents/components/trajectory distance,
mesh <-> pose <-> prior transform, native depth units). Usage: ms_measure.py <scan_dir>"""
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

root = os.path.abspath(sys.argv[1])
sid = os.path.basename(root)
recon = os.path.join(root, "recon")
pool = os.path.realpath(os.path.join(recon, "frames"))


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum() * len(src)
    res = np.linalg.norm(s * (R @ a.T).T + md - dst, axis=1)
    return s, np.median(res)


r = np.load(os.path.join(recon, "metashape_result.npz"))
keep, raw = r["keep"], r["poses"]
s = float(r["scale"][0])
prior = np.stack([np.loadtxt(os.path.join(pool, "pose", f"{k}.txt")) for k in keep])
traj = lambda p: np.linalg.norm(np.diff(p[:, :3, 3], axis=0), axis=1).sum()
print(f"chunk scale s={s:.6f} aligned={len(keep)} depth={len(r['depth_keep'])}")
print(f"prior traj {traj(prior):.2f} m | T*cam translation traj {traj(raw):.2f} m "
      f"| T*cam/s traj {traj(raw) / s:.2f} m")
print("T*cam -> prior: similarity scale %.4f, median residual %.3f m"
      % umeyama(raw[:, :3, 3], prior[:, :3, 3]))
final_path = os.path.join(recon, "final_poses.npy")
final = np.load(final_path)
print(f"final_poses n={len(final)} traj {traj(final):.2f} m, "
      f"ext {np.ptp(final[:, :3, 3], axis=0).round(2)}")

paths = [("engine_native", os.path.join(recon, "engine_native.ply")),
         ("mesh_aligned", os.path.join(recon, "mesh_aligned.ply")),
         ("vh_clean_2", os.path.join(root, f"{sid}_vh_clean_2.ply"))]
for name, p in paths:
    if not os.path.isfile(p):
        print(name, "absent")
        continue
    m = o3d.io.read_triangle_mesh(p)
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    tc = np.asarray(m.cluster_connected_triangles()[0])
    cnt = np.bincount(tc)
    print(f"{name}: {len(V)}v/{len(F)}f ext {np.ptp(V, axis=0).round(2)} "
          f"min {V.min(0).round(2)} max {V.max(0).round(2)} "
          f"area {m.get_surface_area():.0f} m2 comps {len(cnt)} "
          f"dominant {100 * cnt.max() / len(F):.2f}% tris")
    ref = final if name != "engine_native" else None
    if ref is None:
        continue
    d = cKDTree(ref[:, :3, 3]).query(V)[0]
    print(f"  vertex distance to trajectory: median {np.median(d):.2f} p99 "
          f"{np.percentile(d, 99):.2f} max {d.max():.2f} m; >6 m {100 * (d > 6).mean():.2f}% "
          f"z pcts {np.percentile(V[:, 2], [1, 50, 99]).round(2)}")

# Native depth: units, and residual to the final mesh under final_poses.
import cv2  # noqa: E402

K = np.loadtxt(os.path.join(pool, "intrinsic_depth.txt"))
m = o3d.io.read_triangle_mesh(os.path.join(recon, "mesh_aligned.ply"))
tree = cKDTree(np.asarray(m.vertices))
K0 = np.load(os.path.join(recon, "keep.npy"))
zero, med, res = [], [], []
for j in np.linspace(0, len(K0) - 1, 40).astype(int):
    d = cv2.imread(os.path.join(recon, "metashape_frames", "depth", f"{K0[j]}.png"),
                   cv2.IMREAD_UNCHANGED).astype(float)
    zero.append((d == 0).mean())
    ok = d > 0
    if ok.sum() < 100:
        continue
    med.append(np.median(d[ok]) / 1000)
    v, u = np.nonzero(ok)
    pick = np.random.default_rng(0).choice(len(u), 3000)
    u, v = u[pick], v[pick]
    z = d[v, u] / 1000
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)])
    pw = (final[j] @ pc)[:3].T
    res.append(np.median(tree.query(pw)[0]))
print(f"depth png: mean zero fraction {np.mean(zero):.3f}; frames with depth {len(med)}/40; "
      f"median-of-medians depth {np.median(med) if med else float('nan'):.3f} m; "
      f"backprojection-to-mesh median-of-medians "
      f"{np.median(res) if res else float('nan'):.3f} m")
