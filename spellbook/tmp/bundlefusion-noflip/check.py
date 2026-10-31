"""Back-project depth as stored in <sid>.sens through its poses; distance to the output mesh,
plain vs mirrored pixels. Then CAD score of the mesh vs its reflection (same scorer functions)."""
import struct, sys, zlib, numpy as np, open3d as o3d, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, '/data/ScanNet'); sys.path.insert(0, '/data/ScanNet/spellbook')
sid = sys.argv[1]; R = f'/data/scannetdata/scans/{sid}'
keep = np.load(R + '/recon/keep.npy'); final = np.load(R + '/recon/final_poses.npy')
frames = {}
with open(f'{R}/{sid}.sens', 'rb') as f:
    ver, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8)); f.read(n)
    mats = np.frombuffer(f.read(16*4*4), '<f4').reshape(4, 4, 4); Kd = mats[2]
    cc, dc = struct.unpack('<ii', f.read(8)); cw, ch, dw, dh = struct.unpack('<IIII', f.read(16))
    shift, = struct.unpack('<f', f.read(4)); nf, = struct.unpack('<Q', f.read(8))
    assert dc == 1, dc  # zlib_ushort
    for i in range(nf):
        T = np.frombuffer(f.read(64), '<f4').reshape(4, 4).astype(float)
        f.read(16); cs, ds = struct.unpack('<QQ', f.read(16)); f.seek(cs, 1)
        if i % 10 == 0:
            frames[i] = (T, np.frombuffer(zlib.decompress(f.read(ds)), '<u2').reshape(dh, dw).astype(float) / shift)
        else:
            f.seek(ds, 1)
fx, fy, cx, cy = Kd[0,0], Kd[1,1], Kd[0,2], Kd[1,2]
print(f'sens frames {nf} depth {dw}x{dh} K {fx:.2f} {cx:.2f} {cy:.2f} keep {len(keep)} final {len(final)}', 'pose match', all(np.allclose(frames[i][0], final[i], atol=1e-3) for i in frames))
V = np.asarray(o3d.io.read_triangle_mesh(f'{R}/{sid}_vh_clean.ply').vertices); t = cKDTree(V)
rng = np.random.default_rng(0); rows = []
for i, (T, d) in frames.items():
    v, u = np.nonzero((d > 0.1) & (d < 4.0))
    if len(u) < 500: continue
    s = rng.choice(len(u), 3000, replace=False); u, v, z = u[s], v[s], d[v[s], u[s]]
    r = []
    for uu in (u, dw - 1 - u):
        X = np.stack([(uu - cx)*z/fx, (v - cy)*z/fy, z, np.ones_like(z)], 1)
        dist, _ = t.query((T @ X.T).T[:, :3], distance_upper_bound=2.0); dist[~np.isfinite(dist)] = 2.0
        r += [np.median(dist), (dist < 0.05).mean()]
    rows.append([i] + r)
a = np.array(rows)
def rep(name, a):
    print(f'{name}: frames {len(a)} plain median {100*np.median(a[:,1]):.2f} cm overlap5cm {np.median(a[:,2]):.3f} | mirrored median {100*np.median(a[:,3]):.2f} cm overlap5cm {np.median(a[:,4]):.3f} | plain better on {(a[:,1]<a[:,3]).mean()*100:.0f}% of frames')
rep('first 200', a[a[:,0] < 200]); rep('all', a)
from reconstruct import geometry_score as g
cfg = g.load_config(); sc = g.scene_config(cfg, 9004)
mesh = g.transform_mesh(g.load_mesh(f'{R}/{sid}_vh_clean_2.ply'), g.parse_axis_alignment(f'{R}/{sid}.txt'))
vm = float(cfg['voxel_mm'])/1000; rc = g.recon_surface_centres(mesh, vm, cfg.get('max_surface_voxels'))
o = np.asarray(sc['grid_origin_m'], float); cad = g.load_mesh(sc['cad_ply'])
ci = g.surface_voxels(cad, o, vm, sc.get('bounds_min_m'), sc.get('bounds_max_m'), cfg.get('max_surface_voxels'))
cs = set(map(tuple, ci.tolist())); vi = np.array([i for i in np.load(sc['visible_voxels'])['indices'].astype(np.int32) if tuple(i) in cs], dtype=np.int32)
cc_ = g.voxel_centres(vi, o, vm)
for name, pts in (('as-is', rc), ('reflected', rc*np.array([-1., 1., 1.]))):
    T, y = g.align_cad_to_recon(cc_, pts)
    sc_, acc, comp = g.mean_bidirectional_distance_cm(cc_, g.apply_T(pts, np.linalg.inv(T)))
    print(f'CAD {name}: mean {sc_:.3f} acc {acc:.3f} comp {comp:.3f} yaw {y}')
