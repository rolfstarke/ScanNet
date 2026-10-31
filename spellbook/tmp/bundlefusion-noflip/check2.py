import sys, numpy as np, cv2, open3d as o3d
from scipy.spatial import cKDTree
R='/data/scannetdata/scans/scene9004_50'; P='/data/scannetdata/derived/reconstruction/frames/scene9004/frames'
keep=np.load(R+'/recon/keep.npy'); F=np.load(R+'/recon/final_poses.npy')
K=np.loadtxt(P+'/intrinsic_depth.txt'); fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
rng=np.random.default_rng(0)
for name in ('recon/mesh_aligned.ply','scene9004_50_vh_clean.ply','scene9004_50_vh_clean_2.ply'):
    V=np.asarray(o3d.io.read_triangle_mesh(f'{R}/{name}').vertices); t=cKDTree(V); print(name,len(V))
    rows=[]
    for idx in list(range(0,200,20))+list(range(400,3500,400)):
        i=int(keep[idx]); d=cv2.imread(f'{P}/depth/{i}.png',cv2.IMREAD_UNCHANGED).astype(float)/1000
        v,u=np.nonzero((d>0.1)&(d<4)); s=rng.choice(len(u),3000,replace=False); u,v=u[s],v[s]; z=d[v,u]
        r=[idx,i]
        for uu in (u,1919-u):
            X=np.stack([(uu-cx)*z/fx,(v-cy)*z/fy,z,np.ones_like(z)],1); dist,_=t.query((F[idx]@X.T).T[:,:3],distance_upper_bound=2.0); dist[~np.isfinite(dist)]=2.0
            r+=[round(100*np.median(dist),1),round((dist<.05).mean(),2)]
        rows.append(r)
    print(np.array(rows))
