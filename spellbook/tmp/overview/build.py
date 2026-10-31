#!/usr/bin/env python3
"""Regenerate the overview's datasets. Edit SETUP/RUNS state here, then re-upload both JSON files."""
import json, sys
SETUP = [
 # component, kind, state, detail
 ["Scene data + ground truth","Data","ready","16 val scenes, label map, ScanNet200 GT"],
 ["ZED recordings + CAD reference","Data","ready","SVOs in custom/raw; CAD PLY matches logged hash"],
 ["Frame pools","Data","ready","9004 and 9009 match the logged frame counts"],
 ["Visible-voxel mask","Data","ready","Rebuilt from pool poses; differs from the lost original"],
 ["ZED SDK 5.4.1 + spellbook env","Env","ready","User-local install; both depth models optimized"],
 ["Metashape 2.3.1","Reconstruction","ready","Licence active; export argument fixed"],
 ["Open3D adapter","Reconstruction","ready","Bounded B1"],
 ["ZED adapter","Reconstruction","ready","native-default, b5-gen1-neural, s2b8-fixedcalib"],
 ["Docker GPU access","Containers","ready","Snap docker: helper handles GPU and /data mounts"],
 ["BundleFusion image + adapter","Containers","running","Rebuilding image; original Dockerfile is lost"],
 ["RTAB-Map image + adapter","Containers","running","Adapter rebuilt from issue logs"],
 ["Isaac nvblox image + adapter","Containers","running","Built from public debs, as logged"],
 ["OpenYOLO3D","Prediction","ready","Env, weights, 16 precomputed proposal files"],
 ["OpenMask3D","Prediction","ready","Env, weights, segment pooling restored"],
 ["Mosaic3D","Prediction","ready","Needs OpenIns3D env for its proposals"],
 ["sam3dis","Prediction","ready","Rewritten in /data/sam3dis from issue 69"],
 ["Open3DIS","Prediction","running","Finishing single-scene test"],
 ["OpenIns3D","Prediction","running","Wiring to logged commit 56ccc22, then single-scene test"],
]
Q = "queued"
RUNS = [
 # stage, method, run, scene, settings, logged, unit, state, result, runtime_s, limit_s
 ["Reconstruction","ZED","baseline","9004","native-default",30.662521,"cm",Q,None,None,361.2],
 ["Reconstruction","ZED","baseline","9009","native-default",None,"",Q,None,None,471.1],
 ["Reconstruction","ZED","best","9004","s2b8-fixedcalib",21.306363,"cm",Q,None,None,361.2],
 ["Reconstruction","Metashape","baseline","9004","native-default",18.096334,"cm",Q,None,None,None],
 ["Reconstruction","Metashape","best","9004","b5-kf350",14.627121,"cm",Q,None,None,None],
 ["Reconstruction","RTAB-Map","baseline","9004","assisted-hybrid s90",20.515862,"cm",Q,None,None,361.2],
 ["Reconstruction","RTAB-Map","baseline","9009","assisted-hybrid s90",None,"",Q,None,None,471.1],
 ["Reconstruction","Isaac","baseline","9004","official_default",38.914,"cm",Q,None,None,361.2],
 ["Reconstruction","Isaac","baseline","9009","official_default",None,"",Q,None,None,471.1],
 ["Reconstruction","Isaac","best","9004","conf100_best",20.505585,"cm",Q,None,None,361.2],
 ["Reconstruction","Open3D","baseline","9004","bounded B1",29.676032,"cm",Q,None,None,361.2],
 ["Reconstruction","Open3D","baseline","9009","bounded B1",None,"",Q,None,None,471.1],
 ["Reconstruction","BundleFusion","baseline","9004","official image",50.745,"cm",Q,None,None,361.2],
 ["Reconstruction","BundleFusion","baseline","9009","official image",None,"",Q,None,None,471.1],
 ["Prediction","OpenYOLO3D","baseline","16 scenes","defaults",0.2129,"AP",Q,None,None,None],
 ["Prediction","OpenYOLO3D","best","16 scenes","official_precomputed",0.3022,"AP",Q,None,None,None],
 ["Prediction","OpenMask3D","baseline","16 scenes","segment pooling",0.1809,"AP",Q,None,None,None],
 ["Prediction","Open3DIS","baseline","16 scenes","official 2D+3D",0.1465,"AP",Q,None,None,None],
 ["Prediction","OpenIns3D","baseline","16 scenes","use_precomputed",0.0496,"AP",Q,None,None,None],
 ["Prediction","OpenIns3D","best","16 scenes","yoloworld",0.05245,"AP",Q,None,None,None],
 ["Prediction","Mosaic3D","baseline","16 scenes","defaults",0.0969,"AP",Q,None,None,None],
 ["Prediction","Mosaic3D","best","16 scenes","b2-geom2",0.10784,"AP",Q,None,None,None],
 ["Prediction","sam3dis","baseline","16 scenes","r4",0.17072,"AP",Q,None,None,None],
 ["Prediction","sam3dis","best","16 scenes","pool engine",0.27519,"AP",Q,None,None,None],
]
overrides = json.load(open("state.json")) if len(sys.argv) > 1 else {}
setup = [dict(zip(("component","kind","state","detail"), r)) for r in SETUP]
runs = []
for i, r in enumerate(RUNS, 1):
    d = dict(zip(("order","stage","method","run","scene","settings","logged","unit","state","result","runtime_s","limit_s"), [i]+r))
    d.update(overrides.get(str(i), {}))
    runs.append(d)
for s in setup:
    s.update(overrides.get(s["component"], {}))
json.dump({"rows": setup}, open("setup.json","w"), indent=0)
json.dump({"rows": runs}, open("runs.json","w"), indent=0)
print(len(setup), len(runs))
