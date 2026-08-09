# Scout Report: SOTA Alternatives for Multi-Room Indoor 3D Reconstruction (2024-2026)

**Date:** 2026-08-08
**Context:** ZED X .svo2 stereo recordings. Target: dense colored point cloud + mesh + posed RGB-D frames, ScanNet-quality, good for pretrained open-vocab 3D instance segmentation. Known failure modes of current stack: VSLAM drift across rooms (no revisits), glass-window tracking loss -> spike artifacts, sparse/blobby geometry. HW: RTX A4000-class 16 GB (Ampere), multiple GPUs, driver 580, no sudo (rootless podman + conda only), Ubuntu 24.04.

**Our unique input capability:** rectified left/right frames + ZED SDK metric depth + ZED SDK poses + known intrinsics, at any resolution. Candidates that exploit these priors rank higher.

---

## 1. Feed-forward 3D foundation models

Common pattern: a transformer regresses camera poses + dense metric point maps/depth from N images in one pass. All are conda/pip installable (no sudo; compiled CUDA kernels via conda cuda-toolkit work fine). Key scaling fact: **VRAM grows roughly linearly with frame count** (global attention over frame tokens). Measured proxy (VGGT-Omega-1B, A100, 624x416 inputs, official README):

| frames | 1 | 10 | 25 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| peak GB | 6.0 | 6.7 | 7.8 | 9.7 | 13.4 | 20.8 | 43.2 |

=> on 16 GB, ~100-140 frames per pass. Thousands of frames require sliding-window / chunked inference with overlap, or the SLAM-style variants with submap graphs. VGGT repo note (May 2026): a memory fix now allows 2-3x more frames at same budget.

| Model | Venue / repo | Metric scale | Ingests K / depth / poses? | 16 GB VRAM | Multi-room verdict |
|---|---|---|---|---|---|
| DUSt3R | CVPR'24, naver/dust3r | metric (up to scale) | K optional; no depth | pair-wise only | superseded |
| MASt3R | ECCV'24, naver/mast3r | metric | K optional; no depth | pair-wise only | superseded (use MASt3R-SLAM) |
| Spann3R | arXiv 2408.16061, HengyiWang/spann3r | metric | K optional; no depth | ~10 GB, sequential | incremental; quality below MASt3R-SLAM |
| CUT3R | CVPR'25 Oral, CUT3R/CUT3R | metric | no conditioning | linear in frames; ~50-100 @512 on 16 GB | strong offline sequence recon, but ignores depth |
| Fast3R | CVPR'25, facebookresearch/fast3r | metric | no conditioning | efficient attention, hundreds of frames | same class as CUT3R; less maintained |
| VGGT | CVPR'25 Best Paper, facebookresearch/vggt | metric | K optional; no depth | see table above | foundation; direct use limited |
| VGGT-Omega | 2026, facebookresearch/vggt-omega | metric | checkpoints gated (auto approval form) | same table | successor of VGGT |
| Pi3 / Pi3X | 2025, yyfz/Pi3 | Pi3X: approx metric | Pi3X: K+depth+poses conditioning | ~10-14 GB, batch-dependent | weights CC-BY-NC; below MapAnything |
| MapAnything | 3DV'26, facebookresearch/map-anything | metric | **yes - any combo of K, depth, poses, metric flags** | minibatch=1 => linear; up to 2000 views on 140 GB; est. ~100-200 views on 16 GB | **best fit for our priors** |
| MASt3R-SLAM | CVPR'25, rmurai0610/MASt3R-SLAM | metric | K via calib file; RGB-only (no depth) | tested RTX 4090; 512x384 feasible on 16 GB (drop res if needed) | **dense SLAM + global submap alignment** |
| VGGT-SLAM 2.0 | NeurIPS'25 / RSS'26, MIT-SPARK/VGGT-SLAM | metric | RGB-only; no depth/poses | keyframes only; fits 16 GB | submap-based + loop closure, large-scale |
| StreamVGGT | ICLR'26, wzzheng/StreamVGGT | metric | no conditioning | causal attention, low mem | streaming; offline chunked is weaker than MapAnything/CUT3R |

### Section verdicts (foundation models)

- **MapAnything** (Meta + CMU, Keetha et al., arXiv 2509.13414): the only model that ingests our *exact priors* - intrinsics + metric depth + poses in any per-view combination - and outputs refined metric geometry **plus refined camera poses** (MVS-style regression). Has MVS mode that runs on COLMAP outputs (calibration+poses as input). Apache-2.0-licensed checkpoint variant available (`facebook/map-anything-apache`). Includes COLMAP export + GLB + profiling scripts. Actively maintained (v1 Sept 2025, current release 2026). Multi-GPU inference possible.
- **MASt3R-SLAM** (Imperial, Murai et al., arXiv 2412.12392): real-time dense SLAM whose tracking/mapping is driven by learned MASt3R 3D priors, not feature matching. This is exactly the fix for textureless/low-texture indoor scenes and glass-adjacent regions where classical VSLAM loses track: the network hallucinates coherent geometry from appearance (metric scale). Global optimization runs over a submap graph -> multi-room drift handled via overlapping submaps even without explicit revisits. No depth input (RGB only), but we can feed ZED rectified frames + intrinsics.
- **VGGT-SLAM 2.0** (MIT-SPARK): dense RGB SLAM on SL(4), submap-based, loop closures, built on VGGT; NeurIPS'25 + RSS'26, actively developed (v2.0 Jan 2026, realtime mode added). RGB-only; no depth/pose priors; VGGT per-keyframe cost at 518px ~8-13 GB. Promising but younger codebase; MASt3R-SLAM is more battle-tested (3.1k stars, many users).
- **CUT3R / Fast3R / StreamVGGT / VGGT / Pi3X:** all solid offline/streaming reconstruction models, but none condition on metric depth or poses, so they discard our stereo advantage; their scale recovery also degrades on long unconstrained sequences. Pi3/Pi3X weights are CC-BY-NC. Use only if MapAnything fails.

---

## 2. Global SfM: GLOMAP / COLMAP / hloc / OpenMVS

- **GLOMAP** (ECCV'24, arXiv 2407.20219) is **deprecated/archived (Mar 2026)**: "fully migrated to COLMAP, where GLOMAP functionality is exposed as the **'global' mapper**". https://github.com/colmap/glomap -> use modern COLMAP (>= 3.12) / pycolmap with `--mapper-type global`. Still 1-2 orders of magnitude faster than incremental mapper with on-par or better quality.
- **hloc** (cvg/Hierarchical-Localization, MIT): SuperPoint + LightGlue + NetVLAD retrieval as the feature front-end, feeding COLMAP/GLOMAP. This is the documented mitigation for low-texture indoor video: learned features + retrieval-based global matching find correspondences across rooms/doors that SIFT+exhaustive matching miss, enabling true global BA (which is the classical answer to "no revisits drift" - one joint optimization over all keyframes, drift cannot accumulate by construction).
- **OpenMVS** (cdcseacave/openmvs): dense MVS over COLMAP poses; can use external depth maps as priors. Room-scale indoor MVS is workable but slower and noisier on textureless walls than RGB-D fusion; useful only as a colored-mesh fallback.
- **Known indoor failure modes (documented):** textureless walls/ceilings -> few features; repetitive structure -> wrong matches (mitigated by learned descriptors + geometric verification); glass/reflections -> phantom matches (mitigate: mask glass regions in feature extraction, skip frames while facing glass); scale gauge / camera motion degeneracy on pure rotation. Metrics: GLOMAP paper explicitly evaluates indoor scenes; with hloc front-end it is the strongest classical option.
- **Fitting our pipeline:** ZED poses + intrinsics can be given to COLMAP as priors (`--use_poses` style absolute pose priors / prior focal), then global BA refines them; alternatively use MapAnything MVS mode on COLMAP output. Cheap to try (pip pycolmap + hloc; no GPU needed, no sudo).

---

## 3. Gaussian-splatting-to-mesh pipelines

All of these **assume accurate input poses** (they optimize radiance/geometry, not camera drift, except GS-SLAM variants which are room-scale). They do NOT solve our drift problem; they are downstream geometry refinements. Colored mesh: yes (baked appearance / extracted surface + texture).

| Method | Venue / repo | Surface quality | Room-scale @16 GB | Verdict |
|---|---|---|---|---|
| SuGaR | ICCV'23, Anttwo/SuGaR | good, older | yes | superseded by GOF/2DGS |
| 2DGS | SIGGRAPH'24, hbb1/2d-gaussian-splatting | very good (surf splatting) | yes | solid choice as post-step |
| GOF | SIGGRAPH Asia'24, autonomousvision/gaussian-opacity-fields | best of class (Marching Tetra) | yes, unbounded scenes | strongest mesh extraction today |
| RaDe-GS | ECCV'24, BaowenZ/RaDe-GS | good, focus on radiance | yes | no advantage for us |
| Nerfstudio splatfacto | nerfstudio-project/nerfstudio | mesh via export only | yes | weaker surface prior; not for mesh |
| PGSR | ECCV'24, zju3dv/PGSR | excellent (planar GS, DTU CD 0.47) | yes; coarse-to-fine helps | strong; explicitly suggests `--max_abs_split_points 0` for weakly textured scenes |

Verdict: use one of these **only after** poses are fixed (e.g. by MapAnything/GO-SLAM/global SfM), as a "colored mesh polish" step. Whole-apartment in one scene on 16 GB: risky - expect to process per-room or heavily downsample. Multi-GPU splitting is not standard for these trainers.

---

## 4. RGB-D fusion successors of BundleFusion

**We already have metric depth + poses; the question here is fusion + drift correction, not tracking.** GLOMAP-style global SfM and these tools are complementary.

| Method | Repo / docs | Loop closure / drift correction | Multi-room | 16 GB / no sudo | Verdict |
|---|---|---|---|---|---|
| **Open3D "Reconstruction System"** | open3d.org/docs/release/tutorial/reconstruction_system/index.html | **fragment-based: local TSDF per fragment, global registration (RGBD feature + ICP) between all fragments, pose graph optimization, global TSDF integration** | yes (designed for it) | trivial (CPU ok) | **the direct, documented successor of the ScanNet/BundleFusion workflow** |
| VDBFusion | PRBonn/vdbfusion, pip | none (assumes poses) | yes (memory-bounded VDB) | trivial, CPU | great fusion backend once poses are fixed |
| Open3D Scalable TSDF (GPU) | open3d python API | none | yes | yes | same role as VDBFusion |
| ElasticFusion | mp3guy/ElasticFusion (2015) | local loop closures only | no (single room, feature-less) | yes but CUDA 8-era | dead, skip |
| Voxblox | ethz-asl/voxblox | none | yes (robotics-scale) | yes (CPU) | inferior mesh quality; skip |
| KinectFusion variants | — | none | no | — | historical only |
| NICE-SLAM | cvg/nice-slam (CVPR'22) | no | no (Replica-room scale) | yes | skip |
| Co-SLAM | NVlabs/co-slam (CVPR'23) | no | no | yes | skip |
| Point-SLAM | eriksandstroem/Point-SLAM (CVPR'23) | no | no | yes | skip |
| SplaTAM | spla-tam/SplaTAM (CVPR'24) | no | no | yes (~10 GB) | skip for our scale |
| GS-ICP-SLAM | Lab-RI/GS-ICP-SLAM (RA-L'24) | no | no (room) | yes | skip |
| **GO-SLAM** | youmi-zym/GO-SLAM (ICCV'23) | **loop closing + full BA, DROID-based**; ScanNet-native input format (color/depth/intrinsic/pose dirs!) | medium (tested on ScanNet rooms, handles long seqs) | yes (~10-12 GB; needs tiny-cuda-nn + libopenexr-dev - apt dep, workaround via conda openexr or ignore) | **best RGB-D-native SLAM candidate with loop closure** |

**Open3D Reconstruction System details (why it targets our drift problem):** it is the maintained, official incarnation of the BundleFusion-style fragment workflow used by ScanNet: split sequence into ~100-frame fragments, TSDF-fuse each fragment (so intra-fragment drift is bounded), then **register fragments against each other globally** (features + ICP, no revisits required - overlap between consecutive fragments suffices), build a pose graph with loop constraints, optimize with robust kernels, and re-integrate everything globally. This directly produces ScanNet-quality colored TSDF meshes at 2-5 mm voxel. Purely pip-installable. Multi-GPU not needed. Documented failure mode: fragment registration fails on very low-texture/glass-heavy fragments -> documented mitigations: larger fragments, more overlap (stride), feature-based (RGB) + geometric registration combined, manual alignment for degenerate cases.

**GO-SLAM details:** consumes ScanNet directory format (color/depth/intrinsic/pose) - i.e. exactly our artifact contract. Stereo mode exists (could use ZED left-right directly). Loop closing + online full BA corrects accumulated drift **when revisits or overlapping geometry exist**; without any revisit across rooms, it degrades like classical VSLAM (honest limitation). Produces colored mesh via fusion. Repo marked "development phase"; single-GPU.

---

## 5. Multi-room / whole-apartment / drift-free-by-design approaches

| Method | Venue / repo | Concept | Verdict |
|---|---|---|---|
| ManhattanSDF | CVPR'22 paper (zhou13.github.io/manhattansdf) | online whole-apartment TSDF with Manhattan-world + room-prior constraints | **official code deleted (404)** - only paper; skip |
| ManhattanSLAM | razayunus/ManhattanSLAM (2019-2021) | planar RGB-D SLAM exploiting Manhattan-world structure -> strong drift reduction in rectangular rooms | old C++ build, realtime, low-res dense output; a niche experiment for structural regularization, moderate effort |
| SceneGraphFusion | CVPR'22, ShunChengWu/SceneGraphFusion | real-time room-level submap fusion with semantic room priors + multiway alignment; multi-room by design | C++ + TensorRT era build; room prior requires segmented room maps; moderate-high effort; concept (submap + room prior) is what we want, implementation dated |
| VGGT-SLAM 2.0 submaps | MIT-SPARK/VGGT-SLAM | learned submaps + loop closures + SL(4) BA; explicitly targets large environments (office loop demo) | RGB-only; see section 1 |
| MASt3R-SLAM submap graph | rmurai0610/MASt3R-SLAM | global optimization over submap graph with 3D-prior matches | see section 1; closest thing to "no-revisit drift correction" among released SLAMs |
| Floorplan/structural priors (general) | research line (e.g. Floor-SP, structural SfM) | constrain geometry/poses to floorplan or dominant planes | no maintained turnkey release; not worth engineering ourselves |
| FOUND-IT / scene graphs | arXiv 2605.25371 (MIT-SPARK) | 3D scene graph on top of VGGT-SLAM 2.0 | downstream of VGGT-SLAM; optional later |

**Glass windows - honest statement across ALL candidates:** no released method handles specular glass natively; every approach (classical and learned) fails on the transparent pane itself. Documented mitigations: (1) mask glass regions (VGGT: mask pixels -> excluded from reconstruction; ZED depth already invalid there), (2) fill holes afterwards from neighboring views / plane completion, (3) record trajectories so glass is never the sole cue. Any spike-artifact rejection must remain in our post-processing regardless of backend. ZED depth spikes at glass will also poison MapAnything/GO-SLAM if fed unfiltered - feed masked depth.

---

## Ranking per candidate: does it solve (a) multiroom drift, (b) 16 GB no-sudo, (c) maturity, (d) effort

| Candidate | (a) drift | (a) glass | (a) dense colored geo | (b) 16 GB | (c) maturity | (d) effort |
|---|---|---|---|---|---|---|
| Open3D RecSys (multiway) | strong (global reg, no revisits) | weak (feature registration) | strong (TSDF colored mesh) | yes (CPU) | excellent, official docs | **L** |
| MapAnything (conditioned MVS) | strong if poses fed in + refined; chunked | medium (masking supported) | strong (metric dense pts + GLB) | yes (~100-200 f/chunk) | high (Meta, active) | **M** |
| MASt3R-SLAM | strong (submap graph global) | medium (learned priors help textureless; glass still bad) | medium (dense pts; need own fusion) | borderline (verify 512x384) | high (CVPR'25, 3.1k stars) | **M** |
| GO-SLAM (RGB-D) | medium (LC needs overlap) | medium | strong (colored mesh out-of-box, ScanNet format) | yes | medium (2023, dev-phase repo) | **M** (low if we ignore apt dep) |
| VGGT-SLAM 2.0 | strong (submaps + LC) | medium | medium | yes | medium-high (RSS'26) | **M** |
| GLOMAP/COLMAP global + hloc | strong (global BA) | medium (mask features) | weak alone (needs OpenMVS/MapAnything for dense) | yes (CPU) | excellent (COLMAP) | **M** |
| CUT3R / Fast3R | medium | medium | medium | borderline | high (CVPR'25) | M |
| PGSR / GOF / 2DGS (pose-refined) | none (pose assumption) | weak | strong meshes | yes (per room) | high | H (only as polish) |
| NICE/Co/Point-SLAM, SplaTAM, GS-ICP-SLAM | none (single room) | weak | strong in room | yes | medium | M (not applicable) |
| ElasticFusion / Voxblox / KinectFusion | none | — | medium | yes | dead | skip |
| ManhattanSDF | (idea is right) | — | — | — | **code deleted** | skip |
| ManhattanSLAM | medium (MW constraint) | medium | low-res surfels | yes | old | M-H (experiment only) |
| SceneGraphFusion | strong concept | medium | strong | yes (RTX) | dated (2022) | H (experiment only) |

---

## Ranked shortlist (top 3 worth adding to the plan)

1. **Open3D Reconstruction System (fragment multiway registration + pose graph)** - the officially documented, maintained successor of the ScanNet/BundleFusion pipeline. Solves multiroom drift via global fragment registration without needing revisits, outputs ScanNet-quality colored TSDF mesh, runs on CPU, pure pip/conda, effort L. Do this FIRST as the new baseline backend (replaces the failed ZED-pose+Open3D TSDF fake with the real documented workflow; also the fragment outputs map 1:1 to our per-room scanning).
2. **MapAnything in conditioned/MVS mode** - feed ZED intrinsics + masked metric depth + ZED poses into the model; get refined metric point clouds + refined poses per chunk; Apache-2.0 checkpoint; COLMAP export feeds step 1 or the GS polish step. This is the highest-value NEW technology: it is the only released model that consumes our stereo priors and it simultaneously upgrades geometry (dense, smooth, metric) and kills drift by re-estimating poses globally per chunk.
3. **MASt3R-SLAM** - learned-prior dense SLAM with global submap alignment; strongest released answer to tracking loss on textureless indoor regions (the class of failure our VSLAM/glass artifacts come from). Metric scale, calib input, conda-only. Verify VRAM at 512x384 on 16 GB first (RTX 4090 is the documented testbed); reduce to 448x336 if needed. Output is dense points - fuse with VDBFusion/Open3D for the colored mesh.

Honorable mentions: GO-SLAM (if we want a depth-native, ScanNet-format, loop-closure SLAM with colored mesh out-of-box - lowest effort after Open3D), VGGT-SLAM 2.0 (watch for 2026 maturity), GOF/PGSR as post-pose mesh polish.

**Explicitly hype / not applicable:**
- DUSt3R, MASt3R, Spann3R standalone (superseded by MASt3R-SLAM/MapAnything/CUT3R)
- StreamVGGT, VGGT, VGGT-Omega plain (no depth/pose conditioning; Omega checkpoints gated)
- Pi3/Pi3X (CC-BY-NC weights, below MapAnything)
- NICE-SLAM, Co-SLAM, Point-SLAM, SplaTAM, GS-ICP-SLAM (single-room benchmarks, no loop closure, no multiroom story)
- ElasticFusion, Voxblox, KinectFusion variants (dead/robotics-only)
- SuGaR, RaDe-GS, splatfacto-for-mesh (superseded by GOF/2DGS/PGSR)
- ManhattanSDF (code removed)
- GLOMAP standalone (archived; use COLMAP global mapper)

---

## Key URLs

- GLOMAP (archived, merged into COLMAP): https://github.com/colmap/glomap | paper https://arxiv.org/abs/2407.20219
- MASt3R: https://github.com/naver/mast3r | arXiv 2406.09756
- MASt3R-SLAM: https://github.com/rmurai0610/MASt3R-SLAM | arXiv 2412.12392 | project https://edexheim.github.io/mast3r-slam/
- Spann3R: https://github.com/HengyiWang/spann3r | arXiv 2408.16061
- CUT3R: https://github.com/CUT3R/CUT3R | arXiv 2501.12387
- Fast3R: https://github.com/facebookresearch/fast3r | arXiv 2501.13928
- VGGT: https://github.com/facebookresearch/vggt | arXiv 2503.11651 | https://vgg-t.github.io/
- VGGT-Omega: https://github.com/facebookresearch/vggt-omega | https://vggt-omega.github.io/ | arXiv 2605.15195
- VGGT-SLAM: https://github.com/MIT-SPARK/VGGT-SLAM | arXiv 2505.12549 (v2.0: RSS 2026)
- Pi3/Pi3X: https://github.com/yyfz/Pi3 | arXiv 2507.13347
- MapAnything: https://github.com/facebookresearch/map-anything | arXiv 2509.13414 | https://map-anything.github.io/
- StreamVGGT: https://github.com/wzzheng/StreamVGGT | arXiv 2507.11539
- hloc: https://github.com/cvg/Hierarchical-Localization
- OpenMVS: https://github.com/cdcseacave/openmvs
- Open3D Reconstruction System: https://www.open3d.org/docs/release/tutorial/reconstruction_system/index.html
- VDBFusion: https://github.com/PRBonn/vdbfusion
- GO-SLAM: https://github.com/youmi-zym/GO-SLAM | arXiv 2309.02436
- NICE-SLAM https://github.com/cvg/nice-slam | Co-SLAM https://github.com/NVlabs/co-slam | Point-SLAM https://github.com/eriksandstroem/Point-SLAM | SplaTAM https://github.com/spla-tam/SplaTAM | GS-ICP-SLAM https://github.com/Lab-RI/GS-ICP-SLAM
- ElasticFusion: https://github.com/mp3guy/ElasticFusion | Voxblox: https://github.com/ethz-asl/voxblox
- SuGaR: https://github.com/Anttwo/SuGaR | 2DGS: https://github.com/hbb1/2d-gaussian-splatting | GOF: https://github.com/autonomousvision/gaussian-opacity-fields | RaDe-GS: https://github.com/BaowenZ/RaDe-GS | PGSR: https://github.com/zju3dv/PGSR | nerfstudio: https://github.com/nerfstudio-project/nerfstudio
- ManhattanSLAM: https://github.com/razayunus/ManhattanSLAM
- SceneGraphFusion: https://github.com/ShunChengWu/SceneGraphFusion | paper: SceneGraphFusion (CVPR 2022)
- ManhattanSDF paper: https://zhou13.github.io/manhattansdf/ (code deleted)
- FOUND-IT: https://arxiv.org/abs/2605.25371
