"""Isaac ROS 4.5 cuVSLAM + nvblox on the SVO2 replay (snap docker, public-deb image).

Image zed-isaac-nvblox:spellbook comes from containers/isaac/Dockerfile (public apt repos, no NGC
login, #22 #42). Inside one component_container_mt: ZED SVO playback (half-res, positional tracking
off), cuVSLAM stereo VO (tracking_mode 0) on rectified gray, nvblox static_tsdf fed by cuVSLAM TF.
A line-buffered rclpy recorder maps poses to frames through /zed/zed_node/status/svo (frame_ts ==
image stamp with use_svo_timestamps). The mesh comes from /nvblox_node/save_ply.

Presets: SPELLBOOK_ISAAC_PRESET = official_default (#48 baseline) | optimized_cad_best (#42) |
conf100_best (default, #42 2026-09-26). Keys are the seven varied by the campaigns; everything
else stays at the nvblox node defaults, which is what the "official default" runs used.
"""
import json
import os
import shutil
import subprocess
import time
from collections import deque

import numpy as np
import yaml

from .. import preset_env
from ..containers import gpu_args, mount, user_args

IMAGE = "zed-isaac-nvblox:spellbook"
CONTAINER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "containers", "isaac")
# ZED AI models are downloaded and optimized per GPU on first use; keep them across runs.
ZED_RESOURCES = "/data/spellbook/cache/zed_resources"  # [inferred]
TIMEOUT = 1800
SAVE_PLY_TIMEOUT_S = 600
COVERAGE_MIN = 0.80  # #42 "<80% gate"
MESH_VERTICES_MIN = 100_000  # #42 "<100k gate"

GPU_POLICY = "managed"
SERIAL = True

PRESETS = {
    "official_default": dict(depth_mode="NEURAL_LIGHT", map_clearing_radius_m=5.0,
                             decay_tsdf_rate_hz=5.0, voxel_size=0.05,
                             max_integration_distance_m=7.0, truncation_distance_vox=4.0,
                             max_weight=5.0),
    # Adapter hash 383834ab in #42 (source lost, values verbatim).
    "optimized_cad_best": dict(depth_mode="NEURAL", map_clearing_radius_m=0.0,
                               decay_tsdf_rate_hz=5.0, voxel_size=0.05,
                               max_integration_distance_m=7.0, truncation_distance_vox=4.0,
                               max_weight=10.0),
}
PRESETS["conf100_best"] = dict(PRESETS["optimized_cad_best"], depth_confidence=100)
DEFAULT_PRESET = "conf100_best"
DEFAULT_DEPTH_CONFIDENCE = 95  # wrapper default, echoed in the #42 official-default run
VARIANT_NAMES = {"official_default": "official-default",
                 "optimized_cad_best": "neural-noclear-w10",
                 "conf100_best": "conf100_best"}  # [inferred] #42 names no run.json variant for it

CAMERA_MODEL = "zedx"
BASE_FRAME = "zed_camera_link"  # [inferred] URDF root: a second parent for camera_center breaks TF
CENTER_FRAME = "zed_camera_center"
LEFT_OPTICAL = "zed_left_camera_frame_optical"
RIGHT_OPTICAL = "zed_right_camera_frame_optical"
STATUS_END = 2  # zed_msgs/SvoStatus


def selected_preset():
    return preset_env("isaac") or DEFAULT_PRESET


def preset_config(name=None):
    name = name or selected_preset()
    if name not in PRESETS:
        raise ValueError(f"unknown SPELLBOOK_ISAAC_PRESET {name!r}; choose from {sorted(PRESETS)}")
    return dict(PRESETS[name])


def domain_id(gpu):
    """Per-lease DDS domain: a concurrent ZED stream on domain 0 contaminated a run (#22)."""
    return 100 + int(gpu)


def nvblox_params(cfg):
    return {
        "voxel_size": cfg["voxel_size"],
        "mapping_type": "static_tsdf",
        "global_frame": "odom",
        "use_tf_transforms": True,
        "num_cameras": 1,
        "use_depth": True,
        "use_color": False,  # ZED publishes bgr8, which nvblox rejects: meshes are gray (#22)
        "use_lidar": False,
        "input_qos": "SENSOR_DATA",  # official specialization nvblox_zed.yaml
        "map_clearing_radius_m": cfg["map_clearing_radius_m"],
        "map_clearing_frame_id": BASE_FRAME,
        "decay_tsdf_rate_hz": cfg["decay_tsdf_rate_hz"],
        "static_mapper.projective_integrator_max_integration_distance_m":
            cfg["max_integration_distance_m"],
        "static_mapper.projective_integrator_truncation_distance_vox":
            cfg["truncation_distance_vox"],
        "static_mapper.projective_integrator_max_weight": cfg["max_weight"],
    }


def cuvslam_params():
    return {
        "tracking_mode": 0,
        "num_cameras": 2,
        "min_num_images": 2,
        "rectified_images": True,
        "enable_image_denoising": False,
        "enable_slam_visualization": False,
        "enable_landmarks_view": False,
        "enable_observations_view": False,
        "enable_localization_n_mapping": False,  # [inferred] official nvblox vslam.launch.py
        "publish_odom_to_base_tf": True,
        "publish_map_to_odom_tf": False,  # [inferred] mesh and poses both live in odom
        "odom_frame": "odom",
        "base_frame": BASE_FRAME,
        "camera_optical_frames": [LEFT_OPTICAL, RIGHT_OPTICAL],
    }


def zed_override(cfg, svo_name):
    """ZED wrapper override file (node parameters, on top of the stock stereo yaml)."""
    params = {
        "svo": {"svo_path": f"/svo/{svo_name}", "use_svo_timestamps": True,
                "publish_svo_clock": False, "svo_loop": False, "svo_realtime": False},
        "general": {"camera_name": "zed", "camera_model": CAMERA_MODEL,
                    "grab_frame_rate": 15, "pub_resolution": "CUSTOM",
                    "pub_downscale_factor": 2.0},  # [inferred] half-res; #42 b2 full-res was the delta
        # Stock yaml advertises neither: cuVSLAM subscribes to the rectified gray pair.
        "video": {"publish_left_right": True, "publish_gray": True},
        "depth": {"depth_mode": cfg["depth_mode"], "depth_stabilization": 0,
                  "depth_confidence": cfg.get("depth_confidence", DEFAULT_DEPTH_CONFIDENCE)},
        "pos_tracking": {"pos_tracking_enabled": False, "publish_tf": False,
                         "publish_map_tf": False},
        "sensors": {"publish_imu_tf": False},
    }
    return {"/**": {"ros__parameters": params}}


def write_run_files(export_dir, cfg, svo_name):
    """Config consumed by containers/isaac/graph.launch.py."""
    with open(os.path.join(export_dir, "run_params.json"), "w") as f:
        json.dump({"camera_model": CAMERA_MODEL, "cuvslam": cuvslam_params(),
                   "nvblox": nvblox_params(cfg)}, f, indent=2, sort_keys=True)
    with open(os.path.join(export_dir, "zed_override.yaml"), "w") as f:
        yaml.safe_dump(zed_override(cfg, svo_name), f, sort_keys=False)


def _image_id():
    r = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else None


def preflight():
    name = selected_preset()
    if name not in PRESETS:
        return f"unknown SPELLBOOK_ISAAC_PRESET {name!r}; choose from {sorted(PRESETS)}"
    r = subprocess.run(["docker", "version", "-f", "{{.Server.Version}}"], capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        return f"docker daemon not reachable: {(r.stderr or r.stdout).strip()[-200:]}"
    if _image_id() is None:
        return (f"docker image {IMAGE} missing: docker build -t {IMAGE} "
                f"{CONTAINER_DIR}")
    return None


def provenance():
    name = selected_preset()
    return {"variant": VARIANT_NAMES.get(name, name), "preset": name,
            "config": preset_config(name), "image": IMAGE, "image_id": _image_id()}


def provenance_after(work):
    path = os.path.join(work, "isaac_export", "track_stats.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {"track": yaml.safe_load(f)}


def gpu_check(gpu=None, lease_fd=None, hold_seconds=5):
    """Native distribution check under the lease (no results)."""
    if gpu is None or lease_fd is None:
        return dict(status="fail", reason="managed check requires gpu + lease_fd")
    reason = preflight()
    if reason:
        return dict(status="blocked", reason=reason)
    start = time.time()
    r = subprocess.run(
        ["docker", "run", "--rm", "--name", f"spellbook-gpu-check-isaac-{os.getpid()}"]
        + gpu_args(gpu) + [IMAGE, "bash", "-c",
                           'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; '
                           f"nvidia-smi -L; sleep {int(hold_seconds)}"],
        capture_output=True, text=True, timeout=hold_seconds + 120, pass_fds=(lease_fd,))
    if r.returncode != 0:
        return dict(status="fail",
                    reason=f"isaac gpu_check rc={r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    if f"CUDA_VISIBLE_DEVICES={gpu}" not in r.stdout:
        return dict(status="fail", reason=f"assigned env missing in output: {r.stdout[-300:]}")
    return dict(status="pass", policy="managed", physical_gpu=gpu, visible_gpu=gpu,
                lease_fd=lease_fd, runtime=f"docker {IMAGE}",
                seconds=round(time.time() - start, 1), reason=None)


def quat_to_rot(qx, qy, qz, qw):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def pose_matrix(x, y, z, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
    T[:3, 3] = (x, y, z)
    return T


def _rows(path):
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


def load_static_edges(path):
    """tf_static.txt -> [(parent, child, T_parent_child)]."""
    return [(r[0], r[1], pose_matrix(*map(float, r[2:9]))) for r in _rows(path) if len(r) >= 9]


def static_transform(edges, source, target):
    """T_source_target by breadth-first search over the static graph. The tree branches at the
    camera center (left/right/imu), so a greedy parent chain finds the wrong path (#22)."""
    graph = {}
    for parent, child, T in edges:
        graph.setdefault(parent, []).append((child, T))
        graph.setdefault(child, []).append((parent, np.linalg.inv(T)))
    todo = deque([(source, np.eye(4))])
    seen = {source}
    while todo:
        frame, T = todo.popleft()
        if frame == target:
            return T
        for neighbour, step in graph.get(frame, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                todo.append((neighbour, T @ step))
    raise RuntimeError(f"[isaac] no static TF path {source} -> {target} "
                       f"(frames: {sorted(graph)})")


def status_frame_offset(status_rows):
    """Subtract this from SvoStatus.frame_id to get the 0-based grab index the pool uses. The
    wrapper publishes (last frame_ts, getSVOPosition()) before each grab; the SDK docs allow both
    an index and a count of frames read, so read it off the data: a playing row at frame_id ==
    total only exists for the count form."""  # [inferred]
    playing = [(int(r[1]), int(r[2])) for r in status_rows if int(r[3]) != STATUS_END]
    return 1 if any(frame >= total for frame, total in playing) else 0


def match_frames(status_rows, pose_rows, pool_size):
    """Pool frame index -> pose row by exact stamp: SvoStatus.frame_ts == image stamp == pose stamp
    with use_svo_timestamps. The first pose for a repeated stamp wins; frames past the pool (the
    extractor drops the last SVO frame) and END status rows are ignored."""
    offset = status_frame_offset(status_rows)
    stamp_to_frame = {}
    for r in status_rows:
        if int(r[3]) == STATUS_END:
            continue
        frame = int(r[1]) - offset
        if 0 <= frame < pool_size:
            stamp_to_frame.setdefault(r[0], frame)
    matched = {}
    for r in pose_rows:
        frame = stamp_to_frame.get(r[0])
        if frame is not None and frame not in matched:
            matched[frame] = r
    return matched


def svo_end_ok(final_frame, total):
    """Strict gate: playback must reach the last SVO frame (#22)."""
    return total > 0 and final_frame == total - 1


def world_optical(P_world_base, T_base_center, T_center_optical):
    """P_world_center @ T_center_optical."""
    P_world_center = P_world_base @ T_base_center
    return P_world_center @ T_center_optical


def ply_counts(path):
    """(vertices, faces) from a PLY header."""
    verts = faces = 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith(b"element vertex"):
                verts = int(line.split()[-1])
            elif line.startswith(b"element face"):
                faces = int(line.split()[-1])
            elif line == b"end_header":
                return verts, faces
    raise RuntimeError(f"[isaac] {path}: no PLY header")


def _flatten(tree, prefix=""):
    out = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def effective_params(export_dir):
    """Flat {nvblox: {...}, zed: {...}} from the `ros2 param dump` files; missing -> empty."""
    result = {}
    for key, name in (("nvblox", "effective_nvblox.yaml"), ("zed", "effective_zed.yaml")):
        path = os.path.join(export_dir, name)
        doc = None
        if os.path.isfile(path) and os.path.getsize(path):
            try:
                with open(path) as f:
                    doc = yaml.safe_load(f)
            except yaml.YAMLError:
                doc = None
        flat = {}
        for node in (doc or {}).values():
            if isinstance(node, dict) and "ros__parameters" in node:
                flat = _flatten(node["ros__parameters"])
        result[key] = flat
    return result


def effective_mismatches(cfg, effective):
    """Requested settings that the running nodes did not take. An absent dump is not a mismatch
    (it is best effort under load); a differing value is."""
    want = {("nvblox", k): v for k, v in nvblox_params(cfg).items()
            if k in ("voxel_size", "map_clearing_radius_m", "decay_tsdf_rate_hz")
            or k.startswith("static_mapper.")}
    want[("zed", "depth.depth_mode")] = cfg["depth_mode"]
    want[("zed", "depth.depth_confidence")] = cfg.get("depth_confidence",
                                                      DEFAULT_DEPTH_CONFIDENCE)
    bad = []
    for (node, key), value in want.items():
        got = effective.get(node, {}).get(key)
        if got is None:
            continue
        same = abs(float(got) - float(value)) < 1e-6 if isinstance(value, (int, float)) \
            else got == value
        if not same:
            bad.append(f"{node}:{key} want {value!r} got {got!r}")
    return bad


def seed_zed_resources():
    """The run mounts ZED_RESOURCES over the image's model dir: copy the shipped models once."""
    os.makedirs(ZED_RESOURCES, exist_ok=True)
    if os.listdir(ZED_RESOURCES):
        return
    subprocess.run(["docker", "run", "--rm"] + user_args() + mount(ZED_RESOURCES, "/cache")
                   + [IMAGE, "bash", "-c", "cp -r /usr/local/zed/resources/. /cache/"],
                   check=True, capture_output=True, timeout=300)


def docker_command(name, gpu, export_dir, log_dir, svo_dir, overall_s, prime=False):
    return (["docker", "run", "--rm", "--name", name, "--network", "host", "--shm-size", "2g",
             "-e", f"ISAAC_PRIME={int(prime)}",
             "-e", f"ROS_DOMAIN_ID={domain_id(gpu)}",
             "-e", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
             "-e", "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
             "-e", "HOME=/tmp/isaac_home"]
            + gpu_args(gpu) + user_args()
            + mount(export_dir, "/output") + mount(log_dir, "/logs")
            + mount(svo_dir, "/svo", read_only=True)
            + mount(CONTAINER_DIR, "/isaac", read_only=True)
            + mount(ZED_RESOURCES, "/usr/local/zed/resources")
            + [IMAGE, "/bin/bash", "/isaac/inside.sh", str(overall_s), str(SAVE_PLY_TIMEOUT_S)])


def _run_container(cmd, name, log_path, lease_fd):
    with open(log_path, "wb") as logf:
        try:
            return subprocess.run(cmd, timeout=TIMEOUT, stdout=logf, stderr=subprocess.STDOUT,
                                  pass_fds=(lease_fd,)).returncode
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"[isaac] container timed out after {TIMEOUT}s, see {log_path}")
        finally:
            # Killing the CLI (timeout, runtime cap) leaves the container on the leased GPU.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=120)


def reconstruct(work, root, gpu=None, lease_fd=None):
    if gpu is None or lease_fd is None:
        raise RuntimeError("isaac is GPU_POLICY managed: gpu + lease_fd required "
                           "(run.py acquires the settings-pool lease)")
    marker = os.path.join(work, "svo_path.txt")
    if not os.path.exists(marker):
        raise RuntimeError("[isaac] no svo_path marker; run prepare first")
    with open(marker) as f:
        svo = f.read().strip()
    pool_size = len(os.listdir(os.path.join(work, "frames", "color")))

    preset = selected_preset()
    cfg = preset_config(preset)
    export_dir = os.path.join(work, "isaac_export")
    log_dir = os.path.join(work, "logs")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    seed_zed_resources()
    write_run_files(export_dir, cfg, os.path.basename(svo))

    name = f"spellbook-isaac-{os.getpid()}"
    cmd = docker_command(name, gpu, export_dir, log_dir, os.path.dirname(svo), TIMEOUT - 60)
    log_path = os.path.join(log_dir, "isaac.log")
    print(f"[isaac] preset={preset} gpu={gpu} domain={domain_id(gpu)} {cfg}")
    rc = _run_container(cmd, name, log_path, lease_fd)
    if rc != 0:
        raise RuntimeError(f"[isaac] container exit {rc}, see {log_path} + {log_dir}/graph.log"
                           + ("; ZED model not optimized yet? python -m "
                              "spellbook.reconstruct.engines.isaac --prime" if rc == 3 else ""))

    native, poses, keep, stats = collect_export(export_dir, work, pool_size, cfg)
    stats.update(preset=preset, gpu=int(gpu), domain=domain_id(gpu))
    with open(os.path.join(export_dir, "track_stats.yaml"), "w") as f:
        yaml.safe_dump(stats, f, sort_keys=False)
    print(f"[isaac] {stats}")
    return native, poses, keep, "cuvslam_map"


def prime_presets():
    """One preset per distinct ZED depth mode."""
    by_mode = {}
    for name, cfg in PRESETS.items():
        by_mode.setdefault(cfg["depth_mode"], name)
    return sorted(by_mode.values())


def prime_models(scenes):
    """Fill ZED_RESOURCES outside the timed region: the SDK optimizes a model for ~6 min on first
    open, per depth mode and resolution, and the cap would kill a real run first. Replays each
    scene's SVO through the real graph until its first frame.
    python -m spellbook.reconstruct.engines.isaac --prime --scene 9004 9009"""  # [inferred]
    import sys
    _spellbook = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _spellbook)
    import jobs
    from evaluation.benchmark import load_settings
    from utils.gpu import gpu_lease
    from .. import svo_path

    settings = load_settings()
    argv = [sys.executable, "-m", "spellbook.reconstruct.engines.isaac", "--prime", "--scene"] \
        + [str(s) for s in scenes]
    with jobs.job_context(argv, kind="command"), \
            gpu_lease(settings["gpu_pool"], settings["scannet_root"]) as lease:
        seed_zed_resources()
        for scene in scenes:
            svo = svo_path(scene)
            for preset in prime_presets():
                work = os.path.join(os.path.dirname(ZED_RESOURCES), "isaac_prime",
                                    f"scene{scene}_{preset}")
                shutil.rmtree(work, ignore_errors=True)
                os.makedirs(os.path.join(work, "logs"))
                write_run_files(work, preset_config(preset), os.path.basename(svo))
                name = f"spellbook-isaac-prime-{os.getpid()}"
                started = time.monotonic()
                cmd = docker_command(name, lease.index, work, os.path.join(work, "logs"),
                                     os.path.dirname(svo), TIMEOUT - 60, prime=True)
                rc = _run_container(cmd, name, os.path.join(work, "logs", "isaac.log"),
                                    lease.fileno())
                print(f"[isaac-prime] scene{scene} {preset} rc={rc} "
                      f"{time.monotonic() - started:.0f}s", flush=True)
                if rc != 0:
                    raise RuntimeError(f"[isaac] prime failed, see {work}/logs")


def collect_export(export_dir, work, pool_size, cfg):
    """Gates + poses from a finished container run."""
    def path(name):
        return os.path.join(export_dir, name)

    if not os.path.isfile(path("svo_end.txt")):
        raise RuntimeError(f"[isaac] no svo_end.txt in {export_dir}")
    final_frame, total = map(int, _rows(path("svo_end.txt"))[0][:2])
    if not svo_end_ok(final_frame, total):
        raise RuntimeError(f"[isaac] SVO stopped at {final_frame}/{total}")

    bad = effective_mismatches(cfg, effective_params(export_dir))
    if bad:
        raise RuntimeError(f"[isaac] settings not applied: {bad}")

    if not os.path.isfile(path("mesh.ply")):
        raise RuntimeError(f"[isaac] no mesh.ply from save_ply in {export_dir}")
    vertices, faces = ply_counts(path("mesh.ply"))
    if vertices < MESH_VERTICES_MIN:
        raise RuntimeError(f"[isaac] mesh {vertices} vertices < {MESH_VERTICES_MIN} gate")

    pose_rows = [r for r in _rows(path("poses.txt")) if len(r) >= 8]
    status_rows = [r for r in _rows(path("svo_status.txt")) if len(r) >= 4]
    matched = match_frames(status_rows, pose_rows, pool_size)
    coverage = len(matched) / pool_size if pool_size else 0.0
    if coverage < COVERAGE_MIN:
        raise RuntimeError(f"[isaac] {len(matched)}/{pool_size} frames with a pose "
                           f"({coverage:.1%}) < {COVERAGE_MIN:.0%} gate")

    edges = load_static_edges(path("tf_static.txt"))
    T_base_center = static_transform(edges, BASE_FRAME, CENTER_FRAME)
    T_center_optical = static_transform(edges, CENTER_FRAME, LEFT_OPTICAL)
    keep = sorted(matched)
    poses = np.stack([world_optical(pose_matrix(*map(float, matched[i][1:8])), T_base_center,
                                    T_center_optical) for i in keep])
    if not np.isfinite(poses).all():
        raise RuntimeError("[isaac] non-finite pose")

    native = os.path.join(work, "engine_native.ply")
    shutil.copyfile(path("mesh.ply"), native)
    vo = [r[1] for r in _rows(path("vo_status.txt")) if len(r) >= 2]
    stats = dict(matched=len(keep), pool=pool_size, coverage=round(coverage, 4),
                 pose_messages=len(pose_rows), vo_messages=len(vo),
                 vo_state_counts={s: vo.count(s) for s in sorted(set(vo))},
                 traj_m=round(float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0),
                                                   axis=1).sum()), 3),
                 mesh_vertices=vertices, mesh_faces=faces, svo_end=final_frame, svo_total=total,
                 status_frame_offset=status_frame_offset(status_rows),
                 tf_static=len(edges))
    return native, poses, keep, stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prime the ZED model cache for the Isaac engine")
    ap.add_argument("--prime", action="store_true", required=True)
    ap.add_argument("--scene", type=int, nargs="+", default=[9004, 9009])
    prime_models(ap.parse_args().scene)
