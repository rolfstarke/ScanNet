import os

import numpy as np


CAMERA_OFF = "off"
CAMERA_PATH = "path"
CAMERA_REPLAY = "replay"
REPLAY_FPS = 10.0
VIDEO_MAX = (664, 498)
FRUSTUM_DEPTH = 0.35
FRUSTUM_LINES = ((0, 1), (0, 2), (0, 3), (0, 4),
                 (1, 2), (2, 3), (3, 4), (4, 1))
REPLAY_TRANSPORT = ("back", "toggle", "forward")
CACHE_WINDOW = 8


def _stems(directory, ext):
    if not os.path.isdir(directory):
        return []
    out = []
    suffix = ext if ext.startswith(".") else "." + ext
    for name in os.listdir(directory):
        if not name.endswith(suffix):
            continue
        stem = name[:-len(suffix)]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


def load_matrix(path):
    if not os.path.isfile(path):
        return None
    try:
        mat = np.loadtxt(path)
    except (OSError, ValueError):
        return None
    mat = np.asarray(mat, dtype=np.float64)
    if mat.size == 16:
        mat = mat.reshape(4, 4)
    if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
        return None
    if abs(np.linalg.det(mat[:3, :3])) < 1e-8:
        return None
    return mat


def is_valid_pose(mat):
    if mat is None:
        return False
    mat = np.asarray(mat)
    if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
        return False
    return abs(np.linalg.det(mat[:3, :3])) >= 1e-8


def discover_frames(frames_dir):
    color = _stems(os.path.join(frames_dir, "color"), ".jpg")
    depth = _stems(os.path.join(frames_dir, "depth"), ".png")
    pose = _stems(os.path.join(frames_dir, "pose"), ".txt")
    rgb_ids = [i for i in color if i in pose]
    proj_ids = [i for i in rgb_ids if i in depth]
    return {
        "dir": frames_dir,
        "rgb_ids": rgb_ids,
        "proj_ids": proj_ids,
        "color": set(color),
        "depth": set(depth),
        "pose": set(pose),
    }


def load_calibration(frames_dir):
    k_color = load_matrix(os.path.join(frames_dir, "intrinsic_color.txt"))
    k_depth = load_matrix(os.path.join(frames_dir, "intrinsic_depth.txt"))
    e_depth = load_matrix(os.path.join(frames_dir, "extrinsic_depth.txt"))
    if k_color is None or k_depth is None:
        return None
    if e_depth is None:
        e_depth = np.eye(4, dtype=np.float64)
    return {
        "K_color": k_color[:3, :3],
        "K_depth": k_depth[:3, :3],
        "E_depth": e_depth,
    }


def load_pose(frames_dir, frame_id):
    return load_matrix(os.path.join(frames_dir, "pose", f"{int(frame_id)}.txt"))


def camera_center(pose):
    return np.asarray(pose, dtype=np.float64)[:3, 3].copy()


def frustum_corners(pose, k_color, width, height, depth=FRUSTUM_DEPTH):
    """Classical camera pyramid for a world-from-camera ScanNet pose.

    Returns (origin, corners): the camera centre and the four world-space
    image-plane corners for pixels [0,0], [w-1,0], [w-1,h-1], [0,h-1]
    projected at positive camera-Z `depth`. Pure function.
    """
    pose = np.asarray(pose, dtype=np.float64)
    k_inv = np.linalg.inv(np.asarray(k_color, dtype=np.float64))
    w = max(1, int(width))
    h = max(1, int(height))
    d = float(depth)
    origin = pose[:3, 3].copy()
    corners = []
    for u, v in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)):
        cam = k_inv @ np.array([float(u), float(v), 1.0]) * d
        corners.append(pose[:3, :3] @ cam + pose[:3, 3])
    return origin, np.stack(corners)


def move_replay_selection(index, direction):
    """Wrap the Back/Pause-Play/Forward selection. Pure function."""
    return (int(index) + int(direction)) % len(REPLAY_TRANSPORT)


def step_replay_frame(frame, count, delta):
    """Wrap a frame step around [0, count). Pure function."""
    count = int(count)
    if count <= 0:
        return int(frame)
    return (int(frame) + int(delta)) % count


def fill_replay_cache(adapter, ids, cached, requested, composite,
                      window=CACHE_WINDOW):
    """Incrementally prefetch every frame into `cached` (in RAM only).

    `composite(entry)` turns an adapter entry into the final RGBA frame.
    Submits at most `window` unanswered requests, then composites every
    ready-but-uncached frame. Returns (newly_cached, status). Never raises:
    per-frame failures are reported through `status`. Never touches disk.
    """
    try:
        adapter.poll()
    except Exception:
        pass
    status = ""
    inflight = 0
    for fid in ids:
        if fid in cached:
            continue
        if fid in requested:
            inflight += 1
            continue
        if inflight >= max(1, int(window)):
            continue
        try:
            entry = adapter.fetch(fid)
        except Exception as exc:
            status = f"replay failed: {exc}"
            continue
        requested.add(fid)
        inflight += 1
        status = entry.get("status") or status
    done = []
    for fid in ids:
        if fid in cached:
            continue
        try:
            ready = bool(adapter.ready(fid))
        except Exception:
            ready = False
        if not ready:
            continue
        try:
            entry = adapter.fetch(fid)
        except Exception as exc:
            status = f"replay failed: {exc}"
            continue
        try:
            cached[fid] = composite(entry)
        except Exception as exc:
            status = f"replay failed: {exc}"
            continue
        requested.discard(fid)
        done.append(fid)
        status = entry.get("status") or status
    return done, status


def trajectory_polylines(poses):
    lines = []
    current = []
    for pose in poses:
        if pose is None or not is_valid_pose(pose):
            if len(current) >= 2:
                lines.append(np.stack(current))
            current = []
            continue
        current.append(camera_center(pose))
    if len(current) >= 2:
        lines.append(np.stack(current))
    return lines


def load_color_rgb(path):
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_rgb(rgb, max_wh=VIDEO_MAX):
    from PIL import Image
    rgb = np.asarray(rgb, dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    target = thumbnail_size((rgb.shape[1], rgb.shape[0]), max_wh)
    if image.size != target:
        image = image.resize(target, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def thumbnail_size(src_wh, max_wh=VIDEO_MAX):
    width, height = (int(src_wh[0]), int(src_wh[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(1.0, max_wh[0] / width, max_wh[1] / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def rgb_to_rgba_bytes(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def advance_playback(playing, frame, count, last_t, now, fps=REPLAY_FPS):
    if not playing or count <= 0:
        return int(frame), last_t
    if last_t is None:
        return int(frame) % count, now
    step = int((now - last_t) * fps)
    if step <= 0:
        return int(frame) % count, last_t
    return (int(frame) + step) % count, last_t + step / fps
