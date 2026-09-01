import os

import numpy as np

from utils.prediction_masks import pack_mask


CAMERA_OFF = "off"
CAMERA_PATH = "path"
CAMERA_REPLAY = "replay"
VIS_THRESH_M = 0.10
CUT_BOUND = 10
REPLAY_FPS = 10.0
VIDEO_MAX = (420, 315)
OVERLAY_ALPHA = 0.55


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


def frustum_corners(pose, k_color, width, height, depth=0.35):
    pose = np.asarray(pose, dtype=np.float64)
    k_inv = np.linalg.inv(np.asarray(k_color, dtype=np.float64))
    pixels = np.array([
        [0.0, 0.0, 1.0],
        [width - 1.0, 0.0, 1.0],
        [width - 1.0, height - 1.0, 1.0],
        [0.0, height - 1.0, 1.0],
    ], dtype=np.float64)
    corners = []
    origin = camera_center(pose)
    for pixel in pixels:
        cam = (k_inv @ pixel) * depth
        world = pose @ np.array([cam[0], cam[1], cam[2], 1.0])
        corners.append(world[:3])
    return origin, np.stack(corners)


def load_depth_m(path):
    import cv2
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return raw.astype(np.float64) / 1000.0


def load_color_rgb(path):
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def project_points(points, pose, calib, depth_m, color_wh, vis_thresh=VIS_THRESH_M,
                   cut_bound=CUT_BOUND):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    valid = np.zeros(n, dtype=bool)
    uv = np.zeros((n, 2), dtype=np.int32)
    z_color = np.full(n, np.inf, dtype=np.float64)
    if n == 0 or pose is None or calib is None or depth_m is None:
        return uv, z_color, valid
    hom = np.concatenate([points, np.ones((n, 1))], axis=1)
    try:
        w2c = np.linalg.inv(np.asarray(pose, dtype=np.float64))
    except np.linalg.LinAlgError:
        return uv, z_color, valid
    p_depth = (w2c @ hom.T).T
    z = p_depth[:, 2]
    front = z > 1e-6
    k_d = calib["K_depth"]
    u_d = np.round(k_d[0, 0] * p_depth[:, 0] / np.where(front, z, 1.0) + k_d[0, 2]).astype(np.int32)
    v_d = np.round(k_d[1, 1] * p_depth[:, 1] / np.where(front, z, 1.0) + k_d[1, 2]).astype(np.int32)
    dh, dw = depth_m.shape[:2]
    inside = (front
              & (u_d >= cut_bound) & (v_d >= cut_bound)
              & (u_d < dw - cut_bound) & (v_d < dh - cut_bound))
    visible = inside.copy()
    if np.any(inside):
        sensor = depth_m[v_d[inside], u_d[inside]]
        visible[inside] = (sensor > 0) & (np.abs(sensor - z[inside]) <= vis_thresh)
    p_color = (np.asarray(calib["E_depth"], dtype=np.float64) @ p_depth.T).T
    zc = p_color[:, 2]
    color_front = zc > 1e-6
    k_c = calib["K_color"]
    cw, ch = color_wh
    u_c = np.round(k_c[0, 0] * p_color[:, 0] / np.where(color_front, zc, 1.0) + k_c[0, 2]).astype(np.int32)
    v_c = np.round(k_c[1, 1] * p_color[:, 1] / np.where(color_front, zc, 1.0) + k_c[1, 2]).astype(np.int32)
    in_color = color_front & (u_c >= 0) & (v_c >= 0) & (u_c < cw) & (v_c < ch)
    valid = visible & in_color
    uv[:, 0] = u_c
    uv[:, 1] = v_c
    z_color = zc
    return uv, z_color, valid


def zbuffer_colors(uv, z, colors, valid, height, width):
    overlay = np.zeros((height, width, 3), dtype=np.float64)
    mask = np.zeros((height, width), dtype=bool)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return overlay, mask
    u = uv[idx, 0]
    v = uv[idx, 1]
    lin = v.astype(np.int64) * int(width) + u.astype(np.int64)
    order = np.lexsort((z[idx], lin))
    lin_sorted = lin[order]
    keep = np.ones(lin_sorted.size, dtype=bool)
    keep[1:] = lin_sorted[1:] != lin_sorted[:-1]
    chosen = idx[order[keep]]
    overlay[uv[chosen, 1], uv[chosen, 0]] = np.asarray(colors[chosen], dtype=np.float64)
    mask[uv[chosen, 1], uv[chosen, 0]] = True
    return overlay, mask


def resize_rgb(rgb, max_wh=VIDEO_MAX):
    from PIL import Image
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    image.thumbnail(max_wh, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def composite_overlay(rgb, overlay, mask, alpha=OVERLAY_ALPHA):
    rgb = np.asarray(rgb, dtype=np.float32)
    out = rgb.copy()
    if mask is not None and np.any(mask):
        color = np.asarray(overlay, dtype=np.float32)
        if color.max() <= 1.0:
            color = color * 255.0
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def rgb_to_rgba_bytes(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def scale_uv(uv, src_wh, dst_wh):
    sx = dst_wh[0] / max(src_wh[0], 1)
    sy = dst_wh[1] / max(src_wh[1], 1)
    scaled = np.asarray(uv, dtype=np.float64).copy()
    scaled[:, 0] = np.round(scaled[:, 0] * sx)
    scaled[:, 1] = np.round(scaled[:, 1] * sy)
    return scaled.astype(np.int32)


def advance_playback(playing, frame, count, last_t, now, fps=REPLAY_FPS):
    if not playing or count <= 0:
        return int(frame), last_t
    if last_t is None:
        return int(frame) % count, now
    step = int((now - last_t) * fps)
    if step <= 0:
        return int(frame) % count, last_t
    return (int(frame) + step) % count, last_t + step / fps


def packed_visible(valid, vertex_count=None):
    mask = np.asarray(valid, dtype=bool)
    if vertex_count is not None and mask.size != int(vertex_count):
        raise ValueError("visible mask length mismatch")
    return pack_mask(mask)


def update_seen_tp(seen, visible_packed, objects):
    seen = set(seen)
    added = 0
    width = int(np.asarray(visible_packed).shape[0])
    for obj in objects:
        if obj.get("verdict") != "tp":
            continue
        key = obj.get("key")
        if key is None or key in seen:
            continue
        packed = obj.get("packed")
        if packed is None:
            sel = obj.get("sel")
            if sel is None:
                continue
            packed = pack_mask(sel)
        packed = np.asarray(packed, dtype=np.uint8)
        if packed.shape[0] != width:
            continue
        if np.any(np.bitwise_and(packed, visible_packed)):
            seen.add(key)
            added += 1
    return seen, added


def owner_colors(vertex_count, objects, color_of):
    owners = np.full(int(vertex_count), -1, dtype=np.int32)
    confs = np.full(int(vertex_count), -np.inf, dtype=np.float64)
    colors = np.zeros((int(vertex_count), 3), dtype=np.float64)
    for index, obj in enumerate(objects):
        sel = obj.get("sel")
        if sel is None:
            continue
        sel = np.asarray(sel, dtype=bool)
        if sel.size != vertex_count:
            continue
        conf = float(obj.get("score") or 0.0)
        better = sel & (conf >= confs)
        owners[better] = index
        confs[better] = conf
        colors[better] = np.asarray(color_of(index, obj), dtype=np.float64)
    return owners, colors
