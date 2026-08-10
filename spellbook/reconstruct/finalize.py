"""Finalize: write <id>.sens and <id>.txt once the engine's optimized poses exist.

The .sens color stream reuses the exact JPEG bytes already on disk (frames/color/<i>.jpg);
depth is re-compressed from the PNG pixels (lossless). Frame indices = engine `keep` order.
"""
import os
import struct
import zlib

import numpy as np

from .extract import _COLOR_JPEG, _DEPTH_ZLIB, _DEPTH_SHIFT, write_sens

TXT_KEYS = ("axisAlignment", "colorHeight", "colorWidth", "depthHeight", "depthWidth",
            "fx_color", "fx_depth", "fy_color", "fy_depth", "mx_color", "mx_depth",
            "my_color", "my_depth", "numColorFrames", "numDepthFrames",
            "numIMUmeasurements", "sceneType")


def _read_frame_bytes(frames_dir, i):
    import cv2
    color_path = os.path.join(frames_dir, "color", f"{i}.jpg")
    depth_path = os.path.join(frames_dir, "depth", f"{i}.png")
    with open(color_path, "rb") as f:
        color = f.read()
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None or d.dtype != np.uint16:
        raise RuntimeError(f"bad depth frame {i}")
    return color, zlib.compress(d.astype("<u2").tobytes(), level=6)


def _frame_dims(frames_dir):
    import cv2
    d = cv2.imread(os.path.join(frames_dir, "depth", "0.png"), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError("cannot read depth frame 0 for dimensions")
    return d.shape[1], d.shape[0]


def write_txt(path, K, num_frames, axis_alignment, scene_type, width, height):
    rows = [
        "axisAlignment = " + " ".join(f"{x:.6f}" for x in np.asarray(axis_alignment).ravel()),
        f"colorHeight = {height}",
        f"colorWidth = {width}",
        f"depthHeight = {height}",
        f"depthWidth = {width}",
        f"fx_color = {K[0, 0]:.6f}",
        f"fx_depth = {K[0, 0]:.6f}",
        f"fy_color = {K[1, 1]:.6f}",
        f"fy_depth = {K[1, 1]:.6f}",
        f"mx_color = {K[0, 2]:.6f}",
        f"mx_depth = {K[0, 2]:.6f}",
        f"my_color = {K[1, 2]:.6f}",
        f"my_depth = {K[1, 2]:.6f}",
        f"numColorFrames = {num_frames}",
        f"numDepthFrames = {num_frames}",
        "numIMUmeasurements = 0",
        f"sceneType = {scene_type}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")


def finalize(scan_root, scan_id, frames_dir, K, poses, keep):
    """Write <id>.sens for frames `keep` (original extract indices) with engine poses."""
    frames_dir = os.path.abspath(frames_dir)
    width, height = _frame_dims(frames_dir)
    color_bytes = []
    depth_bytes = []
    for i in keep:
        c, d = _read_frame_bytes(frames_dir, i)
        color_bytes.append(c)
        depth_bytes.append(d)
    sens_path = os.path.join(scan_root, f"{scan_id}.sens")
    write_sens(sens_path, poses, color_bytes, depth_bytes,
               [0] * len(keep), [0] * len(keep), K, width, height)
    recon = os.path.join(scan_root, "recon")
    os.makedirs(recon, exist_ok=True)
    np.save(os.path.join(recon, "final_poses.npy"), poses)
    return sens_path, width, height
