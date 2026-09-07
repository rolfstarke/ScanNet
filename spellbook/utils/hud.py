import math
import os
import subprocess

from utils.compute_time import format_elapsed


LEFT_WIDTH = 440
RIGHT_WIDTH = 680
VIEWER_TITLE = "ScanNet visualizer"
ROW_H = 20
ELLIPSIS = "..."


def width_for(side):
    return LEFT_WIDTH if side == "left" else RIGHT_WIDTH


def ellipsize(text, max_width, measure):
    if measure(text) <= max_width:
        return text
    if measure(ELLIPSIS) >= max_width:
        return ELLIPSIS
    kept = text
    while kept and measure(kept + ELLIPSIS) > max_width:
        kept = kept[:-1]
    return kept + ELLIPSIS


def format_metric(value):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    return f"{value:.3f}"


def _u32(imgui, r, g, b, a=1.0):
    return imgui.get_color_u32_rgba(r, g, b, a)


def _draw_tree_row(imgui, row, width):
    draw = imgui.get_window_draw_list()
    x, y = imgui.get_cursor_screen_pos()
    indent = 12 * row.get("depth", 0)
    marker = ""
    if row.get("folder"):
        marker = "[-] " if row.get("expanded") else "[+] "
    measure = lambda s: imgui.calc_text_size(s).x
    suffix = row.get("suffix") or ""
    suffix_w = measure(suffix) + 8 if suffix else 0
    avail = max(40.0, width - 20 - indent - suffix_w)
    label = ellipsize(marker + row["label"], avail, measure)
    imgui.set_cursor_screen_pos((x + indent, y))
    active = bool(row.get("path_level"))
    if active:
        imgui.text(label)
    else:
        imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.55, 0.55, 1.0)
        imgui.text(label)
        imgui.pop_style_color()
    if suffix:
        imgui.same_line()
        imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.55, 0.55, 1.0)
        imgui.text(suffix)
        imgui.pop_style_color()
    if row.get("cursor"):
        draw.add_rect(x - 4, y - 2, x + width - 8, y + ROW_H - 2,
                      _u32(imgui, 0.08, 0.08, 0.08, 1.0))
    imgui.dummy(0, 2)


def _draw_left(imgui, payload, height, width, view_start):
    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE |
             imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_INPUTS)
    imgui.set_next_window_position(0, 0)
    imgui.set_next_window_size(width, height)
    imgui.begin("##scannet_left", flags=flags)
    imgui.text("Scenes")
    imgui.separator()
    rows = payload.get("tree") or []
    cursor = next((i for i, row in enumerate(rows) if row.get("cursor")), 0)
    max_rows = max(3, int((height - 36) // ROW_H))
    start, end = 0, len(rows)
    lead = trail = False
    if len(rows) > max_rows:
        body = max(1, max_rows - 2)
        start = view_start[0]
        if cursor < start:
            start = cursor
        elif cursor >= start + body:
            start = cursor - body + 1
        start = min(max(0, start), len(rows) - body)
        end = start + body
        view_start[0] = start
        lead = start > 0
        trail = end < len(rows)
    if lead:
        imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.55, 0.55, 1.0)
        imgui.text(ELLIPSIS)
        imgui.pop_style_color()
    for row in rows[start:end]:
        _draw_tree_row(imgui, row, width)
    if trail:
        imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.55, 0.55, 1.0)
        imgui.text(ELLIPSIS)
        imgui.pop_style_color()
    imgui.end()


TP_TEXT = (0.10, 0.55, 0.25)
ERR_LIGHT = (1.0, 0.72, 0.68)
ERR_DARK = (0.62, 0.08, 0.08)


def _err_shade(count, scale):
    try:
        count = int(count)
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    t = min(1.0, count / max(1, int(scale)))
    return tuple(round(l + (d - l) * t, 3) for l, d in zip(ERR_LIGHT, ERR_DARK))


def _draw_class_row(imgui, name, color, stats=None, scale=1, track=None,
                    selected=False):
    """Text-only class row with a colored [TP/FP/FN] confusion suffix.

    ``stats`` holds {"pred", "gt", "tp", "fp", "fn"}; TP renders green, FP/FN
    render in a white-to-dark-red shade shared across rows (light red = few
    errors, dark red = many). ``track`` is an optional {"fitted",
    "effective", "is_manual"} threshold shown as trailing ``t`` text.
    """
    draw = imgui.get_window_draw_list()
    x, y = imgui.get_cursor_screen_pos()
    dark = imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0)
    swatch = imgui.get_color_u32_rgba(*color, 1.0)
    draw.add_rect_filled(x, y + 2, x + 11, y + 13, swatch, 2)
    pred = int((stats or {}).get("pred") or 0)
    gt = None if stats is None else stats.get("gt")
    head = ("> " if selected else "") + f"{name}  P {pred}  "
    head += f"G {int(gt)}  [" if gt is not None else "G -"
    cx = x + 17
    for text, rgb in _row_segments(head, stats, scale):
        draw.add_text(cx, y, imgui.get_color_u32_rgba(*rgb, 1.0), text)
        try:
            cx += imgui.calc_text_size(text).x
        except Exception:
            cx += 8.0 * len(text)
    tail = ""
    if track is not None:
        eff = track.get("effective")
        if eff is not None:
            tail = f"  t {float(eff):.2f}{'*' if track.get('is_manual') else ''}"
        else:
            tail = "  t n/a"
    if tail:
        draw.add_text(cx, y, dark, tail)
    imgui.dummy(315, 17)


def _row_segments(head, stats, scale):
    """(text, rgb) segments for one class row."""
    dark = (0.08, 0.08, 0.08)
    segments = [(head, dark)]
    if stats is None or stats.get("gt") is None:
        return segments
    try:
        tp = int(stats.get("tp") or 0)
    except (TypeError, ValueError):
        tp = 0
    segments.append((f"TP {tp}", TP_TEXT if tp else dark))
    segments.append((" / ", dark))
    try:
        fp = int(stats.get("fp") or 0)
    except (TypeError, ValueError):
        fp = 0
    segments.append((f"FP {fp}", _err_shade(fp, scale) or dark))
    segments.append((" / ", dark))
    fn = stats.get("fn")
    try:
        fn = int(fn)
    except (TypeError, ValueError):
        fn = None
    if fn is None:
        segments.append(("FN -", dark))
    else:
        segments.append((f"FN {fn}", _err_shade(fn, scale) or dark))
    segments.append(("]", dark))
    return segments


def _draw_threshold_track(imgui, track):
    # Retained for API compatibility; the track is now trailing text.
    return


def _draw_setting_row(imgui, row, width, settings_focus):
    draw = imgui.get_window_draw_list()
    x, y = imgui.get_cursor_screen_pos()
    focused = bool(row.get("focused") and settings_focus)
    imgui.text(f"{row['name']}:")
    first = True
    max_x = x + width - 12
    for opt in row.get("options") or []:
        disabled = not opt.get("enabled", True)
        token = opt["text"] if first else f" | {opt['text']}"
        tw = imgui.calc_text_size(token).x
        cx, cy = imgui.get_cursor_screen_pos()
        wrap = (not first) and cx + tw > max_x
        if wrap:
            imgui.new_line()
            token = opt["text"]
            tw = imgui.calc_text_size(token).x
            cx, cy = imgui.get_cursor_screen_pos()
        else:
            imgui.same_line()
            cx, cy = imgui.get_cursor_screen_pos()
        first = False
        if opt.get("sel") and focused and not disabled:
            draw.add_rect(cx - 3, cy - 2, cx + tw + 3, cy + 16,
                          imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0))
        active = opt.get("kind") in ("applied", "pending")
        if disabled:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.70, 0.70, 0.70, 1.0)
            imgui.text(token)
            imgui.pop_style_color()
        elif active:
            imgui.text(token)
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.55, 0.55, 1.0)
            imgui.text(token)
            imgui.pop_style_color()
    imgui.dummy(0, 4)


def _send(keys_out, payload):
    if keys_out is None:
        return
    try:
        keys_out.send(payload)
    except Exception:
        pass


def _draw_video(imgui, overlay, width):
    if not overlay or overlay.get("kind") != "video" or not overlay.get("id"):
        return
    iw, ih = overlay["wh"]
    dw = max(1.0, width - 16.0)
    dh = dw * ih / max(iw, 1)
    imgui.image(overlay["id"], dw, dh)
    imgui.dummy(0, 8)


def _draw_query_block(imgui, payload, hud_state, keys_out):
    query = payload.get("query")
    if not query:
        return
    mode = query.get("mode") or "off"
    can_search = bool(query.get("can_search") or query.get("can_image") or mode != "off")
    if not can_search:
        return
    imgui.text("Query")
    prompt = query.get("input") or ""
    if mode != "off" and prompt:
        try:
            count = int(query.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        imgui.text(f"'{prompt}'  {count} masks")
    if query.get("status"):
        imgui.text(query["status"])
    results = query.get("results") or []
    for line in results[:5]:
        imgui.text(str(line))
    imgui.separator()


def _draw_replay_block(imgui, payload, keys_out):
    camera = payload.get("camera")
    if not camera or camera.get("mode") != "replay":
        return
    phase = camera.get("phase") or "loading"
    count = int(camera.get("count") or 0)
    frame = int(camera.get("frame") or 0)
    playing = bool(camera.get("playing"))
    imgui.text("Replay")
    if camera.get("caption"):
        imgui.text(str(camera["caption"]))
    if phase == "loading":
        total = max(1, int(camera.get("total") or 0))
        done = min(total, max(0, int(camera.get("done") or 0)))
        imgui.text(f"Caching video {done}/{total}…")
        try:
            imgui.progress_bar(done / total)
        except Exception:
            imgui.text(f"{done}/{total}")
        try:
            elapsed = float(camera.get("elapsed") or 0.0)
        except (TypeError, ValueError):
            elapsed = 0.0
        imgui.text(f"{elapsed:.0f}s elapsed")
        if camera.get("status"):
            imgui.text(str(camera["status"]))
        imgui.separator()
        return
    if phase != "ready":
        if camera.get("status"):
            imgui.text(str(camera["status"]))
        else:
            imgui.text("Replay unavailable")
        imgui.separator()
        return
    labels = ("Back", "Pause" if playing else "Play", "Forward")
    options = ("back", "toggle", "forward")
    sel = int(camera.get("replay_index") or 0) % len(options)
    focused = payload.get("focus") == "replay"
    for i, (option, label) in enumerate(zip(options, labels)):
        if i:
            imgui.same_line()
        text = f"> {label} <" if (focused and i == sel) else label
        if imgui.button(text):
            _send(keys_out, {"type": "replay_transport", "option": option})
    imgui.same_line()
    imgui.text(f"{frame}/{max(count - 1, 0)}")
    if camera.get("status"):
        imgui.text(str(camera["status"]))
    if count > 1:
        changed, value = imgui.slider_int("##replay_frame", frame, 0, count - 1)
        if changed and value is not None:
            _send(keys_out, {"type": "replay_frame", "frame": int(value)})
    imgui.separator()


def _draw_pipeline_block(imgui, payload):
    pipe = payload.get("pipeline")
    if not pipe or not pipe.get("text"):
        return
    imgui.text(f"Pipeline: {pipe.get('method')}")
    imgui.push_style_color(imgui.COLOR_TEXT, 0.45, 0.45, 0.45, 1.0)
    imgui.text_wrapped(str(pipe["text"]))
    imgui.pop_style_color()
    imgui.separator()


def _filter_status_line(payload):
    filt = payload.get("filter") or {}
    mode = filt.get("mode", "off")
    if mode != "on":
        reason = filt.get("reason") or ""
        return f"Filter: Off ({reason})" if reason else "Filter: Off"
    if filt.get("paused_by_query"):
        return "Filter: On (paused by Query)"
    if not filt.get("active"):
        reason = filt.get("reason") or "unavailable"
        return f"Filter: On ({reason})"
    return "Filter: On"


def _draw_right(imgui, payload, height, width, overlay=None, hud_state=None, keys_out=None):
    interactive = not payload.get("cad_overlay")
    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE |
             imgui.WINDOW_NO_SAVED_SETTINGS)
    if not interactive:
        flags |= imgui.WINDOW_NO_INPUTS
    imgui.set_next_window_position(0, 0)
    imgui.set_next_window_size(width, height)
    imgui.begin("##scannet_hud", flags=flags)

    if overlay and overlay.get("id") and payload.get("cad_overlay"):
        iw, ih = overlay["wh"]
        scale = min(width / max(iw, 1), height / max(ih, 1))
        dw, dh = iw * scale, ih * scale
        imgui.set_cursor_pos(((width - dw) * 0.5, (height - dh) * 0.5))
        imgui.image(overlay["id"], dw, dh)
        imgui.end()
        return

    if payload.get("status"):
        imgui.text(payload["status"])
        imgui.separator()

    imgui.text("Information")
    info = payload.get("info") or {}
    imgui.text(info.get("scene") or "")
    reconstruction = format_elapsed(info.get("reconstruction_s"))
    if reconstruction:
        imgui.text(f"Reconstruction {reconstruction}")
    method, run = info.get("method"), info.get("run")
    if method and run:
        imgui.text(f"{method} / {run}")
        prediction = format_elapsed(info.get("prediction_s"))
        if prediction:
            imgui.text(f"Prediction {prediction}")
        scene_ap = info.get("scene_ap")
        scene_classes = info.get("scene_ap_classes")
        if scene_ap is None:
            imgui.text("Scene AP -")
        else:
            try:
                n = int(scene_classes)
            except (TypeError, ValueError):
                n = 0
            imgui.text(f"Scene AP {format_metric(scene_ap)} ({n} evaluated classes)")
        run_ap = info.get("run_ap")
        run_classes = info.get("run_ap_classes")
        run_scenes = info.get("run_scenes")
        if run_ap is None:
            imgui.text("Run AP -")
        else:
            try:
                rn = int(run_scenes) if run_scenes is not None else None
            except (TypeError, ValueError):
                rn = None
            try:
                rc = int(run_classes) if run_classes is not None else None
            except (TypeError, ValueError):
                rc = None
            if rn is not None and rc is not None:
                imgui.text(f"Run AP {format_metric(run_ap)} ({rn} scenes, {rc} evaluated classes)")
            else:
                imgui.text(f"Run AP {format_metric(run_ap)}")
        visible = info.get("visible_pred")
        eligible = info.get("eligible_gt")
        visible_txt = "-" if visible is None else str(visible)
        eligible_txt = "-" if eligible is None else str(eligible)
        imgui.text(f"Visible predictions {visible_txt} | Eligible GT {eligible_txt}")
    imgui.separator()

    imgui.text("Settings")
    settings_focus = payload.get("focus") == "settings"
    for row in payload.get("settings") or []:
        _draw_setting_row(imgui, row, width, settings_focus)
    imgui.separator()

    _draw_query_block(imgui, payload, hud_state or {}, keys_out)
    _draw_replay_block(imgui, payload, keys_out)
    panel = payload.get("panel", "classes")
    if panel == "preview":
        _draw_video(imgui, overlay, width)
    elif panel == "pipeline":
        _draw_pipeline_block(imgui, payload)
    else:
        imgui.text(_filter_status_line(payload))
        if payload.get("color_mode") == "tp_fn":
            imgui.text("TP green | FP dark red | FN light red")
        query = payload.get("query") or {}
        query_mode = query.get("mode") or "off"
        if query_mode in ("search", "image"):
            prompt = query.get("input") or ""
            try:
                count = int(query.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            imgui.text(f"Query '{prompt}'  P {count}  G -")
            if query.get("status"):
                imgui.text(query["status"])
            imgui.end()
            return
        imgui.text("Classes   P prediction   G eligible GT   [TP green / errors red]")
        classes = payload.get("classes") or []
        if classes:
            class_stats = payload.get("class_stats") or {}
            filt = payload.get("filter") or {}
            rows = {row.get("name"): row for row in filt.get("rows") or []}
            selected = filt.get("selected")
            scale = 1
            for name in classes:
                stats = class_stats.get(name)
                if stats is not None:
                    for key in ("fp", "fn"):
                        try:
                            value = int(stats.get(key) or 0)
                        except (TypeError, ValueError):
                            value = 0
                        scale = max(scale, value)
            for name in classes:
                row = rows.get(name) or {}
                track = None
                if filt.get("mode") == "on":
                    track = {
                        "fitted": row.get("fitted"),
                        "effective": row.get("effective")
                        if row.get("is_fitted") or row.get("is_manual") else None,
                        "is_manual": bool(row.get("is_manual")),
                    }
                _draw_class_row(imgui, name, payload["colors"][name],
                                class_stats.get(name), scale, track,
                                selected=(selected is not None and name == selected))
            if filt.get("mode") == "on" and not filt.get("paused_by_query"):
                imgui.text("Up/Down select, Left/Right adjust, Enter reset")
    imgui.end()


def _draw(imgui, payload, height, side, width, overlay=None, view_start=None,
          hud_state=None, keys_out=None):
    if side == "left":
        _draw_left(imgui, payload, height, width, view_start or [0])
    else:
        _draw_right(imgui, payload, height, width, overlay, hud_state, keys_out)


def _display_size():
    try:
        result = subprocess.run(["xdotool", "getdisplaygeometry"],
                                capture_output=True, text=True, timeout=1)
        if result.returncode:
            return 1920, 1080
        w, h = result.stdout.split()
        return int(w), int(h)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 1920, 1080


def _load_overlay_rgba(path, max_w, max_h):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGBA")
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
        im.thumbnail((max(1, max_w), max(1, max_h)), resample)
        arr = np.asarray(im, dtype=np.uint8)
    return np.ascontiguousarray(arr)


def _upload_overlay_texture(gl, rgba):
    h, w = rgba.shape[:2]
    tex = int(gl.glGenTextures(1))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE, rgba)
    return tex, w, h


def run(updates, parent_pid, side="right", viewer_rect=None, keys_out=None):
    import glfw
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
    from OpenGL.GL import GL_COLOR_BUFFER_BIT, glClear, glClearColor
    import OpenGL.GL as gl

    dock_w = width_for(side)
    vx, vy, vw, vh = viewer_rect or (LEFT_WIDTH, 0, 800, 800)
    if side == "left":
        dock_x, dock_y, dock_w, dock_h = vx - dock_w, vy, dock_w, vh
    else:
        dock_x, dock_y, dock_w, dock_h = vx + vw, vy, dock_w, vh

    if not glfw.init():
        return
    glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.FOCUSED, False)
    glfw.window_hint(glfw.FOCUS_ON_SHOW, False)
    glfw.window_hint(glfw.RESIZABLE, False)
    glfw.window_hint(glfw.FLOATING, True)
    window = glfw.create_window(dock_w, dock_h, f"ScanNet {side}", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    glfw.set_window_pos(window, max(0, dock_x), max(0, dock_y))
    glfw.show_window(window)

    imgui.create_context()
    style = imgui.get_style()
    style.window_rounding = 0
    style.window_border_size = 0
    style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (1.0, 1.0, 1.0, 1.0)
    style.colors[imgui.COLOR_TEXT] = (0.08, 0.08, 0.08, 1.0)
    style.colors[imgui.COLOR_BORDER] = (0.30, 0.30, 0.30, 1.0)
    renderer = GlfwRenderer(window, attach_callbacks=(side == "right"))

    def _on_key(win, key, scancode, action, mods):
        if side == "right":
            callback = getattr(renderer, "keyboard_callback", None) or getattr(
                renderer, "_keyboard_callback", None)
            if callback:
                callback(win, key, scancode, action, mods)
            try:
                if imgui.get_io().want_capture_keyboard:
                    return
            except Exception:
                pass
        if keys_out is None or action != glfw.PRESS:
            return
        try:
            keys_out.send(key)
        except Exception:
            pass

    glfw.set_key_callback(window, _on_key)
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    imgui.get_io().display_size = (float(max(fb_w, 1)), float(max(fb_h, 1)))

    empty = {"tree": [], "settings": [], "classes": [], "colors": {},
              "status": "", "focus": "tree", "cad_overlay": None,
              "panel": "classes"}
    payload = empty
    width, height = dock_w, dock_h
    dirty = True
    overlay = {"id": None, "path": None, "wh": None, "kind": None,
               "frame_id": None, "revision": None}
    fullscreen = False
    view_start = [0]
    hud_state = {}

    def _clear_overlay():
        if overlay["id"] is not None:
            try:
                gl.glDeleteTextures(1, [overlay["id"]])
            except Exception:
                pass
        overlay["id"] = None
        overlay["path"] = None
        overlay["wh"] = None
        overlay["kind"] = None
        overlay["frame_id"] = None
        overlay["revision"] = None

    def _set_texture(rgba, kind, path=None, frame_id=None, revision=None):
        h, w = rgba.shape[:2]
        if overlay["id"] is not None and overlay.get("wh") == (w, h):
            gl.glBindTexture(gl.GL_TEXTURE_2D, overlay["id"])
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, gl.GL_RGBA,
                               gl.GL_UNSIGNED_BYTE, rgba)
        else:
            tex, iw, ih = _upload_overlay_texture(gl, rgba)
            _clear_overlay()
            overlay["id"] = tex
            overlay["wh"] = (iw, ih)
        overlay["kind"] = kind
        overlay["path"] = path
        overlay["frame_id"] = frame_id
        overlay["revision"] = revision

    def _sync_overlay(path, video):
        if path:
            if overlay.get("kind") == "cad" and path == overlay["path"]:
                return
            dw, dh = _display_size()
            try:
                rgba = _load_overlay_rgba(path, dw, dh)
                _set_texture(rgba, "cad", path=path)
            except Exception:
                _clear_overlay()
            return
        if video and video.get("rgba") is not None:
            frame_id = video.get("frame_id")
            revision = video.get("revision")
            if (overlay.get("kind") == "video" and overlay.get("frame_id") == frame_id
                    and overlay.get("revision") == revision):
                return
            try:
                import numpy as np
                rgba = np.ascontiguousarray(video["rgba"], dtype=np.uint8)
                _set_texture(rgba, "video", frame_id=frame_id, revision=revision)
            except Exception:
                _clear_overlay()
            return
        if overlay["id"] is not None:
            _clear_overlay()

    def _dock():
        nonlocal width, height, fullscreen
        width, height = dock_w, dock_h
        glfw.set_window_size(window, dock_w, dock_h)
        glfw.set_window_pos(window, max(0, dock_x), max(0, dock_y))
        glfw.show_window(window)
        fullscreen = False

    while not glfw.window_should_close(window):
        try:
            if updates.poll(0.05):
                while True:
                    value = updates.recv()
                    if value is None:
                        raise KeyboardInterrupt
                    payload = value
                    dirty = True
                    if not updates.poll():
                        break
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        try:
            os.kill(parent_pid, 0)
        except OSError:
            break

        cad = payload.get("cad_overlay") if payload else None
        video = payload.get("video") if payload else None
        if cad:
            dw, dh = _display_size()
            if not fullscreen or height != dh:
                glfw.set_window_pos(window, 0, 0)
                glfw.set_window_size(window, dw, dh)
                glfw.show_window(window)
                width, height = dw, dh
                fullscreen = True
                dirty = True
        elif fullscreen:
            _dock()
            dirty = True

        glfw.poll_events()
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        if fb_w < 8 or fb_h < 8:
            continue
        glfw.make_context_current(window)
        if side == "right" and payload:
            _sync_overlay(cad, video)
        imgui.get_io().display_size = (float(fb_w), float(fb_h))
        renderer.process_inputs()
        imgui.new_frame()
        if payload:
            _draw(imgui, payload, height, side, width, overlay, view_start, hud_state, keys_out)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        imgui.render()
        renderer.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        dirty = False

    _clear_overlay()
    renderer.shutdown()
    glfw.destroy_window(window)
    glfw.terminate()
