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


def _draw_class_row(imgui, name, color):
    draw = imgui.get_window_draw_list()
    x, y = imgui.get_cursor_screen_pos()
    text = imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0)
    swatch = imgui.get_color_u32_rgba(*color, 1.0)
    draw.add_rect_filled(x, y + 2, x + 11, y + 13, swatch, 2)
    draw.add_text(x + 17, y, text, name)
    imgui.dummy(315, 17)


def _draw_setting_row(imgui, row, width, settings_focus):
    draw = imgui.get_window_draw_list()
    x, y = imgui.get_cursor_screen_pos()
    focused = bool(row.get("focused") and settings_focus)
    imgui.text(f"{row['name']}:")
    first = True
    max_x = x + width - 12
    for opt in row.get("options") or []:
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
        if opt.get("sel") and focused:
            draw.add_rect(cx - 3, cy - 2, cx + tw + 3, cy + 16,
                          imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0))
        active = opt.get("kind") in ("applied", "pending")
        if active:
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


def _sync_query_text(hud_state, payload):
    query = payload.get("query") or {}
    reset_id = query.get("reset_id")
    if hud_state.get("reset_id") != reset_id:
        hud_state["reset_id"] = reset_id
        hud_state["query_text"] = query.get("input") or ""


def _draw_video(imgui, overlay, width):
    if not overlay or overlay.get("kind") != "video" or not overlay.get("id"):
        return
    iw, ih = overlay["wh"]
    scale = min(1.0, 420 / max(iw, 1), 315 / max(ih, 1))
    dw, dh = iw * scale, ih * scale
    imgui.set_cursor_pos((max(8.0, width - dw - 8), 8.0))
    imgui.image(overlay["id"], dw, dh)
    imgui.set_cursor_pos((8.0, dh + 16.0))


def _draw_query_block(imgui, payload, hud_state, keys_out):
    query = payload.get("query")
    if not query:
        return
    imgui.text("Query")
    mode = query.get("mode") or "off"
    imgui.text(f"Mode {mode}")
    if query.get("status"):
        imgui.text(query["status"])
    if mode != "off":
        _sync_query_text(hud_state, payload)
        label = "Image path" if mode == "image" else "Prompt"
        imgui.text(label)
        current = hud_state.get("query_text") or ""
        changed, value = imgui.input_text("##query_field", current, 512)
        if changed and value is not None:
            hud_state["query_text"] = value
        if imgui.button("Apply"):
            _send(keys_out, {"type": "query_apply", "text": hud_state.get("query_text") or ""})
        imgui.same_line()
        if imgui.button("Clear"):
            hud_state["query_text"] = ""
            _send(keys_out, {"type": "query_clear"})
        results = query.get("results") or []
        for line in results[:5]:
            imgui.text(str(line))
    imgui.separator()


def _draw_replay_block(imgui, payload, keys_out):
    camera = payload.get("camera")
    if not camera or camera.get("mode") != "replay":
        return
    count = int(camera.get("count") or 0)
    frame = int(camera.get("frame") or 0)
    playing = bool(camera.get("playing"))
    imgui.text("Replay")
    if imgui.button("Pause" if playing else "Play"):
        _send(keys_out, {"type": "replay_toggle"})
    imgui.same_line()
    imgui.text(f"{frame}/{max(count - 1, 0)}")
    live = camera.get("live_tp")
    live_gt = camera.get("live_gt")
    if live is not None and live_gt is not None:
        imgui.text(f"Replay TP/GT {live}/{live_gt}")
    if count > 1:
        changed, value = imgui.slider_int("##replay_frame", frame, 0, count - 1)
        if changed and value is not None:
            _send(keys_out, {"type": "replay_frame", "frame": int(value)})
    imgui.separator()


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

    _draw_video(imgui, overlay, width)

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
        imgui.text(
            f"AP {format_metric(info.get('ap'))} · "
            f"AP50 {format_metric(info.get('ap50'))} · "
            f"AP25 {format_metric(info.get('ap25'))}")
        tp, gt, fp = info.get("tp"), info.get("gt"), info.get("fp")
        if tp is None or gt is None:
            imgui.text("TP/GT -")
        else:
            fp_text = "-" if fp is None else str(fp)
            imgui.text(f"TP/GT {tp}/{gt} · FP {fp_text}")
    imgui.separator()

    _draw_replay_block(imgui, payload, keys_out)

    imgui.text("Settings")
    settings_focus = payload.get("focus") == "settings"
    for row in payload.get("settings") or []:
        _draw_setting_row(imgui, row, width, settings_focus)
    imgui.separator()

    _draw_query_block(imgui, payload, hud_state or {}, keys_out)

    imgui.text("Classes")
    classes = payload.get("classes") or []
    if classes:
        imgui.columns(2, "class_columns", border=False)
        for name in classes:
            _draw_class_row(imgui, name, payload["colors"][name])
            imgui.next_column()
        imgui.columns(1)
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
             "status": "", "focus": "tree", "cad_overlay": None}
    payload = empty
    width, height = dock_w, dock_h
    dirty = True
    overlay = {"id": None, "path": None, "wh": None, "kind": None, "frame_id": None}
    fullscreen = False
    view_start = [0]
    hud_state = {"query_text": "", "reset_id": None}

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

    def _set_texture(rgba, kind, path=None, frame_id=None):
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
            if overlay.get("kind") == "video" and overlay.get("frame_id") == frame_id:
                return
            try:
                import numpy as np
                rgba = np.ascontiguousarray(video["rgba"], dtype=np.uint8)
                _set_texture(rgba, "video", frame_id=frame_id)
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
        if payload and dirty:
            glfw.make_context_current(window)
            if side == "right":
                _sync_overlay(cad, video)
            imgui.get_io().display_size = (float(fb_w), float(fb_h))
            renderer.process_inputs()
            imgui.new_frame()
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
