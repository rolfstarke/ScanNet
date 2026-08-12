import os
import queue
import subprocess
import time


WIDTH = 680


def _window_geometry(parent_pid, scene_id):
    result = subprocess.run(
        ["xdotool", "search", "--pid", str(parent_pid), "--name", f"^ScanNet - {scene_id}"],
        capture_output=True, text=True)
    if result.returncode:
        return None
    window_id = result.stdout.splitlines()[-1]
    result = subprocess.run(["xdotool", "getwindowgeometry", "--shell", window_id],
                            capture_output=True, text=True)
    if result.returncode:
        return None
    values = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = int(value)
    return values["X"], values["Y"], values["WIDTH"], values["HEIGHT"]


def _draw_class_row(imgui, name, color):
    draw = imgui.get_window_draw_list()
    x, y = imgui.get_cursor_screen_pos()
    text = imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 1.0)
    swatch = imgui.get_color_u32_rgba(*color, 1.0)

    draw.add_rect_filled(x, y + 2, x + 11, y + 13, swatch, 2)
    draw.add_text(x + 17, y, text, name)
    imgui.dummy(315, 17)


def _draw_setting_row(imgui, row):
    draw = imgui.get_window_draw_list()
    x, y = imgui.get_cursor_screen_pos()
    if row["selected"]:
        draw.add_rect_filled(x - 6, y - 3, x + WIDTH - 6, y + 17,
                             imgui.get_color_u32_rgba(0.82, 0.88, 1.0, 1.0))
    marker = "> " if row["selected"] else "  "
    imgui.text(f"{marker}{row['name']}: {row['value']}")
    imgui.dummy(0, 4)


def _draw(imgui, payload, height):
    imgui.set_next_window_position(0, 0)
    imgui.set_next_window_size(WIDTH, height)
    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE |
             imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_INPUTS)
    imgui.begin("##scannet_hud", flags=flags)

    # Block 1: Information
    imgui.text("ScanNet")
    imgui.separator()
    imgui.text(f"Scene: {payload['scene']}")
    imgui.text(f"Source: {payload['source']}")
    imgui.text(f"Instances: {payload['visible_count']}")
    if payload["tp"] is not None:
        imgui.text(f"TP: {payload['tp']}   FP: {payload['fp']}")
    if payload["status"]:
        imgui.text(payload["status"])
    imgui.text(payload["help"])
    imgui.separator()

    # Block 2: Settings
    imgui.text("Settings")
    for row in payload["settings"]:
        _draw_setting_row(imgui, row)
    imgui.separator()

    # Block 3: Classes
    imgui.text("Classes")
    imgui.columns(2, "class_columns", border=False)
    for name in payload["classes"]:
        _draw_class_row(imgui, name, payload["colors"][name])
        imgui.next_column()
    imgui.columns(1)
    imgui.end()


def run(updates, parent_pid, scene_id):
    import glfw
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
    from OpenGL.GL import GL_COLOR_BUFFER_BIT, glClear, glClearColor

    if not glfw.init():
        return
    glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.FOCUSED, False)
    glfw.window_hint(glfw.FOCUS_ON_SHOW, False)
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(WIDTH, 300, "ScanNet legend", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    imgui.create_context()
    style = imgui.get_style()
    style.window_rounding = 0
    style.window_border_size = 0
    style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (1.0, 1.0, 1.0, 1.0)
    style.colors[imgui.COLOR_TEXT] = (0.08, 0.08, 0.08, 1.0)
    style.colors[imgui.COLOR_BORDER] = (0.30, 0.30, 0.30, 1.0)
    renderer = GlfwRenderer(window, attach_callbacks=False)
    glfw.hide_window(window)

    payload = None
    height = 300
    next_position_update = 0.0
    while not glfw.window_should_close(window):
        try:
            while True:
                value = updates.get_nowait()
                if value is None:
                    raise KeyboardInterrupt
                payload = value
        except queue.Empty:
            pass
        except KeyboardInterrupt:
            break

        try:
            os.kill(parent_pid, 0)
        except OSError:
            break

        now = time.monotonic()
        if now >= next_position_update:
            geometry = _window_geometry(parent_pid, scene_id)
            if geometry:
                x, y, width, viewer_height = geometry
                if viewer_height != height:
                    height = viewer_height
                    glfw.set_window_size(window, WIDTH, height)
                glfw.set_window_pos(window, x + width, y)
                glfw.show_window(window)
            next_position_update = now + 0.25

        if payload:
            glfw.poll_events()
            renderer.process_inputs()
            imgui.new_frame()
            _draw(imgui, payload, height)
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            imgui.render()
            renderer.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
        else:
            time.sleep(0.05)

    renderer.shutdown()
    glfw.destroy_window(window)
    glfw.terminate()
