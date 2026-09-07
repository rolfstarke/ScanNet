"""SceneWidget backend for the ScanNet visualizer (Stage B).

Official Open3D 0.19 gui/rendering APIs only. No private bindings, no
monkeypatch. Importing this module must not initialize a GUI and must not
require DISPLAY; all window/scene creation happens in
:meth:`SceneWidgetBackend.create_window`.
"""

import numpy as np

from open3d.visualization import gui, rendering

# Screen-space label scales (plan D2/H). ``off`` means no handle.
LABEL_SCALE = {"off": None, "small": 1.0, "large": 1.4}

# Explicit key routing: every gui.KeyName maps to a symbolic dispatch action.
# No GLFW integer codes are used anywhere in this module.
KEY_TO_ACTION = {
    gui.KeyName.UP: "up",
    gui.KeyName.DOWN: "down",
    gui.KeyName.LEFT: "left",
    gui.KeyName.RIGHT: "right",
    gui.KeyName.ENTER: "enter",
    gui.KeyName.TAB: "tab",
    gui.KeyName.Q: "q",
    gui.KeyName.ESCAPE: "escape",
}


def map_key_event(event):
    """Map a ``gui.KeyEvent`` to a dispatch action string or ``None``.

    Key-up events and unhandled keys return ``None``. Repeat key-down
    events are treated like first presses (repeat behavior preserved).
    """
    if event is None:
        return None
    etype = getattr(event, "type", None)
    try:
        up = gui.KeyEvent.Type.UP
    except Exception:  # pragma: no cover - defensive, enum always exists
        up = None
    if up is not None and etype == up:
        return None
    return KEY_TO_ACTION.get(getattr(event, "key", None))


def make_key_handler(dispatch):
    """Build a ``Window.set_on_key`` callback from a ``dispatch(action)`` fn.

    Returns ``True`` for handled keys so camera controls do not consume
    navigation keys, ``False`` otherwise.
    """

    def _on_key(event):
        action = map_key_event(event)
        if action is None:
            return False
        dispatch(action)
        return True

    return _on_key


def tick_result(dirty):
    """Tick redraw contract: truthy state change -> True, else False."""
    return bool(dirty)


class SceneWidgetBackend:
    """Thin name-keyed wrapper around one gui.SceneWidget + Open3DScene."""

    def __init__(self, window=None, widget=None, scene=None, scaling=None):
        self._window = window
        self._widget = widget
        if scene is not None:
            self._scene = scene
        elif widget is not None and getattr(widget, "scene", None) is not None:
            self._scene = widget.scene
        else:
            self._scene = None
        if scaling is not None:
            self._scaling = float(scaling)
        elif window is not None and getattr(window, "scaling", None):
            try:
                self._scaling = float(window.scaling)
            except Exception:
                self._scaling = 1.0
        else:
            self._scaling = 1.0
        self._names = set()
        self._labels = set()
        self._materials = {}
        self._closed = False
        self._cleanup = None

    # -- construction -------------------------------------------------
    @classmethod
    def create_window(cls, title, width, height, x, y, background=(0, 0, 0, 1)):
        """Create a real window + SceneWidget + Open3DScene (needs DISPLAY)."""
        app = gui.Application.instance
        try:
            app.initialize()
        except Exception:
            pass
        # Default flags (0): official create_window(title,w,h,x,y) path.
        window = app.create_window(title, width, height, x, y)
        widget = gui.SceneWidget()
        widget.scene = rendering.Open3DScene(window.renderer)
        window.add_child(widget)

        def _on_layout(_ctx):
            rect = window.content_rect
            widget.frame = gui.Rect(rect.x, rect.y, rect.width, rect.height)

        window.set_on_layout(_on_layout)
        widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        widget.enable_scene_caching(True)
        backend = cls(window=window, widget=widget, scene=widget.scene)
        backend.set_background(background)
        backend.attach_close()
        return backend

    # -- background / materials ---------------------------------------
    def set_background(self, rgba=(0, 0, 0, 1)):
        color = np.asarray(rgba, dtype=np.float32).reshape(4)
        self._scene.set_background(color)

    def _scaled_px(self, px):
        return max(1, int(round(float(px) * self._scaling)))

    def get_material(self, shader, rgba, point_size=1, line_width=1):
        rgba_key = tuple(float(v) for v in rgba)
        key = (shader, rgba_key, self._scaled_px(point_size),
               self._scaled_px(line_width))
        cached = self._materials.get(key)
        if cached is not None:
            return cached
        mat = rendering.MaterialRecord()
        mat.shader = shader
        mat.base_color = list(rgba_key)
        mat.point_size = key[2]
        mat.line_width = key[3]
        self._materials[key] = mat
        return mat

    def mesh_material(self, rgba=(1, 1, 1, 1)):
        return self.get_material("defaultLit", rgba)

    def points_material(self, rgba, point_size=3):
        return self.get_material("defaultUnlit", rgba,
                                 point_size=point_size)

    def line_material(self, rgba, line_width=2):
        return self.get_material("unlitLine", rgba,
                                 line_width=line_width)

    # -- geometry registry (Open3DScene name-keyed only) --------------
    def add(self, name, geom, mat):
        self._scene.add_geometry(name, geom, mat)
        self._names.add(name)

    def remove(self, name):
        if name not in self._names:
            return
        try:
            self._scene.remove_geometry(name)
        finally:
            self._names.discard(name)

    def show(self, name, visible):
        self._scene.show_geometry(name, bool(visible))

    def has(self, name):
        scene = self._scene
        if scene is not None and hasattr(scene, "has_geometry"):
            try:
                return bool(scene.has_geometry(name))
            except Exception:
                pass
        return name in self._names

    def set_material(self, name, mat):
        self._scene.modify_geometry_material(name, mat)

    def set_transform(self, name, matrix_4x4):
        self._scene.set_geometry_transform(
            name, np.asarray(matrix_4x4, dtype=np.float64))

    def clear_generation(self, prefix):
        for name in sorted(n for n in list(self._names)
                           if n.startswith(prefix)):
            self.remove(name)

    def clear_all(self):
        for name in sorted(self._names):
            self.remove(name)

    @property
    def names(self):
        return set(self._names)

    # -- labels (Label3D handles, not geometry) ------------------------
    @staticmethod
    def _to_gui_color(color):
        r, g, b = (float(color[0]), float(color[1]), float(color[2]))
        a = float(color[3]) if len(color) > 3 else 1.0
        return gui.Color(r, g, b, a)

    def add_label(self, pos, text, color=(1, 1, 1, 1), scale=1.0):
        handle = self._widget.add_3d_label(
            np.asarray(pos, dtype=np.float32), str(text))
        handle.color = self._to_gui_color(color)
        handle.scale = float(scale)
        self._labels.add(handle)
        return handle

    def update_label(self, handle, pos=None, text=None, color=None,
                     scale=None):
        if handle is None or handle not in self._labels:
            return
        if pos is not None:
            handle.position = np.asarray(pos, dtype=np.float32)
        if text is not None:
            handle.text = str(text)
        if color is not None:
            handle.color = self._to_gui_color(color)
        if scale is not None:
            handle.scale = float(scale)

    def remove_label(self, handle):
        if handle is None or handle not in self._labels:
            return
        try:
            self._widget.remove_3d_label(handle)
        finally:
            self._labels.discard(handle)

    def clear_labels(self):
        for handle in list(self._labels):
            self.remove_label(handle)

    @property
    def labels(self):
        return set(self._labels)

    # -- camera --------------------------------------------------------
    def setup_camera(self, fov, bounds, center):
        self._widget.setup_camera(float(fov), bounds,
                                  np.asarray(center, dtype=np.float32))

    def look_at(self, center, eye, up):
        self._scene.camera.look_at(
            np.asarray(center, dtype=float),
            np.asarray(eye, dtype=float),
            np.asarray(up, dtype=float))

    def request_redraw(self):
        self._widget.force_redraw()

    # -- close ----------------------------------------------------------
    def attach_close(self, cleanup=None):
        if cleanup is not None:
            self._cleanup = cleanup

        def _on_close():
            self._do_close()
            return True

        self._window.set_on_close(_on_close)
        return _on_close

    def _do_close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._cleanup is not None:
                self._cleanup()
        finally:
            self.clear_labels()
            try:
                gui.Application.instance.quit()
            except Exception:
                pass

    def close(self):
        self._do_close()
        try:
            if self._window is not None:
                self._window.close()
        except Exception:
            pass
