"""Headless SceneWidget backend tests (Stage B). Fakes only, no DISPLAY."""

import os
import sys
import unittest

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from open3d.visualization import gui  # noqa: E402

from utils.scene_widget import (  # noqa: E402
    KEY_TO_ACTION,
    LABEL_SCALE,
    SceneWidgetBackend,
    make_key_handler,
    map_key_event,
    tick_result,
)


class FakeCamera:
    def __init__(self):
        self.calls = []

    def look_at(self, center, eye, up):
        self.calls.append((np.asarray(center), np.asarray(eye),
                           np.asarray(up)))


class FakeScene:
    def __init__(self):
        self.geoms = {}
        self.visible = {}
        self.materials = {}
        self.transforms = {}
        self.camera = FakeCamera()
        self.background = None

    def add_geometry(self, name, geom, mat):
        self.geoms[name] = geom
        self.materials[name] = mat
        self.visible[name] = True

    def remove_geometry(self, name):
        self.geoms.pop(name, None)
        self.visible.pop(name, None)

    def show_geometry(self, name, flag):
        self.visible[name] = bool(flag)

    def has_geometry(self, name):
        return name in self.geoms

    def modify_geometry_material(self, name, mat):
        self.materials[name] = mat

    def set_geometry_transform(self, name, mat):
        self.transforms[name] = np.asarray(mat)

    def set_background(self, color):
        self.background = np.asarray(color)


class FakeLabel:
    def __init__(self, pos, text):
        self.position = np.asarray(pos, dtype=np.float32)
        self.text = text
        self.color = None
        self.scale = 1.0


class FakeWidget:
    def __init__(self, scene):
        self.scene = scene
        self.labels = []
        self.camera_args = None
        self.redraws = 0

    def add_3d_label(self, pos, text):
        handle = FakeLabel(pos, text)
        self.labels.append(handle)
        return handle

    def remove_3d_label(self, handle):
        if handle in self.labels:
            self.labels.remove(handle)

    def setup_camera(self, fov, bounds, center):
        self.camera_args = (fov, bounds, np.asarray(center))

    def force_redraw(self):
        self.redraws += 1


class FakeWindow:
    def __init__(self, scaling=2.0):
        self.scaling = scaling
        self.on_close = None
        self.closed = False

    def set_on_close(self, fn):
        self.on_close = fn

    def close(self):
        self.closed = True


class FakeKeyEvent:
    def __init__(self, key, etype):
        self.key = key
        self.type = etype


def _backend(scaling=2.0):
    scene = FakeScene()
    return SceneWidgetBackend(window=FakeWindow(scaling=scaling),
                              widget=FakeWidget(scene), scene=scene)


class RegistryTests(unittest.TestCase):
    def test_add_show_remove_has(self):
        back = _backend()
        geom, mat = object(), object()
        back.add("object/1/0", geom, mat)
        self.assertTrue(back.has("object/1/0"))
        back.show("object/1/0", False)
        self.assertFalse(back._scene.visible["object/1/0"])
        back.show("object/1/0", True)
        self.assertTrue(back._scene.visible["object/1/0"])
        back.remove("object/1/0")
        self.assertFalse(back.has("object/1/0"))
        back.remove("object/1/0")  # idempotent

    def test_set_material_and_transform(self):
        back = _backend()
        back.add("box/1/0", object(), "m0")
        back.set_material("box/1/0", "m1")
        self.assertEqual(back._scene.materials["box/1/0"], "m1")
        back.set_transform("box/1/0", np.eye(4))
        np.testing.assert_allclose(back._scene.transforms["box/1/0"],
                                   np.eye(4))

    def test_clear_generation(self):
        back = _backend()
        back.add("object/1/0", object(), None)
        back.add("object/1/1", object(), None)
        back.add("object/2/0", object(), None)
        back.clear_generation("object/1/")
        self.assertFalse(back.has("object/1/0"))
        self.assertFalse(back.has("object/1/1"))
        self.assertTrue(back.has("object/2/0"))


class MaterialCacheTests(unittest.TestCase):
    def test_key_reuse_and_scaling(self):
        back = _backend(scaling=2.0)
        first = back.get_material("defaultUnlit", (1, 0, 0, 1),
                                  point_size=3)
        second = back.get_material("defaultUnlit", (1, 0, 0, 1),
                                   point_size=3)
        self.assertIs(first, second)
        self.assertEqual(first.point_size, 6)  # scaled once by 2.0
        other = back.get_material("defaultUnlit", (0, 1, 0, 1),
                                  point_size=3)
        self.assertIsNot(first, other)
        line = back.line_material((1, 1, 1, 1), line_width=2)
        self.assertEqual(line.shader, "unlitLine")
        mesh = back.mesh_material()
        self.assertEqual(mesh.shader, "defaultLit")


class LabelTests(unittest.TestCase):
    def test_create_mutate_remove_idempotent(self):
        back = _backend()
        handle = back.add_label([1, 2, 3], "chair", (1, 0, 0, 1), 1.4)
        self.assertIn(handle, back.labels)
        self.assertEqual(handle.text, "chair")
        self.assertAlmostEqual(handle.scale, 1.4)
        back.update_label(handle, text="table", scale=1.0)
        self.assertEqual(handle.text, "table")
        self.assertAlmostEqual(handle.scale, 1.0)
        back.update_label(handle, pos=[4, 5, 6])
        np.testing.assert_allclose(handle.position, [4, 5, 6])
        back.remove_label(handle)
        self.assertNotIn(handle, back.labels)
        back.remove_label(handle)  # idempotent
        back.remove_label(None)  # idempotent
        back.update_label(handle, text="noop")  # removed: no-op
        back.update_label(None, text="noop")  # no-op

    def test_label_scale_map(self):
        self.assertIsNone(LABEL_SCALE["off"])
        self.assertEqual(LABEL_SCALE["small"], 1.0)
        self.assertEqual(LABEL_SCALE["large"], 1.4)


class TickTests(unittest.TestCase):
    def test_bool_contract(self):
        self.assertTrue(tick_result(True))
        self.assertFalse(tick_result(False))
        self.assertFalse(tick_result(None))
        self.assertTrue(tick_result(1))
        self.assertIsInstance(tick_result(True), bool)
        self.assertIsInstance(tick_result(False), bool)


class KeyMapTests(unittest.TestCase):
    def test_table_uses_keyname_not_ints(self):
        for key in KEY_TO_ACTION:
            self.assertIsInstance(key, gui.KeyName)
        for value in KEY_TO_ACTION.values():
            self.assertIsInstance(value, str)
        expected = {gui.KeyName.UP, gui.KeyName.DOWN, gui.KeyName.LEFT,
                    gui.KeyName.RIGHT, gui.KeyName.ENTER, gui.KeyName.TAB,
                    gui.KeyName.Q, gui.KeyName.ESCAPE}
        self.assertEqual(set(KEY_TO_ACTION), expected)

    def test_map_and_handler(self):
        down = FakeKeyEvent(gui.KeyName.UP, gui.KeyEvent.Type.DOWN)
        self.assertEqual(map_key_event(down), "up")
        keyup = FakeKeyEvent(gui.KeyName.UP, gui.KeyEvent.Type.UP)
        self.assertIsNone(map_key_event(keyup))
        unknown = FakeKeyEvent(gui.KeyName.F1, gui.KeyEvent.Type.DOWN)
        self.assertIsNone(map_key_event(unknown))
        seen = []
        handler = make_key_handler(seen.append)
        self.assertTrue(handler(down))
        self.assertEqual(seen, ["up"])
        self.assertFalse(handler(keyup))
        self.assertFalse(handler(unknown))

    def test_close_is_idempotent_and_quits(self):
        back = _backend()
        calls = []
        back.attach_close(cleanup=lambda: calls.append(1))
        self.assertTrue(back._window.on_close())
        self.assertTrue(back._window.on_close())
        self.assertEqual(calls, [1])
        back.close()
        self.assertTrue(back._window.closed)


if __name__ == "__main__":
    unittest.main()
