"""Recon recovery: presets/profiles, keep rules, runtime cap, fast .sens writer, env passthrough.
CPU only; no SDK, Metashape licence or GPU is touched."""
import os
import struct
import sys
import tempfile
import time
import unittest
import zlib
from unittest import mock

import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

import jobs  # noqa: E402
from reconstruct import preset_env, runtime_cap_s  # noqa: E402
from reconstruct import finalize  # noqa: E402
from reconstruct.engines import metashape, open3d as o3d_engine, zed  # noqa: E402
from reconstruct import run  # noqa: E402

_ENV_KEYS = ("SPELLBOOK_ZED_PRESET", "SPELLBOOK_ZED_VARIANT", "ZED_VARIANT",
             "SPELLBOOK_METASHAPE_PRESET", "SPELLBOOK_METASHAPE_PROFILE",
             "SPELLBOOK_METASHAPE_SCALE_BARS", "SPELLBOOK_METASHAPE_MAX_KEYFRAMES",
             "SPELLBOOK_OPEN3D_PRESET", "OPEN3D_BOUNDED_BACKEND")


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


class ZedPresetTests(unittest.TestCase):
    def test_native_default_is_official_baseline(self):
        c = zed.PRESETS["native-default"]
        self.assertEqual((c["passes"], c["tracking"], c["depth_mode"]), (2, "GEN_1", "NEURAL_PLUS"))
        self.assertEqual((c["resolution_meter"], c["range_meter"]), (0.05, 0.0))
        self.assertEqual((c["confidence_threshold"], c["stability_counter"]), (95, 0))
        self.assertEqual((c["max_memory_usage"], c["mesh_filter"]), (2048, "LOW"))

    def test_derivation_chain(self):
        b5 = zed.PRESETS["b5-gen1-neural"]
        self.assertEqual((b5["tracking"], b5["depth_mode"], b5["range_meter"],
                          b5["confidence_threshold"], b5["stability_counter"]),
                         ("GEN_1", "NEURAL", 3.5, 50, 10))
        diff = {k for k in b5 if b5[k] != zed.PRESETS["s2b8-fixedcalib"][k]}
        self.assertEqual(diff, {"camera_disable_self_calib"})
        one = zed.PRESETS["s2b7-native-onepass"]
        self.assertEqual({k for k in one if one[k] != zed.PRESETS["native-default"][k]},
                         {"passes"})

    def test_selection_precedence(self):
        with _clean_env():
            self.assertEqual(zed.selected_variant(), "b5-gen1-neural")
        with _clean_env(ZED_VARIANT="native-default"):
            self.assertEqual(zed.selected_variant(), "native-default")
        with _clean_env(ZED_VARIANT="native-default", SPELLBOOK_ZED_PRESET="s2b8-fixedcalib"):
            self.assertEqual(zed.selected_variant(), "s2b8-fixedcalib")
            self.assertEqual(zed.provenance()["variant"], "s2b8-fixedcalib")
        with _clean_env(SPELLBOOK_ZED_PRESET="nope"):
            with self.assertRaises(ValueError):
                zed.selected_variant()


class MetashapeProfileTests(unittest.TestCase):
    def test_profiles_chain(self):
        nd, b5 = metashape.PROFILES["native-default"], metashape.PROFILES["b5-kf350"]
        self.assertEqual((nd["depth_downscale"], nd["max_keyframes"], nd["scale_bars"]),
                         (2, 400, 0))
        self.assertFalse(nd["rematch"] or nd["sparse_cleanup"])
        self.assertEqual((b5["depth_downscale"], b5["interpolation"], b5["max_keyframes"],
                          b5["scale_bars"]), (1, "DisabledInterpolation", 350, 3))
        self.assertTrue(b5["rematch"] and b5["sparse_cleanup"])
        self.assertEqual(metashape.DEFAULT_PROFILE, "b5-kf350")
        self.assertTrue(metashape.RUNTIME_CAP_EXEMPT)
        self.assertFalse(hasattr(metashape, "TIMEOUT_S"))

    def test_overrides_and_run0_pin(self):
        with _clean_env(SPELLBOOK_METASHAPE_SCALE_BARS="0",
                        SPELLBOOK_METASHAPE_MAX_KEYFRAMES="400"):
            name, cfg, ov = metashape.selected_profile()
            self.assertEqual((name, cfg["scale_bars"], cfg["max_keyframes"]), ("b5-kf350", 0, 400))
            self.assertEqual(ov, {"scale_bars": 0, "max_keyframes": 400})
        with _clean_env(SPELLBOOK_METASHAPE_PRESET="native-default",
                        SPELLBOOK_METASHAPE_SCALE_BARS="3"):
            name, cfg, ov = metashape.selected_profile()
            self.assertEqual((name, cfg["scale_bars"], ov), ("native-default", 0, {}))
        with _clean_env(SPELLBOOK_METASHAPE_PROFILE="b2-h3"):
            self.assertEqual(metashape.selected_profile()[0], "b2-h3")

    def test_keyframe_cap(self):
        poses = np.stack([np.eye(4)] * 2000)
        poses[:, 0, 3] = np.arange(2000) * 0.1
        sel = metashape.select_keyframes(poses, 350)
        self.assertLessEqual(len(sel), 351)
        self.assertEqual((sel[0], sel[-1]), (0, 1999))

    def test_scale_bars(self):
        pos = {i: np.array([float(i), 0.0, 0.0]) for i in range(0, 101, 5)}
        bars = metashape.select_scale_bars(pos, 3)
        self.assertEqual(len(bars), 3)
        ends = [e for a, b, _ in bars for e in (a, b)]
        self.assertEqual(len(set(ends)), 6)
        self.assertTrue(all(d >= 0.7 * 100 for _, _, d in bars))
        self.assertEqual(metashape.select_scale_bars(pos, 0), [])

    def test_metric_frame_check(self):
        priors = np.random.default_rng(0).normal(size=(50, 3))
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(metashape.check_metric_frame(priors @ R + 5.0, priors), 1.0)
        with self.assertRaises(RuntimeError):  # poses wrongly divided by the chunk scale
            metashape.check_metric_frame(priors / 0.16, priors)

    def test_rendered_script_scales_depth_not_mesh(self):
        with tempfile.TemporaryDirectory() as work:
            name, cfg, ov = metashape.selected_profile()
            script, _ = metashape._render(work, 3, name, cfg, ov, [])
        self.assertIn("d * scale * 1000.0", script)
        self.assertNotIn("/ scale", script)

    def test_rendered_script_compiles_and_verifies(self):
        with tempfile.TemporaryDirectory() as work:
            name, cfg = "b5-kf350", dict(metashape.B5_KF350)
            bars = [[20, 1, 9.5]]
            script, doc = metashape._render(work, 1, name, cfg, {}, bars)
            compile(script, "metashape_run.py", "exec")
            with open(os.path.join(work, "metashape_run.py"), "w") as f:
                f.write(script)
            import json
            with open(os.path.join(work, "metashape_settings.json"), "w") as f:
                json.dump(doc, f)
            metashape._verify_rendered(work, 1, name, cfg, {}, bars)
            # poison: slot rendered for another profile must fail loudly
            with self.assertRaises(RuntimeError):
                metashape._verify_rendered(work, 1, "native-default",
                                           dict(metashape.NATIVE_DEFAULT), {}, [])
            with self.assertRaises(RuntimeError):
                metashape._verify_rendered(work, 1, name, cfg, {}, bars,
                                           {"settings_sha256": np.array(["0" * 64])})


class Open3dB1Tests(unittest.TestCase):
    def test_keep_rule_9004(self):
        idx = o3d_engine.keep_indices(3612, o3d_engine._BOUNDED_MAX_KEEP_BY_SCENE[9004])
        self.assertEqual(len(idx), 80)
        self.assertEqual(list(idx[:3]), [0, 46, 91])
        self.assertEqual(list(idx[-2:]), [3565, 3611])

    def test_keep_rule_9009_matches_published(self):
        published = [0, 40, 80, 120, 160, 200, 239, 279, 319, 359, 399, 439, 479, 519, 559,
                     599, 639, 679, 718, 758, 798, 838, 878, 918, 958, 998, 1038, 1078, 1118,
                     1158, 1197, 1237, 1277, 1317, 1357, 1397, 1437, 1477, 1517, 1557, 1597,
                     1637, 1676, 1716, 1756, 1796, 1836, 1876, 1916, 1956, 1996, 2036, 2076,
                     2116, 2155, 2195, 2235, 2275, 2315, 2355]
        idx = o3d_engine.keep_indices(2356, o3d_engine._BOUNDED_MAX_KEEP_BY_SCENE[9009])
        self.assertEqual(list(idx), published)

    def test_backend_selection(self):
        with _clean_env():
            self.assertEqual(o3d_engine.selected_backend(), "tensor")
            self.assertEqual(o3d_engine.provenance()["variant"], "b1")
        with _clean_env(OPEN3D_BOUNDED_BACKEND="legacy"):
            self.assertEqual(o3d_engine.selected_backend(), "legacy")
        with _clean_env(OPEN3D_BOUNDED_BACKEND="legacy", SPELLBOOK_OPEN3D_PRESET="tensor"):
            self.assertEqual(o3d_engine.selected_backend(), "tensor")

    def test_b1_fusion_constants(self):
        b = o3d_engine.B1
        self.assertEqual((b["width"], b["height"], b["voxel_size"], b["depth_min"],
                          b["depth_max"], b["block_resolution"], b["block_count"],
                          b["trunc_voxel_multiplier"], b["weight_threshold"]),
                         (320, 240, 0.010, 0.1, 4.0, 16, 50000, 6.0, 3.0))
        self.assertAlmostEqual(b["voxel_size"] * b["trunc_voxel_multiplier"], b["sdf_trunc"])

    def test_run0_guard(self):
        with self.assertRaises(RuntimeError):
            o3d_engine.reconstruct("/nonexistent/recon", "/x/scene9004_41", gpu=1, lease_fd=3)


class CapTests(unittest.TestCase):
    def test_cap_values(self):
        self.assertAlmostEqual(runtime_cap_s(9004), 361.200330, places=6)
        self.assertAlmostEqual(runtime_cap_s(9009), 471.131361, places=6)
        self.assertIsNone(runtime_cap_s(1234))

    def test_cap_for_exempt(self):
        self.assertIsNone(run.cap_for(metashape, 9004))
        self.assertAlmostEqual(run.cap_for(zed, 9004), 361.200330, places=6)
        self.assertAlmostEqual(run.cap_for(o3d_engine, 9009), 471.131361, places=6)

    def test_runtime_cap_fires(self):
        with self.assertRaises(run.RuntimeCapExceeded):
            with run.runtime_cap(0.05):
                time.sleep(1.0)
        with run.runtime_cap(5.0):
            pass  # disarmed on exit
        with run.runtime_cap(None):
            pass

    def test_preset_env(self):
        with _clean_env(SPELLBOOK_ZED_PRESET=" ", ZED_VARIANT="x"):
            self.assertEqual(preset_env("zed", "ZED_VARIANT"), "x")


class SensWriterTests(unittest.TestCase):
    def _frames(self, root, n):
        import cv2
        for d in ("color", "depth"):
            os.makedirs(os.path.join(root, d))
        for i in range(n):
            cv2.imwrite(os.path.join(root, "color", f"{i}.jpg"),
                        np.full((6, 8, 3), i, np.uint8))
            cv2.imwrite(os.path.join(root, "depth", f"{i}.png"),
                        np.full((6, 8), 100 + i, np.uint16))

    def test_ordered_threaded_sens(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = os.path.join(tmp, "frames")
            scan = os.path.join(tmp, "scan")
            os.makedirs(scan)
            self._frames(frames, 40)
            keep = list(range(39, -1, -3))
            poses = np.stack([np.eye(4) * (1 + i) for i in range(len(keep))])
            with mock.patch.object(finalize, "FINALIZE_WINDOW", 2):
                path, w, h = finalize.finalize(scan, "scene9999_00", frames, np.eye(3),
                                               poses, keep)
            self.assertEqual((w, h), (8, 6))
            self.assertEqual(sorted(os.listdir(scan)), ["recon", "scene9999_00.sens"])
            np.testing.assert_array_equal(np.load(os.path.join(scan, "recon", "keep.npy")),
                                          keep)
            with open(path, "rb") as f:
                buf = f.read()
            off = 4
            (nl,) = struct.unpack_from("<Q", buf, off)
            off += 8 + nl + 4 * 64 + 8 + 16 + 4
            (n,) = struct.unpack_from("<Q", buf, off)
            off += 8
            self.assertEqual(n, len(keep))
            for j, i in enumerate(keep):
                pose = np.array(struct.unpack_from("<16f", buf, off)).reshape(4, 4)
                off += 64 + 16
                cl, dl = struct.unpack_from("<QQ", buf, off)
                off += 16 + cl
                depth = np.frombuffer(zlib.decompress(buf[off:off + dl]), "<u2")
                off += dl
                self.assertTrue(np.allclose(pose, poses[j]))
                self.assertTrue((depth == 100 + i).all())
            self.assertEqual(struct.unpack_from("<Q", buf, off)[0], 0)

    def test_failed_write_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = os.path.join(tmp, "frames")
            self._frames(frames, 3)
            with self.assertRaises(Exception):
                finalize.finalize(tmp, "scene9999_00", frames, np.eye(3),
                                  np.stack([np.eye(4)] * 2), [0, 7])
            self.assertFalse([p for p in os.listdir(tmp) if p.endswith(".sens")
                              or ".tmp." in p])


class JobsEnvTests(unittest.TestCase):
    def test_legacy_selectors_pass_detached_jobs(self):
        for name in ("ZED_VARIANT", "OPEN3D_BOUNDED_BACKEND", "ZED_DIR"):
            self.assertIn(name, jobs.ENV_ALLOWLIST)
        self.assertTrue("SPELLBOOK_METASHAPE_PRESET".startswith(jobs.ENV_PREFIX_ALLOWLIST))


if __name__ == "__main__":
    unittest.main()
