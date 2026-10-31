"""BundleFusion adapter: presets, overlays, crop-correct input, solver cap, docker command.
CPU only; docker and GPUs are mocked."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from reconstruct import runtime_cap_s  # noqa: E402
from reconstruct import run  # noqa: E402
from reconstruct.engines import bundlefusion as bf  # noqa: E402

_ENV = {k: v for k, v in os.environ.items() if not k.startswith("SPELLBOOK_BUNDLEFUSION")}


def _clean_env(**extra):
    return mock.patch.dict(os.environ, {**_ENV, **extra}, clear=True)


def _K(fx, cx, cy):
    return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])


def _make_pool(root, width, height, n):
    """A frame pool in the /data/scannetdata layout: <root>/frames + frames_manifest.json."""
    pool = os.path.join(root, "frames")
    os.makedirs(os.path.join(pool, "color"))
    os.makedirs(os.path.join(pool, "depth"))
    K = np.eye(4)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = 735.307861, 735.307861, 941.765747, 599.485962
    np.savetxt(os.path.join(pool, "intrinsic_depth.txt"), K)
    with open(os.path.join(root, "frames_manifest.json"), "w") as f:
        json.dump({"color_n": n}, f)
    for i in range(n):
        color = np.full((height, width, 3), 40 + i, np.uint8)
        color[:, width // 2:] = 200  # right half bright, to see the crop
        depth = np.full((height, width), 1000 + i, np.uint16)
        cv2.imwrite(os.path.join(pool, "color", f"{i}.jpg"), color)
        cv2.imwrite(os.path.join(pool, "depth", f"{i}.png"), depth)
    return pool


class BaselineTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(bf._OPT_ROUNDS, 0)
        self.assertEqual(bf._TIMEOUT, 10800)
        self.assertEqual(bf.GPU_COUNT, 2)
        self.assertEqual(bf.GPU_PAIR, (1, 2))
        self.assertEqual(bf.GPU_POOL, [1])

    def test_pipeline_cap_exempt_but_solver_capped(self):
        self.assertIsNone(run.cap_for(bf, 9004))
        self.assertAlmostEqual(runtime_cap_s(9004), 361.20033)
        self.assertIn(bf.SolverCapExceeded, bf.CAP_ERRORS)

    def test_default_preset_is_canonical(self):
        with _clean_env():
            eff = bf.effective_preset()
        self.assertEqual((eff["preset"], eff["image"]), ("official-default", "bundlefusion:noflip"))
        self.assertEqual((eff["app"], eff["bundling"]), ({}, {}))

    def test_historic_presets(self):
        self.assertEqual(bf.PRESETS["batch1-scene9004_51"]["bundling"]["s_minKeyScale"], "1.0f")
        self.assertEqual(bf.PRESETS["batch2-scene9004_52"]["bundling"]["s_minKeyScale"], "0.5f")
        self.assertEqual(bf.PRESETS["s2b4-surfarea-004"]["bundling"]["s_surfAreaPcaThresh"], "0.04f")
        self.assertEqual(bf.PRESETS["batch1-scene9004_51"]["app"]["s_hashNumSDFBlocks"], "1500000")

    def test_unknown_preset_fails(self):
        with _clean_env(SPELLBOOK_BUNDLEFUSION_PRESET="nope"):
            with self.assertRaises(RuntimeError):
                bf.effective_preset()

    def test_overlays_merge_over_preset(self):
        with _clean_env(SPELLBOOK_BUNDLEFUSION_PRESET="batch2-scene9004_52",
                        SPELLBOOK_BUNDLEFUSION_BUNDLING="s_minKeyScale=2.0f; s_submapSize=15",
                        SPELLBOOK_BUNDLEFUSION_APP="s_maxFrameFixes=10",
                        SPELLBOOK_BUNDLEFUSION_IMAGE="x:y",
                        SPELLBOOK_BUNDLEFUSION_NAME="cand",
                        SPELLBOOK_BUNDLEFUSION_HYPOTHESIS="why"):
            eff = bf.effective_preset()
            prov = bf.provenance()
        self.assertEqual(eff["bundling"], {"s_minKeyScale": "2.0f", "s_submapSize": "15"})
        self.assertEqual(eff["app"]["s_maxFrameFixes"], "10")
        self.assertEqual(eff["app"]["s_hashNumSDFBlocks"], "1500000")
        self.assertEqual(eff["image"], "x:y")
        self.assertEqual((prov["variant"], prov["hypothesis"]), ("cand", "why"))
        self.assertEqual(prov["config"]["opt_rounds"], 0)

    def test_bad_overlay_fails(self):
        with _clean_env(SPELLBOOK_BUNDLEFUSION_APP="s_x"):
            with self.assertRaises(RuntimeError):
                bf.effective_preset()


class ConfigTests(unittest.TestCase):
    def setUp(self):
        with open(bf._APP_CFG_SRC) as f:
            self.app = f.read()
        with open(bf._BUNDLE_CFG_SRC) as f:
            self.bundling = f.read()

    def test_official_app_only_adds_sensor_and_k(self):
        out = bf._patch_app_config(self.app, 1, 2, 3, 4)
        self.assertIn("s_sensorIdx = 1;", out)
        self.assertIn("s_cameraIntrinsicCy = 4.000000;", out)
        self.assertEqual(out.split("\ns_cameraIntrinsicFx")[0],
                         self.app.replace("s_sensorIdx = 8;", "s_sensorIdx = 1;"))

    def test_official_bundling_is_byte_identical(self):
        self.assertEqual(bf._patch_bundling_config(self.bundling), self.bundling)

    def test_overrides_and_unknown_key(self):
        out = bf._patch_bundling_config(self.bundling, {"s_minKeyScale": "0.5f"})
        self.assertRegex(out, r"(?m)^s_minKeyScale = 0\.5f;")
        out = bf._patch_app_config(self.app, 1, 2, 3, 4, {"s_hashNumSDFBlocks": "1500000"})
        self.assertRegex(out, r"(?m)^s_hashNumSDFBlocks = 1500000;")
        with self.assertRaises(RuntimeError):
            bf._patch_bundling_config(self.bundling, {"s_typo": "1"})


class PrepareTests(unittest.TestCase):
    def test_logged_k_both_scenes(self):
        """Expected K quoted in #52 2026-09-17."""
        *_, s, k = bf.crop_and_scale(1920, 1200, _K(735.307861, 941.765747, 599.485962))
        self.assertAlmostEqual(s, 0.4)
        np.testing.assert_allclose(k, (294.12, 294.12, 312.71, 239.79), atol=0.005)
        x0, y0, cw, ch, s, k = bf.crop_and_scale(1920, 1080, _K(741.178345, 941.904175, 539.248291))
        self.assertEqual((x0, y0, cw, ch), (240, 0, 1440, 1080))
        np.testing.assert_allclose(k, (329.41, 329.41, 311.96, 239.67), atol=0.005)

    def test_prepare_caches_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = _make_pool(tmp, 192, 120, 3)
            out = os.path.join(tmp, "prepared")
            doc = bf.prepare(9004, pool, out)
            self.assertEqual(doc["crop"], [16, 0, 160, 120])
            self.assertEqual(doc["n_frames"], 3)
            self.assertEqual(len(doc["files"]), 6)
            color = cv2.imread(os.path.join(out, "color", "1.jpg"))
            depth = cv2.imread(os.path.join(out, "depth", "2.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(color.shape, (480, 640, 3))
            self.assertEqual((depth.shape, depth.dtype), ((480, 640), np.uint16))
            self.assertEqual(int(depth[0, 0]), 1002)  # NEAREST keeps the depth value

            stamp = os.path.getmtime(os.path.join(out, "manifest.json"))
            bf.prepare(9004, pool, out)  # cache hit
            self.assertEqual(os.path.getmtime(os.path.join(out, "manifest.json")), stamp)

            bf.validate_prepared(out, pool)
            with open(os.path.join(out, "depth", "0.png"), "ab") as f:
                f.write(b"x")
            with self.assertRaises(RuntimeError):
                bf.validate_prepared(out, pool)

    def test_changed_pool_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = _make_pool(tmp, 192, 120, 2)
            out = os.path.join(tmp, "prepared")
            bf.prepare(9004, pool, out)
            with open(os.path.join(tmp, "frames_manifest.json"), "w") as f:
                json.dump({"color_n": 99}, f)
            with self.assertRaises(RuntimeError):
                bf.validate_prepared(out, pool)
            self.assertEqual(bf.prepare(9004, pool, out)["n_frames"], 2)

    def test_gap_in_frame_ids_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = _make_pool(tmp, 192, 120, 3)
            os.remove(os.path.join(pool, "color", "1.jpg"))
            with self.assertRaises(RuntimeError):
                bf.prepare(9004, pool, os.path.join(tmp, "prepared"))


class PoseTests(unittest.TestCase):
    def test_non_finite_dropped_never_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = np.eye(4)
            good[0, 3] = 2.5
            np.savetxt(os.path.join(tmp, "0.txt"), good)
            np.savetxt(os.path.join(tmp, "4.txt"), good)
            np.savetxt(os.path.join(tmp, "2.txt"), np.full((4, 4), -np.inf))
            poses, keep = bf._parse_poses(tmp, 5, "log")
            self.assertEqual(keep, [0, 4])
            np.testing.assert_array_equal(poses[1], good)
            with self.assertRaises(RuntimeError):
                bf._parse_poses(tmp, 3, "log")  # index past the frame count


class DockerTests(unittest.TestCase):
    def test_pair_and_rounds_in_command(self):
        gpu = ["--device", "/dev/nvidia1", "-e", "CUDA_VISIBLE_DEVICES=1", "-e", "X=y"]
        done = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(bf, "gpu_args", return_value=gpu), \
                mock.patch.object(bf.subprocess, "run", return_value=done) as sub:
            bf._run_docker("img:1", "/data/p", "/data/c", "/data/o", (1, 2), (7, 8))
        cmd, kwargs = sub.call_args[0][0], sub.call_args[1]
        self.assertIn("CUDA_VISIBLE_DEVICES=1,2", cmd)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=1", cmd)
        self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID", cmd)
        self.assertEqual(kwargs["pass_fds"], (7, 8))
        self.assertEqual(kwargs["timeout"], 10800)
        self.assertEqual(cmd[-3:], ["/input", "/output", "0"])
        self.assertIn("/var/lib/snapd/hostfs/data/p:/input:ro", cmd)  # the snap mount helper


class ReconstructTests(unittest.TestCase):
    """reconstruct() with docker mocked: a fake solver writes the output directory."""

    def _run(self, solver_s, returncode=0, log="[info] processed 2/2 frames\n", write=True):
        tmp = self.tmp
        root = os.path.join(tmp, "scene9004_50")
        work = os.path.join(root, "recon")
        os.makedirs(work)
        os.makedirs(os.path.join(tmp, "pool"))
        pool = _make_pool(os.path.join(tmp, "pool"), 192, 120, 2)
        os.symlink(pool, os.path.join(work, "frames"))

        def fake_docker(image, prepared, cfg_dir, out_dir, gpus, lease_fds):
            self.call = (image, gpus, lease_fds)
            if write:
                os.makedirs(os.path.join(out_dir, "pose"))
                np.savetxt(os.path.join(out_dir, "pose", "1.txt"), np.eye(4))
                with open(os.path.join(out_dir, "mesh.ply"), "w") as f:
                    f.write("ply")
            self.clock += solver_s
            return subprocess.CompletedProcess([], returncode, log, "")

        lease = mock.MagicMock()
        lease.fileno.return_value = 9
        lease.__enter__.return_value = lease
        self.clock = 0.0
        with mock.patch.object(bf, "_run_docker", fake_docker), \
                mock.patch.object(bf, "_image_id", return_value="sha256:new"), \
                mock.patch.object(bf, "PREPARED_ROOT", os.path.join(tmp, "prepared")), \
                mock.patch.object(bf.time, "monotonic", lambda: self.clock), \
                mock.patch("utils.gpu.gpu_lease", return_value=lease) as gpu_lease, \
                mock.patch("utils.scan_lock.exclusive_lock"):
            result = bf.reconstruct(work, root, gpu=1, lease_fd=5)
        self.gpu_lease = gpu_lease
        return work, result

    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.tmp = self._d.name
        self.addCleanup(self._d.cleanup)
        self._env = _clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_success_records_solver_scope(self):
        work, (native, poses, keep, conv) = self._run(100.0)
        self.assertEqual((keep, conv), ([1], "bundlefusion_world"))
        self.assertEqual(self.call[1:], ((1, 2), (5, 9)))
        self.assertEqual(self.gpu_lease.call_args[0][0], (2,))
        with open(os.path.join(work, "bundlefusion_run.json")) as f:
            doc = json.load(f)
        self.assertEqual((doc["solver_pass"], doc["physical_gpus"]), (True, [1, 2]))
        self.assertEqual(doc["solver_s"], 100.0)
        self.assertAlmostEqual(doc["solver_limit_s"], 361.20033)
        self.assertEqual(bf.provenance_after(work)["image_id"], "sha256:new")
        with open(os.path.join(work, "stage_timings.json")) as f:
            self.assertEqual(json.load(f)["solver"], 100.0)
        self.assertFalse(os.path.exists(os.path.join(work, "bf_out")))
        with open(os.path.join(work, "logs", "zParametersScanNet.effective.txt")) as f:
            self.assertIn("s_cameraIntrinsicFx = 2941.231", f.read())  # crop 160 -> s=4

    def test_solver_over_cap_raises(self):
        with self.assertRaises(bf.SolverCapExceeded):
            self._run(400.0)

    def test_docker_failure_raises(self):
        with self.assertRaisesRegex(RuntimeError, "exit 139"):
            self._run(100.0, returncode=139)

    def test_skipped_frame_raises(self):
        with self.assertRaisesRegex(RuntimeError, "skipped"):
            self._run(100.0, log="[warn] skipping frame 1\n[info] processed 1/2 frames\n")

    def test_wrong_gpu_refused(self):
        with self.assertRaises(RuntimeError):
            bf.reconstruct("w", "r", gpu=3, lease_fd=5)


if __name__ == "__main__":
    unittest.main()
