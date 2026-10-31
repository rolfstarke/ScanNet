"""Isaac adapter: presets, poses, gates, docker command. CPU only, no docker daemon or GPU."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from reconstruct.engines import isaac  # noqa: E402


def _rot_x(deg):
    T = np.eye(4)
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T[1:3, 1:3] = [[c, -s], [s, c]]
    return T


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k != "SPELLBOOK_ISAAC_PRESET"}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


class PresetTests(unittest.TestCase):
    def test_official_default_verbatim(self):
        self.assertEqual(isaac.PRESETS["official_default"], dict(
            depth_mode="NEURAL_LIGHT", map_clearing_radius_m=5.0, decay_tsdf_rate_hz=5.0,
            voxel_size=0.05, max_integration_distance_m=7.0, truncation_distance_vox=4.0,
            max_weight=5.0))

    def test_optimized_cad_best_verbatim(self):
        self.assertEqual(isaac.PRESETS["optimized_cad_best"], dict(
            depth_mode="NEURAL", map_clearing_radius_m=0.0, decay_tsdf_rate_hz=5.0,
            voxel_size=0.05, max_integration_distance_m=7.0, truncation_distance_vox=4.0,
            max_weight=10.0))

    def test_conf100_is_optimized_plus_confidence(self):
        conf = dict(isaac.PRESETS["conf100_best"])
        self.assertEqual(conf.pop("depth_confidence"), 100)
        self.assertEqual(conf, isaac.PRESETS["optimized_cad_best"])

    def test_selection(self):
        with _env():
            self.assertEqual(isaac.selected_preset(), "conf100_best")
        with _env(SPELLBOOK_ISAAC_PRESET="official_default"):
            self.assertEqual(isaac.selected_preset(), "official_default")
            self.assertEqual(isaac.preset_config()["max_weight"], 5.0)
        with _env(SPELLBOOK_ISAAC_PRESET="nope"):
            self.assertIn("unknown", isaac.preflight())
            with self.assertRaises(ValueError):
                isaac.preset_config()

    def test_variant_names(self):
        with _env(SPELLBOOK_ISAAC_PRESET="official_default"), \
                mock.patch.object(isaac, "_image_id", return_value="sha256:x"):
            prov = isaac.provenance()
        self.assertEqual((prov["variant"], prov["image_id"]), ("official-default", "sha256:x"))
        self.assertEqual(isaac.VARIANT_NAMES["optimized_cad_best"], "neural-noclear-w10")


class ParameterTests(unittest.TestCase):
    def test_nvblox_keys(self):
        p = isaac.nvblox_params(isaac.PRESETS["optimized_cad_best"])
        self.assertEqual(p["map_clearing_radius_m"], 0.0)
        self.assertEqual(p["static_mapper.projective_integrator_max_weight"], 10.0)
        self.assertEqual(p["static_mapper.projective_integrator_max_integration_distance_m"], 7.0)
        self.assertEqual(p["mapping_type"], "static_tsdf")
        self.assertTrue(p["use_tf_transforms"])
        self.assertFalse(p["use_color"])
        self.assertNotIn("static_mapper.projective_integrator_weighting_mode", p)

    def test_cuvslam_stereo_vo(self):
        p = isaac.cuvslam_params()
        self.assertEqual(p["tracking_mode"], 0)
        self.assertTrue(p["rectified_images"])
        self.assertEqual(p["base_frame"], isaac.BASE_FRAME)

    def test_zed_override(self):
        doc = isaac.zed_override(isaac.PRESETS["conf100_best"], "scene9004.svo2")
        p = doc["/**"]["ros__parameters"]
        self.assertEqual(p["svo"]["svo_path"], "/svo/scene9004.svo2")
        self.assertTrue(p["svo"]["use_svo_timestamps"])
        self.assertFalse(p["pos_tracking"]["pos_tracking_enabled"])
        self.assertEqual(p["depth"]["depth_stabilization"], 0)
        self.assertEqual(p["depth"]["depth_confidence"], 100)
        self.assertEqual(p["general"]["pub_downscale_factor"], 2.0)
        self.assertTrue(p["video"]["publish_gray"] and p["video"]["publish_left_right"])
        base = isaac.zed_override(isaac.PRESETS["official_default"], "x")["/**"]["ros__parameters"]
        self.assertEqual(base["depth"]["depth_confidence"], 95)

    def test_run_files_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            isaac.write_run_files(d, isaac.PRESETS["official_default"], "a.svo2")
            with open(os.path.join(d, "run_params.json")) as f:
                doc = json.load(f)
            self.assertEqual(doc["nvblox"]["voxel_size"], 0.05)
            with open(os.path.join(d, "zed_override.yaml")) as f:
                self.assertEqual(yaml.safe_load(f)["/**"]["ros__parameters"]["depth"]["depth_mode"],
                                 "NEURAL_LIGHT")

    def test_domain_per_gpu(self):
        self.assertEqual([isaac.domain_id(g) for g in (1, 2, 4)], [101, 102, 104])


class DockerCommandTests(unittest.TestCase):
    def test_docker_helpers_and_pin(self):
        with mock.patch.object(isaac, "gpu_args", return_value=["-e", "CUDA_VISIBLE_DEVICES=3"]):
            cmd = isaac.docker_command("spellbook-isaac-7", 3, "/data/x/export", "/data/x/logs",
                                       "/data/raw", 100)
        text = " ".join(cmd)
        self.assertEqual(cmd[:2], ["docker", "run"])
        self.assertNotIn("podman", text)
        self.assertIn("--name spellbook-isaac-7", text)
        self.assertIn("ROS_DOMAIN_ID=103", text)
        self.assertIn("CUDA_VISIBLE_DEVICES=3", text)
        self.assertIn("/var/lib/snapd/hostfs/data/x/export:/output", text)
        self.assertIn("/var/lib/snapd/hostfs/data/raw:/svo:ro", text)
        self.assertIn(f"--user {os.getuid()}:{os.getgid()}", text)
        self.assertEqual(cmd[-2:], ["100", str(isaac.SAVE_PLY_TIMEOUT_S)])
        self.assertEqual(isaac.SAVE_PLY_TIMEOUT_S, 600)

    def test_prime_flag_and_presets(self):
        with mock.patch.object(isaac, "gpu_args", return_value=[]):
            self.assertIn("ISAAC_PRIME=1", isaac.docker_command("n", 1, "/a", "/b", "/c", 9,
                                                                prime=True))
            self.assertIn("ISAAC_PRIME=0", isaac.docker_command("n", 1, "/a", "/b", "/c", 9))
        modes = {isaac.PRESETS[p]["depth_mode"] for p in isaac.prime_presets()}
        self.assertEqual(modes, {"NEURAL", "NEURAL_LIGHT"})
        self.assertEqual(len(isaac.prime_presets()), 2)

    def test_container_removed_on_cap_exception(self):
        class Boom(BaseException):
            pass

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(isaac.subprocess, "run", side_effect=[Boom(), mock.Mock()]) as run:
            with self.assertRaises(Boom):
                isaac._run_container(["docker", "run"], "n", os.path.join(d, "log"), 0)
        self.assertEqual(run.call_args_list[-1].args[0], ["docker", "rm", "-f", "n"])


class PoseTests(unittest.TestCase):
    def test_quaternion(self):
        s = np.sqrt(0.5)
        np.testing.assert_allclose(isaac.quat_to_rot(s, 0, 0, s), _rot_x(90)[:3, :3], atol=1e-12)
        np.testing.assert_allclose(isaac.quat_to_rot(0, 0, 0, 2), np.eye(3))

    def test_static_search_branched(self):
        a, b, c = np.eye(4), np.eye(4), np.eye(4)
        a[0, 3], b[1, 3], c[2, 3] = 1, 2, 3
        edges = [("link", "center", a), ("center", "imu", _rot_x(10)),
                 ("center", "left", b), ("left", "optical", c)]
        T = isaac.static_transform(edges, "link", "optical")
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(isaac.static_transform(edges, "optical", "link"),
                                   np.linalg.inv(T))
        np.testing.assert_allclose(isaac.static_transform(edges, "imu", "left"),
                                   np.linalg.inv(_rot_x(10)) @ b)
        with self.assertRaises(RuntimeError):
            isaac.static_transform(edges, "link", "absent")

    def test_optical_composition_order(self):
        P = _rot_x(30)
        P[:3, 3] = (1, 0, 0)
        center = np.eye(4)
        center[2, 3] = 0.1
        optical = _rot_x(-90)
        optical[1, 3] = 0.06
        np.testing.assert_allclose(isaac.world_optical(P, center, optical), P @ center @ optical)

    def test_match_frames(self):
        status = [["100", "0", "4", "0"], ["200", "1", "4", "0"], ["300", "2", "4", "0"],
                  ["400", "3", "4", "0"], ["400", "3", "4", "2"]]
        poses = [["100", "a"], ["100", "dup"], ["300", "c"], ["400", "past"], ["999", "x"]]
        got = isaac.match_frames(status, poses, pool_size=3)
        self.assertEqual({k: v[1] for k, v in got.items()}, {0: "a", 2: "c"})

    def test_match_frames_count_form_position(self):
        status = [["0", "0", "3", "0"], ["100", "1", "3", "0"], ["200", "2", "3", "0"],
                  ["300", "3", "3", "0"], ["300", "2", "3", "2"]]
        poses = [["100", "a"], ["200", "b"], ["300", "c"]]
        got = isaac.match_frames(status, poses, pool_size=2)
        self.assertEqual({k: v[1] for k, v in got.items()}, {0: "a", 1: "b"})

    def test_end_status_never_maps(self):
        self.assertEqual(isaac.match_frames([["5", "1", "4", "2"]], [["5", "p"]], 4), {})

    def test_svo_end_gate(self):
        self.assertTrue(isaac.svo_end_ok(3612, 3613))
        self.assertFalse(isaac.svo_end_ok(3000, 3613))
        self.assertFalse(isaac.svo_end_ok(0, 0))


class GateTests(unittest.TestCase):
    def _export(self, d, n=10, vertices=200_000, end=None, total=11):
        with open(os.path.join(d, "svo_status.txt"), "w") as f:
            for i in range(n):
                f.write(f"{1000 + i} {i} {total} 0\n")
        with open(os.path.join(d, "poses.txt"), "w") as f:
            for i in range(n):
                f.write(f"{1000 + i} {i * 0.1} 0 0 0 0 0 1\n")
        with open(os.path.join(d, "vo_status.txt"), "w") as f:
            for i in range(n):
                f.write(f"{1000 + i} 2\n")
        with open(os.path.join(d, "tf_static.txt"), "w") as f:
            f.write(f"{isaac.BASE_FRAME} {isaac.CENTER_FRAME} 0 0 0.015 0 0 0 1\n")
            f.write(f"{isaac.CENTER_FRAME} zed_left_camera_frame 0 0.06 0 0 0 0 1\n")
            f.write(f"zed_left_camera_frame {isaac.LEFT_OPTICAL} 0 0 0 -0.5 0.5 -0.5 0.5\n")
        with open(os.path.join(d, "svo_end.txt"), "w") as f:
            f.write(f"{total - 1 if end is None else end} {total}\n")
        with open(os.path.join(d, "mesh.ply"), "wb") as f:
            f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {vertices}\n"
                    "element face 5\nend_header\n".encode())

    def test_collect_ok(self):
        with tempfile.TemporaryDirectory() as d:
            self._export(d)
            native, poses, keep, stats = isaac.collect_export(
                d, d, 10, isaac.PRESETS["official_default"])
            self.assertTrue(os.path.isfile(native))
            self.assertEqual(keep, list(range(10)))
            self.assertEqual(poses.shape, (10, 4, 4))
            self.assertEqual(stats["mesh_vertices"], 200_000)
            self.assertEqual(stats["vo_state_counts"], {"2": 10})
            self.assertAlmostEqual(stats["traj_m"], 0.9)
            # optical z axis = camera forward = base +x
            np.testing.assert_allclose(poses[0, :3, 2], [1, 0, 0], atol=1e-9)

    def test_gates_refuse(self):
        cfg = isaac.PRESETS["official_default"]
        for kwargs, needle in ((dict(end=5), "SVO stopped"), (dict(vertices=47_114), "100000"),
                               (dict(n=7), "gate")):
            with tempfile.TemporaryDirectory() as d:
                self._export(d, **kwargs)
                with self.assertRaisesRegex(RuntimeError, needle):
                    isaac.collect_export(d, d, 10, cfg)

    def test_effective_mismatch(self):
        cfg = isaac.PRESETS["conf100_best"]
        self.assertEqual(isaac.effective_mismatches(cfg, {}), [])
        ok = {"zed": {"depth.depth_confidence": 100, "depth.depth_mode": "NEURAL"},
              "nvblox": {"voxel_size": 0.05, "map_clearing_radius_m": 0.0}}
        self.assertEqual(isaac.effective_mismatches(cfg, ok), [])
        ok["zed"]["depth.depth_confidence"] = 95
        self.assertEqual(len(isaac.effective_mismatches(cfg, ok)), 1)

    def test_effective_dump_parse(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "effective_nvblox.yaml"), "w") as f:
                f.write("/nvblox_node:\n  ros__parameters:\n    voxel_size: 0.05\n"
                        "    static_mapper:\n      projective_integrator_max_weight: 5.0\n")
            got = isaac.effective_params(d)
            self.assertEqual(got["nvblox"]["static_mapper.projective_integrator_max_weight"], 5.0)
            self.assertEqual(got["zed"], {})


class ContainerFilesTests(unittest.TestCase):
    def test_container_files_present(self):
        for name in ("Dockerfile", "graph.launch.py", "pose_recorder.py", "inside.sh"):
            self.assertTrue(os.path.isfile(os.path.join(isaac.CONTAINER_DIR, name)), name)


if __name__ == "__main__":
    unittest.main()
