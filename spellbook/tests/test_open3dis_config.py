"""CPU-only tests for the Open3DIS wrapper config derivation, run-scoped scratch layout and
tracker reconciliation (#3, #15, #24, #60, 0ca2762)."""
import importlib.util
import os
import sys
import tempfile
import unittest

import yaml

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_SPELLBOOK, "predict", "models")
sys.path.insert(0, _SPELLBOOK)


def _module():
    spec = importlib.util.spec_from_file_location(
        "spellbook_open3dis_config", os.path.join(_MODELS, "_open3dis_config.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _module()

# trimmed copy of the official configs/scannet200.yaml keys the derivation touches
BASE = {
    "foundation_model": {
        "grounded_config_file": "./segmenter2d/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "grounded_checkpoint": "../pretrains/foundation_models/groundingdino_swint_ogc.pth",
        "box_threshold": 0.4, "text_threshold": 0.4,
        "sam_checkpoint": "../pretrains/foundation_models/sam_vit_h_4b8939.pth",
        "clip_model": "ViT-L/14@336px", "device": "cuda", "clip_dim": 768,
    },
    "segmenter2d": {"model": "Grounded-SAM"},
    "data": {"dataset_name": "scannet200", "split_path": "./x.txt", "datapath": "./data",
             "img_dim": [640, 480], "rgb_img_dim": [1296, 968], "img_interval": 2,
             "num_classes": 198},
    "refine_grounding": {"top_k": 5},
    "final_instance": {"spp_level": True, "duplicate": True, "iou_overlap": 0.9,
                       "top_k": 300, "scale_semantic_score": 300.0},
    "exp": {"exp_name": "v", "mask2d_output": "maskyoloworldsam",
            "grounded_feat_output": "grounded_feat",
            "refined_grounded_feat_output": "refined_grounded_feat",
            "clustering_3d_output": "hier_agglo", "final_output": "final_result_hier_agglo",
            "save_dir": "../exp_scannet200"},
    "proposals": {"p2d": True, "p3d": True, "agnostic": True, "refined": True},
    "evaluate": {"evalvocab": False, "evalagnostic": False},
    "fp16": True,
}


class ScratchLayoutTests(unittest.TestCase):
    def test_scratch_is_run_and_scene_scoped_under_scannet_root(self):
        path = C.scratch_dir("/data/scannetdata", "run-a", "scene0618_00")
        self.assertEqual(path, "/data/scannetdata/derived/open3dis/run-a/scene0618_00")
        self.assertNotEqual(path, C.scratch_dir("/data/scannetdata", "run-b", "scene0618_00"))
        self.assertFalse(path.startswith(C.OPEN3DIS_REPO))

    def test_scannet_root_from_pointcloud(self):
        self.assertEqual(C.scannet_root_from_pointcloud(
            "/data/scannetdata/scans/scene0618_00/scene0618_00_vh_clean_2.ply"),
            "/data/scannetdata")

    def test_layout_paths(self):
        lay = C.scene_layout("/s", "scene0618_00")
        self.assertEqual(lay["cwd"], "/s/cwd")
        self.assertEqual(lay["scene2d"],
                         "/s/data/Scannet200/Scannet200_2D_5interval/val/scene0618_00")
        # ISBNet forward_test saves to ../../data/Scannet200/Scannet200_3D/val/<dir>
        for key, name in (("dc_feat", "dc_feat_scannet200"),
                          ("proposals3d", "isbnet_clsagnostic_scannet200")):
            rel = os.path.normpath(os.path.join(
                lay["isbnet_cwd"], "../../data/Scannet200/Scannet200_3D/val", name))
            self.assertEqual(rel, lay[key])

    def test_sampled_frames_every_fifth_zero_padded(self):
        ids = [str(i) for i in range(23)]
        self.assertEqual(C.sampled_frames(ids),
                         [(0, "00000"), (5, "00005"), (10, "00010"), (15, "00015"),
                          (20, "00020")])
        names = [n for _, n in C.sampled_frames([str(i) for i in range(2000)])]
        self.assertEqual(names, sorted(names))


class DeriveConfigTests(unittest.TestCase):
    def setUp(self):
        self.lay = C.scene_layout("/s", "scene0618_00")

    def test_base_config_is_official_scannet200(self):
        self.assertTrue(C.BASE_CONFIG.endswith("configs/scannet200.yaml"))

    def test_cluster_config(self):
        cfg = C.derive_config(BASE, self.lay, "scene0618_00", "run-a")
        self.assertEqual(cfg["data"]["img_dim"], [640, 480])
        self.assertEqual(cfg["data"]["rgb_img_dim"], [1296, 968])
        self.assertEqual(cfg["data"]["num_classes"], 198)
        self.assertEqual(cfg["data"]["split_path"], "/s/open3dis_split.txt")
        self.assertEqual(cfg["data"]["datapath"], self.lay["val2d"])
        self.assertEqual(cfg["data"]["spp_path"], self.lay["superpoints"])
        self.assertEqual(cfg["exp"]["save_dir"], "/s/exp")
        self.assertEqual(cfg["exp"]["exp_name"], "run-a")
        self.assertFalse(cfg["exp"]["skip_clustering"])
        self.assertFalse(cfg["exp"]["emit_final"])
        self.assertTrue(cfg["proposals"]["agnostic"])
        self.assertEqual(cfg["refine_grounding"]["top_k"], 5)
        self.assertEqual(cfg["foundation_model"]["box_threshold"], 0.4)

    def test_final_config_is_step4_fusion(self):
        cfg = C.derive_config(BASE, self.lay, "scene0618_00", "run-a", final=True)
        self.assertTrue(cfg["exp"]["skip_clustering"])
        self.assertTrue(cfg["exp"]["emit_final"])
        self.assertEqual(cfg["proposals"],
                         {"p2d": True, "p3d": True, "agnostic": False, "refined": True})
        self.assertEqual(cfg["final_instance"]["top_k"], 300)
        self.assertEqual(cfg["final_instance"]["iou_overlap"], 0.9)

    def test_relative_repo_paths_become_absolute(self):
        fm = C.derive_config(BASE, self.lay, "s", "r", repo="/data/Open3DIS")["foundation_model"]
        self.assertEqual(fm["grounded_checkpoint"],
                         "/data/pretrains/foundation_models/groundingdino_swint_ogc.pth")
        self.assertEqual(fm["sam_checkpoint"],
                         "/data/pretrains/foundation_models/sam_vit_h_4b8939.pth")
        self.assertTrue(fm["grounded_config_file"].startswith("/data/Open3DIS/segmenter2d/"))
        self.assertEqual(fm["clip_model"], "ViT-L/14@336px")

    def test_base_not_mutated(self):
        before = yaml.safe_dump(BASE)
        C.derive_config(BASE, self.lay, "s", "r", final=True)
        self.assertEqual(yaml.safe_dump(BASE), before)

    def test_isbnet_config_absolute_roots(self):
        base = {"data": {"train": {"type": "scannet200", "data_root": "../x"},
                         "test": {"type": "scannet200", "data_root": "../x",
                                  "prefix": "groundtruth"}},
                "dataloader": {"test": {"batch_size": 1, "num_workers": 1}}}
        cfg = C.derive_isbnet_config(base, self.lay)
        self.assertEqual(cfg["data"]["test"]["data_root"], self.lay["val3d"])
        self.assertEqual(cfg["data"]["train"]["type"], "scannet200")


class TrackerTests(unittest.TestCase):
    def test_reconcile_drops_entries_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            lay = C.scene_layout(tmp, "scene0001_00")
            cfg = C.derive_config(BASE, lay, "scene0001_00", "run-a")
            out = os.path.dirname(C.output_path(cfg, "mask2d_output", "scene0001_00"))
            os.makedirs(out)
            open(os.path.join(out, "scene0001_00.pth"), "w").close()
            tracker = os.path.join(lay["cwd"], "tracker_2d.txt")
            C.ensure_tracker(tracker)
            with open(tracker, "a") as f:
                f.write("scene0001_00.pth\nscene0002_00.pth\n")
            self.assertEqual(C.reconcile_tracker(tracker, cfg), ["scene0002_00.pth"])
            with open(tracker) as f:
                self.assertEqual(f.read(), "Processed Scenes .\nscene0001_00.pth\n")
            self.assertEqual(C.reconcile_tracker(tracker, cfg), [])

    def test_missing_tracker_is_noop_and_every_tracker_has_an_output(self):
        cfg = C.derive_config(BASE, C.scene_layout("/s", "x"), "x", "r")
        self.assertEqual(C.reconcile_tracker("/nonexistent/tracker_lifted.txt", cfg), [])
        for key in C.TRACKER_OUTPUTS.values():
            self.assertIn(key, cfg["exp"])


if __name__ == "__main__":
    unittest.main()
