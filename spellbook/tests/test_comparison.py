import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml

_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from evaluation.benchmark import BENCHMARKS  # noqa: E402
from evaluation.comparison import (  # noqa: E402
    build_dashboard_payload,
    collect_prediction_rows,
    collect_reconstruction_rows,
    comparison_path,
    custom_scene_groups,
    generate_report,
    incomplete_items,
    is_custom_scan,
    open_report,
    prediction_groups,
    reconstruction_groups,
    render_page,
)
from evaluation.runs import write_run_manifest  # noqa: E402


def _write_csv(path, ap="0.5", ap50="0.6", ap25="0.7"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "class id", "ap", "ap50", "ap25"])
        writer.writeheader()
        writer.writerow({"class": "chair", "class id": "5",
                         "ap": ap, "ap50": ap50, "ap25": ap25})
        writer.writerow({"class": "table", "class id": "7",
                         "ap": "nan", "ap50": "nan", "ap25": "nan"})


def _geometry_doc(scan_id, mean="15.0", acc="20.0", comp="10.0"):
    return {
        "metric": "observed_surface_voxel_mean_bidirectional_distance_v1",
        "mean_bidirectional_distance_cm": float(mean),
        "accuracy_mean_cm": float(acc),
        "completeness_mean_cm": float(comp),
        "scene": scan_id,
        "reference_sha256": "766131128d262b8a7c0cf75243f88c3b269f4c88058c62d9750b380c8b4eb24d",
        "visible_voxels_sha256": "9e47eb8667b43d0f2604f119909489d05db7989f469159368c5ab4414d724d9c",
        "voxel_mm": 10,
    }


def _scan(root, scan_id, mean="15.0"):
    recon = os.path.join(root, "scans", scan_id, "recon")
    os.makedirs(recon, exist_ok=True)
    with open(os.path.join(recon, "geometry_score.yaml"), "w") as f:
        yaml.safe_dump(_geometry_doc(scan_id, mean=mean), f)
    with open(os.path.join(recon, "run.json"), "w") as f:
        json.dump({"engine": "metashape", "scan_id": scan_id}, f)


class PredictionDiscoveryTests(unittest.TestCase):
    def test_historical_manifest_and_csv_discovered(self):
        spec = BENCHMARKS["ScanNet20"]
        with tempfile.TemporaryDirectory() as root:
            write_run_manifest(spec, "run-a", ["scene0568_01"], ["mosaic3d"],
                               scannet_root=root)
            _write_csv(os.path.join(
                root, "derived", "evaluations", "ScanNet20", "run-a", "mosaic3d.csv"))
            rows = collect_prediction_rows(root)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["domain"], "prediction")
            self.assertEqual(row["status"], "scored")
            self.assertAlmostEqual(row["ap"], 0.5)
            self.assertEqual(row["rank"], 1)

    def test_orphan_csv_listed_without_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            _write_csv(os.path.join(
                root, "derived", "evaluations", "ScanNet20", "old", "mosaic3d.csv"))
            rows = collect_prediction_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "missing_manifest")
            self.assertIsNone(rows[0]["rank"])

    def test_methods_sharing_scenes_share_rank_group(self):
        spec = BENCHMARKS["ScanNet20"]
        with tempfile.TemporaryDirectory() as root:
            write_run_manifest(spec, "run-a", ["scene0568_01"], ["mosaic3d"],
                               scannet_root=root)
            _write_csv(os.path.join(
                root, "derived", "evaluations", "ScanNet20", "run-a", "mosaic3d.csv"),
                ap="0.9")
            write_run_manifest(spec, "run-b", ["scene0568_01"], ["openyolo3d"],
                               scannet_root=root)
            _write_csv(os.path.join(
                root, "derived", "evaluations", "ScanNet20", "run-b", "openyolo3d.csv"),
                ap="0.1")
            groups = prediction_groups(collect_prediction_rows(root))
            self.assertEqual(len(groups), 1)
            self.assertEqual([r["run_id"] for r in groups[0]["rows"]], ["run-a", "run-b"])
            self.assertEqual([r["rank"] for r in groups[0]["rows"]], [1, 2])

    def test_different_scenes_never_share_rank(self):
        spec = BENCHMARKS["ScanNet20"]
        with tempfile.TemporaryDirectory() as root:
            write_run_manifest(spec, "run-a", ["scene0568_01"], ["mosaic3d"],
                               scannet_root=root)
            _write_csv(os.path.join(
                root, "derived", "evaluations", "ScanNet20", "run-a", "mosaic3d.csv"),
                ap="0.9")
            write_run_manifest(spec, "run-b", ["scene0575_00"], ["mosaic3d"],
                               scannet_root=root)
            _write_csv(os.path.join(
                root, "derived", "evaluations", "ScanNet20", "run-b", "mosaic3d.csv"),
                ap="0.1")
            groups = prediction_groups(collect_prediction_rows(root))
            self.assertEqual(len(groups), 2)
            for group in groups:
                self.assertEqual([r["rank"] for r in group["rows"]], [1])


class ReconstructionDiscoveryTests(unittest.TestCase):
    def test_scored_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as root:
            _scan(root, "scene9004_10", mean="17.0")
            os.makedirs(os.path.join(root, "scans", "scene9009_10", "recon"))
            rows = collect_reconstruction_rows(os.path.join(root, "scans"), root)
            by_id = {r["scan_id"]: r for r in rows}
            self.assertEqual(by_id["scene9004_10"]["status"], "scored")
            self.assertEqual(by_id["scene9009_10"]["status"], "missing_score")
            self.assertIsNone(by_id["scene9009_10"]["rank"])

    def test_custom_classification_from_svo(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "custom", "raw"))
            open(os.path.join(root, "custom", "raw", "scene9004.svo2"), "w").close()
            self.assertTrue(is_custom_scan("scene9004_10", root))
            self.assertFalse(is_custom_scan("scene0568_00", root))

    def test_best_mean_ranks_first(self):
        with tempfile.TemporaryDirectory() as root:
            _scan(root, "scene9004_10", mean="17.0")
            _scan(root, "scene9004_17", mean="15.0")
            rows = collect_reconstruction_rows(os.path.join(root, "scans"), root)
            by_id = {r["scan_id"]: r for r in rows}
            self.assertEqual(by_id["scene9004_17"]["rank"], 1)
            self.assertEqual(by_id["scene9004_10"]["rank"], 2)

    def test_unscored_custom_only_in_custom_groups(self):
        with tempfile.TemporaryDirectory() as root:
            _scan(root, "scene9004_10", mean="17.0")
            os.makedirs(os.path.join(root, "custom", "raw"))
            open(os.path.join(root, "custom", "raw", "scene9009.svo2"), "w").close()
            os.makedirs(os.path.join(root, "scans", "scene9009_10", "recon"))
            rows = collect_reconstruction_rows(os.path.join(root, "scans"), root)
            customs = custom_scene_groups(rows)
            self.assertIn("scene9009", customs)
            self.assertEqual(len(reconstruction_groups(rows)[0]["rows"]), 1)
            missing = incomplete_items([], rows)
            self.assertFalse(any("scene9009_10" in item["label"] for item in missing))


class HtmlReportTests(unittest.TestCase):
    def _payload(self, root):
        spec = BENCHMARKS["ScanNet20"]
        write_run_manifest(spec, "run-a", ["scene0568_01"], ["mosaic3d"],
                           scannet_root=root)
        _write_csv(os.path.join(
            root, "derived", "evaluations", "ScanNet20", "run-a", "mosaic3d.csv"))
        _scan(root, "scene9004_10", mean="17.0")
        return build_dashboard_payload(root, os.path.join(root, "scans"))

    def test_report_contains_three_sections_and_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            payload = self._payload(root)
            page = render_page(payload["prediction_groups"],
                               payload["reconstruction_groups"],
                               payload["custom_groups"], payload["incomplete"],
                               "2026-09-07 00:00")
            for section in ("Predictions", "Reconstructions", "Custom scans",
                            "Incomplete artifacts"):
                self.assertIn(section, page)
            self.assertIn("run-a", page)
            self.assertIn("scene9004_10", page)

    def test_dynamic_values_are_escaped(self):
        page = render_page(
            [{"benchmark": "ScanNet20", "scenes": ["scene0568_01"], "scene_count": 1,
              "rows": [{"rank": 1, "method": "<b>mosaic</b>", "run_id": "run-a",
                        "ap": 0.5, "ap50": 0.6, "ap25": 0.7, "scene_count": 1,
                        "elapsed_s": None}]}],
            [], {}, [], "now")
        self.assertNotIn("<b>mosaic</b>", page)
        self.assertIn("&lt;b&gt;mosaic&lt;/b&gt;", page)

    def test_no_external_references(self):
        with tempfile.TemporaryDirectory() as root:
            payload = self._payload(root)
            page = render_page(payload["prediction_groups"],
                               payload["reconstruction_groups"],
                               payload["custom_groups"], payload["incomplete"],
                               "now").lower()
            for token in ("<script", "http://", "https://", "@import", "url("):
                self.assertNotIn(token, page)

    def test_tables_deterministic_and_ap_scale_fixed(self):
        with tempfile.TemporaryDirectory() as root:
            payload = self._payload(root)
            first = render_page(payload["prediction_groups"],
                                payload["reconstruction_groups"],
                                payload["custom_groups"], payload["incomplete"],
                                "now")
            second = render_page(payload["prediction_groups"],
                                 payload["reconstruction_groups"],
                                 payload["custom_groups"], payload["incomplete"],
                                 "now")
            self.assertEqual(first, second)
            self.assertIn("width:50.0%", first)  # AP 0.5 on a fixed 0..1 scale

    def test_empty_sections_show_message(self):
        page = render_page([], [], {}, [], "now")
        self.assertIn("No scored predictions found.", page)
        self.assertIn("No scored reconstructions found.", page)
        self.assertIn("No custom scans found.", page)

    def test_generate_report_is_atomic_and_html_only(self):
        with tempfile.TemporaryDirectory() as root:
            self._payload(root)
            out = generate_report(root, os.path.join(root, "scans"))
            self.assertEqual(out, comparison_path(root))
            self.assertTrue(out.endswith("comparison.html"))
            self.assertFalse(os.path.isfile(out + ".tmp"))
            with open(out) as f:
                text = f.read()
            self.assertIn("<!DOCTYPE html>", text)
            for banned in ("chamfer", "normal consistency", "f-score"):
                self.assertNotIn(banned, text.lower())
            for banned_word in ("ate", "rpe"):
                self.assertIsNone(
                    __import__("re").search(r"\b" + banned_word + r"\b",
                                            text.lower()),
                    f"unsupported metric {banned_word!r} in report")

    def test_open_report_without_display(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "comparison.html")
            with open(path, "w") as f:
                f.write("<html></html>")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertFalse(open_report(path))

    def test_open_report_failure_keeps_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "comparison.html")
            with open(path, "w") as f:
                f.write("<html></html>")
            env = {"DISPLAY": ":10", "PATH": os.environ.get("PATH", "")}
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch("evaluation.comparison._resolve_browser",
                               return_value=None):
                self.assertFalse(open_report(path))
            self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
