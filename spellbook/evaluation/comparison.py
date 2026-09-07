"""Minimal offline comparison dashboard for reconstruction and prediction runs.

Read-only discovery over existing artifacts (no rescoring, no new runs):

- Predictions: run.json manifests + official evaluator CSVs + per-scene timing.
- Reconstructions: scans/*/recon/geometry_score.yaml + recon/run.json + timing.json.

``python spellbook/main.py --compare`` regenerates one self-contained HTML
file; historical and future runs appear automatically. Only metrics already
produced by the current pipelines are reported (AP/AP50/AP25, mean
bidirectional distance, accuracy/completeness, elapsed seconds). ATE/RPE,
Chamfer, normal consistency and F-score have no producers and are never
invented here.
"""

import html
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime

COMPARISON_HTML_NAME = "comparison.html"

STATUS_SCORED = "scored"
STATUS_MISSING_SCORE = "missing_score"
STATUS_MISSING_MANIFEST = "missing_manifest"
STATUS_INVALID_SCORE = "invalid_score"

BROWSER_CANDIDATES = ("chromium", "/snap/bin/chromium", "google-chrome", "firefox")

METHOD_COLORS = {
    "mosaic3d": "#2f6fed",
    "openins3d": "#0e9f6e",
    "openyolo3d": "#e02424",
    "open3dis": "#c27803",
    "openmask3d": "#7c3aed",
    "zed": "#2f6fed",
    "metashape": "#0e9f6e",
    "rtabmap": "#e02424",
    "isaac": "#7c3aed",
    "open3d": "#c27803",
    "bundlefusion": "#0694a2",
}
FALLBACK_COLOR = "#6b7280"

_CSS = """
:root { color-scheme: light; }
body { background: #fff; color: #111827; font-family: system-ui, -apple-system,
  "Segoe UI", Roboto, Ubuntu, Cantarell, sans-serif; margin: 0; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 20px 64px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
p.sub { color: #4b5563; margin: 0 0 24px; }
h2 { font-size: 1.15rem; margin: 32px 0 8px; }
h3 { font-size: 1rem; margin: 20px 0 8px; font-weight: 600; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
thead th { position: sticky; top: 0; background: #f9fafb; text-align: left;
  font-weight: 600; border-bottom: 1px solid #e5e7eb; padding: 8px 10px;
  white-space: nowrap; }
tbody td { padding: 7px 10px; border-bottom: 1px solid #f1f2f4;
  white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.85rem; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 9999px;
  margin-right: 6px; vertical-align: baseline; }
.bar { display: block; height: 4px; background: #eef2f7; border-radius: 2px;
  margin-top: 4px; min-width: 72px; }
.bar > span { display: block; height: 4px; border-radius: 2px; }
.status { color: #4b5563; font-size: 0.8rem; }
.empty { color: #4b5563; }
details { margin-top: 12px; }
summary { cursor: pointer; font-weight: 600; }
ul.plain { list-style: none; padding-left: 0; }
ul.plain li { padding: 3px 0; border-bottom: 1px solid #f1f2f4; font-size: 0.88rem; }
"""


def _spellbook():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def comparison_path(scannet_root=None):
    from evaluation.benchmark import load_settings
    root = scannet_root or load_settings()["scannet_root"]
    return os.path.join(root, "derived", "evaluations", COMPARISON_HTML_NAME)


def _atomic_write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.isfile(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_ap(value):
    number = _finite(value)
    return "-" if number is None else f"{number:.3f}"


def _fmt_cm(value):
    number = _finite(value)
    return "-" if number is None else f"{number:.2f}"


def _fmt_runtime(seconds):
    number = _finite(seconds)
    if number is None:
        return "-"
    try:
        from utils.compute_time import format_elapsed
        text = format_elapsed(float(number))
    except (ImportError, OSError, TypeError, ValueError):
        text = None
    return text or "-"


def _color_for(name):
    return METHOD_COLORS.get(name or "", FALLBACK_COLOR)


def _prediction_protocol_id(benchmark, scenes):
    return "pred:{}:{}".format(benchmark, ";".join(scenes))


def _reconstruction_protocol_id(metric, source_scene, reference_sha, visible_sha, voxel_mm):
    return "recon:{}:{}:{}:{}:{}".format(
        metric, source_scene, reference_sha or "-", visible_sha or "-", voxel_mm or "-")


def _prediction_total_elapsed(spec, run_id, method, scenes, scannet_root):
    from utils.compute_time import load_timing, prediction_timing_path
    total = 0.0
    for scene in scenes:
        elapsed = load_timing(
            prediction_timing_path(spec, run_id, method, scene, scannet_root),
            "prediction", scene_id=scene, run_id=run_id, model=method)
        if elapsed is None:
            return None
        total += float(elapsed)
    return total


def _sidecar_scenes(eval_dir, run_id, method):
    sidecar_dir = os.path.join(eval_dir, run_id, method)
    if not os.path.isdir(sidecar_dir):
        return []
    scenes = []
    for name in sorted(os.listdir(sidecar_dir)):
        if name.endswith(".tp50.json"):
            scenes.append(name[:-len(".tp50.json")])
    return scenes


def collect_prediction_rows(scannet_root=None):
    from evaluation.benchmark import BENCHMARKS, artifact_paths
    from evaluation.runs import _is_comparable, load_manifest, read_evaluator_csv
    rows = []
    for spec in BENCHMARKS.values():
        eval_root = artifact_paths(spec, scannet_root)["evaluations"]
        if not os.path.isdir(eval_root):
            continue
        for run_id in sorted(os.listdir(eval_root)):
            run_dir = os.path.join(eval_root, run_id)
            if not os.path.isdir(run_dir):
                continue
            manifest_path = os.path.join(run_dir, "run.json")
            manifest = None
            if os.path.isfile(manifest_path):
                try:
                    manifest = load_manifest(manifest_path, spec=spec, run_id=run_id)
                except (OSError, ValueError) as exc:
                    print(f"[WARN] skipping manifest {run_id}: {exc}", file=sys.stderr)
                    continue
            if manifest is not None:
                for method in sorted(manifest["methods"]):
                    csv_path = os.path.join(run_dir, f"{method}.csv")
                    scenes = list(manifest["scenes"])
                    protocol = _prediction_protocol_id(spec.name, scenes)
                    comparable = bool(_is_comparable(manifest))
                    issue = manifest.get("issue")
                    if not os.path.isfile(csv_path):
                        rows.append({
                            "domain": "prediction",
                            "benchmark": spec.name,
                            "protocol_id": protocol,
                            "method": method,
                            "run_id": run_id,
                            "scan_id": "",
                            "source_scene": "",
                            "scenes": scenes,
                            "scene_count": len(scenes),
                            "ap": None, "ap50": None, "ap25": None,
                            "mean_bidirectional_distance_cm": None,
                            "accuracy_mean_cm": None,
                            "completeness_mean_cm": None,
                            "elapsed_s": _prediction_total_elapsed(
                                spec, run_id, method, scenes, scannet_root),
                            "comparable": comparable,
                            "status": STATUS_MISSING_SCORE,
                            "rank": None,
                            "issue": issue,
                            "source_path": csv_path,
                        })
                        continue
                    try:
                        metrics = read_evaluator_csv(csv_path)
                    except (OSError, ValueError) as exc:
                        print(f"[WARN] skipping {run_id}/{method}: {exc}", file=sys.stderr)
                        rows.append({
                            "domain": "prediction",
                            "benchmark": spec.name,
                            "protocol_id": protocol,
                            "method": method,
                            "run_id": run_id,
                            "scan_id": "",
                            "source_scene": "",
                            "scenes": scenes,
                            "scene_count": len(scenes),
                            "ap": None, "ap50": None, "ap25": None,
                            "mean_bidirectional_distance_cm": None,
                            "accuracy_mean_cm": None,
                            "completeness_mean_cm": None,
                            "elapsed_s": _prediction_total_elapsed(
                                spec, run_id, method, scenes, scannet_root),
                            "comparable": comparable,
                            "status": STATUS_INVALID_SCORE,
                            "rank": None,
                            "issue": issue,
                            "source_path": csv_path,
                        })
                        continue
                    rows.append({
                        "domain": "prediction",
                        "benchmark": spec.name,
                        "protocol_id": protocol,
                        "method": method,
                        "run_id": run_id,
                        "scan_id": "",
                        "source_scene": "",
                        "scenes": scenes,
                        "scene_count": len(scenes),
                        "ap": metrics["ap"],
                        "ap50": metrics["ap50"],
                        "ap25": metrics["ap25"],
                        "mean_bidirectional_distance_cm": None,
                        "accuracy_mean_cm": None,
                        "completeness_mean_cm": None,
                        "elapsed_s": _prediction_total_elapsed(
                            spec, run_id, method, scenes, scannet_root),
                        "comparable": comparable,
                        "status": STATUS_SCORED,
                        "rank": None,
                        "issue": issue,
                        "source_path": csv_path,
                    })
            else:
                # Orphan evaluator CSV without a manifest: historical visibility only.
                for name in sorted(os.listdir(run_dir)):
                    if not name.endswith(".csv") or name == "ranking.csv":
                        continue
                    method = name[:-4]
                    csv_path = os.path.join(run_dir, name)
                    try:
                        metrics = read_evaluator_csv(csv_path)
                    except (OSError, ValueError):
                        continue
                    scenes = _sidecar_scenes(eval_root, run_id, method)
                    try:
                        from evaluation.benchmark import validate_prediction_methods
                        validate_prediction_methods([method])
                    except ValueError:
                        continue
                    rows.append({
                        "domain": "prediction",
                        "benchmark": spec.name,
                        "protocol_id": _prediction_protocol_id(
                            spec.name, scenes) if scenes else "",
                        "method": method,
                        "run_id": run_id,
                        "scan_id": "",
                        "source_scene": "",
                        "scenes": scenes,
                        "scene_count": len(scenes),
                        "ap": metrics["ap"],
                        "ap50": metrics["ap50"],
                        "ap25": metrics["ap25"],
                        "mean_bidirectional_distance_cm": None,
                        "accuracy_mean_cm": None,
                        "completeness_mean_cm": None,
                        "elapsed_s": None,
                        "comparable": False,
                        "status": STATUS_MISSING_MANIFEST,
                        "rank": None,
                        "issue": None,
                        "source_path": csv_path,
                    })
    _assign_prediction_ranks(rows)
    return rows


def _assign_prediction_ranks(rows):
    groups = {}
    for row in rows:
        if row["domain"] != "prediction" or row["status"] != STATUS_SCORED:
            continue
        if not row["protocol_id"] or row["ap"] is None:
            continue
        groups.setdefault(row["protocol_id"], []).append(row)
    for protocol in sorted(groups):
        members = sorted(groups[protocol], key=lambda r: (-float(r["ap"]), r["run_id"]))
        for index, row in enumerate(members, start=1):
            row["rank"] = index


def _load_geometry_reference():
    import yaml
    path = os.path.join(_spellbook(), "reconstruct", "geometry_reference.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _reconstruction_identity(scan_id, recon_dir):
    engine = None
    issue = None
    run_path = os.path.join(recon_dir, "run.json")
    if os.path.isfile(run_path):
        try:
            with open(run_path) as f:
                doc = json.load(f)
            if isinstance(doc, dict):
                if isinstance(doc.get("engine"), str):
                    engine = doc["engine"]
                if isinstance(doc.get("issue"), int) and doc["issue"] > 0:
                    issue = doc["issue"]
        except (OSError, ValueError):
            pass
    if engine is None:
        try:
            from reconstruct import parse_scan_id
            _, engine, _ = parse_scan_id(scan_id)
        except (ValueError, ImportError):
            engine = ""
    return engine or "", issue


def is_custom_scan(scan_id, scannet_root=None, reference=None):
    try:
        from reconstruct import parse_scan_id
        scene_num, _, _ = parse_scan_id(scan_id)
    except (ValueError, ImportError):
        return False
    custom_raw = os.path.join(
        scannet_root or "/data/scannet", "custom", "raw", f"scene{scene_num:04d}.svo2")
    if os.path.isfile(custom_raw):
        return True
    try:
        reference = reference if reference is not None else _load_geometry_reference()
    except OSError:
        return False
    return f"scene{scene_num:04d}" in (reference.get("scenes") or {})


def _read_geometry_score(path):
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError("geometry score is not a mapping")
    return doc


def collect_reconstruction_rows(scans_dir=None, scannet_root=None):
    from utils.compute_time import load_timing, reconstruction_timing_path
    root = scannet_root or "/data/scannet"
    scans = scans_dir or os.path.join(root, "scans")
    try:
        reference = _load_geometry_reference()
    except OSError:
        reference = None
    rows = []
    if not os.path.isdir(scans):
        return rows
    for scan_id in sorted(os.listdir(scans)):
        scan_dir = os.path.join(scans, scan_id)
        recon_dir = os.path.join(scan_dir, "recon")
        if not os.path.isdir(recon_dir):
            continue
        try:
            from reconstruct import parse_scan_id
            scene_num, _, _ = parse_scan_id(scan_id)
            source_scene = f"scene{scene_num:04d}"
        except (ValueError, ImportError):
            continue
        engine, issue = _reconstruction_identity(scan_id, recon_dir)
        score_path = os.path.join(recon_dir, "geometry_score.yaml")
        elapsed = load_timing(
            reconstruction_timing_path(scan_dir), "reconstruction", scene_id=scan_id)
        custom = is_custom_scan(scan_id, root, reference)
        if not os.path.isfile(score_path):
            rows.append({
                "domain": "reconstruction",
                "benchmark": source_scene,
                "protocol_id": "",
                "method": engine,
                "run_id": "",
                "scan_id": scan_id,
                "source_scene": source_scene,
                "scenes": [scan_id],
                "scene_count": 1,
                "ap": None, "ap50": None, "ap25": None,
                "mean_bidirectional_distance_cm": None,
                "accuracy_mean_cm": None,
                "completeness_mean_cm": None,
                "elapsed_s": elapsed,
                "comparable": False,
                "status": STATUS_MISSING_SCORE,
                "rank": None,
                "issue": issue,
                "source_path": score_path,
                "custom": custom,
                "voxel_mm": None,
            })
            continue
        try:
            doc = _read_geometry_score(score_path)
            metric = doc.get("metric")
            mean = _finite(doc.get("mean_bidirectional_distance_cm"))
            acc = _finite(doc.get("accuracy_mean_cm"))
            comp = _finite(doc.get("completeness_mean_cm"))
            reference_sha = doc.get("reference_sha256")
            visible_sha = doc.get("visible_voxels_sha256")
            voxel_mm = doc.get("voxel_mm")
            try:
                voxel_mm = int(voxel_mm)
            except (TypeError, ValueError):
                voxel_mm = None
            if mean is None or acc is None or comp is None:
                raise ValueError("non-finite geometry metrics")
            if mean < 0 or acc < 0 or comp < 0:
                raise ValueError("negative geometry metrics")
            comparable = False
            if reference is not None:
                scene_ref = (reference.get("scenes") or {}).get(source_scene)
                comparable = bool(
                    scene_ref is not None
                    and metric == reference.get("metric")
                    and reference_sha == scene_ref.get("reference_sha256")
                    and visible_sha == scene_ref.get("visible_voxels_sha256")
                    and int(voxel_mm) == int(reference.get("voxel_mm")))
            protocol = _reconstruction_protocol_id(
                metric, source_scene, reference_sha, visible_sha, voxel_mm)
            rows.append({
                "domain": "reconstruction",
                "benchmark": source_scene,
                "protocol_id": protocol,
                "method": engine,
                "run_id": "",
                "scan_id": scan_id,
                "source_scene": source_scene,
                "scenes": [scan_id],
                "scene_count": 1,
                "ap": None, "ap50": None, "ap25": None,
                "mean_bidirectional_distance_cm": mean,
                "accuracy_mean_cm": acc,
                "completeness_mean_cm": comp,
                "elapsed_s": elapsed,
                "comparable": comparable,
                "status": STATUS_SCORED if comparable else STATUS_INVALID_SCORE,
                "rank": None,
                "issue": issue,
                "source_path": score_path,
                "custom": custom,
                "voxel_mm": voxel_mm,
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[WARN] skipping geometry {scan_id}: {exc}", file=sys.stderr)
            rows.append({
                "domain": "reconstruction",
                "benchmark": source_scene,
                "protocol_id": "",
                "method": engine,
                "run_id": "",
                "scan_id": scan_id,
                "source_scene": source_scene,
                "scenes": [scan_id],
                "scene_count": 1,
                "ap": None, "ap50": None, "ap25": None,
                "mean_bidirectional_distance_cm": None,
                "accuracy_mean_cm": None,
                "completeness_mean_cm": None,
                "elapsed_s": elapsed,
                "comparable": False,
                "status": STATUS_INVALID_SCORE,
                "rank": None,
                "issue": issue,
                "source_path": score_path,
                "custom": custom,
                "voxel_mm": None,
            })
    _assign_reconstruction_ranks(rows)
    return rows


def _assign_reconstruction_ranks(rows):
    groups = {}
    for row in rows:
        if row["domain"] != "reconstruction" or row["status"] != STATUS_SCORED:
            continue
        value = row.get("mean_bidirectional_distance_cm")
        if value is None or not row["protocol_id"]:
            continue
        groups.setdefault(row["protocol_id"], []).append(row)
    for protocol in sorted(groups):
        members = sorted(
            groups[protocol],
            key=lambda r: (float(r["mean_bidirectional_distance_cm"]), r["scan_id"]))
        for index, row in enumerate(members, start=1):
            row["rank"] = index


def prediction_groups(rows):
    groups = {}
    for row in rows or []:
        if row.get("domain") != "prediction" or row.get("status") != STATUS_SCORED:
            continue
        if _finite(row.get("ap")) is None:
            continue
        key = (row.get("benchmark"), tuple(row.get("scenes") or []))
        groups.setdefault(key, []).append(row)
    out = []
    for (benchmark, scenes) in sorted(groups, key=lambda k: (k[0], -len(k[1]), list(k[1]))):
        members = sorted(groups[(benchmark, scenes)],
                         key=lambda r: (-float(r["ap"]), r["run_id"]))
        out.append({
            "benchmark": benchmark,
            "scenes": list(scenes),
            "scene_count": len(scenes),
            "rows": members,
        })
    return out


def reconstruction_groups(rows):
    groups = {}
    for row in rows or []:
        if row.get("domain") != "reconstruction" or row.get("status") != STATUS_SCORED:
            continue
        if _finite(row.get("mean_bidirectional_distance_cm")) is None:
            continue
        protocol = row.get("protocol_id")
        if not protocol:
            continue
        groups.setdefault(protocol, []).append(row)
    out = []
    for protocol in sorted(groups):
        members = sorted(groups[protocol],
                         key=lambda r: (float(r["mean_bidirectional_distance_cm"]),
                                        r["scan_id"]))
        first = members[0]
        try:
            voxel_mm = int(first.get("voxel_mm"))
        except (TypeError, ValueError):
            voxel_mm = ""
        out.append({
            "protocol_id": protocol,
            "source_scene": first.get("source_scene") or "",
            "voxel_mm": voxel_mm,
            "rows": members,
        })
    out.sort(key=lambda g: (g["source_scene"], str(g["voxel_mm"])))
    return out


def custom_scene_groups(rows):
    groups = {}
    for row in rows or []:
        if not row.get("custom"):
            continue
        groups.setdefault(row.get("source_scene") or "", []).append(row)
    out = {}
    for scene in sorted(groups):
        members = sorted(
            groups[scene],
            key=lambda r: (
                0 if _finite(r.get("mean_bidirectional_distance_cm")) is not None else 1,
                float(r["mean_bidirectional_distance_cm"])
                if _finite(r.get("mean_bidirectional_distance_cm")) is not None
                else float("inf"),
                r.get("scan_id") or ""))
        out[scene] = members
    return out


def incomplete_items(predictions, reconstructions):
    items = []
    for row in predictions or []:
        if row.get("status") == STATUS_SCORED and _finite(row.get("ap")) is not None:
            continue
        items.append({
            "label": "prediction \u00b7 {} \u00b7 {} \u00b7 {}".format(
                row.get("benchmark") or "-",
                row.get("method") or "-",
                row.get("run_id") or "-"),
            "status": row.get("status") or "unknown",
        })
    for row in reconstructions or []:
        if row.get("custom"):
            continue
        if row.get("status") == STATUS_SCORED:
            continue
        items.append({
            "label": "reconstruction \u00b7 {} \u00b7 {}".format(
                row.get("scan_id") or "-",
                row.get("method") or "-"),
            "status": row.get("status") or "unknown",
        })
    items.sort(key=lambda item: (item["label"], item["status"]))
    return items


def _prediction_table(group):
    lines = []
    benchmark = _esc(group["benchmark"])
    lines.append(f"<h3>{benchmark} \u2013 {group['scene_count']} scenes</h3>")
    lines.append('<div class="table-wrap"><table>')
    lines.append("<thead><tr><th>#</th><th>Method</th><th>Run</th>"
                 '<th class="num">AP</th><th class="num">AP50</th>'
                 '<th class="num">AP25</th><th class="num">Scenes</th>'
                 "<th>Runtime</th></tr></thead><tbody>")
    for row in group["rows"]:
        ap = _finite(row.get("ap")) or 0.0
        pct = max(0.0, min(100.0, ap * 100.0))
        color = _color_for(row.get("method"))
        lines.append(
            "<tr>"
            f"<td>{_esc(row.get('rank'))}</td>"
            f"<td><span class=\"dot\" style=\"background:{color}\"></span>{_esc(row.get('method'))}</td>"
            f"<td class=\"mono\">{_esc(row.get('run_id'))}</td>"
            f"<td class=\"num\">{_esc(_fmt_ap(row.get('ap')))}"
            f"<span class=\"bar\"><span style=\"width:{pct:.1f}%;background:{color}\"></span></span></td>"
            f"<td class=\"num\">{_esc(_fmt_ap(row.get('ap50')))}</td>"
            f"<td class=\"num\">{_esc(_fmt_ap(row.get('ap25')))}</td>"
            f"<td class=\"num\">{_esc(row.get('scene_count'))}</td>"
            f"<td>{_esc(_fmt_runtime(row.get('elapsed_s')))}</td>"
            "</tr>")
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def _reconstruction_table(group):
    lines = []
    src = _esc(group["source_scene"])
    voxel = _esc(group["voxel_mm"])
    heading = f"{src} \u2013 CAD surface, {voxel} mm" if voxel != "" else f"{src}"
    lines.append(f"<h3>{heading}</h3>")
    lines.append('<div class="table-wrap"><table>')
    lines.append("<thead><tr><th>#</th><th>Engine</th><th>Scan</th>"
                 '<th class="num">Mean, cm (lower is better)</th>'
                 '<th class="num">Accuracy, cm</th><th class="num">Completeness, cm</th>'
                 "<th>Runtime</th></tr></thead><tbody>")
    values = [_finite(r.get("mean_bidirectional_distance_cm")) or 0.0
              for r in group["rows"]]
    top = max(values) if values else 0.0
    for row in group["rows"]:
        mean = _finite(row.get("mean_bidirectional_distance_cm")) or 0.0
        pct = max(2.0, min(100.0, (mean / top * 100.0 if top > 0 else 0.0)))
        color = _color_for(row.get("method"))
        lines.append(
            "<tr>"
            f"<td>{_esc(row.get('rank'))}</td>"
            f"<td><span class=\"dot\" style=\"background:{color}\"></span>{_esc(row.get('method'))}</td>"
            f"<td class=\"mono\">{_esc(row.get('scan_id'))}</td>"
            f"<td class=\"num\">{_esc(_fmt_cm(row.get('mean_bidirectional_distance_cm')))}"
            f"<span class=\"bar\"><span style=\"width:{pct:.1f}%;background:{color}\"></span></span></td>"
            f"<td class=\"num\">{_esc(_fmt_cm(row.get('accuracy_mean_cm')))}</td>"
            f"<td class=\"num\">{_esc(_fmt_cm(row.get('completeness_mean_cm')))}</td>"
            f"<td>{_esc(_fmt_runtime(row.get('elapsed_s')))}</td>"
            "</tr>")
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


def _custom_tables(groups):
    lines = []
    if not groups:
        return '<p class="empty">No custom scans found.</p>'
    for scene in sorted(groups):
        lines.append(f"<h3>{_esc(scene)}</h3>")
        lines.append('<div class="table-wrap"><table>')
        lines.append("<thead><tr><th>Engine</th><th>Scan</th>"
                     '<th class="num">Mean, cm</th><th class="num">Accuracy, cm</th>'
                     '<th class="num">Completeness, cm</th><th>Runtime</th>'
                     "<th>Status</th></tr></thead><tbody>")
        for row in groups[scene]:
            mean = _finite(row.get("mean_bidirectional_distance_cm"))
            status = "scored" if row.get("status") == STATUS_SCORED and mean is not None \
                else "No geometry score"
            color = _color_for(row.get("method"))
            lines.append(
                "<tr>"
                f"<td><span class=\"dot\" style=\"background:{color}\"></span>{_esc(row.get('method'))}</td>"
                f"<td class=\"mono\">{_esc(row.get('scan_id'))}</td>"
                f"<td class=\"num\">{_esc(_fmt_cm(row.get('mean_bidirectional_distance_cm')))}</td>"
                f"<td class=\"num\">{_esc(_fmt_cm(row.get('accuracy_mean_cm')))}</td>"
                f"<td class=\"num\">{_esc(_fmt_cm(row.get('completeness_mean_cm')))}</td>"
                f"<td>{_esc(_fmt_runtime(row.get('elapsed_s')))}</td>"
                f"<td class=\"status\">{_esc(status)}</td>"
                "</tr>")
        lines.append("</tbody></table></div>")
    return "\n".join(lines)


def _incomplete_block(items):
    lines = [f"<details><summary>Incomplete artifacts ({len(items or [])})</summary>"]
    if not items:
        lines.append('<p class="empty">No incomplete artifacts.</p>')
        lines.append("</details>")
        return "\n".join(lines)
    lines.append('<ul class="plain">')
    for item in items:
        lines.append(f"<li>{_esc(item['label'])} "
                     f"<span class=\"status\">[{_esc(item['status'])}]</span></li>")
    lines.append("</ul></details>")
    return "\n".join(lines)


def render_page(pred_groups, recon_groups, custom_groups, incomplete, generated):
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>ScanNet comparison</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<h1>ScanNet comparison</h1>",
        f"<p class=\"sub\">Generated {_esc(generated)} from existing artifacts. "
        "No models or evaluators were run.</p>",
        "<h2>Predictions</h2>",
    ]
    if pred_groups:
        parts.extend(_prediction_table(group) for group in pred_groups)
    else:
        parts.append('<p class="empty">No scored predictions found.</p>')
    parts.append("<h2>Reconstructions</h2>")
    if recon_groups:
        parts.extend(_reconstruction_table(group) for group in recon_groups)
    else:
        parts.append('<p class="empty">No scored reconstructions found.</p>')
    parts.append("<h2>Custom scans</h2>")
    parts.append(_custom_tables(custom_groups))
    parts.append(_incomplete_block(incomplete))
    parts.append("</main></body></html>")
    return "\n".join(part for part in parts if part) + "\n"


def build_dashboard_payload(scannet_root=None, scans_dir=None):
    predictions = collect_prediction_rows(scannet_root)
    reconstructions = collect_reconstruction_rows(scans_dir, scannet_root)
    pred_groups = prediction_groups(predictions)
    recon_groups = reconstruction_groups(reconstructions)
    customs = custom_scene_groups(reconstructions)
    missing = incomplete_items(predictions, reconstructions)
    return {
        "predictions": predictions,
        "reconstructions": reconstructions,
        "prediction_groups": pred_groups,
        "reconstruction_groups": recon_groups,
        "custom_groups": customs,
        "incomplete": missing,
    }


def build_compare_payload(scannet_root=None, scans_dir=None):
    payload = build_dashboard_payload(scannet_root, scans_dir)
    return {
        "schema": 1,
        "predictions": payload["predictions"],
        "reconstructions": payload["reconstructions"],
        "custom_scans": [row for rows in payload["custom_groups"].values()
                         for row in rows],
        "custom_groups": payload["custom_groups"],
        "prediction_groups": payload["prediction_groups"],
        "reconstruction_groups": payload["reconstruction_groups"],
        "incomplete": payload["incomplete"],
    }


def generate_report(scannet_root=None, scans_dir=None):
    from evaluation.benchmark import load_settings
    root = scannet_root or load_settings()["scannet_root"]
    scans = scans_dir or os.path.join(root, "scans")
    payload = build_dashboard_payload(root, scans)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = render_page(payload["prediction_groups"], payload["reconstruction_groups"],
                       payload["custom_groups"], payload["incomplete"], generated)
    out = os.path.join(root, "derived", "evaluations", COMPARISON_HTML_NAME)
    _atomic_write_text(out, page)
    return out


def _resolve_browser():
    for candidate in BROWSER_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def open_report(path):
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    exe = _resolve_browser()
    if exe is None:
        return False
    try:
        from pathlib import Path
        uri = Path(os.path.abspath(path)).as_uri()
        subprocess.Popen([exe, uri],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, ValueError):
        return False
