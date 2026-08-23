"""Smoke test for spellbook/evaluation/ package migration. Exit 1 on any FAIL."""
from __future__ import annotations

import os
import subprocess
import sys

SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SPELLBOOK)
PY = sys.executable
ok = fail = skip = 0


def report(status: str, msg: str) -> None:
    global ok, fail, skip
    print(f"[{status}] {msg}")
    if status == "OK":
        ok += 1
    elif status == "FAIL":
        fail += 1
    else:
        skip += 1


def main() -> int:
    eval_dir = os.path.join(SPELLBOOK, "evaluation")
    required = ("__init__.py", "benchmark.py", "evaluate.py", "scannet200_evaluator.py")
    for name in required:
        path = os.path.join(eval_dir, name)
        if os.path.isfile(path):
            report("OK", f"layout has evaluation/{name}")
        else:
            report("FAIL", f"missing evaluation/{name}")

    for stale in ("benchmark.py", "evaluate.py", "scannet200_evaluator.py"):
        path = os.path.join(SPELLBOOK, stale)
        if os.path.exists(path):
            report("FAIL", f"stale top-level still present: {stale}")
        else:
            report("OK", f"no top-level {stale}")

    if SPELLBOOK not in sys.path:
        sys.path.insert(0, SPELLBOOK)

    try:
        from evaluation.benchmark import (
            BENCHMARKS,
            load_settings,
            resolve_benchmark,
        )
        s20 = resolve_benchmark("ScanNet20")
        s200 = resolve_benchmark("ScanNet200")
        assert s20.name == "ScanNet20" and len(s20.class_labels) == 18
        assert s200.name == "ScanNet200" and len(s200.class_labels) == 198
        assert set(BENCHMARKS) >= {"ScanNet20", "ScanNet200"}
        cfg = load_settings()
        assert "scannet_root" in cfg and "gpu_pool" in cfg
        report("OK", "package import resolve_benchmark + load_settings")
    except Exception as exc:
        report("FAIL", f"package import: {exc}")

    try:
        import importlib.util

        path = os.path.join(SPELLBOOK, "evaluation", "benchmark.py")
        spec = importlib.util.spec_from_file_location("spellbook_benchmark", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.resolve_benchmark("ScanNet20").label_to_id
        report("OK", "foreign-env absolute load of evaluation/benchmark.py")
    except Exception as exc:
        report("FAIL", f"foreign-env absolute load: {exc}")

    try:
        models_dir = os.path.join(SPELLBOOK, "predict", "models")
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        from common import _benchmark_spec

        assert _benchmark_spec("ScanNet200").name == "ScanNet200"
        report("OK", "common._benchmark_spec foreign-env helper")
    except Exception as exc:
        report("FAIL", f"common._benchmark_spec: {exc}")

    sc200 = os.path.join(eval_dir, "scannet200_evaluator.py")
    if os.path.isfile(sc200):
        report("OK", f"scannet200 script path {sc200}")
    else:
        report("FAIL", "scannet200 script missing")

    for args in (
        [PY, os.path.join(eval_dir, "evaluate.py")],
        [PY, os.path.join(eval_dir, "evaluate.py"), "export-gt", "--help"],
        [PY, os.path.join(eval_dir, "evaluate.py"), "evaluate", "--help"],
    ):
        proc = subprocess.run(args, capture_output=True, text=True)
        label = " ".join(args[1:])
        # bare invoke exits 2 with usage; --help exits 0
        if "--help" in args:
            if proc.returncode == 0:
                report("OK", f"CLI help: {label}")
            else:
                report("FAIL", f"CLI help: {label} rc={proc.returncode} {proc.stderr[:200]}")
        else:
            if proc.returncode == 2 and "usage:" in (proc.stdout + proc.stderr):
                report("OK", f"CLI usage: {label}")
            else:
                report("FAIL", f"CLI usage: {label} rc={proc.returncode}")

    scene = "scene0568_00"
    scan = os.path.join("/data/scannet", "scans", scene)
    if os.path.isdir(scan):
        proc = subprocess.run(
            [PY, os.path.join(eval_dir, "evaluate.py"), "export-gt",
             "--scene", "0568_00", "--benchmark", "ScanNet20"],
            capture_output=True, text=True)
        gt = os.path.join("/data/scannet", "derived", "ground_truth", "ScanNet20", f"{scene}.txt")
        if proc.returncode == 0 and os.path.isfile(gt) and os.path.getsize(gt) > 0:
            report("OK", f"GT export smoke -> {gt}")
        else:
            report("FAIL", f"GT export rc={proc.returncode} stderr={proc.stderr[:300]}")
    else:
        report("SKIP", f"no {scan} for GT export")

    try:
        from evaluation.scannet200_evaluator import Evaluator

        assert hasattr(Evaluator, "evaluate")
        report("OK", "lazy Evaluator import path")
    except Exception as exc:
        report("FAIL", f"Evaluator import: {exc}")

    proc = subprocess.run(
        ["grep", "-rn",
         r"from benchmark import\|from scannet200_evaluator import\|spellbook/benchmark\.py\|spellbook/evaluate\.py\|spellbook/scannet200_evaluator",
         SPELLBOOK, "--include=*.py", "--include=*.md"],
        capture_output=True, text=True)
    hits = []
    for line in (proc.stdout or "").splitlines():
        if "/archive/" in line or "/tmp/" in line or "__pycache__" in line:
            continue
        if "evaluation/scannet200_evaluator.py" in line and "RozDavid" in line:
            continue
        hits.append(line)
    if hits:
        report("FAIL", f"stale path/import hits ({len(hits)}):\n  " + "\n  ".join(hits[:12]))
    else:
        report("OK", "grep gate: no stale imports/paths")

    print(f"\nsummary: OK={ok} FAIL={fail} SKIP={skip}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
