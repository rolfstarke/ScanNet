#!/usr/bin/env python3
"""Restore official ScanNet validation scenes with parallel, resumable downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request


_SPELLBOOK = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SPELLBOOK)
sys.path.insert(0, _SPELLBOOK)

from evaluation.benchmark import load_settings, normalize_scene_id, official_val_scenes  # noqa: E402


BASE_URL = "https://kaldir.vc.in.tum.de/scannet"
CANONICAL_FILE_TYPES = (
    ".aggregation.json",
    ".txt",
    "_vh_clean.ply",
    "_vh_clean_2.0.010000.segs.json",
    "_vh_clean_2.ply",
    "_vh_clean.segs.json",
    "_vh_clean.aggregation.json",
    "_vh_clean_2.labels.ply",
    ".sens",
)
METADATA_FILE_TYPES = (".aggregation.json", ".txt")
_SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")
MIN_SEGMENTED_BYTES = 64 * 1024 * 1024


def file_url(scene_id, file_type):
    release = "v1" if file_type == ".sens" else "v2"
    return f"{BASE_URL}/{release}/scans/{scene_id}/{scene_id}{file_type}"


def validate_scenes(scenes):
    official = official_val_scenes()
    out = []
    seen = set()
    for value in scenes:
        scene = normalize_scene_id(value)
        if not _SCENE_RE.fullmatch(scene):
            raise ValueError(f"invalid scene id {value!r}")
        if scene not in official:
            raise ValueError(f"scene {scene!r} is not in the official ScanNet v2 val split")
        if scene in seen:
            raise ValueError(f"duplicate scene id {scene!r}")
        seen.add(scene)
        out.append(scene)
    if not out:
        raise ValueError("at least one scene is required")
    return out


def _remote_size(url, attempts=4):
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=60) as response:
                size = int(response.headers["Content-Length"])
            if size <= 0:
                raise RuntimeError(f"invalid Content-Length for {url}: {size}")
            return size
        except Exception as exc:  # urllib exposes several transport-specific exceptions
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"cannot read remote size for {url}: {last_error}")


def _byte_ranges(size, segments):
    chunk_size = (size + segments - 1) // segments
    return [
        (start, min(start + chunk_size - 1, size - 1))
        for start in range(0, size, chunk_size)
    ]


def _download_segmented(url, destination, partial, expected, segments):
    ranges = _byte_ranges(expected, segments)
    paths = [f"{partial}.{start}-{end}" for start, end in ranges]
    missing = []
    for (start, end), path in zip(ranges, paths):
        segment_size = end - start + 1
        if os.path.isfile(path) and os.path.getsize(path) == segment_size:
            continue
        if os.path.exists(path):
            os.remove(path)
        missing.append((start, end, path))

    if missing:
        command = ["curl", "--parallel", "--parallel-immediate",
                   "--parallel-max", str(len(missing))]
        for index, (start, end, path) in enumerate(missing):
            if index:
                command.append("--next")
            command.extend([
                "--fail", "--location", "--silent", "--show-error",
                "--retry", "6", "--retry-all-errors", "--retry-delay", "2",
                "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "90",
                "--range", f"{start}-{end}", "--output", path, url,
            ])
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"segmented download failed for {os.path.basename(destination)}: "
                f"curl rc={result.returncode}")

    for (start, end), path in zip(ranges, paths):
        actual = os.path.getsize(path) if os.path.isfile(path) else 0
        expected_segment = end - start + 1
        if actual != expected_segment:
            raise RuntimeError(
                f"incomplete range {start}-{end} for {os.path.basename(destination)}: "
                f"{actual}/{expected_segment} bytes")

    with open(partial, "wb") as output:
        for path in paths:
            with open(path, "rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    for path in paths:
        os.remove(path)


def _download_one(scene_id, file_type, scannet_root, replace=False, segments=1):
    url = file_url(scene_id, file_type)
    scene_dir = os.path.join(scannet_root, "scans", scene_id)
    destination = os.path.join(scene_dir, scene_id + file_type)
    partial = destination + ".part"
    os.makedirs(scene_dir, exist_ok=True)

    expected = _remote_size(url)
    if replace:
        for path in (destination, partial, *glob.glob(partial + ".*")):
            if os.path.exists(path):
                os.remove(path)

    if os.path.isfile(destination):
        actual = os.path.getsize(destination)
        if actual == expected:
            for path in (partial, *glob.glob(partial + ".*")):
                if os.path.exists(path):
                    os.remove(path)
            return {"path": destination, "status": "skipped", "bytes": actual}
        raise RuntimeError(
            f"existing file has wrong size: {destination} ({actual} != {expected}); use --replace")

    if os.path.isfile(partial) and os.path.getsize(partial) > expected:
        raise RuntimeError(f"partial file is larger than remote file: {partial}; use --replace")

    if segments > 1 and expected >= MIN_SEGMENTED_BYTES:
        if not os.path.isfile(partial) or os.path.getsize(partial) != expected:
            _download_segmented(url, destination, partial, expected, segments)
    elif not os.path.isfile(partial) or os.path.getsize(partial) < expected:
        command = [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "6", "--retry-all-errors", "--retry-delay", "2",
            "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "90",
            "--continue-at", "-", "--output", partial, url,
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            actual = os.path.getsize(partial) if os.path.isfile(partial) else 0
            raise RuntimeError(
                f"download failed for {scene_id}{file_type}: curl rc={result.returncode}, "
                f"partial={actual}/{expected} bytes")

    actual = os.path.getsize(partial) if os.path.isfile(partial) else 0
    if actual != expected:
        raise RuntimeError(
            f"incomplete download for {scene_id}{file_type}: {actual}/{expected} bytes")
    os.replace(partial, destination)
    for path in glob.glob(partial + ".*"):
        os.remove(path)
    return {"path": destination, "status": "downloaded", "bytes": actual}


def download_scenes(scenes, file_types=None, workers=4, scannet_root=None, replace=False,
                    segments=1):
    scenes = validate_scenes(scenes)
    file_types = tuple(file_types or CANONICAL_FILE_TYPES)
    unknown = [file_type for file_type in file_types if file_type not in CANONICAL_FILE_TYPES]
    if unknown:
        raise ValueError(f"unsupported file types: {unknown}")
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer in [1, 16]")
    if not isinstance(segments, int) or isinstance(segments, bool) or not 1 <= segments <= 16:
        raise ValueError("segments must be an integer in [1, 16]")
    if shutil.which("curl") is None:
        raise FileNotFoundError("curl is required for resumable ScanNet downloads")

    root = scannet_root or load_settings()["scannet_root"]
    jobs = [(scene, file_type) for scene in scenes for file_type in file_types]
    results = []
    print(f"restoring {len(scenes)} scenes ({len(jobs)} files, {workers} parallel downloads)")
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {
            pool.submit(_download_one, scene, file_type, root, replace, segments): (scene, file_type)
            for scene, file_type in jobs
        }
        try:
            for future in as_completed(futures):
                scene, file_type = futures[future]
                result = future.result()
                results.append(result)
                size_mb = result["bytes"] / (1024 * 1024)
                print(f"{result['status']:10s} {scene}{file_type} ({size_mb:.1f} MiB)", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download canonical files for official ScanNet v2 validation scenes")
    parser.add_argument("--scene", nargs="+", required=True,
                        help="official val scene IDs, e.g. scene0458_01")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--metadata-only", action="store_true",
                       help="download only .txt and .aggregation.json files")
    group.add_argument("--types", nargs="+", choices=CANONICAL_FILE_TYPES,
                       help="download only the selected canonical file types")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel file downloads (default: 4; maximum: 16)")
    parser.add_argument("--segments", type=int, default=1,
                        help="parallel byte ranges per file >=64 MiB (default: 1; maximum: 16)")
    parser.add_argument("--scannet-root", help="override settings.yaml scannet_root")
    parser.add_argument("--replace", action="store_true",
                        help="redownload files, discarding final and partial copies")
    args = parser.parse_args(argv)

    file_types = METADATA_FILE_TYPES if args.metadata_only else args.types
    try:
        download_scenes(
            args.scene, file_types=file_types, workers=args.workers,
            scannet_root=args.scannet_root, replace=args.replace, segments=args.segments)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
