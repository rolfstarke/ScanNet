"""Orchestrates --predict: runs each requested model as a subprocess in its own conda env,
writing ScanNet official benchmark submissions to
/data/scannet/predictions/<Benchmark>/<run-id>/<model>/ (flat <scene>.txt files +
predicted_masks/, directly zippable for scan-net.org).

Tasks are the (model x scene) cross-product, dispatched in parallel over the shared
automatic GPU pool from spellbook/settings.yaml (gpu_pool): each worker blocks on a
cross-process flock lease for one of GPUs 1-4, runs one task, then releases it. Physical
GPU 0 is never allocated (reserved for the ZED SDK, issue #18). The lease descriptor is
forwarded to the model child with pass_fds so the GPU job survives the runner; children
see a single visible GPU via CUDA_VISIBLE_DEVICES=<physical index>. Completed (scene)
tasks are appended to a per-run marker file under
derived/evaluations/<Benchmark>/<run-id>/<model>.tasks as the resume state.
"""
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from benchmark import load_settings

SCANS_DIR = "/data/scannet/scans"

LEASE_FD_ENV = "SPELLBOOK_GPU_LEASE_FD"

MODEL_PYTHON = {
    "mosaic3d": "/data/mosaic3d/conda/envs/mosaic3d/bin/python",
    "openins3d": "/data/openins3d/conda/envs/openins3d/bin/python",
    "openmask3d": "/home/rolf/anaconda3/envs/openmask3d/bin/python",
    "openyolo3d": "/data/openyolo3D/conda/envs/openyolo3d/bin/python",
    "open3dis": "/data/open3dis/conda/envs/open3dis/bin/python",
}

MODEL_RUN_SCRIPT = {
    "mosaic3d": "models/_mosaic3d_run.py",
    "openins3d": "models/_openins3d_run.py",
    "openmask3d": "models/_openmask3d_run.py",
    "openyolo3d": "models/_openyolo3d_run.py",
    "open3dis": "models/_open3dis_run.py",
}

MODEL_NEEDS_FRAMES = {
    "mosaic3d": False,
    "openins3d": False,
    "openmask3d": True,
    "openyolo3d": True,
    "open3dis": True,
}

OPENYOLO3D_CUDA_HOME = "/data/openyolo3D/cuda-11.3"


def _pointcloud_path(scene_id):
    return os.path.join(SCANS_DIR, scene_id, f"{scene_id}_vh_clean_2.ply")


def _run_one(model, scene_id, frames_dir, classes, gpu, out_dir, benchmark, tasks_log, lease):
    """Run one (model, scene) task in a subprocess; appends scene_id to the task marker file
    on success. Returns (model, scene_id, out_dir, elapsed, ok)."""
    os.makedirs(out_dir, exist_ok=True)

    run_script = os.path.join(os.path.dirname(__file__), MODEL_RUN_SCRIPT[model])
    args = [MODEL_PYTHON[model], run_script, "--pointcloud", _pointcloud_path(scene_id),
            "--classes", *classes, "--out", out_dir, "--benchmark", benchmark]
    if MODEL_NEEDS_FRAMES.get(model, False):
        if frames_dir is None:
            raise RuntimeError(f"{model} needs frames but extraction produced none for {scene_id}")
        args += ["--frames", frames_dir]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if lease is not None:
        env[LEASE_FD_ENV] = str(lease.fileno())
    if model == "openyolo3d":
        env["LD_LIBRARY_PATH"] = os.path.join(OPENYOLO3D_CUDA_HOME, "lib64") + ":" + env.get("LD_LIBRARY_PATH", "")

    pass_fds = () if lease is None else (lease.fileno(),)

    print(f"[INFO] running {model} on {scene_id} (auto GPU {gpu}) ...")
    start = time.time()
    try:
        subprocess.run(args, env=env, check=True, pass_fds=pass_fds)
        elapsed = time.time() - start
        print(f"[INFO] {model} on {scene_id} done in {elapsed:.1f}s -> {out_dir}")
        with open(tasks_log, "a") as f:
            f.write(scene_id + "\n")
        return model, scene_id, out_dir, elapsed, True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] {model} on {scene_id} FAILED (gpu {gpu}, rc={e.returncode}) after {time.time() - start:.1f}s")
        return model, scene_id, None, time.time() - start, False


def predict(scene_ids, models, classes, benchmark="ScanNet20", run_id=None, replace=False):
    """Run predictions for `models` on all `scene_ids`, in parallel over the automatic
    settings GPU pool.

    Each (benchmark, run, model) writes an official ScanNet submission directly to
    /data/scannet/predictions/<Benchmark>/<run-id>/<model>/. `classes` is the prompt class
    list; None -> the benchmark's official protocol classes. `run_id` isolates outputs;
    None -> one auto-generated run-YYYYMMDD-HHMMSS id for the whole call.
    Returns a list of (model, scene_id, out_dir, elapsed, ok) per task."""
    from benchmark import artifact_paths, resolve_benchmark, submission_dir
    from utils.gpu import gpu_lease

    settings = load_settings()
    gpu_pool = settings["gpu_pool"]
    scannet_root = settings["scannet_root"]

    spec = resolve_benchmark(benchmark)
    if classes is None:
        classes = spec.class_labels
    if run_id is None:
        run_id = time.strftime("run-%Y%m%d-%H%M%S")
    print(f"[INFO] benchmark={spec.name} classes={len(classes)} run_id={run_id}")

    eval_root = artifact_paths(spec)["evaluations"]

    unknown = [m for m in models if m not in MODEL_PYTHON]
    for m in unknown:
        print(f"[WARN] unknown model '{m}', skipping")
    models = [m for m in models if m in MODEL_PYTHON]

    for scene_id in scene_ids:
        if not os.path.isdir(os.path.join(SCANS_DIR, scene_id)):
            raise ValueError(f"scene directory not found: {os.path.join(SCANS_DIR, scene_id)}")
        if not os.path.isfile(_pointcloud_path(scene_id)):
            raise ValueError(f"pointcloud not found: {_pointcloud_path(scene_id)}")

    # Shared per-scene frame extraction, sequential up front (parallel dispatch would race on it).
    need_frames = any(MODEL_NEEDS_FRAMES.get(m, False) for m in models)
    frames_dir_by_scene = {}
    if need_frames:
        from predict.frames import extract_frames
        for scene_id in scene_ids:
            frames_dir_by_scene[scene_id] = extract_frames(scene_id, replace)

    tasks = [(m, s) for m in models for s in scene_ids]
    print(f"[INFO] {len(tasks)} tasks, automatic GPU pool {gpu_pool}")

    def _task_args(model):
        tasks_log = os.path.join(eval_root, run_id, f"{model}.tasks")
        os.makedirs(os.path.dirname(tasks_log), exist_ok=True)
        return submission_dir(spec, run_id, model), tasks_log

    results = []

    def work(task):
        model, scene_id = task
        out_dir, tasks_log = _task_args(model)
        with gpu_lease(gpu_pool, scannet_root) as lease:
            return _run_one(model, scene_id, frames_dir_by_scene.get(scene_id), classes,
                            lease.index, out_dir, benchmark, tasks_log, lease)

    max_workers = max(1, min(len(gpu_pool), len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, t) for t in tasks]
        for fut in futures:
            results.append(fut.result())

    return results
