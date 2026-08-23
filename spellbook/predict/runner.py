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

SUPPORTED_MODELS = ["mosaic3d", "openins3d", "openyolo3d", "open3dis", "openmask3d"]

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

OPENMASK3D_RESOURCES = (
    "/home/rolf/GIT/openmask3d",
    "/data/openmask3d/resources/scannet200_model.ckpt",
    "/data/openmask3d/resources/sam_vit_h_4b8939.pth",
)

OPENYOLO3D_CUDA_HOME = "/data/openyolo3D/cuda-11.3"


def preflight_model(model):
    """Return a reason string when `model` cannot run, else None. No CUDA context is
    ever initialized here."""
    if model not in SUPPORTED_MODELS:
        return f"unknown model {model!r} (expected one of {SUPPORTED_MODELS})"
    py = MODEL_PYTHON[model]
    if not os.path.isfile(py) or not os.access(py, os.X_OK):
        return f"{model}: python not found: {py}"
    script = os.path.join(os.path.dirname(__file__), MODEL_RUN_SCRIPT[model])
    if not os.path.isfile(script):
        return f"{model}: wrapper missing: {script}"
    if model == "openmask3d":
        for p in OPENMASK3D_RESOURCES:
            if not os.path.exists(p):
                return f"openmask3d: required resource missing: {p}"
    return None


def child_env(lease, model):
    """Shared child environment: one visible physical GPU + lease fd (pass_fds must
    mirror this). Used identically by normal prediction and GPU-check probes."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(lease.index)
    env[LEASE_FD_ENV] = str(lease.fileno())
    if model == "openyolo3d":
        env["LD_LIBRARY_PATH"] = os.path.join(OPENYOLO3D_CUDA_HOME, "lib64") + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def _pointcloud_path(scene_id):
    return os.path.join(SCANS_DIR, scene_id, f"{scene_id}_vh_clean_2.ply")


def _run_one(model, scene_id, frames_dir, classes, gpu, out_dir, benchmark, tasks_log, lease):
    """Run one (model, scene) task in a subprocess; appends scene_id to the task marker file
    on success. Returns (model, scene_id, out_dir, elapsed, ok)."""
    from utils.scan_lock import exclusive_lock, prediction_index_lock_path

    os.makedirs(out_dir, exist_ok=True)

    run_script = os.path.join(os.path.dirname(__file__), MODEL_RUN_SCRIPT[model])
    args = [MODEL_PYTHON[model], run_script, "--pointcloud", _pointcloud_path(scene_id),
            "--classes", *classes, "--out", out_dir, "--benchmark", benchmark]
    if MODEL_NEEDS_FRAMES.get(model, False):
        if frames_dir is None:
            raise RuntimeError(f"{model} needs frames but extraction produced none for {scene_id}")
        args += ["--frames", frames_dir]

    env = child_env(lease, model)
    pass_fds = (lease.fileno(),)

    print(f"[INFO] running {model} on {scene_id} (auto GPU {gpu}) ...")
    start = time.time()
    try:
        subprocess.run(args, env=env, check=True, pass_fds=pass_fds)
        elapsed = time.time() - start
        print(f"[INFO] {model} on {scene_id} done in {elapsed:.1f}s -> {out_dir}")
        # append after lease release path is handled by caller; lock the tasks file
        with exclusive_lock(prediction_index_lock_path()):
            os.makedirs(os.path.dirname(tasks_log), exist_ok=True)
            with open(tasks_log, "a") as f:
                f.write(scene_id + "\n")
        return model, scene_id, out_dir, elapsed, True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] {model} on {scene_id} FAILED (gpu {gpu}, rc={e.returncode}) after {time.time() - start:.1f}s")
        return model, scene_id, None, time.time() - start, False


def gpu_check_model(model, lease, hold_seconds=5):
    """Distribution probe for one model in its own conda env under the lease: the probe
    imports torch, requires exactly one visible CUDA device, allocates on it, prints a
    JSON line, and holds briefly. No model code, frames, outputs, or task markers."""
    probe = os.path.join(os.path.dirname(__file__), "models", "_gpu_check.py")
    args = [MODEL_PYTHON[model], probe, "--method", model, "--hold", str(hold_seconds)]
    env = child_env(lease, model)
    start = time.time()
    try:
        r = subprocess.run(args, env=env, capture_output=True, text=True,
                           pass_fds=(lease.fileno(),), timeout=hold_seconds + 90)
    except subprocess.TimeoutExpired:
        return dict(status="fail", reason=f"gpu_check timed out after {hold_seconds + 90}s")
    import json
    if r.returncode != 0:
        return dict(status="fail", reason=f"probe rc={r.returncode}: "
                                          f"{(r.stderr or r.stdout).strip()[-400:]}")
    try:
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return dict(status="fail", reason=f"unparseable probe output: {r.stdout[-400:]}")
    if info.get("physical") != lease.index:
        return dict(status="fail", reason=f"probe saw device index {info.get('physical')}, "
                                          f"expected {lease.index}")
    return dict(status="pass", policy="managed", physical_gpu=lease.index,
                visible_gpu=info.get("visible"), lease_fd=lease.fileno(),
                runtime=info.get("device") or "torch",
                seconds=round(time.time() - start, 1), reason=None)


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

    reasons = [preflight_model(m) for m in models]
    bad = [(m, r) for m, r in zip(models, reasons) if r]
    if bad:
        raise ValueError("prediction preflight failed:\n" +
                         "\n".join(f"  {m}: {r}" for m, r in bad))
    models = [m for m in models if m in SUPPORTED_MODELS]

    from utils.scan_lock import ScanLock, scan_lock_path

    unique_scenes = sorted(set(scene_ids))
    locks = []
    try:
        for scene_id in unique_scenes:
            lk = ScanLock(scan_lock_path(scene_id, scannet_root), exclusive=True)
            lk.acquire()
            locks.append(lk)

        for scene_id in unique_scenes:
            if not os.path.isdir(os.path.join(SCANS_DIR, scene_id)):
                raise ValueError(f"scene directory not found: {os.path.join(SCANS_DIR, scene_id)}")
            if not os.path.isfile(_pointcloud_path(scene_id)):
                raise ValueError(f"pointcloud not found: {_pointcloud_path(scene_id)}")

        need_frames = any(MODEL_NEEDS_FRAMES.get(m, False) for m in models)
        frames_dir_by_scene = {}
        if need_frames:
            from predict.frames import extract_frames
            for scene_id in unique_scenes:
                frames_dir_by_scene[scene_id] = extract_frames(scene_id, replace)

        for lk in locks:
            lk.downgrade_to_shared()

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
    finally:
        for lk in reversed(locks):
            lk.release()
