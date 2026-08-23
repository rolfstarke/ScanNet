"""Exact prediction/evaluation artifact cleanup for one scan ID."""
import glob
import os
import shutil

from evaluation.benchmark import BENCHMARKS, artifact_paths, load_settings
from utils.scan_lock import exclusive_lock, prediction_index_lock_path


def _safe_unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def purge_scan_predictions(scene_id, scannet_root=None):
    """Delete only exact prediction indexes/masks and matching task lines/CSVs for scene_id."""
    root = scannet_root or load_settings()["scannet_root"]
    with exclusive_lock(prediction_index_lock_path(root)):
        for spec in BENCHMARKS.values():
            paths = artifact_paths(spec, root)
            pred_root = paths["predictions"]
            eval_root = paths["evaluations"]
            if os.path.isdir(pred_root):
                for run_id in os.listdir(pred_root):
                    run_dir = os.path.join(pred_root, run_id)
                    if not os.path.isdir(run_dir):
                        continue
                    for model in os.listdir(run_dir):
                        model_dir = os.path.join(run_dir, model)
                        if not os.path.isdir(model_dir):
                            continue
                        changed = False
                        idx = os.path.join(model_dir, f"{scene_id}.txt")
                        if os.path.isfile(idx):
                            # delete referenced masks inside model_dir
                            try:
                                with open(idx) as f:
                                    for line in f:
                                        p = line.strip().split()[0] if line.strip() else ""
                                        if not p:
                                            continue
                                        full = p if os.path.isabs(p) else os.path.join(model_dir, p)
                                        full = os.path.normpath(full)
                                        if full.startswith(os.path.normpath(model_dir) + os.sep):
                                            _safe_unlink(full)
                            except OSError:
                                pass
                            _safe_unlink(idx)
                            changed = True
                        mask_dir = os.path.join(model_dir, "predicted_masks")
                        if os.path.isdir(mask_dir):
                            for p in glob.glob(os.path.join(mask_dir, f"{scene_id}_*.txt")):
                                _safe_unlink(p)
                                changed = True
                        if changed:
                            # stale aggregate CSV under evaluations
                            csv_path = os.path.join(eval_root, run_id, f"{model}.csv")
                            _safe_unlink(csv_path)

            if os.path.isdir(eval_root):
                for run_id in os.listdir(eval_root):
                    run_dir = os.path.join(eval_root, run_id)
                    if not os.path.isdir(run_dir):
                        continue
                    for name in os.listdir(run_dir):
                        if not name.endswith(".tasks"):
                            continue
                        tasks = os.path.join(run_dir, name)
                        try:
                            with open(tasks) as f:
                                lines = f.readlines()
                        except OSError:
                            continue
                        new = [ln for ln in lines if ln.strip() != scene_id]
                        if len(new) != len(lines):
                            tmp = tasks + ".tmp"
                            with open(tmp, "w") as f:
                                f.writelines(new)
                            os.replace(tmp, tasks)
                            csv_path = os.path.join(run_dir, name.replace(".tasks", ".csv"))
                            _safe_unlink(csv_path)


def remove_scan_dir(path):
    if os.path.isdir(path) or os.path.islink(path):
        if os.path.islink(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
