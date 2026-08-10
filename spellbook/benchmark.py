"""Benchmark backend resolution: ScanNet20 (official 18-class instance protocol) and
ScanNet200 (official 198-class instance protocol = 200 minus wall/floor).

One source of truth for prediction prompts, label ids, GT export, and evaluation.
Protocol details are fixed in code (never in settings.yaml) so invalid combinations
(e.g. ScanNet200 prompts with the ScanNet20 evaluator) are impossible by construction.

settings.yaml (spellbook/settings.yaml):
    default: ScanNet20
    scannet_root: /data/scannet
"""
import importlib.util
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.yaml")
_SCANNET200_CONSTANTS = os.path.join(
    _REPO_ROOT, "BenchmarkScripts", "ScanNet200", "scannet200_constants.py")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_official_constants():
    """Official ScanNet class tables (pure data files). Fail loudly if unavailable --
    never fall back to hand-copied literals (see issue #12)."""
    c = _load_module("scannet200_constants", _SCANNET200_CONSTANTS)
    try:
        s = _load_module("scannet200_splits",
                         os.path.join(os.path.dirname(_SCANNET200_CONSTANTS), "scannet200_splits.py"))
    except FileNotFoundError:
        s = None
    return c, s


class BenchmarkSpec:
    def __init__(self, name, class_labels, valid_ids, gt_column, evaluator):
        self.name = name
        self.class_labels = list(class_labels)
        self.valid_ids = list(valid_ids)
        self.gt_column = gt_column          # label-map tsv column: "nyu40id" | "id"
        self.evaluator = evaluator          # "official" | "scannet200"
        self.label_to_id = dict(zip(self.class_labels, self.valid_ids))
        self.id_to_label = dict(zip(self.valid_ids, self.class_labels))

    def __repr__(self):
        return f"BenchmarkSpec({self.name}, {len(self.class_labels)} classes, {self.evaluator})"


def _build_specs():
    c, _ = _load_official_constants()
    # ScanNet20 instance protocol: 20-class set minus wall/floor (NYU40 ids).
    labels_20 = [l for l in c.CLASS_LABELS_20 if l not in ("wall", "floor")]
    ids_20 = [i for i in c.VALID_CLASS_IDS_20 if i not in (1, 2)]
    # ScanNet200 instance protocol: 200-class set minus wall(id 1)/floor(id 3) -> 198.
    labels_200 = [l for l in c.CLASS_LABELS_200 if l not in ("wall", "floor")]
    ids_200 = [i for i in c.VALID_CLASS_IDS_200 if i not in (1, 3)]
    return {
        "ScanNet20": BenchmarkSpec("ScanNet20", labels_20, ids_20, "nyu40id", "official"),
        "ScanNet200": BenchmarkSpec("ScanNet200", labels_200, ids_200, "id", "scannet200"),
    }


BENCHMARKS = _build_specs()


def _validate():
    for name, spec in BENCHMARKS.items():
        assert len(spec.class_labels) == len(spec.valid_ids), name
        assert len(set(spec.class_labels)) == len(spec.class_labels), f"{name} dup labels"
        assert len(set(spec.valid_ids)) == len(spec.valid_ids), f"{name} dup ids"
    assert len(BENCHMARKS["ScanNet20"].class_labels) == 18, "ScanNet20 must have 18 classes"
    assert len(BENCHMARKS["ScanNet200"].class_labels) == 198, "ScanNet200 must have 198 classes"
    assert 1 not in BENCHMARKS["ScanNet200"].valid_ids and 3 not in BENCHMARKS["ScanNet200"].valid_ids
    assert 1 not in BENCHMARKS["ScanNet20"].valid_ids and 2 not in BENCHMARKS["ScanNet20"].valid_ids


_validate()


def load_settings():
    """Returns {default, scannet_root} from spellbook/settings.yaml."""
    import yaml
    with open(_SETTINGS_FILE) as f:
        cfg = yaml.safe_load(f) or {}
    default = cfg.get("default", "ScanNet20")
    root = cfg.get("scannet_root", "/data/scannet")
    if default not in BENCHMARKS:
        raise ValueError(f"settings.yaml: unknown default benchmark {default!r} "
                         f"(expected one of {list(BENCHMARKS)})")
    if not os.path.isdir(root):
        raise ValueError(f"settings.yaml: scannet_root {root!r} is not a directory")
    return {"default": default, "scannet_root": root}


def resolve_benchmark(name=None):
    """Resolve a benchmark name (or settings default) to a BenchmarkSpec."""
    if name is None:
        name = load_settings()["default"]
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r} (expected one of {list(BENCHMARKS)})")
    return BENCHMARKS[name]


def artifact_paths(spec, scannet_root=None):
    """Official-format artifact roots for a benchmark. `predictions` mirrors the official
    submission layout; `gt`/`evaluations` are project-side derived data."""
    root = scannet_root or load_settings()["scannet_root"]
    return {
        "predictions": os.path.join(root, "predictions", spec.name),
        "gt": os.path.join(root, "derived", "ground_truth", spec.name),
        "evaluations": os.path.join(root, "derived", "evaluations", spec.name),
    }


def submission_dir(spec, run_id, model, scannet_root=None):
    """Official submission root for one (benchmark, run, model): flat <scene>.txt files +
    predicted_masks/. This directory is directly zippable as a benchmark submission."""
    return os.path.join(artifact_paths(spec, scannet_root)["predictions"], run_id, model)
