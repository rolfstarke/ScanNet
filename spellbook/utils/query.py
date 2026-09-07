import hashlib
import json
import os
import subprocess
import tempfile
import threading

import numpy as np


SCHEMA = 1
OPENINS_SCRATCH = "/data/openins3d/scratch"
INSTANT_MODELS = frozenset({"openmask3d", "open3dis", "mosaic3d"})
NATIVE_LOOKUP_MODELS = frozenset({"openins3d"})
ENCODERS = {
    "openmask3d": ("openai_clip", "ViT-L/14@336px", 768),
    "open3dis": ("openai_clip", "ViT-L/14@336px", 768),
    "mosaic3d": ("open_clip", "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP", 768),
}
QUERY_OFF = "off"
QUERY_SEARCH = "search"
QUERY_IMAGE = "image"
SEARCH_TOP_K = 10


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_features_path(spec, run_id, model, scene_id, scannet_root=None):
    from evaluation.benchmark import artifact_paths
    return os.path.join(
        artifact_paths(spec, scannet_root)["evaluations"],
        run_id, model, scene_id + ".clip.npz")


def l2_normalize_rows(features):
    features = np.ascontiguousarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be 2-D")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(~np.isfinite(features)) or np.any(norms[:, 0] < 1e-6):
        raise ValueError("features must be finite with non-zero rows")
    return features / norms


def _scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        item = arr.reshape(-1)[0]
        return item.item() if hasattr(item, "item") else item
    raise ValueError("expected scalar field")


def validate_clip_features(doc, expected):
    if int(_scalar(doc["schema"])) != SCHEMA:
        raise ValueError("unsupported clip feature schema")
    for key in ("scene_id", "benchmark", "run_id", "model", "encoder_family", "encoder_name"):
        if str(_scalar(doc[key])) != str(expected[key]):
            raise ValueError(f"clip feature {key} mismatch")
    if int(_scalar(doc["feature_dim"])) != int(expected["feature_dim"]):
        raise ValueError("clip feature_dim mismatch")
    stored = str(_scalar(doc["index_sha256"]))
    if len(stored) != 64 or stored != expected["index_sha256"]:
        raise ValueError("clip feature index hash mismatch")
    keys = doc["keys"]
    if keys.dtype.kind not in "SU":
        raise ValueError("clip feature keys must be strings")
    features = np.asarray(doc["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != keys.shape[0]:
        raise ValueError("clip feature row count mismatch")
    if features.shape[1] != int(expected["feature_dim"]):
        raise ValueError("clip feature width mismatch")
    features = l2_normalize_rows(features)
    return {
        "keys": [str(key) for key in keys.tolist()],
        "features": features,
        "encoder_family": str(expected["encoder_family"]),
        "encoder_name": str(expected["encoder_name"]),
        "feature_dim": int(expected["feature_dim"]),
        "index_sha256": stored,
    }


def write_clip_features(path, features, keys, index_path, scene_id, benchmark, run_id, model,
                        encoder_family, encoder_name, feature_dim):
    features = l2_normalize_rows(features)
    keys = [str(key) for key in keys]
    if features.shape[0] != len(keys):
        raise ValueError("feature/key count disagree")
    if features.shape[1] != int(feature_dim):
        raise ValueError("feature_dim disagrees with array width")
    index_keys = prediction_index_keys(index_path)
    if keys != index_keys:
        raise ValueError("feature keys do not match prediction index")
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".npz", dir=directory)
    try:
        os.close(fd)
        os.unlink(tmp)
        np.savez(
            tmp,
            schema=np.int32(SCHEMA),
            scene_id=np.asarray(scene_id),
            benchmark=np.asarray(benchmark),
            run_id=np.asarray(run_id),
            model=np.asarray(model),
            encoder_family=np.asarray(encoder_family),
            encoder_name=np.asarray(encoder_name),
            feature_dim=np.int32(feature_dim),
            index_sha256=np.asarray(sha256_file(index_path)),
            keys=np.asarray(keys),
            features=features,
        )
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        for leftover in (tmp, tmp + ".npz"):
            try:
                os.unlink(leftover)
            except OSError:
                pass
        raise


def write_aligned_features(path, feature_rows, source_indices, keys, index_path, **meta):
    if not path:
        return None
    if not keys:
        return None
    rows = [np.asarray(feature_rows[int(i)], dtype=np.float32) for i in source_indices]
    features = np.stack(rows, axis=0)
    write_clip_features(path, features, keys, index_path, **meta)
    return path


def prediction_index_keys(index_path):
    keys = []
    with open(index_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rel = line.split()[0]
            if rel.endswith(".json"):
                continue
            keys.append(rel.replace("\\", "/"))
    return keys


def load_clip_features(path, spec, run_id, model, scene_id, index_path):
    if not path or not os.path.isfile(path) or not os.path.isfile(index_path):
        return None
    encoder = ENCODERS.get(model)
    if encoder is None:
        return None
    family, name, dim = encoder
    expected = {
        "scene_id": scene_id,
        "benchmark": spec.name,
        "run_id": run_id,
        "model": model,
        "encoder_family": family,
        "encoder_name": name,
        "feature_dim": dim,
        "index_sha256": sha256_file(index_path),
    }
    try:
        with np.load(path, allow_pickle=False) as doc:
            loaded = validate_clip_features(doc, expected)
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None
    if loaded["keys"] != prediction_index_keys(index_path):
        return None
    return loaded


def cosine_search(features, query_vec, top_k=SEARCH_TOP_K):
    query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    query_vec = query_vec / max(float(np.linalg.norm(query_vec)), 1e-6)
    scores = features @ query_vec
    order = np.argsort(-scores)
    k = min(int(top_k), int(order.size))
    picked = order[:k]
    return picked.astype(np.int32), scores[picked].astype(np.float32)


def clip_path_ready(path):
    return bool(path) and os.path.isfile(path)


def query_modes_for(model, has_features=False, has_snap=False, source_pred=False):
    if not source_pred:
        return [QUERY_OFF]
    if model in INSTANT_MODELS and has_features:
        return [QUERY_OFF, QUERY_SEARCH, QUERY_IMAGE]
    if model in NATIVE_LOOKUP_MODELS and has_snap:
        return [QUERY_OFF, QUERY_SEARCH]
    return [QUERY_OFF]


def openins_snap_root(run_id, scene_id):
    return os.path.join(OPENINS_SCRATCH, run_id, scene_id)


def openins_snap_scene_dir(run_id, scene_id):
    return os.path.join(openins_snap_root(run_id, scene_id), scene_id)


def openins_snap_complete(run_id, scene_id):
    scene_dir = openins_snap_scene_dir(run_id, scene_id)
    image_dir = os.path.join(scene_dir, "image")
    pose_dir = os.path.join(scene_dir, "pose")
    if not os.path.isdir(image_dir) or not os.path.isdir(pose_dir):
        return False
    images = [name for name in os.listdir(image_dir) if name.endswith(".png")]
    return bool(images)


def method_python(model):
    from predict.runner import MODEL_PYTHON
    return MODEL_PYTHON[model]


class QueryClient:
    def __init__(self):
        self.proc = None
        self.python = None
        self.family = None
        self.name = None
        self.device = None
        self._next_id = 0
        self._lock = threading.Lock()

    def close(self):
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def ensure(self, python, family, name, device="cuda:0"):
        if (self.proc is not None and self.proc.poll() is None
                and self.python == python and self.family == family and self.name == name
                and self.device == device):
            return
        self.close()
        script = os.path.abspath(__file__)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env.pop("SPELLBOOK_GPU_LEASE_FD", None)
        self.proc = subprocess.Popen(
            [python, script, "worker", "--family", family, "--name", name,
             "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True)
        self.python = python
        self.family = family
        self.name = name
        self.device = device

    def submit(self, payload):
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError("query worker is not running")
            self._next_id += 1
            request_id = self._next_id
            payload = dict(payload)
            payload["id"] = request_id
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            return request_id

    def poll(self):
        proc = self.proc
        if proc is None or proc.stdout is None:
            return None
        if proc.poll() is not None:
            return {"ok": False, "error": "query worker exited"}
        try:
            import select
            ready, _, _ = select.select([proc.stdout], [], [], 0)
            if not ready:
                return None
            line = proc.stdout.readline()
        except Exception:
            return None
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None


def _load_worker_encoder(family, name, device):
    if family == "openai_clip":
        import clip
        import torch
        model, preprocess = clip.load(name, device=device)
        model.eval()
        return {"family": family, "model": model, "preprocess": preprocess,
                "tokenize": clip.tokenize, "torch": torch, "device": device}
    if family == "open_clip":
        import open_clip
        import torch
        model, preprocess = open_clip.create_model_from_pretrained(name, device=device)
        tokenizer = open_clip.get_tokenizer(name)
        model.eval()
        return {"family": family, "model": model, "preprocess": preprocess,
                "tokenize": tokenizer, "torch": torch, "device": device}
    raise ValueError(f"unsupported encoder family {family!r}")


def _encode_texts(encoder, texts):
    torch = encoder["torch"]
    device = encoder["device"]
    with torch.no_grad():
        if encoder["family"] == "openai_clip":
            tokens = encoder["tokenize"](list(texts)).to(device)
            feats = encoder["model"].encode_text(tokens)
        else:
            tokens = encoder["tokenize"](list(texts)).to(device)
            feats = encoder["model"].encode_text(tokens)
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def _encode_image(encoder, path):
    from PIL import Image
    torch = encoder["torch"]
    device = encoder["device"]
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = encoder["preprocess"](image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = encoder["model"].encode_image(tensor).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)[0]


def _worker_features(cache, req):
    path = req["features_path"]
    index_path = req["index_path"]
    stamp = (path, sha256_file(index_path), os.path.getmtime(path))
    loaded = cache.get("features")
    if loaded is None or cache.get("stamp") != stamp:
        with np.load(path, allow_pickle=False) as doc:
            features = l2_normalize_rows(doc["features"])
            keys = [str(key) for key in doc["keys"].tolist()]
        cache["features"] = (features, keys)
        cache["stamp"] = stamp
        loaded = cache["features"]
    return loaded


def worker_main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("--family", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    encoder = _load_worker_encoder(args.family, args.name, args.device)
    cache = {}
    import sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        try:
            features, keys = _worker_features(cache, req)
            op = req.get("op")
            if op == "search":
                vec = _encode_texts(encoder, [req["text"]])[0]
                ranks, scores = cosine_search(features, vec, req.get("top_k", SEARCH_TOP_K))
                out = {
                    "ok": True, "id": req["id"], "op": op,
                    "ranks": ranks.tolist(), "scores": scores.tolist(),
                    "keys": [keys[i] for i in ranks],
                }
            elif op == "image":
                vec = _encode_image(encoder, req["path"])
                ranks, scores = cosine_search(features, vec, req.get("top_k", SEARCH_TOP_K))
                out = {
                    "ok": True, "id": req["id"], "op": op,
                    "ranks": ranks.tolist(), "scores": scores.tolist(),
                    "keys": [keys[i] for i in ranks],
                }
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception as exc:
            out = {"ok": False, "id": req.get("id"), "error": str(exc)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    import sys
    worker_main(sys.argv[1:])
