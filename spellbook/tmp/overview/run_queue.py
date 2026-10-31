#!/usr/bin/env python3
"""Run the 24 baseline/best runs one at a time, in overview order, and record results in state.json.

Unattended: before each run it uses only GPUs with free memory; a run whose GPUs stay busy is
deferred to a later pass; a run or scoring step past its time limit is killed; a failure retries
once and then the queue moves on. Restartable: runs already marked done or failed are skipped.
Run from anywhere with the spellbook env active, from a shell that has the docker group.
"""
import glob, json, os, shutil, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SPELLBOOK = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(SPELLBOOK)
ROOT = "/data/scannetdata"
PARAMS = os.path.join(SPELLBOOK, "tmp", "recovery", "params")
STATE = os.path.join(HERE, "state.json")
LOGS = os.path.join(HERE, "logs")
RESERVED = os.path.join(HERE, "reserved_gpus")
ENGINE_DIGIT = {"zed": 0, "metashape": 1, "rtabmap": 2, "isaac": 3, "open3d": 4, "bundlefusion": 5}
GPU_POOL = [1, 2, 3, 4]
GPU_BUSY_MIB = 2048  # used memory above this means someone else is on the GPU
GPU_WAIT_S = 1800  # first pass: wait this long for GPUs, then defer the run
DEFERRED_WAIT_S = 6 * 3600  # later pass, nothing else left to do
RECON_LIMIT_S = {"metashape": 3 * 3600}  # whole run; other engines default below
RECON_DEFAULT_LIMIT_S = 1800
PRED_LIMIT_GPU_S = 4 * 24 * 3600  # GPU-seconds, divided by the GPUs a prediction run gets
SCORE_LIMIT_S = 1200
CONTAINER_IMAGES = ("localhost/zed-rtabmap:jazzy", "zed-isaac-nvblox:spellbook", "bundlefusion:noflip")


def recon(engine, scene, issue, env=None, best=False):
    return dict(kind="recon", engine=engine, scene=scene, issue=issue, best=best,
                env=dict(env or {}, **({} if best else {"SPELLBOOK_TARGET_RUN0": "1"})))


def pred(model, kind, issue, params=None):
    return dict(kind="pred", model=model, run_id=f"repro-{kind}-{model}", issue=issue, params=params)


QUEUE = {
    1: recon("zed", 9004, 47, {"SPELLBOOK_ZED_PRESET": "native-default"}),
    2: recon("zed", 9009, 47, {"SPELLBOOK_ZED_PRESET": "native-default"}),
    3: recon("zed", 9004, 39, {"SPELLBOOK_ZED_PRESET": "s2b8-fixedcalib"}, best=True),
    4: recon("metashape", 9004, 53, {"SPELLBOOK_METASHAPE_PRESET": "native-default"}),
    5: recon("metashape", 9004, 40, {"SPELLBOOK_METASHAPE_PRESET": "b5-kf350"}, best=True),
    6: recon("rtabmap", 9004, 49, {"SPELLBOOK_RTABMAP_PROFILE": "assisted-hybrid", "SPELLBOOK_RTABMAP_STRIDE": "90"}),
    7: recon("rtabmap", 9009, 49, {"SPELLBOOK_RTABMAP_PROFILE": "assisted-hybrid", "SPELLBOOK_RTABMAP_STRIDE": "90"}),
    8: recon("isaac", 9004, 48, {"SPELLBOOK_ISAAC_PRESET": "official_default"}),
    9: recon("isaac", 9009, 48, {"SPELLBOOK_ISAAC_PRESET": "official_default"}),
    10: recon("isaac", 9004, 42, {"SPELLBOOK_ISAAC_PRESET": "conf100_best"}, best=True),
    11: recon("open3d", 9004, 51, {"SPELLBOOK_OPEN3D_PRESET": "tensor"}),
    12: recon("open3d", 9009, 51, {"SPELLBOOK_OPEN3D_PRESET": "tensor"}),
    13: recon("bundlefusion", 9004, 52),
    14: recon("bundlefusion", 9009, 52),
    15: pred("openyolo3d", "baseline", 44),
    16: pred("openyolo3d", "best", 35, "openyolo3d-h1.json"),
    17: pred("openmask3d", "baseline", 46),
    18: pred("open3dis", "baseline", 60),
    19: pred("openins3d", "baseline", 45, "openins3d-baseline.json"),
    20: pred("openins3d", "best", 34, "openins3d-best.json"),
    21: pred("mosaic3d", "baseline", 43),
    22: pred("mosaic3d", "best", 33, "mosaic3d-best.json"),
    23: pred("sam3dis", "baseline", 69),
    24: pred("sam3dis", "best", 69, "sam3dis-best.json"),
}


def load_state():
    return json.load(open(STATE)) if os.path.isfile(STATE) else {}


def set_state(order, **fields):
    state = load_state()
    state.setdefault(str(order), {}).update(fields)
    tmp = STATE + ".tmp"
    json.dump(state, open(tmp, "w"), indent=0)
    os.replace(tmp, STATE)
    subprocess.run([sys.executable, "build.py", "s"], cwd=HERE, stdout=subprocess.DEVNULL)


def command(spec):
    main = [sys.executable, os.path.join(SPELLBOOK, "main.py")]
    if spec["kind"] == "recon":
        return main + ["--engine", spec["engine"], "--scene", str(spec["scene"]), "--issue", str(spec["issue"]),
                       "--offline-score", "--foreground"]
    cmd = main + ["--predict", "--benchmark", "ScanNet200", "--models", spec["model"], "--run-id", spec["run_id"],
                  "--issue", str(spec["issue"]), "--foreground"]
    if spec["params"]:
        cmd += ["--run-parameters", os.path.join(PARAMS, spec["params"])]
    return cmd


def clean_previous(spec):
    """Drop attempts of this exact run, so a redo replaces them instead of filling another slot."""
    if spec["kind"] == "pred":
        for base in ("predictions", "derived/evaluations"):
            path = os.path.join(ROOT, base, "ScanNet200", spec["run_id"])
            if os.path.isdir(path):
                shutil.rmtree(path)
    elif spec["best"]:
        digit = ENGINE_DIGIT[spec["engine"]]
        for scan in glob.glob(os.path.join(ROOT, "scans", f"scene{spec['scene']}_{digit}[1-9]")):
            shutil.rmtree(scan)


def recon_result(spec):
    digit = ENGINE_DIGIT[spec["engine"]]
    scans = glob.glob(os.path.join(ROOT, "scans", f"scene{spec['scene']}_{digit}?", "recon", "run.json"))
    path = max(scans, key=os.path.getmtime)
    doc = json.load(open(path))
    scan = os.path.dirname(os.path.dirname(path))
    fields = dict(runtime_s=round(doc.get("solver_elapsed_s") or doc.get("elapsed_s") or 0, 1), scan=os.path.basename(scan))
    if spec["scene"] == 9004:
        subprocess.run([sys.executable, "-m", "spellbook.reconstruct.geometry_score", "--scan-dir", scan],
                       cwd=REPO, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=SCORE_LIMIT_S)
        import yaml
        score = yaml.safe_load(open(os.path.join(scan, "recon", "geometry_score.yaml")))
        fields["result"] = score["mean_bidirectional_distance_cm"]
    return fields


def pred_result(spec, wall):
    sys.path.insert(0, SPELLBOOK)
    from evaluation.runs import read_evaluator_csv
    path = os.path.join(ROOT, "derived", "evaluations", "ScanNet200", spec["run_id"], spec["model"] + ".csv")
    metrics = read_evaluator_csv(path)
    return dict(result=round(metrics["ap"], 4), ap50=round(metrics["ap50"], 4), ap25=round(metrics["ap25"], 4),
                runtime_s=round(wall))


def free_gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    used = {int(a): int(b) for a, b in (line.split(",") for line in out.strip().splitlines())}
    reserved = set()  # GPUs held back for debugging: one index per line, read before every run
    if os.path.isfile(RESERVED):
        reserved = {int(x) for x in open(RESERVED).read().split()}
    return [g for g in GPU_POOL if g not in reserved and used.get(g, GPU_BUSY_MIB) < GPU_BUSY_MIB]


def gpus_for(spec, deadline):
    """Free GPUs this run can use, waiting until `deadline`; None if they stay busy."""
    while True:
        free = free_gpus()
        if spec.get("engine") == "bundlefusion":  # the adapter pins this pair
            if {1, 2} <= set(free):
                return [1, 2]
        elif free:
            return free if spec["kind"] == "pred" else free[:1]
        if time.time() >= deadline:
            return None
        time.sleep(60)


def time_limit(spec, gpus):
    if spec["kind"] == "pred":
        return PRED_LIMIT_GPU_S / len(gpus)
    return RECON_LIMIT_S.get(spec["engine"], RECON_DEFAULT_LIMIT_S)


def run_limited(cmd, env, log_path, limit):
    """Return the exit code, or None after killing the whole process group at the limit."""
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    for image in CONTAINER_IMAGES:  # containers outlive the killed process group
        ids = subprocess.run(["docker", "ps", "-q", "--filter", f"ancestor={image}"],
                             capture_output=True, text=True).stdout.split()
        if ids:
            subprocess.run(["docker", "kill"] + ids, capture_output=True)
    return None


def run_one(order, spec, wait_s):
    """Returns False when the run was deferred because its GPUs stayed busy."""
    for attempt in (1, 2):
        gpus = gpus_for(spec, time.time() + wait_s)
        if gpus is None:
            print(f"[queue] {order} deferred: GPUs busy ({time.strftime('%H:%M:%S')})", flush=True)
            set_state(order, state="queued")
            return False
        set_state(order, state="running")
        print(f"[queue] {order} start {time.strftime('%H:%M:%S')} attempt {attempt} gpus {gpus}", flush=True)
        env = dict(os.environ, **spec.get("env", {}), SPELLBOOK_GPU_POOL=",".join(map(str, gpus)))
        if attempt == 1:  # a retry resumes: the runner skips scenes that already finished
            clean_previous(spec)
        start = time.time()
        rc = run_limited(command(spec), env, os.path.join(LOGS, f"{order:02d}-a{attempt}.log"), time_limit(spec, gpus))
        wall = time.time() - start
        if rc != 0:
            why = "time limit" if rc is None else f"rc={rc}"
            print(f"[queue] {order} attempt {attempt} failed: {why} ({wall:.0f}s)", flush=True)
            if attempt == 2 or rc in (None, 124):  # both repeat deterministically: move on
                set_state(order, state="failed", runtime_s=round(wall), reason=why)
                return True
            continue
        try:
            fields = recon_result(spec) if spec["kind"] == "recon" else pred_result(spec, wall)
            set_state(order, state="done", **fields)
            print(f"[queue] {order} done {fields}", flush=True)
        except Exception as exc:
            # The run itself finished: keep its output for diagnosis instead of retrying over it.
            print(f"[queue] {order} failed in scoring: {str(exc)[-300:]} ({wall:.0f}s)", flush=True)
            set_state(order, state="failed", runtime_s=round(wall), reason="scoring")
        return True
    return True


def main():
    os.makedirs(LOGS, exist_ok=True)
    only = {int(a) for a in sys.argv[1:]}
    pending = [o for o in QUEUE if (not only or o in only)
               and load_state().get(str(o), {}).get("state") not in ("done", "failed")]
    for wait_s in (GPU_WAIT_S, DEFERRED_WAIT_S):
        deferred = []
        for order in pending:
            try:
                if not run_one(order, QUEUE[order], wait_s):
                    deferred.append(order)
            except Exception as exc:  # nothing may stop the queue
                print(f"[queue] {order} failed: queue error {exc}", flush=True)
                set_state(order, state="failed", reason="queue error")
        pending = deferred
    for order in pending:
        set_state(order, state="blocked", reason="GPUs stayed busy")
    print(f"[queue] finished; blocked: {pending}", flush=True)


if __name__ == "__main__":
    main()
