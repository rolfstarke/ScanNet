"""Docker helpers shared by the container engines (rtabmap, isaac, bundlefusion)."""
import os
import subprocess

_HOSTFS = "/var/lib/snapd/hostfs"
GPU_ARGS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_args.sh")


def host_path(path):
    """Bind-mount source for a host path. Snap docker mounts plain /data paths as empty dirs."""
    return _HOSTFS + os.path.abspath(path)


def mount(path, target=None, read_only=False):
    """`docker run` args binding host `path` at `target` (default: the same path)."""
    spec = f"{host_path(path)}:{target or os.path.abspath(path)}"
    return ["-v", spec + (":ro" if read_only else "")]


def gpu_args(gpu):
    """`docker run` args exposing the NVIDIA driver and pinning the leased GPU."""
    return subprocess.run([GPU_ARGS_SCRIPT, str(gpu)], check=True, capture_output=True,
                          text=True).stdout.split("\n")[:-1]


def user_args():
    """Run as the invoking user so bind-mounted outputs are not root-owned."""
    return ["--user", f"{os.getuid()}:{os.getgid()}"]
