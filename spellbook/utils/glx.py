"""Select the NVIDIA GLX provider for the Open3D viewer on remote displays (#55).

xrdp sessions default to llvmpipe, which renders dense scenes at under 1 FPS. Call this before
importing Open3D: it switches only when the default renderer is proven software and a short
trial proves NVIDIA direct rendering. An explicit __GLX_VENDOR_LIBRARY_NAME is left alone.
"""
import os
import subprocess

_VENDOR_ENV = "__GLX_VENDOR_LIBRARY_NAME"
_TRIAL_S = 3


def _renderer(env):
    """(renderer string, direct rendering) from glxinfo, or (None, False) if it can't run."""
    try:
        out = subprocess.run(["glxinfo", "-B"], env=env, capture_output=True, text=True,
                             timeout=_TRIAL_S).stdout
    except (OSError, subprocess.SubprocessError):
        return None, False
    renderer, direct = None, False
    for line in out.splitlines():
        if line.startswith("OpenGL renderer string:"):
            renderer = line.split(":", 1)[1].strip()
        elif line.startswith("direct rendering:"):
            direct = line.split(":", 1)[1].strip() == "Yes"
    return renderer, direct


def _is_software(renderer):
    return renderer is not None and any(
        name in renderer.lower() for name in ("llvmpipe", "softpipe", "swrast"))


def ensure_nvidia_glx():
    """Returns True when the NVIDIA provider was selected for this process."""
    if os.environ.get(_VENDOR_ENV):
        return False
    renderer, _ = _renderer(os.environ)
    if not _is_software(renderer):
        return False
    trial = dict(os.environ, **{_VENDOR_ENV: "nvidia"})
    renderer, direct = _renderer(trial)
    if not direct or renderer is None or "nvidia" not in renderer.lower():
        return False
    os.environ[_VENDOR_ENV] = "nvidia"
    print(f"[glx] using {renderer}")
    return True
