"""Engine adapters (one engine per file).

Contract for every adapter module:

    GPU_POLICY = "managed" | "cpu" | "blocked"
        "managed"      GPU-backed; run.py acquires a shared settings-pool lease
                       (gpu_pool, never GPU 0) and passes the physical index + fd.
        "cpu"          CPU-only pipeline (Open3D legacy TSDF). No lease.
        "blocked"      Non-runnable (ZED: its SDK default-device path would use
                       user-reserved physical GPU 0, see #18). preflight() returns
                       the reason; no lease is ever taken.
    SERIAL = bool      At most one task of this engine at a time (batch-level; license,
                       container, or tracking constraints).
    preflight() -> None | reason string
                       Runtime presence check used by the batch runner to skip an
                       engine with a clear reason instead of failing the batch.
    reconstruct(work, root, gpu=None, lease_fd=None)
                       -> (mesh_native_path, poses (M,4,4), keep, convention)
    gpu_check(gpu=None, lease_fd=None, hold_seconds=5) -> dict
                       Native-runtime distribution check (no results). Managed
                       adapters require gpu + lease_fd; cpu adapters require None.

`gpu` is the automatic physical index from the shared allocator for "managed" engines
(always in settings gpu_pool, never 0); None for "cpu" and "blocked". `lease_fd` must
be forwarded with subprocess pass_fds to the blocking GPU child (Metashape python,
podman/docker run) so the lease outlives the adapter if the parent dies; the child
must keep the descriptor open for the whole workload. Adapters must not import one
another; shared integration code lives in spellbook/reconstruct/tsdf.py.
"""
