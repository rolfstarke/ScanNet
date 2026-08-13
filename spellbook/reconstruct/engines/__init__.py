"""Engine adapters (one engine per file).

Contract for every adapter module:

    GPU_POLICY = "zed-default" | "cpu" | "managed"
        "zed-default"  ZED SDK positional tracking: runs on the default CUDA device
                       (SDK 5.4 bug #18: sdk_gpu_id != 0 -> constant poses). No lease.
        "cpu"          CPU-only pipeline (Open3D legacy TSDF). No lease.
        "managed"      GPU-backed; run.py acquires a shared settings-pool lease
                       (gpu_pool, never GPU 0) and passes the physical index + fd.
    SERIAL = bool      At most one task of this engine at a time (batch-level; license,
                       container, or tracking constraints).
    preflight() -> None | reason string
                       Runtime presence check used by the batch runner to skip an
                       engine with a clear reason instead of failing the batch.
    reconstruct(work, root, gpu=None, lease_fd=None)
                       -> (mesh_native_path, poses (M,4,4), keep, convention)

`gpu` is the automatic physical index from the shared allocator for "managed" engines
(always in settings gpu_pool, never 0); None for "zed-default"/"cpu". `lease_fd` must
be forwarded with subprocess pass_fds to the blocking GPU child (Metashape python,
podman/docker run) so the lease outlives the adapter if the parent dies; the child
must keep the descriptor open for the whole workload. Adapters must not import one
another; shared integration code lives in spellbook/reconstruct/tsdf.py.
"""
