# Reconstruction engines. Each module exposes:
#   reconstruct(work, root, gpu=None) -> (mesh_native_path, poses (M,4,4), keep, convention)
# poses are in the engine's native frame, one per kept frame index (original extract indices).
