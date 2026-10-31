# Headless Linux BundleFusion: public FangGet/BundleFusion_Ubuntu_Pangolin port (the fork named
# by ce25997; the original image and /data/zed-bundlefusion are lost) + headless.patch (ce25997
# fork patch list + the bundle_fusion_headless binary the adapter calls).
# [inferred] the whole headless driver: finishBundleFusion order, pose rule (optimized if finite, else
# integrated, else drop) and exit path are rebuilt from the docs; the original sources are lost.
# Build: docker build -t bundlefusion:latest spellbook/reconstruct/containers/bundlefusion
# Run:   only through spellbook/reconstruct/containers helpers (snap docker has no --gpus).
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake make g++ libeigen3-dev libfreeimage-dev libncurses-dev libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

ARG FORK_SHA=f350ab06a3bdf47c4f1592d4518eac371218f327
WORKDIR /app
RUN git init -q . && \
    git fetch -q --depth 1 https://github.com/FangGet/BundleFusion_Ubuntu_Pangolin "$FORK_SHA" && \
    git checkout -q FETCH_HEAD

COPY headless.patch /tmp/headless.patch
RUN git apply --whitespace=nowarn /tmp/headless.patch && \
    mkdir build && cd build && \
    cmake -DVISUALIZATION=OFF .. && \
    make -j"$(nproc)" bundle_fusion_headless && \
    test -x bundle_fusion_headless
