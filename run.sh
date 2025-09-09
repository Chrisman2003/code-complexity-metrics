#!/usr/bin/env bash
set -euo pipefail

# Where to place fake headers
FAKE_DIR="$PWD/fake_cuda"

echo "Creating fake headers in: $FAKE_DIR"
mkdir -p "$FAKE_DIR"

# Minimal safe host_defines.h that avoids the __noinline__ macro clash
cat > "$FAKE_DIR/host_defines.h" <<'EOF'
#ifndef FAKE_HOST_DEFINES_H
#define FAKE_HOST_DEFINES_H
/* Provide safe tokens so attribute lists in libstdc++ remain valid */
#define __host__
#define __device__
#define __global__
#define __noinline__ noinline
#define __forceinline__ always_inline
#endif
EOF

# Minimal stub for cuda_runtime.h: only declarations, no inline bodies
cat > "$FAKE_DIR/cuda_runtime.h" <<'EOF'
#ifndef FAKE_CUDA_RUNTIME_H
#define FAKE_CUDA_RUNTIME_H
#include <stddef.h>
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 0
#ifdef __cplusplus
extern "C" {
#endif
extern cudaError_t cudaMallocHost(void **ptr, size_t size);
extern cudaError_t cudaMalloc(void **devPtr, size_t size);
extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind);
extern cudaError_t cudaFree(void *devPtr);
#ifdef __cplusplus
}
#endif
#endif
EOF

# Export CPATH so clang/gcc will search fake headers before standard system headers.
# CPATH is used by compilers as additional include directories.
export CPATH="$FAKE_DIR:${CPATH:-}"

echo "Running code-metrics with CPATH=$CPATH"
# run your tool (adjust path if needed)
code-metrics samples/online_cuda.cu
