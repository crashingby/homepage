---
title: CUDA GEMM Optimization Notes
date: 2026-06-11
tags: [CUDA, GEMM, GPU]
summary: A first blog draft for recording GEMM optimization experiments from naive kernels to tiled implementations.
---

# CUDA GEMM Optimization Notes

This is a placeholder post. You can replace it with your own notes, benchmark results, and code snippets.

## Roadmap

- Naive global-memory implementation
- Shared-memory tiling
- Register tiling
- Tensor Core path
- CuTe, CUTLASS, and Triton comparisons

```cpp
// Example snippet.
__global__ void gemm_kernel(const float* a, const float* b, float* c) {
    // Write the real kernel here.
}
```
