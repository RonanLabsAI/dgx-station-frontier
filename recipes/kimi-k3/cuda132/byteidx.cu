// Copyright 2026 RonanLabs. SPDX-License-Identifier: MIT
// The kernel body reproduces the indexing pattern of vec_dot_iq3_s_q8_1 and get_int_b2 from llama.cpp
// ggml/src/ggml-cuda/vecdotq.cuh (MIT, Copyright (c) 2023-2026 The ggml authors); this file is MIT for that reason.
//
// Standalone reproducer for the nvcc 13.2.51 / 13.2.78 (CUDA 13.2.0 / 13.2.1) NVVM miscompile behind garbage output of
// llama.cpp IQ1_S / IQ2_S / IQ3_S kernels. At -O2 and above cicc drops the "& 0xFF" when a byte is read out of a
// packed 32-bit word through a uint8_t* cast and used as a table index, so the index can reach 0xFFFF.
//   nvcc -O3 -arch=sm_103 byteidx.cu -o byteidx && ./byteidx     (any arch shows it; sm_120 PTX is identical)
//   nvcc 13.0.88 / 13.1.115 / 13.2.86 / 13.3.73 / 13.4.92: mismatches=0/4096    PASS
//   nvcc 13.2.51 / 13.2.78:                                mismatches=4096/4096 FAIL
// One-line detector on the PTX: nvcc -O3 -arch=sm_103 -ptx byteidx.cu -o b.ptx && grep -c and.b16 b.ptx   (4 = good, 0 = bad)
// Upstream: llama.cpp issues #21255 (closed) and #28581 (open), PR #28784 (__byte_perm workaround, closed unmerged).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define N 4096
__device__ uint32_t grid_d[512];
static uint32_t grid_h[512];

static __device__ __forceinline__ int get_int_b2(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32 = x16[2*i32 + 0] << 0;
    x32    |= x16[2*i32 + 1] << 16;
    return x32;
}

__global__ void k_byteidx(const uint16_t * __restrict__ qs_all, const uint8_t * __restrict__ qh_all, int * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int2      qs_packed = make_int2(get_int_b2(qs_all + 4*i, 0), get_int_b2(qs_all + 4*i, 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = qh_all[i];
    unsigned acc = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const unsigned a = grid_d[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const unsigned b = grid_d[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        acc = acc * 31u + a;
        acc = acc * 31u + b;
    }
    out[i] = (int) acc;
}

int main() {
    srand(1234);
    for (int j = 0; j < 512; ++j) grid_h[j] = (uint32_t) rand() * 2654435761u + j;
    uint16_t * qs = (uint16_t *) malloc(N * 4 * sizeof(uint16_t));
    uint8_t  * qh = (uint8_t  *) malloc(N);
    for (int i = 0; i < N * 4; ++i) qs[i] = (uint16_t) rand();
    for (int i = 0; i < N; ++i)     qh[i] = (uint8_t) rand();

    // CPU reference (same byte order: little-endian int2 = x then y)
    int * ref = (int *) malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        uint8_t b[8];
        memcpy(b, qs + 4*i, 8);
        unsigned acc = 0;
        for (int l0 = 0; l0 < 8; l0 += 2) {
            acc = acc * 31u + grid_h[b[l0 + 0] | ((qh[i] << (8 - l0)) & 0x100)];
            acc = acc * 31u + grid_h[b[l0 + 1] | ((qh[i] << (7 - l0)) & 0x100)];
        }
        ref[i] = (int) acc;
    }

    uint16_t * d_qs; uint8_t * d_qh; int * d_out;
    cudaMalloc(&d_qs, N * 4 * sizeof(uint16_t)); cudaMalloc(&d_qh, N); cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_qs, qs, N * 4 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qh, qh, N, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(grid_d, grid_h, sizeof(grid_h));
    k_byteidx<<<(N + 127) / 128, 128>>>(d_qs, d_qh, d_out);
    cudaError_t e = cudaDeviceSynchronize();
    int * out = (int *) malloc(N * sizeof(int));
    cudaMemcpy(out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < N; ++i) bad += out[i] != ref[i];
    int rtv = 0; cudaRuntimeGetVersion(&rtv);
    printf("runtime=%d err=%s mismatches=%d/%d -> %s\n", rtv, cudaGetErrorString(e), bad, N, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
