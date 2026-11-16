#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "chimera/config.cuh"

namespace Chimera { namespace split_token_score_detail {
static_assert(Q_DOCLEN == 32 && PADDED_DIM == 128 && NUM_CHUNKS == 32,
              "Split query-group token scoring requires Q32/D128");
// The helper and scoring body come from the qualified token microkernel.
// Only the CTA index is flattened from dim3(token_blocks,4); each token's
// c0..31 accumulation and q-major output are unchanged.
template<int QT>
__device__ __forceinline__ void add_lookup_rows(
    const float* lut, uint64_t code0, uint64_t code1, float (&ip)[QT]) {
    #pragma unroll 1
    for (int c = 0; c < 32; ++c) {
        const uint64_t word = c < 16 ? code0 : code1;
        const int nibble = int((word >> ((c & 15) * 4)) & 15);
        const int address = c * 16 + nibble;
        #pragma unroll
        for (int q = 0; q < QT; ++q)
            ip[q] = __fadd_rn(ip[q], lut[q * 512 + address]);
    }
}

template<int QT>
__global__ void token_split_query_group(
    const float* __restrict__ lut, const char* __restrict__ codes,
    const float* __restrict__ factors, const float* __restrict__ cb,
    const uint32_t* __restrict__ positions, float* __restrict__ output,
    size_t total_tokens, size_t batch_tokens) {
    static_assert(QT == 4 || QT == 8, "Q32 groups");
    constexpr unsigned groups = 32 / QT;
    // Flatten the requested (token_blocks,4) grid to preserve the measured
    // one-dimensional variant's interleaved logical query-group ownership.
    const size_t linear_block = size_t(blockIdx.x) + size_t(gridDim.x) * blockIdx.y;
    const unsigned group = unsigned(linear_block % groups);
    const size_t token = (linear_block / groups) * blockDim.x + threadIdx.x;
    extern __shared__ float smem[];
    float* local_cb = smem + QT * 512;
    uint64_t code0 = 0, code1 = 0;
    float factor = 0.0f;
    const bool valid = token < batch_tokens;
    if (valid) {
        const uint32_t pos = positions[token];
        const uint64_t* words = reinterpret_cast<const uint64_t*>(codes + size_t(pos) * 16);
        code0 = words[0]; code1 = words[1]; factor = factors[pos];
    }
    for (int i = threadIdx.x; i < QT * 512; i += blockDim.x)
        smem[i] = lut[group * QT * 512 + i];
    if (threadIdx.x < QT) local_cb[threadIdx.x] = cb[group * QT + threadIdx.x];
    __syncthreads();
    if (valid) {
        float ip[QT] = {};
        add_lookup_rows<QT>(smem, code0, code1, ip);
        #pragma unroll
        for (int q = 0; q < QT; ++q)
            output[size_t(group * QT + q) * total_tokens + token] =
                __fmul_rn(__fsub_rn(ip[q], local_cb[q]), factor);
    }
}
} }
