#include "chimera/cpu_document_reduce_pool.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Chimera {
namespace {
inline float maximum_contiguous_tokens(const float* row, size_t start, size_t end) {
    float maximum = -FLT_MAX;
    size_t token = start;
#if defined(__aarch64__)
    // Four independent 4-token chains avoid one long SIMD max dependency.
    float32x4_t m0 = vdupq_n_f32(-FLT_MAX);
    float32x4_t m1 = m0, m2 = m0, m3 = m0;
    for (; end - token >= 16; token += 16) {
        m0 = vmaxnmq_f32(m0, vld1q_f32(row + token));
        m1 = vmaxnmq_f32(m1, vld1q_f32(row + token + 4));
        m2 = vmaxnmq_f32(m2, vld1q_f32(row + token + 8));
        m3 = vmaxnmq_f32(m3, vld1q_f32(row + token + 12));
    }
    m0 = vmaxnmq_f32(vmaxnmq_f32(m0, m1), vmaxnmq_f32(m2, m3));
    for (; end - token >= 4; token += 4)
        m0 = vmaxnmq_f32(m0, vld1q_f32(row + token));
    maximum = vmaxnmvq_f32(m0);
#endif
    for (; token < end; ++token) maximum = std::fmax(maximum, row[token]);
    return maximum;
}
} // namespace

// Shared worker-pool leaf; ATS producer/consumer handoff is the caller's chunk event.
// CPU-only optimization: sweep contiguous query rows over a document tile,
// then use exactly the original GPU query-sum tree. No per-query heap storage.
void score_one_bit_documents_cpu_pool(
    const float* token_scores, const size_t* offsets, float* document_scores,
    size_t total_tokens, size_t num_documents) {
    constexpr size_t kDocumentTile = 32;
    alignas(64) float sums[kDocumentTile][32]; // 4 KiB per worker task; no allocation.
    for (size_t first = 0; first < num_documents; first += kDocumentTile) {
        const size_t count = std::min(kDocumentTile, num_documents - first);
        // Each row's token accesses advance monotonically across this tile's
        // contiguous documents; hardware prefetch does not juggle 32 rows per
        // document as in the original doc-outer traversal.
        for (size_t q = 0; q < 32; ++q) {
            const float* row = token_scores + q * total_tokens;
            for (size_t local = 0; local < count; ++local) {
                const size_t doc = first + local;
                float value = maximum_contiguous_tokens(row, offsets[doc], offsets[doc + 1]);
                value = 0.0f + value;
                value = value + 0.0f; // original stride64
                value = value + 0.0f; // original stride32
                sums[local][q] = value;
            }
        }
        for (size_t local = 0; local < count; ++local) {
            // Preserve all FP32 addition pairs; do not replace by vaddvq.
            for (size_t stride = 16; stride != 0; stride >>= 1)
                for (size_t q = 0; q < stride; ++q)
                    sums[local][q] = sums[local][q] + sums[local][q + stride];
            document_scores[first + local] = sums[local][0];
        }
    }
}
} // namespace Chimera
