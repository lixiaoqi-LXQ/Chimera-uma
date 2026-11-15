#pragma once

#include "chimera/config.cuh"
#include "rabitqlib/utils/space.hpp"
#include "cpu_gemv_neon.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace Chimera {

// Immutable borrowed inputs. The submitter joins every task before any query
// storage can be reused. Each task writes only its own output candidate.
struct FullbitCpuInputs {
    const char* codes;
    const float* factors;
    const int* doc_ptrs;
    const float* queries_flat;
    const float* cbex_sumq;
    size_t code_stride;
    void (*unpack)(const uint8_t*, float*, size_t);
};

// Same FP32 operation/reduction order as compute_cpu_full_bit_scores. Keep the
// original CPU backend as the qualification oracle and default implementation.
// This leaf has no OpenMP region and is safe for a shared persistent task pool.
inline std::pair<float, int> score_fullbit_document_cpu(
    const FullbitCpuInputs& input, int doc_id) {
    const size_t doc_start = input.doc_ptrs[doc_id];
    const size_t n_tok = input.doc_ptrs[doc_id + 1] - doc_start;

    alignas(64) float decoded[PADDED_DIM];
    alignas(64) float full_ip[Q_DOCLEN];
    alignas(64) float max_ts[Q_DOCLEN];
    __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (size_t j = 0; j < Q_DOCLEN; j += 16)
        _mm512_store_ps(&max_ts[j], neg_inf);

    for (size_t t = 0; t < n_tok; t++) {
        const size_t tid = doc_start + t;
        input.unpack(reinterpret_cast<const uint8_t*>(
            &input.codes[tid * input.code_stride]), decoded, PADDED_DIM);
        if (t + 1 < n_tok) {
            __builtin_prefetch(&input.codes[(tid + 1) * input.code_stride], 0, 3);
            __builtin_prefetch(&input.factors[tid + 1], 0, 1);
        }
#if defined(__aarch64__)
        rabitqlib::gemv_batch4_neon(
#else
        rabitqlib::gemv_batch8_avx512(
#endif
            input.queries_flat, decoded, full_ip, Q_DOCLEN, PADDED_DIM);

        const float full_bit_factor = input.factors[tid];
        const __m512 full_bit_factor_vector = _mm512_set1_ps(full_bit_factor);
        for (size_t j = 0; j < Q_DOCLEN; j += 16) {
            __m512 full_ip_v = _mm512_load_ps(&full_ip[j]);
            __m512 cbex = _mm512_load_ps(&input.cbex_sumq[j]);
            __m512 combined = _mm512_mul_ps(
                _mm512_sub_ps(full_ip_v, cbex), full_bit_factor_vector);
            _mm512_store_ps(&max_ts[j],
                _mm512_max_ps(_mm512_load_ps(&max_ts[j]), combined));
        }
    }

    __m512 sum = _mm512_load_ps(&max_ts[0]);
    for (size_t j = 16; j < Q_DOCLEN; j += 16)
        sum = _mm512_add_ps(sum, _mm512_load_ps(&max_ts[j]));
    return {_mm512_reduce_add_ps(sum), doc_id};
}

} // namespace Chimera
