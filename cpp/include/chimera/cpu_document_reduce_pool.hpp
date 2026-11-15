#pragma once
#include <cstddef>
namespace Chimera {
// CPU-only leaf. Offsets may point at a document subrange; token offsets remain
// global into the Q-major matrix. Caller guarantees GPU completion before use.
void score_one_bit_documents_cpu_pool(const float* token_scores,
    const std::size_t* offsets, float* document_scores,
    std::size_t total_tokens, std::size_t num_documents);
}
