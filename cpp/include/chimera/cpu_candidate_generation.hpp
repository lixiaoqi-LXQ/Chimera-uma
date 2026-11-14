#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace Chimera {
// Integer-only counterpart of mark/count/scan/materialize. Each cluster's
// virtual token permutation is bijective, so contiguous traversal sets the
// identical bitmap. IDs are emitted in increasing order, including ties.
inline uint32_t cpu_generate_candidates(
    const uint32_t* labels, size_t label_count, const size_t* cluster_pos,
    size_t clusters, const int* clustered_doc_ids, size_t docs,
    uint32_t* bitmap, size_t buckets, uint32_t* offsets,
    int* ids, size_t capacity) {
    std::memset(bitmap, 0, buckets * sizeof(uint32_t));
    for (size_t j = 0; j < label_count; ++j) {
        const uint32_t c = labels[j];
        if (c >= clusters) continue;
        for (size_t t = cluster_pos[c]; t < cluster_pos[c + 1]; ++t) {
            const int d = clustered_doc_ids[t];
            if (static_cast<size_t>(d) < docs)
                bitmap[size_t(d) / 32] |= uint32_t(1) << (d & 31);
        }
    }
    uint32_t count = 0;
    for (size_t b = 0; b < buckets; ++b) {
        if ((b & 7) == 0) offsets[b / 8] = count;
        count += static_cast<uint32_t>(__builtin_popcount(bitmap[b]));
    }
    offsets[buckets / 8] = count;  // buckets padded to multiple of eight.
    if (count > capacity) throw std::runtime_error("CPU CG exceeded workspace candidate bound");
    size_t out = 0;
    for (size_t b = 0; b < buckets; ++b) {
        uint32_t bits = bitmap[b];
        while (bits) {
            const unsigned bit = static_cast<unsigned>(__builtin_ctz(bits));
            ids[out++] = static_cast<int>(32 * b + bit);
            bits &= bits - 1;
        }
    }
    return count;
}
}
