#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <functional>
#include <limits>
namespace Chimera {
// Matches CUB float radix ordering, including equivalent signed zeros and
// bitwise ordered NaNs. Original row breaks ties: stable CUB preserves it.
inline uint32_t cpu_refine_radix_key(float score) {
    uint32_t bits;
    std::memcpy(&bits, &score, sizeof(bits));
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits ^ ((bits & 0x80000000U) ? 0xffffffffU : 0x80000000U);
}
// order[] holds packed unsigned keys, not row indices. High 32 bits sort
// scores descending; complemented low 32 bits preserve ascending stable rows.
inline size_t cpu_refine_row(size_t packed) {
    static_assert(sizeof(size_t) == sizeof(uint64_t), "Packed CPU top-k requires 64-bit size_t");
    return size_t(0xffffffffU - static_cast<uint32_t>(packed));
}
inline void cpu_refine_topk(const float* scores, const int* ids, size_t n,
                            size_t k, size_t* order, int* selected_ids) {
    static_assert(sizeof(size_t) == sizeof(uint64_t), "Packed CPU top-k requires 64-bit size_t");
    assert(n <= std::numeric_limits<uint32_t>::max());
    assert(k <= n);
    // One sequential score read/key transform per row; selection comparisons
    // operate only on compact words and never chase scores at random indices.
    for (size_t row = 0; row < n; ++row) {
        order[row] = static_cast<size_t>(
            (uint64_t(cpu_refine_radix_key(scores[row])) << 32) |
            uint64_t(0xffffffffU - static_cast<uint32_t>(row)));
    }
    const auto before = std::greater<size_t>{};
    if (k < n) std::nth_element(order, order + k, order + n, before);
    std::sort(order, order + k, before);
    for (size_t i = 0; i < k; ++i) selected_ids[i] = ids[cpu_refine_row(order[i])];
}
}
