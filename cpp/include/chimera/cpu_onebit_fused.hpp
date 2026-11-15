#pragma once
#include <array>
#include <cstdint>
namespace Chimera {
// Full-query transposed LUT; CPU-only implementation keeps NEON out of NVCC.
struct alignas(64) CpuOnebitLut { std::array<float,32*16*32> values; };
void build_cpu_onebit_lut(const float* rotated_queries, CpuOnebitLut& lut);
float score_cpu_onebit_document(const uint8_t* codes, const float* factors,
    const uint32_t* inverse, const int* doc_ptrs, int doc,
    const CpuOnebitLut& lut, const float* cb1_sumq);
}
