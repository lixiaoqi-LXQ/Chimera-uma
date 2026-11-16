#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace Chimera { namespace doc_order_detail {
static __global__ void reorder(const uint4* source_codes, const float* source_factors,
        const uint32_t* mapping, uint4* ordered_codes, float* ordered_factors, size_t count) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<count;
         i+=size_t(gridDim.x)*blockDim.x) {
        const uint32_t pos=mapping[i];
        if (size_t(pos)>=count) continue;
        ordered_codes[i]=source_codes[pos]; ordered_factors[i]=source_factors[pos];
    }
}
static __global__ void verify(const uint4* source_codes, const float* source_factors,
        const uint32_t* mapping, const uint4* ordered_codes, const float* ordered_factors,
        size_t count, unsigned* mismatch) {
    unsigned bad=0;
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<count;
         i+=size_t(gridDim.x)*blockDim.x) {
        const uint32_t pos=mapping[i];
        if (size_t(pos)>=count) { bad=1; continue; }
        const uint4 a=source_codes[pos], b=ordered_codes[i];
        bad |= a.x!=b.x || a.y!=b.y || a.z!=b.z || a.w!=b.w ||
            __float_as_uint(source_factors[pos])!=__float_as_uint(ordered_factors[i]);
    }
    if (bad) atomicOr(mismatch,1u);
}
static __global__ void gather_positions(const int* docs, const int* doc_ptrs,
        const size_t* offsets, uint32_t* positions, size_t count) {
    const size_t doc=blockIdx.x;
    if(doc>=count)return;
    const int begin=doc_ptrs[docs[doc]], end=doc_ptrs[docs[doc]+1];
    for (int t=threadIdx.x; t<end-begin; t+=blockDim.x)
        positions[offsets[doc]+t]=uint32_t(begin+t);
}
} }
