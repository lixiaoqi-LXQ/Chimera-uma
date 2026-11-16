#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdint>
#include <cmath>

namespace Chimera { namespace tensor_half_oracle_detail {
// Diagnostic-only independent arithmetic. It deliberately uses original
// materialized positions/offsets even when the primary path uses direct IDs.
// One warp/document; lane=q. No WMMA, shared bit matrix or TC max code is used.
// Binary dot of128 finite halves is an integer multiple of2^-24 with magnitude
// below2^47 in those units, so its double scalar accumulation is exact.
__global__ void scalar_half_query_maxima_reference(
    const float* __restrict__ lut,
    const uint4* __restrict__ codes,
    const float* __restrict__ factors,
    const float* __restrict__ cb,
    const uint32_t* __restrict__ original_positions,
    const size_t* __restrict__ original_offsets,
    double* __restrict__ reference_maxima,
    double* __restrict__ maximum_error_bounds,
    size_t docs) {
    const size_t doc=blockIdx.x;
    const int q=threadIdx.x;
    if(doc>=docs || q>=32)return;
    const size_t begin=original_offsets[doc],end=original_offsets[doc+1];
    // WMMA does not fix a rounding mode: use a one-ulp single-precision
    // accumulation model, and separately cover flushing half-subnormal inputs.
    constexpr double utc=0x1p-23,u32=0x1p-24,u64=0x1p-53;
    constexpr double gamma128=(128*utc)/(1-128*utc);
    constexpr double gamma2=(2*u32)/(1-2*u32);
    constexpr double gamma2d=(2*u64)/(1-2*u64);
    constexpr double eta32=0x1p-149;
    double maximum=-DBL_MAX,error_bound=0;
    const double correction=static_cast<double>(cb[q]);
    for(size_t t=begin;t<end;++t) {
        const uint32_t pos=original_positions[t];
        const uint4 packed=codes[pos];
        const double factor=static_cast<double>(factors[pos]);
        double dot=0,absolute_sum=0,subnormal_sum=0;
        // Feature0..127, independently decoded from the full binary code.
        // Conversion from the LUT onehot item defines the same half input
        // domain, while the sum and max remain independent scalar double.
        for(int feature=0;feature<128;++feature) {
            const int word_index=feature/32;
            const uint32_t word=word_index==0?packed.x:word_index==1?packed.y:word_index==2?packed.z:packed.w;
            if((word>>(feature%32))&1u) {
                const float value=lut[q*512+(feature/4)*16+(1<<(feature%4))];
                const double half=static_cast<double>(__half2float(__float2half_rn(value)));
                dot=__dadd_rn(dot,half);
                absolute_sum=__dadd_rn(absolute_sum,fabs(half));
                if(fabs(half)<0x1p-14)subnormal_sum=__dadd_rn(subnormal_sum,fabs(half));
            }
        }
        const double score=__dmul_rn(__dadd_rn(dot,-correction),factor);
        const double dot_bound=gamma128*absolute_sum+subnormal_sum;
        const double scale=fabs(factor),magnitudes=fabs(dot)+fabs(correction);
        // Covers the WMMA sum, two FP32 cb/factor roundings, gradual FP32
        // underflow, and the two FP64 operations in this scalar reference.
        double bound=scale*((1+gamma2)*dot_bound+gamma2*magnitudes)
                    +scale*eta32+eta32+gamma2d*scale*magnitudes;
        // Upward safety margin for the small number of FP64 bound operations.
        bound=__dmul_ru(bound,1.00000000000001);
        maximum=fmax(maximum,score);
        error_bound=fmax(error_bound,bound);
    }
    // Primary uses -FLT_MAX query maxima for an empty document; finalsum is
    // -inf. Preserve this sentinel for exact empty-querymax QA.
    if(begin==end)maximum=-static_cast<double>(FLT_MAX);
    reference_maxima[doc*32+q]=maximum;
    maximum_error_bounds[doc*32+q]=error_bound;
}

// CPU diagnostic after the final event. Require finite nonempty maxima;
// empty documents are checked separately as exact -FLT_MAX/-inf sentinels.
inline double finalsum_error_bound(const double* reference_maxima,
                                   const double* maximum_error_bounds) noexcept {
    constexpr double u32=0x1p-24;
    constexpr double gamma8=(8*u32)/(1-8*u32);
    constexpr double u64=0x1p-53,gamma32d=(32*u64)/(1-32*u64);
    double magnitude=0,error=0;
    for(int q=0;q<32;++q){magnitude+=std::abs(reference_maxima[q]);error+=maximum_error_bounds[q];}
    return ((1+gamma8)*error+(gamma8+gamma32d)*magnitude+32*0x1p-149)*1.00000000000001;
}
} }  // namespace Chimera::tensor_half_oracle_detail
