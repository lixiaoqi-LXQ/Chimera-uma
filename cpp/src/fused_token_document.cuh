#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include "chimera/config.cuh"

namespace Chimera { namespace fused_token_document_detail {
static_assert(Q_DOCLEN == 32 && PADDED_DIM == 128 && NUM_CHUNKS == 32,
              "Fused token/document scoring requires Q32/D128");
// One CTA per document and contiguous eight-query group. The code/factor
// inputs are loaded once for each group; no 32xN score matrix is materialized.
// Per-query LUT additions retain c=0..31 order. Maxima are associative for the
// finite nonzero inputs of production; signed-zero fixtures verify the target.
template<int Threads>
__global__ void fused_token_doc_groups(
    const float* __restrict__ lut,
    const uint4* __restrict__ codes,
    const float* __restrict__ factors,
    const float* __restrict__ cb,
    const uint32_t* __restrict__ positions,
    const size_t* __restrict__ offsets,
    float* __restrict__ query_maxima,
    size_t docs) {
    static_assert(Threads == 128 || Threads == 256, "Supported CTA choices");
    extern __shared__ float shared[];
    float* tile_lut=shared;
    float* tile_cb=shared+4096;
    float* warp_maxima=shared+4104;
    const size_t doc=blockIdx.x;
    const int q0=int(blockIdx.y)*8, tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    if(doc>=docs)return;
    for(int i=tid;i<4096;i+=Threads)tile_lut[i]=lut[q0*512+i];
    if(tid<8)tile_cb[tid]=cb[q0+tid];
    __syncthreads();
    float maxima[8];
    #pragma unroll
    for(int q=0;q<8;++q)maxima[q]=-FLT_MAX;
    for(size_t t=offsets[doc]+tid;t<offsets[doc+1];t+=Threads) {
        const uint32_t pos=positions[t];
        const uint4 code=codes[pos];
        const uint64_t low=(uint64_t(code.y)<<32)|code.x;
        const uint64_t high=(uint64_t(code.w)<<32)|code.z;
        const float factor=factors[pos];
        float ip[8]={};
        // Keeping this loop rolled bounds decoded-value live ranges while
        // keeping every query's FP32 accumulation sequence unchanged.
        #pragma unroll 1
        for(int c=0;c<32;++c) {
            const uint64_t packed=c<16?low:high;
            const int nibble=int((packed>>((c&15)*4))&15);
            const int address=c*16+nibble;
            #pragma unroll
            for(int q=0;q<8;++q)ip[q]+=tile_lut[q*512+address];
        }
        #pragma unroll
        for(int q=0;q<8;++q)maxima[q]=fmaxf(maxima[q],(ip[q]-tile_cb[q])*factor);
    }
    #pragma unroll
    for(int q=0;q<8;++q) {
        float value=maxima[q];
        #pragma unroll
        for(int delta=16;delta;delta>>=1)value=fmaxf(value,__shfl_down_sync(0xffffffffu,value,delta));
        if(lane==0)warp_maxima[warp*8+q]=value;
    }
    __syncthreads();
    if(tid<8) {
        float value=-FLT_MAX;
        #pragma unroll
        for(int w=0;w<Threads/32;++w)value=fmaxf(value,warp_maxima[w*8+tid]);
        query_maxima[doc*32+q0+tid]=value;
    }
}

// Match original document kernel's 128-lane reduction ordering: its first
// 32 lanes do (0+max)+0+0, followed by the 16/8/4/2/1 tree. The added zeros
// matter for signed-zero qualification; explicit round-to-nearest additions
// also prevent a fast-math build from reassociating the final tree.
__global__ void fused_document_sum(
    const float* __restrict__ query_maxima,
    float* __restrict__ scores,
    size_t docs) {
    const size_t doc=blockIdx.x;
    const int lane=threadIdx.x;
    if(doc>=docs)return;
    float value=__fadd_rn(0.0f,query_maxima[doc*32+lane]);
    value=__fadd_rn(value,0.0f);
    value=__fadd_rn(value,0.0f);
    #pragma unroll
    for(int delta=16;delta;delta>>=1)value=__fadd_rn(value,__shfl_down_sync(0xffffffffu,value,delta));
    if(lane==0)scores[doc]=value;
}


template<int Threads>
__global__ void fused_token_doc_groups_interleaved(
    const float* __restrict__ lut,
    const uint4* __restrict__ codes,
    const float* __restrict__ factors,
    const float* __restrict__ cb,
    const uint32_t* __restrict__ positions,
    const size_t* __restrict__ offsets,
    float* __restrict__ query_maxima,
    size_t docs) {
    static_assert(Threads == 128 || Threads == 256, "Supported CTA choices");
    extern __shared__ float shared[];
    float* tile_lut=shared;
    float* tile_cb=shared+4096;
    float* warp_maxima=shared+4104;
    const size_t linear=size_t(blockIdx.x)+size_t(gridDim.x)*blockIdx.y;
    const size_t doc=linear/4;
    const int q0=int(linear%4)*8, tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    if(doc>=docs)return;
    for(int i=tid;i<4096;i+=Threads)tile_lut[i]=lut[q0*512+i];
    if(tid<8)tile_cb[tid]=cb[q0+tid];
    __syncthreads();
    float maxima[8];
    #pragma unroll
    for(int q=0;q<8;++q)maxima[q]=-FLT_MAX;
    for(size_t t=offsets[doc]+tid;t<offsets[doc+1];t+=Threads) {
        const uint32_t pos=positions[t];
        const uint4 code=codes[pos];
        const uint64_t low=(uint64_t(code.y)<<32)|code.x;
        const uint64_t high=(uint64_t(code.w)<<32)|code.z;
        const float factor=factors[pos];
        float ip[8]={};
        // Keeping this loop rolled bounds decoded-value live ranges while
        // keeping every query's FP32 accumulation sequence unchanged.
        #pragma unroll 1
        for(int c=0;c<32;++c) {
            const uint64_t packed=c<16?low:high;
            const int nibble=int((packed>>((c&15)*4))&15);
            const int address=c*16+nibble;
            #pragma unroll
            for(int q=0;q<8;++q)ip[q]+=tile_lut[q*512+address];
        }
        #pragma unroll
        for(int q=0;q<8;++q)maxima[q]=fmaxf(maxima[q],(ip[q]-tile_cb[q])*factor);
    }
    #pragma unroll
    for(int q=0;q<8;++q) {
        float value=maxima[q];
        #pragma unroll
        for(int delta=16;delta;delta>>=1)value=fmaxf(value,__shfl_down_sync(0xffffffffu,value,delta));
        if(lane==0)warp_maxima[warp*8+q]=value;
    }
    __syncthreads();
    if(tid<8) {
        float value=-FLT_MAX;
        #pragma unroll
        for(int w=0;w<Threads/32;++w)value=fmaxf(value,warp_maxima[w*8+tid]);
        query_maxima[doc*32+q0+tid]=value;
    }
}


__device__ __forceinline__ void fused_v2_update_maxima(
    uint64_t low,uint64_t high,float factor,const float* lut,const float* cb,float (&maxima)[8]) {
    float ip[8]={};
    #pragma unroll 1
    for(int c=0;c<32;++c) {
        const uint64_t packed=c<16?low:high;
        const int nibble=int((packed>>((c&15)*4))&15);
        const int address=c*16+nibble;
        #pragma unroll
        for(int q=0;q<8;++q)ip[q]+=lut[q*512+address];
    }
    #pragma unroll
    for(int q=0;q<8;++q)maxima[q]=fmaxf(maxima[q],(ip[q]-cb[q])*factor);
}

// One CTA per document. The first two per-thread token slots retain their
// two packed words and factors across all four eight-query LUT tiles. This
// avoids rereading the common <=2*Threads document input from its scattered
// corpus positions. Longer document tails remain correct via per-tile loops.
template<int Threads>
__global__ void fused_token_doc_cached(
    const float* __restrict__ lut,
    const uint4* __restrict__ codes,
    const float* __restrict__ factors,
    const float* __restrict__ cb,
    const uint32_t* __restrict__ positions,
    const size_t* __restrict__ offsets,
    float* __restrict__ query_maxima,
    size_t docs) {
    static_assert(Threads==128||Threads==256,"Supported CTA choices");
    extern __shared__ float shared[];
    float* tile_lut=shared;
    float* tile_cb=shared+4096;
    float* warp_maxima=shared+4104;
    const size_t doc=blockIdx.x;
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    if(doc>=docs)return;
    const size_t begin=offsets[doc],end=offsets[doc+1];
    uint64_t cache_low[2]={},cache_high[2]={};
    float cache_factor[2]={};
    bool cache_valid[2];
    #pragma unroll
    for(int slot=0;slot<2;++slot) {
        const size_t t=begin+tid+size_t(slot)*Threads;
        cache_valid[slot]=t<end;
        if(cache_valid[slot]) {
            const uint32_t pos=positions[t];const uint4 code=codes[pos];
            cache_low[slot]=(uint64_t(code.y)<<32)|code.x;
            cache_high[slot]=(uint64_t(code.w)<<32)|code.z;
            cache_factor[slot]=factors[pos];
        }
    }
    // Keep the four-tile loop rolled so the two cached code slots are explicit
    // live state and so LUT loads/barriers are shared across all tile bodies.
    #pragma unroll 1
    for(int tile=0;tile<4;++tile) {
        const int q0=tile*8;
        // Safe without an additional barrier here: the preceding tile's first
        // eight threads only read warp_maxima, disjoint from LUT/cb. Those
        // reads must finish before these same threads can reach this tile's
        // first barrier. No warp_maxima writer for this tile can pass it early.
        for(int i=tid;i<4096;i+=Threads)tile_lut[i]=lut[q0*512+i];
        if(tid<8)tile_cb[tid]=cb[q0+tid];
        __syncthreads();
        float maxima[8];
        #pragma unroll
        for(int q=0;q<8;++q)maxima[q]=-FLT_MAX;
        #pragma unroll
        for(int slot=0;slot<2;++slot)
            if(cache_valid[slot])fused_v2_update_maxima(cache_low[slot],cache_high[slot],cache_factor[slot],tile_lut,tile_cb,maxima);
        // For >2*Threads tokens the tail is read each tile. It neither indexes
        // past the document nor silently drops large-document tokens.
        for(size_t t=begin+size_t(2)*Threads+tid;t<end;t+=Threads) {
            const uint32_t pos=positions[t];const uint4 code=codes[pos];
            const uint64_t low=(uint64_t(code.y)<<32)|code.x;
            const uint64_t high=(uint64_t(code.w)<<32)|code.z;
            fused_v2_update_maxima(low,high,factors[pos],tile_lut,tile_cb,maxima);
        }
        #pragma unroll
        for(int q=0;q<8;++q) {
            float value=maxima[q];
            #pragma unroll
            for(int delta=16;delta;delta>>=1)value=fmaxf(value,__shfl_down_sync(0xffffffffu,value,delta));
            if(lane==0)warp_maxima[warp*8+q]=value;
        }
        __syncthreads();
        if(tid<8) {
            float value=-FLT_MAX;
            #pragma unroll
            for(int w=0;w<Threads/32;++w)value=fmaxf(value,warp_maxima[w*8+tid]);
            query_maxima[doc*32+q0+tid]=value;
        }
    }
}

} }  // namespace Chimera::fused_token_document_detail
