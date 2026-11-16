#pragma once
#include "tensor_token_document.cuh"

namespace Chimera { namespace tensor_token_document_max4_detail {
inline constexpr int threads=tensor_token_document_detail::threads;
inline constexpr int token_tile=tensor_token_document_detail::token_tile;
inline constexpr int half_ld=tensor_token_document_detail::half_ld;
inline constexpr int product_ld=tensor_token_document_detail::product_ld;
inline constexpr size_t shared_bytes=tensor_token_document_detail::shared_bytes;
static_assert(threads==128 && token_tile==32 && half_ld==136 && product_ld==36 && shared_bytes==22784,
              "Max4 requires the qualified padded WMMA layout");
// Same inputs/WMMA/direct single-writer mapping as TensorPadded. Only max is
// split across four lanes/query; exact max4/control micro qualification precedes
// adoption. Core QA retains the original half-scalar bound and final fullbit.
template<bool DirectDocs>
__global__ void tensor_token_document_max4(
    const float* __restrict__ lut,
    const uint4* __restrict__ codes,
    const float* __restrict__ factors,
    const float* __restrict__ cb,
    const uint32_t* __restrict__ positions,
    const size_t* __restrict__ offsets,
    const int* __restrict__ refined_doc_ids,
    const int* __restrict__ document_ptrs,
    int* __restrict__ mapped_refined_doc_ids,
    int* __restrict__ diagnostic_lengths,
    float* __restrict__ query_maxima,
    size_t docs) {
    namespace wmma=nvcuda::wmma;
    __shared__ __align__(32) __half query[32*136];
    // Column-major K128 x N32, with 8 half padding slots per token.
    __shared__ __align__(32) __half bits[32*136];
    __shared__ __align__(32) float products[32*36];
    __shared__ __align__(32) uint4 tile_codes[32];
    __shared__ float tile_factors[32];
    __shared__ float tile_cb[32];
    const size_t doc=blockIdx.x;
    if(doc>=docs)return;  // CTA-uniform; all four warps execute every WMMA call.
    const int tid=threadIdx.x,warp=tid/32;
    const int q0=(warp/2)*16,t0=(warp%2)*16;
    for(int i=tid;i<32*128;i+=128) {
        const int q=i/128,k=i%128;
        query[q*136+k]=__float2half_rn(lut[q*512+(k/4)*16+(1<<(k%4))]);
    }
    if(tid<32)tile_cb[tid]=cb[tid];
    __syncthreads();
    size_t begin,end;
    if constexpr (DirectDocs) {
        const int doc_id=refined_doc_ids[doc];
        begin=static_cast<size_t>(document_ptrs[doc_id]);
        end=static_cast<size_t>(document_ptrs[doc_id+1]);
        // Exactly one writer per document. CPU readers wait for the caller's
        // existing final completion event. Input GPU IDs and ATS output IDs
        // must be separate allocations (the no-CPU-refine direct contract).
        if(tid==0) {
            mapped_refined_doc_ids[doc]=doc_id;
            if(diagnostic_lengths)diagnostic_lengths[doc]=static_cast<int>(end-begin);
        }
    } else {
        begin=offsets[doc];end=offsets[doc+1];
    }
    const int query_id=tid/4,lane4=tid%4;
    const float query_cb=tile_cb[query_id];
    float maximum=-FLT_MAX;  // Four strided lanes/query, retained across tiles.
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    for(size_t base=begin;base<end;base+=32) {
        const int valid=static_cast<int>((end-base)<32?(end-base):32);
        if(tid<32) {
            if(tid<valid) {
                uint32_t pos;
                if constexpr (DirectDocs)pos=static_cast<uint32_t>(base+tid);
                else pos=positions[base+tid];
                tile_codes[tid]=codes[pos];
                tile_factors[tid]=factors[pos];
            } else {
                tile_codes[tid]=make_uint4(0,0,0,0);
                tile_factors[tid]=0.0f;
            }
        }
        // This barrier also finishes prior-tile product/max readers before the
        // next WMMA store. Codes/factors and products occupy disjoint ranges.
        __syncthreads();
        for(int i=tid;i<32*128;i+=128) {
            const int token=i/128,k=i%128;
            const uint4 packed=tile_codes[token];
            const uint32_t word=(k/32)==0?packed.x:(k/32)==1?packed.y:(k/32)==2?packed.z:packed.w;
            bits[token*136+k]=__float2half_rn(float((word>>(k%32))&1u));
        }
        // Each thread captures its eight factors before barrier B. Warp0
        // can then produce the next tile while other warps finish max without
        // racing their factor reads. No shared factor reads are allowed after B.
        float factor_registers[8];
        #pragma unroll
        for(int j=0;j<8;++j)factor_registers[j]=tile_factors[lane4+4*j];
        __syncthreads();  // B: all factor/codes readers finish before reuse
        wmma::fill_fragment(acc,0.0f);
        #pragma unroll
        for(int k=0;k<128;k+=16) {
            wmma::load_matrix_sync(a,query+q0*136+k,136);
            wmma::load_matrix_sync(b,bits+t0*136+k,136);
            wmma::mma_sync(acc,a,b,acc);
        }
        wmma::store_matrix_sync(products+q0*36+t0,acc,36,wmma::mem_row_major);
        __syncthreads();
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int token=lane4+4*j;
            if(token<valid)
                maximum=fmaxf(maximum,(products[query_id*36+token]-query_cb)*factor_registers[j]);
        }
    }
    // CTA-uniform control flow: all four warps execute both width4 shuffles,
    // including empty documents. One writer per query keeps the original scratch.
    maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,2,4));
    maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,1,4));
    if(lane4==0)query_maxima[doc*32+query_id]=maximum;
}
} }  // namespace Chimera::tensor_token_document_max4_detail
