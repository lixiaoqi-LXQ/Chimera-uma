#pragma once
#include "tensor_token_document.cuh"

namespace Chimera { namespace tensor_token_document_hoist_decode_detail {
inline constexpr int threads=tensor_token_document_detail::threads;
inline constexpr int token_tile=tensor_token_document_detail::token_tile;
inline constexpr int half_ld=tensor_token_document_detail::half_ld;
inline constexpr int product_ld=tensor_token_document_detail::product_ld;
inline constexpr size_t shared_bytes=32*136*2+32*36*4+32*16+32*4+32*4;
static_assert(threads==128 && token_tile==32 && half_ld==136 && product_ld==36 && shared_bytes==14080,
              "A hoist requires the qualified padded WMMA shape and query/bits union");
union QueryBits {
    __half query[32*136];
    __half bits[32*136];
};
static_assert(sizeof(QueryBits)==8704,"One 32x136 half shared union");
// Combine the qualified A hoist/query-bits union with the qualified code-word
// cache/warp broadcast decode. Preserve original warp0 max, K order, LUT rounding,
// barriers, direct mapping and completion; do not stack Max4.
template<bool DirectDocs>
__global__ void tensor_token_document_hoist_decode(
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
    __shared__ __align__(32) QueryBits query_bits;
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
        query_bits.query[q*136+k]=__float2half_rn(lut[q*512+(k/4)*16+(1<<(k%4))]);
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
    float maximum=-FLT_MAX;  // Only first 32 threads keep a query maximum.
    // Constant fragment names/K offsets avoid dynamic fragment-array indexing.
    // Actual native registers and spills are measured after this integration.
    using AFragment=wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major>;
    AFragment a0,a1,a2,a3,a4,a5,a6,a7;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::load_matrix_sync(a0,query_bits.query+q0*136+0,136);
    wmma::load_matrix_sync(a1,query_bits.query+q0*136+16,136);
    wmma::load_matrix_sync(a2,query_bits.query+q0*136+32,136);
    wmma::load_matrix_sync(a3,query_bits.query+q0*136+48,136);
    wmma::load_matrix_sync(a4,query_bits.query+q0*136+64,136);
    wmma::load_matrix_sync(a5,query_bits.query+q0*136+80,136);
    wmma::load_matrix_sync(a6,query_bits.query+q0*136+96,136);
    wmma::load_matrix_sync(a7,query_bits.query+q0*136+112,136);
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
        // First tile: all four warps finish every A/query shared read before
        // bits decode aliases that storage. Later tiles keep the original
        // product/max-reader boundary; codes/factors remain disjoint. Empty
        // documents skip this loop CTA-uniformly and never overwrite query.
        __syncthreads();
        // A published all32 code slots, including zeroed invalid tail slots.
        // Warp w owns bits32*w..32*w+31. Lane l loads only word w of token l.
        const int lane=tid&31;
        const uint4* own_token=&tile_codes[lane];
        const uint32_t* own_word_address=warp==0?&own_token->x:
            warp==1?&own_token->y:warp==2?&own_token->z:&own_token->w;
        const uint32_t own_word=*own_word_address;
        #pragma unroll
        for(int token=0;token<32;++token) {
            const uint32_t word=__shfl_sync(0xffffffffu,own_word,token);
            query_bits.bits[token*136+tid]=__float2half_rn(float((word>>lane)&1u));
        }
        // Existing B below completes every shuffle/decode before code reuse.
        __syncthreads();
        wmma::fill_fragment(acc,0.0f);
        wmma::load_matrix_sync(b,query_bits.bits+t0*136+0,136);
        wmma::mma_sync(acc,a0,b,acc);
        wmma::load_matrix_sync(b,query_bits.bits+t0*136+16,136);
        wmma::mma_sync(acc,a1,b,acc);
        wmma::load_matrix_sync(b,query_bits.bits+t0*136+32,136);
        wmma::mma_sync(acc,a2,b,acc);
        wmma::load_matrix_sync(b,query_bits.bits+t0*136+48,136);
        wmma::mma_sync(acc,a3,b,acc);
        wmma::load_matrix_sync(b,query_bits.bits+t0*136+64,136);
        wmma::mma_sync(acc,a4,b,acc);
        wmma::load_matrix_sync(b,query_bits.bits+t0*136+80,136);
        wmma::mma_sync(acc,a5,b,acc);
        wmma::load_matrix_sync(b,query_bits.bits+t0*136+96,136);
        wmma::mma_sync(acc,a6,b,acc);
        wmma::load_matrix_sync(b,query_bits.bits+t0*136+112,136);
        wmma::mma_sync(acc,a7,b,acc);
        wmma::store_matrix_sync(products+q0*36+t0,acc,36,wmma::mem_row_major);
        __syncthreads();
        if(tid<32) {
            // Padding columns must not enter max, even when every valid score
            // is negative. FP32 cb/factor operations retain original form.
            for(int token=0;token<valid;++token)
                maximum=fmaxf(maximum,(products[tid*36+token]-tile_cb[tid])*tile_factors[token]);
        }
    }
    if(tid<32)query_maxima[doc*32+tid]=maximum;
}
} }  // namespace Chimera::tensor_token_document_hoist_decode_detail
