#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>

#include "chimera/chimera_index.cuh"

#define EIGEN_NO_CUDA

#include "chimera/config.cuh"
#include "chimera/chunk_partition.hpp"
#include "chimera/cpu_candidate_generation.hpp"
#include "chimera/cpu_document_reduce_pool.hpp"
#include "cpu_task_pool.hpp"
#include "chimera/cpu_refine_topk.hpp"
#include "chimera/profiling.hpp"
#include "chimera/ivf_pg.hpp"
#include "chimera/kernels.cuh"
#include "chimera_index_internal.cuh"
#include "doc_order_onebit.cuh"
#include "fused_token_document.cuh"
#include "split_token_score.cuh"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

#include <algorithm>
#include <cstdio>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "rabitqlib/utils/chimera_simd.hpp"
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Chimera {

namespace {

struct cast_int_size_t {
    __host__ __device__ size_t operator()(int value) const {
        return static_cast<size_t>(value);
    }
};

}  // namespace

std::vector<size_t> chimera_index::search(
    const std::vector<float>& query,
    size_t k) {
    return search(query.data(), query.size(), k);
}

std::vector<size_t> chimera_index::search(
    const float* query,
    size_t query_size,
    size_t k) {
    if (!has_index_) {
        throw std::runtime_error("Cannot search an empty Chimera index");
    }
    if (query == nullptr || query_size != Q_DOCLEN * d) {
        throw std::invalid_argument(
            "A Chimera query must have shape [" +
            std::to_string(Q_DOCLEN) + ", " + std::to_string(d) + "]");
    }
    if (k == 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (!search_ready_) {
        try {
            initialize_search_state();
        } catch (...) {
            release_resources();
            throw;
        }
    }
    return search_one(query, k);
}

std::vector<size_t> chimera_index::search_one(const float* queries, size_t k) {
    if (profiling_enabled_) {
        static thread_local bool named = false;
        if (!named) {
            char name[32];
            std::snprintf(name, sizeof(name), "Chimera_lane_%d", omp_get_thread_num());
            nvtxNameOsThreadA(static_cast<uint32_t>(::syscall(SYS_gettid)), name);
            named = true;
        }
    }
    std::vector<QueryState> query_states;
    {
    ProfileRange cpu_prepare(profiling_enabled_, "Chimera/cpu_query_prepare");
    for (size_t i = 0; i < Q_DOCLEN; ++i) {
        rotator_->rotate(&queries[i * d], &ws_->h_pinned_queries[i * PADDED_DIM]);
    }

    query_states.resize(Q_DOCLEN);
    for (size_t i = 0; i < Q_DOCLEN; ++i) {
        query_states[i] =
            QueryState(&ws_->h_pinned_queries[i * PADDED_DIM], PADDED_DIM, ex_bits);
        ws_->h_pinned_cb1_sumq[i] = query_states[i].cb1_sumq;
    }
    }

    if (!ws_->uma_query_buffers) {
        check_cuda(cudaMemcpyAsync(ws_->d_queries, ws_->h_pinned_queries,
                                   Q_DOCLEN * PADDED_DIM * sizeof(float),
                                   cudaMemcpyHostToDevice, ws_->stream_h2d));
        check_cuda(cudaMemcpyAsync(ws_->d_cb1_sumq, ws_->h_pinned_cb1_sumq,
                                   Q_DOCLEN * sizeof(float),
                                   cudaMemcpyHostToDevice, ws_->stream_h2d));
        check_cuda(cudaEventRecord(ws_->event_h2d_done, ws_->stream_h2d));
        check_cuda(cudaStreamWaitEvent(ws_->stream_compute, ws_->event_h2d_done));
    }

    precompute_one_bit_lut_kernel<<<Q_DOCLEN, 256, 0, ws_->stream_compute>>>(
        ws_->d_queries, ws_->d_lut);
    check_cuda(cudaGetLastError());


    int num_refined_candidates = 0;
    generate_and_refine_candidates(
        options_.nprobe,
        options_.k_refine,
        num_refined_candidates,
        ws_->stream_compute);


    std::vector<size_t> result;
    collaborative_document_scoring(
        num_refined_candidates,
        k,
        options_.k_full_bit,
        query_states.data(),
        result);


    return result;
}

void chimera_index::generate_and_refine_candidates(
    size_t nprobe,
    size_t k_refine,
    int& num_refined_candidates,
    cudaStream_t stream) {
    ProfileRange range(profiling_enabled_, "Chimera/candidate_generation_refinement");
    if (!options_.uma_cpu_candidates)
        check_cuda(cudaMemsetAsync(d_candidate_bitmap_, 0,
                                   candidate_bitmap_bucket_count_ * sizeof(candidate_bitmap_bucket_t),
                                   ws_->stream_d2h));
    if (ws_->uma_query_buffers) {
        // The count kernel overwrites every group count. Only the final scan
        // sentinel needs resetting; the previous synchronous query is done.
        ws_->h_candidate_bitmap_offsets[candidate_bitmap_offset_count_ - 1] = 0;
    } else {
        check_cuda(cudaMemsetAsync(d_candidate_bitmap_offsets_, 0,
                                   candidate_bitmap_offset_count_ * sizeof(candidate_bitmap_offset_t),
                                   ws_->stream_d2h));
    }
    check_cuda(cudaEventRecord(ws_->event_h2d_done, ws_->stream_d2h));

    {
    ProfileRange route(profiling_enabled_, "Chimera/route");
    if (uses_flat_centroid_route(options_.algorithm_variant)) {
        ws_->flat_router->search(*cagra_res_, ws_->d_queries, Q_DOCLEN, nprobe, ws_->d_cagra_dists, ws_->d_cagra_labels, stream);
    } else {
    ivf->search_centroids(
        *cagra_res_,
        ws_->d_queries,
        Q_DOCLEN,
        nprobe,
        ws_->d_cagra_dists,
        ws_->d_cagra_labels,
        stream,
        static_cast<size_t>(options_.cagra_itopk_size),
        options_.cagra_algo,
        static_cast<size_t>(options_.cagra_team_size),
        static_cast<size_t>(options_.cagra_thread_block_size),
        static_cast<size_t>(options_.cagra_search_width));
    }
    }

    check_cuda(cudaStreamWaitEvent(stream, ws_->event_h2d_done));

    const int threads_per_block = 256;
    const int blocks_x = std::max(
        (max_cluster_size + threads_per_block - 1) / threads_per_block,
        16);
    const dim3 grid(blocks_x, Q_DOCLEN);

    candidate_bitmap_offset_t num_candidate_docs_u32 = 0;
    if (options_.uma_cpu_candidates) {
        // CAGRA publishes labels into ATS; all subsequent CPU writes are
        // published before the same stream's dependent GPU launches.
        check_cuda(cudaStreamSynchronize(stream));
        ProfileRange cpu_cg(profiling_enabled_, "Chimera/uma_cpu_candidates");
        num_candidate_docs_u32 = cpu_generate_candidates(
            ws_->h_cpu_cagra_labels, size_t(Q_DOCLEN) * nprobe,
            ivf->cluster_pos.data(), ivf->n_clusters, d_clustered_doc_ids_, num_docs,
            ws_->h_cpu_candidate_bitmap, candidate_bitmap_bucket_count_,
            ws_->h_candidate_bitmap_offsets, ws_->h_cpu_materialized_ids,
            max_candidate_docs_);
    } else {
    mark_candidate_documents_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_->d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        d_candidate_bitmap_,
        num_docs,
        static_cast<int>(nprobe),
        ivf->n_clusters);
    check_cuda(cudaGetLastError());

    count_candidate_bitmap_kernel<<<
        (candidate_bitmap_bucket_count_ + threads_per_block - 1) / threads_per_block,
        threads_per_block,
        0,
        stream>>>(
            d_candidate_bitmap_,
            candidate_bitmap_bucket_count_,
            d_candidate_bitmap_offsets_);
    check_cuda(cudaGetLastError());

    if (ws_->uma_query_buffers) {
        // count_candidate_bitmap_kernel overwrites all group counts. Wait for
        // those GPU writes, then scan the same ATS-visible storage on the CPU.
        // The sentinel input is zero, so its exclusive prefix is the doc count.
        check_cuda(cudaStreamSynchronize(stream));
        candidate_bitmap_offset_t prefix = 0;
        for (size_t i = 0; i < candidate_bitmap_offset_count_; ++i) {
            const candidate_bitmap_offset_t count = ws_->h_candidate_bitmap_offsets[i];
            ws_->h_candidate_bitmap_offsets[i] = prefix;
            prefix += count;
        }
        num_candidate_docs_u32 = ws_->h_candidate_bitmap_offsets[candidate_bitmap_offset_count_ - 1];
    } else {
        size_t scan_temp = ws_->cub_temp_storage_bytes;
        cub::DeviceScan::ExclusiveSum(
            ws_->d_cub_temp_storage,
            scan_temp,
            d_candidate_bitmap_offsets_,
            d_candidate_bitmap_offsets_,
            candidate_bitmap_offset_count_,
            stream);
        check_cuda(cudaMemcpyAsync(&num_candidate_docs_u32,
                                   d_candidate_bitmap_offsets_ + candidate_bitmap_offset_count_ - 1,
                                   sizeof(candidate_bitmap_offset_t),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        check_cuda(cudaStreamSynchronize(stream));
    }

    }

    const int num_candidate_docs = static_cast<int>(num_candidate_docs_u32);
    if (num_candidate_docs == 0) {
        num_refined_candidates = 0;
        return;
    }
    if (static_cast<size_t>(num_candidate_docs) > max_candidate_docs_) {
        throw std::runtime_error(
            "Candidate document count " + std::to_string(num_candidate_docs) +
            " exceeded workspace bound " + std::to_string(max_candidate_docs_));
    }

    ensure_candidate_capacity(static_cast<size_t>(num_candidate_docs));
    if (!skips_candidate_refinement(options_.algorithm_variant))
    check_cuda(cudaMemsetAsync(ws_->d_partial_query_scores,
                               0,
                               static_cast<size_t>(num_candidate_docs) * Q_DOCLEN * sizeof(float),
                               stream));

    if (options_.uma_cpu_candidates && options_.verify_full_bit) {
        // Diagnostic only: run the original GPU integer pipeline on independent
        // buffers. This path is excluded from all throughput measurements.
        struct Reference {
            uint32_t* bitmap = nullptr;
            uint32_t* offsets = nullptr;
            int* ids = nullptr;
            ~Reference() { (void)cudaFree(bitmap); (void)cudaFree(offsets); (void)cudaFree(ids); }
        } ref;
        check_cuda(cudaMalloc(&ref.bitmap, candidate_bitmap_bucket_count_ * sizeof(uint32_t)));
        check_cuda(cudaMalloc(&ref.offsets, candidate_bitmap_offset_count_ * sizeof(uint32_t)));
        check_cuda(cudaMalloc(&ref.ids, size_t(num_candidate_docs) * sizeof(int)));
        check_cuda(cudaMemsetAsync(ref.bitmap, 0, candidate_bitmap_bucket_count_ * sizeof(uint32_t), stream));
        check_cuda(cudaMemsetAsync(ref.offsets, 0, candidate_bitmap_offset_count_ * sizeof(uint32_t), stream));
        mark_candidate_documents_kernel<<<grid, threads_per_block, 0, stream>>>(
            ws_->d_cagra_labels, d_cluster_pos_, d_clustered_doc_ids_, ref.bitmap,
            num_docs, static_cast<int>(nprobe), ivf->n_clusters);
        check_cuda(cudaGetLastError());
        count_candidate_bitmap_kernel<<<
            (candidate_bitmap_bucket_count_ + threads_per_block - 1) / threads_per_block,
            threads_per_block, 0, stream>>>(ref.bitmap, candidate_bitmap_bucket_count_, ref.offsets);
        check_cuda(cudaGetLastError());
        size_t scan_temp = ws_->cub_temp_storage_bytes;
        cub::DeviceScan::ExclusiveSum(ws_->d_cub_temp_storage, scan_temp, ref.offsets,
            ref.offsets, candidate_bitmap_offset_count_, stream);
        // First compare count; an incorrect reference count must not overrun IDs.
        uint32_t reference_count = 0;
        check_cuda(cudaMemcpyAsync(&reference_count, ref.offsets + candidate_bitmap_offset_count_ - 1,
            sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaStreamSynchronize(stream));
        if (reference_count != num_candidate_docs_u32)
            throw std::runtime_error("UMA CPU CG count differs from GPU reference");
        materialize_candidate_documents_kernel<<<
            (candidate_bitmap_bucket_count_ + threads_per_block - 1) / threads_per_block,
            threads_per_block, 0, stream>>>(ref.bitmap, candidate_bitmap_bucket_count_, ref.offsets, ref.ids);
        check_cuda(cudaGetLastError());
        std::vector<uint32_t> host_bitmap(candidate_bitmap_bucket_count_);
        std::vector<uint32_t> host_offsets(candidate_bitmap_offset_count_);
        std::vector<int> host_ids(num_candidate_docs);
        check_cuda(cudaMemcpyAsync(host_bitmap.data(), ref.bitmap, host_bitmap.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaMemcpyAsync(host_offsets.data(), ref.offsets, host_offsets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaMemcpyAsync(host_ids.data(), ref.ids, host_ids.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaStreamSynchronize(stream));
        if (std::memcmp(host_bitmap.data(), ws_->h_cpu_candidate_bitmap, host_bitmap.size() * sizeof(uint32_t)) ||
            std::memcmp(host_offsets.data(), ws_->h_candidate_bitmap_offsets, host_offsets.size() * sizeof(uint32_t)) ||
            std::memcmp(host_ids.data(), ws_->h_cpu_materialized_ids, host_ids.size() * sizeof(int)))
            throw std::runtime_error("UMA CPU CG bitmap/prefix/IDs differ from GPU reference");
        std::cout << "[UMA_CANDIDATES_VERIFY] candidates=" << num_candidate_docs
                  << " bitmap_prefix_ids_bitwise=passed\n";
    }
    // Keep a local pointer: ensure_candidate_capacity rebinds the arena alias.
    int* materialized_ids = options_.uma_cpu_candidates
        ? ws_->h_cpu_materialized_ids : ws_->d_materialized_candidate_doc_ids;
    if (!options_.uma_cpu_candidates) {
    materialize_candidate_documents_kernel<<<
        (candidate_bitmap_bucket_count_ + threads_per_block - 1) / threads_per_block,
        threads_per_block,
        0,
        stream>>>(
            d_candidate_bitmap_,
            candidate_bitmap_bucket_count_,
            d_candidate_bitmap_offsets_,
            ws_->d_materialized_candidate_doc_ids);
    check_cuda(cudaGetLastError());

    }

    if (skips_candidate_refinement(options_.algorithm_variant)) {
        // True no-CR: every materialized CG document enters unchanged DS.
        ProfileRange bypass(profiling_enabled_, "Chimera/no_cr_all_candidates", num_candidate_docs);
        num_refined_candidates = num_candidate_docs;
        check_cuda(cudaMemcpyAsync(ws_->d_candidate_doc_ids, materialized_ids,
                                   size_t(num_candidate_docs) * sizeof(int), (options_.uma_cpu_candidates ? cudaMemcpyDefault : cudaMemcpyDeviceToDevice), stream));
        return;
    }

    partial_score_kernel<<<grid, threads_per_block, 0, stream>>>(
        ws_->d_lut,
        d_clustered_one_bit_codes_,
        d_clustered_one_bit_factors_,
        ws_->d_cb1_sumq,
        ws_->d_cagra_labels,
        d_cluster_pos_,
        d_clustered_doc_ids_,
        ws_->d_partial_query_scores,
        d_candidate_bitmap_,
        d_candidate_bitmap_offsets_,
        num_candidate_docs,
        num_docs,
        static_cast<int>(nprobe),
        ivf->n_clusters);
    check_cuda(cudaGetLastError());

    num_refined_candidates =
        std::min(static_cast<int>(k_refine), num_candidate_docs);

    const int thread_count = PARTIAL_SCORE_WARPS_PER_BLOCK * 32;
    const int score_blocks =
        (num_candidate_docs + PARTIAL_SCORE_WARPS_PER_BLOCK - 1) /
        PARTIAL_SCORE_WARPS_PER_BLOCK;
    float* partial_scores = options_.uma_cpu_refine
        ? ws_->h_cpu_refine_scores : ws_->d_partial_scores;
    sum_partial_scores_kernel<<<score_blocks, thread_count, 0, stream>>>(
        ws_->d_partial_query_scores,
        materialized_ids,
        partial_scores,
        ws_->d_candidate_doc_ids,
        num_candidate_docs);
    check_cuda(cudaGetLastError());
    if (options_.uma_cpu_refine) {
        check_cuda(cudaStreamSynchronize(stream));
        {
            ProfileRange cpu_topk(profiling_enabled_, "Chimera/uma_cpu_refine");
            cpu_refine_topk(partial_scores, materialized_ids, size_t(num_candidate_docs),
                size_t(num_refined_candidates), ws_->cpu_refine_order.data(), ws_->h_refined_doc_ids);
        }
        if (options_.verify_full_bit) {
    cub::DeviceRadixSort::SortPairsDescending(
        ws_->d_cub_temp_storage,
        ws_->cub_temp_storage_bytes,
        partial_scores,
        ws_->d_sorted_partial_scores,
        ws_->d_candidate_doc_ids,
        ws_->d_sorted_candidate_doc_ids,
        num_candidate_docs,
        0,
        32,
        stream);
    check_cuda(cudaGetLastError());
            std::vector<int> reference_ids(num_refined_candidates);
            std::vector<float> reference_scores(num_refined_candidates);
            check_cuda(cudaMemcpyAsync(reference_ids.data(), ws_->d_sorted_candidate_doc_ids,
                size_t(num_refined_candidates) * sizeof(int), cudaMemcpyDeviceToHost, stream));
            check_cuda(cudaMemcpyAsync(reference_scores.data(), ws_->d_sorted_partial_scores,
                size_t(num_refined_candidates) * sizeof(float), cudaMemcpyDeviceToHost, stream));
            check_cuda(cudaStreamSynchronize(stream));
            for (int i = 0; i < num_refined_candidates; ++i) {
                const float selected = partial_scores[cpu_refine_row(ws_->cpu_refine_order[size_t(i)])];
                if (reference_ids[size_t(i)] != ws_->h_refined_doc_ids[i] ||
                    std::memcmp(&selected, &reference_scores[size_t(i)], sizeof(float)))
                    throw std::runtime_error("UMA CPU top-k differs from CUB reference IDs/score bits");
            }
            std::cout << "[UMA_REFINE_VERIFY] candidates=" << num_candidate_docs
                      << " selected=" << num_refined_candidates << " ids_scores_bitwise=passed\n";
        }
    } else {
    cub::DeviceRadixSort::SortPairsDescending(
        ws_->d_cub_temp_storage,
        ws_->cub_temp_storage_bytes,
        partial_scores,
        ws_->d_sorted_partial_scores,
        ws_->d_candidate_doc_ids,
        ws_->d_sorted_candidate_doc_ids,
        num_candidate_docs,
        0,
        32,
        stream);
    check_cuda(cudaGetLastError());
    check_cuda(cudaMemcpyAsync(ws_->d_candidate_doc_ids,
                               ws_->d_sorted_candidate_doc_ids,
                               num_refined_candidates * sizeof(int),
                               cudaMemcpyDeviceToDevice,
                               stream));
    }
}

void chimera_index::collaborative_document_scoring(
    int num_refined_candidates,
    size_t k,
    size_t k_full_bit,
    QueryState* queries,
    std::vector<size_t>& result
) {
    ProfileRange range(profiling_enabled_, "Chimera/collaborative_scoring", num_refined_candidates);
    if (num_refined_candidates == 0) {
        result.clear();
        last_result_scores_.clear();
        return;
    }

    cudaStream_t stream = ws_->stream_compute;
    cudaStream_t stream_d2h = ws_->stream_d2h;
    const int full_bit_target = static_cast<int>(
        std::min(k_full_bit, static_cast<size_t>(num_refined_candidates)));

    const int* refined_ids = options_.uma_cpu_refine
        ? ws_->h_refined_doc_ids : ws_->d_candidate_doc_ids;
    const bool direct_docs = options_.fused_direct_docs;
    // The old metadata path is retained only for ordinary scoring or an
    // explicit original-kernel oracle. The primary direct kernel still runs
    // during verification; only its reference needs materialized positions.
    const bool metadata_ready = !direct_docs || options_.verify_full_bit;
    const bool collect_direct_lengths = direct_docs &&
        (options_.verify_full_bit || (profiling_enabled_ && profile_query_id_ >= 0));
    size_t total_tokens = 0;
    if (metadata_ready) {
    if (options_.uma_cpu_refine) {
        // Selected IDs already belong to the CPU. Derive the tiny metadata
        // directly; do not round-trip to a GPU length gather and host wait.
        ProfileRange metadata(profiling_enabled_, "Chimera/uma_cpu_refined_metadata");
        const auto& ptrs = corpus().doc_ptrs_;
        ws_->h_pinned_refined_doc_offsets[0] = 0;
        for (int i = 0; i < num_refined_candidates; ++i) {
            const int doc = refined_ids[i];
            total_tokens += static_cast<size_t>(ptrs[doc + 1] - ptrs[doc]);
            ws_->h_pinned_refined_doc_offsets[i + 1] = total_tokens;
        }
    } else {
        // Prepare the refined candidate documents for collaborative scoring.
        int threads = 256;
        int blocks = (num_refined_candidates + threads - 1) / threads;

        gather_refined_doc_lengths_kernel<<<blocks, threads, 0, stream>>>(
            refined_ids, d_doc_ptrs_,
            ws_->d_refined_doc_lengths,
            num_refined_candidates,
            options_.uma_cpu_refine ? nullptr : ws_->d_mapped_refined_doc_ids
        );
        check_cuda(cudaGetLastError());

        if (ws_->uma_query_buffers) {
            // Lengths and mapped IDs were both written by the preceding gather.
            // Move the existing metadata handoff here, before the CPU prefix sum.
            check_cuda(cudaStreamSynchronize(stream));
            ws_->h_pinned_refined_doc_offsets[0] = 0;
            for (int i = 0; i < num_refined_candidates; ++i) {
                total_tokens += static_cast<size_t>(ws_->h_refined_doc_lengths[i]);
                ws_->h_pinned_refined_doc_offsets[i + 1] = total_tokens;
            }
        } else {
            check_cuda(cudaMemsetAsync(ws_->d_refined_doc_offsets, 0, sizeof(size_t), stream));
            thrust::device_ptr<int> len_ptr(ws_->d_refined_doc_lengths);
            auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t());
            size_t scan_temp = ws_->cub_temp_storage_bytes;
            cub::DeviceScan::InclusiveScan(
                ws_->d_cub_temp_storage, scan_temp,
                xform_iter, ws_->d_refined_doc_offsets + 1,
                thrust::plus<size_t>(), num_refined_candidates, stream);
        }
    }

    if(options_.doc_order_onebit) {
        doc_order_detail::gather_positions<<<num_refined_candidates,256,0,stream>>>(
            refined_ids,d_doc_ptrs_,ws_->d_refined_doc_offsets,
            ws_->d_refined_token_positions,num_refined_candidates);
    } else {
    gather_refined_token_positions_kernel<<<
        num_refined_candidates, 256, 0, stream>>>(
        refined_ids,
        d_doc_ptrs_,
        d_token_to_cluster_pos_,
        ws_->d_refined_doc_offsets,
        ws_->d_refined_token_positions,
        num_refined_candidates
    );
    }
    check_cuda(cudaGetLastError());

    if (!ws_->uma_query_buffers) {
        check_cuda(cudaMemcpyAsync(
            ws_->h_num_refined_tokens,
            ws_->d_refined_doc_offsets + num_refined_candidates,
            sizeof(size_t),
            cudaMemcpyDeviceToHost,
            stream));
        check_cuda(cudaMemcpyAsync(
            ws_->h_pinned_refined_doc_offsets,
            ws_->d_refined_doc_offsets,
            (num_refined_candidates + 1) * sizeof(size_t),
            cudaMemcpyDeviceToHost,
            stream));

        check_cuda(cudaMemcpyAsync(
            ws_->h_refined_doc_ids,
            refined_ids,
            num_refined_candidates * sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));

        check_cuda(cudaStreamSynchronize(stream));
        total_tokens = *ws_->h_num_refined_tokens;
    }
    }
    // Ordinary/verification metadata is CPU-ready here. Direct scoring leaves
    // mapped IDs/diagnostic lengths GPU-owned until the existing final event;
    // constructing this CPU pointer does not read its not-yet-ready contents.
    const int* refined_doc_ids = ws_->h_refined_doc_ids;
    std::vector<int> direct_reference_ids;
    if (direct_docs && options_.verify_full_bit)
        direct_reference_ids.assign(refined_doc_ids, refined_doc_ids + num_refined_candidates);
    if (profiling_enabled_ && profile_query_id_ >= 0) {
        profile_refined_docs_ = size_t(num_refined_candidates);
        // The ordinary metadata handoff already provides the exact scan total.
        // Observe it in O(1), without adding a random doc-pointer walk before
        // the GPU token launch or an additional CUDA wait.
        if (!direct_docs) {
            profile_refined_tokens_ = total_tokens;
            if (ws_->h_pinned_refined_doc_offsets[num_refined_candidates] != total_tokens)
                throw std::runtime_error("Profile refined token count disagrees with actual document offsets");
        } // Direct counts are written by the scoring kernel and read after its existing event below.
    }

    // Score the chunks on the GPU while completed chunks stream to the CPU.
    int score_threads = 128;
    while (score_threads < Q_DOCLEN && score_threads < 256) score_threads *= 2;
    score_threads = std::min(score_threads, 256);

    const auto partition = ceil_chunk_partition(num_refined_candidates, options_.num_chunks);
    const int num_chunks = partition.first;
    const int chunk_size = partition.second;

    auto& chunk_compute_done = ws_->chunk_compute_done;
    auto& chunk_d2h_done = ws_->chunk_d2h_done;
    if (static_cast<size_t>(num_chunks) > chunk_compute_done.size() ||
        static_cast<size_t>(num_chunks) > chunk_d2h_done.size()) {
        throw std::runtime_error("Chunk event workspace capacity exceeded");
    }

    auto launch_chunk_compute = [&](int chunk) {
        ProfileRange gpu_range(profiling_enabled_, "Chimera/gpu_chunk_submission", chunk);
        const auto bounds = clamped_chunk_bounds(num_refined_candidates, chunk_size, chunk);
        const int chunk_start = bounds.first;
        const int chunk_end = bounds.second;
        const int chunk_count = chunk_end - chunk_start;
        const size_t token_start = metadata_ready
            ? ws_->h_pinned_refined_doc_offsets[chunk_start] : 0;
        const size_t token_end = metadata_ready
            ? ws_->h_pinned_refined_doc_offsets[chunk_end] : 0;
        const size_t token_count = token_end - token_start;

        if (options_.fused_token_document) {
            // chunks1 keeps these offsets and token positions relative to the
            // full query. Reuse only the first docs*32 floats as query-max
            // scratch; finalsum consumes it before an optional oracle rewrites
            // the same allocation as the full token matrix below.
            if (num_chunks != 1 || chunk_start != 0 || token_start != 0 ||
                static_cast<size_t>(chunk_count) > ws_->max_refined_tokens)
                throw std::runtime_error("Fused token/document scratch or chunks contract differs");
            const auto* codes = reinterpret_cast<const uint4*>(options_.doc_order_onebit
                ? d_doc_order_onebit_codes_ : d_clustered_one_bit_codes_);
            const float* factors = options_.doc_order_onebit
                ? d_doc_order_onebit_factors_ : d_clustered_one_bit_factors_;
            if (chunk_count > 0) {
                const int threads = options_.fused_token_document_threads;
                const size_t shared_bytes = options_.fused_token_document_layout == FusedTokenDocumentLayout::Warp4Doc
                    ? size_t(4104) * sizeof(float)
                    : (4104 + (threads / 32) * 8) * sizeof(float);
                if (direct_docs) {
                    if (!ws_->uma_query_buffers || !ws_->d_mapped_refined_doc_ids ||
                        (collect_direct_lengths && (!ws_->d_refined_doc_lengths || !ws_->h_refined_doc_lengths)))
                        throw std::runtime_error("Missing direct fusion mapped metadata buffers");
                    if (options_.fused_token_document_layout == FusedTokenDocumentLayout::Warp4Doc) {
                        if (threads != 128) throw std::runtime_error("Warp4doc direct launch requires 128 threads");
                        const dim3 grid((static_cast<unsigned>(chunk_count) + 3) / 4, 4);
                        fused_token_document_detail::fused_warp4_documents_direct<<<
                            grid, 128, shared_bytes, stream>>>(
                            ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                            refined_ids, d_doc_ptrs_, ws_->d_one_bit_token_scores,
                            ws_->d_mapped_refined_doc_ids,
                            collect_direct_lengths ? ws_->d_refined_doc_lengths : nullptr,
                            static_cast<size_t>(chunk_count));
                    } else {
                        const dim3 grid(static_cast<unsigned>(chunk_count), 4);
                        if (threads == 128) {
                            fused_token_document_detail::fused_token_doc_direct_groups<128><<<
                                grid, 128, shared_bytes, stream>>>(
                                ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                                refined_ids, d_doc_ptrs_, ws_->d_one_bit_token_scores,
                                ws_->d_mapped_refined_doc_ids,
                                collect_direct_lengths ? ws_->d_refined_doc_lengths : nullptr,
                                static_cast<size_t>(chunk_count));
                        } else {
                            fused_token_document_detail::fused_token_doc_direct_groups<256><<<
                                grid, 256, shared_bytes, stream>>>(
                                ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                                refined_ids, d_doc_ptrs_, ws_->d_one_bit_token_scores,
                                ws_->d_mapped_refined_doc_ids,
                                collect_direct_lengths ? ws_->d_refined_doc_lengths : nullptr,
                                static_cast<size_t>(chunk_count));
                        }
                    }
                } else {
                switch (options_.fused_token_document_layout) {
                    case FusedTokenDocumentLayout::Grouped: {
                        const dim3 grid(static_cast<unsigned>(chunk_count), 4);
                        if (threads == 128) {
                            fused_token_document_detail::fused_token_doc_groups<128><<<
                                grid, 128, shared_bytes, stream>>>(
                                    ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                                    ws_->d_refined_token_positions, ws_->d_refined_doc_offsets,
                                    ws_->d_one_bit_token_scores, static_cast<size_t>(chunk_count));
                        } else {
                            fused_token_document_detail::fused_token_doc_groups<256><<<
                                grid, 256, shared_bytes, stream>>>(
                                    ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                                    ws_->d_refined_token_positions, ws_->d_refined_doc_offsets,
                                    ws_->d_one_bit_token_scores, static_cast<size_t>(chunk_count));
                        }
                        break;
                    }
                    case FusedTokenDocumentLayout::Interleaved: {
                        const dim3 grid(static_cast<unsigned>(chunk_count) * 4, 1);
                        if (threads == 128) {
                            fused_token_document_detail::fused_token_doc_groups_interleaved<128><<<
                                grid, 128, shared_bytes, stream>>>(
                                    ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                                    ws_->d_refined_token_positions, ws_->d_refined_doc_offsets,
                                    ws_->d_one_bit_token_scores, static_cast<size_t>(chunk_count));
                        } else {
                            fused_token_document_detail::fused_token_doc_groups_interleaved<256><<<
                                grid, 256, shared_bytes, stream>>>(
                                    ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                                    ws_->d_refined_token_positions, ws_->d_refined_doc_offsets,
                                    ws_->d_one_bit_token_scores, static_cast<size_t>(chunk_count));
                        }
                        break;
                    }
                    case FusedTokenDocumentLayout::Cached: {
                        const dim3 grid(static_cast<unsigned>(chunk_count), 1);
                        if (threads == 128) {
                            fused_token_document_detail::fused_token_doc_cached<128><<<
                                grid, 128, shared_bytes, stream>>>(
                                    ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                                    ws_->d_refined_token_positions, ws_->d_refined_doc_offsets,
                                    ws_->d_one_bit_token_scores, static_cast<size_t>(chunk_count));
                        } else {
                            fused_token_document_detail::fused_token_doc_cached<256><<<
                                grid, 256, shared_bytes, stream>>>(
                                    ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                                    ws_->d_refined_token_positions, ws_->d_refined_doc_offsets,
                                    ws_->d_one_bit_token_scores, static_cast<size_t>(chunk_count));
                        }
                        break;
                    }
                    case FusedTokenDocumentLayout::Warp4Doc: {
                        if (threads != 128) throw std::runtime_error("Warp4doc launch requires 128 threads");
                        const dim3 grid((static_cast<unsigned>(chunk_count) + 3) / 4, 4);
                        fused_token_document_detail::fused_warp4_documents<<<
                            grid, 128, shared_bytes, stream>>>(
                            ws_->d_lut, codes, factors, ws_->d_cb1_sumq,
                            ws_->d_refined_token_positions, ws_->d_refined_doc_offsets,
                            ws_->d_one_bit_token_scores, static_cast<size_t>(chunk_count));
                        break;
                    }
                    default: throw std::runtime_error("Unknown fused token/document layout at launch");
                }
                }
                check_cuda(cudaGetLastError());
                fused_token_document_detail::fused_document_sum<<<chunk_count, 32, 0, stream>>>(
                    ws_->d_one_bit_token_scores, ws_->d_one_bit_document_scores,
                    static_cast<size_t>(chunk_count));
                check_cuda(cudaGetLastError());
                if (options_.verify_full_bit) {
                    if (!ws_->d_document_reduce_reference || !ws_->h_document_reduce_reference)
                        throw std::runtime_error("Missing fused token/document oracle buffers");
                    if (token_count > 0) {
                        const int token_blocks = static_cast<int>((token_count + 255) / 256);
                        const size_t reference_shared =
                            (DOCUMENT_SCORING_LUT_SMEM_FLOATS + DOCUMENT_SCORING_LUT_TILE_Q) * sizeof(float);
                        one_bit_token_score_kernel<<<token_blocks, 256, reference_shared, stream>>>(
                            ws_->d_lut, reinterpret_cast<const char*>(codes), factors,
                            ws_->d_cb1_sumq, ws_->d_refined_token_positions,
                            ws_->d_one_bit_token_scores, total_tokens, token_count);
                        check_cuda(cudaGetLastError());
                    }
                    one_bit_document_score_kernel<<<chunk_count, score_threads, 0, stream>>>(
                        ws_->d_one_bit_token_scores, ws_->d_refined_doc_offsets,
                        ws_->d_document_reduce_reference, total_tokens, chunk_count);
                    check_cuda(cudaGetLastError());
                    check_cuda(cudaMemcpyAsync(
                        ws_->h_document_reduce_reference, ws_->d_document_reduce_reference,
                        chunk_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
                }
            }
        } else {
        if (token_count > 0) {
            const int token_blocks = (token_count + 255) / 256;
            const size_t lut_shared_memory =
                DOCUMENT_SCORING_LUT_SMEM_FLOATS * sizeof(float) +
                DOCUMENT_SCORING_LUT_TILE_Q * sizeof(float);
            if (options_.split_token_score) {
                split_token_score_detail::token_split_query_group<8><<<
                    dim3(token_blocks, 4), 256, lut_shared_memory, stream>>>(
                    ws_->d_lut,
                    options_.doc_order_onebit ? d_doc_order_onebit_codes_ : d_clustered_one_bit_codes_,
                    options_.doc_order_onebit ? d_doc_order_onebit_factors_ : d_clustered_one_bit_factors_,
                    ws_->d_cb1_sumq,
                    ws_->d_refined_token_positions + token_start,
                    ws_->d_one_bit_token_scores + token_start,
                    total_tokens, token_count);
            } else if (options_.compact_token_score) {
                one_bit_token_score_compact_kernel<<<
                token_blocks, 256, lut_shared_memory, stream>>>(
                ws_->d_lut,
                options_.doc_order_onebit ? d_doc_order_onebit_codes_ : d_clustered_one_bit_codes_,
                options_.doc_order_onebit ? d_doc_order_onebit_factors_ : d_clustered_one_bit_factors_,
                ws_->d_cb1_sumq,
                ws_->d_refined_token_positions + token_start,
                ws_->d_one_bit_token_scores + token_start,
                total_tokens,
                token_count);
            } else {
            one_bit_token_score_kernel<<<
                token_blocks, 256, lut_shared_memory, stream>>>(
                ws_->d_lut,
                options_.doc_order_onebit ? d_doc_order_onebit_codes_ : d_clustered_one_bit_codes_,
                options_.doc_order_onebit ? d_doc_order_onebit_factors_ : d_clustered_one_bit_factors_,
                ws_->d_cb1_sumq,
                ws_->d_refined_token_positions + token_start,
                ws_->d_one_bit_token_scores + token_start,
                total_tokens,
                token_count);
            }
            check_cuda(cudaGetLastError());
        }

        if (chunk_count > 0) {
            if (ws_->cpu_document_enabled) {
                if (options_.verify_full_bit) {
                    one_bit_document_score_kernel<<<chunk_count, score_threads, 0, stream>>>(
                        ws_->d_one_bit_token_scores,
                        ws_->d_refined_doc_offsets + chunk_start,
                        ws_->d_document_reduce_reference + chunk_start,
                        total_tokens, chunk_count);
                    check_cuda(cudaGetLastError());
                    check_cuda(cudaMemcpyAsync(
                        ws_->h_document_reduce_reference + chunk_start,
                        ws_->d_document_reduce_reference + chunk_start,
                        chunk_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
                }
            } else if (options_.fast_document_reduce) {
                one_bit_document_score_fast_kernel<<<chunk_count, 128, 0, stream>>>(
                    ws_->d_one_bit_token_scores,
                    ws_->d_refined_doc_offsets + chunk_start,
                    ws_->d_one_bit_document_scores + chunk_start,
                    total_tokens, chunk_count);
                check_cuda(cudaGetLastError());
                if (options_.verify_full_bit && !options_.split_token_score) {
                    one_bit_document_score_kernel<<<chunk_count, score_threads, 0, stream>>>(
                        ws_->d_one_bit_token_scores,
                        ws_->d_refined_doc_offsets + chunk_start,
                        ws_->d_document_reduce_reference + chunk_start,
                        total_tokens, chunk_count);
                    check_cuda(cudaGetLastError());
                    check_cuda(cudaMemcpyAsync(
                        ws_->h_document_reduce_reference + chunk_start,
                        ws_->d_document_reduce_reference + chunk_start,
                        chunk_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
                }
            } else {
                one_bit_document_score_kernel<<<chunk_count, score_threads, 0, stream>>>(
                    ws_->d_one_bit_token_scores,
                    ws_->d_refined_doc_offsets + chunk_start,
                    ws_->d_one_bit_document_scores + chunk_start,
                    total_tokens, chunk_count);
                check_cuda(cudaGetLastError());
            }
        }

        if (options_.split_token_score && options_.verify_full_bit && chunk_count > 0) {
            // The primary token + GPU document result is complete in this
            // stream before reusing the token matrix for the original oracle.
            // CPU document is rejected because it would still need that matrix.
            if (!ws_->d_document_reduce_reference || !ws_->h_document_reduce_reference)
                throw std::runtime_error("Missing split token score oracle buffers");
            if (token_count > 0) {
                const int reference_blocks = static_cast<int>((token_count + 255) / 256);
                const size_t reference_shared =
                    (DOCUMENT_SCORING_LUT_SMEM_FLOATS + DOCUMENT_SCORING_LUT_TILE_Q) * sizeof(float);
                one_bit_token_score_kernel<<<reference_blocks, 256, reference_shared, stream>>>(
                    ws_->d_lut,
                    options_.doc_order_onebit ? d_doc_order_onebit_codes_ : d_clustered_one_bit_codes_,
                    options_.doc_order_onebit ? d_doc_order_onebit_factors_ : d_clustered_one_bit_factors_,
                    ws_->d_cb1_sumq, ws_->d_refined_token_positions + token_start,
                    ws_->d_one_bit_token_scores + token_start, total_tokens, token_count);
                check_cuda(cudaGetLastError());
            }
            one_bit_document_score_kernel<<<chunk_count, score_threads, 0, stream>>>(
                ws_->d_one_bit_token_scores, ws_->d_refined_doc_offsets + chunk_start,
                ws_->d_document_reduce_reference + chunk_start, total_tokens, chunk_count);
            check_cuda(cudaGetLastError());
            check_cuda(cudaMemcpyAsync(
                ws_->h_document_reduce_reference + chunk_start,
                ws_->d_document_reduce_reference + chunk_start,
                chunk_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }

        }

        check_cuda(cudaEventRecord(chunk_compute_done[chunk], stream));

        if (!ws_->uma_query_buffers && !ws_->cpu_document_enabled) {
            check_cuda(cudaStreamWaitEvent(
                stream_d2h, chunk_compute_done[chunk], 0));
            if (chunk_count > 0) check_cuda(cudaMemcpyAsync(
                ws_->h_mapped_one_bit_document_scores + chunk_start,
                ws_->d_one_bit_document_scores + chunk_start,
                chunk_count * sizeof(float),
                cudaMemcpyDeviceToHost, stream_d2h));
            check_cuda(cudaEventRecord(chunk_d2h_done[chunk], stream_d2h));
        }
    };

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        launch_chunk_compute(chunk);
    }

    std::priority_queue<std::pair<float, size_t>> full_bit_result_heap;
    auto& seen_doc_map = ws_->seen_doc_map;
    std::vector<int> seen_doc_list;
    seen_doc_list.reserve(full_bit_target * 2);
    int previous_order_size = 0;

    auto& one_bit_score_order = ws_->one_bit_score_order;
    auto& full_bit_candidate_indices = ws_->full_bit_candidate_indices;
    auto& full_bit_scores = ws_->full_bit_scores;

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const auto bounds = clamped_chunk_bounds(num_refined_candidates, chunk_size, chunk);
        const int chunk_start = bounds.first;
        const int chunk_end = bounds.second;
        const int chunk_count = chunk_end - chunk_start;
        {
            ProfileRange wait(profiling_enabled_, "Chimera/wait_chunk", chunk);
            check_cuda(cudaEventSynchronize((ws_->uma_query_buffers || ws_->cpu_document_enabled)
                ? chunk_compute_done[chunk] : chunk_d2h_done[chunk]));
        }

        if (direct_docs && chunk_count > 0) {
            if (options_.verify_full_bit) {
                if (direct_reference_ids.size() != static_cast<size_t>(num_refined_candidates))
                    throw std::runtime_error("Missing direct fusion reference document IDs");
                for (int i = chunk_start; i < chunk_end; ++i)
                    if (refined_doc_ids[i] != direct_reference_ids[i])
                        throw std::runtime_error("Direct fusion mapped document ID differs at candidate " + std::to_string(i));
            }
            if (collect_direct_lengths) {
                size_t actual_tokens = 0;
                for (int i = chunk_start; i < chunk_end; ++i) {
                    const int length = ws_->h_refined_doc_lengths[i];
                    if (length < 0) throw std::runtime_error("Direct fusion wrote a negative document length");
                    actual_tokens += static_cast<size_t>(length);
                }
                if (options_.verify_full_bit && actual_tokens != total_tokens)
                    throw std::runtime_error("Direct fusion token count differs from original metadata");
                if (profiling_enabled_ && profile_query_id_ >= 0)
                    profile_refined_tokens_ = actual_tokens;
                if (options_.verify_full_bit)
                    std::cout << "[FUSED_DIRECT_DOCS_VERIFY] candidates=" << chunk_count
                              << " tokens=" << actual_tokens << " ids_exact=passed token_count=passed\n";
            }
        }

        if (ws_->cpu_document_enabled && chunk_count > 0) {
            if (!cpu_task_pool_) throw std::runtime_error("CPU document worker pool is unavailable");
            const bool qid_profile = profiling_enabled_ && profile_query_id_ >= 0;
            const int64_t qid = profile_query_id_;
            ProfileRange cpu_reduce(profiling_enabled_, "Chimera/cpu_document_pool_reduce",
                                    qid_profile ? qid : int64_t(chunk));
            constexpr size_t docs_per_tile = 32;
            const size_t tile_count = (static_cast<size_t>(chunk_count) + docs_per_tile - 1) / docs_per_tile;
            if (qid_profile) profile_document_tiles_ += tile_count;
            const float* scores = ws_->h_cpu_document_token_scores;
            const size_t* offsets = ws_->h_pinned_refined_doc_offsets;
            float* outputs = ws_->h_mapped_one_bit_document_scores;
            auto group = cpu_task_pool_->submit(tile_count, 1,
                [scores, offsets, outputs, total_tokens, chunk_start, chunk_end, qid_profile, qid](size_t tile) {
                    ProfileRange work(qid_profile, "Chimera/cpu_document_worker", qid);
                    const size_t first = static_cast<size_t>(chunk_start) + tile * size_t(32);
                    const size_t count = std::min(size_t(32), static_cast<size_t>(chunk_end) - first);
                    score_one_bit_documents_cpu_pool(scores, offsets + first,
                        outputs + first, total_tokens, count);
                });
            group->wait();
        }

        if ((options_.split_token_score || options_.fast_document_reduce || options_.fused_token_document ||
             ws_->cpu_document_enabled) && options_.verify_full_bit) {
            for (int i = chunk_start; i < chunk_end; ++i) {
                if (std::memcmp(&ws_->h_document_reduce_reference[i],
                                &ws_->h_mapped_one_bit_document_scores[i], sizeof(float))) {
                    throw std::runtime_error("Document reduction differs from original at candidate " +
                                             std::to_string(i));
                }
            }
            std::cout << (options_.split_token_score ? "[SPLIT_TOKEN_SCORE_VERIFY] candidates="
                            : options_.fused_token_document ? "[FUSED_TOKEN_DOCUMENT_VERIFY] candidates="
                            : ws_->cpu_document_enabled ? "[CPU_DOCUMENT_POOL_VERIFY] candidates="
                                                        : "[FAST_DOCUMENT_REDUCE_VERIFY] candidates=") << chunk_count;
            if (options_.fused_token_document)
                std::cout << " threads=" << options_.fused_token_document_threads
                          << " layout=" << fused_token_document_layout_name(options_.fused_token_document_layout)
                          << " direct_docs=" << (options_.fused_direct_docs ? 1 : 0);
            std::cout << " scores_fp32_bitwise=passed\n";
        }

        int num_full_bit_candidates = 0;
        {
        ProfileRange cpu_select(profiling_enabled_, "Chimera/cpu_candidate_select", chunk);
        std::iota(
            one_bit_score_order.begin() + previous_order_size,
            one_bit_score_order.begin() + previous_order_size + chunk_count,
            chunk_start);
        const int merge_size = previous_order_size + chunk_count;
        const int carry_size = std::min(full_bit_target, merge_size);
        const float* one_bit_scores = ws_->h_mapped_one_bit_document_scores;
        auto descending_one_bit_score = [one_bit_scores](int a, int b) {
            return one_bit_scores[a] > one_bit_scores[b];
        };
        std::nth_element(
            one_bit_score_order.begin(),
            one_bit_score_order.begin() + carry_size,
            one_bit_score_order.begin() + merge_size,
            descending_one_bit_score);
        const int cumulative_full_bit_target = std::min(
            int((int64_t(full_bit_target) * (chunk + 1)) / num_chunks), carry_size);
        std::nth_element(
            one_bit_score_order.begin(),
            one_bit_score_order.begin() + cumulative_full_bit_target,
            one_bit_score_order.begin() + carry_size,
            descending_one_bit_score);
        previous_order_size = carry_size;

        std::vector<std::pair<float, int>> newly_selected_candidates;
        newly_selected_candidates.reserve(cumulative_full_bit_target);
        for (int i = 0; i < cumulative_full_bit_target; ++i) {
            const int refined_index = one_bit_score_order[i];
            const int doc_id = refined_doc_ids[refined_index];
            if (!seen_doc_map[doc_id]) {
                newly_selected_candidates.emplace_back(
                    one_bit_scores[refined_index], refined_index);
            }
        }

        if (newly_selected_candidates.empty()) {
            continue;
        }
        std::sort(
            newly_selected_candidates.begin(),
            newly_selected_candidates.end(),
            std::greater<>());
        num_full_bit_candidates = static_cast<int>(newly_selected_candidates.size());
        for (const auto& candidate : newly_selected_candidates) {
            const int doc_id = refined_doc_ids[candidate.second];
            seen_doc_map[doc_id] = true;
            seen_doc_list.push_back(doc_id);
        }

        for (int i = 0; i < num_full_bit_candidates; ++i) {
            full_bit_candidate_indices[i] =
                newly_selected_candidates[i].second;
        }

        }

        if (profiling_enabled_ && profile_query_id_ >= 0) {
            profile_fullbit_candidates_ += size_t(num_full_bit_candidates);
            const auto& ptrs = corpus().doc_ptrs_;
            for (int i = 0; i < num_full_bit_candidates; ++i) {
                const int doc = refined_doc_ids[full_bit_candidate_indices[i]];
                profile_fullbit_tokens_ += size_t(ptrs[doc + 1] - ptrs[doc]);
            }
        }

        if (defer_fullbit_in_batch_) {
            if (num_chunks != 1 || options_.uma_cpu_fullbit_fraction != 1.0 || pending_fullbit_)
                throw std::logic_error("Invalid deferred fullbit batch state");
            // Copy every per-query input before releasing this workspace.
            // The factory also captures GPU verification while d_queries is
            // still the current query, then submits independent CPU work.
            if (profiling_enabled_ && profile_query_id_ >= 0)
                profile_fullbit_tasks_ += size_t(num_full_bit_candidates);
            pending_fullbit_ = submit_deferred_fullbit(
                refined_doc_ids, full_bit_candidate_indices.data(),
                num_full_bit_candidates, ws_->h_pinned_queries, queries);
            for (int doc_id : seen_doc_list) seen_doc_map[doc_id] = false;
            result.clear();
            last_result_scores_.clear();
            return;
        }

        compute_full_bit_scores(
            refined_doc_ids,
            full_bit_candidate_indices.data(),
            num_full_bit_candidates,
            ws_->h_pinned_queries,
            queries,
            full_bit_scores.data());

        {
        ProfileRange cpu_heap(profiling_enabled_, "Chimera/cpu_heap_insert", chunk);
        for (int i = 0; i < num_full_bit_candidates; ++i) {
            full_bit_result_heap.emplace(
                full_bit_scores[i].first,
                static_cast<size_t>(full_bit_scores[i].second));
        }
        }
    }

    {
    ProfileRange cpu_final(profiling_enabled_, "Chimera/cpu_final_select");
    result.clear();
    last_result_scores_.clear();
    if (capture_scores_) last_result_scores_.reserve(k);
    for (size_t i = 0; i < k && !full_bit_result_heap.empty(); ++i) {
        result.push_back(full_bit_result_heap.top().second);
        if (capture_scores_) last_result_scores_.push_back(full_bit_result_heap.top().first);
        full_bit_result_heap.pop();
    }

    for (int doc_id : seen_doc_list) seen_doc_map[doc_id] = false;
    }

}


}  // namespace Chimera
