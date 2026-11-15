#include "chimera/spilled_full_codes.hpp"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>

#include "chimera/chimera_index.cuh"

#define EIGEN_NO_CUDA

#include "chimera/config.cuh"
#include "chimera/ivf_pg.hpp"
#include "chimera/kernels.cuh"
#include "chimera/index_file_io.hpp"
#include "chimera_index_internal.cuh"
#include "rabitqlib/utils/rotator.hpp"

#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>

namespace Chimera {

namespace {

// Ordinary CPU-cached storage, addressed directly through ATS host page tables.
// This path is enabled only after checking pageableMemoryAccess/hostPageTables.
// The host allocation is the only owner; neither alias is passed to cudaFree.
template <typename T>
void allocate_ats_buffer(T*& host, T*& device, size_t count) {
    void* storage = nullptr;
    const size_t bytes = count * sizeof(T);
    if (::posix_memalign(&storage, 256, std::max(bytes, size_t(256))) != 0) {
        throw std::bad_alloc();
    }
    host = static_cast<T*>(storage);
    // Initialize and CPU-first-touch every page outside query timing.
    std::memset(host, 0, bytes);
    device = host;
}

struct cast_int_size_t {
    __host__ __device__ size_t operator()(int value) const {
        return static_cast<size_t>(value);
    }
};

constexpr size_t kCandidateRowAlignment = 1u << 16;  // 65536 rows
constexpr size_t kCandidateMinSlackRows = 1u << 15;  // 32768 rows
constexpr size_t kCandidateArenaAlignment = 128;
constexpr size_t kCagraWorkspacePoolFloorBytes = 64ull * 1024ull * 1024ull;

void configure_cagra_workspace(raft::resources& res, size_t estimated_workspace_bytes) {
    constexpr size_t kPoolAlignment = 16ull * 1024ull * 1024ull;
    const size_t aligned_estimate =
        ((estimated_workspace_bytes + kPoolAlignment - 1) / kPoolAlignment) * kPoolAlignment;
    const size_t cagra_pool_bytes = std::max(
        kCagraWorkspacePoolFloorBytes,
        aligned_estimate);
    raft::resource::set_workspace_to_pool_resource(res, cagra_pool_bytes);
}

size_t sum_largest_counts(std::vector<size_t>& counts, size_t top_n) {
    if (counts.empty() || top_n == 0) {
        return 0;
    }

    std::sort(counts.begin(), counts.end(), std::greater<size_t>());
    top_n = std::min(top_n, counts.size());
    return std::accumulate(counts.begin(), counts.begin() + top_n, size_t{0});
}

size_t compute_refined_token_bound(
    const std::vector<int>& doc_ptrs,
    size_t k_refine) {
    if (doc_ptrs.size() < 2 || k_refine == 0) {
        return 0;
    }

    std::vector<size_t> doc_lengths(doc_ptrs.size() - 1, 0);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int doc_idx = 0; doc_idx < static_cast<int>(doc_lengths.size()); ++doc_idx) {
        doc_lengths[doc_idx] =
            static_cast<size_t>(doc_ptrs[doc_idx + 1] - doc_ptrs[doc_idx]);
    }

    std::sort(doc_lengths.begin(), doc_lengths.end(), std::greater<size_t>());

    return std::accumulate(
        doc_lengths.begin(),
        doc_lengths.begin() + std::min(k_refine, doc_lengths.size()),
        size_t{0});
}

size_t align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

char* align_up_ptr(char* ptr, size_t alignment) {
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const auto aligned = align_up(addr, alignment);
    return reinterpret_cast<char*>(aligned);
}

size_t compute_initial_candidate_capacity(
    size_t num_tokens,
    size_t num_clusters,
    size_t max_nprobe,
    size_t max_candidate_docs) {
    if (max_candidate_docs == 0 || num_clusters == 0 || max_nprobe == 0) {
        return 1;
    }

    const size_t avg_cluster_tokens =
        std::max<size_t>(1, (num_tokens + num_clusters - 1) / num_clusters);

    size_t estimate = avg_cluster_tokens;
    if (estimate > std::numeric_limits<size_t>::max() / max_nprobe) {
        estimate = max_candidate_docs;
    } else {
        estimate *= max_nprobe;
    }

    if (estimate > std::numeric_limits<size_t>::max() / 4) {
        estimate = max_candidate_docs;
    } else {
        estimate *= 4;
    }

    estimate = std::min(max_candidate_docs, estimate);
    estimate = std::min(
        max_candidate_docs, align_up(estimate, kCandidateRowAlignment));
    return std::max<size_t>(estimate, 1);
}

size_t grow_candidate_capacity(size_t current, size_t required, size_t limit) {
    const size_t aligned_required =
        align_up(std::max(required, size_t{1}), kCandidateRowAlignment);
    const size_t required_slack =
        std::max(required / 16, kCandidateMinSlackRows);
    const size_t current_slack =
        std::max(current / 8, kCandidateMinSlackRows);
    size_t target = aligned_required;
    target = std::max(
        target,
        align_up(required + required_slack, kCandidateRowAlignment));
    if (current > 0) {
        target = std::max(
            target,
            align_up(current + current_slack, kCandidateRowAlignment));
    }
    return std::min(limit, target);
}

size_t candidate_arena_bytes(
    size_t row_capacity,
    size_t score_buffer_capacity,
    bool no_cr) {
    if (no_cr) return align_up(row_capacity * sizeof(int), kCandidateArenaAlignment) + row_capacity * sizeof(int);
    size_t offset = 0;
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += row_capacity * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += row_capacity * Q_DOCLEN * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += score_buffer_capacity * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    offset += score_buffer_capacity * sizeof(int);
    return offset;
}


int doc_id_from_doc_ptrs_host_binary_range(
    const std::vector<int>& doc_ptrs,
    int lo,
    int hi,
    uint32_t token_id) {
    while (lo < hi) {
        const int mid = lo + ((hi - lo + 1) >> 1);
        const uint32_t mid_start = static_cast<uint32_t>(doc_ptrs[mid]);
        if (mid_start <= token_id) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

int doc_id_from_doc_ptrs_host(
    const std::vector<int>& doc_ptrs,
    uint32_t token_id) {
    return doc_id_from_doc_ptrs_host_binary_range(
        doc_ptrs,
        0,
        static_cast<int>(doc_ptrs.size()) - 2,
        token_id);
}

}  // namespace

void chimera_index::bind_candidate_arena(
    char* base,
    size_t row_capacity,
    size_t score_buffer_capacity) {
    if (skips_candidate_refinement(options_.algorithm_variant)) {
        ws_->d_materialized_candidate_doc_ids = reinterpret_cast<int*>(base);
        ws_->d_candidate_doc_ids = reinterpret_cast<int*>(base + align_up(row_capacity * sizeof(int), kCandidateArenaAlignment));
        ws_->d_sorted_candidate_doc_ids = nullptr;
        ws_->d_partial_scores = nullptr;
        ws_->d_partial_query_scores = nullptr;
        ws_->d_sorted_partial_scores = nullptr;
        return;
    }
    size_t offset = 0;
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_sorted_candidate_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_materialized_candidate_doc_ids = reinterpret_cast<int*>(base + offset);
    offset += row_capacity * sizeof(int);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_partial_scores = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_partial_query_scores = reinterpret_cast<float*>(base + offset);
    offset += row_capacity * Q_DOCLEN * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_sorted_partial_scores = reinterpret_cast<float*>(base + offset);
    offset += score_buffer_capacity * sizeof(float);
    offset = align_up(offset, kCandidateArenaAlignment);
    ws_->d_candidate_doc_ids = reinterpret_cast<int*>(base + offset);
}

void chimera_index::compute_candidate_bounds() {
    const size_t probe_cluster_bound =
        std::min(static_cast<size_t>(options_.nprobe), n_clusters);
    max_probed_clusters_ = probe_cluster_bound;
    if (probe_cluster_bound == 0) {
        max_retrieved_tokens_per_query_ = 0;
        max_candidate_docs_per_query_ = 0;
        return;
    }

    std::vector<size_t> cluster_token_counts(n_clusters, 0);
    std::vector<size_t> cluster_unique_doc_counts(n_clusters, 0);

#pragma omp parallel for schedule(dynamic, 256)
    for (int cluster_idx = 0; cluster_idx < static_cast<int>(n_clusters); ++cluster_idx) {
        const size_t cluster_id = static_cast<size_t>(cluster_idx);
        const size_t start = ivf->cluster_pos[cluster_id];
        const size_t end = ivf->cluster_pos[cluster_id + 1];
        const size_t token_count = end - start;
        cluster_token_counts[cluster_id] = token_count;

        if (token_count == 0) {
            cluster_unique_doc_counts[cluster_id] = 0;
            continue;
        }

        const auto first_token_id = static_cast<uint32_t>(ivf->inv_list[start]);
        int prev_doc_id = doc_id_from_doc_ptrs_host(doc_ptrs_, first_token_id);
        size_t unique_doc_count = 1;

        for (size_t pos = start + 1; pos < end; ++pos) {
            const uint32_t token_id = static_cast<uint32_t>(ivf->inv_list[pos]);
            const int current_doc_id = doc_id_from_doc_ptrs_host_binary_range(
                doc_ptrs_, prev_doc_id, static_cast<int>(num_docs) - 1, token_id);
            if (current_doc_id != prev_doc_id) {
                ++unique_doc_count;
                prev_doc_id = current_doc_id;
            }
        }

        cluster_unique_doc_counts[cluster_id] = unique_doc_count;
    }

    max_retrieved_tokens_per_query_ =
        sum_largest_counts(cluster_token_counts, probe_cluster_bound);
    max_candidate_docs_per_query_ =
        sum_largest_counts(cluster_unique_doc_counts, probe_cluster_bound);
}

void chimera_index::ensure_candidate_capacity(size_t required_rows) {
    required_rows = std::max<size_t>(required_rows, 1);
    if (required_rows <= candidate_capacity_ &&
        d_candidate_arena_raw_ != nullptr) {
        return;
    }

    const size_t grown_rows =
        (candidate_capacity_ == 0)
            ? std::min(
                  static_cast<size_t>(num_docs),
                  align_up(required_rows, kCandidateRowAlignment))
            : grow_candidate_capacity(
                  candidate_capacity_, required_rows, static_cast<size_t>(num_docs));
    const size_t score_buffer_capacity =
        std::max(grown_rows, ws_->max_refined_candidates);
    const size_t required_buffer_bytes =
        candidate_arena_bytes(grown_rows, score_buffer_capacity, skips_candidate_refinement(options_.algorithm_variant));

    check_cuda(cudaFree(d_candidate_arena_raw_));
    d_candidate_arena_raw_ = nullptr;
    d_candidate_arena_ = nullptr;
    check_cuda(cudaMalloc(
        &d_candidate_arena_raw_,
        required_buffer_bytes + kCandidateArenaAlignment - 1));
    d_candidate_arena_ =
        align_up_ptr(d_candidate_arena_raw_, kCandidateArenaAlignment);
    bind_candidate_arena(
        d_candidate_arena_, grown_rows, score_buffer_capacity);

    candidate_capacity_ = grown_rows;
}

void chimera_index::allocate_workspace() {
    if (options_.uma_cpu_document) {
        int device = 0, pageable = 0, host_tables = 0;
        check_cuda(cudaGetDevice(&device));
        check_cuda(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device));
        check_cuda(cudaDeviceGetAttribute(&host_tables,
                   cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
        if (!pageable || !host_tables)
            throw std::runtime_error("CPU document scoring requires ATS host-page-table access");
        ws_->cpu_document_enabled = true;
    }
    if (options_.uma_query_buffers) {
        int device = 0, pageable = 0, host_tables = 0;
        check_cuda(cudaGetDevice(&device));
        check_cuda(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device));
        check_cuda(cudaDeviceGetAttribute(&host_tables,
                   cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
        if (!pageable || !host_tables) {
            throw std::runtime_error("UMA query buffers require ATS host-page-table access");
        }
        ws_->uma_query_buffers = true;
    }
    ws_->max_full_bit_candidates = options_.k_full_bit;
    const size_t max_scored_candidate_docs = std::min(
        static_cast<size_t>(num_docs),
        max_retrieved_tokens_per_query_ * Q_DOCLEN);

    max_candidate_docs_ = std::min(
        static_cast<size_t>(num_docs),
        max_candidate_docs_per_query_ * Q_DOCLEN);
    ws_->max_refined_candidates = skips_candidate_refinement(options_.algorithm_variant)
        ? max_candidate_docs_ : size_t(options_.k_refine);
    ws_->max_refined_tokens = (batch_owner_ ? batch_owner_->ws_->max_refined_tokens :
        compute_refined_token_bound(doc_ptrs_, ws_->max_refined_candidates));
    candidate_bitmap_bucket_count_ = candidate_bitmap_num_buckets(num_docs);
    candidate_bitmap_offset_count_ =
        candidate_bitmap_num_offsets(candidate_bitmap_bucket_count_);
    const size_t initial_candidate_capacity = compute_initial_candidate_capacity(
        n,
        n_clusters,
        max_probed_clusters_,
        max_candidate_docs_);
    const size_t candidate_bound_capacity =
        (max_candidate_docs_per_query_ > 0)
            ? grow_candidate_capacity(
                  0,
                  std::min(max_candidate_docs_per_query_, max_candidate_docs_),
                  max_candidate_docs_)
            : initial_candidate_capacity;
    const size_t candidate_rows =
        std::max(initial_candidate_capacity, candidate_bound_capacity);

    const size_t doc_lengths_bytes =
        ws_->max_refined_candidates * sizeof(int);

    const size_t estimated_size =
        Q_DOCLEN * PADDED_DIM * sizeof(float) +
        Q_DOCLEN * sizeof(float) +
        doc_lengths_bytes +
        candidate_rows * sizeof(int) +
        candidate_rows * sizeof(int) +
        (skips_candidate_refinement(options_.algorithm_variant) ? 0 : candidate_rows * sizeof(float)) +
        (skips_candidate_refinement(options_.algorithm_variant) ? 0 : candidate_rows * Q_DOCLEN * sizeof(float)) +
        candidate_bitmap_bucket_count_ * sizeof(candidate_bitmap_bucket_t) +
        candidate_bitmap_offset_count_ * sizeof(candidate_bitmap_offset_t);
    configure_cagra_workspace(*cagra_res_, estimated_size);

    if (ws_->uma_query_buffers) {
        allocate_ats_buffer(ws_->h_pinned_queries, ws_->d_queries, Q_DOCLEN * PADDED_DIM);
        allocate_ats_buffer(ws_->h_pinned_cb1_sumq, ws_->d_cb1_sumq, size_t(Q_DOCLEN));
        allocate_ats_buffer(ws_->h_pinned_refined_doc_offsets, ws_->d_refined_doc_offsets,
                               ws_->max_refined_candidates + 1);
        allocate_ats_buffer(ws_->h_candidate_bitmap_offsets, d_candidate_bitmap_offsets_,
                               candidate_bitmap_offset_count_);
    } else {
        check_cuda(cudaMalloc(&ws_->d_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
        check_cuda(cudaMalloc(&ws_->d_cb1_sumq, Q_DOCLEN * sizeof(float)));
    }
    if (ws_->uma_query_buffers) {
        allocate_ats_buffer(ws_->h_refined_doc_lengths, ws_->d_refined_doc_lengths,
                            ws_->max_refined_candidates);
    } else {
        check_cuda(cudaMalloc(&ws_->d_refined_doc_lengths, ws_->max_refined_candidates * sizeof(int)));
    }
    if (!ws_->uma_query_buffers) {
        check_cuda(cudaMalloc(&ws_->d_refined_doc_offsets,
                              (ws_->max_refined_candidates + 1) * sizeof(size_t)));
    }
    if (options_.uma_cpu_candidates) {
        allocate_ats_buffer(ws_->h_cpu_candidate_bitmap, d_candidate_bitmap_, candidate_bitmap_bucket_count_);
        int* alias = nullptr;
        allocate_ats_buffer(ws_->h_cpu_materialized_ids, alias, max_candidate_docs_);
        if (options_.uma_cpu_refine) {
            float* score_alias = nullptr;
            allocate_ats_buffer(ws_->h_cpu_refine_scores, score_alias, max_candidate_docs_);
            ws_->cpu_refine_order.resize(max_candidate_docs_);
        }
    } else {
        check_cuda(cudaMalloc(&d_candidate_bitmap_,
                              candidate_bitmap_bucket_count_ * sizeof(candidate_bitmap_bucket_t)));
    }
    if (!ws_->uma_query_buffers) {
        check_cuda(cudaMalloc(&d_candidate_bitmap_offsets_,
                              candidate_bitmap_offset_count_ * sizeof(candidate_bitmap_offset_t)));
    }

    ws_->d_cub_temp_storage = nullptr;
    ws_->cub_temp_storage_bytes = 0;
    candidate_capacity_ = 0;
    size_t temp_bytes = 0;

    if (!skips_candidate_refinement(options_.algorithm_variant)) {
    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        temp_bytes,
        ws_->d_partial_scores,
        ws_->d_sorted_partial_scores,
        ws_->d_candidate_doc_ids,
        ws_->d_sorted_candidate_doc_ids,
        static_cast<int>(std::max<size_t>(max_scored_candidate_docs, 1)),
        0,
        32);
    ws_->cub_temp_storage_bytes = std::max(ws_->cub_temp_storage_bytes, temp_bytes);
    }

    {
        thrust::device_ptr<int> len_ptr(ws_->d_refined_doc_lengths);
        auto xform_iter = thrust::make_transform_iterator(len_ptr, cast_int_size_t());
        temp_bytes = 0;
        cub::DeviceScan::InclusiveScan(
            nullptr,
            temp_bytes,
            xform_iter,
            ws_->d_refined_doc_offsets + 1,
            thrust::plus<size_t>(),
            static_cast<int>(ws_->max_refined_candidates));
        ws_->cub_temp_storage_bytes = std::max(ws_->cub_temp_storage_bytes, temp_bytes);
    }

    temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_bytes,
        d_candidate_bitmap_offsets_,
        d_candidate_bitmap_offsets_,
        candidate_bitmap_offset_count_);
    ws_->cub_temp_storage_bytes = std::max(ws_->cub_temp_storage_bytes, temp_bytes);

    check_cuda(cudaMalloc(&ws_->d_cub_temp_storage, ws_->cub_temp_storage_bytes));

    ensure_candidate_capacity(candidate_rows);


    check_cuda(cudaMalloc(&ws_->d_lut, LUT_TOTAL_FLOATS * sizeof(float)));
    const size_t document_scoring_lut_smem =
        DOCUMENT_SCORING_LUT_SMEM_FLOATS * sizeof(float) +
        DOCUMENT_SCORING_LUT_TILE_Q * sizeof(float);
    check_cuda(cudaFuncSetAttribute(
        one_bit_token_score_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        document_scoring_lut_smem));
    if (options_.compact_token_score) {
        check_cuda(cudaFuncSetAttribute(one_bit_token_score_compact_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, document_scoring_lut_smem));
    }


    check_cuda(cudaMalloc(
        &ws_->d_cagra_dists,
        Q_DOCLEN * max_probed_clusters_ * sizeof(float)));
    if (options_.uma_cpu_candidates) {
        allocate_ats_buffer(ws_->h_cpu_cagra_labels, ws_->d_cagra_labels,
                            Q_DOCLEN * max_probed_clusters_);
    } else {
        check_cuda(cudaMalloc(&ws_->d_cagra_labels,
                             Q_DOCLEN * max_probed_clusters_ * sizeof(uint32_t)));
    }

    if (!ws_->uma_query_buffers) {
        check_cuda(cudaMallocHost(&ws_->h_pinned_queries, Q_DOCLEN * PADDED_DIM * sizeof(float)));
        check_cuda(cudaMallocHost(&ws_->h_pinned_cb1_sumq, Q_DOCLEN * sizeof(float)));
    }

    if (!options_.uma_cpu_onebit_fused || options_.verify_full_bit)
    check_cuda(cudaMalloc(&ws_->d_refined_token_positions,
                          ws_->max_refined_tokens * sizeof(uint32_t)));
    if (ws_->cpu_document_enabled) {
        constexpr size_t page = size_t(2) << 20;
        const size_t factors = static_cast<size_t>(Q_DOCLEN) * sizeof(float);
        if (ws_->max_refined_tokens > (std::numeric_limits<size_t>::max() - (page - 1)) / factors)
            throw std::overflow_error("CPU document ATS matrix size overflow");
        const size_t bytes = ws_->max_refined_tokens * factors;
        const size_t rounded = std::max(page, (bytes + page - 1) & ~(page - 1));
        void* storage = nullptr;
        if (::posix_memalign(&storage, page, rounded) != 0) throw std::bad_alloc();
        ws_->h_cpu_document_token_scores = static_cast<float*>(storage);
        (void)::madvise(storage, rounded, MADV_HUGEPAGE);
        std::memset(storage, 0, rounded);
        ws_->d_one_bit_token_scores = ws_->h_cpu_document_token_scores;
    } else if (!options_.uma_cpu_onebit_fused || options_.verify_full_bit) {
        check_cuda(cudaMalloc(&ws_->d_one_bit_token_scores,
                              ws_->max_refined_tokens * static_cast<size_t>(Q_DOCLEN) * sizeof(float)));
    }
    if (ws_->uma_query_buffers) {
        allocate_ats_buffer(ws_->h_mapped_one_bit_document_scores,
                               ws_->d_one_bit_document_scores, ws_->max_refined_candidates);
        allocate_ats_buffer(ws_->h_refined_doc_ids, ws_->d_mapped_refined_doc_ids,
                               ws_->max_refined_candidates);
    } else {
        check_cuda(cudaMalloc(&ws_->d_one_bit_document_scores, ws_->max_refined_candidates * sizeof(float)));
        check_cuda(cudaMallocHost(&ws_->h_mapped_one_bit_document_scores,
                                  ws_->max_refined_candidates * sizeof(float)));
        check_cuda(cudaMallocHost(&ws_->h_pinned_refined_doc_offsets,
                                  (ws_->max_refined_candidates + 1) * sizeof(size_t)));
        check_cuda(cudaMallocHost(&ws_->h_refined_doc_ids,
                                  ws_->max_refined_candidates * sizeof(int)));
        check_cuda(cudaMallocHost(&ws_->h_num_refined_tokens, sizeof(size_t)));
    }

    if ((options_.fast_document_reduce || ws_->cpu_document_enabled || options_.uma_cpu_onebit_fused || options_.uma_cpu_onebit_half) && options_.verify_full_bit) {
        check_cuda(cudaMalloc(&ws_->d_document_reduce_reference,
                              ws_->max_refined_candidates * sizeof(float)));
        check_cuda(cudaMallocHost(&ws_->h_document_reduce_reference,
                                  ws_->max_refined_candidates * sizeof(float)));
    }

    if (!skips_candidate_refinement(options_.algorithm_variant)) check_cuda(cudaMemset(
        ws_->d_partial_query_scores,
        0,
        candidate_rows * Q_DOCLEN * sizeof(float)));
    if (options_.uma_cpu_candidates)
        std::memset(d_candidate_bitmap_, 0, candidate_bitmap_bucket_count_ * sizeof(candidate_bitmap_bucket_t));
    else
        check_cuda(cudaMemset(d_candidate_bitmap_, 0,
                              candidate_bitmap_bucket_count_ * sizeof(candidate_bitmap_bucket_t)));
    if (ws_->uma_query_buffers) {
        std::memset(ws_->h_candidate_bitmap_offsets, 0,
                    candidate_bitmap_offset_count_ * sizeof(candidate_bitmap_offset_t));
    } else {
        check_cuda(cudaMemset(d_candidate_bitmap_offsets_, 0,
                              candidate_bitmap_offset_count_ * sizeof(candidate_bitmap_offset_t)));
    }

    int least_prio, greatest_prio;
    check_cuda(cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio));
    check_cuda(cudaStreamCreateWithPriority(&ws_->stream_compute, cudaStreamNonBlocking, least_prio));
    check_cuda(cudaStreamCreateWithFlags(&ws_->stream_h2d, cudaStreamNonBlocking));
    check_cuda(cudaStreamCreateWithFlags(&ws_->stream_d2h, cudaStreamNonBlocking));
    check_cuda(cudaEventCreate(&ws_->event_h2d_done));

    // A query can use fewer chunks than requested when candidate count is
    // small. Reserve the maximum possible count once, including for NoCR.
    const size_t max_chunks = std::min({
        static_cast<size_t>(options_.num_chunks),
        ws_->max_refined_candidates,
        num_docs});
    // Initialize every slot before creating handles: release_resources() also
    // handles a failure halfway through either vector's event creation.
    ws_->chunk_compute_done.resize(max_chunks, nullptr);
    ws_->chunk_d2h_done.resize(max_chunks, nullptr);
    for (size_t chunk = 0; chunk < max_chunks; ++chunk) {
        check_cuda(cudaEventCreateWithFlags(
            &ws_->chunk_compute_done[chunk], cudaEventDisableTiming));
        check_cuda(cudaEventCreateWithFlags(
            &ws_->chunk_d2h_done[chunk], cudaEventDisableTiming));
    }


    ws_->one_bit_score_order.resize(ws_->max_refined_candidates);
    ws_->full_bit_candidate_indices.resize(ws_->max_full_bit_candidates);
    ws_->full_bit_scores.resize(ws_->max_full_bit_candidates);
    ws_->seen_doc_map.resize(num_docs, false);
    if (options_.gpu_full_bit || options_.verify_full_bit) initialize_gpu_full_bit();
}

void chimera_index::set_doc_mapping(const std::vector<int>& doc_lens) {
    if (doc_lens.empty()) {
        throw std::runtime_error("Chimera index has no documents");
    }
    num_docs = doc_lens.size();
    doc_ptrs_.resize(num_docs + 1, 0);
    for (size_t i = 0; i < num_docs; ++i) {
        if (doc_lens[i] <= 0 ||
            doc_ptrs_[i] > std::numeric_limits<int>::max() - doc_lens[i]) {
            throw std::runtime_error("Invalid document lengths in Chimera index");
        }
        doc_ptrs_[i + 1] = doc_ptrs_[i] + doc_lens[i];
    }
    if (static_cast<size_t>(doc_ptrs_.back()) != n) {
        throw std::runtime_error(
            "Document lengths do not match the number of indexed embeddings");
    }
}

void chimera_index::release_resources() noexcept {
    if (search_ready_) {
        (void)cudaDeviceSynchronize();
    }

    // Internal batch workers must finish and release their private workspaces
    // before any shared corpus pointer or host registration is released.
    batch_workers_.clear();
    // Workers drop their shared references first; the owner's final reference
    // drains and joins the pool while borrowed corpus/workspace storage exists.
    cpu_task_pool_.reset();
    last_batch_result_scores_.clear();

    // Release exact-flat asynchronous users before their borrowed centroid storage.
    if (ws_) ws_->flat_router.reset();
    if (!batch_owner_) delete ivf;
    ivf = nullptr;
    cagra_res_.reset();

    if (ws_ && ws_->h_cpu_candidate_bitmap) std::free(ws_->h_cpu_candidate_bitmap);
    else (void)cudaFree(d_candidate_bitmap_);
    if (ws_ && ws_->uma_query_buffers) {
        std::free(ws_->h_candidate_bitmap_offsets);
    } else {
        (void)cudaFree(d_candidate_bitmap_offsets_);
    }
    (void)cudaFree(d_candidate_arena_raw_);
    if (!batch_owner_) {
    (void)cudaFree(d_doc_ptrs_);
    (void)cudaFree(d_cluster_pos_);
    if (h_shared_onebit_codes_) std::free(h_shared_onebit_codes_);
    else (void)cudaFree(d_clustered_one_bit_codes_);
    if (h_shared_onebit_factors_) std::free(h_shared_onebit_factors_);
    else (void)cudaFree(d_clustered_one_bit_factors_);
    h_shared_onebit_codes_ = nullptr; h_shared_onebit_factors_ = nullptr;
    if (h_uma_clustered_doc_ids_) std::free(h_uma_clustered_doc_ids_);
    else (void)cudaFree(d_clustered_doc_ids_);
    h_uma_clustered_doc_ids_ = nullptr;
    if (h_shared_onebit_inverse_) std::free(h_shared_onebit_inverse_);
    else (void)cudaFree(d_token_to_cluster_pos_);
    h_shared_onebit_inverse_ = nullptr;
    }

    if (ws_ != nullptr) {
        release_gpu_full_bit();
        (void)cudaFree(ws_->d_document_reduce_reference);
        (void)cudaFreeHost(ws_->h_document_reduce_reference);
        if (!ws_->uma_query_buffers) {
            (void)cudaFree(ws_->d_queries);
            (void)cudaFree(ws_->d_cb1_sumq);
            (void)cudaFree(ws_->d_refined_doc_offsets);
            (void)cudaFree(ws_->d_one_bit_document_scores);
        }
        (void)cudaFree(ws_->d_lut);
        if (ws_->uma_query_buffers) {
            std::free(ws_->h_refined_doc_lengths);
        } else {
            (void)cudaFree(ws_->d_refined_doc_lengths);
        }
        (void)cudaFree(ws_->d_cagra_dists);
        if (ws_->h_cpu_cagra_labels) std::free(ws_->h_cpu_cagra_labels);
        else (void)cudaFree(ws_->d_cagra_labels);
        std::free(ws_->h_cpu_materialized_ids);
        std::free(ws_->h_cpu_refine_scores);
        (void)cudaFree(ws_->d_cub_temp_storage);
        if (ws_->uma_query_buffers) {
            std::free(ws_->h_pinned_queries);
            std::free(ws_->h_pinned_cb1_sumq);
        } else {
            (void)cudaFreeHost(ws_->h_pinned_queries);
            (void)cudaFreeHost(ws_->h_pinned_cb1_sumq);
        }
        if (ws_->h_cpu_document_token_scores) std::free(ws_->h_cpu_document_token_scores);
        else (void)cudaFree(ws_->d_one_bit_token_scores);
        (void)cudaFree(ws_->d_refined_token_positions);
        if (ws_->uma_query_buffers) {
            std::free(ws_->h_mapped_one_bit_document_scores);
            std::free(ws_->h_pinned_refined_doc_offsets);
            std::free(ws_->h_refined_doc_ids);
        } else {
            (void)cudaFreeHost(ws_->h_mapped_one_bit_document_scores);
            (void)cudaFreeHost(ws_->h_pinned_refined_doc_offsets);
            (void)cudaFreeHost(ws_->h_refined_doc_ids);
            (void)cudaFreeHost(ws_->h_num_refined_tokens);
        }
        for (cudaEvent_t event : ws_->chunk_compute_done) {
            if (event != nullptr) (void)cudaEventDestroy(event);
        }
        for (cudaEvent_t event : ws_->chunk_d2h_done) {
            if (event != nullptr) (void)cudaEventDestroy(event);
        }
        (void)cudaStreamDestroy(ws_->stream_compute);
        (void)cudaStreamDestroy(ws_->stream_h2d);
        (void)cudaStreamDestroy(ws_->stream_d2h);
        (void)cudaEventDestroy(ws_->event_h2d_done);
    }
    ws_.reset();

    if (!batch_owner_) delete rotator_;
    rotator_ = nullptr;

    d_candidate_bitmap_ = nullptr;
    d_candidate_bitmap_offsets_ = nullptr;
    d_candidate_arena_raw_ = nullptr;
    d_candidate_arena_ = nullptr;
    d_doc_ptrs_ = nullptr;
    d_cluster_pos_ = nullptr;
    d_clustered_one_bit_codes_ = nullptr;
    d_clustered_one_bit_factors_ = nullptr;
    d_clustered_doc_ids_ = nullptr;
    d_token_to_cluster_pos_ = nullptr;

    spilled_full_bit_codes_.reset();
    mapped_full_bit_data_ = nullptr;
    mapped_full_bit_codes_.reset();
    low_memory_load_ = false;
    load_options_ = LoadOptions {};
    load_directory_.clear();
    full_bit_codes_.clear();
    full_bit_factors_.clear();
    one_bit_codes_.clear();
    one_bit_factors_.clear();
    clustered_data_.clear();
    doc_lens_.clear();
    doc_ptrs_.clear();
    n = 0;
    d = 0;
    n_clusters = 0;
    ex_bits = 0;
    num_docs = 0;
    max_cluster_size = 0;
    batch_owner_ = nullptr;
    has_index_ = false;
    search_ready_ = false;
}

}  // namespace Chimera

namespace Chimera {
const char* chimera_index::full_bit_codes_data() const {
    if (batch_owner_) return batch_owner_->full_bit_codes_data();
    if (spilled_full_bit_codes_) return spilled_full_bit_codes_->data();
    return mapped_full_bit_data_ ? mapped_full_bit_data_ : full_bit_codes_.data();
}
}
