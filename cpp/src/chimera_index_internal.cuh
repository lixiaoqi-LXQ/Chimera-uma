#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "chimera/chimera_index.cuh"
#include "chimera/flat_l2_router.cuh"
#include "partial_topk.cuh"

#include <numeric>
#include <utility>
#include <vector>

namespace Chimera {

struct chimera_index::Workspace {
    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    // Actual enabled state; set only after verifying ATS/host-page-table access.
    bool uma_query_buffers = false;
    bool cpu_document_enabled = false;
    float* h_cpu_document_token_scores = nullptr;  // Sole owner; d_one_bit_token_scores aliases ATS.
    uint32_t* h_cpu_candidate_bitmap = nullptr;
    uint32_t* h_cpu_cagra_labels = nullptr;
    int* h_cpu_materialized_ids = nullptr;
    float* h_cpu_refine_scores = nullptr;
    std::vector<size_t> cpu_refine_order;
    int* d_mapped_refined_doc_ids = nullptr;
    int* h_refined_doc_lengths = nullptr;
    uint32_t* h_candidate_bitmap_offsets = nullptr;
    std::unique_ptr<FlatL2Router> flat_router;
    const char* gpu_full_codes = nullptr;
    const float* gpu_full_factors = nullptr;
    void* registered_full_codes = nullptr;
    void* registered_full_factors = nullptr;
    int* d_full_doc_ids = nullptr;
    float* d_full_cbex = nullptr;
    float* d_full_scores = nullptr;
    int* h_full_doc_ids = nullptr;
    float* h_full_cbex = nullptr;
    float* h_full_scores = nullptr;
    float* d_queries = nullptr;
    // Sole lane-local ATS owner is h_; d_ is the same address, never cudaFree'd.
    __half* h_cagra_half_queries = nullptr;
    __half* d_cagra_half_queries = nullptr;
    size_t cagra_half_query_bytes = 0;
    float* d_cb1_sumq = nullptr;
    int* d_refined_doc_lengths = nullptr;
    float* d_one_bit_token_scores = nullptr;
    float* d_one_bit_document_scores = nullptr;
    float* d_document_reduce_reference = nullptr;  // Diagnostic only.
    float* h_document_reduce_reference = nullptr;
    // Private lane-owned diagnostics, allocated only for TensorPadded+verify.
    float* d_tensor_querymax_capture = nullptr;
    float* h_tensor_querymax_capture = nullptr;
    double* d_tensor_half_reference = nullptr;
    double* h_tensor_half_reference = nullptr;
    double* d_tensor_half_bounds = nullptr;
    double* h_tensor_half_bounds = nullptr;
    float* d_lut = nullptr;
    float* d_cagra_dists = nullptr;
    uint32_t* d_cagra_labels = nullptr;
    int* d_sorted_candidate_doc_ids = nullptr;
    int* d_materialized_candidate_doc_ids = nullptr;
    float* d_partial_scores = nullptr;
    float* d_partial_query_scores = nullptr;
    void* d_cub_temp_storage = nullptr;
    size_t cub_temp_storage_bytes = 0;
    float* d_sorted_partial_scores = nullptr;
    PartialTopKWorkspace partial_topk;  // Sole lane owner; no shared batch-owner buffers.
    int* d_candidate_doc_ids = nullptr;
    float* h_pinned_queries = nullptr;
    float* h_pinned_cb1_sumq = nullptr;
    size_t max_refined_candidates = 0;
    size_t max_refined_tokens = 0;
    size_t max_full_bit_candidates = 0;
    size_t* d_refined_doc_offsets = nullptr;
    uint32_t* d_refined_token_positions = nullptr;
    float* h_mapped_one_bit_document_scores = nullptr;
    size_t* h_pinned_refined_doc_offsets = nullptr;
    int* h_refined_doc_ids = nullptr;
    size_t* h_num_refined_tokens = nullptr;
    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_h2d = nullptr;
    cudaStream_t stream_d2h = nullptr;
    cudaEvent_t event_h2d_done = nullptr;
    // Private to this query workspace/lane. A synchronous search waits for all
    // active chunk D2H events before returning, so handles can be re-recorded
    // by the next query without cross-query dependencies.
    std::vector<cudaEvent_t> chunk_compute_done;
    std::vector<cudaEvent_t> chunk_d2h_done;
    std::vector<int> one_bit_score_order;
    std::vector<int> full_bit_candidate_indices;
    std::vector<std::pair<float, int>> full_bit_scores;
    std::vector<bool> seen_doc_map;
};

struct chimera_index::QueryState {
    float* rotated_query;
    float cb1_sumq;
    float cbex_sumq;

    QueryState() = default;

    explicit QueryState(float* rotated_query, size_t padded_dim, size_t ex_bits)
        : rotated_query(rotated_query) {
        const float sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, 0.0f);
        cb1_sumq = sumq * ((1 << 1) - 1) / 2.0f;
        cbex_sumq = sumq * ((1 << (ex_bits + 1)) - 1) / 2.0f;
    }
};

}  // namespace Chimera
