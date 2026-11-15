#include "chimera/chimera_index.cuh"
#include "chimera/config.cuh"
#include "chimera/profiling.hpp"
#include "chimera_index_internal.cuh"
#include "rabitqlib/utils/space.hpp"
#include "cpu_gemv_neon.hpp"
#include "cpu_fullbit_pool_helpers.hpp"
#include "cpu_task_pool.hpp"
#include <algorithm>
#include <array>
#include <queue>
#include <cmath>
#include <exception>
#include <limits>
#include <cstring>
#include <iostream>
namespace Chimera {

// No task borrows query_states, candidate arrays or workspace query storage.
// Corpus inputs remain read-only and are drained before the batch owner frees
// them. Definition stays in this C++ TU so CUDA TUs never include NEON helpers.
struct alignas(64) chimera_index::PendingFullbit {
    alignas(64) std::array<float, Q_DOCLEN * PADDED_DIM> rotated_query;
    alignas(64) std::array<float, Q_DOCLEN> cbex_sumq;
    std::vector<int> doc_ids;
    std::vector<std::pair<float, int>> scores;
    std::vector<std::pair<float, int>> cpu_reference;
    std::vector<std::pair<float, int>> gpu_reference;
    std::shared_ptr<CpuTaskPool::Group> group;
    bool profile = false;

    ~PendingFullbit() noexcept {
        // Protect owned input storage on every exception/unwinding path too.
        // Normal completion already called wait(), which may report errors.
        if (group) {
            try { group->wait(); } catch (...) {}
        }
    }
};

std::shared_ptr<chimera_index::PendingFullbit> chimera_index::submit_deferred_fullbit(
    const int* refined_doc_ids, const int* candidate_indices, int count,
    const float* queries_flat, const QueryState* queries) {
    ProfileRange submit(profiling_enabled_, "Chimera/deferred_fullbit_submit", count);
    if (!cpu_task_pool_ || count <= 0)
        throw std::logic_error("Deferred fullbit requires a pool and nonempty candidates");
    auto job = std::make_shared<PendingFullbit>();
    job->profile = profiling_enabled_;
    std::copy_n(queries_flat, Q_DOCLEN * PADDED_DIM, job->rotated_query.data());
    for (size_t q = 0; q < Q_DOCLEN; ++q) job->cbex_sumq[q] = queries[q].cbex_sumq;
    job->doc_ids.resize(count);
    job->scores.resize(count);
    for (int i = 0; i < count; ++i)
        job->doc_ids[i] = refined_doc_ids[candidate_indices[i]];

    if (options_.verify_full_bit) {
        // Capture both references before this lane advances and replaces
        // d_queries/h_pinned_queries or any candidate metadata.
        job->cpu_reference.resize(count);
        compute_cpu_full_bit_scores(refined_doc_ids, candidate_indices, count,
            queries_flat, queries, job->cpu_reference.data());
        job->gpu_reference.resize(count);
        compute_gpu_full_bit_scores(refined_doc_ids, candidate_indices, count,
            queries, job->gpu_reference.data());
    }

    const auto& owner = corpus();
    const FullbitCpuInputs input{
        full_bit_codes_data(), owner.full_bit_factors_.data(), owner.doc_ptrs_.data(),
        job->rotated_query.data(), job->cbex_sumq.data(),
        PADDED_DIM * (1 + ex_bits) / 8, unpack_func_};
    PendingFullbit* const output = job.get();
    // submit is transactional. Once published, job's destructor joins the
    // group before destroying any storage referenced by this raw pointer.
    job->group = cpu_task_pool_->submit(size_t(count), size_t{1},
        [input, output](size_t index) {
            const int doc = output->doc_ids[index];
            ProfileRange work(output->profile, "Chimera/deferred_fullbit_worker",
                              input.doc_ptrs[doc + 1] - input.doc_ptrs[doc]);
            output->scores[index] = score_fullbit_document_cpu(input, doc);
        });
    return job;
}

void chimera_index::wait_deferred_fullbit(const std::shared_ptr<PendingFullbit>& job) {
    if (!job) return;
    ProfileRange join(job->profile, "Chimera/deferred_fullbit_join", job->doc_ids.size());
    job->group->wait();
}

void chimera_index::finish_deferred_fullbit(
    const std::shared_ptr<PendingFullbit>& job, size_t k,
    std::vector<size_t>& result, std::vector<float>* captured_scores) {
    wait_deferred_fullbit(job);
    ProfileRange finish(job->profile, "Chimera/deferred_fullbit_finish", job->scores.size());
    if (!job->gpu_reference.empty()) {
        for (size_t i = 0; i < job->scores.size(); ++i) {
            const auto& actual = job->scores[i];
            const auto& cpu = job->cpu_reference[i];
            const auto& gpu = job->gpu_reference[i];
            if (cpu.second != actual.second || gpu.second != actual.second ||
                std::memcmp(&cpu.first, &actual.first, sizeof(float)) ||
                std::memcmp(&gpu.first, &actual.first, sizeof(float)))
                throw std::runtime_error("Deferred fullbit differs from CPU/GPU at candidate " +
                                         std::to_string(i));
        }
        std::cout << "[DEFERRED_FULLBIT_VERIFY] candidates=" << job->scores.size()
                  << " cpu_oracle_fp32_bitwise=passed gpu_oracle_fp32_bitwise=passed\n";
    }

    // Preserve the original pair<float,size_t> heap and its tie ordering.
    std::priority_queue<std::pair<float, size_t>> heap;
    for (const auto& score : job->scores)
        heap.emplace(score.first, static_cast<size_t>(score.second));
    result.clear();
    if (captured_scores) {
        captured_scores->clear();
        captured_scores->reserve(k);
    }
    for (size_t i = 0; i < k && !heap.empty(); ++i) {
        result.push_back(heap.top().second);
        if (captured_scores) captured_scores->push_back(heap.top().first);
        heap.pop();
    }
}

void chimera_index::compute_full_bit_scores(
    const int* refined_doc_ids,
    const int* full_bit_candidate_indices,
    int num_full_bit_candidates,
    const float* queries_flat,
    const QueryState* queries,
    std::pair<float, int>* full_bit_scores
) {
    if (options_.uma_cpu_fullbit_fraction > 0.0) {
        ProfileRange range(profiling_enabled_, "Chimera/uma_cpu_fullbit", num_full_bit_candidates);
        if (num_full_bit_candidates <= 0) return;
        if (!options_.gpu_full_bit)
            throw std::logic_error("CPU full-bit cooperation requires the GPU full-bit backend");
        if (!cpu_task_pool_)
            throw std::logic_error("CPU full-bit cooperation requires a shared CPU task pool");

        const int cpu_count = std::min(num_full_bit_candidates, static_cast<int>(
            std::ceil(num_full_bit_candidates * options_.uma_cpu_fullbit_fraction)));
        const int gpu_count = num_full_bit_candidates - cpu_count;
        alignas(64) float cbex_sumq_arr[Q_DOCLEN];
        for (size_t j = 0; j < Q_DOCLEN; ++j)
            cbex_sumq_arr[j] = queries[j].cbex_sumq;
        const auto& full_bit_factors = corpus().full_bit_factors_;
        const auto& doc_ptrs = corpus().doc_ptrs_;
        const FullbitCpuInputs input{
            full_bit_codes_data(), full_bit_factors.data(), doc_ptrs.data(),
            queries_flat, cbex_sumq_arr, PADDED_DIM * (1 + ex_bits) / 8,
            unpack_func_};
        const bool profile = profiling_enabled_;

        // Pool submission is transactional: on failure no task has borrowed
        // these inputs. After successful submission every path below joins.
        auto group = cpu_task_pool_->submit(size_t(cpu_count), size_t{1},
            [=](size_t index) {
                const int i = gpu_count + static_cast<int>(index);
                const int doc = refined_doc_ids[full_bit_candidate_indices[i]];
                ProfileRange work(profile, "Chimera/uma_cpu_fullbit_worker",
                                  input.doc_ptrs[doc + 1] - input.doc_ptrs[doc]);
                full_bit_scores[i] = score_fullbit_document_cpu(input, doc);
            });

        std::exception_ptr failure;
        try {
            if (gpu_count > 0)
                compute_gpu_full_bit_scores(refined_doc_ids, full_bit_candidate_indices,
                    gpu_count, queries, full_bit_scores);
        } catch (...) {
            failure = std::current_exception();
        }
        try {
            ProfileRange join(profile, "Chimera/uma_cpu_fullbit_join", cpu_count);
            group->wait();
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
        if (failure) std::rethrow_exception(failure);

        if (options_.verify_full_bit) {
            std::vector<std::pair<float, int>> reference(num_full_bit_candidates);
            compute_cpu_full_bit_scores(refined_doc_ids, full_bit_candidate_indices,
                num_full_bit_candidates, queries_flat, queries, reference.data());
            for (int i = 0; i < num_full_bit_candidates; ++i) {
                if (reference[i].second != full_bit_scores[i].second ||
                    std::memcmp(&reference[i].first, &full_bit_scores[i].first, sizeof(float)))
                    throw std::runtime_error("Mixed full-bit score differs from original CPU at candidate " +
                                             std::to_string(i));
            }
            // Qualify the CPU shard against the unchanged GPU scorer too;
            // this checks pool execution and both score implementations.
            std::vector<std::pair<float, int>> gpu_reference(cpu_count);
            compute_gpu_full_bit_scores(refined_doc_ids,
                full_bit_candidate_indices + gpu_count, cpu_count, queries,
                gpu_reference.data());
            for (int i = 0; i < cpu_count; ++i) {
                const auto& actual = full_bit_scores[gpu_count + i];
                if (gpu_reference[i].second != actual.second ||
                    std::memcmp(&gpu_reference[i].first, &actual.first, sizeof(float)))
                    throw std::runtime_error("Pooled CPU full-bit score differs from GPU at candidate " +
                                             std::to_string(gpu_count + i));
            }
            std::cout << "[UMA_FULLBIT_VERIFY] candidates=" << num_full_bit_candidates
                      << " cpu_candidates=" << cpu_count << " gpu_candidates=" << gpu_count
                      << " cpu_oracle_fp32_bitwise=passed gpu_shard_fp32_bitwise=passed\n";
        }
        return;
    }
    if (options_.gpu_full_bit) {
        compute_gpu_full_bit_scores(refined_doc_ids, full_bit_candidate_indices,
                                    num_full_bit_candidates, queries, full_bit_scores);
        if (options_.verify_full_bit) {
            std::vector<std::pair<float, int>> reference(num_full_bit_candidates);
            compute_cpu_full_bit_scores(refined_doc_ids, full_bit_candidate_indices,
                num_full_bit_candidates, queries_flat, queries, reference.data());
            for (int i = 0; i < num_full_bit_candidates; ++i) {
                if (reference[i].second != full_bit_scores[i].second ||
                    std::memcmp(&reference[i].first, &full_bit_scores[i].first, sizeof(float)))
                    throw std::runtime_error("GPU full-bit score differs from CPU at candidate " + std::to_string(i));
            }
            std::cout << "[FULLBIT_VERIFY] candidates=" << num_full_bit_candidates
                      << " scores_fp32_bitwise=passed\n";
        }
        return;
    }
    compute_cpu_full_bit_scores(refined_doc_ids, full_bit_candidate_indices,
        num_full_bit_candidates, queries_flat, queries, full_bit_scores);
}

void chimera_index::compute_cpu_full_bit_scores(
    const int* refined_doc_ids, const int* full_bit_candidate_indices,
    int num_full_bit_candidates, const float* queries_flat,
    const QueryState* queries, std::pair<float, int>* full_bit_scores) {
    ProfileRange range(profiling_enabled_, "Chimera/cpu_fullbit", num_full_bit_candidates);
    const size_t full_bit_codes_stride = PADDED_DIM * (1 + ex_bits) / 8;
    const char* full_bit_codes = full_bit_codes_data();
    const auto& full_bit_factors = corpus().full_bit_factors_;
    const auto& doc_ptrs = corpus().doc_ptrs_;
    alignas(64) float cbex_sumq_arr[Q_DOCLEN];
    for (size_t j = 0; j < Q_DOCLEN; j++)
        cbex_sumq_arr[j] = queries[j].cbex_sumq;

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < num_full_bit_candidates; ++i) {
        int doc_id = refined_doc_ids[full_bit_candidate_indices[i]];
        size_t doc_start = doc_ptrs[doc_id];
        size_t n_tok = doc_ptrs[doc_id + 1] - doc_start;

        alignas(64) float decoded[PADDED_DIM];
        alignas(64) float full_ip[Q_DOCLEN];
        alignas(64) float max_ts[Q_DOCLEN];
        __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        for (size_t j = 0; j < Q_DOCLEN; j += 16)
            _mm512_store_ps(&max_ts[j], neg_inf);

        for (size_t t = 0; t < n_tok; t++) {
            size_t tid = doc_start + t;
            unpack_func_(
                reinterpret_cast<const uint8_t*>(
                    &full_bit_codes[tid * full_bit_codes_stride]),
                decoded, PADDED_DIM);
            if (t + 1 < n_tok) {
                __builtin_prefetch(
                    &full_bit_codes[(tid + 1) * full_bit_codes_stride], 0, 3);
                __builtin_prefetch(&full_bit_factors[tid + 1], 0, 1);
            }
#if defined(__aarch64__)
            rabitqlib::gemv_batch4_neon(
#else
            rabitqlib::gemv_batch8_avx512(
#endif
                queries_flat, decoded,
                full_ip,
                Q_DOCLEN, PADDED_DIM);

            const float full_bit_factor = full_bit_factors[tid];
            const __m512 full_bit_factor_vector =
                _mm512_set1_ps(full_bit_factor);

            for (size_t j = 0; j < Q_DOCLEN; j += 16) {
                __m512 full_ip_v = _mm512_load_ps(&full_ip[j]);
                __m512 cbex    = _mm512_load_ps(&cbex_sumq_arr[j]);
                __m512 combined = _mm512_mul_ps(
                    _mm512_sub_ps(full_ip_v, cbex),
                    full_bit_factor_vector);
                _mm512_store_ps(&max_ts[j],
                    _mm512_max_ps(_mm512_load_ps(&max_ts[j]), combined));
            }
        }

        __m512 sum = _mm512_load_ps(&max_ts[0]);
        for (size_t j = 16; j < Q_DOCLEN; j += 16)
            sum = _mm512_add_ps(sum, _mm512_load_ps(&max_ts[j]));
        full_bit_scores[i] = {_mm512_reduce_add_ps(sum), doc_id};
    }
}

} // namespace Chimera
