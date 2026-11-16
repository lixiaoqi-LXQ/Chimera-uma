#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace raft {
class resources;
}

namespace rabitqlib {
template <typename T>
class Rotator;
}

namespace Chimera {

struct IVF_PG;
class FloatInputView;
class ReadOnlyMappedFile;
class SpilledFullCodes;
class CpuTaskPool;

class BuildCheckpointStop final : public std::exception {
 public:
    const char* what() const noexcept override { return "Assignment prefix committed; intentional checkpoint stop"; }
};

struct BuildOptions {
    std::string spill_full_codes_path;
    size_t spill_flush_bytes = 64ull * 1024ull * 1024ull;
    std::string checkpoint_dir;
    std::string checkpoint_identity_sha256;
    bool resume_checkpoint = false;
    size_t stop_after_assignment_batches = 0;
};

struct LoadOptions {
    bool low_memory = false;
    size_t transfer_chunk_bytes = 64ull * 1024ull * 1024ull;
    // True reads mapped CPU pages once during load, before query timing.
    bool prefault_full_codes = true;
};

// Diagnostic borrow of the allocation already used by GPU fullbit scoring.
// Read-only; valid only while this index and its prepared lanes remain alive.
struct ReadOnlyFullbitCodesView {
    const char* cpu_data = nullptr;
    const char* gpu_data = nullptr;
    size_t bytes = 0;
    size_t token_count = 0;
    size_t document_count = 0;
    const int* document_offsets = nullptr;
    std::vector<const char*> lane_gpu_data;
    std::string backing_file;
    bool ats_alias = false;
};

enum class AlgorithmVariant { Full, NoCR, NoIVFPG, Neither };
inline bool skips_candidate_refinement(AlgorithmVariant v) { return v == AlgorithmVariant::NoCR || v == AlgorithmVariant::Neither; }
inline bool uses_flat_centroid_route(AlgorithmVariant v) { return v == AlgorithmVariant::NoIVFPG || v == AlgorithmVariant::Neither; }

enum class CagraMemoryPlacement { Device, AtsDataset, AtsDatasetAndGraph };
// Existing cuVS search implementations; this only controls host dispatch.
enum class CagraSearchAlgorithm { Auto = 100, SingleCta = 0, MultiCta = 1, MultiKernel = 2 };
enum class FusedTokenDocumentLayout { Grouped, Interleaved, Cached, Warp4Doc };
inline const char* fused_token_document_layout_name(FusedTokenDocumentLayout layout) {
    switch (layout) {
        case FusedTokenDocumentLayout::Grouped: return "grouped";
        case FusedTokenDocumentLayout::Interleaved: return "interleaved";
        case FusedTokenDocumentLayout::Cached: return "cached";
        case FusedTokenDocumentLayout::Warp4Doc: return "warp4doc";
        default: throw std::invalid_argument("Unknown fused token/document layout");
    }
}

struct SearchOptions {
    AlgorithmVariant algorithm_variant = AlgorithmVariant::Full;
    CagraMemoryPlacement cagra_memory_placement = CagraMemoryPlacement::Device;
    size_t expected_coarse_clusters = 0;
    int nprobe = 128;
    int k_refine = 3000;
    int k_full_bit = 300;
    int cagra_itopk_size = 150;
    CagraSearchAlgorithm cagra_algo = CagraSearchAlgorithm::Auto;
    int cagra_team_size = 0;
    int cagra_thread_block_size = 0;
    int cagra_search_width = 1;
    int num_chunks = 5;
    bool gpu_full_bit = true;  // Chimera+: only the full-bit scoring backend changes.
    bool verify_full_bit = false;  // Diagnostic: replay identical candidates on CPU.
    bool doc_order_onebit = false;  // Optional document-order replica, built and verified once at load.
    bool fused_token_document = false;  // Opt-in Q8 groups, small query-max scratch, then original-order sum.
    bool fused_direct_docs = false;  // Grouped/warp4doc fusion reads natural doc ranges; bypass refined metadata handoff.
    int fused_token_document_threads = 128;  // Supported CTA sizes: 128 or 256.
    FusedTokenDocumentLayout fused_token_document_layout = FusedTokenDocumentLayout::Grouped;
    bool split_token_score = false;  // Opt-in independent Q8 CTA groups, original score matrix.
    bool compact_token_score = false;  // Independent opt-in bounded-register token scorer.
    bool fast_document_reduce = false;  // Independent opt-in GPU warp document reduction.
    bool uma_cpu_refine = false;  // CPU exact stable-radix-equivalent top-k.
    bool uma_cpu_candidates = false;  // CPU integer candidate generation; requires UMA query buffers.
    bool uma_query_buffers = false;  // Opt-in: ATS-coherent mapped query/metadata buffers.
    double uma_cpu_fullbit_fraction = 0.0;  // Shared CPU pool scores this fraction of fullbit candidates.
    bool uma_cpu_fullbit_deferred = false;  // Batch-only detached fullbit tails; fraction=1, chunks=1.
    bool uma_cpu_document = false;  // Shared CPU pool reduces GPU token scores to document scores.
    int uma_cpu_threads = 10;
    std::string uma_cpu_pool_cpus;  // Optional comma-separated CPU IDs; empty keeps the E-core default.
};

inline void check_cuda(
    cudaError_t status,
    const char* operation = "CUDA operation") {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

// Optional per-query diagnostic counters; allocated only for explicit qid profiling.
struct QueryProfileStats {
    int64_t qid = -1;
    size_t lane = 0;
    size_t lane_ordinal = 0;
    size_t expected_document_tiles = 0;
    size_t expected_fullbit_tasks = 0;
    size_t refined_docs = 0;
    size_t refined_tokens = 0;
    size_t fullbit_candidates = 0;
    size_t fullbit_tokens = 0;
};

class chimera_index {
  public:
    chimera_index();
    ~chimera_index();

    void build(
        const std::vector<float>& embeddings,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options = {});

    // Non-owning FP32 input; keep storage alive until this synchronous call returns.
    // embedding_values is the number of floats, not the number of token vectors.
    void build_from_ptr(
        const float* embeddings,
        size_t embedding_values,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options = {});

    // IEEE FP16 bit patterns. Expanded exactly to FP32 in bounded CPU buffers.
    void build_from_f16_ptr(
        const uint16_t* embeddings,
        size_t embedding_values,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options = {});

    void build(
        const std::vector<float>& embeddings,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options,
        const BuildOptions& build_options);

    // Non-owning FP32 input; keep storage alive until this synchronous call returns.
    // embedding_values is the number of floats, not the number of token vectors.
    void build_from_ptr(
        const float* embeddings,
        size_t embedding_values,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options,
        const BuildOptions& build_options);

    // IEEE FP16 bit patterns. Expanded exactly to FP32 in bounded CPU buffers.
    void build_from_f16_ptr(
        const uint16_t* embeddings,
        size_t embedding_values,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options,
        const BuildOptions& build_options);

    std::vector<size_t> search(
        const std::vector<float>& query,
        size_t k);
    std::vector<size_t> search(
        const float* query,
        size_t query_size,
        size_t k);

    // Synchronous multi-stream batch. Contiguous [B, 32, d] FP32 input;
    // query_values counts floats. Results retain input order. A single index
    // object must not be called concurrently (including load/save/search).
    // Allocate reusable worker workspaces outside throughput timing.
    void prepare_batch(size_t workers);
    ReadOnlyFullbitCodesView borrow_fullbit_codes() const;
    std::vector<std::vector<size_t>> search_batch(
        const float* queries, size_t query_values, size_t k, size_t workers);
    const std::vector<std::vector<float>>& last_batch_result_scores() const {
        return last_batch_result_scores_;
    }

    // Optional observation only; disabled for original API callers.
    void set_profiling_enabled(bool enabled) { profiling_enabled_ = enabled; }
    // Set >=0 before each profiled batch; -1 preserves existing profiling payloads.
    void set_profile_query_base(int64_t base) {
        if (base < -1) throw std::invalid_argument("Profile query base must be >= -1");
        profile_query_base_ = base;
    }
    // Allocate once outside capture; qid is also its bounds-checked output slot.
    void prepare_profile_query_stats(size_t count) {
        last_batch_profile_stats_.assign(count, QueryProfileStats{});
    }
    const std::vector<QueryProfileStats>& last_batch_profile_stats() const {
        return last_batch_profile_stats_;
    }
    // Diagnostic only; call between synchronous searches, after loading with
    // verify_full_bit=true so every lane has its verification scratch storage.
    void disable_diagnostic_candidate_verification() {
        options_.verify_full_bit = false;
        for (auto& worker : batch_workers_) worker->options_.verify_full_bit = false;
    }
    void set_capture_scores(bool enabled) { capture_scores_ = enabled; }
    const std::vector<float>& last_result_scores() const { return last_result_scores_; }

    void save(const std::string& index_dir) const;
    void load(
        const std::string& index_dir,
        const SearchOptions& options = {});
    void load(
        const std::string& index_dir,
        const SearchOptions& options,
        const LoadOptions& load_options);

    chimera_index(const chimera_index&) = delete;
    chimera_index& operator=(const chimera_index&) = delete;

  private:
    friend struct AlgorithmAblationQualification;  // Isolated synthetic GPU qualification; no runtime hook.
    void build_from_view(
        const FloatInputView& embeddings,
        size_t dimension,
        const std::vector<int>& doc_lens,
        size_t n_clusters,
        size_t ex_bits,
        const SearchOptions& options,
        const BuildOptions& build_options = {});
    const char* full_bit_codes_data() const;

    struct QueryState;
    struct PendingFullbit;  // CPU-only definition; owns all per-query task storage.

    struct IndexHeader {
        size_t num_embeddings;
        size_t dimension;
        size_t num_clusters;
        size_t ex_bits;
        size_t padded_dimension;
    };

    struct Workspace;

    static void validate_search_options(const SearchOptions& options);
    void begin_initialization(const SearchOptions& options);
    void initialize_search_state();
    std::vector<size_t> search_one(const float* queries, size_t k);
    void release_resources() noexcept;
    void set_doc_mapping(const std::vector<int>& doc_lens);
    void compute_candidate_bounds();
    void allocate_workspace();
    void ensure_candidate_capacity(size_t required_rows);
    void bind_candidate_arena(
        char* base,
        size_t row_capacity,
        size_t score_buffer_capacity);

    void generate_and_refine_candidates(
        size_t nprobe,
        size_t k_refine,
        int& num_refined_candidates,
        cudaStream_t stream = nullptr);
    void collaborative_document_scoring(
        int num_refined_candidates,
        size_t k,
        size_t k_full_bit,
        QueryState* queries,
        std::vector<size_t>& result);
    void compute_full_bit_scores(
        const int* refined_doc_ids,
        const int* full_bit_candidate_indices,
        int num_full_bit_candidates,
        const float* queries_flat,
        const QueryState* queries,
        std::pair<float, int>* full_bit_scores);

    void compute_cpu_full_bit_scores(
        const int* refined_doc_ids, const int* candidate_indices, int count,
        const float* queries_flat, const QueryState* queries,
        std::pair<float, int>* scores);
    std::shared_ptr<PendingFullbit> submit_deferred_fullbit(
        const int* refined_doc_ids, const int* candidate_indices, int count,
        const float* queries_flat, const QueryState* queries);
    static void wait_deferred_fullbit(const std::shared_ptr<PendingFullbit>& job);
    static void finish_deferred_fullbit(
        const std::shared_ptr<PendingFullbit>& job, size_t k,
        std::vector<size_t>& result, std::vector<float>* captured_scores);
    void initialize_gpu_full_bit();
    void release_gpu_full_bit() noexcept;
    void compute_gpu_full_bit_scores(
        const int* refined_doc_ids, const int* candidate_indices, int count,
        const QueryState* queries, std::pair<float, int>* scores);

    // Internal workers borrow immutable corpus storage from this owner. They
    // are destroyed before the owner frees that storage; no index is reloaded.
    const chimera_index* batch_owner_ = nullptr;
    std::vector<std::unique_ptr<chimera_index>> batch_workers_;
    std::shared_ptr<CpuTaskPool> cpu_task_pool_;  // Shared by owner and all query lanes.
    bool defer_fullbit_in_batch_ = false;  // Scoped by search_batch, never enabled by ordinary search.
    std::shared_ptr<PendingFullbit> pending_fullbit_;  // Published into the current batch before reuse.
    std::vector<std::vector<float>> last_batch_result_scores_;
    const chimera_index& corpus() const { return batch_owner_ ? *batch_owner_ : *this; }

    bool profiling_enabled_ = false;
    int64_t profile_query_base_ = -1;
    int64_t profile_query_id_ = -1;  // Lane-local current query; never shared across lanes.
    size_t profile_document_tiles_ = 0;
    size_t profile_fullbit_tasks_ = 0;
    size_t profile_refined_docs_ = 0;
    size_t profile_refined_tokens_ = 0;
    size_t profile_fullbit_candidates_ = 0;
    size_t profile_fullbit_tokens_ = 0;
    std::vector<QueryProfileStats> last_batch_profile_stats_;
    bool capture_scores_ = false;
    std::vector<float> last_result_scores_;

    size_t n = 0;
    size_t d = 0;
    size_t n_clusters = 0;
    size_t ex_bits = 0;
    size_t num_docs = 0;
    int max_cluster_size = 0;
    size_t max_probed_clusters_ = 0;
    size_t max_retrieved_tokens_per_query_ = 0;
    size_t max_candidate_docs_per_query_ = 0;

    rabitqlib::Rotator<float>* rotator_ = nullptr;
    IVF_PG* ivf = nullptr;
    std::vector<char> full_bit_codes_;
    std::unique_ptr<SpilledFullCodes> spilled_full_bit_codes_;
    std::unique_ptr<ReadOnlyMappedFile> mapped_full_bit_codes_;
    const char* mapped_full_bit_data_ = nullptr;
    LoadOptions load_options_;
    bool low_memory_load_ = false;
    std::string load_directory_;
    std::vector<float> full_bit_factors_;
    std::vector<std::uint64_t> one_bit_codes_;
    std::vector<float> one_bit_factors_;
    std::vector<char> clustered_data_;
    std::vector<int> doc_lens_;
    std::vector<int> doc_ptrs_;
    void (*unpack_func_)(const uint8_t*, float*, size_t) = nullptr;

    int* d_doc_ptrs_ = nullptr;
    size_t* d_cluster_pos_ = nullptr;
    char* d_clustered_one_bit_codes_ = nullptr;
    char* d_doc_order_onebit_codes_ = nullptr;
    float* d_doc_order_onebit_factors_ = nullptr;
    float* d_clustered_one_bit_factors_ = nullptr;
    int* d_clustered_doc_ids_ = nullptr;
    int* h_uma_clustered_doc_ids_ = nullptr;  // Corpus owner only; d_ aliases this ATS allocation.
    uint32_t* d_token_to_cluster_pos_ = nullptr;

    uint32_t* d_candidate_bitmap_ = nullptr;
    uint32_t* d_candidate_bitmap_offsets_ = nullptr;
    size_t candidate_bitmap_bucket_count_ = 0;
    size_t candidate_bitmap_offset_count_ = 0;
    size_t max_candidate_docs_ = 0;
    size_t candidate_capacity_ = 0;
    char* d_candidate_arena_raw_ = nullptr;
    char* d_candidate_arena_ = nullptr;

    SearchOptions options_;
    std::unique_ptr<raft::resources> cagra_res_;
    std::unique_ptr<Workspace> ws_;
    bool has_index_ = false;
    bool search_ready_ = false;
};

}  // namespace Chimera
