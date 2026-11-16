#include "chimera/spilled_full_codes.hpp"
#include <cuda_runtime.h>

#include "chimera/chimera_index.cuh"

#define EIGEN_NO_CUDA

#include "chimera/config.cuh"
#include "chimera/io.hpp"
#include "chimera/profiling.hpp"
#include "chimera/index_file_io.hpp"
#include "chimera/ivf_pg.hpp"
#include "chimera/quantization.hpp"
#include "chimera_index_internal.cuh"
#include "doc_order_onebit.cuh"
#include "cpu_task_pool.hpp"
#include "rabitqlib/utils/rotator.hpp"

#include <raft/core/resources.hpp>

#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <omp.h>
#include <exception>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace Chimera {

void chimera_index::validate_search_options(const SearchOptions& options) {
    if (options.split_token_score &&
        (options.compact_token_score || options.fused_token_document || options.uma_cpu_document))
        throw std::invalid_argument("Split token scoring cannot combine compact token, fused token/document or CPU document scoring");
    if (options.fused_token_document_threads != 128 && options.fused_token_document_threads != 256)
        throw std::invalid_argument("Fused token/document CTA size must be 128 or 256");
    if (!options.fused_token_document && options.fused_token_document_threads != 128)
        throw std::invalid_argument("Fused token/document CTA selection requires fused scoring");
    if (options.fused_token_document && options.num_chunks != 1)
        throw std::invalid_argument("Fused token/document scoring requires chunks=1");
    if (options.fused_token_document &&
        (options.uma_cpu_document || options.fast_document_reduce || options.compact_token_score))
        throw std::invalid_argument("Fused token/document scoring cannot combine CPU document, fast document or compact token scoring");
    if (!std::isfinite(options.uma_cpu_fullbit_fraction) ||
        options.uma_cpu_fullbit_fraction < 0 || options.uma_cpu_fullbit_fraction > 1)
        throw std::invalid_argument("UMA CPU fullbit fraction must be in [0,1]");
    if (options.uma_cpu_fullbit_fraction > 0 && !options.gpu_full_bit)
        throw std::invalid_argument("UMA CPU fullbit cooperation requires the GPU fullbit backend");
    if (options.uma_cpu_fullbit_deferred &&
        (options.uma_cpu_fullbit_fraction != 1.0 || options.num_chunks != 1))
        throw std::invalid_argument("Deferred CPU fullbit requires fraction=1 and chunks=1");
    CpuTaskPool::validate_cpu_selection(static_cast<size_t>(options.uma_cpu_threads),
        CpuTaskPool::parse_cpu_list(options.uma_cpu_pool_cpus));
    if (options.uma_cpu_refine && (!options.uma_cpu_candidates || !options.uma_query_buffers))
        throw std::invalid_argument("UMA CPU refine requires CPU candidates and UMA query buffers");
    if (options.uma_cpu_refine && skips_candidate_refinement(options.algorithm_variant))
        throw std::invalid_argument("UMA CPU refine requires candidate refinement");
    if (options.uma_cpu_candidates && !options.uma_query_buffers)
        throw std::invalid_argument("UMA CPU candidates require uma_query_buffers");
    const auto placement = static_cast<int>(options.cagra_memory_placement);
    if (placement < 0 || placement > static_cast<int>(CagraMemoryPlacement::AtsDatasetAndGraph))
        throw std::invalid_argument("Unknown CAGRA memory placement");
    if (options.cagra_memory_placement != CagraMemoryPlacement::Device &&
        uses_flat_centroid_route(options.algorithm_variant))
        throw std::invalid_argument("CAGRA memory placement requires the CAGRA graph route");
    const auto variant = static_cast<int>(options.algorithm_variant);
    if (variant < static_cast<int>(AlgorithmVariant::Full) || variant > static_cast<int>(AlgorithmVariant::Neither))
        throw std::invalid_argument("Unknown algorithm ablation enum");
    if (uses_flat_centroid_route(options.algorithm_variant)) {
        if (!options.expected_coarse_clusters || options.expected_coarse_clusters > size_t(2147483647))
            throw std::invalid_argument("Flat algorithm ablation requires a positive int-sized coarse cluster count");
    } else if (options.expected_coarse_clusters) {
        throw std::invalid_argument("Coarse cluster declaration is only valid for flat algorithm modes");
    }

    switch (options.cagra_algo) {
        case CagraSearchAlgorithm::Auto:
        case CagraSearchAlgorithm::SingleCta:
        case CagraSearchAlgorithm::MultiCta:
        case CagraSearchAlgorithm::MultiKernel: break;
        default: throw std::invalid_argument("Unknown CAGRA search algorithm");
    }
    if (options.cagra_team_size != 0 && options.cagra_team_size != 4 &&
        options.cagra_team_size != 8 && options.cagra_team_size != 16 && options.cagra_team_size != 32)
        throw std::invalid_argument("CAGRA team size must be 0, 4, 8, 16 or 32");
    if (options.cagra_thread_block_size != 0 && options.cagra_thread_block_size != 64 &&
        options.cagra_thread_block_size != 128 && options.cagra_thread_block_size != 256 &&
        options.cagra_thread_block_size != 512 && options.cagra_thread_block_size != 1024)
        throw std::invalid_argument("CAGRA block size must be 0, 64, 128, 256, 512 or 1024");
    if (options.cagra_search_width < 1)
        throw std::invalid_argument("CAGRA search width must be positive");
    // cuVS 26.04's actual SINGLE_CTA implementation supports <=512, even
    // though the shared base validator has a looser <=1024 bound.
    if (options.cagra_algo == CagraSearchAlgorithm::SingleCta && options.cagra_itopk_size > 512)
        throw std::invalid_argument("CAGRA SINGLE_CTA requires itopk_size <= 512");
    // MULTI_KERNEL uses the bundled top-k selector on itopk_size itself.
    if (options.cagra_algo == CagraSearchAlgorithm::MultiKernel && options.cagra_itopk_size > 1024)
        throw std::invalid_argument("CAGRA MULTI_KERNEL requires itopk_size <= 1024");
    if (options.cagra_itopk_size > 0) {
        const int64_t rounded_itopk = ((int64_t(options.cagra_itopk_size) + 31) / 32) * 32;
        if ((options.cagra_algo == CagraSearchAlgorithm::SingleCta ||
             options.cagra_algo == CagraSearchAlgorithm::MultiKernel) && options.nprobe > rounded_itopk)
            throw std::invalid_argument("CAGRA route output nprobe must not exceed rounded itopk_size");
        if (options.cagra_algo == CagraSearchAlgorithm::MultiCta &&
            int64_t(options.nprobe) > 32 * std::max(int64_t(options.cagra_search_width), rounded_itopk / 32))
            throw std::invalid_argument("CAGRA MULTI_CTA output exceeds CTA result capacity");
    }

    if (options.nprobe <= 0 || options.k_refine <= 0 ||
        options.k_full_bit <= 0 || options.cagra_itopk_size <= 0 ||
        options.num_chunks <= 0) {
        throw std::invalid_argument("All Chimera search options must be positive");
    }
    if (options.nprobe > kCandidateRefinementMaxNprobe) {
        throw std::invalid_argument(
            "nprobe must not exceed candidate refinement limit " +
            std::to_string(kCandidateRefinementMaxNprobe));
    }
    if (!skips_candidate_refinement(options.algorithm_variant) && options.k_full_bit > options.k_refine) {
        throw std::invalid_argument("k_full_bit must not exceed k_refine");
    }
}

chimera_index::chimera_index() = default;

ReadOnlyFullbitCodesView chimera_index::borrow_fullbit_codes() const {
    const auto& owner = corpus();
    if (!owner.has_index_ || !owner.search_ready_ || !owner.options_.gpu_full_bit ||
        !owner.low_memory_load_ || !owner.mapped_full_bit_data_ ||
        owner.ex_bits != 3 || PADDED_DIM != 128 ||
        !owner.ws_ || !owner.ws_->gpu_full_codes) {
        throw std::logic_error("Fullbit borrow requires prepared, mmap-backed 4bit GPU scoring");
    }
    const char* cpu_data = owner.full_bit_codes_data();
    int device = 0, pageable_access = 0, host_page_tables = 0;
    check_cuda(cudaGetDevice(&device), "Inspecting fullbit borrow device");
    check_cuda(cudaDeviceGetAttribute(&pageable_access, cudaDevAttrPageableMemoryAccess, device),
               "Inspecting fullbit ATS access");
    check_cuda(cudaDeviceGetAttribute(&host_page_tables,
               cudaDevAttrPageableMemoryAccessUsesHostPageTables, device),
               "Inspecting fullbit host page tables");
    if (owner.ws_->gpu_full_codes != cpu_data ||
        !pageable_access || !host_page_tables ||
        owner.n > std::numeric_limits<size_t>::max() / 64 ||
        owner.doc_ptrs_.size() != owner.num_docs + 1) {
        throw std::logic_error("Fullbit borrow requires the unchanged CPU/GPU ATS allocation");
    }
    ReadOnlyFullbitCodesView view;
    view.cpu_data = cpu_data;
    view.gpu_data = owner.ws_->gpu_full_codes;
    view.bytes = owner.n * 64;
    view.token_count = owner.n;
    view.document_count = owner.num_docs;
    view.document_offsets = owner.doc_ptrs_.data();
    view.backing_file = (std::filesystem::path(owner.load_directory_) / "doc_full.bin").string();
    view.ats_alias = true;
    // prepare_batch(4) creates the owner lane plus three borrowing workspaces.
    view.lane_gpu_data.push_back(owner.ws_->gpu_full_codes);
    for (const auto& lane : owner.batch_workers_) {
        if (!lane->search_ready_ || !lane->ws_ || lane->ws_->gpu_full_codes != cpu_data) {
            throw std::logic_error("A prepared lane does not alias the borrowed fullbit allocation");
        }
        view.lane_gpu_data.push_back(lane->ws_->gpu_full_codes);
    }
    return view;
}

void chimera_index::begin_initialization(const SearchOptions& options) {
    release_resources();
    options_ = options;
    validate_search_options(options_);
    cagra_res_ = std::make_unique<raft::resources>();
    ws_ = std::make_unique<Workspace>();
}

chimera_index::~chimera_index() {
    release_resources();
}

void chimera_index::load(
    const std::string& index_dir,
    const SearchOptions& options) {
    load(index_dir, options, LoadOptions {});
}

void chimera_index::load(
    const std::string& index_dir,
    const SearchOptions& options,
    const LoadOptions& load_options) {
    validate_search_options(options);
    if (load_options.transfer_chunk_bytes == 0) {
        throw std::invalid_argument("Load transfer_chunk_bytes must be positive");
    }
    const std::filesystem::path directory(index_dir);
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error(
            "Chimera index directory does not exist: " + index_dir);
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "Querying CUDA devices");
    if (device_count == 0) {
        throw std::runtime_error("No CUDA device is available");
    }
    check_cuda(cudaSetDevice(0), "Selecting CUDA device 0");
    begin_initialization(options);
    load_options_ = load_options;
    low_memory_load_ = load_options.low_memory;
    load_directory_ = std::filesystem::absolute(directory).string();
    if (low_memory_load_) {
        // release_resources clears vector sizes; drop capacities from a previous
        // ordinary load too so switching modes on the same object saves memory.
        std::vector<char>().swap(full_bit_codes_);
        std::vector<uint64_t>().swap(one_bit_codes_);
        std::vector<float>().swap(one_bit_factors_);
        std::vector<char>().swap(clustered_data_);
        std::vector<float>().swap(full_bit_factors_);
    }

    const auto doclens_path = (directory / "doclens.bin").string();
    const auto doc_one_bit_path = (directory / "doc_1bit.bin").string();
    const auto doc_full_path = (directory / "doc_full.bin").string();
    const auto cluster_one_bit_path =
        (directory / "cluster_1bit.bin").string();
    const auto ivf_path = (directory / "ivf.bin").string();
    const auto centroids_path = (directory / "centroids.carga").string();

    std::ifstream inf(doc_one_bit_path, std::ios::binary);
    IndexHeader header {};
    inf.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!inf || header.num_embeddings == 0 ||
        header.num_embeddings > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        header.dimension < 64 || header.dimension > PADDED_DIM ||
        header.num_clusters == 0 ||
        header.num_clusters > header.num_embeddings ||
        header.ex_bits < 1 || header.ex_bits > 7) {
        throw std::runtime_error("Invalid Chimera index header: " + doc_one_bit_path);
    }
    n = header.num_embeddings;
    d = header.dimension;
    n_clusters = header.num_clusters;
    ex_bits = header.ex_bits;
    if (static_cast<size_t>(options_.nprobe) > n_clusters) {
        throw std::invalid_argument("nprobe must not exceed the number of clusters");
    }

    if (header.padded_dimension != PADDED_DIM) {
        throw std::runtime_error(
            "Index file padded_dim=" +
            std::to_string(header.padded_dimension) +
            " does not match compiled PADDED_DIM=" + std::to_string(PADDED_DIM) +
            ". Please recompile with matching PADDED_DIM in chimera/config.cuh"
        );
    }

    rotator_ = rabitqlib::choose_rotator<float>(
        d,
        rabitqlib::RotatorType::FhtKacRotator,
        header.padded_dimension);
    rotator_->load(inf);
    if (!inf) {
        delete rotator_;
        rotator_ = nullptr;
        throw std::runtime_error("Failed to read rotator from " + doc_one_bit_path);
    }

    full_bit_factors_.resize(n);
    if (low_memory_load_) {
        // doc_1bit.bin after its rotator stores codes[16*N], one-bit
        // factors[4*N], then CPU full-bit factors[4*N]. Search uses only
        // the last segment on CPU; GPU 1-bit data comes from cluster_1bit.bin.
        const size_t skip = n * (PADDED_DIM / 8 + sizeof(float));
        inf.seekg(static_cast<std::streamoff>(skip), std::ios::cur);
    } else {
        one_bit_codes_.resize(n * (PADDED_DIM / 64));
        one_bit_factors_.resize(n);
        inf.read(reinterpret_cast<char*>(one_bit_codes_.data()),
                 static_cast<std::streamsize>(one_bit_codes_.size() * sizeof(uint64_t)));
        inf.read(reinterpret_cast<char*>(one_bit_factors_.data()),
                 static_cast<std::streamsize>(one_bit_factors_.size() * sizeof(float)));
    }
    inf.read(reinterpret_cast<char*>(full_bit_factors_.data()),
             static_cast<std::streamsize>(full_bit_factors_.size() * sizeof(float)));
    if (!inf) {
        throw std::runtime_error("Failed to read quantized payload from " + doc_one_bit_path);
    }
    const size_t full_code_bytes = n * PADDED_DIM * (1 + ex_bits) / 8;
    if (low_memory_load_) {
        mapped_full_bit_codes_ = std::make_unique<ReadOnlyMappedFile>(doc_full_path, full_code_bytes);
        mapped_full_bit_data_ = mapped_full_bit_codes_->data();
        if (load_options_.prefault_full_codes) {
            const auto prefault_begin = std::chrono::steady_clock::now();
            mapped_full_bit_codes_->prefault();
            const double prefault_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prefault_begin).count();
            std::cout << "[PREFAULT] bytes=" << full_code_bytes
                      << " elapsed_ms=" << prefault_ms << '\n';
        }
    } else {
        full_bit_codes_.resize(full_code_bytes);
        std::ifstream full_bit_codes_file(doc_full_path, std::ios::binary);
        full_bit_codes_file.read(full_bit_codes_.data(), full_bit_codes_.size());
        if (!full_bit_codes_file) {
            throw std::runtime_error("Failed to read full-code payload from " + doc_full_path);
        }
    }
    ivf = new IVF_PG(n_clusters, d);
    ivf->load(ivf_path, centroids_path);
    doc_lens_ = load_doclens(doclens_path);

    const size_t clustered_bytes =
        n * (PADDED_DIM / 8 + sizeof(float) + sizeof(int) + sizeof(uint32_t));
    std::ifstream clustered_file(
        cluster_one_bit_path,
        std::ios::binary | std::ios::ate);
    if (!clustered_file || clustered_file.tellg() < 0 ||
        static_cast<size_t>(clustered_file.tellg()) != clustered_bytes) {
        throw std::runtime_error(
            "Invalid clustered index payload: " + cluster_one_bit_path);
    }
    if (!low_memory_load_) {
        clustered_data_.resize(clustered_bytes);
        clustered_file.seekg(0);
        clustered_file.read(clustered_data_.data(),
                            static_cast<std::streamsize>(clustered_data_.size()));
        if (!clustered_file) {
            throw std::runtime_error("Failed to read clustered index: " + cluster_one_bit_path);
        }
    }

    has_index_ = true;
    try {
        initialize_search_state();
    } catch (...) {
        release_resources();
        throw;
    }
}

void chimera_index::initialize_search_state() {
    if (search_ready_) {
        return;
    }
    if (!has_index_ || ivf == nullptr || rotator_ == nullptr || ws_ == nullptr) {
        throw std::runtime_error("Cannot initialize an empty Chimera index");
    }
    if (static_cast<size_t>(options_.nprobe) > n_clusters) {
        throw std::invalid_argument("nprobe must not exceed the number of clusters");
    }

    unpack_func_ = rabitqlib::select_excode_unpackfunc(1 + ex_bits);
    max_cluster_size = ivf->max_cluster_size();
    set_doc_mapping(doc_lens_);

    check_cuda(cudaMalloc(&d_doc_ptrs_, (num_docs + 1) * sizeof(int)));
    check_cuda(cudaMalloc(&d_cluster_pos_, (ivf->n_clusters + 1) * sizeof(size_t)));
    check_cuda(cudaMemcpy(
        d_cluster_pos_,
        ivf->cluster_pos.data(),
        (ivf->n_clusters + 1) * sizeof(size_t),
        cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(
        d_doc_ptrs_,
        doc_ptrs_.data(),
        (num_docs + 1) * sizeof(int),
        cudaMemcpyHostToDevice));

    const size_t code_per_vec = PADDED_DIM / 8;
    const size_t expected_clustered_bytes =
        n * (code_per_vec + sizeof(float) + sizeof(int) + sizeof(uint32_t));
    if (!low_memory_load_ && clustered_data_.size() != expected_clustered_bytes) {
        throw std::runtime_error("Invalid in-memory clustered index payload");
    }

    // On UMA, cudaMemGetInfo excludes reclaimable OS page cache and can
    // underreport allocatable memory. Let the checked allocations decide;
    // initialize_search_state's caller releases partial allocations on error.
    // https://nvidia.custhelp.com/app/answers/detail/a_id/5728
    check_cuda(cudaMalloc(&d_clustered_one_bit_codes_, n * code_per_vec));
    check_cuda(cudaMalloc(&d_clustered_one_bit_factors_, n * sizeof(float)));
    check_cuda(cudaMalloc(&d_clustered_doc_ids_, n * sizeof(int)));
    check_cuda(cudaMalloc(&d_token_to_cluster_pos_, n * sizeof(uint32_t)));

    if (low_memory_load_) {
        const auto path = (std::filesystem::path(load_directory_) / "cluster_1bit.bin").string();
        const std::vector<size_t> sizes = {
            n * code_per_vec, n * sizeof(float), n * sizeof(int), n * sizeof(uint32_t)};
        char* destinations[] = {
            d_clustered_one_bit_codes_,
            reinterpret_cast<char*>(d_clustered_one_bit_factors_),
            reinterpret_cast<char*>(d_clustered_doc_ids_),
            reinterpret_cast<char*>(d_token_to_cluster_pos_)};
        stream_index_segments(path, sizes, load_options_.transfer_chunk_bytes,
            [&](size_t segment, size_t offset, const char* data, size_t count) {
                check_cuda(cudaMemcpy(destinations[segment] + offset, data, count,
                                      cudaMemcpyHostToDevice));
            });
    } else {
        const char* clustered_code = clustered_data_.data();
        const auto* clustered_factor = reinterpret_cast<const float*>(
            clustered_code + n * code_per_vec);
        const auto* clustered_doc_ids = reinterpret_cast<const int*>(
            reinterpret_cast<const char*>(clustered_factor) + n * sizeof(float));
        const auto* token_to_cluster_pos = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const char*>(clustered_doc_ids) + n * sizeof(int));
        check_cuda(cudaMemcpy(d_clustered_one_bit_codes_, clustered_code,
                              n * code_per_vec, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_clustered_one_bit_factors_, clustered_factor,
                              n * sizeof(float), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_clustered_doc_ids_, clustered_doc_ids,
                              n * sizeof(int), cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_token_to_cluster_pos_, token_to_cluster_pos,
                              n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    if (options_.doc_order_onebit) {
        static_assert(PADDED_DIM == 128, "Document-order replica expects D128/16B codes");
        check_cuda(cudaMalloc(&d_doc_order_onebit_codes_, n*16));
        check_cuda(cudaMalloc(&d_doc_order_onebit_factors_, n*sizeof(float)));
        const int blocks=int(std::min(size_t(65535),(n+255)/256));
        doc_order_detail::reorder<<<blocks,256>>>(
            reinterpret_cast<const uint4*>(d_clustered_one_bit_codes_),
            d_clustered_one_bit_factors_,d_token_to_cluster_pos_,
            reinterpret_cast<uint4*>(d_doc_order_onebit_codes_),d_doc_order_onebit_factors_,n);
        check_cuda(cudaGetLastError());
        unsigned* mismatch=nullptr;
        check_cuda(cudaMalloc(&mismatch,sizeof(unsigned)));
        try {
            check_cuda(cudaMemset(mismatch,0,sizeof(unsigned)));
            doc_order_detail::verify<<<blocks,256>>>(
                reinterpret_cast<const uint4*>(d_clustered_one_bit_codes_),
                d_clustered_one_bit_factors_,d_token_to_cluster_pos_,
                reinterpret_cast<const uint4*>(d_doc_order_onebit_codes_),d_doc_order_onebit_factors_,n,mismatch);
            check_cuda(cudaGetLastError());
            unsigned bad=0; check_cuda(cudaMemcpy(&bad,mismatch,sizeof(bad),cudaMemcpyDeviceToHost));
            if(bad) throw std::runtime_error("Document-order onebit full payload verification failed");
        } catch (...) { (void)cudaFree(mismatch); throw; }
        check_cuda(cudaFree(mismatch));
        std::cout << "[DOC_ORDER_ONEBIT] tokens="<<n<<" replica_bytes="<<n*20
                  <<" full_payload_bitwise=passed initialization_only=1"<<std::endl;
    }

    if (uses_flat_centroid_route(options_.algorithm_variant)) {
        if (!options_.expected_coarse_clusters || n_clusters != options_.expected_coarse_clusters)
            throw std::invalid_argument("Flat algorithm ablation requires the explicitly declared coarse cluster count");
        const auto& graph = *ivf->pg_index->index_cagra;
        if (graph.size() != n_clusters || graph.dim() != PADDED_DIM)
            throw std::invalid_argument("Exact flat ablation centroid matrix shape mismatch");
        if (graph.metric() != cuvs::distance::DistanceType::L2Expanded)
            throw std::invalid_argument("Exact flat ablation requires the original default L2Expanded metric");
    }
    if (options_.cagra_memory_placement != CagraMemoryPlacement::Device)
        ivf->pg_index->place_in_ats_hugepages(
            options_.cagra_memory_placement == CagraMemoryPlacement::AtsDatasetAndGraph);
    compute_candidate_bounds();
    allocate_workspace();
    if (options_.uma_cpu_candidates) {
        // Initialization only: replace the device allocation with one shared ATS
        // allocation. RAII retains the old device owner if allocation/copy fails.
        constexpr size_t page = size_t(2) << 20;
        const size_t bytes = n * sizeof(int);
        if (bytes > std::numeric_limits<size_t>::max() - (page - 1))
            throw std::overflow_error("UMA corpus allocation size overflow");
        const size_t rounded = (bytes + page - 1) & ~(page - 1);
        void* raw = nullptr;
        if (posix_memalign(&raw, page, std::max(page, rounded)) != 0)
            throw std::bad_alloc();
        std::unique_ptr<void, decltype(&std::free)> allocation(raw, &std::free);
        if (madvise(raw, std::max(page, rounded), MADV_HUGEPAGE) != 0)
            throw std::runtime_error("Cannot request huge pages for UMA candidate corpus");
        std::memset(raw, 0, std::max(page, rounded));
        check_cuda(cudaMemcpy(raw, d_clustered_doc_ids_, bytes, cudaMemcpyDeviceToHost));
        check_cuda(cudaFree(d_clustered_doc_ids_));
        h_uma_clustered_doc_ids_ = static_cast<int*>(allocation.release());
        d_clustered_doc_ids_ = h_uma_clustered_doc_ids_;
        std::cout << "[UMA_ALLOC] name=clustered_doc_ids ptr=" << raw
                  << " requested_bytes=" << bytes << " allocated_bytes=" << std::max(page, rounded)
                  << " alignment=" << page << std::endl;
    }
    if (uses_flat_centroid_route(options_.algorithm_variant)) {
        const auto matrix = ivf->pg_index->index_cagra->dataset();
        ws_->flat_router = std::make_unique<FlatL2Router>(matrix.data_handle(), n_clusters, PADDED_DIM, matrix.stride(0), Q_DOCLEN);
    }
    std::cout << "[ALGORITHM_WORKSPACE] max_CG_documents=" << max_candidate_docs_
              << " max_DS_documents=" << ws_->max_refined_candidates
              << " max_DS_tokens=" << ws_->max_refined_tokens
              << " token_score_bytes=" << ws_->max_refined_tokens * size_t(Q_DOCLEN) * sizeof(float)
              << " CR_arena=" << (skips_candidate_refinement(options_.algorithm_variant) ? "absent" : "original")
              << " flat_scratch_bytes=" << (ws_->flat_router ? ws_->flat_router->scratch_bytes() : 0) << '\n';
    if (!batch_owner_ && (options_.uma_cpu_fullbit_fraction > 0 || options_.uma_cpu_document)) {
        cpu_task_pool_ = std::make_shared<CpuTaskPool>(options_.uma_cpu_threads,
            CpuTaskPool::parse_cpu_list(options_.uma_cpu_pool_cpus));
        std::cout << "[CPU_POOL] workers=" << options_.uma_cpu_threads
                  << " cores=" << (options_.uma_cpu_pool_cpus.empty() ? "E-default" : options_.uma_cpu_pool_cpus)
                  << " fullbit_fraction=" << options_.uma_cpu_fullbit_fraction
                  << " document_reduce=" << options_.uma_cpu_document << '\n';
    }
    search_ready_ = true;
}

}  // namespace Chimera

namespace Chimera {

void chimera_index::prepare_batch(size_t workers) {
    if (!has_index_ || batch_owner_)
        throw std::runtime_error("Batch search requires a loaded owner index");
    if (workers == 0 || workers > 20)
        throw std::invalid_argument("Batch workers must be in [1, 20]");
    if (options_.algorithm_variant != AlgorithmVariant::Full)
        throw std::invalid_argument("Multi-stream batch currently supports the full algorithm");
    if (!search_ready_) initialize_search_state();
    // Worker zero is the owner itself, preserving the existing B1 path.
    while (batch_workers_.size() + 1 < workers) {
        auto worker = std::make_unique<chimera_index>();
        worker->batch_owner_ = this;
        worker->options_ = options_;
        worker->cpu_task_pool_ = cpu_task_pool_;
        worker->n = n;
        worker->d = d;
        worker->n_clusters = n_clusters;
        worker->ex_bits = ex_bits;
        worker->num_docs = num_docs;
        worker->max_cluster_size = max_cluster_size;
        worker->max_probed_clusters_ = max_probed_clusters_;
        worker->max_retrieved_tokens_per_query_ = max_retrieved_tokens_per_query_;
        worker->max_candidate_docs_per_query_ = max_candidate_docs_per_query_;
        worker->unpack_func_ = unpack_func_;
        worker->rotator_ = rotator_;
        worker->ivf = ivf;
        worker->d_doc_ptrs_ = d_doc_ptrs_;
        worker->d_cluster_pos_ = d_cluster_pos_;
        worker->d_clustered_one_bit_codes_ = d_clustered_one_bit_codes_;
        worker->d_doc_order_onebit_codes_ = d_doc_order_onebit_codes_;
        worker->d_doc_order_onebit_factors_ = d_doc_order_onebit_factors_;
        worker->d_clustered_one_bit_factors_ = d_clustered_one_bit_factors_;
        worker->d_clustered_doc_ids_ = d_clustered_doc_ids_;
        worker->d_token_to_cluster_pos_ = d_token_to_cluster_pos_;
        worker->cagra_res_ = std::make_unique<raft::resources>();
        worker->ws_ = std::make_unique<Workspace>();
        worker->has_index_ = true;
        worker->allocate_workspace();
        worker->search_ready_ = true;
        batch_workers_.push_back(std::move(worker));
    }
}

std::vector<std::vector<size_t>> chimera_index::search_batch(
    const float* queries, size_t query_values, size_t k, size_t workers) {
    if (!has_index_ || !d)
        throw std::runtime_error("Cannot search an empty Chimera index");
    const size_t stride = Q_DOCLEN * d;
    if (query_values % stride || (query_values && !queries) || !k)
        throw std::invalid_argument("Batch queries must have shape [B, 32, d] and k > 0");
    if (!workers || workers > 20)
        throw std::invalid_argument("Batch workers must be in [1, 20]");
    const size_t count = query_values / stride;
    last_batch_result_scores_.clear();
    const bool query_profile = profiling_enabled_ && profile_query_base_ >= 0;
    if (query_profile && count && uint64_t(count - 1) >
            uint64_t(std::numeric_limits<int64_t>::max() - profile_query_base_))
        throw std::overflow_error("Profile query ID overflow");
    if (query_profile && (uint64_t(profile_query_base_) > last_batch_profile_stats_.size() ||
            count > last_batch_profile_stats_.size() - size_t(profile_query_base_)))
        throw std::invalid_argument("Prepare profile query stats before capture");
    if (!count) return {};
    workers = std::min(workers, count);
    prepare_batch(workers);
    const bool deferred = options_.uma_cpu_fullbit_deferred && count > 1;
    std::vector<std::shared_ptr<PendingFullbit>> jobs(deferred ? count : 0);
    std::vector<std::vector<size_t>> results(count);
    if (capture_scores_) last_batch_result_scores_.resize(count);
    const bool capture = capture_scores_;
    const bool profile = profiling_enabled_;
    std::vector<std::exception_ptr> errors(workers);
    int device = 0;
    check_cuda(cudaGetDevice(&device));
    // One host submitter per lane allows the existing stream synchronizations
    // to wait independently. OpenMP reuses its thread team between batches.
    // Each loop iteration owns its lane even when OMP_DYNAMIC reduces the team.
#pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int lane = 0; lane < static_cast<int>(workers); ++lane) {
        auto& worker = lane == 0 ? *this : *batch_workers_[lane - 1];
        worker.capture_scores_ = capture;
        worker.profiling_enabled_ = profile;
        worker.defer_fullbit_in_batch_ = deferred;
        size_t active_query = count;
        try {
            check_cuda(cudaSetDevice(device));
            for (size_t q = lane; q < count; q += workers) {
                active_query = q;
                worker.profile_query_id_ = query_profile ? profile_query_base_ + int64_t(q) : -1;
                if (query_profile) {
                    worker.profile_document_tiles_ = 0;
                    worker.profile_fullbit_tasks_ = 0;
                    worker.profile_refined_docs_ = 0;
                    worker.profile_refined_tokens_ = 0;
                    worker.profile_fullbit_candidates_ = 0;
                    worker.profile_fullbit_tokens_ = 0;
                }
                ProfileRange query_range(query_profile, "Chimera/query", worker.profile_query_id_);
                if (worker.options_.verify_full_bit) {
                    try { results[q] = worker.search_one(queries + q * stride, k); }
                    catch (const std::exception& error) {
                        throw std::runtime_error("Candidate verification query_in_batch=" +
                            std::to_string(q) + " lane=" + std::to_string(lane) + ": " + error.what());
                    }
                } else {
                    results[q] = worker.search_one(queries + q * stride, k);
                }
                if (query_profile) last_batch_profile_stats_[size_t(profile_query_base_) + q] = QueryProfileStats{
                    worker.profile_query_id_, size_t(lane), q / workers,
                    worker.profile_document_tiles_, worker.profile_fullbit_tasks_,
                    worker.profile_refined_docs_, worker.profile_refined_tokens_,
                    worker.profile_fullbit_candidates_, worker.profile_fullbit_tokens_};
                if (deferred && worker.pending_fullbit_) {
                    // The job owns its query/IDs/scores. This lane can reuse
                    // its workspace immediately for the next GPU front end.
                    jobs[q] = std::move(worker.pending_fullbit_);
                } else if (capture) {
                    last_batch_result_scores_[q] = worker.last_result_scores_;
                }
            }
        } catch (...) {
            // Retain a job even if an exception occurs after publication but
            // before the ordinary hand-off. Every published group is drained.
            if (deferred && active_query < count && worker.pending_fullbit_)
                jobs[active_query] = std::move(worker.pending_fullbit_);
            errors[lane] = std::current_exception();
        }
        worker.defer_fullbit_in_batch_ = false;
        worker.profile_query_id_ = -1;
    }
    std::exception_ptr failure;
    for (const auto& error : errors) {
        if (error && !failure) failure = error;
    }
    // This is still a synchronous public API. Complete every detached tail
    // before returning, releasing workspace/corpus, or rethrowing an error.
    for (size_t q = 0; q < jobs.size(); ++q) {
        if (!jobs[q]) continue;
        try {
            if (failure) {
                wait_deferred_fullbit(jobs[q]);
            } else {
                finish_deferred_fullbit(jobs[q], k, results[q],
                    capture ? &last_batch_result_scores_[q] : nullptr);
            }
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
        jobs[q].reset();
    }
    if (deferred && capture && !failure) {
        try {
            // Preserve each lane's last_result_scores() observation as well
            // as the explicit input-ordered batch score arrays.
            for (size_t lane = 0; lane < workers; ++lane) {
                const size_t last_query = lane + ((count - 1 - lane) / workers) * workers;
                auto& worker = lane == 0 ? *this : *batch_workers_[lane - 1];
                worker.last_result_scores_ = last_batch_result_scores_[last_query];
            }
        } catch (...) {
            failure = std::current_exception();
        }
    }
    if (failure) {
        // CPU groups have all joined; now drain GPU users before freeing the
        // borrowed corpus. The batch has no outstanding cross-call work.
        (void)cudaDeviceSynchronize();
        release_resources();
        std::rethrow_exception(failure);
    }
    return results;
}
}  // namespace Chimera
