#include "chimera/ivf_pg.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <new>
#include <sys/mman.h>
#include <system_error>
#include <utility>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <rmm/cuda_stream_view.hpp>

namespace Chimera {

using cagra_index_t = cuvs::neighbors::cagra::index<float, uint32_t>;

namespace {
constexpr size_t kRouteHugePage = size_t{2} << 20;
using AtsRouteOwner = std::unique_ptr<void, PG_CAGRA::FreeAts>;

void route_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

AtsRouteOwner allocate_ats_route(size_t bytes, const char* name) {
    if (!bytes || bytes > std::numeric_limits<size_t>::max() - (kRouteHugePage - 1))
        throw std::overflow_error("CAGRA ATS allocation size overflow");
    const size_t rounded = (bytes + kRouteHugePage - 1) & ~(kRouteHugePage - 1);
    void* raw = nullptr;
    if (::posix_memalign(&raw, kRouteHugePage, rounded) != 0) throw std::bad_alloc();
    AtsRouteOwner allocation(raw);
    if (::madvise(raw, rounded, MADV_HUGEPAGE) != 0)
        throw std::system_error(errno, std::generic_category(), "CAGRA MADV_HUGEPAGE");
    // CPU first touch before the initialization-only device-to-CPU copy.
    std::memset(raw, 0, rounded);
    std::cout << "[UMA_ALLOC] name=" << name << " ptr=" << raw
              << " requested_bytes=" << bytes << " allocated_bytes=" << rounded
              << " alignment=" << kRouteHugePage << std::endl;
    return allocation;
}
}


PG_CAGRA::PG_CAGRA(size_t n, size_t d) : n(n), d(d) {}

void PG_CAGRA::build_index(const float* data) {
    if (d == 0 || n > std::numeric_limits<size_t>::max() / d / sizeof(float))
        throw std::invalid_argument("Explicit centroid dataset size overflow");
    index_cagra.reset();
    explicit_dataset_.reset();
    ats_dataset_.reset();
    ats_graph_.reset();
    float* device_data = nullptr;
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&device_data), n * d * sizeof(float));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    explicit_dataset_.reset(device_data);
    status = cudaMemcpy(device_data, data, n * d * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    auto dataset = raft::make_device_matrix_view(
        static_cast<const float*>(device_data),
        static_cast<std::int64_t>(n),
        static_cast<std::int64_t>(d));
    cuvs::neighbors::cagra::index_params index_params;
    index_cagra = std::make_unique<cagra_index_t>(
        cuvs::neighbors::cagra::build(res_, index_params, dataset));
}


void PG_CAGRA::place_in_ats_hugepages(bool include_graph) {
    if (!index_cagra || ats_dataset_ || ats_graph_)
        throw std::logic_error("CAGRA ATS placement requires a fresh loaded index");
    int device = 0, pageable = 0, host_tables = 0;
    route_cuda(cudaGetDevice(&device), "CAGRA ATS device");
    route_cuda(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device),
               "CAGRA ATS pageable access");
    route_cuda(cudaDeviceGetAttribute(&host_tables,
                   cudaDevAttrPageableMemoryAccessUsesHostPageTables, device),
               "CAGRA ATS host page tables");
    if (!pageable || !host_tables)
        throw std::runtime_error("CAGRA ATS placement requires hardware host-page-table access");
    const auto data = index_cagra->dataset();
    const auto graph = index_cagra->graph();
    if (index_cagra->metric() != cuvs::distance::DistanceType::L2Expanded ||
        data.extent(0) != static_cast<int64_t>(n) ||
        data.extent(1) != static_cast<int64_t>(d) ||
        data.stride(0) != static_cast<int64_t>(d) || data.stride(1) != 1 ||
        (d * sizeof(float)) % 16 != 0 || !data.data_handle() ||
        graph.extent(0) != static_cast<int64_t>(n) || !graph.data_handle() ||
        index_cagra->source_indices().has_value() ||
        index_cagra->dataset_fd().has_value() || index_cagra->graph_fd().has_value() ||
        index_cagra->mapping_fd().has_value())
        throw std::invalid_argument("CAGRA ATS placement requires original contiguous FP32 L2 graph/data");
    const size_t degree = static_cast<size_t>(graph.extent(1));
    if (!d || !degree || n > std::numeric_limits<size_t>::max() / d / sizeof(float) ||
        n > std::numeric_limits<size_t>::max() / degree / sizeof(uint32_t))
        throw std::overflow_error("CAGRA ATS payload size overflow");
    const size_t data_bytes = n * d * sizeof(float);
    const size_t graph_bytes = n * degree * sizeof(uint32_t);
    raft::resource::sync_stream(res_);
    auto dataset_owner = allocate_ats_route(data_bytes, "cagra_dataset");
    route_cuda(cudaMemcpy(dataset_owner.get(), data.data_handle(), data_bytes,
                           cudaMemcpyDeviceToHost), "CAGRA ATS copy dataset");
    AtsRouteOwner graph_owner;
    if (include_graph) {
        graph_owner = allocate_ats_route(graph_bytes, "cagra_graph");
        route_cuda(cudaMemcpy(graph_owner.get(), graph.data_handle(), graph_bytes,
                               cudaMemcpyDeviceToHost), "CAGRA ATS copy graph");
    }

    // Do not call make_aligned_dataset: its cudaPointerGetAttributes probe can
    // classify an ordinary ATS pointer as unregistered and silently make a GPU copy.
    auto shared_dataset = std::make_unique<cuvs::neighbors::non_owning_dataset<float, int64_t>>(
        raft::make_device_strided_matrix_view<const float, int64_t>(
            static_cast<const float*>(dataset_owner.get()), n, d, d));
    if (include_graph) {
        // update_graph(view) alone retains the old index's private GPU graph owner.
        // A fresh thin index releases both old GPU owners after the pointer switch.
        auto replacement = std::make_unique<cagra_index_t>(res_, index_cagra->metric());
        replacement->update_dataset(res_, std::move(shared_dataset));
        replacement->update_graph(res_, raft::make_device_matrix_view<const uint32_t, int64_t>(
            static_cast<const uint32_t*>(graph_owner.get()), n, degree));
        if (replacement->dataset().data_handle() != dataset_owner.get() ||
            replacement->graph().data_handle() != graph_owner.get() ||
            replacement->data().is_owning())
            throw std::runtime_error("CAGRA ATS thin-wrapper pointer/ownership mismatch");
        index_cagra.swap(replacement);
        replacement.reset();
    } else {
        // Dataset replacement releases the index-owned GPU dataset immediately.
        index_cagra->update_dataset(res_, std::move(shared_dataset));
        if (index_cagra->dataset().data_handle() != dataset_owner.get() ||
            index_cagra->data().is_owning())
            throw std::runtime_error("CAGRA ATS dataset pointer/ownership mismatch");
    }
    explicit_dataset_.reset();
    ats_dataset_ = std::move(dataset_owner);
    ats_graph_ = std::move(graph_owner);
    std::cout << "[CAGRA_UMA] placement=" << (include_graph ? "ats-both" : "ats-dataset")
              << " dataset_alias_verified=1 dataset_is_owning=0 graph_alias_verified="
              << (include_graph ? 1 : 0) << " dataset_bytes=" << data_bytes
              << " graph_bytes=" << (include_graph ? graph_bytes : 0)
              << " rows=" << n << " dimension=" << d << " graph_degree=" << degree
              << " initialization_copy_bytes=" << (data_bytes + (include_graph ? graph_bytes : 0))
              << " per_query_corpus_copy_bytes=0" << std::endl;
}

void PG_CAGRA::search_centroids(
    const float* d_queries,
    size_t n_queries,
    size_t nprobe,
    float* d_dists,
    uint32_t* d_labels,
    cudaStream_t stream,
    size_t cagra_itopk_size) {
    search_centroids(
        res_, d_queries, n_queries, nprobe, d_dists, d_labels, stream,
        cagra_itopk_size);
}

void PG_CAGRA::search_centroids(
    const raft::resources& res,
    const float* d_queries,
    size_t n_queries,
    size_t nprobe,
    float* d_dists,
    uint32_t* d_labels,
    cudaStream_t stream,
    size_t cagra_itopk_size) {
    raft::resource::set_cuda_stream(res, rmm::cuda_stream_view(stream));

    auto queries = raft::make_device_matrix_view(
        d_queries,
        static_cast<std::int64_t>(n_queries),
        static_cast<std::int64_t>(d));
    auto distances = raft::make_device_matrix_view(
        d_dists,
        static_cast<std::int64_t>(n_queries),
        static_cast<std::int64_t>(nprobe));
    auto labels = raft::make_device_matrix_view(
        d_labels,
        static_cast<std::int64_t>(n_queries),
        static_cast<std::int64_t>(nprobe));

    cuvs::neighbors::cagra::search_params search_params;
    search_params.itopk_size = cagra_itopk_size;
    cuvs::neighbors::cagra::search(
        res, search_params, *index_cagra, queries, labels, distances);
}

void PG_CAGRA::save(const std::string& graph_path) const {
    cuvs::neighbors::cagra::serialize(res_, graph_path, *index_cagra);
}

void PG_CAGRA::load(const std::string& graph_path) {
    // Destroy the old view before releasing its explicit backing allocation.
    index_cagra.reset();
    explicit_dataset_.reset();
    ats_dataset_.reset();
    ats_graph_.reset();
    index_cagra = std::make_unique<cagra_index_t>(res_);
    cuvs::neighbors::cagra::deserialize(res_, graph_path, index_cagra.get());
}

IVF_PG::IVF_PG(size_t n_clusters, size_t d)
    : n_clusters(n_clusters), d(d), pg_index(new PG_CAGRA(n_clusters, d)) {}

IVF_PG::~IVF_PG() {
    delete pg_index;
}

void IVF_PG::search_centroids(
    const float* d_queries,
    size_t n_queries,
    size_t nprobe,
    float* d_dists,
    uint32_t* d_labels,
    cudaStream_t stream,
    size_t cagra_itopk_size) {
    pg_index->search_centroids(
        d_queries, n_queries, nprobe, d_dists, d_labels, stream,
        cagra_itopk_size);
}

void IVF_PG::search_centroids(
    const raft::resources& res,
    const float* d_queries,
    size_t n_queries,
    size_t nprobe,
    float* d_dists,
    uint32_t* d_labels,
    cudaStream_t stream,
    size_t cagra_itopk_size) {
    pg_index->search_centroids(
        res, d_queries, n_queries, nprobe, d_dists, d_labels, stream,
        cagra_itopk_size);
}

void IVF_PG::build_from_assignments(const uint32_t* list_nos, size_t n_vectors) {
    cluster_pos.assign(n_clusters + 1, 0);
    for (size_t i = 0; i < n_vectors; ++i) {
        ++cluster_pos[list_nos[i] + 1];
    }

    for (size_t i = 0; i < n_clusters; ++i) {
        cluster_pos[i + 1] += cluster_pos[i];
    }

    inv_list.resize(n_vectors);
    std::vector<size_t> write_pos = cluster_pos;
    for (size_t i = 0; i < n_vectors; ++i) {
        const auto cluster_id = list_nos[i];
        inv_list[write_pos[cluster_id]++] = static_cast<uint32_t>(i);
    }
}

int IVF_PG::max_cluster_size() const {
    int max_size = 0;
    for (size_t i = 0; i < n_clusters; ++i) {
        max_size = std::max(max_size, static_cast<int>(cluster_pos[i + 1] - cluster_pos[i]));
    }
    return max_size;
}

void IVF_PG::save(const std::string& ivf_path, const std::string& graph_path) const {
    pg_index->save(graph_path);

    std::ofstream output(ivf_path, std::ios::binary);
    const size_t n = inv_list.size();
    output.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
    output.write(
        reinterpret_cast<const char*>(inv_list.data()),
        static_cast<std::streamsize>(n * sizeof(uint32_t)));
    output.write(
        reinterpret_cast<const char*>(cluster_pos.data()),
        static_cast<std::streamsize>(cluster_pos.size() * sizeof(size_t)));
}

void IVF_PG::load(const std::string& ivf_path, const std::string& graph_path) {
    std::ifstream input(ivf_path, std::ios::binary);
    size_t n;
    input.read(reinterpret_cast<char*>(&n), sizeof(size_t));
    inv_list.resize(n);
    input.read(
        reinterpret_cast<char*>(inv_list.data()),
        static_cast<std::streamsize>(n * sizeof(uint32_t)));
    cluster_pos.resize(n_clusters + 1);
    input.read(
        reinterpret_cast<char*>(cluster_pos.data()),
        static_cast<std::streamsize>((n_clusters + 1) * sizeof(size_t)));

    pg_index->load(graph_path);
}

}  // namespace Chimera
