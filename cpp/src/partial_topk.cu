#include "partial_topk.cuh"
#include "chimera/chimera_index.cuh"

#include <cub/version.cuh>
#if CUB_VERSION >= 300300 && __has_include(<cub/device/device_topk.cuh>)
#define CHIMERA_HAS_PARTIAL_TOPK 1
#include <cub/device/device_topk.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_radix_sort.cuh>
#else
#define CHIMERA_HAS_PARTIAL_TOPK 0
#endif
#include <algorithm>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Chimera {

bool partial_topk_available() noexcept { return CHIMERA_HAS_PARTIAL_TOPK != 0; }
int partial_topk_cub_version() noexcept { return CUB_VERSION; }

#if CHIMERA_HAS_PARTIAL_TOPK
namespace partial_topk_detail {

// Same packed comparison as the independently qualified standalone micro.
// Canonicalize zero only for comparison; preserve original score payload bits.
__device__ __forceinline__ uint32_t float_rank(uint32_t bits) {
    if ((bits & 0x7fffffffu) == 0) return 0x80000000u;
    return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__global__ void pack_rank_row(const uint32_t* bits, uint64_t* keys,
                              int* oracle_rows, int rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < static_cast<uint32_t>(rows)) {
        keys[row] = (uint64_t(float_rank(bits[row])) << 32) | uint32_t(~uint32_t(row));
        if (oracle_rows) oracle_rows[row] = static_cast<int>(row);
    }
}

__global__ void finish(const uint64_t* selected, const uint32_t* input_bits,
                       const int* input_ids, uint32_t* output_bits, int* output_ids,
                       int* output_rows, int count) {
    using Sort = cub::BlockRadixSort<uint64_t, 256, 4>;
    __shared__ typename Sort::TempStorage scratch;
    uint64_t local[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int rank = static_cast<int>(threadIdx.x) * 4 + i;
        local[i] = rank < count ? selected[rank] : uint64_t(0);
    }
    Sort(scratch).SortDescending(local);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int rank = static_cast<int>(threadIdx.x) * 4 + i;
        if (rank < count) {
            const uint32_t row = ~uint32_t(local[i]);
            output_ids[rank] = input_ids[row];
            output_bits[rank] = input_bits[row];
            if (output_rows) output_rows[rank] = static_cast<int>(row);
        }
    }
}

// Explicit types are required with this installed CCCL/C++17 + nvcc13.0 build.
using Requirements = decltype(cuda::execution::require(
    cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted));
using Stream = cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>;
using Environment = cuda::std::execution::env<Requirements, Stream>;
Environment environment(cudaStream_t stream) {
    return Environment{
        cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                  cuda::execution::output_ordering::unsorted),
        Stream{cuda::get_stream, cuda::stream_ref{stream}}};
}

}  // namespace partial_topk_detail
#endif

PartialTopKWorkspace::~PartialTopKWorkspace() { release(); }

void PartialTopKWorkspace::swap(PartialTopKWorkspace& other) noexcept {
    using std::swap;
    swap(keys, other.keys); swap(selected_keys, other.selected_keys);
    swap(score_bits, other.score_bits); swap(ids, other.ids);
    swap(selected_rows, other.selected_rows); swap(oracle_input_rows, other.oracle_input_rows);
    swap(temporary, other.temporary); swap(temporary_bytes, other.temporary_bytes);
    swap(row_capacity, other.row_capacity); swap(selected_capacity, other.selected_capacity);
    swap(verification_storage, other.verification_storage);
}

void PartialTopKWorkspace::release() noexcept {
    if (keys) { (void)cudaFree(keys); keys = nullptr; }
    if (selected_keys) { (void)cudaFree(selected_keys); selected_keys = nullptr; }
    if (score_bits) { (void)cudaFree(score_bits); score_bits = nullptr; }
    if (ids) { (void)cudaFree(ids); ids = nullptr; }
    if (selected_rows) { (void)cudaFree(selected_rows); selected_rows = nullptr; }
    if (oracle_input_rows) { (void)cudaFree(oracle_input_rows); oracle_input_rows = nullptr; }
    if (temporary) { (void)cudaFree(temporary); temporary = nullptr; }
    temporary_bytes = row_capacity = selected_capacity = 0;
    verification_storage = false;
}

void PartialTopKWorkspace::reserve(size_t rows, size_t selected, bool verification) {
#if !CHIMERA_HAS_PARTIAL_TOPK
    throw std::invalid_argument("Partial TopK requires CUB >=3.3.0 with DeviceTopK support");
#else
    if (rows == 0 || rows > size_t(std::numeric_limits<int>::max()) ||
        selected == 0 || selected > 1000)
        throw std::invalid_argument("Partial TopK requires int-sized positive N and 1..1000 selected capacity");
    if (row_capacity >= rows && selected_capacity >= selected &&
        (!verification || verification_storage)) return;
    // Stage all allocations before changing a live lane. On failure the staged
    // owner's destructor releases only its allocations, preserving the old owner.
    PartialTopKWorkspace next;
    next.row_capacity = rows;
    next.selected_capacity = selected;
    next.verification_storage = verification;
    check_cuda(cudaMalloc(&next.keys, rows * sizeof(uint64_t)), "Partial TopK input keys");
    check_cuda(cudaMalloc(&next.selected_keys, selected * sizeof(uint64_t)), "Partial TopK selected keys");
    check_cuda(cudaMalloc(&next.score_bits, selected * sizeof(uint32_t)), "Partial TopK selected scorebits");
    check_cuda(cudaMalloc(&next.ids, selected * sizeof(int)), "Partial TopK selected IDs");
    if (verification) {
        check_cuda(cudaMalloc(&next.selected_rows, selected * sizeof(int)), "Partial TopK QA selected rows");
        check_cuda(cudaMalloc(&next.oracle_input_rows, rows * sizeof(int)), "Partial TopK QA original rows");
    }
    check_cuda(cub::DeviceTopK::MaxKeys(nullptr, next.temporary_bytes, next.keys,
        next.selected_keys, static_cast<int>(rows), static_cast<int>(std::min(rows, selected)),
        partial_topk_detail::environment(nullptr)), "Partial TopK temporary query");
    check_cuda(cudaMalloc(&next.temporary, next.temporary_bytes), "Partial TopK temporary allocation");
    swap(next);
#endif
}

void PartialTopKWorkspace::select(
    const float* scores, const int* original_ids, int rows, int count,
    cudaStream_t stream, bool verification, void* reference_temp,
    size_t reference_temp_bytes, float* reference_scores, int* reference_ids) {
#if !CHIMERA_HAS_PARTIAL_TOPK
    throw std::invalid_argument("Partial TopK requires CUB >=3.3.0 with DeviceTopK support");
#else
    if (!scores || !original_ids || rows <= 0 || size_t(rows) > row_capacity ||
        count <= 0 || count > rows || size_t(count) > selected_capacity || count > 1000 ||
        (verification && !verification_storage))
        throw std::logic_error("Partial TopK workspace/input extent mismatch");
    if (verification && (!oracle_input_rows || !selected_rows || !reference_temp ||
                         !reference_scores || !reference_ids))
        throw std::logic_error("Partial TopK QA storage is absent");
    const auto* bits = reinterpret_cast<const uint32_t*>(scores);
    partial_topk_detail::pack_rank_row<<<(static_cast<uint32_t>(rows) + 255u) / 256u, 256, 0, stream>>>(
        bits, keys, verification ? oracle_input_rows : nullptr, rows);
    check_cuda(cudaGetLastError(), "Partial TopK pack launch");
    size_t available = temporary_bytes;
    check_cuda(cub::DeviceTopK::MaxKeys(temporary, available, keys, selected_keys,
        rows, count, partial_topk_detail::environment(stream)), "Partial TopK selection");
    partial_topk_detail::finish<<<1, 256, 0, stream>>>(selected_keys, bits, original_ids,
        score_bits, ids, verification ? selected_rows : nullptr, count);
    check_cuda(cudaGetLastError(), "Partial TopK finish launch");
    if (!verification) return;

    // Both full-N CUB oracles consume the same original score/ID arrays.
    // Candidate outputs own separate K storage, so references cannot overwrite them.
    std::vector<int> actual_ids(count), actual_rows(count), expected_ids(count), expected_rows(count);
    std::vector<uint32_t> actual_bits(count), expected_bits(count), row_oracle_bits(count);
    check_cuda(cudaMemcpyAsync(actual_ids.data(), ids, size_t(count) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaMemcpyAsync(actual_rows.data(), selected_rows, size_t(count) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaMemcpyAsync(actual_bits.data(), score_bits, size_t(count) * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    size_t reference_available = reference_temp_bytes;
    check_cuda(cub::DeviceRadixSort::SortPairsDescending(reference_temp, reference_available,
        scores, reference_scores, original_ids, reference_ids, rows, 0, 32, stream),
        "Partial TopK full-N ID/score oracle");
    check_cuda(cudaMemcpyAsync(expected_ids.data(), reference_ids, size_t(count) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaMemcpyAsync(expected_bits.data(), reference_scores, size_t(count) * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    reference_available = reference_temp_bytes;
    check_cuda(cub::DeviceRadixSort::SortPairsDescending(reference_temp, reference_available,
        scores, reference_scores, oracle_input_rows, reference_ids, rows, 0, 32, stream),
        "Partial TopK full-N original-row oracle");
    check_cuda(cudaMemcpyAsync(expected_rows.data(), reference_ids, size_t(count) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaMemcpyAsync(row_oracle_bits.data(), reference_scores, size_t(count) * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaStreamSynchronize(stream), "Partial TopK actual outputs and two full-N oracles");
    for (int rank = 0; rank < count; ++rank) {
        if (actual_rows[rank] < 0 || actual_rows[rank] >= rows ||
            actual_rows[rank] != expected_rows[rank] || actual_ids[rank] != expected_ids[rank] ||
            actual_bits[rank] != expected_bits[rank] || expected_bits[rank] != row_oracle_bits[rank])
            throw std::runtime_error("Partial TopK differs from original stable CUB row/ID/scorebits at rank " + std::to_string(rank));
    }
    std::ostringstream line;
    line << "[PARTIAL_TOPK_VERIFY] candidates=" << rows << " selected=" << count
         << " full_N_ID_oracle=passed full_N_row_oracle=passed rows_exact=passed"
         << " ids_exact=passed scores_fp32_bitwise=passed\n";
    std::fprintf(stdout, "%s", line.str().c_str());
#endif
}

}  // namespace Chimera
