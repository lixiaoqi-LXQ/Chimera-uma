#include "chimera/cagra_half.hpp"
#include "chimera/chimera_index.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace Chimera {
namespace {

// FP32 bits for 65504, the greatest finite FP16 value. An unsigned magnitude
// comparison also rejects every Inf/NaN, independent of host FP exceptions.
constexpr uint32_t kHalfMaxFloatBits = 0x477fe000u;

__global__ void cagra_dataset_float_to_half(const float* source, __half* destination,
                                            size_t count, unsigned* invalid) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        const float x = source[i];
        if ((__float_as_uint(x) & 0x7fffffffu) > kHalfMaxFloatBits) {
            atomicExch(invalid, 1u);
            destination[i] = __float2half_rn(0.0f);
        } else {
            destination[i] = __float2half_rn(x);
        }
    }
}

struct FreeCuda {
    void operator()(unsigned* p) const noexcept { if (p) (void)cudaFree(p); }
};

}  // namespace

void initialize_cagra_half_dataset(const float* source, __half* destination,
                                   size_t count, cudaStream_t stream) {
    if (!source || !destination || !count)
        throw std::invalid_argument("CAGRA half conversion requires nonempty buffers");
    unsigned* raw_invalid = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&raw_invalid), sizeof(unsigned)),
               "Allocate CAGRA half domain flag");
    std::unique_ptr<unsigned, FreeCuda> invalid(raw_invalid);
    unsigned bad = 0;  // Must stay alive through the catch-path stream drain.
    try {
        check_cuda(cudaMemsetAsync(invalid.get(), 0, sizeof(unsigned), stream),
                   "Clear CAGRA half domain flag");
        constexpr unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>(
            65535, 1 + (count - 1) / threads));
        cagra_dataset_float_to_half<<<blocks, threads, 0, stream>>>(
            source, destination, count, invalid.get());
        check_cuda(cudaGetLastError(), "Launch CAGRA half dataset conversion");
        check_cuda(cudaMemcpyAsync(&bad, invalid.get(), sizeof(bad),
                                   cudaMemcpyDeviceToHost, stream),
                   "Read CAGRA half domain flag");
        check_cuda(cudaStreamSynchronize(stream), "Complete CAGRA half dataset initialization");
        if (bad)
            throw std::invalid_argument(
                "CAGRA half dataset requires finite FP32 values within [-65504,65504]");
    } catch (...) {
        // Drain before the caller's local half owner or our flag can be freed.
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

bool prepare_cagra_half_queries(const float* source, __half* destination,
                                size_t count) noexcept {
    if ((!source || !destination) && count) return false;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, source + i, sizeof(bits));
        if ((bits & 0x7fffffffu) > kHalfMaxFloatBits) return false;
        destination[i] = __float2half_rn(source[i]);
    }
    return true;
}

}  // namespace Chimera
