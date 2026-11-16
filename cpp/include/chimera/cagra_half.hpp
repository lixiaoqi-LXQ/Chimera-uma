#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstddef>

namespace Chimera {

// Initialization only: convert the original dense FP32 rows with RN-even,
// reject nonfinite/out-of-range values, and synchronize this stream before
// returning (including the exception path). Does not modify the source.
void initialize_cagra_half_dataset(const float* source, __half* destination,
                                   size_t count, cudaStream_t stream);

// CPU-only, after rotation and before any query GPU submission. No CUDA API,
// allocation, synchronization, or change to the original FP32 query buffer.
// False rejects nonfinite values or |x| > 65504; destination may be partial.
bool prepare_cagra_half_queries(const float* source, __half* destination,
                                size_t count) noexcept;

}  // namespace Chimera
