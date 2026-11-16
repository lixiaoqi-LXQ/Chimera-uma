#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace Chimera {

// Every Workspace owns one instance. Nothing is borrowed from the batch owner.
// Reserve follows candidate-arena growth; all allocation is outside the selector.
struct PartialTopKWorkspace {
    PartialTopKWorkspace() = default;
    ~PartialTopKWorkspace();
    PartialTopKWorkspace(const PartialTopKWorkspace&) = delete;
    PartialTopKWorkspace& operator=(const PartialTopKWorkspace&) = delete;

    void reserve(size_t rows, size_t selected, bool verification);
    void release() noexcept;
    void select(const float* scores, const int* original_ids, int rows, int selected,
                cudaStream_t stream, bool verification, void* reference_temp,
                size_t reference_temp_bytes, float* reference_scores, int* reference_ids);

    const int* selected_ids() const noexcept { return ids; }

 private:
    void swap(PartialTopKWorkspace& other) noexcept;
    uint64_t* keys = nullptr;
    uint64_t* selected_keys = nullptr;
    uint32_t* score_bits = nullptr;
    int* ids = nullptr;
    int* selected_rows = nullptr;  // QA only; no timed row-output store.
    int* oracle_input_rows = nullptr;  // QA only, initialized by the actual pack kernel.
    void* temporary = nullptr;
    size_t temporary_bytes = 0;
    size_t row_capacity = 0;
    size_t selected_capacity = 0;
    bool verification_storage = false;
};

}  // namespace Chimera
