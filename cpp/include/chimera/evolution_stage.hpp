#pragma once

#include "chimera/chimera_index.cuh"
#include <stdexcept>
#include <string>

namespace Chimera {

enum class EvolutionStage { Custom, ChimeraBatch, GpuRerank, UmaCollaboration };

inline EvolutionStage parse_evolution_stage(const std::string& name) {
    if (name == "custom") return EvolutionStage::Custom;
    if (name == "chimera-batch") return EvolutionStage::ChimeraBatch;
    if (name == "gpu-rerank") return EvolutionStage::GpuRerank;
    if (name == "uma-collab") return EvolutionStage::UmaCollaboration;
    throw std::invalid_argument("Stage must be chimera-batch, gpu-rerank or uma-collab");
}

// Quality parameters remain caller-controlled. Mainline stage selection fixes
// the execution backend and excludes the archived kernel/metadata experiments.
inline void apply_evolution_stage(EvolutionStage stage, SearchOptions& options) {
    if (stage == EvolutionStage::Custom) return;
    if (options.shared_onebit_corpus || options.uma_cpu_onebit_fused)
        throw std::invalid_argument("Shared onebit experiments require the custom stage");
    if (options.compact_token_score || options.fast_document_reduce ||
        options.uma_query_buffers || options.uma_cpu_candidates || options.uma_cpu_refine ||
        options.cagra_memory_placement != CagraMemoryPlacement::Device)
        throw std::invalid_argument("Archived side optimizations are disabled in the three evolution stages");
    if (options.uma_cpu_document || options.uma_cpu_fullbit_deferred ||
        options.uma_cpu_fullbit_fraction != 0.0)
        throw std::invalid_argument("Choose the evolution stage without additional CPU backend switches");
    options.gpu_full_bit = stage != EvolutionStage::ChimeraBatch;
    if (stage == EvolutionStage::UmaCollaboration) {
        options.num_chunks = 1;
        options.uma_cpu_document = true;
        options.uma_cpu_fullbit_fraction = 1.0;
        options.uma_cpu_fullbit_deferred = true;
        options.uma_cpu_threads = 16;
        options.uma_cpu_pool_cpus = "0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17";
    }
}

}  // namespace Chimera
