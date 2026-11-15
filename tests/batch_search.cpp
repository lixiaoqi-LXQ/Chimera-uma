#include "chimera/chimera_index.cuh"
#include "chimera/evolution_stage.hpp"
#include "chimera/config.cuh"
#include "chimera/profiling.hpp"
#include "chimera/io.hpp"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ids = std::vector<std::vector<size_t>>;
// Pressure samples use the same Linux monotonic epoch, unlike wall-clock UTC.
static uint64_t monotonic_ns() {
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        throw std::runtime_error("CLOCK_MONOTONIC read failed");
    return uint64_t(value.tv_sec) * 1000000000ull + uint64_t(value.tv_nsec);
}

static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now()-t).count();
}
static std::vector<float> read_queries(const std::string& path, size_t& count, size_t& len, size_t& dim) {
    if (path.size() < 4 || path.substr(path.size()-4) != ".npy")
        return Chimera::load_query(len, count, dim, path);
    std::ifstream f(path, std::ios::binary);
    char magic[6]; f.read(magic, 6);
    if (std::memcmp(magic,"\x93NUMPY",6)) throw std::runtime_error("Invalid NPY magic");
    unsigned char version[2]; f.read(reinterpret_cast<char*>(version),2);
    uint32_t n = 0;
    if (version[0] == 1) { uint16_t x=0; f.read(reinterpret_cast<char*>(&x),2); n=x; }
    else if (version[0] == 2 || version[0] == 3) f.read(reinterpret_cast<char*>(&n),4);
    else throw std::runtime_error("Unsupported NPY version");
    std::string header(n,' '); f.read(header.data(),n);
    if (header.find("'<f4'")==std::string::npos && header.find("'=f4'")==std::string::npos)
        throw std::runtime_error("Queries require little-endian FP32 NPY");
    if (header.find("'fortran_order': False")==std::string::npos)
        throw std::runtime_error("Queries require C-order NPY");
    std::smatch m;
    if (!std::regex_search(header,m,std::regex("'shape': \\(([0-9]+), ([0-9]+), ([0-9]+),?\\)")))
        throw std::runtime_error("Queries require shape [N,32,D]");
    count=std::stoull(m[1]); len=std::stoull(m[2]); dim=std::stoull(m[3]);
    if (!count || len!=32 || dim!=128) throw std::runtime_error("Expected nonempty [N,32,128] queries");
    std::vector<float> values(count*len*dim);
    f.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(float));
    if (!f) throw std::runtime_error("Truncated query file");
    return values;
}
static void write_ids(std::ostream& out, const Ids& rows) {
    out << '[';
    for(size_t i=0;i<rows.size();++i) {
        if(i) out << ','; out << '[';
        for(size_t j=0;j<rows[i].size();++j) { if(j)out << ','; out << rows[i][j]; }
        out << ']';
    }
    out << ']';
}
static void require_rows(const Ids& rows, size_t count, size_t k) {
    if(rows.size()!=count)throw std::runtime_error("Incomplete query count");
    for(const auto& row:rows)if(row.size()!=k)throw std::runtime_error("Incomplete Top-k result");
}
int main(int argc,char** argv) try {
    std::string index_path, query_path, output_path, worker_list="1,2,4,8";
    size_t batch=32, rounds=3, limit=0, k=10, query_repeats=1;
    bool batch_only=false, profile_range=false;
    std::string cagra_memory="device", cagra_algo="auto";
    std::string stage_name="custom";
    bool custom_backend_switch=false, explicit_workers=false;
    Chimera::SearchOptions options;
    options.nprobe=32; options.k_refine=1000; options.k_full_bit=20;
    options.cagra_itopk_size=64; options.num_chunks=1;
    for(int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if(arg=="--cpu-full-bit") { options.gpu_full_bit=false;custom_backend_switch=true;continue; }
        if(arg=="--compact-token-score") { options.compact_token_score=true;continue; }
        if(arg=="--fast-document-reduce") { options.fast_document_reduce=true;continue; }
        if(arg=="--uma-cpu-refine") { options.uma_cpu_refine=true; options.uma_cpu_candidates=true; options.uma_query_buffers=true;continue; }
        if(arg=="--uma-cpu-candidates") { options.uma_cpu_candidates=true; options.uma_query_buffers=true;continue; }
        if(arg=="--uma-query-buffers") { options.uma_query_buffers=true;continue; }
        if(arg=="--uma-cpu-onebit-half") { options.uma_cpu_onebit_half=true;options.shared_onebit_corpus=true;custom_backend_switch=true;continue; }
        if(arg=="--shared-onebit-corpus") { options.shared_onebit_corpus=true;custom_backend_switch=true;continue; }
        if(arg=="--shared-onebit-base-pages") { options.shared_onebit_base_pages=true;custom_backend_switch=true;continue; }
        if(arg=="--uma-cpu-onebit-fused") { options.uma_cpu_onebit_fused=true;options.shared_onebit_corpus=true;custom_backend_switch=true;continue; }
        if(arg=="--uma-cpu-document") { options.uma_cpu_document=true;custom_backend_switch=true;continue; }
        if(arg=="--uma-cpu-fullbit-deferred") { options.uma_cpu_fullbit_deferred=true; options.uma_cpu_fullbit_fraction=1.0;custom_backend_switch=true;continue; }
        if(arg=="--verify-full-bit") { options.verify_full_bit=true;continue; }
        if(arg=="--batch-only") { batch_only=true;continue; }
        if(arg=="--profile-range") { profile_range=true;continue; }
        if(arg=="--help") {
            std::cout
                << "batch_search_benchmark --index DIR --queries FILE.npy|FILE.bin --output FILE.json\n"
                << "  --stage chimera-batch|gpu-rerank|uma-collab\n"
                << "    chimera-batch: original CPU fullbit + batch adaptation\n"
                << "    gpu-rerank:   move fullbit to GPU, original 1bit kernels\n"
                << "    uma-collab:   CPU document reduction + deferred CPU fullbit, 16-worker pool\n"
                << "  [--shared-onebit-corpus | --uma-cpu-onebit-fused | --uma-cpu-onebit-half] (custom stage, chunks=1)\n"
                << "  [--shared-onebit-base-pages] (requires shared corpus or fused; MADV_NOHUGEPAGE)\n"
                << "  [--workers 4 --batch-size 32 --rounds 4 --query-repeats 1]\n"
                << "  [--topk 10 --nprobe 32 --k-refine 1000 --k-full-bit 20 --itopk 64]\n"
                << "  [--cagra-algo auto|single-cta|multi-cta|multi-kernel]\n"
                << "  [--cagra-team-size 0|4|8|16|32 --cagra-thread-block-size 0|64|128|256|512|1024 --cagra-search-width N]\n"
                << "  [--verify-full-bit --limit N --batch-only --profile-range]\n"
                << "Native B1 and batch are both measured unless --batch-only is selected.\n"
                << "Named stages exclude archived GPU-kernel and UMA-buffer experiments.\n"
                << "Legacy individual backend/experimental switches remain available with --stage custom (default).\n";
            return 0;
        }
        if(++i>=argc)throw std::invalid_argument("Missing value for "+arg);
        std::string value=argv[i];
        if(arg=="--index")index_path=value;
        else if(arg=="--queries")query_path=value;
        else if(arg=="--output")output_path=value;
        else if(arg=="--workers") { worker_list=value;explicit_workers=true; }
        else if(arg=="--stage")stage_name=value;
        else if(arg=="--cagra-algo") {
            cagra_algo=value;
            if(value=="auto") options.cagra_algo=Chimera::CagraSearchAlgorithm::Auto;
            else if(value=="single-cta") options.cagra_algo=Chimera::CagraSearchAlgorithm::SingleCta;
            else if(value=="multi-cta") options.cagra_algo=Chimera::CagraSearchAlgorithm::MultiCta;
            else if(value=="multi-kernel") options.cagra_algo=Chimera::CagraSearchAlgorithm::MultiKernel;
            else throw std::invalid_argument("Unknown CAGRA algorithm");
        }
        else if(arg=="--cagra-team-size") options.cagra_team_size=std::stoi(value);
        else if(arg=="--cagra-thread-block-size") options.cagra_thread_block_size=std::stoi(value);
        else if(arg=="--cagra-search-width") options.cagra_search_width=std::stoi(value);
        else if(arg=="--cagra-memory") {
            cagra_memory=value;
            if(value=="device") options.cagra_memory_placement=Chimera::CagraMemoryPlacement::Device;
            else if(value=="ats-dataset") options.cagra_memory_placement=Chimera::CagraMemoryPlacement::AtsDataset;
            else if(value=="ats-both") options.cagra_memory_placement=Chimera::CagraMemoryPlacement::AtsDatasetAndGraph;
            else throw std::invalid_argument("CAGRA memory must be device, ats-dataset or ats-both");
        }
        else if(arg=="--batch-size")batch=std::stoull(value);
        else if(arg=="--rounds")rounds=std::stoull(value);
        else if(arg=="--query-repeats")query_repeats=std::stoull(value);
        else if(arg=="--limit")limit=std::stoull(value);
        else if(arg=="--topk")k=std::stoull(value);
        else if(arg=="--nprobe")options.nprobe=std::stoi(value);
        else if(arg=="--k-refine")options.k_refine=std::stoi(value);
        else if(arg=="--k-full-bit")options.k_full_bit=std::stoi(value);
        else if(arg=="--itopk")options.cagra_itopk_size=std::stoi(value);
        else if(arg=="--uma-cpu-fullbit-fraction") { options.uma_cpu_fullbit_fraction=std::stod(value);custom_backend_switch=true; }
        else if(arg=="--uma-cpu-threads") { options.uma_cpu_threads=std::stoi(value);custom_backend_switch=true; }
        else if(arg=="--uma-cpu-pool-cpus") { options.uma_cpu_pool_cpus=value;custom_backend_switch=true; }
        else throw std::invalid_argument("Unknown option "+arg);
    }
    const auto stage=Chimera::parse_evolution_stage(stage_name);
    if(stage!=Chimera::EvolutionStage::Custom) {
        if(custom_backend_switch)throw std::invalid_argument("Do not mix --stage with custom CPU backend/pool switches");
        Chimera::apply_evolution_stage(stage,options);
        if(!explicit_workers)worker_list="4";
    }
    if(index_path.empty()||query_path.empty()||output_path.empty()||!batch||!rounds||!k||!query_repeats)
        throw std::invalid_argument("Require index, queries, output, positive batch/rounds/topk/query-repeats");
    std::vector<size_t> workers;
    std::istringstream input(worker_list); std::string part;
    while(std::getline(input,part,',')) {
        size_t w=std::stoull(part);
        if(!w || w>batch)throw std::invalid_argument("Workers must be between 1 and batch-size");
        workers.push_back(w);
    }
    if(workers.empty())throw std::invalid_argument("Empty workers list");
    size_t n=0,len=0,dim=0;
    auto queries=read_queries(query_path,n,len,dim);
    if(len!=Q_DOCLEN || dim!=128)throw std::invalid_argument("Expected Q32 D128 queries");
    if(limit)n=std::min(n,limit);
    if(query_repeats>std::numeric_limits<size_t>::max()/n)
        throw std::invalid_argument("Timed query count overflows size_t");
    const size_t timed_queries=n*query_repeats;
    const size_t stride=len*dim, check_n=std::min(n,size_t(32));
    Chimera::chimera_index index;
    Chimera::LoadOptions load; load.low_memory=true;
    index.load(index_path,options,load);
    std::string pool_cpu_report = options.uma_cpu_pool_cpus;
    if (pool_cpu_report.empty()) {
        constexpr int efficiency_cpus[] = {0,10,1,11,2,12,3,13,4,14};
        for (int i=0;i<options.uma_cpu_threads;++i) {
            if(i) pool_cpu_report += ',';
            pool_cpu_report += std::to_string(efficiency_cpus[i]);
        }
    }
    index.set_capture_scores(true);
    Ids reference(check_n); std::vector<std::vector<float>> ref_scores(check_n);
    for(size_t q=0;q<check_n;++q) {
        reference[q]=index.search(queries.data()+q*stride,stride,k);
        ref_scores[q]=index.last_result_scores();
    }
    require_rows(reference,check_n,k);
    // A completed output file is the run receipt; interrupted work retains .partial.
    const std::string partial_path=output_path+".partial";
    std::ofstream out(partial_path);
    if(!out)throw std::runtime_error("Cannot open output");
    out << std::setprecision(12) << "{\"stage\":\""<<stage_name<<"\",\"queries\":" << timed_queries
        <<",\"distinct_queries\":"<<n<<",\"query_repeats\":"<<query_repeats
        << ",\"query_length\":"<<len
        <<",\"dimension\":"<<dim<<",\"topk\":"<<k<<",\"batch_size\":"<<batch
        <<",\"rounds\":"<<rounds<<",\"nprobe\":"<<options.nprobe
        <<",\"k_refine\":"<<options.k_refine<<",\"k_full_bit\":"<<options.k_full_bit
        <<",\"cagra_itopk_size\":"<<options.cagra_itopk_size<<",\"chunks\":1"
        <<",\"cagra_algo\":\""<<cagra_algo<<"\""
        <<",\"cagra_team_size\":"<<options.cagra_team_size
        <<",\"cagra_thread_block_size\":"<<options.cagra_thread_block_size
        <<",\"cagra_search_width\":"<<options.cagra_search_width
        <<",\"gpu_full_bit\":"<<(options.gpu_full_bit?"true":"false")
        <<",\"compact_token_score\":"<<(options.compact_token_score?"true":"false")
        <<",\"fast_document_reduce\":"<<(options.fast_document_reduce?"true":"false")
        <<",\"cagra_memory\":\""<<cagra_memory<<"\""
        <<",\"uma_cpu_refine\":"<<(options.uma_cpu_refine?"true":"false")
        <<",\"uma_cpu_candidates\":"<<(options.uma_cpu_candidates?"true":"false")
        <<",\"uma_query_buffers\":"<<(options.uma_query_buffers?"true":"false")
        <<",\"uma_cpu_fullbit_fraction\":"<<options.uma_cpu_fullbit_fraction
        <<",\"uma_cpu_fullbit_deferred\":"<<(options.uma_cpu_fullbit_deferred?"true":"false")
        <<",\"shared_onebit_corpus\":"<<(options.shared_onebit_corpus?"true":"false")
        <<",\"shared_onebit_base_pages\":"<<(options.shared_onebit_base_pages?"true":"false")
        <<",\"uma_cpu_onebit_half\":"<<(options.uma_cpu_onebit_half?"true":"false")
        <<",\"uma_cpu_onebit_fused\":"<<(options.uma_cpu_onebit_fused?"true":"false")
        <<",\"uma_cpu_document\":"<<(options.uma_cpu_document?"true":"false")
        <<",\"uma_cpu_threads\":"<<options.uma_cpu_threads
        <<",\"uma_cpu_pool_cpus\":\""<<pool_cpu_report<<"\""
        <<",\"verify_full_bit\":"<<(options.verify_full_bit?"true":"false")
        <<",\"batch_only\":"<<(batch_only?"true":"false")
        <<",\"profile_range\":"<<(profile_range?"true":"false")
        <<",\"preflight_queries_per_point\":"<<n
        <<",\"warmup_queries_per_round\":"<<std::min(n,std::max(size_t(32),batch))
        <<",\"timing\":\"host FP32 queries to completed host IDs; QPS from whole query-loop wall time; no encoding/load/warmup/output/recall\""
        <<",\"latency_unit\":\"batch completion ms, B1 per-query ms\""
        <<",\"round_window_clock\":\"CLOCK_MONOTONIC\",\"points\":[";
    bool first_point=true;
    // workers=0 denotes the original single-query API reference.
    std::vector<size_t> modes;
    if(!batch_only)modes.push_back(0);
    modes.insert(modes.end(),workers.begin(),workers.end());
    for(size_t w:modes) {
        const auto point_start=Clock::now();
        size_t bad_ids=0,bad_scores=0;
        if(w) {
            index.prepare_batch(w);
            index.set_capture_scores(true);
            for(size_t base=0;base<check_n;base+=batch) {
                size_t count=std::min(batch,check_n-base);
                auto actual=index.search_batch(queries.data()+base*stride,count*stride,k,w);
                const auto& scores=index.last_batch_result_scores();
                require_rows(actual,count,k);
                if(scores.size()!=count)throw std::runtime_error("Missing batch scores");
                for(size_t q=0;q<count;++q) {
                    if(actual[q]!=reference[base+q])++bad_ids;
                    if(scores[q].size()!=ref_scores[base+q].size() ||
                       std::memcmp(scores[q].data(),ref_scores[base+q].data(),scores[q].size()*sizeof(float)))++bad_scores;
                }
            }
        }
        std::cout << "[CHECK] mode="<<(w?"batch":"B1")<<" workers="<<w<<" checked_queries="<<check_n
                  <<" id_mismatch_queries="<<bad_ids<<" score_mismatch_queries="<<bad_scores<<std::endl;
        if(!w) {
            index.set_capture_scores(true);
            for(size_t q=0;q<check_n;++q) {
                auto actual=index.search(queries.data()+q*stride,stride,k);
                const auto& scores=index.last_result_scores();
                if(actual!=reference[q])++bad_ids;
                if(scores.size()!=ref_scores[q].size() ||
                   std::memcmp(scores.data(),ref_scores[q].data(),scores.size()*sizeof(float)))++bad_scores;
            }
            std::cout << "[B1_REPEAT] id_mismatch_queries="<<bad_ids
                      <<" score_mismatch_queries="<<bad_scores<<std::endl;
        }
        index.set_capture_scores(false);
        // Exercise the full fixed workload once on every lane before timing.
        // This grows candidate arenas and initializes CUDA per-thread state;
        // startup allocation/synchronization must not masquerade as throughput.
        for(size_t base=0;base<n;base+=(w?batch:1)) {
            if(w) index.search_batch(queries.data()+base*stride,
                std::min(batch,n-base)*stride,k,w);
            else index.search(queries.data()+base*stride,stride,k);
        }
        Chimera::check_cuda(cudaDeviceSynchronize());
        if(!first_point)out<<','; first_point=false;
        out << "{\"mode\":\""<<(w?"batch":"B1")<<"\",\"workers\":"<<w
            <<",\"checked_queries\":"<<check_n<<",\"id_mismatch_queries\":"<<bad_ids
            <<",\"score_mismatch_queries\":"<<bad_scores<<",\"measurements\":[";
        for(size_t round=0;round<rounds;++round) {
            const size_t warm_n=std::min(n,std::max(size_t(32),batch));
            for(size_t base=0;base<warm_n;base+=(w?batch:1)) {
                size_t count=std::min(batch,warm_n-base);
                if(w)index.search_batch(queries.data()+base*stride,count*stride,k,w);
                else index.search(queries.data()+base*stride,stride,k);
            }
            Chimera::check_cuda(cudaDeviceSynchronize());
            std::vector<double> latencies;
            Ids results(timed_queries);
            if(profile_range) {
                index.set_profiling_enabled(true);
                Chimera::check_cuda(cudaProfilerStart(), "Starting timed round capture");
            }
            if(profile_range) nvtxRangePushA("ChimeraBatch/measured_round");
            const uint64_t monotonic_start_ns = monotonic_ns();
            const auto loop_start=Clock::now();
            for(size_t repeat=0;repeat<query_repeats;++repeat) {
            for(size_t base=0;base<n;base+=(w?batch:1)) {
                Chimera::ProfileRange call_range(profile_range, w ? "ChimeraBatch/batch" : "ChimeraBatch/B1");
                const auto call_start=Clock::now();
                if(w) {
                    const size_t count=std::min(batch,n-base);
                    auto rows=index.search_batch(queries.data()+base*stride,count*stride,k,w);
                    latencies.push_back(elapsed(call_start));
                    for(size_t q=0;q<count;++q)results[repeat*n+base+q]=std::move(rows[q]);
                } else {
                    results[repeat*n+base]=index.search(queries.data()+base*stride,stride,k);
                    latencies.push_back(elapsed(call_start));
                }
            }
            }
            Chimera::check_cuda(cudaDeviceSynchronize());
            const double wall_ms=elapsed(loop_start);
            const uint64_t monotonic_end_ns = monotonic_ns();
            if(profile_range) {
                nvtxRangePop();
                Chimera::check_cuda(cudaProfilerStop(), "Stopping timed round capture");
                index.set_profiling_enabled(false);
            }
            require_rows(results,timed_queries,k);
            const double search_ms=std::accumulate(latencies.begin(),latencies.end(),0.0);
            std::sort(latencies.begin(),latencies.end());
            const double p99=latencies[std::min(latencies.size()-1,size_t(std::ceil(latencies.size()*0.99)-1))];
            if(round)out<<',';
            out << "{\"round\":"<<round+1<<",\"completed_queries\":"<<timed_queries
                <<",\"monotonic_start_ns\":"<<monotonic_start_ns
                <<",\"monotonic_end_ns\":"<<monotonic_end_ns<<",\"loop_wall_ms\":"<<wall_ms
                <<",\"search_ms\":"<<search_ms<<",\"qps\":"<<timed_queries*1000.0/wall_ms
                <<",\"batch_latency_mean_ms\":"<<search_ms/latencies.size()<<",\"batch_latency_p99_ms\":"<<p99
                <<",\"result_ids\":";
            write_ids(out,results); out << '}'; out.flush();
            std::cout << "[RESULT] mode="<<(w?"batch":"B1")<<" workers="<<w<<" round="<<round+1
                      <<" completed="<<timed_queries<<" qps="<<timed_queries*1000.0/wall_ms<<" batch_mean_ms="<<search_ms/latencies.size()
                      <<" batch_p99_ms="<<p99<<std::endl;
        }
        out << "],\"point_wall_ms\":"<<elapsed(point_start)<<'}'; out.flush();
    }
    out << "],\"completed\":true}\n";
    out.close();
    if(!out)throw std::runtime_error("Failed writing output");
    if(std::rename(partial_path.c_str(),output_path.c_str()))throw std::runtime_error("Cannot finalize output");
    return 0;
} catch(const std::exception& e) {
    std::cerr<<"[FAILED] "<<e.what()<<std::endl; return 1;
}
