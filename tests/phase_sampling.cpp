#include "chimera/chimera_index.cuh"
#include "chimera/evolution_stage.hpp"
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <ctime>
#include <exception>
#include <iterator>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace {
constexpr size_t sample_count = 128, batch_size = 32, workers = 4, stride = 32 * 128;
volatile sig_atomic_t interrupted = 0;
void signal_handler(int) { interrupted = 1; }
void require(bool value, const std::string& text) { if (!value) throw std::runtime_error(text); }
uint64_t mono_ns() {
    timespec t{}; require(!clock_gettime(CLOCK_MONOTONIC, &t), "CLOCK_MONOTONIC failed");
    return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}
std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
void save(const fs::path& path, const std::string& value) {
    auto tmp = path.string() + ".partial";
    { std::ofstream out(tmp); out << value << '\n'; require(bool(out), "Cannot write " + tmp); }
    fs::rename(tmp, path);
}
void u32(std::ostream& out, uint32_t x) { for (int i = 0; i < 4; ++i) out.put(char(x >> (8 * i))); }
void u64(std::ostream& out, uint64_t x) { for (int i = 0; i < 8; ++i) out.put(char(x >> (8 * i))); }
void lease(int fd) {
    struct stat a{}, b{};
    require(fd >= 0 && !fstat(fd, &a) && !stat("/home/lxq/device.lock", &b) &&
            a.st_dev == b.st_dev && a.st_ino == b.st_ino, "Inherited device.lock inode mismatch");
    std::ifstream info("/proc/self/fdinfo/" + std::to_string(fd));
    std::string line; bool held = false;
    while (std::getline(info, line)) if (line.rfind("lock:", 0) == 0 &&
            line.find("FLOCK") != line.npos && line.find("WRITE") != line.npos) held = true;
    require(held, "Inherited EX flock missing");
}
void affinity() {
    cpu_set_t requested, actual; CPU_ZERO(&requested); CPU_ZERO(&actual);
    for (int c : {8, 9, 18, 19}) CPU_SET(c, &requested);
    require(!sched_setaffinity(0, sizeof(requested), &requested), "P4 affinity failed");
    require(!sched_getaffinity(0, sizeof(actual), &actual) && CPU_EQUAL(&requested, &actual), "P4 readback failed");
}
struct Faults {
    long minor = 0, major = 0;
    static Faults read() { rusage u{}; require(!getrusage(RUSAGE_SELF, &u), "getrusage failed"); return {u.ru_minflt, u.ru_majflt}; }
    std::string json() const { return "{\"ru_minflt\":" + std::to_string(minor) + ",\"ru_majflt\":" + std::to_string(major) + "}"; }
};

// Strictly parse the tiny wrapper-generated map, retaining the external ID's
// JSON scalar representation. No query payload or arbitrary configuration parser.
struct QueryMap {
    std::string text; size_t pos = 0;
    void ws() { while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\r' || text[pos] == '\t')) ++pos; }
    void take(char c) { ws(); require(pos < text.size() && text[pos++] == c, "Malformed query map"); }
    std::string string_token() {
        ws(); const size_t begin = pos; take('"');
        while (pos < text.size()) {
            const unsigned char c = text[pos++];
            if (c == '"') return text.substr(begin, pos - begin);
            require(c >= 32, "Control character in query map");
            if (c == '\\') {
                require(pos < text.size(), "Truncated query map escape");
                const char escape = text[pos++];
                if (escape == 'u') for (int i = 0; i < 4; ++i) {
                    require(pos < text.size() && std::isxdigit(static_cast<unsigned char>(text[pos])), "Invalid Unicode escape"); ++pos;
                } else require(std::string("\"\\/bfnrt").find(escape) != std::string::npos, "Invalid JSON escape");
            }
        }
        throw std::runtime_error("Unterminated query map string");
    }
    std::string integer_token() {
        ws(); const size_t begin = pos;
        if (pos < text.size() && text[pos] == '-') ++pos;
        require(pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])), "Expected query map integer");
        if (text[pos] == '0') ++pos;
        else while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) ++pos;
        return text.substr(begin, pos - begin);
    }
    std::array<std::string, sample_count> read(const std::string& path) {
        std::ifstream input(path); require(bool(input), "Cannot open query map");
        input.seekg(0, std::ios::end); const auto size = input.tellg();
        require(size >= 0 && size <= 1024 * 1024, "Query map too large"); input.seekg(0);
        text.assign(std::istreambuf_iterator<char>(input), {});
        std::array<std::string, sample_count> ids; take('[');
        for (size_t row = 0; row < sample_count; ++row) {
            if (row) take(','); take('{');
            require(string_token() == "\"input_row\"", "query map needs input_row first"); take(':');
            require(integer_token() == std::to_string(row), "Query map must map prefix rows0..127"); take(',');
            require(string_token() == "\"input_id\"", "query map needs input_id second"); take(':'); ws();
            ids[row] = pos < text.size() && text[pos] == '"' ? string_token() : integer_token(); take('}');
        }
        take(']'); ws(); require(pos == text.size(), "Trailing query map content"); return ids;
    }
};
std::vector<float> prefix(const std::string& path, const std::string& dataset) {
    std::ifstream in(path, std::ios::binary); require(bool(in), "Cannot open querybin");
    int32_t shape[3]{}; in.read(reinterpret_cast<char*>(shape), sizeof(shape));
    require(bool(in) && shape[0] == (dataset == "lotte" ? 2931 : 6980) && shape[1] == 32 && shape[2] == 128,
            "Expected full dataset querybin header Q32D128");
    require(fs::file_size(path) == sizeof(shape) + uint64_t(shape[0]) * stride * sizeof(float), "Querybin size/header mismatch");
    std::vector<float> out(sample_count * stride);
    in.read(reinterpret_cast<char*>(out.data()), out.size() * sizeof(float)); require(bool(in), "Truncated query prefix");
    for (float x : out) require(std::isfinite(x), "Nonfinite query value");
    return out;
}
struct Result { uint64_t id; uint32_t bits; };
std::vector<Result> collect(Chimera::chimera_index& index, const float* query, size_t count, size_t k) {
    std::vector<Result> out(count * k);
    for (size_t base = 0; base < count; base += batch_size) {
        auto ids = index.search_batch(query + base * stride, batch_size * stride, k, workers);
        const auto& scores = index.last_batch_result_scores();
        require(ids.size() == batch_size && scores.size() == batch_size, "Missing query rows");
        for (size_t q = 0; q < batch_size; ++q) {
            require(ids[q].size() == k && scores[q].size() == k, "Incomplete TopK");
            for (size_t rank = 0; rank < k; ++rank) {
                require(std::isfinite(scores[q][rank]), "Nonfinite score"); uint32_t bits;
                std::memcpy(&bits, &scores[q][rank], sizeof(bits)); out[(base + q) * k + rank] = {ids[q][rank], bits};
            }
        }
    }
    return out;
}
void result_file(const fs::path& path, const std::vector<Result>& rows, size_t count, size_t k) {
    std::ofstream out(path, std::ios::binary); out.write("CHMRRD01", 8); u32(out, 1); u32(out, k); u64(out, count); u64(out, 1);
    for (const auto& row : rows) { u64(out, row.id); u32(out, row.bits); }
    require(bool(out), "Result write failed");
}
struct Args { bool run = false, profile = false; int fd = -1; size_t warm_passes = 1; std::string dataset, quality, stage, index, queries, map, output; };
Chimera::SearchOptions options(const Args& a, size_t& k) {
    Chimera::SearchOptions out; out.num_chunks = 1;
    if (a.quality == "r10p90") { k = 10; out.nprobe = 32; out.k_refine = 1000; out.k_full_bit = 20; out.cagra_itopk_size = 64; }
    else if (a.quality == "r10p95") { k = 10; out.nprobe = 128; out.k_refine = 1000; out.k_full_bit = 50; out.cagra_itopk_size = 160; }
    else if (a.quality == "r100p90") { k = 100; out.nprobe = 64; out.k_refine = 3000; out.k_full_bit = 300; out.cagra_itopk_size = 96; }
    else if (a.quality == "r100p95") { k = 100; out.nprobe = 256; out.k_refine = 3000; out.k_full_bit = 300; out.cagra_itopk_size = 288; }
    else throw std::runtime_error("Unknown quality");
    Chimera::apply_evolution_stage(a.stage == "B" ? Chimera::EvolutionStage::GpuRerank : Chimera::EvolutionStage::UmaCollaboration, out);
    out.verify_full_bit = true; return out;
}
std::string flags(const Chimera::SearchOptions& o) {
    std::ostringstream s; s << std::boolalpha << "{\"gpu_full_bit\":" << o.gpu_full_bit
        << ",\"compact_token_score\":false,\"fast_document_reduce\":false,\"uma_query_buffers\":false,\"uma_cpu_candidates\":false,\"uma_cpu_refine\":false,\"cagra_memory\":\"device\",\"verify_full_bit\":false,\"uma_cpu_document\":" << o.uma_cpu_document
        << ",\"uma_cpu_fullbit_deferred\":" << o.uma_cpu_fullbit_deferred << ",\"uma_cpu_fullbit_fraction\":" << o.uma_cpu_fullbit_fraction
        << ",\"uma_cpu_threads\":" << o.uma_cpu_threads << ",\"uma_cpu_pool_cpus\":" << quote(o.uma_cpu_pool_cpus) << "}"; return s.str();
}
}  // namespace

int main(int argc, char** argv) try {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--run") a.run = true;
        else if (key == "--profile-range") a.profile = true;
        else if (key == "--plan") {}
        else if (key == "--help") { std::cout << "--run --lock-fd N --dataset lotte|marco --quality r10p90|r10p95|r100p90|r100p95 --stage B|C --index PATH --queries QUERYBIN --query-map JSON128 --output-dir FRESH [--profile-range] [--warm-passes 1|2]\nDefault plan; no CUDA initialization. Fixed128/B32/W4/P4; no fullcode prefault. Mandatory same-candidate verify32 outside capture.\n"; return 0; }
        else {
            require(i + 1 < argc, "Missing option value"); std::string value = argv[++i];
            if (key == "--lock-fd") a.fd = std::stoi(value); else if (key == "--warm-passes") { require(value == "1" || value == "2", "Warm passes must be1 or2"); a.warm_passes = size_t(value[0] - '0'); } else if (key == "--dataset") a.dataset = value;
            else if (key == "--quality") a.quality = value; else if (key == "--stage") a.stage = value;
            else if (key == "--index") a.index = value; else if (key == "--queries") a.queries = value;
            else if (key == "--query-map") a.map = value; else if (key == "--output-dir") a.output = value;
            else throw std::runtime_error("Unknown option " + key);
        }
    }
    require(a.dataset == "lotte" || a.dataset == "marco", "Expected dataset lotte|marco");
    require(a.warm_passes == 1 || a.warm_passes == 2, "Warm passes must be1 or2");
    require(a.stage == "B" || a.stage == "C", "Expected stage B|C"); size_t k = 0; auto opts = options(a, k);
    if (!a.run) { std::cout << "{\"plan\":true,\"queries\":128,\"batch_size\":32,\"workers\":4,\"stage\":" << quote(a.stage) << ",\"quality\":" << quote(a.quality) << ",\"warm_passes\":" << a.warm_passes << ",\"flags\":" << flags(opts) << "}\n"; return 0; }
    lease(a.fd); affinity(); std::signal(SIGTERM, signal_handler); std::signal(SIGINT, signal_handler);
    require(!a.output.empty() && !fs::exists(a.output), "Fresh output required");
    auto input_ids = QueryMap{}.read(a.map); auto query = prefix(a.queries, a.dataset); fs::create_directories(a.output);
    const Faults before_load = Faults::read();
    Chimera::chimera_index index; Chimera::LoadOptions load; load.low_memory = true; load.prefault_full_codes = false;
    index.load(a.index, opts, load); index.prepare_batch(workers); index.set_capture_scores(true);
    const Faults after_load = Faults::read();
    const auto view = index.borrow_fullbit_codes();
    require(view.document_count == (a.dataset == "lotte" ? size_t(2428854) : size_t(8841823)),
            "Loaded index document count does not match dataset");
    // These first32 calls verify identical candidates on original CPU/GPU,
    // including C's CPU document leaf. All verification completes before warmup.
    auto qa = collect(index, query.data(), 32, k); Chimera::check_cuda(cudaDeviceSynchronize());
    index.disable_diagnostic_candidate_verification(); opts.verify_full_bit = false;
    auto repeated = collect(index, query.data(), 32, k); Chimera::check_cuda(cudaDeviceSynchronize());
    size_t id_mismatch = 0, score_mismatch = 0, common_mismatch = 0, common_items = 0;
    for (size_t row = 0; row < 32; ++row) {
        std::map<uint64_t, uint32_t> previous;
        for (size_t rank = 0; rank < k; ++rank) previous.emplace(qa[row*k+rank].id, qa[row*k+rank].bits);
        for (size_t rank = 0; rank < k; ++rank) {
            const auto& old = qa[row*k+rank]; const auto& now = repeated[row*k+rank];
            id_mismatch += old.id != now.id; score_mismatch += old.bits != now.bits;
            auto found = previous.find(now.id); if (found != previous.end()) { ++common_items; common_mismatch += found->second != now.bits; }
        }
    }
    result_file(fs::path(a.output)/"qualification-verify32.bin", qa, 32, k);
    result_file(fs::path(a.output)/"qualification-repeat32.bin", repeated, 32, k);
    require(common_mismatch == 0, "Repeated32 common-document scorebits differ");
    // Touch only128 sampled query paths, then drain every deferred job/GPU user.
    for (size_t pass = 1; pass <= a.warm_passes; ++pass) {
        require(!interrupted, "Interrupted sample warmup");
        auto warm = collect(index, query.data(), sample_count, k); Chimera::check_cuda(cudaDeviceSynchronize());
        result_file(fs::path(a.output)/("warm128-pass-" + std::to_string(pass) + ".bin"), warm, sample_count, k);
        if (pass == a.warm_passes) result_file(fs::path(a.output)/"warm128.bin", warm, sample_count, k);
    }
    index.prepare_profile_query_stats(sample_count);
    index.set_profiling_enabled(true);
    std::array<std::vector<std::vector<size_t>>, 4> ids;
    std::array<std::vector<std::vector<float>>, 4> scores;
    bool profiler_started = false, round_open = false;
    if (a.profile) { Chimera::check_cuda(cudaProfilerStart()); profiler_started = true; }
    const Faults before_capture = Faults::read();
    nvtxRangePushA("ChimeraBatch/measured_round"); round_open = true; const uint64_t begin = mono_ns();
    try {
        for (size_t batch = 0; batch < 4; ++batch) {
            require(!interrupted, "Interrupted phase capture"); index.set_profile_query_base(int64_t(batch * batch_size));
            ids[batch] = index.search_batch(query.data() + batch * batch_size * stride, batch_size * stride, k, workers);
            scores[batch] = index.last_batch_result_scores();
        }
        Chimera::check_cuda(cudaDeviceSynchronize());
    } catch (...) {
        (void)cudaDeviceSynchronize(); if (round_open) nvtxRangePop(); if (profiler_started) (void)cudaProfilerStop();
        index.set_profiling_enabled(false); index.set_profile_query_base(-1); throw;
    }
    const uint64_t end = mono_ns();
    const Faults after_capture = Faults::read();
    nvtxRangePop(); round_open = false;
    if (profiler_started) { Chimera::check_cuda(cudaProfilerStop()); profiler_started = false; }
    index.set_profiling_enabled(false); index.set_profile_query_base(-1);
    const auto profile_stats = index.last_batch_profile_stats();
    require(profile_stats.size() == sample_count, "Missing profile counters");
    std::vector<Result> output(sample_count * k);
    for (size_t batch = 0; batch < 4; ++batch) {
        require(ids[batch].size() == batch_size && scores[batch].size() == batch_size, "Incomplete measured batch");
        for (size_t q = 0; q < batch_size; ++q) {
            require(ids[batch][q].size() == k && scores[batch][q].size() == k, "Incomplete measured TopK");
            for (size_t rank = 0; rank < k; ++rank) {
                const auto value = scores[batch][q][rank];
                require(std::isfinite(value) && ids[batch][q][rank] < view.document_count, "Invalid measured ID/score"); uint32_t bits;
                std::memcpy(&bits, &value, sizeof(bits)); output[((batch*batch_size+q)*k)+rank] = {ids[batch][q][rank], bits};
            }
        }
    }
    result_file(fs::path(a.output)/"results.bin", output, sample_count, k);
    std::ostringstream sidecar; sidecar << "{\"schema\":\"chimera-phase-sample-v1\",\"dataset\":" << quote(a.dataset)
        << ",\"quality\":" << quote(a.quality) << ",\"stage\":" << quote(a.stage)
        << ",\"expected_queries\":128,\"batch_size\":32,\"workers\":4,\"completed\":true,\"queries\":[";
    for (size_t q = 0; q < sample_count; ++q) {
        const auto& s = profile_stats[q]; require(s.qid == int64_t(q) && s.lane == q%4 && s.lane_ordinal == (q%32)/4, "Query profile assignment mismatch");
        require(a.stage == "C" ? s.expected_document_tiles > 0 && s.expected_fullbit_tasks > 0 :
                s.expected_document_tiles == 0 && s.expected_fullbit_tasks == 0, "Wrong CPU task counts");
        if (q) sidecar << ','; sidecar << "{\"qid\":" << q << ",\"input_row\":" << q << ",\"input_id\":" << input_ids[q]
            << ",\"batch\":" << q/32 << ",\"batch_position\":" << q%32 << ",\"lane\":" << s.lane
            << ",\"lane_ordinal\":" << s.lane_ordinal << ",\"expected_document_tiles\":" << s.expected_document_tiles
            << ",\"expected_fullbit_tasks\":" << s.expected_fullbit_tasks
            << ",\"refined_docs\":" << s.refined_docs << ",\"refined_tokens\":" << s.refined_tokens
            << ",\"fullbit_candidates\":" << s.fullbit_candidates << ",\"fullbit_tokens\":" << s.fullbit_tokens << '}';
    }
    sidecar << "]}"; save(fs::path(a.output)/"sidecar.json", sidecar.str());
    std::ostringstream summary; summary << std::setprecision(15) << "{\"completed\":true,\"diagnostic_only\":true,\"formal_QPS_eligible\":false,\"completed_queries\":128,\"query_repeats\":1,\"rounds\":1,\"dataset\":" << quote(a.dataset)
        << ",\"quality\":" << quote(a.quality) << ",\"stage\":" << quote(a.stage) << ",\"topk\":" << k
        << ",\"batch_size\":32,\"workers\":4,\"query_length\":32,\"dimension\":128,\"search_cpus\":[8,9,18,19],\"pool_cpus\":"
        << (a.stage == "C" ? "[0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17]" : "[]")
        << ",\"low_memory\":true,\"prefault_full_codes\":false,\"warm_passes\":" << a.warm_passes
        << ",\"distinct_warmup_queries\":128,\"warmup_queries\":" << sample_count * a.warm_passes
        << ",\"actual_warm_query_calls\":" << sample_count * a.warm_passes << ",\"profile_range\":" << (a.profile ? "true" : "false")
        << ",\"nprobe\":" << opts.nprobe << ",\"k_refine\":" << opts.k_refine << ",\"k_full_bit\":" << opts.k_full_bit
        << ",\"cagra_itopk_size\":" << opts.cagra_itopk_size << ",\"chunks\":1,\"flags\":" << flags(opts)
        << ",\"round_window_clock\":\"CLOCK_MONOTONIC\",\"monotonic_start_ns\":" << begin << ",\"monotonic_end_ns\":" << end
        << ",\"diagnostic_qps\":" << sample_count*1e9/double(end-begin)
        << ",\"qualification\":{\"queries\":32,\"same_candidate_verification_passed\":true,\"cpu_document_verified\":" << (a.stage == "C" ? "true" : "false")
        << ",\"id_mismatch_items\":" << id_mismatch << ",\"score_mismatch_items\":" << score_mismatch << ",\"common_document_score_items\":" << common_items
        << ",\"common_document_score_mismatches\":" << common_mismatch << "},\"faults\":{\"before_load\":" << before_load.json()
        << ",\"after_load\":" << after_load.json() << ",\"before_capture\":" << before_capture.json() << ",\"after_capture\":" << after_capture.json() << "}}";
    save(fs::path(a.output)/"summary.json", summary.str()); return 0;
} catch (const std::exception& error) { std::cerr << "[FAILED] " << error.what() << '\n'; return 1; }
