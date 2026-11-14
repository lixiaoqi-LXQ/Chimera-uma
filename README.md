# Chimera+

Chimera+ replaces only the CPU full-bit document scorer in the accepted ARM Chimera baseline with a CUDA scorer. Routing, candidate generation/refinement, 1bit scoring, chunk selection and the final CPU heap are unchanged. The existing index is loaded without conversion or rebuilding.

The current GPU scorer supports the experiment's 4bit, D128, 32-query-token representation. It preserves the CPU nibble layout, FP32 FMA/addition tree, correction factors, per-query-token maxima and score ties. One CUDA block handles each selected document. On GB10, ATS lets the GPU read the existing full-code/factor arrays through host page tables; no second corpus allocation or FP32 expansion is made. Other supported CUDA mappings use registered host storage.

- `--full-bit-device gpu` is the Chimera+ default.
- `--full-bit-device cpu` invokes the unchanged CPU scoring body for comparison.
- `--verify-full-bit` additionally replays exactly the same selected candidates on CPU and requires bitwise equal FP32 scores. This diagnostic must be disabled for performance measurements.

`BASELINE.json` identifies the imported experimental baseline. Git branch `baseline` preserves it; `Chimera+` contains this change. The GPU scoring change is independent of the optional multi-stream batch API described below.

Build outside the source tree with CUDA architecture 121, the existing cuVS/RMM toolchain and `CHIMERA_BUILD_PYTHON=OFF`. Targets: `gpu_search`, `fullbit_gpu_qualification`. The qualification compares actual CPU SIMD scoring with GPU output on nibble boundaries, reordered/repeated IDs, document lengths 1–300, negative scores, zero/tied scores and empty candidate batches.

Heavy builds and all GPU validation/performance measurements use an exclusive nonblocking `~/device.lock`, held until all associated work completes. Build directories, logs, test specifications and outputs live in `/home/lxq/multi-vector-research/ai-workspace/`; they are not source artifacts.

The four frozen LoTTE configurations (CPU20, chunks1) remain:

| Returned k | Recall target | nprobe | refine | full-bit | CAGRA itopk |
|---:|---:|---:|---:|---:|---:|
| 10 | .90 | 32 | 1000 | 20 | 64 |
| 10 | .95 | 128 | 1000 | 50 | 160 |
| 100 | .90 | 64 | 3000 | 300 | 96 |
| 100 | .95 | 256 | 3000 | 300 | 288 |

The single experiment report remains `/home/lxq/multi-vector-research/reports/lotte-r10-brief.md`.

## Multi-stream batch search

`prepare_batch(workers)` allocates reusable private query workspaces. Then
`search_batch(queries, query_values, k, workers)` accepts contiguous FP32
`[B, 32, D]` input and synchronously returns one ordered ID list per query.
`query_values` is the total float count, not the query count. `workers` is 1–20;
up to that many queries execute concurrently. Tail batches and empty batches
are supported. Enable `set_capture_scores(true)` to inspect
`last_batch_result_scores()` outside throughput timing.

```cpp
index.prepare_batch(4);                    // outside timing
// queries contains B * 32 * 128 floats
const auto ids = index.search_batch(queries.data(), queries.size(), 10, 4);
```

Each lane uses independent nonblocking compute/upload/download streams,
candidate bitmap/arena, RAFT resources and CPU scratch. A host OpenMP team
submits the existing single-query kernels concurrently; the corpus, CAGRA graph,
full-bit codes and factors are shared. No new index or quantization is needed.
The index object is synchronous and must have only one caller at a time;
internal workers are destroyed before the shared corpus during load/destruction.
The initial version supports the full algorithm (no algorithm ablations).

Build target `batch_search_benchmark` compares the original `search()` B1 API
with batch sizes and stream counts, checks initial IDs/FP32 scores, and writes
all per-round IDs, QPS and batch completion latency to JSON. It performs one
full untimed preflight per point to initialize worker state and grow candidate
buffers, then 32 or `batch_size` warmup queries before each timed round.

```sh
# Hold an exclusive ~/device.lock for the complete load/test/process lifetime.
OMP_NUM_THREADS=20 OMP_MAX_ACTIVE_LEVELS=1 OMP_DYNAMIC=FALSE \
OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
  ./batch_search_benchmark --index INDEX --queries queries.bin \
  --output batch.json --workers 1,2,4,8,10 --batch-size 32 --rounds 3 \
  --query-repeats 5 \
  --nprobe 32 --k-refine 1000 --k-full-bit 20 --itopk 64 --topk 10
```

QPS uses the wall time for all host queries through completed host IDs,
including query rotation/upload and batch submission overhead. Batch latency
is measured until the whole batch completes; it is not a single-query latency.
The output is renamed from `.partial` only after all points finish. Allocation
failures or GPU errors invalidate the index; reload it before retrying.

For diagnostics, `--batch-only` skips the timed original B1 control while
retaining initial result checks. `--profile-range` enables NVTX and brackets
each measured round with CUDA profiler start/stop; use one round when capturing
with `nsys --capture-range=cudaProfilerApi --capture-range-end=stop`.
Profiler measurements are diagnostic; throughput results use ordinary runs.
The independent batch report at `reports/chimera-batch-search.md` includes
long-duration lane sweeps at R10 .90 / R100 .95 and the plateau analysis.

### Reusable chunk events

Each workspace now owns its chunk compute/D2H completion events. They are
created during workspace initialization and destroyed during cleanup. Search
records and waits on these events as before, without creating/destroying CUDA
event objects per query. Each batch lane has its own event set. This also
applies to the original CPU full-bit backend and preserves chunk selection,
scoring, and completion dependencies.

The fixed-B32 full-query comparison and bottleneck evidence are in
`reports/chimera-batch-search.md`. Auxiliary C2C calibration and CPU memory
pressure programs are maintained in `systems/chimera/bottleneck/` in the
research repository.

### Optional UMA paths and independent GPU kernels

The default path uses the original GPU routing/candidate/refinement/1bit kernels and the Chimera+ fullbit scorer. All switches below are opt-in and fixed when the index is loaded. Multi-stream workers inherit the owner's configuration and share its read-only corpus.

| SearchOptions field | Batch CLI | Default | Work moved or storage changed |
|---|---|---|---|
| `uma_query_buffers` | `--uma-query-buffers` | `false` | ATS query/metadata buffers; CPU integer candidate-count and refined-length prefixes |
| `uma_cpu_candidates` | `--uma-cpu-candidates` | `false` | CPU bitmap mark/count/prefix/materialization after CAGRA publishes labels into ATS |
| `uma_cpu_refine` | `--uma-cpu-refine` | `false` | CPU exact stable-radix-equivalent Top-k after the unchanged GPU partial-score sum |
| `cagra_memory_placement` | `--cagra-memory device\|ats-dataset\|ats-both` | `Device` / `device` | CAGRA dataset, or dataset and graph, in CPU-owned ATS storage |
| `fast_document_reduce` | `--fast-document-reduce` | `false` | Four-warp GPU document reduction |
| `compact_token_score` | `--compact-token-score` | `false` | GPU token scoring with bounded live accumulators |

The two GPU kernel switches are independent of UMA. Attribute UMA gains using the same GPU kernels, batch size, worker count and search parameters in both arms. Report GPU kernel gains separately.

#### CPU collaboration through ATS

`uma_query_buffers` requires CUDA pageable-memory access using host page tables. Every lane owns its query, offsets and result metadata once; device pointers alias the same CPU allocations. GPU token-score storage remains device-allocated. CPU reads follow the corresponding stream/event completion; CPU writes precede dependent GPU launches.

`uma_cpu_candidates` requires `uma_query_buffers`. It reads completed route labels and a shared corpus document-ID array, builds the candidate bitmap and emits IDs in ascending row order. Initialization replaces the corpus document-ID allocation with one CPU-owned ATS allocation; workers borrow that owner. The CLI enables its buffer prerequisite automatically.

`uma_cpu_refine` requires both previous switches and candidate refinement. The unchanged GPU partial-score sum writes ATS scores. CPU precomputes packed64 radix keys, selects Top-k and sorts the selected prefix. The key includes the original row tie-break, preserving stable radix ordering, equivalent signed zeros and NaN bit ordering. Selected IDs already reside on CPU, so CPU also derives document offsets directly; token-position gathering remains on GPU. The CLI enables both prerequisites automatically. Library callers set them explicitly:

```cpp
SearchOptions options;
options.uma_query_buffers = true;
options.uma_cpu_candidates = true;
options.uma_cpu_refine = true;
index.load(index_path, options, load_options);
```

`--verify-full-bit` additionally compares CPU candidate arrays and CPU Top-k IDs/score bits against their original GPU counterparts when those paths are enabled. It also retains the fullbit CPU/GPU score check and the fast-document-reduction check. Run it for qualification, then disable it for throughput.

#### CAGRA memory placement

`--cagra-memory ats-dataset` keeps the graph on device and exposes the unchanged FP32 centroid dataset through ATS. `ats-both` exposes both arrays through ATS using a non-owning CAGRA view. Initialization copies the original bytes once and releases the replaced device owners; query-time corpus copies are zero. Search kernels, metric and routing parameters stay the same.

The backing allocations are 2MiB-aligned and request `MADV_HUGEPAGE`. Actual huge-page coverage is recorded separately from `/proc/PID/smaps`. The owning `PG_CAGRA` outlives every worker; its view is destroyed before the ATS backing allocations. This placement can be tested independently of the CPU-collaboration switches.

#### Interpretation of current results

The completed same-kernel W8 R100 comparison measured about **1.5%** UMA gain. Larger gains from compact token scoring and fast document reduction are GPU kernel results and do not meet the UMA-only ≥10% objective. CPU candidate/refinement and CAGRA ATS paths remain measured alternatives, with the per-configuration evidence maintained in `reports/chimera-batch-search.md`.

The current prerequisite study measures CPU scheduling availability and sustained read-pressure tolerance with all experimental switches disabled. It uses complete query sets, discards the first conditioning round and scores three rounds of at least six seconds. Full recall verification runs after formal pressure measurements finish.

`--query-repeats N` runs N complete passes continuously per round, including
each pass's tail batch. Output records distinct/total query counts and preserves
pass/query order. Increase N until every scored round exceeds six seconds.
`token_score_qualification` compares original/compact GPU scores bitwise on
192 generated cases and reports theoretical residency limits.
