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
  --output batch.json --workers 1,2,4,8 --batch-size 32 --rounds 3 \
  --nprobe 32 --k-refine 1000 --k-full-bit 20 --itopk 64 --topk 10
```

QPS uses the wall time for all host queries through completed host IDs,
including query rotation/upload and batch submission overhead. Batch latency
is measured until the whole batch completes; it is not a single-query latency.
The output is renamed from `.partial` only after all points finish. Allocation
failures or GPU errors invalidate the index; reload it before retrying.
