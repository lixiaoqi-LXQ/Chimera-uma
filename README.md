# Chimera: batch → GPU rerank → UMA collaboration

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

The main experiment report is `/home/lxq/multi-vector-research/reports/performance.md`.

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

### Three selectable evolution stages

Use one stage preset in `batch_search_benchmark`:

| Stage | Document reduction | Fullbit scoring | CPU organization |
|---|---|---|---|
| `--stage chimera-batch` | Original GPU kernel | Original CPU implementation | Native B1 OpenMP20; batch outer4 with inner1 or inner5 |
| `--stage gpu-rerank` | Original GPU kernel | GPU | Four batch submitters, CPU20 allowed |
| `--stage uma-collab` | Shared CPU pool | Deferred CPU pool jobs | Four P-core submitters + remaining16-core pool |

Named stages reject compact-token / fast-document GPU kernels, UMA small
query/candidate/refinement buffers and CAGRA ATS placement. They also reject
individual CPU backend/pool switches. Quality parameters remain caller-controlled.
Use the same four quality configurations above to compare these stages.
The main report records native B1 call latency and B32/W4 throughput separately.

The original CPU baseline compares serial inner scoring with four teams of
five threads. The latter uses `OMP_NUM_THREADS=20,5`,
`OMP_MAX_ACTIVE_LEVELS=2` and `OMP_THREAD_LIMIT=20`; native B1 still uses20
threads, and batch has at most20 participating OpenMP threads. LoTTE R10 is
faster with inner1, while R100 is faster with inner5. The performance driver
selects the faster measured batch policy and retains B1 from that same case.

```sh
# After acquiring the global device lock; hold it through process teardown.
taskset -c 8,9,18,19 ./batch_search_benchmark \
  --stage uma-collab --index INDEX --queries queries.bin --output uma.json \
  --workers 4 --batch-size 32 --rounds 4 --query-repeats 2 \
  --nprobe 256 --k-refine 3000 --k-full-bit 300 --itopk 288 --topk 100
```

`apply_evolution_stage` in `cpp/include/chimera/evolution_stage.hpp` also
provides these backend presets to C++ callers. Set the submitter affinity and
OpenMP policy outside the library. The UMA preset assigns the persistent pool
to `0..7,10..17`; submitters use `8,9,18,19`.

### CPU collaboration

One persistent task pool is shared by all query lanes. The original GPU token
scorer writes a CPU-owned, 2MiB-aligned ATS score matrix. After its completion
event, CPU tasks reduce documents using the original FP32 addition tree.
CPU fullbit jobs read the existing 4bit corpus and factors directly, own their
query/candidate/result storage and overlap with later queries on the GPU.
Every job joins before batch return, including failure cleanup. Native B1 keeps
synchronous completion. There is no deliberate memory-read throttle.

`--verify-full-bit` qualifies CPU document scores and fullbit scores bitwise
against original CPU/GPU scorers; the original CPU stage is checked against
the GPU oracle as well. Disable verification during performance measurement.

Prior R100 .95 confirmation measured 424.9→479.7 QPS (+12.9%), holding the
original GPU kernels and submitter cores fixed. Six scored rounds were
477.79–480.77 QPS with actual recall about .9529. R10 .90 was approximately
flat; the main report measures every requested quality point.

### Archived side experiments

The `archive/side-optimizations-20261115` branch retains the complete earlier
GPU-kernel and UMA-buffer exploration with its documentation. Legacy switches
remain opt-in under `--stage custom` for compatibility; all are disabled in
the three named stages. Their gains are kept outside the UMA collaboration
comparison. The research repository tracks the snapshot in
`configs/performance-stages.json`.

### Optional CAGRA search parameters

`batch_search_benchmark` exposes the installed cuVS search implementations with
`--cagra-algo auto|single-cta|multi-cta|multi-kernel`, `--cagra-team-size`,
`--cagra-thread-block-size`, and `--cagra-search-width`. Their defaults are
`auto`, `0`, `0`, and `1`, matching the original route. These settings apply
equally to the three named stages through `SearchOptions`; the original
Chimera scoring kernels remain unchanged. Routing parameter gains are reported
separately from CPU cooperation, with recall checked on the complete query set.


`tensor-padded-max4` is an opt-in native C++ custom-stage candidate. It shares the
TensorPadded FP16-input/FP32-accum and half-scalar QA contracts, with four lanes per
query for max reduction. See [source/validation scope](tensor-padded-max4.md).

`tensor-padded-hoist` caches the eight A fragments before the token-tile loop
and shares query/bits storage while retaining the original warp0 maximum.
See [source and validation scope](tensor-padded-hoist.md).
The default layout remains `grouped`; this candidate has no accepted performance
result. The legacy Python adapter does not expose the experimental layout options.
