# Optional exact partial-score selection

`SearchOptions::partial_topk` defaults to false. `--partial-topk` in the custom
batch benchmark and search CLI replaces only the first partial-score full sort.
The existing FP32 score reduction, materialized document order, downstream
document scoring and fullbit scoring remain the same.

The option requires GPU candidate generation/refinement and `k_refine` in
1..1000. It rejects CPU candidate/refinement replacements and modes that bypass
candidate refinement. Named evolution stages retain their original selector.

```text
original FP32 partial scores + original row/ID sequence
  → packed64 (CUB-compatible float radix rank, complement original row)
  → CUB DeviceTopK::MaxKeys, unsorted unique keys
  → one CTA256×4 integer sort + original ID/scorebits gather
  → existing K-ID D2D copy and downstream pipeline
```

The comparison rank merges ±0, while the payload keeps the original zero sign
and all other bits. The original row breaks equal-score ties, including repeated
document IDs. Scores are read from the original input after selection; the code
does not reconstruct floating-point values from the rank. The unpacked row is
valid even for a smallest-rank key because every real packed key is greater
than the zero sentinel at these int-sized row counts.

For nonempty input, `K=min(k_refine,N)`; the existing empty-candidate return
handles N=0 before calling the selector. The pack grid uses unsigned arithmetic
and guards every row. The finish CTA reads only K selected keys and writes only
K results, using 1024 shared-sort slots with zero sentinels.

Each `Workspace` has a noncopyable `PartialTopKWorkspace` owner. It reserves its
input keys, K-result buffers and CUB temp when the existing candidate arena
grows. New allocations are staged before swapping owners; failed allocation
releases only the staged buffers. The selector itself allocates no device
memory. QA row buffers exist only with `verify_full_bit=true`. Index cleanup
drains the existing GPU work, then releases these private buffers before stream
destruction; a later destructor release is harmless. Batch lanes never borrow
the owner's TopK buffers.

With `--verify-full-bit`, every real nonempty partial-selector call runs both
original full-N CUB pair sorts: one carries the original IDs, the other carries
the original row. Independent K-result buffers protect the primary output from
the references. The host compares ordered rows, IDs and FP32 scorebits before
printing:

```text
[PARTIAL_TOPK_VERIFY] candidates=N selected=K full_N_ID_oracle=passed full_N_row_oracle=passed rows_exact=passed ids_exact=passed scores_fp32_bitwise=passed
```

The existing limit32 benchmark geometry has five phases of32 calls, so a QA
case with all candidates nonempty produces160 such markers. The timed path
runs no reference sort, host readback or marker. It has one pack launch, the
installed CUB TopK launches and one finish launch; this implementation does not
fuse packing into the original reduction kernel.

The specialized path is compiled only with CUB version>=3.3.0 and an available
`device_topk.cuh`. Older CUB builds retain the original full sort and reject an
explicit TopK request before search initialization. `partial_topk_available()`
and `partial_topk_cub_version()` query this compiled capability without
initializing CUDA. Benchmark JSON records the option, availability, compiled
CUB version, selection path and `fp32-scorebits/radix-exact/stable-row` math.

The packed algorithm and ±0/Inf/NaN/tie rules were first checked in the standalone
micro against original CUB, including row/ID/scorebit and guard checks. This
integration requires a fresh build and actual same-candidate QA before timing.
