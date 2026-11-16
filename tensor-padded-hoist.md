# TensorPaddedHoist: opt-in native layout

Select `--fused-token-document-layout tensor-padded-hoist` with fused scoring,
doc-order, UMA query buffers, chunks1 and exactly128threads. The native API uses
`FusedTokenDocumentLayout::TensorPaddedHoist`, appended as enum value6. Existing
enum values and defaults are unchanged. Named benchmark stages keep their
existing custom-experiment restriction.

The kernel caches eight named WMMA A fragments per warp before the token-tile
loop and aliases query/bits shared storage in one32x136 half union. Declared
shared storage is14,080B instead of22,784B. The original LUT-to-half RN path,
K=0,16,...,112 accumulation order, warp0 maximum order, half stride136, product
stride36, per-tile three barriers and original finalsum are retained. The first
tile's existing barrier A waits until all warps finish every query/A load before
bits decode overwrites the union. Empty documents skip this loop CTA-uniformly.
This mode includes neither Max4 nor decode caching.

Direct=false retains matching positions/offsets. Direct=true retains natural
document ranges and exactly one mapped-ID/optional-length writer per document;
GPU input IDs and ATS output IDs remain separate. Existing lane scratch, stream
ordering, completion events and deferred fullbit lifetime are unchanged.

The shared Tensor-layout helper includes this enum. It therefore uses the same
finite/range domain gate, private per-worker diagnostic allocations, independent
FP16-input/FP32-accum half-scalar querymax/docscore bounds, original-FP32 error
and top20 diagnostics, direct-ID/token checks and strict fullbit oracles.
Batch workers copy the entire SearchOptions object. JSON reports the new layout
name and the existing Tensor math/stride fields. No Tensor path claims original
FP32 onebit-score bitwise equivalence.

The isolated micro passed all16 fixture groups and real packed/span complete
querymax/docscore raw-bit comparisons with output guards. This native integration
is source-only: it still requires compilation, native direct off/on QA, full GT
recall/returned-ID validation and a paired screen. Native registers, spill and
occupancy must be measured on the integrated binary; the micro's register count
is not assigned to the native kernel.

Base source is B09 `b786ea1a7cf6fd98aa0dc08c2d47baff59c457b9`. Original
`tensor_token_document.cuh`, Max4, scalar-half oracle and `kernels.cu` are byte
unchanged. The existing native API/custom CLI expose the mode; no Python API or
optional Python CMake target changes are included.
