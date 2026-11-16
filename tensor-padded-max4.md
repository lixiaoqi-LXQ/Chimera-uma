# TensorPaddedMax4: opt-in native candidate

Select `--fused-token-document-layout tensor-padded-max4` with the existing fused
scoring, doc-order and UMA flags, chunks1 and exactly128threads. The C++ API uses
`FusedTokenDocumentLayout::TensorPaddedMax4`. This enum is appended after the five
existing values; defaults remain unchanged. Named benchmark stages reject this
custom experiment. Direct=false uses matching positions/offsets; Direct=true
keeps the existing no-CPU-refine constraint and one metadata writer per document.

The new kernel retains WMMA Q32/K128/Ntile32, FP16 inputs, FP32 accumulation,
A/B stride136, C stride36, four warps and the original finalsum. It assigns four
lanes to each query, scans eight tokens per lane and reduces with width4 shuffles.
Each lane captures eight factors in registers before the original barrier B;
there are no shared factor reads after B. The next tile may overwrite only the
disjoint code/factor storage before barrier A waits for all product/max readers.
Shared storage and CTA barrier count are unchanged. Empty/tail documents keep the
same sentinel, validity predicate and single scratch writer per query.

Both Tensor layouts use the same finite/range check before GPU submission,
per-lane private diagnostic buffers, independent half-scalar querymax/docscore
bounds, original-FP32 error/top20 diagnostics, direct ID/token verification and
strict original-precision fullbit verification. Tensor layouts do not emit the
FP32 fused-score bitwise marker. New mode selection and math labels are reported
in benchmark JSON. Batch workers copy the full SearchOptions and allocate their
own diagnostic workspace.

This source is based on E1 `e1a2dc0276ab05d55c3788bb41995b6685f48694`. The original
TensorPadded header and `kernels.cu` are unchanged. The independent micro passed
bitwise querymax/docscore fixtures and a real packed snapshot; timing with external
activity is ineligible. This integrated source has not been compiled or GPU-tested.
Before adoption, run native32QA for direct off/on, complete GT/returned-ID checks
and an interference-free paired screen. All existing half-scalar bounds and
fullbit oracles remain required.

The current optional Python CMake target has no `Chimera/python/` implementation
in this branch; the old `systems/chimera/arm-source/python/bindings.cpp` exposes only
legacy arguments. This candidate is exposed through the existing native C++ API
and custom benchmark CLI. No Python API or optional-CMake repair is included.
