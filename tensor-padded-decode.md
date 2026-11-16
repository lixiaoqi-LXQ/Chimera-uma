# TensorPaddedDecode and TensorPaddedHoistDecode

The custom native CLI accepts `tensor-padded-decode` (appended enum7) and
`tensor-padded-hoist-decode` (appended enum8). Both use the existing fused scoring,
doc-order, UMA, chunks1 and exactly128thread contract. Default layout and enum0..6
remain unchanged.

Decode caches one selected uint32 code word per lane after barrier A. Each warp
owns one32-bit feature segment; 32 full-warp broadcasts produce the same32x128
half bit matrix. Invalid tail code slots are initialized to zero before A, and
the existing barrier B finishes all shared-code/shuffle reads before reuse.
Decode alone retains22,784B declared shared storage. The combination adds exactly
this qualified decode to Hoist's eight cached A fragments/query-bits union,
retaining14,080B declared shared storage.

Both keep the original warp0 max order, K order, LUT-to-half RN conversion,
half stride136, product stride36, initialization barrier, three barriers per
token tile and finalsum. They do not add Max4. Empty-document paths are
CTA-uniform. Direct/position addressing, the single mapped-ID/optional-length
writer per document, per-worker scratch and final event/deferred job lifetimes
are unchanged.

The shared Tensor-layout helper covers both modes, so native validation, finite
FP16-input domain checks, private scalar-half querymax/docscore bounds,
original-FP32 diagnostic/top20 comparison, direct ID/token checks and final
fullbit verification follow the existing Tensor contract. JSON reports the
actual layout name and existing math/stride fields. Batch workers forward the
layout through the complete SearchOptions copy.

Source is based on Hoist commit `632fb48dc9a98794446acf30d9ee2611c022240c`.
The qualified Decode and combined micro passed all16 fixture groups plus
real packed/span complete querymax/docscore raw-bit and guard checks before
this integration. Original TensorPadded, Hoist, Max4, scalar-half oracle and
`kernels.cu` remain byte unchanged. This integration still requires a native
build, direct off/on QA, complete GT/ID validation and a same-build screen.
Native register/shared/occupancy measurements are taken from that build.
