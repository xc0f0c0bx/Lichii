# Game-DLSS replacement

The Vulkan renderer can replace a recognized game-DLSS evaluation with NVIDIA DLSS on
Windows, then Intel XeSS on Windows, then AMD FSR 3.1.4 if those are disabled or unavailable.
FSR uses AMD's eight temporal-upscaling passes, including depth, motion, history and optional RCAS.
This is separate from the presentation filters that scale the finished window image.

Fire Emblem uses the jitter-only spatial workaround by default. Explicit `fsr` and `dlss`
comparison modes enable temporal replacement for its validated input contract. The upstream
temporal blend and RCAS fallback remain available if a temporal backend is unavailable.
Invalid consumed inputs leave the native dispatch enabled. An already-rejected native
tensor shader remains an explicit dispatch failure. General tensor emulation remains.

## Supported input contract

Selection currently recognizes Fire Emblem: Fortune's Weave's captured prepass and output
shader, under its exact program ID. Temporal mode requires matching B10G11R11 input/output,
R16G16 motion, D32 depth and a fresh jitter sample. Spatial mode requires only the matching
color/output and fresh jitter. The upstream temporal contract uses reversed depth,
normalized motion toward the previous frame, HDR color and jitter in render pixels.
Resolution scaling scales the render extent and jitter together. History resets on size
changes, first use and evaluation gaps longer than 250 ms. An explicit game camera-cut
reset is not yet extracted.

Fast Fusion's shaders still execute natively. Its captured shader identities are known,
but motion scale/direction, jitter units, depth convention, reset and resource roles have
not been proven for this LLE path. The RGBA16F menu/DLAA path also requires alpha handling
that the Fire Emblem contract does not provide. Turning on NVIDIA DLSS does not override
these safety checks and is not evidence of a Fast Fusion visual fix.

For a diagnostic run, `EDEN_NX2_DLSS_CONTRACT_TRACE=1` logs at most 24 matching Fast Fusion
dispatches per compute engine. `EDEN_NX2_DLSS_CONTRACT_SKIP=N` skips the first N matching
dispatches, capped at one million. Logs include exact shader identity, QMD launch dimensions,
bounded constant-buffer words and raw texture descriptors. They contain guest addresses
and should remain private. No shaders are skipped or game files changed by this capture.

## Tensor instructions on the host

A game's own DLSS network, and any other Switch 2 shader using the tensor (HMMA) instructions,
runs those as `TensorMulAdd` in the shader recompiler. By default they are emulated in software
(`emit_spirv_tensor.cpp`): a loop of subgroup shuffles and fused multiply-adds reproducing
NVIDIA's fp16 rounding after every K=8 group, which needs compute shaders 32 threads wide.

When the device offers `VK_KHR_cooperative_matrix` with a subgroup-scope fp16 shape of
16x8x16 (the guest's own shape) or 8x16x16 (Intel Arc, through the transposed product), the
compute stage uses the matrix hardware instead: XMX units on Arc, tensor cores on NVIDIA. Each
lane stores its m16n8k16 fragments to a per-subgroup shared-memory scratch, one
`OpCooperativeMatrixMulAddKHR` computes the product, and each lane reads its result back.
Sirit has no KHR cooperative matrix instructions, so they are emitted as markers and rewritten
after assembly. The hardware accumulates differently from the software path (no fp16 rounding at
the K=8 midpoint), so results are close but not bit-identical; the network's output checks still
apply. A depth-8 product uses the same K=16 shape with its zeroed upper registers.
`EDEN_NX2_TENSOR=software` forces the software path for A/B runs. The path is skipped outside
compute, without fp16 support, and when the scratch would push shared memory past 32 KiB.

## Settings and comparison runs

"Use NVIDIA DLSS in place of a game's DLSS" defaults to off and takes effect when the game
starts. Disabled means no NVIDIA runtime is loaded; an explicitly selected temporal
replacement uses FSR 3.1. Missing DLLs and unsupported GPUs fall back to FSR. Recording is
asynchronous. A failed DLSS recording clears that frame's output, restores input layouts,
and makes later evaluations fall back; it does not wait for the scheduler worker.

`EDEN_NX2_DLSS_REPLACEMENT=fsr` selects temporal replacement and forces FSR for comparison.
`dlss` selects temporal replacement with NVIDIA preferred when enabled and available.
`native` preserves the guest evaluation. With no override, Fire Emblem uses spatial de-jitter.
The old `EDEN_NX2_DLSS_STANDIN` variable is not read. The internal `standin` backend override
does not select Fire Emblem's intercepted dispatch. The RCAS sharpness setting controls FSR
and fallback sharpening. NVIDIA model-selection overrides
from the upstream implementation remain available for diagnostic comparisons.

## Intel XeSS

"Use Intel XeSS in place of a game's DLSS" defaults to off and takes effect when the game
starts. It runs after NVIDIA DLSS (which wins when both are on and available) and before FSR
3.1. XeSS super resolution uses the XMX units of Intel Arc GPUs; on other vendors it runs its
DP4a path, so it is mainly worth turning on for Arc. `EDEN_NX2_DLSS_REPLACEMENT=xess` selects
temporal replacement and forces XeSS over NVIDIA DLSS; `fsr` and `dlss` skip it.

The backend (`renderer_vulkan/vk_xess_upscaler.cpp`) hands XeSS the same inputs as the other
two: jittered linear colour, reversed depth (`XESS_INIT_FLAG_INVERTED_DEPTH`) and render-size
motion vectors scaled to pixels, with XeSS's auto exposure. The quality setting is the one whose
input range holds the render size. `EDEN_NX2_XESS_MV_SCALE` and `EDEN_NX2_XESS_JITTER_SCALE`
(default 1; -1 flips a convention), `EDEN_NX2_XESS_AUTO_EXPOSURE=0` and `EDEN_NX2_XESS_LDR=1`
are for A/B runs. A failed initialization or evaluation clears that frame's output and makes
later evaluations fall back to FSR 3.1.

`vulkan_common/xess.cpp` loads `libxess.dll` from `EDEN_XESS_DIR`, the executable's folder or
its `xess` subfolder before the Vulkan instance is created, and asks XeSS for the instance
version, instance and device extensions and device features it needs (the device is created with
them). Intel ships the library for Windows only; the code compiles elsewhere so it is checked on
every platform, and XeSS reports itself unavailable there.

## Build and distribution

FSR needs no external runtime. Keep `externals/FidelityFX-SDK/LICENSE.txt` and its attribution
with distributions. The vendored files are unmodified AMD MIT sources from FidelityFX SDK
v1.1.4, commit `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`.

For the optional Windows NVIDIA path, obtain only the MIT Streamline headers and license
from NVIDIA-RTX/Streamline tag `v2.14.16`, commit
`625db07d4bca5abc67d781c42af7e7fb0761f1b6`. The SDK directory must contain `include/sl.h`,
`include/sl_dlss.h`, `include/sl_helpers_vk.h`, `include/sl_security.h` and their companion
headers. MSVC x86_64 builds enable `ENABLE_NVIDIA_DLSS` by default and fetch the headers;
`-DEGLANTINE_STREAMLINE_SDK=<header-root>` selects a local copy. This adds Windows
`version`, `wintrust` and `crypt32` libraries, not NVIDIA import libraries. Retain Streamline's
MIT `license.txt` in the packaged notices.

For Intel XeSS, Windows x86_64 builds enable `ENABLE_INTEL_XESS` by default and fetch the
XeSS SDK 3.0.2 release zip (`cpmfile.json`, `xess`); `-DEGLANTINE_XESS_SDK=<sdk-root>` selects a
local copy. The SDK's headers are compiled in and its `bin/libxess.dll` is copied next to the
executables and installed with them, as the Intel Simplified Software License allows for the
unmodified binary; its text ships as `LICENSE-XeSS.txt` next to the executables and must stay
with distributions. Nothing of Intel's is linked: the library is resolved at run time.

Do not bundle NVIDIA's proprietary DLLs. The Graphics panel obtains missing files from
NVIDIA's Streamline release only after the user reviews and accepts the license texts.
The runtime searches `EDEN_STREAMLINE_DIR`, the user `streamline` directory, then the
executable's `streamline` directory. It verifies the interposer signature and supports the
NVIDIA app's signed DLSS model. The UI and Windows loading path require Windows validation.

## Provenance

This backport is AI-assisted and requires human review and testing before distribution.
It carries upstream work by Duck912 from Eglantine commits `08dbcc7e3b3d`, `11cebf56014a`
and PR #434 through `f9d0c802e8e4`, plus Jonas's warning fix `537c1f4e173a`. The empty-extension requirement
fix `1889602c7a0d` is a separately included follow-up. Existing SPDX notices and AMD/NVIDIA
license text are retained. The upstream nx2 merge also carries newer asynchronous recording
and fallback changes while retaining the fork's signature checks and input validation.
