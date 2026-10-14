# A770 decode: where the time goes, what the hardware allows, and the plan

Arc A770 16 GB (Alchemist / DG2, `xe`), Ryzen 7 3800X, 30 GB DDR4, Coder IQ1_M at 128K, `--resident-experts`,
engine `v0.1.44.1` + #2207 + #2215 (`a770/next`). Measured 2026-10-14 unless marked. **Measured** numbers come from the
runs and probes named; **estimates** are labelled as such.

## 1. One decode round today (stepped window, 8K prompt)

About 100 ms for 2.8 tokens (decode 24.4-27.6 tok/s):

| part | ms / round | what it is |
|---|---|---|
| wait for rings | 43-44 | the host waits for each of 97 graph segments to drain (dense part of every layer, the VRAM hit experts, launch + drain latency) |
| pool | 35-43 | after a drain: the CPU computes the experts the VRAM cache does not hold (24% miss), thread create/join, copies, flags |
| -- of which CPU pool call | 30-34 | job building + miss prefetch 9.5-11.7, row compute 21 (29 GB/s) |
| MTP drafting | 10.8 | |
| adaptive tier swaps | 4.4 | |

The stepped window already overlaps the VRAM hit experts of a layer with the CPU's misses of the same layer
(verify.cpp, serve path: the split segment is submitted before the pool runs). What it cannot overlap is the chain
between layers: layer l+1 needs layer l's CPU experts. (Our earlier reading of `R4 overlap 0 of 0` as "no overlap" was
wrong: that counter belongs to the single-token hit path.)

## 2. What the hardware allows (`sycl/probe/a770/README.md`)

| resource | measured | per round today | floor |
|---|---|---|---|
| VRAM read | 427 GB/s | ~4.5 GB (ledger estimate) in ~44 ms (~24% of the bandwidth) | ~10.5 ms |
| RAM read, CPU | 34-42 GB/s | 6.35 distinct misses/layer x 48 x 1.95 MiB = ~595 MiB in 21 ms (29 GB/s) | ~15 ms |
| RAM read, GPU zero-copy over PCIe | 17.3 GB/s alone | not used | **shares the RAM with the CPU: together 28-36 GB/s, less than the CPU alone** |
| GPU <-> host handshake | 3.9 us per round trip (uncached-hint 32-bit store/load) | 97 segment launches + 48 blocking drains + 48 thread spawns | |

Estimate: a round bounded by the hardware would be around max(CPU ~15-21 ms, GPU ~10-20 ms) per layer chain plus
drafting, i.e. 2-3x today's decode. The CPU side is already within ~1.4x of the RAM limit; the GPU side and the
per-layer synchronisation are where the gap is.

## 3. What the probes rule in and out

- **Streaming misses to the GPU over PCIe (`--pcie-frac` on a chunked pinned resident set): rejected.** It is
  possible (a 4 GiB+ single pinned allocation fails, 2 GiB chunks work, 17.3 GB/s), but the GPU's PCIe reads and the
  CPU pool read the same DDR4: together 28-36 GB/s against 34-42 GB/s for the CPU alone. It would move work without
  adding bandwidth, and slow the CPU share down.
- **Copy engine as the host -> GPU data plane: rejected for now.** The copy engine runs beside a spinning kernel, but
  the GPU then reads the VRAM block stale (its L3 keeps the old lines; uncached read hints did not help). Without an L3
  invalidate this is unsafe.
- **A second SYCL queue to raise a device flag: unsafe.** It runs on the same compute engine, queues behind the
  spinning kernel and the GuC resets the engine (`DEVICE_LOST`).
- **Ordinary host USM for bulk data mid-kernel: unsafe.** Read fresh once, stale afterwards, whatever the load hint or
  fence. BIAS_UNCACHED host USM is the only data plane that stays fresh (what upstream's single-graph window already
  allocates), at ~2.3 GB/s in 256 KiB blocks and ~6.7 GB/s in 4 MiB blocks.
- **Flags work.** A 32-bit uncached-hint load (after an acquire fence) of ordinary host USM sees a host store in
  ~9 us; an uncached-hint store reaches the host mid-kernel; a full round trip is 3.9 us.
- **The engine's existing single-graph window works on this card but is slower** (`STRATA_VERIFY_STEPPED=0`,
  `STRATA_SPIN_MAX=2000000`): 19.98-21.48 tok/s against 24.41-25.12 stepped, same output ids; its GPU wait is 76-80 ms
  against 43-44 ms. Not caused by the shared-expert fork (`STRATA_SH_STREAM=0`: 76 ms). With plain host memory instead
  of BIAS_UNCACHED it drops to 13.5 tok/s and the ids change, which is the stale-read result above showing up in the
  engine.
- `--spec-split` in the stepped window: 1.63 tok/s and different ids (one run failed). Not usable as is.

## 4. The plan

### A. CPU pool overhead (no architectural change; low risk)

`dispatch ... jobs` is 9.5-11.7 ms of the 30-34 ms pool call, before any row is computed. It times the job list and
`src->prefetch()` of the misses. Under `STRATA_RESIDENT_HEADROOM_GIB=7` about 1,600 blobs per decode come from the GGUF
through the page cache (`expert tiers ... files 1600 blobs`), so part of this is file reads. Measure it with the
service's headroom (default 4), then profile the rest (perf). Expected: a few ms per round (estimate).

### B. A DG2 single-graph window (the architectural change)

Goal: one graph per round as on other GPUs, with the DG2-safe primitives from section 3, so the 97 launches, 48
drains and 48 thread spawns per round go away while the overlap of hits and misses stays.

1. **Phase 0 - find the 32-36 ms.** The single-graph window's GPU work is the same kernels as the stepped one, yet its
   GPU wait is 76-80 ms against 43-44. Candidates, each with an A/B or a device timestamp
   (device timestamps written into a device buffer at the stage boundaries, e.g. `sycl_ext_oneapi_clock` if this
   compiler has it - not checked yet; the existing
   `STRATA_VERIFY_PROFILE` changes the graph and inflates times):
   - the CPU rows read back from BIAS_UNCACHED memory with 64-bit loads (`copy_rows_from_mapped`): ~0.2 MB per layer
     at ~2.3 GB/s is ~4-5 ms per round (estimate from the probe);
   - the payload publish with uncached stores, and the plan copy from mapped memory;
   - the 1x1 spin kernels between graph nodes (3 per layer), and how long each waits for the host;
   - the host side: polling BIAS_UNCACHED memory, the pool's start latency after the GPU's publish.
2. **Phase 1 - DG2 primitives, behind one switch** (`STRATA_VERIFY_DG2=1`, opt-in):
   - flags (seq, A, B, M) in ordinary host USM, read and written with 32-bit uncached-hint accesses plus fences
     (measured 3.9 us round trip), instead of BIAS_UNCACHED;
   - bulk data (payload, plan, CPU rows) stays in BIAS_UNCACHED host USM, read with wide coalesced loads by many
     work-items (the probe's bandwidth rises with block size), not by a 1x1 kernel;
   - spin bound for Alchemist on `xe` raised (today 20,000 reads, too short for a CPU layer);
   - every other path unchanged.
3. **Phase 2 - make it the Alchemist default** only after it measures faster than stepped on the A770 and is checked
   on an A750 (i915) by the maintainers.

Expected gain (estimate, to be replaced by Phase 0 numbers): the stepped window's per-layer launch/drain/spawn cost is
the difference between its 44 ms GPU wait and the GPU's actual kernel time; if the single graph matched the stepped
window's kernel time without that cost, 10-30 ms per round, i.e. +10-40% decode. If Phase 0 shows the extra time is the
BIAS_UNCACHED reads themselves, the gain is smaller and the option is to read less (only the rows the combine needs,
in larger blocks).

### C. GPU kernel efficiency (after B)

At ~24% of VRAM bandwidth the decode kernels have headroom (ledger: GPU expert gate/up and down at ~100 GB/s). Today
part of it is hidden behind the per-segment overhead, so this comes after B.

## 5. Risk to other platforms, and getting it upstream

- **Files:** B touches SYCL-only code (`sycl/src/core/verify.cpp`, `sycl/include/strata/sycl_doorbell.hpp`, the
  wait/copy kernels). CUDA and HIP do not compile these files: no risk there.
- **Other Intel cards:** everything is behind `STRATA_VERIFY_DG2` (Phase 1) and then an Alchemist gate (Phase 2), the
  pattern upstream accepted for #2193 #2199 #2203. Battlemage and every default path keep today's code. The risk is in
  shared helpers if they are edited in place; keep the change additive (new wait/copy kernels, not modified ones).
- **Correctness on Alchemist:** the failure mode is stale data (wrong tokens) or a GPU reset, not a crash.
  Mitigations: bounded spins (already), output ids identical to the stepped window over a prompt set,
  `STRATA_DOORBELL_CHECK=1` in validation, and a **startup self-test** (the flag and BIAS_UNCACHED freshness probes,
  ~1 ms) that falls back to the stepped window if it fails. The self-test also covers driver versions that behave
  differently (the cache-hint behaviour is a driver/compiler property).
- **Driver dependence:** the results are for compute runtime 26.05 / oneAPI 2026.1 on `xe`. i915 (A750) is untested
  here; the maintainers have one.
- **Upstream order** (one change per PR):
  1. the probes + this measured table as docs (no code risk; gives the maintainers the DG2 memory semantics);
  2. Phase 0 instrumentation, if it is useful beyond the A770;
  3. the DG2 window, opt-in, with the self-test and a parity test;
  4. the Alchemist default, with A770 numbers and the maintainers' A750 run.

Likelihood of breaking other platforms with this sequence: low (opt-in, SYCL-only, additive, self-tested). The real
risk is that B does not pay: Phase 0 decides that before the bulk of the work.

## 7. The architecture, measured (2026-10-14)

Where the A770 time goes once every measured structural alternative is tried. Coder IQ1_M, 128K, `--resident-experts`,
service headroom (4 GiB), v0.1.44.1 + #2207 + #2215.

**What was architecturally wrong, and is fixed** (`a770/up-resident-chunked`): the RAM-resident expert copy. The CUDA
engine registers it (`cudaHostRegister`); the SYCL port could not (DPCT1027) and DG2 refuses one pinned allocation of
>= 4 GiB, so the copy was pageable and the GPU copied from it at 2.4-4 GB/s through the host stager. Pinned in 2 GiB
chunks: prompt experts 9,352/9,352 by DMA (was 0), host staging -95%, lent-slot refill -55%, tier swaps -78%,
prompt +1.5%, decode +0.5-1.3%, identical ids. (Registering the pageable copy with `prepare_for_device_copy` instead
works in a probe at 18 GiB but failed in the engine above 4-10 GiB with `UR_RESULT_ERROR_OUT_OF_RESOURCES` from the
adapter's pool; rejected.)

**Prompt (12 s for 8K): GPU-bound, and the cost is the MoE dispatch structure.** The compute engine is 93-96% busy;
the host thread waits on oneMKL's copy-based FP16 GEMM driver (synchronous: `mkl_blas_gpu_pool_complete_request_usm`
-> `urEventWait`, 82% of its waiting, all from the dense projections), which costs no GPU idle. The expert path is
~5.7 s of 11.9: ~12,300 per-expert oneMKL calls + a staging copy per GEMM (oneMKL misreads operands > 4 GiB into an
allocation on this card; the prompt path's buffers are borrowed from the cache arena past 4 GiB) + a separate
dequant pass. A synthetic layer (`moe_layer_probe`, 256 experts, 79,866 uneven rows): engine-shaped 87 ms, without
staging 77, without dequant 31, one GEMM pair over all rows 13.6 ms (57.8 TFLOPS; oneMKL peaks at 75-82 TFLOPS).
**The fix is a grouped MoE GEMM** (all experts of a layer in one or two launches, dequant fused, operands where the
oneMKL bug cannot reach): the expert path's ~5.7 s toward ~1.5 s, i.e. a prompt of ~12 s -> ~8 s (estimate). The
existing grouped XMX prototype is short of it: `STRATA_PF_XMX=1` 710 tok/s (+4%), `=2` 658 (DG2 8x8x16 joint_matrix,
untuned).

**Decode (~100 ms/round, 2.8 tokens): the stepped window is the right structure; the limits are compute.** Measured
and rejected: the single-graph window (20-23 tok/s, +35 ms/round GPU for an unexplained reason), `--spec-split` (two
groups pipelined: 1.6 tok/s stepped, 17.4 single-graph), the GPU reading misses over PCIe from the pinned copy
(`--pcie-frac` 0.2 / 0.35: 30.96 / 29.04 vs 31.30 tok/s; the GPU's PCIe reads cost more than the CPU time they save,
as the RAM contention probe predicts), host-overhead removal in the stepped serve (hidden behind the CPU pool). What
is left per round: the GPU's dense layers ~43 ms (~25% of VRAM bandwidth), the CPU's misses ~22-25 ms (within ~1.4x of
the RAM limit), MTP ~11 ms, tier swaps ~1 ms after the fix. The decode levers are GPU kernel efficiency and the miss
rate (VRAM for more cache slots), not the window architecture.

## 8. Prompt: the expert cache in segments (2026-10-14)

The prompt path borrows the top expert-cache slots (5.4-7.6 GiB into the ~8.6 GiB arena), and oneMKL's GEMM misreads an
operand more than 4 GiB into one allocation, so `staged_gemm` copied nearly every expert GEMM operand first: 24,576 staged
GEMMs in an 8K prompt, 123 GB of W and 76 GB of X copied (`STRATA_SYCL_STAGE_STATS`, diagnostic, not kept).  Running
direct and staged side by side in the engine (`STRATA_SYCL_GEMM_VERIFY`, diagnostic) reproduced the misread for the
dense prompt GEMMs at T = 7987, and `gemm_far_probe` placed it: ONE allocation only.  The same GEMMs in physical
segments behind one reserved address range read right at any offset.

Change (`strata::malloc_device_segmented`, `sycl_queue.hpp`; `ExpertCache::open`): on Alchemist the arena is 512 MiB
segments behind one range (`STRATA_ARENA_SEGMENTED=0`: one allocation as before, `=1`: other devices too) and is not
registered for staging.  Level Zero refused a smaller last segment, so the segments are equal and of whole 2 MiB pages.
A copy kernel in place of the staging memcpy gained nothing (bandwidth, not engine switches).

Same-day A/B, deployed config (`STRATA_PF_XMX=2`, resident RAM, learned24), segmented vs `STRATA_ARENA_SEGMENTED=0`:

| prompt | prefill seg | prefill one | decode seg | decode one | output |
|---|---|---|---|---|---|
| 8K #1 | 726.0 | 657.3 | 32.57 | 32.52 | identical |
| 8K #2 | 730.8 | 656.8 | 32.36 | 32.46 | identical |
| 32K #1 | 742.5 | 663.1 | 26.77 | 22.32 (outlier) | identical |
| 32K #2 | 742.2 | 663.3 | 26.79 | 26.80 | identical |

Prompt +11-12%, decode unchanged.  `STRATA_SYCL_GEMM_NO_STAGE=1`'s 1,122 tok/s was not a ceiling: its garbage
activations collapsed the routing (365 experts streamed instead of 9,352).  The custom grouped XMX GEMM was measured too
(`xmx_v`/`xmx_g` harnesses, 16 experts x 312 rows): SLM-tiled, double-buffered and direct-global variants with VNNI-packed
B all plateau at 10-12 TFLOPS on DG2 (no 2D block loads), against 23 for per-expert oneMKL and 59 dense, so it is not the
prompt's lever on this card; the remaining expert-GEMM gap (engine ~13 vs isolated 20-33 TFLOPS) is next.

## 9. Decode: where the GPU time goes, and pre-attention expert prefetch (2026-10-14)

GPU-clocked stage table (Phase 0 clock, branch `a770/decode-prof`, T ~4, ms per window, stamps included): VRAM-hit
experts 12.4, hc0 norm/down/up 7.3, shared + quant 5.9, the GEMVs (qkv, out-proj, z, ab) ~11, recurrence / attention /
top-k / small kernels ~15. No single kernel dominates; the hit experts run at about a third of VRAM bandwidth and most
small stages are launch-sized (8-12 us). fdinfo shows the compute engine ~99.5% busy all through decode (spin-waits
included), so it cannot separate work from waiting.

The serial part of a round is the CPU's out-of-VRAM experts (pool ~24.5 ms of ~85). A pre-attention prediction would
let them be copied to VRAM during the layer's attention and computed as hits. `STRATA_PREDICT_PROBE=1` (diagnostic,
`a770/decode-prof`) runs each layer's router on its attention input and compares with the real routing, per layer-window:

| | value |
|---|---|
| distinct experts routed | 22.6 |
| of them out of VRAM | 6.45 |
| recall of those, predicted top-10 / 16 / 20 | 0.654 / 0.794 / 0.842 |
| prefetches out of VRAM, top-10 / 16 / 20 | 8.8 / 15.2 / 19.7 experts (17 / 30 / 38 MB) |

At top-10 the DMA is ~825 MB per round, ~48 ms at 17 GB/s: as long as the GPU part of the round, and sharing RAM
bandwidth with the CPU's remaining misses. Estimate: ~85 -> 65-70 ms per round (+15-25% decode), bounded by the copy
volume, not the recall. Cost: copies into the captured stepped graphs (graph copy nodes beside a spinning kernel gave
DEVICE_LOST, section 7), device residency updates inside a window, prefetch slots. A multi-day engine feature; not
started.

## 10. Decode: the VRAM prefetch, built and measured (2026-10-14) - rejected

Built on branch `a770/vram-prefetch` (opt-in `STRATA_VRAM_PREFETCH=<top-k per token>`, `_SLOTS=<experts per layer>`):
`RouterLookahead` (CS-T, the next layer's router on this layer's MoE input; recall of the out-of-VRAM experts 0.64 /
0.80 / 0.86 at top-10 / 16 / 20, `STRATA_PREDICT_PROBE=2`) copies its predicted out-of-VRAM experts from the pinned RAM
copy into a two-layer VRAM ring on its own queue; the next layer's pool plans a miss found there as a VRAM group (the
hits' own kernel).  First version put them in the PCIe-group list: with `--pcie-mode auto` (the in-graph staging copy,
16 slots) the extra groups overran staging and the output turned to zeros after 79 tokens; as VRAM groups the output
is right (ids identical to the baseline in most runs; one run diverged at token 138 by the GPU/CPU rounding the cache
note describes, no garbage).

Same-day, 8K, `STRATA_PF_XMX=2`, resident RAM, learned24 (baseline 32.53 / 32.56 tok/s decode):

| prefetch (top-k / slots) | decode | GPU wait | pool | CPU misses / layer | used |
|---|---|---|---|---|---|
| 16 / 20 | 20.81 | 108.3 | 10.9 | 1.61 | 4.92 of 12.37 |
| 10 / 6 | 27.06 | 67.9 | 17.9 | 3.45 | 2.96 of 4.77 |
| 10 / 6, copy kernel | 25.73 | 62.4 | 28.9 | 3.38 | 2.97 of 4.76 |
| 6 / 3 (x2) | 30.06 / 30.08 | 53.1 | 22.6 | 4.70 | 1.71 of 2.25 |
| 10 / 6, copies never used | 26.69 | 58.4 | 29.1 | 6.29 | 0 |

It does move the misses off the CPU (pool 24.5 -> 11-23 ms), but the copies themselves - unused - add ~14 ms of GPU
wait and ~4.6 ms of pool per round at ~220 MB/round (CPU row bandwidth 29.4 -> 24.3 GB/s; the copies run at 14.6
GB/s, on the copy engine or a copy kernel alike).  Every configuration loses (-8% to -36%).  Same lesson as the
zero-copy PCIe share (section 7): on this machine host->GPU traffic during decode costs the GPU's critical path more
than the CPU time it saves.  Not proposed upstream; off by default.

**Decode status:** the measured candidates for an architectural decode change on the A770 - single-graph window,
spec-split, PCIe zero-copy misses, pre-attention / next-layer VRAM prefetch - all lose.  What remains is per-kernel GPU
efficiency (no kernel dominates; VRAM hits ~123 GB/s, launch-sized small stages) and a higher hit rate (more VRAM).

## 11. Decode kernels and the runtime (2026-10-14, evening)

Where the decode round's GPU time is on the critical path: the stepped window runs the VRAM-hit experts and the
shared expert BESIDE the CPU pool (hidden while the pool is longer); the critical part is each layer's dense segment
(hc reads, the qkv / z / out-proj / q GEMVs, the recurrence, attention, the combine) plus the segment submissions.

| change | where | measured (A770, same day) | bits |
|---|---|---|---|
| large GRF for the grouped expert kernels (`STRATA_EXPERT_LARGE_GRF`, Alchemist default) | iq_kernels | `native_grouped_parity --bench2`: 218 -> 179 / 215 -> 197 / 280 -> 232 us a call; decode +1.3% (hidden behind the pool) | same |
| the GR down reduce folded into the sliced down kernel (last slice of a row block reduces; `STRATA_GR_FUSED_REDUCE`) | fused_gr | gr_bench 1-6 tokens 88.4-150.8 -> 76.8-146.5 us | same (checksums) |
| GR up kernel at 4 columns a block (`STRATA_GR_UP_MORE_COLS`) | fused_gr | gr_bench 4-6 tokens 111.3-146.7 -> 106.8-136.3 us | same |
| the three together | | decode 33.64 / 33.41 -> 34.80 / 34.84 tok/s (+3.9%), ids identical | same |
| the up kernel reads the lo values the fused reduce computes once (`STRATA_GR_LO_READY`) | fused_gr | gr_bench 4 / 6 tokens 106.7 -> 104.9, 137 -> 133 us | same |
| the norm folded into the sliced down kernel (`STRATA_GR_NORM_FOLD`, Alchemist default): slices stage residual x w_norm unscaled + sums of squares, the reduce applies each stream's rs | fused_gr | gr_bench 1 / 4 / 6 tokens 74 -> 62, 105 -> 90, 134 -> 115 us | fp32 order (sums equal ~1e-7) |
| the up projection one row a lane (`STRATA_GR_UP_V5`, Alchemist default): lo read as a local-memory broadcast instead of re-read per row (the old kernel was local-memory bound) | fused_gr | gr_bench 4 / 6 tokens 90.3 -> 83.3, 115.4 -> 100.5 us; 8 lanes a row (v4): slower | fp32 order |
| norm fold + row-a-lane up, engine | | GPU wait 39.2-39.9 -> 38.1-39.2 ms/round at 8K / 32K / 2K; decode within noise (~+1%); output text equivalent, 2K ids identical | fp32 order |
| Level Zero v2 adapter (`UR_LOADER_USE_LEVEL_ZERO_V2=1`, launcher default) | runtime | a graph segment submits in ~25 us instead of 53-200 (instrumented serve); GPU wait 43-48 -> 39.4-39.7 ms/round; decode +7%, prompt +3% (3 pairs 8K + 32K); the stepped window's 3x-slower outlier runs (5 of ~50 with v1) not seen with v2 | same kernels |
| restricted draft vocabulary (`--mtp-draft-vocab data/draft_vocab_en.bin`, service config) | config | drafting 10.7 -> 7.5 ms/round; decode +5.5% (4 pairs); output coherent (greedy verification) | other window sizes |

Measured and rejected: the existing lane / split knobs of the expert kernels (+-1%, split -38%), `use_root_sync`
(no cost), a persistent pool helper thread (no difference under v2), the single-graph window under v2 (22 tok/s),
`--spec-min-p` 0 / 0.3 / 0.7 and `--spec` 3 / 5 (all slower than spec 4 / min-p 0.5), all-80-in-flight reduce loads
(spills), the up kernel's ILP variant (3x slower). The hc projections of this GGUF are BF16, so upstream's
`STRATA_HC_Q8` (the GGUF's own Q8_0 bytes) does not apply.

Cumulative, same day, against the morning's configuration (PF_XMX=2, one-allocation arena, v1 adapter, no draft
vocab, kernel switches off): 8K prefill 654 / 656 -> 791 / 786 tok/s, decode 29.0 / 31.8 -> 33.8 / 35.2; 32K prefill
664 -> 794, decode 26.3 -> 29.6.  Remaining GPU levers: the hc reads (~106-137 us each, two a layer, ~125 GB/s), the
dense GEMVs at 4 columns (180-240 GB/s). After the norm fold and the row-a-lane up kernel the hc read is 62 / 83 / 100
us at 1 / 4 / 6 tokens (from 88 / 123 / 151 in the morning).

## 12. GPU saturation: where the decode GPU idles (2026-10-14, night)

unitrace device timing of a 2K-prompt decode (minus an 8-token run: decode only, ~62 rounds): ~55 ms of kernels in an
~87 ms round; the rest is the GPU idle while the CPU computes the out-of-VRAM experts (~25 ms) and the segment gaps.
Biggest kernels per round: the dense Q6_K wide GEMVs 8.6 ms (68.5 us a call, ~300 GB/s), the hc read 10 ms (down 63 +
up 40 us a call in the engine), the GDN recurrence 6.7 ms (step 118 + commit 75 us), the VRAM-hit experts ~7 ms
(beside the CPU), router / top-k / other GEMVs ~9 ms.

- upstream's split GDN recurrence (`STRATA_GDN_SPLIT`, S25/S26, bitwise - gdn_parity 47 cases) is now the Alchemist
  default: GPU wait 39.5 / 38.8 -> 37.6 / 37.9 ms a round.
- `--vram-reserve-mib` (the cache's headroom): 300 / 800 MiB give 4,959 / 4,701 slots instead of 4,495 (hits 0.755 ->
  0.79 / 0.78) and +3-4% decode at 8K, but at 32K the serves stall (VRAM too full: 15.4 / 26.7 against 31.9 / 28.9
  tok/s). The 1200 default stays.
- Benchmark hygiene: other jobs on this host (8 x `yes` under `timeout 420`, `test-feature-provenance`) cut the CPU
  pool's row bandwidth from ~30 to 14-17 GB/s in some runs; `~/strata-probes/quiet.sh` now gates each bench.

The GPU cannot be kept busy through the CPU's share while ~24% of the experts are out of VRAM: that share is the
remaining structural idle, bounded by the card's 16 GB against ~27 GB of experts.
