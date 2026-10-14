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
