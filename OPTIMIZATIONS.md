# openEMS performance work — GPU and AVX2 engines

Running record of optimization work on the Vulkan GPU engine and the AVX2
engines: what was done, what it was worth, and — just as important — what was
tried and rejected, so it is not rediscovered blind.

Scope of this document is deliberately narrow: `Engine_Vulkan` /
`Operator_Vulkan` and `Engine_AVX2` / `Engine_AVX2_Multithread`. The SSE and
basic engines are out of scope.

Measurements come from three hosts and are not comparable across them:

- **Host A (historic, CPU)** — i7-6700K, 4 cores / 8 threads (SMT2),
  dual-channel DDR4. Practical STREAM-Triad ceiling ~29 GB/s. Everything in
  §1's AVX2 table was measured here. **This host is gone**; see §4.
- **Host B (historic, GPU)** — Radeon RX 6800, 512 GB/s peak, 128 MB Infinity
  Cache, **256 MB BAR** (no resizable BAR). Everything in §1's GPU table was
  measured here.
- **Host C (current, 2026-09-04)** — i9-13900K (8 P-cores + 16 E-cores,
  32 threads), 32 GB dual-channel DDR5, STREAM-Triad ceiling **~49 GB/s** at
  6–8 threads. Two GPUs: the same **Radeon RX 6800** — still a 256 MB BAR,
  still no resizable BAR, so `fbd95a6` is load-bearing here too — and an
  integrated **Intel UHD 770**. Mesa 26.2.1, RADV and ANV. §4 re-verifies
  everything here.

Benchmark drivers: `python/Tests/benchmark_avx2_engine.py`,
`python/Tests/benchmark_gpu_engine.py` (`--gpu-index` selects the device),
`python/Tests/test_gpu_engine.py`, `python/Tests/test_upml_engines.py`,
`OPENEMS_RUN_STABILITY_TESTS=1 python/Tests/test_stability_testsuite.py`.

---

## 1. Completed work

### GPU

| Commit | Change | Measured |
|---|---|---|
| `fbd95a6` | Don't place field buffers in an undersized BAR aperture | **6.4–6.5×** on grids exceeding the 256 MB aperture (803 → 5115 MC/s at 256×224×224) |
| `ea82f59` | Async field-dump download; request `HOST_CACHED` staging | **~18×** wall clock on a dump-heavy 291³ run (74 s → 16 s) |
| `2ed6523` | Reuse shared curl taps in the update shaders | Neutral on RADV (its NIR already CSE'd them); kernels shrink 11–14%. Portability insurance |
| `49c7efc` | Narrow the compressed-operator index to 8/16/32 bits | **+3.5% to +6.2%**; Coax index buffer 2793 → 698 KB |
| `fabcaa3` | Single Vulkan 1.3 baseline, negotiate device features | Enablement, not throughput |
| `6b571da` | Per-cell operator storage when deduplication stops paying | **1.93×–2.76×** on graded meshes; uniform meshes unchanged |
| `00ba1c6` | Deduplicate UPML coefficients into shared tables | **1.08×–1.17×**; 19.2 MB → 1.6 MB, 53.5 MB → 4.5 MB |

Two structural results worth carrying forward:

- **The Yee kernel is at roofline.** After `fbd95a6`, ~5.2 GCells/s × ~80
  B/cell/timestep ≈ 415 GB/s against a 512 GB/s peak. Grids that fit the
  128 MB Infinity Cache run 12–14 GCells/s. There is no headroom left in the
  plain PEC update path; remaining GPU wins must come from *reducing traffic
  or passes*, not from tightening arithmetic.
- **Operator-table locality dominates on graded meshes.** `6b571da` came from
  observing a 4× collapse (383 → 95 MC/s) when 493,040 unique coefficient
  sets in a 23 MB table turned six per-cell gathers into six scattered cache
  lines. Table *size relative to cache* is the thing that matters, not the
  compression ratio.

### AVX2

| Commit | Change | Measured |
|---|---|---|
| `e3932e8` | Eliminate redundant field loads in the update kernels | **+7–8%** single-threaded (338 → 365 MC/s); flat at 8 threads (already bandwidth-bound) |
| `586a8ac` | Add the missing AVX2 fast field-extraction path | Full-domain dump every timestep: **336 → 426 MC/s**; removes an AVX2-specific tax SSE never paid |
| `574f714` | Document the MT bandwidth ceiling (comment only) | — |
| `08fa968` | Vectorize the UPML passes | See below |
| `fd555ee` | Calibrate the thread count on real timesteps instead of on 4 s `NextInterval` callbacks | **+50%** on a 400-timestep run (507 → 759 MC/s), **+6.4%** at 20000 timesteps. Host C only; Host A was never hurt by the old search |

**The multithreaded AVX2 engine is memory-bound and already at the host
ceiling.** Measured DRAM traffic at its optimal 4-thread point is 27–38 GB/s
against a ~29 GB/s STREAM ceiling; IPC ~1.06. Thread count peaks at 4 and
*drops* 6% at 8 — SMT siblings add no bandwidth. **Do not pursue compute-side
optimizations for the multithreaded AVX2 engine on this class of host**;
single-threaded and low-thread-count configurations still respond.

> **Amended on Host C.** The bandwidth-bound conclusion survives — the engine
> still tracks the STREAM curve, including its roll-off past 8 threads. The
> clause *"the runtime auto-tune already converges to 3–4 threads unassisted"*
> did not: it was true only because Host A's optimum is near the search's
> starting point. See §4.2.

---

## 2. Tried and rejected — do not retry without new information

- **Non-temporal stores (`_mm256_stream_ps`) in the AVX2 update kernels.**
  Measured *worse* everywhere: −15% single-threaded, −2% at 8 threads. Root
  cause is structural: the per-polarisation stride is 3 `f8vector`s = **96
  bytes**, which does not align to 64-byte cache lines, so write-combining
  buffers cannot accumulate full lines. Revisiting this requires a
  cache-line-aligned field layout first (see candidate A3).
- **Subgroup-uniform coefficient fetch on the GPU.** Looked worth 26% on a
  uniform synthetic mesh but only 2.6% on Coax's graded mesh — the 26% was a
  fixture artifact, not a lever.
- **Shrinking the GPU coefficient tables themselves** (as opposed to the
  index): they are a few KB and already cache-resident.
- **Transferring the AVX2 UPML vectorization findings to the GPU.**
  Investigated and rejected on inspection: the GPU is already coalesced and
  already fuses UPML, and its PML penalty is 2.0× versus the CPU's 4.8×. The
  CPU problem was specifically the `f8vector` z-lane layout, which has no GPU
  analogue.

---

## 3. Open candidates

Ordered by expected value. Each states how to *falsify* it cheaply before
committing to an implementation.

### A1 — AVX2 dispersive extension  ✅ **done (`2bfb77d`)**

Priced, implemented and closed. No dispersive fixture existed, so one was
built: 96³ grid, Drude box (`eps_plasma = 3e10`, order 1) over the central
half of the domain — **12.5% of cells** — against an identical model with a
plain dielectric box.

The extension cost **33–34% of runtime**. The cause was not the arithmetic
but the addressing: the extension reaches the field only through
`GetVolt`/`SetVolt`, and on `Engine_AVX2` those carry an integer div and mod
against the runtime member `numVectors` — **18 divisions and 18 moduli per
cell per timestep** across the four passes. Precomputing the vector offset
and lane per list entry (the list is fixed after operator build) removed all
of them:

| Engine | before | after | |
|---|---|---|---|
| `avx2` | 239.95 MC/s | 286.54 MC/s | **+19.4%** |
| `avx2-multithreaded` | 259.45 MC/s | 311.26 MC/s | **+20.0%** |

Bit-identical on both engines; divergence from the untouched SSE engine is
unchanged at 9.235e-03.

**A2 (sort the cell list for locality) is closed out as not worth doing.**
93.7% of consecutive entries already land within 3 f8vector units — median
delta is exactly 3, i.e. one cell's three polarisations — so the list is
already effectively sequential in engine address space. Sorting would take
that to 99.96%.

**The threading claim is withdrawn.** The extension genuinely has no
threading, but on this grid threading adds little to *either* model (plain
+2.5%, dispersive +8.6% ST→MT), so a serial-section cost is not demonstrated
and would need a larger grid to assess. It was speculation and is not
supported by the measurement.

### A1b — is anything left in the dispersive extension?

After the addressing fix the extension still costs ~22% of runtime on 12.5%
of cells. A traffic estimate suggests this is now close to inherent: per cell
per timestep the ADE passes stream three operator coefficient arrays and
read-modify-write two state arrays across three components (~200 B), against
the Yee update's ~80 B/cell. A dispersive cell is simply ~3× a normal cell in
memory traffic, and those arrays are already dense and contiguous.

**Confirmed — closed.** Stubbing the scattered field read out entirely
(wrong physics, timing only) recovers only 5 of the 74 MC/s gap, **+1.8%**:
277.22 → 282.21 MC/s against a 351.41 MC/s non-dispersive baseline. The
addressing is no longer the cost; ~93% of what remains is dense streaming of
the ADE arrays. Supporting evidence: cost is linear in dispersive volume
above ~10% of cells, and order-2 dispersion costs exactly **2.00×** order-1's
overhead — the loop scales with the number of ADE arrays and nothing else.

This is a traffic floor, matching the two structural results in §1. The only
remaining levers change the physics (fewer ADE arrays, lower precision) and
are out of scope. **The AVX2 side is done.**

### A3 — Cache-line-aligned AVX2 field layout

The 96-byte per-polarisation stride is implicated in the NT-store failure and
also fragments prefetch. Padding 3 `f8vector`s to 4 (128 B) costs 25% more
memory on a bandwidth-bound kernel — plausibly a net loss on its own, and only
interesting *combined with* NT stores.

High risk, wide blast radius (touches every engine and extension indexing
scheme). Listed for completeness; **do not start here.**

### Hardware caveat — historic, and now lifted

> **Superseded on 2026-09-04.** The RX 6800 is on Host C, and §4.1 reproduces
> the headline §1 figure to within 0.2%. The caveat below is kept because G1
> and G2 were written under it: those two were deliberately argued from
> dispatch and barrier *counts* rather than from throughput, and that
> reasoning is still sound. But it is why neither of them measured what the
> minimal pipeline actually costs — which §4.5/C3 now does.

When G1–G3 were written, the RX 6800 that produced every GPU measurement in §1
was not on the machine. The only GPU available was an Intel HD Graphics 530
(Skylake GT2, integrated, sharing the CPU's DDR4, no large last-level cache).
It ran the GPU engine correctly but in a completely different performance
regime — 386 MC/s on a 96³ PEC model, i.e. about the same as the AVX2 CPU
engine on that host, because it was the same memory. Throughput conclusions
drawn there would not have transferred, so the findings below were restricted
to *architecture-independent* facts.

### G1 — Fuse the dispersive passes into the main GPU kernels  ❌ closed

Measured dispatch counts (probe build, batch of 28 timesteps):

| Model | dispatches/timestep | barriers/timestep |
|---|---|---|
| PEC, no dispersion | 3 | 3 |
| PEC + dispersion | 5 | 5 |
| PML + dispersion | 5 | 6 |

Dispersion costs **2 extra dispatches per timestep**, not the four the shader
list suggests, and PML adds no dispatches at all (it is already fused) — only
barriers.

Fusing those two away cannot pay. The dispersive dispatches cover only the
sparse cell list, so they are small; what fusion would save is 2 barriers and
2 launches per timestep. What it would cost is a per-cell membership test in
the main kernel, which runs over the *entire* grid — either a full-size lookup
array (extra traffic on every cell, including the ~87% that are not
dispersive) or a hash. Adding whole-grid traffic to remove two barriers is
the wrong trade on hardware that is already at its bandwidth roofline.

### G2 — Dispatch and barrier count on small grids  ❌ closed

Answered by the same measurement: the per-timestep pipeline is **3 dispatches
and 3 barriers** for a plain model, rising to 5 and 6 with dispersion and PML.
That is already minimal — there is no dispatch bloat to remove. The earlier
suspicion that several active extension families would multiply dispatches is
wrong, because UPML is fused and the rest are sparse-list kernels.

### G3 — `volt_flux` traffic in the PML path  ❌ closed

Structural, not measurable away. A UPML cell's fused voltage update reads and
writes both `volt` and `volt_flux` (3 components each), roughly doubling
per-cell field traffic versus a Yee cell. The flux array is inherent to the
UPML formulation and the flux-swap needs the previous value, so the traffic
cannot be removed without changing the algorithm. After `00ba1c6` made the
coefficients cheap, this *is* the PML cost — it is a floor, not an
inefficiency.

---

## 3b. Status of the candidates above

Every candidate in §3 was either implemented or closed on evidence, and none of
those closures is reopened by the Host C re-verification:

- AVX2 — at the traffic floor. The update kernels (`574f714`), and the
  dispersive extension (§A1b), are both bandwidth-bound with no addressing
  or arithmetic overhead left to remove.
- GPU — the Yee kernel is at roofline (§1), the per-timestep dispatch *count*
  is already minimal (G2), and the two remaining structural costs (PML flux
  traffic, dispersive passes) are inherent to their algorithms (G1, G3).

The one claim that did not survive was not a candidate at all but an aside in
§1 — that the runtime thread auto-tune needs no help. It needed a lot; see
§4.2. **New candidates opened by the Host C measurements are in §4.6.**

---

## 4. Re-verification on Host C (2026-09-04)

Host A (i7-6700K) is gone. Host C is an i9-13900K carrying **both** the RX 6800
from Host B and an integrated UHD 770, so for the first time the GPU and CPU
numbers come from one machine and the two GPUs are directly comparable.

Every number below is a fresh measurement on Host C. Benchmark drivers are
unchanged: `python/Tests/benchmark_avx2_engine.py` and
`python/Tests/benchmark_gpu_engine.py`, 300–400 timesteps, PEC unless stated.

### 4.1 The GPU results reproduce exactly

| Grid | cells | doc value (Host B) | Host C | |
|---|---|---|---|---|
| 256×224×224 | 12.8 M | 5115 MC/s | **5126 MC/s** | reproduced |

That is 0.2% apart on the headline `fbd95a6` figure, which retires any doubt
that §1's GPU table describes this hardware. The roofline argument holds
unchanged: 5126 MC/s × ~80 B/cell ≈ 410 GB/s against a 512 GB/s peak.

Full sweep, both GPUs, same fixture:

| cells | RX 6800 | UHD 770 | AVX2 MT (8 threads) |
|---|---|---|---|
| 64³ = 0.26 M | 9605 MC/s | 497 MC/s | 1851 MC/s |
| 96³ = 0.88 M | 14397 MC/s | 591 MC/s | 3449 MC/s |
| 128³ = 2.1 M | 16165 MC/s | 575 MC/s | 1881 MC/s |
| 160×128×192 = 3.9 M | 16888 MC/s | 510 MC/s | 887 MC/s |
| 224³ = 11.2 M | 5284 MC/s | 506 MC/s | 619 MC/s |
| 256×224×224 = 12.8 M | 5181 MC/s | 506 MC/s | — |
| 320×288×288 = 26.5 M | 5368 MC/s | 501 MC/s | — |

Three things to read off it:

- **The Infinity Cache cliff is between 3.9 M and 11.2 M cells**, exactly where
  the ~24 B/cell of field state crosses 128 MB. Above it the GPU sits flat at
  ~5.2 GC/s out to 26.5 M cells — the roofline regime, and stable there.
- **The UHD 770 is flat at ~0.5 GC/s regardless of grid size**, i.e. it is
  bounded by the same DDR5 the CPU uses and is beaten by the AVX2 engine on
  every grid up to 3.9 M cells. It is useful as a portability check on a
  second Vulkan driver (ANV rather than RADV) and for nothing else. Device
  selection already prefers the discrete GPU, which was confirmed here.
- **PML costs 15–20% on the RX 6800**: 4334 vs 5126 MC/s at 256×224×224, 9718
  vs 16888 MC/s at 160×128×192. Consistent with the 2.0× per-PML-cell figure
  in §2 once the PML's share of the volume is accounted for.

### 4.2 The AVX2 thread auto-tune was badly wrong on a many-core host

This is the one place the document was actively misleading. §1 asserted that
"the runtime auto-tune already converges to 3–4 threads unassisted." On Host A
that was true. On Host C the same code cost **43%** of throughput.

The old search hill-climbed inside `NextInterval()`, which `openems.cpp` calls
only when four seconds of wall time have elapsed, adding one thread per call
from a start of one. Reaching Host A's optimum of 3–4 takes a few intervals.
Reaching Host C's takes 44 s, during which the run is nowhere near peak:

| run length | old auto-tune | hand-pinned best | new calibration |
|---|---|---|---|
| 400 TS (~8 s) | 507 MC/s | 884 MC/s | **759 MC/s** |
| 20000 TS (~90 s) | 843 MC/s | 875 MC/s (8 thr) | **896 MC/s** |

`fd555ee` replaces it with a search that scores candidates on real timesteps
inside `IterateTS()` — legitimate because thread count cannot change results —
walking a geometric ladder and then bisecting around the winner. It settles in
about 250 timesteps rather than 44 s. The rewrite is worth roughly nothing on
Host A and 1.5× on Host C, which is the whole point.

One implementation detail that mattered more than expected: the first batch
after each thread respawn must be discarded. Fresh threads start with cold
private caches, which penalises precisely the wide configurations under
judgement; without the warm-up the search chose anywhere between 4 and 12
threads run to run, and with it, 8 every time.

### 4.3 Thread scaling, and why it rolls off

| threads | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 20 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| MC/s | 486 | 706 | 814 | 865 | 884 | 884 | 707 | 673 | 677 | 702 |

Single-threaded `avx2` (not `avx2-multithreaded` at width 1) reaches
**535 MC/s**, up from Host A's 365.

The 20% collapse past 12 threads is *not* an engine defect and not a P-core /
E-core scheduling artifact. A plain OpenMP STREAM-Triad on the same host has
the same shape:

| threads | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| GB/s | 28.8 | 33.2 | 44.8 | 48.2 | **48.9** | 45.0 | 41.8 | 39.3 | 39.3 |

Both peak at 6–8 and both lose ~20% by 16. The memory controller, not the
engine, sets this curve. There is nothing to fix — only a search that must not
walk into the far side of it, which §4.2 now provides.

### 4.4 How much a 36 MB L3 is worth

At a fixed 8 threads, throughput against grid size:

| cells | 0.26 M | 0.88 M | 2.1 M | 3.9 M | 4.1 M | 11.2 M |
|---|---|---|---|---|---|---|
| MC/s | 1851 | **3449** | 1881 | 887 | 862 | 619 |

A 96³ grid runs **3.9× faster per cell** than a 160×128×192 one. The field
state at 96³ is a few tens of MB and lives in the 36 MB L3; past that
everything streams from DDR5. Host A's 8 MB L3 could not show this — its
in-cache regime was too small to be interesting — which is why §1 concluded
"memory-bound" full stop and stopped there.

Read this as an upper bound on what removing DRAM traffic would buy, not as
evidence that traffic is currently being wasted. §4.5 measures the traffic and
shows it is not: the out-of-cache regime really is at the DRAM ceiling, so the
3.9× is only reachable by eliminating traffic (§4.6/C2), not by rearranging
it (§4.6/C1).

### 4.5 Measured DRAM traffic — §1's 80 B/cell was too high, its conclusion was not

§1 and §A1b both rest on an assumed ~80 B/cell/timestep that had never been
measured. `perf_event_paranoid` is 2 here so the `uncore_imc` CAS counters are
out of reach, but the *core* PMU is not, and `longest_lat_cache.miss`
differenced between two run lengths isolates the stepping phase cleanly
(500 vs 2500 timesteps, 8 threads pinned to the P-cores, 160×128×192):

| | 500 TS | 2500 TS | difference |
|---|---|---|---|
| L3 misses | 1.041e9 | 4.902e9 | 3.861e9 over 2000 TS |
| cell-timesteps | | | 7.864e9 |

That is **0.491 L3 misses per cell-timestep = 31.4 B/cell of DRAM fills.** The
ideal is 24 B/cell — read each cell's `volt` and `curr` once — so the sweep is
already within 31% of perfect read locality, with the neighbour accesses
absorbed by cache as intended. Add the ~24 B/cell of dirty writeback that a
94 MB working set must eventually push to DRAM and the total is **~55 B/cell**,
which at 873 MC/s is **~48 GB/s against the 49 GB/s STREAM ceiling.**

So: the constant in §1 was wrong by ~45%, and the conclusion it was used to
support is right anyway. The multithreaded AVX2 engine is at the DRAM roofline
on Host C, not merely near it.

### 4.6 Open candidates (new, from Host C)

#### C1 — Spatial blocking of the AVX2 sweep  ❌ closed on measurement

This was the obvious reading of §4.4: a 3.9× in-cache/out-of-cache gap looks
like scattered access waiting to be tiled. §4.5 closes it. Read traffic is
already 31.4 B/cell against a 24 B/cell floor — there is no scattered-access
waste to recover, because the x-line decomposition and the y–z plane sweep
already give the stencil the locality it needs. Tiling would rearrange traffic
that is not being wasted.

The 3.9× is simply what disappearing off DRAM buys, and reaching it needs
traffic *elimination*, not rearrangement. That is C2.

#### C2 — Temporal blocking of the AVX2 sweep

The only lever left on the CPU side. A wavefront or trapezoidal scheme that
advances a cache-resident tile through *k* consecutive timesteps before moving
on divides DRAM traffic by roughly *k*. §4.4 bounds the prize: 96³ runs at
3449 MC/s with DRAM out of the picture against 887 MC/s streaming, so a
successful k=4 blocking would be worth somewhere between 2× and 3.5×.

This is by far the largest single number anywhere in this document, and also by
far the most invasive change in it. Every engine extension is written against
a per-thread contract of "you own this contiguous x-range for this one
timestep", and temporal blocking breaks that contract for all of them.

**How to falsify cheaply, before touching the engine:** write a standalone
3-point-stencil microbenchmark on a 94 MB array with the same 8-thread
partitioning, once flat and once with a k=4 trapezoidal schedule, and measure
whether the DRAM traffic actually falls by ~4× using the same
`longest_lat_cache.miss` differencing as §4.5. If it does not — if the halo
re-reads eat the saving at realistic tile sizes — the engine work cannot pay
either, and this closes for the same reason C1 did.

#### C3 — The GPU has a ~12.6 µs fixed cost per timestep

New, and it does not contradict G2. G2 established that the per-timestep
dispatch *count* is minimal (3 dispatches, 3 barriers for a plain model). It
never measured what that fixed pipeline *costs*. Fitting the RX 6800 sweep in
§4.1 as `time/TS = cells / rate + overhead` gives a marginal rate of
**17.9 GC/s** and an intercept of **12.6 µs**, with the marginal rate
consistent to within 2% across all three intervals — so the fit is real, not an
artifact of two endpoints.

That overhead is 46% of a 64³ timestep, 20% of a 96³ one, and 5% at
160×128×192. It is *not* submission latency: `IterateTS()` already batches many
timesteps into one command buffer and splits only at `m_maxTSPerSubmit`. It is
in-command-buffer dispatch and full-grid barrier drain, ~4 µs apiece.

**How to falsify cheaply:** record a command buffer with the same three
dispatches over a 64³ grid but with the two intermediate barriers removed
(wrong physics, timing only). If the timestep does not drop by roughly 8 µs the
cost is dispatch launch rather than barrier drain, and there is nothing to
fuse. Note that G1 already closed *dispersive* fusion on a traffic argument;
this is the plain three-pass pipeline on small grids, where the trade goes the
other way because there is so little work per pass.

Practical caveat: even at 64³ the RX 6800 is 5× faster than the AVX2 engine, so
this is a real inefficiency but not a competitive one.

#### C4 — Is the GPU Yee kernel actually at roofline?

§1 says yes, on the strength of 5126 MC/s × 80 B/cell ≈ 410 GB/s against a
512 GB/s peak. §4.5 has just shown that 80 B/cell is too high by ~45% on the
CPU, and the GPU stores the same `volt` and `curr` state. At the measured
~55 B/cell the same throughput is **282 GB/s, i.e. 55% of peak** — which is not
a roofline, it is a kernel with headroom.

The counter-evidence is real too: §4.1 shows throughput flat at 5.2–5.4 GC/s
from 11.2 M to 26.5 M cells, and flatness under growing working set is exactly
what a bandwidth limit looks like. So one of the two readings is wrong and the
arithmetic cannot settle it.

**How to falsify cheaply:** RADV exposes memory counters — `RADV_PERFTEST`
plus a GPU profiler, or simply `radeontop`'s memory-controller utilisation
sampled during a 256×224×224 run. If it sits near 100%, §1 is right by luck and
C4 closes. If it sits near 55%, the flatness has another cause (occupancy, or
the ~12.6 µs of C3 scaling with dispatch size) and the Yee kernel has room —
which would reopen a large part of §1's "no headroom left" conclusion.

Do this before any further GPU kernel work. It is one measurement and it
decides whether §1's central structural claim stands.


---

## 5. Method notes

Hard-won test-harness lessons; all three have produced false results in this
repo.

- **RPATH is absolute.** Copying a build tree does not give an independent
  binary — the copy still loads the original's `libopenEMS.so`. A/B builds
  need separately *configured* build directories.
- **`LD_LIBRARY_PATH` set from inside Python has no effect** on the running
  process's linker. The Cython extension resolves `libopenEMS.so` through its
  baked-in RUNPATH. Re-exec (see `_bootstrap_local_openems_runtime` in
  `test_gpu_engine.py`) or drive the `openEMS` CLI on a `Write2XML()` file.
- **The Python module links the *installed* library**, so `--engine gpu` is
  unavailable through it unless the installed build has `WITH_GPU=ON`.
- **Degenerate meshes make A/B tests pass vacuously.** Grading with `t**1.7`
  from zero produces near-zero first cells, collapsing the timestep so nothing
  propagates and every field dump is identically zero. Use geometric grading
  and assert a non-zero peak.
- **`uncore_imc` is unavailable, the core PMU is not.** `perf_event_paranoid`
  is 2 on Host C and there is no passwordless sudo, so the memory-controller
  CAS counters that would give DRAM traffic directly cannot be read. The core
  PMU *does* work unprivileged for one's own process:
  `longest_lat_cache.miss` is the usable substitute (§4.5). On this hybrid CPU
  perf splits every event into `cpu_core/` and `cpu_atom/` variants — pin with
  `taskset -c 0-15` and count only `cpu_core/`, otherwise the atom counter is
  multiplexed down to a few percent enable time and silently scaled back up.
- **Difference two run lengths instead of trusting one.** A whole-process
  counter includes operator build, mesh setup and HDF5 writes, which on short
  runs are a large fraction of the total. Running 500 and 2500 timesteps and
  dividing the difference by 2000 isolates the stepping phase without needing
  to instrument anything.
- **Guard STREAM-style microbenchmarks against dead-code elimination.** The
  first cut of the bandwidth benchmark in §4.3 reported 14 *million* GB/s,
  because nothing read the output array and GCC deleted the loop. Consume the
  result and print it.
- **Cross-engine comparison is structurally blind to operator bugs.** Every
  engine routes through the same `Calc_ECOperatorPos`, so an indexing error
  there is invisible to all cross-engine tests (this is how the `81005f8`
  regression survived). Absolute-value tests on inhomogeneous models are the
  only guard.
