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
`python/Tests/bench_temporal_blocking.c` (standalone, see its header),
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
| `05ce5f0` | Stop allocating the async dump ring when the model has no dump boxes | **2.8× less GPU memory** on dump-free runs (224³: 803 → 288 MB VRAM, 657 → 142 MB host). Throughput unchanged — the buffers were never used |
| `b87a1f7` | Give each TF/SF field index to one thread instead of racing on `+=` | Correctness, not speed: the Vulkan engine was **nondeterministic** on any TF/SF model. GPU–CPU H-field disagreement 5.62e-06 → 4.64e-07 |

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
| *(prototype, opt-in)* | Trapezoidal temporal blocking of the update sweep, `OPENEMS_AVX2_TEMPORAL_BLOCK=<k>` | **3.2×–3.6×** on grids that exceed L3 (873 → 2805, 613 → 2171, 610 → 2194 MC/s). Bit-identical to the flat sweep. Engages only for the plain update path plus the excitation — see §4.7/C2 |

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
> reasoning is still sound. §4.7/C3 has since measured what the minimal
> pipeline actually costs on the real hardware — nothing — so the
> count-based argument turns out to have been conservative rather than
> merely careful.

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

Both GPU closures were re-tested directly on Host C rather than taken on
trust, and both hold: the Yee kernel runs at 89% of a *measured* bandwidth
ceiling (§4.7/C4), and an extra dispatch plus barrier is free to within noise,
so dispatch count is not merely minimal but not a lever at all (§4.7/C3).

The one claim that did not survive was not a candidate at all but an aside in
§1 — that the runtime thread auto-tune needs no help. It needed a lot; see
§4.2. **New candidates opened by the Host C measurements are in §4.7. All are
now resolved; the one that pays is C2 — temporal blocking of the AVX2 sweep,
validated at 3–4× with a bit-identical reference schedule, and awaiting
its engine port.**

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
3.9× is only reachable by eliminating traffic (§4.7/C2), not by rearranging
it (§4.7/C1).

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

### 4.6 GPU memory footprint — 2.8× more than the model needs

Not a throughput finding, but it sets the largest grid the card can hold, which
is the GPU's hard limit. Sampling `mem_info_vram_used` / `mem_info_gtt_used`
through a run with no dump boxes at all:

| grid | field buffer | VRAM before | VRAM after | host before | host after |
|---|---|---|---|---|---|
| 224³ | 128 MB | 803 MB | **288 MB** | 657 MB | **142 MB** |
| 320×288×288 | 303 MB | 1869 MB | **653 MB** | 1533 MB | **318 MB** |

The gap was exactly `DUMP_RING × 2 × m_fieldBufSize` in each of VRAM and host
memory — the async field-download ring from `ea82f59`, allocated for
simulations that never download a field. `RunFDTD()` decides whether to arm
that pipeline by scanning the `ProcessingArray` for a `ProcessFields`, and the
scan also matched `ProcField`, the bare `ProcessFields` registered a hundred
lines earlier purely to give the energy estimate a processing cadence. Fixed in
`05ce5f0`.

Two details worth keeping:

- **It only bit grids above the BAR aperture.** Below it the field buffers are
  host-mapped directly and `EnsureDumpBuffers()` bails out at its
  `m_voltMapped && m_currMapped` check. So the waste scaled precisely with the
  models where VRAM is the binding constraint, and never showed up on the
  small fixtures.
- **Throughput was untouched** (5633 vs 5663 MC/s at 224³). The buffers were
  allocated and then sat idle, which is why nothing in §1's timing work ever
  pointed at it. Footprint needs its own measurement; speed benchmarks will
  not find this class of bug.

### 4.7 Open candidates (new, from Host C)

#### C1 — Spatial blocking of the AVX2 sweep  ❌ closed on measurement

This was the obvious reading of §4.4: a 3.9× in-cache/out-of-cache gap looks
like scattered access waiting to be tiled. §4.5 closes it. Read traffic is
already 31.4 B/cell against a 24 B/cell floor — there is no scattered-access
waste to recover, because the x-line decomposition and the y–z plane sweep
already give the stencil the locality it needs. Tiling would rearrange traffic
that is not being wasted.

The 3.9× is simply what disappearing off DRAM buys, and reaching it needs
traffic *elimination*, not rearrangement. That is C2.

#### C2 — Temporal blocking of the AVX2 sweep  ✅ **validated, and prototyped in the engine at 3.2–3.6×**

The only lever left on the CPU side, and the largest number in this document.
A trapezoidal scheme that advances a cache-resident tile through *k*
consecutive timesteps before moving on divides DRAM traffic by roughly *k*.
§4.4 bounded the prize at 2×–3.5×; measurement beats the bound.

**The vehicle.** `python/Tests/bench_temporal_blocking.c` reproduces the
*access pattern* of `Engine_AVX2_Multithread` — two arrays of three components
in N-I-J-K layout with z contiguous, the E pass reading H at
(x,y,z),(x−1,y,z),(x,y−1,z),(x,y,z−1) and the H pass the mirror image — but not
its physics. Values are meaningless; only the loads, stores and their order
matter, and those set cache behaviour. Two things say it is faithful: its flat
baseline is **920 MC/s** against the engine's 887, and its DRAM traffic is
**31.0 B/cell** against the engine's measured 31.4 (§4.5). Both within ~4%.

**Traffic — the falsification criterion asked for ~4×:**

| schedule | B/cell/timestep | |
|---|---|---|
| flat (what the engine does today) | 30.95 | |
| trapezoidal, W=72, k=24 | **2.40** | **12.9× less** |

**Throughput, 8 threads, best (W,k) per grid:**

| grid | field state | flat | blocked | | best (W,k) |
|---|---|---|---|---|---|
| 96³ | 21 MB | 2915 MC/s | — | **skip: already fits L3** | — |
| 160×128×192 | 94 MB | 864 MC/s | **3675 MC/s** | 4.25× | (72,24) |
| 224³ | 270 MB | 652 MC/s | **2270 MC/s** | 3.48× | (32,8) |
| 320×288×288 | 637 MB | 586 MC/s | **1679 MC/s** | 2.86× | (18,8) |

It helps at every thread count, and it removes the roll-off past 8 threads that
§4.3 traced to the memory controller — once the sweep is not DRAM-bound, extra
threads stop fighting each other:

| threads | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| flat | 562 | 753 | 846 | 925 | 756 | 687 |
| trapezoidal (72,24) | 824 | 1402 | 2410 | **3642** | 2866 | 2812 |
| gain | 1.47× | 1.86× | 2.85× | 3.94× | 3.79× | 4.09× |

**The schedule is verified, not just plausible.** `bench_tb verify` runs the
trapezoid and the flat sweep from the same pseudorandom initial state and
compares every element: **bit-identical** across W ∈ {24,26,32,72},
k ∈ {4,8,12,24}, threads ∈ {1,4,8}, and tile widths that do not divide the
domain. That property is what the engine port has to preserve, and the
benchmark is the reference implementation for it.

Two geometry details cost real debugging time and are worth carrying over:

- **Domain edges must be held, not sloped.** The trapezoid narrows because data
  outside the tile is at the wrong time; at x=0 and x=NX−1 there is no outside
  (the update clamps), so sloping there leaves edge cells un-advanced.
- **The wedge's E range is not symmetric with the core's.** The two passes must
  partition the domain — every cell updated exactly once per timestep. A
  narrowing core leaves gaps of width 2t+1 in E but 2t+2 in H, so the widening
  wedge sweeps E on `[An+1, Bn)`, one cell narrower per side than the
  natural-looking `[An, Bn+1)`. The symmetric choice double-updates those cells
  and was the first version's bug.

**Guards any implementation needs, both measured:**

- **Turn it off when the grid already fits L3.** At 96³ there is no traffic to
  remove; the overlapped-tiling prototype lost 32% there. The crossover is
  where field state (24 B/cell) exceeds ~36 MB.
- **Derive the tile width from the cross-section.** What must fit cache is
  `W × Ny × Nz × 24 B`; the optimum lands at L3/4–L3/2, so a fixed W is wrong.
  W=32 is right at 224³ and 27% off at 320×288×288, where W=18 wins. `k` also
  has an optimum (k=24 at 160×128×192, k=32 already worse), and `W ≥ 2k` is a
  hard constraint of the geometry.

**The engine prototype.** `Engine_AVX2_Multithread` now carries this schedule,
opt-in via `OPENEMS_AVX2_TEMPORAL_BLOCK=<k>` (or `<k>:<W>` to set the tile
width by hand). Measured on the real engine, 8 threads, 1200 timesteps:

| grid | flat | blocked | |
|---|---|---|---|
| 160×128×192 | 873 MC/s | **2805 MC/s** | 3.21× |
| 224³ | 613 MC/s | **2171 MC/s** | 3.54× |
| 256×224×224 | 610 MC/s | **2194 MC/s** | 3.59× |

Short of the microbenchmark's 4.25×, as expected: the real kernel also streams
a compressed operator index and is therefore not quite as purely
bandwidth-bound as the proxy.

**Bit-identical to the flat engine**, checked by dumping fields on a full-x
slab (the axis the trapezoid tiles, so a boundary error shows there) and
comparing every element: identical across grids 160×128×192, 224³, 256×224×224
and a deliberately non-round 173×131×197; k ∈ {6,8,16}; 1, 4, 8 and 12 threads;
and both auto-derived and hand-set tile widths.

**The tile width is derived from the last-level cache**, not fixed. One tile of
both field arrays should be about one L3: the measured optimum was W=64 on a
grid with 0.56 MB x-planes and W=32 on one with 1.20 MB — 36 and 38 MB against
this host's 36 MB. Getting this wrong is expensive rather than fatal: at 224³ a
tile sized for the *other* grid (W=64, 77 MB) gives 1267 MC/s where W=32 gives
2249.

**It refuses rather than risks.** Blocking engages only when every active
engine extension reports `SupportsSlabApply()`, which today means the
excitation and nothing else — so UPML, dispersive materials, Mur, TF/SF and
lumped elements all fall back to the flat sweep automatically, and a new
extension is safe by omission. It also declines when the grid already fits
cache (measured a 32% loss there) or when there are too few x-lines to tile.
Verified: a PML run reports *"disabled, extension 'Uniaxial PML Extension'
cannot be applied per x-range"*, and a 96³ run reports *"disabled, grid fits
cache (20.25 MB of field state)"*.

**What the prototype does not yet do**, and what full adoption needs:

- **Only the excitation is slab-capable.** UPML and the dispersive extension
  are per-cell and could follow the trapezoid with the same
  `Apply2VoltagesSlab` interface; probes and field dumps need globally coherent
  state, so blocks would have to align to the processing horizon. Until then
  the models that benefit are PEC/homogeneous ones — which is most of the
  benchmark suite but not most real simulations.
- **Thread calibration now runs on whichever schedule the run will use.** It
  had been calibrating on the flat path, which is the wrong constraint:
  §4.3's roll-off past 8 threads is DRAM contention, and blocking removes it.
  Calibrating blocked, the search picks **16 threads instead of 8** and auto
  thread selection reaches 2412 MC/s against the flat path's 876 — **2.75×**
  end to end, up from 2.44×. Still short of the 3.21× a hand-pinned thread
  count gets, because the calibration timesteps are themselves spent
  measuring.
- **It is opt-in and prints what it decided.** Nothing changes unless the
  environment variable is set.


#### C3 — A fixed per-timestep GPU cost  ❌ closed, the candidate was an artifact

**Withdrawn.** The claim was a marginal rate of 17.9 GC/s and a 12.6 µs
intercept, fitted from the §4.1 sweep. Both the fit and the data were bad.

The data: that sweep used 300-timestep runs, short enough that startup is a
visible share of the total. Re-measured at 4000 timesteps the 64³ point moves
from 9605 to 13380 MC/s — a 39% difference from run length alone.

The fit: a single line was fitted across what turn out to be **three** distinct
cache regimes, so the "consistent marginal rate" was an average over regimes
that do not share one. Clean sweep, 4000 timesteps, GPU timestamps:

| grid | cells | µs/TS | throughput | marginal rate vs previous |
|---|---|---|---|---|
| 48³ | 0.11 M | 12.19 | 9.1 GC/s | — |
| 64³ | 0.26 M | 19.59 | 13.4 GC/s | 20.5 GC/s |
| 80³ | 0.51 M | 32.65 | 15.7 GC/s | 19.1 GC/s |
| 96³ | 0.88 M | 60.36 | 14.7 GC/s | 13.4 GC/s |
| 128³ | 2.1 M | 162.89 | 12.9 GC/s | 11.8 GC/s |
| 160×128×192 | 3.9 M | 309.71 | 12.7 GC/s | 12.5 GC/s |

The marginal rate falls from 20.5 to 12.5 GC/s — it is not one line. Below
~0.5 M cells the 24 B/cell of field state fits the 4 MB L2; from there to
~4 M it is Infinity-Cache-resident; past ~5 M it is DRAM (§4.1). Each regime is
bandwidth-limited by its own cache level. Extrapolating a line through all
three back to zero produces an "intercept" that is an artifact of the regime
change, and fitting only the Infinity-Cache points gives an intercept of
**−10 µs** — small grids are faster than the asymptote, not slower.

**Measured directly instead.** An extra pipeline barrier plus an extra dispatch
were recorded into the per-timestep command buffer, using the excitation
pipeline with `excCount=0` so every invocation returns immediately and writes
nothing. Field evolution is bit-identical to the baseline, so the delta is the
pure marginal cost of a dispatch boundary:

| grid | baseline | +1 dispatch +1 barrier | delta |
|---|---|---|---|
| 64³ | 19.53 µs/TS | 19.51 µs/TS | −0.1% |
| 96³ | 60.58 µs/TS | 59.73 µs/TS | −1.4% |
| 160×128×192 | 311.73 µs/TS | 317.37 µs/TS | +1.8% |

A whole extra dispatch and barrier is **free to within run-to-run noise**.
There is no fixed per-timestep overhead to remove at any grid size, and G2's
conclusion is strengthened rather than qualified: not only is the dispatch
count minimal, dispatch count is not a lever on this hardware at all.

#### C4 — Is the GPU Yee kernel at roofline?  ✅ closed, yes

§1 is right; the doubt raised here was mine and it was wrong. It came from
importing the CPU's measured ~55 B/cell (§4.5) to the GPU, where it does not
apply: the CPU's two passes share a 36 MB L3 and get cross-pass reuse, the
GPU's do not at these grid sizes.

Counting the traffic from `update_voltages.comp` instead of assuming it — one
thread per cell, all three components:

| per cell, per pass | bytes |
|---|---|
| read `curr` (3 components; the six neighbour taps hit cache) | 12 |
| read `volt` (read-modify-write) | 12 |
| write `volt` | 12 |
| `opIdx` (8-bit specialization constant on this fixture) | 1 |
| compressed coefficients (6 unique sets, L2-resident) | ~0 |
| **per pass** | **37** |

The current pass is symmetric, so **74 B/cell/timestep**. Against a measured
ceiling — `clpeak --bandwidth` on this RX 6800 reports **477 GB/s** global
memory (float4; 459 GB/s for float) — the out-of-cache figure is

    5729 MC/s × 74 B = 424 GB/s = **89% of achievable bandwidth**

which is at roofline for a memory-bound kernel. Independent corroboration: the
comment in `update_voltages.comp` records that a 32-bit `opIdx` was measured at
"~10% of all memory traffic", and 2 passes × 4 B = 8 B against ~80 B/cell is
exactly 10%.

So §1's 80 B/cell was right *for the GPU* and wrong *for the CPU* — the
document used one constant for two engines whose cache behaviour differs. §4.5
stands for the CPU, §1 stands for the GPU, and they are not in conflict.

Corollary: `clpeak` also reports **48.7 GB/s** for the UHD 770, against the
CPU's 49 GB/s STREAM-Triad ceiling (§4.3). The iGPU and the CPU are the same
memory system to within measurement error, which is why §4.1's iGPU column is
flat at ~0.5 GC/s regardless of grid size.


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
- **An intermittently failing tolerance check is a race until proven
  otherwise.** `test_tfsf_field_probes` had been failing about one run in
  three, which reads like a tolerance that wants loosening. It was not: six
  runs of the same TF/SF model gave six different GPU results, while every CPU
  engine gave one identical result across six runs each. Hashing whole probe
  traces across repeated runs of *one* engine — rather than comparing two
  engines to each other — is what separates "these two disagree" from "this one
  disagrees with itself", and it took three commands. The bug was a non-atomic
  `+=` on field elements shared by two TF/SF box faces (`b87a1f7`).

  The rest of the engine was then audited the same way rather than by
  inspection. Only three shaders accumulate into shared field memory at all
  (`apply_excitation`, `tfsf_voltage`, `tfsf_current`), and the excitation's
  index list has no duplicates. Repeating the hash-across-runs check on a plain
  model and on the dispersive, lumped-RLC, conducting-sheet and local-ABC paths
  found all of them reproducible, so TF/SF was the only one.
- **A GPU A/B that silences the source measures the source, not the change.**
  This one cost most of an afternoon and nearly produced a fake 26% win. The
  first C3 experiment removed the excitation dispatch and measured
  318 → 244 µs/TS, seemingly proving that one dispatch boundary was worth 26%.
  It proved nothing of the sort: with no excitation the fields stay *exactly
  zero*, and an all-zero grid runs ~28% faster than the identical kernel on
  real data. Controls, all on the same 160×128×192 fixture at 4000 timesteps:

  | configuration | µs/TS |
  |---|---|
  | dispatch present, real fields | 318.4 |
  | dispatch present, **amplitudes zeroed** | 241.3 |
  | dispatch absent, fields zero | 235.7 |

  The dispatch is worth 5.6 µs; being zero is worth 77 µs. Ruled out as
  explanations: iteration count (both ran exactly 4000), clock and power
  (2174 vs 2205 MHz, 202 vs 201 W sampled through the run), excitation
  waveform (gaussian and sinusoidal agree to 1%), excitation *size* (19,140
  points vs 110 changes nothing), and denormals — scaling all amplitudes by
  1e10, which pushes every value out of denormal range, leaves the time
  unchanged at 318 µs. What is left is datapath switching activity: zero
  operands flip almost no bits, so the same instruction stream costs far less
  energy and sustains more throughput inside the same power envelope.

  **Rule: never A/B a GPU change by disabling the excitation, and check that
  both arms carry real field data.** To isolate a dispatch's cost, add a no-op
  dispatch (the excitation pipeline with `excCount=0`) to the arm that lacks
  one, so field evolution stays bit-identical on both sides.
- **Fit within one cache regime or not at all.** The withdrawn C3 intercept
  came from a straight line through L2-resident, Infinity-Cache-resident and
  DRAM-resident grids at once. Check where the working set (24 B/cell) sits
  relative to L2 (4 MB), Infinity Cache (128 MB) and VRAM before fitting
  anything, and use run lengths long enough that startup is negligible —
  300-timestep runs put the 64³ point 39% off its 4000-timestep value.
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
- **Speed benchmarks do not find footprint bugs.** §4.6 was 515 MB of VRAM
  allocated and never touched, invisible to every timing measurement in this
  document. Sample `mem_info_vram_used` and `mem_info_gtt_used` under
  `/sys/class/drm/card*/device` through a whole run and compare against
  `3 * N * sizeof(float)` per field buffer; anything unexplained is real.
  Sample continuously rather than at one fixed delay — allocation only reaches
  steady state after `CalcECOperator`, which takes 16 s on a 26 M-cell grid.
- **Cross-engine comparison is structurally blind to operator bugs.** Every
  engine routes through the same `Calc_ECOperatorPos`, so an indexing error
  there is invisible to all cross-engine tests (this is how the `81005f8`
  regression survived). Absolute-value tests on inhomogeneous models are the
  only guard.
