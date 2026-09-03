# openEMS performance work — GPU and AVX2 engines

Running record of optimization work on the Vulkan GPU engine and the AVX2
engines: what was done, what it was worth, and — just as important — what was
tried and rejected, so it is not rediscovered blind.

Scope of this document is deliberately narrow: `Engine_Vulkan` /
`Operator_Vulkan` and `Engine_AVX2` / `Engine_AVX2_Multithread`. The SSE and
basic engines are out of scope.

Measurements come from two hosts and are not comparable across them:

- **CPU host** — i7-6700K, 4 cores / 8 threads (SMT2), dual-channel DDR4.
  Practical STREAM-Triad ceiling ~29 GB/s.
- **GPU host** — Radeon RX 6800, 512 GB/s peak, 128 MB Infinity Cache,
  **256 MB BAR** (no resizable BAR).

Benchmark drivers: `python/Tests/benchmark_avx2_engine.py`,
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

**The multithreaded AVX2 engine is memory-bound and already at the host
ceiling.** Measured DRAM traffic at its optimal 4-thread point is 27–38 GB/s
against a ~29 GB/s STREAM ceiling; IPC ~1.06. Thread count peaks at 4 and
*drops* 6% at 8 — SMT siblings add no bandwidth. The runtime auto-tune
already converges to 3–4 threads unassisted. **Do not pursue compute-side
optimizations for the multithreaded AVX2 engine on this class of host**;
single-threaded and low-thread-count configurations still respond.

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

### Hardware caveat — read before trusting any GPU number below

The RX 6800 that produced every GPU measurement in §1 **is not on the current
machine.** The only GPU here is an Intel HD Graphics 530 (Skylake GT2,
integrated, sharing the same DDR4 as the CPU, no large last-level cache). It
runs the GPU engine correctly but in a completely different performance
regime — it reaches 386 MC/s on a 96³ PEC model, i.e. *about the same as the
AVX2 CPU engine on the same host*, because it is the same memory.

Throughput conclusions measured here do not transfer to the discrete-GPU
regime the GPU engine was tuned for. The findings below are therefore
restricted to *architecture-independent* facts — dispatch and barrier counts,
and structural arguments — which do transfer.

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

## 3b. Status: no open candidates

Every candidate raised has been either implemented or closed on evidence:

- AVX2 — at the traffic floor. The update kernels (`574f714`), and now the
  dispersive extension (§A1b), are both bandwidth-bound with no addressing
  or arithmetic overhead left to remove.
- GPU — the Yee kernel is at roofline (§1), the dispatch pipeline is already
  minimal (G2), and the two remaining structural costs (PML flux traffic,
  dispersive passes) are inherent to their algorithms (G1, G3).

Further GPU work would need the discrete GPU to evaluate. Optimizing against
the integrated GPU available here would repeat the failure already recorded in
§2 — the "26% subgroup-uniform" win that turned out to be an artifact of an
unrepresentative fixture.

**Restarting this work is worthwhile when:** the RX 6800 host is available
again, a profile of a real production model (rather than synthetic fixtures)
points somewhere specific, or a new engine feature adds a hot path.

## 4. Method notes

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
- **Cross-engine comparison is structurally blind to operator bugs.** Every
  engine routes through the same `Calc_ECOperatorPos`, so an indexing error
  there is invisible to all cross-engine tests (this is how the `81005f8`
  regression survived). Absolute-value tests on inhomogeneous models are the
  only guard.
