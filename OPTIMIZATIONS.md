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

### A1 — AVX2 path for the dispersive (Lorentz/Drude) extension  ★ highest

`Engine_Ext_LorentzMaterial::DoPreVoltageUpdatesImpl` /
`DoPreCurrentUpdatesImpl` are scalar loops over a sparse cell list. UPML is
currently **the only** extension with a vectorized path
(`engine_ext_upml_avx2.cpp`); every other extension runs scalar on the AVX2
engines.

Structurally this is favourable: the ADE state arrays (`volt_ADE`,
`volt_Lor_ADE`, `v_int_ADE`, `v_ext_ADE`, `v_Lor_ADE`) are all indexed by a
dense contiguous `i`, so six of the eight operands vectorize directly. Only
the field term `eng->GetVolt(n, pos[0][i], pos[1][i], pos[2][i])` is a
gather — and on `Engine_AVX2` that call carries a runtime `%`/`/` against
`numVectors` per element, the same cost `586a8ac` removed from the field
extraction path.

Affects every model with frequency-dependent materials, which includes the
bio/MRI phantom fixtures this repo ships.

**Priced — confirmed worth pursuing.** No dispersive fixture existed, so one
was built: 96³ grid, a Drude box (`eps_plasma = 3e10`, order 1) spanning the
central half of the domain, i.e. **12.5% of cells**, against an identical
model with a plain dielectric box of the same geometry and `epsilon`/`kappa`.

| Engine | plain | Drude | cost |
|---|---|---|---|
| `avx2` | 360.2 MC/s | 241.9 MC/s | **−33%** |
| `avx2-multithreaded` | 386.9 MC/s | 256.6 MC/s | **−34%** |

A third of total runtime spent on an eighth of the cells — a per-cell cost
comparable to the 4.8× UPML penalty that motivated `08fa968`. Far above the
3% threshold that would have closed this out.

Second finding from the same measurement: the extension contains **no
threading at all** — no tiling, no thread partitioning, no engine thread
awareness. On `avx2-multithreaded` it is a serial section between parallel
Yee updates, so it is an Amdahl bottleneck whose relative cost *grows* with
thread count. Vectorization and threading are separable pieces of work here;
threading may well be the larger of the two.

Model generator kept at `scratchpad/disp/gen.py` — should be promoted into
`python/Tests/` as a fixture regardless of what happens next, since the test
suite currently has no dispersive-material coverage at all.

### A2 — Sort the dispersive cell list for locality

Independent of A1 and cheaper. If `m_LM_pos` is not ordered to match the
engine's memory layout, the gather is scattered. Sorting the cell list once at
operator setup costs nothing at runtime.

*Cheap falsification:* instrument the position list and measure how far
consecutive `i` land apart in engine address space. If already near-sequential,
drop it.

### A3 — Cache-line-aligned AVX2 field layout

The 96-byte per-polarisation stride is implicated in the NT-store failure and
also fragments prefetch. Padding 3 `f8vector`s to 4 (128 B) costs 25% more
memory on a bandwidth-bound kernel — plausibly a net loss on its own, and only
interesting *combined with* NT stores.

High risk, wide blast radius (touches every engine and extension indexing
scheme). Listed for completeness; **do not start here.**

### G1 — Fuse the dispersive passes into the main GPU kernels

Only UPML is fused (`m_hasFusedUPML`). Dispersive costs four separate
dispatches per timestep (`dispersive_pre_voltage`, `dispersive_apply_voltage`,
`dispersive_pre_current`, `dispersive_apply_current`) plus barriers, each
re-reading the global field arrays.

Caveat that must be settled first: UPML fusion worked because PML regions are
axis-aligned boxes testable in-register. Dispersive cells are a *sparse list*,
so fusion needs a per-cell membership lookup in the main kernel — which may
cost more than the saved passes when the dispersive fraction is small.

*Cheap falsification:* measure the dispersive dispatches' share of frame time
on a phantom model. If the four dispatches are <5% of the timestep, the
fusion cannot pay for the lookup.

### G2 — Dispatch and barrier count on small grids

At 12–14 GCells/s on cache-resident grids, a timestep is short enough that
per-dispatch overhead may be material once several extension families are
active. Nobody has counted dispatches per timestep or measured the floor.

*Cheap falsification:* count dispatches per timestep, then time an empty-kernel
run at the same dispatch count to establish the launch floor.

### G3 — `volt_flux` traffic in the PML path

PML cells carry an extra read-modify-write of the flux array per component.
After `00ba1c6` the coefficients are cheap, so flux traffic is now the
dominant PML-specific cost. Whether it can be reduced at all is unclear — the
flux-swap trick needs the previous value.

*Cheap falsification:* compute the flux array's byte traffic as a fraction of
total PML-cell traffic. If it is small, close this out.

---

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
