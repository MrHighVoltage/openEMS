# Performance

This fork adds two engine families to openEMS — **AVX2/FMA CPU engines** and a
**Vulkan GPU engine** — plus a trapezoidal temporal-blocking schedule for the
AVX2 multithreaded engine. This document reports what they are worth, measured
against a from-scratch build of upstream openEMS on the same machine, with the
same compiler and the same models.

The engineering record behind these numbers — including what was tried and
rejected — is in [OPTIMIZATIONS.md](OPTIMIZATIONS.md). This document is the
summary; that one is the evidence.

---

## 1. At a glance

Throughput in **MCells/s** (million cell-updates per second), higher is better.
"Upstream best" is the fastest of the three upstream configurations measured,
which is always its SSE multithreaded engine at a hand-picked thread count —
not its slower out-of-the-box default.

| Model | Cells | Upstream<br>best | Fork AVX2<br>8 threads | Fork AVX2<br>+ temporal blocking | Fork Vulkan<br>RX 6800 | Best speedup |
|---|---|---|---|---|---|---|
| 64&sup3; PEC | 0.26 M | 2081 | 3397 | 3407 | 12384 | **5.95&times;** |
| 160&times;128&times;192 PEC | 3.93 M | 789 | 896 | 3284 | 13211 | **16.75&times;** |
| 224&sup3; PEC | 11.24 M | 585 | 630 | 2644 | 5673 | **9.70&times;** |
| 160&times;128&times;192 PML_8 | 3.93 M | 280 | 266 | 775 | 9574 | **34.20&times;** |
| 224&sup3; PML_8 | 11.24 M | 261 | 237 | 830 | 4620 | **17.68&times;** |
| 224&sup3; Mur | 11.24 M | 530 | 565 | 1813 | 5126 | **9.68&times;** |

Two separate results are stacked in that table, and they are worth keeping
apart:

- **On the CPU**, temporal blocking is the large win: **2.8&times;–4.5&times;**
  over upstream's best on every grid that does not fit last-level cache. It is
  an opt-in prototype (§5.4).
- **On a discrete GPU**, the Vulkan engine is worth **6&times;–34&times;**
  over upstream's best CPU engine. The spread is wide because the two sides
  respond very differently to a PML: on the 224&sup3; model, adding PML_8 costs
  the CPU **2.24&times;** (585 → 261 MC/s) and the GPU only **1.23&times;**
  (5673 → 4620 MC/s), so the ratio between them widens wherever PML dominates.

---

## 2. What was measured, and how

**Host.** Intel Core i9-13900K (8 P-cores + 16 E-cores, 32 threads, 36 MB L3),
32 GB dual-channel DDR5, `powersave` governor with turbo active. Two GPUs:
Radeon RX 6800 (RADV) and integrated UHD 770 (ANV), Mesa 26.2.1, Vulkan 1.4.

**Builds.** Both configured from scratch and compiled with the same toolchain
(GCC 16.2.1, CMake Release, `-O3 -DNDEBUG`) against the same CSXCAD, HDF5, VTK
and Boost:

| | commit | engines available |
|---|---|---|
| upstream | `d3d2a49` (2026-08-15) | `basic`, `sse`, `sse-compressed`, `multithreaded` |
| this fork | `42e3d2e` (2026-09-05) | the above **+** `avx2`, `avx2-multithreaded`, `gpu` |

The fork's merge-base with upstream *is* upstream's head, so this is exactly
upstream plus 93 commits, with no divergence to account for.

**Method.** Both builds' `openEMS` **binaries** are driven over byte-identical
XML models, written once with `Write2XML()`. Nothing depends on which Python
bindings happen to be installed, and the models contain nothing a stock
upstream build cannot parse — a uniform Cartesian grid, one excitation box, a
boundary condition. There are **no probes, no field dumps and no
post-processing**, so what is reported is the time-stepping loop alone: the
binary's own final `Speed: <x> MCells/s` line. Operator build time is reported
separately in §6.

Each binary was checked to load its own `libopenEMS.so.0` rather than the
installed one — separately *configured* build directories, because RPATH is
absolute and a copied build tree is not an independent binary.

**The harness is in the repo**: `python/Tests/benchmark_fork_vs_upstream.py`.
See §7 to reproduce.

### 2.1 Why some rows are pinned

Rows that compare engines to each other pin to one hardware thread on each of
the eight P-cores (`taskset -c 0,2,4,6,8,10,12,14`) and pass
`--numThreads=8` on both sides. This is **not** noise reduction — the one
genuinely noisy configuration stays noisy when pinned (§8) — it is to make the
comparison controlled, so that both builds run the same number of threads on
the same cores and a column difference is the engine and nothing else.

Eight threads is where both builds peak. From a separate calibration sweep on
the 224&sup3; PEC model (600 timesteps, unpinned): upstream reaches
543 / 570 / **572** / 499 MC/s at 4 / 6 / 8 / 12 threads, the fork's AVX2
engine 600 / 621 / **622** / 552, and the temporal-blocked schedule
1765 / **2305** / 2109 at 4 / 8 / 12. Both roll off past 8 for the same reason:
a plain OpenMP STREAM-Triad on this host peaks at 6–8 threads and loses ~20% by
16 ([OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.3). The memory controller sets that
curve, not the engines.

Rows labelled **default** run exactly as a user gets them: no pinning, no
`--numThreads`, each build using its own thread auto-tune. Both families are
reported; §5.3 is specifically about the difference between them.

Every figure is the best of 3 repeats, or 6 on the 64&sup3; grid, for the
reason in §8.

---

## 3. Full results

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 8t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 8t | fork<br>default | fork<br>+tblock 8t | fork<br>+tblock default | fork GPU<br>RX 6800 | fork GPU<br>UHD 770 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64&sup3; PEC | 40000 | 478 | 2081 | 942 | 989 | 3397 | 2786 | 3407 | 2981 | 12384 | 633 |
| 160&times;128&times;192 PEC | 3000 | 267 | 789 | 558 | 498 | 896 | 905 | 3284 | 2345 | 13211 | 520 |
| 224&sup3; PEC | 1200 | 251 | 585 | 497 | 449 | 630 | 611 | 2644 | 2110 | 5673 | 519 |
| 160&times;128&times;192 PML_8 | 1500 | 110 | 280 | 206 | 148 | 266 | 257 | 775 | 673 | 9574 | 317 |
| 224&sup3; PML_8 | 600 | 111 | 261 | 179 | 147 | 237 | 225 | 830 | 647 | 4620 | 321 |
| 224&sup3; Mur | 1000 | 212 | 530 | 423 | 362 | 565 | 541 | 1813 | 1458 | 5126 | 518 |

---

## 4. Where the speed comes from

### 4.1 The AVX2 engines

`Engine_AVX2` and `Engine_AVX2_Multithread` replace the 4-wide SSE Yee kernels
with 8-wide AVX2+FMA ones, keeping the compressed-operator storage of the SSE
engines. On top of the raw width, three changes mattered:

- **Redundant field loads eliminated** in the update kernels (`e3932e8`) —
  +7–8% single-threaded, flat at 8 threads, which was the first clear sign the
  multithreaded engine was already bandwidth-bound.
- **The missing AVX2 fast field-extraction path** (`586a8ac`). Field extraction
  had an SSE fast path and no AVX2 one, so every dump fell back to the generic
  scalar path — an AVX2-specific tax SSE never paid. Dumping the full domain
  every timestep went 336 → 426 MC/s.
- **Vectorized UPML passes** (`08fa968`).

Single-threaded, where the engine is not yet fighting the memory system, the
width shows up directly:

| Model | upstream SSE | fork AVX2 | |
|---|---|---|---|
| 64&sup3; PEC | 478 | 989 | 2.07&times; |
| 160&times;128&times;192 PEC | 267 | 498 | 1.86&times; |
| 224&sup3; PEC | 251 | 449 | 1.79&times; |
| 160&times;128&times;192 PML_8 | 110 | 148 | 1.34&times; |
| 224&sup3; PML_8 | 111 | 147 | 1.33&times; |
| 224&sup3; Mur | 212 | 362 | 1.71&times; |

The 1.79–2.07&times; on PEC is close to what doubling the vector width can
give. The 1.33–1.34&times; on PML is lower because the PML passes are a larger
share of the work and vectorize less well — see §5.1, where this becomes a
problem at high thread counts.

### 4.2 Trapezoidal temporal blocking

This is the largest CPU result in this document, and the reasoning behind it is
worth stating because it is not "make the arithmetic faster".

The multithreaded AVX2 engine **is already at the DRAM roofline.** Traffic
measured with the core PMU on this host is 31.4 bytes per cell per timestep
against a 24 B/cell ideal, which at its optimum is ~48 GB/s against a ~49 GB/s
STREAM-Triad ceiling. There is no arithmetic left to win and no scattered
access left to tile. (Traffic and roofline figures in this section are from
[OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.5 and §4.7, measured separately on this
same host; the throughput tables below are from this run.)

What is left is to *stop going to DRAM*. A trapezoidal schedule advances a
cache-resident tile of the grid through *k* consecutive timesteps before moving
on, which divides DRAM traffic by roughly *k*. Measured on a proxy benchmark:
30.95 → 2.40 B/cell, **12.9&times; less traffic.** In the real engine:

| Model | field state | flat AVX2-MT | + temporal blocking | |
|---|---|---|---|---|
| 64&sup3; PEC | 6 MB | 3397 | 3407 | 1.00&times; |
| 160&times;128&times;192 PEC | 94 MB | 896 | 3284 | 3.67&times; |
| 224&sup3; PEC | 270 MB | 630 | 2644 | 4.20&times; |
| 160&times;128&times;192 PML_8 | 94 MB | 266 | 775 | 2.91&times; |
| 224&sup3; PML_8 | 270 MB | 237 | 830 | 3.51&times; |
| 224&sup3; Mur | 270 MB | 565 | 1813 | 3.21&times; |

The 64&sup3; row is the schedule declining to engage, and saying so:

```
AVX2 temporal blocking: disabled, only 64 x-lines for a tile width of 384.
```

The tile width is derived from the last-level cache, and at this grid's small
x-planes that comes out wider than the grid itself. There is nothing to gain
here anyway — 6 MB of field state is already L3-resident, and blocking a
cache-resident grid was measured to *lose* 32% — but the guard that fires first
is the geometric one. Either way the run falls back to the flat sweep, which is
why the row reads 1.00&times; rather than a regression.

Three properties make this usable rather than merely fast:

- **It is bit-identical to the flat sweep.** Not "within tolerance" —
  element-for-element identical. `python/Tests/test_temporal_blocking.py`
  checks this for PEC, PML, Mur, Drude, lumped RLC, a local absorbing sheet, a
  TF/SF plane wave, a combined case, and a sweep over
  (k,W) &isin; {2:8, 4:16, 3:13, 6:20} &times; {1,3,4} threads; all 9 cases pass
  on this build. Each case asserts that blocking actually *engaged* before
  comparing fields — otherwise the comparison passes vacuously every time an
  extension declines and the run silently falls back to the flat sweep.
- **It refuses rather than risks.** Blocking engages only when *every* active
  engine extension implements the per-slab hooks. An extension that has not
  returns `false` by default and sends the whole run back to the flat sweep, so
  a newly added extension is safe by omission rather than by enumeration. The
  startup line names every extension that declined.
- **It carries the extensions**, not just the bare PEC kernel: UPML, Mur and
  local absorbing BCs, dispersive materials, lumped RLC and TF/SF plane-wave
  sources all have slab paths. Steady-state detection is the one extension that
  genuinely cannot be blocked — it integrates energy over the whole grid every
  period, and a blocked schedule has no moment at which the whole grid is at one
  timestep.

### 4.3 The Vulkan GPU engine

A full Vulkan 1.3 compute implementation — Yee update, UPML, Mur, excitation,
TF/SF, dispersive materials, lumped RLC, energy reduction and probe gather all
run as compute shaders, with the operator compressed on the device.

| Model | RX 6800 | vs upstream best | UHD 770 | vs upstream best |
|---|---|---|---|---|
| 64&sup3; PEC | 12384 | 5.95&times; | 633 | 0.30&times; |
| 160&times;128&times;192 PEC | 13211 | 16.75&times; | 520 | 0.66&times; |
| 224&sup3; PEC | 5673 | 9.70&times; | 519 | 0.89&times; |
| 160&times;128&times;192 PML_8 | 9574 | 34.20&times; | 317 | 1.13&times; |
| 224&sup3; PML_8 | 4620 | 17.68&times; | 321 | 1.23&times; |
| 224&sup3; Mur | 5126 | 9.68&times; | 518 | 0.98&times; |

The changes that produced those numbers, in order of what they were worth:

- **Don't put field buffers in an undersized BAR aperture** (`fbd95a6`) —
  **6.4&times;** on grids exceeding the 256 MB aperture. This card has no
  resizable BAR, so above that size the buffers were being serviced across
  PCIe rather than from VRAM.
- **Per-cell operator storage when deduplication stops paying** (`6b571da`) —
  **1.93–2.76&times;** on graded meshes. Found by watching throughput collapse
  4&times; when 493,040 unique coefficient sets in a 23 MB table turned six
  per-cell gathers into six scattered cache lines. What matters is table size
  *relative to cache*, not the compression ratio.
- **Async field-dump download** (`ea82f59`) — **~18&times; wall clock** on a
  dump-heavy run (74 s → 16 s). Not visible in this document's benchmarks,
  which deliberately dump nothing, but it is the difference that matters most
  in practice on dump-heavy models.
- **Narrowed the compressed-operator index to 8/16/32 bits** (`49c7efc`) —
  +3.5–6.2%.
- **Deduplicated UPML coefficients into shared tables** (`00ba1c6`) —
  1.08–1.17&times;, and 19.2 MB → 1.6 MB of tables.
- **Stopped allocating the async dump ring for runs that never dump**
  (`05ce5f0`) — **2.8&times; less GPU memory** (224&sup3;: 803 → 288 MB VRAM).
  Throughput unchanged; this sets the largest grid the card can hold, which is
  the GPU's hard limit.

**The GPU Yee kernel is at roofline.** Counting traffic from the shader source
rather than assuming it gives 74 B/cell/timestep ([OPTIMIZATIONS.md](OPTIMIZATIONS.md)
§4.7/C4), and this run's 5673 MC/s &times; 74 B = **420 GB/s** against a
`clpeak` ceiling of 477 GB/s on this card — **88% of achievable bandwidth**.
Further GPU gains have to come from reducing traffic or passes, not from
tightening arithmetic.

The 64&sup3; and 160&times;128&times;192 rows run 12–13 GC/s because their
field state fits the card's 128 MB Infinity Cache; the cliff between 3.9 M and
11.2 M cells is exactly where ~24 B/cell crosses that boundary (94 MB vs
270 MB of field state). Above it the GPU sits flat at ~5.2 GC/s out to at least
26 M cells ([OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.1).

### 4.4 Correctness work that came with it

Two fixes worth naming, because performance work that breaks results is worth
nothing:

- **A TF/SF write race** (`b87a1f7`): a non-atomic `+=` on field elements
  shared by two TF/SF box faces made the Vulkan engine **nondeterministic** on
  any TF/SF model — six runs gave six different answers. Found by hashing probe
  traces across repeated runs of *one* engine, which separates "these two
  engines disagree" from "this one disagrees with itself".
- **An operator indexing regression**, fixed in `81005f8` by restoring the
  x-fastest EC index in `Calc_ECOperatorPos`. Every engine routes through that
  function, so cross-engine comparison is structurally blind to this class of
  bug — all engines agree, and all are wrong together. Only absolute-value
  tests on inhomogeneous models catch it.

---

## 5. Where this fork is *not* faster

### 5.1 The AVX2 multithreaded engine loses to SSE on PML

At 8 threads on a PML model, the AVX2 engine is **5–9% slower** than the SSE
engine it was meant to replace — despite being 1.33&times; faster
single-threaded on the same model.

| Model | upstream SSE-MT | fork SSE-MT | fork AVX2-MT | fork AVX2-MT<br>+ temporal blocking |
|---|---|---|---|---|
| 160&times;128&times;192 PML_8 | 280 | 281 | 266 | 775 |
| 224&sup3; PML_8 | 261 | 261 | 237 | 830 |
| 224&sup3; PEC | 585 | 587 | 630 | 2644 |

The fork's SSE-MT column was measured the same way as the rest — same pinning,
same 8 threads, best of 3. It matches upstream's to within 0.4%, so this is
not a fork-wide regression; it is specific to the AVX2 engine at high thread
counts. Both engines are bandwidth-bound there, and the AVX2 kernel's wider
per-lane footprint buys nothing once DRAM sets the pace, while its PML passes
cost slightly more. The right fix is not more vectorization — temporal blocking
removes the constraint entirely and takes the same case to 775–830 MC/s.

**Practical guidance:** on a PML-dominated model with many threads and without
temporal blocking, `--engine=multithreaded` is the better choice.

### 5.2 The integrated GPU is not a compute device

Measured here: **317–633 MC/s on every grid**, essentially flat with grid size,
i.e. **0.30&times;–1.23&times;** upstream's best CPU configuration (§4.3 table).

That flatness has a cause. `clpeak` reports 48.7 GB/s for the UHD 770 against
the CPU's ~49 GB/s STREAM-Triad ceiling: the iGPU and the CPU are the same
memory system, so there is no cache hierarchy of its own for grid size to
interact with. It is worth having as a portability check on a second Vulkan
driver — ANV rather than RADV, which is how driver-specific assumptions get
caught — and for nothing else. Device selection already prefers the discrete
GPU.

### 5.3 Thread auto-tune: fixed here, still worth knowing

Upstream's thread search hill-climbs inside `NextInterval()`, which is called
only after four seconds of wall time, adding one thread per call starting from
one. On a 4-core machine it converges quickly; on this 32-thread machine
reaching the optimum takes ~44 s, during which the run is nowhere near peak
([OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.2). `fd555ee` replaces it
with a search that scores candidates on real timesteps inside `IterateTS()` —
legitimate because thread count cannot change results — settling in ~250
timesteps instead. One detail mattered more than expected: the first batch after
each thread respawn must be discarded, or cold private caches penalise exactly
the wide configurations under judgement.

How close each build's default gets to its own pinned optimum:

| Model | upstream default | upstream best | reached | fork default | fork best | reached |
|---|---|---|---|---|---|---|
| 64&sup3; PEC | 942 | 2081 | 45% | 2786 | 3397 | 82% |
| 160&times;128&times;192 PEC | 558 | 789 | 71% | 905 | 896 | 101% |
| 224&sup3; PEC | 497 | 585 | 85% | 611 | 630 | 97% |
| 160&times;128&times;192 PML_8 | 206 | 280 | 74% | 257 | 266 | 97% |
| 224&sup3; PML_8 | 179 | 261 | 68% | 225 | 237 | 95% |
| 224&sup3; Mur | 423 | 530 | 80% | 541 | 565 | 96% |

Upstream's default leaves 15–55% on the table; the fork's lands within 3–5% on
every stable configuration. This is a fork improvement, but note what it means
for reading §3: **comparing the two `default` columns measures the auto-tune as
much as the engines.** That is why the pinned columns exist.

### 5.4 Temporal blocking is opt-in, and capped by probe cadence

It is a prototype behind `OPENEMS_AVX2_TEMPORAL_BLOCK=<k>` (here, `k=16`) and
does nothing unless the variable is set. Two limits are worth knowing before
relying on the numbers in §4.2:

- **Block depth is capped by the probe interval.** `openems.cpp` calls
  `IterateTS()` with the number of timesteps until the next probe or dump, and
  the schedule clamps *k* to it — which is what keeps probes and dumps correct
  with no work on their side. That interval is typically 5–25 timesteps, which
  happens to bracket the measured optimum (*k* = 6–24). A run oversampled far
  past Nyquist will not reach these figures.
- **These models have no probes or dumps at all**, so they see the ceiling of
  what blocking can do. A probe-dense model will land lower.

---

## 6. Operator build time

Time-stepping throughput is not the whole cost. Operator construction, measured
as wall clock outside the stepping loop:

| Model | upstream SSE-MT | fork AVX2-MT | fork Vulkan |
|---|---|---|---|
| 224&sup3; PEC | 6.0 s | 5.1 s | 6.9 s |
| 224&sup3; PML_8 | 8.3 s | 5.6 s | 6.6 s |
| 224&sup3; Mur | 6.0 s | 5.1 s | 6.9 s |

Operator setup is parallelized in this fork, which is where the PML case's
8.3 s → 5.6 s comes from.

For the GPU, setup is the dominant fixed cost of a short run: at 224&sup3; the
operator takes ~6.9 s while all 1200 timesteps take only ~2.4 s. Since the CPU
pays ~6.0 s of setup too, though, the crossover is early — the GPU is ahead of
upstream's best CPU configuration after about **52 timesteps**. Against the
fork's own temporal-blocked CPU engine the crossover is ~790 timesteps, so on
short runs the two are closer than §3 suggests.

---

## 7. Reproducing

```bash
# 1. build upstream from scratch, out of tree
git worktree add --detach ../openEMS-upstream-bench upstream/master
cmake -S ../openEMS-upstream-bench -B ../openEMS-upstream-bench/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCSXCAD_ROOT_DIR=$HOME/opt -DFPARSER_ROOT_DIR=$HOME/opt
make -C ../openEMS-upstream-bench/build -j$(nproc)

# 2. build this fork from scratch, with the GPU engine
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON \
      -DCSXCAD_ROOT_DIR=$HOME/opt -DFPARSER_ROOT_DIR=$HOME/opt
make -C build-bench -j$(nproc)

# 3. run the matrix (~2 h)
python python/Tests/benchmark_fork_vs_upstream.py --out /tmp/bench.json
```

The script writes incrementally and skips configurations already present in the
output file, so it can be interrupted and resumed. `--only <substring>` runs a
single cell of the matrix.

The raw output behind every table in this document is committed as
`python/Tests/results/fork_vs_upstream_2026-09-05.json`, including the
individual repeats, so the best-of-N choices can be checked rather than taken
on trust.

Related drivers: `python/Tests/benchmark_avx2_engine.py`,
`python/Tests/benchmark_gpu_engine.py` (`--gpu-index` selects the device),
`python/Tests/bench_temporal_blocking.c`.

---

## 8. Measurement notes

Two hazards on this host produced wrong numbers before they were understood,
and both are worth repeating for anyone re-running this.

**Cache-resident grids are bimodal on this CPU, for a reason not identified.**
The 64&sup3; model returns either ~480 or ~150–310 MC/s from run to run — a
3&times; swing on identical input. What it is *not*, each ruled out by direct
measurement rather than by argument:

| candidate | test | result |
|---|---|---|
| competing load | sample `ps`/loadavg through the run | machine otherwise idle |
| P-core vs E-core placement | `taskset -c 0` | still bimodal |
| ASLR-driven cache aliasing | `setarch -R` | still bimodal |
| CPU frequency / thermal throttle | sample `cpu MHz` through the run | constant 5.3 GHz in fast *and* slow runs |
| hugepage backing | `smaps_rollup` per run | identical `AnonHugePages` and RSS |

The leading remaining candidate is the **uncore/ring clock**, which ranges
800 MHz–5 GHz on this part and would affect only L3-resident work — but this is
a hypothesis, not a finding: `current_freq_khz` is not readable unprivileged
here, so it was never confirmed, and the effect shows up at 8 threads as well
as at 1, which a simple "one core does not raise the uncore" story does not
explain.

What matters for reading this document is the scope, and that *is* established:
only the cache-resident grid is affected. Every DRAM-bound configuration is
stable to **0.5–4%** across repeats. Hence 6 repeats on the 64&sup3; grid and 3
elsewhere, and hence the 64&sup3; row should be read as "about this" while the
rest can be read as measured.

**Never A/B a GPU change by disabling the excitation.** With no source the
fields stay exactly zero, and an all-zero grid runs ~28% faster than the
identical kernel on real data — zero operands flip almost no bits. This nearly
produced a fake 26% win once. Both arms of any comparison must carry real field
data.

---

*Measurements taken 2026-09-05. Upstream `d3d2a49`, fork `42e3d2e`. Every
throughput figure in the tables above comes from
`python/Tests/results/fork_vs_upstream_2026-09-05.json`; the two prose figures
that do not (the fork's SSE-MT column in §5.1 and the thread sweep in §2.1) are
labelled where they appear.*
