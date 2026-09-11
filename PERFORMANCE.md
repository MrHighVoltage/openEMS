# Performance

This fork adds two engine families to openEMS — **AVX2/FMA CPU engines** and a
**Vulkan GPU engine** — plus a trapezoidal temporal-blocking schedule that now
runs on **both** of them. This document reports what they are worth, measured
against a from-scratch build of upstream openEMS on the same machine, with the
same compiler and the same models.

Everything here was re-measured after the GPU engine gained the blocked
schedule: Host C on **2026-09-07**, Host A on **2026-09-08** from a clean
rebuild of the same commit.

On **2026-09-11** the CPU half of the matrix was extended to five more
machines — Hosts D–H, §2 — chosen because they are *unlike* the first two:
an AMD Zen 2 desktop part, three dual-socket servers and a six-core client
CPU, spanning last-level caches from 9 MB to 4×16 MB and including one host
with no AVX2 at all. They cost the document two of its conclusions. CPU
temporal blocking does not simply track last-level cache size (§5.4); it
tracks the cache the running threads *collectively* have, and the tile width
is derived from one cache instance, which is wrong on every machine whose LLC
is partitioned — worth up to **2×** where it bites (§5.7). And the fork's
AVX2 engine has no runtime ISA check, so on a pre-Haswell host it does not run
slowly, it dies (§5.8).

The engineering record behind these numbers — including what was tried and
rejected — is in [OPTIMIZATIONS.md](OPTIMIZATIONS.md). This document is the
summary; that one is the evidence.

---

## 1. At a glance

Throughput in **MCells/s** (million cell-updates per second), higher is better.
"Upstream best" is the fastest of the three upstream configurations measured,
which is almost always its SSE multithreaded engine at a hand-picked thread
count — not its slower out-of-the-box default.

### Host C — i9-13900K + Radeon RX 6800

| Model | Cells | Upstream<br>best | Fork AVX2<br>8 threads | Fork AVX2<br>+ temporal blocking | Fork Vulkan<br>RX 6800 | Fork Vulkan<br>+ temporal blocking | Best speedup |
|---|---|---|---|---|---|---|---|
| 64&sup3; PEC | 0.26 M | 2039 | 3507 | 3458 | 12433 | 12385 | **6.10&times;** |
| 160&times;128&times;192 PEC | 3.93 M | 789 | 889 | 3249 | 13070 | 12950 | **16.57&times;** |
| 224&sup3; PEC | 11.24 M | 586 | 632 | 2607 | 5674 | 12319 | **21.01&times;** |
| 160&times;128&times;192 PML_8 | 3.93 M | 280 | 266 | 800 | 9462 | 9844 | **35.15&times;** |
| 224&sup3; PML_8 | 11.24 M | 262 | 237 | 828 | 4619 | 10212 | **38.98&times;** |
| 224&sup3; Mur | 11.24 M | 531 | 569 | 1848 | 5125 | 5125 | **9.66&times;** |

Three separate results are stacked in that table, and they are worth keeping
apart:

- **On the CPU**, temporal blocking is the large win: **3.0&times;–4.1&times;**
  over the flat AVX2 sweep on every grid that does not fit last-level cache. It
  is an opt-in prototype (§5.5).
- **On a discrete GPU**, temporal blocking is worth **2.2&times;** on the grids
  that exceed the card's Infinity Cache, and roughly nothing on the ones that
  do not (§4.3). This is new since the last revision of this document, and it
  is what moves 224&sup3; PML_8 from 17.7&times; to **39.0&times;**.
- **Together**, the Vulkan engine is worth **6&times;–39&times;** over
  upstream's best CPU engine. The spread is wide because the two sides respond
  very differently to a PML: on the 224&sup3; model, adding PML_8 costs the CPU
  **2.24&times;** (586 → 262 MC/s) and the blocked GPU only **1.21&times;**
  (12319 → 10212 MC/s), so the ratio between them widens wherever PML
  dominates.

### Host A — i7-6700K + integrated HD 530

A four-core Skylake desktop part with an 8 MB L3 and a Gen9 iGPU. It is in this
document because *every conclusion above is a property of the hardware as much
as of the code*, and a second machine is the cheapest way to say which is
which.

| Model | Upstream<br>best | Fork AVX2<br>4 threads | Fork AVX2<br>+ tblock (default threads) | Fork Vulkan<br>HD 530 | Best speedup |
|---|---|---|---|---|---|
| 64&sup3; PEC | 544 | 937 | 1322 | 505 | **2.45&times;** |
| 160&times;128&times;192 PEC | 448 | 497 | 862 | 381 | **1.93&times;** |
| 224&sup3; PEC | 423 | 475 | 577 | 484 | **1.37&times;** |
| 160&times;128&times;192 PML_8 | 146 | 164 | 251 | 225 | **1.72&times;** |
| 224&sup3; PML_8 | 168 | 183 | 227 | 270 | **1.61&times;** |
| 224&sup3; Mur | 358 | 388 | 382 | 503 | **1.40&times;** |

Two things do **not** carry over from Host C, and both have the same cause —
an 8 MB last-level cache instead of 36 MB:

- **CPU temporal blocking gains far less** (1.05&times;–1.42&times; instead of
  3.0&times;–4.1&times;), and on the Mur model it is a **13% loss**. An 8 MB L3
  forces tiles six x-lines wide at k=3, and at that size the trapezoid's wedges
  cost more than the traffic they save (§5.4).
- **GPU temporal blocking gains nothing at all** on the HD 530 (1.00&times;–
  1.04&times;), because that device is compute-bound rather than
  bandwidth-bound, so removing DRAM traffic removes nothing that was limiting
  it (§4.3).

What *does* carry over is the AVX2 width itself (1.65&times;–2.59&times;
single-threaded, §4.1), and — unexpectedly — the value of the GPU engine on a
weak CPU: the HD 530 is the **fastest engine on this machine** on the 224&sup3;
PML_8 and Mur models, which the much stronger UHD 770 never is on Host C
(§5.3).

### Hosts D–H — five servers, CPU only

None of these five has a Vulkan driver stack, so the GPU engine is out of the
picture and these are CPU-only numbers. Best fork configuration against best
upstream configuration, both as defined above:

| Model | compute5<br>Zen 2, 16C | loki<br>2&times;Skylake-SP, 32C | maxwell<br>2&times;Sandy Bridge, 12C | saturn<br>Coffee Lake, 6C | tierwater<br>2&times;Haswell, 12C |
|---|---|---|---|---|---|
| 64&sup3; PEC | **2.52&times;** | **2.35&times;** | *does not run* | **2.16&times;** | **1.89&times;** |
| 160&times;128&times;192 PEC | **2.68&times;** | 1.35&times; | *does not run* | **2.53&times;** | 1.36&times; |
| 224&sup3; PEC | 1.57&times; | 1.18&times; | *does not run* | 1.59&times; | 1.15&times; |
| 160&times;128&times;192 PML_8 | 2.16&times; | 1.32&times; | *does not run* | 1.85&times; | 1.16&times; |
| 224&sup3; PML_8 | 1.68&times; | 1.09&times; | *does not run* | 1.60&times; | 1.08&times; |
| 224&sup3; Mur | 1.19&times; | 1.37&times; | *does not run* | 1.46&times; | 1.10&times; |

Three things to read out of that table, in descending order of how much they
should change what anyone does:

- **maxwell runs nothing.** It is a pre-Haswell host, the AVX2 engine is
  selected at compile time with no runtime check, and every fork configuration
  — including the *default* one — terminates with SIGILL. Not a slow row, an
  absent one. §5.8.
- **The two hosts that gain least, loki and tierwater, are the two whose
  last-level cache is split across sockets** — and the loss is not inherent to
  them. Told the cache its threads actually have, loki's blocked schedule goes
  from 636 to 1280 MC/s and tierwater's from 356 to 569 on the 224&sup3; PEC
  model. The matrix above reports the defaults, so it reports the defect.
  §5.7.
- **saturn, the weakest machine here, gains the most consistently**
  (1.46&times;–2.53&times;) for the same reason in reverse: six cores behind a
  single undivided 9 MB L3 is exactly the shape the tile heuristic was written
  for.

---

## 2. What was measured, and how

**Host C.** Intel Core i9-13900K (8 P-cores + 16 E-cores, 32 threads, 36 MB
L3), 32 GB dual-channel DDR5, `powersave` governor with turbo active. Two GPUs:
Radeon RX 6800 (RADV, 128 MB Infinity Cache, 256 MB BAR, no resizable BAR) and
integrated UHD 770 (ANV). Mesa 26.2.1, Vulkan 1.4. Otherwise idle.

**Host A.** Intel Core i7-6700K (4 cores / 8 threads, 8 MB L3), 64 GB
dual-channel DDR4, `powersave` governor. One GPU: integrated Intel HD Graphics
530 (Skylake GT2, ANV). Mesa 26.2.1, Vulkan 1.4. This is the machine
[OPTIMIZATIONS.md](OPTIMIZATIONS.md) calls Host A and describes as "gone" — it
is not; it is a remote server, and §1's AVX2 table there was measured on it.
It is a **shared** machine and was not quiesced for this run: `rclone`,
Jellyfin and an idle BOINC client were resident throughout, at a load average
of 0.6–2.8. See §8 for what that does and does not affect.

Host A's matrix was run twice, a day apart, the second time from a clean
rebuild of the same commit fetched from the remote rather than from a local
copy. The two agree to **1.7% worst case** and to 0.8% on every DRAM-bound
row; the tables here are the second run, and the first is committed alongside
it as the corroborating repeat (§7).

**Hosts D–H** are five servers added on 2026-09-11, all running only the
benchmark and otherwise idle (loki had other users logged in but no load).
None of them has a Vulkan ICD installed, so all five are CPU-only. They were
picked for cache and socket topology, which is what §5.4 and §5.7 turn on:

| | machine | cores | last-level cache | `getconf LEVEL3_CACHE_SIZE` | OS |
|---|---|---|---|---|---|
| **D** | compute5 — Ryzen 9 3950X | 16C/32T, 1 socket | 4 &times; 16 MB (one per CCX), **64 MB total** | 16 MB | Rocky 8.10, glibc 2.28 |
| **E** | loki — 2&times; Xeon Gold 6130 | 32C/64T, 2 sockets | 2 &times; 22 MB, non-inclusive, **+ 1 MB L2 per core** | 22 MB | Rocky 8.10, glibc 2.28 |
| **F** | maxwell — 2&times; Xeon E5-2630 | 12C/24T, 2 sockets | 2 &times; 15 MB | 15 MB | Rocky 8.10, glibc 2.28 |
| **G** | saturn — Core i5-8500 | 6C/6T, 1 socket | 9 MB, undivided | 9 MB | CentOS 7.9, glibc 2.17, **kernel 3.10** |
| **H** | tierwater — 2&times; Xeon E5-2620 v3 | 12C/24T, 2 sockets | 2 &times; 15 MB | 15 MB | Rocky 8.10, glibc 2.28 |

That fifth column is the whole of §5.7: on D, E, F and H it is the size of
*one* cache instance, not of the cache a full-machine run has.

**maxwell (F) is pre-Haswell** — it has AVX and SSE4.2 but no AVX2 and no FMA.
It is in this document precisely for that: it is the only machine here that can
answer what a binary carrying an AVX2 engine does on hardware that cannot
execute one. §5.8.

**Builds.** For Hosts A and C, all four builds were configured from scratch and
compiled with the same toolchain (GCC 16.2.1, CMake Release, `-O3 -DNDEBUG`)
against each host's own CSXCAD, HDF5, VTK and Boost:

| | commit | engines available |
|---|---|---|
| upstream | `d3d2a49` (2026-08-15) | `basic`, `sse`, `sse-compressed`, `multithreaded` |
| this fork | `6c795de` (2026-09-07) | the above **+** `avx2`, `avx2-multithreaded`, `gpu` |

The fork's merge-base with upstream *is* upstream's head, so this is exactly
upstream plus 99 commits, with no divergence to account for.

**Hosts D–H could not be built on**, and that constraint shaped the method.
They are shared machines that must not be modified, none of them has CSXCAD,
HDF5, VTK, CGAL or Boost installed, and saturn is a CentOS 7 box — glibc 2.17,
kernel 3.10 — which no binary from a current distribution will even start on.
So both sides were built *once*, on the workstation, inside a glibc-2.17
container (`quay.io/pypa/manylinux2014_x86_64`, GCC 10.2.1) against a
dependency stack compiled from source in the same container, and linked
statically against everything except libc. The resulting bundle needs nothing
on the target but a glibc ≥ 2.17 x86-64 system: its highest versioned symbol
requirement is exactly `GLIBC_2.17`, and its only dynamic dependencies are
`libc`, `libm`, `libdl`, `libpthread` and the three project libraries shipped
beside it. `python/Tests/build_portable_bench.sh` is that build, end to end.

Nothing was installed on any of the five hosts. The bundle was copied to the
NFS home they share and staged to node-local `/tmp` for the runs, so no NFS
round trip is anywhere near a measurement.

**The older compiler is not a variable.** GCC 10 against the GCC 16 used for
Hosts A and C is a real difference, and since it applies to both sides of every
comparison it cannot manufacture a fork-vs-upstream result — but it could
still distort the *absolute* numbers this document quotes. It does not. The
two builds were compared on Host C in a paired test, the arms alternating run
by run on the same model so that machine state cancels:

| Host C, 224&sup3; PEC | GCC 16 | GCC 10 | |
|---|---|---|---|
| `sse` 1 thread | 235 | 234 | 1.00&times; |
| `avx2` 1 thread | 439 | 438 | 1.00&times; |
| `avx2-multithreaded` 8t | 612 | 619 | 1.01&times; |
| `avx2-multithreaded` +tblock 8t | 2639 | 2703 | 1.02&times; |

Across all sixteen model/engine combinations tested the ratio stays within
**0.98–1.04**, with no systematic direction. Full output in
`python/Tests/results/toolchain_gcc16_vs_gcc10_2026-09-11.txt`. Note the form
of that control: two *matrix runs* on different days would have conflated the
toolchain with the machine's state, and an attempt at exactly that comparison
came out 15–29% low on the PEC rows purely because the workstation was busy
that afternoon. Alternating arms is what makes it a measurement.

**Method.** Both builds' `openEMS` **binaries** are driven over byte-identical
XML models, written once with `Write2XML()` and then copied to the other hosts,
so every machine runs the same bytes. The models are committed as
`python/Tests/models/*.xml` for that reason — on Hosts D–H nothing could
regenerate them, since none of them has the Python bindings. Nothing depends on which Python
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
Everything host-specific in it is a command-line option, which is what let the
same matrix run unchanged on both machines. See §7 to reproduce.

### 2.1 Why some rows are pinned, and at what thread count

Rows that compare engines to each other pin both builds to one hardware thread
per physical core and pass the same `--numThreads` on both sides. This is
**not** noise reduction — the one genuinely noisy configuration stays noisy
when pinned (§8) — it is to make the comparison controlled, so that both builds
run the same number of threads on the same cores and a column difference is the
engine and nothing else.

The pinned count is where **both builds peak**, from a separate calibration
sweep on the 224&sup3; PEC model (600–1200 timesteps, unpinned):

| host | pinning | upstream SSE-MT | fork AVX2-MT | fork +tblock |
|---|---|---|---|---|
| C | `-c 0,2,4,6,8,10,12,14`, 8 threads | 543 / 570 / **572** / 499 at 4/6/8/12 | 600 / 621 / **622** / 552 | 1765 / **2305** / 2109 at 4/8/12 |
| A | `-c 0,2,4,6`, 4 threads | 384 / **441** / 394 / 377 at 2/4/6/8 | **487** / 480 / 431 / 391 | 484 / 554 / **592** / 575 |
| D compute5 | `-c 0-15`, 16 threads | 423 / 410 / **424** / 422 / 372 at 4/8/12/16/32 | 436 / 424 / **441** / 439 / 392 | **1132** / 584 / 570 / 520 / 447 |
| E loki | `-c 0-31`, 32 threads | 559 / 856 / 856 / 929 / **1006** / 614 at 4/8/16&nbsp;s0/16&nbsp;spread/32/64 | 707 / 952 / 935 / 1048 / **1144** / 912 | 769 / **989** / 986 / 742 / 620 / 425 |
| F maxwell | `-c 0-11`, 12 threads | 332 / 351 / 439 / **538** / 421 at 4/6&nbsp;s0/6&nbsp;spread/12/24 | *SIGILL* | *SIGILL* |
| G saturn | `-c 0-1`, 2 threads | **207** / 201 / 180 at 2/4/6 | **225** / 211 / 190 | 304 / 316 / **337** |
| H tierwater | `-c 0-11`, 12 threads | 206 / 251 / 303 / **433** / 370 at 4/6&nbsp;s0/6&nbsp;spread/12/24 | 285 / 291 / 455 / **471** / 433 | 308 / **361** / 324 / 358 / 284 |

Host C's two engines agree on 8, and both roll off past it for the same reason:
a plain OpenMP STREAM-Triad on that host peaks at 6–8 threads and loses ~20% by
16 ([OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.3). The memory controller sets that
curve, not the engines.

The dual-socket rows carry an extra column the single-socket hosts do not: on
maxwell and tierwater, six threads *spread across both sockets*
(`-c 0-2,6-8`) beat six threads on one (`-c 0-5`) by **25%** and **21%** — two
memory controllers rather than one, for the same core count. That is why the
pinned sets for those hosts span both sockets.

**Three of the five hosts' blocked schedules peak somewhere other than their
flat ones, and by a lot**: compute5 at 4 threads against 16, loki at 8 against
32, saturn at 6 against 2. On Host A the same disagreement was read as SMT
helping a less bandwidth-bound kernel (§2.1 above). That reading does not
survive these hosts: the blocked schedule prefers *fewer* threads here, and
§5.7 shows why — at low thread counts the run fits inside one cache instance,
which is the only case in which the derived tile width is the right one. The
pinned `+tblock` columns in §3 therefore understate the schedule on Hosts D, E
and H, and §5.7 is where the corrected figures are.

**Host A's engines do not agree**, and that is itself a result. Its flat AVX2
engine peaks at **2** threads, upstream's SSE engine at **4**, and the
temporally blocked schedule at **6** — past the physical core count, into SMT.
Blocking makes the engine less bandwidth-bound, and a less bandwidth-bound
kernel is exactly the kind that SMT siblings help. Pinning to 4 was chosen
because it is where the two *flat* engines are closest to their peaks, so the
engine-vs-engine columns stay controlled; the consequence is that Host A's
`tblock 4t` column is **16% below** that schedule's own optimum, and its
`tblock default` column (unpinned, auto-tuned, free to use all 8 logical CPUs)
is the more representative one. Host A's summary table in §1 therefore quotes
the default column, and both are in §3.

Rows labelled **default** run exactly as a user gets them: no pinning, no
`--numThreads`, each build using its own thread auto-tune. Both families are
reported; §5.6 is specifically about the difference between them.

GPU rows are neither pinned nor thread-limited: the CPU is not the resource
under test there, so constraining it would only add a variable.

Every figure is the best of 3 repeats, or 6 on the 64&sup3; grid, for the
reason in §8.

---

## 3. Full results

### Host C — i9-13900K, RX 6800, UHD 770

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 8t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 8t | fork<br>default | fork<br>+tblock 8t | fork<br>+tblock default | GPU<br>RX 6800 | GPU RX 6800<br>+tblock | GPU<br>UHD 770 | GPU UHD 770<br>+tblock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64&sup3; PEC | 40000 | 476 | 2039 | 956 | 972 | 3507 | 2928 | 3458 | 2846 | 12433 | 12385 | 633 | 633 |
| 160&times;128&times;192 PEC | 3000 | 267 | 789 | 569 | 494 | 889 | 865 | 3249 | 2723 | 13070 | 12950 | 524 | 611 |
| 224&sup3; PEC | 1200 | 247 | 586 | 496 | 449 | 632 | 612 | 2607 | 1960 | 5674 | 12319 | 520 | 601 |
| 160&times;128&times;192 PML_8 | 1500 | 111 | 280 | 207 | 148 | 266 | 256 | 800 | 685 | 9462 | 9844 | 317 | 344 |
| 224&sup3; PML_8 | 600 | 112 | 262 | 180 | 147 | 237 | 224 | 828 | 640 | 4619 | 10212 | 323 | 347 |
| 224&sup3; Mur | 1000 | 219 | 531 | 431 | 364 | 569 | 543 | 1848 | 1466 | 5125 | 5125 | 518 | 518 |

### Host A — i7-6700K, HD 530

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 4t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 4t | fork<br>default | fork<br>+tblock 4t | fork<br>+tblock default | GPU<br>HD 530 | GPU HD 530<br>+tblock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64&sup3; PEC | 40000 | 198 | 512 | 544 | 513 | 937 | 1335 | 938 | 1322 | 505 | 504 |
| 160&times;128&times;192 PEC | 3000 | 199 | 448 | 393 | 374 | 497 | 492 | 706 | 862 | 368 | 381 |
| 224&sup3; PEC | 1200 | 200 | 423 | 387 | 376 | 475 | 489 | 497 | 577 | 478 | 484 |
| 160&times;128&times;192 PML_8 | 1500 | 70 | 146 | 141 | 124 | 164 | 169 | 221 | 251 | 225 | 224 |
| 224&sup3; PML_8 | 600 | 83 | 168 | 133 | 137 | 183 | 186 | 205 | 227 | 270 | 269 |
| 224&sup3; Mur | 1000 | 173 | 358 | 321 | 286 | 388 | 406 | 339 | 382 | 503 | 502 |

### Hosts D–H — CPU only

Same matrix, same models, same binaries; no GPU columns because none of these
machines has a Vulkan driver. The pinned thread count per host is the one
§2.1 derives.

**D — compute5, Ryzen 9 3950X, 16C/32T, 4&times;16 MB L3**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 16t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 16t | fork<br>default | fork<br>+tblock 16t | fork<br>+tblock default |
|---|---|---|---|---|---|---|---|---|---|
| 64&sup3; PEC | 40000 | 273 | 586 | 631 | 588 | 990 | 1588 | 964 | 1587 |
| 160&times;128&times;192 PEC | 3000 | 207 | 430 | 403 | 338 | 473 | 616 | 705 | 1153 |
| 224&sup3; PEC | 1200 | 212 | 422 | 390 | 348 | 440 | 460 | 513 | 661 |
| 160&times;128&times;192 PML_8 | 1500 | 73 | 184 | 153 | 96 | 175 | 177 | 292 | 397 |
| 224&sup3; PML_8 | 600 | 82 | 201 | 128 | 109 | 174 | 171 | 270 | 337 |
| 224&sup3; Mur | 1000 | 193 | 376 | 335 | 289 | 388 | 400 | 319 | 446 |

**E — loki, 2&times; Xeon Gold 6130, 32C/64T, 2&times;22 MB L3**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 32t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 32t | fork<br>default | fork<br>+tblock 32t | fork<br>+tblock default |
|---|---|---|---|---|---|---|---|---|---|
| 64&sup3; PEC | 40000 | 181 | 184 | 488 | 335 | 370 | 1002 | 374 | 1148 |
| 160&times;128&times;192 PEC | 3000 | 128 | 1057 | 473 | 221 | 1426 | 1242 | 909 | 1324 |
| 224&sup3; PEC | 1200 | 128 | 1083 | 371 | 200 | 1282 | 992 | 637 | 807 |
| 160&times;128&times;192 PML_8 | 1500 | 49 | 275 | 139 | 74 | 364 | 344 | 190 | 294 |
| 224&sup3; PML_8 | 600 | 55 | 396 | 100 | 77 | 432 | 309 | 160 | 261 |
| 224&sup3; Mur | 1000 | 108 | 608 | 271 | 153 | 835 | 696 | 221 | 455 |

**F — maxwell, 2&times; Xeon E5-2630, 12C/24T, no AVX2**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 12t | upstream<br>default | every fork AVX2 row |
|---|---|---|---|---|---|
| 64&sup3; PEC | 40000 | 100 | 192 | 222 | **SIGILL** |
| 160&times;128&times;192 PEC | 3000 | 99 | 549 | 283 | **SIGILL** |
| 224&sup3; PEC | 1200 | 98 | 566 | 238 | **SIGILL** |
| 160&times;128&times;192 PML_8 | 1500 | 33 | 191 | 75 | **SIGILL** |
| 224&sup3; PML_8 | 600 | 37 | 231 | 59 | **SIGILL** |
| 224&sup3; Mur | 1000 | 77 | 407 | 157 | **SIGILL** |

All 30 fork rows on this host exit with status 132 — 128 + SIGILL — including
the two `default` ones. §5.8. What the fork's *portable* engine does there,
which is the question the crash hides, is in §5.8's second table.

**G — saturn, Core i5-8500, 6C/6T, 9 MB L3**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 2t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 2t | fork<br>default | fork<br>+tblock 2t | fork<br>+tblock default |
|---|---|---|---|---|---|---|---|---|---|
| 64&sup3; PEC | 40000 | 166 | 430 | 530 | 436 | 772 | 1144 | 782 | 1128 |
| 160&times;128&times;192 PEC | 3000 | 124 | 210 | 206 | 226 | 230 | 230 | 474 | 531 |
| 224&sup3; PEC | 1200 | 123 | 208 | 201 | 221 | 225 | 225 | 303 | 332 |
| 160&times;128&times;192 PML_8 | 1500 | 53 | 78 | 78 | 71 | 74 | 75 | 140 | 144 |
| 224&sup3; PML_8 | 600 | 59 | 88 | 83 | 79 | 83 | 84 | 142 | 141 |
| 224&sup3; Mur | 1000 | 110 | 182 | 175 | 179 | 192 | 190 | 265 | 259 |

**H — tierwater, 2&times; Xeon E5-2620 v3, 12C/24T, 2&times;15 MB L3**

| Model | TS | upstream<br>SSE 1t | upstream<br>SSE-MT 12t | upstream<br>default | fork<br>AVX2 1t | fork<br>AVX2-MT 12t | fork<br>default | fork<br>+tblock 12t | fork<br>+tblock default |
|---|---|---|---|---|---|---|---|---|---|
| 64&sup3; PEC | 40000 | 143 | 238 | 175 | 338 | 449 | 364 | 446 | 358 |
| 160&times;128&times;192 PEC | 3000 | 114 | 430 | 244 | 194 | 519 | 440 | 583 | 461 |
| 224&sup3; PEC | 1200 | 114 | 447 | 201 | 187 | 514 | 422 | 365 | 320 |
| 160&times;128&times;192 PML_8 | 1500 | 41 | 145 | 61 | 62 | 168 | 150 | 148 | 133 |
| 224&sup3; PML_8 | 600 | 46 | 175 | 67 | 66 | 190 | 156 | 118 | 112 |
| 224&sup3; Mur | 1000 | 87 | 349 | 136 | 125 | 383 | 332 | 204 | 219 |

Two rows in there are worth pausing on because they are not engine results:

- **loki's 64&sup3; row collapses when pinned** — 184 MC/s at 32 threads
  against 488 for the same build left to auto-tune. A 6 MB grid spread over 32
  cores on two sockets is barrier traffic and cross-socket coherence, nothing
  else; both builds suffer it equally, so the column is still a controlled
  comparison, but the absolute number says more about the pinning than the
  engine. The `default` columns are the ones to read on that row.
- **tierwater and loki's `+tblock` columns are below their flat ones** on the
  large grids — 365 against 514, 637 against 1282. That is the §5.7 defect, not
  a property of the schedule; corrected, the same configurations reach 569 and
  1280.

Raw output for all seven hosts, including every individual repeat and the
engine's own "temporal blocking active / disabled — …" line for each blocked
row, is committed under `python/Tests/results/`:
`fork_vs_upstream_hostC_2026-09-07.json`, `…_hostA_2026-09-07.json`,
`…_hostA_2026-09-08.json` and
`fork_vs_upstream_{compute5,loki,maxwell,saturn,tierwater}_2026-09-11.json`.
The thread-count sweeps behind §2.1 are beside them as
`calibration_<host>_2026-09-11.txt`.

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
width shows up directly — and it shows up on both machines:

| Model | Host C<br>SSE → AVX2 | | Host A<br>SSE → AVX2 | |
|---|---|---|---|---|
| 64&sup3; PEC | 476 → 972 | 2.04&times; | 198 → 513 | 2.59&times; |
| 160&times;128&times;192 PEC | 267 → 494 | 1.85&times; | 199 → 374 | 1.88&times; |
| 224&sup3; PEC | 247 → 449 | 1.82&times; | 200 → 376 | 1.88&times; |
| 160&times;128&times;192 PML_8 | 111 → 148 | 1.34&times; | 70 → 124 | 1.77&times; |
| 224&sup3; PML_8 | 112 → 147 | 1.31&times; | 83 → 137 | 1.65&times; |
| 224&sup3; Mur | 219 → 364 | 1.66&times; | 173 → 286 | 1.65&times; |

The 1.82–2.59&times; on PEC is close to what doubling the vector width can
give. On Host C the PML rows drop to 1.31–1.34&times; because the PML passes
are a larger share of the work and vectorize less well — see §5.1, where this
becomes a problem at high thread counts. On Host A the same rows hold
1.65–1.77&times;, which is the clearest single sign that §5.1 is a
memory-system effect and not a property of the kernel: the slower machine, with
proportionally more compute per byte of bandwidth, keeps the vector win the
faster machine loses.

### 4.2 Trapezoidal temporal blocking on the CPU

This was the largest CPU result in this document, and the reasoning behind it is
worth stating because it is not "make the arithmetic faster".

The multithreaded AVX2 engine **is already at the DRAM roofline.** Traffic
measured with the core PMU on Host C is 31.4 bytes per cell per timestep
against a 24 B/cell ideal, which at its optimum is ~48 GB/s against a ~49 GB/s
STREAM-Triad ceiling. There is no arithmetic left to win and no scattered
access left to tile. (Traffic and roofline figures in this section are from
[OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.5 and §4.7, measured separately on that
same host; the throughput tables below are from this run.)

What is left is to *stop going to DRAM*. A trapezoidal schedule advances a
cache-resident tile of the grid through *k* consecutive timesteps before moving
on, which divides DRAM traffic by roughly *k*. Measured on a proxy benchmark:
30.95 → 2.40 B/cell, **12.9&times; less traffic.** In the real engine:

| Model | field state | Host C<br>flat → blocked | | Host A<br>flat → blocked | |
|---|---|---|---|---|---|
| 64&sup3; PEC | 6 MB | 3507 → 3458 | 0.99&times; | 937 → 938 | 1.00&times; |
| 160&times;128&times;192 PEC | 94 MB | 889 → 3249 | 3.66&times; | 497 → 706 | 1.42&times; |
| 224&sup3; PEC | 270 MB | 632 → 2607 | 4.13&times; | 475 → 497 | 1.05&times; |
| 160&times;128&times;192 PML_8 | 94 MB | 266 → 800 | 3.00&times; | 164 → 221 | 1.34&times; |
| 224&sup3; PML_8 | 270 MB | 237 → 828 | 3.50&times; | 183 → 205 | 1.12&times; |
| 224&sup3; Mur | 270 MB | 569 → 1848 | 3.25&times; | 388 → 339 | **0.87&times;** |

(Both columns at each host's pinned thread count, so the comparison is
controlled. Host A's blocked schedule prefers 6 threads and reaches
1.18&times;–1.75&times; when allowed them on the four models where it helps at
all; on Mur it still loses, 406 → 382. See §2.1 and the `+tblock default`
column in §3.)

The five hosts added later put the same comparison on a wider range of cache
topologies, and split it in two. Flat → blocked at each host's pinned thread
count, 224&sup3; PEC:

| host | last-level cache | flat → blocked | | |
|---|---|---|---|---|
| C | 36 MB, one cache | 632 → 2607 | **4.13&times;** | |
| G saturn | 9 MB, one cache | 225 → 303 | 1.35&times; | |
| A | 8 MB, one cache | 475 → 497 | 1.05&times; | |
| D compute5 | 4 &times; 16 MB | 440 → 513 | 1.17&times; | → **2.27&times;** corrected |
| H tierwater | 2 &times; 15 MB | 514 → 365 | **0.71&times;** | → 1.11&times; corrected |
| E loki | 2 &times; 22 MB | 1282 → 637 | **0.50&times;** | → 1.00&times; corrected |

The first three rows are the story §5.4 already tells: gains scale with how big
a tile the cache allows, and 8–9 MB is near the bottom of what pays. The last
three are a different story, and the "corrected" column is the point — those
are the same hosts with the tile width set by hand to the cache their threads
actually have, and that is §5.7. Taken at face value the bottom two rows say
temporal blocking is a 30–50% *regression* on a dual-socket server; corrected,
they say it is roughly break-even there. Neither is the 3–4&times; of Host C,
but the difference between "loses half your throughput" and "does nothing"
matters, and only one of them is a property of the schedule.

The 64&sup3; row is the schedule declining to engage, and saying so:

```
AVX2 temporal blocking: disabled, only 64 x-lines for a tile width of 384.
```

The tile width is derived from the last-level cache, and at this grid's small
x-planes that comes out wider than the grid itself. There is nothing to gain
here anyway — 6 MB of field state is already L3-resident, and blocking a
cache-resident grid was measured to *lose* 32% — but the guard that fires first
is the geometric one. Either way the run falls back to the flat sweep, which is
why the row reads ~1.00&times; rather than a regression.

Three properties make this usable rather than merely fast:

- **It is bit-identical to the flat sweep.** Not "within tolerance" —
  element-for-element identical. `python/Tests/test_temporal_blocking.py`
  checks this for PEC, PML, Mur, Drude, lumped RLC, a local absorbing sheet, a
  TF/SF plane wave, a combined case, and a sweep over
  (k,W) &isin; {2:8, 4:16, 3:13, 6:20} &times; {1,3,4} threads; all 9 cases pass
  on the build measured here. Each case asserts that blocking actually
  *engaged* before comparing fields — otherwise the comparison passes vacuously
  every time an extension declines and the run silently falls back.
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

### 4.3 Trapezoidal temporal blocking on the GPU — new

The same schedule now runs in `Engine_Vulkan`, enabled by
`OPENEMS_GPU_TEMPORAL_BLOCK=<k>`. §4.4 concluded that the GPU Yee kernel is at
roofline and that further gains have to come from *reducing traffic* rather
than from tightening arithmetic. This is that reduction, and it is the largest
new result in this revision:

| Model | field state | RX 6800<br>flat → blocked | | UHD 770<br>flat → blocked | | HD 530<br>flat → blocked | |
|---|---|---|---|---|---|---|---|
| 64&sup3; PEC | 6 MB | 12433 → 12385 | 1.00&times; | 633 → 633 | 1.00&times; | 505 → 504 | 1.00&times; |
| 160&times;128&times;192 PEC | 94 MB | 13070 → 12950 | 0.99&times; | 524 → 611 | 1.17&times; | 368 → 381 | 1.04&times; |
| 224&sup3; PEC | 270 MB | 5674 → **12319** | **2.17&times;** | 520 → 601 | 1.16&times; | 478 → 484 | 1.01&times; |
| 160&times;128&times;192 PML_8 | 94 MB | 9462 → 9844 | 1.04&times; | 317 → 344 | 1.08&times; | 225 → 224 | 1.00&times; |
| 224&sup3; PML_8 | 270 MB | 4619 → **10212** | **2.21&times;** | 323 → 347 | 1.08&times; | 270 → 269 | 1.00&times; |
| 224&sup3; Mur | 270 MB | 5125 → 5125 | 1.00&times; | 518 → 518 | 1.00&times; | 503 → 502 | 1.00&times; |

Read the RX 6800 column top to bottom and the mechanism is visible directly.
The card has a 128 MB Infinity Cache. The two small grids are already inside it
and there is nothing to save — and on the 94 MB PEC row the schedule engages
anyway and loses 1%, which is a defaulting bug and is §5.2. The two 270 MB
grids are outside the cache, and those are
exactly the two rows that gain **2.2&times;** — which erases the throughput
cliff at the cache boundary: 224&sup3; PEC went from 5674 (below it) to 12319,
which is within 6% of the 13070 the 94 MB grid reaches *inside* the cache. The
blocked schedule makes a grid that does not fit run at close to the speed of one
that does. That is the whole claim of the method, and this is the cleanest
demonstration of it in this document.

Three device classes, three outcomes, all consistent with the same explanation:

- **RX 6800 (discrete, bandwidth-bound, 128 MB cache)** — 2.2&times; where the
  grid exceeds the cache, nothing where it does not.
- **UHD 770 (integrated, shares 36 MB of CPU L3)** — a flat 1.08–1.17&times;
  everywhere above 6 MB. Its cache is small enough that every real grid is
  outside it, so the gain does not switch on and off with grid size; but the
  device is partly latency-bound, so the gain is small.
- **HD 530 (integrated, Gen9)** — nothing, anywhere. `clpeak`-class bandwidth
  is not what limits this part; it is 24 execution units. Removing DRAM traffic
  from a compute-bound kernel removes nothing that was in the way.

The correctness apparatus is the same as the CPU's, and for the same reason.
`python/Tests/test_gpu_temporal_blocking.py` checks the blocked sweep
**bit-identical** to the flat one for PEC, fused UPML, unfused UPML, a
dispersive material with UPML, and a sweep of (k,W) schedules including one
where W does not divide NX — 11 tests, all passing on the build measured here.
Two of them exist specifically to close gaps the others cannot see: one asserts
the flat reference is not all zeros (a comparison against zero passes no matter
what), and one checks the fused and unfused PML paths against *each other*,
because each of the first two compares a path against itself and both would
pass if the slab range were wrong in a way they shared.

The Mur row is the refusal mechanism working:

```
Engine_Vulkan: temporal blocking disabled — Mur ABC has no slab path yet
```

Mur, TF/SF, lumped RLC and GPU probes have no slab path on the GPU yet, so any
model using them falls back to the flat sweep and reports the flat number. What
remains to be done, in the order that pays, is in
`plans/GPU_TEMPORAL_BLOCKING_PLAN.md`.

### 4.4 The Vulkan GPU engine

A full Vulkan 1.3 compute implementation — Yee update, UPML, Mur, excitation,
TF/SF, dispersive materials, lumped RLC, energy reduction and probe gather all
run as compute shaders, with the operator compressed on the device.

| Model | RX 6800<br>best | vs upstream best | UHD 770<br>best | vs upstream best | HD 530<br>best | vs Host A<br>upstream best |
|---|---|---|---|---|---|---|
| 64&sup3; PEC | 12433 | 6.10&times; | 633 | 0.31&times; | 505 | 0.93&times; |
| 160&times;128&times;192 PEC | 13070 | 16.57&times; | 611 | 0.78&times; | 381 | 0.85&times; |
| 224&sup3; PEC | 12319 | 21.01&times; | 601 | 1.03&times; | 484 | 1.14&times; |
| 160&times;128&times;192 PML_8 | 9844 | 35.15&times; | 344 | 1.23&times; | 225 | 1.54&times; |
| 224&sup3; PML_8 | 10212 | 38.98&times; | 347 | 1.33&times; | 270 | 1.61&times; |
| 224&sup3; Mur | 5125 | 9.66&times; | 518 | 0.98&times; | 503 | 1.40&times; |

The changes that produced those numbers, in order of what they were worth:

- **Trapezoidal temporal blocking** (`99ac18a`, `d9a6431`, `88032e4`) —
  **2.2&times;** on grids exceeding the card's cache. §4.3.
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

**The flat GPU Yee kernel is at roofline.** Counting traffic from the shader
source rather than assuming it gives 74 B/cell/timestep
([OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.7/C4), and the flat 5674 MC/s
&times; 74 B = **420 GB/s** against a `clpeak` ceiling of 477 GB/s on this
card — **88% of achievable bandwidth**. That is precisely why §4.3 was worth
doing and why it works: the only way past a roofline is to stop needing the
resource it measures.

### 4.5 Correctness work that came with it

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

### 5.1 The AVX2 multithreaded engine loses to SSE on PML — on Host C only

At 8 threads on a PML model, the AVX2 engine is **5–10% slower** than the SSE
engine it was meant to replace — despite being 1.31&times; faster
single-threaded on the same model.

| Model | upstream SSE-MT | fork AVX2-MT | fork AVX2-MT<br>+ temporal blocking |
|---|---|---|---|
| 160&times;128&times;192 PML_8 | 280 | 266 | 800 |
| 224&sup3; PML_8 | 262 | 237 | 828 |
| 224&sup3; PEC | 586 | 632 | 2607 |

Both engines are bandwidth-bound there, and the AVX2 kernel's wider per-lane
footprint buys nothing once DRAM sets the pace, while its PML passes cost
slightly more. That this is a memory-system effect and not a kernel defect is
now directly confirmed: on **Host A the same comparison never inverts** —
164 vs 146 and 183 vs 168 MC/s, the AVX2 engine ahead by 9–13% on exactly the
models where it falls behind on Host C. Four cores cannot saturate DDR4 the way
eight P-cores saturate DDR5, so the crossover never happens.

The right fix is not more vectorization — temporal blocking removes the
constraint entirely and takes the same case to 800–828 MC/s.

**Practical guidance:** on a many-core host, on a PML-dominated model, without
temporal blocking, `--engine=multithreaded` is the better choice.

### 5.2 GPU temporal blocking engages when it should not, and costs ~5%

The tile target is a **tunable with a per-device-class default** — 64 MB
discrete, 16 MB integrated — because Vulkan exposes no last-level-cache size to
query. On the RX 6800 the real cache is 128 MB, so a grid between 64 and 128 MB
is already cache-resident but still looks too large to the guard. It engages,
adds the trapezoid's wedge overhead, and saves traffic that was not being spent:

| 160&times;128&times;192 PEC, RX 6800 | result |
|---|---|
| flat | 13070 |
| blocked, default 64 MB target | 12950 |
| blocked, `OPENEMS_GPU_TEMPORAL_BLOCK_MB=64` (single run) | 12508 |
| blocked, `OPENEMS_GPU_TEMPORAL_BLOCK_MB=128` (single run) | **13143** — *refuses, correctly* |

Told the card's actual cache size, the guard declines and full flat throughput
comes back. Told the default, it costs about 1% in the best-of-3 matrix row and
about 5% in the direct single-run A/B above. Either way it is a loss, and it is
avoidable per host with one environment variable.

This is a defaulting problem, not a schedule problem, and the fix is known: a
device-ID table or a startup micro-probe instead of a constant. Until then, on
a card with an unusually large last-level cache, set
`OPENEMS_GPU_TEMPORAL_BLOCK_MB` to it.

### 5.3 The integrated GPUs are not compute devices — but the comparison is relative

On Host C the UHD 770 measures **317–633 MC/s on every grid**, essentially flat
with grid size, i.e. **0.31&times;–1.33&times;** upstream's best CPU
configuration. That flatness has a cause: `clpeak` reports 48.7 GB/s for it
against the CPU's ~49 GB/s STREAM-Triad ceiling, because the iGPU and the CPU
*are the same memory system*, so there is no cache hierarchy of its own for
grid size to interact with.

The Host A result is the same device class against a much weaker CPU, and it
lands differently. The HD 530 is **slower in absolute terms** than the UHD 770
on every one of the six models — and yet on two of them, 224&sup3; PML_8 and
224&sup3; Mur, it is the **fastest engine on its machine**, beating every CPU
configuration including the blocked one. Against upstream's best it reaches
1.54&times;–1.61&times; on the PML models and 1.40&times; on Mur. An iGPU is not
fast; it is simply not competing with much on a four-core desktop part, and it
does not slow down when a PML is added the way the CPU does (224&sup3;:
478 → 270 MC/s on the GPU, 423 → 168 on the CPU).

The qualifier matters. On the smaller 160&times;128&times;192 PML_8 model the
blocked CPU schedule reaches 251 MC/s against the GPU's 225, so the GPU is
*not* simply the right answer for PML on this machine — it wins where the grid
is large enough that the CPU's 8 MB L3 has stopped helping.

So the honest statement is not "integrated GPUs are useless" but: **an
integrated GPU is worth trying whenever the CPU it shares a die with is weak
and the model is both PML-dominated and large.** On Host C it remains a portability check on a
second Vulkan driver — ANV rather than RADV, which is how driver-specific
assumptions get caught — and little more. Device selection already prefers the
discrete GPU where one exists.

### 5.4 CPU temporal blocking needs a large — and undivided — last-level cache

On Host A's 8 MB L3, the schedule's gains collapse and one model regresses:

| Model | tile geometry chosen | flat → blocked (4t) | |
|---|---|---|---|
| 160&times;128&times;192 PEC | k=7, W=14, 11 tiles, 7.9 MB/tile | 497 → 706 | 1.42&times; |
| 224&sup3; PEC | k=3, W=6, 37 tiles, 6.9 MB/tile | 475 → 497 | 1.05&times; |
| 224&sup3; PML_8 | k=3, W=6, 37 tiles, 6.9 MB/tile | 183 → 205 | 1.12&times; |
| 224&sup3; Mur | k=3, W=6, 37 tiles, 6.9 MB/tile | 388 → 339 | **0.87&times;** |

Compare Host C, where the same models get k=15–16 and 31–64-line tiles. The
tile width is derived from the L3, so a 4.5&times; smaller cache buys a
4.5&times; narrower tile, and the depth *k* has to shrink with it to satisfy
the geometric `W >= 2k` constraint. At k=3 the schedule is paying the
trapezoid's full wedge and re-traversal overhead to divide DRAM traffic by
three, and on the Mur model — whose boundary work is per-face and does not
shrink with the tile — that trade is a net loss.

The schedule is opt-in (§5.5), so this costs nothing unless it is asked for.
But it does mean the 3–4&times; figures in §4.2 should be read as a property of
a 36 MB L3, not of the method: **check it on your own hardware before relying
on it**, which is a two-command exercise (§7).

saturn (Host G) is the same story on a 9 MB L3 and confirms it —
k=3, W=7, 8.0 MB per tile, and 1.35&times; on 224&sup3; PEC. What it adds is
that the floor is not as low as Host A suggested: at the same tile geometry
saturn gains where Host A barely did, so "small cache" is not on its own a
prediction. The variable this section names — *cache size* — turned out to be
the wrong one, or at least an incomplete one. §5.7 names the other half.

### 5.5 Temporal blocking is opt-in, and capped by probe cadence

Both schedules are prototypes behind environment variables —
`OPENEMS_AVX2_TEMPORAL_BLOCK=<k>` and `OPENEMS_GPU_TEMPORAL_BLOCK=<k>` (here,
k=16 on the CPU and on the RX 6800, k=6 on the integrated devices) — and do
nothing unless set. Two limits are worth knowing before relying on the numbers
in §4.2 and §4.3:

- **Block depth is capped by the probe interval.** `openems.cpp` calls
  `IterateTS()` with the number of timesteps until the next probe or dump, and
  both schedules clamp *k* to it — which is what keeps probes and dumps correct
  with no work on their side. That interval is typically 5–25 timesteps, which
  happens to bracket the measured optimum (*k* = 6–24). A run oversampled far
  past Nyquist will not reach these figures.
- **These models have no probes or dumps at all**, so they see the ceiling of
  what blocking can do. A probe-dense model will land lower.

### 5.6 Thread auto-tune: fixed here, still worth knowing

Upstream's thread search hill-climbs inside `NextInterval()`, which is called
only after four seconds of wall time, adding one thread per call starting from
one. On a 4-core machine it converges quickly; on a 32-thread machine reaching
the optimum takes ~44 s, during which the run is nowhere near peak
([OPTIMIZATIONS.md](OPTIMIZATIONS.md) §4.2). `fd555ee` replaces it with a
search that scores candidates on real timesteps inside `IterateTS()` —
legitimate because thread count cannot change results — settling in ~250
timesteps instead. One detail mattered more than expected: the first batch after
each thread respawn must be discarded, or cold private caches penalise exactly
the wide configurations under judgement.

How close each build's default gets to its own pinned optimum:

| Model | Host C<br>upstream | Host C<br>fork | Host A<br>upstream | Host A<br>fork |
|---|---|---|---|---|
| 64&sup3; PEC | 47% | 83% | 100% | 100% |
| 160&times;128&times;192 PEC | 72% | 97% | 88% | 99% |
| 224&sup3; PEC | 85% | 97% | 91% | 100% |
| 160&times;128&times;192 PML_8 | 74% | 96% | 97% | 100% |
| 224&sup3; PML_8 | 69% | 95% | 79% | 100% |
| 224&sup3; Mur | 81% | 95% | 90% | 100% |

The two hosts together are the whole argument. On Host C upstream's default
leaves 15–53% on the table and the fork's lands within 3–5%. On Host A —
4 cores, exactly the shape of machine upstream's search was written for —
upstream loses at most 21% and the fork reaches 99–100%. The fix is worth most
where the old search was worst, which is the many-core case, and costs nothing
where it was already fine.

Note what this means for reading §3: **comparing the two `default` columns
measures the auto-tune as much as the engines.** That is why the pinned columns
exist.

On Hosts D–H the auto-tune is no longer uniformly the safer choice, and on the
two biggest machines it is the *worse* one: loki's default reaches 992 MC/s on
224&sup3; PEC against 1282 pinned, and tierwater's 422 against 514. Both are
dual-socket, and the search scores thread counts without any notion of socket
placement, so it settles on a count that a NUMA-aware placement would beat. In
the other direction the 64&sup3; row on loki has the default at 1002 against
370 pinned, because there the pinned count is the bad choice (§3). The honest
summary is that on a dual-socket host neither column is reliably the one a user
should expect; on the single-socket hosts D and G the default lands within
0–5% of pinned, as it does on A and C.

### 5.7 The blocking tile is sized from one cache, not from the cache the run has

This is the largest defect these five hosts exposed, and it is worth up to
**2&times;**.

The tile width comes from `sysconf(_SC_LEVEL3_CACHE_SIZE)`
(`FDTD/engine_avx2_multithread.cpp`), which reports the size of **one**
last-level cache instance. On a machine whose LLC is a single shared cache —
Hosts A, C and G — that is also the cache the whole run has, and the derived
width is right. On a machine whose LLC is partitioned it is off by the number
of partitions the threads span: a quarter of the truth on a Ryzen 3950X
(4 CCXs), a half on any dual-socket server.

The tile is divided across cores in x, so each core's slab lands in its own
cache instance; the capacity a run has is the *sum* over the instances its
threads occupy, not one of them. Sweeping the width by hand
(`OPENEMS_AVX2_TEMPORAL_BLOCK=k:W`) against thread count says so directly —
224&sup3; PEC, MC/s, single runs:

| host | LLC the threads span | auto (derived) | best hand-set | |
|---|---|---|---|---|
| C, 8t | 36 MB &times; 1 | **2544** (35.6 MB) | 2160 (52.8 MB) | auto is best |
| G saturn, 2t | 9 MB &times; 1 | **306** (8.0 MB) | 274 (18.4 MB) | auto is best |
| G saturn, 6t | 9 MB &times; 1 | **335** (8.0 MB) | 252 (18.4 MB) | auto is best |
| H tierwater, 6t — one socket | 15 MB &times; 1 | **363** (14.9 MB) | 326 (29.9 MB) | auto is best |
| H tierwater, 12t — two sockets | 15 MB &times; 2 | 356 (14.9 MB) | **569** (44.8 MB) | **1.60&times;** |
| D compute5, 4t — one CCX | 16 MB &times; 1 | **1133** (14.9 MB) | 600 (29.9 MB) | auto is best |
| D compute5, 16t — four CCXs | 16 MB &times; 4 | 504 (14.9 MB) | **1000** (59.7 MB) | **1.99&times;** |
| E loki, 16t — one socket | 22 MB &times; 1 | 983 (21.8 MB) | **1430** (43.6 MB) | **1.46&times;** |
| E loki, 32t — two sockets | 22 MB &times; 2 | 636 (21.8 MB) | **1280** (59.7 MB) | **2.01&times;** |

Every row where the thread set fits inside one cache instance picks the derived
width, and every row where it spans several wants a wider one. tierwater makes
the point without leaving the machine: the *same* binary on the *same* model
has the derived width optimal at 6 threads on one socket and 60% short at
12 threads across two.

Two details keep this from being a one-line fix. The optimum tracks the
aggregate capacity but not exactly — loki at 16 threads on a single 22 MB
socket prefers a 43.6 MB tile, which is explained by Skylake-SP's
**non-inclusive** L3 and 1 MB private L2 per core (16 &times; 1 + 22 ≈ 38 MB of
real capacity, not 22), so the right quantity is aggregate capacity across the
whole hierarchy the threads own, not the L3 line from `sysconf`. And the
trapezoid's live set is smaller than *W* planes — it shrinks from *W* to
*W*&nbsp;−&nbsp;2*k* as the tile advances — so the mapping from capacity to *W*
already carries a fudge factor that these hosts show is not universal.

**Until it is fixed, one environment variable recovers it per host.** Set the
width explicitly to about (number of LLC instances the run spans) &times; the
derived width:

```bash
OPENEMS_AVX2_TEMPORAL_BLOCK=16:52 openEMS model.xml --engine=avx2-multithreaded
```

Raw sweep output is `python/Tests/results/tile_width_sweep_2026-09-11.txt`.
This is the CPU twin of §5.2, which is the same mistake on the GPU: a tile
target that is a constant where it should be a property of the device and of
how much of the device the run is using.

### 5.8 The AVX2 engine has no runtime ISA guard — it does not fall back, it crashes

maxwell (Host F) is a 2&times; Xeon E5-2630, Sandy Bridge: SSE4.2 and AVX, no
AVX2 and no FMA. On it, **every** fork configuration in the matrix terminates
with SIGILL — exit status 132 — including `--engine=fastest`, which is what a
user gets by typing nothing:

```
$ openEMS model.xml
...
Create FDTD operator (AVX2 + FMA + multi-threading)
Illegal instruction (core dumped)        # exit 132
```

It does not even reach the stepping loop: the default path dies in operator
construction. Asked for `--engine=avx2` explicitly it gets one line further,
printing `Create FDTD engine (AVX2 + FMA, compressed flat arrays)` and
`Running FDTD engine...` before the same signal.

The guard that exists (`987fcb7`) is `OPENEMS_ENABLE_AVX2`, set by CMake from
`CMAKE_SYSTEM_PROCESSOR` and a `check_cxx_compiler_flag("-mavx2 -mfma")` probe.
Both questions are about the **build** machine and its compiler, not about the
machine that will run the binary. So the guard does what it was written for —
it keeps `<immintrin.h>` out of an ARM build — and does nothing at all for the
case that actually bites: an x86-64 binary built anywhere with a modern GCC and
run on a pre-Haswell host. `openems.cpp` then makes AVX2-multithreaded the
default engine unconditionally, and the `--engine=avx2` fallback message
(`"AVX2+FMA engine is unavailable on this platform; using multithreaded
engine"`) is inside `#if OPENEMS_ENABLE_AVX2 … #else`, so on this binary it is
compiled out.

This matters beyond one old server: it is every distribution package, every
container image and every shared cluster filesystem — anywhere the build host
and the run host are not the same machine.

**The capability is not missing, only the dispatch.** Asking maxwell for the
portable engine explicitly, on both builds, at 12 threads:

| 224&sup3; model | upstream `multithreaded` | fork `multithreaded` | |
|---|---|---|---|
| PEC | 556 | 590 | 1.06&times; |
| PML_8 | 228 | 226 | 0.99&times; |
| Mur | 381 | 401 | 1.05&times; |

So the fork is a small gain on this host, from the host-side work in
[OPTIMIZATIONS.md](OPTIMIZATIONS.md) §1.1 and the thread auto-tune — it just
cannot be reached without `--engine=multithreaded` on the command line. The fix
is a runtime check (`__builtin_cpu_supports("avx2") && …("fma")`) at engine
selection, falling back the way the compile-time path already does; that is a
code change and has not been made here. Raw output:
`python/Tests/results/maxwell_non_avx2_2026-09-11.txt`.

**Practical guidance until then:** on any pre-Haswell x86 host, pass
`--engine=multithreaded` explicitly, or build on that host so CMake's probe
matches it.

---

## 6. Operator build time

Time-stepping throughput is not the whole cost. Operator construction, measured
as wall clock outside the stepping loop:

| Model | | upstream SSE-MT | fork AVX2-MT | fork Vulkan |
|---|---|---|---|---|
| 224&sup3; PEC | Host C | 6.0 s | 5.1 s | 7.0 s |
| 224&sup3; PML_8 | Host C | 8.2 s | 5.6 s | 6.8 s |
| 224&sup3; Mur | Host C | 6.0 s | 5.1 s | 7.0 s |
| 224&sup3; PEC | Host A | 13.9 s | 12.9 s | 12.2 s |
| 224&sup3; PML_8 | Host A | 17.3 s | 14.3 s | 13.1 s |
| 224&sup3; Mur | Host A | 13.9 s | 12.6 s | 12.1 s |

Operator setup is parallelized in this fork, which is where the PML case's
8.2 s → 5.6 s comes from; on Host A's four cores the same change is worth
17.3 s → 14.3 s.

For the GPU, setup is the dominant fixed cost of a short run: on Host C at
224&sup3; the operator takes ~7.0 s while all 1200 timesteps take only ~1.1 s
with blocking on. Since the CPU pays ~6.0 s of setup too, the crossover is
early — the blocked GPU is ahead of upstream's best CPU configuration after
about **55 timesteps**, and ahead of the fork's own blocked CPU engine after
about **530**.

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

# 3. run the matrix (~2 h) -- Host C
python python/Tests/benchmark_fork_vs_upstream.py --out /tmp/bench.json

# 3'. the same matrix on a different machine: everything host-specific is a flag
python python/Tests/benchmark_fork_vs_upstream.py \
    --tuned-threads 4 --pin-single 0 --pin-tuned 0,2,4,6 \
    --gpu hd530:0:6 --csxcad-lib ~/openEMS/build-deps/install/lib \
    --out /tmp/bench-hostA.json
```

`--gpu LABEL:INDEX[:K]` is repeatable and adds both a flat and a
`+tblock` row for each device; `--gpu none` gives a CPU-only matrix. The script
writes incrementally and skips configurations already present in the output
file, so it can be interrupted and resumed. `--only <substring>` runs a single
cell of the matrix.

### 7.1 On a machine you cannot build on

Hosts D–H had no openEMS dependencies and could not have any installed, and one
of them is old enough (CentOS 7, glibc 2.17) that a binary from a current
distribution will not start. Both builds therefore come out of one container
and travel as a self-contained bundle:

```bash
# 1. build fork + upstream + every dependency inside a glibc-2.17 container.
#    ~40 min from cold; stages are stamped, so a rerun resumes.
python/Tests/build_portable_bench.sh            # -> ~/openems-portable/bundle

# 2. copy the bundle to the target and run it there. Nothing is installed;
#    the bundle only needs a glibc >= 2.17 x86-64 host.
rsync -a ~/openems-portable/bundle/ target:openems-bench/
ssh target 'cp -a ~/openems-bench /tmp/oebench && cd /tmp/oebench &&
            ./run_portable_bench.sh calib 4:4:0-3 8:8:0-7 12:12:0-11 &&
            ./run_portable_bench.sh matrix 12 0-11 0'
```

`run_portable_bench.sh calib <label>:<threads>:<cpulist> …` is the thread sweep
behind §2.1 and prints upstream-MT, fork-AVX2 and fork-+tblock at each point;
`matrix <threads> <pin_tuned> [pin_single]` then runs the full matrix with
`--gpu none` and writes `results-<hostname>.json`. Staging to node-local disk
matters if `$HOME` is on NFS, as it was here.

The bundle is deliberately boring to verify: `ldd` on either binary should show
only `libc`, `libm`, `libdl`, `libpthread` and the three project libraries, and

```bash
objdump -T bundle/fork/openEMS | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1
```

should print no more than the oldest target's glibc. Everything else — VTK,
HDF5, CGAL, Boost, TinyXML, fparser, GMP, MPFR, libstdc++ — is linked in
statically. GMP is built `--disable-assembly` on purpose: its configure script
otherwise selects assembly for the *build* host's microarchitecture, which is
exactly the portability bug this bundle exists to avoid.

**The toolchain differs from Hosts A and C** (GCC 10.2.1 against 16.2.1),
because glibc 2.17 constrains it. §2 shows that costs 0.98–1.04&times; in a
paired test, with no systematic direction. If you repeat that control, pair the
arms run by run — two matrix runs on different days measure the machine's
afternoon, not the compiler.

The raw output behind every table in this document is committed as
`python/Tests/results/fork_vs_upstream_hostC_2026-09-07.json` and
`python/Tests/results/fork_vs_upstream_hostA_2026-09-08.json`, including the
individual repeats and each blocked row's schedule announcement, so the
best-of-N choices and the engage/refuse decisions can be checked rather than
taken on trust. Host A's first, independently built run is kept beside it as
`fork_vs_upstream_hostA_2026-09-07.json` — the two are a full-matrix
reproducibility check, not a duplicate. The previous single-host revision is
preserved as `fork_vs_upstream_2026-09-05.json`.

Hosts D–H are
`fork_vs_upstream_{compute5,loki,maxwell,saturn,tierwater}_2026-09-11.json`,
with `calibration_<host>_2026-09-11.txt` for the thread sweeps,
`tile_width_sweep_2026-09-11.txt` for §5.7,
`maxwell_non_avx2_2026-09-11.txt` for §5.8 and
`toolchain_gcc16_vs_gcc10_2026-09-11.txt` for the toolchain control. The models
themselves are committed as `python/Tests/models/*.xml`, so all seven hosts
demonstrably ran the same bytes.

Related drivers: `python/Tests/benchmark_avx2_engine.py`,
`python/Tests/benchmark_gpu_engine.py` (`--gpu-index` selects the device),
`python/Tests/bench_temporal_blocking.c`. Correctness:
`python/Tests/test_temporal_blocking.py` and
`python/Tests/test_gpu_temporal_blocking.py`.

---

## 8. Measurement notes

Three hazards produced wrong numbers before they were understood, and all three
are worth repeating for anyone re-running this.

**Cache-resident grids are bimodal on Host C, for a reason not identified.**
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
explain. It reproduces on Host A's 64&sup3; rows as well (198 vs 162 MC/s
single-threaded in the run tabulated here, and 199 vs 146 in the repeat), on a
completely different microarchitecture, which weakens any explanation specific
to Raptor Lake.

Hosts D–H weaken it further: the same bimodality appears on the 64&sup3;
single-threaded rows of **maxwell** (100 / 100 / 100 / 78.5 / 99.6 / 81.7
MC/s), **tierwater** (142 / 142 / 143 / 106 / 143 / 97.7) and **saturn**
(128 / 166 / 160 / 166 / 160 / 142), and on saturn's `upstream default` row
(266 among five repeats at 521–530). That is now five microarchitectures —
Sandy Bridge, Haswell, Skylake, Coffee Lake, Raptor Lake — and it never
appears on a DRAM-bound row on any of them. Whatever it is, it is a property of
cache-resident FDTD work on x86 generally and not of any one part, which makes
the uncore-clock hypothesis more plausible without confirming it. compute5
(Zen 2) is the one exception: its 64&sup3; repeats are tight throughout.

**A second, unrelated source of spread on the 64&sup3; row** shows up on the
`default` columns of the dual-socket hosts, and should not be mistaken for the
above: upstream's thread auto-tune lands somewhere different on each run.
loki's six repeats of `upstream default` are 225 / 488 / 483 / 234 / 172 / 176
MC/s and maxwell's are 204 / 222 / 156 / 124 / 112 / 66 — a 2.2&times; and
3.4&times; spread, on the pinned-thread-count *sibling* of a row that is stable
to a percent. That is a search converging to different thread counts on a
cache-resident grid where the right count is small and the penalty for
overshooting is large, and it is the same weakness §5.6 describes; it is
visible here rather than there because the 64&sup3; grid punishes it hardest.
The fork's search is not immune on these hosts either — tierwater's
`fork tblock default` repeats spread 252–358 — but it is consistently
narrower.

What matters for reading this document is the scope, and that *is* established:
only the cache-resident grid is affected. Every DRAM-bound configuration is
stable to **0.5–4%** across repeats — visible directly in the committed JSON,
whose `samples` arrays are within a percent of each other on every large grid.
Hence 6 repeats on the 64&sup3; grid and 3 elsewhere, and hence the 64&sup3;
row should be read as "about this" while the rest can be read as measured.

**Host A was not quiesced.** It is a shared server and ran `rclone`, Jellyfin
and an idle BOINC client throughout, at a load average of 0.6–2.8. This is a
real caveat and it is not hidden: it means Host A's absolute numbers could be a
few percent low. It does **not** undermine what Host A is used for here, which
is entirely *ratios measured within the host* — AVX2 vs SSE, blocked vs flat,
GPU vs CPU — all of which are best-of-3 over interleaved runs that shared the
same background load. The one place to be careful is the 64&sup3; row, where
that noise compounds with the bimodality above: its 8-thread repeats spread
491–512 MC/s in the run tabulated here and 380–512 in the repeat, which is why
that row moved 1.4% between the two while every other row moved under 0.8%.
Nothing in §1 or §5 turns on it.

**Hosts D–H were idle**, which is the one thing that makes them easier to read
than Hosts A and C. All five are servers with no desktop session; load average
was 0.00–0.19 before each run started, they ran one matrix at a time, and the
binaries were staged to node-local disk so that no NFS traffic could land in a
measurement. loki had other users logged in but none of them running anything.

**Do not compare a matrix run on one day against one on another to answer a
build question.** That was tried here — the portable bundle against the
2026-09-07 Host C run, to check the compiler — and it produced a confident,
entirely spurious 15–29% deficit on the PEC rows, because the workstation was
busy that afternoon and the PEC rows are the most bandwidth-sensitive in the
matrix. The PML and Mur rows of the same comparison agreed to 2–5%, which is
exactly how such an artefact looks when it is not recognised: partly
consistent. The paired alternating test in §2 put the same two builds within
0.98–1.04&times;. The general lesson is the one already implicit in §2.1's
pinning: when the question is about the binary, vary only the binary.

**Never A/B a GPU change by disabling the excitation.** With no source the
fields stay exactly zero, and an all-zero grid runs ~28% faster than the
identical kernel on real data — zero operands flip almost no bits. This nearly
produced a fake 26% win once. Both arms of any comparison must carry real field
data.

**A harness bug worth naming**, because it silently produced missing data
rather than wrong data: the regex that reads the loop time out of openEMS's
output accepted `1200` and `262144.00` but not `1.12394e+07`, and the cell
count prints in scientific notation exactly when the run is too fast to emit an
interval line first — that is, on every GPU run on a large grid. The effect was
that `setup_s` came back `None` on precisely the rows with the most setup to
report. Fixed in this revision; §6's GPU column is the first that did not have
to be measured by hand.
