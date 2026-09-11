<title>Precision Abstraction Plan</title>

# Multi-Precision openEMS — Abstraction & Implementation Plan

**Goal:** make the FDTD scalar type a runtime choice (`fp16` / `fp32` / `fp64`) instead of a
compile-time typedef, in two stages: first a clean abstraction with *zero* behaviour change,
then the multi-type implementations.

**Explicitly out of scope:** fixed-point / integer quantization. The dynamic range of E&M
field decay across a simulation is logarithmic; floating-point is the right representation.

---

## 0. Where we actually stand

Reconnaissance of the tree, because the plan depends on it.

| Fact | Location | Consequence |
|---|---|---|
| `using FDTD_FLOAT = float;` | `tools/constants.h:21` | The only nominal knob — and it is leaky. |
| 317 `FDTD_FLOAT` sites | tree-wide | Already-abstracted surface. Good. |
| 155 raw `float` in `FDTD/` + `Common/` | sim path | The leak. Must be converted. |
| 245 raw `float` in HDF5/VTK/nf2ff/SAR | I/O path | **Must NOT be converted** — see §1.2. |
| `ArrayLib::ArrayNIJK<T>` already templated | `tools/arraylib/array_nijk.h:34` | Storage layer is *free*. Biggest single win. |
| `Engine_Interface_Base` is already all-`double` | `Common/engine_interface_base.h:52-82` | The precision-agnostic public seam **already exists**. |
| `processfields_calc.h` promotes to `double` | `Common/processfields_calc.h:59` | Post-processing convention is already "compute wide". |
| Extensions use `FDTD_FLOAT` + virtual accessors | `FDTD/extensions/*` | ~18 of 20 extensions need **no changes**. |
| Only cylinder extensions touch SIMD directly | `engine_ext_cylinder.*`, `engine_ext_cylindermultigrid.cpp`, `engine_extension_dispatcher.h` | Contains the SIMD blast radius to 3 files. |
| `f4vector` = hardcoded 4×`float` union | `tools/array_ops.h:39-47` (89 sites) | Hard blocker for SIMD engines. |
| `f8vector` = hardcoded `__m256` | `FDTD/operator_avx2.h:29` (44 sites) | Same. |
| `numVectors = ceil(numLines[2]/4.0)` **duplicated** | `operator_sse.cpp:88` **and** `engine_sse.cpp:40` | Lane count hardcoded in two places that must agree. Latent bug today. |
| `EC_C/G/L/R` stored `FDTD_FLOAT`, arithmetic in `double` | `operator.h:349-352`, `operator.cpp:964-982` | **Existing precision loss** during operator build. Independently worth fixing. |
| All 25 GPU shaders declare `float` buffers | `FDTD/gpu/shaders/*.comp` | Needs per-type shader variants. |
| Spec constants already used (`OPIDX_BITS`) | `update_voltages.comp:41` | Precedent exists, but type ≠ spec constant — needs `-D` variants. |
| Python options go via `SetLibraryArguments(vector[string])` | `python/openEMS/openEMS.pxd:46` | A `--precision=` CLI flag reaches Python **for free**. |

### 0.1 The one architectural decision that matters

`Engine` and `Operator` are runtime-polymorphic class trees with `FDTD_FLOAT` **in the vtable
signatures** (`engine.h:58-104`, `operator.h:64-125`). There are two ways out:

**(A) Template the hierarchy on `T`.** Rejected. The engine matrix is
`{basic, sse, sse_compressed, avx2, avx2_mt, multithread, mpi, gpu} × {cartesian, cylinder,
multigrid}`, plus ~20 extension classes that each hold `Engine*`. Templating multiplies all of
it by 3 and forces every extension to become a template too. Compile times and binary size
explode, and `Engine*` is passed around as a plain pointer everywhere.

**(B) Widen the virtual boundary to `double`; template only storage and the kernel.** Chosen.

The virtual accessors (`GetVolt`/`SetVolt`/`GetVV`/…) are **not on the hot path** — the inner
update loops touch the arrays directly. They serve probes, extensions and post-processing,
all of which already work in `double`. Widening them costs nothing measurable and buys a
precision-agnostic interface that every extension already speaks.

```
             ┌──────────────────────────────────────────┐
 probes,     │ Engine / Operator virtual API  → double  │   precision-agnostic
 extensions, │ Engine_Interface_Base          → double  │   (already double today!)
 processing  └──────────────────────────────────────────┘
                              ▲  convert on the boundary
             ┌────────────────┴─────────────────────────┐
 hot loops   │ Storage<Store_T> + Kernel<Compute_T>     │   precision-parameterized
             └──────────────────────────────────────────┘
```

### 0.2 Two types, not one

A single scalar type is the wrong abstraction. The FDTD kernel is **memory-bound**, so the
payoff comes from narrowing *storage*, while accuracy is governed by *arithmetic*. Split them:

- **`Store_T`** — what sits in RAM/VRAM: `fp16` | `fp32` | `fp64`
- **`Compute_T`** — what the arithmetic runs in: `fp32` | `fp64`

This makes *fp16 storage / fp32 math* — the configuration most likely to actually win — a
first-class combination rather than a hack. Valid pairings:

| Mode | Store | Compute | Expected |
|---|---|---|---|
| `fp16` | half | float | ~2× bandwidth. **Accuracy at risk — see §7.** |
| `fp32` | float | float | today's behaviour, bit-identical |
| `fp64` | double | double | ~0.5× speed, reference accuracy |

`bf16` is deliberately not offered: it trades mantissa for exponent range, and FDTD needs
mantissa. fp16's range is adequate once fields are normalized (§7).

---

## Phase 0 — Numerical regression harness *(do this first)*

Nothing else is safe without it. This phase changes no production code.

1. **Golden-reference capture.** Pick ~6 cases spanning the physics: simple waveguide,
   patch antenna (radiating, long decay), lumped-RLC port, dispersive/Lorentz material,
   UPML-heavy, cylindrical multigrid. Run each at current `float`, dump full field state at
   fixed timestep checkpoints plus final S-parameters.
2. **Comparison tool.** `TESTSUITE/precision/compare.py` reporting per-checkpoint relative
   L2 and L∞ field error, S-parameter dB/phase deviation, and total-energy drift vs. reference.
3. **Bit-exactness gate for Phase 1.** Phase 1 must reproduce the golden `float` results
   **bit-for-bit**. Any deviation is a bug, not a rounding difference. This is the single
   most valuable property of the whole plan: it turns a large refactor into a mechanically
   verifiable one.
4. **Accuracy budget for Phase 2+.** Define pass thresholds up front, e.g. S11 within
   0.05 dB and 0.5° of the fp64 reference over the band of interest.

**Deliverable:** `TESTSUITE/precision/` + CI target. **Depends on:** nothing.

---

## Phase 1 — Clean abstraction, zero behaviour change

`FDTD_FLOAT` stays `float` throughout. Every step here is independently reviewable and
gated on bit-exactness. **This phase is worth doing even if multi-precision never ships.**

### 1.1 Introduce the type vocabulary — `tools/precision.h`

```cpp
namespace openEMS::precision {

enum class Mode { FP16, FP32, FP64 };

template <Mode M> struct Traits;
template <> struct Traits<Mode::FP32> {
    using Store   = float;
    using Compute = float;
    static constexpr size_t simd_lanes_sse  = 4;
    static constexpr size_t simd_lanes_avx2 = 8;
    static constexpr const char* name = "fp32";
};
// FP64 / FP16 specializations follow in Phase 2 / Phase 5.

}  // namespace openEMS::precision
```

Keep `FDTD_FLOAT` as an alias for `Traits<FP32>::Store` so nothing breaks mid-migration.

### 1.2 Eradicate raw `float` — but *only* in the simulation path

This is the step most likely to be done wrong. **Two distinct populations of `float`:**

- **Simulation-path `float` (155 sites, `FDTD/` + `Common/`)** — a precision choice.
  Convert to `FDTD_FLOAT`. Notably `Operator::m_epsR_ptr/m_kappa_ptr/m_mueR_ptr/m_sigma_ptr`
  (`operator.h:339-342`), which are spelled `float` literally today.
- **I/O-path `float` (245 sites: `tools/hdf5_file_*`, `tools/vtk_file_writer*`, `nf2ff/*`,
  `tools/sar_calculation*`)** — a *file-format and interop* choice, dictated by HDF5
  datatypes, VTK output and the Python/MATLAB ABI. **Leave alone.** Sweeping these up
  would silently change on-disk formats and break every downstream script.

Add a short comment at the top of each I/O file stating that its `float` is a format
contract, so a future reader does not "fix" it.

### 1.3 Fix `EC_*` to `double` *(real bug fix, ship independently)*

`EC_C/EC_G/EC_L/EC_R` are `FDTD_FLOAT*` (`operator.h:349-352`) but every consumer promotes
to `double` (`operator.cpp:964-982`) for expressions like `(1-dT*G/2/C)/(1+dT*G/2/C)`, which
is catastrophically cancellation-prone when `dT*G/2/C` is small. The equivalent-circuit values
are accumulated per-cell from material integrals and then rounded to `float` before that
division. Storing them as `double` costs 4 transient arrays during setup only — they are
freed before the timestep loop — and measurably improves coefficient accuracy at **every**
precision, fp32 included.

*This will not be bit-exact against the golden reference.* Ship it as its own commit with its
own before/after accuracy numbers, then re-baseline Phase 0 goldens on top of it.

### 1.4 Unify SIMD lane count — `SIMD_Traits`

Replace the duplicated `ceil(numLines[2]/4.0)` in `operator_sse.cpp:88` and
`engine_sse.cpp:40` with one shared `NumVectors<T, ISA>()` helper. Two independent
copies of a value that *must* agree is a bug waiting to happen, and it becomes a certainty
once lane count starts varying with type (fp64/SSE = 2, fp64/AVX2 = 4, fp16 = 16).

### 1.5 Widen the virtual accessor boundary to `double`

Change `GetVolt/GetCurr/SetVolt/SetCurr` (`engine.h:58-104`) and `GetVV/GetVI/GetII/GetIV` +
setters (`operator.h:64-125`) to take and return `double`. Storage stays `float`; the
conversion is implicit and free.

Verify with a targeted benchmark that no hot loop regresses — the SSE/AVX2 inner loops go
through `f4_volt(...)` directly, not through these, so the expectation is zero delta. If any
extension turns out to call an accessor per-cell per-timestep, fix *that* extension to use a
range/block accessor rather than reverting this step.

**Phase 1 exit criteria:** bit-identical results on all Phase 0 cases (with §1.3 re-baselined),
no benchmark regression >1%, zero raw `float` remaining in `FDTD/` and `Common/`.

---

## Phase 2 — Parameterize storage & kernel; land fp64 first

fp64 before fp16, deliberately: it exercises the entire abstraction with **no hardware
feature negotiation, no range concerns, and an obvious correctness oracle** (it should be
*more* accurate). It proves the seam. fp16 then becomes a contained experiment.

1. **Template the basic engine/operator.** `Engine_Basic<Store, Compute>` and
   `Operator_Basic<Store, Compute>` in the `.cpp`, with explicit instantiation for the
   supported pairs to keep compile times sane. The public `Engine`/`Operator` base classes
   stay non-template — only the leaf implementations are templated.
2. **Runtime factory.** `Engine::New(op)` / `Operator::New()` dispatch on
   `precision::Mode` obtained from global settings. One `switch` per factory.
3. **CLI + API.** `--precision=fp16|fp32|fp64` in `openems.cpp` (near the existing
   `--engine=` handling, ~line 243). Because Python routes through
   `SetLibraryArguments`, `openEMS(..., precision='fp64')` needs only a thin `.pyx` wrapper.
   Default stays `fp32` forever — this must be opt-in.
4. **Engine × precision matrix.** Not every combination is worth building. Declare the
   supported set explicitly and fail loudly with a clear message on unsupported pairs
   (e.g. "fp64 is not available for the GPU engine on this device") rather than silently
   downgrading. Silent precision downgrade would be the worst possible failure mode.
5. **Report the active mode** in the startup banner and write it into the HDF5 output
   attributes, so a result file records the precision it was produced at.

**Exit criteria:** fp64 runs all Phase 0 cases and becomes the new accuracy reference;
fp32 remains bit-identical to Phase 1.

---

## Phase 3 — SIMD engines (SSE / AVX2)

The heaviest mechanical phase: 89 `f4vector` + 44 `f8vector` sites.

1. Replace both unions with `simd_vec<T, Width>` carrying `lanes`, `load`, `store`,
   arithmetic operators, and a `widen`/`narrow` pair for mixed store/compute.
2. `Engine_SSE<Store, Compute>` / `Engine_AVX2<Store, Compute>`. Lane count comes from
   `SIMD_Traits` (§1.4), so the `z%numVectors` / `z/numVectors` layout arithmetic in
   `engine_sse.h:38-83` follows automatically.
3. **The three cylinder files** (`engine_ext_cylinder.*`,
   `engine_ext_cylindermultigrid.cpp`, `engine_extension_dispatcher.h`) reach directly into
   `f4_volt_ptr`. They need matching templating — the only extensions that do.
4. The tail-handling code (`engine_sse.cpp:136-153`, `230-238`) hardcodes lane indices
   `f[0..3]`. Must become width-generic; this is fiddly and deserves a dedicated test.
5. AVX2 fp64 = 4 lanes: real but modest speedup vs. scalar fp64.

**Risk:** highest mechanical-error density in the plan. Mitigation: Phase 0 bit-exactness on
fp32 catches essentially any mistake here immediately.

---

## Phase 4 — GPU (Vulkan)

1. **Shader variants.** `#define SCALAR float` → build each of the 25 `.comp` files once per
   supported type via `glslc -DSCALAR_TYPE=…` in `CMakeLists.txt:107-116`. Naming becomes
   `update_voltages_fp32.spv`; `embed_spirv.cmake` globs, so it picks them up unchanged.
   25 shaders × 3 types = 75 SPIR-V blobs — check the binary-size impact and consider
   building only the types actually enabled at configure time.
2. **Feature negotiation.** Extend the existing device-feature negotiation (commit `fabcaa3`)
   with `shaderFloat16` + `storageBuffer16BitAccess` (fp16) and `shaderFloat64` (fp64).
   fp64 throughput is 1/32 of fp32 on consumer GeForce/Radeon — **warn explicitly**, because
   a user selecting fp64 on a gaming GPU will otherwise think the run has hung.
3. **Coefficient traffic is already solved on GPU.** The compressed-operator path keeps
   `vv_comp`/`vi_comp` L2-resident (`update_voltages.comp:1-16`), so narrowing *coefficients*
   buys the GPU almost nothing. On GPU, fp16 only pays off on the **field** buffers — which is
   exactly the risky half (§7). Set expectations accordingly.

---

## Phase 5 — fp16 storage on CPU

Only now, with the abstraction proven and fp64 as a reference oracle.

- Storage `_Float16`; convert on load/store with F16C (`_mm256_cvtph_ps` /
  `_mm256_cvtps_ph`), accumulate in fp32. Runtime F16C detection with scalar fallback.
- **Prefer fp16 for coefficients over fields.** On the CPU non-compressed path, `vv/vi/ii/iv`
  are roughly half of all memory traffic, and they are **read-only** — they never accumulate
  error across timesteps. Narrowing them is a one-time quantization of the operator, which is
  far more defensible than narrowing the fields. Make coefficient and field precision
  *independently* selectable; the coefficient-only mode is the most likely production win.
- Add per-component field normalization if fields are narrowed (§7).

---

## Phase 6 — Validation & UX

- Full sweep of Phase 0 cases × every supported precision × every engine.
- **Accuracy vs. speed table published in the docs**, per case, measured — not estimated.
- Automatic guard: if the selected precision drifts beyond the configured energy-conservation
  tolerance, warn at runtime with the timestep at which it occurred.
- Document *when* to pick each mode, with honest guidance about what fp16 costs.

---

## 7. The fp16 accuracy problem — read before committing to Phase 5

This is the part of the plan most likely to fail, and it should be understood now.

**Range** is manageable. fp16 max is 65504, min normal 6.1e-5 — about 8 usable decades. Field
amplitudes in openEMS are voltages (E·dl) and currents (H·dl), whose absolute magnitudes
depend on mesh size and excitation amplitude, so they can easily sit outside that window.
Fixable by normalizing fields to O(1) at operator-build time and folding the scale into the
coefficients — a one-time transformation with no runtime cost.

**Accumulated round-off is the real risk.** The leapfrog update accumulates error roughly as
`√N · ε` over `N` timesteps. With `ε_fp16 ≈ 4.9e-4` and a typical `N = 10⁵`:

```
√1e5 × 4.9e-4  ≈  0.15   →  ~15% error
```

versus `√1e5 × 6e-8 ≈ 2e-5` for fp32. Even with fp32 accumulation, storing the field back to
fp16 each timestep re-injects `ε_fp16` per step, so the picture does not fundamentally change.
Runs that terminate on a `-60 dB` energy criterion are exactly the long ones where this bites.

**Therefore:**
- fp16 **coefficients** (read-only, no accumulation) — low risk, real bandwidth win on CPU.
  This is the recommended fp16 target.
- fp16 **fields** — high risk. May prove viable only for short, strongly-driven runs
  (steady-state, high-Q resonators excluded).
- The honest framing: Phases 1–4 are unconditionally worthwhile. Phase 5 is an **experiment**,
  and the value of the preceding phases is that they make it a cheap one to run and a cheap
  one to abandon.

Stochastic rounding on the fp32→fp16 store would convert the `√N·ε` bias-accumulating term
into a smaller random walk, and is the first thing to try if the naive version fails.

---

## 8. Sequencing

```
Phase 0  harness                     ─┐ blocks everything
Phase 1  abstraction (bit-exact)     ─┤ valuable standalone; 1.3 ships separately
Phase 2  fp64 + factory + CLI        ─┤ proves the seam
Phase 3  SIMD  ──┐
Phase 4  GPU   ──┤ independent of each other, parallelizable
Phase 5  fp16  ──┘ gated on §7 experiment
Phase 6  validation & docs
```

Phases 1.3 (EC precision fix) and 1.4 (lane-count dedup) are net improvements today and can
land immediately, independent of the rest.
