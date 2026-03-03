# GPU Full-Pipeline Plan: Eliminating CPU Stalls

## Executive Summary

The current `Engine_Vulkan` has two performance problems:

1. **Pure-GPU mode (~2.5% GPU utilization):** GPU computes for ~50 µs, then sits
   idle while the CPU runs `PA->Process()` for 1–5 ms.  The GPU cannot start the
   next batch until `PA->Process()` finishes and calls `IterateTS()` again.

2. **Hybrid mode (any extension present = catastrophic):** Every timestep requires
   2–4 full-field GPU↔CPU round-trips (352 MB each direction), serialising all
   work and making the GPU slower than the CPU engine.

Almost every real simulation uses UPML (PML boundary), which forces hybrid mode.
This plan describes how to move extensions onto the GPU and restructure the main
loop to keep the GPU continuously occupied.

---

## 1. Current Architecture — What Happens Per Timestep

### 1.1 Engine Base Class Update Order

```
for each timestep:
    DoPreVoltageUpdates()     ← extensions (reverse priority order)
    UpdateVoltages()          ← main Yee curl update
    DoPostVoltageUpdates()    ← extensions (normal priority order)
    Apply2Voltages()          ← extensions (normal priority order)
    DoPreCurrentUpdates()     ← extensions (reverse priority order)
    UpdateCurrents()          ← main Yee curl update
    DoPostCurrentUpdates()    ← extensions (normal priority order)
    Apply2Current()           ← extensions (normal priority order)
```

### 1.2 Extension Hook Usage Summary

| Extension          | PreVolt | PostVolt | Apply2V | PreCurr | PostCurr | Apply2I | Aux State              | Access Pattern |
|--------------------|---------|----------|---------|---------|----------|---------|------------------------|----------------|
| **UPML (PML)**     |    ✓    |    ✓     |         |    ✓    |    ✓     |         | `volt_flux`, `curr_flux` (3D PML vol) | local cell only |
| **Lorentz/Drude**  |    ✓    |          |    ✓    |    ✓    |          |    ✓    | `volt_ADE`, `curr_ADE`, `volt_Lor_ADE`, `curr_Lor_ADE` | local cell only (sparse) |
| **ConductSheet**   |         |          |    ✓    |         |          |    ✓    | `volt_ADE`, `curr_ADE` (from base) | local cell only (sparse) |
| **LumpedRLC**      |    ✓    |          |    ✓    |         |          |         | `v_Vdn`, `v_Jn`, `v_Il` (history) | local cell only (sparse) |
| **TFSF**           |         |    ✓     |         |         |    ✓     |         | `m_DelayLookup` (recomputed) | surface cells only |
| **Mur ABC**        |    ✓    |    ✓     |    ✓    |         |          |         | `m_volt_nyP/PP` (2D boundary) | 1 neighbor (shifted) |
| **AbsorbingBC**    |    ✓    |    ✓     |    ✓    |  (SA)   |   (SA)   |  (SA)   | `m_V_nyP/PP`, `m_I_nyP/PP` | 1 neighbor (shifted) |
| **Excitation**     |         |          |    ✓    |         |          |    ✓    | *already on GPU*       | sparse |
| **SteadyState**    |    ✓    |          |         |         |          |         | energy tracking | read-only |
| **Cylinder**       |         |    ✓     |         |         |    ✓     |         | r=0 fixup | boundary cells |

### 1.3 Key Observation

**Every extension accesses only local cells or at most 1 neighbor.**
No extension performs a stencil wider than the Yee curl itself.  This means
all extensions can be fused into the existing GPU compute shaders (or added as
separate shaders in the same command buffer) without requiring any additional
communication pattern beyond what the Yee update already needs.

---

## 2. Proposed Architecture: All-GPU Pipeline

### 2.1 Design Principle

Move **all physics extensions** into GPU compute shaders so that a full timestep
(Yee update + PML + dispersive + ABC + TF/SF + excitation) executes entirely on
the GPU.  The CPU only handles:
- Probe readback (sparse — see §4)
- HDF5/VTK file I/O
- Convergence checks

### 2.2 New Shader Pipeline Per Timestep

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Command Buffer                           │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │ 1. PML PreVolt   │  upml_pre_voltage.comp                   │
│  │    (PML region)  │  reads: volt, volt_flux, PML coeffs      │
│  │                  │  writes: volt (replace w/ flux), volt_flux│
│  └────────┬─────────┘                                           │
│           │ barrier                                              │
│  ┌────────▼─────────┐                                           │
│  │ 2. Yee Volt + Ext│  update_voltages_ext.comp                 │
│  │    (all cells)   │  Yee curl + inline dispersive PreVolt     │
│  │                  │  + Mur/ABC pre-save                       │
│  └────────┬─────────┘                                           │
│           │ barrier                                              │
│  ┌────────▼─────────┐                                           │
│  │ 3. PostVolt exts │  post_voltage_ext.comp                    │
│  │                  │  PML PostVolt, TFSF PostVolt,             │
│  │                  │  Mur PostVolt, ABC PostVolt               │
│  └────────┬─────────┘                                           │
│           │ barrier                                              │
│  ┌────────▼─────────┐                                           │
│  │ 4. Apply2Volt    │  apply2_voltages.comp                     │
│  │                  │  Dispersive Apply2V, LumpedRLC Apply2V,   │
│  │                  │  Mur/ABC Apply2V, Excitation              │
│  └────────┬─────────┘                                           │
│           │ barrier                                              │
│  ┌────────▼─────────┐                                           │
│  │ 5. PML PreCurr   │  upml_pre_current.comp                   │
│  └────────┬─────────┘                                           │
│           │ barrier                                              │
│  ┌────────▼─────────┐                                           │
│  │ 6. Yee Curr + Ext│  update_currents_ext.comp                 │
│  └────────┬─────────┘                                           │
│           │ barrier                                              │
│  ┌────────▼─────────┐                                           │
│  │ 7. PostCurr exts │  post_current_ext.comp                    │
│  └────────┬─────────┘                                           │
│           │ barrier                                              │
│  ┌────────▼─────────┐                                           │
│  │ 8. Apply2Curr    │  apply2_currents.comp                     │
│  └────────┬─────────┘                                           │
│           │ barrier                                              │
│  ┌────────▼─────────┐                                           │
│  │ 9. Probe gather  │  gather_probes.comp (optional)            │
│  │    (sparse read) │  writes small probe_result_buf            │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

Multiple timesteps are recorded into a single command buffer (as done today in
pure-GPU mode), because none of the extension shaders require CPU intervention.

### 2.3 Alternative: Fused Kernels

Instead of 8-9 separate dispatches, many operations can be **fused** since they
touch the same cells.  For example:

**Fused Voltage Kernel** — one dispatch, one thread per cell:
```glsl
// Pseudocode for a fused voltage update kernel
void main() {
    uint gid = gl_GlobalInvocationID.x;
    // ... decompose to (x, y, z)

    // 1. PML pre-voltage (if cell is in PML region)
    if (isPMLCell(x, y, z)) {
        pml_pre_voltage(x, y, z);  // swap flux ↔ volt
    }

    barrier();  // subgroup/workgroup not needed — each cell is independent

    // 2. Yee voltage update (all cells)
    float curl = compute_curl_H(x, y, z);
    volt[idx] = vv[idx] * volt[idx] + vi[idx] * curl;

    // 3. PML post-voltage (if PML cell)
    if (isPMLCell(x, y, z)) {
        pml_post_voltage(x, y, z);  // combine flux with Yee result
    }

    // 4. Dispersive ADE pre-update (if dispersive cell)
    if (isDispersiveCell(idx)) {
        dispersive_pre_voltage(idx);
    }

    // 5. Apply2Voltages: dispersive, RLC, excitation
    if (isDispersiveCell(idx)) volt[idx] -= volt_ADE[idx];
    if (isRLCCell(idx)) rlc_apply_voltage(idx);
    excitation_apply(idx);  // sparse — most threads skip
}
```

**Trade-off:** Fusing reduces dispatch overhead and barrier stalls but creates
divergent branches.  On modern GPUs (RDNA3, Ada, etc.) with 32/64-wide waves,
PML cells are typically a small fraction (~5-15% of total cells, concentrated at
boundaries), so most waves will take the non-PML fast path.  The divergence
penalty of a few waves at PML boundaries is far smaller than the cost of a
separate dispatch + full pipeline drain.

**Recommendation: Start with separate dispatches (8 shaders), profile, then
selectively fuse based on measured overhead.**

> **UPDATE (2026-03):** Fused Yee+UPML kernels have been implemented as
> `fused_update_voltages.comp` and `fused_update_currents.comp`, along with a
> pipelined main loop using speculative submission. See
> `GPU_FUSION_AND_PIPELINING.md` for the implementation details.

---

## 3. Extension-by-Extension GPU Porting Analysis

### 3.1 UPML (PML) — HIGHEST PRIORITY

**Current cost:** Forces hybrid mode → 4 sync points/timestep → ~700 MB/s PCIe
  transfers kill all GPU benefit.

**GPU port complexity: MEDIUM**

The UPML algorithm is a split-field PML with auxiliary flux variables.  Per cell:

```
PreVoltage:
  f_help = vv_pml * volt[pos] - vvfo * volt_flux[loc_pos]
  volt[pos] = volt_flux[loc_pos]     // save old flux
  volt_flux[loc_pos] = f_help

PostVoltage:
  f_help = volt_flux[loc_pos]        // partial from pre
  volt_flux[loc_pos] = volt[pos]     // save Yee result as new flux
  volt[pos] = f_help + vvfn * volt_flux[loc_pos]
```

**GPU requirements:**
- **Buffers:** Upload `volt_flux` (3×PML_cells × 4B) and `curr_flux` to GPU.
  For a typical PML of 8 cells on each of 6 faces of a 150³ grid:
  PML volume ≈ 6 × 8 × 150² = 1.08M cells → 3 × 1.08M × 4B ≈ 13 MB per flux array.
  Total: ~26 MB for flux + ~78 MB for 6 coefficient arrays = ~104 MB.
  This easily fits in VRAM alongside the main fields.

- **Coefficient buffers (read-only):** `vv`, `vvfo`, `vvfn`, `ii`, `iifo`, `iifn`
  — 6 arrays, each 3×PML_cells. Upload once during Init.

- **Index mapping:** PML cells are a contiguous sub-region of the grid
  (`m_StartPos[3]` to `m_StartPos[3] + m_numLines[3]`).  The shader needs:
  - Push constants or UBO: `pml_start[3]`, `pml_size[3]`
  - Linear index mapping from PML-local coords to global field index

- **Shader:** Two options:
  1. **Separate PML dispatch** over PML cells only (small dispatch, ~1M cells)
  2. **Merged into Yee kernel** with a branch (slightly divergent, but
     PML bands are axis-aligned → waves near boundaries diverge, interior
     waves execute fast path)

- **No neighbor access** — all PML operations read/write only the local cell
  plus the PML auxiliary arrays.

**Data lifetime:** `volt_flux` and `curr_flux` evolve every timestep and must
persist on GPU between timesteps.  Never need to come back to CPU.

**Verdict: Straightforward GPU port.  This alone would eliminate hybrid mode
for the majority of simulations.**

### 3.2 Lorentz/Drude Dispersive Material — HIGH PRIORITY

**GPU port complexity: LOW-MEDIUM**

Sparse: only cells with dispersive material.  Auxiliary state is `volt_ADE`,
`curr_ADE`, and optionally `volt_Lor_ADE`, `curr_Lor_ADE`.

**GPU requirements:**
- **Index buffer:** Upload the list of dispersive cell linear indices (sparse).
  Each order has its own index list (`m_LM_Count[o]` cells).
- **Coefficient buffers:** `v_int_ADE`, `v_ext_ADE`, `v_Lor_ADE`, etc.
  — per order, per direction, per cell.
- **ADE state buffers:** `volt_ADE[order][3][cell]`, `curr_ADE[order][3][cell]`,
  `volt_Lor_ADE`, `curr_Lor_ADE` — GPU-resident, evolve each timestep.

**Shader approach:** Sparse dispatch over dispersive cells only.  Each thread
handles one dispersive cell, reads the engine voltage/current, updates ADE state,
writes the correction back.

**Multi-order support:** Can either:
- Loop over orders inside a single dispatch (push constant `numOrders`)
- Separate dispatch per order (simpler, small overhead)

**Verdict: Clean GPU port.  Sparse pattern means minimal divergence.**

### 3.3 Conducting Sheet — LOW PRIORITY

Uses the `Engine_Ext_Dispersive` base class directly.  Once Lorentz/Drude is
ported, conducting sheet comes for free — same ADE pattern, same shader.

### 3.4 Lumped RLC — MEDIUM PRIORITY

**GPU port complexity: LOW**

Very sparse (individual circuit elements).  Small auxiliary state (3-deep history
per element).  Z-transform coefficient update is purely local.

**GPU requirements:**
- Index buffer of RLC cell positions
- Coefficient buffers (operator data, uploaded once)
- State buffers: `v_Vdn[3]`, `v_Jn[3]`, `v_Il` — tiny (one value per element)
- Pointer rotation of history buffers → on GPU, just swap buffer bindings or
  use a ring-buffer index in push constants

**Verdict: Trivial GPU port.  Very few cells, so overhead matters — may want
to fuse into the main voltage shader with a sparse-index lookup.**

### 3.5 TF/SF (Total-Field/Scattered-Field) — MEDIUM PRIORITY

**GPU port complexity: LOW**

Adds incident field on 6 faces of a box.  Pattern is nearly identical to excitation
(sparse indexed additions with signal lookup + delay interpolation).

**GPU requirements:**
- Per-face arrays: amplitude, delay, index (already exist in operator)
- Signal buffer (same as excitation signal - may be the same signal)
- `m_DelayLookup` table: recompute per timestep → push constant `numTS` + inline
  computation in shader (same approach as `apply_excitation.comp`)

**Shader:** One dispatch per face direction, or single dispatch with
face-direction push constant.  Surface area is small → tiny dispatch.

**Verdict: Same pattern as the excitation shader we already have.  Easy.**

### 3.6 Mur ABC / Absorbing BC — MEDIUM PRIORITY

**GPU port complexity: LOW-MEDIUM**

2D boundary operations.  Saves pre-update boundary values, then after the Yee
update, constructs the ABC boundary condition from old + new values.

**GPU requirements:**
- 2D auxiliary state buffers: `m_volt_nyP`, `m_volt_nyPP` (one per boundary face)
- Mur coefficients: 2D arrays `m_Mur_Coeff_nyP`, `m_Mur_Coeff_nyPP`
- Access pattern: reads `volt(pos)` and `volt(pos_shifted)` — one neighbor.
  The shifted cell is one mesh step inward from the boundary.

**Shader:** Dispatch over boundary surface cells.  Pre-voltage shader saves
`volt[shifted] - K*volt[boundary]` into aux buffer.  Post-voltage shader adds
`K*volt[shifted_updated]`.  Apply2 shader overwrites boundary value.

**Complication:** Needs to interleave with the Yee update (pre, post, apply
happen at specific points).  This naturally maps to 3 separate small dispatches
or a fused kernel with phase tracking.

**Verdict: Small data, small dispatch.  Clean GPU port.**

### 3.7 Cylindrical Coord Fixup — LOW PRIORITY (niche)

Only relevant for cylindrical coordinates.  Small correction at r=0 axis.
Straightforward sparse shader.

### 3.8 Steady-State Detection — TRIVIAL

Read-only energy computation.  Can be done as a GPU reduction shader or
deferred to the CPU (reads via ReBAR, tiny probe count).

---

## 4. Sparse Probe Readback — Eliminating the PA->Process() Stall

### 4.1 The Problem

Even in pure-GPU mode, `PA->Process()` calls `GetVolt()`/`GetCurr()` which
calls `DrainGPU()` → GPU stalls while CPU reads probes.

For a voltage probe: 3 ReBAR reads (~900 ns) — negligible.
For a mode-match probe integrating over a cross section: 50,000+ reads
→ ~15 ms of PCIe latency → GPU idle for 300× its compute time.

### 4.2 Solution: GPU Probe Gather Shader

Add a `gather_probes.comp` shader at the end of each timestep batch:

```glsl
layout(set=0, binding=0) buffer FieldBuf { float field[]; };
layout(set=0, binding=1) buffer ProbeBuf { float results[]; };
layout(set=0, binding=2) buffer IdxBuf   { uint  indices[]; };
// ...weights, types, etc.

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= probeCount) return;

    uint fieldIdx = indices[gid];
    results[gid] = field[fieldIdx];
    // Or: results[gid] = weight[gid] * field[fieldIdx]; for weighted integrals
}
```

For DFT accumulation (frequency-domain probes), the GPU can also accumulate:
```glsl
// Per frequency bin per probe point:
dft_real[bin] += field[idx] * cos(omega * dt * ts);
dft_imag[bin] += field[idx] * sin(omega * dt * ts);
```

### 4.3 Readback Strategy

After the gather shader, only the `results[]` buffer (~probeCount × 4 bytes,
typically a few KB) needs to be read by the CPU.  This can be:

1. **ReBAR direct read** — instantaneous for small buffers
2. **Host-visible buffer** — the results buffer is allocated as host-visible,
   CPU reads after fence
3. **Async DMA** — copy results to a host buffer using a transfer queue while
   the next compute batch runs

### 4.4 Revised Main Loop

```
                         time ─────────────────────────────────────────►

GPU:  ┌──FDTD-batch─1──┬──gather──┐┌──FDTD-batch─2──┬──gather──┐
      │ N×(Yee+PML+Ext)│probe read││ N×(Yee+PML+Ext)│probe read│
      └────────────────┴──────────┘└────────────────┴──────────┘

CPU:                    ┌──Process──┐                ┌──Process──┐
                        │read result│                │read result│
                        │accum DFT  │                │accum DFT  │
                        │HDF5 write │                │HDF5 write │
                        └───────────┘                └───────────┘
```

**Key change:** The CPU reads from the **previous batch's** gathered results
while the GPU is already computing the **next batch**.  This is the classic
"pipelining" approach — the fence wait is for batch N-1 (already long done),
not batch N (still in progress).

```cpp
// Pseudocode for new main loop
int step = PA->Process();
while (running) {
    // Submit GPU batch (records step timesteps + all extensions + probe gather)
    FDTD_Eng->IterateTS(step);            // non-blocking, fires off GPU work

    // CPU processes probe results from PREVIOUS batch (not the one just submitted)
    // This was gathered by the GPU into probe_result_buf in the previous IterateTS call
    step = PA->ProcessFromGPUResults();    // reads tiny buffer, no DrainGPU needed

    // By the time Process finishes, the GPU batch is typically done already
}
```

**GPU utilization goes from 2.5% → ~95%+** because the GPU never waits for
the CPU and vice versa (except the initial pipeline fill latency).

---

## 5. Implementation Roadmap

### Phase 1: GPU-Native Extensions (eliminate hybrid mode)

**Goal:** Move PML + all common extensions into GPU shaders so that
`m_hasCPUExtensions` is always false.

| Task | Effort | Impact |
|------|--------|--------|
| 1a. UPML GPU shader + buffer management | 2-3 weeks | Eliminates hybrid mode for ~90% of sims |
| 1b. TF/SF GPU shader (like excitation) | 3-5 days | Covers plane-wave sims |
| 1c. Mur/Absorbing BC GPU shader | 1 week | Free-space boundary alternative |
| 1d. Lorentz/Drude/Dispersive GPU shader | 1-2 weeks | Covers material sims |
| 1e. LumpedRLC GPU shader | 3-5 days | Covers circuit co-sim |
| 1f. Conducting sheet (via dispersive) | 0 days | Free with 1d |
| 1g. Cylindrical coord fixup | 3 days | Niche |

**Deliverable:** All physics extensions execute on GPU.  No CPU extension hooks
needed.  `IterateTS` always takes the pure-GPU path.

### Phase 2: Sparse Probe Readback (eliminate PA->Process() stall)

| Task | Effort | Impact |
|------|--------|--------|
| 2a. `gather_probes.comp` shader | 1 week | Eliminates massive ReBAR latency |
| 2b. GPU DFT accumulation shader | 1 week | Moves FD processing to GPU |
| 2c. Pipelined main loop (N-1 readback) | 1 week | Achieves >95% GPU utilization |
| 2d. Integrate with ProcessingArray | 1 week | Wire up existing probe infrastructure |

**Deliverable:** Main loop is fully pipelined.  GPU utilization >95%.

### Phase 3: Advanced Optimizations

| Task | Effort | Impact |
|------|--------|--------|
| 3a. Kernel fusion (benchmark-driven) | 2 weeks | 10-30% kernel overhead reduction |
| 3b. Async transfer queue for field dumps | 1 week | Overlaps VTK/HDF5 writes with compute |
| 3c. Multi-timestep batching with mid-batch probes | 2 weeks | Handle step=1 efficiently |
| 3d. GPU-side convergence detection | 3 days | Avoid CPU energy reads |

---

## 6. Buffer Summary for Full GPU Pipeline

### New GPU Buffers Required

| Buffer | Size (typical 150³ grid, 8-cell PML) | Lifetime |
|--------|--------------------------------------|----------|
| `volt_flux` | 3 × PML_cells × 4B ≈ 13 MB | Persistent, evolves |
| `curr_flux` | 13 MB | Persistent, evolves |
| PML coeffs (vv,vvfo,vvfn,ii,iifo,iifn) | 6 × 13 MB = 78 MB | Upload once |
| Dispersive ADE state | 3 × dispersive_cells × orders × 4B ≈ 1-20 MB | Persistent, evolves |
| Dispersive coeffs | ~2-10 MB | Upload once |
| Mur/ABC aux | 2 × boundary_face × 4B ≈ 0.3 MB | Persistent |
| TF/SF (like excitation) | ~0.1 MB | Upload once |
| LumpedRLC state | ~0.01 MB | Persistent |
| Probe index + results | ~0.1 MB | Index: once; Results: per-batch |
| DFT accumulators | 2 × freq_bins × probe_points × 4B ≈ 1-10 MB | Persistent |

**Total additional VRAM: ~120-150 MB** for a typical simulation.

    Compared to the ~350 MB already used for volt+curr buffers, this is a ~35-40%
    increase.  A 4 GB GPU can handle grids up to ~300³ cells; an 8 GB GPU up to
    ~400³+ cells — well within the typical use case.

---

## 7. Shader Architecture Decision: Separate vs. Fused

### Option A: Separate Shaders (Recommended for Phase 1)

```
Dispatches per timestep: 8-12 (depending on active extensions)
Barriers: 7-11
```

**Pros:**
- Each shader is simple and independently testable
- Easy to conditionally skip unused extensions (just don't dispatch)
- Maps directly to the existing extension hook model
- Easier to maintain and extend

**Cons:**
- Dispatch overhead: ~2-5 µs per dispatch on modern GPUs
- With 10 dispatches: 20-50 µs overhead vs ~50 µs compute = significant
- Barrier flush between dispatches can hurt occupancy

### Option B: Fused Mega-Kernels (Phase 3 optimization)

```
Dispatches per timestep: 2 (one for volt half, one for curr half)
Barriers: 1
```

**Pros:**
- Minimal dispatch overhead
- Single read/write of `volt[]`/`curr[]` per half-step (better cache)
- Maximum occupancy

**Cons:**
- Complex branching (PML regions, dispersive cells, boundaries)
- Harder to maintain; every extension change requires shader rebuild
- VGPR pressure from having all extension state in registers

### Option C: Hybrid — Fuse Yee+PML, Separate Others

```
Dispatches per timestep: 4-6
Barriers: 3-5
```

The main Yee kernel is modified to **inline PML** (since PML touches the same
cells), while other extensions remain separate small dispatches.

**This is likely the sweet spot** — PML is the most common extension and benefits
most from fusion, while other extensions are sparse and benefit from targeted
dispatches.

---

## 8. Compatibility and Fallback Strategy

### Extension Auto-Detection

During `Init()`, classify each extension:

```cpp
enum class ExtensionGPUSupport {
    GPU_NATIVE,      // has a GPU shader implementation
    GPU_UNSUPPORTED, // must use CPU hybrid path
};
```

If ALL extensions are `GPU_NATIVE`, use the all-GPU pipeline.
If ANY extension is `GPU_UNSUPPORTED`, fall back to the current hybrid path
(but only the unsupported extension requires CPU sync — GPU-native extensions
still run their GPU shaders).

### Graceful Degradation

For Phase 1, port the 6 most common extensions.  Any custom/third-party
extension that isn't ported falls back to hybrid mode.  Over time, the
"unsupported" set shrinks.

### Validation

Each GPU extension should be validated against the CPU reference by:
1. Running the same simulation with CPU engine and GPU engine
2. Comparing field dumps at multiple timesteps
3. Ensuring field differences are within float32 precision (~1e-6 relative)

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| PML numerical precision (float32 vs float64) | Medium | PML coefficients involve products close to 1.0; test with real PML configs. Fall back to CPU PML if precision insufficient. |
| VRAM exhaustion from additional buffers | Low | ~150 MB overhead; monitor and warn at Init() |
| Dispatch overhead dominates for small grids | Medium | For grids < 50³, the CPU engine is likely faster anyway. Provide a threshold. |
| Divergence penalty in fused kernels | Low | PML is boundary-aligned; most warps don't diverge |
| Third-party extensions require CPU | Low | Hybrid fallback preserved; only the specific extension forces sync |

---

## 10. Summary: What Changes and Why

| Component | Current | Proposed | Why |
|-----------|---------|----------|-----|
| UPML | CPU hybrid (4 syncs/ts) | GPU shader | Eliminates 352 MB/ts transfers |
| Dispersive | CPU hybrid | GPU shader (sparse) | No sync needed |
| TF/SF | CPU hybrid | GPU shader (like excitation) | Surface-only, trivial |
| ABC/Mur | CPU hybrid | GPU shader (boundary) | Small data, clean port |
| LumpedRLC | CPU hybrid | GPU shader (sparse) | Tiny, trivial |
| Excitation | Already GPU | Already GPU | No change |
| Probe readback | ReBAR per-element | GPU gather + tiny readback | Eliminates 15 ms PCIe stall |
| DFT accumulation | CPU per-element | GPU reduction shader | Moves compute to GPU |
| Main loop | Sequential | Pipelined (N-1 readback) | GPU utilization 2.5% → 95%+ |

**Expected speedup over current GPU engine:**
- Simulations with PML: **50-200×** (eliminating hybrid mode)
- Pure-GPU simulations: **10-40×** (eliminating probe stall via pipelining)
- Overall vs CPU engine (4-core): **5-20×** for typical antenna/waveguide sims
