# GPU Pipeline Optimization: Energy Reduction, Probe Gather & Async Processing

## Problem Statement

The `RunFDTD()` main loop exhibited periodic stalls (~300 ms) caused by two operations
that forced full GPU↔CPU synchronization and data transfer every few seconds:

1. **`CalcTotalEnergyEstimate()` → `CalcFastEnergy()`**: Scans ALL grid cells
   (`3×Nx×Ny×Nz` floats ≈ 650 MB for a typical 27M-cell mesh) on the CPU side.
   Even with ReBAR, this means reading ~650 MB over PCIe via random-access patterns
   through memory-mapped VRAM — extremely slow.

2. **`PA->Process()` → `CalcVoltageIntegral()` → `GetVolt()` → `DrainGPU()` +
   `SyncFieldsToHost()`**: Voltage/current probes trigger a GPU drain and, without
   ReBAR, a full field download just to read a handful of cells along a line.

3. **No threading separation**: CPU processing (file I/O, FFT, data writing) runs
   sequentially with GPU submission — the GPU sits idle while CPUs process data.

## Solution Architecture

Three optimizations, implemented in priority order:

### 1. GPU Energy Reduction (highest impact)

**Shader**: `FDTD/gpu/shaders/reduce_energy.comp`

Computes total field energy entirely on-GPU using a parallel reduction:
- 256 workgroups × 256 threads = 65,536 threads
- Grid-stride loop: each thread accumulates `v0²+v1²+v2²` (E) and `c0²+c1²+c2²` (H)
  over its assigned cells
- Shared memory tree reduction within each workgroup (7 barrier-synchronized steps)
- Each workgroup writes one (E_partial, H_partial) float pair → 2 KB total output

The CPU sums 256 partial sums (trivially fast), applies `ε₀` and `μ₀` constants.

**Data flow**:
```
GPU field buffers (650 MB, already resident)
    ↓ reduce_energy.comp dispatch (256 workgroups)
GPU partial sums buffer (2 KB)
    ↓ ReBAR direct read or staging download
CPU: sum 256 + 256 floats → return ε₀·E + μ₀·H
```

**Performance**: Eliminates ~650 MB PCIe transfer. GPU reduction runs in ~100 μs.
Total CalcFastEnergy time drops from ~300 ms to ~1 ms.

### 2. GPU Voltage/Current Integral (moderate impact)

**Methods**: `CalcVoltageIntegralGPU()`, `CalcCurrentIntegralGPU()`

For voltage probes (line integrals), we bypass the generic `GetVolt()` virtual dispatch:
- **With ReBAR**: After `DrainGPU()`, read directly from `m_voltMapped` / `m_currMapped`.
  A line integral touches only a few dozen cells — just a few PCIe reads, negligible.
- **Without ReBAR**: Fall back to `SyncFieldsToHost()` + `Engine::GetVolt()` (cached
  after the first call per batch, so subsequent probes are free).

The key improvement is avoiding the per-cell virtual dispatch overhead and ensuring
the integration uses the optimal path for the memory configuration.

### 3. Async Processing Thread (removed)

An earlier design explored a background `std::thread` with mutex/condition_variable
synchronization to run `PA->Process()` concurrently with the next GPU batch. It was
never wired into `RunFDTD()` and was superseded before that integration happened by
the pipelined main loop with speculative submission (see `GPU_FUSION_AND_PIPELINING.md`),
which provides the same overlap without a second OS thread or thread-safety concerns.
The unused scaffolding (`StartAsyncProcessing()`, `AsyncWorkerLoop()`,
`TriggerAsyncProcessing()`, `WaitAsyncProcessing()`) has been removed.

### 4. Async Field-Dump Download Pipeline

Full-field dumps (`ProcessFields`/`AddDump`) need the *entire* volt/curr buffers,
not the sparse cells the probe cache (§2) covers. Before this, that download
(`SyncFieldsToHost()`) was a single blocking call on the compute queue: drain,
`vkCmdCopyBuffer`, `vkWaitForFences`, with nothing else in flight — and it
happened lazily, mid-`PA->Process()`, wherever the dump's `FillFieldData()`
call landed.

**Async pipeline** (`BeginAsyncFieldDownload()` / `FinishAsyncFieldDownload()`,
`EnsureDumpBuffers()`): a second, genuinely independent hardware queue is
selected in `InitVulkan()` (on this hardware: the `COMPUTE|TRANSFER`,
non-`GRAPHICS` queue family with 4 queues that `m_computeQueueFamily`
selection didn't consider before, since it stops at the first
`VK_QUEUE_COMPUTE_BIT` family). Two stages, double-buffered:

1. Live `volt`/`curr` → device-local **snapshot** buffers, on the compute
   queue. Fast (VRAM-bandwidth bound), and must complete before the *next*
   speculative batch overwrites the live buffers in place.
2. Snapshot → host-visible **staging** buffers, on the independent dump
   queue. This is the slow, PCIe-bound leg, and runs fully concurrently with
   whatever the caller submits to the compute queue next.

`SetHasFieldDumps()`/`HasPipelinedProcessing()` (`engine_vulkan.h`) now
enable the speculative-submission main loop (`GPU_FUSION_AND_PIPELINING.md`
§3) for dump-only simulations too, not just ones with probes — the
submit-ahead structure is what lets `BeginAsyncFieldDownload()`'s download
overlap with real compute, so it needs to run in `openems.cpp`'s pipelined
loop, not the sequential fallback. Ordering matters: `BeginAsyncFieldDownload()`
is called once a state is *fully committed* (right after
`CommitSpeculative()`/`SnapshotProbeCache()`), **before** the next
`SubmitSpeculative()` — calling it after would make its internal `DrainGPU()`
block on the batch just submitted, serializing the two and defeating the
whole point.

**The dominant cost turned out not to be scheduling.** Once the async
pipeline was wired up and benchmarked end-to-end (291³ grid, 24.6M cells,
`--gpu-no-rebar-fields` to force the non-ReBAR path on hardware where ReBAR
would otherwise cover the whole buffer), it showed no measurable improvement
over the original blocking code — timing instrumentation traced this to
`FinishAsyncFieldDownload()`'s host-side `memcpy()` out of the mapped
staging buffer taking **~2.1 seconds for 564 MB (~260 MB/s)**, dwarfing both
the GPU-side PCIe copy (~26 ms) and the snapshot copy. Root cause:
`FindMemoryTypeSoft()` picks the *first* memory type satisfying the
requested property flags, and `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` alone is satisfied by a
write-combined/uncached type on this driver *before* it reaches a
`HOST_CACHED` one — fine for the CPU *writing* into GPU-visible memory
(uploads), catastrophic for *reading* it back. Requesting
`VK_MEMORY_PROPERTY_HOST_CACHED_BIT` as well (falling back to the plain
flags if a driver doesn't expose a cached host-visible type) for both the
new dump staging buffers and the pre-existing `m_stagingBuf` (used by
`DownloadFromDeviceBuffer()` — the same bug affected every non-ReBAR field
sync, not just dumps) turned the same benchmark from **~78 MC/s to ~1450
MC/s, an ~18x wall-clock speedup** (74s → 16s for a 200-timestep run with
periodic dumps). The async queue/double-buffering plumbing itself measured
a statistically-insignificant ±3% either way once the memcpy was fixed —
it's kept as correctness-safe infrastructure for cases where the transfer
itself is large enough to matter again (bigger grids, slower PCIe links),
gated behind `OPENEMS_GPU_DISABLE_DUMP_ASYNC=1` for A/B comparison.

**Correctness**: verified two ways — (1) GPU dump output compared against
the CPU (AVX2) engine's dump for the same simulation, exact match; (2) the
async and non-async GPU paths produce bit-identical dump output on the same
run. Both existing GPU-vs-CPU regression tests (`test_gpu_engine.py`) and
this dump-specific check pass.

## Files Modified

### New Files
| File | Purpose |
|------|---------|
| `FDTD/gpu/shaders/reduce_energy.comp` | GPU parallel energy reduction shader |
| `FDTD/GPU_PIPELINE_OPTIMIZATION.md` | This documentation |

### Modified Files

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Added `reduce_energy` to `GPU_SHADER_NAMES` |
| `FDTD/engine_vulkan.h` | Added energy reduction members (`m_energyPartialBuf`, `m_energyMapped`, descriptor/pipeline objects, `EnergyPC` struct, `ENERGY_NUM_WG`), public methods (`CalcFastEnergyGPU`, `CalcVoltageIntegralGPU`, `CalcCurrentIntegralGPU`). Async dump download: `SetHasFieldDumps()`/`HasFieldDumps()`, `HasDumpTransferQueue()`, `BeginAsyncFieldDownload()`, `HasPipelinedProcessing()` now also true for dump-only sims, dump queue/buffer/sync members (`m_dumpQueue`, `m_dumpSnapVolt/Curr`, `m_dumpStageVolt/Curr`, fences/semaphores, `DUMP_RING`). |
| `FDTD/engine_vulkan.cpp` | Added `#include "tools/constants.h"`. Init: calls `SetupGPU_EnergyReduction()`. CleanupVulkan: calls `CleanupGPU_EnergyReduction()`. New method implementations: `SetupGPU_EnergyReduction()`, `CleanupGPU_EnergyReduction()`, `CalcFastEnergyGPU()`, `CalcVoltageIntegralGPU()`, `CalcCurrentIntegralGPU()`. Async dump download: independent queue family selection in `InitVulkan()` (`OPENEMS_GPU_DISABLE_DUMP_ASYNC` opt-out), `EnsureDumpBuffers()`, `BeginAsyncFieldDownload()`, `FinishAsyncFieldDownload()`, `CleanupDumpAsync()`; `SyncFieldsToHost()` checks for a pending async request first; `m_stagingBuf` and the new dump staging buffers now prefer a `HOST_CACHED` memory type. |
| `FDTD/engine_interface_fdtd.cpp` | Added `#include "engine_vulkan.h"` (under `WITH_GPU`). GPU fast-path dispatch at top of `CalcFastEnergy()` and `CalcVoltageIntegral()` |
| `openems.cpp` | `RunFDTD()`: detects registered `ProcessFields` dumps and calls `SetHasFieldDumps()`; pipelined loop calls `BeginAsyncFieldDownload()` once each committed state is stable (before the next `SubmitSpeculative()`); the sequential fallback loop does the same after `IterateTS()` as a defensive no-op path for `m_hasCPUExtensions` sims. |

## Resource Lifecycle

### Energy Reduction Resources
```
Init()
  └─ SetupGPU_EnergyReduction()
       ├─ CreateGpuBuf(m_energyPartialBuf, 2KB, ReBAR or device-local)
       ├─ vkMapMemory if ReBAR → m_energyMapped
       ├─ Create descriptor layout (3 bindings: volt, curr, partials)
       ├─ Create pipeline layout (EnergyPC push constant, 8 bytes)
       ├─ Create m_energyDescPool (maxSets=1, 3 storage buffers)
       ├─ Allocate + write m_energyDescSet
       └─ Create compute pipeline from reduce_energy SPIR-V

~Engine_Vulkan()
  └─ CleanupVulkan()
       └─ CleanupGPU_EnergyReduction()
            ├─ vkUnmapMemory(m_energyPartialBuf)
            ├─ DestroyGpuBuf(m_energyPartialBuf)
            ├─ vkDestroyPipeline(m_energyPipeline)
            ├─ vkDestroyPipelineLayout(m_energyPipeLayout)
            ├─ vkDestroyDescriptorSetLayout(m_energyDescLayout)
            └─ vkDestroyDescriptorPool(m_energyDescPool)
```

## Descriptor Layout

### Energy Reduction (set 0)
| Binding | Type | Content |
|---------|------|---------|
| 0 | STORAGE_BUFFER (readonly) | volt[] — 3×N×sizeof(float) |
| 1 | STORAGE_BUFFER (readonly) | curr[] — 3×N×sizeof(float) |
| 2 | STORAGE_BUFFER (writeonly) | partials[] — 2×256×sizeof(float) |

**Push constant** (8 bytes): `{ uint32_t N; uint32_t numWG; }`

## Future Work

- ~~**Async processing integration**~~ — Superseded by the pipelined main loop
  with speculative submission (see `GPU_FUSION_AND_PIPELINING.md`), which
  provides better overlap without thread-safety concerns.

- ~~**GPU probe gather shader**~~ — ✅ Implemented. `gather_probes.comp` is now
  wired up via `SetupProbeCache()` using a recording-based cell discovery
  approach.  See `GPU_FUSION_AND_PIPELINING.md` §3.

- **Batched energy reduction**: Instead of a separate `RunSingleCommand()` dispatch,
  the energy reduction could be appended to the main FDTD command buffer every N
  timesteps, eliminating the extra submit/fence overhead entirely.

- **Full kernel fusion**: Extend the fused Yee+UPML shaders to also inline
  dispersive, Mur, and RLC operations for further dispatch reduction.
  See `GPU_FUSION_AND_PIPELINING.md` §8.

- **Audit other `HOST_VISIBLE|HOST_COHERENT`-only allocations for the same
  uncached-read issue** (§4): the excitation/coefficient upload buffers are
  write-only from the CPU so they're unaffected, but any future buffer the
  CPU reads back from should request `HOST_CACHED` too.
