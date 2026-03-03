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

### 3. Async Processing Thread (latency hiding)

**Infrastructure**: `StartAsyncProcessing()`, `AsyncWorkerLoop()`,
`TriggerAsyncProcessing()`, `WaitAsyncProcessing()`

A background `std::thread` with mutex/condition_variable synchronization that can
run `PA->Process()` concurrently with the next GPU batch. The pattern:

```
GPU batch N submits → fence
CPU: DrainGPU (wait fence) → fields now stable
CPU: TriggerAsyncProcessing() → background thread processes data
CPU: submit GPU batch N+1 → GPU runs while background thread works
CPU: WaitAsyncProcessing() → ensure previous processing done before next drain
```

Currently provides the thread infrastructure — the actual integration into `RunFDTD()`
requires hooking the `PA->Process()` call (future work, minor changes to `openems.cpp`).

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
| `FDTD/engine_vulkan.h` | Added energy reduction members (`m_energyPartialBuf`, `m_energyMapped`, descriptor/pipeline objects, `EnergyPC` struct, `ENERGY_NUM_WG`), async thread members (`m_asyncThread`, mutex, condition variables), public methods (`CalcFastEnergyGPU`, `CalcVoltageIntegralGPU`, `CalcCurrentIntegralGPU`, async methods) |
| `FDTD/engine_vulkan.cpp` | Added `#include "tools/constants.h"`. Init: calls `SetupGPU_EnergyReduction()`. Destructor: async thread shutdown. CleanupVulkan: calls `CleanupGPU_EnergyReduction()`. New method implementations: `SetupGPU_EnergyReduction()`, `CleanupGPU_EnergyReduction()`, `CalcFastEnergyGPU()`, `CalcVoltageIntegralGPU()`, `CalcCurrentIntegralGPU()`, `StartAsyncProcessing()`, `AsyncWorkerLoop()`, `TriggerAsyncProcessing()`, `WaitAsyncProcessing()` |
| `FDTD/engine_interface_fdtd.cpp` | Added `#include "engine_vulkan.h"` (under `WITH_GPU`). GPU fast-path dispatch at top of `CalcFastEnergy()` and `CalcVoltageIntegral()` |

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

### Async Thread Lifecycle
```
StartAsyncProcessing()
  └─ spawns m_asyncThread running AsyncWorkerLoop()

~Engine_Vulkan()
  └─ m_asyncShutdown = true
     m_asyncCond.notify_one()
     m_asyncThread.join()
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
