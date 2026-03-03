# Engine_Vulkan — GPU FDTD Architecture

## 1. Overview

`Engine_Vulkan` is a Vulkan compute-based FDTD engine that replaces the CPU
field-update loop with GPU dispatches.  It operates in two modes:

| Mode | Condition | Sync cost |
|------|-----------|-----------|
| **Pure GPU** | No non-excitation CPU extensions | 1 fence wait per `PA->Process()` call |
| **Hybrid** | CPU extensions present (PML, etc.) | 2 full GPU↔CPU round-trips per timestep |

Excitation is always applied on the GPU (dedicated compute shader) to avoid a
CPU round-trip.

---

## 2. Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                         GPU  VRAM                           │
│                                                             │
│  ┌──────────┐ ┌──────────┐   ┌────┐┌────┐┌────┐┌────┐     │
│  │ volt_buf │ │ curr_buf │   │ vv ││ vi ││ ii ││ iv │     │
│  │ 176 MB   │ │ 176 MB   │   │coef││coef││coef││coef│     │
│  │ RW+Xfer  │ │ RW+Xfer  │   │ RO │ │ RO ││ RO ││ RO │    │
│  └─┬────────┘ └─┬────────┘   └────┘└────┘└────┘└────┘     │
│    │ReBAR map   │ReBAR map                                  │
│    │            │           ┌────────────┐                  │
│    ▼            ▼           │ staging_buf│ (host-visible)   │
│  m_voltMapped  m_currMapped │  176 MB    │                  │
│  (CPU pointer) (CPU pointer)└────────────┘                  │
│                                                             │
│  ┌──────────────────────────────────────────────┐           │
│  │ Excitation buffers (idx, amp, delay, signal) │           │
│  │ Per volt + per curr                          │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
         │ PCIe BAR (ReBAR)
         ▼
┌─────────────────────────────────────────────────────────────┐
│                        CPU  RAM                             │
│                                                             │
│  volt_ptr (Engine base)    curr_ptr (Engine base)           │
│  Shadow copy — used only   Shadow copy — used only          │
│  by hybrid path & no-ReBAR by hybrid path & no-ReBAR        │
│  fallback                  fallback                         │
└─────────────────────────────────────────────────────────────┘
```

### ReBAR (Resizable BAR)

The volt/curr buffers are allocated as `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`.
The CPU can read directly from the GPU's VRAM through the PCIe BAR — no staging
buffer copy needed.  Each scalar read takes ~300 ns over PCIe.  Without ReBAR,
a full 176 MB download through the staging buffer is required.

---

## 3. Compute Shaders

Three GLSL compute shaders, compiled to SPIR-V at build time:

| Shader | Bindings | Push Constants | Workgroup | Dispatch |
|--------|----------|----------------|-----------|----------|
| `update_voltages.comp` | volt, curr, vv, vi | `{Nx, Ny, Nz}` | 256×1×1 | `⌈N/256⌉` |
| `update_currents.comp` | curr, volt, ii, iv | `{Nx, Ny, Nz}` | 256×1×1 | `⌈cN/256⌉` |
| `apply_excitation.comp` | field, amp, idx, delay, signal | `{count, ts, sigLen, period}` | 256×1×1 | `⌈count/256⌉` |

Where `N = Nx·Ny·Nz` (all cells) and `cN = (Nx-1)·(Ny-1)·(Nz-1)` (interior cells).

---

## 4. Command Buffer Architecture

Double-buffered command buffers with a utility buffer for one-shot operations:

```
┌────────────────────────────────────┐
│          VkCommandPool             │
│                                    │
│  m_cmdBufs[0]   m_cmdBufs[1]      │    m_utilCmdBuf
│  ┌───────────┐  ┌───────────┐     │    ┌───────────┐
│  │ FDTD work │  │ FDTD work │     │    │ Uploads,  │
│  │ slot 0    │  │ slot 1    │     │    │ downloads,│
│  └─────┬─────┘  └─────┬─────┘     │    │ hybrid    │
│        │              │           │    └─────┬─────┘
│  m_fences[0]    m_fences[1]       │    m_utilFence
│  (start signaled) (start signaled)│    (start unsignaled)
└────────────────────────────────────┘
         │              │
     m_cmdIdx alternates (0 → 1 → 0 → ...)
```

- `m_cmdBufs[m_cmdIdx]` — used by `IterateTS()` for FDTD compute
- `m_utilCmdBuf` — used by `RunSingleCommand()` for uploads, downloads, hybrid path
- Fences start **signaled** so the first `WaitForFences` is a no-op

---

## 5. Pure GPU Path — Main Loop Timeline

This is the critical path for performance.  The main loop in `RunFDTD()` is:

```cpp
while (running) {
    FDTD_Eng->IterateTS(step);   // GPU dispatch
    step = PA->Process();         // CPU probe processing
}
```

### What happens step by step:

```
 Call #    IterateTS(step)                         PA->Process()
 ──────   ─────────────────────────────────       ──────────────────────
   1      record cmdBufs[0], submit+fence[0]      DrainGPU: wait fence[0]
          flip → cmdIdx=1                         then fence[1] (no-op)
          [GPU starts FDTD batch 1]               read m_voltMapped[]
                                                  (probes, HDF5, etc.)

   2      wait fence[1] (no-op, signaled)         DrainGPU: wait fence[1]
          record cmdBufs[1], submit+fence[1]      then fence[0] (no-op)
          flip → cmdIdx=0                         read m_voltMapped[]
          [GPU starts FDTD batch 2]

   3      wait fence[0] (blocks if batch 1        ...
          not done during PA->Process #2)
          record cmdBufs[0], submit+fence[0]
          flip → cmdIdx=1
```

### Timeline diagram (ideal case where compute < process time):

```
         time ──────────────────────────────────────────────────────────►

GPU:     ┌──FDTD─1──┐         ┌──FDTD─2──┐         ┌──FDTD─3──┐
         │ Volt+Exc  │         │ Volt+Exc  │         │ Volt+Exc  │
         │ Curr+Exc  │         │ Curr+Exc  │         │ Curr+Exc  │
         └───────────┘         └───────────┘         └───────────┘

CPU:     ┌record┐              ┌record┐              ┌record┐
         │submit│              │submit│              │submit│
         └──┬───┘              └──┬───┘              └──┬───┘
            │  ┌──PA->Process──┐  │  ┌──PA->Process──┐  │
            │  │ DrainGPU:wait │  │  │ DrainGPU:wait │  │
            │  │ GetVolt (bar) │  │  │ GetVolt (bar) │  │
            │  │ HDF5 writes   │  │  │ HDF5 writes   │  │
            │  └───────────────┘  │  └───────────────┘  │
            ▼                     ▼                     ▼
Fence:   [0]=submit            [1]=submit            [0]=submit
```

### Timeline diagram (actual bottleneck: process < compute):

```
         time ──────────────────────────────────────────────────────────►

GPU:     ┌──FDTD─1──┐┌──FDTD─2──┐┌──FDTD─3──┐┌──FDTD─4──┐
         │ ~50 µs    ││ ~50 µs    ││ ~50 µs    ││ ~50 µs    │
         └───────────┘└───────────┘└───────────┘└───────────┘

CPU:     rec│wait│PA │rec│wait│PA │rec│wait│PA │rec│wait│PA │
                 │        │        │        │        │
                 ▼        ▼        ▼        ▼        ▼
             Drain:    Drain:   Drain:   Drain:   ...
             wait[0]   wait[1]  wait[0]  wait[1]
             ~50 µs    ~50 µs   ~50 µs   ~50 µs
             STALL!    STALL!   STALL!   STALL!
```

**This is the current bottleneck.** `PA->Process()` finishes in microseconds
for most probe types between sync points, but `DrainGPU()` blocks until the
GPU compute completes.  The GPU spends ~50 µs per timestep (for ~15M cells),
and the CPU can process probes faster than that, so the GPU is always the
bottleneck → `DrainGPU()` always stalls.

---

## 6. DrainGPU / FlushGPU — Synchronization

```
DrainGPU()                          FlushGPU()
┌─────────────────────────┐         ┌────────────────────┐
│ for i in {0, 1}:        │         │ calls DrainGPU()   │
│   if gpuInFlight[i]:    │         └────────────────────┘
│     vkWaitForFences(    │
│       fences[i])        │
│     gpuInFlight[i]=false│
└─────────────────────────┘
```

`DrainGPU()` is called from:
- **`GetVolt()` / `GetCurr()`** — every probe read
- **`SyncFieldsToHost()`** — full field download (no-ReBAR fallback)
- **`SyncFieldsToDevice()`** — before CPU→GPU upload

Since probes call `GetVolt`/`GetCurr` hundreds of times per `PA->Process()`,
`DrainGPU()` is called hundreds of times — but only the first call actually
waits (subsequent calls see `gpuInFlight[i] == false`).

---

## 7. Field Access Paths

### GetVolt / GetCurr (Read)

```
GetVolt(n, x, y, z)
        │
        ▼
   DrainGPU()  ◄─── wait for both fences (usually already done)
        │
        ├── m_voltMapped != null (ReBAR)?
        │         │
        │    YES  ▼
        │   return m_voltMapped[linear index]  ◄── PCIe BAR read (~300 ns)
        │
        └── NO
             │
             ▼
        m_hostDirty?
             │
        YES  ▼
        SyncFieldsToHost()  ◄── staging buffer: 2 × 176 MB DMA download
             │
             ▼
        return Engine::GetVolt()  ◄── read from CPU shadow array
```

### SetVolt / SetCurr (Write)

```
SetVolt(n, x, y, z, v)
        │
        ▼
   m_hostDirty? ──YES──► SyncFieldsToHost()
        │
        ▼
   Engine::SetVolt()  ◄── write to CPU shadow array
        │
        ▼
   m_deviceDirty = true  ◄── next IterateTS will upload
```

---

## 8. Hybrid Path (with CPU Extensions)

When non-excitation CPU extensions exist (PML, lumped elements, etc.), the
engine must sync after every voltage update and every current update:

```
 Per timestep (1 of iterTS):

 CPU                          GPU                          Transfers
 ───                          ───                          ─────────
 DoPreVoltageUpdates()
 SyncFieldsToDevice()                                     ◄── upload if dirty
                              ┌─────────────────┐
                              │ Voltage update   │
                              │ + V excitation   │
                              └────────┬────────┘
                                       │
 SyncFieldsToHost()                    ▼          ◄── 176 MB download × 2
 DoPostVoltageUpdates()
 Apply2Voltages()
 DoPreCurrentUpdates()
 SyncFieldsToDevice()                                     ◄── upload if dirty
                              ┌─────────────────┐
                              │ Current update   │
                              │ + I excitation   │
                              └────────┬────────┘
                                       │
 SyncFieldsToHost()                    ▼          ◄── 176 MB download × 2
 DoPostCurrentUpdates()
 Apply2Current()
 m_deviceDirty = true
 ─── next timestep ───
```

This path has **4 GPU↔CPU synchronization points** per timestep.
Each sync involves `RunSingleCommand` (submit + fence wait) plus potentially
a staging-buffer transfer.  This is inherently slow.

---

## 9. Initialization Sequence

```
Engine_Vulkan::New(op)
    │
    ▼
 Constructor
    │  Set all handles to VK_NULL_HANDLE
    │  Set all pointers to nullptr
    │
    ▼
 Init()
    ├── Engine::Init()         ◄── allocate CPU shadow arrays, create extensions
    ├── InitVulkan()           ◄── VkInstance → PhysDevice → Device → Queue
    │                               CmdPool → 3 CmdBufs → 3 Fences
    ├── CreateBuffers()        ◄── volt/curr (ReBAR probe → fallback device-local)
    │                               vv/vi/ii/iv (device-local), staging (host-visible)
    ├── CreateDescriptors()    ◄── descriptor sets for FDTD + excitation
    ├── CreatePipelines()      ◄── compile SPIR-V → 3 compute pipelines
    ├── UploadCoefficients()   ◄── CPU vv/vi/ii/iv → staging → GPU
    ├── SetupGPUExcitation()   ◄── convert CPU excitation → GPU buffers
    │                               remove CPU excitation extension
    │                               detect hasCPUExtensions
    ├── Upload initial fields  ◄── via ReBAR memcpy or staging buffer
    └── Print GPU info
```

---

## 10. Where the Performance Issues Arise

### Issue 1: Sequential Main Loop

```
┌──────────────────────────────────────────────────────────┐
│  while (running) {                                       │
│      FDTD_Eng->IterateTS(step);  // GPU: ~50 µs         │
│      step = PA->Process();        // CPU: ~1-5 ms        │
│  }                                    ▲                  │
│                                       │                  │
│  PA->Process calls GetVolt/GetCurr    │                  │
│  which calls DrainGPU() ──────────────┘                  │
│  GPU is IDLE during entire PA->Process                   │
└──────────────────────────────────────────────────────────┘
```

The GPU finishes in ~50 µs but the CPU takes 1-5 ms to process probes.
During that entire CPU-side processing, the GPU sits idle.  This means
**GPU utilization is roughly `50µs / (50µs + 2000µs) ≈ 2.5%`**.

### Issue 2: DrainGPU Blocks Before PA->Process Can Start

With the current double-buffered design, `IterateTS` submits fire-and-forget,
then `PA->Process()` immediately calls `GetVolt` → `DrainGPU()` → fence wait.
In practice, the GPU hasn't finished the just-submitted work yet, so the CPU
blocks for the full ~50 µs compute time.

### Issue 3: PCIe BAR Read Latency

Each `GetVolt`/`GetCurr` call reads 4 bytes over PCIe (~300 ns per read).
A voltage probe at 1 point = 3 reads.  A mode-match probe integrating over
a cross-section might do 50,000+ reads = ~15 ms, which actually exceeds the
GPU compute time significantly.

### Possible Future Improvements

1. **Batch multiple timesteps** between `PA->Process()` calls — the `step`
   variable from `PA->Process()` already tells us how many steps until the
   next probe triggers.  More timesteps per batch = higher GPU utilization.
   (This already works: `IterateTS(step)` records `step` timesteps into one
   command buffer.  The issue is when `step == 1`.)

2. **Move probe processing to GPU** — compute integrals, DFTs, and field
   probes as additional compute shaders.  This would eliminate the
   GPU→CPU→GPU round-trip entirely.

3. **Asynchronous probe download** with double-buffered snapshots (was tried;
   the 2×176 MB VRAM→VRAM copy took longer than the compute itself).

4. **Sparse field readback** — instead of reading the full field through
   ReBAR, have the GPU write only the probed field values into a small
   buffer that the CPU can read quickly.
