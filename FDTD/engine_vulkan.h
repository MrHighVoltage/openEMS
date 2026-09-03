/*
*	Copyright (C) 2026 openEMS contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifndef ENGINE_VULKAN_H
#define ENGINE_VULKAN_H

#ifdef WITH_GPU

#include "engine.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <functional>
#include <algorithm>
#include <string>
#include <chrono>

class Operator_Ext_Excitation;
class Operator_Ext_UPML;
class Operator_Ext_LorentzMaterial;
class Engine_Ext_SteadyState;
class Operator_Ext_TFSF;
class Operator_Ext_Mur_ABC;
class Operator_Ext_LumpedRLC;

/*!
 * \brief GPU-accelerated FDTD engine using Vulkan compute shaders.
 *
 * The voltage and current fields live on the GPU as storage buffers.
 * The CPU-side volt_ptr / curr_ptr (inherited from Engine) serve as a
 * shadow copy that is lazily synchronised on the first GetVolt / GetCurr
 * call after a GPU update.
 *
 * Two iteration modes:
 *  - **Pure GPU** (no non-excitation extensions): all timesteps are batched
 *    in a single Vulkan command buffer with zero CPU involvement.
 *  - **Hybrid** (non-excitation extensions present): per-timestep with
 *    GPU↔CPU sync around extension hooks.
 *
 * The excitation is always applied on the GPU through a dedicated compute
 * shader to avoid CPU round-trips.
 */
class Engine_Vulkan : public Engine
{
public:
	//! Single Vulkan baseline for the whole engine -- instance creation,
	//! preflight probe, and device selection all negotiate against this one
	//! value.  1.3 is the lowest version that provides, as core, everything
	//! the engine's performance work needs: subgroup size control and
	//! synchronization2 (1.3), timeline semaphores and 8-bit storage (1.2),
	//! subgroup arithmetic and 16-bit storage (1.1).
	static constexpr uint32_t API_VERSION = VK_API_VERSION_1_3;

	//! Optional device capabilities negotiated at device creation.  Each flag
	//! is only true when the feature was reported supported *and* enabled.
	struct DeviceCaps
	{
		bool subgroupSizeControl   = false;
		bool computeFullSubgroups  = false;
		bool synchronization2      = false;
		bool timelineSemaphore     = false;
		bool storageBuffer16       = false;
		bool storageBuffer8        = false;
		bool shaderInt16           = false;
		bool shaderInt8            = false;
		uint32_t minSubgroupSize   = 0;
		uint32_t maxSubgroupSize   = 0;
	};
	const DeviceCaps& GetDeviceCaps() const { return m_caps; }
	//! Width actually chosen for the compressed-operator index (8/16/32 bits).
	uint32_t GetOpIdxBits() const { return m_opIdxBits; }

	static Engine_Vulkan* New(const Operator* op);
	//! Prefer ReBAR HOST_VISIBLE allocations for main field buffers.
	//! When false, volt/curr stay device-local for maximum dGPU bandwidth.
	static void SetPreferReBARFieldBuffers(bool prefer) { s_preferReBARFieldBuffers = prefer; }
	static bool GetPreferReBARFieldBuffers() { return s_preferReBARFieldBuffers; }
	//! Enable per-batch GPU timestamp profiling and fence-wait accounting.
	static void SetEnableProfiling(bool enable) { s_enableProfiling = enable; }
	static bool GetEnableProfiling() { return s_enableProfiling; }
	//! Fast-fail probe that allocates and frees representative GPU buffers
	//! for the given grid size before costly operator preparation begins.
	//! strictCoeffReserve=true also reserves a conservative coefficient/index budget.
	static bool PreflightAllocationForGrid(unsigned int Nx, unsigned int Ny, unsigned int Nz,
	                                      bool strictCoeffReserve, std::string* errMsg = nullptr);
	virtual ~Engine_Vulkan();

	virtual void Init();
	virtual void Reset();

	virtual bool IterateTS(unsigned int iterTS);

	//! GPU-side energy estimation — avoids downloading entire field arrays.
	double CalcFastEnergyGPU() const;

	//! GPU-side voltage line integral — avoids full field download for probes.
	double CalcVoltageIntegralGPU(const unsigned int* start, const unsigned int* stop) const;
	//! GPU-side current line integral.
	double CalcCurrentIntegralGPU(const unsigned int* start, const unsigned int* stop) const;

	// --- Pipelined processing support -------------------------------------
	//! Begin recording cell accesses. Call before PA->Process().
	void StartProbeRecording() { m_recordedCells.clear(); m_recordingProbeAccess = true; }
	//! Stop recording and build probe cache from recorded cells.
	//! Must be called after PA->Process(). Creates GPU gather resources.
	void SetupProbeCache(class ProcessingArray* PA);
	//! After DrainGPU, snapshot gathered probe values into CPU cache.
	void SnapshotProbeCache();
	//! Enable/disable pipelined read mode. When enabled, GetVolt/GetCurr
	//! return from the CPU-side probe cache instead of VRAM.
	void EnablePipelinedReading(bool enable) { m_pipelinedReading = enable; }
	//! Check if pipelined processing is active and usable.
	//! Probes use the probe cache; field dumps use the async download pipeline
	//! (BeginAsyncFieldDownload). Either alone is enough to make the
	//! speculative-submission loop worthwhile -- it's the submit-ahead
	//! structure (next batch queued before this state's data is consumed)
	//! that lets both overlap with GPU compute, not just the probe cache.
	bool HasPipelinedProcessing() const
	{ return (m_hasGPU_Probes || m_hasFieldDumps) && !m_hasCPUExtensions; }
	//! Submit a speculative batch that does NOT advance the public numTS.
	void SubmitSpeculative(unsigned int nTS);
	//! Drain the speculative batch and advance numTS.
	void CommitSpeculative();
	//! Wait for all in-flight GPU work to complete.
	void DrainGPU() const;

	// --- Field access (lazy sync from GPU) --------------------------------
	// Pure-GPU mode with ReBAR: DrainGPU() (typically a no-op since the
	// ~50 µs compute is already done) then read directly from the
	// persistently mapped VRAM pointer — zero copies.
	// Hybrid mode / no-ReBAR fallback: download via staging buffer.
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		uint64_t linIdx = (uint64_t)n * m_fieldN + x * m_strideYZ + y * numLines[2] + z;
		if (m_recordingProbeAccess)
		{
			m_recordedCells.push_back((uint64_t)0 << 48 | linIdx);
			// still need to return a value — use zero during recording
			if (!m_gpuDrained) DrainGPU();
			if (m_voltMapped) return m_voltMapped[linIdx];
			if (m_hostDirty) SyncFieldsToHost();
			return Engine::GetVolt(n,x,y,z);
		}
		if (m_pipelinedReading)
		{
			const uint64_t key = (uint64_t)0 << 48 | linIdx;
			auto it = std::lower_bound(m_probeCacheKeys.begin(), m_probeCacheKeys.end(), key);
			if (it != m_probeCacheKeys.end() && *it == key)
			{
				m_probeCacheHits++;
				return m_probeCacheCPU[(size_t)(it - m_probeCacheKeys.begin())];
			}
			m_probeCacheMisses++;
			// Fallback: not in cache — must drain GPU
		}
		if (!m_gpuDrained) DrainGPU();
		if (m_voltMapped)
			return m_voltMapped[linIdx];
		if (m_hostDirty) SyncFieldsToHost();
		return Engine::GetVolt(n,x,y,z);
	}
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const
	{ return GetVolt(n, pos[0], pos[1], pos[2]); }
	inline virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		uint64_t linIdx = (uint64_t)n * m_fieldN + x * m_strideYZ + y * numLines[2] + z;
		if (m_recordingProbeAccess)
		{
			m_recordedCells.push_back((uint64_t)1 << 48 | linIdx);
			if (!m_gpuDrained) DrainGPU();
			if (m_currMapped) return m_currMapped[linIdx];
			if (m_hostDirty) SyncFieldsToHost();
			return Engine::GetCurr(n,x,y,z);
		}
		if (m_pipelinedReading)
		{
			const uint64_t key = (uint64_t)1 << 48 | linIdx;
			auto it = std::lower_bound(m_probeCacheKeys.begin(), m_probeCacheKeys.end(), key);
			if (it != m_probeCacheKeys.end() && *it == key)
			{
				m_probeCacheHits++;
				return m_probeCacheCPU[(size_t)(it - m_probeCacheKeys.begin())];
			}
			m_probeCacheMisses++;
		}
		if (!m_gpuDrained) DrainGPU();
		if (m_currMapped)
			return m_currMapped[linIdx];
		if (m_hostDirty) SyncFieldsToHost();
		return Engine::GetCurr(n,x,y,z);
	}
	inline virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const
	{ return GetCurr(n, pos[0], pos[1], pos[2]); }

	inline virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT v)
	{ if (m_hostDirty) SyncFieldsToHost(); Engine::SetVolt(n,x,y,z,v); m_deviceDirty = true; }
	inline virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT v)
	{ if (m_hostDirty) SyncFieldsToHost(); Engine::SetVolt(n,pos,v); m_deviceDirty = true; }
	inline virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT v)
	{ if (m_hostDirty) SyncFieldsToHost(); Engine::SetCurr(n,x,y,z,v); m_deviceDirty = true; }
	inline virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT v)
	{ if (m_hostDirty) SyncFieldsToHost(); Engine::SetCurr(n,pos,v); m_deviceDirty = true; }

	//! Copy GPU field data → CPU shadow.  Logically const (mutable internals).
	void SyncFieldsToHost() const;
	//! Copy CPU shadow → GPU buffers (after CPU-side extension modifications).
	void SyncFieldsToDevice();

	// --- Async field-dump download (overlaps large dump readback with GPU compute) ---
	//! Tell the engine whether any GPU field dumps (ProcessFields) are registered.
	//! Gates the speculative pre-fetch below; a no-op (safe) when left false.
	void SetHasFieldDumps(bool v) { m_hasFieldDumps = v; }
	bool HasFieldDumps() const { return m_hasFieldDumps; }
	//! True if a second, independent hardware queue is available for the
	//! dump download pipeline (see SetupDumpTransferQueue()).
	bool HasDumpTransferQueue() const { return m_hasDumpQueue; }
	//! Kick off (non-blocking on the CPU) a snapshot + download of the
	//! current committed field state for an upcoming dump, so the slow PCIe
	//! leg runs concurrently with the next speculative GPU batch and with
	//! CPU-side processing. SyncFieldsToHost() picks up the result lazily.
	//! Safe to call even when no dump is imminent (self-flushes any stale
	//! pending request first) or when the dump queue isn't available (no-op).
	void BeginAsyncFieldDownload();

protected:
	Engine_Vulkan(const Operator* op);

private:
	static constexpr int CMD_RING_SIZE = 16;

	// ---- Vulkan core --------------------------------------------------
	VkInstance       m_instance;
	VkPhysicalDevice m_physDevice;
	VkDevice         m_device;
	VkQueue          m_computeQueue;
	uint32_t         m_computeQueueFamily;
	VkQueue          m_transferQueue;
	uint32_t         m_transferQueueFamily;
	bool             m_hasDedicatedTransferQueue;
	DeviceCaps       m_caps;

	// ---- GPU buffers --------------------------------------------------
	VkBuffer       m_voltBuf, m_currBuf;
	VkDeviceMemory m_voltMem, m_currMem;
	// Compressed operator coefficient buffers (much smaller than full-size)
	VkBuffer       m_opIndexBuf;    //!< Per-cell index into compressed tables [N]
	VkDeviceMemory m_opIndexMem;
	VkBuffer       m_vvCompBuf, m_viCompBuf, m_iiCompBuf, m_ivCompBuf;
	VkDeviceMemory m_vvCompMem, m_viCompMem, m_iiCompMem, m_ivCompMem;
	VkDeviceSize   m_opIndexBufSize; //!< packed size of the opIdx buffer, 4-byte aligned
	uint32_t       m_opIdxBits;      //!< 8, 16 or 32 -- width of one packed opIdx entry
	VkDeviceSize   m_coeffCompBufSize; //!< 3 * numCompressed * sizeof(float)
	VkBuffer       m_stagingBuf;
	VkDeviceMemory m_stagingMem;
	VkDeviceSize   m_fieldBufSize; // 3 * Nx * Ny * Nz * sizeof(float)
	FDTD_FLOAT*    m_voltMapped;   //!< Persistently mapped VRAM pointer (ReBAR), or nullptr
	FDTD_FLOAT*    m_currMapped;   //!< Persistently mapped VRAM pointer (ReBAR), or nullptr
	uint32_t       m_fieldN;       //!< Nx * Ny * Nz (cached for fast index computation)
	uint32_t       m_strideYZ;     //!< Ny * Nz (cached for fast index computation)

	// ---- Excitation GPU data ------------------------------------------
	VkBuffer       m_excVoltIdxBuf, m_excVoltAmpBuf, m_excVoltDelayBuf;
	VkDeviceMemory m_excVoltIdxMem, m_excVoltAmpMem, m_excVoltDelayMem;
	VkBuffer       m_excCurrIdxBuf, m_excCurrAmpBuf, m_excCurrDelayBuf;
	VkDeviceMemory m_excCurrIdxMem, m_excCurrAmpMem, m_excCurrDelayMem;
	VkBuffer       m_excSignalVoltBuf, m_excSignalCurrBuf;
	VkDeviceMemory m_excSignalVoltMem, m_excSignalCurrMem;
	unsigned int   m_excVoltCount, m_excCurrCount;
	unsigned int   m_excSignalLen;
	int            m_excSignalPeriod;

	// ---- Descriptors --------------------------------------------------
	VkDescriptorSetLayout m_fdtdDescLayout;
	VkDescriptorSetLayout m_excDescLayout;
	VkDescriptorPool      m_descPool;
	VkDescriptorSet       m_updateVoltDescSet;
	VkDescriptorSet       m_updateCurrDescSet;
	VkDescriptorSet       m_excVoltDescSet;
	VkDescriptorSet       m_excCurrDescSet;

	// ---- Pipelines ----------------------------------------------------
	VkPipelineLayout m_fdtdPipeLayout;
	VkPipelineLayout m_excPipeLayout;
	VkPipeline       m_updateVoltPipeline;
	VkPipeline       m_updateCurrPipeline;
	VkPipeline       m_excPipeline;

	// ---- Command / sync -----------------------------------------------
	// Double-buffered command submission: IterateTS records pure FDTD
	// compute into m_cmdBufs[m_cmdIdx] and submits fire-and-forget.
	// The next call uses the other slot. GetVolt/GetCurr calls DrainGPU()
	// (usually a no-op, ~50 µs compute already done) then reads directly
	// from the ReBAR-mapped VRAM pointer — zero copies, zero stalls.
	VkCommandPool   m_cmdPool;
	VkCommandBuffer m_cmdBufs[CMD_RING_SIZE];   //!< Ring-buffered FDTD + copy command buffers
	VkCommandBuffer m_utilCmdBuf;   //!< Utility cmd buf (uploads, hybrid path)
	VkFence         m_fences[CMD_RING_SIZE];    //!< One per m_cmdBufs slot
	VkFence         m_utilFence;    //!< Fence for utility (upload/download) work
	int             m_cmdIdx;       //!< Index into m_cmdBufs/m_fences ring
	mutable bool    m_gpuInFlight[CMD_RING_SIZE]; //!< Whether m_cmdBufs[i] is submitted but not waited

	// Optional transfer-queue path for probe output copies
	VkCommandPool   m_transferCmdPool;
	VkCommandBuffer m_transferCmdBufs[CMD_RING_SIZE];
	VkFence         m_transferFences[CMD_RING_SIZE];
	VkSemaphore     m_probeCopySem[CMD_RING_SIZE];
	mutable bool    m_probeTransferInFlight[CMD_RING_SIZE];

	// ---- State --------------------------------------------------------
	mutable bool m_hostDirty;
	bool         m_deviceDirty;
	bool         m_hasCPUExtensions;
	bool         m_hasGPUExcitation;
	bool         m_validateFields;
	bool         m_validateStages;
	bool         m_validateMagnitude;
	uint32_t     m_validateTraceIndex;
	mutable bool m_gpuDrained;       //!< true after DrainGPU(), cleared on submit
	unsigned int m_maxTSPerSubmit;   //!< Max timesteps per pure-GPU vkQueueSubmit
	double       m_chunkProgressIntervalSec; //!< Minimum interval between chunk-progress prints
	mutable bool m_chunkProgressHasLastPrint;
	mutable std::chrono::steady_clock::time_point m_chunkProgressLastPrint;

	// ---- Pipeline cache (faster shader load on restart) ---------------
	VkPipelineCache m_pipelineCache;

	// ---- Runtime configuration (static) ---------------------------------
	static bool s_preferReBARFieldBuffers;
	static bool s_enableProfiling;

	// ---- Optional profiling ---------------------------------------------
	bool m_profileEnabled = false;
	VkQueryPool m_tsQueryPool = VK_NULL_HANDLE;  //!< 2 queries per command slot
	float m_timestampPeriodNs = 0.0f;
	bool m_profilePhaseEnabled = false;
	uint32_t m_profileQueriesPerSlot = 2;
	bool m_profilePending[CMD_RING_SIZE] = {};
	unsigned int m_profileSubmittedTS[CMD_RING_SIZE] = {};
	mutable uint64_t m_profileBatchCount = 0;
	mutable uint64_t m_profileTSCount = 0;
	mutable double m_profileGpuMs = 0.0;
	mutable double m_profileWaitMs = 0.0;
	mutable double m_profileComputeMs = 0.0;
	mutable double m_profileProbeMs = 0.0;
	mutable double m_profileTailMs = 0.0;

	// ---- Internal helpers ---------------------------------------------
	void InitVulkan();
	void CreateBuffers();
	void CreateDescriptors();
	void CreatePipelines();
	void UploadCoefficients();
	void SetupGPUExcitation();
	void SetupGPU_SteadyState();
	void RecordSteadyStateSample(VkCommandBuffer cmd, uint32_t ts) const;
	void UpdateSteadyStateResult();
	void SetupGPU_LocalABC();
	void RecordLocalABCPhase(VkCommandBuffer cmd, uint32_t phase) const;
	void CleanupVulkan();
	void SetupProfiling();
	void CollectProfileForSlot(int slot) const;
	void PrintProfileSummary() const;

	//! Submit any pending batched GPU command buffer and wait for completion.
	//! Used only by hybrid path and non-ReBAR fallback.
	void FlushGPU() const;

	//! Upload CPU data to a device-local buffer via the staging buffer.
	void UploadToDeviceBuffer(VkBuffer dst, const void* data, VkDeviceSize size);
	//! Download device-local buffer to CPU via the staging buffer.
	void DownloadFromDeviceBuffer(VkBuffer src, void* data, VkDeviceSize size) const;

	//! Create a VkBuffer + allocate and bind VkDeviceMemory.
	void CreateBufferWithMemory(VkDeviceSize size, VkBufferUsageFlags usage,
	                            VkMemoryPropertyFlags memProps,
	                            VkBuffer& buf, VkDeviceMemory& mem);
	//! Allocation probe used for optional memory classes such as ReBAR.
	bool TryCreateBufferWithMemory(VkDeviceSize size, VkBufferUsageFlags usage,
	                               VkMemoryPropertyFlags memProps,
	                               VkBuffer& buf, VkDeviceMemory& mem);

	VkShaderModule CreateShaderModule(const unsigned char* code, size_t codeSize);
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
	//! Like FindMemoryType but returns UINT32_MAX on failure instead of throwing.
	uint32_t FindMemoryTypeSoft(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;

	//! Record and submit a one-shot command buffer, wait for completion.
	void RunSingleCommand(std::function<void(VkCommandBuffer)> func) const;

	//! Push constant structs matching the shader layouts.
	struct GridPC   { uint32_t Nx, Ny, Nz, numComp; };
	struct FusedPC  { uint32_t Nx, Ny, Nz, numComp, numPmlRegions; };
	struct ExcPC    { uint32_t count; int32_t numTS; uint32_t sigLen; int32_t period; };

	//! Push constant struct for UPML shaders (36 bytes = 9 × uint32).
	struct PmlPC    { uint32_t Nx, Ny, Nz, pNx, pNy, pNz, pStartX, pStartY, pStartZ; };
	//! Push constant struct for dispersive material shaders.
	struct DispPC   { uint32_t count, Nx, Ny, Nz, hasLorADE; };
	//! Push constant struct for TF/SF shaders (same as ExcPC).
	struct TfsfPC   { uint32_t count; int32_t numTS; uint32_t sigLen; int32_t sigPeriod; };
	//! Push constant struct for Mur ABC shaders.
	struct MurPC    { uint32_t Nx, Ny, Nz, ny, lineNr, lineNrShift, numLinesP, numLinesPP; };
	//! Push constant struct for LumpedRLC shaders.
	struct RlcPC    { uint32_t count; };
	//! Push constant struct for probe gather shader.
	struct ProbePC  { uint32_t count; };
	struct SteadyPC { uint32_t count, ringSize, timestep; };
	struct LocalAbcPC {
		uint32_t Nx, Ny, Nz, ny, nyP, nyPP;
		uint32_t startX, startY, startZ;
		uint32_t voltBoundary, voltShift, currBoundary, currShift;
		uint32_t sizeP, sizePP, countVolt, countCurr, phase;
	};

	//! Helper pairing a VkBuffer with its VkDeviceMemory.
	struct GpuBuf {
		VkBuffer buffer       = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
	};
	void CreateGpuBuf(GpuBuf& gb, VkDeviceSize size, VkBufferUsageFlags usage,
	                  VkMemoryPropertyFlags memProps);
	void DestroyGpuBuf(GpuBuf& gb);
	void UploadGpuBuf(GpuBuf& gb, const void* data, VkDeviceSize size);

	// ---- Per-instance GPU data for each extension type ----------------

	//! Per-UPML-region GPU state.
	struct GpuUPMLData {
		GpuBuf voltFlux, currFlux;                       // auxiliary flux fields
		GpuBuf pmlIdx;                                   // narrow index into the shared coeff tables
		VkDeviceSize pmlIdxBufSize = 0;
		//! Table index per coefficient entry, [3][pNx][pNy][pNz], kept host-side
		//! so SetupFusedUPML can concatenate the regions without a readback.
		std::vector<uint32_t> coefIdx;
		VkDescriptorSet preVoltDesc  = VK_NULL_HANDLE;
		VkDescriptorSet postVoltDesc = VK_NULL_HANDLE;
		VkDescriptorSet preCurrDesc  = VK_NULL_HANDLE;
		VkDescriptorSet postCurrDesc = VK_NULL_HANDLE;
		PmlPC pc{};
		uint32_t totalCells = 0;                          // pNx*pNy*pNz for dispatch
	};

	//! Per-order dispersive material GPU state.
	struct GpuDispData {
		GpuBuf voltADE, voltLorADE;                       // voltage ADE state
		GpuBuf currADE, currLorADE;                       // current ADE state
		GpuBuf posIdx;                                    // [3][count] flat position indices
		GpuBuf vIntADE, vExtADE, vLorADE;                 // voltage coefficients
		GpuBuf iIntADE, iExtADE, iLorADE;                 // current coefficients
		VkDescriptorSet preVoltDesc   = VK_NULL_HANDLE;
		VkDescriptorSet preCurrDesc   = VK_NULL_HANDLE;
		VkDescriptorSet applyVoltDesc = VK_NULL_HANDLE;
		VkDescriptorSet applyCurrDesc = VK_NULL_HANDLE;
		DispPC voltPC{}, currPC{};
		uint32_t count = 0;
		bool voltLorADEOn = false, currLorADEOn = false;
		bool voltADEOn = false, currADEOn = false;
	};

	//! TF/SF GPU state (all injection points flattened).
	struct GpuTFSFData {
		GpuBuf voltAmp, voltDelta, voltIdx, voltDelay, voltSignal;
		GpuBuf currAmp, currDelta, currIdx, currDelay, currSignal;
		VkDescriptorSet voltDesc = VK_NULL_HANDLE;
		VkDescriptorSet currDesc = VK_NULL_HANDLE;
		uint32_t voltCount = 0, currCount = 0;
	};

	//! Per-face Mur ABC GPU state.
	struct GpuMurData {
		GpuBuf murNyP, murNyPP;                           // intermediate saved values
		GpuBuf coeffNyP, coeffNyPP;                       // Mur coefficients
		VkDescriptorSet preDesc   = VK_NULL_HANDLE;
		VkDescriptorSet postDesc  = VK_NULL_HANDLE;
		VkDescriptorSet applyDesc = VK_NULL_HANDLE;
		MurPC pc{};
		uint32_t totalCells = 0;
		unsigned int startTS = 0;                         // timestep after which Mur activates
	};

	//! Lumped RLC GPU state.
	struct GpuRLCData {
		GpuBuf il, vdn0, vdn1, vdn2, jn0, jn1, jn2;     // ADE state
		GpuBuf linIdx;                                    // pre-linearized positions
		GpuBuf i2v, ilv, vvd, vv2, vj1, vj2, ib0, b1, b2; // coefficients
		VkDescriptorSet preDesc   = VK_NULL_HANDLE;
		VkDescriptorSet applyDesc = VK_NULL_HANDLE;
		uint32_t count = 0;
	};

	//! Probe gather GPU state.
	struct GpuProbeData {
		GpuBuf probeIdx, fieldSel, output;
		float* outputMapped = nullptr;                    // ReBAR-mapped for readback
		VkDescriptorSet desc = VK_NULL_HANDLE;
		uint32_t count = 0;
	};

	// ---- Fused Yee+UPML pipeline (concatenated PML buffers) ----------
	bool m_hasFusedUPML = false;
	GpuBuf m_fusedVoltFlux;     //!< Concatenated volt flux (all PML regions)
	GpuBuf m_fusedCurrFlux;     //!< Concatenated curr flux (all PML regions)
	GpuBuf m_fusedPmlIdx;       //!< Concatenated per-entry PML coefficient index
	VkDeviceSize m_fusedPmlIdxBufSize = 0;
	GpuBuf m_fusedPmlRegionInfo; //!< PML region metadata SSBO
	uint32_t m_fusedTotalPmlCells = 0;  //!< Sum of all PML regions' cell counts
	VkDescriptorSetLayout m_fusedDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout      m_fusedPipeLayout = VK_NULL_HANDLE;
	VkPipeline            m_fusedVoltPipeline = VK_NULL_HANDLE;
	VkPipeline            m_fusedCurrPipeline = VK_NULL_HANDLE;
	VkDescriptorPool      m_fusedDescPool = VK_NULL_HANDLE;
	VkDescriptorSet       m_fusedVoltDescSet = VK_NULL_HANDLE;
	VkDescriptorSet       m_fusedCurrDescSet = VK_NULL_HANDLE;
	void SetupFusedUPML();       //!< Create concatenated PML buffers + fused pipeline
	void CleanupFusedUPML();     //!< Destroy fused UPML resources

	// ---- Deduplicated UPML coefficient tables -------------------------
	// Shared by every region and by both the fused and the separate-dispatch
	// path, so an index fetched anywhere means the same table entry.
	GpuBuf m_pmlTabVv, m_pmlTabVvfo, m_pmlTabVvfn;
	GpuBuf m_pmlTabIi, m_pmlTabIifo, m_pmlTabIifn;
	VkDeviceSize m_pmlTabBufSize = 0;
	uint32_t m_pmlTabCount = 0;      //!< number of distinct coefficient tuples
	uint32_t m_pmlIdxBits  = 32;     //!< width of one packed table index
	void BuildUPMLCoeffTables();     //!< Deduplicate the UPML coefficients

	// ---- Extension GPU instances --------------------------------------
	VkDescriptorPool      m_extDescPool = VK_NULL_HANDLE;
	std::vector<GpuUPMLData>  m_gpuUPML;
	std::vector<GpuDispData>  m_gpuDisp;
	GpuTFSFData               m_gpuTFSF;
	std::vector<GpuMurData>   m_gpuMur;
	GpuRLCData                m_gpuRLC;
	GpuProbeData              m_gpuProbes;

	// ---- GPU steady-state observer -----------------------------------
	Engine_Ext_SteadyState* m_gpuSteadyStateExt = nullptr; // owned by Engine::m_Eng_exts
	GpuBuf m_steadyProbeIdx, m_steadyHistory;
	float* m_steadyHistoryMapped = nullptr;
	uint32_t m_steadyProbeCount = 0;
	uint32_t m_steadyPeriod = 0;
	uint32_t m_steadyRingSize = 0;
	uint32_t m_steadyLastCompletedPeriods = 0;
	VkDescriptorSetLayout m_steadyDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_steadyPipeLayout = VK_NULL_HANDLE;
	VkPipeline m_steadyPipeline = VK_NULL_HANDLE;
	VkDescriptorPool m_steadyDescPool = VK_NULL_HANDLE;
	VkDescriptorSet m_steadyDescSet = VK_NULL_HANDLE;

	struct GpuLocalABCData {
		GpuBuf voltState, currState, k1, k2;
		VkDescriptorSet voltDesc = VK_NULL_HANDLE;
		VkDescriptorSet currDesc = VK_NULL_HANDLE;
		LocalAbcPC pc{};
		bool superAbsorption = false;
	};
	std::vector<GpuLocalABCData> m_gpuLocalABC;
	VkDescriptorSetLayout m_localAbcDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_localAbcPipeLayout = VK_NULL_HANDLE;
	VkPipeline m_localAbcPipeline = VK_NULL_HANDLE;
	VkDescriptorPool m_localAbcDescPool = VK_NULL_HANDLE;

	// ---- Extension descriptor set layouts -----------------------------
	VkDescriptorSetLayout m_upmlPreVoltDescLayout  = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_upmlPostVoltDescLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_upmlPreCurrDescLayout  = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_upmlPostCurrDescLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_dispPreDescLayout      = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_dispApplyDescLayout    = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_tfsfDescLayout         = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_murPreDescLayout       = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_murPostDescLayout      = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_murApplyDescLayout     = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_rlcPreDescLayout       = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_rlcApplyDescLayout     = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_probeDescLayout        = VK_NULL_HANDLE;

	// ---- Extension pipeline layouts -----------------------------------
	VkPipelineLayout m_upmlPrePipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_upmlPostPipeLayout= VK_NULL_HANDLE;
	VkPipelineLayout m_dispPrePipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_dispApplyPipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_tfsfPipeLayout    = VK_NULL_HANDLE;
	VkPipelineLayout m_murUpdatePipeLayout = VK_NULL_HANDLE;
	VkPipelineLayout m_murApplyPipeLayout  = VK_NULL_HANDLE;
	VkPipelineLayout m_rlcPrePipeLayout  = VK_NULL_HANDLE;
	VkPipelineLayout m_rlcApplyPipeLayout= VK_NULL_HANDLE;
	VkPipelineLayout m_probePipeLayout   = VK_NULL_HANDLE;

	// ---- Extension compute pipelines ----------------------------------
	VkPipeline m_upmlPreVoltPipeline   = VK_NULL_HANDLE;
	VkPipeline m_upmlPostVoltPipeline  = VK_NULL_HANDLE;
	VkPipeline m_upmlPreCurrPipeline   = VK_NULL_HANDLE;
	VkPipeline m_upmlPostCurrPipeline  = VK_NULL_HANDLE;
	VkPipeline m_dispPreVoltPipeline   = VK_NULL_HANDLE;
	VkPipeline m_dispPreCurrPipeline   = VK_NULL_HANDLE;
	VkPipeline m_dispApplyVoltPipeline = VK_NULL_HANDLE;
	VkPipeline m_dispApplyCurrPipeline = VK_NULL_HANDLE;
	VkPipeline m_tfsfVoltPipeline      = VK_NULL_HANDLE;
	VkPipeline m_tfsfCurrPipeline      = VK_NULL_HANDLE;
	VkPipeline m_murPreVoltPipeline    = VK_NULL_HANDLE;
	VkPipeline m_murPostVoltPipeline   = VK_NULL_HANDLE;
	VkPipeline m_murApplyVoltPipeline  = VK_NULL_HANDLE;
	VkPipeline m_rlcPreVoltPipeline    = VK_NULL_HANDLE;
	VkPipeline m_rlcApplyVoltPipeline  = VK_NULL_HANDLE;
	VkPipeline m_probePipeline         = VK_NULL_HANDLE;

	// ---- GPU extension state flags ------------------------------------
	bool m_hasGPU_UPML       = false;
	bool m_hasGPU_Dispersive = false;
	bool m_hasGPU_TFSF       = false;
	bool m_hasGPU_Mur        = false;
	bool m_hasGPU_RLC        = false;
	bool m_hasGPU_Probes     = false;

	// ---- Pipelined processing state -----------------------------------
	bool m_pipelinedReading = false;   //!< When true, GetVolt/GetCurr use probe cache
	mutable bool m_recordingProbeAccess = false; //!< When true, GetVolt/GetCurr record cell indices
	mutable std::vector<uint64_t> m_recordedCells; //!< Recorded (fieldSel<<48 | linearIdx) during dry run
	std::vector<float> m_probeCacheCPU;  //!< CPU-side snapshot of gathered probes
	std::vector<uint64_t> m_probeCacheKeys; //!< Sorted keys for binary search (fieldSel<<48 | linearIdx)
	uint32_t m_probeCacheCount = 0;     //!< Number of probe points
	mutable uint64_t m_probeCacheHits = 0;
	mutable uint64_t m_probeCacheMisses = 0;
	GpuBuf m_probeIdxBuf;              //!< GPU probe index buffer
	GpuBuf m_probeSelBuf;              //!< GPU field selector (0=volt, 1=curr)
	GpuBuf m_probeOutBuf;              //!< Coherent host-visible GPU probe output
	GpuBuf m_probeReadbackBuf[CMD_RING_SIZE];      //!< Per-submit-slot host-visible probe readback ring
	GpuBuf m_probeStagingBuf;          //!< Staging buffer for non-ReBAR probe download
	float* m_probeOutMapped = nullptr;  //!< Persistently mapped probe output
	float* m_probeReadbackMapped[CMD_RING_SIZE] = {}; //!< Persistently mapped readback ring
	int m_lastProbeSubmitSlot = 0;      //!< Last submitted cmd slot carrying probe results
	VkDescriptorSet  m_probeGatherDescSet = VK_NULL_HANDLE;
	VkDescriptorPool m_probeDescPool     = VK_NULL_HANDLE;
	unsigned int m_speculativeTS = 0;   //!< Timesteps in speculative batch

	// ---- GPU energy reduction -----------------------------------------
	static constexpr uint32_t ENERGY_NUM_WG = 256; //!< workgroups for energy reduction
	GpuBuf       m_energyPartialBuf;   //!< [2 * ENERGY_NUM_WG] float partials (E, H)
	float*       m_energyMapped = nullptr; //!< ReBAR-mapped readback (or nullptr)
	VkDescriptorSetLayout m_energyDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout      m_energyPipeLayout = VK_NULL_HANDLE;
	VkPipeline            m_energyPipeline   = VK_NULL_HANDLE;
	VkDescriptorPool      m_energyDescPool   = VK_NULL_HANDLE;
	VkDescriptorSet       m_energyDescSet    = VK_NULL_HANDLE;
	//! Push constant struct for energy reduction shader.
	struct EnergyPC { uint32_t N; uint32_t numWG; };
	void SetupGPU_EnergyReduction();     //!< Create energy reduction resources
	void CleanupGPU_EnergyReduction();   //!< Destroy energy reduction resources

	// ---- Optional raw-field finite-value scan -------------------------
	GpuBuf       m_validateBadBuf;       //!< [V0,V1,V2,C0,C1,C2] first bad cell indices
	uint32_t*    m_validateBadMapped = nullptr;
	VkDescriptorSetLayout m_validateDescLayout = VK_NULL_HANDLE;
	VkPipelineLayout      m_validatePipeLayout = VK_NULL_HANDLE;
	VkPipeline            m_validatePipeline = VK_NULL_HANDLE;
	VkDescriptorPool      m_validateDescPool = VK_NULL_HANDLE;
	VkDescriptorSet       m_validateDescSet = VK_NULL_HANDLE;
	struct ValidatePC { uint32_t N, outputBase, traceIndex, numComp, Ny, Nz, checkMagnitude; };
	void SetupGPU_FieldValidation();
	void CleanupGPU_FieldValidation();
	void RecordGPUFieldValidation(VkCommandBuffer cmd, uint32_t outputBase) const;
	void ValidateGPUFields(unsigned int timestep) const; //!< Debug-only finite-value scan
	void ReportGPUFieldValidation(unsigned int timestep) const; //!< Report staged scans

	// ---- Extension setup methods --------------------------------------
	void SetupGPUExtensions();     //!< Detect and setup all GPU-native extensions
	void SetupGPU_UPML();          //!< Upload PML data and create descriptors/pipelines
	void SetupGPU_Dispersive();    //!< Upload dispersive ADE data
	void SetupGPU_TFSF();          //!< Flatten and upload TF/SF injection points
	void SetupGPU_Mur();           //!< Upload Mur ABC data
	void SetupGPU_RLC();           //!< Upload lumped RLC data
	void SetupGPU_Probes();        //!< Build probe gather descriptor
	void CreateExtensionDescriptorLayouts();   //!< Create all extension descriptor set layouts
	void CreateExtensionPipelines();           //!< Create all extension compute pipelines
	void CleanupExtensions();      //!< Destroy all extension GPU resources

	//! Record extension dispatches into a command buffer for the voltage half-step.
	void RecordVoltageExtensions(VkCommandBuffer cmd, uint32_t numTS) const;
	//! Record extension dispatches into a command buffer for the current half-step.
	void RecordCurrentExtensions(VkCommandBuffer cmd, uint32_t numTS) const;

	static void WriteDescriptorBuffers(VkDevice dev, VkDescriptorSet set,
	                                   const VkBuffer* bufs, const VkDeviceSize* sizes,
	                                   uint32_t count);

	// ---- Async field-dump download --------------------------------------
	// Full-field dumps (ProcessFields) need the *entire* volt/curr buffers,
	// not the sparse cells the probe cache covers. On a discrete GPU that
	// download is large (hundreds of MB) and, before this, was a single
	// blocking copy on the compute queue: DrainGPU() + vkCmdCopyBuffer +
	// vkWaitForFences with nothing else in flight. The pipeline below
	// replaces that with two stages on an independent hardware queue:
	//   1. live buffers -> device-local "snapshot" buffers (VRAM-VRAM, fast,
	//      bandwidth-bound only; must finish before the *next* speculative
	//      batch overwrites the live buffers in place).
	//   2. snapshot -> host-visible staging (PCIe, slow) -- runs fully
	//      concurrently with subsequent GPU compute and CPU-side PA::Process()
	//      work, since stage 1 already froze the source.
	// FinishAsyncFieldDownload() (called lazily from SyncFieldsToHost) just
	// waits on stage 2's fence and memcpy's into the CPU shadow arrays.
	bool     m_hasFieldDumps = false;   //!< Set via SetHasFieldDumps() by the caller
	bool     m_hasDumpQueue  = false;   //!< True once a 2nd hw queue was found & buffers created
	uint32_t m_dumpQueueFamily = 0;
	VkQueue  m_dumpQueue = VK_NULL_HANDLE;
	VkCommandPool m_dumpCmdPool     = VK_NULL_HANDLE; //!< On m_computeQueueFamily (stage 1)
	VkCommandPool m_dumpXferCmdPool = VK_NULL_HANDLE; //!< On m_dumpQueueFamily (stage 2)

	static constexpr int DUMP_RING = 2; //!< Double-buffered so a still-in-flight
	                                     //!< previous request never blocks a new one.
	VkCommandBuffer m_dumpSnapshotCmdBuf[DUMP_RING] = {};
	VkCommandBuffer m_dumpStagingCmdBuf[DUMP_RING]  = {};
	VkFence     m_dumpSnapshotFence[DUMP_RING] = {};
	VkFence     m_dumpStagingFence[DUMP_RING]  = {};
	VkSemaphore m_dumpSnapshotSem[DUMP_RING]   = {}; //!< stage1 signal -> stage2 wait
	bool        m_dumpSlotInFlight[DUMP_RING]  = {}; //!< stage2 submitted, fence not yet waited

	// Snapshot buffers must be readable from both queue families, so they
	// use VK_SHARING_MODE_CONCURRENT (avoids manual ownership-transfer
	// barriers for what is a rare, non-hot-path transfer).
	GpuBuf m_dumpSnapVolt[DUMP_RING];
	GpuBuf m_dumpSnapCurr[DUMP_RING];
	GpuBuf m_dumpStageVolt[DUMP_RING]; //!< Host-visible+coherent, persistently mapped
	GpuBuf m_dumpStageCurr[DUMP_RING];
	FDTD_FLOAT* m_dumpStageVoltMapped[DUMP_RING] = {};
	FDTD_FLOAT* m_dumpStageCurrMapped[DUMP_RING] = {};

	bool m_dumpBuffersReady  = false; //!< Lazily created on first BeginAsyncFieldDownload()
	bool m_dumpAsyncPending  = false;
	int  m_dumpAsyncIdx      = 0;     //!< Slot currently holding the pending request
	int  m_dumpNextIdx       = 0;     //!< Slot to use for the next request

	void EnsureDumpBuffers();
	void FinishAsyncFieldDownload();
	void CleanupDumpAsync();
};

#endif // WITH_GPU
#endif // ENGINE_VULKAN_H
