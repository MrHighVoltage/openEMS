/*
*	Copyright (C) 2026 openEMS contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifdef WITH_GPU

#include "engine_vulkan.h"
#include "operator_vulkan.h"
#include "operator.h"
#include "extensions/engine_extension.h"
#include "extensions/engine_ext_excitation.h"
#include "extensions/operator_ext_excitation.h"
#include "extensions/engine_ext_upml.h"
#include "extensions/operator_ext_upml.h"
#include "extensions/engine_ext_lorentzmaterial.h"
#include "extensions/operator_ext_lorentzmaterial.h"
#include "extensions/engine_ext_tfsf.h"
#include "extensions/operator_ext_tfsf.h"
#include "extensions/engine_ext_mur_abc.h"
#include "extensions/operator_ext_mur_abc.h"
#include "extensions/engine_ext_lumpedRLC.h"
#include "extensions/operator_ext_lumpedRLC.h"
#include "extensions/engine_ext_steadystate.h"
#include "extensions/operator_ext_steadystate.h"
#include "extensions/engine_ext_absorbing_bc.h"
#include "extensions/operator_ext_absorbing_bc.h"
#include "Common/processing.h"
#include "gpu_shader_spirv.h"
#include "tools/constants.h"

#include <iostream>
#include <cstring>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>

using std::cerr;
using std::cout;
using std::endl;

bool Engine_Vulkan::s_preferReBARFieldBuffers = true;
bool Engine_Vulkan::s_enableProfiling = false;

// ---------------------------------------------------------------------------
// Vulkan error checking
// ---------------------------------------------------------------------------
static const char* VkResultName(VkResult result)
{
	switch (result)
	{
		case VK_SUCCESS: return "VK_SUCCESS";
		case VK_NOT_READY: return "VK_NOT_READY";
		case VK_TIMEOUT: return "VK_TIMEOUT";
		case VK_EVENT_SET: return "VK_EVENT_SET";
		case VK_EVENT_RESET: return "VK_EVENT_RESET";
		case VK_INCOMPLETE: return "VK_INCOMPLETE";
		case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
		case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
		case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
		case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
		case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
		case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
		case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
		case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
		case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
		case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
		case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
		case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
		case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
		case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
		default: return "VK_ERROR_UNRECOGNIZED";
	}
}

#define VK_CHECK(call) \
	do { \
		VkResult _r = (call); \
		if (_r != VK_SUCCESS) { \
			cerr << "Vulkan error " << VkResultName(_r) << " (" << _r << ") at " << __FILE__ << ":" << __LINE__ << endl; \
			throw std::runtime_error("Vulkan call failed"); \
		} \
	} while(0)

// ===========================================================================
// Construction / Destruction
// ===========================================================================

Engine_Vulkan* Engine_Vulkan::New(const Operator* op)
{
	cout << "Create Vulkan GPU engine" << endl;
	Engine_Vulkan* e = new Engine_Vulkan(op);
	e->Init();
	return e;
}

bool Engine_Vulkan::PreflightAllocationForGrid(unsigned int Nx, unsigned int Ny, unsigned int Nz,
                                               bool strictCoeffReserve, std::string* errMsg)
{
	VkInstance instance = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkBuffer bufA = VK_NULL_HANDLE, bufB = VK_NULL_HANDLE, bufStaging = VK_NULL_HANDLE, bufCoeff = VK_NULL_HANDLE;
	VkDeviceMemory memA = VK_NULL_HANDLE, memB = VK_NULL_HANDLE, memStaging = VK_NULL_HANDLE, memCoeff = VK_NULL_HANDLE;

	auto cleanup = [&]()
	{
		if (bufCoeff) vkDestroyBuffer(device, bufCoeff, nullptr);
		if (memCoeff) vkFreeMemory(device, memCoeff, nullptr);
		if (bufStaging) vkDestroyBuffer(device, bufStaging, nullptr);
		if (memStaging) vkFreeMemory(device, memStaging, nullptr);
		if (bufB) vkDestroyBuffer(device, bufB, nullptr);
		if (memB) vkFreeMemory(device, memB, nullptr);
		if (bufA) vkDestroyBuffer(device, bufA, nullptr);
		if (memA) vkFreeMemory(device, memA, nullptr);
		if (device) vkDestroyDevice(device, nullptr);
		if (instance) vkDestroyInstance(instance, nullptr);
	};
	auto fail = [&](const std::string& msg) -> bool
	{
		cleanup();
		if (errMsg) *errMsg = msg;
		return false;
	};

	try
	{
		VkApplicationInfo app{};
		app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app.pApplicationName = "openEMS Vulkan preflight";
		app.apiVersion = VK_API_VERSION_1_1;

		VkInstanceCreateInfo ici{};
		ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		ici.pApplicationInfo = &app;
		if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS)
			return fail("Vulkan preflight: failed to create Vulkan instance");

		uint32_t devCount = 0;
		vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
		if (devCount == 0)
			return fail("Vulkan preflight: no Vulkan physical devices found");
		std::vector<VkPhysicalDevice> devs(devCount);
		vkEnumeratePhysicalDevices(instance, &devCount, devs.data());

		struct Candidate { VkPhysicalDevice dev; uint32_t qf; VkPhysicalDeviceProperties props; };
		std::vector<Candidate> cands;
		for (auto& d : devs)
		{
			uint32_t qfCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, nullptr);
			if (qfCount == 0) continue;
			std::vector<VkQueueFamilyProperties> qfProps(qfCount);
			vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, qfProps.data());
			for (uint32_t i = 0; i < qfCount; ++i)
			{
				if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
				{
					Candidate c{};
					c.dev = d;
					c.qf = i;
					vkGetPhysicalDeviceProperties(d, &c.props);
					cands.push_back(c);
					break;
				}
			}
		}
		if (cands.empty())
			return fail("Vulkan preflight: no compute-capable GPU found");

		int forcedIndex = -1;
		if (const char* envGpuIndex = std::getenv("OPENEMS_GPU_INDEX"))
		{
			char* endPtr = nullptr;
			long idx = std::strtol(envGpuIndex, &endPtr, 10);
			if (endPtr && *endPtr == '\0' && idx >= 0 && idx < (long)cands.size())
				forcedIndex = (int)idx;
		}

		uint32_t computeQf = cands[0].qf;
		phys = cands[0].dev;
		if (forcedIndex >= 0)
		{
			phys = cands[forcedIndex].dev;
			computeQf = cands[forcedIndex].qf;
		}
		else
		{
			for (const auto& c : cands)
			{
				if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
				{
					phys = c.dev;
					computeQf = c.qf;
					break;
				}
			}
		}

		float priority = 1.0f;
		VkDeviceQueueCreateInfo qci{};
		qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		qci.queueFamilyIndex = computeQf;
		qci.queueCount = 1;
		qci.pQueuePriorities = &priority;

		VkDeviceCreateInfo dci{};
		dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		dci.queueCreateInfoCount = 1;
		dci.pQueueCreateInfos = &qci;
		if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS)
			return fail("Vulkan preflight: failed to create logical device");

		auto findMemType = [&](uint32_t bits, VkMemoryPropertyFlags flags) -> uint32_t
		{
			VkPhysicalDeviceMemoryProperties mp{};
			vkGetPhysicalDeviceMemoryProperties(phys, &mp);
			for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
			{
				if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
					return i;
			}
			return UINT32_MAX;
		};

		auto allocBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags,
		                      VkBuffer& outBuf, VkDeviceMemory& outMem) -> bool
		{
			VkBufferCreateInfo bci{};
			bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bci.size = size;
			bci.usage = usage;
			bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			if (vkCreateBuffer(device, &bci, nullptr, &outBuf) != VK_SUCCESS)
				return false;

			VkMemoryRequirements req{};
			vkGetBufferMemoryRequirements(device, outBuf, &req);
			uint32_t memType = findMemType(req.memoryTypeBits, memFlags);
			if (memType == UINT32_MAX)
				return false;

			VkMemoryAllocateInfo mai{};
			mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			mai.allocationSize = req.size;
			mai.memoryTypeIndex = memType;
			if (vkAllocateMemory(device, &mai, nullptr, &outMem) != VK_SUCCESS)
				return false;

			if (vkBindBufferMemory(device, outBuf, outMem, 0) != VK_SUCCESS)
				return false;

			return true;
		};

		const uint64_t N = (uint64_t)Nx * (uint64_t)Ny * (uint64_t)Nz;
		const VkDeviceSize fieldSize = (VkDeviceSize)(3ull * N * sizeof(FDTD_FLOAT));
		const VkBufferUsageFlags fieldUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		const VkBufferUsageFlags stagingUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		const VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		const VkMemoryPropertyFlags rebar = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
		                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		bool fieldsOk = false;
		if (s_preferReBARFieldBuffers)
		{
			VkBuffer tmpA = VK_NULL_HANDLE, tmpB = VK_NULL_HANDLE;
			VkDeviceMemory tmpAM = VK_NULL_HANDLE, tmpBM = VK_NULL_HANDLE;
			if (allocBuffer(fieldSize, fieldUsage, rebar, tmpA, tmpAM) &&
			    allocBuffer(fieldSize, fieldUsage, rebar, tmpB, tmpBM))
			{
				bufA = tmpA; memA = tmpAM;
				bufB = tmpB; memB = tmpBM;
				fieldsOk = true;
			}
			else
			{
				if (tmpB) vkDestroyBuffer(device, tmpB, nullptr);
				if (tmpBM) vkFreeMemory(device, tmpBM, nullptr);
				if (tmpA) vkDestroyBuffer(device, tmpA, nullptr);
				if (tmpAM) vkFreeMemory(device, tmpAM, nullptr);
			}
		}

		if (!fieldsOk)
		{
			if (!allocBuffer(fieldSize, fieldUsage, dLoc, bufA, memA) ||
			    !allocBuffer(fieldSize, fieldUsage, dLoc, bufB, memB))
				return fail("Vulkan preflight: unable to allocate field buffers on selected GPU");
		}

		if (!allocBuffer(fieldSize, stagingUsage, host, bufStaging, memStaging))
			return fail("Vulkan preflight: unable to allocate staging buffer on selected GPU/host memory");

		if (strictCoeffReserve)
		{
			const VkDeviceSize coeffUpper = (VkDeviceSize)(N * (uint64_t)sizeof(uint32_t) +
			                                               4ull * 3ull * N * (uint64_t)sizeof(float));
			const VkBufferUsageFlags coeffUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
			                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			if (!allocBuffer(coeffUpper, coeffUsage, dLoc, bufCoeff, memCoeff))
				return fail("Vulkan preflight: strict coefficient reserve failed (GPU memory likely insufficient)");
		}

		cleanup();
		return true;
	}
	catch (...)
	{
		return fail("Vulkan preflight: unexpected error while probing GPU allocations");
	}
}

Engine_Vulkan::Engine_Vulkan(const Operator* op) : Engine(op)
{
	m_type = GPU;

	m_instance       = VK_NULL_HANDLE;
	m_physDevice     = VK_NULL_HANDLE;
	m_device         = VK_NULL_HANDLE;
	m_computeQueue   = VK_NULL_HANDLE;
	m_computeQueueFamily = 0;
	m_transferQueue  = VK_NULL_HANDLE;
	m_transferQueueFamily = 0;
	m_hasDedicatedTransferQueue = false;

	m_voltBuf = m_currBuf = VK_NULL_HANDLE;
	m_voltMem = m_currMem = VK_NULL_HANDLE;
	m_opIndexBuf = VK_NULL_HANDLE;
	m_opIndexMem = VK_NULL_HANDLE;
	m_vvCompBuf = m_viCompBuf = m_iiCompBuf = m_ivCompBuf = VK_NULL_HANDLE;
	m_vvCompMem = m_viCompMem = m_iiCompMem = m_ivCompMem = VK_NULL_HANDLE;
	m_opIndexBufSize = 0;
	m_coeffCompBufSize = 0;
	m_stagingBuf = VK_NULL_HANDLE;
	m_stagingMem = VK_NULL_HANDLE;
	m_fieldBufSize = 0;
	m_voltMapped = nullptr;
	m_currMapped = nullptr;
	m_fieldN     = 0;
	m_strideYZ   = 0;

	m_excVoltIdxBuf = m_excVoltAmpBuf = m_excVoltDelayBuf = VK_NULL_HANDLE;
	m_excVoltIdxMem = m_excVoltAmpMem = m_excVoltDelayMem = VK_NULL_HANDLE;
	m_excCurrIdxBuf = m_excCurrAmpBuf = m_excCurrDelayBuf = VK_NULL_HANDLE;
	m_excCurrIdxMem = m_excCurrAmpMem = m_excCurrDelayMem = VK_NULL_HANDLE;
	m_excSignalVoltBuf = m_excSignalCurrBuf = VK_NULL_HANDLE;
	m_excSignalVoltMem = m_excSignalCurrMem = VK_NULL_HANDLE;
	m_excVoltCount = m_excCurrCount = 0;
	m_excSignalLen = 0;
	m_excSignalPeriod = 0;

	m_fdtdDescLayout = VK_NULL_HANDLE;
	m_excDescLayout  = VK_NULL_HANDLE;
	m_descPool       = VK_NULL_HANDLE;
	m_updateVoltDescSet = m_updateCurrDescSet = VK_NULL_HANDLE;
	m_excVoltDescSet = m_excCurrDescSet = VK_NULL_HANDLE;

	m_fdtdPipeLayout = VK_NULL_HANDLE;
	m_excPipeLayout  = VK_NULL_HANDLE;
	m_updateVoltPipeline = m_updateCurrPipeline = m_excPipeline = VK_NULL_HANDLE;

	m_cmdPool = VK_NULL_HANDLE;
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		m_cmdBufs[i] = VK_NULL_HANDLE;
	m_utilCmdBuf = VK_NULL_HANDLE;
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		m_fences[i] = VK_NULL_HANDLE;
	m_utilFence = VK_NULL_HANDLE;
	m_cmdIdx = 0;
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		m_gpuInFlight[i] = false;

	m_transferCmdPool = VK_NULL_HANDLE;
	for (int i = 0; i < CMD_RING_SIZE; ++i)
	{
		m_transferCmdBufs[i] = VK_NULL_HANDLE;
		m_transferFences[i] = VK_NULL_HANDLE;
		m_probeCopySem[i] = VK_NULL_HANDLE;
		m_probeTransferInFlight[i] = false;
	}

	m_hostDirty   = false;
	m_deviceDirty = false;
	m_hasCPUExtensions  = false;
	m_hasGPUExcitation  = false;
	m_validateFields    = false;
	m_validateStages    = false;
	m_validateMagnitude = false;
	m_validateTraceIndex = std::numeric_limits<uint32_t>::max();
	m_gpuDrained  = true;
	m_pipelineCache = VK_NULL_HANDLE;
	m_maxTSPerSubmit = 64;
	m_chunkProgressIntervalSec = 4.0;
	m_chunkProgressHasLastPrint = false;
}

Engine_Vulkan::~Engine_Vulkan()
{
	DrainGPU();
	Reset();
}

// ===========================================================================
// Init / Reset
// ===========================================================================

void Engine_Vulkan::Init()
{
	bool startupTrace = (g_settings.GetVerboseLevel() > 0);
	if (const char* env = std::getenv("OPENEMS_GPU_STARTUP_TRACE"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			startupTrace = (v != 0);
	}

	auto startupBegin = std::chrono::steady_clock::now();
	unsigned int startupStep = 0;
	auto runStartupStep = [&](const char* label, const std::function<void()>& fn)
	{
		auto t0 = std::chrono::steady_clock::now();
		if (startupTrace)
			cout << "[Vulkan startup] step " << (++startupStep) << ": " << label << "..." << endl;
		fn();
		auto t1 = std::chrono::steady_clock::now();
		if (startupTrace)
		{
			double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
			cout << "[Vulkan startup] done " << label << " in " << ms << " ms" << endl;
		}
	};

	// Base class allocates CPU arrays (volt_ptr, curr_ptr) and extensions
	runStartupStep("Engine::Init", [&]() { Engine::Init(); });

	// Read once: pure-GPU submit chunk size to avoid overly long command buffers.
	if (const char* env = std::getenv("OPENEMS_GPU_MAX_TS_PER_SUBMIT"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0' && v > 0)
			m_maxTSPerSubmit = (unsigned int)v;
	}
	if (const char* env = std::getenv("OPENEMS_GPU_CHUNK_PROGRESS_SEC"))
	{
		char* endPtr = nullptr;
		double v = std::strtod(env, &endPtr);
		if (endPtr && *endPtr == '\0' && v > 0.0)
			m_chunkProgressIntervalSec = v;
	}
	if (const char* env = std::getenv("OPENEMS_GPU_VALIDATE_FIELDS"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			m_validateFields = (v != 0);
	}
	if (const char* env = std::getenv("OPENEMS_GPU_VALIDATE_STAGES"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			m_validateStages = (v != 0);
	}
	if (const char* env = std::getenv("OPENEMS_GPU_VALIDATE_MAGNITUDE"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			m_validateMagnitude = (v != 0);
	}
	m_validateFields = m_validateFields || m_validateStages || m_validateMagnitude;
	if (const char* env = std::getenv("OPENEMS_GPU_TRACE_CELL"))
	{
		unsigned int x = 0, y = 0, z = 0;
		if (std::sscanf(env, "%u,%u,%u", &x, &y, &z) == 3 &&
		    x < numLines[0] && y < numLines[1] && z < numLines[2])
			m_validateTraceIndex = x * numLines[1] * numLines[2] + y * numLines[2] + z;
	}

	if (startupTrace)
	{
		cout << "[Vulkan startup] config: max_ts_per_submit=" << m_maxTSPerSubmit
		     << ", chunk_progress_sec=" << m_chunkProgressIntervalSec
		     << ", validate_fields=" << (m_validateFields ? 1 : 0)
		     << ", validate_stages=" << (m_validateStages ? 1 : 0) << endl;
	}

	// Vulkan resources
	runStartupStep("InitVulkan", [&]() { InitVulkan(); });
	runStartupStep("CreateBuffers", [&]() { CreateBuffers(); });
	runStartupStep("CreateDescriptors", [&]() { CreateDescriptors(); });
	runStartupStep("CreatePipelines", [&]() { CreatePipelines(); });
	runStartupStep("UploadCoefficients", [&]() { UploadCoefficients(); });
	runStartupStep("SetupGPUExcitation", [&]() { SetupGPUExcitation(); });
	runStartupStep("SetupGPUExtensions", [&]() { SetupGPUExtensions(); });
	runStartupStep("SetupFusedUPML", [&]() { SetupFusedUPML(); });
	runStartupStep("SetupGPU_EnergyReduction", [&]() { SetupGPU_EnergyReduction(); });
	if (m_validateFields)
		runStartupStep("SetupGPU_FieldValidation", [&]() { SetupGPU_FieldValidation(); });

	// Upload initial (zero) field data to GPU
	runStartupStep("UploadInitialFields", [&]() {
		if (m_voltMapped)
		{
			// ReBAR: direct memcpy through the persistently mapped pointer
			memcpy(m_voltMapped, volt_ptr->data(), m_fieldBufSize);
			memcpy(m_currMapped, curr_ptr->data(), m_fieldBufSize);
		}
		else
		{
			UploadToDeviceBuffer(m_voltBuf, volt_ptr->data(), m_fieldBufSize);
			UploadToDeviceBuffer(m_currBuf, curr_ptr->data(), m_fieldBufSize);
		}
	});

	m_hostDirty  = false;
	m_deviceDirty = false;

	// Print GPU info
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(m_physDevice, &props);
	cout << "Engine_Vulkan: using " << props.deviceName << endl;
	cout << "  Field buffer size: " << (m_fieldBufSize / (1024*1024)) << " MB per field" << endl;
	{
		const Operator_Vulkan* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
		cout << "  Compressed operator: " << opVk->GetNumCompressed() << " unique coeff sets"
		     << " (index: " << (m_opIndexBufSize / 1024) << " KB, tables: "
		     << (4 * m_coeffCompBufSize / 1024) << " KB)" << endl;
	}
	if (m_voltMapped)
		cout << "  ReBAR active — zero-copy CPU field reads from VRAM" << endl;
	else
		cout << "  ReBAR not available — using staging buffer for field sync" << endl;
	if (!s_preferReBARFieldBuffers)
		cout << "  Field-buffer ReBAR disabled by configuration (device-local volt/curr)" << endl;
	if (m_hasCPUExtensions)
		cout << "  WARNING: CPU extensions present — hybrid mode (slower)" << endl;
	else
		cout << "  Pure GPU mode — double-buffered fire-and-forget submission" << endl;
	if (m_profileEnabled)
		cout << "  GPU profiling enabled (timestamp queries + fence wait accounting)" << endl;
	if (m_hasGPU_UPML)
		cout << "  GPU UPML active (" << m_gpuUPML.size() << " region(s))" << endl;
	if (m_hasFusedUPML)
		cout << "  Fused Yee+UPML kernels active" << endl;
	if (m_hasGPU_Dispersive)
		cout << "  GPU dispersive materials active (" << m_gpuDisp.size() << " order(s))" << endl;
	if (m_hasGPU_TFSF)
		cout << "  GPU TF/SF active (V:" << m_gpuTFSF.voltCount << " I:" << m_gpuTFSF.currCount << " points)" << endl;
	if (m_hasGPU_Mur)
		cout << "  GPU Mur ABC active (" << m_gpuMur.size() << " face(s))" << endl;
	if (m_hasGPU_RLC)
		cout << "  GPU lumped RLC active (" << m_gpuRLC.count << " elements)" << endl;
	if (m_steadyPipeline)
		cout << "  GPU steady-state sampling active (" << m_steadyProbeCount << " probes)" << endl;
	if (!m_gpuLocalABC.empty())
		cout << "  GPU local absorbing sheets active (" << m_gpuLocalABC.size() << " sheet(s))" << endl;
	cout << "  GPU energy reduction active (" << ENERGY_NUM_WG << " workgroups)" << endl;

	if (startupTrace)
	{
		auto startupEnd = std::chrono::steady_clock::now();
		double totalMs = std::chrono::duration<double, std::milli>(startupEnd - startupBegin).count();
		cout << "[Vulkan startup] total init time: " << totalMs << " ms" << endl;
	}
}

void Engine_Vulkan::Reset()
{
	if (m_hasGPU_Probes && (m_probeCacheHits > 0 || m_probeCacheMisses > 0))
	{
		uint64_t total = m_probeCacheHits + m_probeCacheMisses;
		double missPct = total ? (100.0 * (double)m_probeCacheMisses / (double)total) : 0.0;
		cout << "Engine_Vulkan probe cache stats: hits=" << m_probeCacheHits
		     << ", misses=" << m_probeCacheMisses
		     << " (" << missPct << "% miss)" << endl;
		if (m_probeCacheMisses > 0)
			cout << "  Note: probe cache misses trigger fallback field reads (can be slow with ReBAR on dGPU)." << endl;
	}
	CleanupVulkan();
	Engine::Reset();
}

// ===========================================================================
// Vulkan Initialisation
// ===========================================================================

void Engine_Vulkan::InitVulkan()
{
	// --- Instance ---
	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "openEMS";
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo instInfo{};
	instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instInfo.pApplicationInfo = &appInfo;
	VK_CHECK(vkCreateInstance(&instInfo, nullptr, &m_instance));

	// --- Physical device ---
	uint32_t devCount = 0;
	vkEnumeratePhysicalDevices(m_instance, &devCount, nullptr);
	if (devCount == 0)
		throw std::runtime_error("Engine_Vulkan: no Vulkan-capable GPU found");

	std::vector<VkPhysicalDevice> devs(devCount);
	vkEnumeratePhysicalDevices(m_instance, &devCount, devs.data());

	struct GpuCandidate {
		VkPhysicalDevice dev;
		uint32_t computeQueueFamily;
		VkPhysicalDeviceProperties props;
	};

	std::vector<GpuCandidate> candidates;
	candidates.reserve(devs.size());

	for (auto& dev : devs)
	{
		uint32_t qfCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, nullptr);
		std::vector<VkQueueFamilyProperties> qfProps(qfCount);
		vkGetPhysicalDeviceQueueFamilyProperties(dev, &qfCount, qfProps.data());

		for (uint32_t i = 0; i < qfCount; i++)
		{
			if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				GpuCandidate c{};
				c.dev = dev;
				c.computeQueueFamily = i;
				vkGetPhysicalDeviceProperties(dev, &c.props);
				candidates.push_back(c);
				break;
			}
		}
	}

	if (candidates.empty())
		throw std::runtime_error("Engine_Vulkan: no compute-capable GPU found");

	// Optional override: OPENEMS_GPU_INDEX=<n> selects the n-th compute-capable
	// Vulkan device from vkEnumeratePhysicalDevices() order.
	int forcedIndex = -1;
	if (const char* envGpuIndex = std::getenv("OPENEMS_GPU_INDEX"))
	{
		char* endPtr = nullptr;
		long idx = std::strtol(envGpuIndex, &endPtr, 10);
		if (endPtr && *endPtr == '\0' && idx >= 0 && idx < (long)candidates.size())
			forcedIndex = (int)idx;
		else
			cerr << "Engine_Vulkan: invalid OPENEMS_GPU_INDEX='" << envGpuIndex
			     << "' (valid range: 0.." << (int)candidates.size()-1
			     << "), using automatic GPU selection." << endl;
	}

	if (forcedIndex >= 0)
	{
		m_physDevice = candidates[forcedIndex].dev;
		m_computeQueueFamily = candidates[forcedIndex].computeQueueFamily;
	}
	else
	{
		// Default policy: prefer discrete GPU, fall back to first compute-capable.
		m_physDevice = candidates[0].dev;
		m_computeQueueFamily = candidates[0].computeQueueFamily;
		for (const auto& c : candidates)
		{
			if (c.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			{
				m_physDevice = c.dev;
				m_computeQueueFamily = c.computeQueueFamily;
				break;
			}
		}
	}

	// Print Vulkan device inventory once for easier multi-GPU selection.
	cout << "Engine_Vulkan: compute-capable Vulkan devices:" << endl;
	for (size_t i = 0; i < candidates.size(); ++i)
	{
		cout << "  [" << i << "] " << candidates[i].props.deviceName;
		if (candidates[i].props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			cout << " (discrete)";
		else if (candidates[i].props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
			cout << " (integrated)";
		cout << endl;
	}
	if (forcedIndex >= 0)
		cout << "Engine_Vulkan: forcing GPU index " << forcedIndex << " via OPENEMS_GPU_INDEX" << endl;
	if (m_physDevice == VK_NULL_HANDLE)
		throw std::runtime_error("Engine_Vulkan: no compute-capable GPU found");

	// Probe and energy outputs are tiny coherent host-visible buffers. Keeping
	// their work on the compute queue avoids cross-family ownership transfers
	// and semaphore submissions, which cost more than these readbacks.
	m_transferQueueFamily = m_computeQueueFamily;
	m_hasDedicatedTransferQueue = false;

	bool dumpAsyncDisabled = false;
	if (const char* env = std::getenv("OPENEMS_GPU_DISABLE_DUMP_ASYNC"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		dumpAsyncDisabled = (endPtr && *endPtr == '\0' && v != 0);
	}

	// A second, genuinely independent queue for the large field-dump download
	// pipeline (see SetupDumpTransferQueue()). Unlike the probe/transfer queue
	// above, dump downloads move hundreds of MB, so a real concurrent hardware
	// queue is worth the extra device-queue setup. Prefer a family that has
	// COMPUTE+TRANSFER but not GRAPHICS (an async-compute/DMA-capable family
	// on most discrete GPUs) and more than one queue, distinct from whichever
	// family m_computeQueueFamily landed on above.
	uint32_t qfCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(m_physDevice, &qfCount, nullptr);
	std::vector<VkQueueFamilyProperties> qfProps(qfCount);
	vkGetPhysicalDeviceQueueFamilyProperties(m_physDevice, &qfCount, qfProps.data());

	bool haveDumpFamily = false;
	uint32_t bestScore = 0;
	for (uint32_t i = 0; i < qfCount && !dumpAsyncDisabled; i++)
	{
		if (i == m_computeQueueFamily) continue;
		const auto& p = qfProps[i];
		if (!(p.queueFlags & VK_QUEUE_TRANSFER_BIT) && !(p.queueFlags & VK_QUEUE_COMPUTE_BIT))
			continue;
		bool noGraphics = !(p.queueFlags & VK_QUEUE_GRAPHICS_BIT);
		uint32_t score = (noGraphics ? 2u : 0u) + ((p.queueFlags & VK_QUEUE_COMPUTE_BIT) ? 1u : 0u);
		if (!haveDumpFamily || score > bestScore)
		{
			m_dumpQueueFamily = i;
			haveDumpFamily = true;
			bestScore = score;
		}
	}
	// m_hasDumpQueue is only latched true once the device/queue/pool below
	// are actually created; EnsureDumpBuffers() checks it again before
	// touching any of this, so a failure anywhere just disables the feature.
	m_hasDumpQueue = haveDumpFamily;

	// --- Logical device ---
	float priority = 1.0f;
	std::vector<VkDeviceQueueCreateInfo> queueInfos;
	VkDeviceQueueCreateInfo computeQueueInfo{};
	computeQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	computeQueueInfo.queueFamilyIndex = m_computeQueueFamily;
	computeQueueInfo.queueCount = 1;
	computeQueueInfo.pQueuePriorities = &priority;
	queueInfos.push_back(computeQueueInfo);
	if (m_hasDedicatedTransferQueue)
	{
		VkDeviceQueueCreateInfo transferQueueInfo{};
		transferQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		transferQueueInfo.queueFamilyIndex = m_transferQueueFamily;
		transferQueueInfo.queueCount = 1;
		transferQueueInfo.pQueuePriorities = &priority;
		queueInfos.push_back(transferQueueInfo);
	}
	if (m_hasDumpQueue)
	{
		VkDeviceQueueCreateInfo dumpQueueInfo{};
		dumpQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		dumpQueueInfo.queueFamilyIndex = m_dumpQueueFamily;
		dumpQueueInfo.queueCount = 1;
		dumpQueueInfo.pQueuePriorities = &priority;
		queueInfos.push_back(dumpQueueInfo);
	}

	VkDeviceCreateInfo devInfo{};
	devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	devInfo.queueCreateInfoCount = (uint32_t)queueInfos.size();
	devInfo.pQueueCreateInfos = queueInfos.data();
	VK_CHECK(vkCreateDevice(m_physDevice, &devInfo, nullptr, &m_device));

	vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_computeQueue);
	vkGetDeviceQueue(m_device, m_transferQueueFamily, 0, &m_transferQueue);
	if (m_hasDumpQueue)
	{
		vkGetDeviceQueue(m_device, m_dumpQueueFamily, 0, &m_dumpQueue);

		VkCommandPoolCreateInfo dumpPoolInfo{};
		dumpPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		dumpPoolInfo.queueFamilyIndex = m_computeQueueFamily;
		dumpPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VkCommandPoolCreateInfo dumpXferPoolInfo = dumpPoolInfo;
		dumpXferPoolInfo.queueFamilyIndex = m_dumpQueueFamily;
		if (vkCreateCommandPool(m_device, &dumpPoolInfo, nullptr, &m_dumpCmdPool) != VK_SUCCESS ||
		    vkCreateCommandPool(m_device, &dumpXferPoolInfo, nullptr, &m_dumpXferCmdPool) != VK_SUCCESS)
		{
			m_hasDumpQueue = false;
		}
	}

	// --- Command pool + buffer ---
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = m_computeQueueFamily;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_cmdPool));
	if (m_hasDedicatedTransferQueue)
	{
		VkCommandPoolCreateInfo transferPoolInfo{};
		transferPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		transferPoolInfo.queueFamilyIndex = m_transferQueueFamily;
		transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK(vkCreateCommandPool(m_device, &transferPoolInfo, nullptr, &m_transferCmdPool));
	}

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = m_cmdPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &m_cmdBufs[i]));
	VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &m_utilCmdBuf));
	if (m_hasDedicatedTransferQueue)
	{
		VkCommandBufferAllocateInfo transferAllocInfo{};
		transferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		transferAllocInfo.commandPool = m_transferCmdPool;
		transferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		transferAllocInfo.commandBufferCount = 1;
		for (int i = 0; i < CMD_RING_SIZE; ++i)
			VK_CHECK(vkAllocateCommandBuffers(m_device, &transferAllocInfo, &m_transferCmdBufs[i]));
	}

	// --- Fences ---
	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;   // start signaled so first wait is a no-op
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		VK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &m_fences[i]));
	if (m_hasDedicatedTransferQueue)
	{
		for (int i = 0; i < CMD_RING_SIZE; ++i)
			VK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &m_transferFences[i]));
	}
	fenceInfo.flags = 0;   // utility fence starts unsignaled
	VK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &m_utilFence));
	if (m_hasDedicatedTransferQueue)
	{
		VkSemaphoreCreateInfo semInfo{};
		semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		for (int i = 0; i < CMD_RING_SIZE; ++i)
			VK_CHECK(vkCreateSemaphore(m_device, &semInfo, nullptr, &m_probeCopySem[i]));
	}

	// --- Pipeline cache (speeds up subsequent runs) ---
	VkPipelineCacheCreateInfo cacheInfo{};
	cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	VK_CHECK(vkCreatePipelineCache(m_device, &cacheInfo, nullptr, &m_pipelineCache));

	SetupProfiling();
}

void Engine_Vulkan::SetupProfiling()
{
	m_profileEnabled = false;
	if (!s_enableProfiling)
		return;

	m_profilePhaseEnabled = false;
	if (const char* envPhase = std::getenv("OPENEMS_GPU_PROFILE_PHASES"))
	{
		int enabled = std::atoi(envPhase);
		m_profilePhaseEnabled = (enabled != 0);
	}
	m_profileQueriesPerSlot = m_profilePhaseEnabled ? 4u : 2u;

	uint32_t qfCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(m_physDevice, &qfCount, nullptr);
	if (qfCount == 0 || m_computeQueueFamily >= qfCount)
		return;

	std::vector<VkQueueFamilyProperties> qfProps(qfCount);
	vkGetPhysicalDeviceQueueFamilyProperties(m_physDevice, &qfCount, qfProps.data());
	if (qfProps[m_computeQueueFamily].timestampValidBits == 0)
	{
		cerr << "Engine_Vulkan: GPU profiling requested, but queue timestamps are unsupported." << endl;
		return;
	}

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(m_physDevice, &props);
	m_timestampPeriodNs = props.limits.timestampPeriod;

	VkQueryPoolCreateInfo qpi{};
	qpi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
	qpi.queryCount = m_profileQueriesPerSlot * CMD_RING_SIZE;
	VK_CHECK(vkCreateQueryPool(m_device, &qpi, nullptr, &m_tsQueryPool));

	m_profileEnabled = true;
}

void Engine_Vulkan::CollectProfileForSlot(int slot) const
{
	if (!m_profileEnabled || !m_tsQueryPool || slot < 0 || slot >= CMD_RING_SIZE || !m_profilePending[slot])
		return;

	uint64_t ts[4] = {0, 0, 0, 0};
	VkResult r = vkGetQueryPoolResults(
		m_device,
		m_tsQueryPool,
		(uint32_t)(slot * m_profileQueriesPerSlot),
		m_profileQueriesPerSlot,
		sizeof(ts),
		ts,
		sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

	if (r == VK_SUCCESS && ts[m_profileQueriesPerSlot - 1] >= ts[0])
	{
		double gpuNs = (double)(ts[m_profileQueriesPerSlot - 1] - ts[0]) * (double)m_timestampPeriodNs;
		m_profileGpuMs += gpuNs * 1e-6;
		m_profileBatchCount++;
		m_profileTSCount += m_profileSubmittedTS[slot];

		if (m_profilePhaseEnabled && m_profileQueriesPerSlot == 4 &&
		    ts[1] >= ts[0] && ts[2] >= ts[1] && ts[3] >= ts[2])
		{
			double computeNs = (double)(ts[1] - ts[0]) * (double)m_timestampPeriodNs;
			double probeNs   = (double)(ts[2] - ts[1]) * (double)m_timestampPeriodNs;
			double tailNs    = (double)(ts[3] - ts[2]) * (double)m_timestampPeriodNs;
			m_profileComputeMs += computeNs * 1e-6;
			m_profileProbeMs += probeNs * 1e-6;
			m_profileTailMs += tailNs * 1e-6;
		}
	}

	const_cast<Engine_Vulkan*>(this)->m_profilePending[slot] = false;
	const_cast<Engine_Vulkan*>(this)->m_profileSubmittedTS[slot] = 0;
}

void Engine_Vulkan::PrintProfileSummary() const
{
	if (!m_profileEnabled || m_profileBatchCount == 0)
		return;

	double avgBatchMs = m_profileGpuMs / (double)m_profileBatchCount;
	double avgTsUs = (m_profileTSCount > 0)
	               ? (m_profileGpuMs * 1000.0) / (double)m_profileTSCount
	               : 0.0;
	double waitPerBatchMs = m_profileWaitMs / (double)m_profileBatchCount;

	cout << "Engine_Vulkan profiling summary:" << endl;
	cout << "  Batches: " << m_profileBatchCount
	     << ", timesteps: " << m_profileTSCount << endl;
	cout << "  GPU time: " << m_profileGpuMs << " ms total"
	     << ", avg " << avgBatchMs << " ms/batch"
	     << ", " << avgTsUs << " us/TS" << endl;
	cout << "  Fence wait: " << m_profileWaitMs << " ms total"
	     << ", avg " << waitPerBatchMs << " ms/batch" << endl;
	if (m_profilePhaseEnabled && m_profileGpuMs > 0.0)
	{
		double compPct = 100.0 * (m_profileComputeMs / m_profileGpuMs);
		double probePct = 100.0 * (m_profileProbeMs / m_profileGpuMs);
		double tailPct = 100.0 * (m_profileTailMs / m_profileGpuMs);
		cout << "  Phase split (GPU): compute=" << m_profileComputeMs << " ms (" << compPct << "%), "
		     << "probe=" << m_profileProbeMs << " ms (" << probePct << "%), "
		     << "tail=" << m_profileTailMs << " ms (" << tailPct << "%)" << endl;
	}
}

// ===========================================================================
// Buffer helpers
// ===========================================================================

uint32_t Engine_Vulkan::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
{
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(m_physDevice, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) &&
		    (memProps.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	throw std::runtime_error("Engine_Vulkan: failed to find suitable memory type");
}

uint32_t Engine_Vulkan::FindMemoryTypeSoft(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
{
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(m_physDevice, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) &&
		    (memProps.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	return UINT32_MAX;
}

void Engine_Vulkan::CreateBufferWithMemory(VkDeviceSize size, VkBufferUsageFlags usage,
                                           VkMemoryPropertyFlags memProps,
                                           VkBuffer& buf, VkDeviceMemory& mem)
{
	if (size == 0) { buf = VK_NULL_HANDLE; mem = VK_NULL_HANDLE; return; }
	if (!TryCreateBufferWithMemory(size, usage, memProps, buf, mem))
		throw std::runtime_error("Engine_Vulkan: buffer memory allocation failed");
}

bool Engine_Vulkan::TryCreateBufferWithMemory(VkDeviceSize size, VkBufferUsageFlags usage,
                                              VkMemoryPropertyFlags memProps,
                                              VkBuffer& buf, VkDeviceMemory& mem)
{
	buf = VK_NULL_HANDLE;
	mem = VK_NULL_HANDLE;
	if (size == 0)
		return true;

	VkBufferCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	ci.size  = size;
	ci.usage = usage;
	ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	if (vkCreateBuffer(m_device, &ci, nullptr, &buf) != VK_SUCCESS)
		return false;

	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(m_device, buf, &memReq);

	uint32_t memoryType = FindMemoryTypeSoft(memReq.memoryTypeBits, memProps);
	if (memoryType == UINT32_MAX)
	{
		vkDestroyBuffer(m_device, buf, nullptr);
		buf = VK_NULL_HANDLE;
		return false;
	}

	VkMemoryAllocateInfo ai{};
	ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize  = memReq.size;
	ai.memoryTypeIndex = memoryType;
	if (vkAllocateMemory(m_device, &ai, nullptr, &mem) != VK_SUCCESS)
	{
		vkDestroyBuffer(m_device, buf, nullptr);
		buf = VK_NULL_HANDLE;
		return false;
	}
	if (vkBindBufferMemory(m_device, buf, mem, 0) != VK_SUCCESS)
	{
		vkFreeMemory(m_device, mem, nullptr);
		vkDestroyBuffer(m_device, buf, nullptr);
		mem = VK_NULL_HANDLE;
		buf = VK_NULL_HANDLE;
		return false;
	}
	return true;
}

void Engine_Vulkan::CreateBuffers()
{
	size_t N = (size_t)numLines[0] * numLines[1] * numLines[2];
	m_fieldBufSize = 3 * N * sizeof(FDTD_FLOAT);
	m_fieldN   = (uint32_t)N;
	m_strideYZ = numLines[1] * numLines[2];

	VkBufferUsageFlags devRW  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
	                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags devRO  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	// --- Try ReBAR (resizable BAR): DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT ---
	// This allows the CPU to read field data directly from VRAM without any
	// staging-buffer copy, eliminating the 352 MB download bottleneck.
	VkMemoryPropertyFlags rebar = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
	                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	// Probe whether a suitable memory type exists for the volt buffer
	if (s_preferReBARFieldBuffers)
	{
		VkBufferCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		ci.size  = m_fieldBufSize;
		ci.usage = devRW;
		ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VkBuffer probeBuf;
		vkCreateBuffer(m_device, &ci, nullptr, &probeBuf);
		VkMemoryRequirements memReq;
		vkGetBufferMemoryRequirements(m_device, probeBuf, &memReq);
		uint32_t rebarIdx = FindMemoryTypeSoft(memReq.memoryTypeBits, rebar);
		vkDestroyBuffer(m_device, probeBuf, nullptr);

			if (rebarIdx != UINT32_MAX)
			{
				// A compatible memory type does not guarantee that the aperture is
				// large enough for both fields. Keep every step recoverable.
				if (TryCreateBufferWithMemory(m_fieldBufSize, devRW, rebar, m_voltBuf, m_voltMem) &&
				    vkMapMemory(m_device, m_voltMem, 0, m_fieldBufSize, 0, (void**)&m_voltMapped) == VK_SUCCESS &&
				    TryCreateBufferWithMemory(m_fieldBufSize, devRW, rebar, m_currBuf, m_currMem) &&
				    vkMapMemory(m_device, m_currMem, 0, m_fieldBufSize, 0, (void**)&m_currMapped) == VK_SUCCESS)
				{
					// Both mappings are ready.
				}
				else
				{
					if (m_currMapped) vkUnmapMemory(m_device, m_currMem);
					if (m_voltMapped) vkUnmapMemory(m_device, m_voltMem);
					m_currMapped = nullptr;
					m_voltMapped = nullptr;
					if (m_currBuf) vkDestroyBuffer(m_device, m_currBuf, nullptr);
					if (m_currMem) vkFreeMemory(m_device, m_currMem, nullptr);
					if (m_voltBuf) vkDestroyBuffer(m_device, m_voltBuf, nullptr);
					if (m_voltMem) vkFreeMemory(m_device, m_voltMem, nullptr);
					m_voltBuf = m_currBuf = VK_NULL_HANDLE;
					m_voltMem = m_currMem = VK_NULL_HANDLE;
				}
		}
	}

	// Fallback: plain device-local (requires staging for CPU access)
	if (m_voltBuf == VK_NULL_HANDLE)
		CreateBufferWithMemory(m_fieldBufSize, devRW, dLoc, m_voltBuf, m_voltMem);
	if (m_currBuf == VK_NULL_HANDLE)
		CreateBufferWithMemory(m_fieldBufSize, devRW, dLoc, m_currBuf, m_currMem);

	// Compressed coefficient buffers (read-only, always device-local)
	const Operator_Vulkan* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
	unsigned int numComp = opVk->GetNumCompressed();
	m_opIndexBufSize  = N * sizeof(uint32_t);
	m_coeffCompBufSize = 3 * numComp * sizeof(float);
	CreateBufferWithMemory(m_opIndexBufSize,  devRO, dLoc, m_opIndexBuf,  m_opIndexMem);
	CreateBufferWithMemory(m_coeffCompBufSize, devRO, dLoc, m_vvCompBuf, m_vvCompMem);
	CreateBufferWithMemory(m_coeffCompBufSize, devRO, dLoc, m_viCompBuf, m_viCompMem);
	CreateBufferWithMemory(m_coeffCompBufSize, devRO, dLoc, m_iiCompBuf, m_iiCompMem);
	CreateBufferWithMemory(m_coeffCompBufSize, devRO, dLoc, m_ivCompBuf, m_ivCompMem);

	// Staging buffer (still needed for coefficient uploads, excitation data,
	// and fallback field sync when ReBAR is not available)
	VkBufferUsageFlags stgUse = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
	                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	// Prefer a HOST_CACHED type: this buffer is read back by the CPU in
	// DownloadFromDeviceBuffer() (field sync without ReBAR, energy/probe
	// fallback), and a plain HOST_VISIBLE|HOST_COHERENT type can land on
	// write-combined memory -- fast to write, ~10x slower than a cached
	// read for the CPU to read back. Falls back to the plain flags on
	// drivers without a cached host-visible type.
	if (!TryCreateBufferWithMemory(m_fieldBufSize, stgUse, host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
	                                m_stagingBuf, m_stagingMem))
		CreateBufferWithMemory(m_fieldBufSize, stgUse, host, m_stagingBuf, m_stagingMem);
}

// ===========================================================================
// Descriptor setup
// ===========================================================================

void Engine_Vulkan::WriteDescriptorBuffers(VkDevice dev, VkDescriptorSet set,
                                           const VkBuffer* bufs, const VkDeviceSize* sizes,
                                           uint32_t count)
{
	std::vector<VkDescriptorBufferInfo> infos(count);
	std::vector<VkWriteDescriptorSet>   writes(count);
	for (uint32_t i = 0; i < count; i++)
	{
		infos[i]  = {bufs[i], 0, sizes[i]};
		writes[i] = {};
		writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet          = set;
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[i].pBufferInfo     = &infos[i];
	}
	vkUpdateDescriptorSets(dev, count, writes.data(), 0, nullptr);
}

void Engine_Vulkan::CreateDescriptors()
{
	// --- Layouts ---
	auto makeLayout = [&](uint32_t nBindings) -> VkDescriptorSetLayout
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings(nBindings);
		for (uint32_t i = 0; i < nBindings; i++)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = nBindings;
		ci.pBindings    = bindings.data();
		VkDescriptorSetLayout layout;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &layout));
		return layout;
	};
	m_fdtdDescLayout = makeLayout(5);
	m_excDescLayout  = makeLayout(5);

	// --- Pool ---
	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5*2 + 5*2};
	VkDescriptorPoolCreateInfo pi{};
	pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pi.maxSets       = 4;
	pi.poolSizeCount = 1;
	pi.pPoolSizes    = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &pi, nullptr, &m_descPool));

	// --- Allocate sets ---
	VkDescriptorSetLayout layouts[4] = {m_fdtdDescLayout, m_fdtdDescLayout,
	                                     m_excDescLayout,  m_excDescLayout};
	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_descPool;
	ai.descriptorSetCount = 4;
	ai.pSetLayouts        = layouts;
	VkDescriptorSet sets[4];
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, sets));
	m_updateVoltDescSet = sets[0];
	m_updateCurrDescSet = sets[1];
	m_excVoltDescSet    = sets[2];
	m_excCurrDescSet    = sets[3];

	// --- Write FDTD update descriptors ---
	{
		VkBuffer      bufs[5] = {m_voltBuf, m_currBuf, m_opIndexBuf, m_vvCompBuf, m_viCompBuf};
		VkDeviceSize sizes[5] = {m_fieldBufSize, m_fieldBufSize, m_opIndexBufSize,
		                          m_coeffCompBufSize, m_coeffCompBufSize};
		WriteDescriptorBuffers(m_device, m_updateVoltDescSet, bufs, sizes, 5);
	}
	{
		VkBuffer      bufs[5] = {m_currBuf, m_voltBuf, m_opIndexBuf, m_iiCompBuf, m_ivCompBuf};
		VkDeviceSize sizes[5] = {m_fieldBufSize, m_fieldBufSize, m_opIndexBufSize,
		                          m_coeffCompBufSize, m_coeffCompBufSize};
		WriteDescriptorBuffers(m_device, m_updateCurrDescSet, bufs, sizes, 5);
	}
	// Excitation descriptors are written in SetupGPUExcitation()
}

// ===========================================================================
// Pipeline setup
// ===========================================================================

VkShaderModule Engine_Vulkan::CreateShaderModule(const unsigned char* code, size_t codeSize)
{
	// Copy to aligned uint32_t buffer
	std::vector<uint32_t> aligned((codeSize + 3) / 4);
	memcpy(aligned.data(), code, codeSize);

	VkShaderModuleCreateInfo ci{};
	ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	ci.codeSize = codeSize;
	ci.pCode    = aligned.data();
	VkShaderModule mod;
	VK_CHECK(vkCreateShaderModule(m_device, &ci, nullptr, &mod));
	return mod;
}

void Engine_Vulkan::CreatePipelines()
{
	// --- Pipeline layouts ---
	// FDTD: push constant = GridPC (12 bytes)
	VkPushConstantRange fdtdPC = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GridPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount         = 1;
	pli.pSetLayouts            = &m_fdtdDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges    = &fdtdPC;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_fdtdPipeLayout));

	// Excitation: push constant = ExcPC (16 bytes)
	VkPushConstantRange excPC = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ExcPC)};
	pli.pSetLayouts         = &m_excDescLayout;
	pli.pPushConstantRanges = &excPC;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_excPipeLayout));

	// --- Shader modules ---
	VkShaderModule voltMod = CreateShaderModule(gpu_spirv::update_voltages_data,
	                                            gpu_spirv::update_voltages_size);
	VkShaderModule currMod = CreateShaderModule(gpu_spirv::update_currents_data,
	                                            gpu_spirv::update_currents_size);
	VkShaderModule excMod  = CreateShaderModule(gpu_spirv::apply_excitation_data,
	                                            gpu_spirv::apply_excitation_size);

	auto makePipeline = [&](VkShaderModule mod, VkPipelineLayout layout) -> VkPipeline
	{
		VkPipelineShaderStageCreateInfo stage{};
		stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = mod;
		stage.pName  = "main";

		VkComputePipelineCreateInfo ci{};
		ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ci.stage  = stage;
		ci.layout = layout;
		VkPipeline pipe;
		VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &pipe));
		return pipe;
	};

	m_updateVoltPipeline = makePipeline(voltMod, m_fdtdPipeLayout);
	m_updateCurrPipeline = makePipeline(currMod, m_fdtdPipeLayout);
	m_excPipeline        = makePipeline(excMod,  m_excPipeLayout);

	vkDestroyShaderModule(m_device, voltMod, nullptr);
	vkDestroyShaderModule(m_device, currMod, nullptr);
	vkDestroyShaderModule(m_device, excMod,  nullptr);
}

// ===========================================================================
// Data upload helpers
// ===========================================================================

void Engine_Vulkan::RunSingleCommand(std::function<void(VkCommandBuffer)> func) const
{
	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkResetCommandBuffer(m_utilCmdBuf, 0);
	VK_CHECK(vkBeginCommandBuffer(m_utilCmdBuf, &bi));

	func(m_utilCmdBuf);

	VK_CHECK(vkEndCommandBuffer(m_utilCmdBuf));

	VkSubmitInfo si{};
	si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &m_utilCmdBuf;
	vkResetFences(m_device, 1, &m_utilFence);
	VK_CHECK(vkQueueSubmit(m_computeQueue, 1, &si, m_utilFence));
	VK_CHECK(vkWaitForFences(m_device, 1, &m_utilFence, VK_TRUE, UINT64_MAX));
}

void Engine_Vulkan::UploadToDeviceBuffer(VkBuffer dst, const void* data, VkDeviceSize size)
{
	void* mapped;
	VK_CHECK(vkMapMemory(m_device, m_stagingMem, 0, size, 0, &mapped));
	memcpy(mapped, data, size);
	vkUnmapMemory(m_device, m_stagingMem);

	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		VkBufferCopy region = {0, 0, size};
		vkCmdCopyBuffer(cmd, m_stagingBuf, dst, 1, &region);
	});
}

void Engine_Vulkan::DownloadFromDeviceBuffer(VkBuffer src, void* data, VkDeviceSize size) const
{
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		VkBufferCopy region = {0, 0, size};
		vkCmdCopyBuffer(cmd, src, m_stagingBuf, 1, &region);
	});

	void* mapped;
	VK_CHECK(vkMapMemory(m_device, m_stagingMem, 0, size, 0, &mapped));
	memcpy(data, mapped, size);
	vkUnmapMemory(m_device, m_stagingMem);
}

void Engine_Vulkan::UploadCoefficients()
{
	const Operator_Vulkan* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
	UploadToDeviceBuffer(m_opIndexBuf, opVk->GetOpIndex(), m_opIndexBufSize);
	UploadToDeviceBuffer(m_vvCompBuf,  opVk->GetVVComp(),  m_coeffCompBufSize);
	UploadToDeviceBuffer(m_viCompBuf,  opVk->GetVIComp(),  m_coeffCompBufSize);
	UploadToDeviceBuffer(m_iiCompBuf,  opVk->GetIIComp(),  m_coeffCompBufSize);
	UploadToDeviceBuffer(m_ivCompBuf,  opVk->GetIVComp(),  m_coeffCompBufSize);
}

// ===========================================================================
// GpuBuf helpers
// ===========================================================================

void Engine_Vulkan::CreateGpuBuf(GpuBuf& gb, VkDeviceSize size,
                                 VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps)
{
	CreateBufferWithMemory(size, usage, memProps, gb.buffer, gb.memory);
}

void Engine_Vulkan::DestroyGpuBuf(GpuBuf& gb)
{
	if (gb.buffer) { vkDestroyBuffer(m_device, gb.buffer, nullptr); gb.buffer = VK_NULL_HANDLE; }
	if (gb.memory) { vkFreeMemory(m_device, gb.memory, nullptr);    gb.memory = VK_NULL_HANDLE; }
}

void Engine_Vulkan::UploadGpuBuf(GpuBuf& gb, const void* data, VkDeviceSize size)
{
	UploadToDeviceBuffer(gb.buffer, data, size);
}

// ===========================================================================
// GPU Excitation Setup
// ===========================================================================

void Engine_Vulkan::SetupGPUExcitation()
{
	// Find and remove the CPU excitation extension; upload its data to GPU.
	Operator_Ext_Excitation* opExc = const_cast<Operator*>(Op)->GetExcitationExtension();
	Excitation* exc = const_cast<Operator*>(Op)->GetExcitationSignal();

	// Remove CPU excitation first; excitation is always handled on the GPU.
	for (auto it = m_Eng_exts.begin(); it != m_Eng_exts.end(); )
	{
		Engine_Ext_Excitation* eext = dynamic_cast<Engine_Ext_Excitation*>(*it);
		if (eext)
		{
			// Remove CPU excitation; we handle it on GPU
			delete eext;
			it = m_Eng_exts.erase(it);
			continue;
		}

		++it;
	}

	bool forceCpuLocalABC = false;
	if (const char* env = std::getenv("OPENEMS_GPU_CPU_LOCAL_ABC"))
		forceCpuLocalABC = std::strcmp(env, "0") != 0;
	auto isGpuNative = [forceCpuLocalABC](Engine_Extension* ext)
	{
		if (forceCpuLocalABC && dynamic_cast<Engine_Ext_Absorbing_BC*>(ext))
			return false;
		return dynamic_cast<Engine_Ext_UPML*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_LorentzMaterial*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_TFSF*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_Mur_ABC*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_LumpedRLC*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_SteadyState*>(ext) != nullptr ||
		       dynamic_cast<Engine_Ext_Absorbing_BC*>(ext) != nullptr;
	};

	// The hybrid loop executes extensions on the CPU. Keep every extension
	// CPU-owned when even one extension lacks a GPU implementation, otherwise
	// mixed configurations would silently omit the GPU-native phases.
	m_hasCPUExtensions = std::any_of(m_Eng_exts.begin(), m_Eng_exts.end(),
		[&](Engine_Extension* ext) { return !isGpuNative(ext); });
	if (!m_hasCPUExtensions)
	{
		for (auto it = m_Eng_exts.begin(); it != m_Eng_exts.end(); )
		{
			// openEMS keeps a non-owning pointer to this observer. Leave it in
			// the engine-owned list, but do not execute it on the pure GPU path.
			if (dynamic_cast<Engine_Ext_SteadyState*>(*it))
			{
				++it;
				continue;
			}
			if (!isGpuNative(*it))
			{
				++it;
				continue;
			}
			delete *it;
			it = m_Eng_exts.erase(it);
		}
	}

	if (!opExc || !exc)
	{
		m_hasGPUExcitation = false;
		return;
	}

	m_hasGPUExcitation = true;
	m_excSignalLen    = exc->GetLength();
	m_excSignalPeriod = (exc->GetSignalPeriod() > 0)
	                    ? (int)(exc->GetSignalPeriod() / exc->GetTimestep())
	                    : 0;
	for (unsigned int i = 0; i < m_excSignalLen; ++i)
	{
		if (!std::isfinite(exc->GetVoltageSignal()[i]) ||
		    !std::isfinite(exc->GetCurrentSignal()[i]))
			throw std::runtime_error("Engine_Vulkan: non-finite excitation signal sample");
	}

	size_t N = (size_t)numLines[0] * numLines[1] * numLines[2];
	uint32_t sYZ = numLines[1] * numLines[2];

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	// --- Voltage excitation ---
	m_excVoltCount = opExc->GetVoltCount();
	if (m_excVoltCount > 0)
	{
		// Precompute linear indices
		std::vector<uint32_t> linIdx(m_excVoltCount);
		for (unsigned int n = 0; n < m_excVoltCount; n++)
		{
			uint32_t dir = opExc->GetVoltDir()[n];
			uint32_t x   = opExc->GetVoltIndex(0)[n];
			uint32_t y   = opExc->GetVoltIndex(1)[n];
			uint32_t z   = opExc->GetVoltIndex(2)[n];
			if (dir >= 3 || x >= numLines[0] || y >= numLines[1] || z >= numLines[2] ||
			    !std::isfinite(opExc->GetVoltAmp()[n]))
				throw std::runtime_error("Engine_Vulkan: invalid voltage excitation entry");
			linIdx[n] = dir * (uint32_t)N + x * sYZ + y * numLines[2] + z;
		}

		VkDeviceSize idxSize = m_excVoltCount * sizeof(uint32_t);
		VkDeviceSize ampSize = m_excVoltCount * sizeof(FDTD_FLOAT);
		CreateBufferWithMemory(idxSize, usage, dLoc, m_excVoltIdxBuf,   m_excVoltIdxMem);
		CreateBufferWithMemory(ampSize, usage, dLoc, m_excVoltAmpBuf,   m_excVoltAmpMem);
		CreateBufferWithMemory(idxSize, usage, dLoc, m_excVoltDelayBuf, m_excVoltDelayMem);
		UploadToDeviceBuffer(m_excVoltIdxBuf,   linIdx.data(), idxSize);
		UploadToDeviceBuffer(m_excVoltAmpBuf,   opExc->GetVoltAmp(), ampSize);
		UploadToDeviceBuffer(m_excVoltDelayBuf, opExc->GetVoltDelay(), idxSize);
	}

	// --- Current excitation ---
	m_excCurrCount = opExc->GetCurrCount();
	if (m_excCurrCount > 0)
	{
		std::vector<uint32_t> linIdx(m_excCurrCount);
		for (unsigned int n = 0; n < m_excCurrCount; n++)
		{
			uint32_t dir = opExc->GetCurrDir()[n];
			uint32_t x   = opExc->GetCurrIndex(0)[n];
			uint32_t y   = opExc->GetCurrIndex(1)[n];
			uint32_t z   = opExc->GetCurrIndex(2)[n];
			if (dir >= 3 || x >= numLines[0] || y >= numLines[1] || z >= numLines[2] ||
			    !std::isfinite(opExc->GetCurrAmp()[n]))
				throw std::runtime_error("Engine_Vulkan: invalid current excitation entry");
			linIdx[n] = dir * (uint32_t)N + x * sYZ + y * numLines[2] + z;
		}

		VkDeviceSize idxSize = m_excCurrCount * sizeof(uint32_t);
		VkDeviceSize ampSize = m_excCurrCount * sizeof(FDTD_FLOAT);
		CreateBufferWithMemory(idxSize, usage, dLoc, m_excCurrIdxBuf,   m_excCurrIdxMem);
		CreateBufferWithMemory(ampSize, usage, dLoc, m_excCurrAmpBuf,   m_excCurrAmpMem);
		CreateBufferWithMemory(idxSize, usage, dLoc, m_excCurrDelayBuf, m_excCurrDelayMem);
		UploadToDeviceBuffer(m_excCurrIdxBuf,   linIdx.data(), idxSize);
		UploadToDeviceBuffer(m_excCurrAmpBuf,   opExc->GetCurrAmp(), ampSize);
		UploadToDeviceBuffer(m_excCurrDelayBuf, opExc->GetCurrDelay(), idxSize);
	}

	// --- Excitation signal ---
	VkDeviceSize sigSize = m_excSignalLen * sizeof(FDTD_FLOAT);
	CreateBufferWithMemory(sigSize, usage, dLoc, m_excSignalVoltBuf, m_excSignalVoltMem);
	CreateBufferWithMemory(sigSize, usage, dLoc, m_excSignalCurrBuf, m_excSignalCurrMem);
	UploadToDeviceBuffer(m_excSignalVoltBuf, exc->GetVoltageSignal(), sigSize);
	UploadToDeviceBuffer(m_excSignalCurrBuf, exc->GetCurrentSignal(), sigSize);

	// --- Write excitation descriptor sets ---
	if (m_excVoltCount > 0)
	{
		VkBuffer      bufs[5] = {m_voltBuf, m_excVoltAmpBuf, m_excVoltIdxBuf,
		                          m_excVoltDelayBuf, m_excSignalVoltBuf};
		VkDeviceSize sizes[5] = {m_fieldBufSize,
		                          m_excVoltCount * sizeof(FDTD_FLOAT),
		                          m_excVoltCount * sizeof(uint32_t),
		                          m_excVoltCount * sizeof(uint32_t),
		                          sigSize};
		WriteDescriptorBuffers(m_device, m_excVoltDescSet, bufs, sizes, 5);
	}
	if (m_excCurrCount > 0)
	{
		VkBuffer      bufs[5] = {m_currBuf, m_excCurrAmpBuf, m_excCurrIdxBuf,
		                          m_excCurrDelayBuf, m_excSignalCurrBuf};
		VkDeviceSize sizes[5] = {m_fieldBufSize,
		                          m_excCurrCount * sizeof(FDTD_FLOAT),
		                          m_excCurrCount * sizeof(uint32_t),
		                          m_excCurrCount * sizeof(uint32_t),
		                          sigSize};
		WriteDescriptorBuffers(m_device, m_excCurrDescSet, bufs, sizes, 5);
	}
}

// ===========================================================================
// GPU Extension Setup (UPML, Dispersive, TF/SF, Mur ABC, LumpedRLC, Probes)
// ===========================================================================

void Engine_Vulkan::SetupGPUExtensions()
{
	// Hybrid execution retains all extensions on the CPU so their priority and
	// phase ordering remain identical to the reference engine.
	if (m_hasCPUExtensions)
		return;

	bool startupTrace = (g_settings.GetVerboseLevel() > 0);
	if (const char* env = std::getenv("OPENEMS_GPU_STARTUP_TRACE"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0')
			startupTrace = (v != 0);
	}

	auto extBegin = std::chrono::steady_clock::now();

	// Scan operator extensions and setup GPU resources for each
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();

	// Count what we need for descriptor pool sizing
	unsigned int numUPML = 0, numDispOrders = 0, numTFSF = 0, numMur = 0;
	unsigned int numRLC = 0;
	unsigned int totalDescSets = 0;
	unsigned int totalStorageBindings = 0;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		if (auto* upml = dynamic_cast<Operator_Ext_UPML*>(ext))
		{
			numUPML++;
			totalDescSets += 4;
			totalStorageBindings += 14;  // 4+3+4+3
		}
		else if (auto* lor = dynamic_cast<Operator_Ext_LorentzMaterial*>(ext))
		{
			numDispOrders += lor->m_Order;
			totalDescSets += lor->m_Order * 4;
			totalStorageBindings += lor->m_Order * 20;
		}
		else if (auto* tfsf = dynamic_cast<Operator_Ext_TFSF*>(ext))
		{
			numTFSF++;
			totalDescSets += 2;
			totalStorageBindings += 12;
		}
		else if (auto* mur = dynamic_cast<Operator_Ext_Mur_ABC*>(ext))
		{
			numMur++;
			totalDescSets += 3;
			totalStorageBindings += 13;
		}
		else if (auto* rlc = dynamic_cast<Operator_Ext_LumpedRLC*>(ext))
		{
			if (rlc->RLC_count > 0) {
				numRLC++;
				totalDescSets += 2;
				totalStorageBindings += 22;
			}
		}
	}
	// Probes (placeholder, will be set up later)
	// totalDescSets += 1;
	// totalStorageBindings += 5;

	if (startupTrace)
	{
		cout << "[Vulkan startup] extension scan: op_ext=" << nExts
		     << ", upml=" << numUPML
		     << ", dispersive_orders=" << numDispOrders
		     << ", tfsf=" << numTFSF
		     << ", mur=" << numMur
		     << ", rlc=" << numRLC
		     << ", desc_sets=" << totalDescSets
		     << ", storage_bindings=" << totalStorageBindings
		     << endl;
	}

	if (totalDescSets == 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] no GPU extension resources required" << endl;
		SetupGPU_LocalABC();
		SetupGPU_SteadyState();
		return;
	}

	// Create extension descriptor set layouts
	CreateExtensionDescriptorLayouts();

	// --- Create extension descriptor pool ---
	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, totalStorageBindings + 32};
	VkDescriptorPoolCreateInfo pi{};
	pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pi.maxSets       = totalDescSets + 8;
	pi.poolSizeCount = 1;
	pi.pPoolSizes    = &poolSize;
	pi.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	VK_CHECK(vkCreateDescriptorPool(m_device, &pi, nullptr, &m_extDescPool));

	// Setup each extension family exactly once.
	// Each SetupGPU_* function scans all operator extensions internally.
	if (numUPML > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: UPML" << endl;
		SetupGPU_UPML();
	}
	if (numDispOrders > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: Dispersive" << endl;
		SetupGPU_Dispersive();
	}
	if (numTFSF > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: TFSF" << endl;
		SetupGPU_TFSF();
	}
	if (numMur > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: Mur" << endl;
		SetupGPU_Mur();
	}
	if (numRLC > 0)
	{
		if (startupTrace)
			cout << "[Vulkan startup] setup extension family: RLC" << endl;
		SetupGPU_RLC();
	}

	// Create extension compute pipelines
	CreateExtensionPipelines();
	SetupGPU_LocalABC();
	SetupGPU_SteadyState();

	if (startupTrace)
	{
		auto extEnd = std::chrono::steady_clock::now();
		double extMs = std::chrono::duration<double, std::milli>(extEnd - extBegin).count();
		cout << "[Vulkan startup] SetupGPUExtensions total: " << extMs << " ms" << endl;
	}
}

void Engine_Vulkan::CreateExtensionDescriptorLayouts()
{
	auto makeLayout = [&](uint32_t nBindings) -> VkDescriptorSetLayout
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings(nBindings);
		for (uint32_t i = 0; i < nBindings; i++)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = nBindings;
		ci.pBindings    = bindings.data();
		VkDescriptorSetLayout layout;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &layout));
		return layout;
	};

	// UPML: preVolt=4, postVolt=3, preCurr=4, postCurr=3
	m_upmlPreVoltDescLayout  = makeLayout(4);
	m_upmlPostVoltDescLayout = makeLayout(3);
	m_upmlPreCurrDescLayout  = makeLayout(4);
	m_upmlPostCurrDescLayout = makeLayout(3);

	// Dispersive: pre=7 (field, ADE, LorADE, posIdx, int, ext, Lor), apply=3
	m_dispPreDescLayout   = makeLayout(7);
	m_dispApplyDescLayout = makeLayout(3);

	// TF/SF: 6 bindings (field, amp, delta, idx, delay, signal)
	m_tfsfDescLayout = makeLayout(6);

	// Mur: pre=5, post=5, apply=3
	m_murPreDescLayout   = makeLayout(5);
	m_murPostDescLayout  = makeLayout(5);
	m_murApplyDescLayout = makeLayout(3);

	// RLC: pre=6, apply=16
	m_rlcPreDescLayout   = makeLayout(6);
	m_rlcApplyDescLayout = makeLayout(16);

	// Probes: 5 bindings
	m_probeDescLayout = makeLayout(5);
}

void Engine_Vulkan::CreateExtensionPipelines()
{
	auto makePipeline = [&](VkShaderModule mod, VkPipelineLayout layout) -> VkPipeline
	{
		VkPipelineShaderStageCreateInfo stage{};
		stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = mod;
		stage.pName  = "main";
		VkComputePipelineCreateInfo ci{};
		ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ci.stage  = stage;
		ci.layout = layout;
		VkPipeline pipe;
		VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &pipe));
		return pipe;
	};

	auto makePipeLayout = [&](VkDescriptorSetLayout descLayout, uint32_t pcSize) -> VkPipelineLayout
	{
		VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize};
		VkPipelineLayoutCreateInfo pli{};
		pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pli.setLayoutCount         = 1;
		pli.pSetLayouts            = &descLayout;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges    = &pcRange;
		VkPipelineLayout layout;
		VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &layout));
		return layout;
	};

	// --- UPML pipelines ---
	if (m_hasGPU_UPML)
	{
		m_upmlPrePipeLayout  = makePipeLayout(m_upmlPreVoltDescLayout, sizeof(PmlPC));
		m_upmlPostPipeLayout = makePipeLayout(m_upmlPostVoltDescLayout, sizeof(PmlPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::upml_pre_voltage_data, gpu_spirv::upml_pre_voltage_size);
		m_upmlPreVoltPipeline = makePipeline(mod, m_upmlPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::upml_post_voltage_data, gpu_spirv::upml_post_voltage_size);
		m_upmlPostVoltPipeline = makePipeline(mod, m_upmlPostPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::upml_pre_current_data, gpu_spirv::upml_pre_current_size);
		m_upmlPreCurrPipeline = makePipeline(mod, m_upmlPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::upml_post_current_data, gpu_spirv::upml_post_current_size);
		m_upmlPostCurrPipeline = makePipeline(mod, m_upmlPostPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);
	}

	// --- Dispersive pipelines ---
	if (m_hasGPU_Dispersive)
	{
		m_dispPrePipeLayout   = makePipeLayout(m_dispPreDescLayout, sizeof(DispPC));
		m_dispApplyPipeLayout = makePipeLayout(m_dispApplyDescLayout, sizeof(DispPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::dispersive_pre_voltage_data, gpu_spirv::dispersive_pre_voltage_size);
		m_dispPreVoltPipeline = makePipeline(mod, m_dispPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::dispersive_pre_current_data, gpu_spirv::dispersive_pre_current_size);
		m_dispPreCurrPipeline = makePipeline(mod, m_dispPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::dispersive_apply_voltage_data, gpu_spirv::dispersive_apply_voltage_size);
		m_dispApplyVoltPipeline = makePipeline(mod, m_dispApplyPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::dispersive_apply_current_data, gpu_spirv::dispersive_apply_current_size);
		m_dispApplyCurrPipeline = makePipeline(mod, m_dispApplyPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);
	}

	// --- TF/SF pipelines ---
	if (m_hasGPU_TFSF)
	{
		m_tfsfPipeLayout = makePipeLayout(m_tfsfDescLayout, sizeof(TfsfPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::tfsf_voltage_data, gpu_spirv::tfsf_voltage_size);
		m_tfsfVoltPipeline = makePipeline(mod, m_tfsfPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::tfsf_current_data, gpu_spirv::tfsf_current_size);
		m_tfsfCurrPipeline = makePipeline(mod, m_tfsfPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);
	}

	// --- Mur ABC pipelines ---
	if (m_hasGPU_Mur)
	{
		m_murUpdatePipeLayout = makePipeLayout(m_murPreDescLayout, sizeof(MurPC));
		m_murApplyPipeLayout  = makePipeLayout(m_murApplyDescLayout, sizeof(MurPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::mur_pre_voltage_data, gpu_spirv::mur_pre_voltage_size);
		m_murPreVoltPipeline = makePipeline(mod, m_murUpdatePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::mur_post_voltage_data, gpu_spirv::mur_post_voltage_size);
		m_murPostVoltPipeline = makePipeline(mod, m_murUpdatePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::mur_apply_voltage_data, gpu_spirv::mur_apply_voltage_size);
		m_murApplyVoltPipeline = makePipeline(mod, m_murApplyPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);
	}

	// --- Lumped RLC pipelines ---
	if (m_hasGPU_RLC)
	{
		m_rlcPrePipeLayout   = makePipeLayout(m_rlcPreDescLayout, sizeof(RlcPC));
		m_rlcApplyPipeLayout = makePipeLayout(m_rlcApplyDescLayout, sizeof(RlcPC));

		VkShaderModule mod;
		mod = CreateShaderModule(gpu_spirv::rlc_pre_voltage_data, gpu_spirv::rlc_pre_voltage_size);
		m_rlcPreVoltPipeline = makePipeline(mod, m_rlcPrePipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);

		mod = CreateShaderModule(gpu_spirv::rlc_apply_voltage_data, gpu_spirv::rlc_apply_voltage_size);
		m_rlcApplyVoltPipeline = makePipeline(mod, m_rlcApplyPipeLayout);
		vkDestroyShaderModule(m_device, mod, nullptr);
	}

	// --- Probe gather pipeline ---
	// (not created here; will be created on demand when probes are set up)
}

// ---------------------------------------------------------------------------
// Per-extension GPU data upload and descriptor writing
// ---------------------------------------------------------------------------

void Engine_Vulkan::SetupGPU_UPML()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		auto* upml = dynamic_cast<Operator_Ext_UPML*>(ext);
		if (!upml) continue;

		GpuUPMLData data;
		uint32_t pNx = upml->m_numLines[0];
		uint32_t pNy = upml->m_numLines[1];
		uint32_t pNz = upml->m_numLines[2];
		uint32_t pN  = pNx * pNy * pNz;
		VkDeviceSize pmlBufSize = 3 * pN * sizeof(FDTD_FLOAT);  // [3][pNx][pNy][pNz]

		if (pN == 0) continue;

		data.pc.Nx = numLines[0]; data.pc.Ny = numLines[1]; data.pc.Nz = numLines[2];
		data.pc.pNx = pNx; data.pc.pNy = pNy; data.pc.pNz = pNz;
		data.pc.pStartX = upml->m_StartPos[0];
		data.pc.pStartY = upml->m_StartPos[1];
		data.pc.pStartZ = upml->m_StartPos[2];
		data.totalCells  = pN;

		auto validatePml = [pN](const ArrayLib::ArrayNIJK<FDTD_FLOAT>& values,
		                             const char* name)
		{
			for (uint32_t i = 0; i < 3u * pN; ++i)
			{
				if (!std::isfinite(values.data(i)))
					throw std::runtime_error(std::string("Engine_Vulkan: non-finite UPML ") +
					                         name + " coefficient at index " + std::to_string(i));
			}
		};
		validatePml(upml->vv, "VV");
		validatePml(upml->vvfo, "VVFO");
		validatePml(upml->vvfn, "VVFN");
		validatePml(upml->ii, "II");
		validatePml(upml->iifo, "IIFO");
		validatePml(upml->iifn, "IIFN");

		// Create and upload auxiliary flux buffers (initialized to zero)
		VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		CreateGpuBuf(data.voltFlux, pmlBufSize, rwUsage, dLoc);
		CreateGpuBuf(data.currFlux, pmlBufSize, rwUsage, dLoc);
		{
			std::vector<float> zeros(3 * pN, 0.0f);
			UploadGpuBuf(data.voltFlux, zeros.data(), pmlBufSize);
			UploadGpuBuf(data.currFlux, zeros.data(), pmlBufSize);
		}

		// Upload PML coefficient arrays (contiguous [3][pNx][pNy][pNz])
		CreateGpuBuf(data.pmlVv,   pmlBufSize, usage, dLoc);
		CreateGpuBuf(data.pmlVvfo, pmlBufSize, usage, dLoc);
		CreateGpuBuf(data.pmlVvfn, pmlBufSize, usage, dLoc);
		CreateGpuBuf(data.pmlIi,   pmlBufSize, usage, dLoc);
		CreateGpuBuf(data.pmlIifo, pmlBufSize, usage, dLoc);
		CreateGpuBuf(data.pmlIifn, pmlBufSize, usage, dLoc);
		UploadGpuBuf(data.pmlVv,   upml->vv.data(),   pmlBufSize);
		UploadGpuBuf(data.pmlVvfo, upml->vvfo.data(), pmlBufSize);
		UploadGpuBuf(data.pmlVvfn, upml->vvfn.data(), pmlBufSize);
		UploadGpuBuf(data.pmlIi,   upml->ii.data(),   pmlBufSize);
		UploadGpuBuf(data.pmlIifo, upml->iifo.data(), pmlBufSize);
		UploadGpuBuf(data.pmlIifn, upml->iifn.data(), pmlBufSize);

		// Allocate descriptor sets
		auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
		{
			VkDescriptorSetAllocateInfo ai{};
			ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			ai.descriptorPool     = m_extDescPool;
			ai.descriptorSetCount = 1;
			ai.pSetLayouts        = &layout;
			VkDescriptorSet set;
			VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
			return set;
		};

		data.preVoltDesc  = allocDescSet(m_upmlPreVoltDescLayout);
		data.postVoltDesc = allocDescSet(m_upmlPostVoltDescLayout);
		data.preCurrDesc  = allocDescSet(m_upmlPreCurrDescLayout);
		data.postCurrDesc = allocDescSet(m_upmlPostCurrDescLayout);

		// Write descriptors: preVolt = [volt, volt_flux, pml_vv, pml_vvfo]
		{
			VkBuffer bufs[4]      = {m_voltBuf, data.voltFlux.buffer, data.pmlVv.buffer, data.pmlVvfo.buffer};
			VkDeviceSize sizes[4] = {m_fieldBufSize, pmlBufSize, pmlBufSize, pmlBufSize};
			WriteDescriptorBuffers(m_device, data.preVoltDesc, bufs, sizes, 4);
		}
		// postVolt = [volt, volt_flux, pml_vvfn]
		{
			VkBuffer bufs[3]      = {m_voltBuf, data.voltFlux.buffer, data.pmlVvfn.buffer};
			VkDeviceSize sizes[3] = {m_fieldBufSize, pmlBufSize, pmlBufSize};
			WriteDescriptorBuffers(m_device, data.postVoltDesc, bufs, sizes, 3);
		}
		// preCurr = [curr, curr_flux, pml_ii, pml_iifo]
		{
			VkBuffer bufs[4]      = {m_currBuf, data.currFlux.buffer, data.pmlIi.buffer, data.pmlIifo.buffer};
			VkDeviceSize sizes[4] = {m_fieldBufSize, pmlBufSize, pmlBufSize, pmlBufSize};
			WriteDescriptorBuffers(m_device, data.preCurrDesc, bufs, sizes, 4);
		}
		// postCurr = [curr, curr_flux, pml_iifn]
		{
			VkBuffer bufs[3]      = {m_currBuf, data.currFlux.buffer, data.pmlIifn.buffer};
			VkDeviceSize sizes[3] = {m_fieldBufSize, pmlBufSize, pmlBufSize};
			WriteDescriptorBuffers(m_device, data.postCurrDesc, bufs, sizes, 3);
		}

		m_gpuUPML.push_back(std::move(data));
	}

	m_hasGPU_UPML = !m_gpuUPML.empty();
}

void Engine_Vulkan::SetupGPU_Dispersive()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	uint32_t N  = m_fieldN;
	uint32_t sYZ = m_strideYZ;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		auto* lor = dynamic_cast<Operator_Ext_LorentzMaterial*>(ext);
		if (!lor) continue;

		for (int order = 0; order < lor->m_Order; order++)
		{
			uint32_t cnt = lor->m_LM_Count[order];
			if (cnt == 0) continue;

			GpuDispData data;
			data.count     = cnt;
			data.voltADEOn = lor->m_volt_ADE_On[order];
			data.currADEOn = lor->m_curr_ADE_On[order];
			data.voltLorADEOn = data.voltADEOn && lor->m_volt_Lor_ADE_On && lor->m_volt_Lor_ADE_On[order];
			data.currLorADEOn = data.currADEOn && lor->m_curr_Lor_ADE_On && lor->m_curr_Lor_ADE_On[order];

			data.voltPC = {cnt, numLines[0], numLines[1], numLines[2], data.voltLorADEOn ? 1u : 0u};
			data.currPC = {cnt, numLines[0], numLines[1], numLines[2], data.currLorADEOn ? 1u : 0u};

			VkDeviceSize adeSize = 3 * cnt * sizeof(FDTD_FLOAT);
			VkDeviceSize idxSize = 3 * cnt * sizeof(uint32_t);

			// Flatten position indices: [3][count]
			std::vector<uint32_t> posIdx(3 * cnt);
			for (uint32_t i = 0; i < cnt; i++)
			{
				posIdx[i]           = lor->m_LM_pos[order][0][i];
				posIdx[cnt + i]     = lor->m_LM_pos[order][1][i];
				posIdx[2*cnt + i]   = lor->m_LM_pos[order][2][i];
			}
			CreateGpuBuf(data.posIdx, idxSize, usage, dLoc);
			UploadGpuBuf(data.posIdx, posIdx.data(), idxSize);

			// ADE state (zero-initialized)
			std::vector<float> zeros(3 * cnt, 0.0f);
			CreateGpuBuf(data.voltADE, adeSize, rwUsage, dLoc);
			UploadGpuBuf(data.voltADE, zeros.data(), adeSize);
			CreateGpuBuf(data.currADE, adeSize, rwUsage, dLoc);
			UploadGpuBuf(data.currADE, zeros.data(), adeSize);

			if (data.voltLorADEOn)
			{
				CreateGpuBuf(data.voltLorADE, adeSize, rwUsage, dLoc);
				UploadGpuBuf(data.voltLorADE, zeros.data(), adeSize);
			}
			if (data.currLorADEOn)
			{
				CreateGpuBuf(data.currLorADE, adeSize, rwUsage, dLoc);
				UploadGpuBuf(data.currLorADE, zeros.data(), adeSize);
			}

			// Flatten coefficient arrays: [3][count]
			auto flattenCoef = [&](FDTD_FLOAT*** arr) -> std::vector<float>
			{
				std::vector<float> flat(3 * cnt);
				for (int n = 0; n < 3; n++)
					for (uint32_t i = 0; i < cnt; i++)
						flat[n * cnt + i] = arr[order][n][i];
				return flat;
			};

			auto uploadCoef = [&](GpuBuf& gb, FDTD_FLOAT*** arr)
			{
				auto flat = flattenCoef(arr);
				CreateGpuBuf(gb, adeSize, usage, dLoc);
				UploadGpuBuf(gb, flat.data(), adeSize);
			};

			if (data.voltADEOn)
			{
				uploadCoef(data.vIntADE, lor->v_int_ADE);
				uploadCoef(data.vExtADE, lor->v_ext_ADE);
			}
			if (data.currADEOn)
			{
				uploadCoef(data.iIntADE, lor->i_int_ADE);
				uploadCoef(data.iExtADE, lor->i_ext_ADE);
			}
			if (data.voltLorADEOn)
				uploadCoef(data.vLorADE, lor->v_Lor_ADE);
			if (data.currLorADEOn)
				uploadCoef(data.iLorADE, lor->i_Lor_ADE);

			// Allocate descriptor sets
			auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
			{
				VkDescriptorSetAllocateInfo ai{};
				ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				ai.descriptorPool     = m_extDescPool;
				ai.descriptorSetCount = 1;
				ai.pSetLayouts        = &layout;
				VkDescriptorSet set;
				VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
				return set;
			};

			// Use dummy buffer for unused Lorentz slots when hasLorADE=false
			VkBuffer voltDummy = data.voltADE.buffer;
			VkBuffer currDummy = data.currADE.buffer;
			VkBuffer lorVoltBuf = data.voltLorADEOn ? data.voltLorADE.buffer : voltDummy;
			VkBuffer lorCurrBuf = data.currLorADEOn ? data.currLorADE.buffer : currDummy;
			VkBuffer vIntBuf = data.voltADEOn ? data.vIntADE.buffer : voltDummy;
			VkBuffer vExtBuf = data.voltADEOn ? data.vExtADE.buffer : voltDummy;
			VkBuffer iIntBuf = data.currADEOn ? data.iIntADE.buffer : currDummy;
			VkBuffer iExtBuf = data.currADEOn ? data.iExtADE.buffer : currDummy;
			VkBuffer vLorBuf = data.voltLorADEOn ? data.vLorADE.buffer : voltDummy;
			VkBuffer iLorBuf = data.currLorADEOn ? data.iLorADE.buffer : currDummy;

			data.preVoltDesc = allocDescSet(m_dispPreDescLayout);
			{
				VkBuffer bufs[7]      = {m_voltBuf, data.voltADE.buffer, lorVoltBuf,
				                          data.posIdx.buffer, vIntBuf, vExtBuf, vLorBuf};
				VkDeviceSize sizes[7] = {m_fieldBufSize, adeSize, adeSize, idxSize, adeSize, adeSize, adeSize};
				WriteDescriptorBuffers(m_device, data.preVoltDesc, bufs, sizes, 7);
			}

			data.preCurrDesc = allocDescSet(m_dispPreDescLayout);
			{
				VkBuffer bufs[7]      = {m_currBuf, data.currADE.buffer, lorCurrBuf,
				                          data.posIdx.buffer, iIntBuf, iExtBuf, iLorBuf};
				VkDeviceSize sizes[7] = {m_fieldBufSize, adeSize, adeSize, idxSize, adeSize, adeSize, adeSize};
				WriteDescriptorBuffers(m_device, data.preCurrDesc, bufs, sizes, 7);
			}

			data.applyVoltDesc = allocDescSet(m_dispApplyDescLayout);
			{
				VkBuffer bufs[3]      = {m_voltBuf, data.voltADE.buffer, data.posIdx.buffer};
				VkDeviceSize sizes[3] = {m_fieldBufSize, adeSize, idxSize};
				WriteDescriptorBuffers(m_device, data.applyVoltDesc, bufs, sizes, 3);
			}

			data.applyCurrDesc = allocDescSet(m_dispApplyDescLayout);
			{
				VkBuffer bufs[3]      = {m_currBuf, data.currADE.buffer, data.posIdx.buffer};
				VkDeviceSize sizes[3] = {m_fieldBufSize, adeSize, idxSize};
				WriteDescriptorBuffers(m_device, data.applyCurrDesc, bufs, sizes, 3);
			}

			m_gpuDisp.push_back(std::move(data));
		}
	}

	m_hasGPU_Dispersive = !m_gpuDisp.empty();
}

void Engine_Vulkan::SetupGPU_TFSF()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	uint32_t N = m_fieldN;
	uint32_t sYZ = m_strideYZ;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		auto* tfsf = dynamic_cast<Operator_Ext_TFSF*>(ext);
		if (!tfsf) continue;

		Excitation* exc = tfsf->m_Exc;
		if (!exc) continue;

		// Count total injection points for voltage and current
		uint32_t voltTotal = 0, currTotal = 0;
		for (int n = 0; n < 3; n++)
		{
			int nP = (n+1)%3, nPP = (n+2)%3;
			uint32_t faceSize = tfsf->m_numLines[nP] * tfsf->m_numLines[nPP];
			for (int side = 0; side < 2; side++)
			{
				if (tfsf->m_ActiveDir[n][side])
				{
					voltTotal += 2 * faceSize;  // 2 tangential components
					currTotal += 2 * faceSize;
				}
			}
		}

		if (voltTotal == 0 && currTotal == 0) continue;

		// Flatten voltage injection points
		std::vector<float>    vAmp(voltTotal), vDelta(voltTotal);
		std::vector<uint32_t> vIdx(voltTotal), vDelay(voltTotal);
		uint32_t vi = 0;

		for (int n = 0; n < 3; n++)
		{
			int nP = (n+1)%3, nPP = (n+2)%3;
			for (int side = 0; side < 2; side++)
			{
				if (!tfsf->m_ActiveDir[n][side]) continue;

				uint32_t linePos = (side == 0) ? tfsf->m_Start[n] : tfsf->m_Stop[n];
				unsigned int ui_pos = 0;
				unsigned int pos[3];

				for (unsigned int i = 0; i < tfsf->m_numLines[nP]; i++)
				{
					for (unsigned int j = 0; j < tfsf->m_numLines[nPP]; j++)
					{
						pos[nP]  = tfsf->m_Start[nP] + i;
						pos[nPP] = tfsf->m_Start[nPP] + j;
						pos[n]   = linePos;

						// nP component
						uint32_t gIdx_nP = nP * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2];
						vIdx[vi]   = gIdx_nP;
						vAmp[vi]   = tfsf->m_VoltAmp[n][side][0][ui_pos];
						vDelta[vi] = tfsf->m_VoltDelayDelta[n][side][0][ui_pos];
						vDelay[vi] = tfsf->m_VoltDelay[n][side][0][ui_pos];
						vi++;

						// nPP component
						uint32_t gIdx_nPP = nPP * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2];
						vIdx[vi]   = gIdx_nPP;
						vAmp[vi]   = tfsf->m_VoltAmp[n][side][1][ui_pos];
						vDelta[vi] = tfsf->m_VoltDelayDelta[n][side][1][ui_pos];
						vDelay[vi] = tfsf->m_VoltDelay[n][side][1][ui_pos];
						vi++;

						ui_pos++;
					}
				}
			}
		}

		// Flatten current injection points
		std::vector<float>    cAmp(currTotal), cDelta(currTotal);
		std::vector<uint32_t> cIdx(currTotal), cDelay(currTotal);
		uint32_t ci_idx = 0;

		for (int n = 0; n < 3; n++)
		{
			int nP = (n+1)%3, nPP = (n+2)%3;
			for (int side = 0; side < 2; side++)
			{
				if (!tfsf->m_ActiveDir[n][side]) continue;

				// Current injection: lower face uses Start[n]-1, upper face uses Stop[n]
				uint32_t linePos = (side == 0) ? tfsf->m_Start[n] - 1 : tfsf->m_Stop[n];
				unsigned int ui_pos = 0;
				unsigned int pos[3];

				for (unsigned int i = 0; i < tfsf->m_numLines[nP]; i++)
				{
					for (unsigned int j = 0; j < tfsf->m_numLines[nPP]; j++)
					{
						pos[nP]  = tfsf->m_Start[nP] + i;
						pos[nPP] = tfsf->m_Start[nPP] + j;
						pos[n]   = linePos;

						uint32_t gIdx_nP = nP * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2];
						cIdx[ci_idx]   = gIdx_nP;
						cAmp[ci_idx]   = tfsf->m_CurrAmp[n][side][0][ui_pos];
						cDelta[ci_idx] = tfsf->m_CurrDelayDelta[n][side][0][ui_pos];
						cDelay[ci_idx] = tfsf->m_CurrDelay[n][side][0][ui_pos];
						ci_idx++;

						uint32_t gIdx_nPP = nPP * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2];
						cIdx[ci_idx]   = gIdx_nPP;
						cAmp[ci_idx]   = tfsf->m_CurrAmp[n][side][1][ui_pos];
						cDelta[ci_idx] = tfsf->m_CurrDelayDelta[n][side][1][ui_pos];
						cDelay[ci_idx] = tfsf->m_CurrDelay[n][side][1][ui_pos];
						ci_idx++;

						ui_pos++;
					}
				}
			}
		}

		m_gpuTFSF.voltCount = voltTotal;
		m_gpuTFSF.currCount = currTotal;

		// Upload voltage TF/SF data
		if (voltTotal > 0)
		{
			VkDeviceSize fSize = voltTotal * sizeof(float);
			VkDeviceSize iSize = voltTotal * sizeof(uint32_t);
			CreateGpuBuf(m_gpuTFSF.voltAmp,   fSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.voltDelta,  fSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.voltIdx,    iSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.voltDelay,  iSize, usage, dLoc);
			UploadGpuBuf(m_gpuTFSF.voltAmp,   vAmp.data(),   fSize);
			UploadGpuBuf(m_gpuTFSF.voltDelta,  vDelta.data(), fSize);
			UploadGpuBuf(m_gpuTFSF.voltIdx,    vIdx.data(),   iSize);
			UploadGpuBuf(m_gpuTFSF.voltDelay,  vDelay.data(), iSize);

			// Signal: H-field signal for voltage injection
			VkDeviceSize sigSize = exc->GetLength() * sizeof(FDTD_FLOAT);
			CreateGpuBuf(m_gpuTFSF.voltSignal, sigSize, usage, dLoc);
			UploadGpuBuf(m_gpuTFSF.voltSignal, exc->GetCurrentSignal(), sigSize);
		}

		// Upload current TF/SF data
		if (currTotal > 0)
		{
			VkDeviceSize fSize = currTotal * sizeof(float);
			VkDeviceSize iSize = currTotal * sizeof(uint32_t);
			CreateGpuBuf(m_gpuTFSF.currAmp,   fSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.currDelta,  fSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.currIdx,    iSize, usage, dLoc);
			CreateGpuBuf(m_gpuTFSF.currDelay,  iSize, usage, dLoc);
			UploadGpuBuf(m_gpuTFSF.currAmp,   cAmp.data(),   fSize);
			UploadGpuBuf(m_gpuTFSF.currDelta,  cDelta.data(), fSize);
			UploadGpuBuf(m_gpuTFSF.currIdx,    cIdx.data(),   iSize);
			UploadGpuBuf(m_gpuTFSF.currDelay,  cDelay.data(), iSize);

			// Signal: E-field signal for current injection
			VkDeviceSize sigSize = exc->GetLength() * sizeof(FDTD_FLOAT);
			CreateGpuBuf(m_gpuTFSF.currSignal, sigSize, usage, dLoc);
			UploadGpuBuf(m_gpuTFSF.currSignal, exc->GetVoltageSignal(), sigSize);
		}

		// Descriptor sets
		auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
		{
			VkDescriptorSetAllocateInfo ai{};
			ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			ai.descriptorPool     = m_extDescPool;
			ai.descriptorSetCount = 1;
			ai.pSetLayouts        = &layout;
			VkDescriptorSet set;
			VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
			return set;
		};

		VkDeviceSize sigSize = exc->GetLength() * sizeof(FDTD_FLOAT);

		if (voltTotal > 0)
		{
			m_gpuTFSF.voltDesc = allocDescSet(m_tfsfDescLayout);
			VkDeviceSize fSize = voltTotal * sizeof(float);
			VkDeviceSize iSize = voltTotal * sizeof(uint32_t);
			VkBuffer bufs[6]      = {m_voltBuf, m_gpuTFSF.voltAmp.buffer, m_gpuTFSF.voltDelta.buffer,
			                          m_gpuTFSF.voltIdx.buffer, m_gpuTFSF.voltDelay.buffer,
			                          m_gpuTFSF.voltSignal.buffer};
			VkDeviceSize sizes[6] = {m_fieldBufSize, fSize, fSize, iSize, iSize, sigSize};
			WriteDescriptorBuffers(m_device, m_gpuTFSF.voltDesc, bufs, sizes, 6);
		}

		if (currTotal > 0)
		{
			m_gpuTFSF.currDesc = allocDescSet(m_tfsfDescLayout);
			VkDeviceSize fSize = currTotal * sizeof(float);
			VkDeviceSize iSize = currTotal * sizeof(uint32_t);
			VkBuffer bufs[6]      = {m_currBuf, m_gpuTFSF.currAmp.buffer, m_gpuTFSF.currDelta.buffer,
			                          m_gpuTFSF.currIdx.buffer, m_gpuTFSF.currDelay.buffer,
			                          m_gpuTFSF.currSignal.buffer};
			VkDeviceSize sizes[6] = {m_fieldBufSize, fSize, fSize, iSize, iSize, sigSize};
			WriteDescriptorBuffers(m_device, m_gpuTFSF.currDesc, bufs, sizes, 6);
		}

		m_hasGPU_TFSF = true;
		break;  // Only one TF/SF extension expected
	}
}

void Engine_Vulkan::SetupGPU_Mur()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		auto* mur = dynamic_cast<Operator_Ext_Mur_ABC*>(ext);
		if (!mur) continue;

		GpuMurData data;
		uint32_t numLinesP  = mur->m_numLines[0];
		uint32_t numLinesPP = mur->m_numLines[1];
		uint32_t total      = numLinesP * numLinesPP;
		if (total == 0) continue;

		data.totalCells = total;
		data.pc.Nx = numLines[0]; data.pc.Ny = numLines[1]; data.pc.Nz = numLines[2];
		data.pc.ny          = mur->m_ny;
		data.pc.lineNr      = mur->m_LineNr;
		data.pc.lineNrShift = mur->m_LineNr_Shift;
		data.pc.numLinesP   = numLinesP;
		data.pc.numLinesPP  = numLinesPP;

		VkDeviceSize bufSize = total * sizeof(FDTD_FLOAT);

		// Intermediate value buffers (zero-init)
		CreateGpuBuf(data.murNyP,  bufSize, rwUsage, dLoc);
		CreateGpuBuf(data.murNyPP, bufSize, rwUsage, dLoc);
		{
			std::vector<float> zeros(total, 0.0f);
			UploadGpuBuf(data.murNyP,  zeros.data(), bufSize);
			UploadGpuBuf(data.murNyPP, zeros.data(), bufSize);
		}

		// Coefficient buffers
		CreateGpuBuf(data.coeffNyP,  bufSize, usage, dLoc);
		CreateGpuBuf(data.coeffNyPP, bufSize, usage, dLoc);
		UploadGpuBuf(data.coeffNyP,  mur->m_Mur_Coeff_nyP.data(),  bufSize);
		UploadGpuBuf(data.coeffNyPP, mur->m_Mur_Coeff_nyPP.data(), bufSize);

		auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
		{
			VkDescriptorSetAllocateInfo ai{};
			ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			ai.descriptorPool     = m_extDescPool;
			ai.descriptorSetCount = 1;
			ai.pSetLayouts        = &layout;
			VkDescriptorSet set;
			VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
			return set;
		};

		// pre = [volt, murNyP, murNyPP, coeffNyP, coeffNyPP]
		data.preDesc = allocDescSet(m_murPreDescLayout);
		{
			VkBuffer bufs[5]      = {m_voltBuf, data.murNyP.buffer, data.murNyPP.buffer,
			                          data.coeffNyP.buffer, data.coeffNyPP.buffer};
			VkDeviceSize sizes[5] = {m_fieldBufSize, bufSize, bufSize, bufSize, bufSize};
			WriteDescriptorBuffers(m_device, data.preDesc, bufs, sizes, 5);
		}

		// post = same layout as pre
		data.postDesc = allocDescSet(m_murPostDescLayout);
		{
			VkBuffer bufs[5]      = {m_voltBuf, data.murNyP.buffer, data.murNyPP.buffer,
			                          data.coeffNyP.buffer, data.coeffNyPP.buffer};
			VkDeviceSize sizes[5] = {m_fieldBufSize, bufSize, bufSize, bufSize, bufSize};
			WriteDescriptorBuffers(m_device, data.postDesc, bufs, sizes, 5);
		}

		// apply = [volt, murNyP, murNyPP]
		data.applyDesc = allocDescSet(m_murApplyDescLayout);
		{
			VkBuffer bufs[3]      = {m_voltBuf, data.murNyP.buffer, data.murNyPP.buffer};
			VkDeviceSize sizes[3] = {m_fieldBufSize, bufSize, bufSize};
			WriteDescriptorBuffers(m_device, data.applyDesc, bufs, sizes, 3);
		}

		m_gpuMur.push_back(std::move(data));
	}

	m_hasGPU_Mur = !m_gpuMur.empty();
}

void Engine_Vulkan::SetupGPU_RLC()
{
	// Iterate operator extensions
	size_t nExts = Op->GetNumberOfExtentions();
	VkBufferUsageFlags usage   = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	uint32_t N   = m_fieldN;
	uint32_t sYZ = m_strideYZ;

	for (size_t eIdx = 0; eIdx < nExts; eIdx++)
	{
		Operator_Extension* ext = Op->GetExtension(eIdx);
		auto* rlc = dynamic_cast<Operator_Ext_LumpedRLC*>(ext);
		if (!rlc || rlc->RLC_count == 0) continue;

		uint32_t cnt = rlc->RLC_count;
		m_gpuRLC.count = cnt;

		VkDeviceSize fSize = cnt * sizeof(float);
		VkDeviceSize iSize = cnt * sizeof(uint32_t);

		// Pre-linearize indices: dir*N + x*sYZ + y*Nz + z
		std::vector<uint32_t> linIdx(cnt);
		for (uint32_t i = 0; i < cnt; i++)
		{
			uint32_t dir = rlc->v_RLC_dir[i];
			uint32_t x   = rlc->v_RLC_pos[0][i];
			uint32_t y   = rlc->v_RLC_pos[1][i];
			uint32_t z   = rlc->v_RLC_pos[2][i];
			linIdx[i] = dir * N + x * sYZ + y * numLines[2] + z;
		}

		CreateGpuBuf(m_gpuRLC.linIdx, iSize, usage, dLoc);
		UploadGpuBuf(m_gpuRLC.linIdx, linIdx.data(), iSize);

		// State buffers (zero-init)
		std::vector<float> zeros(cnt, 0.0f);
		CreateGpuBuf(m_gpuRLC.il,   fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.il,   zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.vdn0, fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.vdn0, zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.vdn1, fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.vdn1, zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.vdn2, fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.vdn2, zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.jn0,  fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.jn0,  zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.jn1,  fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.jn1,  zeros.data(), fSize);
		CreateGpuBuf(m_gpuRLC.jn2,  fSize, rwUsage, dLoc); UploadGpuBuf(m_gpuRLC.jn2,  zeros.data(), fSize);

		// Coefficient buffers
		auto uploadCoef = [&](GpuBuf& gb, FDTD_FLOAT* data_ptr)
		{
			CreateGpuBuf(gb, fSize, usage, dLoc);
			UploadGpuBuf(gb, data_ptr, fSize);
		};
		uploadCoef(m_gpuRLC.i2v, rlc->v_RLC_i2v);
		uploadCoef(m_gpuRLC.ilv, rlc->v_RLC_ilv);
		uploadCoef(m_gpuRLC.vvd, rlc->v_RLC_vvd);
		uploadCoef(m_gpuRLC.vv2, rlc->v_RLC_vv2);
		uploadCoef(m_gpuRLC.vj1, rlc->v_RLC_vj1);
		uploadCoef(m_gpuRLC.vj2, rlc->v_RLC_vj2);
		uploadCoef(m_gpuRLC.ib0, rlc->v_RLC_ib0);
		uploadCoef(m_gpuRLC.b1,  rlc->v_RLC_b1);
		uploadCoef(m_gpuRLC.b2,  rlc->v_RLC_b2);

		// Descriptor sets
		auto allocDescSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet
		{
			VkDescriptorSetAllocateInfo ai{};
			ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			ai.descriptorPool     = m_extDescPool;
			ai.descriptorSetCount = 1;
			ai.pSetLayouts        = &layout;
			VkDescriptorSet set;
			VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &set));
			return set;
		};

		// preDesc: [il, vdn0, vdn1, vdn2, i2v, ilv]
		m_gpuRLC.preDesc = allocDescSet(m_rlcPreDescLayout);
		{
			VkBuffer bufs[6]      = {m_gpuRLC.il.buffer, m_gpuRLC.vdn0.buffer,
			                          m_gpuRLC.vdn1.buffer, m_gpuRLC.vdn2.buffer,
			                          m_gpuRLC.i2v.buffer, m_gpuRLC.ilv.buffer};
			VkDeviceSize sizes[6] = {fSize, fSize, fSize, fSize, fSize, fSize};
			WriteDescriptorBuffers(m_device, m_gpuRLC.preDesc, bufs, sizes, 6);
		}

		// applyDesc: [volt, il, vdn0, vdn1, vdn2, jn0, jn1, jn2, linIdx, vvd, vv2, vj1, vj2, ib0, b1, b2]
		m_gpuRLC.applyDesc = allocDescSet(m_rlcApplyDescLayout);
		{
			VkBuffer bufs[16] = {
				m_voltBuf, m_gpuRLC.il.buffer,
				m_gpuRLC.vdn0.buffer, m_gpuRLC.vdn1.buffer, m_gpuRLC.vdn2.buffer,
				m_gpuRLC.jn0.buffer, m_gpuRLC.jn1.buffer, m_gpuRLC.jn2.buffer,
				m_gpuRLC.linIdx.buffer,
				m_gpuRLC.vvd.buffer, m_gpuRLC.vv2.buffer,
				m_gpuRLC.vj1.buffer, m_gpuRLC.vj2.buffer,
				m_gpuRLC.ib0.buffer, m_gpuRLC.b1.buffer, m_gpuRLC.b2.buffer
			};
			VkDeviceSize sizes[16] = {
				m_fieldBufSize, fSize,
				fSize, fSize, fSize,
				fSize, fSize, fSize,
				iSize,
				fSize, fSize,
				fSize, fSize,
				fSize, fSize, fSize
			};
			WriteDescriptorBuffers(m_device, m_gpuRLC.applyDesc, bufs, sizes, 16);
		}

		m_hasGPU_RLC = true;
		break;  // Only one RLC extension expected
	}
}

void Engine_Vulkan::SetupGPU_SteadyState()
{
	Operator_Ext_SteadyState* opSteady = nullptr;
	for (size_t i = 0; i < Op->GetNumberOfExtentions(); ++i)
	{
		opSteady = dynamic_cast<Operator_Ext_SteadyState*>(Op->GetExtension(i));
		if (opSteady) break;
	}
	for (Engine_Extension* ext : m_Eng_exts)
	{
		m_gpuSteadyStateExt = dynamic_cast<Engine_Ext_SteadyState*>(ext);
		if (m_gpuSteadyStateExt) break;
	}
	if (!opSteady || !m_gpuSteadyStateExt || opSteady->m_TS_period == 0 ||
	    opSteady->m_E_probe_dir.empty())
		return;

	m_steadyProbeCount = (uint32_t)opSteady->m_E_probe_dir.size();
	m_steadyPeriod = opSteady->m_TS_period;
	m_steadyRingSize = 3 * m_steadyPeriod;
	const uint64_t historyValues = (uint64_t)m_steadyProbeCount * m_steadyRingSize;
	if (historyValues > UINT32_MAX)
		throw std::runtime_error("Vulkan steady-state history exceeds 32-bit shader indexing");

	std::vector<uint32_t> indices(m_steadyProbeCount);
	for (uint32_t i = 0; i < m_steadyProbeCount; ++i)
	{
		indices[i] = opSteady->m_E_probe_dir[i] * m_fieldN +
		             opSteady->m_E_probe_pos[0][i] * m_strideYZ +
		             opSteady->m_E_probe_pos[1][i] * numLines[2] +
		             opSteady->m_E_probe_pos[2][i];
	}

	const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	const VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	const VkDeviceSize indexSize = indices.size() * sizeof(uint32_t);
	const VkDeviceSize historySize = historyValues * sizeof(float);
	CreateGpuBuf(m_steadyProbeIdx, indexSize, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceLocal);
	UploadGpuBuf(m_steadyProbeIdx, indices.data(), indexSize);
	CreateGpuBuf(m_steadyHistory, historySize, storage, hostVisible);
	VK_CHECK(vkMapMemory(m_device, m_steadyHistory.memory, 0, historySize, 0,
	                     (void**)&m_steadyHistoryMapped));
	std::memset(m_steadyHistoryMapped, 0, historySize);

	VkDescriptorSetLayoutBinding bindings[3];
	for (uint32_t i = 0; i < 3; ++i)
		bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo dli{};
	dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dli.bindingCount = 3;
	dli.pBindings = bindings;
	VK_CHECK(vkCreateDescriptorSetLayout(m_device, &dli, nullptr, &m_steadyDescLayout));

	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SteadyPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount = 1;
	pli.pSetLayouts = &m_steadyDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_steadyPipeLayout));

	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets = 1;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_steadyDescPool));
	VkDescriptorSetAllocateInfo ai{};
	ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool = m_steadyDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts = &m_steadyDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &m_steadyDescSet));
	VkBuffer bufs[3] = {m_voltBuf, m_steadyProbeIdx.buffer, m_steadyHistory.buffer};
	VkDeviceSize sizes[3] = {m_fieldBufSize, indexSize, historySize};
	WriteDescriptorBuffers(m_device, m_steadyDescSet, bufs, sizes, 3);

	VkShaderModule mod = CreateShaderModule(gpu_spirv::steady_state_sample_data,
	                                        gpu_spirv::steady_state_sample_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName = "main";
	VkComputePipelineCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage = stage;
	ci.layout = m_steadyPipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_steadyPipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

	m_gpuSteadyStateExt->SetGpuUpdater([this]() { UpdateSteadyStateResult(); });
}

void Engine_Vulkan::SetupGPU_LocalABC()
{
	std::vector<Operator_Ext_Absorbing_BC*> sheets;
	for (size_t i = 0; i < Op->GetNumberOfExtentions(); ++i)
	{
		if (auto* sheet = dynamic_cast<Operator_Ext_Absorbing_BC*>(Op->GetExtension(i)))
			sheets.push_back(sheet);
	}
	if (sheets.empty()) return;

	VkDescriptorSetLayoutBinding bindings[4];
	for (uint32_t i = 0; i < 4; ++i)
		bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo dli{};
	dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dli.bindingCount = 4;
	dli.pBindings = bindings;
	VK_CHECK(vkCreateDescriptorSetLayout(m_device, &dli, nullptr, &m_localAbcDescLayout));

	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LocalAbcPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount = 1;
	pli.pSetLayouts = &m_localAbcDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_localAbcPipeLayout));

	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)sheets.size() * 8};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets = (uint32_t)sheets.size() * 2;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_localAbcDescPool));

	VkShaderModule mod = CreateShaderModule(gpu_spirv::local_absorbing_bc_data,
	                                        gpu_spirv::local_absorbing_bc_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName = "main";
	VkComputePipelineCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage = stage;
	ci.layout = m_localAbcPipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_localAbcPipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

	const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	const VkMemoryPropertyFlags deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	for (Operator_Ext_Absorbing_BC* sheet : sheets)
	{
		GpuLocalABCData data;
		data.superAbsorption = sheet->m_ABCtype == Operator_Ext_Absorbing_BC::MUR_1ST_SA;
		data.pc.Nx = numLines[0]; data.pc.Ny = numLines[1]; data.pc.Nz = numLines[2];
		data.pc.ny = sheet->m_ny; data.pc.nyP = sheet->m_nyP; data.pc.nyPP = sheet->m_nyPP;
		data.pc.startX = sheet->m_sheetX0[0];
		data.pc.startY = sheet->m_sheetX0[1];
		data.pc.startZ = sheet->m_sheetX0[2];
		data.pc.voltBoundary = sheet->m_sheetX0[sheet->m_ny];
		data.pc.voltShift = data.pc.voltBoundary + (sheet->m_normalSignPositive ? 1 : -1);
		data.pc.currBoundary = data.pc.voltBoundary + (sheet->m_normalSignPositive ? 0 : -1);
		data.pc.currShift = data.pc.voltBoundary + (sheet->m_normalSignPositive ? 1 : -2);
		data.pc.sizeP = sheet->m_numLines[0];
		data.pc.sizePP = sheet->m_numLines[1];
		data.pc.countVolt = data.pc.sizeP * data.pc.sizePP;
		data.pc.countCurr = (data.pc.sizeP - 1) * (data.pc.sizePP - 1);
		data.pc.phase = 0;

		std::vector<float> k1(2 * data.pc.countVolt);
		std::vector<float> k2(2 * data.pc.countVolt, 0.0f);
		for (uint32_t i = 0; i < data.pc.sizeP; ++i)
		{
			for (uint32_t j = 0; j < data.pc.sizePP; ++j)
			{
				const uint32_t idx = i * data.pc.sizePP + j;
				k1[idx] = sheet->m_K1_nyP(i,j);
				k1[data.pc.countVolt + idx] = sheet->m_K1_nyPP(i,j);
				if (data.superAbsorption)
				{
					k2[idx] = sheet->m_K2_nyP(i,j);
					k2[data.pc.countVolt + idx] = sheet->m_K2_nyPP(i,j);
				}
			}
		}

		const VkDeviceSize coeffSize = k1.size() * sizeof(float);
		const VkDeviceSize voltStateSize = 2 * (VkDeviceSize)data.pc.countVolt * sizeof(float);
		const VkDeviceSize currStateSize = 2 * (VkDeviceSize)data.pc.countCurr * sizeof(float);
		CreateGpuBuf(data.k1, coeffSize, usage, deviceLocal);
		CreateGpuBuf(data.k2, coeffSize, usage, deviceLocal);
		UploadGpuBuf(data.k1, k1.data(), coeffSize);
		UploadGpuBuf(data.k2, k2.data(), coeffSize);
		std::vector<float> voltZeros(2 * data.pc.countVolt, 0.0f);
		std::vector<float> currZeros(2 * data.pc.countCurr, 0.0f);
		CreateGpuBuf(data.voltState, voltStateSize, usage, deviceLocal);
		CreateGpuBuf(data.currState, currStateSize, usage, deviceLocal);
		UploadGpuBuf(data.voltState, voltZeros.data(), voltStateSize);
		UploadGpuBuf(data.currState, currZeros.data(), currStateSize);

		VkDescriptorSetLayout layouts[2] = {m_localAbcDescLayout, m_localAbcDescLayout};
		VkDescriptorSet sets[2];
		VkDescriptorSetAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		ai.descriptorPool = m_localAbcDescPool;
		ai.descriptorSetCount = 2;
		ai.pSetLayouts = layouts;
		VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, sets));
		data.voltDesc = sets[0]; data.currDesc = sets[1];
		VkBuffer voltBufs[4] = {m_voltBuf, data.voltState.buffer, data.k1.buffer, data.k2.buffer};
		VkDeviceSize voltSizes[4] = {m_fieldBufSize, voltStateSize, coeffSize, coeffSize};
		WriteDescriptorBuffers(m_device, data.voltDesc, voltBufs, voltSizes, 4);
		VkBuffer currBufs[4] = {m_currBuf, data.currState.buffer, data.k1.buffer, data.k2.buffer};
		VkDeviceSize currSizes[4] = {m_fieldBufSize, currStateSize, coeffSize, coeffSize};
		WriteDescriptorBuffers(m_device, data.currDesc, currBufs, currSizes, 4);
		m_gpuLocalABC.push_back(std::move(data));
	}
}

void Engine_Vulkan::RecordLocalABCPhase(VkCommandBuffer cmd, uint32_t phase) const
{
	if (!m_localAbcPipeline || m_gpuLocalABC.empty()) return;
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_localAbcPipeline);
	bool dispatched = false;
	for (const GpuLocalABCData& data : m_gpuLocalABC)
	{
		if (phase >= 3 && !data.superAbsorption) continue;
		LocalAbcPC pc = data.pc;
		pc.phase = phase;
		const uint32_t count = phase >= 3 ? pc.countCurr : pc.countVolt;
		const VkDescriptorSet desc = phase >= 3 ? data.currDesc : data.voltDesc;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_localAbcPipeLayout, 0, 1, &desc, 0, nullptr);
		vkCmdPushConstants(cmd, m_localAbcPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(LocalAbcPC), &pc);
		vkCmdDispatch(cmd, (count + 255) / 256, 1, 1);
		dispatched = true;
	}
	if (dispatched)
	{
		VkMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
		                     1, &barrier, 0, nullptr, 0, nullptr);
	}
}

void Engine_Vulkan::RecordSteadyStateSample(VkCommandBuffer cmd, uint32_t ts) const
{
	if (!m_steadyPipeline) return;
	const SteadyPC pc = {m_steadyProbeCount, m_steadyRingSize, ts};
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_steadyPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
	                        m_steadyPipeLayout, 0, 1, &m_steadyDescSet, 0, nullptr);
	vkCmdPushConstants(cmd, m_steadyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(SteadyPC), &pc);
	vkCmdDispatch(cmd, (m_steadyProbeCount + 255) / 256, 1, 1);

	VkMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
	                     1, &barrier, 0, nullptr, 0, nullptr);
}

void Engine_Vulkan::UpdateSteadyStateResult()
{
	if (!m_gpuSteadyStateExt || !m_steadyHistoryMapped || m_steadyPeriod == 0)
		return;
	const uint32_t completedPeriods = numTS / m_steadyPeriod;
	if (completedPeriods < 2 || completedPeriods == m_steadyLastCompletedPeriods)
		return;
	DrainGPU();
	const double energy = CalcFastEnergyGPU();
	m_gpuSteadyStateExt->UpdateGpuSamples(m_steadyHistoryMapped, 3, numTS, energy);
	m_steadyLastCompletedPeriods = completedPeriods;
}

void Engine_Vulkan::SetupGPU_Probes()
{
	// Placeholder — actual probe setup happens in SetupProbeCache()
	// after the initial PA->Process() recording run.
}

void Engine_Vulkan::SetupProbeCache(ProcessingArray* /*PA*/)
{
	// Finalize probe recording: stop recording mode and build GPU resources
	// from the cell accesses captured during the initial PA->Process().
	m_recordingProbeAccess = false;

	// Probe resources are used even when there are no other GPU extensions.
	// In that case SetupGPUExtensions() may have skipped creating layouts.
	if (m_probeDescLayout == VK_NULL_HANDLE)
	{
		VkDescriptorSetLayoutBinding bindings[5];
		for (uint32_t i = 0; i < 5; ++i)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = 5;
		ci.pBindings = bindings;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_probeDescLayout));
	}

	if (m_recordedCells.empty())
	{
		cerr << "[Vulkan] No probe cell accesses recorded — pipelining disabled." << endl;
		m_hasGPU_Probes = false;
		return;
	}

	// Deduplicate recorded cells
	std::sort(m_recordedCells.begin(), m_recordedCells.end());
	m_recordedCells.erase(std::unique(m_recordedCells.begin(), m_recordedCells.end()),
	                      m_recordedCells.end());

	m_probeCacheCount = (uint32_t)m_recordedCells.size();
	cerr << "[Vulkan] Probe cache: " << m_probeCacheCount << " unique cells recorded." << endl;

	// Build CPU-side cache keys: sorted vector (already sorted by dedup step above)
	// Index into m_probeCacheCPU is simply the position in the sorted vector.
	m_probeCacheKeys.resize(m_probeCacheCount);
	m_probeCacheHits = 0;
	m_probeCacheMisses = 0;
	std::vector<uint32_t> probeIndices(m_probeCacheCount);
	std::vector<uint32_t> probeFieldSel(m_probeCacheCount);

	for (uint32_t i = 0; i < m_probeCacheCount; ++i)
	{
		uint64_t cell = m_recordedCells[i];
		m_probeCacheKeys[i] = cell;
		uint32_t fieldSel = (uint32_t)(cell >> 48);
		uint32_t linearIdx = (uint32_t)(cell & 0x0000FFFFFFFFFFFF);
		probeIndices[i]  = linearIdx;
		probeFieldSel[i] = fieldSel;
	}

	m_probeCacheCPU.resize(m_probeCacheCount, 0.0f);

	// Free the recording vector
	m_recordedCells.clear();
	m_recordedCells.shrink_to_fit();

	// Create GPU buffers for the gather shader
	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VkDeviceSize idxSize = m_probeCacheCount * sizeof(uint32_t);
	VkDeviceSize outSize = m_probeCacheCount * sizeof(float);

	// Upload probe index buffer (read-only on GPU)
	CreateGpuBuf(m_probeIdxBuf, idxSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dLoc);
	UploadGpuBuf(m_probeIdxBuf, probeIndices.data(), idxSize);

	// Upload field selector buffer (read-only on GPU)
	CreateGpuBuf(m_probeSelBuf, idxSize, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, dLoc);
	UploadGpuBuf(m_probeSelBuf, probeFieldSel.data(), idxSize);

	// Probe results are compact. Let the GPU write them directly to coherent
	// host-visible memory instead of coupling readback to the field allocation.
	CreateGpuBuf(m_probeOutBuf, outSize, usage, hostVisible);
	VK_CHECK(vkMapMemory(m_device, m_probeOutBuf.memory, 0, outSize, 0,
	                     (void**)&m_probeOutMapped));

	// Create probe gather pipeline (re-use existing m_probeDescLayout with 5 bindings)
	// Descriptor pool for 1 set, 5 storage buffer descriptors
	VkDescriptorPoolSize poolSz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets       = 1;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes    = &poolSz;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_probeDescPool));

	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_probeDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts        = &m_probeDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &m_probeGatherDescSet));

	// Write descriptors: volt, curr, probeIdx, fieldSel, probeOut
	VkBuffer bufs[5] = {m_voltBuf, m_currBuf, m_probeIdxBuf.buffer,
	                     m_probeSelBuf.buffer, m_probeOutBuf.buffer};
	VkDeviceSize sizes[5] = {m_fieldBufSize, m_fieldBufSize, idxSize, idxSize, outSize};
	WriteDescriptorBuffers(m_device, m_probeGatherDescSet, bufs, sizes, 5);

	// Create the probe gather compute pipeline using m_probePipeLayout
	// The push constant is a single uint (count)
	if (m_probePipeLayout == VK_NULL_HANDLE)
	{
		VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
		VkPipelineLayoutCreateInfo pli{};
		pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pli.setLayoutCount         = 1;
		pli.pSetLayouts            = &m_probeDescLayout;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges    = &pcRange;
		VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_probePipeLayout));
	}

	VkShaderModule mod = CreateShaderModule(gpu_spirv::gather_probes_data,
	                                        gpu_spirv::gather_probes_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName  = "main";

	VkComputePipelineCreateInfo ci{};
	ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage  = stage;
	ci.layout = m_probePipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_probePipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

	m_hasGPU_Probes = true;
	cerr << "[Vulkan] Probe gather pipeline created (" << m_probeCacheCount
	     << " cells, host-visible direct readback)." << endl;
}

void Engine_Vulkan::SnapshotProbeCache()
{
	if (!m_hasGPU_Probes || m_probeCacheCount == 0) return;
	if (m_hasDedicatedTransferQueue && m_probeTransferInFlight[m_lastProbeSubmitSlot])
	{
		vkWaitForFences(m_device, 1, &m_transferFences[m_lastProbeSubmitSlot], VK_TRUE, UINT64_MAX);
		m_probeTransferInFlight[m_lastProbeSubmitSlot] = false;
	}

	if (m_probeOutMapped)
	{
		// ReBAR: direct copy from mapped VRAM
		memcpy(m_probeCacheCPU.data(), m_probeOutMapped,
		       m_probeCacheCount * sizeof(float));
	}
	else if (m_probeReadbackMapped[m_lastProbeSubmitSlot])
	{
		// Readback ring: data was copied in the main command buffer for this slot.
		memcpy(m_probeCacheCPU.data(), m_probeReadbackMapped[m_lastProbeSubmitSlot],
		       m_probeCacheCount * sizeof(float));
	}
	else
	{
		// No ReBAR: copy via staging buffer with a transfer command
		VkDeviceSize sz = m_probeCacheCount * sizeof(float);
		RunSingleCommand([&](VkCommandBuffer cmd)
		{
			VkBufferCopy region = {0, 0, sz};
			vkCmdCopyBuffer(cmd, m_probeOutBuf.buffer,
			                m_probeStagingBuf.buffer, 1, &region);
		});

		void* mapped;
		VK_CHECK(vkMapMemory(m_device, m_probeStagingBuf.memory, 0, sz, 0, &mapped));
		memcpy(m_probeCacheCPU.data(), mapped, sz);
		vkUnmapMemory(m_device, m_probeStagingBuf.memory);
	}
}

void Engine_Vulkan::SubmitSpeculative(unsigned int nTS)
{
	// Submit a batch of nTS timesteps without advancing the public numTS.
	// We save/restore numTS around the call to IterateTS so the engine
	// external state doesn't change until CommitSpeculative().
	m_speculativeTS = nTS;
	unsigned int savedTS = numTS;
	IterateTS(nTS);      // internally advances numTS
	numTS = savedTS;     // revert public numTS
}

void Engine_Vulkan::CommitSpeculative()
{
	if (m_speculativeTS == 0) return;
	numTS += m_speculativeTS;
	m_speculativeTS = 0;
}

// ===========================================================================
// Record extension shader dispatches into command buffer
// ===========================================================================

void Engine_Vulkan::RecordVoltageExtensions(VkCommandBuffer cmd, uint32_t ts) const
{
	VkMemoryBarrier barrier{};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	bool dispatchedPreVolt = false;

	// --- Pre-voltage: UPML pre-voltage, Mur pre-voltage ---
	// UPML pre-voltage (highest priority, runs first)
	if (!m_gpuUPML.empty())
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPreVoltPipeline);
	for (const auto& u : m_gpuUPML)
	{
		uint32_t groups = (u.totalCells + 255) / 256;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_upmlPrePipeLayout, 0, 1, &u.preVoltDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_upmlPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PmlPC), &u.pc);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreVolt = true;
	}

	// Mur pre-voltage (saves boundary values before Yee update)
	if (!m_gpuMur.empty())
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murPreVoltPipeline);
	for (const auto& m : m_gpuMur)
	{
		if (ts < m.startTS) continue;  // Mur not yet active
		uint32_t groups = (m.totalCells + 255) / 256;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_murUpdatePipeLayout, 0, 1, &m.preDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_murUpdatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreVolt = true;
	}

	// RLC pre-voltage (inductor current update + history rotation)
	if (m_hasGPU_RLC)
	{
		RlcPC pc = {m_gpuRLC.count};
		uint32_t groups = (m_gpuRLC.count + 255) / 256;
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcPreVoltPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_rlcPrePipeLayout, 0, 1, &m_gpuRLC.preDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_rlcPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreVolt = dispatchedPreVolt || (groups > 0);
	}

	// Dispersive pre-voltage (ADE update before Yee)
	if (!m_gpuDisp.empty())
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPreVoltPipeline);
	for (const auto& d : m_gpuDisp)
	{
		if (!d.voltADEOn) continue;
		uint32_t groups = (d.count + 255) / 256;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_dispPrePipeLayout, 0, 1, &d.preVoltDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_dispPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &d.voltPC);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreVolt = true;
	}
	RecordLocalABCPhase(cmd, 0);

	// Barrier: all pre-voltage writes must complete before Yee update
	if (dispatchedPreVolt)
	{
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0, 1, &barrier, 0, nullptr, 0, nullptr);
	}
}

void Engine_Vulkan::RecordCurrentExtensions(VkCommandBuffer cmd, uint32_t ts) const
{
	(void)ts;
	VkMemoryBarrier barrier{};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	bool dispatchedPreCurr = false;

	// --- Pre-current: UPML pre-current ---
	if (!m_gpuUPML.empty())
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPreCurrPipeline);
	for (const auto& u : m_gpuUPML)
	{
		uint32_t groups = (u.totalCells + 255) / 256;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_upmlPrePipeLayout, 0, 1, &u.preCurrDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_upmlPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PmlPC), &u.pc);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreCurr = true;
	}

	// Dispersive pre-current
	if (!m_gpuDisp.empty())
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPreCurrPipeline);
	for (const auto& d : m_gpuDisp)
	{
		if (!d.currADEOn) continue;
		uint32_t groups = (d.count + 255) / 256;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_dispPrePipeLayout, 0, 1, &d.preCurrDesc, 0, nullptr);
		vkCmdPushConstants(cmd, m_dispPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &d.currPC);
		vkCmdDispatch(cmd, groups, 1, 1);
		dispatchedPreCurr = true;
	}
	RecordLocalABCPhase(cmd, 3);

	if (dispatchedPreCurr)
	{
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0, 1, &barrier, 0, nullptr, 0, nullptr);
	}
}

void Engine_Vulkan::FlushGPU() const
{
	DrainGPU();
}

void Engine_Vulkan::DrainGPU() const
{
	if (m_gpuDrained) return;  // Already drained — skip redundant fence syscalls
	for (int i = 0; i < CMD_RING_SIZE; ++i)
	{
		if (m_gpuInFlight[i])
		{
			auto t0 = std::chrono::steady_clock::now();
			vkWaitForFences(m_device, 1, &m_fences[i], VK_TRUE, UINT64_MAX);
			auto t1 = std::chrono::steady_clock::now();
			if (m_profileEnabled)
			{
				m_profileWaitMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
				CollectProfileForSlot(i);
			}
			m_gpuInFlight[i] = false;
		}
		if (m_probeTransferInFlight[i])
		{
			vkWaitForFences(m_device, 1, &m_transferFences[i], VK_TRUE, UINT64_MAX);
			m_probeTransferInFlight[i] = false;
		}
	}
	m_gpuDrained = true;
}

void Engine_Vulkan::SyncFieldsToHost() const
{
	if (m_dumpAsyncPending)
	{
		// A BeginAsyncFieldDownload() request is already in flight (or done) on
		// the dump queue -- just wait for it and copy into the CPU shadow, no
		// need to also drain/re-copy on the compute queue.
		const_cast<Engine_Vulkan*>(this)->FinishAsyncFieldDownload();
		return;
	}
	DrainGPU();  // ensure all pending GPU work is complete
	if (!m_hostDirty) return;
	if (m_voltMapped && m_currMapped)
	{
		std::memcpy(volt_ptr->data(), m_voltMapped, m_fieldBufSize);
		std::memcpy(curr_ptr->data(), m_currMapped, m_fieldBufSize);
	}
	else
	{
		const_cast<Engine_Vulkan*>(this)->DownloadFromDeviceBuffer(
			m_voltBuf, volt_ptr->data(), m_fieldBufSize);
		const_cast<Engine_Vulkan*>(this)->DownloadFromDeviceBuffer(
			m_currBuf, curr_ptr->data(), m_fieldBufSize);
	}
	m_hostDirty = false;
}

void Engine_Vulkan::SyncFieldsToDevice()
{
	DrainGPU();  // must complete any pending GPU work before overwriting
	if (!m_deviceDirty) return;
	if (m_voltMapped && m_currMapped)
	{
		std::memcpy(m_voltMapped, volt_ptr->data(), m_fieldBufSize);
		std::memcpy(m_currMapped, curr_ptr->data(), m_fieldBufSize);
	}
	else
	{
		UploadToDeviceBuffer(m_voltBuf, volt_ptr->data(), m_fieldBufSize);
		UploadToDeviceBuffer(m_currBuf, curr_ptr->data(), m_fieldBufSize);
	}
	m_deviceDirty = false;
}

void Engine_Vulkan::EnsureDumpBuffers()
{
	if (m_dumpBuffersReady || !m_hasDumpQueue) return;
	// ReBAR fields are already directly host-mapped -- the async pipeline's
	// whole point is to hide a staging-buffer PCIe copy, so there is nothing
	// to gain here (and DrainGPU() + memcpy is already fast in that case).
	if (m_voltMapped && m_currMapped) { m_hasDumpQueue = false; return; }

	VkBufferUsageFlags snapUsage  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags dLoc    = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VkMemoryPropertyFlags host    = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	// HOST_VISIBLE|HOST_COHERENT alone is satisfied by write-combined memory
	// on this driver -- fine for the CPU *writing* into GPU-visible memory
	// (the upload path elsewhere), catastrophic for *reading* it back: a
	// plain memcpy() out of uncached mapped memory measured ~260 MB/s here
	// (2+ seconds for a 564 MB dump) vs. the ~26 ms the underlying PCIe copy
	// itself takes. HOST_CACHED turns that memcpy into an ordinary cached
	// read. Try it first; fall back to the plain flags on drivers that don't
	// expose a cached host-visible type.
	VkMemoryPropertyFlags hostCached = host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

	uint32_t sharingFamilies[2] = {m_computeQueueFamily, m_dumpQueueFamily};

	auto createStaging = [&](GpuBuf& gb) -> bool
	{
		return TryCreateBufferWithMemory(m_fieldBufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		                                  hostCached, gb.buffer, gb.memory) ||
		       TryCreateBufferWithMemory(m_fieldBufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		                                  host, gb.buffer, gb.memory);
	};

	auto createConcurrent = [&](GpuBuf& gb, VkDeviceSize size) -> bool
	{
		VkBufferCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		ci.size  = size;
		ci.usage = snapUsage;
		ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
		ci.queueFamilyIndexCount = 2;
		ci.pQueueFamilyIndices   = sharingFamilies;
		if (vkCreateBuffer(m_device, &ci, nullptr, &gb.buffer) != VK_SUCCESS)
			return false;
		VkMemoryRequirements memReq;
		vkGetBufferMemoryRequirements(m_device, gb.buffer, &memReq);
		uint32_t memType = FindMemoryTypeSoft(memReq.memoryTypeBits, dLoc);
		if (memType == UINT32_MAX)
		{
			vkDestroyBuffer(m_device, gb.buffer, nullptr);
			gb.buffer = VK_NULL_HANDLE;
			return false;
		}
		VkMemoryAllocateInfo ai{};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize  = memReq.size;
		ai.memoryTypeIndex = memType;
		if (vkAllocateMemory(m_device, &ai, nullptr, &gb.memory) != VK_SUCCESS)
		{
			vkDestroyBuffer(m_device, gb.buffer, nullptr);
			gb.buffer = VK_NULL_HANDLE;
			return false;
		}
		vkBindBufferMemory(m_device, gb.buffer, gb.memory, 0);
		return true;
	};

	bool ok = true;
	for (int i = 0; i < DUMP_RING && ok; ++i)
	{
		ok = ok && createConcurrent(m_dumpSnapVolt[i], m_fieldBufSize);
		ok = ok && createConcurrent(m_dumpSnapCurr[i], m_fieldBufSize);
		if (!ok) break;

		// Staging is only ever touched by m_dumpQueue (writer) and the host
		// (reader), so it can stay VK_SHARING_MODE_EXCLUSIVE.
		ok = ok && createStaging(m_dumpStageVolt[i]);
		ok = ok && createStaging(m_dumpStageCurr[i]);
		if (!ok) break;

		if (vkMapMemory(m_device, m_dumpStageVolt[i].memory, 0, m_fieldBufSize, 0,
		                 (void**)&m_dumpStageVoltMapped[i]) != VK_SUCCESS) { ok = false; break; }
		if (vkMapMemory(m_device, m_dumpStageCurr[i].memory, 0, m_fieldBufSize, 0,
		                 (void**)&m_dumpStageCurrMapped[i]) != VK_SUCCESS) { ok = false; break; }

		VkCommandBufferAllocateInfo snapAlloc{};
		snapAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		snapAlloc.commandPool = m_dumpCmdPool;
		snapAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		snapAlloc.commandBufferCount = 1;
		if (vkAllocateCommandBuffers(m_device, &snapAlloc, &m_dumpSnapshotCmdBuf[i]) != VK_SUCCESS) { ok = false; break; }

		VkCommandBufferAllocateInfo xferAlloc = snapAlloc;
		xferAlloc.commandPool = m_dumpXferCmdPool;
		if (vkAllocateCommandBuffers(m_device, &xferAlloc, &m_dumpStagingCmdBuf[i]) != VK_SUCCESS) { ok = false; break; }

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signaled: slot is initially idle
		if (vkCreateFence(m_device, &fenceInfo, nullptr, &m_dumpSnapshotFence[i]) != VK_SUCCESS) { ok = false; break; }
		if (vkCreateFence(m_device, &fenceInfo, nullptr, &m_dumpStagingFence[i]) != VK_SUCCESS) { ok = false; break; }

		VkSemaphoreCreateInfo semInfo{};
		semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		if (vkCreateSemaphore(m_device, &semInfo, nullptr, &m_dumpSnapshotSem[i]) != VK_SUCCESS) { ok = false; break; }
	}

	if (!ok)
	{
		cerr << "Engine_Vulkan: dump-queue buffer setup failed, falling back to "
		        "synchronous field-dump downloads." << endl;
		CleanupDumpAsync();
		m_hasDumpQueue = false;
		return;
	}
	m_dumpBuffersReady = true;
}

void Engine_Vulkan::BeginAsyncFieldDownload()
{
	// Flush any request from a previous call that nothing has consumed yet,
	// so the data SyncFieldsToHost() eventually returns is never more than
	// one BeginAsyncFieldDownload() cycle stale.
	if (m_dumpAsyncPending)
		FinishAsyncFieldDownload();

	if (!m_hostDirty) return;              // nothing new on the GPU side
	if (!m_hasDumpQueue) return;            // no independent queue -- caller falls back
	EnsureDumpBuffers();
	if (!m_dumpBuffersReady) return;

	DrainGPU();  // must see the fully-committed state before snapshotting it

	const int idx = m_dumpNextIdx;
	m_dumpNextIdx = (m_dumpNextIdx + 1) % DUMP_RING;

	// Slot reuse safety: with the flush-before-new-request rule above, at
	// most one slot is ever genuinely in flight, but wait defensively in
	// case a caller ever bypasses BeginAsyncFieldDownload's own pending flush.
	vkWaitForFences(m_device, 1, &m_dumpSnapshotFence[idx], VK_TRUE, UINT64_MAX);
	vkWaitForFences(m_device, 1, &m_dumpStagingFence[idx], VK_TRUE, UINT64_MAX);
	vkResetFences(m_device, 1, &m_dumpSnapshotFence[idx]);
	vkResetFences(m_device, 1, &m_dumpStagingFence[idx]);

	// --- Stage 1 (compute queue, family == field buffers' family): ---
	// live volt/curr -> device-local snapshot. Fast, VRAM-bandwidth bound.
	// Signals m_dumpSnapshotSem[idx] so stage 2 can start without a CPU wait.
	VkCommandBuffer snapCmd = m_dumpSnapshotCmdBuf[idx];
	VkCommandBufferBeginInfo bi{};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkResetCommandBuffer(snapCmd, 0);
	VK_CHECK(vkBeginCommandBuffer(snapCmd, &bi));
	VkBufferCopy region = {0, 0, m_fieldBufSize};
	vkCmdCopyBuffer(snapCmd, m_voltBuf, m_dumpSnapVolt[idx].buffer, 1, &region);
	vkCmdCopyBuffer(snapCmd, m_currBuf, m_dumpSnapCurr[idx].buffer, 1, &region);
	VK_CHECK(vkEndCommandBuffer(snapCmd));

	VkSubmitInfo snapSubmit{};
	snapSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	snapSubmit.commandBufferCount = 1;
	snapSubmit.pCommandBuffers = &snapCmd;
	snapSubmit.signalSemaphoreCount = 1;
	snapSubmit.pSignalSemaphores = &m_dumpSnapshotSem[idx];
	VK_CHECK(vkQueueSubmit(m_computeQueue, 1, &snapSubmit, m_dumpSnapshotFence[idx]));

	// --- Stage 2 (independent dump queue): snapshot -> host-visible staging. ---
	// Slow (PCIe-bound) but runs fully concurrently with whatever the caller
	// submits to m_computeQueue next, since the source is already frozen.
	VkCommandBuffer xferCmd = m_dumpStagingCmdBuf[idx];
	vkResetCommandBuffer(xferCmd, 0);
	VK_CHECK(vkBeginCommandBuffer(xferCmd, &bi));
	vkCmdCopyBuffer(xferCmd, m_dumpSnapVolt[idx].buffer, m_dumpStageVolt[idx].buffer, 1, &region);
	vkCmdCopyBuffer(xferCmd, m_dumpSnapCurr[idx].buffer, m_dumpStageCurr[idx].buffer, 1, &region);
	VK_CHECK(vkEndCommandBuffer(xferCmd));

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	VkSubmitInfo xferSubmit{};
	xferSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	xferSubmit.waitSemaphoreCount = 1;
	xferSubmit.pWaitSemaphores = &m_dumpSnapshotSem[idx];
	xferSubmit.pWaitDstStageMask = &waitStage;
	xferSubmit.commandBufferCount = 1;
	xferSubmit.pCommandBuffers = &xferCmd;
	VK_CHECK(vkQueueSubmit(m_dumpQueue, 1, &xferSubmit, m_dumpStagingFence[idx]));

	m_dumpSlotInFlight[idx] = true;
	m_dumpAsyncPending = true;
	m_dumpAsyncIdx = idx;
}

void Engine_Vulkan::FinishAsyncFieldDownload()
{
	if (!m_dumpAsyncPending)
	{
		// Nothing was pre-fetched (e.g. HasDumpTransferQueue() was false) --
		// fall back to the original fully-synchronous path.
		DrainGPU();
		if (!m_hostDirty) return;
		if (m_voltMapped && m_currMapped)
		{
			std::memcpy(volt_ptr->data(), m_voltMapped, m_fieldBufSize);
			std::memcpy(curr_ptr->data(), m_currMapped, m_fieldBufSize);
		}
		else
		{
			DownloadFromDeviceBuffer(m_voltBuf, volt_ptr->data(), m_fieldBufSize);
			DownloadFromDeviceBuffer(m_currBuf, curr_ptr->data(), m_fieldBufSize);
		}
		m_hostDirty = false;
		return;
	}

	const int idx = m_dumpAsyncIdx;
	vkWaitForFences(m_device, 1, &m_dumpStagingFence[idx], VK_TRUE, UINT64_MAX);
	m_dumpSlotInFlight[idx] = false;
	std::memcpy(volt_ptr->data(), m_dumpStageVoltMapped[idx], m_fieldBufSize);
	std::memcpy(curr_ptr->data(), m_dumpStageCurrMapped[idx], m_fieldBufSize);
	m_hostDirty = false;
	m_dumpAsyncPending = false;
}

void Engine_Vulkan::CleanupDumpAsync()
{
	if (m_device == VK_NULL_HANDLE) return;
	for (int i = 0; i < DUMP_RING; ++i)
	{
		if (m_dumpSlotInFlight[i])
			vkWaitForFences(m_device, 1, &m_dumpStagingFence[i], VK_TRUE, UINT64_MAX);
		if (m_dumpStageVoltMapped[i]) { vkUnmapMemory(m_device, m_dumpStageVolt[i].memory); m_dumpStageVoltMapped[i] = nullptr; }
		if (m_dumpStageCurrMapped[i]) { vkUnmapMemory(m_device, m_dumpStageCurr[i].memory); m_dumpStageCurrMapped[i] = nullptr; }
		DestroyGpuBuf(m_dumpSnapVolt[i]);
		DestroyGpuBuf(m_dumpSnapCurr[i]);
		DestroyGpuBuf(m_dumpStageVolt[i]);
		DestroyGpuBuf(m_dumpStageCurr[i]);
		if (m_dumpSnapshotSem[i]) { vkDestroySemaphore(m_device, m_dumpSnapshotSem[i], nullptr); m_dumpSnapshotSem[i] = VK_NULL_HANDLE; }
		if (m_dumpSnapshotFence[i]) { vkDestroyFence(m_device, m_dumpSnapshotFence[i], nullptr); m_dumpSnapshotFence[i] = VK_NULL_HANDLE; }
		if (m_dumpStagingFence[i]) { vkDestroyFence(m_device, m_dumpStagingFence[i], nullptr); m_dumpStagingFence[i] = VK_NULL_HANDLE; }
		m_dumpSnapshotCmdBuf[i] = VK_NULL_HANDLE; // freed with their pool below
		m_dumpStagingCmdBuf[i]  = VK_NULL_HANDLE;
	}
	if (m_dumpCmdPool) { vkDestroyCommandPool(m_device, m_dumpCmdPool, nullptr); m_dumpCmdPool = VK_NULL_HANDLE; }
	if (m_dumpXferCmdPool) { vkDestroyCommandPool(m_device, m_dumpXferCmdPool, nullptr); m_dumpXferCmdPool = VK_NULL_HANDLE; }
	m_dumpBuffersReady = false;
	m_dumpAsyncPending = false;
}

void Engine_Vulkan::ValidateGPUFields(unsigned int timestep) const
{
	const uint32_t invalid = std::numeric_limits<uint32_t>::max();
	std::fill_n(m_validateBadMapped, 6, invalid);
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		RecordGPUFieldValidation(cmd, 0);
		VkMemoryBarrier hostBarrier{};
		hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
			0, 1, &hostBarrier, 0, nullptr, 0, nullptr);
	});

	for (unsigned int component = 0; component < 6; ++component)
	{
		const uint32_t index = m_validateBadMapped[component];
		if (index == invalid)
			continue;

		const uint32_t x = index / m_strideYZ;
		const uint32_t rem = index % m_strideYZ;
		const uint32_t y = rem / numLines[2];
		const uint32_t z = rem % numLines[2];
		const bool voltage = component < 3;
		const unsigned int fieldComponent = voltage ? component : component - 3;
		throw std::runtime_error(
			std::string("Engine_Vulkan: non-finite ") +
			(voltage ? "voltage" : "current") + " field at timestep " +
			std::to_string(timestep) + " component " +
			std::to_string(fieldComponent) + " [" + std::to_string(x) + "," +
			std::to_string(y) + "," + std::to_string(z) + "]");
	}
}

void Engine_Vulkan::RecordGPUFieldValidation(VkCommandBuffer cmd, uint32_t outputBase) const
{
	const auto* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
	const uint32_t numComp = opVk ? opVk->GetNumCompressed() : 0u;
	ValidatePC pc = {m_fieldN, outputBase, m_validateTraceIndex, numComp,
	                 numLines[1], numLines[2], m_validateMagnitude ? 1u : 0u};
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_validatePipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
	                        m_validatePipeLayout, 0, 1, &m_validateDescSet, 0, nullptr);
	vkCmdPushConstants(cmd, m_validatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(ValidatePC), &pc);
	vkCmdDispatch(cmd, (m_fieldN + 255) / 256, 1, 1);

	VkMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Engine_Vulkan::ReportGPUFieldValidation(unsigned int timestep) const
{
	DrainGPU();
	const uint32_t invalid = std::numeric_limits<uint32_t>::max();
	static constexpr const char* stages[] = {
		"after voltage pre-extensions", "after voltage update",
		"after voltage extensions", "after current pre-extensions",
		"after current update", "after current extensions"};
	auto traceFloat = [](uint32_t bits) {
		float value;
		std::memcpy(&value, &bits, sizeof(value));
		return value;
	};

	for (unsigned int stage = 0; stage < 6; ++stage)
	{
		if (m_validateTraceIndex != invalid)
		{
			static constexpr const char* stages[] = {
				"after voltage pre-extensions", "after voltage update",
				"after voltage extensions", "after current pre-extensions",
				"after current update", "after current extensions"};
			const uint32_t* trace = m_validateBadMapped + stage * 16 + 6;
			cout << "Engine_Vulkan: trace timestep=" << timestep
			     << " stage=" << stages[stage]
			     << " V0=" << traceFloat(trace[0])
			     << " Cz=" << traceFloat(trace[1])
			     << " Cz_m=" << traceFloat(trace[2])
			     << " Cy=" << traceFloat(trace[3])
			     << " Cy_zm=" << traceFloat(trace[4])
			     << " curl0=" << traceFloat(trace[5])
			     << " op=" << trace[6]
			     << " vv0=" << traceFloat(trace[7])
			     << " vi0=" << traceFloat(trace[8])
			     << " result=" << traceFloat(trace[9]) << endl;
		}
		for (unsigned int component = 0; component < 6; ++component)
		{
			const uint32_t index = m_validateBadMapped[stage * 16 + component];
			if (index == invalid)
				continue;

			const uint32_t x = index / m_strideYZ;
			const uint32_t rem = index % m_strideYZ;
			const uint32_t y = rem / numLines[2];
			const uint32_t z = rem % numLines[2];
			const bool voltage = component < 3;
			const unsigned int fieldComponent = voltage ? component : component - 3;
			throw std::runtime_error(
				std::string("Engine_Vulkan: ") +
				(m_validateMagnitude ? "non-finite or excessive-magnitude " : "non-finite ") +
				(voltage ? "voltage" : "current") + " field " + stages[stage] +
				" at timestep " + std::to_string(timestep) + " component " +
				std::to_string(fieldComponent) + " [" + std::to_string(x) + "," +
				std::to_string(y) + "," + std::to_string(z) + "]");
		}
	}
}

// ===========================================================================
// IterateTS — the main time-stepping loop
// ===========================================================================

bool Engine_Vulkan::IterateTS(unsigned int iterTS)
{
	if (iterTS == 0) return true;

	// On some drivers (notably RADV/AMDGPU), very long command buffers can
	// trigger GPU watchdog resets. Split pure-GPU batches into smaller submits.
	if (!m_hasCPUExtensions)
	{
		if (iterTS > m_maxTSPerSubmit)
		{
			if (g_settings.GetVerboseLevel() > 1)
				cout << "Engine_Vulkan: splitting " << iterTS
				     << " timesteps into chunks of " << m_maxTSPerSubmit
				     << " (progress every ~" << m_chunkProgressIntervalSec << " s)"
				     << endl;

			unsigned int submitted = 0;
			unsigned int remaining = iterTS;
			while (remaining > 0)
			{
				unsigned int chunk = std::min(remaining, m_maxTSPerSubmit);
				if (!IterateTS(chunk))
					return false;
				remaining -= chunk;
				submitted += chunk;

				if (g_settings.GetVerboseLevel() > 0)
				{
					auto now = std::chrono::steady_clock::now();
					bool printNow = !m_chunkProgressHasLastPrint;
					if (!printNow)
					{
						double dt = std::chrono::duration<double>(now - m_chunkProgressLastPrint).count();
						printNow = (dt >= m_chunkProgressIntervalSec);
					}
					if (printNow)
					{
						unsigned int pct = (iterTS > 0)
						                 ? (unsigned int)((100ULL * submitted + (iterTS / 2)) / iterTS)
						                 : 100U;
						cout << "Engine_Vulkan: chunk progress " << submitted << "/" << iterTS
						     << " TS (" << pct << "%)";
						if (m_speculativeTS > 0)
							cout << " [speculative]";
						cout << endl;
						m_chunkProgressLastPrint = now;
						m_chunkProgressHasLastPrint = true;
					}
				}
			}
			return true;
		}
	}

	uint32_t N  = numLines[0] * numLines[1] * numLines[2];
	uint32_t cN = (numLines[0]-1) * (numLines[1]-1) * (numLines[2]-1);
	const Operator_Vulkan* opVk = dynamic_cast<const Operator_Vulkan*>(Op);
	GridPC gridPC = {numLines[0], numLines[1], numLines[2], opVk->GetNumCompressed()};
	FusedPC fusedPC = {numLines[0], numLines[1], numLines[2], opVk->GetNumCompressed(),
	                   m_hasFusedUPML ? (uint32_t)m_gpuUPML.size() : 0u};
	uint32_t voltGroups = (N  + 255) / 256;
	uint32_t currGroups = (cN + 255) / 256;
	uint32_t excVoltGroups = (m_excVoltCount + 255) / 256;
	uint32_t excCurrGroups = (m_excCurrCount + 255) / 256;
	if (m_validateStages)
		std::fill_n(m_validateBadMapped, 6 * 16, std::numeric_limits<uint32_t>::max());

	VkMemoryBarrier barrier{};
	barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

	if (!m_hasCPUExtensions)
	{
		// ========== Pure GPU path: fire-and-forget double-buffered ==========
		// Record FDTD compute into m_cmdBufs[m_cmdIdx], submit immediately,
		// flip to the other slot.  GetVolt/GetCurr calls DrainGPU() (usually
		// a no-op since the ~50 µs compute is already done) then reads
		// directly from the ReBAR-mapped VRAM pointer — zero copies.
		if (m_deviceDirty) SyncFieldsToDevice();

		bool hasExcVolt = m_hasGPUExcitation && m_excVoltCount > 0;
		bool hasExcCurr = m_hasGPUExcitation && m_excCurrCount > 0;
		bool hasDispVoltADE = false;
		bool hasDispCurrADE = false;
		for (const auto& d : m_gpuDisp)
		{
			hasDispVoltADE = hasDispVoltADE || d.voltADEOn;
			hasDispCurrADE = hasDispCurrADE || d.currADEOn;
			if (hasDispVoltADE && hasDispCurrADE) break;
		}
		bool needPostVoltBarrierFused = (m_hasGPU_Mur || m_hasGPU_TFSF || hasDispVoltADE || m_hasGPU_RLC);
		bool needPostCurrBarrierFused = (m_hasGPU_TFSF || hasDispCurrADE);
		bool needPostVoltBarrier = (m_hasGPU_UPML || m_hasGPU_Mur || m_hasGPU_TFSF || hasDispVoltADE || m_hasGPU_RLC);
		bool needPostCurrBarrier = (m_hasGPU_UPML || m_hasGPU_TFSF || hasDispCurrADE);
		bool hasVoltLateWritesFused = hasExcVolt || m_hasGPU_Mur || m_hasGPU_TFSF || hasDispVoltADE || m_hasGPU_RLC;
		bool hasCurrLateWritesFused = hasExcCurr || m_hasGPU_TFSF || hasDispCurrADE;
		bool hasVoltLateWrites = hasExcVolt || m_hasGPU_UPML || m_hasGPU_Mur || m_hasGPU_TFSF || hasDispVoltADE || m_hasGPU_RLC;
		bool hasCurrLateWrites = hasExcCurr || m_hasGPU_UPML || m_hasGPU_TFSF || hasDispCurrADE;
		bool useAsyncProbeCopy = m_hasDedicatedTransferQueue && m_hasGPU_Probes &&
		                         (m_probeCacheCount > 0) && !m_probeOutMapped &&
		                         (m_probeReadbackMapped[m_cmdIdx] != nullptr);

		// Cache TF/SF signal info (invariant across iterations)
		Excitation* tfsfExc = nullptr;
		int         tfsfSigPeriod = 0;
		uint32_t    tfsfSigLen = 0;
		if (m_hasGPU_TFSF)
		{
			tfsfExc = const_cast<Operator*>(Op)->GetExcitationSignal();
			if (tfsfExc)
			{
				tfsfSigLen = (uint32_t)tfsfExc->GetLength();
				tfsfSigPeriod = (tfsfExc->GetSignalPeriod() > 0)
				              ? (int)(tfsfExc->GetSignalPeriod() / tfsfExc->GetTimestep()) : 0;
			}
		}

		// Wait for this slot if it is still in-flight from a previous round
		if (m_gpuInFlight[m_cmdIdx])
		{
			auto t0 = std::chrono::steady_clock::now();
			vkWaitForFences(m_device, 1, &m_fences[m_cmdIdx], VK_TRUE, UINT64_MAX);
			auto t1 = std::chrono::steady_clock::now();
			if (m_profileEnabled)
			{
				m_profileWaitMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
				CollectProfileForSlot(m_cmdIdx);
			}
			m_gpuInFlight[m_cmdIdx] = false;
		}
		if (m_probeTransferInFlight[m_cmdIdx])
		{
			vkWaitForFences(m_device, 1, &m_transferFences[m_cmdIdx], VK_TRUE, UINT64_MAX);
			m_probeTransferInFlight[m_cmdIdx] = false;
		}

		// Begin recording
		VkCommandBuffer cmd = m_cmdBufs[m_cmdIdx];
		VkCommandBufferBeginInfo bi{};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkResetCommandBuffer(cmd, 0);
		VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
		uint32_t qBaseProfile = 0;

		if (useAsyncProbeCopy)
		{
			// Acquire probe output buffer ownership from transfer queue before compute writes.
			VkBufferMemoryBarrier ownAcquire{};
			ownAcquire.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			ownAcquire.srcAccessMask = 0;
			ownAcquire.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			ownAcquire.srcQueueFamilyIndex = m_transferQueueFamily;
			ownAcquire.dstQueueFamilyIndex = m_computeQueueFamily;
			ownAcquire.buffer = m_probeOutBuf.buffer;
			ownAcquire.offset = 0;
			ownAcquire.size = VK_WHOLE_SIZE;
			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0, 0, nullptr, 1, &ownAcquire, 0, nullptr);
		}

		if (m_profileEnabled && m_tsQueryPool)
		{
			qBaseProfile = (uint32_t)(m_cmdIdx * m_profileQueriesPerSlot);
			vkCmdResetQueryPool(cmd, m_tsQueryPool, qBaseProfile, m_profileQueriesPerSlot);
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_tsQueryPool, qBaseProfile);
		}

		// Append all FDTD timesteps
		for (unsigned int iter = 0; iter < iterTS; ++iter)
		{
			uint32_t ts = numTS + iter;

			if (m_hasFusedUPML)
			{
				// ============= FUSED Yee+UPML path =============
				// Single dispatch replaces: UPML pre-volt + Yee volt + UPML post-volt
				// Mur/RLC/Dispersive pre-voltage still dispatch separately before fused
				if (m_hasGPU_Mur || m_hasGPU_RLC || m_hasGPU_Dispersive)
				{
					// Record non-UPML pre-voltage extensions
					// Mur pre-voltage
					if (!m_gpuMur.empty())
						vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murPreVoltPipeline);
					for (const auto& m : m_gpuMur)
					{
						if (ts < m.startTS) continue;
						uint32_t groups = (m.totalCells + 255) / 256;
						vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
						                        m_murUpdatePipeLayout, 0, 1, &m.preDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_murUpdatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
						vkCmdDispatch(cmd, groups, 1, 1);
					}
					// RLC pre-voltage
					if (m_hasGPU_RLC)
					{
						RlcPC pc = {m_gpuRLC.count};
						uint32_t groups = (m_gpuRLC.count + 255) / 256;
						vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcPreVoltPipeline);
						vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
						                        m_rlcPrePipeLayout, 0, 1, &m_gpuRLC.preDesc, 0, nullptr);
						vkCmdPushConstants(cmd, m_rlcPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
						vkCmdDispatch(cmd, groups, 1, 1);
					}
					// Dispersive pre-voltage
					if (!m_gpuDisp.empty())
						vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPreVoltPipeline);
					for (const auto& d : m_gpuDisp)
					{
						if (!d.voltADEOn) continue;
						uint32_t groups = (d.count + 255) / 256;
						vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
						                        m_dispPrePipeLayout, 0, 1, &d.preVoltDesc, 0, nullptr);
						vkCmdPushConstants(cmd, m_dispPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &d.voltPC);
						vkCmdDispatch(cmd, groups, 1, 1);
					}
					vkCmdPipelineBarrier(cmd,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						0, 1, &barrier, 0, nullptr, 0, nullptr);
				}
				RecordLocalABCPhase(cmd, 0);

				// --- Fused voltage update (Yee + UPML pre + UPML post in one dispatch) ---
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedVoltPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_fusedPipeLayout, 0, 1, &m_fusedVoltDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_fusedPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(FusedPC), &fusedPC);
				vkCmdDispatch(cmd, voltGroups, 1, 1);

				vkCmdPipelineBarrier(cmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0, 1, &barrier, 0, nullptr, 0, nullptr);
				if (m_validateStages)
					RecordGPUFieldValidation(cmd, 16);

				// --- Post-voltage: non-UPML only (Mur, TF/SF) ---
				// (UPML post-voltage is fused into the main dispatch above)
				if (m_hasGPU_Mur || m_hasGPU_TFSF)
				{
					// Mur post-voltage
					if (!m_gpuMur.empty())
						vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murPostVoltPipeline);
					for (const auto& m : m_gpuMur)
					{
						if (ts < m.startTS) continue;
						uint32_t groups = (m.totalCells + 255) / 256;
						vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
						                        m_murUpdatePipeLayout, 0, 1, &m.postDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_murUpdatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
						vkCmdDispatch(cmd, groups, 1, 1);
					}
					// TF/SF voltage injection
					if (m_hasGPU_TFSF && m_gpuTFSF.voltCount > 0 && tfsfExc)
					{
						TfsfPC tpc = {m_gpuTFSF.voltCount, (int32_t)(ts+1), tfsfSigLen, tfsfSigPeriod};
						uint32_t groups = (m_gpuTFSF.voltCount + 255) / 256;
						vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfVoltPipeline);
						vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
						                        m_tfsfPipeLayout, 0, 1, &m_gpuTFSF.voltDesc, 0, nullptr);
						vkCmdPushConstants(cmd, m_tfsfPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TfsfPC), &tpc);
						vkCmdDispatch(cmd, groups, 1, 1);
					}
					vkCmdPipelineBarrier(cmd,
						VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						0, 1, &barrier, 0, nullptr, 0, nullptr);
				}
				RecordLocalABCPhase(cmd, 1);

				// --- Apply2Voltages (dispersive, Mur, RLC — same as non-fused path) ---
				if (!m_gpuDisp.empty())
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyVoltPipeline);
				for (const auto &d : m_gpuDisp)
				{
					if (!d.voltADEOn)
						continue;
					DispPC dpc = {d.count, numLines[0], numLines[1], numLines[2], 0};
					uint32_t groups = (d.count + 255) / 256;
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyPipeLayout, 0, 1,
											&d.applyVoltDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_dispApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC),
									   &dpc);
					vkCmdDispatch(cmd, groups, 1, 1);
					vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
					                     1, &barrier, 0, nullptr, 0, nullptr);
				}
				if (!m_gpuMur.empty())
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murApplyVoltPipeline);
				for (const auto &m : m_gpuMur)
				{
					if (ts < m.startTS)
						continue;
					uint32_t groups = (m.totalCells + 255) / 256;
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murApplyPipeLayout, 0, 1,
											&m.applyDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_murApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
					vkCmdDispatch(cmd, groups, 1, 1);
				}
				if (m_hasGPU_RLC)
				{
					RlcPC pc = {m_gpuRLC.count};
					uint32_t groups = (m_gpuRLC.count + 255) / 256;
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcApplyVoltPipeline);
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcApplyPipeLayout, 0, 1,
											&m_gpuRLC.applyDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_rlcApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
					vkCmdDispatch(cmd, groups, 1, 1);
				}
				RecordLocalABCPhase(cmd, 2);

				// --- Voltage excitation ---
				// Match Engine::Apply2Voltages(): dispersive and RLC corrections
				// precede the low-priority excitation extension.
				if (hasExcVolt)
				{
					ExcPC epc = {m_excVoltCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod};
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					                        m_excPipeLayout, 0, 1, &m_excVoltDescSet, 0, nullptr);
					vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
					                   0, sizeof(ExcPC), &epc);
					vkCmdDispatch(cmd, excVoltGroups, 1, 1);
				}

				// Late voltage writes must be visible to pre-current extensions
				// and to the fused Yee current update.
				if (hasVoltLateWritesFused)
				{
					vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
										 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
				}
				if (m_validateStages)
					RecordGPUFieldValidation(cmd, 32);
				RecordSteadyStateSample(cmd, ts);

				// --- Fused current update (Yee + UPML pre + UPML post) ---
				// Pre-current non-UPML extensions (dispersive)
				if (m_hasGPU_Dispersive)
				{
					if (!m_gpuDisp.empty())
						vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPreCurrPipeline);
					for (const auto &d : m_gpuDisp)
					{
						if (!d.currADEOn)
							continue;
						uint32_t groups = (d.count + 255) / 256;
						vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispPrePipeLayout, 0, 1,
												&d.preCurrDesc, 0, nullptr);
						vkCmdPushConstants(cmd, m_dispPrePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC),
										   &d.currPC);
						vkCmdDispatch(cmd, groups, 1, 1);
					}
					vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
										 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
				}
				RecordLocalABCPhase(cmd, 3);

				// Fused current dispatch (Yee + UPML pre/post)
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedCurrPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedPipeLayout, 0, 1,
										&m_fusedCurrDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_fusedPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FusedPC), &fusedPC);
				vkCmdDispatch(cmd, voltGroups, 1, 1); // dispatch N threads (same as voltage)

				vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
									 1, &barrier, 0, nullptr, 0, nullptr);
				if (m_validateStages)
					RecordGPUFieldValidation(cmd, 64);

				// --- Post-current: TF/SF only (UPML is fused) ---
				if (m_hasGPU_TFSF && m_gpuTFSF.currCount > 0 && tfsfExc)
				{
					TfsfPC tpc = {m_gpuTFSF.currCount, (int32_t)(ts), tfsfSigLen, tfsfSigPeriod};
					uint32_t groups = (m_gpuTFSF.currCount + 255) / 256;
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfCurrPipeline);
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfPipeLayout, 0, 1,
											&m_gpuTFSF.currDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_tfsfPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TfsfPC), &tpc);
					vkCmdDispatch(cmd, groups, 1, 1);
				}
				if (m_hasGPU_TFSF && hasDispCurrADE)
				{
					vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
										 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
				}
				RecordLocalABCPhase(cmd, 4);
				// Dispersive apply current
				if (!m_gpuDisp.empty())
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyCurrPipeline);
				for (const auto &d : m_gpuDisp)
				{
					if (!d.currADEOn)
						continue;
					DispPC dpc = {d.count, numLines[0], numLines[1], numLines[2], 0};
					uint32_t groups = (d.count + 255) / 256;
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyPipeLayout, 0, 1,
											&d.applyCurrDesc, 0, nullptr);
					vkCmdPushConstants(cmd, m_dispApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC),
									   &dpc);
					vkCmdDispatch(cmd, groups, 1, 1);
					vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
					                     1, &barrier, 0, nullptr, 0, nullptr);
				}
				RecordLocalABCPhase(cmd, 5);

				// --- Current excitation ---
				// Match Engine::Apply2Current(): dispersive corrections precede
				// the low-priority excitation extension.
				if (hasExcCurr)
				{
					ExcPC epc = {m_excCurrCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod};
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeLayout, 0, 1,
											&m_excCurrDescSet, 0, nullptr);
					vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ExcPC), &epc);
					vkCmdDispatch(cmd, excCurrGroups, 1, 1);
				}
				if (hasCurrLateWritesFused)
				{
					vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
										 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
				}
				if (m_validateStages)
					RecordGPUFieldValidation(cmd, 80);
			}
			else
			{
			// ============= Original separate-dispatch path =============
			// === Pre-voltage extensions (GPU) ===
			RecordVoltageExtensions(cmd, ts);
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 0);

			// --- Voltage update ---
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateVoltPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_fdtdPipeLayout, 0, 1, &m_updateVoltDescSet, 0, nullptr);
			vkCmdPushConstants(cmd, m_fdtdPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			                   0, sizeof(GridPC), &gridPC);
			vkCmdDispatch(cmd, voltGroups, 1, 1);

			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0, 1, &barrier, 0, nullptr, 0, nullptr);
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 16);

			// === Post-voltage extensions (GPU) ===
			// UPML post-voltage
			if (!m_gpuUPML.empty())
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPostVoltPipeline);
			for (const auto& u : m_gpuUPML)
			{
				uint32_t groups = (u.totalCells + 255) / 256;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_upmlPostPipeLayout, 0, 1, &u.postVoltDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_upmlPostPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PmlPC), &u.pc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}

			// Mur post-voltage (accumulates shifted field)
			if (!m_gpuMur.empty())
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murPostVoltPipeline);
			for (const auto& m : m_gpuMur)
			{
				if (ts < m.startTS) continue;
				uint32_t groups = (m.totalCells + 255) / 256;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_murUpdatePipeLayout, 0, 1, &m.postDesc, 0, nullptr);
			vkCmdPushConstants(cmd, m_murUpdatePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}

			// TF/SF voltage injection
			if (m_hasGPU_TFSF && m_gpuTFSF.voltCount > 0 && tfsfExc)
			{
				TfsfPC tpc = {m_gpuTFSF.voltCount, (int32_t)(ts+1), tfsfSigLen, tfsfSigPeriod};
				uint32_t groups = (m_gpuTFSF.voltCount + 255) / 256;
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfVoltPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_tfsfPipeLayout, 0, 1, &m_gpuTFSF.voltDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_tfsfPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TfsfPC), &tpc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}

			// Barrier for post-voltage writes
			if (m_hasGPU_UPML || m_hasGPU_Mur || m_hasGPU_TFSF)
			{
				vkCmdPipelineBarrier(cmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0, 1, &barrier, 0, nullptr, 0, nullptr);
			}
			RecordLocalABCPhase(cmd, 1);

			// === Apply2Voltages extensions (GPU) ===
			// Dispersive apply
			if (!m_gpuDisp.empty())
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyVoltPipeline);
			for (const auto &d : m_gpuDisp)
			{
				if (!d.voltADEOn)
					continue;
				DispPC dpc = {d.count, numLines[0], numLines[1], numLines[2], 0};
				uint32_t groups = (d.count + 255) / 256;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyPipeLayout, 0, 1,
										&d.applyVoltDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_dispApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &dpc);
				vkCmdDispatch(cmd, groups, 1, 1);
				vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
				                     1, &barrier, 0, nullptr, 0, nullptr);
			}

			// Mur apply (overwrites boundary voltages)
			if (!m_gpuMur.empty())
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murApplyVoltPipeline);
			for (const auto &m : m_gpuMur)
			{
				if (ts < m.startTS)
					continue;
				uint32_t groups = (m.totalCells + 255) / 256;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_murApplyPipeLayout, 0, 1, &m.applyDesc,
										0, nullptr);
				vkCmdPushConstants(cmd, m_murApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MurPC), &m.pc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}

			// RLC apply
			if (m_hasGPU_RLC)
			{
				RlcPC pc = {m_gpuRLC.count};
				uint32_t groups = (m_gpuRLC.count + 255) / 256;
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcApplyVoltPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rlcApplyPipeLayout, 0, 1,
										&m_gpuRLC.applyDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_rlcApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RlcPC), &pc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}
			RecordLocalABCPhase(cmd, 2);

			// --- Voltage excitation ---
			// Match Engine::Apply2Voltages(): dispersive and RLC corrections
			// precede the low-priority excitation extension.
			if (hasExcVolt)
			{
				ExcPC epc = {m_excVoltCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod};
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_excPipeLayout, 0, 1, &m_excVoltDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(ExcPC), &epc);
				vkCmdDispatch(cmd, excVoltGroups, 1, 1);
			}

			if (hasVoltLateWrites)
			{
				vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
									 1, &barrier, 0, nullptr, 0, nullptr);
			}
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 32);
			RecordSteadyStateSample(cmd, ts);

			// === Pre-current extensions (GPU) ===
			RecordCurrentExtensions(cmd, ts);
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 48);

			// --- Current update ---
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateCurrPipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fdtdPipeLayout, 0, 1, &m_updateCurrDescSet,
									0, nullptr);
			vkCmdPushConstants(cmd, m_fdtdPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GridPC), &gridPC);
			vkCmdDispatch(cmd, currGroups, 1, 1);

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
								 &barrier, 0, nullptr, 0, nullptr);
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 64);

			// === Post-current extensions (GPU) ===
			// UPML post-current
			if (!m_gpuUPML.empty())
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPostCurrPipeline);
			for (const auto &u : m_gpuUPML)
			{
				uint32_t groups = (u.totalCells + 255) / 256;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upmlPostPipeLayout, 0, 1,
										&u.postCurrDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_upmlPostPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PmlPC), &u.pc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}

			// TF/SF current injection
			if (m_hasGPU_TFSF && m_gpuTFSF.currCount > 0 && tfsfExc)
			{
				TfsfPC tpc = {m_gpuTFSF.currCount, (int32_t)(ts), tfsfSigLen, tfsfSigPeriod};
				uint32_t groups = (m_gpuTFSF.currCount + 255) / 256;
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfCurrPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tfsfPipeLayout, 0, 1,
										&m_gpuTFSF.currDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_tfsfPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TfsfPC), &tpc);
				vkCmdDispatch(cmd, groups, 1, 1);
			}

			if ((m_hasGPU_UPML || m_hasGPU_TFSF) && hasDispCurrADE)
			{
				vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
									 1, &barrier, 0, nullptr, 0, nullptr);
			}
			RecordLocalABCPhase(cmd, 4);

			// Dispersive apply current
			if (!m_gpuDisp.empty())
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyCurrPipeline);
			for (const auto &d : m_gpuDisp)
			{
				if (!d.currADEOn)
					continue;
				DispPC dpc = {d.count, numLines[0], numLines[1], numLines[2], 0};
				uint32_t groups = (d.count + 255) / 256;
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dispApplyPipeLayout, 0, 1,
										&d.applyCurrDesc, 0, nullptr);
				vkCmdPushConstants(cmd, m_dispApplyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DispPC), &dpc);
				vkCmdDispatch(cmd, groups, 1, 1);
				vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
				                     1, &barrier, 0, nullptr, 0, nullptr);
			}
			RecordLocalABCPhase(cmd, 5);

			// --- Current excitation ---
			// Match Engine::Apply2Current(): dispersive corrections precede
			// the low-priority excitation extension.
			if (hasExcCurr)
			{
				ExcPC epc = {m_excCurrCount, (int32_t)ts, m_excSignalLen, m_excSignalPeriod};
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeLayout, 0, 1,
										&m_excCurrDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ExcPC), &epc);
				vkCmdDispatch(cmd, excCurrGroups, 1, 1);
			}

			if (hasCurrLateWrites)
			{
				vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
									 1, &barrier, 0, nullptr, 0, nullptr);
			}
			if (m_validateStages)
				RecordGPUFieldValidation(cmd, 80);
			} // end else (original separate-dispatch path)
		}

		if (m_validateStages)
		{
			VkMemoryBarrier hostBarrier{};
			hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
			hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     VK_PIPELINE_STAGE_HOST_BIT, 0,
			                     1, &hostBarrier, 0, nullptr, 0, nullptr);
		}

		if (m_steadyPipeline)
		{
			VkMemoryBarrier hostBarrier{};
			hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
			hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     VK_PIPELINE_STAGE_HOST_BIT, 0,
			                     1, &hostBarrier, 0, nullptr, 0, nullptr);
		}

		if (m_profileEnabled && m_tsQueryPool && m_profilePhaseEnabled)
		{
			// End of compute/update section (before probe gather/copy section)
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_tsQueryPool, qBaseProfile + 1);
		}

		// --- Probe gather: append at end of batch ----------------------
		if (m_hasGPU_Probes && m_probeCacheCount > 0)
		{
			// Final barrier: ensure all Yee/extension writes are visible
			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0, 1, &barrier, 0, nullptr, 0, nullptr);

			uint32_t probeGroups = (m_probeCacheCount + 255) / 256;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_probePipeline);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
			                        m_probePipeLayout, 0, 1, &m_probeGatherDescSet, 0, nullptr);
			vkCmdPushConstants(cmd, m_probePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			                   0, sizeof(uint32_t), &m_probeCacheCount);
			vkCmdDispatch(cmd, probeGroups, 1, 1);

			if (m_probeOutMapped)
			{
				VkMemoryBarrier hostBarrier{};
				hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
				hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
				vkCmdPipelineBarrier(cmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
					0, 1, &hostBarrier, 0, nullptr, 0, nullptr);
			}

			// If probe output is not directly mapped, enqueue a device->host
			// copy into the readback ring for this submission slot.
			if (useAsyncProbeCopy)
			{
				VkBufferMemoryBarrier ownRelease{};
				ownRelease.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
				ownRelease.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				ownRelease.dstAccessMask = 0;
				ownRelease.srcQueueFamilyIndex = m_computeQueueFamily;
				ownRelease.dstQueueFamilyIndex = m_transferQueueFamily;
				ownRelease.buffer = m_probeOutBuf.buffer;
				ownRelease.offset = 0;
				ownRelease.size = VK_WHOLE_SIZE;
				vkCmdPipelineBarrier(cmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
					0, 0, nullptr, 1, &ownRelease, 0, nullptr);
			}
			else if (!m_probeOutMapped && m_probeReadbackMapped[m_cmdIdx])
			{
				VkMemoryBarrier xferBarrier{};
				xferBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
				xferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				xferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				vkCmdPipelineBarrier(cmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					0, 1, &xferBarrier, 0, nullptr, 0, nullptr);

				VkBufferCopy copyRegion = {0, 0, m_probeCacheCount * sizeof(float)};
				vkCmdCopyBuffer(cmd, m_probeOutBuf.buffer,
				                m_probeReadbackBuf[m_cmdIdx].buffer, 1, &copyRegion);
			}
		}

		if (m_profileEnabled && m_tsQueryPool && m_profilePhaseEnabled)
		{
			// End of probe section (or immediately after compute when no probes)
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_tsQueryPool, qBaseProfile + 2);
		}

		if (m_profileEnabled && m_tsQueryPool)
		{
			vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			                   m_tsQueryPool, qBaseProfile + (m_profileQueriesPerSlot - 1));
		}

		// --- End + submit (fire-and-forget) ---
		VK_CHECK(vkEndCommandBuffer(cmd));

		VkSubmitInfo si{};
		si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		si.commandBufferCount = 1;
		si.pCommandBuffers    = &cmd;
		VkSemaphore signalSem = VK_NULL_HANDLE;
		if (useAsyncProbeCopy)
		{
			signalSem = m_probeCopySem[m_cmdIdx];
			si.signalSemaphoreCount = 1;
			si.pSignalSemaphores = &signalSem;
		}
		vkResetFences(m_device, 1, &m_fences[m_cmdIdx]);
		VK_CHECK(vkQueueSubmit(m_computeQueue, 1, &si, m_fences[m_cmdIdx]));

		if (useAsyncProbeCopy)
		{
			VkCommandBuffer tcmd = m_transferCmdBufs[m_cmdIdx];
			vkResetCommandBuffer(tcmd, 0);
			VkCommandBufferBeginInfo tbi{};
			tbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			tbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			VK_CHECK(vkBeginCommandBuffer(tcmd, &tbi));

			VkBufferMemoryBarrier ownAcquire{};
			ownAcquire.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			ownAcquire.srcAccessMask = 0;
			ownAcquire.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			ownAcquire.srcQueueFamilyIndex = m_computeQueueFamily;
			ownAcquire.dstQueueFamilyIndex = m_transferQueueFamily;
			ownAcquire.buffer = m_probeOutBuf.buffer;
			ownAcquire.offset = 0;
			ownAcquire.size = VK_WHOLE_SIZE;
			vkCmdPipelineBarrier(tcmd,
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, 0, nullptr, 1, &ownAcquire, 0, nullptr);

			VkBufferCopy copyRegion = {0, 0, m_probeCacheCount * sizeof(float)};
			vkCmdCopyBuffer(tcmd, m_probeOutBuf.buffer,
			                m_probeReadbackBuf[m_cmdIdx].buffer, 1, &copyRegion);

			VkBufferMemoryBarrier ownRelease{};
			ownRelease.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			ownRelease.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			ownRelease.dstAccessMask = 0;
			ownRelease.srcQueueFamilyIndex = m_transferQueueFamily;
			ownRelease.dstQueueFamilyIndex = m_computeQueueFamily;
			ownRelease.buffer = m_probeOutBuf.buffer;
			ownRelease.offset = 0;
			ownRelease.size = VK_WHOLE_SIZE;
			vkCmdPipelineBarrier(tcmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				0, 0, nullptr, 1, &ownRelease, 0, nullptr);

			VK_CHECK(vkEndCommandBuffer(tcmd));

			VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			VkSubmitInfo tsi{};
			tsi.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			tsi.waitSemaphoreCount = 1;
			tsi.pWaitSemaphores = &signalSem;
			tsi.pWaitDstStageMask = &waitStage;
			tsi.commandBufferCount = 1;
			tsi.pCommandBuffers = &tcmd;
			vkResetFences(m_device, 1, &m_transferFences[m_cmdIdx]);
			VK_CHECK(vkQueueSubmit(m_transferQueue, 1, &tsi, m_transferFences[m_cmdIdx]));
			m_probeTransferInFlight[m_cmdIdx] = true;
		}
		m_gpuInFlight[m_cmdIdx] = true;
		m_lastProbeSubmitSlot = m_cmdIdx;
		if (m_profileEnabled)
		{
			m_profilePending[m_cmdIdx] = true;
			m_profileSubmittedTS[m_cmdIdx] = iterTS;
		}
		m_gpuDrained = false;  // new submission in flight
		m_cmdIdx = (m_cmdIdx + 1) % CMD_RING_SIZE;

		numTS += iterTS;
		m_hostDirty = true;
	}
	else
	{
		// ========== Hybrid path: per-timestep with CPU extension sync =====
		// Each cycle: GPU update → 1 download → CPU extensions → 1 upload
		// Per timestep: 2 downloads + 2 uploads (minimum possible)
		for (unsigned int iter = 0; iter < iterTS; ++iter)
		{
			// --- Pre-voltage extensions (CPU) ---
			// Host is current from previous iteration's final upload,
			// or from Init() on the first iteration.
			DoPreVoltageUpdates();
			m_deviceDirty = true;
			SyncFieldsToDevice();

			// --- GPU: voltage update + excitation ---
			RunSingleCommand([&](VkCommandBuffer cmd)
			{
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateVoltPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_fdtdPipeLayout, 0, 1, &m_updateVoltDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_fdtdPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(GridPC), &gridPC);
				vkCmdDispatch(cmd, voltGroups, 1, 1);

				vkCmdPipelineBarrier(cmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0, 1, &barrier, 0, nullptr, 0, nullptr);

				if (m_hasGPUExcitation && m_excVoltCount > 0)
				{
					ExcPC epc = {m_excVoltCount, (int32_t)numTS, m_excSignalLen, m_excSignalPeriod};
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					                        m_excPipeLayout, 0, 1, &m_excVoltDescSet, 0, nullptr);
					vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
					                   0, sizeof(ExcPC), &epc);
					vkCmdDispatch(cmd, excVoltGroups, 1, 1);
				}
			});

			// --- Post-voltage + apply + pre-current extensions (CPU) ---
			// Single download, batch all CPU voltage+current-prep work, single upload
			m_hostDirty = true;
			SyncFieldsToHost();
			DoPostVoltageUpdates();
			Apply2Voltages();
			DoPreCurrentUpdates();
			m_deviceDirty = true;
			SyncFieldsToDevice();

			// --- GPU: current update + excitation ---
			RunSingleCommand([&](VkCommandBuffer cmd)
			{
				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateCurrPipeline);
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
				                        m_fdtdPipeLayout, 0, 1, &m_updateCurrDescSet, 0, nullptr);
				vkCmdPushConstants(cmd, m_fdtdPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
				                   0, sizeof(GridPC), &gridPC);
				vkCmdDispatch(cmd, currGroups, 1, 1);

				vkCmdPipelineBarrier(cmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0, 1, &barrier, 0, nullptr, 0, nullptr);

				if (m_hasGPUExcitation && m_excCurrCount > 0)
				{
					ExcPC epc = {m_excCurrCount, (int32_t)numTS, m_excSignalLen, m_excSignalPeriod};
					vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_excPipeline);
					vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
					                        m_excPipeLayout, 0, 1, &m_excCurrDescSet, 0, nullptr);
					vkCmdPushConstants(cmd, m_excPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
					                   0, sizeof(ExcPC), &epc);
					vkCmdDispatch(cmd, excCurrGroups, 1, 1);
				}
			});

			// --- Post-current + apply extensions (CPU) ---
			m_hostDirty = true;
			SyncFieldsToHost();
			DoPostCurrentUpdates();
			Apply2Current();
			// Host is now current; device upload deferred to next iteration
			m_deviceDirty = true;

			++numTS;
		}
		// Final upload so device is consistent
		if (m_deviceDirty) SyncFieldsToDevice();
		m_hostDirty = false;
	}

	if (m_validateStages)
		ReportGPUFieldValidation(numTS);
	else if (m_validateFields)
		ValidateGPUFields(numTS);

	return true;
}

// ===========================================================================
// Cleanup
// ===========================================================================

void Engine_Vulkan::CleanupVulkan()
{
	if (m_device == VK_NULL_HANDLE) return;
	vkDeviceWaitIdle(m_device);

	// Async field-dump download pipeline (own queue/pools/buffers)
	CleanupDumpAsync();

	// First clean up extension resources
	CleanupExtensions();

	auto destroyBufMem = [&](VkBuffer& buf, VkDeviceMemory& mem)
	{
		if (buf) { vkDestroyBuffer(m_device, buf, nullptr); buf = VK_NULL_HANDLE; }
		if (mem) { vkFreeMemory(m_device, mem, nullptr);    mem = VK_NULL_HANDLE; }
	};

	// Pipelines
	if (m_updateVoltPipeline) vkDestroyPipeline(m_device, m_updateVoltPipeline, nullptr);
	if (m_updateCurrPipeline) vkDestroyPipeline(m_device, m_updateCurrPipeline, nullptr);
	if (m_excPipeline)        vkDestroyPipeline(m_device, m_excPipeline, nullptr);
	if (m_fdtdPipeLayout) vkDestroyPipelineLayout(m_device, m_fdtdPipeLayout, nullptr);
	if (m_excPipeLayout)  vkDestroyPipelineLayout(m_device, m_excPipeLayout, nullptr);

	// Descriptors
	if (m_descPool)       vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
	if (m_fdtdDescLayout) vkDestroyDescriptorSetLayout(m_device, m_fdtdDescLayout, nullptr);
	if (m_excDescLayout)  vkDestroyDescriptorSetLayout(m_device, m_excDescLayout, nullptr);

	// Buffers
	if (m_voltMapped) { vkUnmapMemory(m_device, m_voltMem); m_voltMapped = nullptr; }
	if (m_currMapped) { vkUnmapMemory(m_device, m_currMem); m_currMapped = nullptr; }
	destroyBufMem(m_voltBuf, m_voltMem);
	destroyBufMem(m_currBuf, m_currMem);
	destroyBufMem(m_opIndexBuf, m_opIndexMem);
	destroyBufMem(m_vvCompBuf, m_vvCompMem);
	destroyBufMem(m_viCompBuf, m_viCompMem);
	destroyBufMem(m_iiCompBuf, m_iiCompMem);
	destroyBufMem(m_ivCompBuf, m_ivCompMem);
	destroyBufMem(m_stagingBuf, m_stagingMem);
	destroyBufMem(m_excVoltIdxBuf, m_excVoltIdxMem);
	destroyBufMem(m_excVoltAmpBuf, m_excVoltAmpMem);
	destroyBufMem(m_excVoltDelayBuf, m_excVoltDelayMem);
	destroyBufMem(m_excCurrIdxBuf, m_excCurrIdxMem);
	destroyBufMem(m_excCurrAmpBuf, m_excCurrAmpMem);
	destroyBufMem(m_excCurrDelayBuf, m_excCurrDelayMem);
	destroyBufMem(m_excSignalVoltBuf, m_excSignalVoltMem);
	destroyBufMem(m_excSignalCurrBuf, m_excSignalCurrMem);

	// Command / sync
	for (int i = 0; i < CMD_RING_SIZE; ++i)
		if (m_fences[i]) vkDestroyFence(m_device, m_fences[i], nullptr);
	for (int i = 0; i < CMD_RING_SIZE; ++i)
	{
		if (m_transferFences[i]) vkDestroyFence(m_device, m_transferFences[i], nullptr);
		if (m_probeCopySem[i]) vkDestroySemaphore(m_device, m_probeCopySem[i], nullptr);
	}
	if (m_utilFence) vkDestroyFence(m_device, m_utilFence, nullptr);
	if (m_transferCmdPool) vkDestroyCommandPool(m_device, m_transferCmdPool, nullptr);
	if (m_cmdPool) vkDestroyCommandPool(m_device, m_cmdPool, nullptr);

	// Optional raw-field validation
	CleanupGPU_FieldValidation();

	// Energy reduction
	CleanupGPU_EnergyReduction();

	// Fused UPML
	CleanupFusedUPML();

	if (m_profileEnabled)
	{
		for (int i = 0; i < CMD_RING_SIZE; ++i)
			CollectProfileForSlot(i);
		PrintProfileSummary();
	}
	if (m_tsQueryPool)
	{
		vkDestroyQueryPool(m_device, m_tsQueryPool, nullptr);
		m_tsQueryPool = VK_NULL_HANDLE;
	}

	// Pipeline cache
	if (m_pipelineCache) vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);

	// Device / instance
	vkDestroyDevice(m_device, nullptr);
	m_device = VK_NULL_HANDLE;
	vkDestroyInstance(m_instance, nullptr);
	m_instance = VK_NULL_HANDLE;
}

// ===========================================================================
// Extension cleanup
// ===========================================================================

void Engine_Vulkan::CleanupExtensions()
{
	if (m_device == VK_NULL_HANDLE) return;

	// --- Destroy extension pipelines ---
	auto destroyPipeline = [&](VkPipeline& p)
	{
		if (p) { vkDestroyPipeline(m_device, p, nullptr); p = VK_NULL_HANDLE; }
	};
	auto destroyPipeLayout = [&](VkPipelineLayout& pl)
	{
		if (pl) { vkDestroyPipelineLayout(m_device, pl, nullptr); pl = VK_NULL_HANDLE; }
	};
	auto destroyDescLayout = [&](VkDescriptorSetLayout& dl)
	{
		if (dl) { vkDestroyDescriptorSetLayout(m_device, dl, nullptr); dl = VK_NULL_HANDLE; }
	};

	if (m_gpuSteadyStateExt)
		m_gpuSteadyStateExt->SetGpuUpdater({});
	if (m_steadyHistoryMapped)
	{
		vkUnmapMemory(m_device, m_steadyHistory.memory);
		m_steadyHistoryMapped = nullptr;
	}
	destroyPipeline(m_steadyPipeline);
	destroyPipeLayout(m_steadyPipeLayout);
	if (m_steadyDescPool)
	{
		vkDestroyDescriptorPool(m_device, m_steadyDescPool, nullptr);
		m_steadyDescPool = VK_NULL_HANDLE;
	}
	destroyDescLayout(m_steadyDescLayout);
	DestroyGpuBuf(m_steadyProbeIdx);
	DestroyGpuBuf(m_steadyHistory);
	m_steadyDescSet = VK_NULL_HANDLE;
	m_gpuSteadyStateExt = nullptr;
	m_steadyProbeCount = m_steadyPeriod = m_steadyRingSize = 0;
	m_steadyLastCompletedPeriods = 0;

	destroyPipeline(m_localAbcPipeline);
	destroyPipeLayout(m_localAbcPipeLayout);
	if (m_localAbcDescPool)
	{
		vkDestroyDescriptorPool(m_device, m_localAbcDescPool, nullptr);
		m_localAbcDescPool = VK_NULL_HANDLE;
	}
	destroyDescLayout(m_localAbcDescLayout);
	for (GpuLocalABCData& data : m_gpuLocalABC)
	{
		DestroyGpuBuf(data.voltState);
		DestroyGpuBuf(data.currState);
		DestroyGpuBuf(data.k1);
		DestroyGpuBuf(data.k2);
	}
	m_gpuLocalABC.clear();

	destroyPipeline(m_upmlPreVoltPipeline);
	destroyPipeline(m_upmlPostVoltPipeline);
	destroyPipeline(m_upmlPreCurrPipeline);
	destroyPipeline(m_upmlPostCurrPipeline);
	destroyPipeline(m_dispPreVoltPipeline);
	destroyPipeline(m_dispPreCurrPipeline);
	destroyPipeline(m_dispApplyVoltPipeline);
	destroyPipeline(m_dispApplyCurrPipeline);
	destroyPipeline(m_tfsfVoltPipeline);
	destroyPipeline(m_tfsfCurrPipeline);
	destroyPipeline(m_murPreVoltPipeline);
	destroyPipeline(m_murPostVoltPipeline);
	destroyPipeline(m_murApplyVoltPipeline);
	destroyPipeline(m_rlcPreVoltPipeline);
	destroyPipeline(m_rlcApplyVoltPipeline);
	destroyPipeline(m_probePipeline);

	destroyPipeLayout(m_upmlPrePipeLayout);
	destroyPipeLayout(m_upmlPostPipeLayout);
	destroyPipeLayout(m_dispPrePipeLayout);
	destroyPipeLayout(m_dispApplyPipeLayout);
	destroyPipeLayout(m_tfsfPipeLayout);
	destroyPipeLayout(m_murUpdatePipeLayout);
	destroyPipeLayout(m_murApplyPipeLayout);
	destroyPipeLayout(m_rlcPrePipeLayout);
	destroyPipeLayout(m_rlcApplyPipeLayout);
	destroyPipeLayout(m_probePipeLayout);

	// --- Destroy extension descriptor pool (this frees all descriptor sets) ---
	if (m_extDescPool) { vkDestroyDescriptorPool(m_device, m_extDescPool, nullptr); m_extDescPool = VK_NULL_HANDLE; }

	destroyDescLayout(m_upmlPreVoltDescLayout);
	destroyDescLayout(m_upmlPostVoltDescLayout);
	destroyDescLayout(m_upmlPreCurrDescLayout);
	destroyDescLayout(m_upmlPostCurrDescLayout);
	destroyDescLayout(m_dispPreDescLayout);
	destroyDescLayout(m_dispApplyDescLayout);
	destroyDescLayout(m_tfsfDescLayout);
	destroyDescLayout(m_murPreDescLayout);
	destroyDescLayout(m_murPostDescLayout);
	destroyDescLayout(m_murApplyDescLayout);
	destroyDescLayout(m_rlcPreDescLayout);
	destroyDescLayout(m_rlcApplyDescLayout);
	destroyDescLayout(m_probeDescLayout);

	// --- Destroy UPML GPU data ---
	for (auto& u : m_gpuUPML)
	{
		DestroyGpuBuf(u.voltFlux); DestroyGpuBuf(u.currFlux);
		DestroyGpuBuf(u.pmlVv);   DestroyGpuBuf(u.pmlVvfo); DestroyGpuBuf(u.pmlVvfn);
		DestroyGpuBuf(u.pmlIi);   DestroyGpuBuf(u.pmlIifo); DestroyGpuBuf(u.pmlIifn);
	}
	m_gpuUPML.clear();

	// --- Destroy dispersive GPU data ---
	for (auto& d : m_gpuDisp)
	{
		DestroyGpuBuf(d.voltADE);    DestroyGpuBuf(d.voltLorADE);
		DestroyGpuBuf(d.currADE);    DestroyGpuBuf(d.currLorADE);
		DestroyGpuBuf(d.posIdx);
		DestroyGpuBuf(d.vIntADE);    DestroyGpuBuf(d.vExtADE); DestroyGpuBuf(d.vLorADE);
		DestroyGpuBuf(d.iIntADE);    DestroyGpuBuf(d.iExtADE); DestroyGpuBuf(d.iLorADE);
	}
	m_gpuDisp.clear();

	// --- Destroy TF/SF GPU data ---
	DestroyGpuBuf(m_gpuTFSF.voltAmp);   DestroyGpuBuf(m_gpuTFSF.voltDelta);
	DestroyGpuBuf(m_gpuTFSF.voltIdx);   DestroyGpuBuf(m_gpuTFSF.voltDelay);
	DestroyGpuBuf(m_gpuTFSF.voltSignal);
	DestroyGpuBuf(m_gpuTFSF.currAmp);   DestroyGpuBuf(m_gpuTFSF.currDelta);
	DestroyGpuBuf(m_gpuTFSF.currIdx);   DestroyGpuBuf(m_gpuTFSF.currDelay);
	DestroyGpuBuf(m_gpuTFSF.currSignal);
	m_gpuTFSF = GpuTFSFData{};

	// --- Destroy Mur ABC GPU data ---
	for (auto& m : m_gpuMur)
	{
		DestroyGpuBuf(m.murNyP);    DestroyGpuBuf(m.murNyPP);
		DestroyGpuBuf(m.coeffNyP);  DestroyGpuBuf(m.coeffNyPP);
	}
	m_gpuMur.clear();

	// --- Destroy RLC GPU data ---
	DestroyGpuBuf(m_gpuRLC.il);
	DestroyGpuBuf(m_gpuRLC.vdn0); DestroyGpuBuf(m_gpuRLC.vdn1); DestroyGpuBuf(m_gpuRLC.vdn2);
	DestroyGpuBuf(m_gpuRLC.jn0);  DestroyGpuBuf(m_gpuRLC.jn1);  DestroyGpuBuf(m_gpuRLC.jn2);
	DestroyGpuBuf(m_gpuRLC.linIdx);
	DestroyGpuBuf(m_gpuRLC.i2v);  DestroyGpuBuf(m_gpuRLC.ilv);
	DestroyGpuBuf(m_gpuRLC.vvd);  DestroyGpuBuf(m_gpuRLC.vv2);
	DestroyGpuBuf(m_gpuRLC.vj1);  DestroyGpuBuf(m_gpuRLC.vj2);
	DestroyGpuBuf(m_gpuRLC.ib0);  DestroyGpuBuf(m_gpuRLC.b1); DestroyGpuBuf(m_gpuRLC.b2);
	m_gpuRLC = GpuRLCData{};

	// --- Destroy probe GPU data ---
	DestroyGpuBuf(m_gpuProbes.probeIdx);
	DestroyGpuBuf(m_gpuProbes.fieldSel);
	DestroyGpuBuf(m_gpuProbes.output);
	m_gpuProbes = GpuProbeData{};

	// --- Destroy probe cache GPU data (pipelined processing) ---
	if (m_probeOutMapped)
	{
		vkUnmapMemory(m_device, m_probeOutBuf.memory);
		m_probeOutMapped = nullptr;
	}
	DestroyGpuBuf(m_probeIdxBuf);
	DestroyGpuBuf(m_probeSelBuf);
	DestroyGpuBuf(m_probeOutBuf);
	for (int i = 0; i < CMD_RING_SIZE; ++i)
	{
		if (m_probeReadbackMapped[i])
		{
			vkUnmapMemory(m_device, m_probeReadbackBuf[i].memory);
			m_probeReadbackMapped[i] = nullptr;
		}
		DestroyGpuBuf(m_probeReadbackBuf[i]);
	}
	DestroyGpuBuf(m_probeStagingBuf);
	if (m_probeDescPool) { vkDestroyDescriptorPool(m_device, m_probeDescPool, nullptr); m_probeDescPool = VK_NULL_HANDLE; }
	m_probeGatherDescSet = VK_NULL_HANDLE; // freed with pool
	m_probeCacheKeys.clear();
	m_probeCacheCPU.clear();
	m_probeCacheCount = 0;

	// Reset flags
	m_hasGPU_UPML = false;
	m_hasGPU_Dispersive = false;
	m_hasGPU_TFSF = false;
	m_hasGPU_Mur = false;
	m_hasGPU_RLC = false;
	m_hasGPU_Probes = false;
}

// ===========================================================================
// Fused Yee+UPML Pipeline
// ===========================================================================

void Engine_Vulkan::SetupFusedUPML()
{
	if (!m_hasGPU_UPML || m_gpuUPML.empty())
		return;
	if (const char* env = std::getenv("OPENEMS_GPU_DISABLE_FUSED_UPML"))
	{
		char* endPtr = nullptr;
		long v = std::strtol(env, &endPtr, 10);
		if (endPtr && *endPtr == '\0' && v != 0)
		{
			cout << "Engine_Vulkan: fused UPML disabled by environment" << endl;
			return;
		}
	}

	// --- Compute total PML cells and build region metadata ---
	struct PMLRegionGPU { uint32_t startX, startY, startZ, sizeX, sizeY, sizeZ, fluxOffset; };
	std::vector<PMLRegionGPU> regions;
	uint32_t totalPmlCells = 0;

	for (const auto& u : m_gpuUPML)
	{
		PMLRegionGPU r;
		r.startX = u.pc.pStartX;
		r.startY = u.pc.pStartY;
		r.startZ = u.pc.pStartZ;
		r.sizeX  = u.pc.pNx;
		r.sizeY  = u.pc.pNy;
		r.sizeZ  = u.pc.pNz;
		// Each region is packed as [component 0][component 1][component 2].
		// Store an element offset, not a per-component cell offset.
		r.fluxOffset = 3u * totalPmlCells;
		regions.push_back(r);
		totalPmlCells += u.totalCells;
	}
	m_fusedTotalPmlCells = totalPmlCells;

	if (regions.size() > 6)
	{
		cerr << "Engine_Vulkan::SetupFusedUPML: too many PML regions (" << regions.size()
		     << "), max 6 — falling back to separate dispatches" << endl;
		return;
	}

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBufferUsageFlags rwUsage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkMemoryPropertyFlags dLoc = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	VkDeviceSize fluxBufSize = 3 * (VkDeviceSize)totalPmlCells * sizeof(float);
	if (fluxBufSize == 0) return;
	for (size_t ri = 0; ri < regions.size(); ++ri)
	{
		VkDeviceSize regionEnd = ((VkDeviceSize)regions[ri].fluxOffset +
		                              3 * (VkDeviceSize)m_gpuUPML[ri].totalCells) * sizeof(float);
		if (regionEnd > fluxBufSize)
			throw std::runtime_error("Engine_Vulkan::SetupFusedUPML: packed region exceeds buffer");
	}

	// --- Create concatenated buffers ---
	CreateGpuBuf(m_fusedVoltFlux, fluxBufSize, rwUsage, dLoc);
	CreateGpuBuf(m_fusedCurrFlux, fluxBufSize, rwUsage, dLoc);
	CreateGpuBuf(m_fusedPmlVv,    fluxBufSize, usage, dLoc);
	CreateGpuBuf(m_fusedPmlVvfo,  fluxBufSize, usage, dLoc);
	CreateGpuBuf(m_fusedPmlVvfn,  fluxBufSize, usage, dLoc);
	CreateGpuBuf(m_fusedPmlIi,    fluxBufSize, usage, dLoc);
	CreateGpuBuf(m_fusedPmlIifo,  fluxBufSize, usage, dLoc);
	CreateGpuBuf(m_fusedPmlIifn,  fluxBufSize, usage, dLoc);

	// --- Upload concatenated data (copy from per-region buffers via staging) ---
	// Initialize flux to zero
	{
		std::vector<float> zeros(3 * totalPmlCells, 0.0f);
		UploadGpuBuf(m_fusedVoltFlux, zeros.data(), fluxBufSize);
		UploadGpuBuf(m_fusedCurrFlux, zeros.data(), fluxBufSize);
	}

	// Copy every region in one submission; startup latency otherwise scales
	// with the number of PML faces.
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		for (size_t ri = 0; ri < m_gpuUPML.size(); ++ri)
		{
			const auto& u = m_gpuUPML[ri];
			VkDeviceSize regionFluxSize = 3 * (VkDeviceSize)u.totalCells * sizeof(float);
			VkDeviceSize dstOffset = (VkDeviceSize)regions[ri].fluxOffset * sizeof(float);
			VkBufferCopy copyRegion = {0, dstOffset, regionFluxSize};
			vkCmdCopyBuffer(cmd, u.pmlVv.buffer,   m_fusedPmlVv.buffer,   1, &copyRegion);
			vkCmdCopyBuffer(cmd, u.pmlVvfo.buffer, m_fusedPmlVvfo.buffer, 1, &copyRegion);
			vkCmdCopyBuffer(cmd, u.pmlVvfn.buffer, m_fusedPmlVvfn.buffer, 1, &copyRegion);
			vkCmdCopyBuffer(cmd, u.pmlIi.buffer,   m_fusedPmlIi.buffer,   1, &copyRegion);
			vkCmdCopyBuffer(cmd, u.pmlIifo.buffer, m_fusedPmlIifo.buffer, 1, &copyRegion);
			vkCmdCopyBuffer(cmd, u.pmlIifn.buffer, m_fusedPmlIifn.buffer, 1, &copyRegion);
		}
	});

	// --- PML region metadata SSBO ---
	VkDeviceSize regionInfoSize = regions.size() * sizeof(PMLRegionGPU);
	CreateGpuBuf(m_fusedPmlRegionInfo, regionInfoSize, usage, dLoc);
	UploadGpuBuf(m_fusedPmlRegionInfo, regions.data(), regionInfoSize);

	// --- Descriptor set layout (10 bindings) ---
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings(10);
		for (uint32_t i = 0; i < 10; i++)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = 10;
		ci.pBindings    = bindings.data();
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_fusedDescLayout));
	}

	// --- Pipeline layout (push constant = FusedPC, 20 bytes) ---
	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FusedPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount         = 1;
	pli.pSetLayouts            = &m_fusedDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges    = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_fusedPipeLayout));

	// --- Descriptor pool (2 sets × 10 bindings) ---
	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 * 2};
	VkDescriptorPoolCreateInfo pi{};
	pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pi.maxSets       = 2;
	pi.poolSizeCount = 1;
	pi.pPoolSizes    = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &pi, nullptr, &m_fusedDescPool));

	// --- Allocate descriptor sets ---
	VkDescriptorSetLayout layouts[2] = {m_fusedDescLayout, m_fusedDescLayout};
	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_fusedDescPool;
	ai.descriptorSetCount = 2;
	ai.pSetLayouts        = layouts;
	VkDescriptorSet sets[2];
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, sets));
	m_fusedVoltDescSet = sets[0];
	m_fusedCurrDescSet = sets[1];

	// --- Write fused voltage descriptors ---
	{
		VkBuffer bufs[10] = {
			m_voltBuf, m_currBuf, m_opIndexBuf, m_vvCompBuf, m_viCompBuf,
			m_fusedPmlRegionInfo.buffer, m_fusedVoltFlux.buffer,
			m_fusedPmlVv.buffer, m_fusedPmlVvfo.buffer, m_fusedPmlVvfn.buffer
		};
		VkDeviceSize sizes[10] = {
			m_fieldBufSize, m_fieldBufSize, m_opIndexBufSize,
			m_coeffCompBufSize, m_coeffCompBufSize,
			regionInfoSize, fluxBufSize,
			fluxBufSize, fluxBufSize, fluxBufSize
		};
		WriteDescriptorBuffers(m_device, m_fusedVoltDescSet, bufs, sizes, 10);
	}

	// --- Write fused current descriptors ---
	{
		VkBuffer bufs[10] = {
			m_currBuf, m_voltBuf, m_opIndexBuf, m_iiCompBuf, m_ivCompBuf,
			m_fusedPmlRegionInfo.buffer, m_fusedCurrFlux.buffer,
			m_fusedPmlIi.buffer, m_fusedPmlIifo.buffer, m_fusedPmlIifn.buffer
		};
		VkDeviceSize sizes[10] = {
			m_fieldBufSize, m_fieldBufSize, m_opIndexBufSize,
			m_coeffCompBufSize, m_coeffCompBufSize,
			regionInfoSize, fluxBufSize,
			fluxBufSize, fluxBufSize, fluxBufSize
		};
		WriteDescriptorBuffers(m_device, m_fusedCurrDescSet, bufs, sizes, 10);
	}

	// --- Create compute pipelines ---
	auto makePipeline = [&](VkShaderModule mod, VkPipelineLayout layout) -> VkPipeline
	{
		VkPipelineShaderStageCreateInfo stage{};
		stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = mod;
		stage.pName  = "main";
		VkComputePipelineCreateInfo ci{};
		ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ci.stage  = stage;
		ci.layout = layout;
		VkPipeline pipe;
		VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &pipe));
		return pipe;
	};

	VkShaderModule mod;
	mod = CreateShaderModule(gpu_spirv::fused_update_voltages_data,
	                         gpu_spirv::fused_update_voltages_size);
	m_fusedVoltPipeline = makePipeline(mod, m_fusedPipeLayout);
	vkDestroyShaderModule(m_device, mod, nullptr);

	mod = CreateShaderModule(gpu_spirv::fused_update_currents_data,
	                         gpu_spirv::fused_update_currents_size);
	m_fusedCurrPipeline = makePipeline(mod, m_fusedPipeLayout);
	vkDestroyShaderModule(m_device, mod, nullptr);

	m_hasFusedUPML = true;
	// The fused path owns its concatenated state from this point onward.
	// Release the equivalent per-region allocations retained for fallback.
	for (auto& u : m_gpuUPML)
	{
		DestroyGpuBuf(u.voltFlux); DestroyGpuBuf(u.currFlux);
		DestroyGpuBuf(u.pmlVv);   DestroyGpuBuf(u.pmlVvfo); DestroyGpuBuf(u.pmlVvfn);
		DestroyGpuBuf(u.pmlIi);   DestroyGpuBuf(u.pmlIifo); DestroyGpuBuf(u.pmlIifn);
	}
	cout << "Engine_Vulkan: fused Yee+UPML kernels created ("
	     << regions.size() << " regions, " << totalPmlCells << " PML cells)" << endl;
}

void Engine_Vulkan::CleanupFusedUPML()
{
	if (m_device == VK_NULL_HANDLE) return;
	DestroyGpuBuf(m_fusedVoltFlux);
	DestroyGpuBuf(m_fusedCurrFlux);
	DestroyGpuBuf(m_fusedPmlVv);
	DestroyGpuBuf(m_fusedPmlVvfo);
	DestroyGpuBuf(m_fusedPmlVvfn);
	DestroyGpuBuf(m_fusedPmlIi);
	DestroyGpuBuf(m_fusedPmlIifo);
	DestroyGpuBuf(m_fusedPmlIifn);
	DestroyGpuBuf(m_fusedPmlRegionInfo);
	if (m_fusedVoltPipeline) { vkDestroyPipeline(m_device, m_fusedVoltPipeline, nullptr); m_fusedVoltPipeline = VK_NULL_HANDLE; }
	if (m_fusedCurrPipeline) { vkDestroyPipeline(m_device, m_fusedCurrPipeline, nullptr); m_fusedCurrPipeline = VK_NULL_HANDLE; }
	if (m_fusedPipeLayout)   { vkDestroyPipelineLayout(m_device, m_fusedPipeLayout, nullptr); m_fusedPipeLayout = VK_NULL_HANDLE; }
	if (m_fusedDescLayout)   { vkDestroyDescriptorSetLayout(m_device, m_fusedDescLayout, nullptr); m_fusedDescLayout = VK_NULL_HANDLE; }
	if (m_fusedDescPool)     { vkDestroyDescriptorPool(m_device, m_fusedDescPool, nullptr); m_fusedDescPool = VK_NULL_HANDLE; }
	m_fusedVoltDescSet = m_fusedCurrDescSet = VK_NULL_HANDLE;
	m_hasFusedUPML = false;
}

// ===========================================================================
// GPU Energy Reduction
// ===========================================================================

void Engine_Vulkan::SetupGPU_EnergyReduction()
{
	VkDeviceSize partialSize = 2 * ENERGY_NUM_WG * sizeof(float);
	VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	// The reduction writes only 2 KiB. Keep that output directly CPU-visible
	// even when the large field buffers stay in device-local memory on a dGPU.
	CreateGpuBuf(m_energyPartialBuf, partialSize,
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible);
	VK_CHECK(vkMapMemory(m_device, m_energyPartialBuf.memory, 0, partialSize, 0,
	                     (void**)&m_energyMapped));
	// Descriptor layout: 3 bindings (volt, curr, partials)
	auto makeLayout = [&](uint32_t nBindings) -> VkDescriptorSetLayout
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings(nBindings);
		for (uint32_t i = 0; i < nBindings; i++)
			bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo ci{};
		ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		ci.bindingCount = nBindings;
		ci.pBindings    = bindings.data();
		VkDescriptorSetLayout layout;
		VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &layout));
		return layout;
	};
	m_energyDescLayout = makeLayout(3);

	// Pipeline layout: push constant = EnergyPC (8 bytes)
	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(EnergyPC)};
	VkPipelineLayoutCreateInfo pli{};
	pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount         = 1;
	pli.pSetLayouts            = &m_energyDescLayout;
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges    = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pli, nullptr, &m_energyPipeLayout));

	// Descriptor pool + set
	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
	VkDescriptorPoolCreateInfo dpi{};
	dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpi.maxSets       = 1;
	dpi.poolSizeCount = 1;
	dpi.pPoolSizes    = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_energyDescPool));

	VkDescriptorSetAllocateInfo ai{};
	ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool     = m_energyDescPool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts        = &m_energyDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &ai, &m_energyDescSet));

	// Write descriptors: volt, curr, partials
	VkBuffer      bufs[3] = {m_voltBuf, m_currBuf, m_energyPartialBuf.buffer};
	VkDeviceSize sizes[3] = {m_fieldBufSize, m_fieldBufSize, partialSize};
	WriteDescriptorBuffers(m_device, m_energyDescSet, bufs, sizes, 3);

	// Compute pipeline
	VkShaderModule mod = CreateShaderModule(gpu_spirv::reduce_energy_data,
	                                        gpu_spirv::reduce_energy_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = mod;
	stage.pName  = "main";

	VkComputePipelineCreateInfo ci{};
	ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage  = stage;
	ci.layout = m_energyPipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &ci, nullptr, &m_energyPipeline));
	vkDestroyShaderModule(m_device, mod, nullptr);

}

void Engine_Vulkan::SetupGPU_FieldValidation()
{
	const VkDeviceSize resultSize = (m_validateStages ? 6u : 1u) * 16u * sizeof(uint32_t);
	const VkMemoryPropertyFlags hostVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	CreateGpuBuf(m_validateBadBuf, resultSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostVisible);
	VK_CHECK(vkMapMemory(m_device, m_validateBadBuf.memory, 0, resultSize, 0,
	                     (void**)&m_validateBadMapped));

	VkDescriptorSetLayoutBinding bindings[6] = {};
	for (uint32_t i = 0; i < 6; ++i)
		bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 6;
	layoutInfo.pBindings = bindings;
	VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_validateDescLayout));

	VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ValidatePC)};
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &m_validateDescLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pcRange;
	VK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_validatePipeLayout));

	VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	VK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_validateDescPool));

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = m_validateDescPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &m_validateDescLayout;
	VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_validateDescSet));

	VkBuffer buffers[6] = {m_voltBuf, m_currBuf, m_validateBadBuf.buffer,
	                       m_opIndexBuf, m_vvCompBuf, m_viCompBuf};
	VkDeviceSize sizes[6] = {m_fieldBufSize, m_fieldBufSize, resultSize,
	                         m_opIndexBufSize, m_coeffCompBufSize, m_coeffCompBufSize};
	WriteDescriptorBuffers(m_device, m_validateDescSet, buffers, sizes, 6);

	VkShaderModule module = CreateShaderModule(gpu_spirv::validate_fields_data,
	                                           gpu_spirv::validate_fields_size);
	VkPipelineShaderStageCreateInfo stage{};
	stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = module;
	stage.pName = "main";

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage = stage;
	pipelineInfo.layout = m_validatePipeLayout;
	VK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo,
	                                  nullptr, &m_validatePipeline));
	vkDestroyShaderModule(m_device, module, nullptr);
}

void Engine_Vulkan::CleanupGPU_FieldValidation()
{
	if (m_device == VK_NULL_HANDLE) return;
	if (m_validateBadMapped)
	{
		vkUnmapMemory(m_device, m_validateBadBuf.memory);
		m_validateBadMapped = nullptr;
	}
	DestroyGpuBuf(m_validateBadBuf);
	if (m_validatePipeline)   { vkDestroyPipeline(m_device, m_validatePipeline, nullptr); m_validatePipeline = VK_NULL_HANDLE; }
	if (m_validatePipeLayout) { vkDestroyPipelineLayout(m_device, m_validatePipeLayout, nullptr); m_validatePipeLayout = VK_NULL_HANDLE; }
	if (m_validateDescLayout) { vkDestroyDescriptorSetLayout(m_device, m_validateDescLayout, nullptr); m_validateDescLayout = VK_NULL_HANDLE; }
	if (m_validateDescPool)   { vkDestroyDescriptorPool(m_device, m_validateDescPool, nullptr); m_validateDescPool = VK_NULL_HANDLE; }
	m_validateDescSet = VK_NULL_HANDLE;
}

void Engine_Vulkan::CleanupGPU_EnergyReduction()
{
	if (m_device == VK_NULL_HANDLE) return;
	if (m_energyMapped)
	{
		vkUnmapMemory(m_device, m_energyPartialBuf.memory);
		m_energyMapped = nullptr;
	}
	DestroyGpuBuf(m_energyPartialBuf);

	if (m_energyPipeline)   { vkDestroyPipeline(m_device, m_energyPipeline, nullptr); m_energyPipeline = VK_NULL_HANDLE; }
	if (m_energyPipeLayout) { vkDestroyPipelineLayout(m_device, m_energyPipeLayout, nullptr); m_energyPipeLayout = VK_NULL_HANDLE; }
	if (m_energyDescLayout) { vkDestroyDescriptorSetLayout(m_device, m_energyDescLayout, nullptr); m_energyDescLayout = VK_NULL_HANDLE; }
	if (m_energyDescPool)   { vkDestroyDescriptorPool(m_device, m_energyDescPool, nullptr); m_energyDescPool = VK_NULL_HANDLE; }
	m_energyDescSet = VK_NULL_HANDLE; // freed with pool
}

double Engine_Vulkan::CalcFastEnergyGPU() const
{
	// Ensure GPU compute is done
	DrainGPU();

	EnergyPC pc = {m_fieldN, ENERGY_NUM_WG};

	// Dispatch energy reduction
	RunSingleCommand([&](VkCommandBuffer cmd)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_energyPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
		                        m_energyPipeLayout, 0, 1, &m_energyDescSet, 0, nullptr);
		vkCmdPushConstants(cmd, m_energyPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(EnergyPC), &pc);
		vkCmdDispatch(cmd, ENERGY_NUM_WG, 1, 1);

		VkMemoryBarrier hostBarrier{};
		hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
			0, 1, &hostBarrier, 0, nullptr, 0, nullptr);
	});

	// Read back the partial sums
	VkDeviceSize partialSize = 2 * ENERGY_NUM_WG * sizeof(float);
	float partials[2 * ENERGY_NUM_WG];

	memcpy(partials, m_energyMapped, partialSize);

	// CPU-side reduction of the partial sums (256 E + 256 H floats)
	double E_energy = 0.0, H_energy = 0.0;
	for (uint32_t i = 0; i < ENERGY_NUM_WG; ++i)
	{
		E_energy += (double)partials[i];
		H_energy += (double)partials[ENERGY_NUM_WG + i];
	}

	return EPS0 * E_energy + MUE0 * H_energy;
}

// ===========================================================================
// GPU Voltage/Current Integral (probe support)
// ===========================================================================

double Engine_Vulkan::CalcVoltageIntegralGPU(const unsigned int* start, const unsigned int* stop) const
{
	// For voltage/current probes: read directly from mapped VRAM if ReBAR is available,
	// or do a targeted small read.  The key insight is that probes only access a few cells
	// along a line — we just need DrainGPU(), then read directly.  With ReBAR, this is
	// zero-copy.  Without ReBAR, we fall back to the full sync (unavoidable without
	// a dedicated gather dispatch, but this is still better than the generic path because
	// we avoid going through virtual GetVolt calls).

	if (((start[0]!=stop[0]) + (start[1]!=stop[1]) + (start[2]!=stop[2]))!=1)
		return 0.0;

	// When recording probe accesses or reading from probe cache, go through
	// GetVolt() which handles recording and cache lookup.
	if (m_recordingProbeAccess || m_pipelinedReading)
	{
		double result = 0.0;
		for (int n = 0; n < 3; ++n)
		{
			if (start[n] < stop[n])
			{
				unsigned int pos[3] = {start[0], start[1], start[2]};
				for (; pos[n] < stop[n]; ++pos[n])
					result += (double)GetVolt(n, pos[0], pos[1], pos[2]);
			}
			else if (start[n] > stop[n])
			{
				unsigned int pos[3] = {stop[0], stop[1], stop[2]};
				for (; pos[n] < start[n]; ++pos[n])
					result -= (double)GetVolt(n, pos[0], pos[1], pos[2]);
			}
		}
		return result;
	}

	DrainGPU();

	double result = 0.0;
	uint32_t N   = m_fieldN;
	uint32_t sYZ = m_strideYZ;

	if (m_voltMapped)
	{
		// ReBAR: direct VRAM read — no download needed
		for (int n = 0; n < 3; ++n)
		{
			if (start[n] < stop[n])
			{
				unsigned int pos[3] = {start[0], start[1], start[2]};
				for (; pos[n] < stop[n]; ++pos[n])
					result += (double)m_voltMapped[n * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2]];
			}
			else if (start[n] > stop[n])
			{
				unsigned int pos[3] = {stop[0], stop[1], stop[2]};
				for (; pos[n] < start[n]; ++pos[n])
					result -= (double)m_voltMapped[n * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2]];
			}
		}
	}
	else
	{
		// No ReBAR — must sync full fields (cached after first call per batch)
		if (m_hostDirty) SyncFieldsToHost();
		for (int n = 0; n < 3; ++n)
		{
			if (start[n] < stop[n])
			{
				unsigned int pos[3] = {start[0], start[1], start[2]};
				for (; pos[n] < stop[n]; ++pos[n])
					result += Engine::GetVolt(n, pos[0], pos[1], pos[2]);
			}
			else if (start[n] > stop[n])
			{
				unsigned int pos[3] = {stop[0], stop[1], stop[2]};
				for (; pos[n] < start[n]; ++pos[n])
					result -= Engine::GetVolt(n, pos[0], pos[1], pos[2]);
			}
		}
	}
	return result;
}

double Engine_Vulkan::CalcCurrentIntegralGPU(const unsigned int* start, const unsigned int* stop) const
{
	if (((start[0]!=stop[0]) + (start[1]!=stop[1]) + (start[2]!=stop[2]))!=1)
		return 0.0;

	// When recording or in pipelined mode, go through GetCurr() for cache/recording
	if (m_recordingProbeAccess || m_pipelinedReading)
	{
		double result = 0.0;
		for (int n = 0; n < 3; ++n)
		{
			if (start[n] < stop[n])
			{
				unsigned int pos[3] = {start[0], start[1], start[2]};
				for (; pos[n] < stop[n]; ++pos[n])
					result += (double)GetCurr(n, pos[0], pos[1], pos[2]);
			}
			else if (start[n] > stop[n])
			{
				unsigned int pos[3] = {stop[0], stop[1], stop[2]};
				for (; pos[n] < start[n]; ++pos[n])
					result -= (double)GetCurr(n, pos[0], pos[1], pos[2]);
			}
		}
		return result;
	}

	DrainGPU();

	double result = 0.0;
	uint32_t N   = m_fieldN;
	uint32_t sYZ = m_strideYZ;

	if (m_currMapped)
	{
		for (int n = 0; n < 3; ++n)
		{
			if (start[n] < stop[n])
			{
				unsigned int pos[3] = {start[0], start[1], start[2]};
				for (; pos[n] < stop[n]; ++pos[n])
					result += (double)m_currMapped[n * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2]];
			}
			else if (start[n] > stop[n])
			{
				unsigned int pos[3] = {stop[0], stop[1], stop[2]};
				for (; pos[n] < start[n]; ++pos[n])
					result -= (double)m_currMapped[n * N + pos[0] * sYZ + pos[1] * numLines[2] + pos[2]];
			}
		}
	}
	else
	{
		if (m_hostDirty) SyncFieldsToHost();
		for (int n = 0; n < 3; ++n)
		{
			if (start[n] < stop[n])
			{
				unsigned int pos[3] = {start[0], start[1], start[2]};
				for (; pos[n] < stop[n]; ++pos[n])
					result += Engine::GetCurr(n, pos[0], pos[1], pos[2]);
			}
			else if (start[n] > stop[n])
			{
				unsigned int pos[3] = {stop[0], stop[1], stop[2]};
				for (; pos[n] < start[n]; ++pos[n])
					result -= Engine::GetCurr(n, pos[0], pos[1], pos[2]);
			}
		}
	}
	return result;
}

#endif // WITH_GPU
