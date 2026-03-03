/*
*	Copyright (C) 2026 openEMS contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifdef WITH_GPU

#include "operator_vulkan.h"
#include "engine_vulkan.h"

#include <iostream>
#include <map>
#include <cstring>
#include <vector>

using std::cout;
using std::endl;

// ---------------------------------------------------------------------------
// Compression key: 12 floats (vv[3], vi[3], ii[3], iv[3]) per cell.
// Using a struct with operator< for std::map ordering.
// ---------------------------------------------------------------------------
struct VulkanCoeffKey
{
	float vv[3], vi[3], ii[3], iv[3];

	bool operator<(const VulkanCoeffKey& o) const
	{
		return memcmp(this, &o, sizeof(VulkanCoeffKey)) < 0;
	}
};

// ---------------------------------------------------------------------------

Operator_Vulkan::Operator_Vulkan() : Operator()
{
	m_numCompressed = 0;
}

Operator_Vulkan::~Operator_Vulkan()
{
}

Operator_Vulkan* Operator_Vulkan::New()
{
	cout << "Create Vulkan GPU operator" << endl;
	Operator_Vulkan* op = new Operator_Vulkan();
	op->Init();
	return op;
}

Engine* Operator_Vulkan::CreateEngine()
{
	m_Engine = Engine_Vulkan::New(this);
	return m_Engine;
}

int Operator_Vulkan::CalcECOperator(DebugFlags debugFlags)
{
	int errCode = Operator::CalcECOperator(debugFlags);
	CompressOperator();
	return errCode;
}

void Operator_Vulkan::CompressOperator()
{
	if (g_settings.GetVerboseLevel() > 0)
		cout << "Compressing the FDTD operator (GPU)... this may take a while..." << endl;

	size_t N = (size_t)numLines[0] * numLines[1] * numLines[2];

	m_OpIndex.resize(N);

	// Access base class coefficient arrays (N-I-J-K layout)
	const float* vv_data = (const float*)vv_ptr->data();
	const float* vi_data = (const float*)vi_ptr->data();
	const float* ii_data = (const float*)ii_ptr->data();
	const float* iv_data = (const float*)iv_ptr->data();

	std::map<VulkanCoeffKey, uint32_t> lookUpMap;
	std::vector<float> tmp_vv, tmp_vi, tmp_ii, tmp_iv;  // flat, 3 components interleaved

	for (size_t cell = 0; cell < N; ++cell)
	{
		VulkanCoeffKey key;
		for (int n = 0; n < 3; ++n)
		{
			key.vv[n] = vv_data[n * N + cell];
			key.vi[n] = vi_data[n * N + cell];
			key.ii[n] = ii_data[n * N + cell];
			key.iv[n] = iv_data[n * N + cell];
		}

		auto it = lookUpMap.find(key);
		if (it == lookUpMap.end())
		{
			uint32_t idx = (uint32_t)(tmp_vv.size() / 3);
			for (int n = 0; n < 3; ++n)
			{
				tmp_vv.push_back(key.vv[n]);
				tmp_vi.push_back(key.vi[n]);
				tmp_ii.push_back(key.ii[n]);
				tmp_iv.push_back(key.iv[n]);
			}
			lookUpMap[key] = idx;
			m_OpIndex[cell] = idx;
		}
		else
		{
			m_OpIndex[cell] = it->second;
		}
	}

	m_numCompressed = (unsigned int)(tmp_vv.size() / 3);

	// Rearrange to N-major layout: comp[n * numCompressed + idx]
	// Currently tmp_vv is packed as [vv0_0, vv1_0, vv2_0, vv0_1, vv1_1, vv2_1, ...]
	// Need: [vv0_0, vv0_1, ..., vv0_M, vv1_0, vv1_1, ..., vv1_M, vv2_0, ...]
	m_vvComp.resize(3 * m_numCompressed);
	m_viComp.resize(3 * m_numCompressed);
	m_iiComp.resize(3 * m_numCompressed);
	m_ivComp.resize(3 * m_numCompressed);

	for (unsigned int idx = 0; idx < m_numCompressed; ++idx)
	{
		for (int n = 0; n < 3; ++n)
		{
			m_vvComp[n * m_numCompressed + idx] = tmp_vv[idx * 3 + n];
			m_viComp[n * m_numCompressed + idx] = tmp_vi[idx * 3 + n];
			m_iiComp[n * m_numCompressed + idx] = tmp_ii[idx * 3 + n];
			m_ivComp[n * m_numCompressed + idx] = tmp_iv[idx * 3 + n];
		}
	}

	if (g_settings.GetVerboseLevel() > 0)
	{
		size_t origBytes = 4 * 3 * N * sizeof(float);
		size_t compBytes = N * sizeof(uint32_t) + 4 * 3 * m_numCompressed * sizeof(float);
		cout << "  Unique coefficient sets: " << m_numCompressed
		     << " (compression ratio: " << (double)origBytes / compBytes << "x)" << endl;
		cout << "  Original: " << (origBytes / (1024*1024)) << " MB -> Compressed: "
		     << (compBytes / (1024*1024)) << " MB" << endl;
	}
}

void Operator_Vulkan::ShowStat() const
{
	Operator::ShowStat();

	cout << "GPU operator compression\t: yes" << endl;
	cout << "Unique coefficient sets\t\t: " << m_numCompressed << endl;
	cout << "-----------------------------------" << endl;
}

#endif // WITH_GPU
