/*
*	Copyright (C) 2026 openEMS contributors
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*/

#ifndef OPERATOR_VULKAN_H
#define OPERATOR_VULKAN_H

#ifdef WITH_GPU

#include "operator.h"
#include <vector>
#include <cstdint>

/*!
 * \brief Operator companion for Engine_Vulkan with coefficient compression.
 *
 * After CalcECOperator(), scans all cells to find unique coefficient
 * 12-tuples {vv[3], vi[3], ii[3], iv[3]} and builds:
 *   - m_OpIndex[N]: one uint32 per cell -> compressed table index
 *   - m_vvComp[3 * numUnique], m_viComp[...], m_iiComp[...], m_ivComp[...]
 *
 * The compressed coefficient tables are tiny (typically < 100 KB) and
 * fit entirely in GPU L2 cache, eliminating VRAM round-trips for every cell.
 */
class Operator_Vulkan : public Operator
{
	friend class Engine_Vulkan;
public:
	static Operator_Vulkan* New();
	virtual ~Operator_Vulkan();
	virtual Engine* CreateEngine();
	virtual void ShowStat() const;

	unsigned int GetNumCompressed() const { return m_numCompressed; }
	const uint32_t* GetOpIndex() const { return m_OpIndex.data(); }
	const float* GetVVComp() const { return m_vvComp.data(); }
	const float* GetVIComp() const { return m_viComp.data(); }
	const float* GetIIComp() const { return m_iiComp.data(); }
	const float* GetIVComp() const { return m_ivComp.data(); }

protected:
	Operator_Vulkan();
	virtual int CalcECOperator(DebugFlags debugFlags = None);

private:
	void CompressOperator();

	unsigned int m_numCompressed;

	//! Per-cell index into compressed tables.  Size: Nx*Ny*Nz.
	std::vector<uint32_t> m_OpIndex;

	//! Compressed coefficient tables.  Layout: comp[n * numCompressed + idx].
	//! Size: 3 * numCompressed each.
	std::vector<float> m_vvComp, m_viComp, m_iiComp, m_ivComp;
};

#endif // WITH_GPU
#endif // OPERATOR_VULKAN_H
