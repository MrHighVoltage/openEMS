/*
*	Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_EXT_DISPERSIVE_H
#define ENGINE_EXT_DISPERSIVE_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

#include <vector>
#include <cstddef>

class Operator_Ext_Dispersive;

class Engine_Ext_Dispersive : public Engine_Extension
{
public:
	Engine_Ext_Dispersive(Operator_Ext_Dispersive* op_ext_disp);
	virtual ~Engine_Ext_Dispersive();

	virtual void Apply2Voltages();
	virtual void Apply2Current();

protected:
	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng);

	template <typename EngType>
	void Apply2CurrentImpl(EngType* eng);

	Operator_Ext_Dispersive* m_Op_Ext_Disp;

	//! Dispersive order
	int m_Order;

	//! ADE currents
	// Array setup: curr_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***curr_ADE;

	//! ADE voltages
	// Array setup: volt_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***volt_ADE;

#if OPENEMS_ENABLE_AVX2
	// --- AVX2 flat-address cache -------------------------------------------
	// Engine_AVX2 maps a cell to m_volt[n + (z%numVectors)*m_vs_z + y*m_vs_y
	// + x*m_vs_x].f[z/numVectors]. numVectors is a runtime member, so every
	// GetVolt/SetVolt on that engine costs an integer div and mod that the
	// compiler cannot strength-reduce. The dispersive cell list is fixed once
	// the operator is built, so the vector offset and lane are precomputed
	// per list entry and the hot loops address the field arrays directly.
	// Concatenated over orders; m_avx2_start[o] is where order o begins.
	std::vector<unsigned int>  m_avx2_off;
	std::vector<unsigned char> m_avx2_lane;
	std::vector<size_t>        m_avx2_start;
	//! numVectors the cache was built for; 0 = not built yet.
	unsigned int m_avx2_idx_nv = 0;

	//! Build (or rebuild) the flat-address cache for this engine.
	void BuildAVX2Index(class Engine_AVX2* eng);
#endif
};

#endif // ENGINE_EXT_DISPERSIVE_H
