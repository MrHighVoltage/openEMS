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

#include "engine_ext_dispersive.h"
#include "operator_ext_dispersive.h"
#include "FDTD/engine_sse.h"
#if OPENEMS_ENABLE_AVX2
#include "FDTD/engine_avx2.h"
#include <type_traits>
#endif

Engine_Ext_Dispersive::Engine_Ext_Dispersive(Operator_Ext_Dispersive* op_ext_disp) : Engine_Extension(op_ext_disp)
{
	m_Op_Ext_Disp = op_ext_disp;
	int order = m_Op_Ext_Disp->m_Order;
	curr_ADE = new FDTD_FLOAT**[order];
	volt_ADE = new FDTD_FLOAT**[order];
	for (int o=0;o<order;++o)
	{
		curr_ADE[o] = new FDTD_FLOAT*[3];
		volt_ADE[o] = new FDTD_FLOAT*[3];
		for (int n=0; n<3; ++n)
		{
			if (m_Op_Ext_Disp->m_curr_ADE_On[o]==true)
			{
				curr_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Disp->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count[o]; ++i)
					curr_ADE[o][n][i]=0.0;
			}
			else
				curr_ADE[o][n] = NULL;
			if (m_Op_Ext_Disp->m_volt_ADE_On[o]==true)
			{
				volt_ADE[o][n] = new FDTD_FLOAT[m_Op_Ext_Disp->m_LM_Count[o]];
				for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count[o]; ++i)
					volt_ADE[o][n][i]=0.0;
			}
			else
				volt_ADE[o][n] = NULL;
		}
	}
}

Engine_Ext_Dispersive::~Engine_Ext_Dispersive()
{
	if (curr_ADE==NULL && volt_ADE==NULL)
		return;

	for (int o=0;o<m_Op_Ext_Disp->m_Order;++o)
	{
		for (int n=0; n<3; ++n)
		{
			delete[] curr_ADE[o][n];
			delete[] volt_ADE[o][n];
		}
		delete[] curr_ADE[o];
		delete[] volt_ADE[o];
	}
	delete[] curr_ADE;
	curr_ADE=NULL;

	delete[] volt_ADE;
	volt_ADE=NULL;
}

#if OPENEMS_ENABLE_AVX2
void Engine_Ext_Dispersive::BuildAVX2Index(Engine_AVX2* eng)
{
	const unsigned int nv = eng->GetNumVectors();
	if (m_avx2_idx_nv == nv)
		return;

	size_t total = 0;
	m_avx2_start.assign(m_Order, 0);
	for (int o=0; o<m_Order; ++o)
	{
		m_avx2_start[o] = total;
		total += m_Op_Ext_Disp->m_LM_Count.at(o);
	}
	m_avx2_off.resize(total);
	m_avx2_lane.resize(total);

	for (int o=0; o<m_Order; ++o)
	{
		unsigned int **pos = m_Op_Ext_Disp->m_LM_pos[o];
		const size_t base = m_avx2_start[o];
		for (unsigned int i=0; i<m_Op_Ext_Disp->m_LM_Count.at(o); ++i)
		{
			const unsigned int z = pos[2][i];
			m_avx2_off[base+i]  = (z % nv) * eng->m_vs_z
			                    + pos[1][i] * eng->m_vs_y
			                    + pos[0][i] * eng->m_vs_x;
			m_avx2_lane[base+i] = (unsigned char)(z / nv);
		}
	}
	m_avx2_idx_nv = nv;
}
#endif

template <typename EngType>
void Engine_Ext_Dispersive::Apply2VoltagesImpl(EngType* eng)
{
#if OPENEMS_ENABLE_AVX2
	if constexpr (std::is_same<EngType, Engine_AVX2>::value)
		BuildAVX2Index(eng);
#endif
	for (int o=0;o<m_Op_Ext_Disp->m_Order;++o)
	{
		if (m_Op_Ext_Disp->m_volt_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Disp->m_LM_pos[o];
		const unsigned int cnt = m_Op_Ext_Disp->m_LM_Count.at(o);

#if OPENEMS_ENABLE_AVX2
		if constexpr (std::is_same<EngType, Engine_AVX2>::value)
		{
			// Division-free path: address the f8vectors directly through the
			// cache built by BuildAVX2Index(). Same operands, same order.
			const unsigned int*  off  = m_avx2_off.data()  + m_avx2_start[o];
			const unsigned char* lane = m_avx2_lane.data() + m_avx2_start[o];
			f8vector* volt = eng->m_volt;
			for (unsigned int i=0; i<cnt; ++i)
			{
				f8vector* c = volt + off[i];
				const unsigned int l = lane[i];
				c[0].f[l] -= volt_ADE[o][0][i];
				c[1].f[l] -= volt_ADE[o][1][i];
				c[2].f[l] -= volt_ADE[o][2][i];
			}
			continue;
		}
#endif

		for (unsigned int i=0; i<cnt; ++i)
		{
			eng->EngType::SetVolt(0,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetVolt(0,pos[0][i],pos[1][i],pos[2][i]) - volt_ADE[o][0][i]
			);
			eng->EngType::SetVolt(1,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetVolt(1,pos[0][i],pos[1][i],pos[2][i]) - volt_ADE[o][1][i]
			);
			eng->EngType::SetVolt(2,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetVolt(2,pos[0][i],pos[1][i],pos[2][i]) - volt_ADE[o][2][i]
			);
		}
	}
}

void Engine_Ext_Dispersive::Apply2Voltages()
{
	ENG_DISPATCH(Apply2VoltagesImpl);
}

template <typename EngType>
void Engine_Ext_Dispersive::Apply2CurrentImpl(EngType* eng)
{
#if OPENEMS_ENABLE_AVX2
	if constexpr (std::is_same<EngType, Engine_AVX2>::value)
		BuildAVX2Index(eng);
#endif
	for (int o=0;o<m_Op_Ext_Disp->m_Order;++o)
	{
		if (m_Op_Ext_Disp->m_curr_ADE_On[o]==false) continue;

		unsigned int **pos = m_Op_Ext_Disp->m_LM_pos[o];
		const unsigned int cnt = m_Op_Ext_Disp->m_LM_Count.at(o);

#if OPENEMS_ENABLE_AVX2
		if constexpr (std::is_same<EngType, Engine_AVX2>::value)
		{
			const unsigned int*  off  = m_avx2_off.data()  + m_avx2_start[o];
			const unsigned char* lane = m_avx2_lane.data() + m_avx2_start[o];
			f8vector* curr = eng->m_curr;
			for (unsigned int i=0; i<cnt; ++i)
			{
				f8vector* c = curr + off[i];
				const unsigned int l = lane[i];
				c[0].f[l] -= curr_ADE[o][0][i];
				c[1].f[l] -= curr_ADE[o][1][i];
				c[2].f[l] -= curr_ADE[o][2][i];
			}
			continue;
		}
#endif

		for (unsigned int i=0; i<cnt; ++i)
		{
			eng->EngType::SetCurr(0,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetCurr(0,pos[0][i],pos[1][i],pos[2][i]) - curr_ADE[o][0][i]
			);
			eng->EngType::SetCurr(1,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetCurr(1,pos[0][i],pos[1][i],pos[2][i]) - curr_ADE[o][1][i]
			);
			eng->EngType::SetCurr(2,pos[0][i],pos[1][i],pos[2][i],
				eng->EngType::GetCurr(2,pos[0][i],pos[1][i],pos[2][i]) - curr_ADE[o][2][i]
			);
		}
	}
}

void Engine_Ext_Dispersive::Apply2Current()
{
	ENG_DISPATCH(Apply2CurrentImpl);
}
