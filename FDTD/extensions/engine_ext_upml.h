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

#ifndef ENGINE_EXT_UPML_H
#define ENGINE_EXT_UPML_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_UPML;
union f8vector;

class Engine_Ext_UPML : public Engine_Extension
{
public:
	Engine_Ext_UPML(Operator_Ext_UPML* op_ext);
	virtual ~Engine_Ext_UPML();

	virtual void SetEngine(Engine* eng);
	virtual void SetNumberOfThreads(int nrThread);

	virtual void DoPreVoltageUpdates() {Engine_Ext_UPML::DoPreVoltageUpdates(0);};
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() {Engine_Ext_UPML::DoPostVoltageUpdates(0);};
	virtual void DoPostVoltageUpdates(int threadID);

	virtual void DoPreCurrentUpdates() {Engine_Ext_UPML::DoPreCurrentUpdates(0);};
	virtual void DoPreCurrentUpdates(int threadID);
	virtual void DoPostCurrentUpdates() {Engine_Ext_UPML::DoPostCurrentUpdates(0);};
	virtual void DoPostCurrentUpdates(int threadID);

protected:
	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void DoPostVoltageUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void DoPreCurrentUpdatesImpl(EngType* eng, int threadID);

	template <typename EngType>
	void DoPostCurrentUpdatesImpl(EngType* eng, int threadID);

	Operator_Ext_UPML* m_Op_UPML;

	std::vector<unsigned int> m_start;
	std::vector<unsigned int> m_numX;

	ArrayLib::ArrayNIJK<FDTD_FLOAT> volt_flux;
	ArrayLib::ArrayNIJK<FDTD_FLOAT> curr_flux;

#if OPENEMS_ENABLE_AVX2
	// --- AVX2 fast path (engine_ext_upml_avx2.cpp) ---
	//
	// Engine_AVX2 splits the z axis into 8 contiguous lanes of numVectors
	// cells each: global z lives at vector index (z % numVectors), lane
	// (z / numVectors). A PML box therefore lines up with whole f8vectors
	// only if it spans the full z range -- which the four x- and y-normal
	// boxes do (see Operator_Ext_UPML::Create_UPML). For those, the six
	// coefficient arrays and the two flux arrays are repacked into the
	// engine's own layout and the updates run 8 cells at a time. The two
	// z-normal slabs keep the scalar path.
	//
	// Packed layout mirrors the engine exactly, with polarisation fastest:
	//   arr[n + v * 3 + j * m_v_vs_y + i * m_v_vs_x]   (i, j PML-local)
	// so the (v, n) loops fuse into one run of 3 * numVectors contiguous
	// f8vectors in both the field and the coefficient arrays.
	bool BuildAVX2Layout();
	void ReleaseAVX2Layout();

	void PreUpdateAVX2(f8vector* field, f8vector* flux,
	                   const f8vector* a, const f8vector* b, int threadID);
	void PostUpdateAVX2(f8vector* field, f8vector* flux,
	                    const f8vector* c, int threadID);

	void DoPreVoltageUpdatesAVX2(int threadID);
	void DoPostVoltageUpdatesAVX2(int threadID);
	void DoPreCurrentUpdatesAVX2(int threadID);
	void DoPostCurrentUpdatesAVX2(int threadID);

	bool m_avx2_packed;

	unsigned int m_v_numVectors;
	unsigned int m_v_pNy;          //!< PML box extent along y
	unsigned int m_v_sx, m_v_sy;   //!< PML box origin in engine coordinates
	size_t m_v_vs_y, m_v_vs_x;     //!< packed PML strides, in f8vector units
	size_t m_v_e_vs_y, m_v_e_vs_x; //!< engine strides, in f8vector units

	f8vector* m_v_vv;
	f8vector* m_v_vvfo;
	f8vector* m_v_vvfn;
	f8vector* m_v_ii;
	f8vector* m_v_iifo;
	f8vector* m_v_iifn;
	f8vector* m_v_volt_flux;
	f8vector* m_v_curr_flux;
#endif
};

#endif // ENGINE_EXT_UPML_H
