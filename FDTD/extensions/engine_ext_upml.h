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

	//! Every UPML pass is a per-cell read-modify-write of one field cell and its
	//! own flux cell -- no neighbour is touched, in x or in anything else -- so
	//! restricting a pass to an x-range is exact, and slab pattern A applies.
	virtual bool SupportsSlabApply() const {return true;}
	virtual unsigned int SlabHookMask() const
	{return SLAB_PRE_VOLT | SLAB_POST_VOLT | SLAB_PRE_CURR | SLAB_POST_CURR;}

	virtual void DoPreVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);
	virtual void DoPostVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);
	virtual void DoPreCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);
	virtual void DoPostCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);

protected:
	// Both the scalar and the packed AVX2 passes below are parameterised by a
	// half-open range of PML-local x-lines rather than by a thread id, because
	// the two callers disagree about what a thread's share is: the flat path
	// hands out the whole box split m_NrThreads ways, the blocked path hands
	// out only the part of the box inside the current slab. Splitting the range
	// out of the kernels is what keeps one implementation serving both.

	//! This thread's share of the whole PML box, in PML-local x. False if none.
	bool ThreadLocalX(int threadID, unsigned int& locStart, unsigned int& locStop) const;
	//! This thread's share of the box's intersection with the slab. False if none.
	bool SlabLocalX(unsigned int startX, unsigned int stopX, int threadID,
	                unsigned int& locStart, unsigned int& locStop) const;

	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iEnd);

	template <typename EngType>
	void DoPostVoltageUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iEnd);

	template <typename EngType>
	void DoPreCurrentUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iEnd);

	template <typename EngType>
	void DoPostCurrentUpdatesImpl(EngType* eng, unsigned int iStart, unsigned int iEnd);

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
	                   const f8vector* a, const f8vector* b,
	                   unsigned int iStart, unsigned int iEnd);
	void PostUpdateAVX2(f8vector* field, f8vector* flux,
	                    const f8vector* c,
	                    unsigned int iStart, unsigned int iEnd);

	void DoPreVoltageUpdatesAVX2(unsigned int iStart, unsigned int iEnd);
	void DoPostVoltageUpdatesAVX2(unsigned int iStart, unsigned int iEnd);
	void DoPreCurrentUpdatesAVX2(unsigned int iStart, unsigned int iEnd);
	void DoPostCurrentUpdatesAVX2(unsigned int iStart, unsigned int iEnd);

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
