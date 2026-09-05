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
#include <atomic>
#include <mutex>

class Operator_Ext_Dispersive;

class Engine_Ext_Dispersive : public Engine_Extension
{
public:
	Engine_Ext_Dispersive(Operator_Ext_Dispersive* op_ext_disp);
	virtual ~Engine_Ext_Dispersive();

	virtual void Apply2Voltages();
	virtual void Apply2Current();

	//! Every list entry is one cell's ADE state added back into that same cell,
	//! with no reach into a neighbour and no dependence on the absolute
	//! timestep, so an x-range can be applied on its own.
	/*!
	  Subclasses inherit this "yes". Engine_Ext_LorentzMaterial does so
	  deliberately -- it implements the two update hooks it adds. Anything else
	  deriving from this class must either implement its own slab hooks or
	  override this back to false, or the blocked schedule will silently skip
	  whatever it added.
	*/
	virtual bool SupportsSlabApply() const {return true;}
	virtual unsigned int SlabHookMask() const {return SLAB_APPLY_VOLT | SLAB_APPLY_CURR;}
	virtual void Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);
	virtual void Apply2CurrentSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);

protected:
	//! Slab pattern A over the dispersive cell list.
	/*!
	  \a threadID below zero means the whole-grid entry point: the whole list,
	  in list order, exactly as before. From zero up it means a slab, and then
	  only the entries with x in [startX,stopX) that fall in this thread's share
	  of that range are touched.
	*/
	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID);

	template <typename EngType>
	void Apply2CurrentImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID);

	//! Build the slab index, and with it the AVX2 cache below, exactly once.
	void BuildSlabIndex();

	//! Narrow a slab to this thread's x-share; false when it has nothing to do.
	bool SlabXShare(unsigned int startX, unsigned int stopX, int threadID,
	                unsigned int& xLo, unsigned int& xHi) const;

	//! This thread's entries of order \a o, as a range of \a perm (NULL = the
	//! list itself, in list order, for the whole-grid entry points).
	bool SlabEntries(int o, unsigned int xLo, unsigned int xHi, int threadID,
	                 const unsigned int*& perm, unsigned int& j0, unsigned int& j1) const;

	//! Where order \a o's entries with x >= \a x start in m_slab_perm.
	unsigned int SlabLowerBound(int o, unsigned int x) const
	{
		const std::vector<unsigned int>& xoff = m_slab_xoff[o];
		return x < xoff.size() ? xoff[x] : xoff.back();
	}

	Operator_Ext_Dispersive* m_Op_Ext_Disp;

	//! Dispersive order
	int m_Order;

	//! ADE currents
	// Array setup: curr_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***curr_ADE;

	//! ADE voltages
	// Array setup: volt_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***volt_ADE;

	// --- x-ordered index for the slab hooks --------------------------------
	// A slab wants the entries with x in [startX,stopX), and the cell list is
	// not indexed by x, so finding them by scanning would cost a full pass over
	// the list per slab per timestep -- and the blocked schedule calls these
	// hooks far more often than the flat sweep does. m_slab_perm holds the list
	// indices sorted by x (concatenated over orders, order o starting at
	// m_slab_start[o]) and m_slab_xoff[o][x] counts the entries before x, so a
	// slab is a contiguous run found by two lookups.
	//
	// The operator's arrays are shared with its subclasses and have several
	// parallel arrays hanging off the same indices, so this permutes here
	// rather than reordering there.
	std::vector<unsigned int> m_slab_perm;
	std::vector<size_t> m_slab_start;
	std::vector<std::vector<unsigned int> > m_slab_xoff;
	//! x-extent of the whole cell list, [lo,hi), shared by all orders.
	unsigned int m_slab_xlo = 0;
	unsigned int m_slab_xhi = 0;
	//! Guards the one-shot build; the slab hooks run on all threads at once.
	std::atomic<bool> m_slab_idx_built{false};
	std::mutex m_slab_idx_lock;

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
