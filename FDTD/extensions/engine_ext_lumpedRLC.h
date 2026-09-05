/*
*	Copyright (C) 2023 Gadi Lahav (gadi@rfwithcare.com)
*	Copyright (C) 2026 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef ENGINE_EXT_LUMPEDRLC_H
#define ENGINE_EXT_LUMPEDRLC_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_LumpedRLC;

class Engine_Ext_LumpedRLC : public Engine_Extension
{
	friend class Operator_Ext_LumpedRLC;
	friend class Operator;
	friend class ContinuousStructure;

public:

	Engine_Ext_LumpedRLC(Operator_Ext_LumpedRLC *op_ext_RLC);
	virtual ~Engine_Ext_LumpedRLC();

	virtual void DoPreVoltageUpdates();
	virtual void Apply2Voltages();

	//! Every lumped element is a per-cell recurrence over its own history, with
	//! no coupling to a neighbouring cell, so an x-range restriction is exact.
	virtual bool SupportsSlabApply() const {return true;}
	virtual unsigned int SlabHookMask() const {return SLAB_PRE_VOLT | SLAB_APPLY_VOLT;}
	virtual void DoPreVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);
	virtual void Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);

protected:
	void DoPreVoltageUpdatesImpl(int numTS, unsigned int startX, unsigned int stopX);

	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng, int numTS, unsigned int startX, unsigned int stopX);

	//! Ring slot holding the quantity from \a back timesteps before \a numTS.
	/*!
	  The three history slots used to be rotated by swapping pointers once per
	  call, which encodes "one call is one timestep" -- true for the flat sweep,
	  false for the blocked one, where the same timestep is visited once per
	  tile and tiles are at different timesteps at once. Addressing the ring by
	  absolute timestep instead makes the state independent of call order, which
	  is what the slab contract requires. On the flat path it is the identical
	  sequence of reads and writes.
	*/
	static inline int Slot(int numTS, int back) {return ((numTS - back) % 3 + 3) % 3;}

	Operator_Ext_LumpedRLC* m_Op_Ext_RLC;

	// Auxilliary containers

	// Array setup: volt_C_ADE[mesh_pos]
	FDTD_FLOAT *v_Il;		// Container for current on inductor- Parallel RLC

	FDTD_FLOAT **v_Vdn;		// Nodal vd, slot = timestep mod 3 (see Slot())
	FDTD_FLOAT **v_Jn;		// Nodal J,  slot = timestep mod 3 (see Slot())

};

#endif // ENGINE_EXT_LUMPEDRLC_H
