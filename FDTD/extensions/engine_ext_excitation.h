/*
*	Copyright (C) 2011 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef ENGINE_EXT_EXCITATION_H
#define ENGINE_EXT_EXCITATION_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_Excitation;

class Engine_Ext_Excitation : public Engine_Extension
{
public:
	Engine_Ext_Excitation(Operator_Ext_Excitation* op_ext);
	virtual ~Engine_Ext_Excitation();

	virtual void Apply2Voltages();
	virtual void Apply2Current();

	//! The excitation is a plain per-cell add with no cross-cell coupling and no
	//! internal state, so restricting it to an x-range at a chosen timestep is
	//! exact -- which is what lets a temporally blocked engine carry it.
	virtual bool SupportsSlabApply() const {return true;}
	virtual unsigned int SlabHookMask() const {return SLAB_APPLY_VOLT | SLAB_APPLY_CURR;}
	virtual void Apply2VoltagesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);
	virtual void Apply2CurrentSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);

protected:
	//! Slab pattern A over the excitation's cell list, striped by thread.
	/*!
	  The list is not sorted by x, so there is no contiguous share to hand a
	  thread; each takes every m_NrThreads'th entry instead. The entries are
	  independent per-cell adds, so any partition gives the same result, and a
	  list this short does not repay an index.
	*/
	template <typename EngType>
	void Apply2VoltagesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int numTS,
	                        unsigned int first, unsigned int stride);

	template <typename EngType>
	void Apply2CurrentImpl(EngType* eng, unsigned int startX, unsigned int stopX, int numTS,
	                       unsigned int first, unsigned int stride);

	Operator_Ext_Excitation* m_Op_Exc;
};

#endif // ENGINE_EXT_EXCITATION_H
