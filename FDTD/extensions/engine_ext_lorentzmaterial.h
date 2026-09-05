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

#ifndef ENGINE_EXT_LORENTZMATERIAL_H
#define ENGINE_EXT_LORENTZMATERIAL_H

#include "engine_ext_dispersive.h"

class Operator_Ext_LorentzMaterial;

class Engine_Ext_LorentzMaterial : public Engine_Ext_Dispersive
{
public:
	Engine_Ext_LorentzMaterial(Operator_Ext_LorentzMaterial* op_ext_lorentz);
	virtual ~Engine_Ext_LorentzMaterial();

	virtual void DoPreVoltageUpdates();

	virtual void DoPreCurrentUpdates();

	//! SupportsSlabApply() stays true: the two hooks added here are per-cell
	//! recursions over the same list, and both have a slab form below.
	//! Dispersive's two Apply hooks, plus the two this class adds. Inheriting
	//! the base mask unchanged would leave these two never called.
	virtual unsigned int SlabHookMask() const
	{return Engine_Ext_Dispersive::SlabHookMask() | SLAB_PRE_VOLT | SLAB_PRE_CURR;}

	virtual void DoPreVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);
	virtual void DoPreCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);

protected:
	//! \copydoc Engine_Ext_Dispersive::Apply2VoltagesImpl
	template <typename EngType>
	void DoPreVoltageUpdatesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID);

	template <typename EngType>
	void DoPreCurrentUpdatesImpl(EngType* eng, unsigned int startX, unsigned int stopX, int threadID);

	Operator_Ext_LorentzMaterial* m_Op_Ext_Lor;

	//! ADE Lorentz voltages
	// Array setup: volt_Lor_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***volt_Lor_ADE;

	//! ADE Lorentz currents
	// Array setup: curr_Lor_ADE[N_order][direction][mesh_pos]
	FDTD_FLOAT ***curr_Lor_ADE;

};

#endif // ENGINE_EXT_LORENTZMATERIAL_H
