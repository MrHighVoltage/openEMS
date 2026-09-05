/*
*	Copyright (C) 2012 Thorsten Liebig (Thorsten.Liebig@gmx.de)
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

#ifndef ENGINE_EXT_TFSF_H
#define ENGINE_EXT_TFSF_H

#include "engine_extension.h"
#include "tools/constants.h"

class Operator_Ext_TFSF;

class Engine_Ext_TFSF : public Engine_Extension
{
public:
	Engine_Ext_TFSF(Operator_Ext_TFSF* op_ext);
	virtual ~Engine_Ext_TFSF();

	virtual void DoPostVoltageUpdates();
	virtual void DoPostCurrentUpdates();

	//! TF/SF only ever adds a precomputed, delayed excitation value to the six
	//! faces of its box -- it reads no field, so its reach into the grid is
	//! zero and it can be split at any x-range boundary.
	virtual bool SupportsSlabApply() const {return true;}
	virtual unsigned int SlabHookMask() const {return SLAB_POST_VOLT | SLAB_POST_CURR;}
	virtual void DoPostVoltageUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);
	virtual void DoPostCurrentUpdatesSlab(unsigned int startX, unsigned int stopX, int numTS, int threadID);

protected:
	//! Shared body of DoPostVoltageUpdates()/-Slab(): startX/stopX bound the
	//! x-range to touch, numTS is the timestep to evaluate the delay lookup
	//! at, and nThreads/threadID select this thread's share of whichever axis
	//! is not x. The whole-grid entry point passes [0,UINT_MAX), the engine's
	//! current timestep, and a single thread, so there is one implementation.
	void DoPostVoltageUpdatesImpl(unsigned int startX, unsigned int stopX, int numTS, int nThreads, int threadID);
	void DoPostCurrentUpdatesImpl(unsigned int startX, unsigned int stopX, int numTS, int nThreads, int threadID);

	//! One of the box's six faces (direction \a n, low/high side \a lowHigh).
	//! \a posN is the fixed coordinate of the face along axis n (already
	//! resolved by the caller to m_Start[n]/m_Stop[n], since the voltage and
	//! current updates disagree on the low side by one cell). \a nP/\a nPP are
	//! the in-plane axes; whichever of the two is axis 0 carries the global x
	//! coordinate and gets restricted to [startX,stopX), while the other is
	//! split across threadID. Shared between the two low/high calls per
	//! direction so that split, and the ui_pos formula it must stay in step
	//! with, is written once.
	void VoltageFace(int n, int nP, int nPP, int lowHigh, unsigned int posN,
	                  unsigned int startX, unsigned int stopX, int nThreads, int threadID,
	                  const unsigned int* delay, FDTD_FLOAT* signal);
	void CurrentFace(int n, int nP, int nPP, int lowHigh, unsigned int posN,
	                  unsigned int startX, unsigned int stopX, int nThreads, int threadID,
	                  const unsigned int* delay, FDTD_FLOAT* signal);

	Operator_Ext_TFSF* m_Op_TFSF;
};

#endif // ENGINE_EXT_TFSF_H
