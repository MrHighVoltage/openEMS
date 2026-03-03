/*
*	Copyright (C) 2026 openEMS contributors
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

#ifndef ENGINE_INTERFACE_AVX2_FDTD_H
#define ENGINE_INTERFACE_AVX2_FDTD_H

#include "engine_interface_fdtd.h"
#include "operator_avx2.h"
#include "engine_avx2.h"

class Engine_Interface_AVX2_FDTD : public Engine_Interface_FDTD
{
public:
	Engine_Interface_AVX2_FDTD(Operator_AVX2* op);
	virtual ~Engine_Interface_AVX2_FDTD();

	virtual double CalcFastEnergy() const;

protected:
	Operator_AVX2* m_Op_AVX2;
	Engine_AVX2* m_Eng_AVX2;
};

#endif // ENGINE_INTERFACE_AVX2_FDTD_H
