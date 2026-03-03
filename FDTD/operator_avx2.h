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

#ifndef OPERATOR_AVX2_H
#define OPERATOR_AVX2_H

#include "operator.h"
#include "tools/array_ops.h"
#include "tools/arraylib/array_ijk.h"

#include <immintrin.h>
#include <cstring>

// 8-wide float vector for AVX2
union f8vector
{
	__m256 v;
	float f[8];
};

//! Coefficient key for AVX2 operator compression (all 3 polarisations)
class AVX2_coeff
{
public:
	AVX2_coeff(f8vector vv[3], f8vector vi[3], f8vector iv[3], f8vector ii[3]);
	bool operator==(const AVX2_coeff&) const;
	bool operator!=(const AVX2_coeff&) const;
	bool operator<(const AVX2_coeff&) const;
protected:
	f8vector m_vv[3];
	f8vector m_vi[3];
	f8vector m_iv[3];
	f8vector m_ii[3];
};

class Operator_AVX2 : public Operator
{
	friend class Engine_AVX2;
	friend class Engine_AVX2_Multithread;
	friend class Engine_Interface_AVX2_FDTD;
public:
	//! Create a new operator
	static Operator_AVX2* New();
	virtual ~Operator_AVX2();

	virtual Engine* CreateEngine();

	inline virtual FDTD_FLOAT GetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		if (m_Use_Compression)
			return f8_vv_Comp[n][m_Op_index(x, y, z % numVectors)].f[z / numVectors];
		return f8_vv[n][x * m_stride_x + y * m_stride_y + z % numVectors].f[z / numVectors];
	}

	inline virtual FDTD_FLOAT GetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		if (m_Use_Compression)
			return f8_vi_Comp[n][m_Op_index(x, y, z % numVectors)].f[z / numVectors];
		return f8_vi[n][x * m_stride_x + y * m_stride_y + z % numVectors].f[z / numVectors];
	}

	inline virtual FDTD_FLOAT GetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		if (m_Use_Compression)
			return f8_ii_Comp[n][m_Op_index(x, y, z % numVectors)].f[z / numVectors];
		return f8_ii[n][x * m_stride_x + y * m_stride_y + z % numVectors].f[z / numVectors];
	}

	inline virtual FDTD_FLOAT GetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		if (m_Use_Compression)
			return f8_iv_Comp[n][m_Op_index(x, y, z % numVectors)].f[z / numVectors];
		return f8_iv[n][x * m_stride_x + y * m_stride_y + z % numVectors].f[z / numVectors];
	}

	inline virtual void SetVV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		f8_vv[n][x * m_stride_x + y * m_stride_y + z % numVectors].f[z / numVectors] = value;
	}

	inline virtual void SetVI(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		f8_vi[n][x * m_stride_x + y * m_stride_y + z % numVectors].f[z / numVectors] = value;
	}

	inline virtual void SetII(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		f8_ii[n][x * m_stride_x + y * m_stride_y + z % numVectors].f[z / numVectors] = value;
	}

	inline virtual void SetIV(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT value)
	{
		f8_iv[n][x * m_stride_x + y * m_stride_y + z % numVectors].f[z / numVectors] = value;
	}

	virtual void ShowStat() const;

protected:
	//! use New() for creating a new Operator
	Operator_AVX2();

	virtual void Init();
	void Delete();
	virtual void Reset();
	virtual void InitOperator();
	virtual int CalcECOperator(DebugFlags debugFlags = None);

	bool CompressOperator();

	unsigned int numVectors;  //!< number of 8-wide vectors along z
	unsigned int m_stride_y;  //!< flat array y-stride: = numVectors
	unsigned int m_stride_x;  //!< flat array x-stride: = numVectors * numLines[1]

	//! Uncompressed operator coefficients: flat array per polarisation.
	//! f8_vv[n][x * m_stride_x + y * m_stride_y + z_vec]
	//! Allocated during InitOperator, freed after CompressOperator.
	f8vector* f8_vv[3];
	f8vector* f8_vi[3];
	f8vector* f8_iv[3];
	f8vector* f8_ii[3];

	bool m_Use_Compression;

public:
	// Compression data (public for engine access)
	ArrayLib::ArrayIJK<unsigned int> m_Op_index;
	f8vector* f8_vv_Comp[3]; //!< compressed vv lookup table
	f8vector* f8_vi_Comp[3]; //!< compressed vi lookup table
	f8vector* f8_iv_Comp[3]; //!< compressed iv lookup table
	f8vector* f8_ii_Comp[3]; //!< compressed ii lookup table
	unsigned int m_numCompressedEntries;
};

#endif // OPERATOR_AVX2_H
