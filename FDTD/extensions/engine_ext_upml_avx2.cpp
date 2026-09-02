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

// ============================================================================
// AVX2 UPML update kernels
//
// This translation unit is built with -mavx2 -mfma (see the top-level
// CMakeLists.txt). Keep it separate from engine_ext_upml.cpp, which also
// carries the generic paths used by the basic/SSE engines and therefore must
// stay compilable for the baseline ISA.
//
// Why this exists
// ---------------
// Engine_AVX2 stores a field as
//
//     m_volt[n + (z % numVectors) * 3 + y * m_vs_y + x * m_vs_x].f[z / numVectors]
//
// so the z axis is cut into 8 contiguous lanes of numVectors cells each. The
// scalar UPML loop walks z fastest, which in this layout means a 96-byte
// stride that consumes 4 bytes per cache line and revisits every column eight
// times, once per lane. Measured on an 80^3 grid with PML_8, that made the
// four UPML passes 84% of the iterate time and cut the AVX2 engine's lead
// over SSE from 2.0x to 1.2x -- the AVX2 engine lost most of its advantage as
// soon as an open boundary was present. (SSE suffers the same way but less:
// its f4vector layout gives a 16-byte stride over 4 lanes.)
//
// The fix is to stop addressing single cells and update whole f8vectors. That
// requires the PML box to line up with the lane split, which holds exactly
// when the box spans the full z range: local z then equals global z, so one
// f8vector is exactly 8 cells of the box. Operator_Ext_UPML::Create_UPML
// builds the four x- and y-normal boxes over the full z extent, so they all
// qualify; the two z-normal slabs do not and keep the scalar path. A masked
// vector path for those would need 8x the coefficient memory to save no
// instructions at all, since a slab contributes at most one live lane per
// vector.
//
// Arithmetic is deliberately mul+sub / mul+add rather than FMA, matching the
// non-contracted mulss/subss the scalar loop compiles to, so the vector path
// is bit-identical to the scalar one rather than merely close. That needs
// -ffp-contract=off on this file (set in the top-level CMakeLists.txt);
// without it GCC's default contraction folds the intrinsics back into
// vfnmadd/vfmadd and the two paths drift by a few ulp per timestep. The
// kernels are memory-bound, so the saved multiply would buy nothing anyway.
// ============================================================================

#include "engine_ext_upml.h"

#if OPENEMS_ENABLE_AVX2

#include "operator_ext_upml.h"
#include "FDTD/engine_avx2.h"
#include "tools/global.h"

#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

static f8vector* Alloc_f8_upml(size_t count)
{
	void* ptr = NULL;
	if (posix_memalign(&ptr, 32, sizeof(f8vector) * count) != 0)
		return NULL;
	memset(ptr, 0, sizeof(f8vector) * count);
	return static_cast<f8vector*>(ptr);
}

//! Repack one scalar PML coefficient array into the engine's z-lane layout.
/*!
	Source is ArrayNIJK<FDTD_FLOAT>(n, i, j, k) over the PML-local box, target
	is dst[n + v * 3 + j * vs_y + i * vs_x].f[lane] with v = k % numVectors and
	lane = k / numVectors -- the same split Engine_AVX2 applies to global z.
	Entries past numLines[2] are left at zero, which is what the engine's own
	operator holds there, so the padding lanes keep evaluating to zero.
*/
static void PackCoeff(
	f8vector* dst,
	const ArrayLib::ArrayNIJK<FDTD_FLOAT>& src,
	unsigned int pNx, unsigned int pNy, unsigned int pNz,
	unsigned int numVectors, size_t vs_y, size_t vs_x
)
{
	for (unsigned int i = 0; i < pNx; ++i)
	{
		for (unsigned int j = 0; j < pNy; ++j)
		{
			const size_t base = (size_t)i * vs_x + (size_t)j * vs_y;
			for (unsigned int k = 0; k < pNz; ++k)
			{
				const unsigned int v    = k % numVectors;
				const unsigned int lane = k / numVectors;
				for (unsigned int n = 0; n < 3; ++n)
					dst[base + (size_t)v * 3 + n].f[lane] = src(n, i, j, k);
			}
		}
	}
}

bool Engine_Ext_UPML::BuildAVX2Layout()
{
	Engine_AVX2* eng = dynamic_cast<Engine_AVX2*>(m_Eng);
	if (eng == NULL)
		return false;

	// Escape hatch: forces the scalar path so the two can be compared against
	// each other in one binary. They are expected to agree bit for bit; that
	// is what python/Tests/test_upml_engines.py asserts.
	if (const char* env = std::getenv("OPENEMS_UPML_NO_AVX2"))
		if ((env[0] != '\0') && (env[0] != '0'))
			return false;

	const unsigned int pNx = m_Op_UPML->m_numLines[0];
	const unsigned int pNy = m_Op_UPML->m_numLines[1];
	const unsigned int pNz = m_Op_UPML->m_numLines[2];
	if ((pNx == 0) || (pNy == 0) || (pNz == 0))
		return false;

	// Only a box spanning the full z extent maps onto whole f8vectors.
	if (m_Op_UPML->m_StartPos[2] != 0)
		return false;
	if (pNz != eng->GetNumLines(2))
		return false;

	// The operator arrays are handed over to this engine extension and freed;
	// a second engine built from the same operator would find them gone.
	if (!m_Op_UPML->vv.valid())
		return false;

	const unsigned int nv = eng->GetNumVectors();
	if (nv == 0)
		return false;

	m_v_numVectors = nv;
	m_v_pNy        = pNy;
	m_v_sx         = m_Op_UPML->m_StartPos[0];
	m_v_sy         = m_Op_UPML->m_StartPos[1];
	m_v_vs_y       = (size_t)3 * nv;
	m_v_vs_x       = (size_t)3 * nv * pNy;
	m_v_e_vs_y     = eng->m_vs_y;
	m_v_e_vs_x     = eng->m_vs_x;

	const size_t total = (size_t)3 * nv * pNy * pNx;

	f8vector* arrays[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
	for (int n = 0; n < 8; ++n)
	{
		arrays[n] = Alloc_f8_upml(total);
		if (arrays[n] == NULL)
		{
			for (int m = 0; m < n; ++m)
				free(arrays[m]);
			return false;
		}
	}

	m_v_vv        = arrays[0];
	m_v_vvfo      = arrays[1];
	m_v_vvfn      = arrays[2];
	m_v_ii        = arrays[3];
	m_v_iifo      = arrays[4];
	m_v_iifn      = arrays[5];
	m_v_volt_flux = arrays[6];
	m_v_curr_flux = arrays[7];

	PackCoeff(m_v_vv,   m_Op_UPML->vv,   pNx, pNy, pNz, nv, m_v_vs_y, m_v_vs_x);
	PackCoeff(m_v_vvfo, m_Op_UPML->vvfo, pNx, pNy, pNz, nv, m_v_vs_y, m_v_vs_x);
	PackCoeff(m_v_vvfn, m_Op_UPML->vvfn, pNx, pNy, pNz, nv, m_v_vs_y, m_v_vs_x);
	PackCoeff(m_v_ii,   m_Op_UPML->ii,   pNx, pNy, pNz, nv, m_v_vs_y, m_v_vs_x);
	PackCoeff(m_v_iifo, m_Op_UPML->iifo, pNx, pNy, pNz, nv, m_v_vs_y, m_v_vs_x);
	PackCoeff(m_v_iifn, m_Op_UPML->iifn, pNx, pNy, pNz, nv, m_v_vs_y, m_v_vs_x);

	// The packed arrays are a permutation of the scalar ones, so release the
	// originals and the scalar flux arrays to keep the footprint unchanged.
	// Engine_Vulkan reads the operator arrays directly, but it never runs
	// against an Engine_AVX2, so it cannot observe this.
	m_Op_UPML->ReleaseCoeffArrays();
	volt_flux.Reset();
	curr_flux.Reset();

	if (g_settings.GetVerboseLevel() > 0)
	{
		cout << "  UPML [" << m_Op_UPML->m_StartPos[0] << ","
		     << m_Op_UPML->m_StartPos[1] << "," << m_Op_UPML->m_StartPos[2]
		     << "] +[" << pNx << "," << pNy << "," << pNz
		     << "]: AVX2 vector path" << endl;
	}

	return true;
}

void Engine_Ext_UPML::ReleaseAVX2Layout()
{
	free(m_v_vv);        m_v_vv        = NULL;
	free(m_v_vvfo);      m_v_vvfo      = NULL;
	free(m_v_vvfn);      m_v_vvfn      = NULL;
	free(m_v_ii);        m_v_ii        = NULL;
	free(m_v_iifo);      m_v_iifo      = NULL;
	free(m_v_iifn);      m_v_iifn      = NULL;
	free(m_v_volt_flux); m_v_volt_flux = NULL;
	free(m_v_curr_flux); m_v_curr_flux = NULL;
	m_avx2_packed = false;
}

// --- update kernels ---------------------------------------------------------
//
// Polarisation is the fastest index in both the engine and the packed PML
// layout, and the z-vector stride is 3 in both, so the (v, n) loops collapse
// into a single run of 3 * numVectors consecutive f8vectors on either side.
//
// The engine's field arrays are read fresh on every call: Engine_AVX2::Init()
// runs Engine::Init() -- and with it InitExtensions() and BuildAVX2Layout() --
// before it allocates m_volt/m_curr, so only the strides are safe to cache.

//! Pre-update: hand the old flux to the engine, stash a * field - b * flux.
void Engine_Ext_UPML::PreUpdateAVX2(
	f8vector* __restrict field, f8vector* __restrict flux,
	const f8vector* __restrict a, const f8vector* __restrict b,
	int threadID
)
{
	if (threadID >= m_NrThreads)
		return;

	const unsigned int cnt = 3 * m_v_numVectors;

	for (unsigned int i = m_start.at(threadID), iEnd = i + m_numX.at(threadID); i < iEnd; ++i)
	{
		for (unsigned int j = 0; j < m_v_pNy; ++j)
		{
			const size_t e_base = (size_t)(i + m_v_sx) * m_v_e_vs_x + (size_t)(j + m_v_sy) * m_v_e_vs_y;
			const size_t p_base = (size_t)i * m_v_vs_x + (size_t)j * m_v_vs_y;

			for (unsigned int c = 0; c < cnt; ++c)
			{
				const __m256 f_old = field[e_base + c].v;
				const __m256 x_old = flux[p_base + c].v;
				const __m256 f_new = _mm256_sub_ps(
					_mm256_mul_ps(a[p_base + c].v, f_old),
					_mm256_mul_ps(b[p_base + c].v, x_old)
				);
				field[e_base + c].v = x_old;
				flux[p_base + c].v  = f_new;
			}
		}
	}
}

//! Post-update: take the new flux back from the engine, field = stash + c * flux.
void Engine_Ext_UPML::PostUpdateAVX2(
	f8vector* __restrict field, f8vector* __restrict flux,
	const f8vector* __restrict c_arr,
	int threadID
)
{
	if (threadID >= m_NrThreads)
		return;

	const unsigned int cnt = 3 * m_v_numVectors;

	for (unsigned int i = m_start.at(threadID), iEnd = i + m_numX.at(threadID); i < iEnd; ++i)
	{
		for (unsigned int j = 0; j < m_v_pNy; ++j)
		{
			const size_t e_base = (size_t)(i + m_v_sx) * m_v_e_vs_x + (size_t)(j + m_v_sy) * m_v_e_vs_y;
			const size_t p_base = (size_t)i * m_v_vs_x + (size_t)j * m_v_vs_y;

			for (unsigned int c = 0; c < cnt; ++c)
			{
				const __m256 stashed = flux[p_base + c].v;
				const __m256 x_new   = field[e_base + c].v;
				flux[p_base + c].v  = x_new;
				field[e_base + c].v = _mm256_add_ps(
					stashed, _mm256_mul_ps(c_arr[p_base + c].v, x_new)
				);
			}
		}
	}
}

void Engine_Ext_UPML::DoPreVoltageUpdatesAVX2(int threadID)
{
	Engine_AVX2* eng = static_cast<Engine_AVX2*>(m_Eng);
	PreUpdateAVX2(eng->m_volt, m_v_volt_flux, m_v_vv, m_v_vvfo, threadID);
}

void Engine_Ext_UPML::DoPostVoltageUpdatesAVX2(int threadID)
{
	Engine_AVX2* eng = static_cast<Engine_AVX2*>(m_Eng);
	PostUpdateAVX2(eng->m_volt, m_v_volt_flux, m_v_vvfn, threadID);
}

void Engine_Ext_UPML::DoPreCurrentUpdatesAVX2(int threadID)
{
	Engine_AVX2* eng = static_cast<Engine_AVX2*>(m_Eng);
	PreUpdateAVX2(eng->m_curr, m_v_curr_flux, m_v_ii, m_v_iifo, threadID);
}

void Engine_Ext_UPML::DoPostCurrentUpdatesAVX2(int threadID)
{
	Engine_AVX2* eng = static_cast<Engine_AVX2*>(m_Eng);
	PostUpdateAVX2(eng->m_curr, m_v_curr_flux, m_v_iifn, threadID);
}

#endif // OPENEMS_ENABLE_AVX2
