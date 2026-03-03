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

#include "engine_interface_avx2_fdtd.h"

#include <immintrin.h>

using std::cerr;
using std::endl;

Engine_Interface_AVX2_FDTD::Engine_Interface_AVX2_FDTD(Operator_AVX2* op)
	: Engine_Interface_FDTD(op)
{
	m_Op_AVX2 = op;
	m_Eng_AVX2 = dynamic_cast<Engine_AVX2*>(m_Op_AVX2->GetEngine());
	if (m_Eng_AVX2 == nullptr)
	{
		cerr << "Engine_Interface_AVX2_FDTD::Engine_Interface_AVX2_FDTD: "
		     << "Error: AVX2 Engine is not set! Exit!" << endl;
		exit(1);
	}
}

Engine_Interface_AVX2_FDTD::~Engine_Interface_AVX2_FDTD()
{
	m_Op_AVX2 = nullptr;
	m_Eng_AVX2 = nullptr;
}

double Engine_Interface_AVX2_FDTD::CalcFastEnergy() const
{
	if (m_Eng_AVX2->GetType() != Engine::AVX2)
		return Engine_Interface_FDTD::CalcFastEnergy();

	const unsigned int nX = m_Op_AVX2->GetNumberOfLines(0) - 1;
	const unsigned int nY = m_Op_AVX2->GetNumberOfLines(1) - 1;
	const unsigned int nZv = m_Op_AVX2->numVectors;
	const unsigned int vs_z = m_Eng_AVX2->m_vs_z;
	const unsigned int vs_y = m_Eng_AVX2->m_vs_y;
	const unsigned int vs_x = m_Eng_AVX2->m_vs_x;

	f8vector* volt = m_Eng_AVX2->m_volt;
	f8vector* curr = m_Eng_AVX2->m_curr;

	// Use AVX2 to accumulate energy in 8-wide accumulators
	__m256 E_energy_acc = _mm256_setzero_ps();
	__m256 H_energy_acc = _mm256_setzero_ps();

	for (unsigned int x = 0; x < nX; ++x)
	{
		for (unsigned int y = 0; y < nY; ++y)
		{
			for (unsigned int z = 0; z < nZv; ++z)
			{
				unsigned int base = x * vs_x + y * vs_y + z * vs_z;

				// E-field energy: sum of |V|^2 for all 3 polarisations
				__m256 vx = _mm256_load_ps(volt[base].f);
				__m256 vy = _mm256_load_ps(volt[base + 1].f);
				__m256 vz = _mm256_load_ps(volt[base + 2].f);

				E_energy_acc = _mm256_fmadd_ps(vx, vx, E_energy_acc);
				E_energy_acc = _mm256_fmadd_ps(vy, vy, E_energy_acc);
				E_energy_acc = _mm256_fmadd_ps(vz, vz, E_energy_acc);

				// H-field energy: sum of |I|^2 for all 3 polarisations
				__m256 ix = _mm256_load_ps(curr[base].f);
				__m256 iy = _mm256_load_ps(curr[base + 1].f);
				__m256 iz = _mm256_load_ps(curr[base + 2].f);

				H_energy_acc = _mm256_fmadd_ps(ix, ix, H_energy_acc);
				H_energy_acc = _mm256_fmadd_ps(iy, iy, H_energy_acc);
				H_energy_acc = _mm256_fmadd_ps(iz, iz, H_energy_acc);
			}
		}
	}

	// Horizontal sum of 8-wide accumulators
	// E_energy_acc = [e0 e1 e2 e3 e4 e5 e6 e7]
	__m128 e_lo = _mm256_castps256_ps128(E_energy_acc);
	__m128 e_hi = _mm256_extractf128_ps(E_energy_acc, 1);
	__m128 e_sum = _mm_add_ps(e_lo, e_hi); // [e0+e4, e1+e5, e2+e6, e3+e7]
	e_sum = _mm_hadd_ps(e_sum, e_sum);      // [e0+e4+e1+e5, e2+e6+e3+e7, ...]
	e_sum = _mm_hadd_ps(e_sum, e_sum);      // [total, ...]
	float E_total;
	_mm_store_ss(&E_total, e_sum);

	__m128 h_lo = _mm256_castps256_ps128(H_energy_acc);
	__m128 h_hi = _mm256_extractf128_ps(H_energy_acc, 1);
	__m128 h_sum = _mm_add_ps(h_lo, h_hi);
	h_sum = _mm_hadd_ps(h_sum, h_sum);
	h_sum = _mm_hadd_ps(h_sum, h_sum);
	float H_total;
	_mm_store_ss(&H_total, h_sum);

	return __EPS0__ * (double)E_total + __MUE0__ * (double)H_total;
}
