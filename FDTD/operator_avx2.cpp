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

#include "operator_avx2.h"
#include "engine_avx2.h"

#include <cstring>
#include <cmath>
#include <map>

using std::cout;
using std::endl;

// --- Aligned allocation helper for flat f8vector arrays (32-byte for AVX2) ---

static f8vector* Alloc_f8_flat(size_t count)
{
	void* ptr = nullptr;
	if (posix_memalign(&ptr, 32, sizeof(f8vector) * count) != 0)
		return nullptr;
	memset(ptr, 0, sizeof(f8vector) * count);
	return static_cast<f8vector*>(ptr);
}

// --- Operator_AVX2 ---

Operator_AVX2* Operator_AVX2::New()
{
	cout << "Create FDTD operator (AVX2 + FMA)" << endl;
	Operator_AVX2* op = new Operator_AVX2();
	op->Init();
	return op;
}

Operator_AVX2::Operator_AVX2() : Operator()
{
	for (int n = 0; n < 3; ++n)
	{
		f8_vv[n] = nullptr;
		f8_vi[n] = nullptr;
		f8_iv[n] = nullptr;
		f8_ii[n] = nullptr;
		f8_vv_Comp[n] = nullptr;
		f8_vi_Comp[n] = nullptr;
		f8_iv_Comp[n] = nullptr;
		f8_ii_Comp[n] = nullptr;
	}
	numVectors = 0;
	m_stride_y = 0;
	m_stride_x = 0;
	m_Use_Compression = false;
	m_numCompressedEntries = 0;
}

Operator_AVX2::~Operator_AVX2()
{
	Delete();
}

Engine* Operator_AVX2::CreateEngine()
{
	m_Engine = Engine_AVX2::New(this);
	return m_Engine;
}

void Operator_AVX2::Init()
{
	Operator::Init();
	for (int n = 0; n < 3; ++n)
	{
		f8_vv[n] = nullptr;
		f8_vi[n] = nullptr;
		f8_iv[n] = nullptr;
		f8_ii[n] = nullptr;
		f8_vv_Comp[n] = nullptr;
		f8_vi_Comp[n] = nullptr;
		f8_iv_Comp[n] = nullptr;
		f8_ii_Comp[n] = nullptr;
	}
	numVectors = 0;
	m_stride_y = 0;
	m_stride_x = 0;
	m_Use_Compression = false;
	m_numCompressedEntries = 0;
}

void Operator_AVX2::Delete()
{
	for (int n = 0; n < 3; ++n)
	{
		free(f8_vv[n]); f8_vv[n] = nullptr;
		free(f8_vi[n]); f8_vi[n] = nullptr;
		free(f8_iv[n]); f8_iv[n] = nullptr;
		free(f8_ii[n]); f8_ii[n] = nullptr;
		free(f8_vv_Comp[n]); f8_vv_Comp[n] = nullptr;
		free(f8_vi_Comp[n]); f8_vi_Comp[n] = nullptr;
		free(f8_iv_Comp[n]); f8_iv_Comp[n] = nullptr;
		free(f8_ii_Comp[n]); f8_ii_Comp[n] = nullptr;
	}
	m_Op_index.Reset();
	m_Use_Compression = false;
	m_numCompressedEntries = 0;
}

void Operator_AVX2::Reset()
{
	Delete();
	Operator::Reset();
}

void Operator_AVX2::InitOperator()
{
	// Clean up any previous data
	for (int n = 0; n < 3; ++n)
	{
		free(f8_vv[n]); f8_vv[n] = nullptr;
		free(f8_vi[n]); f8_vi[n] = nullptr;
		free(f8_iv[n]); f8_iv[n] = nullptr;
		free(f8_ii[n]); f8_ii[n] = nullptr;
		free(f8_vv_Comp[n]); f8_vv_Comp[n] = nullptr;
		free(f8_vi_Comp[n]); f8_vi_Comp[n] = nullptr;
		free(f8_iv_Comp[n]); f8_iv_Comp[n] = nullptr;
		free(f8_ii_Comp[n]); f8_ii_Comp[n] = nullptr;
	}
	m_Use_Compression = false;

	numVectors = (unsigned int)ceil((double)numLines[2] / 8.0);
	m_stride_y = numVectors;
	m_stride_x = numVectors * numLines[1];

	size_t totalPerPol = (size_t)numLines[0] * numLines[1] * numVectors;

	for (int n = 0; n < 3; ++n)
	{
		f8_vv[n] = Alloc_f8_flat(totalPerPol);
		f8_vi[n] = Alloc_f8_flat(totalPerPol);
		f8_iv[n] = Alloc_f8_flat(totalPerPol);
		f8_ii[n] = Alloc_f8_flat(totalPerPol);
	}
}

int Operator_AVX2::CalcECOperator(DebugFlags debugFlags)
{
	int ErrCode = Operator::CalcECOperator(debugFlags);
	m_Use_Compression = false;
	m_Use_Compression = CompressOperator();
	return ErrCode;
}

bool Operator_AVX2::CompressOperator()
{
	if (g_settings.GetVerboseLevel() > 0)
		cout << "Compressing the FDTD operator (AVX2)... this may take a while..." << endl;

	unsigned int idx_dims[3] = { numLines[0], numLines[1], numVectors };
	m_Op_index.Init("Op_index_avx2", idx_dims);

	std::map<AVX2_coeff, unsigned int> lookUpMap;

	// Temporary storage for compressed coefficient sets
	std::vector<f8vector> tmp_vv[3], tmp_vi[3], tmp_iv[3], tmp_ii[3];

	unsigned int pos[3];
	for (pos[0] = 0; pos[0] < numLines[0]; ++pos[0])
	{
		for (pos[1] = 0; pos[1] < numLines[1]; ++pos[1])
		{
			for (pos[2] = 0; pos[2] < numVectors; ++pos[2])
			{
				unsigned int flatIdx = pos[0] * m_stride_x + pos[1] * m_stride_y + pos[2];

				f8vector vv[3], vi[3], iv[3], ii[3];
				for (int n = 0; n < 3; ++n)
				{
					vv[n] = f8_vv[n][flatIdx];
					vi[n] = f8_vi[n][flatIdx];
					iv[n] = f8_iv[n][flatIdx];
					ii[n] = f8_ii[n][flatIdx];
				}

				AVX2_coeff c(vv, vi, iv, ii);
				std::map<AVX2_coeff, unsigned int>::iterator it;
				it = lookUpMap.find(c);
				if (it == lookUpMap.end())
				{
					// New unique coefficient set
					unsigned int index = tmp_vv[0].size();
					for (int n = 0; n < 3; ++n)
					{
						tmp_vv[n].push_back(vv[n]);
						tmp_vi[n].push_back(vi[n]);
						tmp_iv[n].push_back(iv[n]);
						tmp_ii[n].push_back(ii[n]);
					}
					lookUpMap[c] = index;
					m_Op_index(pos[0], pos[1], pos[2]) = index;
				}
				else
				{
					// Already exists
					m_Op_index(pos[0], pos[1], pos[2]) = it->second;
				}
			}
		}
	}

	// Copy to 32-byte aligned storage
	m_numCompressedEntries = tmp_vv[0].size();
	for (int n = 0; n < 3; ++n)
	{
		f8_vv_Comp[n] = Alloc_f8_flat(m_numCompressedEntries);
		memcpy(f8_vv_Comp[n], tmp_vv[n].data(), m_numCompressedEntries * sizeof(f8vector));

		f8_vi_Comp[n] = Alloc_f8_flat(m_numCompressedEntries);
		memcpy(f8_vi_Comp[n], tmp_vi[n].data(), m_numCompressedEntries * sizeof(f8vector));

		f8_iv_Comp[n] = Alloc_f8_flat(m_numCompressedEntries);
		memcpy(f8_iv_Comp[n], tmp_iv[n].data(), m_numCompressedEntries * sizeof(f8vector));

		f8_ii_Comp[n] = Alloc_f8_flat(m_numCompressedEntries);
		memcpy(f8_ii_Comp[n], tmp_ii[n].data(), m_numCompressedEntries * sizeof(f8vector));
	}

	// Free uncompressed arrays — no longer needed
	for (int n = 0; n < 3; ++n)
	{
		free(f8_vv[n]); f8_vv[n] = nullptr;
		free(f8_vi[n]); f8_vi[n] = nullptr;
		free(f8_iv[n]); f8_iv[n] = nullptr;
		free(f8_ii[n]); f8_ii[n] = nullptr;
	}

	return true;
}

void Operator_AVX2::ShowStat() const
{
	Operator::ShowStat();

	cout << "AVX2 compression enabled\t: " << (m_Use_Compression ? "yes" : "no") << endl;
	if (m_Use_Compression)
		cout << "Unique AVX2 operators\t: " << m_numCompressedEntries << endl;
	cout << "-----------------------------------" << endl;
}


// --- AVX2_coeff ---

AVX2_coeff::AVX2_coeff(f8vector vv[3], f8vector vi[3], f8vector iv[3], f8vector ii[3])
{
	for (int n = 0; n < 3; ++n)
	{
		m_vv[n] = vv[n];
		m_vi[n] = vi[n];
		m_iv[n] = iv[n];
		m_ii[n] = ii[n];
	}
}

bool AVX2_coeff::operator==(const AVX2_coeff& other) const
{
	for (int n = 0; n < 3; ++n)
	{
		if (memcmp(&(m_vv[n]), &(other.m_vv[n]), sizeof(f8vector)) != 0) return false;
		if (memcmp(&(m_vi[n]), &(other.m_vi[n]), sizeof(f8vector)) != 0) return false;
		if (memcmp(&(m_iv[n]), &(other.m_iv[n]), sizeof(f8vector)) != 0) return false;
		if (memcmp(&(m_ii[n]), &(other.m_ii[n]), sizeof(f8vector)) != 0) return false;
	}
	return true;
}

bool AVX2_coeff::operator!=(const AVX2_coeff& other) const
{
	return !(*this == other);
}

bool AVX2_coeff::operator<(const AVX2_coeff& other) const
{
	for (int n = 0; n < 3; ++n)
	{
		for (int c = 0; c < 8; ++c)
		{
			if (m_vv[n].f[c] > other.m_vv[n].f[c]) return false;
			if (m_vv[n].f[c] < other.m_vv[n].f[c]) return true;
			if (m_vi[n].f[c] > other.m_vi[n].f[c]) return false;
			if (m_vi[n].f[c] < other.m_vi[n].f[c]) return true;
			if (m_iv[n].f[c] > other.m_iv[n].f[c]) return false;
			if (m_iv[n].f[c] < other.m_iv[n].f[c]) return true;
			if (m_ii[n].f[c] > other.m_ii[n].f[c]) return false;
			if (m_ii[n].f[c] < other.m_ii[n].f[c]) return true;
		}
	}
	return false;
}
