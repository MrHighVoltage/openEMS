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

#include "operator_avx2_multithread.h"
#include "engine_avx2_multithread.h"
#include "tools/useful.h"

using std::cout;
using std::cerr;
using std::endl;

// --- Operator_AVX2_Multithread ---

Operator_AVX2_Multithread* Operator_AVX2_Multithread::New(unsigned int numThreads)
{
	cout << "Create FDTD operator (AVX2 + FMA + multi-threading)" << endl;
	Operator_AVX2_Multithread* op = new Operator_AVX2_Multithread();
	op->setNumThreads(numThreads);
	op->Init();
	return op;
}

Operator_AVX2_Multithread::~Operator_AVX2_Multithread()
{
	Delete();
}

void Operator_AVX2_Multithread::setNumThreads(unsigned int numThreads)
{
	m_numThreads = numThreads;
	m_orig_numThreads = numThreads;
}

Engine* Operator_AVX2_Multithread::CreateEngine()
{
	m_Engine = Engine_AVX2_Multithread::New(this, m_orig_numThreads);
	return m_Engine;
}

Operator_AVX2_Multithread::Operator_AVX2_Multithread() : Operator_AVX2()
{
	m_CalcEC_Start = nullptr;
	m_CalcEC_Stop = nullptr;
	m_CalcPEC_Start = nullptr;
	m_CalcPEC_Stop = nullptr;
	m_Nr_PEC_thread = nullptr;
}

void Operator_AVX2_Multithread::Init()
{
	Operator_AVX2::Init();

	m_CalcEC_Start = nullptr;
	m_CalcEC_Stop = nullptr;
	m_CalcPEC_Start = nullptr;
	m_CalcPEC_Stop = nullptr;
}

void Operator_AVX2_Multithread::Delete()
{
	m_thread_group.join_all();

	delete m_CalcEC_Start;
	m_CalcEC_Start = nullptr;
	delete m_CalcEC_Stop;
	m_CalcEC_Stop = nullptr;

	delete m_CalcPEC_Start;
	m_CalcPEC_Start = nullptr;
	delete m_CalcPEC_Stop;
	m_CalcPEC_Stop = nullptr;
}

void Operator_AVX2_Multithread::Reset()
{
	Delete();
	Operator_AVX2::Reset();
}

void Operator_AVX2_Multithread::CalcStartStopLines(
	unsigned int &numThreads,
	std::vector<unsigned int> &start,
	std::vector<unsigned int> &stop
) const
{
	std::vector<unsigned int> jpt = AssignJobs2Threads(numLines[0], numThreads, true);

	numThreads = jpt.size();

	start.resize(numThreads);
	stop.resize(numThreads);

	start.at(0) = 0;
	stop.at(0) = jpt.at(0) - 1;

	for (unsigned int n = 1; n < numThreads; n++)
	{
		start.at(n) = stop.at(n - 1) + 1;
		stop.at(n) = start.at(n) + jpt.at(n) - 1;
	}
}

int Operator_AVX2_Multithread::CalcECOperator(DebugFlags debugFlags)
{
	if ((m_numThreads == 0) || (m_numThreads > boost::thread::hardware_concurrency()))
		m_numThreads = boost::thread::hardware_concurrency();

	std::vector<unsigned int> m_Start_Lines;
	std::vector<unsigned int> m_Stop_Lines;
	CalcStartStopLines(m_numThreads, m_Start_Lines, m_Stop_Lines);

	if (g_settings.GetVerboseLevel() > 0)
		cout << "Multithreaded AVX2 operator using " << m_numThreads << " threads." << endl;

	m_thread_group.join_all();
	delete m_CalcEC_Start;
	m_CalcEC_Start = new boost::barrier(m_numThreads + 1);
	delete m_CalcEC_Stop;
	m_CalcEC_Stop = new boost::barrier(m_numThreads + 1);

	delete m_CalcPEC_Start;
	m_CalcPEC_Start = new boost::barrier(m_numThreads + 1);
	delete m_CalcPEC_Stop;
	m_CalcPEC_Stop = new boost::barrier(m_numThreads + 1);

	for (unsigned int n = 0; n < m_numThreads; n++)
	{
		boost::thread* t = new boost::thread(
			Operator_AVX2_Thread(this, m_Start_Lines.at(n), m_Stop_Lines.at(n), n)
		);
		m_thread_group.add_thread(t);
	}

	return Operator_AVX2::CalcECOperator(debugFlags);
}

bool Operator_AVX2_Multithread::Calc_EC()
{
	if (CSX == nullptr)
	{
		cerr << "Operator_AVX2_Multithread::Calc_EC: CSX not given or invalid!!!" << endl;
		return false;
	}

	MainOp->SetPos(0, 0, 0);

	m_CalcEC_Start->wait();
	m_CalcEC_Stop->wait();

	return true;
}

bool Operator_AVX2_Multithread::CalcPEC()
{
	m_Nr_PEC[0] = 0;
	m_Nr_PEC[1] = 0;
	m_Nr_PEC[2] = 0;

	m_Nr_PEC_thread = new unsigned int[m_numThreads][3];

	m_CalcPEC_Start->wait();
	m_CalcPEC_Stop->wait();

	for (unsigned int t = 0; t < m_numThreads; ++t)
		for (int n = 0; n < 3; ++n)
			m_Nr_PEC[n] += m_Nr_PEC_thread[t][n];

	CalcPEC_Curves();

	delete[] m_Nr_PEC_thread;

	return true;
}

// --- Operator_AVX2_Thread ---

Operator_AVX2_Thread::Operator_AVX2_Thread(
	Operator_AVX2_Multithread* ptr, unsigned int start, unsigned int stop, unsigned int threadID)
{
	m_start = start;
	m_stop = stop;
	m_threadID = threadID;
	m_OpPtr = ptr;
}

void Operator_AVX2_Thread::operator()()
{
	// Calculate EC
	m_OpPtr->m_CalcEC_Start->wait();
	m_OpPtr->Calc_EC_Range(m_start, m_stop);
	m_OpPtr->m_CalcEC_Stop->wait();

	// Calculate PEC
	m_OpPtr->m_CalcPEC_Start->wait();
	for (int n = 0; n < 3; ++n)
		m_OpPtr->m_Nr_PEC_thread[m_threadID][n] = 0;

	m_OpPtr->CalcPEC_Range(m_start, m_stop, m_OpPtr->m_Nr_PEC_thread[m_threadID]);
	m_OpPtr->m_CalcPEC_Stop->wait();
}
