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

#include "engine_avx2_multithread.h"
#include "extensions/engine_extension.h"
#include "tools/denormal.h"

#include <iomanip>

using std::cout;
using std::endl;

// ============================================================================
// Engine_AVX2_Multithread
// ============================================================================

Engine_AVX2_Multithread* Engine_AVX2_Multithread::New(
	const Operator_AVX2_Multithread* op, unsigned int numThreads)
{
	cout << "Create FDTD engine (AVX2 + FMA + multi-threading)" << endl;
	Engine_AVX2_Multithread* e = new Engine_AVX2_Multithread(op);
	e->setNumThreads(numThreads);
	e->Init();
	return e;
}

Engine_AVX2_Multithread::Engine_AVX2_Multithread(const Operator_AVX2_Multithread* op)
	: Engine_AVX2(op)
{
	m_Op_MT = op;
	m_IterateBarrier = nullptr;
	m_startBarrier = nullptr;
	m_stopBarrier = nullptr;
	m_max_numThreads = std::thread::hardware_concurrency();
	m_numThreads = 0;
	m_last_speed = 0;
	m_opt_speed = false;
	m_stopThreads = true;
}

Engine_AVX2_Multithread::~Engine_AVX2_Multithread()
{
	Reset();
}

void Engine_AVX2_Multithread::setNumThreads(unsigned int numThreads)
{
	m_numThreads = numThreads;
}

void Engine_AVX2_Multithread::Init()
{
	m_stopThreads = true;
	m_opt_speed = false;
	Engine_AVX2::Init();

	// Initialize threads
	m_stopThreads = false;
	if (m_numThreads == 0)
	{
		m_opt_speed = true;
		m_numThreads = 1;
	}
	else if (m_numThreads > m_max_numThreads)
		m_numThreads = m_max_numThreads;

	this->changeNumThreads(m_numThreads);
}

void Engine_AVX2_Multithread::Reset()
{
	if (!m_threads.empty())
	{
		ClearExtensions();

		m_stopThreads = true;
		m_startBarrier->wait();
		for (auto& t : m_threads) t.join();
		m_threads.clear();
		delete m_IterateBarrier;
		m_IterateBarrier = nullptr;
		delete m_startBarrier;
		m_startBarrier = nullptr;
		delete m_stopBarrier;
		m_stopBarrier = nullptr;
	}

	Engine_AVX2::Reset();
}

void Engine_AVX2_Multithread::changeNumThreads(unsigned int numThreads)
{
	if (!m_threads.empty())
	{
		m_stopThreads = true;
		m_startBarrier->wait();
		for (auto& t : m_threads) t.join();
		m_threads.clear();
	}

	m_numThreads = numThreads;
	m_stopThreads = false;

	if (g_settings.GetVerboseLevel() > 0)
		cout << "AVX2 multithreaded engine using " << m_numThreads << " threads. Utilization: (";

	std::vector<unsigned int> m_Start_Lines;
	std::vector<unsigned int> m_Stop_Lines;
	m_Op_MT->CalcStartStopLines(m_numThreads, m_Start_Lines, m_Stop_Lines);

	delete m_IterateBarrier;
	m_IterateBarrier = new Barrier(m_numThreads);

	delete m_startBarrier;
	m_startBarrier = new Barrier(m_numThreads + 1);

	delete m_stopBarrier;
	m_stopBarrier = new Barrier(m_numThreads + 1);

	for (unsigned int n = 0; n < m_numThreads; n++)
	{
		unsigned int start = m_Start_Lines.at(n);
		unsigned int stop = m_Stop_Lines.at(n);
		unsigned int stop_h = stop;
		if (n == m_numThreads - 1)
		{
			// Last thread processes H-field one row less
			stop_h = stop - 1;
			if (g_settings.GetVerboseLevel() > 0)
				cout << stop - start + 1 << ")" << endl;
		}
		else
			if (g_settings.GetVerboseLevel() > 0)
				cout << stop - start + 1 << ";";

		m_threads.emplace_back(
			NS_Engine_AVX2_Multithread::thread(this, start, stop, stop_h, n)
		);
	}

	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
		m_Eng_exts.at(n)->SetNumberOfThreads(m_numThreads);
}

bool Engine_AVX2_Multithread::IterateTS(unsigned int iterTS)
{
	m_iterTS = iterTS;

	m_startBarrier->wait(); // start threads
	m_stopBarrier->wait();  // wait for completion

	return true;
}

void Engine_AVX2_Multithread::NextInterval(float curr_speed)
{
	Engine_AVX2::NextInterval(curr_speed);
	if (!m_opt_speed)
		return;
	if (curr_speed < m_last_speed)
	{
		this->changeNumThreads(m_numThreads - 1);
		cout << "AVX2 Multithreaded Engine: Best performance found using "
		     << m_numThreads << " threads." << endl;
		m_opt_speed = false;
	}
	else if (m_numThreads < m_max_numThreads)
	{
		m_last_speed = curr_speed;
		this->changeNumThreads(m_numThreads + 1);
	}
}

void Engine_AVX2_Multithread::DoPreVoltageUpdates(int threadID)
{
	for (int n = m_Eng_exts.size() - 1; n >= 0; --n)
	{
		m_Eng_exts.at(n)->DoPreVoltageUpdates(threadID);
		m_IterateBarrier->wait();
	}
}

void Engine_AVX2_Multithread::DoPostVoltageUpdates(int threadID)
{
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		m_Eng_exts.at(n)->DoPostVoltageUpdates(threadID);
		m_IterateBarrier->wait();
	}
}

void Engine_AVX2_Multithread::Apply2Voltages(int threadID)
{
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		m_Eng_exts.at(n)->Apply2Voltages(threadID);
		m_IterateBarrier->wait();
	}
}

void Engine_AVX2_Multithread::DoPreCurrentUpdates(int threadID)
{
	for (int n = m_Eng_exts.size() - 1; n >= 0; --n)
	{
		m_Eng_exts.at(n)->DoPreCurrentUpdates(threadID);
		m_IterateBarrier->wait();
	}
}

void Engine_AVX2_Multithread::DoPostCurrentUpdates(int threadID)
{
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		m_Eng_exts.at(n)->DoPostCurrentUpdates(threadID);
		m_IterateBarrier->wait();
	}
}

void Engine_AVX2_Multithread::Apply2Current(int threadID)
{
	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		m_Eng_exts.at(n)->Apply2Current(threadID);
		m_IterateBarrier->wait();
	}
}

// ============================================================================
// Worker thread
// ============================================================================

namespace NS_Engine_AVX2_Multithread
{

thread::thread(Engine_AVX2_Multithread* ptr, unsigned int start, unsigned int stop,
               unsigned int stop_h, unsigned int threadID)
{
	m_enginePtr = ptr;
	m_start = start;
	m_stop = stop;
	m_stop_h = stop_h;
	m_threadID = threadID;
}

void thread::operator()()
{
	// Flush denormals for this thread
	Denormal::Disable();

	// Note: the loop must be unconditional. Testing m_stopThreads here would let
	// a worker leave without ever arriving at m_startBarrier, while the shutdown
	// path in Reset()/changeNumThreads() blocks on that same barrier expecting
	// m_numThreads+1 arrivals -> deadlock. Shutdown is signalled solely by the
	// check *after* the barrier, which is also the point where m_stopThreads is
	// safely visible (the barrier's mutex provides the happens-before edge).
	for (;;)
	{
		// Wait for start signal
		m_enginePtr->m_startBarrier->wait();

		if (m_enginePtr->m_stopThreads)
			return;

		for (unsigned int iter = 0; iter < m_enginePtr->m_iterTS; ++iter)
		{
			// Pre-voltage extensions
			m_enginePtr->DoPreVoltageUpdates(m_threadID);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after voltage pre-extensions");

			// Voltage update (x-range for this thread)
			m_enginePtr->UpdateVoltages(m_start, m_stop - m_start + 1);

			// Synchronize
			m_enginePtr->m_IterateBarrier->wait();
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after voltage update");

			// Post-voltage extensions
			m_enginePtr->DoPostVoltageUpdates(m_threadID);
			m_enginePtr->Apply2Voltages(m_threadID);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after voltage extensions");

			// Pre-current extensions
			m_enginePtr->DoPreCurrentUpdates(m_threadID);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after current pre-extensions");

			// Current update (last thread processes one fewer x-line for H-field)
			m_enginePtr->UpdateCurrents(m_start, m_stop_h - m_start + 1);

			// Synchronize
			m_enginePtr->m_IterateBarrier->wait();
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after current update");

			// Post-current extensions
			m_enginePtr->DoPostCurrentUpdates(m_threadID);
			m_enginePtr->Apply2Current(m_threadID);
			if (m_threadID == 0)
				m_enginePtr->TraceFieldCell("after current extensions");

			if (m_threadID == 0)
				++m_enginePtr->numTS;
		}

		m_enginePtr->m_stopBarrier->wait();
	}
}

} // namespace NS_Engine_AVX2_Multithread
