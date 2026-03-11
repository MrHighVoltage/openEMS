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

#ifndef ENGINE_AVX2_MULTITHREAD_H
#define ENGINE_AVX2_MULTITHREAD_H

#include "operator_avx2_multithread.h"
#include "engine_avx2.h"

#include <thread>
#include <vector>
#include "tools/barrier.h"

#include "tools/useful.h"
#if defined(_WIN32) && !defined(__GNUC__)
#include <Winsock2.h>
#else
#include <sys/time.h>
#endif

class Engine_AVX2_Multithread;

namespace NS_Engine_AVX2_Multithread
{

class thread
{
public:
	thread(Engine_AVX2_Multithread* ptr, unsigned int start, unsigned int stop,
	       unsigned int stop_h, unsigned int threadID);
	void operator()();

protected:
	unsigned int m_start, m_stop, m_stop_h, m_threadID;
	Engine_AVX2_Multithread* m_enginePtr;
};

} // namespace NS_Engine_AVX2_Multithread


class Engine_AVX2_Multithread : public Engine_AVX2
{
	friend class NS_Engine_AVX2_Multithread::thread;
public:
	static Engine_AVX2_Multithread* New(const Operator_AVX2_Multithread* op,
	                                     unsigned int numThreads = 0);
	virtual ~Engine_AVX2_Multithread();

	virtual void setNumThreads(unsigned int numThreads);
	virtual void Init();
	virtual void Reset();
	virtual void NextInterval(float curr_speed);

	//! Iterate number of timesteps
	virtual bool IterateTS(unsigned int iterTS);

	virtual void DoPreVoltageUpdates()  { throw std::runtime_error("Engine_AVX2_Multithread::DoPreVoltageUpdates without thread ID"); }
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() { throw std::runtime_error("Engine_AVX2_Multithread::DoPostVoltageUpdates without thread ID"); }
	virtual void DoPostVoltageUpdates(int threadID);
	virtual void Apply2Voltages()       { throw std::runtime_error("Engine_AVX2_Multithread::Apply2Voltages without thread ID"); }
	virtual void Apply2Voltages(int threadID);

	virtual void DoPreCurrentUpdates()  { throw std::runtime_error("Engine_AVX2_Multithread::DoPreCurrentUpdates without thread ID"); }
	virtual void DoPreCurrentUpdates(int threadID);
	virtual void DoPostCurrentUpdates() { throw std::runtime_error("Engine_AVX2_Multithread::DoPostCurrentUpdates without thread ID"); }
	virtual void DoPostCurrentUpdates(int threadID);
	virtual void Apply2Current()        { throw std::runtime_error("Engine_AVX2_Multithread::Apply2Current without thread ID"); }
	virtual void Apply2Current(int threadID);

protected:
	Engine_AVX2_Multithread(const Operator_AVX2_Multithread* op);
	void changeNumThreads(unsigned int numThreads);
	const Operator_AVX2_Multithread* m_Op_MT;
	std::vector<std::thread> m_threads;
	Barrier* m_startBarrier;
	Barrier* m_stopBarrier;
	Barrier* m_IterateBarrier;
	volatile unsigned int m_iterTS;
	unsigned int m_numThreads;
	unsigned int m_max_numThreads;
	volatile bool m_stopThreads;
	bool m_opt_speed;
	float m_last_speed;
};

#endif // ENGINE_AVX2_MULTITHREAD_H
