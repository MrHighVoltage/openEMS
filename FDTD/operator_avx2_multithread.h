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

#ifndef OPERATOR_AVX2_MULTITHREAD_H
#define OPERATOR_AVX2_MULTITHREAD_H

#include "operator_avx2.h"
#include <boost/thread.hpp>

class Operator_AVX2_Multithread;

class Operator_AVX2_Thread
{
public:
	Operator_AVX2_Thread(Operator_AVX2_Multithread* ptr, unsigned int start, unsigned int stop, unsigned int threadID);
	void operator()();

protected:
	unsigned int m_start, m_stop, m_threadID;
	Operator_AVX2_Multithread* m_OpPtr;
};

class Operator_AVX2_Multithread : public Operator_AVX2
{
	friend class Engine_AVX2_Multithread;
	friend class Operator_AVX2_Thread;
public:
	//! Create a new operator
	static Operator_AVX2_Multithread* New(unsigned int numThreads = 0);
	virtual ~Operator_AVX2_Multithread();

	virtual void setNumThreads(unsigned int numThreads);

	virtual Engine* CreateEngine();

protected:
	Operator_AVX2_Multithread();
	virtual void Init();
	void Delete();
	virtual void Reset();

	virtual bool Calc_EC(); //!< uses multi-threading

	unsigned int (*m_Nr_PEC_thread)[3]; //!< count PEC edges per thread
	virtual bool CalcPEC(); //!< uses multi-threading

	virtual int CalcECOperator(DebugFlags debugFlags = None);

	//Calc_EC barrier
	boost::barrier* m_CalcEC_Start;
	boost::barrier* m_CalcEC_Stop;
	//CalcPEC barrier
	boost::barrier* m_CalcPEC_Start;
	boost::barrier* m_CalcPEC_Stop;

	boost::thread_group m_thread_group;
	unsigned int m_numThreads;
	unsigned int m_orig_numThreads;

	//! Calculate the start/stop lines for multithreading
	virtual void CalcStartStopLines(
		unsigned int &numThreads,
		std::vector<unsigned int> &start,
		std::vector<unsigned int> &stop
	) const;
};

#endif // OPERATOR_AVX2_MULTITHREAD_H
