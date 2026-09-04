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

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <unistd.h>

using std::cout;
using std::endl;

namespace
{
// Calibration batch granularity. Small enough that a candidate is scored
// promptly even when the caller asks for many timesteps at once, large enough
// that the two barrier round-trips per timestep are not what is being timed.
const unsigned int CAL_BATCH_TS = 8;
// A candidate is only scored once it has run at least this many timesteps and
// this much wall time, so that a fast in-cache grid is not ranked on noise.
const unsigned int CAL_MIN_TS = 16;
const double CAL_MIN_SECONDS = 0.025;
}

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
	m_opt_speed = false;
	m_stopThreads = true;
	m_cal_index = 0;
	m_cal_time = 0.0;
	m_cal_steps = 0;
	m_cal_refined = false;
	m_cal_warmup = true;
	m_blk_k = 0;
	m_blk_W = 0;
	m_blk_active = false;
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
		BuildCalibrationPlan();
		m_numThreads = m_cal_plan.front();
	}
	else if (m_numThreads > m_max_numThreads)
		m_numThreads = m_max_numThreads;

	this->changeNumThreads(m_numThreads);
	ConfigureTemporalBlocking();
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
	if (m_opt_speed)
		return CalibrateIterateTS(iterTS);   // calibrate on the flat path

	m_iterTS = iterTS;
	m_blk_active = (m_blk_k > 1);

	m_startBarrier->wait(); // start threads
	m_stopBarrier->wait();  // wait for completion

	m_blk_active = false;
	return true;
}

// ----------------------------------------------------------------------------
// Trapezoidal temporal blocking  (prototype -- off unless asked for)
//
// The flat sweep streams the whole grid from DRAM once per timestep. Advancing
// a cache-resident tile through k timesteps before moving on divides that
// traffic by roughly k, which on this class of host is worth 3-4x
// (OPTIMIZATIONS.md §4.7/C2). The schedule below is the one verified
// bit-identical to the flat sweep in python/Tests/bench_temporal_blocking.c.
//
// Two passes per block. Pass 1 runs "cores": tiles of width W whose x-range
// narrows by one per side per timestep, because a cell can only be advanced
// while its neighbours are at the right time. That leaves wedges of
// un-advanced cells at every tile boundary, which pass 2 fills with trapezoids
// that widen instead.
//
// Enabled by OPENEMS_AVX2_TEMPORAL_BLOCK=<k>, and only when every active
// extension can be applied to an x-range at a chosen timestep
// (Engine_Extension::SupportsSlabApply), which today means the excitation and
// nothing else. Anything richer -- UPML, dispersive, probes -- falls back to
// the flat sweep, so this is safe by omission rather than by enumeration.
// ----------------------------------------------------------------------------

void Engine_AVX2_Multithread::ConfigureTemporalBlocking()
{
	m_blk_k = 0;
	m_blk_W = 0;

	// OPENEMS_AVX2_TEMPORAL_BLOCK=<k> derives the tile width from the last-level
	// cache; <k>:<W> sets both by hand.
	const char* env = std::getenv("OPENEMS_AVX2_TEMPORAL_BLOCK");
	if (!env)
		return;
	char* endPtr = nullptr;
	long k = std::strtol(env, &endPtr, 10);
	if (!endPtr || k < 2 || (*endPtr != '\0' && *endPtr != ':'))
		return;

	// One tile of both field arrays should be about one last-level cache. That
	// is not obvious a priori -- a trapezoid's swept range shrinks from W to
	// W-2k as it advances, so its live set is smaller than W planes -- but it
	// is what the measurements land on: the optimum was W=64 on a grid with
	// 0.56 MB x-planes and W=32 on one with 1.20 MB, i.e. 36 and 38 MB, against
	// this host's 36 MB L3.
	const double planeBytes = 2.0 * (double)m_vs_x * sizeof(f8vector);
	long cacheBytes = 0;
#ifdef _SC_LEVEL3_CACHE_SIZE
	cacheBytes = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
	if (cacheBytes <= 0)
		cacheBytes = 32L * 1024 * 1024;
	long W = (long)((double)cacheBytes / planeBytes);

	if (*endPtr == ':')
	{
		char* wEnd = nullptr;
		W = std::strtol(endPtr + 1, &wEnd, 10);
		if (!wEnd || *wEnd != '\0')
			return;
	}

	// W >= 2k is a hard constraint of the geometry, not a tuning choice: a
	// wedge's dependencies reach k cells into each neighbouring core, so
	// narrower tiles cannot supply them. Prefer a shallower block over a tile
	// that will not stay resident.
	if (W < 2 * k)
		k = W / 2;
	if (k < 2)
	{
		cout << "AVX2 temporal blocking: disabled, x-planes too large ("
		     << (planeBytes / (1024 * 1024)) << " MB) for a useful tile." << endl;
		return;
	}

	for (size_t n = 0; n < m_Eng_exts.size(); ++n)
	{
		if (!m_Eng_exts.at(n)->SupportsSlabApply())
		{
			cout << "AVX2 temporal blocking: disabled, extension '"
			     << m_Eng_exts.at(n)->GetExtensionName()
			     << "' cannot be applied per x-range." << endl;
			return;
		}
	}

	// W >= 2k is a hard constraint of the geometry, not a tuning choice: a
	// wedge's dependencies reach k cells into each neighbouring core, so
	// narrower tiles cannot supply them.
	// Room for at least two cores, so there is a wedge to fill.
	if ((long)numLines[0] < 2 * W)
	{
		cout << "AVX2 temporal blocking: disabled, only " << numLines[0]
		     << " x-lines for a tile width of " << W << "." << endl;
		return;
	}

	// Below roughly one L3 the flat sweep is already cache-resident and there is
	// no traffic to remove; blocking then costs the wedge work for nothing
	// (measured -32% on a 96^3 grid), so do not engage.
	const double fieldBytes = planeBytes * (double)numLines[0];
	if (fieldBytes < 48.0 * 1024 * 1024)
	{
		cout << "AVX2 temporal blocking: disabled, grid fits cache ("
		     << (fieldBytes / (1024 * 1024)) << " MB of field state)." << endl;
		return;
	}

	m_blk_k = (unsigned int)k;
	m_blk_W = (unsigned int)W;
	cout << "AVX2 temporal blocking ACTIVE (prototype): k=" << m_blk_k
	     << ", tile width " << m_blk_W << " x-lines, "
	     << (planeBytes * m_blk_W / (1024 * 1024)) << " MB per tile" << endl;
}

void Engine_AVX2_Multithread::SplitRange(int lo, int hi, unsigned int nThreads,
                                         unsigned int id, unsigned int& start,
                                         unsigned int& stop)
{
	if (hi <= lo || nThreads == 0) { start = stop = 0; return; }
	const unsigned int n = (unsigned int)(hi - lo);
	const unsigned int base = n / nThreads;
	const unsigned int rem  = n % nThreads;
	const unsigned int off  = id * base + (id < rem ? id : rem);
	const unsigned int cnt  = base + (id < rem ? 1u : 0u);
	start = (unsigned int)lo + off;
	stop  = start + cnt;
}

void Engine_AVX2_Multithread::TrapezoidSweep(int A0, int B0, int k, int dir,
                                             int t0, unsigned int threadID)
{
	const int NX = (int)numLines[0];
	// UpdateCurrents reads volt at x+1, so the engine never updates H on the
	// last x-line -- the same reason the flat path gives its last thread
	// stop_h = stop-1. That line is therefore always "valid" and needs no slope.
	const int NXH = NX - 1;

	int A = A0, B = B0;
	if (A < 0)  A = 0;
	if (B > NX) B = NX;

	for (int t = 0; t < k; ++t)
	{
		// The slope exists only because data outside [A,B) is at the wrong
		// time. At a domain edge there is no outside -- the update kernels
		// clamp -- so the edge is held, not sloped, or its cells never advance.
		int An = (A <= 0)  ? 0  : A + dir;
		int Bn = (B >= NX) ? NX : B - dir;
		if (An < 0)  An = 0;
		if (Bn > NX) Bn = NX;

		// The E sweep range is not symmetric between the two directions: the
		// passes must partition the grid so every cell is updated exactly once
		// per timestep. A narrowing core leaves gaps of width 2t+1 in E but
		// 2t+2 in H, so a widening wedge sweeps E one cell narrower per side.
		int eLo, eHi;
		if (dir > 0) { eLo = An;                     eHi = (Bn >= NX) ? NX : Bn + 1; }
		else         { eLo = (An <= 0) ? 0 : An + 1; eHi = (Bn >= NX) ? NX : Bn; }
		if (eHi > NX)  eHi = NX;
		if (eHi < eLo) eHi = eLo;

		unsigned int s, e;
		SplitRange(eLo, eHi, m_numThreads, threadID, s, e);
		if (e > s) UpdateVoltages(s, e - s);
		m_IterateBarrier->wait();

		if (threadID == 0)
			for (size_t n = 0; n < m_Eng_exts.size(); ++n)
				m_Eng_exts.at(n)->Apply2VoltagesSlab((unsigned int)eLo, (unsigned int)eHi, t0 + t);
		m_IterateBarrier->wait();

		const int hHi = (Bn < NXH) ? Bn : NXH;
		SplitRange(An, hHi, m_numThreads, threadID, s, e);
		if (e > s) UpdateCurrents(s, e - s);
		m_IterateBarrier->wait();

		if (threadID == 0)
			for (size_t n = 0; n < m_Eng_exts.size(); ++n)
				m_Eng_exts.at(n)->Apply2CurrentSlab((unsigned int)An,
				                                    (unsigned int)(hHi < An ? An : hHi), t0 + t);
		m_IterateBarrier->wait();

		A = An;
		B = Bn;
	}
}

void Engine_AVX2_Multithread::BlockedWorker(unsigned int threadID)
{
	const int NX = (int)numLines[0];
	const unsigned int total = m_iterTS;
	const int startTS = (int)numTS;

	unsigned int done = 0;
	while (done < total)
	{
		const int kk = (int)std::min((unsigned int)m_blk_k, total - done);
		const int t0 = startTS + (int)done;

		for (int x0 = 0; x0 < NX; x0 += (int)m_blk_W)
			TrapezoidSweep(x0, x0 + (int)m_blk_W, kk, +1, t0, threadID);
		for (int c = (int)m_blk_W; c < NX; c += (int)m_blk_W)
			TrapezoidSweep(c, c, kk, -1, t0, threadID);

		done += (unsigned int)kk;
	}

	// Every thread read numTS before the block's first barrier and none reads
	// it again, so the last barrier above orders this write after all of them.
	if (threadID == 0)
		numTS += total;
}

// ----------------------------------------------------------------------------
// Thread-count calibration
//
// Thread count does not affect results -- the domain decomposition is by
// x-line and every pass is barrier-separated -- so candidates can be scored on
// *real* timesteps rather than on a warm-up that is thrown away. Calibration is
// therefore free apart from the timesteps that run at a suboptimal width, which
// is why it probes a geometric ladder (a handful of candidates, ~25 ms each)
// instead of climbing one thread at a time.
//
// The predecessor of this code hill-climbed from one thread inside
// NextInterval(), which the main loop only calls every four seconds. On a
// 4-core host that settled in a few intervals; on a 24-core host it needed 44 s
// to reach its optimum, so any run shorter than a minute spent most of its time
// at a fraction of peak throughput.
// ----------------------------------------------------------------------------

void Engine_AVX2_Multithread::BuildCalibrationPlan()
{
	m_cal_plan.clear();
	m_cal_score.clear();
	m_cal_index = 0;
	m_cal_time = 0.0;
	m_cal_steps = 0;
	m_cal_refined = false;
	m_cal_warmup = true;

	if (m_max_numThreads < 1)
		m_max_numThreads = 1;

	for (unsigned int n = 1; n < m_max_numThreads; n *= 2)
		m_cal_plan.push_back(n);
	m_cal_plan.push_back(m_max_numThreads);
}

bool Engine_AVX2_Multithread::CalibrateIterateTS(unsigned int iterTS)
{
	unsigned int remaining = iterTS;

	while (remaining > 0 && m_opt_speed)
	{
		const unsigned int batch = std::min(remaining, CAL_BATCH_TS);

		const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		m_iterTS = batch;
		m_startBarrier->wait();
		m_stopBarrier->wait();
		const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

		remaining -= batch;

		// The first batch at a new width runs on freshly spawned threads with
		// cold private caches, which systematically penalises the wider
		// configurations -- exactly the ones being judged. Throw it away.
		if (m_cal_warmup)
		{
			m_cal_warmup = false;
			continue;
		}

		m_cal_steps += batch;
		m_cal_time += std::chrono::duration<double>(t1 - t0).count();

		if (m_cal_steps >= CAL_MIN_TS && m_cal_time >= CAL_MIN_SECONDS)
			AdvanceCalibration();
	}

	if (remaining > 0)
	{
		m_iterTS = remaining;
		m_startBarrier->wait();
		m_stopBarrier->wait();
	}

	return true;
}

void Engine_AVX2_Multithread::AdvanceCalibration()
{
	m_cal_score.push_back(m_cal_time > 0.0 ? m_cal_steps / m_cal_time : 0.0);

	if (g_settings.GetVerboseLevel() > 0)
		cout << "AVX2 Multithreaded Engine: " << std::setw(3) << m_cal_plan.at(m_cal_index)
		     << " threads -> " << std::setprecision(1) << std::fixed
		     << m_cal_score.back() << " timesteps/s" << endl;

	++m_cal_index;

	if (m_cal_index >= m_cal_plan.size())
	{
		if (m_cal_refined)
		{
			FinishCalibration();
			return;
		}
		m_cal_refined = true;

		// Bisect the two ladder gaps flanking the winner. The throughput plateau
		// around the optimum is broad, so one refinement pass is enough to land
		// within a couple of percent; a full linear scan would cost more in
		// slow timesteps than it recovers.
		const size_t best = std::max_element(m_cal_score.begin(), m_cal_score.end())
		                    - m_cal_score.begin();
		const unsigned int b = m_cal_plan.at(best);
		const unsigned int lo = (best > 0) ? m_cal_plan.at(best - 1) : b;
		const unsigned int hi = (best + 1 < m_cal_plan.size()) ? m_cal_plan.at(best + 1) : b;

		const unsigned int mids[2] = { (lo + b) / 2, (b + hi) / 2 };
		for (unsigned int i = 0; i < 2; ++i)
		{
			const unsigned int m = mids[i];
			if (m == 0 || m > m_max_numThreads)
				continue;
			if (std::find(m_cal_plan.begin(), m_cal_plan.end(), m) != m_cal_plan.end())
				continue;
			m_cal_plan.push_back(m);
		}

		if (m_cal_index >= m_cal_plan.size())
		{
			FinishCalibration();
			return;
		}
	}

	m_cal_time = 0.0;
	m_cal_steps = 0;
	m_cal_warmup = true;
	this->changeNumThreads(m_cal_plan.at(m_cal_index));
}

void Engine_AVX2_Multithread::FinishCalibration()
{
	const size_t best = std::max_element(m_cal_score.begin(), m_cal_score.end())
	                    - m_cal_score.begin();
	const unsigned int bestThreads = m_cal_plan.at(best);

	m_opt_speed = false;
	m_cal_time = 0.0;
	m_cal_steps = 0;

	if (bestThreads != m_numThreads)
		this->changeNumThreads(bestThreads);

	cout << "AVX2 Multithreaded Engine: Best performance found using "
	     << m_numThreads << " threads." << endl;
}

void Engine_AVX2_Multithread::NextInterval(float curr_speed)
{
	// Thread-count tuning happens in CalibrateIterateTS(), which self-times on
	// real timesteps instead of waiting for this callback's four-second cadence.
	Engine_AVX2::NextInterval(curr_speed);
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

		if (m_enginePtr->m_blk_active)
		{
			m_enginePtr->BlockedWorker(m_threadID);
			m_enginePtr->m_stopBarrier->wait();
			continue;
		}

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
