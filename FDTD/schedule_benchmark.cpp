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

#include "schedule_benchmark.h"
#include "engine.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{
const double WARMUP_SECONDS = 2.0;  //!< GPU clocks ramp up, CPU turbo settles
const double WARMUP_STEP    = 0.25; //!< seconds per warm-up step
const double RUN_SECONDS    = 3.0;  //!< one run of the flat sweep
const unsigned int ROUNDS   = 2;
const double NOISE          = 0.02; //!< a spread between rounds worth a warning
const double GAIN           = 0.01; //!< what a blocked schedule has to beat flat by to be recommended

//! \a x rounded up to two significant digits, so the timesteps per run read as chosen.
unsigned int RoundUp2(double x)
{
	if (x < 100)
		return (unsigned int)std::max(1.0, std::ceil(x));
	const double unit = std::pow(10.0, std::floor(std::log10(x)) - 1);
	return (unsigned int)(std::ceil(x / unit) * unit);
}

std::string Percent(double ratio)
{
	std::ostringstream os;
	os << std::showpos << std::fixed << std::setprecision(1) << 100.0 * (ratio - 1.0) << " %";
	return os.str();
}
}

double ScheduleBenchmark::Candidate::Mean() const
{
	if (mcs.empty())
		return 0;
	double sum = 0;
	for (double v : mcs)
		sum += v;
	return sum / mcs.size();
}

std::string ScheduleBenchmark::Candidate::Label(unsigned int maxDepth) const
{
	if (!plan.active)
		return "flat";
	std::ostringstream os;
	os << "k=" << std::min(plan.k, maxDepth) << " W=" << plan.W;
	return os.str();
}

ScheduleBenchmark::ScheduleBenchmark(Engine* eng, double numCells, unsigned int timestepsPerRun)
	: m_eng(eng), m_cells(numCells), m_runTS(timestepsPerRun), m_phase(WARMUP),
	  m_boundary(0), m_startTS(0), m_warmSeconds(0), m_warmRate(0), m_maxInterval(0), m_pos(0)
{
	m_configured = eng->GetTemporalBlocking();
	eng->SetTemporalBlocking(TemporalBlockingRequest::Off(), false);
	std::cout << "Benchmark: warming up on the flat sweep..." << std::endl;
	Start(eng->GetNumberOfTimesteps(), 4);
}

void ScheduleBenchmark::Start(unsigned int currTS, unsigned int length)
{
	m_startTS = currTS;
	m_boundary = currTS + std::max(1u, length);
	m_startTime = std::chrono::steady_clock::now();
}

int ScheduleBenchmark::Clamp(int step, unsigned int currTS) const
{
	if (m_phase == DONE || currTS >= m_boundary)
		return step;
	return std::min(step, (int)(m_boundary - currTS));
}

void ScheduleBenchmark::NoteInterval(int step)
{
	// Only the warm-up's intervals shape the candidates. PA->Process() returns
	// the timesteps to the next probe or dump, and the main loop hands the
	// engine at most that many at a time -- which is the deepest block it
	// can run.
	if (m_phase == WARMUP && step > 0 && (unsigned int)step > m_maxInterval)
		m_maxInterval = (unsigned int)step;
}

unsigned int ScheduleBenchmark::MaxDepth() const
{
	// Beyond 32 no engine blocks anyway; a longer interval means "no cap".
	return (m_maxInterval > 0 && m_maxInterval < 32) ? m_maxInterval : UINT_MAX;
}

void ScheduleBenchmark::BuildCandidates()
{
	const unsigned int maxDepth = MaxDepth();

	Candidate flat;
	flat.req = TemporalBlockingRequest::Off();
	m_cand.push_back(flat);

	// The ladder first and the configured schedule last, so that where the
	// two coincide the row is marked rather than repeated.
	std::vector<TemporalBlockingRequest> reqs = m_eng->TemporalBlockingCandidates(maxDepth);
	reqs.push_back(m_configured);
	for (size_t i = 0; i < reqs.size(); ++i)
	{
		const bool configured = (i + 1 == reqs.size());
		const TemporalBlockingPlan plan = m_eng->PlanTemporalBlocking(reqs[i]);
		if (!plan.active)
		{
			if (configured)
				m_cand.front().configured = true;
			else if (m_noBlocking.empty())
				m_noBlocking = plan.why;
			continue;
		}
		const unsigned int k = std::min(plan.k, maxDepth);
		auto same = std::find_if(m_cand.begin(), m_cand.end(), [&](const Candidate& c)
		{
			return c.plan.active && std::min(c.plan.k, maxDepth) == k && c.plan.W == plan.W;
		});
		if (same != m_cand.end())
		{
			same->configured = same->configured || configured;
			continue;
		}
		Candidate c;
		c.req = TemporalBlockingRequest::Forced(k, plan.W);
		c.plan = plan;
		c.configured = configured;
		m_cand.push_back(c);
	}
	std::sort(m_cand.begin() + 1, m_cand.end(), [](const Candidate& a, const Candidate& b)
	{
		return a.plan.W != b.plan.W ? a.plan.W < b.plan.W : a.plan.k < b.plan.k;
	});

	// Forward, then backward: a drift in clock or temperature over the
	// benchmark then shows as a difference between the rounds instead of
	// favouring one end of the list.
	for (unsigned int r = 0; r < ROUNDS; ++r)
		for (size_t i = 0; i < m_cand.size(); ++i)
			m_order.push_back((r % 2 == 0) ? i : m_cand.size() - 1 - i);
}

bool ScheduleBenchmark::Advance(unsigned int currTS)
{
	const auto now = std::chrono::steady_clock::now();
	const double seconds = std::chrono::duration<double>(now - m_startTime).count();
	const unsigned int steps = currTS - m_startTS;

	if (m_phase == DONE)
		return false;

	if (m_phase == WARMUP)
	{
		m_warmSeconds += seconds;
		if (steps > 0 && seconds > 0)
			m_warmRate = steps / seconds;
		const bool settled = m_warmSeconds >= WARMUP_SECONDS && seconds >= 0.5 * WARMUP_STEP &&
		                     !m_eng->IsCalibrating();
		if (!settled)
		{
			Start(currTS, (unsigned int)std::max(1.0, m_warmRate * WARMUP_STEP));
			return true;
		}

		if (m_runTS == 0)
			m_runTS = RoundUp2(m_warmRate * RUN_SECONDS);
		BuildCandidates();
		std::cout << "Benchmark: warm-up " << std::fixed << std::setprecision(1) << m_warmSeconds
		          << " s, flat " << m_cells * m_warmRate * 1e-6 << " MC/s; " << m_cand.size()
		          << " schedules x " << ROUNDS << " rounds of " << m_runTS << " timesteps" << std::endl;
		m_phase = RUNS;
		m_pos = 0;
	}
	else
	{
		Candidate& c = m_cand[m_order[m_pos]];
		const double mcs = (seconds > 0) ? m_cells * steps / seconds * 1e-6 : 0;
		c.mcs.push_back(mcs);
		std::cout << "Benchmark " << std::setw(3) << m_pos + 1 << "/" << m_order.size() << "  "
		          << std::left << std::setw(14) << c.Label(MaxDepth()) << std::right
		          << std::fixed << std::setprecision(1) << std::setw(10) << mcs << " MC/s" << std::endl;
		if (++m_pos == m_order.size())
		{
			m_phase = DONE;
			return false;
		}
	}

	m_eng->SetTemporalBlocking(m_cand[m_order[m_pos]].req, false);
	Start(currTS, m_runTS);
	return true;
}

void ScheduleBenchmark::Report(std::ostream& os, const std::string& engineName,
                               const std::string& optionName) const
{
	const std::ios::fmtflags flags = os.flags();
	const std::streamsize precision = os.precision();
	const unsigned int maxDepth = MaxDepth();

	os << std::endl << "Benchmark results (--engine=" << engineName << "): " << std::fixed
	   << std::setprecision(2) << m_cells * 1e-6 << " M cells, " << m_runTS << " timesteps per run, "
	   << ROUNDS << " rounds; probes kept, field dumps off" << std::endl;
	if (m_cand.empty())
	{
		os << "  interrupted during the warm-up" << std::endl;
		os.flags(flags);
		os.precision(precision);
		return;
	}
	if (maxDepth != UINT_MAX)
		os << "  probes every " << m_maxInterval << " timesteps cap the block depth at "
		   << m_maxInterval << std::endl;

	const double flatMean = m_cand.front().Mean();
	size_t best = 0;
	double spread = 0;
	for (size_t i = 0; i < m_cand.size(); ++i)
	{
		const Candidate& c = m_cand[i];
		if (c.Mean() > m_cand[best].Mean())
			best = i;
		if (c.mcs.size() > 1)
		{
			const auto mm = std::minmax_element(c.mcs.begin(), c.mcs.end());
			spread = std::max(spread, (*mm.second - *mm.first) / c.Mean());
		}
	}

	os << std::endl << "  schedule         tile      MC/s";
	for (unsigned int r = 0; r < ROUNDS; ++r)
		os << "   round " << r + 1;
	os << "    vs flat" << std::endl;
	for (const Candidate& c : m_cand)
	{
		os << "  " << std::left << std::setw(14) << c.Label(maxDepth) << std::right;
		if (c.plan.active)
			os << std::setw(5) << std::setprecision(0) << c.plan.tileMB << " MB";
		else
			os << std::setw(8) << "-";
		os << std::setprecision(1);
		if (c.mcs.empty())
			os << std::setw(10) << "-";
		else
			os << std::setw(10) << c.Mean();
		for (unsigned int r = 0; r < ROUNDS; ++r)
		{
			if (r < c.mcs.size())
				os << std::setw(10) << c.mcs[r];
			else
				os << std::setw(10) << "-";
		}
		if (c.plan.active && flatMean > 0 && c.Mean() > 0)
			os << std::setw(11) << Percent(c.Mean() / flatMean);
		else
			os << std::setw(11) << "-";
		if (c.configured)
			os << "   as configured";
		os << std::endl;
	}
	os << std::endl;

	if (!Finished())
		os << "  Interrupted: the means above cover only the runs that finished." << std::endl;
	if (m_cand.size() == 1)
	{
		if (optionName.empty())
			os << "  This engine has no temporal blocking; the flat sweep is all it runs." << std::endl;
		else if (!m_noBlocking.empty())
			os << "  No blocked schedule applies: " << m_noBlocking << std::endl;
		os.flags(flags);
		os.precision(precision);
		return;
	}

	// A blocked schedule has to be measurably faster than the flat sweep to be
	// worth recommending: on a GPU that cannot keep a tile in its cache every
	// schedule runs at the flat speed (a Vega 7 within 0.1 %), and the fastest
	// of them is the noise.
	const Candidate& f = m_cand[best];
	const bool gains = f.plan.active && f.Mean() > (1.0 + GAIN) * flatMean;
	const Candidate& r = gains ? f : m_cand.front();
	os << std::setprecision(1) << (Finished() ? "  Fastest: " : "  Fastest so far: ")
	   << f.Label(maxDepth) << ", " << f.Mean() << " MC/s";
	if (f.plan.active && flatMean > 0)
		os << ", " << std::setprecision(2) << f.Mean() / flatMean << "x the flat sweep";
	os << "." << std::endl;
	if (!gains && f.plan.active)
		os << "  No blocked schedule beats the flat sweep by more than " << (int)(100 * GAIN)
		   << " %: blocking gains nothing on this model and machine, so the recommendation is flat." << std::endl;
	for (const Candidate& c : m_cand)
		if (c.configured && &c != &r && r.Mean() > 0 && !c.mcs.empty())
			os << "  As configured (" << c.Label(maxDepth) << "): " << std::setprecision(1) << c.Mean()
			   << " MC/s, " << 100.0 * (1.0 - c.Mean() / r.Mean()) << " % below the recommendation." << std::endl;
	if (spread > NOISE)
		os << "  The rounds differ by up to " << std::setprecision(1) << 100 * spread
		   << " % (other load, or clocks drifting): differences below that are noise." << std::endl;

	const std::string value = r.plan.active ? r.req.ToString() : "off";
	std::string keyword = optionName;
	std::replace(keyword.begin(), keyword.end(), '-', '_');
	os << "  To run this model with the recommendation on this machine:" << std::endl
	   << "    openEMS <model.xml> --engine=" << engineName << " --" << optionName << "=" << value << std::endl
	   << "    FDTD.Run(sim_path, engine='" << engineName << "', " << keyword << "='" << value << "')" << std::endl;
	if (r.configured)
		os << "  (This is what the run uses already.)" << std::endl;

	os.flags(flags);
	os.precision(precision);
}
