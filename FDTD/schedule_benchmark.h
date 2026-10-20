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

#ifndef SCHEDULE_BENCHMARK_H
#define SCHEDULE_BENCHMARK_H

#include <chrono>
#include <ostream>
#include <string>
#include <vector>

#include "temporal_blocking.h"

class Engine;

//! --benchmark: an engine's temporal-blocking schedules, measured on the model it was set up for.
/*!
  Every schedule computes the same fields bit for bit (PERFORMANCE.md §5.1),
  so the schedules need no simulations of their own: the main loop keeps
  stepping the real model, excitation and probes included, and the schedule
  changes at the timesteps this class asks the loop to stop at.

  A warm-up runs the flat sweep for at least two seconds, and until the engine
  has finished calibrating itself (the AVX2 engine's thread count). Its speed
  sets the timesteps per run -- about three seconds of the flat sweep -- unless
  they are given. Then every candidate runs that many timesteps twice, the
  second round in reverse order, so that a clock or thermal drift does not
  favour either end of the list.

  Speed is all it measures, and it recommends the fastest, or the flat sweep
  unless a blocked schedule beats it by more than 1 %. A tie-break
  towards smaller tiles, for the DRAM energy a tile that stays in the cache
  saves at equal speed (PERFORMANCE.md §5.5), was tried and dropped: while
  both tiles fit, the deeper block of the larger one moves less data, and on
  a UHD 770 it chose a schedule 1.6 % slower than the fastest.
*/
class ScheduleBenchmark
{
public:
	//! \a timestepsPerRun 0: from the warm-up.
	ScheduleBenchmark(Engine* eng, double numCells, unsigned int timestepsPerRun);

	//! The timestep the main loop has to stop at next.
	unsigned int Boundary() const {return m_boundary;}
	//! \a step, shortened so that it does not run past Boundary().
	int Clamp(int step, unsigned int currTS) const;
	//! The main loop's next processing interval (PA->Process()); it caps the block depth.
	void NoteInterval(int step);
	//! At Boundary(), with the engine idle: score the run that ended and start
	//! the next. False once every run is done.
	bool Advance(unsigned int currTS);
	bool Finished() const {return m_phase == DONE;}

	//! The table, and the options that reproduce the recommended schedule.
	/*!
	  \a engineName is the --engine value, \a optionName the engine's
	  temporal-blocking option, empty if it has none.
	*/
	void Report(std::ostream& os, const std::string& engineName, const std::string& optionName) const;

private:
	struct Candidate
	{
		TemporalBlockingRequest req;
		TemporalBlockingPlan plan;
		bool configured = false;  //!< what the run would have used
		std::vector<double> mcs;  //!< one per round
		double Mean() const;
		std::string Label(unsigned int maxDepth) const;
	};
	enum Phase {WARMUP, RUNS, DONE};

	void Start(unsigned int currTS, unsigned int length);
	void BuildCandidates();
	unsigned int MaxDepth() const;

	Engine* m_eng;
	double m_cells;
	unsigned int m_runTS;
	Phase m_phase;
	unsigned int m_boundary;   //!< end of the current warm-up step or run
	unsigned int m_startTS;    //!< its first timestep
	std::chrono::steady_clock::time_point m_startTime;
	double m_warmSeconds;      //!< warm-up so far
	double m_warmRate;         //!< timesteps per second of its last step
	unsigned int m_maxInterval; //!< longest processing interval seen; 0 = none
	TemporalBlockingRequest m_configured;
	std::string m_noBlocking;  //!< why no blocked schedule applies, if none does
	std::vector<Candidate> m_cand; //!< flat first, then by tile size
	std::vector<size_t> m_order;   //!< indices into m_cand, all rounds in sequence
	size_t m_pos;
};

#endif // SCHEDULE_BENCHMARK_H
