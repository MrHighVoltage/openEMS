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

#ifndef CPU_TOPOLOGY_H
#define CPU_TOPOLOGY_H

#include <string>
#include <vector>

//! Which cores an engine's worker threads may run on (--cpu-cores).
/*!
  Named by kind rather than by CPU number, so the same value means the same
  thing on another machine and on every operating system: "performance" is
  the P-cores of an Intel hybrid, the big cores of an ARM big.LITTLE, and
  simply every core of a CPU with one kind.
*/
struct CoreSet
{
	enum Kinds
	{
		ALL_KINDS,   //!< every kind of core
		PERFORMANCE, //!< the fastest kind only
		EFFICIENCY   //!< the slowest kind only
	};
	Kinds kinds = ALL_KINDS;
	bool smt = true; //!< every hardware thread of those cores; false: one per core

	//! Read "all", "physical", "performance", "performance+smt", "efficiency"
	//! or "efficiency+smt"; false, and \a set untouched, on anything else.
	static bool Parse(const std::string& text, CoreSet& set);
	//! What Parse() reads back.
	std::string ToString() const;
	bool IsAll() const {return kinds == ALL_KINDS && smt;}
	bool operator==(const CoreSet& o) const {return kinds == o.kinds && smt == o.smt;}
	bool operator!=(const CoreSet& o) const {return !(*this == o);}
};

//! The last-level cache a set of threads can use.
struct LLC_Span
{
	unsigned int level;               //!< 3 for an L3, 2 where the L2 is the last level; 0 if unknown
	unsigned int instances;           //!< instances counted; 0 if unknown
	unsigned long long instanceBytes; //!< the largest of them
	unsigned long long totalBytes;    //!< all of them together
	const char* source;               //!< where the sizes came from
};

//! The processors this process may run on, as the operating system describes them.
/*!
  Read once per process from the operating system's own interfaces, with no
  library: sysfs on Linux (x86 and ARM alike), GetLogicalProcessorInformationEx()
  on Windows, sysctl on macOS. What one of them does not tell is left unknown
  rather than guessed, and the callers fall back accordingly.

  Kinds of core: on Linux the Intel hybrid PMUs (cpu_core, cpu_atom), else
  cpu_capacity (ARM), else the highest clock -- in that order, the first that
  splits the CPUs. A kind starts where a CPU is more than 15 % slower than the
  next faster one, so that the two favoured P-cores of an i9-13900K (5.8 GHz
  instead of 5.5) stay P-cores. Windows has an efficiency class per core,
  macOS a performance level.

  Restricting threads: an affinity mask on Linux and Windows. macOS has none;
  there a thread's QoS class steers it to the performance or the efficiency
  cores of an Apple CPU, so those two sets are offered and nothing finer.
*/
class CpuTopology
{
public:
	struct Cpu
	{
		unsigned int os;   //!< the OS's number for it: Linux CPU number, Windows group * 64 + index
		unsigned int core; //!< the physical core it belongs to, counted from 0
		unsigned int kind; //!< 0 = the fastest kind of core
		int cache;         //!< its last-level cache, an index into Caches(); -1 if unknown
	};
	struct Cache
	{
		unsigned int level;
		unsigned long long bytes;
	};

	//! Detected on first use.
	static const CpuTopology& Get();
	//! Linux: read the sysfs at \a root (normally "/sys") for the CPUs \a allowed. Used by Get(), and by tests.
	static CpuTopology FromSysfs(const std::string& root, const std::vector<unsigned int>& allowed);

	const std::vector<Cpu>& Cpus() const {return m_cpus;}
	const std::vector<Cache>& Caches() const {return m_caches;}
	unsigned int NumCores() const {return m_numCores;}
	unsigned int NumKinds() const {return m_numKinds;}
	const std::string& Source() const {return m_source;}

	//! Whether threads can be kept to \a set here.
	bool Offers(const CoreSet& set) const;
	//! The sets that differ on this machine, in the order --benchmark measures them.
	std::vector<CoreSet> DistinctSets() const;
	//! Indices into Cpus() of the CPUs \a set allows.
	std::vector<unsigned int> Select(const CoreSet& set) const;
	//! Hardware threads in \a set: the most workers worth starting there.
	unsigned int MaxThreads(const CoreSet& set) const;
	//! The last-level caches of \a set's CPUs, at most \a maxThreads instances of them.
	/*!
	  A run cannot occupy more caches than it has threads; which ones it
	  occupies is the scheduler's choice, so these are the first in CPU order.
	*/
	LLC_Span LastLevelCaches(const CoreSet& set, unsigned int maxThreads) const;
	//! Keep the calling thread, worker \a index of \a count, to \a set. False if that cannot be done here.
	bool Restrict(const CoreSet& set, unsigned int index, unsigned int count) const;

	//! "performance", "efficiency", or "" on a CPU with one kind.
	std::string KindName(unsigned int kind) const;
	//! What \a set selects here, in words: "8 performance cores, 16 threads".
	std::string DescribeSet(const CoreSet& set) const;
	//! One line: the kinds of core, their threads, and the last-level caches.
	std::string Describe() const;

private:
	enum Placement {NONE, AFFINITY, QOS};
	struct Raw; //!< one CPU as a detector found it

	static CpuTopology Build(const std::vector<Raw>& raw, Placement placement, const std::string& source);
	static CpuTopology Detect();
	static CpuTopology Fallback();
#ifdef _WIN32
	static CpuTopology DetectWindows();
#endif
#ifdef __APPLE__
	static CpuTopology DetectMacOS();
#endif

	std::vector<Cpu> m_cpus;
	std::vector<Cache> m_caches;
	unsigned int m_numCores = 0;
	unsigned int m_numKinds = 1;
	Placement m_placement = NONE;
	std::string m_source;      //!< of the CPUs and cores
	std::string m_cacheSource; //!< of the cache sizes
};

//! The last-level caches of every CPU the process may run on, at most \a maxThreads instances.
LLC_Span QueryLastLevelCaches(unsigned int maxThreads);

#endif // CPU_TOPOLOGY_H
