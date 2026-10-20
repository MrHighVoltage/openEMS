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

#include "cpu_topology.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__APPLE__)
#include <pthread/qos.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

struct CpuTopology::Raw
{
	unsigned int os = 0;
	std::string core;  //!< CPUs with the same key share a core; "" = a core of its own
	int kind = -1;     //!< smaller is faster; -1 = one kind
	std::string cache; //!< CPUs with the same key share a last-level cache; "" = unknown
	unsigned int cacheLevel = 0;
	unsigned long long cacheBytes = 0;
};

namespace
{
const struct
{
	const char* name;
	CoreSet::Kinds kinds;
	bool smt;
} SET_NAMES[] =
{
	{"all",             CoreSet::ALL_KINDS,   true},
	{"physical",        CoreSet::ALL_KINDS,   false},
	{"performance",     CoreSet::PERFORMANCE, false},
	{"performance+smt", CoreSet::PERFORMANCE, true},
	{"efficiency",      CoreSet::EFFICIENCY,  false},
	{"efficiency+smt",  CoreSet::EFFICIENCY,  true},
};

std::string FormatBytes(unsigned long long bytes)
{
	std::ostringstream os;
	if (bytes >= (1ull << 20))
		os << (bytes + (1ull << 19)) / (1ull << 20) << " MB";
	else
		os << (bytes + 512) / 1024 << " KB";
	return os.str();
}

#if defined(__linux__)
//! Where one kind of core ends: the next is more than this much slower.
const double KIND_GAP = 1.15;

//! Ranks, 0 = fastest, for CPUs of the given speeds; empty if they are all one kind.
std::vector<int> RankBySpeed(const std::vector<double>& speed)
{
	std::vector<double> levels(speed);
	std::sort(levels.begin(), levels.end(), std::greater<double>());
	levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
	if (levels.empty() || levels.back() <= 0)
		return {};
	std::map<double, int> rank;
	int r = 0;
	for (size_t i = 0; i < levels.size(); ++i)
	{
		if (i > 0 && levels[i] * KIND_GAP < levels[i - 1])
			++r;
		rank[levels[i]] = r;
	}
	if (r == 0)
		return {};
	std::vector<int> out;
	for (double s : speed)
		out.push_back(rank[s]);
	return out;
}

bool ReadText(const std::string& path, std::string& text)
{
	std::ifstream in(path.c_str());
	if (!in)
		return false;
	std::ostringstream os;
	os << in.rdbuf();
	text = os.str();
	while (!text.empty() && (text.back() == '\n' || text.back() == ' ' || text.back() == '\r'))
		text.pop_back();
	return true;
}

//! "0-3,8,10-11"
bool ParseCpuList(const std::string& text, std::vector<unsigned int>& cpus)
{
	cpus.clear();
	std::istringstream in(text);
	std::string part;
	while (std::getline(in, part, ','))
	{
		if (part.empty())
			continue;
		const size_t dash = part.find('-');
		char* end = nullptr;
		const unsigned long lo = std::strtoul(part.c_str(), &end, 10);
		if (end == part.c_str())
			return false;
		unsigned long hi = lo;
		if (dash != std::string::npos)
			hi = std::strtoul(part.c_str() + dash + 1, nullptr, 10);
		for (unsigned long c = lo; c <= hi && c < (1ul << 20); ++c)
			cpus.push_back((unsigned int)c);
	}
	return !cpus.empty();
}

bool ReadCpuList(const std::string& path, std::vector<unsigned int>& cpus)
{
	std::string text;
	return ReadText(path, text) && ParseCpuList(text, cpus);
}

//! "36864K"
unsigned long long ParseSize(const std::string& text)
{
	char* end = nullptr;
	unsigned long long v = std::strtoull(text.c_str(), &end, 10);
	if (end && (*end == 'K' || *end == 'k'))
		v *= 1024ull;
	else if (end && *end == 'M')
		v *= 1024ull * 1024;
	else if (end && *end == 'G')
		v *= 1024ull * 1024 * 1024;
	return v;
}

double ReadNumber(const std::string& path)
{
	std::string text;
	return ReadText(path, text) ? std::strtod(text.c_str(), nullptr) : 0;
}

std::vector<unsigned int> AllowedCpus()
{
	std::vector<unsigned int> cpus;
	for (int n = 1024; n <= (1 << 16); n *= 2)
	{
		cpu_set_t* set = CPU_ALLOC(n);
		if (!set)
			break;
		const size_t size = CPU_ALLOC_SIZE(n);
		CPU_ZERO_S(size, set);
		if (sched_getaffinity(0, size, set) == 0)
		{
			for (int c = 0; c < n; ++c)
				if (CPU_ISSET_S(c, size, set))
					cpus.push_back((unsigned int)c);
			CPU_FREE(set);
			return cpus;
		}
		CPU_FREE(set);
		if (errno != EINVAL)
			break;
	}
	ReadCpuList("/sys/devices/system/cpu/online", cpus);
	return cpus;
}
#endif
} // namespace

// ----------------------------------------------------------------------------
// CoreSet
// ----------------------------------------------------------------------------

bool CoreSet::Parse(const std::string& text, CoreSet& set)
{
	for (const auto& n : SET_NAMES)
	{
		if (text == n.name)
		{
			set.kinds = n.kinds;
			set.smt = n.smt;
			return true;
		}
	}
	return false;
}

std::string CoreSet::ToString() const
{
	for (const auto& n : SET_NAMES)
		if (n.kinds == kinds && n.smt == smt)
			return n.name;
	return "all";
}

// ----------------------------------------------------------------------------
// Detection
// ----------------------------------------------------------------------------

const CpuTopology& CpuTopology::Get()
{
	static const CpuTopology topology = Detect();
	return topology;
}

CpuTopology CpuTopology::Detect()
{
	CpuTopology t;
#if defined(__linux__)
	t = FromSysfs("/sys", AllowedCpus());
#elif defined(_WIN32)
	t = DetectWindows();
#elif defined(__APPLE__)
	t = DetectMacOS();
#endif
	if (t.m_cpus.empty())
		t = Fallback();

#ifdef _SC_LEVEL3_CACHE_SIZE
	// glibc on x86 still knows the L3 where sysfs has no cache directories
	// (some containers); one instance, shared by every CPU.
	if (t.m_caches.empty())
	{
		const long bytes = sysconf(_SC_LEVEL3_CACHE_SIZE);
		if (bytes > 0)
		{
			t.m_caches.push_back({3, (unsigned long long)bytes});
			for (Cpu& c : t.m_cpus)
				c.cache = 0;
			t.m_cacheSource = "sysconf";
		}
	}
#endif
	return t;
}

CpuTopology CpuTopology::Fallback()
{
	// Nothing known but the count: every CPU a core of its own, one kind, and
	// no way to keep a thread anywhere.
	std::vector<Raw> raw(std::max(1u, std::thread::hardware_concurrency()));
	for (size_t i = 0; i < raw.size(); ++i)
		raw[i].os = (unsigned int)i;
	return Build(raw, NONE, "hardware_concurrency");
}

CpuTopology CpuTopology::Build(const std::vector<Raw>& raw, Placement placement, const std::string& source)
{
	CpuTopology t;
	t.m_placement = placement;
	t.m_source = source;

	std::set<int> kinds;
	for (const Raw& r : raw)
		kinds.insert(std::max(r.kind, 0));
	std::map<int, unsigned int> kindRank;
	for (int k : kinds)
		kindRank.emplace(k, (unsigned int)kindRank.size());
	t.m_numKinds = std::max<unsigned int>(1, (unsigned int)kindRank.size());

	std::map<std::string, unsigned int> cores, caches;
	for (const Raw& r : raw)
	{
		Cpu c;
		c.os = r.os;
		const std::string coreKey = r.core.empty() ? "#" + std::to_string(r.os) : r.core;
		c.core = cores.emplace(coreKey, (unsigned int)cores.size()).first->second;
		c.kind = kindRank[std::max(r.kind, 0)];
		c.cache = -1;
		if (!r.cache.empty() && r.cacheBytes > 0)
		{
			const auto it = caches.emplace(r.cache, (unsigned int)caches.size());
			if (it.second)
				t.m_caches.push_back({r.cacheLevel, r.cacheBytes});
			c.cache = (int)it.first->second;
		}
		t.m_cpus.push_back(c);
	}
	t.m_numCores = (unsigned int)cores.size();
	if (!t.m_caches.empty())
		t.m_cacheSource = source;
	return t;
}

CpuTopology CpuTopology::FromSysfs(const std::string& root, const std::vector<unsigned int>& allowed)
{
#if defined(__linux__)
	const std::string cpuDir = root + "/devices/system/cpu/cpu";
	std::vector<Raw> raw;
	std::vector<double> capacity, clock;

	// Intel hybrids have a PMU per kind of core.
	std::vector<unsigned int> pList, eList;
	const bool intelHybrid = ReadCpuList(root + "/devices/cpu_core/cpus", pList) &&
	                         ReadCpuList(root + "/devices/cpu_atom/cpus", eList);
	const std::set<unsigned int> eCores(eList.begin(), eList.end());

	for (unsigned int n : allowed)
	{
		const std::string dir = cpuDir + std::to_string(n);
		Raw r;
		r.os = n;
		std::string text;
		if (ReadText(dir + "/topology/core_cpus_list", text) || ReadText(dir + "/topology/thread_siblings_list", text))
			r.core = text;
		if (intelHybrid)
			r.kind = eCores.count(n) ? 1 : 0;
		capacity.push_back(ReadNumber(dir + "/cpu_capacity"));
		clock.push_back(ReadNumber(dir + "/cpufreq/cpuinfo_max_freq"));

		// The highest level of data or unified cache is the last level: an L3
		// on x86, often the L2 on ARM.
		for (int idx = 0; idx < 16; ++idx)
		{
			const std::string c = dir + "/cache/index" + std::to_string(idx);
			std::string level, type, size, shared;
			if (!ReadText(c + "/level", level))
				break;
			ReadText(c + "/type", type);
			if (type == "Instruction")
				continue;
			const unsigned int lv = (unsigned int)std::strtoul(level.c_str(), nullptr, 10);
			if (lv < r.cacheLevel || !ReadText(c + "/size", size) || !ReadText(c + "/shared_cpu_list", shared))
				continue;
			r.cacheLevel = lv;
			r.cacheBytes = ParseSize(size);
			r.cache = "L" + level + ":" + shared;
		}
		raw.push_back(r);
	}

	if (!intelHybrid)
	{
		std::vector<int> rank = RankBySpeed(capacity);
		if (rank.empty())
			rank = RankBySpeed(clock);
		for (size_t i = 0; i < rank.size(); ++i)
			raw[i].kind = rank[i];
	}
	return Build(raw, AFFINITY, "sysfs");
#else
	(void)root;
	(void)allowed;
	return CpuTopology();
#endif
}

#ifdef _WIN32
CpuTopology CpuTopology::DetectWindows()
{
	DWORD len = 0;
	GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
	if (len == 0)
		return CpuTopology();
	std::vector<unsigned char> buf(len);
	if (!GetLogicalProcessorInformationEx(RelationAll,
	        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len))
		return CpuTopology();

	const unsigned int bits = 8 * sizeof(KAFFINITY);
	struct CacheInfo
	{
		unsigned int level;
		unsigned long long bytes;
		WORD group;
		KAFFINITY mask;
	};
	std::vector<Raw> raw;
	std::vector<CacheInfo> caches;
	unsigned int coreNo = 0;
	for (DWORD off = 0; off < len;)
	{
		const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
		if (info->Size == 0)
			break;
		if (info->Relationship == RelationProcessorCore)
		{
			// EfficiencyClass: higher is faster, the same on every core of a CPU
			// with one kind. It is the byte after Flags; older MinGW headers
			// still call it Reserved[0], so it is read by its place.
			const PROCESSOR_RELATIONSHIP& p = info->Processor;
			const BYTE efficiencyClass = reinterpret_cast<const BYTE*>(&p)[1];
			for (WORD g = 0; g < p.GroupCount; ++g)
				for (unsigned int b = 0; b < bits; ++b)
					if (p.GroupMask[g].Mask & ((KAFFINITY)1 << b))
					{
						Raw r;
						r.os = p.GroupMask[g].Group * bits + b;
						r.core = std::to_string(coreNo);
						r.kind = 255 - efficiencyClass;
						raw.push_back(r);
					}
			++coreNo;
		}
		else if (info->Relationship == RelationCache)
		{
			const CACHE_RELATIONSHIP& c = info->Cache;
			if (c.Type == CacheUnified || c.Type == CacheData)
				caches.push_back({c.Level, c.CacheSize, c.GroupMask.Group, c.GroupMask.Mask});
		}
		off += info->Size;
	}

	unsigned int lastLevel = 0;
	for (const CacheInfo& c : caches)
		lastLevel = std::max(lastLevel, c.level);
	for (size_t i = 0; i < caches.size(); ++i)
	{
		if (caches[i].level != lastLevel)
			continue;
		for (Raw& r : raw)
			if (r.os / bits == caches[i].group && (caches[i].mask & ((KAFFINITY)1 << (r.os % bits))))
			{
				r.cache = std::to_string(i);
				r.cacheLevel = caches[i].level;
				r.cacheBytes = caches[i].bytes;
			}
	}

	// The process affinity, where it is one mask (one processor group).
	DWORD_PTR processMask = 0, systemMask = 0;
	if (GetActiveProcessorGroupCount() == 1 &&
	    GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask) && processMask)
		raw.erase(std::remove_if(raw.begin(), raw.end(), [&](const Raw& r)
		{
			return !(processMask & ((DWORD_PTR)1 << (r.os % bits)));
		}), raw.end());

	std::sort(raw.begin(), raw.end(), [](const Raw& a, const Raw& b) {return a.os < b.os;});
	return Build(raw, AFFINITY, "Windows");
}
#endif

#ifdef __APPLE__
CpuTopology CpuTopology::DetectMacOS()
{
	const auto get = [](const std::string& name, long long& value) -> bool
	{
		unsigned char buf[8] = {0};
		size_t size = sizeof(buf);
		if (sysctlbyname(name.c_str(), buf, &size, nullptr, 0) != 0)
			return false;
		if (size == 4)
		{
			int32_t v;
			std::memcpy(&v, buf, 4);
			value = v;
		}
		else if (size == 8)
		{
			int64_t v;
			std::memcpy(&v, buf, 8);
			value = v;
		}
		else
			return false;
		return true;
	};

	// Apple CPUs describe each kind of core as a performance level, 0 the
	// fastest; Intel Macs have none, and one kind.
	long long levels = 0;
	if (!get("hw.nperflevels", levels) || levels < 1)
		levels = 0;
	std::vector<Raw> raw;
	unsigned int os = 0;
	for (long long l = 0; l < std::max(levels, 1ll); ++l)
	{
		const std::string p = levels ? "hw.perflevel" + std::to_string(l) + "." : "hw.";
		long long logical = 0, physical = 0, l2 = 0, perL2 = 0, l3 = 0, perL3 = 0;
		get(p + "logicalcpu", logical);
		get(p + "physicalcpu", physical);
		if (logical <= 0)
			continue;
		if (physical <= 0)
			physical = logical;
		get(p + "l2cachesize", l2);
		get(p + "cpusperl2", perL2);
		get(p + "l3cachesize", l3);
		get(p + "cpusperl3", perL3);
		const long long perCore = std::max(1ll, logical / physical);
		for (long long i = 0; i < logical; ++i)
		{
			Raw r;
			r.os = os++;
			r.core = std::to_string(l) + ":" + std::to_string(i / perCore);
			r.kind = levels ? (int)l : -1;
			if (l3 > 0)
			{
				r.cacheLevel = 3;
				r.cacheBytes = (unsigned long long)l3;
				r.cache = "L3:" + std::to_string(l) + ":" + std::to_string(perL3 > 0 ? i / perL3 : 0);
			}
			else if (l2 > 0)
			{
				r.cacheLevel = 2;
				r.cacheBytes = (unsigned long long)l2;
				r.cache = "L2:" + std::to_string(l) + ":" + std::to_string(perL2 > 0 ? i / perL2 : 0);
			}
			raw.push_back(r);
		}
	}
	return Build(raw, levels >= 2 ? QOS : NONE, "sysctl");
}
#endif

// ----------------------------------------------------------------------------
// Sets of cores
// ----------------------------------------------------------------------------

std::vector<unsigned int> CpuTopology::Select(const CoreSet& set) const
{
	std::vector<unsigned int> out;
	std::set<unsigned int> coresTaken;
	for (size_t i = 0; i < m_cpus.size(); ++i)
	{
		const Cpu& c = m_cpus[i];
		if (set.kinds == CoreSet::PERFORMANCE && c.kind != 0)
			continue;
		if (set.kinds == CoreSet::EFFICIENCY && c.kind != m_numKinds - 1)
			continue;
		if (!set.smt && !coresTaken.insert(c.core).second)
			continue;
		out.push_back((unsigned int)i);
	}
	return out;
}

unsigned int CpuTopology::MaxThreads(const CoreSet& set) const
{
	return (unsigned int)Select(set).size();
}

bool CpuTopology::Offers(const CoreSet& set) const
{
	if (set.IsAll())
		return true;
	switch (m_placement)
	{
	case AFFINITY:
		return !Select(set).empty();
	case QOS:
	{
		// A QoS class picks a kind of core, not single CPUs: one thread per
		// core is only offered where that is all there is anyway.
		CoreSet all = set;
		all.smt = true;
		return set.kinds != CoreSet::ALL_KINDS && m_numKinds > 1 && Select(set).size() == Select(all).size();
	}
	case NONE:
		break;
	}
	return false;
}

std::vector<CoreSet> CpuTopology::DistinctSets() const
{
	std::vector<CoreSet> order;
	const auto add = [&order](CoreSet::Kinds kinds, bool smt)
	{
		CoreSet s;
		s.kinds = kinds;
		s.smt = smt;
		order.push_back(s);
	};
	if (m_numKinds > 1)
	{
		add(CoreSet::PERFORMANCE, false);
		add(CoreSet::PERFORMANCE, true);
		add(CoreSet::EFFICIENCY, false);
		add(CoreSet::EFFICIENCY, true);
	}
	add(CoreSet::ALL_KINDS, false);
	add(CoreSet::ALL_KINDS, true);

	std::vector<CoreSet> out;
	std::vector<std::vector<unsigned int>> taken;
	for (const CoreSet& s : order)
	{
		if (!Offers(s))
			continue;
		const std::vector<unsigned int> cpus = Select(s);
		const auto same = std::find(taken.begin(), taken.end(), cpus);
		if (same != taken.end())
		{
			// "all" names it best: without SMT, "physical" is every CPU too.
			if (s.IsAll())
				out[same - taken.begin()] = s;
			continue;
		}
		taken.push_back(cpus);
		out.push_back(s);
	}
	return out;
}

LLC_Span CpuTopology::LastLevelCaches(const CoreSet& set, unsigned int maxThreads) const
{
	LLC_Span span = {0, 0, 0, 0, m_cacheSource.empty() ? "unknown" : m_cacheSource.c_str()};
	std::vector<int> seen;
	for (unsigned int i : Select(set))
	{
		const int c = m_cpus[i].cache;
		if (c >= 0 && std::find(seen.begin(), seen.end(), c) == seen.end())
			seen.push_back(c);
	}
	if (seen.size() > std::max(1u, maxThreads))
		seen.resize(std::max(1u, maxThreads));
	for (int c : seen)
	{
		span.level = std::max(span.level, m_caches[c].level);
		span.instanceBytes = std::max(span.instanceBytes, m_caches[c].bytes);
		span.totalBytes += m_caches[c].bytes;
	}
	span.instances = (unsigned int)seen.size();
	return span;
}

bool CpuTopology::Restrict(const CoreSet& set, unsigned int index, unsigned int count) const
{
	(void)index;
	(void)count;
	if (!Offers(set))
		return false;
#if defined(__linux__)
	if (m_placement != AFFINITY)
		return set.IsAll();
	const std::vector<unsigned int> sel = Select(set);
	unsigned int maxOs = 0;
	for (unsigned int i : sel)
		maxOs = std::max(maxOs, m_cpus[i].os);
	cpu_set_t* mask = CPU_ALLOC(maxOs + 1);
	if (!mask)
		return false;
	const size_t size = CPU_ALLOC_SIZE(maxOs + 1);
	CPU_ZERO_S(size, mask);
	for (unsigned int i : sel)
		CPU_SET_S(m_cpus[i].os, size, mask);
	const int rc = pthread_setaffinity_np(pthread_self(), size, mask);
	CPU_FREE(mask);
	return rc == 0;
#elif defined(_WIN32)
	if (m_placement != AFFINITY)
		return set.IsAll();
	// A thread's affinity lies within one processor group; with more than one
	// (over 64 CPUs) the workers are spread over them by their share of the set.
	const unsigned int bits = 8 * sizeof(KAFFINITY);
	std::map<WORD, KAFFINITY> groups;
	const std::vector<unsigned int> sel = Select(set);
	for (unsigned int i : sel)
		groups[(WORD)(m_cpus[i].os / bits)] |= (KAFFINITY)1 << (m_cpus[i].os % bits);
	const size_t target = count ? ((2 * (size_t)index + 1) * sel.size()) / (2 * (size_t)count) : 0;
	size_t passed = 0;
	auto g = groups.begin();
	for (; g != groups.end(); ++g)
	{
		size_t n = 0;
		for (unsigned int b = 0; b < bits; ++b)
			n += (g->second >> b) & 1;
		if (target < passed + n)
			break;
		passed += n;
	}
	if (g == groups.end())
		g = groups.begin();
	GROUP_AFFINITY ga;
	std::memset(&ga, 0, sizeof(ga));
	ga.Group = g->first;
	ga.Mask = g->second;
	return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) != 0;
#elif defined(__APPLE__)
	if (set.IsAll())
		return true;
	// Background QoS keeps a thread on the efficiency cores of an Apple CPU;
	// user-interactive prefers the performance cores, without a guarantee.
	const qos_class_t qos = (set.kinds == CoreSet::PERFORMANCE) ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_BACKGROUND;
	return pthread_set_qos_class_self_np(qos, 0) == 0;
#else
	return set.IsAll();
#endif
}

// ----------------------------------------------------------------------------
// Descriptions
// ----------------------------------------------------------------------------

std::string CpuTopology::KindName(unsigned int kind) const
{
	if (m_numKinds <= 1)
		return "";
	if (kind == 0)
		return "performance";
	if (kind + 1 == m_numKinds)
		return "efficiency";
	return "mid-" + std::to_string(kind);
}

std::string CpuTopology::DescribeSet(const CoreSet& set) const
{
	const std::vector<unsigned int> sel = Select(set);
	std::set<unsigned int> cores;
	for (unsigned int i : sel)
		cores.insert(m_cpus[i].core);
	std::ostringstream os;
	os << cores.size();
	if (set.kinds != CoreSet::ALL_KINDS && m_numKinds > 1)
		os << " " << KindName(set.kinds == CoreSet::PERFORMANCE ? 0 : m_numKinds - 1);
	os << (cores.size() == 1 ? " core" : " cores");
	if (sel.size() != cores.size())
		os << ", " << sel.size() << " threads";
	return os.str();
}

std::string CpuTopology::Describe() const
{
	std::ostringstream os;
	for (unsigned int k = 0; k < m_numKinds; ++k)
	{
		std::set<unsigned int> cores;
		unsigned int threads = 0;
		for (const Cpu& c : m_cpus)
			if (c.kind == k)
			{
				cores.insert(c.core);
				++threads;
			}
		if (k > 0)
			os << " + ";
		os << cores.size() << (m_numKinds > 1 ? " " + KindName(k) : std::string())
		   << (cores.size() == 1 ? " core" : " cores");
		if (threads != cores.size())
			os << " (" << threads << " threads)";
	}
	os << "; ";

	// Instances by size: "L3 2 x 32 MB", or "L3 1 x 96 MB + 1 x 32 MB" on a mixed one.
	std::map<unsigned long long, unsigned int, std::greater<unsigned long long>> bySize;
	unsigned int level = 0;
	for (const Cache& c : m_caches)
	{
		++bySize[c.bytes];
		level = std::max(level, c.level);
	}
	if (bySize.empty())
		os << "last-level cache unknown";
	else
	{
		os << "L" << level << " ";
		bool first = true;
		for (const auto& s : bySize)
		{
			os << (first ? "" : " + ") << s.second << " x " << FormatBytes(s.first);
			first = false;
		}
	}
	os << " (" << m_source;
	if (!m_cacheSource.empty() && m_cacheSource != m_source)
		os << ", cache from " << m_cacheSource;
	os << ")";
	if (m_placement == NONE && (m_numKinds > 1 || m_numCores < m_cpus.size()))
		os << "; threads cannot be kept to cores here";
	return os.str();
}

LLC_Span QueryLastLevelCaches(unsigned int maxThreads)
{
	return CpuTopology::Get().LastLevelCaches(CoreSet(), maxThreads);
}
