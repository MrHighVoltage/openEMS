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

#ifndef CACHE_TOPOLOGY_H
#define CACHE_TOPOLOGY_H

//! The L3 capacity a run can actually use.
struct LLC_Span
{
	unsigned int instances;           //!< L3 instances counted; 0 if unknown
	unsigned long long instanceBytes; //!< size of one instance
	unsigned long long totalBytes;    //!< instances * instanceBytes
	const char* source;               //!< "hwloc" or "sysconf"
};

//! Sum the L3 instances the calling process may run on.
/*!
  sysconf(_SC_LEVEL3_CACHE_SIZE) reports one instance, which is the whole
  last-level cache only when that cache is undivided. On a multi-socket server
  or a multi-CCX Ryzen the threads spread over several, and each holds its own
  share of the working set. With hwloc (WITH_HWLOC) this counts the instances
  that intersect the process's CPU affinity, capped at \a maxThreads -- a run
  cannot occupy more caches than it has threads. Without hwloc it falls back to
  one sysconf instance, i.e. the old behaviour.
*/
LLC_Span QueryLastLevelCaches(unsigned int maxThreads);

#endif // CACHE_TOPOLOGY_H
