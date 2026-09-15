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

#include "cache_topology.h"

#include <algorithm>
#include <unistd.h>

#ifdef OPENEMS_HAVE_HWLOC
#include <hwloc.h>

// Works against hwloc 1.11 as well as 2.x: hwloc_get_cache_type_depth() is in
// both, unlike HWLOC_OBJ_L3CACHE, which only 2.x has. The portable benchmark
// bundle builds in a CentOS 7 image whose system hwloc is 1.11.
static LLC_Span QueryHwloc(unsigned int maxThreads)
{
	LLC_Span span = { 0, 0, 0, "hwloc" };
	hwloc_topology_t topo;
	if (hwloc_topology_init(&topo) != 0)
		return span;
	if (hwloc_topology_load(topo) != 0)
	{
		hwloc_topology_destroy(topo);
		return span;
	}

	const int depth = hwloc_get_cache_type_depth(topo, 3, HWLOC_OBJ_CACHE_UNIFIED);
	hwloc_bitmap_t allowed = hwloc_bitmap_alloc();
	if (hwloc_get_cpubind(topo, allowed, HWLOC_CPUBIND_PROCESS) != 0)
		hwloc_bitmap_copy(allowed, hwloc_topology_get_topology_cpuset(topo));

	// Read the sharing sets rather than assuming CPU numbers are contiguous per
	// cache: some dual-socket BIOSes number the sockets alternately (even CPUs
	// on one, odd on the other), so "-c 0-35" spans both.
	unsigned int count = 0;
	if (depth >= 0)
	{
		const unsigned int n = hwloc_get_nbobjs_by_depth(topo, depth);
		for (unsigned int i = 0; i < n; ++i)
		{
			hwloc_obj_t c = hwloc_get_obj_by_depth(topo, depth, i);
			if (!hwloc_bitmap_intersects(c->cpuset, allowed))
				continue;
			span.instanceBytes = std::max(span.instanceBytes, (unsigned long long)c->attr->cache.size);
			++count;
		}
	}

	hwloc_bitmap_free(allowed);
	hwloc_topology_destroy(topo);

	span.instances = std::min(count, std::max(maxThreads, 1u));
	span.totalBytes = span.instances * span.instanceBytes;
	return span;
}
#endif

LLC_Span QueryLastLevelCaches(unsigned int maxThreads)
{
#ifdef OPENEMS_HAVE_HWLOC
	const LLC_Span hw = QueryHwloc(maxThreads);
	if (hw.totalBytes > 0)
		return hw;
#else
	(void)maxThreads;
#endif

	LLC_Span span = { 0, 0, 0, "sysconf" };
#ifdef _SC_LEVEL3_CACHE_SIZE
	const long bytes = sysconf(_SC_LEVEL3_CACHE_SIZE);
	if (bytes > 0)
	{
		span.instances = 1;
		span.instanceBytes = span.totalBytes = (unsigned long long)bytes;
	}
#endif
	return span;
}
