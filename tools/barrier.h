/*
 * Copyright (C) 2025 Georg Wieser
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OPENEMS_BARRIER_H
#define OPENEMS_BARRIER_H

#include <mutex>
#include <condition_variable>
#include <cstddef>

/**
 * @brief Reusable thread barrier (drop-in replacement for boost::barrier).
 *
 * All @p count threads must call wait() before any of them can proceed.
 * The barrier automatically resets after each generation, so it can be
 * reused for repeated synchronisation rounds.
 */
class Barrier
{
public:
	explicit Barrier(unsigned int count)
		: m_threshold(count), m_count(count), m_generation(0) {}

	/// Block until all participants have called wait().
	void wait()
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		unsigned int gen = m_generation;
		if (--m_count == 0)
		{
			// Last thread to arrive: reset and wake everyone
			m_count = m_threshold;
			++m_generation;
			m_cv.notify_all();
		}
		else
		{
			m_cv.wait(lock, [this, gen] { return gen != m_generation; });
		}
	}

private:
	std::mutex              m_mutex;
	std::condition_variable m_cv;
	unsigned int            m_threshold;
	unsigned int            m_count;
	unsigned int            m_generation;
};

#endif // OPENEMS_BARRIER_H
