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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#endif

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

/**
 * @brief Barrier for many short rounds: optionally spins before it sleeps,
 * and can report when each round was released.
 *
 * Barrier above takes a mutex on every arrival and parks every early thread
 * in the kernel; the last one to arrive wakes them all, and they then leave
 * one by one through the same mutex. That costs a few microseconds per round,
 * which is nothing once per timestep and a lot on the temporally blocked
 * schedule, which has tens of rounds per timestep spaced tens of
 * microseconds apart.
 *
 * Here arrival is one atomic decrement. An early thread first polls the
 * generation for up to @p spinMicros (0: not at all), and only then sleeps on
 * the condition variable; the last thread takes the mutex only if somebody is
 * actually asleep. With @p spinMicros = 0 it behaves like Barrier.
 *
 * waitTimed() is wait() plus the steady_clock time (ns) at which the round was
 * released, so a caller can split its time at the barrier into waiting for the
 * slowest thread (arrival until release) and getting going again (release
 * until return).
 */
class HybridBarrier
{
public:
	explicit HybridBarrier(unsigned int count, unsigned int spinMicros = 0)
		: m_threshold(count), m_count(count), m_generation(0), m_sleepers(0),
		  m_release(0), m_spinNs((int64_t)spinMicros * 1000) {}

	void wait() {arrive(false);}
	int64_t waitTimed() {return arrive(true);}

	static int64_t now()
	{
		return std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
	}

private:
	int64_t arrive(bool timed)
	{
		const unsigned int gen = m_generation.load(std::memory_order_acquire);
		if (m_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
		{
			// Last to arrive. The reset is ordered before the release below, so
			// whoever sees the new generation also sees the full count.
			m_count.store(m_threshold, std::memory_order_relaxed);
			const int64_t t = timed ? now() : 0;
			m_release.store(t, std::memory_order_relaxed);
			m_generation.fetch_add(1, std::memory_order_seq_cst);
			// Pairs with the sleeper's increment-then-check below: either it sees
			// the new generation and never sleeps, or this sees it and wakes it.
			if (m_sleepers.load(std::memory_order_seq_cst) > 0)
			{
				std::lock_guard<std::mutex> lock(m_mutex);
				m_cv.notify_all();
			}
			return t;
		}

		if (m_spinNs > 0)
		{
			const int64_t deadline = now() + m_spinNs;
			for (unsigned int n = 1; m_generation.load(std::memory_order_acquire) == gen; ++n)
			{
				pause();
				if ((n & 63) == 0 && now() > deadline)
					break;
			}
		}
		if (m_generation.load(std::memory_order_acquire) == gen)
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_sleepers.fetch_add(1, std::memory_order_seq_cst);
			while (m_generation.load(std::memory_order_seq_cst) == gen)
				m_cv.wait(lock);
			m_sleepers.fetch_sub(1, std::memory_order_relaxed);
		}
		// The next round cannot be released before this thread arrives at it,
		// so the stamp read here is still this round's.
		return m_release.load(std::memory_order_relaxed);
	}

	static void pause()
	{
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
		_mm_pause();
#else
		std::this_thread::yield();
#endif
	}

	const unsigned int        m_threshold;
	std::atomic<unsigned int> m_count;
	std::atomic<unsigned int> m_generation;
	std::atomic<unsigned int> m_sleepers;
	std::atomic<int64_t>      m_release;
	const int64_t             m_spinNs;
	std::mutex                m_mutex;
	std::condition_variable   m_cv;
};

#endif // OPENEMS_BARRIER_H
