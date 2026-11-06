#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <mutex>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		static thread_local const uint32_t tid =
		    static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
		return tid;
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t offset, uint64_t size) {
		const auto [start, end] = GetPageRange(offset, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.FirstRangeFrom(start).first < end) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
			NoteCpuWrite(start, end);
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			// GPU-owned pages must fault on CPU writes again.
			if (m_sticky.FirstRangeFrom(start).first < end) {
				m_sticky.UnsetRange(start, end);
				m_cpu_dirty.UnsetRange(start, end);
				UpdateProtection<true, false>();
			}
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t offset, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(offset, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				UpdateProtection<true, false>();
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// Pages the guest CPU rewrites again and again stay unprotected ("sticky"): instead of a
	// write fault and two protection changes per write, they are marked dirty again by every
	// RefreshSticky (once per guest command buffer) and uploaded when next used. Called with the
	// lock held; returns whether the region still has sticky pages.
	// on_dirty(address, size) receives each sticky range now marked CPU-modified, so callers can
	// queue the work an ordinary write fault would.
	template <typename Func>
	bool RefreshSticky(uint32_t epoch, Func&& on_dirty) {
		if (m_sticky.None()) {
			return false;
		}
		// Writes since the last refresh did not fault: mark the pages modified first.
		auto dirty = m_sticky;
		dirty &= ~m_gpu_dirty;
		m_cpu_dirty |= dirty;
		for (const auto [first, last]: dirty) {
			on_dirty(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
		if (epoch - m_sticky_epoch >= StickyLifetime) {
			// Track the pages again once uploaded; still-hot ones turn sticky after two writes.
			m_sticky.Clear();
			return false;
		}
		return true;
	}

	[[nodiscard]] bool TakeNewSticky() noexcept { return std::exchange(m_has_new_sticky, false); }

	static bool StickyEnabled() {
		// Off by default (KYTY_STICKY_PAGES=1 enables): keeping sticky pages current for BDA
		// readers re-uploads them after every guest command buffer, which cost Tekken 8 more
		// GPU-thread time (~6 us per draw) than the write faults it saves.
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_STICKY_PAGES");
			return value != nullptr && value[0] == '1';
		}();
		return enabled;
	}

	static std::atomic<uint32_t>& StickyEpoch() {
		static std::atomic<uint32_t> epoch {0};
		return epoch;
	}

	TrackingSpinLock lock;

private:
	static constexpr uint32_t StickyFaultWindow = 64;
	static constexpr uint32_t StickyLifetime    = 4096;
	static constexpr size_t   StickyMaxPages    = 16;

	// A guest CPU write of a few clean pages: a page written twice within the window turns
	// sticky.
	void NoteCpuWrite(size_t start, size_t end) {
		if (!StickyEnabled() || end - start > StickyMaxPages) {
			return;
		}
		const auto epoch = StickyEpoch().load(std::memory_order_relaxed);
		if (epoch - m_faulted_epoch >= StickyFaultWindow) {
			m_faulted.Clear();
			m_faulted_epoch = epoch;
		}
		for (size_t page = start; page < end; page++) {
			if (m_cpu_dirty.Get(page) || m_sticky.Get(page)) {
				continue;
			}
			if (!m_faulted.Get(page)) {
				m_faulted.Set(page);
				continue;
			}
			if (m_sticky.None()) {
				m_sticky_epoch = epoch;
			}
			m_sticky.Set(page);
			m_has_new_sticky = true;
		}
	}

	template <bool track, bool is_read>
	void UpdateProtection() {
		auto writable = m_cpu_dirty;
		writable |= m_sticky;
		const auto protection = is_read ? ~m_gpu_dirty : writable;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] static std::pair<size_t, size_t> GetPageRange(uint64_t offset, uint64_t size) {
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	RegionBits   m_writable;
	RegionBits   m_readable;
	RegionBits   m_faulted;
	RegionBits   m_sticky;
	uint32_t     m_faulted_epoch  = 0;
	uint32_t     m_sticky_epoch   = 0;
	bool         m_has_new_sticky = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
