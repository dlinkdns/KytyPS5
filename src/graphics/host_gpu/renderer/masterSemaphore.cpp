#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	m_graphics.device.destroySemaphore(m_semaphore, nullptr);
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	static const bool trace = [] {
		const char* value = std::getenv("KYTY_TRACE_WAITS");
		return value != nullptr && value[0] == '1';
	}();
	const auto started = std::chrono::steady_clock::now();
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
	if (trace) {
		struct Stat {
			double   ms;
			uint64_t count;
		};
		static std::mutex                            mutex;
		static std::unordered_map<std::string, Stat> stats;
		static auto                                  window = std::chrono::steady_clock::now();
		const auto now = std::chrono::steady_clock::now();
		const char* tag = GpuWaitTag::Current();
		std::scoped_lock lock {mutex};
		auto& stat = stats[tag != nullptr ? tag : "untagged"];
		stat.ms += std::chrono::duration<double, std::milli>(now - started).count();
		stat.count++;
		if (now - window >= std::chrono::seconds(2)) {
			const double seconds = std::chrono::duration<double>(now - window).count();
			std::vector<std::pair<std::string, Stat>> sorted(stats.begin(), stats.end());
			std::sort(sorted.begin(), sorted.end(),
			          [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
			std::printf("--- GPU waits over %.1f s ---\n", seconds);
			for (const auto& [name, s]: sorted) {
				std::printf("  %7.1f ms/s  %6.1f waits/s  %s\n", s.ms / seconds, s.count / seconds,
				            name.c_str());
			}
			std::fflush(stdout);
			stats.clear();
			window = now;
		}
	}
}

} // namespace Libs::Graphics
