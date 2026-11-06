#pragma once

// Diagnostics for GPU hangs: with KYTY_GPU_CHECKPOINTS=1 on a device that supports
// VK_NV_device_diagnostic_checkpoints, every dispatch and draw records a checkpoint, and a lost
// device reports which ones the GPU had reached and in which pipeline stage.

#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::GpuCheckpoints {

struct Entry {
	uint32_t kind; // 1 = dispatch, 2 = indirect dispatch, 3 = draw
	uint64_t hash;
	uint64_t hash2;
	uint32_t x, y, z;
};

inline constexpr size_t RingSize = 1u << 16u;

inline bool Requested() {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_GPU_CHECKPOINTS");
		return value != nullptr && value[0] == '1';
	}();
	return requested;
}

inline std::atomic_bool& Enabled() {
	static std::atomic_bool enabled {false};
	return enabled;
}

inline std::array<Entry, RingSize>& Ring() {
	static std::array<Entry, RingSize> ring {};
	return ring;
}

inline std::atomic_uint64_t& Next() {
	static std::atomic_uint64_t next {1};
	return next;
}

// KYTY_GPU_PROFILE=1: timestamps around dispatches, draws and tiler jobs; every ~2 s the GPU is
// drained and the most expensive shaders of that window are printed.
namespace Profile {
struct Pending {
	uint64_t hash;
	uint32_t kind;
	uint32_t query;
	bool     ended;
};
struct Total {
	double   ms;
	uint64_t count;
};
struct State {
	vk::Device                             device;
	vk::QueryPool                          pool;
	double                                 period_ns = 1.0;
	uint32_t                               next      = 0;
	std::vector<Pending>                   pending;
	std::vector<size_t>                    open;
	std::unordered_map<uint64_t, Total>    totals;
	std::chrono::steady_clock::time_point  last = std::chrono::steady_clock::now();
	std::mutex                             mutex;
};
inline constexpr uint32_t Capacity = 1u << 17u;

inline bool Requested() {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_GPU_PROFILE");
		return value != nullptr && value[0] == '1';
	}();
	return requested;
}

inline State& Get() {
	static State state;
	return state;
}

inline void Init(vk::Device device, float timestamp_period) {
	if (!Requested()) {
		return;
	}
	auto& s     = Get();
	s.device    = device;
	s.period_ns = timestamp_period;
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eTimestamp;
	info.queryCount = Capacity;
	if (device.createQueryPool(&info, nullptr, &s.pool) != vk::Result::eSuccess) {
		s.pool = nullptr;
		std::printf("GPU profiler: could not create the timestamp query pool\n");
		return;
	}
	device.resetQueryPool(s.pool, 0, Capacity);
	std::printf("GPU profiler enabled (timestamp period %.2f ns)\n", timestamp_period);
}

inline bool IsBegin(uint32_t kind) { return kind == 1 || kind == 2 || kind == 3 || kind == 15; }
inline bool IsEnd(uint32_t kind) { return kind == 6 || kind == 7 || kind == 8 || kind == 16; }

inline void Record(vk::CommandBuffer buffer, uint32_t kind, uint64_t hash) {
	auto& s = Get();
	if (!s.pool || !buffer) {
		return;
	}
	std::scoped_lock lock {s.mutex};
	if (IsBegin(kind)) {
		if (s.next + 2 > Capacity) {
			return;
		}
		buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, s.pool, s.next);
		s.open.push_back(s.pending.size());
		s.pending.push_back({kind == 15 ? (0x7111e000000000ull | hash) : hash, kind, s.next, false});
		s.next += 2;
	} else if (IsEnd(kind) && !s.open.empty()) {
		auto& entry = s.pending[s.open.back()];
		s.open.pop_back();
		buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, s.pool, entry.query + 1);
		entry.ended = true;
	}
}

// Call right after a queue submit: everything recorded so far has been submitted.
inline void Flush(vk::Queue queue) {
	auto& s = Get();
	if (!s.pool) {
		return;
	}
	std::scoped_lock lock {s.mutex};
	const auto now = std::chrono::steady_clock::now();
	if (now - s.last < std::chrono::seconds(2) && s.next + 4096 < Capacity) {
		return;
	}
	queue.waitIdle();
	std::vector<uint64_t> stamps(s.next);
	if (s.next != 0) {
		(void)s.device.getQueryPoolResults(s.pool, 0, s.next, stamps.size() * sizeof(uint64_t),
		                                   stamps.data(), sizeof(uint64_t),
		                                   vk::QueryResultFlagBits::e64);
	}
	double window_ms = 0.0;
	for (const auto& entry: s.pending) {
		if (!entry.ended || stamps[entry.query + 1] < stamps[entry.query]) {
			continue;
		}
		const double ms = static_cast<double>(stamps[entry.query + 1] - stamps[entry.query]) *
		                  s.period_ns / 1e6;
		auto& total = s.totals[(uint64_t {entry.kind == 3 ? 1u : 0u} << 63u) ^ entry.hash];
		total.ms += ms;
		total.count++;
		window_ms += ms;
	}
	const double seconds = std::chrono::duration<double>(now - s.last).count();
	std::vector<std::pair<uint64_t, Total>> top(s.totals.begin(), s.totals.end());
	std::sort(top.begin(), top.end(),
	          [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
	std::printf("--- GPU profile: %.1f s window, %.1f ms GPU per second in %zu jobs ---\n", seconds,
	            window_ms / seconds, s.pending.size());
	for (size_t i = 0; i < top.size() && i < 20; ++i) {
		const auto key = top[i].first;
		std::printf("  %7.2f ms/s  %6llu calls  %s 0x%016llx\n", top[i].second.ms / seconds,
		            static_cast<unsigned long long>(top[i].second.count),
		            (key >> 63u) ? "draw(vs)" : ((key >> 40u) == 0x7111e0u ? "tiler" : "dispatch"),
		            static_cast<unsigned long long>(key & 0x7fffffffffffffffull));
	}
	std::fflush(stdout);
	s.device.resetQueryPool(s.pool, 0, Capacity);
	s.pending.clear();
	s.open.clear();
	s.totals.clear();
	s.next = 0;
	s.last = std::chrono::steady_clock::now();
}
} // namespace Profile

inline void Mark(vk::CommandBuffer buffer, uint32_t kind, uint64_t hash, uint64_t hash2 = 0,
                 uint32_t x = 0, uint32_t y = 0, uint32_t z = 0) {
	if (Profile::Requested()) {
		Profile::Record(buffer, kind, hash);
	}
	if (!Enabled().load(std::memory_order_relaxed) || !buffer) {
		return;
	}
	const uint64_t id = Next().fetch_add(1, std::memory_order_relaxed);
	Ring()[id % RingSize] = Entry {kind, hash, hash2, x, y, z};
	buffer.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(id)));
}

// KYTY_SKIP_SHADER=<hex hash>[,<hex hash>...] drops dispatches and draws of those shaders.
inline bool SkipShader(uint64_t hash) {
	static const auto hashes = [] {
		std::array<uint64_t, 16> list {};
		const char*              text = std::getenv("KYTY_SKIP_SHADER");
		for (size_t i = 0; text != nullptr && *text != '\0' && i < list.size(); ++i) {
			char* end = nullptr;
			list[i]   = std::strtoull(text, &end, 16);
			text      = (end != nullptr && *end == ',') ? end + 1 : nullptr;
		}
		return list;
	}();
	if (hash == 0) {
		return false;
	}
	for (const auto skipped: hashes) {
		if (skipped == hash) {
			return true;
		}
	}
	return false;
}

inline void Report(vk::Queue queue) {
	if (!Enabled().load(std::memory_order_relaxed)) {
		return;
	}
	const auto data = queue.getCheckpointDataNV();
	std::printf("--- GPU checkpoints (%zu, last id %" PRIu64 ") ---\n", data.size(),
	            Next().load() - 1);
	for (const auto& checkpoint: data) {
		const auto id    = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(checkpoint.pCheckpointMarker));
		const auto entry = Ring()[id % RingSize];
		const char* kind = entry.kind == 1   ? "dispatch"
		                   : entry.kind == 2 ? "dispatch-indirect"
		                   : entry.kind == 3 ? "draw"
		                   : entry.kind == 6 ? "after-dispatch"
		                   : entry.kind == 7 ? "after-dispatch-indirect"
		                   : entry.kind == 8 ? "after-draw"
		                   : entry.kind == 10 ? "draw-begin(count,inst)"
		                   : entry.kind == 11 ? "draw-metadata-op(count,inst)"
		                   : entry.kind == 12 ? "draw-prepared"
		                   : entry.kind == 13 ? "draw-step(1 bindings,2 vb/ib,3 targets,4 commit,5 begin-rendering)"
		                   : entry.kind == 14 ? "copy-depth-to-color-step(1 begin,2 targets,3 pre-copy,4 post-copy)"
		                   : entry.kind == 15 ? "tiler-dispatch(texels, src<<32|dst bytes; groups x,y, slot)"
		                   : entry.kind == 16 ? "after-tiler-dispatch"
		                   : entry.kind == 17 ? "image-upload(addr,size; w,h,format)"
		                   : entry.kind == 18 ? "after-image-upload"
		                   : entry.kind == 19 ? "pre-copy-buffer-to-image(aspect, rowlen<<32|imgheight; w,h,copies)"
		                   : entry.kind == 20 ? "after-copy-buffer-to-image(format)"
		                   : entry.kind == 21 ? "after-upload-transit(format)"
		                                     : "?";
		std::printf("checkpoint id=%" PRIu64 " stage=%s %s hash=0x%016" PRIx64
		            " hash2=0x%016" PRIx64 " groups=%ux%ux%u\n",
		            id, vk::to_string(checkpoint.stage).c_str(), kind, entry.hash, entry.hash2,
		            entry.x, entry.y, entry.z);
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::GpuCheckpoints
