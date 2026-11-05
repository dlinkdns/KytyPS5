#pragma once

// Diagnostics for GPU hangs: with KYTY_GPU_CHECKPOINTS=1 on a device that supports
// VK_NV_device_diagnostic_checkpoints, every dispatch and draw records a checkpoint, and a lost
// device reports which ones the GPU had reached and in which pipeline stage.

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

inline void Mark(vk::CommandBuffer buffer, uint32_t kind, uint64_t hash, uint64_t hash2 = 0,
                 uint32_t x = 0, uint32_t y = 0, uint32_t z = 0) {
	if (!Enabled().load(std::memory_order_relaxed)) {
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
		                                     : "?";
		std::printf("checkpoint id=%" PRIu64 " stage=%s %s hash=0x%016" PRIx64
		            " hash2=0x%016" PRIx64 " groups=%ux%ux%u\n",
		            id, vk::to_string(checkpoint.stage).c_str(), kind, entry.hash, entry.hash2,
		            entry.x, entry.y, entry.z);
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::GpuCheckpoints
