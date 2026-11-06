#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_UNPROTECTABLERANGES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_UNPROTECTABLERANGES_H_

// Guest memory that GPU page tracking must never write-protect.
//
// Games can run code on stacks they allocate themselves (Tekken 8 runs its task system on
// fibers whose stacks sit in direct memory next to GPU resources). If tracking write-protects
// such a page, the next push faults, Windows cannot store the exception record on the same
// protected stack, and the process dies without any report. Pages listed here stay writable;
// the price is that CPU writes to them are not seen by GPU tracking, which is fine for stacks.

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics::UnprotectableRanges {

inline constexpr uint64_t Granularity = 0x4000;

struct State {
	std::mutex                                 mutex;
	std::vector<std::pair<uint64_t, uint64_t>> ranges; // [begin, end), sorted, disjoint
};

inline State& Get() {
	static State state;
	return state;
}

inline void Add(uint64_t address, uint64_t size) {
	if (size == 0) {
		return;
	}
	const uint64_t begin = address & ~(Granularity - 1);
	const uint64_t end   = (address + size + Granularity - 1) & ~(Granularity - 1);
	auto&          s     = Get();
	std::scoped_lock lock {s.mutex};
	s.ranges.emplace_back(begin, end);
	std::sort(s.ranges.begin(), s.ranges.end());
	std::vector<std::pair<uint64_t, uint64_t>> merged;
	for (const auto& range: s.ranges) {
		if (!merged.empty() && range.first <= merged.back().second) {
			merged.back().second = std::max(merged.back().second, range.second);
		} else {
			merged.push_back(range);
		}
	}
	s.ranges = std::move(merged);
}

// Calls fn(begin, size) for each part of [address, address + size) outside every listed range.
template <typename Fn>
inline void ForEachProtectable(uint64_t address, uint64_t size, Fn&& fn) {
	auto&            s = Get();
	std::vector<std::pair<uint64_t, uint64_t>> overlaps;
	{
		std::scoped_lock lock {s.mutex};
		const uint64_t   end = address + size;
		for (const auto& range: s.ranges) {
			if (range.first < end && address < range.second) {
				overlaps.push_back(range);
			}
		}
	}
	uint64_t cursor = address;
	const uint64_t end = address + size;
	for (const auto& range: overlaps) {
		if (range.first > cursor) {
			fn(cursor, range.first - cursor);
		}
		cursor = std::max(cursor, range.second);
	}
	if (cursor < end) {
		fn(cursor, end - cursor);
	}
}

} // namespace Libs::Graphics::UnprotectableRanges

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_UNPROTECTABLERANGES_H_
