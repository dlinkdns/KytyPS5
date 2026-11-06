#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWTRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWTRACE_H_

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// KYTY_TRACE_DRAW=1: every ~2 s, prints how much GPU-thread time each stage of draw processing
// took, per second and per draw. GPU thread only.
namespace Libs::Graphics::DrawTrace {

enum Stage : uint32_t {
	Total,
	RenderState,
	StageBindings,
	GraphicsBindings,
	VertexIndex,
	Pipeline,
	RenderTargets,
	Commit,
	Record,
	Dispatch,
	Count
};

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TRACE_DRAW");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

struct State {
	std::array<uint64_t, Count>           ns {};
	uint64_t                              draws      = 0;
	uint64_t                              dispatches = 0;
	std::chrono::steady_clock::time_point last       = std::chrono::steady_clock::now();
};

inline State& Get() {
	static State state;
	return state;
}

inline void Report() {
	auto&        state   = Get();
	const auto   now     = std::chrono::steady_clock::now();
	const double seconds = std::chrono::duration<double>(now - state.last).count();
	if (seconds < 2.0) {
		return;
	}
	static constexpr const char* names[Count] = {"draw total",  "render state", "stage bindings",
	                                             "graph bindings", "vertex/index", "pipeline",
	                                             "render targets", "commit",       "record",
	                                             "dispatch total"};
	std::printf("--- draw stages over %.1f s: %.0f draws/s, %.0f dispatches/s ---\n", seconds,
	            static_cast<double>(state.draws) / seconds,
	            static_cast<double>(state.dispatches) / seconds);
	for (uint32_t i = 0; i < Count; i++) {
		const double ms_per_s = static_cast<double>(state.ns[i]) / 1e6 / seconds;
		const auto   count    = i == Dispatch ? state.dispatches : state.draws;
		const double us_each  = count != 0 ? static_cast<double>(state.ns[i]) / 1e3 / count : 0.0;
		std::printf("  %-15s %7.1f ms/s  %6.2f us each\n", names[i], ms_per_s, us_each);
	}
	std::fflush(stdout);
	state = {};
	state.last = now;
}

class Scope {
public:
	explicit Scope(Stage stage) noexcept: m_stage(stage), m_enabled(Enabled()) {
		if (m_enabled) {
			m_start = std::chrono::steady_clock::now();
		}
	}
	~Scope() {
		if (!m_enabled) {
			return;
		}
		auto& state = Get();
		state.ns[m_stage] += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
		                                                         m_start)
		        .count());
		if (m_stage == Total) {
			state.draws++;
			Report();
		} else if (m_stage == Dispatch) {
			state.dispatches++;
			Report();
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Stage                                 m_stage;
	bool                                  m_enabled;
	std::chrono::steady_clock::time_point m_start {};
};

} // namespace Libs::Graphics::DrawTrace

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWTRACE_H_
