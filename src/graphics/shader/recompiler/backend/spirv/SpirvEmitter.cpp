#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <unordered_set>
#include <mutex>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

// Whether a wave64 compute program can run one guest lane per host invocation on a 32-wide host.
// The whole workgroup must be one wave, so a workgroup barrier joins exactly its two host
// subgroups (branches are wave-uniform, so both reach every exchange), and every cross-lane
// operation must be one the split emitters handle.
bool SplitMultiWaveEnabled() {
	// On by default; KYTY_SPLIT_WAVE64_MULTI=0 keeps multi-wave workgroups at two lanes per
	// invocation.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPLIT_WAVE64_MULTI");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

uint32_t SplitWaveThreads(ShaderStageInputInfo input_info) {
	const auto& cs = *input_info.compute;
	return std::max(cs.threads_num[0], 1u) * std::max(cs.threads_num[1], 1u) *
	       std::max(cs.threads_num[2], 1u);
}

// One line per multi-wave compute shader: whether it splits, or what stops it.
void ReportSplit(const IR::Program& program, uint32_t threads, const char* outcome) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> seen;
	std::scoped_lock                    lock(mutex);
	if (threads > 64u && seen.insert(program.shader_hash).second) {
		std::printf("split wave64: shader 0x%016llx threads=%u: %s\n",
		            static_cast<unsigned long long>(program.shader_hash), threads, outcome);
	}
}

bool CanSplitWave(const IR::Program& program, ShaderStageInputInfo input_info) {
	if (program.stage != ShaderType::Compute || !input_info.compute->split_wave64) {
		return false;
	}
	const auto& cs      = *input_info.compute;
	const auto  threads = SplitWaveThreads(input_info);
	const auto  waves   = threads / 64u;
	// Several waves need whole waves; their exchange words and handshake counters share LDS.
	if (threads % 64u != 0u || waves == 0u || waves > 16u ||
	    (waves > 1u && !SplitMultiWaveEnabled()) || program.info.compute_derivatives ||
	    program.info.shared_int64_atomics ||
	    (cs.lds_storage ? 0u : cs.lds_size_dwords * 4u) + threads * 4u + waves * 8u + 256u >
	        48u * 1024u) {
		ReportSplit(program, threads, "no (workgroup, derivatives, int64 LDS atomics, swizzle or LDS size)");
		return false;
	}
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const char* blocker = nullptr;
			switch (inst.GetOpcode()) {
				// LDS permutes, swizzles and permlane16 stay within a 32-lane half (the host
				// subgroup); LiveMask joins the halves' ballots.
				case IR::ValueOpcode::DataAppend: blocker = "no (ds_append)"; break;
				case IR::ValueOpcode::DataConsume: blocker = "no (ds_consume)"; break;
				default: break;
			}
			if (blocker != nullptr) {
				ReportSplit(program, threads, blocker);
				return false;
			}
		}
	}
	return true;
}

} // namespace

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, const Profile& profile,
                                  bool emit_debug_info) {
	using namespace Emitter;

	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	IR::ValidateProgram(program);
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	EmitterState state(program, input_info, profile);
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	if (state.lane_count == 2u && CanSplitWave(program, input_info)) {
		state.lane_count  = 1u;
		state.split_wave  = true;
		state.split_waves = SplitWaveThreads(input_info) / 64u;
		ReportSplit(program, SplitWaveThreads(input_info), "yes");
	}
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
