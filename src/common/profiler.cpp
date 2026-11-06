#include "common/profiler.h"

#include "common/emulatorConfig.h"

#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <cstdio>
#include <tracy/Tracy.hpp>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Profiler {

void SetHostThreadName(const char* name) {
#if defined(_WIN32)
	// Windows tools (Process Explorer, thread_cpu.ps1) read names set by SetThreadDescription.
	using SetDescription = HRESULT(WINAPI*)(HANDLE, PCWSTR);
	static const auto set_description = reinterpret_cast<SetDescription>(reinterpret_cast<void*>(
	    GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription")));
	if (set_description == nullptr || name == nullptr) {
		return;
	}
	wchar_t wide[64] {};
	for (size_t i = 0; i + 1 < sizeof(wide) / sizeof(wide[0]) && name[i] != '\0'; i++) {
		wide[i] = static_cast<unsigned char>(name[i]);
	}
	set_description(GetCurrentThread(), wide);
#else
	(void)name;
#endif
}

void SetThreadName(const char* name) {
	SetHostThreadName(name);
	if (tracy::ProfilerAvailable() && name != nullptr) {
		tracy::SetThreadName(name);
	}
}

void Initialize() {
	if (Config::ProfilerEnabled() && !tracy::ProfilerAvailable()) {
		tracy::StartupProfiler();
		TracySetProgramName("KytyPS5");
		::printf("Tracy profiler enabled: client %d.%d.%d, protocol %u, "
		         "broadcast %u, connect to 127.0.0.1:8086\n",
		         tracy::Version::Major, tracy::Version::Minor, tracy::Version::Patch,
		         tracy::ProtocolVersion, tracy::BroadcastVersion);
	}
}

void Shutdown() {
	if (tracy::ProfilerAvailable()) {
		tracy::ShutdownProfiler();
	}
}

} // namespace Profiler
