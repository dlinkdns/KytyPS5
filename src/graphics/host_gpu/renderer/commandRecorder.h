#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "graphics/host_gpu/vulkanCommon.h"

// Records the GPU thread's Vulkan commands on a second thread. While a command buffer is
// active, the dispatcher's vkCmd* entries copy their arguments into a ring that the recorder
// thread replays into the driver, in order. Other command buffers and threads call the
// driver directly. KYTY_RECORD_THREAD=0 disables it.
namespace Libs::Graphics::CommandRecorder {

// Once, right after the device-level dispatcher init.
void Install();
[[nodiscard]] bool Enabled();
// After vkBeginCommandBuffer: this thread's commands into `buffer` go to the recorder.
void Begin(vk::CommandBuffer buffer);
// Before vkEndCommandBuffer: replays everything queued, then records directly again.
void End(vk::CommandBuffer buffer);
// Before code that records into `buffer` without the dispatcher (ImGui): replays everything
// queued so far, so its commands follow them.
void Sync(vk::CommandBuffer buffer);

} // namespace Libs::Graphics::CommandRecorder

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
