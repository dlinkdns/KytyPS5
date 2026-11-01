#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace Libs::Graphics {

namespace {

// One record per draw or dispatch; the checkpoint marker is the record's sequence number.
struct GpuCheckpointRecord {
	uint64_t seq       = 0;
	uint64_t submit_id = 0;
	uint64_t arg4      = 0;
	uint64_t vs_addr   = 0;
	uint64_t ps_addr   = 0;
	uint64_t cs_addr   = 0;
	uint32_t op        = 0;
	uint32_t args[4]   = {};
};

constexpr uint64_t                                    kCheckpointRing = 1u << 16u;
std::array<GpuCheckpointRecord, kCheckpointRing>      g_checkpoints {};
std::atomic<uint64_t>                                 g_checkpoint_seq {0};

void PrintCheckpointLine(const char* fmt_text, const char* where, const GpuCheckpointRecord& r) {
	char line[512];
	std::snprintf(line, sizeof(line),
	              "%s %s seq=%" PRIu64 " op=%u submit=%" PRIu64 " args=%u,%u,%u,%u,0x%" PRIx64
	              " vs=0x%" PRIx64 " ps=0x%" PRIx64 " cs=0x%" PRIx64 "\n",
	              fmt_text, where, r.seq, r.op, r.submit_id, r.args[0], r.args[1], r.args[2],
	              r.args[3], r.arg4, r.vs_addr, r.ps_addr, r.cs_addr);
	LOGF("%s", line);
	std::printf("%s", line);
}

} // namespace

bool GpuDebugSkip(const char* name, uint64_t addr) {
	const char* list = std::getenv(name);
	if (list == nullptr || addr == 0) {
		return false;
	}
	for (const char* p = list; *p != '\0';) {
		char*      end   = nullptr;
		const auto value = std::strtoull(p, &end, 16);
		if (end == p) {
			break;
		}
		if (value == addr) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < 8) {
				std::printf("GPU debug: skipping %s 0x%" PRIx64 "\n", name, addr);
			}
			return true;
		}
		p = (*end == ',') ? end + 1 : end;
	}
	return false;
}

void DumpGpuCheckpoints(GraphicContext& graphics) {
	if (!graphics.nv_checkpoints_enabled ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV == nullptr) {
		return;
	}
	VkQueue  queue = static_cast<VkQueue>(graphics.queue);
	uint32_t count = 0;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, nullptr);
	std::vector<VkCheckpointDataNV> data(count, VkCheckpointDataNV {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, data.data());
	std::printf("GPU checkpoints: %u reported, %" PRIu64 " recorded\n", count,
	            g_checkpoint_seq.load());
	for (const auto& d: data) {
		const auto  seq = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d.pCheckpointMarker));
		const auto& r   = g_checkpoints[seq % kCheckpointRing];
		char        where[64];
		std::snprintf(where, sizeof(where), "stage=0x%08x", static_cast<uint32_t>(d.stage));
		PrintCheckpointLine("GPU checkpoint:", where, r);
	}
	// The draws recorded just before the last reported marker, for context.
	uint64_t last = 0;
	for (const auto& d: data) {
		last = std::max<uint64_t>(last, reinterpret_cast<uintptr_t>(d.pCheckpointMarker));
	}
	for (uint64_t s = (last > 8 ? last - 8 : 1); s <= last + 8 && s <= g_checkpoint_seq.load(); s++) {
		PrintCheckpointLine("  near:", s == last ? "<== last" : "", g_checkpoints[s % kCheckpointRing]);
	}
	std::fflush(stdout);
	Log::Flush();
}

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	return m_buffer;
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;

	if (m_graphics.nv_checkpoints_enabled && m_buffer != nullptr &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV != nullptr) {
		const auto seq = g_checkpoint_seq.fetch_add(1) + 1;
		auto&      r   = g_checkpoints[seq % kCheckpointRing];
		r.seq          = seq;
		r.op           = op;
		r.submit_id    = submit_id;
		r.args[0]      = arg0;
		r.args[1]      = arg1;
		r.args[2]      = arg2;
		r.args[3]      = arg3;
		r.arg4         = arg4;
		if (m_shaders != nullptr) {
			r.vs_addr = m_shaders->GetVs().gs_regs.data_addr;
			r.ps_addr = m_shaders->GetPs().ps_regs.data_addr;
			r.cs_addr = m_shaders->GetCs().cs_regs.data_addr;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV(
		    static_cast<VkCommandBuffer>(m_buffer), reinterpret_cast<const void*>(static_cast<uintptr_t>(seq)));
	}
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state = state;
	for (auto& color: m_render_state.color_attachments) {
		color.is_clear = false;
	}
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering                                           = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
