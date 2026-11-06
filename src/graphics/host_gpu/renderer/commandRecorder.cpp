#include "graphics/host_gpu/renderer/commandRecorder.h"

#include "common/assert.h"

#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <thread>
#include <tuple>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics::CommandRecorder {

namespace {

constexpr uint64_t AlignUp16(uint64_t value) {
	return (value + 15u) & ~uint64_t {15u};
}

inline void CpuRelax() {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#endif
}

using ReplayFn = void (*)(const uint8_t* payload);

struct alignas(16) Header {
	ReplayFn replay;
	uint64_t size; // Whole packet, header included; a null replay marks padding.
};
static_assert(sizeof(Header) == 16);

// Single producer (the GPU thread), single consumer (the recorder thread).
class Ring {
public:
	static constexpr uint64_t Capacity = uint64_t {64} << 20u;

	Ring(): m_data(static_cast<uint8_t*>(::operator new(Capacity, std::align_val_t {64}))) {}

	// Space for one packet; Publish makes it visible.
	uint8_t* Reserve(uint64_t payload_bytes, ReplayFn replay) {
		const uint64_t total = AlignUp16(sizeof(Header) + payload_bytes);
		EXIT_IF(total > Capacity / 4u);
		uint64_t position = m_local_write % Capacity;
		if (position + total > Capacity) {
			const uint64_t pad = Capacity - position;
			WaitForSpace(pad + total);
			auto* padding   = reinterpret_cast<Header*>(m_data.get() + position);
			padding->replay = nullptr;
			padding->size   = pad;
			m_local_write += pad;
			position = 0;
		} else {
			WaitForSpace(total);
		}
		auto* header   = reinterpret_cast<Header*>(m_data.get() + position);
		header->replay = replay;
		header->size   = total;
		m_pending      = total;
		return m_data.get() + position + sizeof(Header);
	}

	void Publish() {
		m_local_write += m_pending;
		m_write.store(m_local_write, std::memory_order_seq_cst);
		if (m_consumer_sleeping.load(std::memory_order_seq_cst)) {
			m_write.notify_one();
		}
	}

	// Returns once the recorder thread has replayed everything published.
	void Drain() const {
		uint32_t spins = 0;
		while (m_read.load(std::memory_order_acquire) != m_local_write) {
			if (++spins < 4096u) {
				CpuRelax();
			} else {
				std::this_thread::yield();
			}
		}
	}

	void Run(const std::stop_token& stop) {
		uint64_t read = 0;
		while (!stop.stop_requested()) {
			uint64_t write = m_write.load(std::memory_order_acquire);
			if (read == write) {
				for (uint32_t i = 0; i < 4096u && write == read; i++) {
					CpuRelax();
					write = m_write.load(std::memory_order_acquire);
				}
				if (write == read) {
					m_consumer_sleeping.store(true, std::memory_order_seq_cst);
					write = m_write.load(std::memory_order_seq_cst);
					if (write == read && !stop.stop_requested()) {
						m_write.wait(read, std::memory_order_seq_cst);
					}
					m_consumer_sleeping.store(false, std::memory_order_seq_cst);
					continue;
				}
			}
			while (read != write) {
				const auto* header = reinterpret_cast<const Header*>(m_data.get() + read % Capacity);
				if (header->replay != nullptr) {
					header->replay(reinterpret_cast<const uint8_t*>(header) + sizeof(Header));
				}
				read += header->size;
				m_read.store(read, std::memory_order_release);
			}
		}
	}

	void Wake() {
		m_write.fetch_add(0, std::memory_order_seq_cst);
		m_write.notify_all();
	}

private:
	struct Deleter {
		void operator()(uint8_t* data) const { ::operator delete(data, std::align_val_t {64}); }
	};

	void WaitForSpace(uint64_t bytes) const {
		uint32_t spins = 0;
		while (m_local_write + bytes - m_read.load(std::memory_order_acquire) > Capacity) {
			if (++spins < 4096u) {
				CpuRelax();
			} else {
				std::this_thread::yield();
			}
		}
	}

	std::unique_ptr<uint8_t[], Deleter>  m_data;
	uint64_t                             m_local_write = 0;
	uint64_t                             m_pending     = 0;
	alignas(64) std::atomic<uint64_t>    m_write {0};
	alignas(64) std::atomic<uint64_t>    m_read {0};
	alignas(64) std::atomic<bool>        m_consumer_sleeping {false};
};

Ring*                        g_ring = nullptr;
std::atomic<VkCommandBuffer> g_active {VK_NULL_HANDLE};
thread_local bool            t_producer = false;
bool                         g_installed = false;

inline bool Deferring(VkCommandBuffer buffer) {
	return t_producer && buffer == g_active.load(std::memory_order_relaxed);
}

// Copies payload pieces into one packet; arrays land after the fixed part, 16-byte aligned.
class PacketWriter {
public:
	explicit PacketWriter(uint8_t* base): m_base(base) {}

	template <typename T>
	T* Fixed() {
		auto* result = reinterpret_cast<T*>(m_base + m_offset);
		m_offset += AlignUp16(sizeof(T));
		return result;
	}

	template <typename T>
	const T* Array(const T* source, uint64_t count) {
		if (source == nullptr || count == 0) {
			return nullptr;
		}
		auto* result = reinterpret_cast<T*>(m_base + m_offset);
		std::memcpy(result, source, sizeof(T) * count);
		m_offset += AlignUp16(sizeof(T) * count);
		return result;
	}

private:
	uint8_t* m_base;
	uint64_t m_offset = 0;
};

template <typename T>
constexpr uint64_t ArrayBytes(const T* source, uint64_t count) {
	return source == nullptr || count == 0 ? 0 : AlignUp16(sizeof(T) * count);
}

template <typename T>
constexpr uint64_t FixedBytes() {
	return AlignUp16(sizeof(T));
}

// Commands whose arguments are all values.
template <auto* real, typename... Args>
struct ValueCommand {
	struct Payload {
		VkCommandBuffer     buffer;
		std::tuple<Args...> args;
	};
	static void Replay(const uint8_t* data) {
		const auto* payload = reinterpret_cast<const Payload*>(data);
		std::apply([&](auto... args) { (*real)(payload->buffer, args...); }, payload->args);
	}
	static VKAPI_ATTR void VKAPI_CALL Thunk(VkCommandBuffer buffer, Args... args) {
		if (!Deferring(buffer)) {
			(*real)(buffer, args...);
			return;
		}
		auto* payload = g_ring->Reserve(sizeof(Payload), &Replay);
		new (payload) Payload {buffer, std::tuple<Args...>(args...)};
		g_ring->Publish();
	}
};

// Real entry points, saved from the dispatcher.
#define KYTY_REAL(name) PFN_##name real_##name = nullptr;
KYTY_REAL(vkCmdBindPipeline)
KYTY_REAL(vkCmdDispatch)
KYTY_REAL(vkCmdEndRendering)
KYTY_REAL(vkCmdResetQueryPool)
KYTY_REAL(vkCmdDraw)
KYTY_REAL(vkCmdSetStencilWriteMask)
KYTY_REAL(vkCmdSetStencilReference)
KYTY_REAL(vkCmdSetStencilCompareMask)
KYTY_REAL(vkCmdSetStencilOp)
KYTY_REAL(vkCmdWriteTimestamp)
KYTY_REAL(vkCmdFillBuffer)
KYTY_REAL(vkCmdDrawMeshTasksEXT)
KYTY_REAL(vkCmdDrawIndexed)
KYTY_REAL(vkCmdSetStencilTestEnable)
KYTY_REAL(vkCmdSetDepthWriteEnable)
KYTY_REAL(vkCmdSetDepthTestEnable)
KYTY_REAL(vkCmdSetDepthBoundsTestEnable)
KYTY_REAL(vkCmdSetDepthBiasEnable)
KYTY_REAL(vkCmdSetLineWidth)
KYTY_REAL(vkCmdSetFrontFace)
KYTY_REAL(vkCmdSetCullMode)
KYTY_REAL(vkCmdSetDepthCompareOp)
KYTY_REAL(vkCmdSetDepthBounds)
KYTY_REAL(vkCmdSetDepthBias)
KYTY_REAL(vkCmdSetCheckpointNV)
KYTY_REAL(vkCmdSetAttachmentFeedbackLoopEnableEXT)
KYTY_REAL(vkCmdEndQuery)
KYTY_REAL(vkCmdBeginQuery)
KYTY_REAL(vkCmdEndConditionalRenderingEXT)
KYTY_REAL(vkCmdDrawIndirect)
KYTY_REAL(vkCmdDrawIndexedIndirect)
KYTY_REAL(vkCmdDispatchIndirect)
KYTY_REAL(vkCmdDispatchBase)
KYTY_REAL(vkCmdBindIndexBuffer)
KYTY_REAL(vkCmdPipelineBarrier)
KYTY_REAL(vkCmdPipelineBarrier2)
KYTY_REAL(vkCmdPushDescriptorSetKHR)
KYTY_REAL(vkCmdPushConstants)
KYTY_REAL(vkCmdCopyImage)
KYTY_REAL(vkCmdCopyBuffer)
KYTY_REAL(vkCmdBeginRendering)
KYTY_REAL(vkCmdCopyBufferToImage)
KYTY_REAL(vkCmdClearColorImage)
KYTY_REAL(vkCmdSetViewport)
KYTY_REAL(vkCmdSetScissor)
KYTY_REAL(vkCmdCopyImageToBuffer)
KYTY_REAL(vkCmdClearDepthStencilImage)
KYTY_REAL(vkCmdBlitImage)
KYTY_REAL(vkCmdBindDescriptorSets)
KYTY_REAL(vkCmdSetViewportWithCount)
KYTY_REAL(vkCmdSetScissorWithCount)
KYTY_REAL(vkCmdSetBlendConstants)
KYTY_REAL(vkCmdResolveImage)
KYTY_REAL(vkCmdBeginConditionalRenderingEXT)
KYTY_REAL(vkCmdClearAttachments)
KYTY_REAL(vkCmdBindVertexBuffers2)
#undef KYTY_REAL

// ---- Commands with pointer arguments: deep copies. ----

struct PipelineBarrierArgs {
	VkCommandBuffer              buffer;
	VkPipelineStageFlags         src;
	VkPipelineStageFlags         dst;
	VkDependencyFlags            flags;
	uint32_t                     memory_count;
	const VkMemoryBarrier*       memory;
	uint32_t                     buffer_count;
	const VkBufferMemoryBarrier* buffers;
	uint32_t                     image_count;
	const VkImageMemoryBarrier*  images;
};

void ReplayPipelineBarrier(const uint8_t* data) {
	const auto* a = reinterpret_cast<const PipelineBarrierArgs*>(data);
	real_vkCmdPipelineBarrier(a->buffer, a->src, a->dst, a->flags, a->memory_count, a->memory,
	                          a->buffer_count, a->buffers, a->image_count, a->images);
}

VKAPI_ATTR void VKAPI_CALL ThunkPipelineBarrier(VkCommandBuffer buffer, VkPipelineStageFlags src,
                                                VkPipelineStageFlags dst, VkDependencyFlags flags,
                                                uint32_t memory_count, const VkMemoryBarrier* memory,
                                                uint32_t                     buffer_count,
                                                const VkBufferMemoryBarrier* buffers,
                                                uint32_t                     image_count,
                                                const VkImageMemoryBarrier*  images) {
	if (!Deferring(buffer)) {
		real_vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, memory, buffer_count,
		                          buffers, image_count, images);
		return;
	}
	const auto size = FixedBytes<PipelineBarrierArgs>() + ArrayBytes(memory, memory_count) +
	                  ArrayBytes(buffers, buffer_count) + ArrayBytes(images, image_count);
	PacketWriter writer(g_ring->Reserve(size, &ReplayPipelineBarrier));
	auto*        a = writer.Fixed<PipelineBarrierArgs>();
	*a = {buffer, src, dst, flags, memory_count, nullptr, buffer_count, nullptr, image_count, nullptr};
	a->memory  = writer.Array(memory, memory_count);
	a->buffers = writer.Array(buffers, buffer_count);
	a->images  = writer.Array(images, image_count);
	g_ring->Publish();
}

struct DependencyArgs {
	VkCommandBuffer  buffer;
	VkDependencyInfo info;
};

void ReplayPipelineBarrier2(const uint8_t* data) {
	const auto* a = reinterpret_cast<const DependencyArgs*>(data);
	real_vkCmdPipelineBarrier2(a->buffer, &a->info);
}

VKAPI_ATTR void VKAPI_CALL ThunkPipelineBarrier2(VkCommandBuffer buffer, const VkDependencyInfo* info) {
	if (!Deferring(buffer) || info->pNext != nullptr) {
		if (Deferring(buffer)) {
			g_ring->Drain();
		}
		real_vkCmdPipelineBarrier2(buffer, info);
		return;
	}
	const auto size = FixedBytes<DependencyArgs>() +
	                  ArrayBytes(info->pMemoryBarriers, info->memoryBarrierCount) +
	                  ArrayBytes(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount) +
	                  ArrayBytes(info->pImageMemoryBarriers, info->imageMemoryBarrierCount);
	PacketWriter writer(g_ring->Reserve(size, &ReplayPipelineBarrier2));
	auto*        a = writer.Fixed<DependencyArgs>();
	a->buffer      = buffer;
	a->info        = *info;
	a->info.pMemoryBarriers = writer.Array(info->pMemoryBarriers, info->memoryBarrierCount);
	a->info.pBufferMemoryBarriers =
	    writer.Array(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount);
	a->info.pImageMemoryBarriers =
	    writer.Array(info->pImageMemoryBarriers, info->imageMemoryBarrierCount);
	g_ring->Publish();
}

struct PushDescriptorArgs {
	VkCommandBuffer             buffer;
	VkPipelineBindPoint         bind_point;
	VkPipelineLayout            layout;
	uint32_t                    set;
	uint32_t                    count;
	const VkWriteDescriptorSet* writes;
};

void ReplayPushDescriptorSet(const uint8_t* data) {
	const auto* a = reinterpret_cast<const PushDescriptorArgs*>(data);
	real_vkCmdPushDescriptorSetKHR(a->buffer, a->bind_point, a->layout, a->set, a->count,
	                               a->writes);
}

VKAPI_ATTR void VKAPI_CALL ThunkPushDescriptorSet(VkCommandBuffer buffer, VkPipelineBindPoint bind_point,
                                                  VkPipelineLayout layout, uint32_t set,
                                                  uint32_t                    count,
                                                  const VkWriteDescriptorSet* writes) {
	bool chained = false;
	for (uint32_t i = 0; i < count; i++) {
		chained = chained || writes[i].pNext != nullptr;
	}
	if (!Deferring(buffer) || chained) {
		if (Deferring(buffer)) {
			g_ring->Drain();
		}
		real_vkCmdPushDescriptorSetKHR(buffer, bind_point, layout, set, count, writes);
		return;
	}
	uint64_t size = FixedBytes<PushDescriptorArgs>() + ArrayBytes(writes, count);
	for (uint32_t i = 0; i < count; i++) {
		const auto& w = writes[i];
		size += ArrayBytes(w.pImageInfo, w.descriptorCount) +
		        ArrayBytes(w.pBufferInfo, w.descriptorCount) +
		        ArrayBytes(w.pTexelBufferView, w.descriptorCount);
	}
	PacketWriter writer(g_ring->Reserve(size, &ReplayPushDescriptorSet));
	auto*        a = writer.Fixed<PushDescriptorArgs>();
	*a             = {buffer, bind_point, layout, set, count, nullptr};
	auto* copies   = const_cast<VkWriteDescriptorSet*>(writer.Array(writes, count));
	for (uint32_t i = 0; i < count; i++) {
		auto& w            = copies[i];
		w.pImageInfo       = writer.Array(writes[i].pImageInfo, w.descriptorCount);
		w.pBufferInfo      = writer.Array(writes[i].pBufferInfo, w.descriptorCount);
		w.pTexelBufferView = writer.Array(writes[i].pTexelBufferView, w.descriptorCount);
	}
	a->writes = copies;
	g_ring->Publish();
}

struct PushConstantsArgs {
	VkCommandBuffer    buffer;
	VkPipelineLayout   layout;
	VkShaderStageFlags stages;
	uint32_t           offset;
	uint32_t           size;
	const uint8_t*     values;
};

void ReplayPushConstants(const uint8_t* data) {
	const auto* a = reinterpret_cast<const PushConstantsArgs*>(data);
	real_vkCmdPushConstants(a->buffer, a->layout, a->stages, a->offset, a->size, a->values);
}

VKAPI_ATTR void VKAPI_CALL ThunkPushConstants(VkCommandBuffer buffer, VkPipelineLayout layout,
                                              VkShaderStageFlags stages, uint32_t offset,
                                              uint32_t size, const void* values) {
	if (!Deferring(buffer)) {
		real_vkCmdPushConstants(buffer, layout, stages, offset, size, values);
		return;
	}
	const auto* bytes = static_cast<const uint8_t*>(values);
	PacketWriter writer(g_ring->Reserve(FixedBytes<PushConstantsArgs>() + ArrayBytes(bytes, size),
	                                    &ReplayPushConstants));
	auto* a   = writer.Fixed<PushConstantsArgs>();
	*a        = {buffer, layout, stages, offset, size, nullptr};
	a->values = writer.Array(bytes, size);
	g_ring->Publish();
}

// Image/buffer copies: handles, layouts, one region array.
template <typename Region>
struct RegionArgs {
	VkCommandBuffer buffer;
	uint64_t        handles[2];
	VkImageLayout   layouts[2];
	uint32_t        count;
	const Region*   regions;
	VkFilter        filter;
};

void ReplayCopyImage(const uint8_t* data) {
	const auto* a = reinterpret_cast<const RegionArgs<VkImageCopy>*>(data);
	real_vkCmdCopyImage(a->buffer, std::bit_cast<VkImage>(a->handles[0]), a->layouts[0],
	                    std::bit_cast<VkImage>(a->handles[1]), a->layouts[1], a->count,
	                    a->regions);
}
void ReplayBlitImage(const uint8_t* data) {
	const auto* a = reinterpret_cast<const RegionArgs<VkImageBlit>*>(data);
	real_vkCmdBlitImage(a->buffer, std::bit_cast<VkImage>(a->handles[0]), a->layouts[0],
	                    std::bit_cast<VkImage>(a->handles[1]), a->layouts[1], a->count,
	                    a->regions, a->filter);
}
void ReplayResolveImage(const uint8_t* data) {
	const auto* a = reinterpret_cast<const RegionArgs<VkImageResolve>*>(data);
	real_vkCmdResolveImage(a->buffer, std::bit_cast<VkImage>(a->handles[0]), a->layouts[0],
	                       std::bit_cast<VkImage>(a->handles[1]), a->layouts[1], a->count,
	                       a->regions);
}
void ReplayCopyBuffer(const uint8_t* data) {
	const auto* a = reinterpret_cast<const RegionArgs<VkBufferCopy>*>(data);
	real_vkCmdCopyBuffer(a->buffer, std::bit_cast<VkBuffer>(a->handles[0]),
	                     std::bit_cast<VkBuffer>(a->handles[1]), a->count, a->regions);
}
void ReplayCopyBufferToImage(const uint8_t* data) {
	const auto* a = reinterpret_cast<const RegionArgs<VkBufferImageCopy>*>(data);
	real_vkCmdCopyBufferToImage(a->buffer, std::bit_cast<VkBuffer>(a->handles[0]),
	                            std::bit_cast<VkImage>(a->handles[1]), a->layouts[1], a->count,
	                            a->regions);
}
void ReplayCopyImageToBuffer(const uint8_t* data) {
	const auto* a = reinterpret_cast<const RegionArgs<VkBufferImageCopy>*>(data);
	real_vkCmdCopyImageToBuffer(a->buffer, std::bit_cast<VkImage>(a->handles[0]),
	                            a->layouts[0], std::bit_cast<VkBuffer>(a->handles[1]),
	                            a->count, a->regions);
}

template <typename Region>
void PushRegions(ReplayFn replay, VkCommandBuffer buffer, uint64_t first, VkImageLayout first_layout,
                 uint64_t second, VkImageLayout second_layout, uint32_t count,
                 const Region* regions, VkFilter filter = VK_FILTER_NEAREST) {
	PacketWriter writer(g_ring->Reserve(
	    FixedBytes<RegionArgs<Region>>() + ArrayBytes(regions, count), replay));
	auto* a       = writer.Fixed<RegionArgs<Region>>();
	*a            = {buffer, {first, second}, {first_layout, second_layout}, count, nullptr, filter};
	a->regions    = writer.Array(regions, count);
	g_ring->Publish();
}

template <typename Handle>
uint64_t HandleBits(Handle handle) {
	static_assert(sizeof(Handle) == sizeof(uint64_t));
	return std::bit_cast<uint64_t>(handle);
}

VKAPI_ATTR void VKAPI_CALL ThunkCopyImage(VkCommandBuffer buffer, VkImage src, VkImageLayout src_layout,
                                          VkImage dst, VkImageLayout dst_layout, uint32_t count,
                                          const VkImageCopy* regions) {
	if (!Deferring(buffer)) {
		real_vkCmdCopyImage(buffer, src, src_layout, dst, dst_layout, count, regions);
		return;
	}
	PushRegions(&ReplayCopyImage, buffer, HandleBits(src), src_layout, HandleBits(dst), dst_layout,
	            count, regions);
}

VKAPI_ATTR void VKAPI_CALL ThunkBlitImage(VkCommandBuffer buffer, VkImage src, VkImageLayout src_layout,
                                          VkImage dst, VkImageLayout dst_layout, uint32_t count,
                                          const VkImageBlit* regions, VkFilter filter) {
	if (!Deferring(buffer)) {
		real_vkCmdBlitImage(buffer, src, src_layout, dst, dst_layout, count, regions, filter);
		return;
	}
	PushRegions(&ReplayBlitImage, buffer, HandleBits(src), src_layout, HandleBits(dst), dst_layout,
	            count, regions, filter);
}

VKAPI_ATTR void VKAPI_CALL ThunkResolveImage(VkCommandBuffer buffer, VkImage src,
                                             VkImageLayout src_layout, VkImage dst,
                                             VkImageLayout dst_layout, uint32_t count,
                                             const VkImageResolve* regions) {
	if (!Deferring(buffer)) {
		real_vkCmdResolveImage(buffer, src, src_layout, dst, dst_layout, count, regions);
		return;
	}
	PushRegions(&ReplayResolveImage, buffer, HandleBits(src), src_layout, HandleBits(dst),
	            dst_layout, count, regions);
}

VKAPI_ATTR void VKAPI_CALL ThunkCopyBuffer(VkCommandBuffer buffer, VkBuffer src, VkBuffer dst,
                                           uint32_t count, const VkBufferCopy* regions) {
	if (!Deferring(buffer)) {
		real_vkCmdCopyBuffer(buffer, src, dst, count, regions);
		return;
	}
	PushRegions(&ReplayCopyBuffer, buffer, HandleBits(src), VK_IMAGE_LAYOUT_UNDEFINED,
	            HandleBits(dst), VK_IMAGE_LAYOUT_UNDEFINED, count, regions);
}

VKAPI_ATTR void VKAPI_CALL ThunkCopyBufferToImage(VkCommandBuffer buffer, VkBuffer src, VkImage dst,
                                                  VkImageLayout dst_layout, uint32_t count,
                                                  const VkBufferImageCopy* regions) {
	if (!Deferring(buffer)) {
		real_vkCmdCopyBufferToImage(buffer, src, dst, dst_layout, count, regions);
		return;
	}
	PushRegions(&ReplayCopyBufferToImage, buffer, HandleBits(src), VK_IMAGE_LAYOUT_UNDEFINED,
	            HandleBits(dst), dst_layout, count, regions);
}

VKAPI_ATTR void VKAPI_CALL ThunkCopyImageToBuffer(VkCommandBuffer buffer, VkImage src,
                                                  VkImageLayout src_layout, VkBuffer dst,
                                                  uint32_t count, const VkBufferImageCopy* regions) {
	if (!Deferring(buffer)) {
		real_vkCmdCopyImageToBuffer(buffer, src, src_layout, dst, count, regions);
		return;
	}
	PushRegions(&ReplayCopyImageToBuffer, buffer, HandleBits(src), src_layout, HandleBits(dst),
	            VK_IMAGE_LAYOUT_UNDEFINED, count, regions);
}

struct RenderingArgs {
	VkCommandBuffer           buffer;
	VkRenderingInfo           info;
	VkRenderingAttachmentInfo depth;
	VkRenderingAttachmentInfo stencil;
};

void ReplayBeginRendering(const uint8_t* data) {
	const auto* a = reinterpret_cast<const RenderingArgs*>(data);
	real_vkCmdBeginRendering(a->buffer, &a->info);
}

VKAPI_ATTR void VKAPI_CALL ThunkBeginRendering(VkCommandBuffer buffer, const VkRenderingInfo* info) {
	bool chained = info->pNext != nullptr ||
	               (info->pDepthAttachment != nullptr && info->pDepthAttachment->pNext != nullptr) ||
	               (info->pStencilAttachment != nullptr &&
	                info->pStencilAttachment->pNext != nullptr);
	for (uint32_t i = 0; i < info->colorAttachmentCount; i++) {
		chained = chained || info->pColorAttachments[i].pNext != nullptr;
	}
	if (!Deferring(buffer) || chained) {
		if (Deferring(buffer)) {
			g_ring->Drain();
		}
		real_vkCmdBeginRendering(buffer, info);
		return;
	}
	PacketWriter writer(g_ring->Reserve(
	    FixedBytes<RenderingArgs>() + ArrayBytes(info->pColorAttachments, info->colorAttachmentCount),
	    &ReplayBeginRendering));
	auto* a   = writer.Fixed<RenderingArgs>();
	a->buffer = buffer;
	a->info   = *info;
	a->info.pColorAttachments =
	    writer.Array(info->pColorAttachments, info->colorAttachmentCount);
	if (info->pDepthAttachment != nullptr) {
		a->depth                 = *info->pDepthAttachment;
		a->info.pDepthAttachment = &a->depth;
	}
	if (info->pStencilAttachment != nullptr) {
		a->stencil                 = *info->pStencilAttachment;
		a->info.pStencilAttachment = &a->stencil;
	}
	g_ring->Publish();
}

struct ClearColorArgs {
	VkCommandBuffer                buffer;
	VkImage                        image;
	VkImageLayout                  layout;
	VkClearColorValue              color;
	uint32_t                       count;
	const VkImageSubresourceRange* ranges;
};

void ReplayClearColorImage(const uint8_t* data) {
	const auto* a = reinterpret_cast<const ClearColorArgs*>(data);
	real_vkCmdClearColorImage(a->buffer, a->image, a->layout, &a->color, a->count, a->ranges);
}

VKAPI_ATTR void VKAPI_CALL ThunkClearColorImage(VkCommandBuffer buffer, VkImage image,
                                                VkImageLayout layout, const VkClearColorValue* color,
                                                uint32_t                       count,
                                                const VkImageSubresourceRange* ranges) {
	if (!Deferring(buffer)) {
		real_vkCmdClearColorImage(buffer, image, layout, color, count, ranges);
		return;
	}
	PacketWriter writer(g_ring->Reserve(FixedBytes<ClearColorArgs>() + ArrayBytes(ranges, count),
	                                    &ReplayClearColorImage));
	auto* a   = writer.Fixed<ClearColorArgs>();
	*a        = {buffer, image, layout, *color, count, nullptr};
	a->ranges = writer.Array(ranges, count);
	g_ring->Publish();
}

struct ClearDepthArgs {
	VkCommandBuffer                buffer;
	VkImage                        image;
	VkImageLayout                  layout;
	VkClearDepthStencilValue       value;
	uint32_t                       count;
	const VkImageSubresourceRange* ranges;
};

void ReplayClearDepthStencilImage(const uint8_t* data) {
	const auto* a = reinterpret_cast<const ClearDepthArgs*>(data);
	real_vkCmdClearDepthStencilImage(a->buffer, a->image, a->layout, &a->value, a->count,
	                                 a->ranges);
}

VKAPI_ATTR void VKAPI_CALL ThunkClearDepthStencilImage(VkCommandBuffer buffer, VkImage image,
                                                       VkImageLayout                   layout,
                                                       const VkClearDepthStencilValue* value,
                                                       uint32_t                        count,
                                                       const VkImageSubresourceRange*  ranges) {
	if (!Deferring(buffer)) {
		real_vkCmdClearDepthStencilImage(buffer, image, layout, value, count, ranges);
		return;
	}
	PacketWriter writer(g_ring->Reserve(FixedBytes<ClearDepthArgs>() + ArrayBytes(ranges, count),
	                                    &ReplayClearDepthStencilImage));
	auto* a   = writer.Fixed<ClearDepthArgs>();
	*a        = {buffer, image, layout, *value, count, nullptr};
	a->ranges = writer.Array(ranges, count);
	g_ring->Publish();
}

struct ViewportArgs {
	VkCommandBuffer   buffer;
	uint32_t          first;
	uint32_t          count;
	const VkViewport* viewports;
	const VkRect2D*   rects;
};

void ReplaySetViewport(const uint8_t* data) {
	const auto* a = reinterpret_cast<const ViewportArgs*>(data);
	real_vkCmdSetViewport(a->buffer, a->first, a->count, a->viewports);
}
void ReplaySetScissor(const uint8_t* data) {
	const auto* a = reinterpret_cast<const ViewportArgs*>(data);
	real_vkCmdSetScissor(a->buffer, a->first, a->count, a->rects);
}
void ReplaySetViewportWithCount(const uint8_t* data) {
	const auto* a = reinterpret_cast<const ViewportArgs*>(data);
	real_vkCmdSetViewportWithCount(a->buffer, a->count, a->viewports);
}
void ReplaySetScissorWithCount(const uint8_t* data) {
	const auto* a = reinterpret_cast<const ViewportArgs*>(data);
	real_vkCmdSetScissorWithCount(a->buffer, a->count, a->rects);
}

void PushViewports(ReplayFn replay, VkCommandBuffer buffer, uint32_t first, uint32_t count,
                   const VkViewport* viewports, const VkRect2D* rects) {
	PacketWriter writer(g_ring->Reserve(FixedBytes<ViewportArgs>() + ArrayBytes(viewports, count) +
	                                        ArrayBytes(rects, count),
	                                    replay));
	auto* a      = writer.Fixed<ViewportArgs>();
	*a           = {buffer, first, count, nullptr, nullptr};
	a->viewports = writer.Array(viewports, count);
	a->rects     = writer.Array(rects, count);
	g_ring->Publish();
}

VKAPI_ATTR void VKAPI_CALL ThunkSetViewport(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                            const VkViewport* viewports) {
	if (!Deferring(buffer)) {
		real_vkCmdSetViewport(buffer, first, count, viewports);
		return;
	}
	PushViewports(&ReplaySetViewport, buffer, first, count, viewports, nullptr);
}
VKAPI_ATTR void VKAPI_CALL ThunkSetScissor(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                           const VkRect2D* rects) {
	if (!Deferring(buffer)) {
		real_vkCmdSetScissor(buffer, first, count, rects);
		return;
	}
	PushViewports(&ReplaySetScissor, buffer, first, count, nullptr, rects);
}
VKAPI_ATTR void VKAPI_CALL ThunkSetViewportWithCount(VkCommandBuffer buffer, uint32_t count,
                                                     const VkViewport* viewports) {
	if (!Deferring(buffer)) {
		real_vkCmdSetViewportWithCount(buffer, count, viewports);
		return;
	}
	PushViewports(&ReplaySetViewportWithCount, buffer, 0, count, viewports, nullptr);
}
VKAPI_ATTR void VKAPI_CALL ThunkSetScissorWithCount(VkCommandBuffer buffer, uint32_t count,
                                                    const VkRect2D* rects) {
	if (!Deferring(buffer)) {
		real_vkCmdSetScissorWithCount(buffer, count, rects);
		return;
	}
	PushViewports(&ReplaySetScissorWithCount, buffer, 0, count, nullptr, rects);
}

struct DescriptorSetsArgs {
	VkCommandBuffer        buffer;
	VkPipelineBindPoint    bind_point;
	VkPipelineLayout       layout;
	uint32_t               first;
	uint32_t               count;
	const VkDescriptorSet* sets;
	uint32_t               dynamic_count;
	const uint32_t*        dynamic_offsets;
};

void ReplayBindDescriptorSets(const uint8_t* data) {
	const auto* a = reinterpret_cast<const DescriptorSetsArgs*>(data);
	real_vkCmdBindDescriptorSets(a->buffer, a->bind_point, a->layout, a->first, a->count, a->sets,
	                             a->dynamic_count, a->dynamic_offsets);
}

VKAPI_ATTR void VKAPI_CALL ThunkBindDescriptorSets(VkCommandBuffer buffer, VkPipelineBindPoint bind_point,
                                                   VkPipelineLayout layout, uint32_t first,
                                                   uint32_t count, const VkDescriptorSet* sets,
                                                   uint32_t        dynamic_count,
                                                   const uint32_t* dynamic_offsets) {
	if (!Deferring(buffer)) {
		real_vkCmdBindDescriptorSets(buffer, bind_point, layout, first, count, sets, dynamic_count,
		                             dynamic_offsets);
		return;
	}
	PacketWriter writer(g_ring->Reserve(FixedBytes<DescriptorSetsArgs>() + ArrayBytes(sets, count) +
	                                        ArrayBytes(dynamic_offsets, dynamic_count),
	                                    &ReplayBindDescriptorSets));
	auto* a = writer.Fixed<DescriptorSetsArgs>();
	*a      = {buffer, bind_point, layout, first, count, nullptr, dynamic_count, nullptr};
	a->sets = writer.Array(sets, count);
	a->dynamic_offsets = writer.Array(dynamic_offsets, dynamic_count);
	g_ring->Publish();
}

struct BlendConstantsArgs {
	VkCommandBuffer buffer;
	float           constants[4];
};

void ReplaySetBlendConstants(const uint8_t* data) {
	const auto* a = reinterpret_cast<const BlendConstantsArgs*>(data);
	real_vkCmdSetBlendConstants(a->buffer, a->constants);
}

VKAPI_ATTR void VKAPI_CALL ThunkSetBlendConstants(VkCommandBuffer buffer, const float constants[4]) {
	if (!Deferring(buffer)) {
		real_vkCmdSetBlendConstants(buffer, constants);
		return;
	}
	auto* a   = reinterpret_cast<BlendConstantsArgs*>(
        g_ring->Reserve(sizeof(BlendConstantsArgs), &ReplaySetBlendConstants));
	a->buffer = buffer;
	std::memcpy(a->constants, constants, sizeof(a->constants));
	g_ring->Publish();
}

struct ConditionalArgs {
	VkCommandBuffer                     buffer;
	VkConditionalRenderingBeginInfoEXT info;
};

void ReplayBeginConditionalRendering(const uint8_t* data) {
	const auto* a = reinterpret_cast<const ConditionalArgs*>(data);
	real_vkCmdBeginConditionalRenderingEXT(a->buffer, &a->info);
}

VKAPI_ATTR void VKAPI_CALL ThunkBeginConditionalRendering(
    VkCommandBuffer buffer, const VkConditionalRenderingBeginInfoEXT* info) {
	if (!Deferring(buffer) || info->pNext != nullptr) {
		if (Deferring(buffer)) {
			g_ring->Drain();
		}
		real_vkCmdBeginConditionalRenderingEXT(buffer, info);
		return;
	}
	auto* a   = reinterpret_cast<ConditionalArgs*>(
        g_ring->Reserve(sizeof(ConditionalArgs), &ReplayBeginConditionalRendering));
	a->buffer = buffer;
	a->info   = *info;
	g_ring->Publish();
}

struct ClearAttachmentsArgs {
	VkCommandBuffer          buffer;
	uint32_t                 attachment_count;
	const VkClearAttachment* attachments;
	uint32_t                 rect_count;
	const VkClearRect*       rects;
};

void ReplayClearAttachments(const uint8_t* data) {
	const auto* a = reinterpret_cast<const ClearAttachmentsArgs*>(data);
	real_vkCmdClearAttachments(a->buffer, a->attachment_count, a->attachments, a->rect_count,
	                           a->rects);
}

VKAPI_ATTR void VKAPI_CALL ThunkClearAttachments(VkCommandBuffer buffer, uint32_t attachment_count,
                                                 const VkClearAttachment* attachments,
                                                 uint32_t rect_count, const VkClearRect* rects) {
	if (!Deferring(buffer)) {
		real_vkCmdClearAttachments(buffer, attachment_count, attachments, rect_count, rects);
		return;
	}
	PacketWriter writer(g_ring->Reserve(FixedBytes<ClearAttachmentsArgs>() +
	                                        ArrayBytes(attachments, attachment_count) +
	                                        ArrayBytes(rects, rect_count),
	                                    &ReplayClearAttachments));
	auto* a        = writer.Fixed<ClearAttachmentsArgs>();
	*a             = {buffer, attachment_count, nullptr, rect_count, nullptr};
	a->attachments = writer.Array(attachments, attachment_count);
	a->rects       = writer.Array(rects, rect_count);
	g_ring->Publish();
}

struct VertexBuffersArgs {
	VkCommandBuffer     buffer;
	uint32_t            first;
	uint32_t            count;
	const VkBuffer*     buffers;
	const VkDeviceSize* offsets;
	const VkDeviceSize* sizes;
	const VkDeviceSize* strides;
};

void ReplayBindVertexBuffers2(const uint8_t* data) {
	const auto* a = reinterpret_cast<const VertexBuffersArgs*>(data);
	real_vkCmdBindVertexBuffers2(a->buffer, a->first, a->count, a->buffers, a->offsets, a->sizes,
	                             a->strides);
}

VKAPI_ATTR void VKAPI_CALL ThunkBindVertexBuffers2(VkCommandBuffer buffer, uint32_t first,
                                                   uint32_t count, const VkBuffer* buffers,
                                                   const VkDeviceSize* offsets,
                                                   const VkDeviceSize* sizes,
                                                   const VkDeviceSize* strides) {
	if (!Deferring(buffer)) {
		real_vkCmdBindVertexBuffers2(buffer, first, count, buffers, offsets, sizes, strides);
		return;
	}
	PacketWriter writer(g_ring->Reserve(
	    FixedBytes<VertexBuffersArgs>() + ArrayBytes(buffers, count) + ArrayBytes(offsets, count) +
	        ArrayBytes(sizes, count) + ArrayBytes(strides, count),
	    &ReplayBindVertexBuffers2));
	auto* a    = writer.Fixed<VertexBuffersArgs>();
	*a         = {buffer, first, count, nullptr, nullptr, nullptr, nullptr};
	a->buffers = writer.Array(buffers, count);
	a->offsets = writer.Array(offsets, count);
	a->sizes   = writer.Array(sizes, count);
	a->strides = writer.Array(strides, count);
	g_ring->Publish();
}

} // namespace

bool Enabled() {
	// On by default; KYTY_RECORD_THREAD=0 records on the GPU thread.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_RECORD_THREAD");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

void Install() {
	if (g_installed || !Enabled()) {
		return;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	// Every entry point used on the GPU thread's command buffers must be replaced; one that
	// stayed direct would record out of order. Missing optional entries stay null.
	bool complete = true;
#define KYTY_SAVE(name) \
	real_##name = d.name; \
	complete    = complete && real_##name != nullptr;
#define KYTY_SAVE_OPTIONAL(name) real_##name = d.name;
	KYTY_SAVE(vkCmdBindPipeline)
	KYTY_SAVE(vkCmdDispatch)
	KYTY_SAVE(vkCmdEndRendering)
	KYTY_SAVE(vkCmdResetQueryPool)
	KYTY_SAVE(vkCmdDraw)
	KYTY_SAVE(vkCmdSetStencilWriteMask)
	KYTY_SAVE(vkCmdSetStencilReference)
	KYTY_SAVE(vkCmdSetStencilCompareMask)
	KYTY_SAVE(vkCmdSetStencilOp)
	KYTY_SAVE(vkCmdWriteTimestamp)
	KYTY_SAVE(vkCmdFillBuffer)
	KYTY_SAVE_OPTIONAL(vkCmdDrawMeshTasksEXT)
	KYTY_SAVE(vkCmdDrawIndexed)
	KYTY_SAVE(vkCmdSetStencilTestEnable)
	KYTY_SAVE(vkCmdSetDepthWriteEnable)
	KYTY_SAVE(vkCmdSetDepthTestEnable)
	KYTY_SAVE(vkCmdSetDepthBoundsTestEnable)
	KYTY_SAVE(vkCmdSetDepthBiasEnable)
	KYTY_SAVE(vkCmdSetLineWidth)
	KYTY_SAVE(vkCmdSetFrontFace)
	KYTY_SAVE(vkCmdSetCullMode)
	KYTY_SAVE(vkCmdSetDepthCompareOp)
	KYTY_SAVE(vkCmdSetDepthBounds)
	KYTY_SAVE(vkCmdSetDepthBias)
	KYTY_SAVE_OPTIONAL(vkCmdSetCheckpointNV)
	KYTY_SAVE_OPTIONAL(vkCmdSetAttachmentFeedbackLoopEnableEXT)
	KYTY_SAVE(vkCmdEndQuery)
	KYTY_SAVE(vkCmdBeginQuery)
	KYTY_SAVE_OPTIONAL(vkCmdEndConditionalRenderingEXT)
	KYTY_SAVE(vkCmdDrawIndirect)
	KYTY_SAVE(vkCmdDrawIndexedIndirect)
	KYTY_SAVE(vkCmdDispatchIndirect)
	KYTY_SAVE(vkCmdDispatchBase)
	KYTY_SAVE(vkCmdBindIndexBuffer)
	KYTY_SAVE(vkCmdPipelineBarrier)
	KYTY_SAVE(vkCmdPipelineBarrier2)
	KYTY_SAVE(vkCmdPushDescriptorSetKHR)
	KYTY_SAVE(vkCmdPushConstants)
	KYTY_SAVE(vkCmdCopyImage)
	KYTY_SAVE(vkCmdCopyBuffer)
	KYTY_SAVE(vkCmdBeginRendering)
	KYTY_SAVE(vkCmdCopyBufferToImage)
	KYTY_SAVE(vkCmdClearColorImage)
	KYTY_SAVE(vkCmdSetViewport)
	KYTY_SAVE(vkCmdSetScissor)
	KYTY_SAVE(vkCmdCopyImageToBuffer)
	KYTY_SAVE(vkCmdClearDepthStencilImage)
	KYTY_SAVE(vkCmdBlitImage)
	KYTY_SAVE(vkCmdBindDescriptorSets)
	KYTY_SAVE(vkCmdSetViewportWithCount)
	KYTY_SAVE(vkCmdSetScissorWithCount)
	KYTY_SAVE(vkCmdSetBlendConstants)
	KYTY_SAVE(vkCmdResolveImage)
	KYTY_SAVE_OPTIONAL(vkCmdBeginConditionalRenderingEXT)
	KYTY_SAVE(vkCmdClearAttachments)
	KYTY_SAVE(vkCmdBindVertexBuffers2)
#undef KYTY_SAVE
#undef KYTY_SAVE_OPTIONAL
	if (!complete) {
		std::printf("Command recorder: a required Vulkan entry point is missing; recording on "
		            "the GPU thread\n");
		return;
	}

#define KYTY_VALUE(name, ...) \
	d.name = &ValueCommand<&real_##name __VA_OPT__(, ) __VA_ARGS__>::Thunk;
#define KYTY_VALUE_OPTIONAL(name, ...)                                                    \
	if (real_##name != nullptr) {                                                         \
		d.name = &ValueCommand<&real_##name __VA_OPT__(, ) __VA_ARGS__>::Thunk;           \
	}
	KYTY_VALUE(vkCmdBindPipeline, VkPipelineBindPoint, VkPipeline)
	KYTY_VALUE(vkCmdDispatch, uint32_t, uint32_t, uint32_t)
	KYTY_VALUE(vkCmdEndRendering)
	KYTY_VALUE(vkCmdResetQueryPool, VkQueryPool, uint32_t, uint32_t)
	KYTY_VALUE(vkCmdDraw, uint32_t, uint32_t, uint32_t, uint32_t)
	KYTY_VALUE(vkCmdSetStencilWriteMask, VkStencilFaceFlags, uint32_t)
	KYTY_VALUE(vkCmdSetStencilReference, VkStencilFaceFlags, uint32_t)
	KYTY_VALUE(vkCmdSetStencilCompareMask, VkStencilFaceFlags, uint32_t)
	KYTY_VALUE(vkCmdSetStencilOp, VkStencilFaceFlags, VkStencilOp, VkStencilOp, VkStencilOp,
	           VkCompareOp)
	KYTY_VALUE(vkCmdWriteTimestamp, VkPipelineStageFlagBits, VkQueryPool, uint32_t)
	KYTY_VALUE(vkCmdFillBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t)
	KYTY_VALUE_OPTIONAL(vkCmdDrawMeshTasksEXT, uint32_t, uint32_t, uint32_t)
	KYTY_VALUE(vkCmdDrawIndexed, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
	KYTY_VALUE(vkCmdSetStencilTestEnable, VkBool32)
	KYTY_VALUE(vkCmdSetDepthWriteEnable, VkBool32)
	KYTY_VALUE(vkCmdSetDepthTestEnable, VkBool32)
	KYTY_VALUE(vkCmdSetDepthBoundsTestEnable, VkBool32)
	KYTY_VALUE(vkCmdSetDepthBiasEnable, VkBool32)
	KYTY_VALUE(vkCmdSetLineWidth, float)
	KYTY_VALUE(vkCmdSetFrontFace, VkFrontFace)
	KYTY_VALUE(vkCmdSetCullMode, VkCullModeFlags)
	KYTY_VALUE(vkCmdSetDepthCompareOp, VkCompareOp)
	KYTY_VALUE(vkCmdSetDepthBounds, float, float)
	KYTY_VALUE(vkCmdSetDepthBias, float, float, float)
	KYTY_VALUE_OPTIONAL(vkCmdSetCheckpointNV, const void*)
	KYTY_VALUE_OPTIONAL(vkCmdSetAttachmentFeedbackLoopEnableEXT, VkImageAspectFlags)
	KYTY_VALUE(vkCmdEndQuery, VkQueryPool, uint32_t)
	KYTY_VALUE(vkCmdBeginQuery, VkQueryPool, uint32_t, VkQueryControlFlags)
	KYTY_VALUE_OPTIONAL(vkCmdEndConditionalRenderingEXT)
	KYTY_VALUE(vkCmdDrawIndirect, VkBuffer, VkDeviceSize, uint32_t, uint32_t)
	KYTY_VALUE(vkCmdDrawIndexedIndirect, VkBuffer, VkDeviceSize, uint32_t, uint32_t)
	KYTY_VALUE(vkCmdDispatchIndirect, VkBuffer, VkDeviceSize)
	KYTY_VALUE(vkCmdDispatchBase, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)
	KYTY_VALUE(vkCmdBindIndexBuffer, VkBuffer, VkDeviceSize, VkIndexType)
#undef KYTY_VALUE
#undef KYTY_VALUE_OPTIONAL
	d.vkCmdPipelineBarrier         = &ThunkPipelineBarrier;
	d.vkCmdPipelineBarrier2        = &ThunkPipelineBarrier2;
	d.vkCmdPushDescriptorSetKHR    = &ThunkPushDescriptorSet;
	d.vkCmdPushConstants           = &ThunkPushConstants;
	d.vkCmdCopyImage               = &ThunkCopyImage;
	d.vkCmdCopyBuffer              = &ThunkCopyBuffer;
	d.vkCmdBeginRendering          = &ThunkBeginRendering;
	d.vkCmdCopyBufferToImage       = &ThunkCopyBufferToImage;
	d.vkCmdClearColorImage         = &ThunkClearColorImage;
	d.vkCmdSetViewport             = &ThunkSetViewport;
	d.vkCmdSetScissor              = &ThunkSetScissor;
	d.vkCmdCopyImageToBuffer       = &ThunkCopyImageToBuffer;
	d.vkCmdClearDepthStencilImage  = &ThunkClearDepthStencilImage;
	d.vkCmdBlitImage               = &ThunkBlitImage;
	d.vkCmdBindDescriptorSets      = &ThunkBindDescriptorSets;
	d.vkCmdSetViewportWithCount    = &ThunkSetViewportWithCount;
	d.vkCmdSetScissorWithCount     = &ThunkSetScissorWithCount;
	d.vkCmdSetBlendConstants       = &ThunkSetBlendConstants;
	d.vkCmdResolveImage            = &ThunkResolveImage;
	d.vkCmdClearAttachments        = &ThunkClearAttachments;
	d.vkCmdBindVertexBuffers2      = &ThunkBindVertexBuffers2;
	if (real_vkCmdBeginConditionalRenderingEXT != nullptr) {
		d.vkCmdBeginConditionalRenderingEXT = &ThunkBeginConditionalRendering;
	}

	g_ring = new Ring();
	static std::jthread recorder([](std::stop_token stop) { g_ring->Run(stop); });
	std::atexit([] {
		recorder.request_stop();
		g_ring->Wake();
	});
	g_installed = true;
	std::printf("Command recorder: recording the GPU thread's commands on a second thread\n");
}

void Begin(vk::CommandBuffer buffer) {
	if (!g_installed) {
		return;
	}
	t_producer = true;
	g_active.store(static_cast<VkCommandBuffer>(buffer), std::memory_order_relaxed);
}

void Sync(vk::CommandBuffer buffer) {
	if (g_installed && t_producer &&
	    g_active.load(std::memory_order_relaxed) == static_cast<VkCommandBuffer>(buffer)) {
		g_ring->Drain();
	}
}

void End(vk::CommandBuffer buffer) {
	if (!g_installed || !t_producer ||
	    g_active.load(std::memory_order_relaxed) != static_cast<VkCommandBuffer>(buffer)) {
		return;
	}
	g_ring->Drain();
	g_active.store(VK_NULL_HANDLE, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::CommandRecorder
