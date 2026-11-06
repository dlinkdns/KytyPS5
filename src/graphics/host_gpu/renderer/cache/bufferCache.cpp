#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// GPU-modified ranges come as many small scattered islands, so a readback widens to a window.
constexpr uint64_t DownloadWindowSize = 512 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Start each packed range on its own 64-byte cache line.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		Libs::Graphics::GpuWaitTag wait_tag_("buffer-download-sync");
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (m_oob_fence) {
		m_graphics.device.destroyFence(m_oob_fence, nullptr);
	}
	if (m_oob_pool) {
		m_graphics.device.destroyCommandPool(m_oob_pool, nullptr);
	}
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	// Prefetched windows leave their pages marked until a guest access; nothing is left to download.
	for (const auto& d: m_prefetched_downloads) {
		m_memory_tracker.UnmarkRegionAsGpuModified(d.begin, d.end - d.begin);
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size);
}

namespace {

// KYTY_TRACE_READBACKS=1: every ~2 s, prints which 64 KiB regions forced synchronous
// GPU-to-CPU downloads (guest CPU reads or writes of GPU-written memory).
struct ReadbackTrace {
	struct Entry {
		uint64_t reads     = 0;
		uint64_t writes    = 0;
		uint64_t downloads = 0;
		uint64_t prefetch_hits = 0;
		uint64_t bytes     = 0;
		uint64_t min_size  = UINT64_MAX;
		uint64_t max_size  = 0;
	};
	std::mutex                            mutex;
	std::map<uint64_t, Entry>             regions;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};

bool ReadbackTraceEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TRACE_READBACKS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void TraceReadback(uint64_t vaddr, uint64_t size, bool is_write, uint64_t downloaded,
                   bool prefetch_hit) {
	static ReadbackTrace trace;
	std::scoped_lock lock {trace.mutex};
	auto& entry = trace.regions[vaddr >> 16u];
	(is_write ? entry.writes : entry.reads)++;
	if (prefetch_hit) {
		entry.prefetch_hits++;
	}
	if (downloaded != 0) {
		entry.downloads++;
		entry.bytes += downloaded;
	}
	entry.min_size = std::min(entry.min_size, size);
	entry.max_size = std::max(entry.max_size, size);
	const auto now = std::chrono::steady_clock::now();
	const double seconds = std::chrono::duration<double>(now - trace.last).count();
	if (seconds < 2.0) {
		return;
	}
	std::vector<std::pair<uint64_t, ReadbackTrace::Entry>> top(trace.regions.begin(),
	                                                           trace.regions.end());
	std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
		return a.second.reads + a.second.writes > b.second.reads + b.second.writes;
	});
	std::printf("--- GPU readbacks over %.1f s (%zu regions) ---\n", seconds, top.size());
	for (size_t i = 0; i < top.size() && i < 12; i++) {
		const auto& [region, e] = top[i];
		std::printf("  0x%010llx  reads %6.1f/s  writes %6.1f/s  downloads %6.1f/s"
		            "  prefetch-hits %6.1f/s  %8.1f KB/s  req %llu..%llu B\n",
		            static_cast<unsigned long long>(region << 16u), e.reads / seconds,
		            e.writes / seconds, e.downloads / seconds, e.prefetch_hits / seconds,
		            e.bytes / seconds / 1024.0,
		            static_cast<unsigned long long>(e.min_size),
		            static_cast<unsigned long long>(e.max_size));
	}
	std::fflush(stdout);
	trace.regions.clear();
	trace.last = now;
}

} // namespace

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		if (IsRegionGpuModified(vaddr, size)) {
			const auto start =
			    std::max(Common::AlignDown(vaddr, DownloadWindowSize), buffer.CpuAddress());
			const auto end = std::min(std::max(start + DownloadWindowSize, vaddr + size),
			                          buffer.CpuAddress() + buffer.Size());
			// A prefetch took its bytes out of the dirty ranges; they reach guest memory when it
			// publishes, which is usually already done.
			const bool prefetched = WaitPrefetchedDownloads(start, end - start);
			bool       downloaded = false;
			if (!DownloadOutOfBand(buffer, start, end - start, downloaded)) {
				downloaded = DownloadBufferMemory<false>(buffer, start, end - start);
			}
			// With no GPU-written bytes left in the window, its pages may lose GPU ownership.
			if (downloaded || !m_gpu_modified_ranges.Intersects(start, end - start)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(start, end - start);
			}
			NoteReadbackFault(vaddr, prefetched && !downloaded);
			if (ReadbackTraceEnabled()) {
				TraceReadback(vaddr, size, is_write, downloaded ? end - start : 0,
				              prefetched && !downloaded);
			}
		} else if (ReadbackTraceEnabled()) {
			TraceReadback(vaddr, size, is_write, 0, false);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size);
		}
	});
}

namespace {

bool ReadbackPrefetchEnabled() {
	// On by default; KYTY_PREFETCH_READBACKS=0 restores purely on-demand downloads.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PREFETCH_READBACKS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

constexpr uint32_t ReadbackCreditOnMiss = 4;
constexpr uint32_t ReadbackCreditOnHit  = 2;
constexpr uint32_t ReadbackCreditMax    = 16;
constexpr size_t   ReadbackHotLimit     = 64;

} // namespace

void BufferCache::NoteReadbackFault(uint64_t vaddr, bool prefetch_hit) {
	if (!ReadbackPrefetchEnabled()) {
		return;
	}
	const auto key = Common::AlignDown(vaddr, DownloadWindowSize);
	auto       it  = m_readback_hot.find(key);
	if (it == m_readback_hot.end()) {
		if (m_readback_hot.size() >= ReadbackHotLimit) {
			return;
		}
		it = m_readback_hot.emplace(key, 0u).first;
	}
	it->second =
	    std::min(ReadbackCreditMax, it->second + (prefetch_hit ? ReadbackCreditOnHit
	                                                           : ReadbackCreditOnMiss));
}

bool BufferCache::HasPendingPrefetch(uint64_t vaddr, uint64_t size) const {
	const auto end = vaddr + size;
	return std::any_of(m_prefetched_downloads.begin(), m_prefetched_downloads.end(),
	                   [&](const PrefetchedDownload& d) { return d.begin < end && vaddr < d.end; });
}

bool BufferCache::WaitPrefetchedDownloads(uint64_t vaddr, uint64_t size) {
	if (m_prefetched_downloads.empty()) {
		return false;
	}
	const auto end  = vaddr + size;
	uint64_t   tick = 0;
	for (const auto& d: m_prefetched_downloads) {
		if (d.begin < end && vaddr < d.end) {
			tick = std::max(tick, d.tick);
		}
	}
	if (tick == 0) {
		return false;
	}
	{
		Libs::Graphics::GpuWaitTag wait_tag_("buffer-prefetch-wait");
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
	}
	// The caller clears the waited range; entries reaching outside it stay (already published, so a
	// later wait on them is free) until an access covers them.
	std::erase_if(m_prefetched_downloads, [&](const PrefetchedDownload& d) {
		return d.tick <= tick && vaddr <= d.begin && d.end <= end;
	});
	return true;
}

void BufferCache::PrefetchReadbacks() {
	// Hot windows only gain GPU-written bytes through ObtainBuffer; skip when there was none.
	if (m_readback_hot.empty() || m_gpu_write_epoch == m_prefetch_epoch ||
	    !ReadbackPrefetchEnabled()) {
		return;
	}
	m_prefetch_epoch = m_gpu_write_epoch;
	for (auto it = m_readback_hot.begin(); it != m_readback_hot.end();) {
		const auto  key   = it->first;
		const auto* owner = m_page_table.Find(key >> PageTable::kPageBits);
		BufferId    id {};
		if (owner != nullptr && *owner) {
			id = *owner;
		} else {
			// The window may begin before the buffer that faulted; look for it inside the window.
			const auto next = m_buffers.lower_bound(key);
			if (next != m_buffers.end() && next->first < key + DownloadWindowSize) {
				id = next->second;
			}
		}
		if (IsBufferInvalid(id)) {
			it = m_readback_hot.erase(it);
			continue;
		}
		auto&      buffer = m_slot_buffers[id];
		const auto start  = std::max(key, buffer.CpuAddress());
		const auto end    = std::min(key + DownloadWindowSize, buffer.CpuAddress() + buffer.Size());
		if (start >= end || !m_gpu_modified_ranges.Intersects(start, end - start)) {
			++it;
			continue;
		}
		if (DownloadBufferMemory<true>(buffer, start, end - start)) {
			// Publishes run in queue order, so the new entry also covers older ones for the window.
			std::erase_if(m_prefetched_downloads, [&](const PrefetchedDownload& d) {
				return start <= d.begin && d.end <= end;
			});
			m_prefetched_downloads.push_back({start, end, m_scheduler.CurrentTick()});
		}
		if (--it->second == 0) {
			it = m_readback_hot.erase(it);
		} else {
			++it;
		}
	}
}

void BufferCache::NoteBatchWrite(uint64_t vaddr, uint64_t size) {
	const auto tick = m_scheduler.CurrentTick();
	if (m_batch_writes_tick != tick) {
		m_batch_writes      = {};
		m_batch_writes_tick = tick;
	}
	m_batch_writes.Add(vaddr, size);
}

bool BufferCache::HasBatchWrite(uint64_t vaddr, uint64_t size) {
	return m_batch_writes_tick == m_scheduler.CurrentTick() &&
	       m_batch_writes.Intersects(vaddr, size);
}

namespace {

bool OutOfBandDownloadsEnabled() {
	// On by default; KYTY_NO_OOB_DOWNLOAD=1 always flushes the batch being recorded.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NO_OOB_DOWNLOAD");
		return value == nullptr || value[0] != '1';
	}();
	return enabled;
}

constexpr uint64_t OutOfBandDownloadSize = 8 * MiB;

} // namespace

bool BufferCache::DownloadOutOfBand(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                    bool& downloaded) {
	downloaded = false;
	if (!OutOfBandDownloadsEnabled() || HasBatchWrite(vaddr, size)) {
		return false;
	}
	// Plan the copies without changing any state yet.
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    total_size += Common::AlignUp(end - start, 64);
		    });
	    });
	if (copies.empty()) {
		return false;
	}
	if (total_size > OutOfBandDownloadSize) {
		return false;
	}
	const auto& device = m_graphics.device;
	if (!m_oob_pool) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.queueFamilyIndex = m_graphics.queue_family;
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
		                  vk::CommandPoolCreateFlagBits::eTransient;
		EXIT_NOT_IMPLEMENTED(device.createCommandPool(&pool_info, nullptr, &m_oob_pool) !=
		                     vk::Result::eSuccess);
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = m_oob_pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		EXIT_NOT_IMPLEMENTED(device.allocateCommandBuffers(&allocate, &m_oob_command) !=
		                     vk::Result::eSuccess);
		vk::FenceCreateInfo fence_info {};
		EXIT_NOT_IMPLEMENTED(device.createFence(&fence_info, nullptr, &m_oob_fence) !=
		                     vk::Result::eSuccess);
		m_oob_download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download,
		                                          0, vk::BufferUsageFlagBits::eTransferDst,
		                                          OutOfBandDownloadSize);
		SetVulkanObjectNameF(device, m_oob_download->Handle(), "Kyty.OutOfBandDownload");
	}

	// Commit: the bytes leave the dirty set now and reach guest memory before returning.
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	EXIT_NOT_IMPLEMENTED(m_oob_command.reset({}) != vk::Result::eSuccess);
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	EXIT_NOT_IMPLEMENTED(m_oob_command.begin(&begin) != vk::Result::eSuccess);
	// Earlier submissions on this queue are in the first scope of the barrier.
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	m_oob_command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                              vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before,
	                              0, nullptr);
	m_oob_command.copyBuffer(buffer.Handle(), m_oob_download->Handle(),
	                         static_cast<uint32_t>(copies.size()), copies.data());
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = m_oob_download->Handle();
	after.offset        = 0;
	after.size          = total_size;
	m_oob_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                              vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &after, 0,
	                              nullptr);
	EXIT_NOT_IMPLEMENTED(m_oob_command.end() != vk::Result::eSuccess);

	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &m_oob_command;
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		const auto        result = m_graphics.queue.submit(1, &submit, m_oob_fence);
		if (result != vk::Result::eSuccess) {
			EXIT("out-of-band download submit failed: %s\n", vk::to_string(result).c_str());
		}
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);
	{
		Libs::Graphics::GpuWaitTag wait_tag_("buffer-download-oob");
		const auto start_time = std::chrono::steady_clock::now();
		EXIT_NOT_IMPLEMENTED(device.waitForFences(1, &m_oob_fence, VK_TRUE, UINT64_MAX) !=
		                     vk::Result::eSuccess);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuWaits);
		Common::DebugCounters::Add(
		    Common::DebugCounters::Counter::GpuWaitNs,
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - start_time)
		                              .count()));
	}
	EXIT_NOT_IMPLEMENTED(device.resetFences(1, &m_oob_fence) != vk::Result::eSuccess);
	m_oob_download->Invalidate(0, total_size);
	const auto* mapped = m_oob_download->Mapped().data();
	for (const auto& copy: copies) {
		Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
		                                      mapped + copy.dstOffset, copy.size);
	}
	downloaded = true;
	return true;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	// Joined contents are copied in this batch.
	NoteBatchWrite(overlap.begin, overlap.end - overlap.begin);
	m_slot_buffers[id].created_epoch = m_command_epoch;
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

namespace {

bool DeferBindUploadsEnabled() {
	// On by default; KYTY_UPLOAD_ON_BIND=1 uploads every CPU-modified page a draw binds.
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_UPLOAD_ON_BIND");
		return value == nullptr || value[0] != '1';
	}();
	return enabled;
}

} // namespace

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	if (!is_written && !is_texel_buffer && buffer.created_epoch < m_command_epoch &&
	    DeferBindUploadsEnabled()) {
		// Pages the guest wrote after this command buffer started hold data for later work
		// (the guest writes before it submits); they upload when the next command buffer
		// starts. Uploading them here ended the render pass for a few pages per draw.
		std::vector<std::pair<uint64_t, uint64_t>> pending;
		{
			std::lock_guard lock(m_bda_sync_mutex);
			m_bda_sync.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) {
				pending.emplace_back(start, end);
			});
		}
		if (!pending.empty()) {
			const auto end    = vaddr + size;
			auto       cursor = vaddr;
			for (auto [start, finish]: pending) {
				start  = std::max(start, vaddr);
				finish = std::min(finish, end);
				if (start > cursor) {
					(void)SynchronizeBuffer(buffer, cursor, start - cursor, false, false);
				}
				cursor = std::max(cursor, finish);
			}
			if (cursor < end) {
				(void)SynchronizeBuffer(buffer, cursor, end - cursor, false, false);
			}
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		m_gpu_write_epoch++;
		NoteBatchWrite(vaddr, size);
		m_gpu_modified_ranges.Add(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		NoteCpuWrite(vaddr, size);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		NoteCpuWrite(dst_vaddr, size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

void BufferCache::NoteCpuWrite(uint64_t vaddr, uint64_t size) {
	// Without sticky pages every tracked page is write-protected, so the store faulted already.
	if (!RegionManager::StickyEnabled()) {
		return;
	}
	if (!GuestRange {vaddr, size}.Valid() || m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		// GPU-owned pages are protected, so the store itself faulted.
		return;
	}
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	QueueBdaSync(vaddr, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size) || HasPendingPrefetch(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty && WaitPrefetchedDownloads(buffer.CpuAddress(), buffer.Size()) &&
		    !m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			// A prefetch already published every GPU-written byte.
			m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
			dirty = false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	Libs::Graphics::GpuWaitTag wait_tag_("buffer-gc-download");
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	m_bda_sync.Add(begin, end - begin);
}

RangeSet BufferCache::TakeBdaSync() {
	std::lock_guard lock(m_bda_sync_mutex);
	return std::exchange(m_bda_sync, {});
}

void BufferCache::SynchronizeBuffersInRanges(
    std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	// Group the ranges by buffer, keeping the buffers in address order.
	std::map<uint64_t, std::pair<BufferId, std::vector<std::pair<uint64_t, uint64_t>>>> groups;
	for (const auto& [vaddr, size]: ranges) {
		const auto end = vaddr + size;
		auto       it  = m_buffers.upper_bound(vaddr);
		if (it != m_buffers.begin()) {
			--it;
		}
		for (; it != m_buffers.end() && it->first < end; ++it) {
			auto&      buffer = m_slot_buffers[it->second];
			const auto start  = std::max(buffer.CpuAddress(), vaddr);
			const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
			if (start < finish) {
				auto& group = groups[it->first];
				group.first = it->second;
				group.second.emplace_back(start, finish - start);
			}
		}
	}
	std::vector<vk::BufferCopy> copies;
	for (auto& [address, group]: groups) {
		(void)address;
		auto& buffer = m_slot_buffers[group.first];
		copies.clear();
		uint64_t total_size = 0;
		for (const auto& [start, size]: group.second) {
			m_memory_tracker.ForEachUploadRange(
			    start, size, false,
			    [&](uint64_t range_address, uint64_t bytes) noexcept {
				    copies.emplace_back(total_size, buffer.Offset(range_address), bytes);
				    total_size += bytes;
			    },
			    []() noexcept {});
		}
		const auto source = UploadCopies(buffer, copies, total_size);
		if (!source) {
			continue;
		}
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto              native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
