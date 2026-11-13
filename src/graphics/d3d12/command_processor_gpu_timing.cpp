/**
 * TEMP DIAGNOSTIC (d3d12_gpu_timing): where the host GPU's time goes, by kind
 * of work. A timestamp is written into the submission wherever the kind of
 * work recorded changes (and only if commands were recorded since the last
 * one), and at each submission's start and end, so the time between two
 * submissions (the GPU idle, or running the presenter's own work) shows too.
 * Read back once the submission's fence has passed; logged every 10 seconds.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/graphics/d3d12/command_processor.h>
#include <rex/logging.h>
#include <rex/ui/d3d12/d3d12_util.h>

REXCVAR_DEFINE_BOOL(d3d12_gpu_timing, false, "GPU/D3D12",
                    "Diagnostics: log the host GPU time spent by kind of work (draws, render "
                    "target transfers, texture loads, uploads, resolves, presenting, frame "
                    "interpolation copies) every 10 seconds, from timestamp queries.");

namespace rex::graphics::d3d12 {

void D3D12CommandProcessor::GpuTimingBeginSubmission() {
  gpu_timing_slot_ = nullptr;
  if (!REXCVAR_GET(d3d12_gpu_timing)) {
    return;
  }
  if (!gpu_timing_heap_) {
    if (gpu_timing_failed_) {
      return;
    }
    const ui::d3d12::D3D12Provider& provider = GetD3D12Provider();
    ID3D12Device* device = provider.GetDevice();
    constexpr uint32_t kCount = kGpuTimingSlots * kGpuTimingPerSlot;
    D3D12_QUERY_HEAP_DESC heap_desc = {};
    heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    heap_desc.Count = kCount;
    D3D12_RESOURCE_DESC buffer_desc;
    ui::d3d12::util::FillBufferResourceDesc(buffer_desc, kCount * sizeof(uint64_t),
                                            D3D12_RESOURCE_FLAG_NONE);
    void* mapping = nullptr;
    const D3D12_RANGE read_range = {0, kCount * sizeof(uint64_t)};
    if (FAILED(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&gpu_timing_heap_))) ||
        FAILED(device->CreateCommittedResource(
            &ui::d3d12::util::kHeapPropertiesReadback, D3D12_HEAP_FLAG_NONE, &buffer_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gpu_timing_readback_))) ||
        FAILED(gpu_timing_readback_->Map(0, &read_range, &mapping)) ||
        FAILED(provider.GetDirectQueue()->GetTimestampFrequency(&gpu_timing_frequency_)) ||
        !gpu_timing_frequency_) {
      REXGPU_ERROR("d3d12_gpu_timing: failed to create the timestamp resources");
      gpu_timing_heap_.Reset();
      gpu_timing_readback_.Reset();
      gpu_timing_failed_ = true;
      return;
    }
    gpu_timing_mapping_ = static_cast<const uint64_t*>(mapping);
    gpu_timing_slots_.resize(kGpuTimingSlots);
    gpu_timing_window_start_ = rex::chrono::Clock::QueryHostTickCount();
  }
  GpuTimingHarvest();
  GpuTimingSlot& slot = gpu_timing_slots_[submission_current_ % kGpuTimingSlots];
  if (slot.pending) {
    return;  // still in flight: this submission goes untimed
  }
  slot.submission = submission_current_;
  slot.count = 0;
  gpu_timing_slot_ = &slot;
  GpuTimingStamp(gpu_zone_);
}

void D3D12CommandProcessor::GpuTimingStamp(GpuZone zone_after) {
  GpuTimingSlot& slot = *gpu_timing_slot_;
  // The last entry is kept for the submission's end.
  if (slot.count + 1 >= kGpuTimingPerSlot) {
    ++gpu_timing_overflows_;
    return;
  }
  const uint32_t index =
      uint32_t(slot.submission % kGpuTimingSlots) * kGpuTimingPerSlot + slot.count;
  deferred_command_list_.D3DEndQuery(gpu_timing_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
  slot.zone[slot.count++] = uint8_t(zone_after);
  gpu_timing_mark_ = deferred_command_list_.size();
}

void D3D12CommandProcessor::SetGpuZoneSlow(GpuZone zone) {
  GpuTimingSlot& slot = *gpu_timing_slot_;
  if (deferred_command_list_.size() == gpu_timing_mark_ && slot.count) {
    slot.zone[slot.count - 1] = uint8_t(zone);  // nothing recorded in between: relabel
    return;
  }
  GpuTimingStamp(zone);
}

void D3D12CommandProcessor::GpuTimingEndSubmission() {
  if (!gpu_timing_slot_) {
    return;
  }
  GpuTimingSlot& slot = *gpu_timing_slot_;
  gpu_timing_slot_ = nullptr;
  const uint32_t index = uint32_t(slot.submission % kGpuTimingSlots) * kGpuTimingPerSlot;
  deferred_command_list_.D3DEndQuery(gpu_timing_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                     index + slot.count);
  slot.zone[slot.count++] = uint8_t(GpuZone::kCount);
  deferred_command_list_.D3DResolveQueryData(gpu_timing_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                             index, slot.count, gpu_timing_readback_.Get(),
                                             uint64_t(index) * sizeof(uint64_t));
  slot.pending = true;
}

void D3D12CommandProcessor::GpuTimingHarvest() {
  std::vector<GpuTimingSlot*> done;
  for (GpuTimingSlot& slot : gpu_timing_slots_) {
    if (slot.pending && slot.submission <= submission_completed_) {
      done.push_back(&slot);
    }
  }
  std::sort(done.begin(), done.end(),
            [](const GpuTimingSlot* a, const GpuTimingSlot* b) { return a->submission < b->submission; });
  for (GpuTimingSlot* slot : done) {
    slot->pending = false;
    const uint64_t* t =
        gpu_timing_mapping_ + uint32_t(slot->submission % kGpuTimingSlots) * kGpuTimingPerSlot;
    for (uint32_t i = 0; i + 1 < slot->count; ++i) {
      if (t[i + 1] > t[i]) {
        gpu_timing_ticks_[slot->zone[i]] += t[i + 1] - t[i];
      }
    }
    if (slot->submission == gpu_timing_last_submission_ + 1 && gpu_timing_last_end_ &&
        t[0] > gpu_timing_last_end_) {
      gpu_timing_ticks_[size_t(GpuZone::kCount)] += t[0] - gpu_timing_last_end_;
    }
    gpu_timing_last_submission_ = slot->submission;
    gpu_timing_last_end_ = t[slot->count - 1];
    ++gpu_timing_submissions_;
  }

  const uint64_t now = rex::chrono::Clock::QueryHostTickCount();
  const uint64_t host_frequency = rex::chrono::Clock::QueryHostTickFrequency();
  if (now - gpu_timing_window_start_ < host_frequency * 10) {
    return;
  }
  const double seconds = double(now - gpu_timing_window_start_) / double(host_frequency);
  auto ms = [&](GpuZone zone) {
    return double(gpu_timing_ticks_[size_t(zone)]) * 1000.0 / double(gpu_timing_frequency_) /
           seconds;
  };
  double busy = 0.0;
  for (size_t i = 0; i < size_t(GpuZone::kCount); ++i) {
    busy += ms(GpuZone(i));
  }
  REXGPU_INFO("d3d12_gpu_timing: ms of GPU time per second: draws {:.1f}, draw setup {:.1f}, "
              "render targets {:.1f}, textures {:.1f}, uploads {:.1f}, resolves {:.1f}, swap "
              "{:.1f}, interp {:.1f}, effects {:.1f}; in submissions {:.1f}, between "
              "consecutive submissions {:.1f}; {} submissions/s, {} over the timestamp limit",
              ms(GpuZone::kDraw), ms(GpuZone::kSetup), ms(GpuZone::kRenderTargets),
              ms(GpuZone::kTextures), ms(GpuZone::kUploads), ms(GpuZone::kResolves),
              ms(GpuZone::kSwap), ms(GpuZone::kInterp), ms(GpuZone::kEffects), busy,
              ms(GpuZone::kCount), uint64_t(double(gpu_timing_submissions_) / seconds),
              gpu_timing_overflows_);
  REXGPU_INFO("d3d12_gpu_timing: shared memory uploads per second: {} calls, {} copies, {:.1f} MB, "
              "{:.1f} MB more kept as versions for frame interpolation",
              uint64_t(double(gpu_timing_upload_calls_) / seconds),
              uint64_t(double(gpu_timing_upload_copies_) / seconds),
              double(gpu_timing_upload_bytes_) / seconds / 1048576.0,
              double(gpu_timing_version_bytes_) / seconds / 1048576.0);
  gpu_timing_upload_calls_ = gpu_timing_upload_copies_ = 0;
  gpu_timing_upload_bytes_ = gpu_timing_version_bytes_ = 0;
  std::fill(std::begin(gpu_timing_ticks_), std::end(gpu_timing_ticks_), 0);
  gpu_timing_submissions_ = 0;
  gpu_timing_overflows_ = 0;
  gpu_timing_window_start_ = now;
}

}  // namespace rex::graphics::d3d12
