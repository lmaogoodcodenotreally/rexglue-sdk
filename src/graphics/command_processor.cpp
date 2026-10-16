/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <bit>
#include <optional>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string_view>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/perf/counter.h>
#include <rex/chrono/clock.h>
#include <rex/graphics/command_processor.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/graphics_system.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/sampler_info.h>
#include <rex/graphics/shared_memory.h>
#include <rex/graphics/xenos.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory.h>
#include <rex/memory/ring_buffer.h>
#include <rex/stream.h>
#include <rex/system/kernel_state.h>
#include <rex/system/user_module.h>

REXCVAR_DEFINE_BOOL(vsync, true, "GPU", "Enable vertical sync");

REXCVAR_DEFINE_BOOL(frame_interp, false, "GPU",
                    "Frame interpolation: replay each guest frame's commands with the shader "
                    "constants halfway to the previous frame and present that between real "
                    "frames (2x the displayed frame rate; the guest's own tick is unchanged).");

REXCVAR_DEFINE_BOOL(clear_memory_page_state, true, "GPU",
                    "Refresh page-valid state from GPU-written memory at frame end. "
                    "Disable for minor CPU overhead reduction, but may break memory coherency.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(occlusion_query_enable, true, "GPU", "Enable host occlusion query handling")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_STRING(readback_resolve, "none", "GPU",
                      "Controls CPU readback of render-to-texture resolve results.\n"
                      " none: Disable readback (default)\n"
                      " fast: Read previous frame (delayed, copy every frame)\n"
                      " some: Read previous frame (delayed, copy on cache miss)\n"
                      " full: Immediate sync readback (accurate but stalls)")
    .allowed({"none", "fast", "some", "full"})
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_resolve_half_pixel_offset, false, "GPU",
                    "When draw resolution scaling is active, sample from the center of each "
                    "scaled block during resolve readback downscale")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_memexport, true, "GPU",
                    "Enable CPU readback of shader memexport writes for guest memory "
                    "coherency (can reduce correctness issues, but may add GPU/CPU sync cost)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(readback_memexport_fast, true, "GPU",
                    "Use fast double-buffered memexport readback when possible, with "
                    "automatic fallback to full synchronous readback")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(query_occlusion_fake_sample_count, 1000, "GPU",
                     "Fake sample count for occlusion queries")
    .range(1, 100000)
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(async_shader_compilation, true, "GPU",
                    "Compile shaders and create pipelines asynchronously in background "
                    "threads. This reduces stutter but may cause brief visual artifacts while "
                    "pipelines are being prepared.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::graphics {

using namespace rex::graphics::xenos;

namespace {

ReadbackResolveMode ParseReadbackResolveMode(std::string_view value) {
  if (value == "fast") {
    return ReadbackResolveMode::kFast;
  }
  if (value == "some") {
    return ReadbackResolveMode::kSome;
  }
  if (value == "full") {
    return ReadbackResolveMode::kFull;
  }
  return ReadbackResolveMode::kDisabled;
}

}  // namespace

CommandProcessor::CommandProcessor(GraphicsSystem* graphics_system,
                                   system::KernelState* kernel_state)
    : memory_(graphics_system->memory()),
      kernel_state_(kernel_state),
      graphics_system_(graphics_system),
      register_file_(graphics_system_->register_file()),
      worker_running_(true),
      write_ptr_index_event_(rex::thread::Event::CreateAutoResetEvent(false)),
      write_ptr_index_(0) {
  assert_not_null(write_ptr_index_event_);
}

CommandProcessor::~CommandProcessor() = default;

bool CommandProcessor::Initialize() {
  // Initialize the gamma ramps to their default (linear) values - taken from
  // what games set when starting with the sRGB (return value 1)
  // VdGetCurrentDisplayGamma.
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t value = i * 0x3FF / 0xFF;
    reg::DC_LUT_30_COLOR& gamma_ramp_entry = gamma_ramp_256_entry_table_[i];
    gamma_ramp_entry.color_10_blue = value;
    gamma_ramp_entry.color_10_green = value;
    gamma_ramp_entry.color_10_red = value;
  }
  for (uint32_t i = 0; i < 128; ++i) {
    reg::DC_LUT_PWL_DATA gamma_ramp_entry = {};
    gamma_ramp_entry.base = (i * 0xFFFF / 0x7F) & ~UINT32_C(0x3F);
    gamma_ramp_entry.delta = i < 0x7F ? 0x200 : 0;
    for (uint32_t j = 0; j < 3; ++j) {
      gamma_ramp_pwl_rgb_[i][j] = gamma_ramp_entry;
    }
  }

  worker_running_ = true;
  worker_thread_ = system::object_ref<system::XHostThread>(
      new system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
        WorkerThreadMain();
        return 0;
      }));
  worker_thread_->set_name("GPU Commands");
  worker_thread_->Create();

  return true;
}

void CommandProcessor::Shutdown() {
  worker_running_ = false;
  write_ptr_index_event_->Set();
  worker_thread_->Wait(0, 0, 0, nullptr);
  worker_thread_.reset();
}

void CommandProcessor::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                               uint32_t title_id, bool blocking) {}

void CommandProcessor::CallInThread(std::function<void()> fn) {
  if (pending_fns_.empty() && system::XThread::IsInThread(worker_thread_.get())) {
    fn();
  } else {
    pending_fns_.push(std::move(fn));
  }
}

void CommandProcessor::ClearCaches() {}

void CommandProcessor::InvalidateGpuMemory() {}

ReadbackResolveMode CommandProcessor::GetReadbackResolveMode(
    bool legacy_readback_resolve_enabled) const {
  ReadbackResolveMode shared_mode = ParseReadbackResolveMode(REXCVAR_GET(readback_resolve));
  bool shared_mode_overrides_legacy = shared_mode != ReadbackResolveMode::kDisabled ||
                                      rex::cvar::HasNonDefaultValue("readback_resolve");
  if (shared_mode_overrides_legacy) {
    return shared_mode;
  }
  return legacy_readback_resolve_enabled ? ReadbackResolveMode::kFast
                                         : ReadbackResolveMode::kDisabled;
}

bool CommandProcessor::IsReadbackMemexportEnabled(bool legacy_backend_flag) const {
  if (legacy_readback_memexport_cvar_name_ &&
      rex::cvar::HasNonDefaultValue(legacy_readback_memexport_cvar_name_)) {
    return legacy_backend_flag;
  }
  return REXCVAR_GET(readback_memexport);
}

void CommandProcessor::SetDesiredSwapPostEffect(SwapPostEffect swap_post_effect) {
  if (swap_post_effect_desired_ == swap_post_effect) {
    return;
  }
  swap_post_effect_desired_ = swap_post_effect;
  CallInThread([this, swap_post_effect]() { swap_post_effect_actual_ = swap_post_effect; });
}

void CommandProcessor::WorkerThreadMain() {
  if (!SetupContext()) {
    rex::FatalError("Unable to setup command processor internal state");
    return;
  }

  while (worker_running_) {
    while (!pending_fns_.empty()) {
      auto fn = std::move(pending_fns_.front());
      pending_fns_.pop();
      fn();
    }

    uint32_t write_ptr_index = write_ptr_index_.load();
    if (write_ptr_index == 0xBAADF00D || read_ptr_index_ == write_ptr_index) {
      SCOPE_profile_cpu_i("gpu", "rex::graphics::CommandProcessor::Stall");
      // We've run out of commands to execute.
      // We spin here waiting for new ones, as the overhead of waiting on our
      // event is too high.
      PrepareForWait();
      const uint64_t wait_start = rex::chrono::Clock::QueryHostTickCount();  // TEMP DIAGNOSTIC
      uint32_t loop_count = 0;
      do {
        // If we spin around too much, revert to a "low-power" state.
        if (loop_count > 500) {
          const int wait_time_ms = interp_held_pending_ ? 1 : 5;
          rex::thread::Wait(write_ptr_index_event_.get(), true,
                            std::chrono::milliseconds(wait_time_ms));
        }

        rex::thread::MaybeYield();
        loop_count++;
        write_ptr_index = write_ptr_index_.load();
        ServiceHeldSwap(false);
      } while (worker_running_ && pending_fns_.empty() &&
               (write_ptr_index == 0xBAADF00D || read_ptr_index_ == write_ptr_index));
      interp_wait_ticks_ += rex::chrono::Clock::QueryHostTickCount() - wait_start;
      ReturnFromWait();
      if (!worker_running_ || !pending_fns_.empty()) {
        continue;
      }
    }
    assert_true(read_ptr_index_ != write_ptr_index);

    // Execute. Note that we handle wraparound transparently.
    read_ptr_index_ = ExecutePrimaryBuffer(read_ptr_index_, write_ptr_index);

    // TODO(benvanik): use reader->Read_update_freq_ and only issue after moving
    //     that many indices.
    if (read_ptr_writeback_ptr_) {
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(read_ptr_writeback_ptr_),
                                       read_ptr_index_);
    }

    // FIXME: We're supposed to process the WAIT_UNTIL register at this point,
    // but no games seem to actually use it.
  }

  ShutdownContext();
}

void CommandProcessor::Pause() {
  if (paused_) {
    return;
  }
  paused_ = true;

  thread::Fence fence;
  CallInThread([&fence]() {
    fence.Signal();
    thread::Thread::GetCurrentThread()->Suspend();
  });

  fence.Wait();
}

void CommandProcessor::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;

  worker_thread_->thread()->Resume();
}

bool CommandProcessor::Save(::rex::stream::ByteStream* stream) {
  assert_true(paused_);

  stream->Write<uint32_t>(primary_buffer_ptr_);
  stream->Write<uint32_t>(primary_buffer_size_);
  stream->Write<uint32_t>(read_ptr_index_);
  stream->Write<uint32_t>(read_ptr_update_freq_);
  stream->Write<uint32_t>(read_ptr_writeback_ptr_);
  stream->Write<uint32_t>(write_ptr_index_.load());

  return true;
}

bool CommandProcessor::Restore(::rex::stream::ByteStream* stream) {
  assert_true(paused_);

  primary_buffer_ptr_ = stream->Read<uint32_t>();
  primary_buffer_size_ = stream->Read<uint32_t>();
  read_ptr_index_ = stream->Read<uint32_t>();
  read_ptr_update_freq_ = stream->Read<uint32_t>();
  read_ptr_writeback_ptr_ = stream->Read<uint32_t>();
  write_ptr_index_.store(stream->Read<uint32_t>());

  return true;
}

bool CommandProcessor::SetupContext() {
  return true;
}

void CommandProcessor::ShutdownContext() {}

void CommandProcessor::InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) {
  read_ptr_index_ = 0;
  primary_buffer_ptr_ = ptr;
  primary_buffer_size_ = uint32_t(1) << (size_log2 + 3);
}

void CommandProcessor::EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) {
  // CP_RB_RPTR_ADDR Ring Buffer Read Pointer Address 0x70C
  // ptr = RB_RPTR_ADDR, pointer to write back the address to.
  read_ptr_writeback_ptr_ = ptr;
  // CP_RB_CNTL Ring Buffer Control 0x704
  // block_size = RB_BLKSZ, log2 of number of quadwords read between updates of
  //              the read pointer.
  read_ptr_update_freq_ = uint32_t(1) << block_size_log2 >> 2;
}

void CommandProcessor::UpdateWritePointer(uint32_t value) {
  write_ptr_index_ = value;
  write_ptr_index_event_->Set();
}

uint32_t CommandProcessor::ReadRegisterValue(uint32_t index) const {
  if (index < RegisterFile::kRegisterCount) {
    return register_file_->values[index];
  }
  auto it = extended_register_values_.find(index);
  return it != extended_register_values_.end() ? it->second : 0;
}

bool CommandProcessor::WriteRegisterHasSideEffects(uint32_t index) {
  // Scratch register writes push the value to guest memory on every write (the
  // guest may have modified the target since the previous writeback).
  if (index >= XE_GPU_REG_SCRATCH_REG0 && index <= XE_GPU_REG_SCRATCH_REG7) {
    return true;
  }
  // Gamma ramp LUT writes advance internal state machines (component/index
  // auto-increment) even when the register value itself is unchanged.
  switch (index) {
    case XE_GPU_REG_DC_LUT_RW_INDEX:
    case XE_GPU_REG_DC_LUT_SEQ_COLOR:
    case XE_GPU_REG_DC_LUT_PWL_DATA:
    case XE_GPU_REG_DC_LUT_30_COLOR:
      return true;
    default:
      return false;
  }
}

void CommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  RegisterFile& regs = *register_file_;
  if (index >= RegisterFile::kRegisterCount) {
    auto [it, inserted] = extended_register_values_.insert_or_assign(index, value);
    (void)it;
    if (inserted) {
      REXGPU_WARN(
          "CommandProcessor::WriteRegister index out of bounds: {} (stored as extended register)",
          index);
    }
    return;
  }

  // Volatile for the WAIT_REG_MEM loop.
  const_cast<volatile uint32_t&>(regs.values[index]) = value;
  if (!regs.GetRegisterInfo(index)) {
    REXGPU_DEBUG("GPU: Write to unknown register ({:04X} = {:08X})", index, value);
  }

  // Scratch register writeback.
  if (index >= XE_GPU_REG_SCRATCH_REG0 && index <= XE_GPU_REG_SCRATCH_REG7) {
    uint32_t scratch_reg = index - XE_GPU_REG_SCRATCH_REG0;
    if ((1 << scratch_reg) & regs.values[XE_GPU_REG_SCRATCH_UMSK]) {
      // Enabled - write to address.
      uint32_t scratch_addr = regs.values[XE_GPU_REG_SCRATCH_ADDR];
      uint32_t mem_addr = scratch_addr + (scratch_reg * 4);
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(mem_addr), value);
    }
  } else {
    switch (index) {
      // If this is a COHER register, set the dirty flag.
      // This will block the command processor the next time it WAIT_REG_MEMs
      // and allow us to synchronize the memory.
      case XE_GPU_REG_COHER_STATUS_HOST: {
        const_cast<volatile uint32_t&>(regs.values[index]) |= UINT32_C(0x80000000);
      } break;

      case XE_GPU_REG_DC_LUT_RW_INDEX: {
        // Reset the sequential read / write component index (see the M56
        // DC_LUT_SEQ_COLOR documentation).
        gamma_ramp_rw_component_ = 0;
      } break;

      case XE_GPU_REG_DC_LUT_SEQ_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // DC_LUT_SEQ_COLOR is in the red, green, blue order, but the write
        // enable mask is blue, green, red.
        bool write_gamma_ramp_component = (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
                                           (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          // Bits 0:5 are hardwired to zero.
          uint32_t gamma_ramp_seq_color = regs.Get<reg::DC_LUT_SEQ_COLOR>().seq_color >> 6;
          switch (gamma_ramp_rw_component_) {
            case 0:
              gamma_ramp_entry.color_10_red = gamma_ramp_seq_color;
              break;
            case 1:
              gamma_ramp_entry.color_10_green = gamma_ramp_seq_color;
              break;
            case 2:
              gamma_ramp_entry.color_10_blue = gamma_ramp_seq_color;
              break;
          }
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          ++new_gamma_ramp_rw_index.rw_index;
          WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                        rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_PWL_DATA: {
        // Should be in the PWL writing mode.
        assert_not_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        // Bit 7 of the index is ignored for PWL.
        uint32_t gamma_ramp_rw_index_pwl = gamma_ramp_rw_index.rw_index & 0x7F;
        // DC_LUT_PWL_DATA is likely in the red, green, blue order because
        // DC_LUT_SEQ_COLOR is, but the write enable mask is blue, green, red.
        bool write_gamma_ramp_component = (regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] &
                                           (UINT32_C(1) << (2 - gamma_ramp_rw_component_))) != 0;
        if (write_gamma_ramp_component) {
          reg::DC_LUT_PWL_DATA& gamma_ramp_entry =
              gamma_ramp_pwl_rgb_[gamma_ramp_rw_index_pwl][gamma_ramp_rw_component_];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_PWL_DATA>();
          // Bits 0:5 are hardwired to zero.
          gamma_ramp_entry.base = gamma_ramp_value.base & ~UINT32_C(0x3F);
          gamma_ramp_entry.delta = gamma_ramp_value.delta & ~UINT32_C(0x3F);
        }
        if (++gamma_ramp_rw_component_ >= 3) {
          gamma_ramp_rw_component_ = 0;
          reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
          // TODO(Triang3l): Should this increase beyond 7 bits for PWL?
          // Direct3D 9 explicitly sets rw_index to 0x80 after writing the last
          // PWL entry. However, the DC_LUT_RW_INDEX documentation says that for
          // PWL, the bit 7 is ignored.
          new_gamma_ramp_rw_index.rw_index = (gamma_ramp_rw_index.rw_index & ~UINT32_C(0x7F)) |
                                             ((gamma_ramp_rw_index_pwl + 1) & 0x7F);
          WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                        rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        }
        if (write_gamma_ramp_component) {
          OnGammaRampPWLValueWritten();
        }
      } break;

      case XE_GPU_REG_DC_LUT_30_COLOR: {
        // Should be in the 256-entry table writing mode.
        assert_zero(regs[XE_GPU_REG_DC_LUT_RW_MODE] & 0b1);
        auto gamma_ramp_rw_index = regs.Get<reg::DC_LUT_RW_INDEX>();
        uint32_t gamma_ramp_write_enable_mask = regs[XE_GPU_REG_DC_LUT_WRITE_EN_MASK] & 0b111;
        if (gamma_ramp_write_enable_mask) {
          reg::DC_LUT_30_COLOR& gamma_ramp_entry =
              gamma_ramp_256_entry_table_[gamma_ramp_rw_index.rw_index];
          auto gamma_ramp_value = regs.Get<reg::DC_LUT_30_COLOR>();
          if (gamma_ramp_write_enable_mask & 0b001) {
            gamma_ramp_entry.color_10_blue = gamma_ramp_value.color_10_blue;
          }
          if (gamma_ramp_write_enable_mask & 0b010) {
            gamma_ramp_entry.color_10_green = gamma_ramp_value.color_10_green;
          }
          if (gamma_ramp_write_enable_mask & 0b100) {
            gamma_ramp_entry.color_10_red = gamma_ramp_value.color_10_red;
          }
        }
        // TODO(Triang3l): Should this reset the component write index? If this
        // increase is assumed to behave like a full DC_LUT_RW_INDEX write, it
        // probably should. Currently this also calls WriteRegister for
        // DC_LUT_RW_INDEX, which resets gamma_ramp_rw_component_ as well.
        gamma_ramp_rw_component_ = 0;
        reg::DC_LUT_RW_INDEX new_gamma_ramp_rw_index = gamma_ramp_rw_index;
        ++new_gamma_ramp_rw_index.rw_index;
        WriteRegister(XE_GPU_REG_DC_LUT_RW_INDEX,
                      rex::memory::Reinterpret<uint32_t>(new_gamma_ramp_rw_index));
        if (gamma_ramp_write_enable_mask) {
          OnGammaRamp256EntryTableValueWritten();
        }
      } break;
    }
  }
}

void CommandProcessor::WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  for (uint32_t i = 0; i < num_registers; ++i) {
    uint32_t data = memory::load_and_swap<uint32_t>(base + i);
    WriteRegister(start_index + i, data);
  }
}

void CommandProcessor::WriteRegisterRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                                  uint32_t num_registers) {
  if (!num_registers) {
    return;
  }
  memory::RingBuffer::ReadRange range = ring->BeginRead(size_t(num_registers) * sizeof(uint32_t));
  if (range.first_length != 0) {
    uint32_t first_count = uint32_t(range.first_length / sizeof(uint32_t));
    WriteRegistersFromMem(base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.first)),
                          first_count);
    base += first_count;
  }
  if (range.second_length != 0) {
    WriteRegistersFromMem(base, reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(range.second)),
                          uint32_t(range.second_length / sizeof(uint32_t)));
  }
  ring->EndRead(range);
}

void CommandProcessor::WriteALURangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                             uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4000, num_registers);
}

void CommandProcessor::WriteFetchRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                               uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4800, num_registers);
}

void CommandProcessor::WriteBoolRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                              uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4900, num_registers);
}

void CommandProcessor::WriteLoopRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                              uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x4908, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromRing(memory::RingBuffer* ring, uint32_t base,
                                                   uint32_t num_registers) {
  WriteRegisterRangeFromRing(ring, base + 0x2000, num_registers);
}

void CommandProcessor::WriteALURangeFromMem(uint32_t start_index, uint32_t* base,
                                            uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4000, base, num_registers);
}

void CommandProcessor::WriteFetchRangeFromMem(uint32_t start_index, uint32_t* base,
                                              uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4800, base, num_registers);
}

void CommandProcessor::WriteBoolRangeFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4900, base, num_registers);
}

void CommandProcessor::WriteLoopRangeFromMem(uint32_t start_index, uint32_t* base,
                                             uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x4908, base, num_registers);
}

void CommandProcessor::WriteREGISTERSRangeFromMem(uint32_t start_index, uint32_t* base,
                                                  uint32_t num_registers) {
  WriteRegistersFromMem(start_index + 0x2000, base, num_registers);
}

void CommandProcessor::MakeCoherent() {
  SCOPE_profile_cpu_f("gpu");

  // Status host often has 0x01000000 or 0x03000000.
  // This is likely toggling VC (vertex cache) or TC (texture cache).
  // Or, it also has a direction in here maybe - there is probably
  // some way to check for dest coherency (what all the COHER_DEST_BASE_*
  // registers are for).
  // Best docs I've found on this are here:
  // https://web.archive.org/web/20160711162346/https://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/10/R6xx_R7xx_3D.pdf
  // https://cgit.freedesktop.org/xorg/driver/xf86-video-radeonhd/tree/src/r6xx_accel.c?id=3f8b6eccd9dba116cc4801e7f80ce21a879c67d2#n454

  // Volatile because this may be called from the WAIT_REG_MEM loop.
  volatile uint32_t* regs_volatile = register_file_->values;
  auto status_host = rex::memory::Reinterpret<reg::COHER_STATUS_HOST>(
      uint32_t(regs_volatile[XE_GPU_REG_COHER_STATUS_HOST]));
  uint32_t base_host = regs_volatile[XE_GPU_REG_COHER_BASE_HOST];
  uint32_t size_host = regs_volatile[XE_GPU_REG_COHER_SIZE_HOST];

  if (!status_host.status) {
    return;
  }

  const char* action = "N/A";
  if (status_host.vc_action_ena && status_host.tc_action_ena) {
    action = "VC | TC";
  } else if (status_host.tc_action_ena) {
    action = "TC";
  } else if (status_host.vc_action_ena) {
    action = "VC";
  }

  // TODO(benvanik): notify resource cache of base->size and type.
  REXGPU_TRACE("Make {:08X} -> {:08X} ({}b) coherent, action = {}", base_host,
               base_host + size_host, size_host, action);

  // Mark coherent.
  regs_volatile[XE_GPU_REG_COHER_STATUS_HOST] = 0;
}

void CommandProcessor::PrepareForWait() {}

void CommandProcessor::ReturnFromWait() {}

uint32_t CommandProcessor::ExecutePrimaryBuffer(uint32_t read_index, uint32_t write_index) {
  SCOPE_profile_cpu_f("gpu");

  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(primary_buffer_ptr_), primary_buffer_size_);
  reader.set_read_offset(read_index * sizeof(uint32_t));
  reader.set_write_offset(write_index * sizeof(uint32_t));
  do {
    if (!ExecutePacket(&reader)) {
      // This probably should be fatal - but we're going to continue anyways.
      REXGPU_ERROR("**** PRIMARY RINGBUFFER: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());

  OnPrimaryBufferEnd();

  return write_index;
}

void CommandProcessor::ExecuteIndirectBuffer(uint32_t ptr, uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
  reader.set_write_offset(count * sizeof(uint32_t));
  do {
    if (!ExecutePacket(&reader)) {
      // Return up a level if we encounter a bad packet.
      REXGPU_ERROR("**** INDIRECT RINGBUFFER: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());
}

void CommandProcessor::ExecutePacket(uint32_t ptr, uint32_t count) {
  // Execute commands!
  memory::RingBuffer reader(memory_->TranslatePhysical(ptr), count * sizeof(uint32_t));
  reader.set_write_offset(count * sizeof(uint32_t));
  do {
    if (!ExecutePacket(&reader)) {
      REXGPU_ERROR("**** ExecutePacket: Failed to execute packet.");
      assert_always();
      break;
    }
  } while (reader.read_count());
}

bool CommandProcessor::ExecutePacket(memory::RingBuffer* reader) {
  if (interp_recording_ && !interp_replaying_) {
    if (SharedMemory* shared_memory = InterpSharedMemory()) {
      shared_memory->interp_packet_ = uint32_t(interp_cur_.size());
    }
    RecordPacket(*reader);
  }
  const uint32_t packet = reader->ReadAndSwap<uint32_t>();
  const uint32_t packet_type = packet >> 30;
  if (packet == 0) {
    return true;
  }

  if (packet == 0xCDCDCDCD) {
    REXGPU_WARN("GPU packet is CDCDCDCD - probably read uninitialized memory!");
  }

  switch (packet_type) {
    case 0x00:
      return ExecutePacketType0(reader, packet);
    case 0x01:
      return ExecutePacketType1(reader, packet);
    case 0x02:
      return ExecutePacketType2(reader, packet);
    case 0x03:
      return ExecutePacketType3(reader, packet);
    default:
      assert_unhandled_case(packet_type);
      return false;
  }
}

bool CommandProcessor::ExecutePacketType0(memory::RingBuffer* reader, uint32_t packet) {
  // Type-0 packet.
  // Write count registers in sequence to the registers starting at
  // (base_index << 2).

  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  if (reader->read_count() < count * sizeof(uint32_t)) {
    REXGPU_ERROR("ExecutePacketType0 overflow (read count {:08X}, packet count {:08X})",
                 reader->read_count(), count * sizeof(uint32_t));
    return false;
  }

  uint32_t base_index = (packet & 0x7FFF);
  uint32_t write_one_reg = (packet >> 15) & 0x1;
  for (uint32_t m = 0; m < count; m++) {
    uint32_t reg_data = reader->ReadAndSwap<uint32_t>();
    uint32_t target_index = write_one_reg ? base_index : base_index + m;
    if (interp_replaying_ && IsReplaySkippedRegister(target_index)) {
      continue;
    }
    WriteRegister(target_index, reg_data);
  }

  return true;
}

bool CommandProcessor::ExecutePacketType1(memory::RingBuffer* reader, uint32_t packet) {
  // Type-1 packet.
  // Contains two registers of data. Type-0 should be more common.
  uint32_t reg_index_1 = packet & 0x7FF;
  uint32_t reg_index_2 = (packet >> 11) & 0x7FF;
  uint32_t reg_data_1 = reader->ReadAndSwap<uint32_t>();
  uint32_t reg_data_2 = reader->ReadAndSwap<uint32_t>();
  if (interp_replaying_ &&
      (IsReplaySkippedRegister(reg_index_1) || IsReplaySkippedRegister(reg_index_2))) {
    return true;  // never seen in practice; both halves go with the real frame
  }
  WriteRegister(reg_index_1, reg_data_1);
  WriteRegister(reg_index_2, reg_data_2);
  return true;
}

bool CommandProcessor::ExecutePacketType2(memory::RingBuffer* reader, uint32_t packet) {
  // Type-2 packet.
  // No-op. Do nothing.
  return true;
}

bool CommandProcessor::ExecutePacketType3(memory::RingBuffer* reader, uint32_t packet) {
  // Type-3 packet.
  uint32_t opcode = (packet >> 8) & 0x7F;
  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  auto data_start_offset = reader->read_offset();

  if (reader->read_count() < count * sizeof(uint32_t)) {
    REXGPU_ERROR("ExecutePacketType3 overflow (read count {:08X}, packet count {:08X})",
                 reader->read_count(), count * sizeof(uint32_t));
    return false;
  }

  // & 1 == predicate - when set, we do bin check to see if we should execute
  // the packet. Only type 3 packets are affected.
  // We also skip predicated swaps, as they are never valid (probably?).
  if (packet & 1) {
    bool any_pass = (bin_select_ & bin_mask_) != 0;
    if (!any_pass || opcode == PM4_XE_SWAP) {
      reader->AdvanceRead(count * sizeof(uint32_t));
      return true;
    }
  }

  if (interp_replaying_) {
    switch (opcode) {
      // Fences, interrupts, waits, memory writes and queries belong to the
      // real frame: the guest must see each exactly once.
      case PM4_XE_SWAP:
      case PM4_INTERRUPT:
      case PM4_WAIT_REG_MEM:
      case PM4_REG_TO_MEM:
      case PM4_MEM_WRITE:
      case PM4_COND_WRITE:
      case PM4_EVENT_WRITE_SHD:
      case PM4_EVENT_WRITE_EXT:
      case PM4_EVENT_WRITE_ZPD:
      case PM4_VIZ_QUERY:
      case PM4_ME_INIT:
        reader->AdvanceRead(count * sizeof(uint32_t));
        return true;
      default:
        break;
    }
  }

  bool result = false;
  switch (opcode) {
    case PM4_ME_INIT:
      result = ExecutePacketType3_ME_INIT(reader, packet, count);
      break;
    case PM4_NOP:
      result = ExecutePacketType3_NOP(reader, packet, count);
      break;
    case PM4_INTERRUPT:
      result = ExecutePacketType3_INTERRUPT(reader, packet, count);
      break;
    case PM4_XE_SWAP:
      result = ExecutePacketType3_XE_SWAP(reader, packet, count);
      break;
    case PM4_INDIRECT_BUFFER:
    case PM4_INDIRECT_BUFFER_PFD:
      result = ExecutePacketType3_INDIRECT_BUFFER(reader, packet, count);
      break;
    case PM4_WAIT_REG_MEM:
      result = ExecutePacketType3_WAIT_REG_MEM(reader, packet, count);
      break;
    case PM4_REG_RMW:
      result = ExecutePacketType3_REG_RMW(reader, packet, count);
      break;
    case PM4_REG_TO_MEM:
      result = ExecutePacketType3_REG_TO_MEM(reader, packet, count);
      break;
    case PM4_MEM_WRITE:
      result = ExecutePacketType3_MEM_WRITE(reader, packet, count);
      break;
    case PM4_COND_WRITE:
      result = ExecutePacketType3_COND_WRITE(reader, packet, count);
      break;
    case PM4_EVENT_WRITE:
      result = ExecutePacketType3_EVENT_WRITE(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_SHD:
      result = ExecutePacketType3_EVENT_WRITE_SHD(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_EXT:
      result = ExecutePacketType3_EVENT_WRITE_EXT(reader, packet, count);
      break;
    case PM4_EVENT_WRITE_ZPD:
      result = ExecutePacketType3_EVENT_WRITE_ZPD(reader, packet, count);
      break;
    case PM4_DRAW_INDX:
      result = ExecutePacketType3_DRAW_INDX(reader, packet, count);
      break;
    case PM4_DRAW_INDX_2:
      result = ExecutePacketType3_DRAW_INDX_2(reader, packet, count);
      break;
    case PM4_SET_CONSTANT:
      result = ExecutePacketType3_SET_CONSTANT(reader, packet, count);
      break;
    case PM4_SET_CONSTANT2:
      result = ExecutePacketType3_SET_CONSTANT2(reader, packet, count);
      break;
    case PM4_LOAD_ALU_CONSTANT:
      result = ExecutePacketType3_LOAD_ALU_CONSTANT(reader, packet, count);
      break;
    case PM4_SET_SHADER_CONSTANTS:
      result = ExecutePacketType3_SET_SHADER_CONSTANTS(reader, packet, count);
      break;
    case PM4_IM_LOAD:
      result = ExecutePacketType3_IM_LOAD(reader, packet, count);
      break;
    case PM4_IM_LOAD_IMMEDIATE:
      result = ExecutePacketType3_IM_LOAD_IMMEDIATE(reader, packet, count);
      break;
    case PM4_INVALIDATE_STATE:
      result = ExecutePacketType3_INVALIDATE_STATE(reader, packet, count);
      break;
    case PM4_VIZ_QUERY:
      result = ExecutePacketType3_VIZ_QUERY(reader, packet, count);
      break;

    case PM4_SET_BIN_MASK_LO: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (bin_mask_ & 0xFFFFFFFF00000000ull) | value;
      result = true;
    } break;
    case PM4_SET_BIN_MASK_HI: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (bin_mask_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(value) << 32);
      result = true;
    } break;
    case PM4_SET_BIN_SELECT_LO: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_select_ = (bin_select_ & 0xFFFFFFFF00000000ull) | value;
      result = true;
    } break;
    case PM4_SET_BIN_SELECT_HI: {
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      bin_select_ = (bin_select_ & 0xFFFFFFFFull) | (static_cast<uint64_t>(value) << 32);
      result = true;
    } break;
    case PM4_SET_BIN_MASK: {
      assert_true(count == 2);
      uint64_t val_hi = reader->ReadAndSwap<uint32_t>();
      uint64_t val_lo = reader->ReadAndSwap<uint32_t>();
      bin_mask_ = (val_hi << 32) | val_lo;
      result = true;
    } break;
    case PM4_SET_BIN_SELECT: {
      assert_true(count == 2);
      uint64_t val_hi = reader->ReadAndSwap<uint32_t>();
      uint64_t val_lo = reader->ReadAndSwap<uint32_t>();
      bin_select_ = (val_hi << 32) | val_lo;
      result = true;
    } break;
    case PM4_CONTEXT_UPDATE: {
      assert_true(count == 1);
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      REXGPU_INFO("GPU context update = {:08X}", value);
      assert_true(value == 0);
      result = true;
      break;
    }
    case PM4_WAIT_FOR_IDLE: {
      // This opcode is used by 5454084E while going / being ingame.
      assert_true(count == 1);
      uint32_t value = reader->ReadAndSwap<uint32_t>();
      REXGPU_INFO("GPU wait for idle = {:08X}", value);
      result = true;
      break;
    }

    default:
      REXGPU_INFO("Unimplemented GPU OPCODE: 0x{:02X}\t\tCOUNT: {}\n", opcode, count);
      assert_always();
      reader->AdvanceRead(count * sizeof(uint32_t));
      break;
  }

  assert_true(reader->read_offset() ==
              (data_start_offset + (count * sizeof(uint32_t))) % reader->capacity());
  return result;
}

bool CommandProcessor::ExecutePacketType3_ME_INIT(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  // initialize CP's micro-engine
  me_bin_.clear();
  for (uint32_t i = 0; i < count; i++) {
    me_bin_.push_back(reader->ReadAndSwap<uint32_t>());
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3_NOP(memory::RingBuffer* reader, uint32_t packet,
                                              uint32_t count) {
  // skip N 32-bit words to get to the next packet
  // No-op, ignore some data.
  reader->AdvanceRead(count * sizeof(uint32_t));
  return true;
}

bool CommandProcessor::ExecutePacketType3_INTERRUPT(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // generate interrupt from the command stream
  uint32_t cpu_mask = reader->ReadAndSwap<uint32_t>();
  for (int n = 0; n < 6; n++) {
    if (cpu_mask & (1 << n)) {
      if (graphics_system_) {
        graphics_system_->DispatchInterruptCallback(1, n);
      }
    }
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_XE_SWAP(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

#ifdef REXGLUE_ENABLE_PERF_COUNTERS
  {
    static uint64_t last_frame_tick = 0;
    uint64_t now = rex::chrono::Clock::QueryHostTickCount();
    if (last_frame_tick) {
      uint64_t freq = rex::chrono::Clock::QueryHostTickFrequency();
      int64_t dt_us = static_cast<int64_t>((now - last_frame_tick) * 1000000 / freq);
      PROFILE_FRAME_TIME_US(dt_us);
      PROFILE_FPS(freq / (now - last_frame_tick));
    }
    last_frame_tick = now;
  }
#endif
  rex::perf::Profiler::Flip();

  // Xenia-specific VdSwap hook.
  // VdSwap will post this to tell us we need to swap the screen/fire an
  // interrupt.
  // 63 words here, but only the first has any data.
  uint32_t magic = reader->ReadAndSwap<memory::fourcc_t>();
  assert_true(magic == kSwapSignature);

  // TODO(benvanik): only swap frontbuffer ptr.
  uint32_t frontbuffer_ptr = reader->ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_width = reader->ReadAndSwap<uint32_t>();
  uint32_t frontbuffer_height = reader->ReadAndSwap<uint32_t>();
  reader->AdvanceRead((count - 4) * sizeof(uint32_t));

  if (REXCVAR_GET(frame_interp) || interp_recording_) {
    SwapWithInterpolation(frontbuffer_ptr, frontbuffer_width, frontbuffer_height);
  } else {
    IssueSwap(frontbuffer_ptr, frontbuffer_width, frontbuffer_height);
    // TEMP DIAGNOSTIC: the same pacing line as frame_interp's, for comparison.
    static uint64_t last = 0, sum = 0, slow = 0, frames = 0;
    const uint64_t now = rex::chrono::Clock::QueryHostTickCount();
    const uint64_t freq = rex::chrono::Clock::QueryHostTickFrequency();
    if (last && now - last < freq) {
      sum += now - last;
      slow += (now - last) * 2000 > freq * 35;
      if (++frames == 600) {
        REXGPU_INFO("swap pacing (no interp): 600 real frames in {} ms, {} over 17.5 ms",
                    sum * 1000 / freq, slow);
        sum = slow = frames = 0;
      }
    }
    last = now;
  }

  ++counter_;
  return true;
}

bool CommandProcessor::IsReplaySkippedRegister(uint32_t index) {
  return WriteRegisterHasSideEffects(index) || index == XE_GPU_REG_SCRATCH_ADDR ||
         index == XE_GPU_REG_SCRATCH_UMSK || index == XE_GPU_REG_COHER_STATUS_HOST ||
         index == XE_GPU_REG_COHER_BASE_HOST || index == XE_GPU_REG_COHER_SIZE_HOST;
}

void CommandProcessor::RecordPacket(const memory::RingBuffer& reader) {
  memory::RingBuffer peek = reader;
  const uint32_t header = peek.ReadAndSwap<uint32_t>();
  if (!header) {
    return;
  }
  uint32_t dwords = 1;
  switch (header >> 30) {
    case 0:
    case 3:
      dwords += ((header >> 16) & 0x3FFF) + 1;
      break;
    case 1:
      dwords = 3;
      break;
  }
  if (header >> 30 == 3) {
    const uint32_t opcode = (header >> 8) & 0x7F;
    // Indirect buffers record as their contents; the swap ends the recording.
    if (opcode == PM4_INDIRECT_BUFFER || opcode == PM4_INDIRECT_BUFFER_PFD ||
        opcode == PM4_XE_SWAP) {
      return;
    }
    // Constant loads record as the plain register writes they perform, with
    // the values of this moment: a replay must not read guest memory the guest
    // has rewritten since, and Interpolate lerps / pairs register writes only.
    static constexpr uint32_t kConstantBase[] = {0x4000, 0x4800, 0x4900, 0x4908, 0x2000};
    const uint32_t count = dwords - 1;
    // Flattened packets lose their predicate: keep only those the real frame
    // executes (predicated tiling sets each tile's state this way).
    const bool executes = !(header & 1) || (bin_select_ & bin_mask_) != 0;
    auto record_writes = [&](uint32_t reg, const uint32_t* be_values, uint32_t n) {
      if (!n || reg + n > RegisterFile::kRegisterCount || n > 0x4000) {
        return false;
      }
      interp_cur_.push_back(std::byteswap(((n - 1) << 16) | reg));
      interp_cur_.insert(interp_cur_.end(), be_values, be_values + n);
      return true;
    };
    if (opcode == PM4_LOAD_ALU_CONSTANT && count >= 3) {
      const uint32_t address = peek.ReadAndSwap<uint32_t>() & 0x3FFFFFFF;
      const uint32_t offset_type = peek.ReadAndSwap<uint32_t>();
      const uint32_t size = peek.ReadAndSwap<uint32_t>() & 0xFFF;
      const uint32_t type = (offset_type >> 16) & 0xFF;
      const auto* values = memory_->TranslatePhysical<const uint32_t*>(address);
      ++interp_pk_stat_[0];
      if (!executes) {
        return;
      }
      if (type < 5 && values &&
          record_writes(kConstantBase[type] + (offset_type & 0x7FF), values, size)) {
        return;
      }
    } else if ((opcode == PM4_SET_CONSTANT || opcode == PM4_SET_CONSTANT2 ||
                opcode == PM4_SET_SHADER_CONSTANTS) &&
               count >= 2 && peek.read_count() >= count * sizeof(uint32_t)) {
      const uint32_t offset_type = peek.ReadAndSwap<uint32_t>();
      uint32_t reg = offset_type & 0xFFFF;
      if (opcode == PM4_SET_CONSTANT) {
        const uint32_t type = (offset_type >> 16) & 0xFF;
        reg = type < 5 ? kConstantBase[type] + (offset_type & 0x7FF) : UINT32_MAX;
      }
      ++interp_pk_stat_[1];
      if (!executes) {
        return;
      }
      std::vector<uint32_t>& be_values = interp_vs_scratch_;
      be_values.resize(count - 1);
      peek.Read(reinterpret_cast<uint8_t*>(be_values.data()), (count - 1) * sizeof(uint32_t));
      if (reg != UINT32_MAX && record_writes(reg, be_values.data(), count - 1)) {
        return;
      }
    }
    peek = reader;  // not recorded as writes: copied as is below
    peek.ReadAndSwap<uint32_t>();
  }
  if (peek.read_count() + sizeof(uint32_t) < dwords * sizeof(uint32_t)) {
    return;  // malformed; ExecutePacket reports it
  }
  const size_t at = interp_cur_.size();
  interp_cur_.resize(at + dwords);
  memory::RingBuffer copy = reader;
  copy.Read(reinterpret_cast<uint8_t*>(interp_cur_.data() + at), dwords * sizeof(uint32_t));
}

void CommandProcessor::ServiceHeldSwap(bool force) {
  if (!interp_held_pending_ ||
      (!force && rex::chrono::Clock::QueryHostTickCount() < interp_held_deadline_)) {
    return;
  }
  interp_held_pending_ = false;
  PresentHeldSwapTexture(interp_held_width_, interp_held_height_);
}

void CommandProcessor::ReplayInterpolatedFrame() {
  InterpBudgetScope budget(interp_budget_[2]);  // TEMP DIAGNOSTIC
  // The replay ends in the real frame's register state except for the lerped
  // constants; put every register it changed back.
  RegisterFile& regs = *register_file_;
  interp_regs_.assign(regs.values, regs.values + RegisterFile::kRegisterCount);
  const uint64_t bin_select = bin_select_, bin_mask = bin_mask_;
  // ...and starts in its starting state: registers set before the first
  // recorded packet (the swap sequence, writes from outside the ring) hold
  // the values the real frame's first draws saw.
  if (interp_start_regs_.size() == RegisterFile::kRegisterCount) {
    for (uint32_t i = 0; i < RegisterFile::kRegisterCount; ++i) {
      if (regs.values[i] != interp_start_regs_[i] && !IsReplaySkippedRegister(i)) {
        WriteRegister(i, interp_start_regs_[i]);
      }
    }
    bin_select_ = interp_start_bin_select_;
    bin_mask_ = interp_start_bin_mask_;
  }

  SharedMemory* shared_memory = InterpSharedMemory();
  interp_replaying_ = true;
  if (shared_memory) {
    shared_memory->interp_replay_ = true;
    if (shared_memory->interp_dump_) {  // TEMP DIAGNOSTIC
      REXGPU_INFO("IDUMP replay");
    }
  }
  memory::RingBuffer reader(reinterpret_cast<uint8_t*>(interp_work_.data()),
                            interp_work_.size() * sizeof(uint32_t));
  reader.set_write_offset(interp_work_.size() * sizeof(uint32_t));
  do {
    if (shared_memory) {
      shared_memory->interp_packet_ = uint32_t(reader.read_offset() / sizeof(uint32_t));
      shared_memory->InterpApplyUploads(shared_memory->interp_packet_);
    }
    if (!ExecutePacket(&reader)) {
      REXGPU_ERROR("frame_interp: replay stopped on a bad packet");
      break;
    }
  } while (reader.read_count());
  if (shared_memory) {
    shared_memory->InterpApplyUploads(UINT32_MAX);  // the frame ends on its last data
    shared_memory->InterpEndReplay();
    shared_memory->interp_replay_ = false;
    interp_vs_written_.clear();
  }
  interp_replaying_ = false;

  bin_select_ = bin_select;
  bin_mask_ = bin_mask;
  for (uint32_t i = 0; i < RegisterFile::kRegisterCount; ++i) {
    if (regs.values[i] != interp_regs_[i] && !IsReplaySkippedRegister(i)) {
      WriteRegister(i, interp_regs_[i]);
    }
  }
}

void CommandProcessor::SwapWithInterpolation(uint32_t frontbuffer_ptr,
                                             uint32_t frontbuffer_width,
                                             uint32_t frontbuffer_height) {
  // A held frame the CP had no idle moment for still goes out first.
  ServiceHeldSwap(true);

  const bool enabled = REXCVAR_GET(frame_interp);
  const uint64_t now = rex::chrono::Clock::QueryHostTickCount();
  const uint64_t freq = rex::chrono::Clock::QueryHostTickFrequency();
  const uint64_t interval = now - interp_last_swap_;
  const uint64_t period = std::clamp<uint64_t>(interval, freq / 60, freq / 30);
  interp_last_swap_ = now;

  const uint64_t parse_start = rex::chrono::Clock::QueryHostTickCount();  // TEMP DIAGNOSTIC
  frame_interp::Parse(interp_cur_.data(), interp_cur_.size(), interp_parsed_cur_);
  bool interpolate = enabled && interp_have_prev_ && !interp_cur_.empty();
  frame_interp::Stats stats;
  struct ParseBudget {  // TEMP DIAGNOSTIC: Parse + Interpolate
    uint64_t& sum;
    uint64_t start;
    ~ParseBudget() { sum += rex::chrono::Clock::QueryHostTickCount() - start; }
  };
  std::optional<ParseBudget> parse_budget(std::in_place, interp_budget_[3], parse_start);
  if (interpolate) {
    interp_work_ = interp_cur_;
    stats = frame_interp::Interpolate(interp_prev_.data(), interp_parsed_prev_,
                                      interp_work_.data(), interp_parsed_cur_, 0.5f,
                                      &interp_pairs_);
    // Most paired draws jumped: a camera cut or scene change, nothing to blend.
    const bool cut = stats.jumped * 4 > stats.matched;
    // TEMP DIAGNOSTIC: also dump frames where a copy of a mesh jumped (at
    // most one a second, 40 a run).
    static uint32_t suspect_dumps = 0, suspect_wait = 0;
    suspect_wait -= suspect_wait != 0;
    const bool suspect = stats.suspect && !cut && !suspect_wait && suspect_dumps < 40;
    if (suspect) {
      ++suspect_dumps;
      suspect_wait = 60;
    }
    if (SharedMemory* sm = InterpSharedMemory(); sm && (sm->interp_dump_ || suspect)) {
      REXGPU_INFO("IDUMP swap segments {} matched {} lerped {} jumped {} suspect {} cut {}",
                  stats.segments, stats.matched, stats.lerped, stats.jumped, stats.suspect, cut);
      // TEMP DIAGNOSTIC: both raw streams, for offline pairing analysis.
      static int dump_index = 0;
      const std::pair<const char*, const std::vector<uint32_t>*> files[] = {
          {"prev", &interp_prev_}, {"cur", &interp_cur_}};
      for (const auto& [name, data] : files) {
        const std::string path =
            fmt::format("C:/Users/User/AppData/Local/Temp/opencode/istream_{}_{}.bin", dump_index, name);
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
          std::fwrite(data->data(), 4, data->size(), f);
          std::fclose(f);
        }
      }
      ++dump_index;
    }
    interp_stat_cuts_ += cut;
    interpolate = !cut && HoldSwapTexture();
  }
  parse_budget.reset();
  if (interpolate) {
    ReplayInterpolatedFrame();
    IssueSwap(frontbuffer_ptr, frontbuffer_width, frontbuffer_height);
    interp_held_width_ = frontbuffer_width;
    interp_held_height_ = frontbuffer_height;
    interp_held_deadline_ = rex::chrono::Clock::QueryHostTickCount() + period / 2;
    interp_held_pending_ = true;
  } else {
    IssueSwap(frontbuffer_ptr, frontbuffer_width, frontbuffer_height);
  }

  if (enabled) {
    interp_stats_.segments += stats.segments;
    interp_stats_.matched += stats.matched;
    interp_stats_.lerped += stats.lerped;
    interp_stats_.jumped += stats.jumped;
    interp_stat_ticks_ += interval < freq ? interval : freq / 60;  // the first swap has no previous
    interp_stat_slow_ += interval * 2000 > freq * 35;
    if (++interp_stat_frames_ == 600) {
      if (SharedMemory* shared_memory = InterpSharedMemory()) {
        REXGPU_INFO("frame_interp: per frame {} uploads kept, {} pages re-uploaded, {} KB restored",
                    std::exchange(shared_memory->interp_noted_, 0) / 600,
                    std::exchange(shared_memory->interp_reupload_pages_, 0) / 600,
                    std::exchange(shared_memory->interp_restored_bytes_, 0) / 600 / 1024);
        uint64_t* d = shared_memory->interp_diag_;  // TEMP DIAGNOSTIC
        REXGPU_INFO("frame_interp: diag x100/frame resolves {}/{}, tex loads {}/{}, blocked {}, "
                    "restores {}",
                    d[0] / 6, d[1] / 6, d[2] / 6, d[3] / 6, d[4] / 6, d[5] / 6);
        std::fill(d, d + 6, 0);
        REXGPU_INFO("frame_interp: vertex streams x100/frame {} float, {} non-float dynamic, "
                    "{} unpaired, {} jumped; {} KB lerped per frame; constant loads x100/frame "
                    "{} from memory, {} inline",
                    interp_vs_stat_[0] / 6, interp_vs_stat_[1] / 6, interp_vs_stat_[3] / 6,
                    interp_vs_stat_[4] / 6, interp_vs_stat_[2] / 600, interp_pk_stat_[0] / 6,
                    interp_pk_stat_[1] / 6);
        std::fill(std::begin(interp_vs_stat_), std::end(interp_vs_stat_), 0);
        std::fill(std::begin(interp_pk_stat_), std::end(interp_pk_stat_), 0);
        REXGPU_INFO("frame_interp: replay state check: {} draws, {} with other fetch constants, "
                    "{} with other render state (dump frames only)",
                    interp_state_stat_[0], interp_state_stat_[1], interp_state_stat_[2]);
        std::fill(std::begin(interp_state_stat_), std::end(interp_state_stat_), 0);
        // TEMP DIAGNOSTIC: GPU thread budget, us per real frame (16667 = all of a 60 Hz tick).
        auto us = [&](uint64_t ticks) { return ticks * 1000000 / freq / 600; };
        REXGPU_INFO("frame_interp: budget us/frame: vertex record {}, vertex replay {}, replay {}, "
                    "parse {}, restores {}, busy {}",
                    us(interp_budget_[0]), us(interp_budget_[1]), us(interp_budget_[2]),
                    us(interp_budget_[3]), us(shared_memory->interp_restore_ticks_),
                    us(interp_stat_ticks_ - std::min(interp_stat_ticks_, interp_wait_ticks_)));
        std::fill(std::begin(interp_budget_), std::end(interp_budget_), 0);
        shared_memory->interp_restore_ticks_ = 0;
        interp_wait_ticks_ = 0;
      }
      REXGPU_INFO(
          "frame_interp: per frame {} draws, {} paired, {} lerped, {} jumped; {} cut frames; 600 "
          "real frames in {} ms (10000 = 60 Hz), {} over 17.5 ms",
          interp_stats_.segments / 600, interp_stats_.matched / 600, interp_stats_.lerped / 600,
          interp_stats_.jumped / 600, interp_stat_cuts_, interp_stat_ticks_ * 1000 / freq,
          interp_stat_slow_);
      interp_stats_ = {};
      interp_stat_frames_ = 0;
      interp_stat_ticks_ = 0;
      interp_stat_slow_ = 0;
      interp_stat_cuts_ = 0;
    }
  }

  std::swap(interp_prev_, interp_cur_);
  std::swap(interp_parsed_prev_, interp_parsed_cur_);
  interp_cur_.clear();
  interp_states_.clear();
  interp_state_values_.clear();
  interp_state_next_ = 0;
  std::swap(interp_vs_prev_, interp_vs_cur_);
  std::swap(interp_vs_data_prev_, interp_vs_data_cur_);
  interp_vs_cur_.clear();
  interp_vs_data_cur_.clear();
  std::swap(interp_vs_copies_prev_, interp_vs_copies_);
  interp_vs_copies_.clear();
  interp_have_prev_ = enabled;
  interp_recording_ = enabled;
  if (enabled) {  // the state the next frame's recording starts from
    interp_start_regs_.assign(register_file_->values,
                              register_file_->values + RegisterFile::kRegisterCount);
    interp_start_bin_select_ = bin_select_;
    interp_start_bin_mask_ = bin_mask_;
  }
  if (SharedMemory* shared_memory = InterpSharedMemory()) {
    shared_memory->InterpEndFrame();
    shared_memory->interp_recording_ = enabled;
    // TEMP DIAGNOSTIC: the next frame and its replay, once per stats window.
    shared_memory->interp_dump_ = enabled && interp_stat_frames_ == 300;
  if (shared_memory->interp_dump_) {
      REXGPU_INFO("IDUMP begin");
    }
  }
}

CommandProcessor::InterpBudgetScope::InterpBudgetScope(uint64_t& s)
    : sum(s), start(rex::chrono::Clock::QueryHostTickCount()) {}
CommandProcessor::InterpBudgetScope::~InterpBudgetScope() {
  sum += rex::chrono::Clock::QueryHostTickCount() - start;
}

void CommandProcessor::InterpCheckDrawState() {
  SharedMemory* shared_memory = InterpSharedMemory();
  if (!interp_recording_ || !shared_memory) {
    return;
  }
  constexpr uint32_t kFetchFirst = 0x4800, kFetchCount = 32 * 6;
  constexpr uint32_t kStateFirst = 0x2000, kStateCount = 0x400;
  const bool dump = shared_memory->interp_dump_;
  const uint32_t* regs = register_file_->values;
  const uint32_t packet = shared_memory->interp_packet_;

  if (!interp_replaying_) {
    interp_states_.push_back({packet, interp_state_values_.size()});
    interp_state_values_.insert(interp_state_values_.end(), regs + kFetchFirst,
                                regs + kFetchFirst + kFetchCount);
    if (dump) {
      interp_state_values_.insert(interp_state_values_.end(), regs + kStateFirst,
                                  regs + kStateFirst + kStateCount);
    }
    return;
  }
  while (interp_state_next_ < interp_states_.size() &&
         interp_states_[interp_state_next_].packet < packet) {
    ++interp_state_next_;
  }
  if (interp_state_next_ >= interp_states_.size() ||
      interp_states_[interp_state_next_].packet != packet) {
    return;
  }
  ++interp_state_stat_[0];
  const uint32_t* real = interp_state_values_.data() + interp_states_[interp_state_next_].at;
  uint32_t logged = 0;
  auto compare = [&](uint32_t first, const uint32_t* real_values, uint32_t count) {
    bool differ = false;
    for (uint32_t i = 0; i < count; ++i) {
      if (regs[first + i] != real_values[i]) {
        differ = true;
        if (dump && logged++ < 8) {
          REXGPU_INFO("IDUMP state pkt={} reg {:04X} replay {:08X} real {:08X}", packet,
                      first + i, regs[first + i], real_values[i]);
        }
      }
    }
    return differ;
  };
  interp_state_stat_[1] += compare(kFetchFirst, real, kFetchCount);
  if (dump) {
    interp_state_stat_[2] += compare(kStateFirst, real + kFetchCount, kStateCount);
  }
}

void CommandProcessor::InterpVertexDraw(const Shader& vertex_shader) {
  SharedMemory* shared_memory = InterpSharedMemory();
  if (!interp_recording_ || !shared_memory) {
    return;
  }
  InterpBudgetScope budget(interp_budget_[interp_replaying_ ? 1 : 0]);  // TEMP DIAGNOSTIC
  const uint32_t packet = shared_memory->interp_packet_;
  auto streams_at = [](const std::vector<InterpVertexStream>& streams, uint32_t at) {
    return std::lower_bound(streams.begin(), streams.end(), at,
                            [](const InterpVertexStream& s, uint32_t p) { return s.packet < p; });
  };

  if (shared_memory->interp_dump_) {
    // TEMP DIAGNOSTIC: per draw, texture bases of fetch slots 0-2 and each
    // vertex binding with a hash of its first 64 guest bytes (the real
    // frame's data) and whether the replay has it valid or restored.
    for (const Shader::VertexBinding& binding : vertex_shader.vertex_bindings()) {
      const xenos::xe_gpu_vertex_fetch_t vf = register_file_->GetVertexFetch(binding.fetch_constant);
      const uint32_t start = uint32_t(vf.address) << 2, bytes = uint32_t(vf.size) << 2;
      uint64_t hash = 0;
      if (const auto* guest = memory_->TranslatePhysical<const uint32_t*>(start)) {
        for (uint32_t i = 0; i < std::min(16u, bytes / 4); ++i) {
          hash = hash * 0x100000001B3ull ^ guest[i];
        }
      }
      REXGPU_INFO("IDUMP draw rp={} pkt={} tex {:08X} {:08X} {:08X} vb {:08X}+{} guest {:016X} "
                  "valid {}",
                  interp_replaying_, packet, register_file_->values[0x4801] & ~0xFFFu,
                  register_file_->values[0x4807] & ~0xFFFu, register_file_->values[0x480D] & ~0xFFFu, start, bytes,
                  hash, shared_memory->InterpReplayCanLoad(start, std::min(bytes, 64u)));
    }
  }

  if (interp_replaying_) {
    auto pair = std::lower_bound(interp_pairs_.begin(), interp_pairs_.end(),
                                 std::make_pair(packet, uint32_t(0)));
    const bool paired = pair != interp_pairs_.end() && pair->first == packet;
    for (auto s = streams_at(interp_vs_cur_, packet);
         s != interp_vs_cur_.end() && s->packet == packet; ++s) {
      const InterpVertexStream* p = nullptr;
      if (paired) {
        for (auto q = streams_at(interp_vs_prev_, pair->second);
             q != interp_vs_prev_.end() && q->packet == pair->second; ++q) {
          if (q->binding == s->binding && q->bytes == s->bytes &&
              q->stride_words == s->stride_words && q->float_words == s->float_words &&
              q->endian == s->endian) {
            p = &*q;
            break;
          }
        }
      }
      if (!p) {
        ++interp_vs_stat_[3];
        if (shared_memory->interp_dump_) {  // TEMP DIAGNOSTIC
          REXGPU_INFO("IDUMP vunpaired pkt={} paired {} b={} at {:08X} bytes {} stride {}", packet,
                      paired, s->binding, s->start, s->bytes, s->stride_words);
        }
        continue;
      }
      if (std::find(interp_vs_written_.begin(), interp_vs_written_.end(), s->start) !=
          interp_vs_written_.end()) {
        continue;  // drawn again (another pass): already lerped
      }
      // Most "CPU-written" streams are static data sharing pages with dynamic
      // data: only the vertices that changed are lerped and uploaded.
      const uint32_t* cur = interp_vs_data_cur_.data() + s->data;
      const uint32_t* prev = interp_vs_data_prev_.data() + p->data;
      const uint32_t words = s->bytes / 4, vertices = words / s->stride_words;
      const uint32_t first_word = uint32_t(std::mismatch(cur, cur + words, prev).first - cur);
      if (first_word == words) {
        continue;
      }
      const auto rcur = std::make_reverse_iterator(cur + words);
      const uint32_t last_word =
          words - uint32_t(std::mismatch(rcur, std::make_reverse_iterator(cur + first_word),
                                         std::make_reverse_iterator(prev + words))
                               .first -
                           rcur);
      const uint32_t first = first_word / s->stride_words;
      const uint32_t last = std::min(vertices, (last_word + s->stride_words - 1) / s->stride_words);
      const uint32_t begin = first * s->stride_words, end = last * s->stride_words;
      interp_vs_scratch_.assign(cur + begin, cur + end);
      if (!frame_interp::LerpVertexFloats(prev + begin, interp_vs_scratch_.data(), last - first,
                                          s->stride_words, s->float_words, s->endian, 0.5f)) {
        ++interp_vs_stat_[4];
        if (shared_memory->interp_dump_) {  // TEMP DIAGNOSTIC
          // The first word that failed: outside the float mask, or a jump.
          const uint32_t* pw = prev + begin;
          const uint32_t* cw = cur + begin;
          uint32_t bad = 0;
          for (; bad < end - begin; ++bad) {
            if (pw[bad] == cw[bad]) continue;
            if (!((s->float_words >> (bad % s->stride_words)) & 1)) break;
            const float c = std::bit_cast<float>(xenos::GpuSwap(cw[bad], xenos::Endian(s->endian)));
            const float q = std::bit_cast<float>(xenos::GpuSwap(pw[bad], xenos::Endian(s->endian)));
            if (std::isfinite(c) && std::isfinite(q) &&
                std::abs(c - q) > std::max(2.0f, 0.5f * std::max(std::abs(c), std::abs(q)))) {
              break;
            }
          }
          REXGPU_INFO("IDUMP vjumped pkt={} b={} at {:08X} prev at {:08X} bytes {} stride {} "
                      "mask {:X} verts {}..{} word {} (vertex {} word {}) {:08X} -> {:08X}",
                      packet, s->binding, s->start, p->start, s->bytes, s->stride_words,
                      s->float_words, first, last, bad, (begin + bad) / s->stride_words,
                      (begin + bad) % s->stride_words, bad < end - begin ? pw[bad] : 0,
                      bad < end - begin ? cw[bad] : 0);
          // TEMP DIAGNOSTIC: the whole failing vertex and the one before, as floats.
          if (bad < end - begin) {
            const uint32_t v0 = bad / s->stride_words;
            for (uint32_t v = v0 ? v0 - 1 : 0; v <= v0; ++v) {
              std::string line;
              for (int which = 0; which < 2; ++which) {
                line += which ? " | cur" : "prev";
                for (uint32_t w = 0; w < s->stride_words; ++w) {
                  const uint32_t word = (which ? cw : pw)[v * s->stride_words + w];
                  line += fmt::format(" {:.3f}", std::bit_cast<float>(
                                                     xenos::GpuSwap(word, xenos::Endian(s->endian))));
                }
              }
              REXGPU_INFO("IDUMP vjvert at {:08X} vertex {} {}", s->start, first + v, line);
            }
          }
        }
        continue;
      }
      shared_memory->InterpWrite(s->start + begin * 4, interp_vs_scratch_.data(),
                                 (end - begin) * 4);
      interp_vs_written_.push_back(s->start);
      interp_vs_stat_[2] += ((end - begin) * 4) >> 10;
      if (shared_memory->interp_dump_) {  // TEMP DIAGNOSTIC
        REXGPU_INFO("IDUMP vlerp pkt={} prev={} b={} at {:08X} prev at {:08X} bytes {} stride {} "
                    "mask {:X} verts {}..{} of {}",
                    packet, pair->second, s->binding, s->start, p->start, s->bytes,
                    s->stride_words, s->float_words, first, last, vertices);
      }
    }
    return;
  }

  const RegisterFile& regs = *register_file_;
  for (const Shader::VertexBinding& binding : vertex_shader.vertex_bindings()) {
    if (!binding.stride_words || binding.stride_words > 64) {
      continue;
    }
    uint64_t float_words = 0;
    for (const Shader::VertexBinding::Attribute& attribute : binding.attributes) {
      const auto& a = attribute.fetch_instr.attributes;
      uint32_t count = 0;
      switch (a.data_format) {
        case xenos::VertexFormat::k_32_FLOAT:
          count = 1;
          break;
        case xenos::VertexFormat::k_32_32_FLOAT:
          count = 2;
          break;
        case xenos::VertexFormat::k_32_32_32_FLOAT:
          count = 3;
          break;
        case xenos::VertexFormat::k_32_32_32_32_FLOAT:
          count = 4;
          break;
        default:
          break;
      }
      for (uint32_t k = 0; k < count; ++k) {
        const int32_t w = a.offset + int32_t(k);
        if (w >= 0 && uint32_t(w) < binding.stride_words) {
          float_words |= uint64_t(1) << w;
        }
      }
    }
    const xenos::xe_gpu_vertex_fetch_t fetch = regs.GetVertexFetch(binding.fetch_constant);
    const uint32_t start = fetch.address << 2, bytes = fetch.size << 2;
    // Only data the CPU wrote this frame can differ from the previous frame's.
    if (!bytes || !shared_memory->InterpUploadedThisFrame(start, bytes)) {
      continue;
    }
    if (!float_words) {
      ++interp_vs_stat_[1];
      continue;
    }
    ++interp_vs_stat_[0];
    const uint32_t vertex_bytes = binding.stride_words * 4;
    const uint32_t lerp_bytes = bytes / vertex_bytes * vertex_bytes;
    const uint8_t* guest = memory_->TranslatePhysical<const uint8_t*>(start);
    if (!lerp_bytes || !guest) {
      continue;
    }
    const uint64_t key = (uint64_t(start) << 32) | lerp_bytes;
    // The same buffer drawn last frame too: static data on pages the CPU
    // rewrote around it (dynamic meshes are double-buffered or ring-allocated).
    if (interp_vs_copies_prev_.count(key)) {
      interp_vs_copies_[key] = SIZE_MAX;
      continue;
    }
    auto [copy, added] = interp_vs_copies_.try_emplace(key, interp_vs_data_cur_.size());
    if (copy->second == SIZE_MAX) {
      continue;
    }
    if (added) {
      interp_vs_data_cur_.resize(interp_vs_data_cur_.size() + lerp_bytes / 4);
      std::memcpy(interp_vs_data_cur_.data() + copy->second, guest, lerp_bytes);
    }
    if (shared_memory->interp_dump_) {  // TEMP DIAGNOSTIC
      REXGPU_INFO("IDUMP vstream pkt={} b={} at {:08X} bytes {} stride {} mask {:X}", packet,
                  binding.binding_index, start, lerp_bytes, binding.stride_words, float_words);
    }
    interp_vs_cur_.push_back({packet, uint32_t(binding.binding_index), start, lerp_bytes,
                              binding.stride_words, uint32_t(fetch.endian), float_words,
                              copy->second});
  }
}

bool CommandProcessor::ExecutePacketType3_INDIRECT_BUFFER(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // indirect buffer dispatch
  uint32_t list_ptr = CpuToGpu(reader->ReadAndSwap<uint32_t>());
  uint32_t list_length = reader->ReadAndSwap<uint32_t>();
  assert_zero(list_length & ~0xFFFFF);
  list_length &= 0xFFFFF;
  ExecuteIndirectBuffer(GpuToCpu(list_ptr), list_length);
  return true;
}

bool CommandProcessor::ExecutePacketType3_WAIT_REG_MEM(memory::RingBuffer* reader, uint32_t packet,
                                                       uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // wait until a register or memory location is a specific value

  uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t ref = reader->ReadAndSwap<uint32_t>();
  uint32_t mask = reader->ReadAndSwap<uint32_t>();
  uint32_t wait = reader->ReadAndSwap<uint32_t>();

  bool is_memory = (wait_info & 0x10) != 0;

  bool matched = false;
  do {
    uint32_t value = 0;
    if (is_memory) {
      value =
          *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(poll_reg_addr & ~uint32_t(0x3)));
      value = xenos::GpuSwap(value, static_cast<xenos::Endian>(poll_reg_addr & 0x3));
    } else {
      value = ReadRegisterValue(poll_reg_addr);
      if (poll_reg_addr == XE_GPU_REG_COHER_STATUS_HOST) {
        MakeCoherent();
        value = ReadRegisterValue(poll_reg_addr);
      }
    }
    switch (wait_info & 0x7) {
      case 0x0:  // Never.
        matched = false;
        break;
      case 0x1:  // Less than reference.
        matched = (value & mask) < ref;
        break;
      case 0x2:  // Less than or equal to reference.
        matched = (value & mask) <= ref;
        break;
      case 0x3:  // Equal to reference.
        matched = (value & mask) == ref;
        break;
      case 0x4:  // Not equal to reference.
        matched = (value & mask) != ref;
        break;
      case 0x5:  // Greater than or equal to reference.
        matched = (value & mask) >= ref;
        break;
      case 0x6:  // Greater than reference.
        matched = (value & mask) > ref;
        break;
      case 0x7:  // Always
        matched = true;
        break;
    }
    if (!matched) {
      ServiceHeldSwap(false);
      // Wait.
      if (wait >= 0x100) {
        PrepareForWait();
        const uint64_t wait_start = rex::chrono::Clock::QueryHostTickCount();  // TEMP DIAGNOSTIC
        if (!REXCVAR_GET(vsync)) {
          // User wants it fast and dangerous.
          rex::thread::MaybeYield();
        } else {
          rex::thread::Sleep(std::chrono::milliseconds(wait / 0x100));
        }
        rex::thread::SyncMemory();
        interp_wait_ticks_ += rex::chrono::Clock::QueryHostTickCount() - wait_start;
        ReturnFromWait();

        if (!worker_running_) {
          // Short-circuited exit.
          return false;
        }
      } else {
        rex::thread::MaybeYield();
      }
    }
  } while (!matched);

  return true;
}

bool CommandProcessor::ExecutePacketType3_REG_RMW(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  // register read/modify/write
  // ? (used during shader upload and edram setup)
  uint32_t rmw_info = reader->ReadAndSwap<uint32_t>();
  uint32_t and_mask = reader->ReadAndSwap<uint32_t>();
  uint32_t or_mask = reader->ReadAndSwap<uint32_t>();
  uint32_t value = register_file_->values[rmw_info & 0x1FFF];
  if ((rmw_info >> 31) & 0x1) {
    // & reg
    value &= register_file_->values[and_mask & 0x1FFF];
  } else {
    // & imm
    value &= and_mask;
  }
  if ((rmw_info >> 30) & 0x1) {
    // | reg
    value |= register_file_->values[or_mask & 0x1FFF];
  } else {
    // | imm
    value |= or_mask;
  }
  WriteRegister(rmw_info & 0x1FFF, value);
  return true;
}

bool CommandProcessor::ExecutePacketType3_REG_TO_MEM(memory::RingBuffer* reader, uint32_t packet,
                                                     uint32_t count) {
  // Copy Register to Memory (?)
  // Count is 2, assuming a Register Addr and a Memory Addr.

  uint32_t reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t mem_addr = reader->ReadAndSwap<uint32_t>();

  uint32_t reg_val = ReadRegisterValue(reg_addr);

  auto endianness = static_cast<xenos::Endian>(mem_addr & 0x3);
  mem_addr &= ~0x3;
  reg_val = GpuSwap(reg_val, endianness);
  memory::store(memory_->TranslatePhysical(mem_addr), reg_val);

  return true;
}

bool CommandProcessor::ExecutePacketType3_MEM_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  uint32_t write_addr = reader->ReadAndSwap<uint32_t>();
  for (uint32_t i = 0; i < count - 1; i++) {
    uint32_t write_data = reader->ReadAndSwap<uint32_t>();

    auto endianness = static_cast<xenos::Endian>(write_addr & 0x3);
    auto addr = write_addr & ~0x3;
    write_data = GpuSwap(write_data, endianness);
    memory::store(memory_->TranslatePhysical(addr), write_data);
    write_addr += 4;
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3_COND_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                     uint32_t count) {
  // conditional write to memory or register
  uint32_t wait_info = reader->ReadAndSwap<uint32_t>();
  uint32_t poll_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t ref = reader->ReadAndSwap<uint32_t>();
  uint32_t mask = reader->ReadAndSwap<uint32_t>();
  uint32_t write_reg_addr = reader->ReadAndSwap<uint32_t>();
  uint32_t write_data = reader->ReadAndSwap<uint32_t>();
  uint32_t value;
  if (wait_info & 0x10) {
    // Memory.
    auto endianness = static_cast<xenos::Endian>(poll_reg_addr & 0x3);
    poll_reg_addr &= ~0x3;
    value = memory::load<uint32_t>(memory_->TranslatePhysical(poll_reg_addr));
    value = GpuSwap(value, endianness);
  } else {
    // Register.
    value = ReadRegisterValue(poll_reg_addr);
  }
  bool matched = false;
  switch (wait_info & 0x7) {
    case 0x0:  // Never.
      matched = false;
      break;
    case 0x1:  // Less than reference.
      matched = (value & mask) < ref;
      break;
    case 0x2:  // Less than or equal to reference.
      matched = (value & mask) <= ref;
      break;
    case 0x3:  // Equal to reference.
      matched = (value & mask) == ref;
      break;
    case 0x4:  // Not equal to reference.
      matched = (value & mask) != ref;
      break;
    case 0x5:  // Greater than or equal to reference.
      matched = (value & mask) >= ref;
      break;
    case 0x6:  // Greater than reference.
      matched = (value & mask) > ref;
      break;
    case 0x7:  // Always
      matched = true;
      break;
  }
  if (matched) {
    // Write.
    if (wait_info & 0x100) {
      // Memory.
      auto endianness = static_cast<xenos::Endian>(write_reg_addr & 0x3);
      write_reg_addr &= ~0x3;
      write_data = GpuSwap(write_data, endianness);
      memory::store(memory_->TranslatePhysical(write_reg_addr), write_data);
    } else {
      // Register.
      WriteRegister(write_reg_addr, write_data);
    }
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE(memory::RingBuffer* reader, uint32_t packet,
                                                      uint32_t count) {
  // generate an event that creates a write to memory when completed
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  if (count == 1) {
    // Just an event flag? Where does this write?
  } else {
    // Write to an address.
    assert_always();
    reader->AdvanceRead((count - 1) * sizeof(uint32_t));
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_SHD(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // generate a VS|PS_done event
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  uint32_t value = reader->ReadAndSwap<uint32_t>();

  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  uint32_t data_value;
  if ((initiator >> 31) & 0x1) {
    // Write counter (GPU vblank counter?).
    data_value = counter_;
  } else {
    // Write value.
    data_value = value;
  }
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;
  data_value = GpuSwap(data_value, endianness);
  memory::store(memory_->TranslatePhysical(address), data_value);
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_EXT(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // generate a screen extent event
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);
  auto endianness = static_cast<xenos::Endian>(address & 0x3);
  address &= ~0x3;

  // Let us hope we can fake this.
  // This callback tells the driver the xy coordinates affected by a previous
  // drawcall.
  // https://www.google.com/patents/US20060055701
  uint16_t extents[] = {
      0 >> 3,                                    // min x
      xenos::kTexture2DCubeMaxWidthHeight >> 3,  // max x
      0 >> 3,                                    // min y
      xenos::kTexture2DCubeMaxWidthHeight >> 3,  // max y
      0,                                         // min z
      1,                                         // max z
  };
  assert_true(endianness == xenos::Endian::k8in16);
  memory::copy_and_swap_16_unaligned(memory_->TranslatePhysical(address), extents,
                                     rex::countof(extents));
  return true;
}

bool CommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader,
                                                          uint32_t packet, uint32_t count) {
  // Set by D3D as BE but struct ABI is LE
  const uint32_t kQueryFinished = rex::byte_swap(0xFFFFFEED);
  assert_true(count == 1);
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  // Writeback initiator.
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);

  // Occlusion queries:
  // This command is send on query begin and end.
  // As a workaround report some fixed amount of passed samples.
  auto fake_sample_count = REXCVAR_GET(query_occlusion_fake_sample_count);
  if (fake_sample_count >= 0) {
    auto* pSampleCounts = memory_->TranslatePhysical<xe_gpu_depth_sample_counts*>(
        register_file_->values[XE_GPU_REG_RB_SAMPLE_COUNT_ADDR]);
    if (!pSampleCounts) {
      return true;
    }
    // 0xFFFFFEED is written to this two locations by D3D only on D3DISSUE_END
    // and used to detect a finished query.
    bool is_end_via_z_pass =
        pSampleCounts->ZPass_A == kQueryFinished || pSampleCounts->ZPass_B == kQueryFinished;
    // Older versions of D3D also checks for ZFail (4D5307D5).
    bool is_end_via_z_fail =
        pSampleCounts->ZFail_A == kQueryFinished || pSampleCounts->ZFail_B == kQueryFinished;
    std::memset(pSampleCounts, 0, sizeof(xe_gpu_depth_sample_counts));
    if (is_end_via_z_pass || is_end_via_z_fail) {
      pSampleCounts->ZPass_A = fake_sample_count;
      pSampleCounts->Total_A = fake_sample_count;
    }
  }

  return true;
}

bool CommandProcessor::ExecutePacketType3Draw(memory::RingBuffer* reader, uint32_t packet,
                                              const char* opcode_name, uint32_t viz_query_condition,
                                              uint32_t count_remaining) {
  // if viz_query_condition != 0, this is a conditional draw based on viz query.
  // This ID matches the one issued in PM4_VIZ_QUERY
  // uint32_t viz_id = viz_query_condition & 0x3F;
  // when true, render conditionally based on query result
  // uint32_t viz_use = viz_query_condition & 0x100;

  assert_not_zero(count_remaining);
  if (!count_remaining) {
    REXGPU_ERROR("{}: Packet too small, can't read VGT_DRAW_INITIATOR", opcode_name);
    return false;
  }
  reg::VGT_DRAW_INITIATOR vgt_draw_initiator;
  vgt_draw_initiator.value = reader->ReadAndSwap<uint32_t>();
  --count_remaining;
  WriteRegister(XE_GPU_REG_VGT_DRAW_INITIATOR, vgt_draw_initiator.value);

  bool draw_succeeded = true;
  // TODO(Triang3l): Remove IndexBufferInfo and replace handling of all this
  // with PrimitiveProcessor when the old Vulkan renderer is removed.
  bool is_indexed = false;
  IndexBufferInfo index_buffer_info;
  switch (vgt_draw_initiator.source_select) {
    case xenos::SourceSelect::kDMA: {
      // Indexed draw.
      is_indexed = true;

      // Two separate bounds checks so if there's only one missing register
      // value out of two, one uint32_t will be skipped in the command buffer,
      // not two.
      assert_not_zero(count_remaining);
      if (!count_remaining) {
        REXGPU_ERROR("{}: Packet too small, can't read VGT_DMA_BASE", opcode_name);
        return false;
      }
      uint32_t vgt_dma_base = reader->ReadAndSwap<uint32_t>();
      --count_remaining;
      WriteRegister(XE_GPU_REG_VGT_DMA_BASE, vgt_dma_base);
      reg::VGT_DMA_SIZE vgt_dma_size;
      assert_not_zero(count_remaining);
      if (!count_remaining) {
        REXGPU_ERROR("{}: Packet too small, can't read VGT_DMA_SIZE", opcode_name);
        return false;
      }
      vgt_dma_size.value = reader->ReadAndSwap<uint32_t>();
      --count_remaining;
      WriteRegister(XE_GPU_REG_VGT_DMA_SIZE, vgt_dma_size.value);

      uint32_t index_size_bytes = vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16
                                      ? sizeof(uint16_t)
                                      : sizeof(uint32_t);
      // The base address must already be word-aligned according to the R6xx
      // documentation, but for safety.
      index_buffer_info.guest_base = vgt_dma_base & ~(index_size_bytes - 1);
      index_buffer_info.endianness = vgt_dma_size.swap_mode;
      index_buffer_info.format = vgt_draw_initiator.index_size;
      index_buffer_info.length = vgt_dma_size.num_words * index_size_bytes;
      index_buffer_info.count = vgt_draw_initiator.num_indices;
    } break;
    case xenos::SourceSelect::kImmediate: {
      // TODO(Triang3l): VGT_IMMED_DATA.
      REXGPU_ERROR(
          "{}: Using immediate vertex indices, which are not supported yet. "
          "Report the game to Xenia developers!",
          opcode_name, uint32_t(vgt_draw_initiator.source_select));
      draw_succeeded = false;
      assert_always();
    } break;
    case xenos::SourceSelect::kAutoIndex: {
      // Auto draw.
      index_buffer_info.guest_base = 0;
      index_buffer_info.length = 0;
    } break;
    default: {
      // Invalid source selection.
      draw_succeeded = false;
      assert_unhandled_case(vgt_draw_initiator.source_select);
    } break;
  }

  // Skip to the next command, for example, if there are immediate indexes that
  // we don't support yet.
  reader->AdvanceRead(count_remaining * sizeof(uint32_t));

  if (draw_succeeded) {
    auto viz_query = register_file_->Get<reg::PA_SC_VIZ_QUERY>();
    if (!(viz_query.viz_query_ena && viz_query.kill_pix_post_hi_z)) {
      // TODO(Triang3l): Don't drop the draw call completely if the vertex
      // shader has memexport.
      // TODO(Triang3l || JoelLinn): Handle this properly in the render
      // backends.

      bool major_mode_explicit =
          xenos::IsMajorModeExplicit(vgt_draw_initiator.major_mode, vgt_draw_initiator.prim_type);
      draw_succeeded = IssueDraw(vgt_draw_initiator.prim_type, vgt_draw_initiator.num_indices,
                                 is_indexed ? &index_buffer_info : nullptr, major_mode_explicit);
      if (!draw_succeeded) {
        auto vgt_output_path_cntl = register_file_->Get<reg::VGT_OUTPUT_PATH_CNTL>();
        auto vgt_hos_cntl = register_file_->Get<reg::VGT_HOS_CNTL>();
        auto rb_modecontrol = register_file_->Get<reg::RB_MODECONTROL>();
        REXGPU_ERROR(
            "{}({}, {}, {}): Failed in backend "
            "(major_mode={}, explicit_major={}, path_select={}, tess_mode={}, edram_mode={})",
            opcode_name, static_cast<uint32_t>(vgt_draw_initiator.num_indices),
            uint32_t(vgt_draw_initiator.prim_type), uint32_t(vgt_draw_initiator.source_select),
            uint32_t(vgt_draw_initiator.major_mode), uint32_t(major_mode_explicit),
            uint32_t(vgt_output_path_cntl.path_select), uint32_t(vgt_hos_cntl.tess_mode),
            uint32_t(rb_modecontrol.edram_mode));
      }
    }
  }

  // If read the packed correctly, but merely couldn't execute it (because of,
  // for instance, features not supported by the host), don't terminate command
  // buffer processing as that would leave rendering in a way more inconsistent
  // state than just a single dropped draw command.
  return true;
}

bool CommandProcessor::ExecutePacketType3_DRAW_INDX(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  // "initiate fetch of index buffer and draw"
  // Generally used by Xbox 360 Direct3D 9 for kDMA and kAutoIndex sources.
  // With a viz query token as the first one.
  uint32_t count_remaining = count;
  assert_not_zero(count_remaining);
  if (!count_remaining) {
    REXGPU_ERROR("PM4_DRAW_INDX: Packet too small, can't read the viz query token");
    return false;
  }
  uint32_t viz_query_condition = reader->ReadAndSwap<uint32_t>();
  --count_remaining;
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX", viz_query_condition,
                                count_remaining);
}

bool CommandProcessor::ExecutePacketType3_DRAW_INDX_2(memory::RingBuffer* reader, uint32_t packet,
                                                      uint32_t count) {
  // "draw using supplied indices in packet"
  // Generally used by Xbox 360 Direct3D 9 for kAutoIndex source.
  // No viz query token.
  return ExecutePacketType3Draw(reader, packet, "PM4_DRAW_INDX_2", 0, count);
}

bool CommandProcessor::ExecutePacketType3_SET_CONSTANT(memory::RingBuffer* reader, uint32_t packet,
                                                       uint32_t count) {
  // load constant into chip and to memory
  // PM4_REG(reg) ((0x4 << 16) | (GSL_HAL_SUBBLOCK_OFFSET(reg)))
  //                                     reg - 0x2000
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t type = (offset_type >> 16) & 0xFF;
  uint32_t count_registers = count - 1;
  switch (type) {
    case 0:  // ALU
      WriteALURangeFromRing(reader, index, count_registers);
      break;
    case 1:  // FETCH
      WriteFetchRangeFromRing(reader, index, count_registers);
      break;
    case 2:  // BOOL
      WriteBoolRangeFromRing(reader, index, count_registers);
      break;
    case 3:  // LOOP
      WriteLoopRangeFromRing(reader, index, count_registers);
      break;
    case 4:  // REGISTERS
      WriteREGISTERSRangeFromRing(reader, index, count_registers);
      break;
    default:
      assert_always();
      reader->AdvanceRead((count - 1) * sizeof(uint32_t));
      return true;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_SET_CONSTANT2(memory::RingBuffer* reader, uint32_t packet,
                                                        uint32_t count) {
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  WriteRegisterRangeFromRing(reader, index, count - 1);
  return true;
}

bool CommandProcessor::ExecutePacketType3_LOAD_ALU_CONSTANT(memory::RingBuffer* reader,
                                                            uint32_t packet, uint32_t count) {
  // load constants from memory
  uint32_t address = reader->ReadAndSwap<uint32_t>();
  address &= 0x3FFFFFFF;
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0x7FF;
  uint32_t size_dwords = reader->ReadAndSwap<uint32_t>();
  size_dwords &= 0xFFF;
  uint32_t type = (offset_type >> 16) & 0xFF;
  uint32_t* xlat_address = memory_->TranslatePhysical<uint32_t*>(address);
  switch (type) {
    case 0:  // ALU
      WriteALURangeFromMem(index, xlat_address, size_dwords);
      break;
    case 1:  // FETCH
      WriteFetchRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 2:  // BOOL
      WriteBoolRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 3:  // LOOP
      WriteLoopRangeFromMem(index, xlat_address, size_dwords);
      break;
    case 4:  // REGISTERS
      WriteREGISTERSRangeFromMem(index, xlat_address, size_dwords);
      break;
    default:
      assert_always();
      return true;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_SET_SHADER_CONSTANTS(memory::RingBuffer* reader,
                                                               uint32_t packet, uint32_t count) {
  uint32_t offset_type = reader->ReadAndSwap<uint32_t>();
  uint32_t index = offset_type & 0xFFFF;
  WriteRegisterRangeFromRing(reader, index, count - 1);
  return true;
}

bool CommandProcessor::ExecutePacketType3_IM_LOAD(memory::RingBuffer* reader, uint32_t packet,
                                                  uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (pointer-based)
  uint32_t addr_type = reader->ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(addr_type & 0x3);
  uint32_t addr = addr_type & ~0x3;
  uint32_t start_size = reader->ReadAndSwap<uint32_t>();
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);

  auto shader =
      LoadShader(shader_type, addr, memory_->TranslatePhysical<uint32_t*>(addr), size_dwords);
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  return true;
}

bool CommandProcessor::ExecutePacketType3_IM_LOAD_IMMEDIATE(memory::RingBuffer* reader,
                                                            uint32_t packet, uint32_t count) {
  SCOPE_profile_cpu_f("gpu");

  // load sequencer instruction memory (code embedded in packet)
  uint32_t dword0 = reader->ReadAndSwap<uint32_t>();
  uint32_t dword1 = reader->ReadAndSwap<uint32_t>();
  auto shader_type = static_cast<xenos::ShaderType>(dword0);
  uint32_t start_size = dword1;
  uint32_t start = start_size >> 16;
  uint32_t size_dwords = start_size & 0xFFFF;  // dwords
  assert_true(start == 0);
  assert_true(reader->read_count() >= size_dwords * 4);
  assert_true(count - 2 >= size_dwords);
  auto shader = LoadShader(shader_type, uint32_t(reader->read_ptr()),
                           reinterpret_cast<uint32_t*>(reader->read_ptr()), size_dwords);
  switch (shader_type) {
    case xenos::ShaderType::kVertex:
      active_vertex_shader_ = shader;
      break;
    case xenos::ShaderType::kPixel:
      active_pixel_shader_ = shader;
      break;
    default:
      assert_unhandled_case(shader_type);
      return false;
  }
  reader->AdvanceRead(size_dwords * sizeof(uint32_t));
  return true;
}

bool CommandProcessor::ExecutePacketType3_INVALIDATE_STATE(memory::RingBuffer* reader,
                                                           uint32_t packet, uint32_t count) {
  // selective invalidation of state pointers
  /*uint32_t mask =*/reader->ReadAndSwap<uint32_t>();
  // driver_->InvalidateState(mask);
  return true;
}

bool CommandProcessor::ExecutePacketType3_VIZ_QUERY(memory::RingBuffer* reader, uint32_t packet,
                                                    uint32_t count) {
  // begin/end initiator for viz query extent processing
  // https://www.google.com/patents/US20050195186
  assert_true(count == 1);

  uint32_t dword0 = reader->ReadAndSwap<uint32_t>();

  uint32_t id = dword0 & 0x3F;
  uint32_t end = dword0 & 0x100;
  if (!end) {
    // begin a new viz query @ id
    // On hardware this clears the internal state of the scan converter (which
    // is different to the register)
    WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, VIZQUERY_START);
    REXGPU_INFO("Begin viz query ID {:02X}", id);
  } else {
    // end the viz query
    WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, VIZQUERY_END);
    REXGPU_INFO("End viz query ID {:02X}", id);
    // The scan converter writes the internal result back to the register here.
    // We just fake it and say it was visible in case it is read back.
    if (id < 32) {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_0] |= uint32_t(1) << id;
    } else {
      register_file_->values[XE_GPU_REG_PA_SC_VIZ_QUERY_STATUS_1] |= uint32_t(1) << (id - 32);
    }
  }

  return true;
}

}  // namespace rex::graphics
