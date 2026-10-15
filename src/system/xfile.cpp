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

#include <rex/cvar.h>
#include <rex/filesystem/vfs.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory.h>
#include <rex/stream.h>
#include <rex/system/kernel_state.h>
#include <rex/system/movie_skip.h>
#include <rex/system/xevent.h>
#include <rex/system/xfile.h>
#include <rex/system/xthread.h>
#include <rex/thread/mutex.h>

#include <span>

REXCVAR_DEFINE_BOOL(movie_skip, true, "Movie",
                    "Allow ending FMV playback early: while an .sfd movie "
                    "streams, any fresh key or button press forces the guest "
                    "movie player's end state and the game takes its own "
                    "movie-finished path");
REXCVAR_DEFINE_BOOL(skip_intro, false, "Movie",
                    "Skip the boot intro: the first .sfd movie opened ends at once "
                    "(the game skips its own disclaimer when it reads this too)");

namespace rex::system {

namespace {

// True for SOFDEC .sfd movie files (the same set the SFDOPEN probe in the VFS
// reports). Deliberately extension-exact: a substring match on "movie" would
// also catch assets like D:\dat\sprite\movie\spmovie.xpr, and serving those
// EOF would corrupt scene loads.
bool IsMovieFilePath(const std::string& path) {
  if (path.size() < 4) {
    return false;
  }
  const char* ext = path.c_str() + path.size() - 4;
  return ext[0] == '.' && (ext[1] == 's' || ext[1] == 'S') &&
         (ext[2] == 'f' || ext[2] == 'F') && (ext[3] == 'd' || ext[3] == 'D');
}

// Retail DOA4 (doa4.xex): live MWPLY (SOFDEC wrapper) movie-player handle.
// Set by the movie open path (sub_8242A2C8 via sub_8242A168), cleared by the
// stop path (sub_8242B6A8 via sub_8242A0F0, which writes through
// &dword_833F8070). Object layout: magic dword at +0 (must read 1 for
// mwPlyGetStat to trust the handle), status dword at +4 (mwPlyGetStat,
// 0x829D06E0: 'lwz r3,4(r10)' after '*handle == 1' check). mwPlyGetStat
// returns any status != 2 verbatim, and the movie scene (sub_824046D8 case
// 3) treats anything outside {1,2} as finished via sub_8242A3A8 and runs its
// own stop -> fade -> next-scene path (sub_8242A0F0, then 90-frame wait).
constexpr uint32_t kMwPlyHandleGuestAddr = 0x833F8070u;
constexpr uint32_t kMwPlyStatusEnd = 4u;
// mwPlyGetStat's handle-validity check: *handle must read 1 before the
// object is trusted. Used here as a write guard - the SDK (and this DLL)
// ships to other titles (UFC3, E3 demo) whose layouts differ, and 0x833F8070
// there holds unrelated data, so a magic mismatch must never be written to.
constexpr uint32_t kMwPlyMagicValid = 1u;

}  // namespace

XFile::XFile(KernelState* kernel_state, rex::filesystem::File* file, bool synchronous)
    : XObject(kernel_state, kObjectType), file_(file), is_synchronous_(synchronous) {
  async_event_ = rex::thread::Event::CreateAutoResetEvent(false);
  assert_not_null(async_event_);
  // FMV skip: remember whether this is a movie file and arm a fresh skip
  // window so input aimed at the disclaimer (or an earlier movie) does not
  // skip this playback. entry() is guarded because the Restore path
  // constructs XFile without a file.
  if (file_ && file_->entry()) {
    is_movie_file_ = IsMovieFilePath(file_->entry()->path());
    if (is_movie_file_) {
      if (movie_skip::IsRequested()) {
        // Diagnostic: a request was already pending when this movie opened -
        // normal if it was aimed at the disclaimer (log it to prove the
        // discard), a leak warning if it appears in the next MOVIESKIP line.
        REXLOG_ERROR("MOVIESKIP discarding pre-open request (from '{}') on '{}'",
                     movie_skip::Source(), path());
      }
      movie_skip::Reset();
      // The first movie a title opens is its boot intro.
      static std::atomic<bool> s_first_movie_seen{false};
      if (!s_first_movie_seen.exchange(true) && REXCVAR_GET(skip_intro) &&
          REXCVAR_GET(movie_skip)) {
        movie_skip::Request("skip_intro");
      }
    }
  }
}

XFile::XFile() : XObject(kObjectType) {
  async_event_ = rex::thread::Event::CreateAutoResetEvent(false);
  assert_not_null(async_event_);
}

XFile::~XFile() {
  // TODO(benvanik): signal that the file is closing?
  async_event_->Set();
  file_->Destroy();
}

uint64_t XFile::position() const {
  std::lock_guard<std::mutex> lock(file_lock_);
  return position_;
}

void XFile::set_position(uint64_t value) {
  std::lock_guard<std::mutex> lock(file_lock_);
  position_ = value;
}

X_STATUS XFile::QueryDirectory(X_FILE_DIRECTORY_INFORMATION* out_info, size_t length,
                               const std::string_view file_name, bool restart) {
  std::lock_guard<std::mutex> lock(file_lock_);
  assert_not_null(out_info);

  rex::filesystem::Entry* entry = nullptr;

  if (!file_name.empty()) {
    // Only queries in the current directory are supported for now.
    assert_true(rex::string::utf8_find_any_of(file_name, "\\") == std::string_view::npos);

    find_engine_.SetRule(file_name);

    // Always restart the search?
    find_index_ = 0;
    entry = file_->entry()->IterateChildren(find_engine_, &find_index_);
    if (!entry) {
      return X_STATUS_NO_SUCH_FILE;
    }
  } else {
    if (restart) {
      find_index_ = 0;
    }

    entry = file_->entry()->IterateChildren(find_engine_, &find_index_);
    if (!entry) {
      return X_STATUS_NO_MORE_FILES;
    }
  }

  auto end = reinterpret_cast<uint8_t*>(out_info) + length;
  const auto& entry_name = entry->name();
  if (reinterpret_cast<uint8_t*>(&out_info->file_name[0]) + entry_name.size() > end) {
    assert_always("Buffer overflow?");
    return X_STATUS_NO_SUCH_FILE;
  }

  out_info->next_entry_offset = 0;
  out_info->file_index = static_cast<uint32_t>(find_index_);
  out_info->creation_time = entry->create_timestamp();
  out_info->last_access_time = entry->access_timestamp();
  out_info->last_write_time = entry->write_timestamp();
  out_info->change_time = entry->write_timestamp();
  out_info->end_of_file = entry->size();
  out_info->allocation_size = entry->allocation_size();
  out_info->attributes = entry->attributes();
  out_info->file_name_length = static_cast<uint32_t>(entry_name.size());
  std::memcpy(out_info->file_name, entry_name.data(), entry_name.size());

  return X_STATUS_SUCCESS;
}

X_STATUS XFile::Read(uint32_t buffer_guest_address, uint32_t buffer_length, uint64_t byte_offset,
                     uint32_t* out_bytes_read, uint32_t apc_context, bool notify_completion) {
  std::lock_guard<std::mutex> lock(file_lock_);
  // TEMP DIAGNOSTIC (retail DOA4 stage-preview hitch hunt, R93+): a felt
  // 150ms stutter at stage-select/main-menu preview switches shows as
  // 3x 44-51ms frames with normal dispatch counts and no upload spike -
  // i.e. the calling guest thread busy or blocked somewhere a kernel-wait
  // probe cannot see. Sync file I/O on that path is the prime suspect; this
  // logs ANY guest read blocking >2ms (host disk/decode work) with the
  // calling guest thread. Bounded: first 60 slow ones + every 100th.
  // Remove after root cause.
  const bool diag_on = rex::diag::enabled();  // TEMP probe gate (REX_DIAG=0)
  const auto sr_t0 = diag_on ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  X_STATUS result = ReadInternal(buffer_guest_address, buffer_length, byte_offset, out_bytes_read,
                                 apc_context, notify_completion);
  if (diag_on) {
    const int64_t sr_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                              sr_t0)
            .count();
    if (sr_us > 2000) {
      static uint32_t s_slowread_n = 0;
      if (s_slowread_n < 60 || (s_slowread_n % 100) == 0) {
        auto thread = XThread::GetCurrentThread();
        REXLOG_ERROR("SLOWREAD n={} tid={:08X} dur_us={} h={:#x} len=0x{:X} off=0x{:X} got={} st={:#x}",
                     s_slowread_n++, thread ? thread->guest_object() : 0u, sr_us, handle(),
                     buffer_length, byte_offset, out_bytes_read ? *out_bytes_read : 0u,
                     static_cast<uint32_t>(result));
      } else {
        s_slowread_n++;
      }
    }
  }
  // TEMP DIAGNOSTIC (issue #7 E3 walker AV): log every read's handle, guest
  // destination, length, offset, byte count and status so the exact slot-file
  // delivery order/sizes can be reconstructed against the GUEST FOPEN lines
  // (the slot pump opens and reads strictly sequentially). Especially: did
  // block(1) at 0x880000 land before/after window(0)'s 0x23000-byte spill at
  // 0x860000, and with what status. Remove after a confirmed passing playtest.
  {
    static uint32_t s_fread_n = 0;
    if (rex::diag::enabled() && s_fread_n < 512) {
      REXLOG_ERROR("GUEST FREAD n={} h={:#x} dst=0x{:08X} len=0x{:X} off=0x{:X} got={} st={:#x}",
                   s_fread_n++, handle(), buffer_guest_address, buffer_length, byte_offset,
                   out_bytes_read ? *out_bytes_read : 0u, static_cast<uint32_t>(result));
    }
  }
  return result;
}

X_STATUS XFile::ReadInternal(uint32_t buffer_guest_address, uint32_t buffer_length,
                             uint64_t byte_offset, uint32_t* out_bytes_read, uint32_t apc_context,
                             bool notify_completion) {
  if (byte_offset == uint64_t(-1)) {
    // Read from current position.
    byte_offset = position_;
  }

  size_t bytes_read = 0;
  X_STATUS result = X_STATUS_SUCCESS;
  // Zero length means success for a valid file object according to Windows
  // tests.
  if (buffer_length) {
    if (UINT32_MAX - buffer_guest_address < buffer_length) {
      result = X_STATUS_ACCESS_VIOLATION;
    } else {
      if (is_movie_file_ && REXCVAR_GET(movie_skip) && movie_skip::IsRequested() &&
          !movie_skip_poked_) {
        // FMV skip (user request): force the guest's own end-of-movie state
        // instead of faking EOF - SOFDEC never treats zero-byte reads as
        // end-of-stream (R45: 250 EOF reads served, no XMP stop, black movie
        // hung for the whole 90 s run). Retail DOA4 keeps the live MWPLY
        // handle in dword_833F8070; the movie scene polls mwPlyGetStat
        // (sub_8242A3A8) every frame and treats any status outside {1,2} as
        // finished, then runs its own stop -> fade -> next-scene path. 4 =
        // END/ERROR passes the game's GetStat wrapper (sub_829D0A68 only
        // overrides statuses 1/3 under streaming-standby fields). Reads keep
        // serving real file data below, so the decoder never stalls while
        // the guest winds down. ReadScatter funnels through here per segment.
        // Guest memory is big-endian (PPC), so all dword access must go
        // through load/store_and_swap - R46 poked a byte-swapped handle
        // (0x835A26AC read as 0xAC265A83) and wrote into thin air.
        uint32_t handle = memory::load_and_swap<uint32_t>(
            memory()->TranslateVirtual(kMwPlyHandleGuestAddr));
        uint32_t magic = 0;
        if (handle && memory()->LookupHeap(handle) && memory()->LookupHeap(handle + 4)) {
          magic = memory::load_and_swap<uint32_t>(
              reinterpret_cast<uint32_t*>(memory()->TranslateVirtual(handle)));
        }
        if (magic == kMwPlyMagicValid) {
          uint32_t* status_host =
              reinterpret_cast<uint32_t*>(memory()->TranslateVirtual(handle + 4));
          uint32_t status_pre = memory::load_and_swap<uint32_t>(status_host);
          memory::store_and_swap<uint32_t>(status_host, kMwPlyStatusEnd);
          movie_skip_poked_ = true;
          REXLOG_ERROR(
              "MOVIESKIP forcing mwPly status END on '{}' (handle=0x{:08X} magic=0x{:08X} "
              "status {}->{}, request from '{}')",
              path(), handle, magic, status_pre, kMwPlyStatusEnd, movie_skip::Source());
        } else {
          // No live player object: either no movie handle is open or this
          // build doesn't lay the player out at the retail address (the DLL
          // ships to other titles - UFC3/E3 - so a magic mismatch must stay
          // inert instead of writing into unrelated guest memory). Not marked
          // poked: a request made at open (skip_intro) can arrive before the
          // player object is live, so later reads retry.
          if (rex::diag::enabled())
            REXLOG_ERROR(
              "MOVIESKIP requested on '{}' but no live mwPly object "
              "(handle=0x{:08X} magic=0x{:08X}), not poking",
              path(), handle, magic);
        }
      }

      // Games often read directly to texture/vertex buffer memory - in this
      // case, invalidation notifications must be sent. However, having any
      // memory callbacks in the range will result in STATUS_ACCESS_VIOLATION at
      // least on Windows, without anything being read or any callbacks being
      // triggered. So for physical memory, host protection must be bypassed,
      // and invalidation callbacks must be triggered manually (it's also wrong
      // to trigger invalidation callbacks before reading in this case, because
      // during the read, the guest may still access the data around the buffer
      // that is located in the same host pages as the buffer's start and end,
      // on the GPU - and that must not trigger a race condition).
      uint32_t buffer_guest_high_address = buffer_guest_address + buffer_length - 1;
      rex::memory::BaseHeap* buffer_start_heap = memory()->LookupHeap(buffer_guest_address);
      const rex::memory::BaseHeap* buffer_end_heap =
          memory()->LookupHeap(buffer_guest_high_address);
      if (!buffer_start_heap || !buffer_end_heap ||
          (buffer_start_heap->heap_type() == memory::HeapType::kGuestPhysical) !=
              (buffer_end_heap->heap_type() == memory::HeapType::kGuestPhysical) ||
          (buffer_start_heap->heap_type() == memory::HeapType::kGuestPhysical &&
           buffer_start_heap != buffer_end_heap)) {
        result = X_STATUS_ACCESS_VIOLATION;
      } else {
        rex::memory::PhysicalHeap* buffer_physical_heap =
            buffer_start_heap->heap_type() == memory::HeapType::kGuestPhysical
                ? static_cast<rex::memory::PhysicalHeap*>(buffer_start_heap)
                : nullptr;
        if (buffer_physical_heap && buffer_physical_heap->QueryRangeAccess(
                                        buffer_guest_address, buffer_guest_high_address) !=
                                        memory::PageAccess::kReadWrite) {
          result = X_STATUS_ACCESS_VIOLATION;
        } else {
          result = file_->ReadSync(
              std::span<uint8_t>(
                  buffer_physical_heap
                      ? memory()->TranslatePhysical(
                            buffer_physical_heap->GetPhysicalAddress(buffer_guest_address))
                      : memory()->TranslateVirtual(buffer_guest_address),
                  buffer_length),
              size_t(byte_offset), &bytes_read);
          if (XSUCCEEDED(result)) {
            if (buffer_physical_heap) {
              buffer_physical_heap->TriggerCallbacks(
                  rex::thread::global_critical_region::AcquireDirect(), buffer_guest_address,
                  buffer_length, true, true);
            }

            if (byte_offset) {
              position_ = byte_offset;
            }
            position_ += bytes_read;
          }
        }
      }
    }
  }

  if (out_bytes_read) {
    *out_bytes_read = uint32_t(bytes_read);
  }

  if (notify_completion) {
    XIOCompletion::IONotification notify;
    notify.apc_context = apc_context;
    notify.num_bytes = uint32_t(bytes_read);
    notify.status = result;

    NotifyIOCompletionPorts(notify);

    async_event_->Set();
  }

  return result;
}

X_STATUS XFile::ReadScatter(uint32_t segments_guest_address, uint32_t length, uint64_t byte_offset,
                            uint32_t* out_bytes_read, uint32_t apc_context) {
  std::lock_guard<std::mutex> lock(file_lock_);
  X_STATUS result = X_STATUS_SUCCESS;

  // segments points to an array of buffer pointers of type
  // "FILE_SEGMENT_ELEMENT", but they can just be treated as normal pointers
  rex::be<uint32_t>* segments =
      reinterpret_cast<rex::be<uint32_t>*>(memory()->TranslateVirtual(segments_guest_address));

  // TODO: not sure if this is meant to change depending on buffer address?
  // (only game seen using this always seems to use 4096-byte buffers)
  uint32_t page_size = 4096;

  uint32_t read_total = 0;
  uint32_t read_remain = length;
  while (read_remain) {
    uint32_t read_length = read_remain;
    uint32_t read_buffer = *segments;
    if (read_length > page_size) {
      read_length = page_size;
      segments++;
    }

    uint32_t bytes_read = 0;
    result = ReadInternal(
        read_buffer, read_length,
        byte_offset
            ? ((byte_offset != -1 && byte_offset != -2) ? byte_offset + read_total : byte_offset)
            : -1,
        &bytes_read, apc_context, false);

    if (result != X_STATUS_SUCCESS) {
      break;
    }

    read_total += bytes_read;
    read_remain -= read_length;
  }

  if (out_bytes_read) {
    *out_bytes_read = uint32_t(read_total);
  }

  XIOCompletion::IONotification notify;
  notify.apc_context = apc_context;
  notify.num_bytes = uint32_t(read_total);
  notify.status = result;

  NotifyIOCompletionPorts(notify);

  async_event_->Set();

  return result;
}

X_STATUS XFile::Write(uint32_t buffer_guest_address, uint32_t buffer_length, uint64_t byte_offset,
                      uint32_t* out_bytes_written, uint32_t apc_context) {
  std::lock_guard<std::mutex> lock(file_lock_);
  if (byte_offset == uint64_t(-1)) {
    // Write from current position.
    byte_offset = position_;
  }

  size_t bytes_written = 0;
  X_STATUS result = file_->WriteSync(
      std::span<const uint8_t>(memory()->TranslateVirtual(buffer_guest_address), buffer_length),
      size_t(byte_offset), &bytes_written);
  if (XSUCCEEDED(result)) {
    position_ += bytes_written;
  }

  XIOCompletion::IONotification notify;
  notify.apc_context = apc_context;
  notify.num_bytes = uint32_t(bytes_written);
  notify.status = result;

  NotifyIOCompletionPorts(notify);

  if (out_bytes_written) {
    *out_bytes_written = uint32_t(bytes_written);
  }

  async_event_->Set();
  return result;
}

X_STATUS XFile::SetLength(size_t length) {
  std::lock_guard<std::mutex> lock(file_lock_);
  return file_->SetLength(length);
}

X_STATUS XFile::Rename(const std::filesystem::path& file_path) {
  std::lock_guard<std::mutex> lock(file_lock_);
  return entry()->Rename(file_path);
}

void XFile::RegisterIOCompletionPort(uint32_t key, object_ref<XIOCompletion> port) {
  std::lock_guard<std::mutex> lock(completion_port_lock_);

  completion_ports_.push_back({key, port});
}

void XFile::RemoveIOCompletionPort(uint32_t key) {
  std::lock_guard<std::mutex> lock(completion_port_lock_);

  for (auto it = completion_ports_.begin(); it != completion_ports_.end(); it++) {
    if (it->first == key) {
      completion_ports_.erase(it);
      break;
    }
  }
}

bool XFile::Save(stream::ByteStream* stream) {
  REXSYS_DEBUG("XFile {:08X} ({})", handle(), file_->entry()->absolute_path().c_str());

  if (!SaveObject(stream)) {
    return false;
  }

  stream->Write(file_->entry()->absolute_path());
  stream->Write<uint64_t>(position_);
  stream->Write(file_access());
  stream->Write<bool>((file_->entry()->attributes() & rex::filesystem::kFileAttributeDirectory) !=
                      0);
  stream->Write<bool>(is_synchronous_);

  return true;
}

object_ref<XFile> XFile::Restore(KernelState* kernel_state, stream::ByteStream* stream) {
  auto file = new XFile();
  file->kernel_state_ = kernel_state;
  if (!file->RestoreObject(stream)) {
    delete file;
    return nullptr;
  }

  auto abs_path = stream->Read<std::string>();
  uint64_t position = stream->Read<uint64_t>();
  auto access = stream->Read<uint32_t>();
  auto is_directory = stream->Read<bool>();
  auto is_synchronous = stream->Read<bool>();

  REXSYS_DEBUG("XFile {:08X} ({})", file->handle(), abs_path);

  rex::filesystem::File* vfs_file = nullptr;
  rex::filesystem::FileAction action;
  auto res = kernel_state->file_system()->OpenFile(nullptr, abs_path,
                                                   rex::filesystem::FileDisposition::kOpen, access,
                                                   is_directory, false, &vfs_file, &action);
  if (XFAILED(res)) {
    REXSYS_ERROR("Failed to open XFile: error {:08X}", res);
    return object_ref<XFile>(file);
  }

  file->file_ = vfs_file;
  file->position_ = position;
  file->is_synchronous_ = is_synchronous;

  return object_ref<XFile>(file);
}

void XFile::NotifyIOCompletionPorts(XIOCompletion::IONotification& notification) {
  std::lock_guard<std::mutex> lock(completion_port_lock_);

  for (auto port : completion_ports_) {
    notification.key_context = port.first;
    port.second->QueueNotification(notification);
  }
}

}  // namespace rex::system
