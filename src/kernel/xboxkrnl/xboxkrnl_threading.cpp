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

// Disable warnings about unused parameters for kernel functions
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <rex/chrono/clock.h>
#include <rex/dbg.h>
#include <rex/kernel/xboxkrnl/private.h>
#include <rex/kernel/xboxkrnl/threading.h>
#include <rex/logging.h>
#include <rex/hook.h>
#include <rex/types.h>
#include <rex/system/kernel_state.h>
#include <rex/system/function_dispatcher.h>
#include <rex/system/thread_state.h>
#include <rex/system/user_module.h>
#include <rex/system/util/string_utils.h>
#include <rex/system/xevent.h>
#include <rex/system/xmutant.h>
#include <rex/system/xsemaphore.h>
#include <rex/system/xthread.h>
#include <rex/system/xtimer.h>
#include <rex/system/xtypes.h>
#include <rex/thread/atomic.h>
#include <rex/thread/fiber.h>
#include <rex/thread/mutex.h>

namespace rex::kernel::xboxkrnl {
using namespace rex::system;
using rex::runtime::current_ppc_context;

// TEMP DIAGNOSTIC (retail DOA4 R78): defined next to the wait entries below;
// forward-declared so KeDelayExecutionThread_entry can log through it too.
static void LogLongWait(const char* api, uint32_t tid, int64_t dur_us, uint32_t obj_addr,
                        int64_t timeout_100ns, uint32_t lr);

// r13 + 0x100: pointer to thread local state
// Thread local state:
//   0x058: kernel time
//   0x14C: thread id
//   0x150: if >0 then error states don't get set
//   0x160: last error

// GetCurrentThreadId:
// lwz       r11, 0x100(r13)
// lwz       r3, 0x14C(r11)

// RtlGetLastError:
// lwz r11, 0x150(r13)
// if (r11 == 0) {
//   lwz r11, 0x100(r13)
//   stw r3, 0x160(r11)
// }

// RtlSetLastError:
// lwz r11, 0x150(r13)
// if (r11 == 0) {
//   lwz r11, 0x100(r13)
//   stw r3, 0x160(r11)
// }

// RtlSetLastNTError:
// r3 = RtlNtStatusToDosError(r3)
// lwz r11, 0x150(r13)
// if (r11 == 0) {
//   lwz r11, 0x100(r13)
//   stw r3, 0x160(r11)
// }

template <typename T>
object_ref<T> LookupNamedObject(KernelState* kernel_state, uint32_t obj_attributes_ptr) {
  // If the name exists and its type matches, we can return that (ref+1)
  // with a success of NAME_EXISTS.
  // If the name exists and its type doesn't match, we do NAME_COLLISION.
  // Otherwise, we add like normal.
  if (!obj_attributes_ptr) {
    return nullptr;
  }
  auto obj_attributes =
      kernel_state->memory()->TranslateVirtual<X_OBJECT_ATTRIBUTES*>(obj_attributes_ptr);
  assert_true(obj_attributes->name_ptr != 0);
  auto name = util::TranslateAnsiStringAddress(kernel_state->memory(), obj_attributes->name_ptr);
  if (!name.empty()) {
    X_HANDLE handle = X_INVALID_HANDLE_VALUE;
    X_RESULT result = kernel_state->object_table()->GetObjectByName(name, &handle);
    if (XSUCCEEDED(result)) {
      // Found something! It's been retained, so return.
      auto obj = kernel_state->object_table()->LookupObject<T>(handle);
      if (obj) {
        // The caller will do as it likes.
        obj->ReleaseHandle();
        return obj;
      }
    }
  }
  return nullptr;
}

u32 ExCreateThread_entry(mapped_u32 handle_ptr, u32 stack_size, mapped_u32 thread_id_ptr,
                         u32 xapi_thread_startup, mapped_void start_address,
                         mapped_void start_context, u32 creation_flags) {
  REXKRNL_IMPORT_TRACE(
      "ExCreateThread", "stack={:#x} xapi_startup={:#x} start={:#x} context={:#x} flags={:#x}",
      (uint32_t)stack_size, (uint32_t)xapi_thread_startup, start_address.guest_address(),
      start_context.guest_address(), (uint32_t)creation_flags);
  // http://jafile.com/uploads/scoop/main.cpp.txt
  // DWORD
  // LPHANDLE Handle,
  // DWORD    StackSize,
  // LPDWORD  ThreadId,
  // LPVOID   XapiThreadStartup, ?? often 0
  // LPVOID   StartAddress,
  // LPVOID   StartContext,
  // DWORD    CreationFlags // 0x80?

  // Determine target process based on creation flags.
  uint32_t guest_process = REX_KERNEL_STATE()->GetTitleProcess();
  if (creation_flags & 2) {
    REXKRNL_WARN("[ExCreateThread] Guest is creating a system thread!");
    guest_process = REX_KERNEL_STATE()->GetSystemProcess();
  }

  // Inherit default stack size
  uint32_t actual_stack_size = stack_size;

  if (actual_stack_size == 0) {
    actual_stack_size = REX_KERNEL_STATE()->GetExecutableModule()->stack_size();
  }

  // Stack must be aligned to 16kb pages
  actual_stack_size = std::max((uint32_t)0x4000, ((actual_stack_size + 0xFFF) & 0xFFFFF000));

  auto thread = object_ref<XThread>(new XThread(
      REX_KERNEL_STATE(), actual_stack_size, xapi_thread_startup, start_address.guest_address(),
      start_context.guest_address(), creation_flags, true, false, guest_process));

  X_STATUS result = thread->Create();
  if (XFAILED(result)) {
    // Failed!
    REXKRNL_ERROR("Thread creation failed: {:08X}", result);
    return result;
  }

  if (XSUCCEEDED(result)) {
    if (handle_ptr) {
      if (creation_flags & 0x80) {
        *handle_ptr = thread->guest_object();
      } else {
        *handle_ptr = thread->handle();
      }
    }
    if (thread_id_ptr) {
      *thread_id_ptr = thread->thread_id();
    }
  }
  REXKRNL_IMPORT_RESULT("ExCreateThread", "{:#x} handle={:#x} tid={}", result,
                        handle_ptr ? (uint32_t)*handle_ptr : 0,
                        thread_id_ptr ? (uint32_t)*thread_id_ptr : 0);
  return result;
}

u32 ExTerminateThread_entry(u32 exit_code) {
  XThread* thread = XThread::GetCurrentThread();

  // NOTE: this kills us right now. We won't return from it.
  return thread->Exit(exit_code);
}

u32 NtResumeThread_entry(u32 handle, mapped_u32 suspend_count_ptr) {
  X_RESULT result = X_STATUS_INVALID_HANDLE;
  uint32_t suspend_count = 0;

  auto thread = REX_KERNEL_OBJECTS()->LookupObject<XThread>(handle);
  if (thread) {
    REXKRNL_TRACE("[NtResumeThread] handle={:08X} thread={}", uint32_t(handle), thread->name());
    result = thread->Resume(&suspend_count);
  } else {
    REXKRNL_WARN("[NtResumeThread] handle={:08X} NOT FOUND", uint32_t(handle));
  }
  if (suspend_count_ptr) {
    *suspend_count_ptr = suspend_count;
  }

  REXKRNL_TRACE("[NtResumeThread] -> {:#x} suspend_count={}", result, suspend_count);
  return result;
}

u32 KeResumeThread_entry(mapped_void thread_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;
  auto thread = XObject::GetNativeObject<XThread>(REX_KERNEL_STATE(), thread_ptr);
  if (thread) {
    REXKRNL_TRACE("[KeResumeThread] ptr={:08X} thread={}", thread_ptr.guest_address(),
                  thread->name());
    result = thread->Resume();
  } else {
    REXKRNL_WARN("[KeResumeThread] ptr={:08X} NOT FOUND", thread_ptr.guest_address());
    result = X_STATUS_INVALID_HANDLE;
  }

  REXKRNL_TRACE("[KeResumeThread] -> {:#x}", result);
  return result;
}

u32 NtSuspendThread_entry(u32 handle, mapped_u32 suspend_count_ptr) {
  X_RESULT result = X_STATUS_SUCCESS;
  uint32_t suspend_count = 0;

  auto thread = REX_KERNEL_OBJECTS()->LookupObject<XThread>(handle);
  if (thread) {
    if (thread->type() != XObject::Type::Thread) {
      return X_STATUS_OBJECT_TYPE_MISMATCH;
    }

    auto guest_thread = thread->guest_object<X_KTHREAD>();
    if (guest_thread->terminated) {
      return X_STATUS_THREAD_IS_TERMINATING;
    }

    REXKRNL_TRACE("[NtSuspendThread] handle={:08X} thread={}", uint32_t(handle), thread->name());
#if REX_PLATFORM_LINUX || REX_PLATFORM_MAC
    auto* current_thread = XThread::GetCurrentThread();
    bool is_self_suspend = current_thread && current_thread == thread.get();
    if (is_self_suspend) {
      REXKRNL_TRACE("[NtSuspendThread] self-suspend for thread={:08X}", uint32_t(handle));
      suspend_count = thread->SelfSuspend();
      result = X_STATUS_SUCCESS;
    } else {
      result = thread->Suspend(&suspend_count);
    }
#else
    result = thread->Suspend(&suspend_count);
#endif
  } else {
    REXKRNL_WARN("[NtSuspendThread] handle={:08X} NOT FOUND", uint32_t(handle));
    result = X_STATUS_INVALID_HANDLE;
  }

  if (suspend_count_ptr) {
    *suspend_count_ptr = suspend_count;
  }

  REXKRNL_TRACE("[NtSuspendThread] -> {:#x} suspend_count={}", result, suspend_count);
  return result;
}

namespace {

void UpdateGuestStackPointers(X_KTHREAD* kthread, X_KPCR* pcr, PPCContext* ctx, uint32_t sp,
                              uint32_t stack_alloc_base, uint32_t stack_base,
                              uint32_t stack_limit) {
  kthread->stack_alloc_base = stack_alloc_base;
  kthread->stack_base = stack_base;
  kthread->stack_limit = stack_limit;
  pcr->stack_base_ptr = stack_base;
  pcr->stack_end_ptr = stack_limit;
  ctx->r1.u64 = sp;
}

//-----------------------------------------------------------------------------
// Guest cooperative context switches -> host fibers.
//
// The DOA4 kernel scheduler switches contexts inside a single KTHREAD: it
// saves the outgoing register file into one context object, restores a
// different one (lr included), adopts that context's stack, then tail branches
// into KeSetCurrentStackPointers. Recompiled code can only unwind its own C++
// frames on the way back out of that tail call, so control would return to the
// *previous* thread's code instead of the restored lr -- and the restored lr is
// frequently mid-function (the idle thread resumes inside its record sweep),
// which no function-table lookup can represent either.
//
// Mirror the switch onto a host fiber instead. The suspended frame resumes
// exactly where it parked, so both the mid-function resume and the C++
// stack discipline come for free -- the same trick SwitchToFiber_entry already
// uses for XAPI fibers.
//
// Keyed by guest stack base: the incoming base is a KSCSP argument, the
// outgoing base is read from the KTHREAD before the update overwrites it, and
// both are the same field the context objects carry at +8.
//
// A stack base is not a context identity: once the guest frees a kernel stack
// the address is handed to the next thread, and the new context's sp starts at
// the fresh stack top while the old chain is still parked below it. So each
// slot also remembers the r1 its chain yielded with -- that is the value the
// guest hands back as KSCSP's stack_ptr when it resumes *that* context, and a
// different sp proves the slot has been adopted by someone else.
struct GuestContextFiber {
  rex::thread::Fiber* fiber = nullptr;
  uint32_t parked_r1 = 0;
  bool parked = false;
};

struct GuestContextFibers {
  std::mutex mutex;
  std::unordered_map<uint32_t, GuestContextFiber> by_stack_base;
};

GuestContextFibers& guest_context_fibers() {
  static GuestContextFibers state;
  return state;
}

struct NewContextFiberArgs {
  uint32_t entry_pc;
  rex::thread::Fiber* fallback;
};

void NewContextFiberEntry(void* raw) {
  const auto args = std::unique_ptr<NewContextFiberArgs>(static_cast<NewContextFiberArgs*>(raw));
  auto* thread = XThread::GetCurrentThread();
  auto* ks = REX_KERNEL_STATE();
  if (!thread || !ks) {
    REXLOG_ERROR("NewContextFiber: no current thread/kernel state for pc {:#010x}",
                 args->entry_pc);
    if (args->fallback) {
      rex::thread::Fiber::SwitchTo(args->fallback);
    }
    return;
  }

  auto* fn = ks->function_dispatcher()->GetFunction(args->entry_pc);
  if (!fn) {
    REXLOG_ERROR("NewContextFiber: no recompiled function at context entry {:#010x}",
                 args->entry_pc);
  } else {
    fn(*thread->thread_state()->context(), ks->memory()->virtual_membase());
    // A scheduler context is expected to switch away rather than ever return.
    REXLOG_WARN("NewContextFiber: guest context {:#010x} returned from its entry",
                args->entry_pc);
  }

  if (args->fallback) {
    rex::thread::Fiber::SwitchTo(args->fallback);
  }
}

/// Mirror a guest context switch onto host fibers, then suspend the caller.
/// Must run after UpdateGuestStackPointers so a resuming frame finds the guest
/// register/stack state that the switcher installed for it.
/// @param outgoing_r1 r1 as it stood on entry to KSCSP, i.e. the r1 of the
///   context being suspended (the guest passes the incoming sp as an argument
///   instead of restoring it, so this still belongs to the outgoing side).
/// @param incoming_sp the KSCSP stack_ptr argument, which is the r1 the guest
///   just restored for the incoming context.
void SwitchHostFiberForGuestContext(uint32_t outgoing_stack_base, uint32_t incoming_stack_base,
                                    uint32_t entry_lr, uint32_t outgoing_r1,
                                    uint32_t outgoing_stack_limit, uint32_t incoming_sp) {
  if (!incoming_stack_base || incoming_stack_base == outgoing_stack_base) {
    return;  // no stack change: bookkeeping call, not a context switch
  }

  auto& registry = guest_context_fibers();
  auto* current = rex::thread::Fiber::Current();
  if (!current) {
    current = rex::thread::Fiber::ConvertCurrentThread();
    if (!current) {
      REXLOG_ERROR("CTXSW: ConvertCurrentThread failed for guest stack {:#010x}",
                   incoming_stack_base);
      return;
    }
  }

  rex::thread::Fiber* incoming = nullptr;
  rex::thread::Fiber* stale = nullptr;
  uint32_t stale_r1 = 0;
  bool created = false;
  bool reused = false;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (outgoing_stack_base) {
      GuestContextFiber& parked = registry.by_stack_base[outgoing_stack_base];
      parked.fiber = current;
      // Only record a yield r1 that actually lies on the outgoing stack; an
      // out-of-range value would poison every later resume check for this key.
      if (outgoing_r1 >= outgoing_stack_limit && outgoing_r1 < outgoing_stack_base) {
        parked.parked_r1 = outgoing_r1;
        parked.parked = true;
      } else {
        parked.parked = false;
      }
    }

    auto it = registry.by_stack_base.find(incoming_stack_base);
    if (it != registry.by_stack_base.end() && it->second.fiber) {
      GuestContextFiber& slot = it->second;
      // The guest can only resume the exact context it suspended, so the sp it
      // restores has to equal the r1 that chain yielded with. A different sp
      // means this stack address has been adopted by a fresh context while the
      // dead chain is still parked on it -- resuming it would run an epilogue
      // against a frame pointer that no longer matches any frame.
      if (!slot.parked || slot.parked_r1 == incoming_sp) {
        incoming = slot.fiber;
        reused = true;
      } else {
        stale = slot.fiber;
        stale_r1 = slot.parked_r1;
        slot.fiber = nullptr;
        slot.parked = false;
        slot.parked_r1 = 0;
      }
    }

    if (!incoming) {
      // First time this context runs: start it at the lr the guest just
      // restored. Created under the lock so two threads cannot race a fiber.
      auto owned = std::make_unique<NewContextFiberArgs>(
          NewContextFiberArgs{entry_lr, current});
      incoming = rex::thread::Fiber::Create(512 * 1024, NewContextFiberEntry, owned.get());
      if (incoming) {
        owned.release();  // NewContextFiberEntry takes ownership
        GuestContextFiber& slot = registry.by_stack_base[incoming_stack_base];
        slot.fiber = incoming;
        slot.parked = false;
        slot.parked_r1 = 0;
        created = true;
      }
    }
  }

  // Drop the abandoned host stack outside the lock. It is suspended, not
  // running, and its guest stack no longer exists.
  if (stale && stale != current) {
    REXLOG_ERROR("CTXSW: discarding parked fiber {} for stack 0x{:08X} (chain yielded r1 0x{:08X}, "
                 "guest restored sp 0x{:08X})",
                 static_cast<const void*>(stale), incoming_stack_base, stale_r1, incoming_sp);
    stale->Destroy();
  }

  if (incoming_stack_base >= 0x70860000u && incoming_stack_base < 0x70890000u) {
    REXLOG_ERROR(
        "CTXSW out=0x{:08X} in=0x{:08X} fiber={} created={} reused={} is_current={} "
        "entry_lr=0x{:08X} in_sp=0x{:08X}",
        outgoing_stack_base, incoming_stack_base, static_cast<const void*>(incoming), created,
        reused, incoming == current, entry_lr, incoming_sp);
  }

  if (!incoming) {
    REXLOG_ERROR("CTXSW: failed to create host fiber for guest stack {:#010x}",
                 incoming_stack_base);
    return;
  }

  if (incoming == current) {
    // TEMP DIAGNOSTIC: the outgoing frame was parked under a different stack
    // key, so returning here unwinds the import stub into a caller whose guest
    // r1 has already moved to another stack.
    REXLOG_ERROR("CTXSW: incoming stack 0x{:08X} resolves to the running fiber (out 0x{:08X})",
                 incoming_stack_base, outgoing_stack_base);
    return;  // already running on that fiber; nothing to switch
  }

  rex::thread::Fiber::SwitchTo(incoming);
}

}  // namespace

// TEMP DIAGNOSTIC: the fiber registry is keyed by guest stack base and is never
// pruned, so freeing a stack leaves its fiber behind. Report the registry at the
// moment a stack is deleted -- a stale key that is later adopted by another
// context resumes a dead host frame chain.
void LogGuestFibersForStack(uint32_t stack_base) {
  // TEMP probe gate (REX_DIAG=0): skip the registry scan entirely.
  if (!rex::diag::enabled()) {
    return;
  }
  auto& registry = guest_context_fibers();
  std::lock_guard<std::mutex> lock(registry.mutex);
  char keys[520] = {};
  size_t kused = 0;
  int shown = 0;
  for (const auto& entry : registry.by_stack_base) {
    if (shown >= 32 || kused + 16 >= sizeof(keys)) {
      break;
    }
    const int n = std::snprintf(keys + kused, sizeof(keys) - kused, " 0x%08X", entry.first);
    if (n <= 0 || static_cast<size_t>(n) >= sizeof(keys) - kused) {
      break;
    }
    kused += static_cast<size_t>(n);
    ++shown;
  }
  const auto it = registry.by_stack_base.find(stack_base);
  REXLOG_ERROR("MMDKS-FIBER stack=0x{:08X} registry={} keys={} fiber={} parked_r1=0x{:08X} "
               "parked={}",
               stack_base, registry.by_stack_base.size(), keys,
               static_cast<const void*>(it != registry.by_stack_base.end() ? it->second.fiber
                                                                           : nullptr),
               it != registry.by_stack_base.end() ? it->second.parked_r1 : 0,
               it != registry.by_stack_base.end() && it->second.parked);
}

void KeSetCurrentStackPointers_entry(mapped_void stack_ptr, ppc_ptr_t<X_KTHREAD> thread,
                                     mapped_void stack_alloc_base, mapped_void stack_base,
                                     mapped_void stack_limit) {
  auto current_thread = XThread::GetCurrentThread();
  auto context = current_thread->thread_state()->context();

  // The outgoing stack base is only readable before UpdateGuestStackPointers
  // overwrites it, and it is the key that maps this switch back onto its host
  // fiber.
  const uint32_t outgoing_stack_base = static_cast<uint32_t>(thread->stack_base);
  const uint32_t outgoing_stack_limit = static_cast<uint32_t>(thread->stack_limit);

  auto pcr =
      REX_KERNEL_MEMORY()->TranslateVirtual<X_KPCR*>(static_cast<uint32_t>(context->r13.u64));

  // TEMP DIAGNOSTIC: this is the only guest-callable path that writes r1
  // absolutely, so any r1 that drifts off a live frame has to come through
  // here (or through ExecuteTrap's whole-context restore). Bounded so a run of
  // switches cannot flood the log.
  const uint32_t r1_before = static_cast<uint32_t>(context->r1.u64);
  const uint32_t incoming_base = stack_base.guest_address();

  UpdateGuestStackPointers(thread, pcr, context, stack_ptr.guest_address(),
                           stack_alloc_base.value(), stack_base.value(), stack_limit.value());

  const uint32_t r1_after = static_cast<uint32_t>(context->r1.u64);
  const uint32_t limit_base = stack_limit.guest_address();
  const bool same_stack = !incoming_base || incoming_base == outgoing_stack_base;
  // Normal switches move r1 between two different stacks, so a bare "r1 moved"
  // test would fire on every one. The interesting case is an sp that lands
  // outside the stack it claims to belong to.
  const bool outside =
      incoming_base && (r1_after < limit_base || r1_after >= incoming_base);
  static uint32_t s_kscsp_count = 0;
  const uint32_t n = s_kscsp_count++;
  // 0x70800000-0x70900000 is the window the faulting frame chain and the two
  // just-released kernel stacks live in, so every switch touching it is kept
  // regardless of the cap -- the clobber has to come through one of these.
  // The stacks the crashing frame chain and the two just-deleted kernel stacks
  // live in. Kept narrow: a wider window logged >14k context switches and the
  // resulting slowdown stopped the teardown sequence from ever happening.
  const bool in_region =
      (stack_ptr.guest_address() >= 0x70860000u && stack_ptr.guest_address() < 0x70890000u) ||
      (incoming_base >= 0x70860000u && incoming_base < 0x70890000u) ||
      (outgoing_stack_base >= 0x70860000u && outgoing_stack_base < 0x70890000u);
  // Besides the first 400, keep every call that does NOT park: those return
  // straight back through the import stub, so a sp change on one of them lands
  // on a caller whose frame still belongs to the old stack.
  if (rex::diag::enabled() && (n < 400 || outside || same_stack || in_region)) {
    REXLOG_ERROR("KSCSP n={} sp=0x{:08X} r1 0x{:08X} -> 0x{:08X} out=0x{:08X} in=0x{:08X} "  // NOLINT
                 "limit=0x{:08X} same_stack={} outside={} lr=0x{:08X}",
                 n, stack_ptr.guest_address(), r1_before, r1_after, outgoing_stack_base,
                 incoming_base, limit_base, same_stack, outside,
                 static_cast<uint32_t>(context->lr));
  }

  // TEMP DIAGNOSTIC: the walker descriptor rows (0x82CE7058 + slot*48) decide
  // which file buffer each slot walks. A fault-time dump only freezes the final
  // state, so log every change as it happens instead: this shows when each
  // slot's block pointer appeared, what every other slot held at that moment,
  // and the raw (pre-stamp) block header the fixup pass will later see.
  // Bounded so a run of switches cannot flood the log.
  {
    static uint32_t s_row_snap[4][12] = {};
    static uint32_t s_row_logs = 0;
    static bool s_row_init = false;
    const auto row_ts = rex::runtime::ThreadState::Get();
    if (row_ts && row_ts->memory()) {
      auto* const membase = row_ts->memory()->virtual_membase();
      const auto row_rd = [&](uint32_t addr) -> uint32_t {
        const uint32_t v = *reinterpret_cast<const uint32_t*>(membase + addr);
        return (v >> 24) | ((v >> 8) & 0x0000FF00u) | ((v << 8) & 0x00FF0000u) | (v << 24);
      };
      const auto heapish = [](uint32_t v) -> bool {
        return (v >= 0x100000u && v < 0x04000000u) || (v >= 0x82000000u && v < 0x84000000u);
      };
      uint32_t cur[4][12];
      for (uint32_t sid = 0; sid < 4; ++sid) {
        for (uint32_t k = 0; k < 12; ++k) {
          cur[sid][k] = row_rd(0x82CE7058u + sid * 48u + k * 4u);
        }
      }
      if (!s_row_init) {
        // Baseline stays zero so any state that already exists at the very
        // first switch is reported as a change on this call instead of being
        // silently adopted as the reference.
        s_row_init = true;
      }
      for (uint32_t sid = 0; sid < 4 && s_row_logs < 64; ++sid) {
        uint32_t mask = 0;
        for (uint32_t k = 0; k < 12; ++k) {
          if (cur[sid][k] != s_row_snap[sid][k]) {
            mask |= 1u << k;
          }
        }
        if (!mask) {
          continue;
        }
        char rw[176] = {};
        size_t rused = 0;
        for (uint32_t k = 0; k < 12; ++k) {
          const int nw = std::snprintf(rw + rused, sizeof(rw) - rused, " %u=%08X", k,
                                       cur[sid][k]);
          if (nw <= 0 || static_cast<size_t>(nw) >= sizeof(rw) - rused) {
            break;
          }
          rused += static_cast<size_t>(nw);
        }
        // A new block pointer means the raw header (before the walker stamps
        // it) is visible right now; a new buffer start identifies the file.
        char extra[176] = "";
        if ((mask & 1u) && heapish(cur[sid][0])) {
          size_t eused = 0;
          for (uint32_t k = 0; k < 6; ++k) {
            const int nw = std::snprintf(extra + eused, sizeof(extra) - eused, " blk%u=%08X", k,
                                         row_rd(cur[sid][0] + k * 4u));
            if (nw <= 0 || static_cast<size_t>(nw) >= sizeof(extra) - eused) {
              break;
            }
            eused += static_cast<size_t>(nw);
          }
        }
        if (mask & (1u << 9)) {
          size_t eused = std::strlen(extra);
          if (heapish(cur[sid][9])) {
            for (uint32_t k = 0; k < 4 && eused < sizeof(extra); ++k) {
              const int nw = std::snprintf(extra + eused, sizeof(extra) - eused, " buf%u=%08X", k,
                                           row_rd(cur[sid][9] + k * 4u));
              if (nw <= 0 || static_cast<size_t>(nw) >= sizeof(extra) - eused) {
                break;
              }
              eused += static_cast<size_t>(nw);
            }
          }
        }
        REXLOG_ERROR("GUEST ROWCHG n={} id={} mask=0x{:03X} row={}{}", n, sid, mask, rw, extra);
        for (uint32_t k = 0; k < 12; ++k) {
          s_row_snap[sid][k] = cur[sid][k];
        }
        ++s_row_logs;
      }
    }
  }

  // Suspend this frame on a host fiber so the restored lr (often mid-function)
  // resumes the way the guest intended instead of unwinding here.
  SwitchHostFiberForGuestContext(outgoing_stack_base, stack_base.guest_address(),
                                 static_cast<uint32_t>(context->lr), r1_before,
                                 outgoing_stack_limit, stack_ptr.guest_address());
}

u32 KeSetAffinityThread_entry(mapped_void thread_ptr, u32 affinity,
                              mapped_u32 previous_affinity_ptr) {
  REXKRNL_IMPORT_TRACE("KeSetAffinityThread", "thread={:#x} affinity={:#x}",
                       thread_ptr.guest_address(), (uint32_t)affinity);
  // The Xbox 360, according to disassembly of KeSetAffinityThread, unlike
  // Windows NT, stores the previous affinity via the pointer provided as an
  // argument, not in the return value - the return value is used for the
  // result.
  if (!affinity) {
    return X_STATUS_INVALID_PARAMETER;
  }
  auto thread = XObject::GetNativeObject<XThread>(REX_KERNEL_STATE(), thread_ptr);
  if (!thread) {
    REXKRNL_IMPORT_WARN("KeSetAffinityThread", "invalid thread handle ptr={:#x}",
                        thread_ptr.guest_address());
    return X_STATUS_INVALID_HANDLE;
  }
  if (previous_affinity_ptr) {
    *previous_affinity_ptr = uint32_t(1) << thread->active_cpu();
  }
  thread->SetAffinity(affinity);
  REXKRNL_IMPORT_RESULT("KeSetAffinityThread", "0x0");
  return X_STATUS_SUCCESS;
}

u32 KeQueryBasePriorityThread_entry(mapped_void thread_ptr) {
  REXKRNL_IMPORT_TRACE("KeQueryBasePriorityThread", "thread={:#x}", thread_ptr.guest_address());
  int32_t priority = 0;

  auto thread = XObject::GetNativeObject<XThread>(REX_KERNEL_STATE(), thread_ptr);
  if (thread) {
    priority = thread->QueryPriority();
  }

  REXKRNL_IMPORT_RESULT("KeQueryBasePriorityThread", "{}", priority);
  return priority;
}

u32 KeSetBasePriorityThread_entry(mapped_void thread_ptr, u32 increment) {
  REXKRNL_IMPORT_TRACE("KeSetBasePriorityThread", "thread={:#x} increment={}",
                       thread_ptr.guest_address(), (int32_t)increment);
  int32_t prev_priority = 0;
  auto thread = XObject::GetNativeObject<XThread>(REX_KERNEL_STATE(), thread_ptr);

  if (thread) {
    prev_priority = thread->QueryPriority();
    thread->SetPriority(increment);
  }
  REXKRNL_IMPORT_RESULT("KeSetBasePriorityThread", "prev_priority={}", prev_priority);

  return prev_priority;
}

u32 KeSetDisableBoostThread_entry(ppc_ptr_t<X_KTHREAD> thread_ptr, u32 disabled) {
  auto old_boost_disabled = reinterpret_cast<std::atomic_uint8_t*>(&thread_ptr->boost_disabled)
                                ->exchange(static_cast<uint8_t>(disabled));
  return old_boost_disabled;
}

u32 KeGetCurrentProcessType_entry() {
  auto current_thread = XThread::GetCurrentThread();
  auto context = current_thread->thread_state()->context();
  auto pcr =
      REX_KERNEL_MEMORY()->TranslateVirtual<X_KPCR*>(static_cast<uint32_t>(context->r13.u64));

  if (pcr->prcb_data.dpc_active) {
    return pcr->processtype_value_in_dpc;
  }

  auto thread = REX_KERNEL_MEMORY()->TranslateVirtual<X_KTHREAD*>(pcr->prcb_data.current_thread);
  return thread->process_type;
}

void KeSetCurrentProcessType_entry(u32 type) {
  // One of X_PROCTYPE_?
  assert_true(type <= 2);

  REX_KERNEL_STATE()->set_process_type(type);

  auto current_thread = XThread::GetCurrentThread();
  auto context = current_thread->thread_state()->context();
  auto pcr =
      REX_KERNEL_MEMORY()->TranslateVirtual<X_KPCR*>(static_cast<uint32_t>(context->r13.u64));

  if (pcr->prcb_data.dpc_active) {
    pcr->processtype_value_in_dpc = static_cast<uint8_t>(type);
  }
}

u32 KeQueryPerformanceFrequency_entry() {
  uint64_t result = chrono::Clock::guest_tick_frequency();
  return static_cast<uint32_t>(result);
}

u32 KeDelayExecutionThread_entry(u32 processor_mode, u32 alertable, mapped_u64 interval_ptr) {
  XThread* thread = XThread::GetCurrentThread();

  if (alertable) {
    thread->DeliverAPCs();
  }

  // TEMP DIAGNOSTIC (retail DOA4 R78): see LogLongWait - delay variant.
  const auto t0 = std::chrono::steady_clock::now();
  X_STATUS result = thread->Delay(processor_mode, alertable, *interval_ptr);
  const int64_t dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
  if (dur_us > 5000) {
    const auto* ppc = current_ppc_context();
    LogLongWait("DELAY", thread->guest_object(), dur_us, 0, static_cast<int64_t>(*interval_ptr),
                ppc ? static_cast<uint32_t>(ppc->lr) : 0u);
  }

  if (alertable && result == X_STATUS_USER_APC) {
    thread->DeliverAPCs();
  }

  return result;
}

u32 NtYieldExecution_entry() {
  rex::thread::MaybeYield();
  return X_STATUS_SUCCESS;
}

void KeQuerySystemTime_entry(mapped_u64 time_ptr) {
  uint64_t time = chrono::Clock::QueryGuestSystemTime();
  if (time_ptr) {
    *time_ptr = time;
  }
}

// https://msdn.microsoft.com/en-us/library/ms686801
u32 KeTlsAlloc_entry() {
  uint32_t slot = REX_KERNEL_STATE()->AllocateTLS(current_ppc_context());
  if (slot != X_TLS_OUT_OF_INDEXES) {
    XThread::GetCurrentThread()->SetTLSValue(slot, 0);
  }
  REXKRNL_IMPORT_RESULT("KeTlsAlloc", "slot={}", slot);
  return slot;
}

// https://msdn.microsoft.com/en-us/library/ms686804
u32 KeTlsFree_entry(u32 tls_index) {
  REXKRNL_IMPORT_TRACE("KeTlsFree", "slot={}", (uint32_t)tls_index);
  if (tls_index == X_TLS_OUT_OF_INDEXES) {
    REXKRNL_IMPORT_RESULT("KeTlsFree", "0 (invalid)");
    return 0;
  }

  REX_KERNEL_STATE()->FreeTLS(current_ppc_context(), tls_index);
  REXKRNL_IMPORT_RESULT("KeTlsFree", "1");
  return 1;
}

// https://msdn.microsoft.com/en-us/library/ms686812
u32 KeTlsGetValue_entry(u32 tls_index) {
  // xboxkrnl doesn't actually have an error branch - it always succeeds, even
  // if it overflows the TLS.
  uint32_t value = 0;
  if (XThread::GetCurrentThread()->GetTLSValue(tls_index, &value)) {
    return value;
  }

  return 0;
}

// https://msdn.microsoft.com/en-us/library/ms686818
u32 KeTlsSetValue_entry(u32 tls_index, u32 tls_value) {
  REXKRNL_IMPORT_TRACE("KeTlsSetValue", "slot={} value={:#x}", (uint32_t)tls_index,
                       (uint32_t)tls_value);
  // xboxkrnl doesn't actually have an error branch - it always succeeds, even
  // if it overflows the TLS.
  if (XThread::GetCurrentThread()->SetTLSValue(tls_index, tls_value)) {
    REXKRNL_IMPORT_RESULT("KeTlsSetValue", "1");
    return 1;
  }

  REXKRNL_IMPORT_RESULT("KeTlsSetValue", "0");
  return 0;
}

void KeInitializeEvent_entry(ppc_ptr_t<X_KEVENT> event_ptr, u32 event_type, u32 initial_state) {
  event_ptr.Zero();
  event_ptr->header.type = event_type;
  event_ptr->header.signal_state = (uint32_t)initial_state;
  auto ev = XObject::GetNativeObject<XEvent>(REX_KERNEL_STATE(), event_ptr, event_type);
  if (!ev) {
    assert_always();
    return;
  }
}

uint32_t xeKeSetEvent(X_KEVENT* event_ptr, uint32_t increment, uint32_t wait) {
  auto ev = XObject::GetNativeObject<XEvent>(REX_KERNEL_STATE(), event_ptr);
  if (!ev) {
    assert_always();
    return 0;
  }

  return ev->Set(increment, !!wait);
}

u32 KeSetEvent_entry(ppc_ptr_t<X_KEVENT> event_ptr, u32 increment, u32 wait) {
  return xeKeSetEvent(event_ptr, increment, wait);
}

u32 KePulseEvent_entry(ppc_ptr_t<X_KEVENT> event_ptr, u32 increment, u32 wait) {
  auto ev = XObject::GetNativeObject<XEvent>(REX_KERNEL_STATE(), event_ptr);
  if (!ev) {
    assert_always();
    return 0;
  }

  return ev->Pulse(increment, !!wait);
}

u32 KeResetEvent_entry(ppc_ptr_t<X_KEVENT> event_ptr) {
  auto ev = XObject::GetNativeObject<XEvent>(REX_KERNEL_STATE(), event_ptr);
  if (!ev) {
    assert_always();
    return 0;
  }

  return ev->Reset();
}

u32 NtCreateEvent_entry(mapped_u32 handle_ptr, ppc_ptr_t<X_OBJECT_ATTRIBUTES> obj_attributes_ptr,
                        u32 event_type, u32 initial_state) {
  // Check for an existing timer with the same name.
  auto existing_object =
      LookupNamedObject<XEvent>(REX_KERNEL_STATE(), obj_attributes_ptr.guest_address());
  if (existing_object) {
    if (existing_object->type() == XObject::Type::Event) {
      if (handle_ptr) {
        existing_object->RetainHandle();
        *handle_ptr = existing_object->handle();
      }
      return X_STATUS_OBJECT_NAME_EXISTS;
    } else {
      return X_STATUS_INVALID_HANDLE;
    }
  }

  auto ev = object_ref<XEvent>(new XEvent(REX_KERNEL_STATE()));
  ev->Initialize(!event_type, !!initial_state);

  // obj_attributes may have a name inside of it, if != NULL.
  if (obj_attributes_ptr) {
    ev->SetAttributes(obj_attributes_ptr.guest_address());
  }

  if (handle_ptr) {
    *handle_ptr = ev->handle();
  }
  return X_STATUS_SUCCESS;
}

uint32_t xeNtSetEvent(uint32_t handle, rex::be<uint32_t>* previous_state_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  auto ev = REX_KERNEL_OBJECTS()->LookupObject<XEvent>(handle);
  if (ev) {
    int32_t was_signalled = ev->Set(0, false);
    if (previous_state_ptr) {
      *previous_state_ptr = static_cast<uint32_t>(was_signalled);
    }
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }

  return result;
}

u32 NtSetEvent_entry(u32 handle, mapped_u32 previous_state_ptr) {
  return xeNtSetEvent(handle, previous_state_ptr);
}

u32 NtPulseEvent_entry(u32 handle, mapped_u32 previous_state_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  auto ev = REX_KERNEL_OBJECTS()->LookupObject<XEvent>(handle);
  if (ev) {
    int32_t was_signalled = ev->Pulse(0, false);
    if (previous_state_ptr) {
      *previous_state_ptr = static_cast<uint32_t>(was_signalled);
    }
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }

  return result;
}

uint32_t xeNtClearEvent(uint32_t handle) {
  X_STATUS result = X_STATUS_SUCCESS;

  auto ev = REX_KERNEL_OBJECTS()->LookupObject<XEvent>(handle);
  if (ev) {
    ev->Reset();
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }

  return result;
}

u32 NtClearEvent_entry(u32 handle) {
  return xeNtClearEvent(handle);
}

// https://msdn.microsoft.com/en-us/library/windows/hardware/ff552150(v=vs.85).aspx
void KeInitializeSemaphore_entry(ppc_ptr_t<X_KSEMAPHORE> semaphore_ptr, u32 count, u32 limit) {
  semaphore_ptr->header.type = 5;  // SemaphoreObject
  semaphore_ptr->header.signal_state = (uint32_t)count;
  semaphore_ptr->limit = (uint32_t)limit;

  auto sem = XObject::GetNativeObject<XSemaphore>(REX_KERNEL_STATE(), semaphore_ptr,
                                                  5 /* SemaphoreObject */);
  if (!sem) {
    assert_always();
    return;
  }
}

uint32_t xeKeReleaseSemaphore(X_KSEMAPHORE* semaphore_ptr, uint32_t increment, uint32_t adjustment,
                              uint32_t wait) {
  auto sem = XObject::GetNativeObject<XSemaphore>(REX_KERNEL_STATE(), semaphore_ptr);
  if (!sem) {
    assert_always();
    return 0;
  }

  // TODO(benvanik): increment thread priority?
  // TODO(benvanik): wait?

  int32_t previous_count = 0;
  [[maybe_unused]] bool success =
      sem->ReleaseSemaphore(static_cast<int32_t>(adjustment), &previous_count);
  return static_cast<uint32_t>(previous_count);
}

u32 KeReleaseSemaphore_entry(ppc_ptr_t<X_KSEMAPHORE> semaphore_ptr, u32 increment, u32 adjustment,
                             u32 wait) {
  return xeKeReleaseSemaphore(semaphore_ptr, increment, adjustment, wait);
}

u32 NtCreateSemaphore_entry(mapped_u32 handle_ptr, mapped_void obj_attributes_ptr, u32 count,
                            u32 limit) {
  // Check for an existing semaphore with the same name.
  auto existing_object =
      LookupNamedObject<XSemaphore>(REX_KERNEL_STATE(), obj_attributes_ptr.guest_address());
  if (existing_object) {
    if (existing_object->type() == XObject::Type::Semaphore) {
      if (handle_ptr) {
        existing_object->RetainHandle();
        *handle_ptr = existing_object->handle();
      }
      return X_STATUS_OBJECT_NAME_EXISTS;
    } else {
      return X_STATUS_INVALID_HANDLE;
    }
  }

  auto sem = object_ref<XSemaphore>(new XSemaphore(REX_KERNEL_STATE()));
  if (!sem->Initialize((int32_t)count, (int32_t)limit)) {
    if (handle_ptr) {
      *handle_ptr = 0;
    }
    sem->ReleaseHandle();
    return X_STATUS_INVALID_PARAMETER;
  }

  // obj_attributes may have a name inside of it, if != NULL.
  if (obj_attributes_ptr) {
    sem->SetAttributes(obj_attributes_ptr.guest_address());
  }

  if (handle_ptr) {
    *handle_ptr = sem->handle();
  }

  return X_STATUS_SUCCESS;
}

u32 NtReleaseSemaphore_entry(u32 sem_handle, u32 release_count, mapped_u32 previous_count_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;
  int32_t previous_count = 0;

  auto sem = REX_KERNEL_OBJECTS()->LookupObject<XSemaphore>(sem_handle);
  if (sem) {
    bool success = sem->ReleaseSemaphore(static_cast<int32_t>(release_count), &previous_count);
    if (!success) {
      result = X_STATUS_SEMAPHORE_LIMIT_EXCEEDED;
    }
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }
  if (previous_count_ptr) {
    *previous_count_ptr = (uint32_t)previous_count;
  }

  return result;
}

u32 NtCreateMutant_entry(mapped_u32 handle_out, ppc_ptr_t<X_OBJECT_ATTRIBUTES> obj_attributes,
                         u32 initial_owner) {
  // Check for an existing timer with the same name.
  auto existing_object =
      LookupNamedObject<XMutant>(REX_KERNEL_STATE(), obj_attributes.guest_address());
  if (existing_object) {
    if (existing_object->type() == XObject::Type::Mutant) {
      if (handle_out) {
        existing_object->RetainHandle();
        *handle_out = existing_object->handle();
      }
      return X_STATUS_OBJECT_NAME_EXISTS;
    } else {
      return X_STATUS_INVALID_HANDLE;
    }
  }

  auto mutant = object_ref<XMutant>(new XMutant(REX_KERNEL_STATE()));
  mutant->Initialize(initial_owner ? true : false);

  // obj_attributes may have a name inside of it, if != NULL.
  if (obj_attributes) {
    mutant->SetAttributes(obj_attributes.guest_address());
  }

  if (handle_out) {
    *handle_out = mutant->handle();
  }

  return X_STATUS_SUCCESS;
}

u32 NtReleaseMutant_entry(u32 mutant_handle, u32 unknown) {
  // This doesn't seem to be supported.
  // int32_t previous_count_ptr = SHIM_GET_ARG_32(2);

  // Whatever arg 1 is all games seem to set it to 0, so whether it's
  // abandon or wait we just say false. Which is good, cause they are
  // both ignored.
  assert_zero(unknown);
  uint32_t priority_increment = 0;
  bool abandon = false;
  bool wait = false;

  X_STATUS result = X_STATUS_SUCCESS;

  auto mutant = REX_KERNEL_OBJECTS()->LookupObject<XMutant>(mutant_handle);
  if (mutant) {
    result = mutant->ReleaseMutant(priority_increment, abandon, wait);
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }

  return result;
}

u32 NtCreateTimer_entry(mapped_u32 handle_ptr, mapped_void obj_attributes_ptr, u32 timer_type) {
  // timer_type = NotificationTimer (0) or SynchronizationTimer (1)

  // Check for an existing timer with the same name.
  auto existing_object =
      LookupNamedObject<XTimer>(REX_KERNEL_STATE(), obj_attributes_ptr.guest_address());
  if (existing_object) {
    if (existing_object->type() == XObject::Type::Timer) {
      if (handle_ptr) {
        existing_object->RetainHandle();
        *handle_ptr = existing_object->handle();
      }
      return X_STATUS_OBJECT_NAME_EXISTS;
    } else {
      return X_STATUS_INVALID_HANDLE;
    }
  }

  auto timer = object_ref<XTimer>(new XTimer(REX_KERNEL_STATE()));
  timer->Initialize(timer_type);

  // obj_attributes may have a name inside of it, if != NULL.
  if (obj_attributes_ptr) {
    timer->SetAttributes(obj_attributes_ptr.guest_address());
  }

  if (handle_ptr) {
    *handle_ptr = timer->handle();
  }

  return X_STATUS_SUCCESS;
}

void KeInitializeTimerEx_entry(ppc_ptr_t<X_KTIMER> timer_ptr, u32 timer_type, u32 proc_type) {
  assert_true(timer_type == 0 || timer_type == 1);
  assert_true(proc_type < 3);
  // Other fields are unmodified; they must carry through multiple calls.
  timer_ptr->header.process_type = static_cast<uint8_t>(proc_type & 0xFF);
  timer_ptr->header.inserted = 0;
  timer_ptr->header.type = static_cast<uint8_t>(timer_type + 8);
  timer_ptr->header.signal_state = 0;
  // Initialize wait list to point to itself (empty list).
  uint32_t wait_list_addr =
      timer_ptr.guest_address() + offsetof(X_DISPATCH_HEADER, wait_list_flink);
  timer_ptr->header.wait_list_flink = wait_list_addr;
  timer_ptr->header.wait_list_blink = wait_list_addr;
  timer_ptr->due_time = 0;
  timer_ptr->period = 0;
}

u32 NtSetTimerEx_entry(u32 timer_handle, mapped_u64 due_time_ptr,
                       mapped_void routine_ptr /*PTIMERAPCROUTINE*/, u32 unk_one,
                       mapped_void routine_arg, u32 resume, u32 period_ms, u32 unk_zero) {
  assert_true(unk_one == 1);
  assert_true(unk_zero == 0);

  uint64_t due_time = *due_time_ptr;

  X_STATUS result = X_STATUS_SUCCESS;

  auto timer = REX_KERNEL_OBJECTS()->LookupObject<XTimer>(timer_handle);
  if (timer) {
    result = timer->SetTimer(due_time, period_ms, routine_ptr.guest_address(),
                             routine_arg.guest_address(), resume ? true : false);
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }

  return result;
}

u32 NtCancelTimer_entry(u32 timer_handle, mapped_u32 current_state_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  auto timer = REX_KERNEL_OBJECTS()->LookupObject<XTimer>(timer_handle);
  if (timer) {
    result = timer->Cancel();
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }
  if (current_state_ptr) {
    *current_state_ptr = 0;
  }

  return result;
}

// TEMP DIAGNOSTIC (retail DOA4 R78 stage-select camera-path stall): log any
// guest kernel wait over 5ms with the guest call site (LR) + object + the
// timeout that was requested, to attribute the ~2s 20Hz guest-starve bursts
// seen at camera flips (R75/R78: 121 half-rate frames per flip; XMA/audio
// paths already ruled out by bounded probes that never fired). Per-LR dedup
// so a site that legitimately waits every frame (the vblank wait, ~16.7ms)
// cannot flood: first 20 occurrences of each distinct LR + every 1000th
// after; coarse cap when the small table is full.
static void LogLongWait(const char* api, uint32_t tid, int64_t dur_us, uint32_t obj_addr,
                        int64_t timeout_100ns, uint32_t lr) {
  // TEMP probe gate (REX_DIAG): some call sites (NTWAIT1/WAITN) measure
  // unconditionally; keep them inert for REX_DIAG=0 runs too.
  if (!rex::diag::enabled()) {
    return;
  }
  struct Seen {
    uint32_t lr;
    uint32_t hits;
  };
  static Seen s_seen[128] = {};
  static uint32_t s_events = 0;
  const uint32_t ev = s_events++;
  const uint32_t key = lr ? lr : 1;
  Seen* entry = nullptr;
  Seen* free_slot = nullptr;
  for (auto& s : s_seen) {
    if (s.lr == key) {
      entry = &s;
      break;
    }
    if (!free_slot && s.lr == 0) {
      free_slot = &s;
    }
  }
  if (!entry) {
    entry = free_slot;
  }
  bool log_it;
  if (entry && entry->lr == 0) {
    entry->lr = key;
    entry->hits = 1;
    log_it = true;
  } else if (entry) {
    entry->hits++;
    log_it = entry->hits <= 20 || (entry->hits % 1000) == 0;
  } else {
    log_it = ev < 20 || (ev % 1000) == 0;
  }
  if (log_it) {
    REXLOG_ERROR("KWAIT {} tid={:08X} dur_us={} obj={:#010x} timeout={} lr={:#010x}", api, tid,
                 dur_us, obj_addr, timeout_100ns, lr);
  }
}

uint32_t xeKeWaitForSingleObject(void* object_ptr, uint32_t wait_reason, uint32_t processor_mode,
                                 uint32_t alertable, uint64_t* timeout_ptr) {
  // TEMP DIAGNOSTIC (retail DOA4 transition-hold hunt, R89+): split the
  // measured KWAIT duration into object lookup (global critical region
  // acquire + table lookup, may queue behind another thread's hold) vs the
  // actual wait (must be instant for guest timeout=0 polls). Bounded (first 40
  // slow ones + every 400th over 5ms), REXLOG_ERROR. Remove after root cause.
  const bool diag_on = rex::diag::enabled();  // TEMP probe gate (REX_DIAG=0)
  const auto sp_t0 = diag_on ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  auto object = XObject::GetNativeObject<XObject>(REX_KERNEL_STATE(), object_ptr);
  const auto sp_t1 = diag_on ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};

  if (!object) {
    // The only kind-of failure code (though this should never happen)
    assert_always();
    return X_STATUS_ABANDONED_WAIT_0;
  }

  X_STATUS result = object->Wait(wait_reason, processor_mode, alertable, timeout_ptr);
  const auto sp_t2 = diag_on ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};

  if (alertable && result == X_STATUS_USER_APC) {
    XThread::GetCurrentThread()->DeliverAPCs();
  }

  if (diag_on) {
    const int64_t lookup_us =
        std::chrono::duration_cast<std::chrono::microseconds>(sp_t1 - sp_t0).count();
    const int64_t wait_us =
        std::chrono::duration_cast<std::chrono::microseconds>(sp_t2 - sp_t1).count();
    if (lookup_us + wait_us > 5000) {
      static uint64_t s_ksplit_n = 0;
      if (++s_ksplit_n <= 40 || (s_ksplit_n % 400) == 0) {
        REXLOG_ERROR("KSPLIT n={} lookup_us={} wait_us={} obj={:#x} st={:#x}", s_ksplit_n,
                     lookup_us, wait_us, reinterpret_cast<uintptr_t>(object_ptr),
                     static_cast<uint32_t>(result));
      }
    }
  }

  return result;
}

u32 KeWaitForSingleObject_entry(mapped_void object_ptr, u32 wait_reason, u32 processor_mode,
                                u32 alertable, mapped_u64 timeout_ptr) {
  uint64_t timeout = timeout_ptr ? static_cast<uint64_t>(*timeout_ptr) : 0u;
  // REXKRNL_IMPORT_TRACE("KeWaitForSingleObject", "obj={:#x} reason={} mode={} alertable={}
  // timeout={}",
  // object_ptr.guest_address(), (uint32_t)wait_reason,
  //(uint32_t)processor_mode, (uint32_t)alertable,
  // timeout_ptr ? (int64_t)timeout : -1);
  // TEMP DIAGNOSTIC (retail DOA4 R78): see LogLongWait - wait variant.
  const int64_t timeout_orig = timeout_ptr ? static_cast<int64_t>(timeout) : -1;
  const bool diag_on = rex::diag::enabled();  // TEMP probe gate (REX_DIAG=0)
  const auto t0 = diag_on ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
  auto result = xeKeWaitForSingleObject(object_ptr, wait_reason, processor_mode, alertable,
                                        timeout_ptr ? &timeout : nullptr);
  if (diag_on) {
    const int64_t dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
    if (dur_us > 5000) {
      const auto* ppc = current_ppc_context();
      LogLongWait("WAIT1", XThread::GetCurrentThread()->guest_object(), dur_us,
                  static_cast<uint32_t>(object_ptr.guest_address()), timeout_orig,
                  ppc ? static_cast<uint32_t>(ppc->lr) : 0u);
    }
  }
  // REXKRNL_IMPORT_RESULT("KeWaitForSingleObject", "{:#x}", result);
  return result;
}

u32 NtWaitForSingleObjectEx_entry(u32 object_handle, u32 wait_mode, u32 alertable,
                                  mapped_u64 timeout_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  auto object = REX_KERNEL_OBJECTS()->LookupObject<XObject>(object_handle);
  if (object) {
    uint64_t timeout = timeout_ptr ? static_cast<uint64_t>(*timeout_ptr) : 0u;
    // TEMP DIAGNOSTIC (retail DOA4 R78): see LogLongWait - Nt wait variant.
    const int64_t timeout_orig = timeout_ptr ? static_cast<int64_t>(timeout) : -1;
    const auto t0 = std::chrono::steady_clock::now();
    result = object->Wait(3, wait_mode, alertable, timeout_ptr ? &timeout : nullptr);
    const int64_t dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
    if (dur_us > 5000) {
      const auto* ppc = current_ppc_context();
      LogLongWait("NTWAIT1", XThread::GetCurrentThread()->guest_object(), dur_us,
                  static_cast<uint32_t>(object->guest_object()), timeout_orig,
                  ppc ? static_cast<uint32_t>(ppc->lr) : 0u);
    }
    if (alertable && result == X_STATUS_USER_APC) {
      XThread::GetCurrentThread()->DeliverAPCs();
    }
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }

  return result;
}

u32 KeWaitForMultipleObjects_entry(u32 count, mapped_u32 objects_ptr, u32 wait_type,
                                   u32 wait_reason, u32 processor_mode, u32 alertable,
                                   mapped_u64 timeout_ptr, mapped_void wait_block_array_ptr) {
  assert_true(wait_type <= 1);

  std::vector<object_ref<XObject>> objects;
  for (uint32_t n = 0; n < count; n++) {
    auto object_ptr = REX_KERNEL_MEMORY()->TranslateVirtual(objects_ptr[n]);
    auto object_ref = XObject::GetNativeObject<XObject>(REX_KERNEL_STATE(), object_ptr);
    if (!object_ref) {
      return X_STATUS_INVALID_PARAMETER;
    }

    objects.push_back(std::move(object_ref));
  }

  uint64_t timeout = timeout_ptr ? static_cast<uint64_t>(*timeout_ptr) : 0u;
  // TEMP DIAGNOSTIC (retail DOA4 R78): see LogLongWait - multi-wait variant.
  const int64_t timeout_orig = timeout_ptr ? static_cast<int64_t>(timeout) : -1;
  const auto t0 = std::chrono::steady_clock::now();
  X_STATUS result = XObject::WaitMultiple(
      uint32_t(objects.size()), reinterpret_cast<XObject**>(objects.data()), wait_type, wait_reason,
      processor_mode, alertable, timeout_ptr ? &timeout : nullptr);
  const int64_t dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
  if (dur_us > 5000) {
    const auto* ppc = current_ppc_context();
    LogLongWait("WAITN", XThread::GetCurrentThread()->guest_object(), dur_us,
                static_cast<uint32_t>(objects[0]->guest_object()), timeout_orig,
                ppc ? static_cast<uint32_t>(ppc->lr) : 0u);
  }
  if (alertable && result == X_STATUS_USER_APC) {
    XThread::GetCurrentThread()->DeliverAPCs();
  }
  return result;
}

uint32_t xeNtWaitForMultipleObjectsEx(uint32_t count, rex::be<uint32_t>* handles,
                                      uint32_t wait_type, uint32_t wait_mode, uint32_t alertable,
                                      uint64_t* timeout_ptr) {
  assert_true(wait_type <= 1);

  std::vector<object_ref<XObject>> objects;
  for (uint32_t n = 0; n < count; n++) {
    uint32_t object_handle = handles[n];
    auto object = REX_KERNEL_OBJECTS()->LookupObject<XObject>(object_handle);
    if (!object) {
      return X_STATUS_INVALID_PARAMETER;
    }
    objects.push_back(std::move(object));
  }

  auto result = XObject::WaitMultiple(count, reinterpret_cast<XObject**>(objects.data()), wait_type,
                                      6, wait_mode, alertable, timeout_ptr);
  if (alertable && result == X_STATUS_USER_APC) {
    XThread::GetCurrentThread()->DeliverAPCs();
  }
  return result;
}

u32 NtWaitForMultipleObjectsEx_entry(u32 count, mapped_u32 handles, u32 wait_type, u32 wait_mode,
                                     u32 alertable, mapped_u64 timeout_ptr) {
  uint64_t timeout = timeout_ptr ? static_cast<uint64_t>(*timeout_ptr) : 0u;
  return xeNtWaitForMultipleObjectsEx(count, handles, wait_type, wait_mode, alertable,
                                      timeout_ptr ? &timeout : nullptr);
}

u32 NtSignalAndWaitForSingleObjectEx_entry(u32 signal_handle, u32 wait_handle, u32 alertable,
                                           u32 r6, mapped_u64 timeout_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  auto signal_object = REX_KERNEL_OBJECTS()->LookupObject<XObject>(signal_handle);
  auto wait_object = REX_KERNEL_OBJECTS()->LookupObject<XObject>(wait_handle);
  if (signal_object && wait_object) {
    uint64_t timeout = timeout_ptr ? static_cast<uint64_t>(*timeout_ptr) : 0u;
    result = XObject::SignalAndWait(signal_object.get(), wait_object.get(), 3, 1, alertable,
                                    timeout_ptr ? &timeout : nullptr);
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }

  if (alertable && result == X_STATUS_USER_APC) {
    XThread::GetCurrentThread()->DeliverAPCs();
  }

  return result;
}

// Guest-memory IRQL helpers - read/write current_irql directly from PCR.
// Take PPCContext* explicitly so they work from any thread (including host threads
// during InitializeGuestObject, dispatch thread creation, etc.).
static unsigned char xeKfRaiseIrql(PPCContext* ctx, unsigned char new_irql) {
  auto* mem = rex::system::kernel_state()->memory();
  auto pcr = mem->TranslateVirtual<X_KPCR*>(static_cast<uint32_t>(ctx->r13.u64));
  uint8_t old_irql = pcr->current_irql;
  pcr->current_irql = new_irql;
  return old_irql;
}

static void xeKfLowerIrql(PPCContext* ctx, unsigned char new_irql) {
  auto* mem = rex::system::kernel_state()->memory();
  auto pcr = mem->TranslateVirtual<X_KPCR*>(static_cast<uint32_t>(ctx->r13.u64));
  pcr->current_irql = new_irql;
}

// Guest-memory spinlock helpers - store PCR address as owner (matching xenia).
// PPCContext* provides r13 (PCR address) without needing XThread::GetCurrentThread().
uint32_t xeKeKfAcquireSpinLock(PPCContext* ctx, X_KSPINLOCK* lock, bool change_irql) {
  uint32_t old_irql = change_irql ? xeKfRaiseIrql(ctx, IRQL_DISPATCH) : 0;
  uint32_t pcr_addr = static_cast<uint32_t>(ctx->r13.u64);
  const uint32_t self = rex::byte_swap(pcr_addr);
  assert_true(lock->prcb_of_owner.value != self);  // self-deadlock detection
  while (!rex::thread::atomic_cas(0u, self, &lock->prcb_of_owner.value)) {
    rex::thread::MaybeYield();
  }
  return old_irql;
}

void xeKeKfReleaseSpinLock(PPCContext* ctx, X_KSPINLOCK* lock, uint32_t old_irql,
                           bool change_irql) {
  assert_true(lock->prcb_of_owner == static_cast<uint32_t>(ctx->r13.u64));
  rex::thread::atomic_store_release(0u, &lock->prcb_of_owner.value);
  if (change_irql && old_irql < IRQL_DISPATCH) {
    xeKfLowerIrql(ctx, static_cast<unsigned char>(old_irql));
  }
}

// Guest-memory APC helpers
void xeKeInitializeApc(XAPC* apc, uint32_t thread_ptr, uint32_t kernel_routine,
                       uint32_t rundown_routine, uint32_t normal_routine, uint32_t apc_mode,
                       uint32_t normal_context) {
  apc->thread_ptr = thread_ptr;
  apc->kernel_routine = kernel_routine;
  apc->rundown_routine = rundown_routine;
  apc->normal_routine = normal_routine;
  apc->type = 18;  // ApcObject
  if (normal_routine) {
    apc->apc_mode = apc_mode;
    apc->normal_context = normal_context;
  } else {
    apc->apc_mode = 0;
    apc->normal_context = 0;
  }
  apc->enqueued = 0;
}

uint32_t xeKeInsertQueueApc(XAPC* apc, uint32_t arg1, uint32_t arg2, uint32_t priority_increment,
                            PPCContext* ctx) {
  uint32_t thread_guest_pointer = apc->thread_ptr;
  if (!thread_guest_pointer) {
    return 0;
  }
  auto* mem = rex::system::kernel_state()->memory();
  auto target_thread = mem->TranslateVirtual<X_KTHREAD*>(thread_guest_pointer);
  auto old_irql = xeKeKfAcquireSpinLock(ctx, &target_thread->apc_lock);
  uint32_t result;
  if (!target_thread->may_queue_apcs || apc->enqueued) {
    result = 0;
  } else {
    apc->arg1 = arg1;
    apc->arg2 = arg2;

    auto& which_list = target_thread->apc_lists[apc->apc_mode];

    if (apc->normal_routine) {
      which_list.InsertTail(apc, mem);
    } else {
      // Kernel-mode APCs without normal_routine go before those with one.
      XAPC* insertion_pos = nullptr;
      for (auto&& sub_apc : which_list.IterateForward(mem)) {
        insertion_pos = &sub_apc;
        if (sub_apc.normal_routine) {
          break;
        }
      }
      if (!insertion_pos) {
        which_list.InsertHead(apc, mem);
      } else {
        util::XeInsertHeadList(insertion_pos->list_entry.blink_ptr, &apc->list_entry, mem);
      }
    }

    apc->enqueued = 1;
    result = 1;
  }
  xeKeKfReleaseSpinLock(ctx, &target_thread->apc_lock, old_irql);
  return result;
}

u32 KfAcquireSpinLock_entry(mapped_u32 lock_ptr) {
  auto lock = reinterpret_cast<X_KSPINLOCK*>(lock_ptr.host_address());
  return xeKeKfAcquireSpinLock(current_ppc_context(), lock);
}

void KfReleaseSpinLock_entry(mapped_u32 lock_ptr, u32 old_irql) {
  auto lock = reinterpret_cast<X_KSPINLOCK*>(lock_ptr.host_address());
  xeKeKfReleaseSpinLock(current_ppc_context(), lock, old_irql);
}

void KeAcquireSpinLockAtRaisedIrql_entry(mapped_u32 lock_ptr) {
  auto lock = reinterpret_cast<X_KSPINLOCK*>(lock_ptr.host_address());
  xeKeKfAcquireSpinLock(current_ppc_context(), lock, false);
}

u32 KeTryToAcquireSpinLockAtRaisedIrql_entry(mapped_u32 lock_ptr) {
  auto lock = reinterpret_cast<X_KSPINLOCK*>(lock_ptr.host_address());
  auto* ctx = current_ppc_context();
  uint32_t pcr_addr = static_cast<uint32_t>(ctx->r13.u64);
  if (!rex::thread::atomic_cas(0u, rex::byte_swap(pcr_addr), &lock->prcb_of_owner.value)) {
    return 0;
  }
  return 1;
}

void KeReleaseSpinLockFromRaisedIrql_entry(mapped_u32 lock_ptr) {
  auto lock = reinterpret_cast<X_KSPINLOCK*>(lock_ptr.host_address());
  xeKeKfReleaseSpinLock(current_ppc_context(), lock, 0, false);
}

void KeEnterCriticalRegion_entry() {
  XThread::GetCurrentThread()->EnterCriticalRegion();
}

void KeLeaveCriticalRegion_entry() {
  XThread::GetCurrentThread()->LeaveCriticalRegion();
}

u32 KeRaiseIrqlToDpcLevel_entry() {
  return xeKfRaiseIrql(current_ppc_context(), IRQL_DISPATCH);
}

void KfLowerIrql_entry(u32 old_value) {
  xeKfLowerIrql(current_ppc_context(),
                static_cast<unsigned char>(static_cast<uint32_t>(old_value)));
}

u32 KfRaiseIrql_entry(u32 new_irql) {
  return xeKfRaiseIrql(current_ppc_context(),
                       static_cast<unsigned char>(static_cast<uint32_t>(new_irql)));
}

void NtQueueApcThread_entry(u32 thread_handle, mapped_void apc_routine,
                            mapped_void apc_routine_context, mapped_void arg1, mapped_void arg2) {
  auto mem = REX_KERNEL_MEMORY();
  auto thread = REX_KERNEL_OBJECTS()->LookupObject<XThread>(thread_handle);

  if (!thread) {
    REXKRNL_ERROR("NtQueueApcThread: Incorrect thread handle! Might cause crash");
    return;
  }

  if (!apc_routine) {
    REXKRNL_ERROR("NtQueueApcThread: Incorrect apc routine! Might cause crash");
    return;
  }

  uint32_t apc_ptr = mem->SystemHeapAlloc(XAPC::kSize);
  if (!apc_ptr) {
    return;
  }
  XAPC* apc = mem->TranslateVirtual<XAPC*>(apc_ptr);
  xeKeInitializeApc(apc, thread->guest_object(), XAPC::kDummyKernelRoutine, 0,
                    apc_routine.guest_address(), 1 /* user apc mode */,
                    apc_routine_context.guest_address());

  if (!xeKeInsertQueueApc(apc, arg1.guest_address(), arg2.guest_address(), 0,
                          current_ppc_context())) {
    mem->SystemHeapFree(apc_ptr);
    return;
  }
  // Match Edge/Canary behavior: callback is only a wakeup hint.
  // APC delivery happens via alertable wait handling.
  thread->thread()->QueueUserCallback([]() {});
}

void KeInitializeApc_entry(ppc_ptr_t<XAPC> apc, mapped_void thread_ptr, mapped_void kernel_routine,
                           mapped_void rundown_routine, mapped_void normal_routine,
                           u32 processor_mode, mapped_void normal_context) {
  xeKeInitializeApc(apc.host_address(), thread_ptr.guest_address(), kernel_routine.guest_address(),
                    rundown_routine.guest_address(), normal_routine.guest_address(), processor_mode,
                    normal_context.guest_address());
}

u32 KeInsertQueueApc_entry(ppc_ptr_t<XAPC> apc, mapped_void arg1, mapped_void arg2,
                           u32 priority_increment) {
  return xeKeInsertQueueApc(apc.host_address(), arg1.guest_address(), arg2.guest_address(),
                            priority_increment, current_ppc_context());
}

u32 KeRemoveQueueApc_entry(ppc_ptr_t<XAPC> apc) {
  uint32_t thread_guest_pointer = apc->thread_ptr;
  if (!thread_guest_pointer) {
    return 0;
  }
  auto* ctx = current_ppc_context();
  auto mem = REX_KERNEL_MEMORY();
  auto target_thread = mem->TranslateVirtual<X_KTHREAD*>(thread_guest_pointer);
  auto old_irql = xeKeKfAcquireSpinLock(ctx, &target_thread->apc_lock);

  bool result = false;
  if (apc->enqueued) {
    apc->enqueued = 0;
    util::XeRemoveEntryList(&apc->list_entry, mem);
    result = true;
  }

  xeKeKfReleaseSpinLock(ctx, &target_thread->apc_lock, old_irql);
  return result ? 1 : 0;
}

u32 KiApcNormalRoutineNop_entry(u32 unk0 /* output? */, u32 unk1 /* 0x13 */) {
  return 0;
}

void KeInitializeDpc_entry(ppc_ptr_t<XDPC> dpc, mapped_void routine, mapped_void context) {
  dpc->Initialize(routine.guest_address(), context.guest_address());
}

u32 KeInsertQueueDpc_entry(ppc_ptr_t<XDPC> dpc, u32 arg1, u32 arg2) {
  assert_always("DPC does not dispatch yet; going to hang!");

  uint32_t list_entry_ptr = dpc.guest_address() + 4;

  // Lock dispatcher.
  auto global_lock = rex::thread::global_critical_region::AcquireDirect();
  auto dpc_list = REX_KERNEL_STATE()->dpc_list();

  // If already in a queue, abort.
  if (dpc_list->IsQueued(list_entry_ptr)) {
    return 0;
  }

  // Prep DPC.
  dpc->arg1 = (uint32_t)arg1;
  dpc->arg2 = (uint32_t)arg2;

  dpc_list->Insert(list_entry_ptr);

  return 1;
}

u32 KeRemoveQueueDpc_entry(ppc_ptr_t<XDPC> dpc) {
  bool result = false;

  uint32_t list_entry_ptr = dpc.guest_address() + 4;

  auto global_lock = rex::thread::global_critical_region::AcquireDirect();
  auto dpc_list = REX_KERNEL_STATE()->dpc_list();
  if (dpc_list->IsQueued(list_entry_ptr)) {
    dpc_list->Remove(list_entry_ptr);
    result = true;
  }

  return result ? 1 : 0;
}

// https://github.com/Cxbx-Reloaded/Cxbx-Reloaded/blob/51e4dfcaacfdbd1a9692272931a436371492f72d/import/OpenXDK/include/xboxkrnl/xboxkrnl.h#L1372
struct X_ERWLOCK {
  be<int32_t> lock_count;              // 0x0
  be<uint32_t> writers_waiting_count;  // 0x4
  be<uint32_t> readers_waiting_count;  // 0x8
  be<uint32_t> readers_entry_count;    // 0xC
  X_KEVENT writer_event;               // 0x10
  X_KSEMAPHORE reader_semaphore;       // 0x20
  X_KSPINLOCK spin_lock;               // 0x34
};
static_assert_size(X_ERWLOCK, 0x38);

void ExInitializeReadWriteLock_entry(ppc_ptr_t<X_ERWLOCK> lock_ptr) {
  lock_ptr->lock_count = -1;
  lock_ptr->writers_waiting_count = 0;
  lock_ptr->readers_waiting_count = 0;
  lock_ptr->readers_entry_count = 0;
  // Create GuestPointers to struct members with correct guest addresses
  ppc_ptr_t<X_KEVENT> event_ptr(&lock_ptr->writer_event,
                                lock_ptr.guest_address() + offsetof(X_ERWLOCK, writer_event));
  ppc_ptr_t<X_KSEMAPHORE> sem_ptr(&lock_ptr->reader_semaphore,
                                  lock_ptr.guest_address() + offsetof(X_ERWLOCK, reader_semaphore));
  KeInitializeEvent_entry(event_ptr, 1, 0);
  KeInitializeSemaphore_entry(sem_ptr, 0, 0x7FFFFFFF);
  lock_ptr->spin_lock.prcb_of_owner = 0;
}

void ExAcquireReadWriteLockExclusive_entry(ppc_ptr_t<X_ERWLOCK> lock_ptr) {
  auto* ctx = current_ppc_context();
  auto old_irql = xeKeKfAcquireSpinLock(ctx, &lock_ptr->spin_lock);

  int32_t lock_count = ++lock_ptr->lock_count;
  if (!lock_count) {
    xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
    return;
  }

  lock_ptr->writers_waiting_count++;

  xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
  xeKeWaitForSingleObject(&lock_ptr->writer_event, 7, 0, 0, nullptr);
}

u32 ExTryToAcquireReadWriteLockExclusive_entry(ppc_ptr_t<X_ERWLOCK> lock_ptr) {
  auto* ctx = current_ppc_context();
  auto old_irql = xeKeKfAcquireSpinLock(ctx, &lock_ptr->spin_lock);

  uint32_t result;
  if (lock_ptr->lock_count < 0) {
    lock_ptr->lock_count = 0;
    result = 1;
  } else {
    result = 0;
  }

  xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
  return result;
}

void ExAcquireReadWriteLockShared_entry(ppc_ptr_t<X_ERWLOCK> lock_ptr) {
  auto* ctx = current_ppc_context();
  auto old_irql = xeKeKfAcquireSpinLock(ctx, &lock_ptr->spin_lock);

  int32_t lock_count = ++lock_ptr->lock_count;
  if (!lock_count || (lock_ptr->readers_entry_count && !lock_ptr->writers_waiting_count)) {
    lock_ptr->readers_entry_count++;
    xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
    return;
  }

  lock_ptr->readers_waiting_count++;

  xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
  xeKeWaitForSingleObject(&lock_ptr->reader_semaphore, 7, 0, 0, nullptr);
}

u32 ExTryToAcquireReadWriteLockShared_entry(ppc_ptr_t<X_ERWLOCK> lock_ptr) {
  auto* ctx = current_ppc_context();
  auto old_irql = xeKeKfAcquireSpinLock(ctx, &lock_ptr->spin_lock);

  uint32_t result;
  if (lock_ptr->lock_count < 0 ||
      (lock_ptr->readers_entry_count && !lock_ptr->writers_waiting_count)) {
    lock_ptr->lock_count++;
    lock_ptr->readers_entry_count++;
    result = 1;
  } else {
    result = 0;
  }

  xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
  return result;
}

void ExReleaseReadWriteLock_entry(ppc_ptr_t<X_ERWLOCK> lock_ptr) {
  auto* ctx = current_ppc_context();
  auto old_irql = xeKeKfAcquireSpinLock(ctx, &lock_ptr->spin_lock);

  int32_t lock_count = --lock_ptr->lock_count;

  if (lock_count < 0) {
    lock_ptr->readers_entry_count = 0;
    xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
    return;
  }

  if (!lock_ptr->readers_entry_count) {
    auto readers_waiting_count = lock_ptr->readers_waiting_count;
    if (readers_waiting_count) {
      lock_ptr->readers_waiting_count = 0;
      lock_ptr->readers_entry_count = readers_waiting_count;
      xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
      xeKeReleaseSemaphore(&lock_ptr->reader_semaphore, 1, readers_waiting_count, 0);
      return;
    }
  }

  auto readers_entry_count = --lock_ptr->readers_entry_count;
  if (readers_entry_count) {
    xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
    return;
  }

  lock_ptr->writers_waiting_count--;
  xeKeKfReleaseSpinLock(ctx, &lock_ptr->spin_lock, old_irql);
  xeKeSetEvent(&lock_ptr->writer_event, 1, 0);
}

// NOTE: This function is very commonly inlined, and probably won't be called!
u32 InterlockedPushEntrySList_entry(ppc_ptr_t<X_SLIST_HEADER> plist_ptr,
                                    ppc_ptr_t<X_SINGLE_LIST_ENTRY> entry) {
  assert_not_null(plist_ptr);
  assert_not_null(entry);

  alignas(8) X_SLIST_HEADER old_hdr = *plist_ptr;
  alignas(8) X_SLIST_HEADER new_hdr = {};
  uint32_t old_head = 0;
  do {
    old_hdr = *plist_ptr;
    new_hdr.depth = old_hdr.depth + 1;
    new_hdr.sequence = old_hdr.sequence + 1;

    old_head = old_hdr.next.next;
    entry->next = old_hdr.next.next;
    new_hdr.next.next = entry.guest_address();
  } while (!rex::thread::atomic_cas(*(uint64_t*)(&old_hdr), *(uint64_t*)(&new_hdr),
                                    reinterpret_cast<uint64_t*>(plist_ptr.host_address())));

  return old_head;
}

u32 InterlockedPopEntrySList_entry(ppc_ptr_t<X_SLIST_HEADER> plist_ptr) {
  assert_not_null(plist_ptr);

  uint32_t popped = 0;
  alignas(8) X_SLIST_HEADER old_hdr = {0};
  alignas(8) X_SLIST_HEADER new_hdr = {0};
  do {
    old_hdr = *plist_ptr;
    auto next = REX_KERNEL_MEMORY()->TranslateVirtual<X_SINGLE_LIST_ENTRY*>(old_hdr.next.next);
    if (!old_hdr.next.next) {
      return 0;
    }
    popped = old_hdr.next.next;

    new_hdr.depth = old_hdr.depth - 1;
    new_hdr.next.next = next->next;
    new_hdr.sequence = old_hdr.sequence;
  } while (!rex::thread::atomic_cas(*(uint64_t*)(&old_hdr), *(uint64_t*)(&new_hdr),
                                    reinterpret_cast<uint64_t*>(plist_ptr.host_address())));

  return popped;
}

u32 InterlockedFlushSList_entry(ppc_ptr_t<X_SLIST_HEADER> plist_ptr) {
  assert_not_null(plist_ptr);

  alignas(8) X_SLIST_HEADER old_hdr = *plist_ptr;
  alignas(8) X_SLIST_HEADER new_hdr = {0};
  uint32_t first = 0;
  do {
    old_hdr = *plist_ptr;
    first = old_hdr.next.next;
    new_hdr.next.next = 0;
    new_hdr.depth = 0;
    new_hdr.sequence = 0;
  } while (!rex::thread::atomic_cas(*(uint64_t*)(&old_hdr), *(uint64_t*)(&new_hdr),
                                    reinterpret_cast<uint64_t*>(plist_ptr.host_address())));

  return first;
}

REX_EXPORT_STUB(__imp__KeInsertByKeyDeviceQueue);
REX_EXPORT_STUB(__imp__KeInsertDeviceQueue);
REX_EXPORT_STUB(__imp__KeInsertHeadQueue);
REX_EXPORT_STUB(__imp__KeInsertQueue);
REX_EXPORT_STUB(__imp__KeReleaseMutant);
REX_EXPORT_STUB(__imp__KeRemoveByKeyDeviceQueue);
REX_EXPORT_STUB(__imp__KeRemoveDeviceQueue);
REX_EXPORT_STUB(__imp__KeRemoveEntryDeviceQueue);
REX_EXPORT_STUB(__imp__KeRemoveQueue);
REX_EXPORT_STUB(__imp__KeSetEventBoostPriority);
REX_EXPORT_STUB(__imp__KiApcNormalRoutineNop_);

u32 KeSuspendThread_entry(mapped_void kthread_ptr) {
  auto thread = XObject::GetNativeObject<XThread>(REX_KERNEL_STATE(), kthread_ptr);
  uint32_t old_suspend_count = 0;

  if (thread) {
    old_suspend_count = thread->suspend_count();
    uint32_t discarded = 0;
    thread->Suspend(&discarded);
  }

  return old_suspend_count;
}

}  // namespace rex::kernel::xboxkrnl

REX_EXPORT(__imp__ExCreateThread, rex::kernel::xboxkrnl::ExCreateThread_entry)
REX_EXPORT(__imp__ExTerminateThread, rex::kernel::xboxkrnl::ExTerminateThread_entry)
REX_EXPORT(__imp__NtResumeThread, rex::kernel::xboxkrnl::NtResumeThread_entry)
REX_EXPORT(__imp__KeResumeThread, rex::kernel::xboxkrnl::KeResumeThread_entry)
REX_EXPORT(__imp__NtSuspendThread, rex::kernel::xboxkrnl::NtSuspendThread_entry)
REX_EXPORT(__imp__KeSuspendThread, rex::kernel::xboxkrnl::KeSuspendThread_entry)
REX_EXPORT(__imp__KeSetCurrentStackPointers, rex::kernel::xboxkrnl::KeSetCurrentStackPointers_entry)
REX_EXPORT(__imp__KeSetAffinityThread, rex::kernel::xboxkrnl::KeSetAffinityThread_entry)
REX_EXPORT(__imp__KeQueryBasePriorityThread, rex::kernel::xboxkrnl::KeQueryBasePriorityThread_entry)
REX_EXPORT(__imp__KeSetBasePriorityThread, rex::kernel::xboxkrnl::KeSetBasePriorityThread_entry)
REX_EXPORT(__imp__KeSetDisableBoostThread, rex::kernel::xboxkrnl::KeSetDisableBoostThread_entry)
REX_EXPORT(__imp__KeGetCurrentProcessType, rex::kernel::xboxkrnl::KeGetCurrentProcessType_entry)
REX_EXPORT(__imp__KeSetCurrentProcessType, rex::kernel::xboxkrnl::KeSetCurrentProcessType_entry)
REX_EXPORT(__imp__KeQueryPerformanceFrequency,
           rex::kernel::xboxkrnl::KeQueryPerformanceFrequency_entry)
REX_EXPORT(__imp__KeDelayExecutionThread, rex::kernel::xboxkrnl::KeDelayExecutionThread_entry)
REX_EXPORT(__imp__NtYieldExecution, rex::kernel::xboxkrnl::NtYieldExecution_entry)
REX_EXPORT(__imp__KeQuerySystemTime, rex::kernel::xboxkrnl::KeQuerySystemTime_entry)
REX_EXPORT(__imp__KeTlsAlloc, rex::kernel::xboxkrnl::KeTlsAlloc_entry)
REX_EXPORT(__imp__KeTlsFree, rex::kernel::xboxkrnl::KeTlsFree_entry)
REX_EXPORT(__imp__KeTlsGetValue, rex::kernel::xboxkrnl::KeTlsGetValue_entry)
REX_EXPORT(__imp__KeTlsSetValue, rex::kernel::xboxkrnl::KeTlsSetValue_entry)
REX_EXPORT(__imp__KeInitializeEvent, rex::kernel::xboxkrnl::KeInitializeEvent_entry)
REX_EXPORT(__imp__KeSetEvent, rex::kernel::xboxkrnl::KeSetEvent_entry)
REX_EXPORT(__imp__KePulseEvent, rex::kernel::xboxkrnl::KePulseEvent_entry)
REX_EXPORT(__imp__KeResetEvent, rex::kernel::xboxkrnl::KeResetEvent_entry)
REX_EXPORT(__imp__NtCreateEvent, rex::kernel::xboxkrnl::NtCreateEvent_entry)
REX_EXPORT(__imp__NtSetEvent, rex::kernel::xboxkrnl::NtSetEvent_entry)
REX_EXPORT(__imp__NtPulseEvent, rex::kernel::xboxkrnl::NtPulseEvent_entry)
REX_EXPORT(__imp__NtClearEvent, rex::kernel::xboxkrnl::NtClearEvent_entry)
REX_EXPORT(__imp__KeInitializeSemaphore, rex::kernel::xboxkrnl::KeInitializeSemaphore_entry)
REX_EXPORT(__imp__KeReleaseSemaphore, rex::kernel::xboxkrnl::KeReleaseSemaphore_entry)
REX_EXPORT(__imp__NtCreateSemaphore, rex::kernel::xboxkrnl::NtCreateSemaphore_entry)
REX_EXPORT(__imp__NtReleaseSemaphore, rex::kernel::xboxkrnl::NtReleaseSemaphore_entry)
REX_EXPORT(__imp__NtCreateMutant, rex::kernel::xboxkrnl::NtCreateMutant_entry)
REX_EXPORT(__imp__NtReleaseMutant, rex::kernel::xboxkrnl::NtReleaseMutant_entry)
REX_EXPORT(__imp__NtCreateTimer, rex::kernel::xboxkrnl::NtCreateTimer_entry)
REX_EXPORT(__imp__NtSetTimerEx, rex::kernel::xboxkrnl::NtSetTimerEx_entry)
REX_EXPORT(__imp__NtCancelTimer, rex::kernel::xboxkrnl::NtCancelTimer_entry)
REX_EXPORT(__imp__KeInitializeTimerEx, rex::kernel::xboxkrnl::KeInitializeTimerEx_entry)
REX_EXPORT(__imp__KeWaitForSingleObject, rex::kernel::xboxkrnl::KeWaitForSingleObject_entry)
REX_EXPORT(__imp__NtWaitForSingleObjectEx, rex::kernel::xboxkrnl::NtWaitForSingleObjectEx_entry)
REX_EXPORT(__imp__KeWaitForMultipleObjects, rex::kernel::xboxkrnl::KeWaitForMultipleObjects_entry)
REX_EXPORT(__imp__NtWaitForMultipleObjectsEx,
           rex::kernel::xboxkrnl::NtWaitForMultipleObjectsEx_entry)
REX_EXPORT(__imp__NtSignalAndWaitForSingleObjectEx,
           rex::kernel::xboxkrnl::NtSignalAndWaitForSingleObjectEx_entry)
REX_EXPORT(__imp__KfAcquireSpinLock, rex::kernel::xboxkrnl::KfAcquireSpinLock_entry)
REX_EXPORT(__imp__KfReleaseSpinLock, rex::kernel::xboxkrnl::KfReleaseSpinLock_entry)
REX_EXPORT(__imp__KeAcquireSpinLockAtRaisedIrql,
           rex::kernel::xboxkrnl::KeAcquireSpinLockAtRaisedIrql_entry)
REX_EXPORT(__imp__KeTryToAcquireSpinLockAtRaisedIrql,
           rex::kernel::xboxkrnl::KeTryToAcquireSpinLockAtRaisedIrql_entry)
REX_EXPORT(__imp__KeReleaseSpinLockFromRaisedIrql,
           rex::kernel::xboxkrnl::KeReleaseSpinLockFromRaisedIrql_entry)
REX_EXPORT(__imp__KeEnterCriticalRegion, rex::kernel::xboxkrnl::KeEnterCriticalRegion_entry)
REX_EXPORT(__imp__KeLeaveCriticalRegion, rex::kernel::xboxkrnl::KeLeaveCriticalRegion_entry)
REX_EXPORT(__imp__KeRaiseIrqlToDpcLevel, rex::kernel::xboxkrnl::KeRaiseIrqlToDpcLevel_entry)
REX_EXPORT(__imp__KfLowerIrql, rex::kernel::xboxkrnl::KfLowerIrql_entry)
REX_EXPORT(__imp__NtQueueApcThread, rex::kernel::xboxkrnl::NtQueueApcThread_entry)
REX_EXPORT(__imp__KeInitializeApc, rex::kernel::xboxkrnl::KeInitializeApc_entry)
REX_EXPORT(__imp__KeInsertQueueApc, rex::kernel::xboxkrnl::KeInsertQueueApc_entry)
REX_EXPORT(__imp__KeRemoveQueueApc, rex::kernel::xboxkrnl::KeRemoveQueueApc_entry)
REX_EXPORT(__imp__KiApcNormalRoutineNop, rex::kernel::xboxkrnl::KiApcNormalRoutineNop_entry)
REX_EXPORT(__imp__KeInitializeDpc, rex::kernel::xboxkrnl::KeInitializeDpc_entry)
REX_EXPORT(__imp__KeInsertQueueDpc, rex::kernel::xboxkrnl::KeInsertQueueDpc_entry)
REX_EXPORT(__imp__KeRemoveQueueDpc, rex::kernel::xboxkrnl::KeRemoveQueueDpc_entry)
REX_EXPORT(__imp__ExInitializeReadWriteLock, rex::kernel::xboxkrnl::ExInitializeReadWriteLock_entry)
REX_EXPORT(__imp__ExAcquireReadWriteLockExclusive,
           rex::kernel::xboxkrnl::ExAcquireReadWriteLockExclusive_entry)
REX_EXPORT(__imp__ExTryToAcquireReadWriteLockExclusive,
           rex::kernel::xboxkrnl::ExTryToAcquireReadWriteLockExclusive_entry)
REX_EXPORT(__imp__ExAcquireReadWriteLockShared,
           rex::kernel::xboxkrnl::ExAcquireReadWriteLockShared_entry)
REX_EXPORT(__imp__ExTryToAcquireReadWriteLockShared,
           rex::kernel::xboxkrnl::ExTryToAcquireReadWriteLockShared_entry)
REX_EXPORT(__imp__ExReleaseReadWriteLock, rex::kernel::xboxkrnl::ExReleaseReadWriteLock_entry)
REX_EXPORT(__imp__InterlockedPushEntrySList, rex::kernel::xboxkrnl::InterlockedPushEntrySList_entry)
REX_EXPORT(__imp__InterlockedPopEntrySList, rex::kernel::xboxkrnl::InterlockedPopEntrySList_entry)
REX_EXPORT(__imp__InterlockedFlushSList, rex::kernel::xboxkrnl::InterlockedFlushSList_entry)

REX_EXPORT_STUB(__imp__KeAlertResumeThread);
REX_EXPORT_STUB(__imp__KeAlertThread);
REX_EXPORT_STUB(__imp__KeBoostPriorityThread);
REX_EXPORT_STUB(__imp__KeCancelTimer);
REX_EXPORT_STUB(__imp__KeConnectInterrupt);
REX_EXPORT_STUB(__imp__KeContextFromKframes);
REX_EXPORT_STUB(__imp__KeContextToKframes);
REX_EXPORT_STUB(__imp__KeDisconnectInterrupt);
REX_EXPORT_STUB(__imp__KeFlushCacheRange);
REX_EXPORT_STUB(__imp__KeFlushCurrentEntireTb);
REX_EXPORT_STUB(__imp__KeFlushEntireTb);
REX_EXPORT_STUB(__imp__KeFlushMultipleTb);
REX_EXPORT_STUB(__imp__KeFlushUserModeCurrentTb);
REX_EXPORT_STUB(__imp__KeFlushUserModeTb);
REX_EXPORT_STUB(__imp__KeInitializeDeviceQueue);
REX_EXPORT_STUB(__imp__KeInitializeInterrupt);
REX_EXPORT_STUB(__imp__KeInitializeMutant);
REX_EXPORT_STUB(__imp__KeInitializeQueue);
// REX_EXPORT_STUB(__imp__KeInitializeTimerEx); -- implemented below
REX_EXPORT_STUB(__imp__KeIpiGenericCall);
REX_EXPORT_STUB(__imp__KeQueryBackgroundProcessors);
REX_EXPORT_STUB(__imp__KeQueryInterruptTime);
REX_EXPORT_STUB(__imp__KeRegisterDriverNotification);
REX_EXPORT_STUB(__imp__KeRestoreFloatingPointState);
REX_EXPORT_STUB(__imp__KeRestoreVectorUnitState);
REX_EXPORT_STUB(__imp__KeRetireDpcList);
REX_EXPORT_STUB(__imp__KeRundownQueue);
REX_EXPORT_STUB(__imp__KeSaveFloatingPointState);
REX_EXPORT_STUB(__imp__KeSaveVectorUnitState);
REX_EXPORT_STUB(__imp__KeSetBackgroundProcessors);
REX_EXPORT_STUB(__imp__KeSetPriorityClassThread);
REX_EXPORT_STUB(__imp__KeSetPriorityThread);
REX_EXPORT_STUB(__imp__KeSetTimer);
REX_EXPORT_STUB(__imp__KeSetTimerEx);
REX_EXPORT_STUB(__imp__KeStallExecutionProcessor);
REX_EXPORT_STUB(__imp__KeSweepDcacheRange);
REX_EXPORT_STUB(__imp__KeSweepIcacheRange);
REX_EXPORT_STUB(__imp__KeTestAlertThread);
REX_EXPORT(__imp__KfRaiseIrql, rex::kernel::xboxkrnl::KfRaiseIrql_entry)
