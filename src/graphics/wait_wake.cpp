#include "wait_wake.h"

#include <atomic>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace rex::graphics::wait_wake {
namespace {

constexpr size_t kMaxWaiters = 16;
std::atomic<void*> g_waiters[kMaxWaiters];
std::atomic<int32_t> g_waiter_count{0};

#ifdef _WIN32
// WaitOnAddress/WakeByAddressAll are Windows 8+ exports not present in every
// link SDK; resolve them dynamically (same pattern as SetThreadDescription in
// core/threading_win.cpp) so a missing symbol degrades to the old Sleep path
// instead of a link error or a startup crash.
typedef BOOL(WINAPI* WaitOnAddressFn)(volatile VOID*, volatile VOID*, SIZE_T, DWORD);
typedef VOID(WINAPI* WakeByAddressAllFn)(volatile VOID*);
WaitOnAddressFn g_wait_on_address = nullptr;
WakeByAddressAllFn g_wake_by_address_all = nullptr;

struct ResolveWaitApi {
  ResolveWaitApi() {
    if (HMODULE kernelbase = GetModuleHandleW(L"api-ms-win-core-synch-l1-2-2.dll")) {
      g_wait_on_address = reinterpret_cast<WaitOnAddressFn>(
          GetProcAddress(kernelbase, "WaitOnAddress"));
      g_wake_by_address_all = reinterpret_cast<WakeByAddressAllFn>(
          GetProcAddress(kernelbase, "WakeByAddressAll"));
    }
    if (!g_wait_on_address || !g_wake_by_address_all) {
      if (HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll")) {
        g_wait_on_address = reinterpret_cast<WaitOnAddressFn>(
            GetProcAddress(kernelbase, "WaitOnAddress"));
        g_wake_by_address_all = reinterpret_cast<WakeByAddressAllFn>(
            GetProcAddress(kernelbase, "WakeByAddressAll"));
      }
    }
  }
};
// Armed on first use (thread-safe static init); nulls everywhere if unavailable.
const ResolveWaitApi g_resolve_wait_api;
#endif

}  // namespace

bool Register(void* host_addr) {
#ifdef _WIN32
  if (!host_addr) {
    return false;
  }
  for (size_t i = 0; i < kMaxWaiters; ++i) {
    void* expected = nullptr;
    if (g_waiters[i].compare_exchange_strong(expected, host_addr,
                                              std::memory_order_acq_rel)) {
      g_waiter_count.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }
  return false;  // slots exhausted - caller keeps the Sleep path
#else
  return false;
#endif
}

void Deregister(void* host_addr) {
#ifdef _WIN32
  if (!host_addr) {
    return;
  }
  for (size_t i = 0; i < kMaxWaiters; ++i) {
    void* cur = g_waiters[i].load(std::memory_order_acquire);
    if (cur == host_addr &&
        g_waiters[i].compare_exchange_strong(cur, nullptr, std::memory_order_acq_rel)) {
      g_waiter_count.fetch_sub(1, std::memory_order_relaxed);
      return;
    }
  }
#endif
}

bool Wait(void* host_addr, uint32_t timeout_us) {
#ifdef _WIN32
  if (!host_addr || !g_wait_on_address) {
    return false;
  }
  // Compare against the current raw (not byte-swapped) dword: WaitOnAddress
  // only uses the snapshot to arm the kernel wait - the caller re-reads and
  // re-evaluates the real match condition after waking.
  uint32_t snapshot = *reinterpret_cast<volatile uint32_t*>(host_addr);
  DWORD timeout_ms = timeout_us / 1000;
  if (timeout_ms == 0) {
    timeout_ms = 1;
  }
  g_wait_on_address(host_addr, &snapshot, sizeof(uint32_t), timeout_ms);
  return true;
#else
  return false;
#endif
}

void WakeAll() {
#ifdef _WIN32
  if (g_waiter_count.load(std::memory_order_relaxed) <= 0) {
    return;  // fast path: nobody waiting (hot path - runs every vblank)
  }
  if (!g_wake_by_address_all) {
    return;
  }
  for (size_t i = 0; i < kMaxWaiters; ++i) {
    void* addr = g_waiters[i].load(std::memory_order_acquire);
    if (addr) {
      g_wake_by_address_all(addr);
    }
  }
#endif
}

}  // namespace rex::graphics::wait_wake
