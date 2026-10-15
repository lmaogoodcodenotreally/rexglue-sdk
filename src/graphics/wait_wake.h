#ifndef REX_GRAPHICS_WAIT_WAKE_H_
#define REX_GRAPHICS_WAIT_WAKE_H_

#include <cstdint>

// TEMP DIAGNOSTIC infrastructure (retail DOA4 frame-pacing work): event-driven
// wake-up for WAIT_REG_MEM memory polls. The guest paces itself by polling a
// vblank/flip flag in guest memory (0x1F56D006-style dwords) on the command
// processor thread with Sleep(wait/0x100) quanta, so the flag's update is only
// detected one sleep-tick late (~1.5 ms) and frame times quantize into sleep
// buckets. A waiter Register()s the host address of the polled dword, sleeps
// on it with WaitOnAddress, and GraphicsSystem::MarkVblank() WakeAll()s right
// after it has dispatched the guest vblank ISR (the code that clears the
// flag). The timeout always equals the old Sleep quantum, so conditions
// cleared by other writers behave exactly as before - never worse. Behind the
// `precise_vsync` cvar (default on); set false to restore plain Sleep polling.
// Supported on Windows only; other platforms fall back via the false returns.
namespace rex::graphics::wait_wake {

// Registers host_addr as about to be waited on. Returns false when
// unsupported (non-Windows, null, or no free slot) - the caller keeps the
// old Sleep() path for this call.
bool Register(void* host_addr);

// Removes a previous Register() (RAII-guarded by the caller).
void Deregister(void* host_addr);

// Sleeps on host_addr until WakeAll() signals it or timeout_us elapses,
// whichever comes first. Returns false when unsupported (caller falls back
// to the old Sleep path).
bool Wait(void* host_addr, uint32_t timeout_us);

// Wakes every registered waiter (called after the guest vblank ISR ran).
// No-op when nobody is waiting.
void WakeAll();

}  // namespace rex::graphics::wait_wake

#endif  // REX_GRAPHICS_WAIT_WAKE_H_
