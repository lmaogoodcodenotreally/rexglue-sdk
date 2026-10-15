/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2015 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/exception_handler.h>

#if REX_PLATFORM_WIN32

#include "platform_win.h"

#include <rex/assert.h>
#include <rex/logging.h>
#include <rex/math.h>

#include <rex/system/thread_state.h>
#include <rex/system/xmemory.h>

#include <cstdarg>
#include <cstddef>
#include <cstdio>

namespace rex::arch {

// Handle of the added VectoredExceptionHandler.
void* veh_handle_ = nullptr;
// Handle of the added VectoredContinueHandler.
void* vch_handle_ = nullptr;

// This can be as large as needed, but isn't often needed.
// As we will be sometimes firing many exceptions we want to avoid having to
// scan the table too much or invoke many custom handlers.
constexpr size_t kMaxHandlerCount = 8;

// All custom handlers, left-aligned and null terminated.
// Executed in order.
std::pair<ExceptionHandler::Handler, void*> handlers_[kMaxHandlerCount];

LONG CALLBACK ExceptionHandlerCallback(PEXCEPTION_POINTERS ex_info) {
  // Visual Studio SetThreadName.
  if (ex_info->ExceptionRecord->ExceptionCode == 0x406D1388) {
    return EXCEPTION_CONTINUE_SEARCH;
  }

  HostThreadContext thread_context;
  thread_context.rip = ex_info->ContextRecord->Rip;
  thread_context.eflags = ex_info->ContextRecord->EFlags;
  std::memcpy(thread_context.int_registers, &ex_info->ContextRecord->Rax,
              sizeof(thread_context.int_registers));
  std::memcpy(thread_context.xmm_registers, &ex_info->ContextRecord->Xmm0,
              sizeof(thread_context.xmm_registers));

  // https://msdn.microsoft.com/en-us/library/ms679331(v=vs.85).aspx
  // https://msdn.microsoft.com/en-us/library/aa363082(v=vs.85).aspx
  // TEMPORARY DIAGNOSTIC (do not commit): reached only on the fatal path.
  // Records which module the RIP actually belongs to (GetModuleHandle(nullptr)
  // is the exe, so faults inside a DLL mis-resolve), plus the instruction
  // bytes, so the faulting guest instruction can be recovered without depending
  // on symbolization.
  const auto log_unhandled = [&]() {
    const uintptr_t rip = static_cast<uintptr_t>(ex_info->ContextRecord->Rip);
    const uint32_t code = ex_info->ExceptionRecord->ExceptionCode;
    const uint32_t tid = ::GetCurrentThreadId();

    HMODULE mod = nullptr;
    wchar_t path[MAX_PATH] = {};
    uint8_t* mod_base = nullptr;
    if (::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCWSTR>(rip), &mod) &&
        mod) {
      ::GetModuleFileNameW(mod, path, MAX_PATH);
      mod_base = reinterpret_cast<uint8_t*>(mod);
    }

    // RIP points at the faulting instruction, which is readable as code.
    const auto* p = reinterpret_cast<const uint8_t*>(rip);
    char bytes[48] = {};
    for (int i = 0; i < 16; ++i) {
      std::snprintf(bytes + i * 3, 4, "%02X ", p[i]);
    }

    char name_a[MAX_PATH] = {};
    for (size_t i = 0; i < MAX_PATH && path[i]; ++i) {
      name_a[i] = static_cast<char>(path[i]);
    }

    const DWORD mxcsr = ex_info->ContextRecord->MxCsr;
    const uintptr_t rdi = static_cast<uintptr_t>(ex_info->ContextRecord->Rdi);

    // TEMPORARY DIAGNOSTIC (do not commit): caller frames, so a fault on a
    // thread that never logged anything else can still be traced back to the
    // SDK path that put it there. Symbolized offline against the PDBs.
    void* frames[24] = {};
    const USHORT frame_count = ::RtlCaptureStackBackTrace(0, 24, frames, nullptr);
    char bt[768] = {};
    size_t used = 0;
    for (USHORT i = 0; i < frame_count && used + 18 < sizeof(bt); ++i) {
      used += static_cast<size_t>(
          std::snprintf(bt + used, sizeof(bt) - used, "%p ", frames[i]));
    }

    if (code == STATUS_ACCESS_VIOLATION) {
      const auto* info = ex_info->ExceptionRecord->ExceptionInformation;
      REXLOG_ERROR(
          "UNHANDLED AV tid={} code 0x{:08X} RIP {:p} RVA 0x{:X} mxcsr 0x{:X} masks=0x{:X} "
          "rdi {:p} bytes [{}] fault {:p} :: {}\n   bt: {}",
          tid, code, reinterpret_cast<const void*>(rip),
          mod_base ? rip - reinterpret_cast<uintptr_t>(mod_base) : rip, mxcsr,
          (mxcsr >> 7) & 0x3F, reinterpret_cast<const void*>(rdi), bytes,
          reinterpret_cast<const void*>(static_cast<uintptr_t>(info[1])), name_a, bt);
    } else {
      REXLOG_ERROR(
          "UNHANDLED EXC tid={} code 0x{:08X} RIP {:p} RVA 0x{:X} mxcsr 0x{:X} masks=0x{:X} "
          "rdi {:p} bytes [{}] :: {}\n   bt: {}",
          tid, code, reinterpret_cast<const void*>(rip),
          mod_base ? rip - reinterpret_cast<uintptr_t>(mod_base) : rip, mxcsr,
          (mxcsr >> 7) & 0x3F, reinterpret_cast<const void*>(rdi), bytes, name_a, bt);
    }

    // TEMPORARY DIAGNOSTIC (do not commit): recompiled code keeps &PPCContext in
    // rsi - the instructions right after the faulting store are
    // `mov rax,[rsi+0x10]; add rax,0x60; mov [rsi+0x10],rax`, and PPCContext
    // puts r1 at +0x10 - so the register file the fault actually used can be
    // read straight off the faulting host frame instead of trusting whichever
    // ThreadState happens to be bound. Both are printed, because a mismatch
    // between them is itself a finding.
    static const char* const kNames[32] = {
        "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",
        "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
        "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
        "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31"};

    const auto readable = [](const void* p, size_t n) -> bool {
      auto* a = reinterpret_cast<const uint8_t*>(p);
      MEMORY_BASIC_INFORMATION mbi{};
      MEMORY_BASIC_INFORMATION mbi2{};
      if (!::VirtualQuery(a, &mbi, sizeof(mbi)) ||
          !::VirtualQuery(a + n - 1, &mbi2, sizeof(mbi2))) {
        return false;
      }
      const MEMORY_BASIC_INFORMATION* ms[2] = {&mbi, &mbi2};
      for (int i = 0; i < 2; ++i) {
        if (ms[i]->State != MEM_COMMIT || (ms[i]->Protect & PAGE_GUARD)) {
          return false;
        }
        const DWORD pr = ms[i]->Protect & 0xFF;
        if (pr == 0 || pr == PAGE_NOACCESS) {
          return false;
        }
      }
      return true;
    };

    const auto* ts = rex::runtime::ThreadState::Get();
    uint8_t* arena = nullptr;
    if (ts && ts->memory()) {
      arena = ts->memory()->virtual_membase();
    }

    const auto dump_ctx = [&](const ::PPCContext* c, const char* tag) {
      const size_t need = offsetof(::PPCContext, last_indirect_target) + sizeof(uint32_t);
      if (!c || !readable(c, need)) {
        REXLOG_ERROR("GUEST REGS {} <unreadable {:p}>", tag, static_cast<const void*>(c));
        return;
      }
      const uint64_t v[32] = {
          c->r0.u64,  c->r1.u64,  c->r2.u64,  c->r3.u64,  c->r4.u64,  c->r5.u64,
          c->r6.u64,  c->r7.u64,  c->r8.u64,  c->r9.u64,  c->r10.u64, c->r11.u64,
          c->r12.u64, c->r13.u64, c->r14.u64, c->r15.u64, c->r16.u64, c->r17.u64,
          c->r18.u64, c->r19.u64, c->r20.u64, c->r21.u64, c->r22.u64, c->r23.u64,
          c->r24.u64, c->r25.u64, c->r26.u64, c->r27.u64, c->r28.u64, c->r29.u64,
          c->r30.u64, c->r31.u64};
      char regs[1400] = {};
      size_t used = 0;
      for (int i = 0; i < 32 && used + 24 < sizeof(regs); ++i) {
        const int n = std::snprintf(regs + used, sizeof(regs) - used, "%s=%llX ", kNames[i],
                                    static_cast<unsigned long long>(v[i]));
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(regs) - used) {
          break;
        }
        used += static_cast<size_t>(n);
      }
      if (used < sizeof(regs)) {
        std::snprintf(regs + used, sizeof(regs) - used, "lr=%llX ctr=%llX msr=%X ind=%X",
                      static_cast<unsigned long long>(c->lr),
                      static_cast<unsigned long long>(c->ctr.u64), static_cast<unsigned>(c->msr),
                      static_cast<unsigned>(c->last_indirect_target));
      }
      REXLOG_ERROR("GUEST REGS {} {}", tag, regs);

      // Slots the epilogues reload from: [-8]/[-16]/[-24] are sub_823E25C0's
      // saved lr/r31/r30, [0] is sub_822F8360's stwu link, [80]/[88] are
      // sub_822F8360's own saved r31/lr. A run of zeros here while the fault is
      // inside that frame means r1 no longer points at that frame.
      const uint32_t sp = c->r1.u32;
      char slots[360] = {};
      size_t sused =
          static_cast<size_t>(std::snprintf(slots, sizeof(slots), "sp=0x%08X", sp));
      static const int kDelta[6] = {-24, -16, -8, 0, 80, 88};
      for (int d : kDelta) {
        if (sused >= sizeof(slots)) {
          break;
        }
        uint64_t val = 0;
        bool ok = false;
        if (arena) {
          auto* const p = arena + static_cast<uint32_t>(static_cast<int32_t>(sp) + d);
          if (readable(p, sizeof(val))) {
            std::memcpy(&val, p, sizeof(val));
            ok = true;
          }
        }
        const int n =
            ok ? std::snprintf(slots + sused, sizeof(slots) - sused, " [%d]=%016llX", d,
                               static_cast<unsigned long long>(val))
               : std::snprintf(slots + sused, sizeof(slots) - sused, " [%d]=?", d);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(slots) - sused) {
          break;
        }
        sused += static_cast<size_t>(n);
      }
      REXLOG_ERROR("GUEST FRAME {} {}", tag, slots);

      // How much of the stack around r1 has ever been written, and the span of
      // the live window. Frame slots ([-8]/[-16]/[0]) sitting inside a live
      // window but still zero means r1 addresses a *different* part of the
      // stack than the frames the faulting code just built.
      if (arena) {
        char hits[520] = {};
        size_t hused = 0;
        int nonzero = 0, unreadable = 0, samples = 0;
        int32_t lo_off = 0, hi_off = 0;
        bool have_range = false;
        for (int32_t off = -65536; off < 65536; off += 4) {
          auto* const p = arena + static_cast<uint32_t>(static_cast<int32_t>(sp) + off);
          if (!readable(p, 4)) {
            ++unreadable;
            continue;
          }
          uint32_t val = 0;
          std::memcpy(&val, p, 4);
          if (val != 0) {
            ++nonzero;
            if (!have_range) {
              lo_off = off;
              have_range = true;
            }
            hi_off = off;
            if (samples < 6 && hused + 44 < sizeof(hits)) {
              const int n = std::snprintf(hits + hused, sizeof(hits) - hused, " nz@%+d=%08X",
                                          off, val);
              if (n > 0) {
                hused += static_cast<size_t>(n);
              }
              ++samples;
            }
          }
        }
        REXLOG_ERROR("GUEST SCAN {} sp=0x{:08X} nonzero={} unreadable={} range=[{:+d},{:+d}]{}",
                     tag, sp, nonzero, unreadable, lo_off, hi_off, hits);
      }
      // Bounded dump of the populated slots between r1-4096 and r1+256, read
      // outward from r1 so the words closest to the live frame win the budget.
      // This is deliberately capped: an unbounded guest-memory walk here leaves
      // the faulting thread stuck inside the handler and the process never
      // terminates, so the run only ever ends by timeout.
      if (arena) {
        static constexpr int kMax = 60;
        static constexpr int kLo = -4096, kHi = 256;
        int32_t offs[kMax] = {};
        uint32_t vals[kMax] = {};
        int n_found = 0;
        for (int32_t off = kHi; off >= kLo && n_found < kMax; off -= 4) {
          auto* const p = arena + static_cast<uint32_t>(static_cast<int32_t>(sp) + off);
          if (!readable(p, 4)) {
            continue;
          }
          uint32_t val = 0;
          std::memcpy(&val, p, 4);
          if (val) {
            offs[n_found] = off;
            vals[n_found] = val;
            ++n_found;
          }
        }
        char dump[1400] = {};
        size_t dused = 0;
        for (int i = n_found - 1; i >= 0 && dused + 24 < sizeof(dump); --i) {
          const int n = std::snprintf(dump + dused, sizeof(dump) - dused, " %+d=%08X", offs[i],
                                      vals[i]);
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(dump) - dused) {
            break;
          }
          dused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST WIN {} sp=0x{:08X} slots={}{}{}", tag, sp, n_found,
                     n_found ? dump : " <zero in [-4096,+256]>",
                     n_found >= kMax ? " (capped)" : "");

        // The frame chain is intact below r1 but the slots r1 itself points at
        // read back as zero, so report the exact zero run containing r1. Its
        // bounds say whether this is a released region, a re-committed page, or
        // a deliberate wipe of the top of the stack.
        int32_t zlo = 0, zhi = 0;
        for (int32_t off = 0; off >= -65536; off -= 4) {
          auto* const p = arena + static_cast<uint32_t>(static_cast<int32_t>(sp) + off);
          if (!readable(p, 4)) {
            break;
          }
          uint32_t val = 0;
          std::memcpy(&val, p, 4);
          if (val) {
            break;
          }
          zlo = off;
        }
        for (int32_t off = 0; off <= 65536; off += 4) {
          auto* const p = arena + static_cast<uint32_t>(static_cast<int32_t>(sp) + off);
          if (!readable(p, 4)) {
            break;
          }
          uint32_t val = 0;
          std::memcpy(&val, p, 4);
          if (val) {
            break;
          }
          zhi = off;
        }
        REXLOG_ERROR("GUEST ZEROBAND {} sp=0x{:08X} [{:+d},{:+d}] addr=0x{:08X}..0x{:08X}", tag,
                     sp, zlo, zhi, static_cast<uint32_t>(static_cast<int32_t>(sp) + zlo),
                     static_cast<uint32_t>(static_cast<int32_t>(sp) + zhi));
      }
      if (arena) {
        auto* const f = arena + 0x82CE80C4u;
        auto* const i = arena + 0x82CE80C8u;
        REXLOG_ERROR("GUEST PROBE {} flag@82CE80C4={} idx@82CE80C8={}", tag,
                     readable(f, 1) ? static_cast<int>(*f) : -1,
                     readable(i, 1) ? static_cast<int>(*i) : -1);
      }
      // TEMP DIAGNOSTIC: the int16 stream walker (sub_822D9020) dereferences
      // the cursor at block+4, with the block pointer in r30 and the slot id in
      // r31. Dump the block header immediately: word0 says whether the offset
      // fixup ran (it stamps 0x80000001/2 before fixing), word1 is the cursor
      // that faulted, and the surrounding words show whether the block holds
      // data of its own type or stale heap contents. The descriptor slot shows
      // which block the caller picked for this id, and the words just before the
      // block fingerprint whatever the heap region held previously.
      if (arena) {
        const auto be32 = [&](uint32_t addr, uint32_t* out) -> bool {
          auto* const p = arena + addr;
          if (!readable(p, 4)) {
            return false;
          }
          uint32_t v = 0;
          std::memcpy(&v, p, 4);
          *out = (v >> 24) | ((v >> 8) & 0x0000FF00u) | ((v << 8) & 0x00FF0000u) |
                 (v << 24);
          return true;
        };
        const uint32_t blk = c->r30.u32;
        char hdr[200] = {};
        size_t hused = 0;
        for (uint32_t k = 0; k < 12; ++k) {
          uint32_t w = 0xDEADBEEFu;
          const bool ok = be32(blk + k * 4, &w);
          const int n = std::snprintf(hdr + hused, sizeof(hdr) - hused, " %u=%08X%s", k, w,
                                      ok ? "" : "?");
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(hdr) - hused) {
            break;
          }
          hused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST BLK {} r30=0x{:08X} hdr={}", tag, blk, hdr);

        const uint32_t id = c->r31.u32;
        if (id < 0x100u) {
          const uint32_t dw = 0x82CE7058u + id * 48u;
          char dsc[200] = {};
          size_t dused = 0;
          for (uint32_t k = 0; k < 12; ++k) {
            uint32_t w = 0xDEADBEEFu;
            const bool ok = be32(dw + k * 4, &w);
            const int n = std::snprintf(dsc + dused, sizeof(dsc) - dused, " %u=%08X%s", k, w,
                                        ok ? "" : "?");
            if (n <= 0 || static_cast<size_t>(n) >= sizeof(dsc) - dused) {
              break;
            }
            dused += static_cast<size_t>(n);
          }
          REXLOG_ERROR("GUEST BLKDESC {} id={} addr=0x{:08X} words={}", tag, id, dw, dsc);
        }

        // TEMP DIAGNOSTIC: dump all four walker descriptor rows, not just the
        // faulting one. The other slots walk without faulting, so their rows show
        // what a valid (block pointer, file window) pair looks like next to the
        // bad one; the words at field9 identify which file each slot owns and the
        // first words of each block say whether that slot's header was stamped.
        const auto heapish = [](uint32_t v) -> bool {
          return (v >= 0x100000u && v < 0x04000000u) || (v >= 0x82000000u && v < 0x84000000u);
        };
        for (uint32_t sid = 0; sid < 4; ++sid) {
          const uint32_t rw_addr = 0x82CE7058u + sid * 48u;
          char rw[176] = {};
          size_t rused = 0;
          for (uint32_t k = 0; k < 12; ++k) {
            uint32_t w = 0xDEADBEEFu;
            const bool ok = be32(rw_addr + k * 4, &w);
            const int n = std::snprintf(rw + rused, sizeof(rw) - rused, " %u=%08X%s", k, w,
                                        ok ? "" : "?");
            if (n <= 0 || static_cast<size_t>(n) >= sizeof(rw) - rused) {
              break;
            }
            rused += static_cast<size_t>(n);
          }
          uint32_t f0 = 0xDEADBEEFu, f9 = 0, f10 = 0;
          be32(rw_addr, &f0);
          be32(rw_addr + 36, &f9);
          be32(rw_addr + 40, &f10);
          char bh[76] = " <range>";
          if (heapish(f9)) {
            size_t bused = 0;
            bh[0] = 0;
            for (uint32_t k = 0; k < 4; ++k) {
              uint32_t w = 0xDEADBEEFu;
              const bool ok = be32(f9 + k * 4, &w);
              const int n = std::snprintf(bh + bused, sizeof(bh) - bused, " %08X%s", w,
                                          ok ? "" : "?");
              if (n <= 0 || static_cast<size_t>(n) >= sizeof(bh) - bused) {
                break;
              }
              bused += static_cast<size_t>(n);
            }
          }
          char bk[76] = "";
          if (heapish(f0)) {
            size_t kused = 0;
            for (uint32_t k = 0; k < 4; ++k) {
              uint32_t w = 0xDEADBEEFu;
              const bool ok = be32(f0 + k * 4, &w);
              const int n = std::snprintf(bk + kused, sizeof(bk) - kused, " %08X%s", w,
                                          ok ? "" : "?");
              if (n <= 0 || static_cast<size_t>(n) >= sizeof(bk) - kused) {
                break;
              }
              kused += static_cast<size_t>(n);
            }
          } else {
            std::snprintf(bk, sizeof(bk), " <range>");
          }
          REXLOG_ERROR("GUEST ROWTAB {} id={} f9=0x{:08X} f10=0x{:08X} row={} buf={} blk={}",
                       tag, sid, f9, f10, rw, bh, bk);
        }

        char pre[160] = {};
        size_t pused = 0;
        for (int k = -8; k < 0; ++k) {
          uint32_t w = 0xDEADBEEFu;
          const bool ok = be32(blk + static_cast<uint32_t>(k) * 4, &w);
          const int n = std::snprintf(pre + pused, sizeof(pre) - pused, " %+d=%08X%s", k * 4, w,
                                      ok ? "" : "?");
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(pre) - pused) {
            break;
          }
          pused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST BLKPRE {} r30=0x{:08X} pre={}", tag, blk, pre);
      }
      // TEMP DIAGNOSTIC: the pointer-relocation walker keeps its block base in a
      // global and derives every object as base+offset before dereferencing it.
      // A fault on such a pointer is either a stale base or corrupt block data,
      // so report the base, the object r8 held, and the words of that object.
      if (arena) {
        const auto g32 = [&](uint32_t addr) -> uint32_t {
          auto* const p = arena + addr;
          if (!readable(p, 4)) {
            return 0xDEADBEEFu;
          }
          uint32_t v = 0;
          std::memcpy(&v, p, 4);
          return (v >> 24) | ((v >> 8) & 0x0000FF00u) | ((v << 8) & 0x00FF0000u) |
                 (v << 24);
        };
        const uint32_t base = g32(0x82C890E0u);
        const uint32_t obj = c->r8.u32;
        const auto g8 = [&](uint32_t addr) -> int {
          auto* const p = arena + addr;
          return readable(p, 1) ? static_cast<int>(*p) : -1;
        };
        // The walker reads [tag:u8][value:u32] entries at base+8k. The tag is
        // what picks the fixup routine, so a mismatch between the tag and the
        // shape of the object it points at is the failure under investigation.
        char tab[340] = {};
        size_t tabused = 0;
        for (uint32_t k = 0; k < 7; ++k) {
          const int n = std::snprintf(tab + tabused, sizeof(tab) - tabused, " [%u]t=%d v=%08X",
                                      k, g8(base + k * 8), g32(base + k * 8 + 4));
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(tab) - tabused) {
            break;
          }
          tabused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST RELTAB {} base=0x{:08X}{}", tag, base, tab);
        // TEMP DIAGNOSTIC: compare the other entries' objects. If they hold
        // pointers while the faulting tag-1 object holds floats, the mismatch
        // is local to that object; if every entry holds floats, the walker is
        // walking the wrong block entirely.
        const uint32_t picks[3] = {1u, 2u, 6u};
        for (uint32_t e : picks) {
          const int et = g8(base + e * 8);
          const uint32_t ev = g32(base + e * 8 + 4);
          char ow[300] = {};
          size_t oused = 0;
          for (uint32_t k = 0; k < 8; ++k) {
            const int n = std::snprintf(ow + oused, sizeof(ow) - oused, " %08X", g32(ev + k * 4));
            if (n <= 0 || static_cast<size_t>(n) >= sizeof(ow) - oused) {
              break;
            }
            oused += static_cast<size_t>(n);
          }
          REXLOG_ERROR("GUEST RELOBJ {} [{}] t={} v=0x{:08X}{}", tag, e, et, ev, ow);
        }
        // sub_822C8D78 was handed *(obj+92) and returned without faulting,
        // which cannot happen for a bogus address; report what lives there.
        const uint32_t sub1 = g32(obj + 92);
        char sw[300] = {};
        size_t sused = 0;
        for (uint32_t k = 0; k < 7; ++k) {
          const int n = std::snprintf(sw + sused, sizeof(sw) - sused, " %08X", g32(sub1 + k * 4));
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(sw) - sused) {
            break;
          }
          sused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST RELSUB {} ptr(obj+92)=0x{:08X}{}", tag, sub1, sw);
        // TEMP DIAGNOSTIC: the caller indexes a 48-byte descriptor table at
        // 0x82CE7068 and passes descriptor[0] (the block) to the walker. The
        // declared size lives in that descriptor, so comparing it against the
        // object address separates "walker used the wrong base" from "block
        // contents are corrupt".
        char blks[300] = {};
        size_t bused = 0;
        char desc[340] = {};
        size_t dused = 0;
        for (uint32_t i2 = 0; i2 < 64; ++i2) {
          const uint32_t dw = 0x82CE7068u + i2 * 48u;
          const uint32_t blk = g32(dw);
          if (blk == 0u || blk == 0xDEADBEEFu) {
            continue;
          }
          const int n = std::snprintf(blks + bused, sizeof(blks) - bused, " %u:%08X", i2, blk);
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(blks) - bused) {
            break;
          }
          bused += static_cast<size_t>(n);
          if (blk != base) {
            continue;
          }
          for (uint32_t w = 0; w < 12 && dused + 14 < sizeof(desc); ++w) {
            const int m = std::snprintf(desc + dused, sizeof(desc) - dused, " %u=%08X", w,
                                        g32(dw + w * 4));
            if (m <= 0 || static_cast<size_t>(m) >= sizeof(desc) - dused) {
              break;
            }
            dused += static_cast<size_t>(m);
          }
        }
        REXLOG_ERROR("GUEST RELBLKS {}{}", tag, blks);
        REXLOG_ERROR("GUEST RELDESC {} i=? base=0x{:08X}{}", tag, base, desc);
        // TEMP DIAGNOSTIC: the walker relocates a stored value only when it is
        // smaller than the block base. A block allocated below its own offsets
        // therefore leaves them looking absolute. Show every entry resolved
        // both ways so the shape that actually holds pointers can be picked.
        char alt[460] = {};
        size_t aused = 0;
        for (uint32_t e = 0; e < 4; ++e) {
          const int et = g8(base + e * 8);
          const uint32_t ev = g32(base + e * 8 + 4);
          const int n = std::snprintf(
              alt + aused, sizeof(alt) - aused,
              " [%u]t=%d v=%08X|a:%08X %08X %08X %08X|b:%08X %08X %08X %08X", e, et, ev, g32(ev),
              g32(ev + 4), g32(ev + 8), g32(ev + 12), g32(base + ev), g32(base + ev + 4),
              g32(base + ev + 8), g32(base + ev + 12));
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(alt) - aused) {
            break;
          }
          aused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST RELALT {}{}", tag, alt);
        // Whether the block carries guest pointers at all: a block made purely
        // of offsets/floats has nothing for a pointer fixup to rewrite.
        char ptrs[360] = {};
        size_t pused = 0;
        uint32_t phits = 0;
        for (uint32_t a = base; a < base + 0x2500u; a += 4) {
          const uint32_t v = g32(a);
          if ((v & 0xFF000000u) != 0x82000000u) {
            continue;
          }
          ++phits;
          if (pused + 22 >= sizeof(ptrs)) {
            continue;
          }
          const int n = std::snprintf(ptrs + pused, sizeof(ptrs) - pused, " %08X=%08X", a, v);
          if (n > 0 && static_cast<size_t>(n) < sizeof(ptrs) - pused) {
            pused += static_cast<size_t>(n);
          }
        }
        REXLOG_ERROR("GUEST RELPTRS {} n={}{}", tag, phits, ptrs);
        // TEMP DIAGNOSTIC: if the stored values are relative offsets the walker
        // failed to add (they are larger than the block base, so the usual
        // "looks like an offset" test misses them), the real objects sit at
        // base+v. Those objects are what the pointer fixup would rewrite, so
        // show exactly the +72..+96 fields it dereferences, resolved both ways.
        char alt2[460] = {};
        size_t a2used = 0;
        for (uint32_t e = 0; e < 3; ++e) {
          const uint32_t ev = g32(base + e * 8 + 4);
          const uint32_t av = base + ev;
          const int n = std::snprintf(
              alt2 + a2used, sizeof(alt2) - a2used,
              " [%u]v=%08X|abs72:%08X abs76:%08X|rel72:%08X %08X %08X %08X %08X %08X %08X", e, ev,
              g32(ev + 72), g32(ev + 76), g32(av + 72), g32(av + 76), g32(av + 80), g32(av + 84),
              g32(av + 88), g32(av + 92), g32(av + 96));
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(alt2) - a2used) {
            break;
          }
          a2used += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST RELALT2 {}{}", tag, alt2);
        // The offset interpretation also predicts a second copy of the data at
        // base+block; look there for guest pointers, and report how much of the
        // region is even mapped so an empty result is meaningful.
        char ptrs2[360] = {};
        size_t p2used = 0;
        uint32_t phits2 = 0;
        uint32_t pread = 0;
        for (uint32_t a = 0x60000u; a < 0x63000u; a += 4) {
          const uint32_t v = g32(a);
          if (v == 0xDEADBEEFu) {
            continue;
          }
          ++pread;
          if ((v & 0xFF000000u) != 0x82000000u) {
            continue;
          }
          ++phits2;
          if (p2used + 22 >= sizeof(ptrs2)) {
            continue;
          }
          const int n = std::snprintf(ptrs2 + p2used, sizeof(ptrs2) - p2used, " %08X=%08X", a, v);
          if (n > 0 && static_cast<size_t>(n) < sizeof(ptrs2) - p2used) {
            p2used += static_cast<size_t>(n);
          }
        }
        REXLOG_ERROR("GUEST RELPTRS2 {} readable={} n={}{}", tag, pread, phits2, ptrs2);
        // TEMP DIAGNOSTIC: attribute the block's very low address. A heap
        // allocation shows a header before it and a neighbour past its declared
        // size; a static placement shows an otherwise empty low region.
        char hraw[170] = {};
        size_t hused = 0;
        for (uint32_t a = 0x30018u; a < 0x30060u; ++a) {
          auto* const hp = arena + a;
          const int n = std::snprintf(hraw + hused, sizeof(hraw) - hused, "%02X",
                                      readable(hp, 1) ? static_cast<unsigned>(*hp) : 0xFFu);
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(hraw) - hused) {
            break;
          }
          hused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST RELHDR {} {}", tag, hraw);
        char prew[210] = {};
        size_t prused = 0;
        for (uint32_t a = 0x2FFF0u; a < 0x30018u; a += 4) {
          const int n = std::snprintf(prew + prused, sizeof(prew) - prused, " %08X=%08X", a,
                                      g32(a));
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(prew) - prused) {
            break;
          }
          prused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST RELPRE {}{}", tag, prew);
        char endp[240] = {};
        size_t eused = 0;
        for (uint32_t a = 0x92540u; a < 0x92580u; a += 4) {
          const int n = std::snprintf(endp + eused, sizeof(endp) - eused, " %08X=%08X", a,
                                      g32(a));
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(endp) - eused) {
            break;
          }
          eused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST RELEND {}{}", tag, endp);
        char lmap[300] = {};
        size_t lused = 0;
        for (uint32_t pg = 0; pg < 64; ++pg) {
          bool nz = false;
          for (uint32_t k = 0; k < 16 && !nz; ++k) {
            const uint32_t v = g32(pg * 0x1000u + k * 0x40u);
            nz = (v != 0u && v != 0xDEADBEEFu);
          }
          if (!nz) {
            continue;
          }
          const int n = std::snprintf(lmap + lused, sizeof(lmap) - lused, " %03X", pg);
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(lmap) - lused) {
            break;
          }
          lused += static_cast<size_t>(n);
        }
        REXLOG_ERROR("GUEST RELLOW {} nonzero-pages{}", tag, lmap);
        // Words of the object spanning the fields sub_822C8EE8 rewrites (+72
        // .. +96); +96 is the pointer handed to the routine that faulted.
        char words[420] = {};
        size_t wused = 0;
        for (uint32_t k = 14; k < 27; ++k) {
          const int n = std::snprintf(words + wused, sizeof(words) - wused, " +%u=%08X",
                                      k * 4, g32(obj + k * 4));
          if (n <= 0 || static_cast<size_t>(n) >= sizeof(words) - wused) {
            break;
          }
          wused += static_cast<size_t>(n);
        }
        REXLOG_ERROR(
            "GUEST RELOC {} base@82C890E0=0x{:08X} obj(r8)=0x{:08X} obj+96=0x{:08X} "
            "walker(r6)=0x{:08X} stored@base+4=0x{:08X}",
            tag, base, obj, g32(obj + 96), c->r6.u32, g32(base + 4));
        REXLOG_ERROR("GUEST RELOCOBJ {}{}", tag, words);
        // TEMP DIAGNOSTIC: sub_822CF0B0 (the byte-stream parser) is the only
        // writer of the parent pointer at array[idx].obj+124, where
        // obj = 0x82BACA20 + idx*0x30010 + 17800. sub_822CDC58 dereferences that
        // parent on entry, so a zero there means child 0 was never created for
        // chunk 0. The parser stamps its own state when it finishes (byte0=0xFF,
        // byte1=0, node count in byte11, per-type counters in bytes 2..10), so
        // reporting those bytes separates "parser never ran" from "the stream
        // it walked simply had no type-16 record". The chunk descriptor slots
        // and the stream head say which byte stream it was given.
        const uint32_t arr = 0x82BACA20u;
        const uint32_t stride = 0x30010u;
        char hdr0[80] = {}, hdr1[80] = {}, tbl[300] = {}, shd[80] = {}, bhd[80] = {};
        auto append = [](char* dst, size_t cap, size_t& used, const char* fmt, ...) {
          va_list ap;
          va_start(ap, fmt);
          const int n = std::vsnprintf(dst + used, cap - used, fmt, ap);
          va_end(ap);
          if (n > 0 && static_cast<size_t>(n) < cap - used) {
            used += static_cast<size_t>(n);
          }
        };
        size_t u0 = 0, u1 = 0, ut = 0, us = 0, ub = 0;
        for (uint32_t k = 0; k < 12; ++k) {
          append(hdr0, sizeof(hdr0), u0, " %d", g8(arr + k));
          append(hdr1, sizeof(hdr1), u1, " %d", g8(arr + stride + k));
        }
        for (uint32_t w = 0; w < 14; ++w) {
          append(tbl, sizeof(tbl), ut, " +%u=%08X", w * 4, g32(0x82CE7040u + w * 4));
        }
        const uint32_t stream = g32(0x82CE7068u);
        for (uint32_t k = 0; k < 16; ++k) {
          append(shd, sizeof(shd), us, " %d", g8(stream + k));
          append(bhd, sizeof(bhd), ub, " %d", g8(0x00120018u + k));
        }
        // The parser walks 8-byte records until it meets a 0xFF type byte, so
        // the type of every record it could have reached says whether it stopped
        // at a real end-of-stream or bailed out in the middle of the table.
        const uint32_t tagbase = (stream > 0x1000u && stream != 0xDEADBEEFu) ? stream : 0x120018u;
        char tags[220] = {};
        size_t ug = 0;
        for (uint32_t k = 0; k < 64 && ug + 8 < sizeof(tags); ++k) {
          append(tags, sizeof(tags), ug, " %d", g8(tagbase + k * 8u));
        }
        REXLOG_ERROR(
            "GUEST PARSE {} hdr0={} obj0+124=0x{:08X} m3718C={} f37189={} hdr1={} "
            "obj1+124=0x{:08X}",
            tag, hdr0, g32(arr + 17800u + 124u), g8(arr + 0x3718Cu), g8(arr + 0x37189u), hdr1,
            g32(arr + stride + 17800u + 124u));
        REXLOG_ERROR("GUEST PARSETBL {}{}", tag, tbl);
        REXLOG_ERROR("GUEST PARSESTR {} stream@82CE7068=0x{:08X} head={} blk@120018={}", tag,
                     stream, shd, bhd);
        REXLOG_ERROR("GUEST PARSETAGS {} base=0x{:08X}{}", tag, tagbase, tags);
      }
    };

    const auto& hctx = *ex_info->ContextRecord;
    REXLOG_ERROR("GUEST META rsi={:p} rbp={:p} rax={:p} rdi={:p} ts={:p} tid={:08X}",
                 reinterpret_cast<const void*>(static_cast<uintptr_t>(hctx.Rsi)),
                 reinterpret_cast<const void*>(static_cast<uintptr_t>(hctx.Rbp)),
                 reinterpret_cast<const void*>(static_cast<uintptr_t>(hctx.Rax)),
                 reinterpret_cast<const void*>(static_cast<uintptr_t>(hctx.Rdi)),
                 static_cast<const void*>(ts ? ts->context() : nullptr),
                 ts ? ts->thread_id() : 0xFFFFFFFFu);
    dump_ctx(reinterpret_cast<const ::PPCContext*>(static_cast<uintptr_t>(hctx.Rsi)), "rsi");
    if (ts && ts->context()) {
      dump_ctx(ts->context(), "ts");
    }
  };

  Exception ex;
  switch (ex_info->ExceptionRecord->ExceptionCode) {
    case STATUS_ILLEGAL_INSTRUCTION:
      ex.InitializeIllegalInstruction(&thread_context);
      break;
    case STATUS_ACCESS_VIOLATION: {
      Exception::AccessViolationOperation access_violation_operation;
      switch (ex_info->ExceptionRecord->ExceptionInformation[0]) {
        case 0:
          access_violation_operation = Exception::AccessViolationOperation::kRead;
          break;
        case 1:
          access_violation_operation = Exception::AccessViolationOperation::kWrite;
          break;
        default:
          access_violation_operation = Exception::AccessViolationOperation::kUnknown;
          break;
      }
      ex.InitializeAccessViolation(&thread_context,
                                   ex_info->ExceptionRecord->ExceptionInformation[1],
                                   access_violation_operation);
    } break;
    default:
      // Unknown/unhandled type (float faults, stack overflow, ...).
      log_unhandled();
      return EXCEPTION_CONTINUE_SEARCH;
  }

  for (size_t i = 0; i < rex::countof(handlers_) && handlers_[i].first; ++i) {
    if (handlers_[i].first(&ex, handlers_[i].second)) {
      // Exception handled.
      ex_info->ContextRecord->Rip = thread_context.rip;
      ex_info->ContextRecord->EFlags = thread_context.eflags;
      uint32_t modified_register_index;
      uint16_t modified_int_registers_remaining = ex.modified_int_registers();
      while (rex::bit_scan_forward(modified_int_registers_remaining, &modified_register_index)) {
        modified_int_registers_remaining &= ~(UINT16_C(1) << modified_register_index);
        (&ex_info->ContextRecord->Rax)[modified_register_index] =
            thread_context.int_registers[modified_register_index];
      }
      uint16_t modified_xmm_registers_remaining = ex.modified_xmm_registers();
      while (rex::bit_scan_forward(modified_xmm_registers_remaining, &modified_register_index)) {
        modified_xmm_registers_remaining &= ~(UINT16_C(1) << modified_register_index);
        std::memcpy(&ex_info->ContextRecord->Xmm0 + modified_register_index,
                    &thread_context.xmm_registers[modified_register_index], sizeof(vec128_t));
      }
      return EXCEPTION_CONTINUE_EXECUTION;
    }
  }

  // No handler claimed it: it is about to terminate the process. Routine MMIO
  // access violations are handled and resumed above, so this only logs fatals.
  log_unhandled();
  return EXCEPTION_CONTINUE_SEARCH;
}

void ExceptionHandler::Install(Handler fn, void* data) {
  if (!veh_handle_) {
    veh_handle_ = AddVectoredExceptionHandler(1, ExceptionHandlerCallback);

    if (IsDebuggerPresent()) {
      // TODO(benvanik): do we need a continue handler if a debugger is
      // attached?
      // vch_handle_ = AddVectoredContinueHandler(1, ExceptionHandlerCallback);
    }
  }

  for (size_t i = 0; i < rex::countof(handlers_); ++i) {
    if (!handlers_[i].first) {
      handlers_[i].first = fn;
      handlers_[i].second = data;
      return;
    }
  }
  assert_always("Too many exception handlers installed");
}

void ExceptionHandler::Uninstall(Handler fn, void* data) {
  for (size_t i = 0; i < rex::countof(handlers_); ++i) {
    if (handlers_[i].first == fn && handlers_[i].second == data) {
      for (; i < rex::countof(handlers_) - 1; ++i) {
        handlers_[i] = handlers_[i + 1];
      }
      handlers_[i].first = nullptr;
      handlers_[i].second = nullptr;
      break;
    }
  }

  bool has_any = false;
  for (size_t i = 0; i < rex::countof(handlers_); ++i) {
    if (handlers_[i].first) {
      has_any = true;
      break;
    }
  }
  if (!has_any) {
    if (veh_handle_) {
      RemoveVectoredExceptionHandler(veh_handle_);
      veh_handle_ = nullptr;
    }
    if (vch_handle_) {
      RemoveVectoredContinueHandler(vch_handle_);
      vch_handle_ = nullptr;
    }
  }
}

}  // namespace rex::arch

#endif  // REX_PLATFORM_WIN32
