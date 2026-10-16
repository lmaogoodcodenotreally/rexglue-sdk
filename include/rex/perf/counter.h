/**
 * @file        perf/counter.h
 * @brief       Performance counter registry and profiler
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include <rex/cvar.h>

#ifdef REXGLUE_ENABLE_PROFILING
#include <tracy/Tracy.hpp>
#endif

namespace rex::perf {

enum class CounterId : uint16_t {
  // Frame
  kFrameTimeUs,
  kFps,

  // GPU
  kDrawCalls,
  kCommandBufferStalls,
  kVerticesProcessed,

  // Audio
  kXmaFramesDecoded,
  kAudioFrameLatencyUs,
  kBufferQueueDepth,

  // Dispatch
  kFunctionsDispatched,
  kInterruptDispatches,

  // Threading
  kActiveThreads,
  kApcQueueDepth,
  kCriticalRegionContentions,

  // Caches
  kTextureCacheHits,
  kTextureCacheMisses,
  kPipelineCacheHits,
  kPipelineCacheMisses,

  // GPU command-processing breakdown (TEMP DIAGNOSTIC: retail DOA4 perf work;
  // splits swap-to-swap frame time into host packet execution, ring-empty
  // wait, and the draw/primitive path). Accumulators, zeroed each frame.
  kGpuExecUs,
  kGpuStallUs,
  kGpuDrawUs,
  kGpuPresentUs,
  kGpuSubmitUs,
  kGpuWaitUs,
  kGpuPackets,
  kGpuRegUs,
  kGpuConstUs,
  kGpuImmUs,
  kGpuType3Us,
  kGpuPrimUs,
  kGpuRtUs,
  kGpuPipeUs,
  kGpuStateUs,
  kGpuBindUs,
  kGpuRecordUs,
  kGpuEntryUs,
  kGpuModsUs,
  kGpuXlatUs,
  kGpuTexUs,
  kGpuResidUs,
  kGpuEvtUs,
  kGpuMiscUs,
  kGpuCopyUs,
  kGpuRegwCount,
  kGpuNopCount,
  kGpuIbCount,
  kGpuImmHitCount,

  // TEMP DIAGNOSTIC (retail DOA4 class-A attribution): run R35 measured
  // gpu_tex_us = 705 + 364 * texture_cache_misses (corr 0.993) -- the
  // first-use texture work burns ~364us per new binding, but correlation
  // alone cannot say which step pays it. These split the RequestTextures
  // path into binding prep (fetch-constant parse, FindOrCreateTexture,
  // shared-memory range requests), the data load/commit loop (GPU untile
  // dispatch + copy recording), and descriptor binding (SRV creation),
  // plus a committed-load count to compare against the miss count.
  // Accumulators, zeroed each frame.
  kGpuTexPrepUs,
  kGpuTexLoadUs,
  kGpuTexBindUs,
  kGpuTexLoadCount,

  // TEMP DIAGNOSTIC (retail DOA4 class-A attribution, second level): R36pre
  // showed gpu_tex_prep_us = 442 + 381 * texture_cache_misses (corr 1.000)
  // -- the binding-prep phase pays the ~380us per new texture, with zero
  // lock contention and load/bind phases nearly free. These bisect prep
  // into the binding loop (fetch parse + FindOrCreateTexture +
  // PrepareTextureLoad) vs the batched shared-memory RequestRanges (page
  // upload), and RequestRanges' upload pipeline into pool page request vs
  // guest-memory memcpy.
  kGpuTexFindUs,
  kGpuTexRangeUs,
  kGpuTexUpPoolUs,
  kGpuTexUpCopyUs,

  // TEMP DIAGNOSTIC (retail DOA4 class-A attribution, third level): R36
  // showed the find loop itself costs 629us per cache miss (corr 0.804) with
  // frames like 31.7ms find / 25 loads / 1 descriptor miss -- so the loop's
  // per-texture work is the driver, not descriptor creation. These split the
  // loop into the texture creation pipeline (CreateCommittedResource + map
  // emplace + trace) and the pending-load queueing (dedup scan +
  // PrepareTextureLoad), leaving fetch parsing + map hit as the remainder.
  kGpuTexMkResUs,
  kGpuTexMkpUs,

  // TEMP DIAGNOSTIC (retail DOA4 residency attribution, fourth level): R40
  // showed gpu_resid_us = 31.25 s (0.84 us/draw) while ALL up_copy memcpy
  // together is only 19.9 s -- at least 11 s inside the residency section is
  // not memcpy. These attribute the vfetch and memexport RequestRange calls
  // (so residency overhead = gpu_resid - vfetch - memexport), the index
  // RequestRange call (the per-draw upstream-TODO path), and the uploaded
  // byte count (to check invalidation-window amplification against actual
  // guest traffic). Accumulators, zeroed each frame.
  kGpuUpVfetchUs,
  kGpuUpMemexportUs,
  kGpuUpIndexUs,
  kGpuUpBytes,

  // TEMP DIAGNOSTIC (retail DOA4 residency attribution, fifth level): R41
  // closed the gpu_resid_us equation (30.0 s = 28.2 s vfetch RequestRange +
  // 1.9 s loop body, memexport never used by DOA4) and showed caller time
  // 42.9 s = 18.4 s memcpy (374 GB uploaded at ~20 GB/s) + ~24.5 s fixed
  // per-call machinery. These count vfetch/index requests and requested bytes
  // (traffic split and requested-vs-uploaded amplification), and split the
  // fixed machinery into prep (validation/sparse alloc) vs scan (lock +
  // bitmap). Accumulators, zeroed each frame.
  kGpuUpVfetchCalls,
  kGpuUpVfetchBytes,
  kGpuUpIndexCalls,
  kGpuUpIndexBytes,
  kGpuUpPrepUs,
  kGpuUpScanUs,

  // TEMP DIAGNOSTIC (retail DOA4 texture first-use, sixth level): R57 showed
  // gpu_tex_mkres_us = 10.7 ms over 340 creates (31 us each) inside the
  // 23.9 ms stage-load frame - 76% of that frame's whole texture path - but
  // the window hides whether the cost is ours or the driver's. These split
  // it: alloc = GetResourceAllocationInfo (queried once when placing and
  // again in the D3D12Texture constructor), drv = CreatePlacedResource /
  // CreateCommittedResource / segment CreateHeap, scan = free-interval
  // first-fit pass (placed-path function span minus drv); remainder of
  // mkres = format/desc fill, constructor tail, map emplace. Accumulators,
  // zeroed each frame.
  kGpuTexMkAllocUs,
  kGpuTexMkDrvUs,
  kGpuTexMkScanUs,

  // TEMP DIAGNOSTIC (retail DOA4 texture first-use, seventh level): the
  // upstream TODO at FindOrCreateTexture - reuse a texture with mip_page
  // unchanged whose base_page was previously 0 (streaming order: mip data
  // lands in guest memory before the base level). Counts create-misses that
  // HAVE such a predecessor (one extra hash probe of the key with
  // base_page zeroed, no behavior change): if frequent, streaming re-keying
  // inflates the create count at every scene load and reuse would cut the
  // mkres burst directly. Accumulator, zeroed each frame.
  kGpuTexReuseCand,

  kCount  // sentinel -- must be last
};

inline constexpr std::size_t kNumCounters = static_cast<std::size_t>(CounterId::kCount);

// TEMP DIAGNOSTIC (retail DOA4 perf work): the probe path touches these
// counters up to ~150k times per frame (scope adds + packet/register counts),
// and the previous out-of-line calls into rexruntime cost ~30ns each -- a
// measurable fraction of every bucket. The array is exported as data and the
// accessors below are inline so the hot path pays only the atomic itself.
// counter.cpp defines REX_PERF_BUILD before including this header so the
// definition side sees dllexport instead of dllimport.
#ifdef REX_PERF_BUILD
#define REX_PERF_DATA __declspec(dllexport)
#else
#define REX_PERF_DATA __declspec(dllimport)
#endif
extern REX_PERF_DATA std::array<std::atomic<int64_t>, kNumCounters> g_counters;

// Returns human-readable name for a counter (e.g. "frame_time_us")
const char* CounterName(CounterId id);

// Set a counter to an absolute value (inline: hot probe path).
inline void SetCounter(CounterId id, int64_t value) {
  g_counters[static_cast<std::size_t>(id)].store(value, std::memory_order_relaxed);
}

// Atomically add to a counter (inline: hot probe path).
inline void IncrementCounter(CounterId id, int64_t delta = 1) {
  g_counters[static_cast<std::size_t>(id)].fetch_add(delta, std::memory_order_relaxed);
}

// Read a counter's current live value (inline: hot probe path).
inline int64_t GetCounter(CounterId id) {
  return g_counters[static_cast<std::size_t>(id)].load(std::memory_order_relaxed);
}

// Snapshot current values into the read buffer and zero the live counters.
// Called once per frame by Profiler::Flip().
void ResetFrameCounters();

// Read a counter from the last-frame snapshot (stable between frames).
int64_t GetSnapshotCounter(CounterId id);

// Initialize the counter system (zeroes everything). Safe to call multiple times.
void Init();

// CSV logging
void SetCsvLogPath(const std::string& path);
void WriteCsvFrame();
void FlushCsv();

// State attribution (TEMP DIAGNOSTIC, retail DOA4 state/stage work): the
// perf_state_addr cvar names a guest u32 that is read once per frame (on
// swap) as the current state id. The per-frame CSV gets a state_id column
// so frame time / fps can be attributed per menu, stage, ... and state
// changes are logged. SetStateLabelFn registers an optional game-provided
// formatter (e.g. `nonmenu`, `stage:SBE (char0=...)`) used by the change
// log, the debug overlay and the exit summary. All no-ops while
// perf_state_addr is 0 (the default).
void SetStateLabelFn(const char* (*fn)(int64_t value));
// State tracker v2 (state/stage work): optional game-provided state poll.
// When set, PollState calls fn(guest_base) once per frame INSTEAD of
// reading perf_state_addr: the game folds several guest bytes (game state,
// stage ids, char ids, screen code, ...) into one int64 so any component
// change is observed as a state transition. guest_base is the kernel
// virtual memory base (guest VA = base + address), non-null. fn must never
// return INT64_MIN for a real state: that value is the "unset" sentinel.
void SetStatePollFn(int64_t (*fn)(const uint8_t* guest_base));
const char* StateLabel(int64_t value);
int64_t GetStateId();
void PollState();
void LogStateSummary();

// Profiler -- coordinates Tracy frame marks and counter snapshots.
// Moved here from rex::debug to consolidate all perf code under rex::perf.
class Profiler {
 public:
  static void Startup() {
#ifdef REXGLUE_ENABLE_PROFILING
    if (!tracy::IsProfilerStarted())
      tracy::StartupProfiler();
#endif
  }
  static void OnThreadEnter(const char* name = nullptr) {
#ifdef REXGLUE_ENABLE_PROFILING
    if (name && tracy::IsProfilerStarted())
      tracy::SetThreadName(name);
#else
    (void)name;
#endif
  }
  static void OnThreadExit() {}
  static void ThreadEnter(const char* name = nullptr) { OnThreadEnter(name); }
  static void ThreadExit() {}
  static void Flip() {
#ifdef REXGLUE_ENABLE_PROFILING
    if (tracy::IsProfilerStarted()) {
      FrameMark;
    }
#endif
#ifdef REXGLUE_ENABLE_PERF_COUNTERS
    ResetFrameCounters();
    PollState();  // state poll + per-state aggregate (state/stage work)
    WriteCsvFrame();
#endif
  }
  static void Flush() {}
  static void Shutdown() {
#ifdef REXGLUE_ENABLE_PROFILING
    if (tracy::IsProfilerStarted())
      tracy::ShutdownProfiler();
#endif
#ifdef REXGLUE_ENABLE_PERF_COUNTERS
    FlushCsv();
#endif
  }
  static bool is_enabled() {
#ifdef REXGLUE_ENABLE_PROFILING
    return tracy::IsProfilerStarted();
#else
    return false;
#endif
  }
};

// TEMP DIAGNOSTIC (retail DOA4 perf work): timestamp for manual split points
// where RAII braces can't be used (a value declared in one region is used in
// another). Pair with PERF_counter_add (a no-op when counters are disabled).
// Remove after a confirmed pass.
inline int64_t PerfNowUs() {
#ifdef REXGLUE_ENABLE_PERF_COUNTERS
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
#else
  return 0;  // every reading feeds a no-op PERF_counter_add (lean: was ~5% of the GPU thread)
#endif
}

#ifdef REXGLUE_ENABLE_PERF_COUNTERS
// TEMP DIAGNOSTIC (retail DOA4 perf work): accumulates microseconds spent in
// a scope into a perf counter (csv column). RAII, so early returns and
// `continue` are counted too. Nested scopes with the SAME counter id
// accumulate EXCLUSIVE time only (the inner scope's inclusive time is added
// to the parent's child_us and subtracted on exit) -- this keeps counters
// like gpu_type3_us correct when indirect buffers re-enter the packet
// dispatcher. Different ids are independent (gpu_draw_us nests inside
// gpu_type3_us and is read as its own column). Remove after a confirmed pass.
struct ScopedUsCounter {
  static inline thread_local ScopedUsCounter* s_active = nullptr;

  CounterId id;
  std::chrono::steady_clock::time_point t0;
  ScopedUsCounter* prev_active;
  ScopedUsCounter* parent_same_id;
  int64_t child_us = 0;

  explicit ScopedUsCounter(CounterId id_in)
      : id(id_in),
        t0(std::chrono::steady_clock::now()),
        prev_active(s_active),
        parent_same_id(FindSameId(s_active, id_in)) {
    s_active = this;
  }
  ~ScopedUsCounter() {
    auto us = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0)
            .count());
    if (parent_same_id) {
      parent_same_id->child_us += us;
    }
    s_active = prev_active;
    // TEMP DIAGNOSTIC (retail DOA4 perf work): skip the zero-delta add -- most
    // per-packet exclusive deltas truncate to 0us, and paying an atomic for
    // them was pure overhead. Semantically identical (adding 0).
    const auto delta = us - child_us;
    if (delta) {
      IncrementCounter(id, delta);
    }
  }
  static ScopedUsCounter* FindSameId(ScopedUsCounter* top, CounterId id_in) {
    while (top) {
      if (top->id == id_in) {
        return top;
      }
      top = top->prev_active;
    }
    return nullptr;
  }
};
#else
// No-op variant so call sites compile unchanged when counters are disabled.
struct ScopedUsCounter {
  explicit ScopedUsCounter(CounterId) {}
};
#endif

}  // namespace rex::perf

// TEMP DIAGNOSTIC (retail DOA4 perf work): the perf_log_csv cvar existed but
// was never declared/consumed anywhere, so per-frame CSV logging was dead
// code. Global-scope declaration (must match the REXCVAR_DEFINE at global
// scope in counter.cpp); consumed at the first swap in command_processor.
REXCVAR_DECLARE(std::string, perf_log_csv);

// Perf counter macros -- compile to no-ops when counters are disabled.
#ifdef REXGLUE_ENABLE_PERF_COUNTERS

// Generic helpers for easily adding new counters
#define PERF_counter_set(id, value) rex::perf::SetCounter(rex::perf::CounterId::id, value)
#define PERF_counter_inc(id) rex::perf::IncrementCounter(rex::perf::CounterId::id)
#define PERF_counter_add(id, delta) rex::perf::IncrementCounter(rex::perf::CounterId::id, delta)

// Purpose-specific macros so callsites stay clean
#define PROFILE_FRAME_TIME_US(value) PERF_counter_set(kFrameTimeUs, value)
#define PROFILE_FPS(value) PERF_counter_set(kFps, value)
#define PROFILE_FUNCTION_DISPATCHED() PERF_counter_inc(kFunctionsDispatched)
#define PROFILE_INTERRUPT_DISPATCHED() PERF_counter_inc(kInterruptDispatches)
#define PROFILE_XMA_FRAME_DECODED() PERF_counter_inc(kXmaFramesDecoded)
#define PROFILE_DRAW_CALL() PERF_counter_inc(kDrawCalls)
#define PROFILE_VERTICES(n) PERF_counter_add(kVerticesProcessed, n)
#define PROFILE_CMD_BUFFER_STALL() PERF_counter_inc(kCommandBufferStalls)
#define PROFILE_AUDIO_LATENCY_US(value) PERF_counter_set(kAudioFrameLatencyUs, value)
#define PROFILE_BUFFER_QUEUE_DEPTH(value) PERF_counter_set(kBufferQueueDepth, value)
#define PROFILE_THREAD_CREATED() PERF_counter_inc(kActiveThreads)
#define PROFILE_THREAD_EXITED() PERF_counter_add(kActiveThreads, -1)
#define PROFILE_APC_QUEUE_DEPTH(value) PERF_counter_set(kApcQueueDepth, value)
#define PROFILE_CRITICAL_REGION_CONTENTION() PERF_counter_inc(kCriticalRegionContentions)
#define PROFILE_TEXTURE_CACHE_HIT() PERF_counter_inc(kTextureCacheHits)
#define PROFILE_TEXTURE_CACHE_MISS() PERF_counter_inc(kTextureCacheMisses)
#define PROFILE_PIPELINE_CACHE_HIT() PERF_counter_inc(kPipelineCacheHits)
#define PROFILE_PIPELINE_CACHE_MISS() PERF_counter_inc(kPipelineCacheMisses)

#else

#define PERF_counter_set(id, value)
#define PERF_counter_inc(id)
#define PERF_counter_add(id, delta)

#define PROFILE_FRAME_TIME_US(value)
#define PROFILE_FPS(value)
#define PROFILE_FUNCTION_DISPATCHED()
#define PROFILE_INTERRUPT_DISPATCHED()
#define PROFILE_XMA_FRAME_DECODED()
#define PROFILE_DRAW_CALL()
#define PROFILE_VERTICES(n)
#define PROFILE_CMD_BUFFER_STALL()
#define PROFILE_AUDIO_LATENCY_US(value)
#define PROFILE_BUFFER_QUEUE_DEPTH(value)
#define PROFILE_THREAD_CREATED()
#define PROFILE_THREAD_EXITED()
#define PROFILE_APC_QUEUE_DEPTH(value)
#define PROFILE_CRITICAL_REGION_CONTENTION()
#define PROFILE_TEXTURE_CACHE_HIT()
#define PROFILE_TEXTURE_CACHE_MISS()
#define PROFILE_PIPELINE_CACHE_HIT()
#define PROFILE_PIPELINE_CACHE_MISS()

#endif
