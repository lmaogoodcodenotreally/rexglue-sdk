/**
 * @file        core/perf/counter.cpp
 * @brief       Performance counter registry implementation
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
// TEMP DIAGNOSTIC (retail DOA4 perf work): mark ourselves as the definition
// side of the exported counter array (see the note in <rex/perf/counter.h>).
// Must be defined before the header include.
#define REX_PERF_BUILD 1
#include <rex/perf/counter.h>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

#include <rex/system/kernel_state.h>

REXCVAR_DEFINE_STRING(perf_log_csv, "", "Perf",
                      "Path to write per-frame CSV log (empty = disabled)");

// TEMP DIAGNOSTIC (retail DOA4 state/stage work): guest u32 address polled
// once per frame (on swap) as the current state id -- e.g. the MSEL screen
// code global (0x82C31C24: >= 0 = menu screen, -1 = outside the menu
// system). 0 = disabled (no poll, no state_id CSV column content). The
// game-specific address is set by the project in OnPostSetup so the SDK
// stays game-agnostic.
REXCVAR_DEFINE_UINT32(perf_state_addr, 0, "Perf",
                      "Guest u32 address polled once per frame as state id (0 = disabled)");


namespace rex::perf {

// Exported data (dllexport comes from the REX_PERF_BUILD declaration in
// <rex/perf/counter.h>).  Definition lives outside the anonymous namespace so
// the linker can export it to the GPU plugin / recomp image; see the TEMP
// DIAGNOSTIC note in the header for why it is accessed inline from hot paths.
std::array<std::atomic<int64_t>, kNumCounters> g_counters{};

// TEMP DIAGNOSTIC (retail DOA4 counter diet, see the note in
// <rex/perf/counter.h>): the diet turned IncrementCounter into a header-inline
// accessor over the exported g_counters array above and dropped the out-of-line
// function export. Exes linked against the pre-diet import library import it
// by name and now fail at process start with "Entry Point Not Found" -- the
// E3 dev-demo exe (built 2026-10-06) does; simpsons.exe (paired with the
// pre-diet DLL copy of 2026-10-02) would after any future DLL sync.
// Taking the address below forces an out-of-line copy of the inline accessor
// to be emitted in this translation unit, and the pragma re-exports that copy
// under the exact pre-diet mangled name (taken verbatim from the E3 exe's
// import table). Freshly linked exes use the inline accessor and never touch
// the export; if the emission or the name ever stops resolving, the DLL link
// fails loudly instead of the game failing at launch. Remove together with
// the counter diet (or once every shipped exe is relinked against the current
// import library).
__attribute__((used)) void (*const g_exportIncrementCounter)(CounterId,
                                                             int64_t) =
    &IncrementCounter;
#pragma comment(linker, "/export:?IncrementCounter@perf@rex@@YAXW4CounterId@12@_J@Z")

namespace {

std::array<std::atomic<int64_t>, kNumCounters> g_snapshot{};

constexpr const char* kCounterNames[] = {
    "frame_time_us",
    "fps",
    "draw_calls",
    "command_buffer_stalls",
    "vertices_processed",
    "xma_frames_decoded",
    "audio_frame_latency_us",
    "buffer_queue_depth",
    "functions_dispatched",
    "interrupt_dispatches",
    "active_threads",
    "apc_queue_depth",
    "critical_region_contentions",
    "texture_cache_hits",
    "texture_cache_misses",
    "pipeline_cache_hits",
    "pipeline_cache_misses",
    "gpu_exec_us",
    "gpu_stall_us",
    "gpu_draw_us",
    "gpu_present_us",
    "gpu_submit_us",
    "gpu_wait_us",
    "gpu_packets",
    "gpu_reg_us",
    "gpu_const_us",
    "gpu_imm_us",
    "gpu_type3_us",
    "gpu_prim_us",
    "gpu_rt_us",
    "gpu_pipe_us",
    "gpu_state_us",
    "gpu_bind_us",
    "gpu_record_us",
    "gpu_entry_us",
    "gpu_mods_us",
    "gpu_xlat_us",
    "gpu_tex_us",
    "gpu_resid_us",
    "gpu_evt_us",
    "gpu_misc_us",
    "gpu_copy_us",
    "gpu_regw_count",
    "gpu_nop_count",
    "gpu_ib_count",
    "gpu_imm_cache_hits",
    "gpu_tex_prep_us",
    "gpu_tex_load_us",
    "gpu_tex_bind_us",
    "gpu_tex_load_count",
    "gpu_tex_find_us",
    "gpu_tex_range_us",
    "gpu_tex_up_pool_us",
    "gpu_tex_up_copy_us",
    "gpu_tex_mkres_us",
    "gpu_tex_mkp_us",
    "gpu_up_vfetch_us",
    "gpu_up_memexport_us",
    "gpu_up_index_us",
    "gpu_up_bytes",
    "gpu_up_vfetch_calls",
    "gpu_up_vfetch_bytes",
    "gpu_up_index_calls",
    "gpu_up_index_bytes",
    "gpu_up_prep_us",
    "gpu_up_scan_us",
    "gpu_tex_mk_alloc_us",
    "gpu_tex_mk_drv_us",
    "gpu_tex_mk_scan_us",
    "gpu_tex_reuse_cand",
};
static_assert(std::size(kCounterNames) == kNumCounters, "kCounterNames must match CounterId enum");

// Gauge counters are snapshotted but NOT zeroed each frame.
// Accumulators (everything else) are zeroed after snapshot.
constexpr bool kIsGauge[] = {
    false,  // kFrameTimeUs       (set each frame)
    false,  // kFps               (set each frame)
    false,  // kDrawCalls
    false,  // kCommandBufferStalls
    false,  // kVerticesProcessed
    false,  // kXmaFramesDecoded
    false,  // kAudioFrameLatencyUs
    false,  // kBufferQueueDepth  (set each frame)
    false,  // kFunctionsDispatched
    false,  // kInterruptDispatches
    true,   // kActiveThreads     (inc/dec over lifetime)
    false,  // kApcQueueDepth
    true,   // kCriticalRegionContentions (running total)
    false,  // kTextureCacheHits
    false,  // kTextureCacheMisses
    false,  // kPipelineCacheHits
    false,  // kPipelineCacheMisses
    false,  // kGpuExecUs
    false,  // kGpuStallUs
    false,  // kGpuDrawUs
    false,  // kGpuPresentUs
    false,  // kGpuSubmitUs
    false,  // kGpuWaitUs
    false,  // kGpuPackets
    false,  // kGpuRegUs
    false,  // kGpuConstUs
    false,  // kGpuImmUs
    false,  // kGpuType3Us
    false,  // kGpuPrimUs
    false,  // kGpuRtUs
    false,  // kGpuPipeUs
    false,  // kGpuStateUs
    false,  // kGpuBindUs
    false,  // kGpuRecordUs
    false,  // kGpuEntryUs
    false,  // kGpuModsUs
    false,  // kGpuXlatUs
    false,  // kGpuTexUs
    false,  // kGpuResidUs
    false,  // kGpuEvtUs
    false,  // kGpuMiscUs
    false,  // kGpuCopyUs
    false,  // kGpuRegwCount
    false,  // kGpuNopCount
    false,  // kGpuIbCount
    false,  // kGpuImmHitCount
    false,  // kGpuTexPrepUs
    false,  // kGpuTexLoadUs
    false,  // kGpuTexBindUs
    false,  // kGpuTexLoadCount
    false,  // kGpuTexFindUs
    false,  // kGpuTexRangeUs
    false,  // kGpuTexUpPoolUs
    false,  // kGpuTexUpCopyUs
    false,  // kGpuTexMkResUs
    false,  // kGpuTexMkpUs
    false,  // kGpuUpVfetchUs
    false,  // kGpuUpMemexportUs
    false,  // kGpuUpIndexUs
    false,  // kGpuUpBytes
    false,  // kGpuUpVfetchCalls
    false,  // kGpuUpVfetchBytes
    false,  // kGpuUpIndexCalls
    false,  // kGpuUpIndexBytes
    false,  // kGpuUpPrepUs
    false,  // kGpuUpScanUs
    false,  // kGpuTexMkAllocUs
    false,  // kGpuTexMkDrvUs
    false,  // kGpuTexMkScanUs
    false,  // kGpuTexReuseCand
};
static_assert(std::size(kIsGauge) == kNumCounters, "kIsGauge must match CounterId enum");

// CSV state
std::FILE* g_csv_file = nullptr;
std::string g_csv_path;
int g_csv_frame_count = 0;

// State attribution (state/stage work, see perf_state_addr above).
constexpr int64_t kStateUnset = INT64_MIN;
std::atomic<int64_t> g_state_id{kStateUnset};
std::atomic<const char* (*)(int64_t)> g_state_label_fn{nullptr};
std::atomic<int64_t (*)(const uint8_t*)> g_state_poll_fn{nullptr};

struct StateAgg {
  int64_t id;
  int64_t frames;
  int64_t sum_us;
  int64_t min_us;
  int64_t max_us;
};
// States are few (menu / stage / ...) so a linear scan beats hashing.
std::vector<StateAgg> g_state_aggs;

}  // anonymous namespace

const char* CounterName(CounterId id) {
  auto idx = static_cast<size_t>(id);
  if (idx < kNumCounters)
    return kCounterNames[idx];
  return "unknown";
}

// SetCounter / IncrementCounter / GetCounter are inline in
// <rex/perf/counter.h> (TEMP DIAGNOSTIC, retail DOA4 perf work): the hot probe
// path must not pay a cross-DLL call per increment.

void ResetFrameCounters() {
  for (size_t i = 0; i < kNumCounters; ++i) {
    if (kIsGauge[i]) {
      // Gauges: snapshot the current value, don't zero
      g_snapshot[i].store(g_counters[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    } else {
      // Accumulators: snapshot and zero for next frame
      g_snapshot[i].store(g_counters[i].exchange(0, std::memory_order_relaxed),
                          std::memory_order_relaxed);
    }
  }
}

int64_t GetSnapshotCounter(CounterId id) {
  return g_snapshot[static_cast<size_t>(id)].load(std::memory_order_relaxed);
}

void Init() {
  for (auto& c : g_counters)
    c.store(0, std::memory_order_relaxed);
  for (auto& s : g_snapshot)
    s.store(0, std::memory_order_relaxed);
}

void SetStateLabelFn(const char* (*fn)(int64_t value)) {
  g_state_label_fn.store(fn, std::memory_order_relaxed);
}

void SetStatePollFn(int64_t (*fn)(const uint8_t*)) {
  g_state_poll_fn.store(fn, std::memory_order_relaxed);
}

const char* StateLabel(int64_t value) {
  if (auto fn = g_state_label_fn.load(std::memory_order_relaxed))
    return fn(value);
  static thread_local char buf[32];
  std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(value));
  return buf;
}

int64_t GetStateId() {
  return g_state_id.load(std::memory_order_relaxed);
}

void PollState() {
  int64_t state = g_state_id.load(std::memory_order_relaxed);

  // Cheap per-frame state poll: one guest load + compare; logging only on
  // change. Two sources, in order of precedence:
  //  - v2: a game-registered poll fn (SetStatePollFn) that packs several
  //    guest bytes into one int64 (so a change in ANY folded field is a
  //    transition),
  //  - v1: the perf_state_addr u32 (kept as the fallback when no fn is
  //    registered).
  // The guest memory is written by the game itself (no recomp hooks in the
  // hot path), so this stays valid across scene changes.
  bool have_value = false;
  int64_t value = kStateUnset;
  if (auto* ks = rex::system::kernel_state()) {
    if (auto* memory = ks->memory()) {
      if (const uint8_t* base = memory->virtual_membase()) {
        if (auto fn = g_state_poll_fn.load(std::memory_order_relaxed)) {
          value = fn(base);
          have_value = true;
        } else if (const uint32_t addr = REXCVAR_GET(perf_state_addr);
                   addr != 0) {
          uint32_t raw = 0;
          std::memcpy(&raw, base + addr, sizeof(raw));
          // Guest memory is big-endian (REX_LOAD_U32 does __builtin_bswap32);
          // swap here so state ids match the guest's own loads (e.g. -1 stays
          // -1, screen code 5 reads 5 instead of 0x05000000).
          const uint32_t be = (raw >> 24) | ((raw >> 8) & 0x0000FF00u) |
                              ((raw << 8) & 0x00FF0000u) | (raw << 24);
          value = static_cast<int32_t>(be);
          have_value = true;
        }
      }
    }
  }
  if (have_value && value != state) {
    if (state == kStateUnset)
      REXLOG_INFO("[state] init {}", StateLabel(value));
    else
      REXLOG_INFO("[state] {} -> {}", StateLabel(state), StateLabel(value));
    state = value;
    g_state_id.store(value, std::memory_order_relaxed);
  }
  if (state == kStateUnset)
    return;

  // Per-state frame-time aggregate for the exit summary (written from the
  // swap thread right after ResetFrameCounters snapshotted this frame).
  const int64_t ft_us =
      g_snapshot[static_cast<size_t>(CounterId::kFrameTimeUs)].load(std::memory_order_relaxed);
  if (ft_us <= 0)
    return;
  for (auto& agg : g_state_aggs) {
    if (agg.id == state) {
      agg.frames++;
      agg.sum_us += ft_us;
      if (ft_us < agg.min_us)
        agg.min_us = ft_us;
      if (ft_us > agg.max_us)
        agg.max_us = ft_us;
      return;
    }
  }
  g_state_aggs.push_back({state, 1, ft_us, ft_us, ft_us});
}

void SetCsvLogPath(const std::string& path) {
  if (g_csv_file) {
    std::fflush(g_csv_file);
    std::fclose(g_csv_file);
    g_csv_file = nullptr;
  }
  g_csv_path = path;
  g_csv_frame_count = 0;
  g_state_aggs.clear();

  if (path.empty())
    return;

  g_csv_file = rex::filesystem::OpenFile(rex::to_path(path), "w");
  if (!g_csv_file) {
    REXLOG_WARN("perf: failed to open CSV log: {}", path);
    g_csv_path.clear();
    return;
  }

  // Write header
  for (size_t i = 0; i < kNumCounters; ++i) {
    if (i > 0)
      std::fputc(',', g_csv_file);
    std::fputs(kCounterNames[i], g_csv_file);
  }
  std::fputs(",state_id", g_csv_file);
  std::fputc('\n', g_csv_file);
}

void WriteCsvFrame() {
  if (!g_csv_file)
    return;

  for (size_t i = 0; i < kNumCounters; ++i) {
    if (i > 0)
      std::fputc(',', g_csv_file);
    std::fprintf(g_csv_file, "%lld",
                 static_cast<long long>(g_snapshot[i].load(std::memory_order_relaxed)));
  }
  std::fprintf(g_csv_file, ",%lld",
               static_cast<long long>(g_state_id.load(std::memory_order_relaxed)));
  std::fputc('\n', g_csv_file);

  if (++g_csv_frame_count % 60 == 0) {
    std::fflush(g_csv_file);
  }
}

void FlushCsv() {
  if (g_csv_file) {
    std::fflush(g_csv_file);
    std::fclose(g_csv_file);
    g_csv_file = nullptr;
  }
  g_csv_path.clear();
  LogStateSummary();
}

void LogStateSummary() {
  // Per-state fps/frame-time summary (state/stage work): one line per
  // observed state, labeled through the game's formatter when set. Called
  // from FlushCsv (normal teardown) and from ReXApp::OnClosing, which
  // hard-exits with std::_Exit and would otherwise never print it. Cleared
  // after printing so double-flush paths stay idempotent.
  if (g_state_aggs.empty())
    return;
  int64_t total = 0;
  for (const auto& agg : g_state_aggs)
    total += agg.frames;
  for (const auto& agg : g_state_aggs) {
    REXLOG_INFO(
        "[state] summary: {} frames={} share={:.1f}% avg={:.2f}ms min={:.2f}ms max={:.2f}ms",
        StateLabel(agg.id), agg.frames, 100.0 * static_cast<double>(agg.frames) / total,
        static_cast<double>(agg.sum_us) / 1000.0 / static_cast<double>(agg.frames),
        static_cast<double>(agg.min_us) / 1000.0, static_cast<double>(agg.max_us) / 1000.0);
  }
  g_state_aggs.clear();
}

}  // namespace rex::perf
