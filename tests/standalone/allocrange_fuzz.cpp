// Standalone differential test for BaseHeap::AllocRange's free-range search.
//
// WHY THIS EXISTS
//
// The 2026-10-03 UFC3 crash log contains a self-contradicting allocation
// failure from the guest physical heap (512 MB, 4096-byte pages, 131072 pages):
//
//   BaseHeap::Alloc failed to find contiguous range
//   pedidas 8 paginas (0.03 MB); livres 28864 de 131072 (112.8 MB),
//   maior faixa contigua 23079 paginas (90.15 MB) -- deveria caber
//
// An 8-page request refused while a 90 MB contiguous run was free. Either the
// search misses ranges that exist, or the free/largest-contiguous figures come
// from a different predicate than the `state == 0` test the search uses.
//
// This harness answers that question without building the game. The search is
// transcribed verbatim from src/system/xmemory.cpp (BaseHeap::AllocRange) and
// compared against a brute-force reference that simply tries every aligned base
// in the window. Any disagreement is a bug, by definition.
//
// Build (MSVC, no LLVM needed):
//   cl /std:c++20 /EHsc /O2 /W3 allocrange_fuzz.cpp && allocrange_fuzz.exe
//
// Not part of the CMake build on purpose: it has no dependencies so it can be
// compiled and run in seconds while iterating on the allocator.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// --- verbatim from include/rex/math.h -------------------------------------------------

template <typename T>
constexpr T align_up(T value, T alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T, typename V>
constexpr T round_up(T value, V multiple, bool force_non_zero = true) {
  if (force_non_zero && !value) {
    return multiple;
  }
  return (value + multiple - 1) / multiple * multiple;
}

// --- verbatim from src/system/xmemory.cpp ----------------------------------------------

constexpr uint32_t get_page_count(uint32_t value, uint32_t page_size, uint32_t page_size_shift) {
  return round_up(value, page_size) >> page_size_shift;
}

constexpr uint32_t kMemoryAllocationReserve = 0x1;
constexpr uint32_t kMemoryAllocationCommit = 0x2;

struct PageEntry {
  uint32_t base_address = 0;
  uint32_t region_page_count = 0;
  uint32_t allocation_protect = 0;
  uint32_t current_protect = 0;
  uint32_t state = 0;
  uint64_t qword() const { return 0; }
};

struct Heap {
  uint32_t heap_base = 0;
  uint32_t heap_size = 0;
  uint32_t page_size = 0;
  uint32_t page_size_shift = 0;
  std::vector<PageEntry> page_table;
  uint32_t unreserved_page_count = 0;

  uint32_t total_page_count() const { return uint32_t(page_table.size()); }
};

// Body of BaseHeap::AllocRange, transcribed from xmemory.cpp:1388-1504 with the
// host allocation, page-state commit and SyncHostPageAccess tail removed: this
// harness is only interested in whether the SEARCH finds a range.
bool SearchRange(const Heap& heap, uint32_t low_address, uint32_t high_address, uint32_t size,
                 uint32_t alignment, bool top_down, uint32_t* out_start, uint32_t* out_end) {
  const uint32_t heap_base = heap.heap_base;
  const uint32_t heap_size = heap.heap_size;
  const uint32_t page_size = heap.page_size;
  const uint32_t page_size_shift = heap.page_size_shift;

  alignment = round_up(alignment, page_size);
  uint32_t page_count = get_page_count(size, page_size, page_size_shift);
  low_address = std::max(heap_base, align_up(low_address, alignment));
  high_address = std::min(heap_base + (heap_size - 1), align_up(high_address, alignment));
  if (high_address < low_address) {
    return false;
  }
  uint32_t low_page_number = (low_address - heap_base) >> page_size_shift;
  uint32_t high_page_number = (high_address - heap_base) >> page_size_shift;
  low_page_number = std::min(uint32_t(heap.page_table.size()) - 1, low_page_number);
  high_page_number = std::min(uint32_t(heap.page_table.size()) - 1, high_page_number);
  if (high_page_number < low_page_number) {
    return false;
  }

  uint32_t available_page_count = high_page_number - low_page_number + 1;
  if (!page_count || page_count > available_page_count) {
    return false;
  }

  uint32_t start_page_number = UINT32_MAX;
  uint32_t end_page_number = UINT32_MAX;
  uint32_t page_scan_stride = alignment >> page_size_shift;
  uint32_t max_base_page_number = high_page_number + 1 - page_count;
  if (top_down) {
    max_base_page_number -= max_base_page_number % page_scan_stride;
    for (int64_t base_page_number = max_base_page_number; base_page_number >= low_page_number;
         base_page_number -= page_scan_stride) {
      if (heap.page_table[base_page_number].state != 0) {
        continue;
      }
      start_page_number = uint32_t(base_page_number);
      end_page_number = uint32_t(base_page_number) + page_count - 1;
      bool any_taken = false;
      for (uint32_t page_number = uint32_t(base_page_number); !any_taken && page_number <= end_page_number;
           ++page_number) {
        bool is_free = heap.page_table[page_number].state == 0;
        if (!is_free) {
          any_taken = true;
          if (page_count > page_number) {
            base_page_number = -1;
          } else {
            base_page_number = page_number - page_count;
            base_page_number -= base_page_number % page_scan_stride;
            base_page_number += page_scan_stride;
          }
          break;
        }
      }
      if (!any_taken) {
        break;
      }
      start_page_number = end_page_number = UINT32_MAX;
    }
  } else {
    for (uint32_t base_page_number = low_page_number; base_page_number <= max_base_page_number;
         base_page_number += page_scan_stride) {
      if (heap.page_table[base_page_number].state != 0) {
        continue;
      }
      start_page_number = base_page_number;
      end_page_number = base_page_number + page_count - 1;
      bool any_taken = false;
      for (uint32_t page_number = base_page_number; !any_taken && page_number <= end_page_number;
           ++page_number) {
        bool is_free = heap.page_table[page_number].state == 0;
        if (!is_free) {
          any_taken = true;
          base_page_number = round_up(page_number + 1, page_scan_stride);
          base_page_number -= page_scan_stride;
          break;
        }
      }
      if (!any_taken) {
        break;
      }
      start_page_number = end_page_number = UINT32_MAX;
    }
  }

  if (start_page_number == UINT32_MAX || end_page_number == UINT32_MAX) {
    return false;
  }
  *out_start = start_page_number;
  *out_end = end_page_number;
  return true;
}

// Reference: try every aligned base page in the window. Deliberately dumb.
bool BruteForceRange(const Heap& heap, uint32_t low_address, uint32_t high_address, uint32_t size,
                     uint32_t alignment, bool top_down, uint32_t* out_start) {
  const uint32_t heap_base = heap.heap_base;
  const uint32_t page_size = heap.page_size;
  const uint32_t page_size_shift = heap.page_size_shift;

  alignment = round_up(alignment, page_size);
  uint32_t page_count = get_page_count(size, page_size, page_size_shift);
  low_address = std::max(heap_base, align_up(low_address, alignment));
  high_address = std::min(heap_base + (heap.heap_size - 1), align_up(high_address, alignment));
  if (high_address < low_address) {
    return false;
  }
  uint32_t low_page_number = (low_address - heap_base) >> page_size_shift;
  uint32_t high_page_number = (high_address - heap_base) >> page_size_shift;
  low_page_number = std::min(uint32_t(heap.page_table.size()) - 1, low_page_number);
  high_page_number = std::min(uint32_t(heap.page_table.size()) - 1, high_page_number);
  if (high_page_number < low_page_number) {
    return false;
  }
  if (!page_count || page_count > (high_page_number - low_page_number + 1)) {
    return false;
  }
  const uint32_t stride = alignment >> page_size_shift;
  const uint32_t max_base = high_page_number + 1 - page_count;

  auto fits = [&](uint32_t base) {
    if (base < low_page_number || base > max_base) {
      return false;
    }
    if (stride && (base % stride) != 0) {
      return false;
    }
    for (uint32_t p = base; p <= base + page_count - 1; ++p) {
      if (heap.page_table[p].state != 0) {
        return false;
      }
    }
    return true;
  };

  if (top_down) {
    for (int64_t base = int64_t(max_base) - int64_t(max_base % stride); base >= int64_t(low_page_number);
         base -= stride) {
      if (fits(uint32_t(base))) {
        *out_start = uint32_t(base);
        return true;
      }
    }
  } else {
    for (uint32_t base = low_page_number; base <= max_base; base += stride) {
      if (fits(base)) {
        *out_start = base;
        return true;
      }
    }
  }
  return false;
}

struct Mismatch {
  uint32_t pages;
  uint32_t page_size;
  uint32_t alignment;
  uint32_t size;
  bool top_down;
  bool search_found;
  bool brute_found;
};

}  // namespace

int main() {
  std::mt19937 rng(0xC0FFEEu);

  // The real heap from the log: 512 MB physical pool, 4 KB pages.
  const std::vector<std::pair<uint32_t, uint32_t>> heap_shapes = {
      {0x20000000u, 4096u},   // heaps_.physical  -- the one in the crash log
      {0x20000000u, 65536u},  // heaps_.vA0000000
      {0x1FD00000u, 4096u},   // heaps_.vE0000000
  };

  std::vector<Mismatch> mismatches;
  uint64_t cases = 0;

  for (const auto& [heap_size, page_size] : heap_shapes) {
    const uint32_t page_size_shift = page_size == 4096u ? 12u : 16u;
    Heap heap;
    heap.heap_base = 0;
    heap.heap_size = heap_size;
    heap.page_size = page_size;
    heap.page_size_shift = page_size_shift;
    const uint32_t pages = heap_size >> page_size_shift;
    heap.page_table.assign(pages, PageEntry{});

    for (int iteration = 0; iteration < 4000; ++iteration) {
      // Random occupancy between 50% and 100% used, which is where the log lives.
      const int pct = 50 + (rng() % 51);
      for (uint32_t p = 0; p < pages; ++p) {
        heap.page_table[p].state =
            (int(rng() % 100) < pct) ? (kMemoryAllocationReserve | kMemoryAllocationCommit) : 0;
      }

      // Alignments the guest can ask for, plus the large ones the physical
      // sub-heaps synthesise when they round up to their own page size.
      const std::vector<uint32_t> alignments = {0x1000u, 0x1000u, 0x10000u, 0x100000u,
                                                0x1000000u, 0x10000000u};
      const std::vector<uint32_t> sizes = {0x1000u, 0x2000u, 0x8000u, 0x100000u};

      for (uint32_t alignment : alignments) {
        for (uint32_t size : sizes) {
          for (bool top_down : {true, false}) {
            ++cases;
            uint32_t s = 0, e = 0, b = 0;
            const bool search_ok = SearchRange(heap, 0, 0xFFFFFFFFu, size, alignment, top_down, &s, &e);
            const bool brute_ok = BruteForceRange(heap, 0, 0xFFFFFFFFu, size, alignment, top_down, &b);
            if (search_ok != brute_ok) {
              if (mismatches.size() < 24) {
                mismatches.push_back({pages, page_size, alignment, size, top_down, search_ok, brute_ok});
              }
            }
          }
        }
      }
    }
  }

  std::printf("cases: %llu\n", (unsigned long long)cases);
  std::printf("mismatches: %zu\n", mismatches.size());
  for (const Mismatch& m : mismatches) {
    std::printf("  pages=%u page_size=%u align=0x%X size=0x%X top_down=%d -> search=%d brute=%d\n",
                m.pages, m.page_size, m.alignment, m.size, (int)m.top_down, (int)m.search_found,
                (int)m.brute_found);
  }
  return mismatches.empty() ? 0 : 1;
}