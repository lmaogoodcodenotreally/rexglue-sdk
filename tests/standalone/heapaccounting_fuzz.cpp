// Standalone differential test for BaseHeap page-table accounting.
//
// WHY THIS EXISTS
//
// The 2026-10-03 UFC3 crash log contains a self-contradicting allocation
// failure from the guest physical heap (512 MB, 4 KB pages, 131072 pages):
//
//   BaseHeap::Alloc failed to find contiguous range
//   pedidas 8 paginas (0.03 MB); livres 28864 de 131072 (112.8 MB),
//   maior faixa contigua 23079 paginas (90.15 MB) -- deveria caber
//
// allocrange_fuzz.cpp proved the free-range SEARCH is correct (576k cases, no
// mismatches). So the search is not lying: either the heap really has no 8-page
// run, or the "livres" / "maior faixa contigua" figures are derived from
// unreserved_page_count, which can drift away from the page table.
//
// This harness models the four mutators verbatim -- AllocRange, AllocFixed,
// Decommit, Release -- and asserts the invariant after every single operation:
//
//   unreserved_page_count  ==  count(pages with state == 0)
//
// Any violation is the desync: the heap would then advertise free memory that
// the allocator will refuse to hand out.
//
// Build (MSVC, no LLVM needed):
//   cl /std:c++20 /EHsc /O2 /W3 heapaccounting_fuzz.cpp && heapaccounting_fuzz.exe

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

namespace {

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

constexpr uint32_t get_page_count(uint32_t value, uint32_t page_size, uint32_t page_size_shift) {
  return round_up(value, page_size) >> page_size_shift;
}

constexpr uint32_t kMemoryAllocationReserve = 0x1;
constexpr uint32_t kMemoryAllocationCommit = 0x2;

struct PageEntry {
  uint32_t base_address = 0;
  uint32_t region_page_count = 0;
  uint32_t state = 0;
};

struct Heap {
  uint32_t heap_base = 0;
  uint32_t heap_size = 0x20000000u;
  uint32_t page_size = 4096;
  uint32_t page_size_shift = 12;
  std::vector<PageEntry> page_table;
  uint32_t unreserved_page_count = 0;

  uint32_t CountFree() const {
    uint32_t n = 0;
    for (const PageEntry& e : page_table) {
      if (e.state == 0) {
        ++n;
      }
    }
    return n;
  }

  uint32_t LargestFreeRun() const {
    uint32_t best = 0, cur = 0;
    for (const PageEntry& e : page_table) {
      if (e.state == 0) {
        ++cur;
        best = std::max(best, cur);
      } else {
        cur = 0;
      }
    }
    return best;
  }

  // First-fit search. allocrange_fuzz.cpp established this is equivalent to the
  // real BaseHeap::AllocRange scan, and only the mutations matter here anyway.
  bool FindRange(uint32_t page_count, uint32_t alignment, uint32_t* out_start) const {
    const uint32_t stride = std::max(1u, alignment >> page_size_shift);
    if (page_count > page_table.size()) {
      return false;
    }
    for (uint32_t base = 0; base + page_count <= page_table.size(); base += stride) {
      bool ok = true;
      for (uint32_t p = base; p < base + page_count; ++p) {
        if (page_table[p].state != 0) {
          ok = false;
          break;
        }
      }
      if (ok) {
        *out_start = base;
        return true;
      }
    }
    return false;
  }

  // xmemory.cpp:1522-1533
  void CommitRange(uint32_t start, uint32_t page_count, uint32_t allocation_type) {
    for (uint32_t p = start; p <= start + page_count - 1; ++p) {
      PageEntry& e = page_table[p];
      if (!e.state) {
        unreserved_page_count--;
      }
      e.base_address = start;
      e.region_page_count = page_count;
      e.state = kMemoryAllocationReserve | allocation_type;
    }
  }

  // xmemory.cpp:1359-1373
  void CommitFixed(uint32_t start, uint32_t page_count, uint32_t allocation_type) {
    for (uint32_t p = start; p <= start + page_count - 1; ++p) {
      PageEntry& e = page_table[p];
      if (allocation_type & kMemoryAllocationReserve) {
        if (!e.state) {
          unreserved_page_count--;
        }
        e.base_address = start;
        e.region_page_count = page_count;
      }
      e.state = kMemoryAllocationReserve | allocation_type;
    }
  }

  // xmemory.cpp:1573-1576
  void Decommit(uint32_t address, uint32_t size) {
    uint32_t page_count = get_page_count(size, page_size, page_size_shift);
    uint32_t start = (address - heap_base) >> page_size_shift;
    start = std::min(uint32_t(page_table.size()) - 1, start);
    uint32_t end = std::min(uint32_t(page_table.size()) - 1, start + page_count - 1);
    for (uint32_t p = start; p <= end; ++p) {
      page_table[p].state &= ~kMemoryAllocationCommit;
    }
  }

  // xmemory.cpp:1585-1645
  bool Release(uint32_t base_address) {
    uint32_t base_page = (base_address - heap_base) >> page_size_shift;
    if (base_page >= page_table.size()) {
      return false;
    }
    const PageEntry entry = page_table[base_page];
    if (entry.base_address != base_page) {
      return false;  // not a region start
    }
    uint32_t end = base_page + entry.region_page_count - 1;
    if (end >= page_table.size()) {
      end = uint32_t(page_table.size()) - 1;
    }
    for (uint32_t p = base_page; p <= end; ++p) {
      page_table[p] = PageEntry{};
      unreserved_page_count++;
    }
    return true;
  }
};

struct Violation {
  const char* op;
  uint32_t unreserved;
  uint32_t counted;
};

}  // namespace

int main() {
  std::mt19937 rng(0x5EED1234u);

  Heap heap;
  heap.page_table.assign(heap.heap_size >> heap.page_size_shift, PageEntry{});
  heap.unreserved_page_count = uint32_t(heap.page_table.size());

  // Live regions, so Release has something to hit.
  std::map<uint32_t, uint32_t> live;  // base page -> page count

  std::vector<Violation> violations;
  uint64_t ops = 0;

  auto note = [&](const char* op) {
    ++ops;
    const uint32_t counted = heap.CountFree();
    if (counted != heap.unreserved_page_count && violations.size() < 20) {
      violations.push_back({op, heap.unreserved_page_count, counted});
    }
  };

  for (int i = 0; i < 200000; ++i) {
    const int roll = int(rng() % 100);

    if (roll < 45) {
      // AllocRange with RESERVE|COMMIT -- what MmAllocatePhysicalMemoryEx does.
      const uint32_t page_count = 1 + (rng() % 64);
      const uint32_t alignment = (rng() % 4 == 0) ? 0x1000u : 0x10000u;
      uint32_t start = 0;
      if (heap.FindRange(page_count, alignment, &start)) {
        heap.CommitRange(start, page_count, kMemoryAllocationReserve | kMemoryAllocationCommit);
        live[start] = page_count;
      }
      note("AllocRange");
    } else if (roll < 60) {
      // AllocRange RESERVE only.
      const uint32_t page_count = 1 + (rng() % 64);
      uint32_t start = 0;
      if (heap.FindRange(page_count, 0x1000u, &start)) {
        heap.CommitRange(start, page_count, kMemoryAllocationReserve);
        live[start] = page_count;
      }
      note("AllocRange-reserve");
    } else if (roll < 70) {
      // AllocFixed COMMIT on already-reserved pages (the reserve-then-commit flow).
      if (!live.empty()) {
        auto it = live.begin();
        std::advance(it, rng() % live.size());
        heap.CommitFixed(it->first, it->second, kMemoryAllocationCommit);
      }
      note("AllocFixed-commit");
    } else if (roll < 78) {
      // AllocFixed RESERVE|COMMIT at a fixed address. The real code REJECTS a
      // reserve that overlaps an already-reserved page (xmemory.cpp:1326), so
      // the model has to reject it too -- otherwise it manufactures overlapping
      // regions that the real allocator would never create, and the resulting
      // accounting desync would be an artefact of this harness rather than a
      // bug in the SDK.
      const uint32_t start = rng() % (heap.page_table.size() - 128);
      const uint32_t page_count = 1 + (rng() % 32);
      bool reserved_overlap = false;
      for (uint32_t p = start; p < start + page_count; ++p) {
        if (heap.page_table[p].state != 0) {
          reserved_overlap = true;
          break;
        }
      }
      if (!reserved_overlap) {
        heap.CommitFixed(start, page_count, kMemoryAllocationReserve | kMemoryAllocationCommit);
        live[start] = page_count;
      }
      note("AllocFixed-reserve-commit");
    } else if (roll < 88) {
      // Decommit.
      if (!live.empty()) {
        auto it = live.begin();
        std::advance(it, rng() % live.size());
        heap.Decommit(heap.heap_base + (it->first << heap.page_size_shift), it->second << heap.page_size_shift);
      }
      note("Decommit");
    } else {
      // Release.
      if (!live.empty()) {
        auto it = live.begin();
        std::advance(it, rng() % live.size());
        heap.Release(heap.heap_base + (it->first << heap.page_size_shift));
        live.erase(it);
      }
      note("Release");
    }
  }

  std::printf("ops: %llu\n", (unsigned long long)ops);
  std::printf("unreserved_page_count = %u\n", heap.unreserved_page_count);
  std::printf("pages with state == 0 = %u\n", heap.CountFree());
  std::printf("largest free run       = %u\n", heap.LargestFreeRun());
  std::printf("violations: %zu\n", violations.size());
  for (const Violation& v : violations) {
    std::printf("  after %-22s unreserved=%u counted=%u  (delta %d)\n", v.op, v.unreserved,
                v.counted, int(v.unreserved) - int(v.counted));
  }
  return violations.empty() ? 0 : 1;
}