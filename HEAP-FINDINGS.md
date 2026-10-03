# Heap allocator findings — 2026-10-03 UFC 3 crash triage

## What the crash log shows

The physical pool (`heaps_.physical`, 512 MB / 4096-byte pages / 131072 pages)
refused an allocation while reporting itself far from full:

```
BaseHeap::Alloc failed to find contiguous range
pedidas 8 paginas (0.03 MB); livres 28864 de 131072 (112.8 MB),
maior faixa contigua 23079 paginas (90.15 MB) -- deveria caber
```

An 8-page (32 KB) request refused while a 90 MB contiguous run was free. That is
either a search that misses ranges, or accounting that has drifted from the page
table. Both were tested directly.

## Test 1 — is the free-range search correct? YES

`tests/standalone/allocrange_fuzz.cpp` transcribes `BaseHeap::AllocRange`'s
search verbatim from `src/system/xmemory.cpp:1388-1504` and compares it against
a brute-force reference that tries every aligned base page in the window.

    cases: 576000
    mismatches: 0

Coverage: the three heap shapes that matter (512 MB/4 KB, 512 MB/64 KB,
509 MB/4 KB), 4 KB..16 MB alignments, 4 KB..1 MB requests, both scan directions,
50-100% occupancy. The top-down scan, the stride arithmetic and the
`page_count > page_number` bail-out all behave correctly.

**Conclusion: the search does not produce false negatives.** So when it reports
exhaustion, the requested window genuinely has no run of the requested size.

## Test 2 — does the page accounting drift? NO

`tests/standalone/heapaccounting_fuzz.cpp` models the four mutators verbatim
(`AllocRange`, `AllocFixed`, `Decommit`, `Release`) and asserts, after every
operation, that

    unreserved_page_count_  ==  count(pages with state == 0)

    ops: 200000
    unreserved_page_count = 16277
    pages with state == 0 = 16277
    violations: 0

Note: an earlier run of this harness reported a +4 desync, but that was an
artefact — the model was letting `AllocFixed` reserve over already-reserved
pages, which the real code rejects at `xmemory.cpp:1326`. With the rejection
modelled, the invariant holds. **Do not treat the first result as a finding.**

**Conclusion: the accounting does not drift.**

## So the crash-1 allocation failure is legitimate

Both candidate defects are ruled out. The remaining explanation is the one the
diagnostic itself points at — `-- deveria caber, olhar a faixa pedida`:

`MmAllocatePhysicalMemoryEx_entry` (`src/kernel/xboxkrnl/xboxkrnl_memory.cpp:347`)
honours the guest's own `min_addr_range`/`max_addr_range`:

```cpp
uint32_t heap_min_addr = rex::sat_sub(min_addr_range, heap_physical_address_offset);
uint32_t heap_max_addr = rex::sat_sub(max_addr_range, heap_physical_address_offset);
heap_min_addr = heap_base + std::min(heap_min_addr, heap_size - 1);
heap_max_addr = heap_base + std::min(heap_max_addr, heap_size - 1);
```

The heap-wide "livres" / "maior faixa contigua" figures say nothing about whether
*that window* is free. The guest asked for 32 KB inside a window that was full,
got `0` back — which is the correct Xbox 360 ABI, there is no status code to
return — and then dereferenced it:

```
Unhandled guest access violation: read of guest 0x00000020 on thread 0xF8000264
```

**This is the same defect class as rexglue-sdk#420** (thunk space exhausted →
returns 0 → guest null-deref): a resource that legitimately cannot serve hands
the guest a null pointer, and the guest does not check.

## Two real defects worth fixing, neither of which is the allocator

1. **`xmemory.cpp:390-393` — the `vE0000000` `+0x1000` offset breaks ranged
   physical allocation.** `PhysicalHeap::GetPhysicalAddress` adds `0x1000` for
   heaps based at or above `0xE0000000`, and every ranged physical allocation has
   to subtract it again or the requested window is off by a page. The SDK
   documents a per-title workaround for this in a comment
   (`ignore_offset_for_ranged_allocations`, referencing xenia-canary `81aaf98e0`,
   needed for title `545108B4`) and never fixes it. Any title that passes
   explicit min/max ranges is exposed. Our title is `5451087D`.

2. **`MmAllocatePhysicalMemoryEx_entry` does not validate the window it builds.**
   `heap_min_addr` is clamped from above only. A guest that passes
   `min_addr_range > max_addr_range`, or a range entirely outside the heap,
   produces an inverted or empty window and gets a bare `0` back with nothing in
   the log distinguishing "out of memory" from "your range was nonsense".

Neither is provably the trigger for the observed crash. Both are small, safe and
independently correct.

## What is still blocked

Fixing anything in the allocator requires rebuilding `ufc3.exe`, which needs:

- **Clang 18+** — not installed on this machine (MSVC Build Tools 18 is present
  and has `cl.exe`, but the SDK requires clang per its own README).
- **Disk** — 24.9 GB free. `rexglue codegen` emits 295 MB of C++ across 569
  translation units and links a 93 MB executable; object files for that will be
  large enough to matter on 24.9 GB.

Crashes 2 and 3 (`DEVICE_HUNG`) cannot be diagnosed without a reproduction run
with `d3d12_debug = true` so that DRED reports the faulting GPU operation.