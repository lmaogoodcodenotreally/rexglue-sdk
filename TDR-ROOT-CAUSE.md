# Root cause: crashes 2 and 3 — GPU timeout in a buffer copy, after a dispatch

## The finding

Two independent runs, different builds, different sessions, the same signature:

    Present failed: HRESULT 0x887A0005; GetDeviceRemovedReason: 0x887A0006
    DRED breadcrumb: completed 3843 of 4825 ops
      [3840] op type 15  RESOURCEBARRIER
      [3841] op type 6   DISPATCH
      [3842] op type 15  RESOURCEBARRIER
      [3843] op type 7   COPYBUFFERREGION   <-- FAULT
    DRED page fault at VA 0x000000024A8AA000

and, from an earlier staging run:

    DRED breadcrumb: completed 2584 of 3808 ops
      [2582] op type 6   DISPATCH
      [2583] op type 15  RESOURCEBARRIER
      [2584] op type 7   COPYBUFFERREGION   <-- FAULT
    DRED page fault at VA 0x0000000280781000

Op codes are `D3D12_AUTO_BREADCRUMB_OP` from `d3d12.h`:

    6 = DISPATCH
    7 = COPYBUFFERREGION
   15 = RESOURCEBARRIER

So: the GPU takes a long compute dispatch, and the **buffer copy that follows it
faults**. `0x887A0006` is `DXGI_ERROR_DEVICE_HUNG`; the GPU stopped consuming and
Windows reset the device.

Both fault addresses are host Direct3D 12 virtual addresses, far above the guest
arena (the guest physical base is `0x200000000` in these runs). So the resource
being faulted is a host-allocated D3D12 buffer, not guest memory.

## Getting here took fixing the reporting first

Every crash before 2026-10-03 was anonymous. Three separate defects had to be
fixed before any of this could be seen, and none of them fixed a crash:

1. **`rex::FatalError` reported only to stderr.** `ufc3.exe` is a GUI-subsystem
   executable (`dumpbin /HEADERS` → subsystem 2), so it has no stderr. Every
   fatal error was discarded and the process aborted with `0xC0000409` and
   nothing in the log.

2. **The logged line did not survive the abort.** `REXLOG_ERROR` buffers and
   `std::abort()` does not flush. The first version of the fix logged the message
   and still lost it; it needed `FlushLogging()`.

3. **Device loss had two entry points and only one was diagnosed.**
   `D3D12CommandProcessor::BeginSubmission` checks `GetDeviceRemovedReason()`,
   logs it, and dumps DRED. The presenter's loss callback went straight to
   `rex::FatalError` with no reason at all. Every one of these crashes came
   through the presenter.

4. **`DXGI_ERROR_DEVICE_HUNG` was not handled by the presenter.** The switch only
   covered `DEVICE_REMOVED` and `DEVICE_RESET`, so a TDR fell into `default`,
   where `SUCCEEDED()` is false, and the presenter returned `kNotPresented` — the
   one loss reason that matters most for a GPU-bound workload, silently ignored.

5. **DRED was enabled but never read on that path.** The command processor only
   dumps DRED when it notices the loss itself.

6. **Logging only `Present`'s HRESULT is not enough.** `Present` returns
   `DXGI_ERROR_DEVICE_REMOVED` (0x887A0005) for *any* loss, including one the OS
   has already reset — so the original cause is gone from it. Only
   `GetDeviceRemovedReason()` still reads back `0x887A0006`.

## What was tried and did not work

`d3d12_readback_fence_before_release`, waiting for the queue before releasing a
replaced resolve readback buffer. The unguarded `Release()` it guards is real and
DRED pointed at the readback path — but enabling the wait made in-match runs die
*sooner* (90s, then 38s) than the unmodified v1.0.2 build (230s, 82s, 75s).
Likely because `AwaitAllQueueOperationsCompletion()` ends the open submission and
`GraphicsUploadBufferPool` reclaims pages by submission index, so disturbing
submission boundaries appears to stop pages being recycled. That mechanism is a
hypothesis, not a measurement.

It is retained, **default off**, with the reversal recorded at the declaration.

Note the fault is *inside* the copy, not on a released buffer, so a fence before
the release was addressing the wrong thing.

## Next step

The copy that faults is the resolve readback:

    deferred_command_list_.D3DCopyBufferRegion(
        rb.buffers[write_index], 0, shared_memory_buffer, written_address, written_length);

Open question: which of those two resources is faulting, and why a `DISPATCH`
immediately precedes it. Worth checking whether a compute UAV is being created on
an upload- or readback-heap resource, which Direct3D 12 does not permit —
`D3D12_HEAP_TYPE_UPLOAD` is CPU-write-only and `D3D12_HEAP_TYPE_READBACK` is
`COPY_DEST`-only. There is also a downscale path in the same function that
creates a `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS` buffer and runs a compute
dispatch over it, though it should be inactive at `draw_resolution_scale` 1.

That is the next thing to investigate, not something this document concludes.

## Crash 1 is unrelated and still open

A guest read of address `0x20` after `MmAllocatePhysicalMemoryEx` returned 0.
The allocator is **exonerated**: the free-range search matches a brute-force
reference over 576,000 cases with no divergence, the page accounting holds
`unreserved_page_count_ == count(state == 0)` across 200,000 operations, and
field data shows the physical pool genuinely exhausted (167 free pages of 131072,
largest run 160). Exonerated means the fix is not in the allocator.
