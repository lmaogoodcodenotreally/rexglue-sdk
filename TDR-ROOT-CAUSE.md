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

It has since been **removed from the tree**, because it was the wrong cure — see
below.

The earlier note in this document said the fault was *inside* the copy rather
than on a released buffer, so a fence before release was addressing the wrong
thing. That inference was wrong: "the copy faulted" and "the copy's destination
was freed" are the same observation seen from two sides, and instrumenting the
copy is what finally separated them.

## Instrumenting the copy ruled out the obvious suspect

Setting `draw_resolution_scale_x/y = 1` makes `IsDrawResolutionScaled()` false,
so the branch taken is the unscaled one:

    shared_memory_->UseAsCopySource();
    SubmitBarriers();
    deferred_command_list_.D3DCopyBufferRegion(
        rb.buffers[write_index], 0, shared_memory_buffer, written_address, written_length);

Shared memory is a **tiled reserved resource**, `kBufferSize = 512 MB`
(`kBufferSizeLog2 = 29`), created with `CreateReservedResource` and mapped on
demand. The copy's source offset is a raw guest physical address. That looked
like the fault: an unmapped tile, or a source offset past the end of the
resource.

Logging the operands disproved it. From a real run:

    shared memory : 0x203700000 + 0x20000000 -> ends 0x223700000
    last copy     : dst 0x226CC0000 (4 MB) | src 0x203700000 + 0x13596000, len 0x398000
    fault VA      : 0x24B382000

The fault is **outside shared memory** and **outside the destination of the last
recorded copy**. A range guard across every one of the 2,309 recorded copies in
that run never fired: every shared-memory range was in bounds. So neither operand
of the resolve copy owns the fault address.

Two runs gave fault VAs (`0x24A8AA000`, `0x24B382000`) that belong to no live
logged resource at all.

## What does release resources while copies are in flight

`readback_buffers_` — keyed by `(written_address, written_length)` — is a cache
with `kMaxReadbackBuffers = 256`. A 30-second UFC 3 session accumulated
**492 distinct resolve keys and 342 buffer creations**, so the map is over its
cap almost immediately and `EvictOldReadbackBuffers` runs **every frame**.

Three sites drop readback buffers. Two had no synchronization whatsoever:

| site | what it did |
|---|---|
| resolve growth | bare `Release()` |
| memexport growth | queue flush, then `Release()` |
| `EvictOldReadbackBuffers` | **bare `Release()`, no fence, no deferral** |

`DeferredCommandList` stores raw `ID3D12Resource*` with no `AddRef`, and a
Direct3D 12 command list does not keep the resources it names alive for the
application. Releasing a buffer that a recorded copy still references means the
GPU writes into memory that no longer exists — which is not reported as an API
error. It surfaces as a DRED page fault, a breadcrumb that stops on
`COPYBUFFERREGION`, and `DXGI_ERROR_DEVICE_HUNG`.

That fits every observation: the faulting op is a copy, and the fault address
belongs to nothing live.

## The fix

`ScheduleResourceDeletion()` hands the resource to `resources_for_deletion_` —
the SDK's own deferred-deletion queue, already used correctly for the resolve
downscale buffer — and all three sites now go through it. It waits one
submission past both the current and the last completed submission, which covers
work that is *recorded but not yet submitted*, the state nearly all of these
buffers are in when dropped.

Crucially there is **no queue flush**. That was the fatal flaw in the fence:
`AwaitAllQueueOperationsCompletion()` closes the open submission, and
`GraphicsUploadBufferPool` reclaims pages by submission index, which is why the
fence made deaths faster rather than slower. Deferral fixes the lifetime bug
without touching submission boundaries, so the cvar is gone — there is nothing
left to toggle.

Also added: a range guard on every `CopyBufferRegion` at the replay choke point
(an out-of-range copy records without complaint and faults on the GPU; now it is
logged with full operands and skipped), operand logging behind
`d3d12_readback_copy_log`, and a per-call eviction count so the churn is
measurable rather than assumed.

**Status: hypothesis with a complete evidence chain, not yet confirmed by a
surviving run.** Two earlier "fixes" were disproven by test. Confirmation means
a match well past the 60–90s mark with evictions still firing; refutation means
another `DEVICE_HUNG`, in which case the watcher attributes the fault VA against
every recorded copy's operand ranges and against readback buffer bases.

## Crash 1 is unrelated and still open

A guest read of address `0x20` after `MmAllocatePhysicalMemoryEx` returned 0.
The allocator is **exonerated**: the free-range search matches a brute-force
reference over 576,000 cases with no divergence, the page accounting holds
`unreserved_page_count_ == count(state == 0)` across 200,000 operations, and
field data shows the physical pool genuinely exhausted (167 free pages of 131072,
largest run 160). Exonerated means the fix is not in the allocator.
