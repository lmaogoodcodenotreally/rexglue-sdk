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
downscale buffer — and all three sites now go through it.

### Why waiting for `GetCurrentSubmission()` is sufficient — checked, not assumed

The queue drains with `if (front().first > submission_completed_) break;`, so a
resource is released only once `submission_completed_` has reached the
submission index it was scheduled with. The scheduling side uses
`max(GetCurrentSubmission(), GetCompletedSubmission() + 1)`. The question is
whether that index is *at least* the submission whose GPU work still names the
resource — because if it were smaller, deferral would be a slower version of the
same bug. Reading `EndSubmission` answers it directly:

    deferred_command_list_.Execute(command_list_, command_list_1_);   // line 3492
    direct_queue->ExecuteCommandLists(1, execute_command_lists);      // line 3495
    direct_queue->Signal(submission_fence_, submission_current_++);   // line 3509

The deferred list is executed and then signalled with `submission_current_`
*before* it is incremented, and `BeginSubmission` resets that list only when it
opens a submission — while the resolve path opens one first (line 2910). So a
copy recorded into the list completes exactly when
`submission_completed_ >= submission_current_`, which is the value
`GetCurrentSubmission()` reports at record time. The `max(...)` can only ever
choose that value or a larger one, so the release is provably never early.

`GetCompletedSubmission() + 1` is the conservative half: if the GPU has already
caught up past the current submission, waiting only for "current" would schedule
the release at a completed index and the drain would free it immediately.

Crucially there is **no queue flush**. That was the fatal flaw in the fence:
`AwaitAllQueueOperationsCompletion()` closes the open submission, and
`GraphicsUploadBufferPool` reclaims pages by submission index, which is why the
fence made deaths faster rather than slower. Deferral fixes the lifetime bug
without touching submission boundaries, so the cvar is gone — there is nothing
left to toggle.

This is also why the three sites are safe from double-release: each one
schedules the pointer and then immediately overwrites or nulls its own slot
(`rb.buffers[write_index] = buffer`, `readback.buffers[i] = nullptr` followed by
`erase`), so no map entry can schedule the same resource twice.

Also added: a range guard on every `CopyBufferRegion` at the replay choke point
(an out-of-range copy records without complaint and faults on the GPU; now it is
logged with full operands and skipped), operand logging behind
`d3d12_readback_copy_log`, and a per-call eviction count so the churn is
measurable rather than assumed.

**Status: the lifetime fix removed `DEVICE_HUNG` from every run — and was
replaced by a different failure, so it is not confirmed.** The test record after
deploying deferral (`87a68d5b…`, since rebuilt):

| run | survived | ended with |
|-----|----------|-----------|
| playtest6 | 5.5 min | no dump, no report, no event |
| playtest7 | 1.6 min | no dump, no report, no event |
| playtest8 | 7.2 min | `.dmp` opened, 0 bytes, no report |
| playtest9 | 3.0 min | **exit `0xC0000374` STATUS_HEAP_CORRUPTION** |
| playtest10 | 5.8 min | **exit `0xC0000374` STATUS_HEAP_CORRUPTION**, duplicate guard never fired |

Before the fix, five of six runs ended in `0xC0000409` with a full dump; the
device-loss signature has not appeared once since. Checked independently of the
logs: the Application event log holds exactly one `D3DDRED2` report for
`ufc3.exe` (`DeviceRemovedReason 0x887A0006`, i.e. `DEVICE_HUNG`) in the last
six hours, and it belongs to **playtest5, which ran 08:11–08:12 — before the
fix was deployed**. Runs 6 through 10 produced none. Whether the absence is
because the lifetime bug is gone, or because the process now dies of something
else before the GPU can complain, is not yet settled — the surviving runs are
not a pass.

One more thing worth recording about run 10: it produced **no Windows event of
any kind** at 09:13:10, so WER is not going to hand over a stack either. The
two `bad_module_info` APPCRASH events in that window belong to other processes
(different pids) and are unrelated.

`0xC0000374` is the first code this project has ever produced that is not
`0xC0000409` or `0xC0000005`. It is worth noticing what it costs us: the heap's
own fail-fast runs past `SetUnhandledExceptionFilter`, past both `signal()`
handlers, and past `MiniDumpWriteDump` — no `.dmp`, no `.txt`, no log line, no
Windows event. The exit code is the *only* evidence it happens at all, which is
why `watch-attrib.ps1` alone would have reported "died but no DRED page fault
recorded" and nothing else.

Two things follow, both committed:

1. `crash_capture.h` now writes the readable `.txt` **before** attempting the
   `.dmp`, and waits briefly instead of tearing the process down when a second
   fault arrives during a write. Order is what makes evidence survive a crash
   inside the dump writer (`6805eb0`).
2. The deletion queue refuses an address it already holds and says so
   (`5964ae1`). A double `Release()` through that queue is the most direct cause
   of `0xC0000374` available in this code, and it was the only new way the fix
   changed resource ownership.
3. `crash_capture.h` installs a **vectored first-chance handler** for
   `0xC0000374`/`0xC0000409`/`0xC0000417`/`0xC0000420` and writes a stack trace
   into the `.txt`. A vectored handler is dispatched before the structured
   search that fail-fast skips, so it is the last place in-process that can see
   the thread with its stack intact. `OnPostInitLogging` logs that the handler
   is armed, so "died with no report" can be told apart from "handler never
   installed". If a run still dies with no report despite that line, Windows is
   not delivering fail-fast to vectored handlers either and the answer has to
   come from Application Verifier — which needs elevation this session does not
   have.

### What run 10 ruled out

Run 10 was the test the paragraph above set up, and it came back
`0xC0000374` with the duplicate guard **never once** logging `handed to the
deletion queue twice`. So, with evidence rather than assumption:

- **Two sites naming one address while it is queued** — ruled out. The guard
  would have said so; it said nothing in 1759 buffer creations and 42 evictions.
- **Unbounded growth of the deletion queue** — ruled out. The eviction line
  prints `N awaiting deletion` and it stayed at 2–4 for the whole run.
- **The range guard corrupting memory** — it is read-only: four
  `GetDesc()`/`GetGPUVirtualAddress()` calls and a comparison, no writes.
- **A `ReadbackBuffer&` outliving its map node** — `EvictOldReadbackBuffers`
  runs only at frame open (`BeginSubmission`, `is_opening_frame`), every caller
  takes its reference *after* `BeginSubmission` has returned, and
  `AwaitAllQueueOperationsCompletion()` is only `CheckSubmissionFence()`, which
  never touches the readback maps.
- **A cross-thread race on the queue** — the class has no mutex anywhere, so
  this was a real concern: `GraphicsSystem` spins up a dedicated command
  processor thread and outside callers reach it only through `CallInThread`.
  All 1804 `[gpu]` lines in run 10 are one thread (`t15680`).
- **The guest heaps** — `xmemory.cpp` allocates no `malloc`/`HeapAlloc`; the
  guest heaps sit in the mapped arena. Guest-side heap damage therefore *cannot*
  produce `0xC0000374`. Whatever corrupts it is on the host side.

Still open, and now the whole of the search: a queue-versus-bare-`Release()`
double free, which the guard structurally cannot see because it only inspects
queue entries (`RequestReadbackBuffer`'s bare release has exactly one caller and
is safe); and every host heap user that has nothing to do with this queue —
spdlog's per-line formatting, the deferred command list's stream, pipeline and
shader creation. The guard closes one door; it does not mean the room is empty.

A note on timing, because it cuts against the "the fix caused it" reading: every
pre-fix run died at 60–90 s of device loss, and `0xC0000374` needs longer than
that to appear. Corruption that predates the fix would have been masked by the
GPU dying first. Longer survival alone can surface an older bug, so attribution
is not yet proven either.

### A fourth bare `Release()` that looks like the bug and is not

`RequestReadbackBuffer` (line 4886) releases its shared `readback_buffer_` with
no fence and no deferral — same shape as the three sites above, and it is
returned straight into a `D3DCopyBufferRegion` destination (line 2783). It is
very tempting to convert it as well. It does not need it, and the reason is
worth writing down because the next person will be tempted too.

Its only caller is `IssueDraw_MemexportReadbackFullPath`, which records the
copies and then *immediately* calls `AwaitAllQueueOperationsCompletion()` at
line 2788. That is not a soft flush. It resolves to
`CheckSubmissionFence(submission_current_)`, which — because
`await_submission >= submission_current_` — takes the branch that calls
`EndSubmission(false)` to close the open submission, and then blocks on
`submission_fence_->SetEventOnCompletion` until `GetCompletedValue()` passes it.
So when that function returns, the GPU has *finished* the copies that name the
buffer. The next call's `Release()` therefore lands on an idle resource.

Every path from recording those copies to returning reaches line 2788 with no
early exit between them, so there is no window where copies are outstanding and
the buffer is releasable.

This is the counter-example that keeps the other three honest: the defect is not
"release without a fence", it is "release while a *deferred* copy still names
it". The memexport singleton is synchronous, the resolve and eviction maps are
not — which is why three sites needed changing and this one must be left alone.

## Crash 1 is unrelated and still open

A guest read of address `0x20` after `MmAllocatePhysicalMemoryEx` returned 0.
The allocator is **exonerated**: the free-range search matches a brute-force
reference over 576,000 cases with no divergence, the page accounting holds
`unreserved_page_count_ == count(state == 0)` across 200,000 operations, and
field data shows the physical pool genuinely exhausted (167 free pages of 131072,
largest run 160). Exonerated means the fix is not in the allocator.
