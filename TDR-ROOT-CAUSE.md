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
| playtest11 | 7.8 min | **exit `0xC0000374`**, but with a `.txt` and a 6.28 MB `.dmp` from the vectored handler |
| playtest12 | 27 min | **no crash** — window closed by hand at 10:37:44; first run to survive |

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
   installed". **Run 11 answered the question this set up: Windows does deliver
   fail-fast to vectored handlers.** The handler fired, wrote the `.txt` and a
   6.28 MB dump (see below), so Application Verifier is no longer the only route
   to a stack.

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

## Run 11: the fail-fast was caught, and it does not point at graphics

The vectored handler worked. Run 11 (pid 12724, 09:34:21 → 09:42:11, 7.8 min,
exit `0xC0000374`) produced `ufc3-20261004-094210-12724.txt` and a 6.28 MB
`.dmp` — the first report this failure has ever generated — and the log carries
the matching `crash capture armed:` line. From here on `0xC0000374` is no longer
an exit code and nothing else.

### Resolving offsets with no PDB

This build emits no debug information, so `llvm-symbolizer` has nothing to read.
The report prints `module+RVA` exactly (`addr - GetModuleHandleEx base`), and
`rexruntime.dll` plus the system DLLs all carry export tables, so a
nearest-export-below lookup names most frames for free. The two that matter are
offsets of +0x1B and +0x51 into export spans of 1472 and 240 bytes:

    [12] ntdll.dll+0x247B1       RtlFreeHeap +0x51
    [13] ucrtbase.dll+0xF05B     _free_base +0x1B
    [14] rexruntime.dll+0x11206D input::InputSystem::RefreshDevices +0xB5D
    [15] rexruntime.dll+0x112696 input::InputSystem::SetState +0x36
    [16] rexruntime.dll+0x2B91F2 ppc::HostToGuestFunction<XamInputSetState_entry> +0x52
    [17] ufc3.exe+0x642CF6       recompiled guest code
    [25] rexruntime.dll+0x245FC4 system::XThread::Execute +0x334
    [26] rexruntime.dll+0x249551 system::XHostThread::Execute +0x6A1
    [27] rexruntime.dll+0xA34E5  thread::ThreadStartRoutine +0xA5

The caveat has to be stated, because it is easy to over-read: frames 6 through
11 are ntdll *internal* heap routines with no exports of their own, so their
labels in the report (`RtlIsZeroMemory`, `RtlpNtSetValueKey`,
`RtlGetCurrentServiceSessionId`) are nearest-export guesses and are wrong. Only
frames landing a small offset into a real export are trustworthy. `_free_base`
is trustworthy, and it says the operation being serviced was a **`free()`, not
an allocation**.

`llvm-objdump -d --disassemble-symbols='?RefreshDevices@InputSystem@input@rex@@AEAAXXZ'`
puts frame 14 at `func+0xB58`, a five-byte `call` into a `free`/`operator delete`
thunk — a tail jump, which is exactly why no thunk frame appears between
`RefreshDevices` and `_free_base`.

### What the stack says

Bottom up: a guest thread (`XHostThread::Execute` → `XThread::Execute`) runs
recompiled guest code, which calls `XamInputSetState` — the rumble API — which
reaches `InputSystem::SetState`, which calls `RefreshDevices()`, whose `free()`
finds a damaged block.

The GPU thread (`t2656`, last line 09:42:10.700) is not on the stack, and the
deletion queue could not have been *detected* here in any case: it releases
`ID3D12Resource`s through the driver's allocator, not through `_free_base`. This
detection site is the CRT heap, in the input subsystem, on a guest thread.

### Why input looks like the cause rather than the victim

`RefreshDevices()` is called from all four query entry points —
`GetCapabilities`, `GetState`, `SetState`, `GetKeystroke` — and it rewrites
`devices_` and `device_owners_` with `clear()` + `push_back()` + `stable_sort`,
which is precisely where heap blocks get freed. The class has **no lock of any
kind**, and neither do the entry points: `XamInputGetState`,
`XamInputSetState`, `XamInputGetCapabilities` and `XamInputGetKeystroke` in
`xam_input.cpp` call straight through to the same `InputSystem` object. Two
guest threads arriving together would both be inside `clear()`/`push_back()` on
one vector — a realloc-while-realloc race on CRT heap blocks, detected by some
later `free()`.

Two independent facts make concurrency real rather than theoretical:

- every input *driver* locks itself (`SDLInputDriver::DrainAndLock()`,
  `MnkInputDriver::state_mutex_`), and `sdl_input_driver.cpp:404` carries a
  comment about a *lock-ordering inversion that caused deadlocks* — you onlyget that if two threads reach the input path at the same time;
- run 11 had at least seven threads alive (log `t` ids 2656, 15756, 1920, 2840,
  1884, 3564, 2680) while the **crashing thread 15712 logged nothing at all** —
  it is a pure guest thread, and there is nothing above it serialising the call.

This is a hypothesis, not a result. The honest statement: the detection site is
inside `RefreshDevices`, that function has a race-shaped hole, and no run has
yet shown two threads actually overlapping in it.

### What is deployed for run 12

Two changes in `input_system.h` / `input_system.cpp`, which only make sense
together:

1. `std::recursive_mutex mutex_` guards the four entry points. Recursive so a
   future caller of `RefreshDevices()` cannot self-deadlock. Deadlock analysis:
   while holding it we only call *down* into drivers, and the one callback that
   could come back *up* (`is_active()` → the ImGui lambda at `rex_app.cpp:254`)
   is invoked at `sdl_input_driver.cpp:197/301`, **before** `DrainAndLock()` at
   203/307, and reads only `WantCaptureMouse`. No reverse order exists.
2. `InputEntryProbe` counts arrivals **before** the mutex is taken and
   decrements **after** it is released, so it measures contention rather than
   protected work. Overlaps are logged with both thread ids and both method
   names — the first 16 in full, then every 500th, so a hot race cannot flood
   the log.

Reading run 12:

| outcome | what it means |
|---|---|
| survives + `concurrent entry` lines | contention is real and now serialised; a second run is needed to call it fixed |
| dies `0xC0000374`, no `concurrent entry` lines | input is single-threaded here; the race is ruled out and `RefreshDevices` was only where the damage was noticed |
| dies `0xC0000374` + `concurrent entry` lines | the race exists and is *not* the cause; the search moves on |
| survives, no lines | inconclusive — the path was never exercised twice |

### Run 12: it survived, and the contention was real

Launched 10:10:39, ran 27 minutes with no crash, and ended only because the
window was closed by hand. The log closes normally — `Window closing, shutting
down...`, `Execution complete`, `Title terminated; hard-exiting process.` at
10:37:44 — and `crashdumps` (formerly `diagnostico`) still holds exactly the same 25 files as before
the run, so no report and no dump were produced.

The probe fired 16 times (the first 16 are printed in full, so the true total
is between 16 and 499), and **10 of those name two different threads**:

    [t2268] InputSystem concurrent entry #1: thread 2268 entering SetState while thread 7792 is inside GetState
    [t2268] InputSystem concurrent entry #3: thread 2268 entering SetState while thread 7792 is inside SetState
    [t7792] InputSystem concurrent entry #7: thread 7792 entering GetState while thread 2268 is inside SetState
    [t7792] InputSystem concurrent entry #10: thread 7792 entering GetState while thread 2268 is inside SetState

Two threads, `GetState` and `SetState`, arriving at the same moment. Before
this change that meant both of them inside `RefreshDevices()` at once, running
`clear()` + `push_back()` on one `std::vector` with nothing between them. The
precondition for the race is not hypothetical: it happened at least ten times
in a single run.

An honesty note about the report format, because it is easy to over-read:
`g_input_owner_*` is only cleared when the count returns to zero, so a later
line can name a stale owner — `#2` reads "thread 7792 entering SetState while
thread 7792 is inside GetState". Only the lines naming two *different* ids
(#1, #3, #5, #7, #9, #10, #12, #13, #15, #16) are unambiguous.

Two supporting numbers from the same run: the error count is 1486, which is
exactly the 1470 benign boot-storm lines plus these 16 probe lines — nothing
else in 27 minutes was logged as an error. And the readback eviction backlog
reached **354 awaiting deletion** (max 215 evicted in one call, cache pinned at
its 256 cap) against 2–4 in run 10. It drained, and the run survived, but that
is a much larger backlog than anything seen before and it is recorded here
without an interpretation.

**Still not verified.** The missing control is a probe-*only* build: because
the mutex went in alongside the probe, survival cannot be attributed to the
mutex on its own — one run is a data point, not a fix. What would settle it:

- a second playtest that survives while the probe keeps showing contention; or
- a run that dies `0xC0000374` *again* with overlap lines present, which would
  mean the race is real but is not the cause, and the mutex stays only because
  the hole is real regardless.

## Crash 1 is unrelated and still open

A guest read of address `0x20` after `MmAllocatePhysicalMemoryEx` returned 0.
The allocator is **exonerated**: the free-range search matches a brute-force
reference over 576,000 cases with no divergence, the page accounting holds
`unreserved_page_count_ == count(state == 0)` across 200,000 operations, and
field data shows the physical pool genuinely exhausted (167 free pages of 131072,
largest run 160). Exonerated means the fix is not in the allocator.
