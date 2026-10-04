# ufc3recomp crash triage — 2026-10-03

Sources: `diagnostico/*.txt` (3 crashes), `userdata/ufc3.log` (3 sessions),
`paulogaab21/ufc3recomp` @ `8341365`, `rexglue/rexglue-sdk` @ `c94f5eb` (v0.10.0).

Installed build: **v1.0.2** (= newest release; repo HEAD is 1 commit ahead, unreleased).
GPU: **AMD Radeon RX 7900 XTX**, D3D12/DirectND.

---

## 1. The three crashes

| # | Time | Exception | Site | Mode |
|---|------|-----------|------|------|
| 1 | 16:06:04 | `0xC0000005` AV read | `ufc3.exe+0x23AA0B5` | guest read of `0x20` |
| 2 | 16:07:46 | `0xC0000409` abort | `ufc3.exe+0x59F1190` | GPU TDR hang |
| 3 | 16:09:07 | `0xC0000409` abort | `ufc3.exe+0x59F1190` | GPU TDR hang |

Two distinct failure modes, not one.

### Crash 1 — guest null-pointer dereference

```
16:06:04.001 [t5800] BaseHeap::Alloc failed to find contiguous range
             heap 'virtual' sem faixa: pedidas 8 paginas (0.03 MB);
             livres 28864 de 131072 (112.8 MB),
             maior faixa contigua 23079 paginas (90.15 MB)
             -- deveria caber, olhar a faixa pedida
16:06:04.728 [t5800] Unhandled guest access violation:
                    read of guest 0x00000020 on thread 0xF8000264
```

**This is the bug.** An 8-page (32 KB) request was refused while the pool had
**112.8 MB free and a single 90 MB contiguous run**. The allocator returns
`false` with `*out_address = 0`, the guest gets a null pointer back, and 0.7 s
later it dereferences `0x20`.

Source: `src/system/xmemory.cpp:1499-1504` (`BaseHeap::AllocRange`) →
`src/system/xmemory.cpp:1915` (`PhysicalHeap::Alloc`) →
`src/system/xmemory.cpp:546` (`AccessViolationCallback`, which then aborts by design).

Same failure *class* as upstream **rexglue-sdk#420** (thunk space exhausted →
returns 0 to the guest → guest null-deref): a resource that reports itself
non-empty but refuses to serve, handing 0 to the guest.

Note: the boot-time `HEAP CHEIO` messages are **benign**. The 512 MB physical
pool really is ~full at boot (`livres 167 de 131072`) and the game backs off
after ~490 retries. The 16:06:04 failure is the only anomalous one.

### Crashes 2 & 3 — GPU hang, then the SDK gives up

```
16:07:46.574 engasgo: quadro de 2361.0 ms
16:07:46.574 D3D12 device removed: 0x887A0006 - DEVICE_HUNG (TDR)
```

The 2361 ms / 2353 ms frames *are* the TDR window — Windows' default `TdrDelay`
is 2 s. The GPU stopped consuming, the OS reset the device, and then:

```
GraphicsSystem::OnHostGpuLossFromAnyThread()   src/graphics/graphics_system.cpp:220
  -> rex::FatalError("Graphics device lost ...")   include/rex/assert.h
     -> std::abort()                                => 0xC0000409
```

So the `abort()` at `ufc3.exe+0x59F1190` is **intentional** — it is the SDK
saying "I lost the GPU". It is a symptom, not the fault.

**The ~1500 `Failed to create a N MB readback buffer` and 1771
`PM4_DRAW_INDX ... Failed in backend` lines are all downstream.** They are
`CreateCommittedResource` and `IssueDraw` failing *after* the device was already
removed. Do not chase them.

---

## 2. Ruled out

- **The repo's shader translation is not the cause.**
  `src/ufc3_shader_translate.cpp:3` — *"This file compiles but does not link into
  the game executable."* It is dead code in v1.0.2. HEAD commit `8341365`
  ("Limita a traducao de shader: ela estava derrubando a GPU") is an
  already-fixed *earlier* instance of that problem, and the cvar default is
  `false`.
- **The readback path is not the trigger.** Failures begin 24 ms *after* the
  stall is reported and resolve 2.3 s *after* the GPU actually hung.
- **`readback_resolve = "some"`** in `ufc3.toml` is a deliberate author choice
  (documented in the file: it fixes the broken Create-a-Fighter torso). Worth
  testing `"none"`, but it is not the bug.
- **Not thunk exhaustion** — no `Thunk address space exhausted` in the log.
- All 1771 failing draws are `tess_mode=1` (tessellated), which matches
  **rexglue-sdk#474** ("tessellated pipelines fail on AMD"). Notable because the
  card is AMD — but those failures are post-TDR, so it is a lead, not a
  conclusion.

---

## 3. The evidence gap

**DRED is not enabled in your build.** `ufc3_provider.cpp:231` enables D3D12
Device Removed Extended Data, but only when the cvar `d3d12_debug` is true
(default `false`). Your log contains neither *"DRED ... enabled"* nor
*"Failed to enable DRED"* — the debug layer block never ran. So
`LogDeviceRemovalDiagnostics` had no breadcrumbs, and there is **no record of
which GPU operation hung the card**.

That is fixable **without recompiling** (see below).

---

## 4. What to do

### Step 1 — turn on DRED (no rebuild, highest value)

`rex_app.cpp:156-158` loads `ufc3.toml` *before* the graphics system is created,
so a cvar set in that file is honoured at device-creation time.

Add to `C:\Users\User\Downloads\QuiverLauncher-win-Portable\Apps\UFCUndisputed3-UFC3Recomp\ufc3.toml`:

```toml
d3d12_debug = true
```

Next TDR will then print, per `command_processor.cpp:3224-3247`:

```
DRED breadcrumb: completed N of M ops
  [k] op type <n>      <-- FAULT
DRED page fault at VA 0x...
```

`<-- FAULT` names the exact GPU op. That is what turns this from guesswork into
a diagnosis. (It costs some performance — diagnosis run, not a permanent setting.)

> **Warning — do not let anything rewrite this file casually.** Upstream
> **SDK defect #7**: `SaveConfig` serialises string cvars with no escaping, so a
> single Windows path written into `ufc3.toml` (`\U` reads as a Unicode escape)
> makes the whole file unparseable and **every setting silently reverts**,
> including `gpu_plugin`. Back the file up before editing, and re-check it after
> any run that opens the in-game options screen.

### Step 2 — mitigate the TDR while diagnosing (no rebuild, admin + reboot)

The stall is ~2.35 s against a 2 s default. Raising the delay often lets slow-but-
correct GPU work finish:

```powershell
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 10 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDdiDelay /t REG_DWORD /d 20 /f
```

Reversible with `/d 2` and `/d 5`. This only buys time — if the GPU is truly dead
it still crashes, just later. Treat it as a band-aid, not the fix.

### Step 3 — cheap A/B while you have the log open

Try `readback_resolve = "none"` (costs the Create-a-Fighter torso rendering) and
see whether the hang moves. One variable, one run, and it separates the resolve
path from the tessellation path.

### Step 4 — report it upstream

Four issues already carry this exact crash and have no maintainer reply:
**#3, #7, #8, #9**. Post to **rexglue-sdk**, not just ufc3recomp — every fault
found is in the SDK, not in `paulogaab21/ufc3recomp`. Include:

- the `pedidas 8 paginas ... -- deveria caber` line (crash 1) — that is the
  allocator bug, and it is the strongest single piece of evidence in the log;
- `DEVICE_HUNG` + the DRED breadcrumbs from step 1;
- GPU `AMD Radeon RX 7900 XTX`, D3D12, SDK `c94f5eb` (v0.10.0).

---

## 5. If you want to patch it yourself

The allocator defect is in your **local SDK** (`xmemory.cpp`), not in
`ufc3recomp` — note that the `heap 'virtual' sem faixa ...` message is your own
diagnostic patch; it is not in upstream `rexglue-sdk`.

Two things in `BaseHeap::AllocRange` are worth a look regardless of which line
is firing — but only one of the two is actually suspect, and saying so matters
because the other one would send a fix attempt somewhere it does not belong:

1. **`xmemory.cpp:1449` — `if (page_count > page_number)` is NOT a bug.** This
   document previously called it "comparing a count against an absolute page
   index, two unrelated quantities". That reading is wrong. The `else` branch
   executes `page_number - page_count` in `uint32_t` arithmetic, which underflows
   exactly when `page_number < page_count` — which is exactly what
   `page_count > page_number` tests. It is an underflow guard, it is correct, and
   the comment above it ("not enough space left to fit entire page range")
   describes it accurately. Left alone deliberately.
2. **`xmemory.cpp:1396` — `rex::align(high_address, alignment)` rounds up**
   (`rex/math.h:43`), then gets clamped back down by the `std::min`. So the
   window's high bound can extend *above* what the caller requested, capped at
   the heap end. It cannot produce a refusal, so it is not crash 1 — but it does
   mean an allocation can land outside the caller's stated `max`. Rounding down
   would state the intent; not changed here because crash 1's cause is still
   unknown and this is not it.

The search-bug question is now decided by measurement rather than by reading
the code: on any refusal the allocator prints which heap refused, the stride it
was stepping by, and one of three verdicts — `BUG IN THE SEARCH` (a
stride-aligned base fitted and was missed), `ALIGNMENT, NOT SPACE` (contiguous
pages existed but none landed on the stride), or `GENUINELY NO ROOM`. That lands
in `src/system/xmemory.cpp` (`BaseHeap::AllocRange`), commit `d45f443`.

Rebuilding `ufc3.exe` needs the XEX plus the full ReXGlue toolchain, so treat
steps 1-3 as the practical path.