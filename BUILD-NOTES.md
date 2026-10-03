# Building on Windows — working recipe

Verified on 2026-10-03 with LLVM 23.1.2 + MSVC Build Tools 18, building
`ufc3recomp`. Two of the seven Windows defects listed in the project README bite
during a fresh clone; this is the order that works and what to do about them.

## Prerequisites

- **Clang 18+** — the SDK's own presets set `CMAKE_CXX_COMPILER=clang`, and
  `-march=x86-64-v2` with clang-cl style MSVC headers. MSVC's `cl.exe` cannot
  build this project. `winget install LLVM.LLVM`.
- **CMake 3.25+ and Ninja** — MSVC Build Tools ships both, no separate install:
  - cmake: `VC\Tools\..\..\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`
  - ninja:  `Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja`
- **Submodules** — `git submodule update --init --recursive --depth 1`. On a
  Windows/D3D12 build the required set is cli11, libmspack, FFmpeg,
  tomlplusplus, simde, xxHash, spdlog, fmt, utfcpp, imgui, sdl3
  (`thirdparty/CMakeLists.txt` hard-fails on any that are missing).

Put all of the above on `PATH` before configuring.

## Defect #1 — `MSPACK_DIR` symlink tree (blocks every Windows clone)

`libmspack` keeps `cabextract/mspack/*` as symlinks into `libmspack/mspack/`.
Git for Windows materialises a symlink as a **29-byte text file containing the
relative path**, and clang then tries to compile that text as C:

```
thirdparty/libmspack/cabextract/mspack/lzxd.c:1:1: error: expected identifier or '('
    1 | ../../libmspack/mspack/lzxd.c
```

Creating real symlinks needs `core.symlinks=true` plus Administrator or
Developer Mode, so on a normal account the fix is to materialise the files:

```powershell
$root = "<sdk>"
$real = "$root\thirdparty\libmspack\libmspack\mspack"
Get-ChildItem "$root\thirdparty\libmspack\cabextract\mspack" -File | ForEach-Object {
  $c = Get-Content $_.FullName -Raw
  if ($c -and $c.Trim() -match '^\.\./\.\./libmspack/mspack/(.+)$') {
    $src = Join-Path $real $Matches[1]
    if (Test-Path $src) { Copy-Item $src $_.FullName -Force }
  }
}
```

This dirties the `libmspack` submodule. Leave it uncommitted — the parent repo
should keep recording the upstream submodule SHA.

`thirdparty/moltenvk` has the same problem (9 stubs) but only matters for
Apple/Vulkan builds, which this configuration skips (`Vulkan=OFF`).

## Order of operations

The codegen CLI is itself a C++ target, so it has to be built before it can
generate anything:

```powershell
# 1. SDK: configure (multi-config Ninja) and build just the CLI
cmake --preset win-amd64 -S <sdk>
cmake --build <sdk>\out\build\win-amd64 --config Release --target rexglue

# 2. Game data. The manifest wants it at <project>/assets/game/default.xex
#    (game_root = "assets/game", file_path = "assets/game/default.xex").
#    A hardlink costs no disk:
#    New-Item -ItemType HardLink <project>\assets\game\default.xex <iso-extracted>\default.xex

# 3. Codegen -> <project>/generated/
Push-Location <project>
<sdk>\out\win-amd64\rexglue.exe codegen ufc3_manifest.toml

# 4. Configure and build the game. Release, not RelWithDebInfo: the generated
#    translation units are ~295 MB across 569 files and debug info for that
#    does not fit comfortably on a normal drive.
cmake --preset win-amd64-release -S <project> -DREXSDK_DIR=<sdk>
cmake --build <project>\out\build\win-amd64-release
```

## Notes

- The SDK reports itself as `0.10.0-dev.g<sha>`; the sha is whatever your branch
  is on, so a local patch shows up in the configure summary. Useful for
  confirming you built your own tree.
- `ufc3recomp` sets `REXGLUE_GPU_STATIC ON`, so the GPU backend is linked into
  `ufc3.exe` rather than loaded as a plugin. Patches to
  `src/graphics/d3d12/*` therefore land in the executable, and no separate
  plugin build is needed.