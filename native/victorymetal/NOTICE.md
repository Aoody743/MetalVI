# victorymetal

MetalVI's Wine unixlib for Metal.

## Origin and licence

`src/` is the `src/winemetal` layer of [3Shain/dxmt](https://github.com/3Shain/dxmt) at commit `faa73e4a0686e432ce5d3bb99a02b5363d76885c` (2026-11-24). It is licensed **LGPL-2.1-or-later**; see `LICENSE.dxmt`.

`third_party/wine_unix_lib.c` is `dlls/winecrt0/unix_lib.c` from Wine 11.0, also **LGPL-2.1-or-later**. It provides `__wine_init_unix_call` and the dispatcher, because Wine installs do not ship winecrt0.

## Changes

- The output is named `victorymetal.dll` / `victorymetal.so` instead of `winemetal`, so it can sit next to an installed DXMT.
- airconv (DXBC→AIR) is stubbed out in `src/unix/airconv_stubs.c`, where every entry point returns -1. MetalVI translates shaders with Apple's Metal Shader Converter; the `SM50*` thunks will be replaced by an MSC interface.

## Build

`./build.sh [--install]`. Set `WINE_ROOT` to choose the wine; the default is `runtime-env/wine-gamesir`. See `metal-backend/T1/REPORT.md` for why each step is needed.
