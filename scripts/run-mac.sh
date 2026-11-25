#!/usr/bin/env bash
set -euo pipefail

# Generic macOS launcher for MetalVI using GameHub (Wine-Proton + Apple GPTK D3DMetal)
# or custom Wine/GPTK installations via environment overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GAMEHUB_ENGINE_DIR="${GAMEHUB_ENGINE_DIR:-$HOME/Library/Application Support/com.gamemac.www/wine-engine}"

if [[ -z "${WINE_ROOT:-}" ]]; then
  WINE_ROOT="$(ls -d "$GAMEHUB_ENGINE_DIR"/containers/wine_installations/* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${GPTK_ROOT:-}" ]]; then
  GPTK_ROOT="$(ls -d "$GAMEHUB_ENGINE_DIR"/downloads/gptk-* 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ -z "${WINE_ROOT:-}" || ! -x "$WINE_ROOT/bin/wine" ]]; then
  echo "error: Wine binary not found. Set WINE_ROOT to your Wine/Wine-Proton directory." >&2
  exit 1
fi
if [[ -z "${GPTK_ROOT:-}" || ! -f "$GPTK_ROOT/external/libd3dshared.dylib" ]]; then
  echo "error: Apple GPTK D3DMetal not found. Set GPTK_ROOT to your gptk directory." >&2
  exit 1
fi

STATE_ROOT="${METALVI_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/metalvi}"
export WINEPREFIX="${WINEPREFIX:-$STATE_ROOT/wineprefix}"
export WINEARCH=win64

case "$WINEPREFIX" in
  "$REPO_ROOT"/*)
    echo "error: WINEPREFIX ($WINEPREFIX) must be outside the MetalVI repository." >&2
    exit 1
    ;;
esac

CACHE_UNIX="${METALVI_CACHE_DIR:-$STATE_ROOT/cache}"
DATA_UNIX="${METALVI_DATA_DIR:-$STATE_ROOT/data}"
mkdir -p "$WINEPREFIX" "$CACHE_UNIX" "$DATA_UNIX"

# Optionally seed native d3dcompiler_47.dll from GameHub shared base if missing in prefix.
PREFIX_D3DC="$WINEPREFIX/drive_c/windows/system32/d3dcompiler_47.dll"
if [[ ! -f "$PREFIX_D3DC" ]]; then
  NATIVE_D3DC="${NATIVE_D3DCOMPILER_47:-$(ls "$GAMEHUB_ENGINE_DIR"/dependencies/shared/*/temp/base/system32/d3dcompiler_47.dll 2>/dev/null | head -n 1 || true)}"
  if [[ -n "$NATIVE_D3DC" && -f "$NATIVE_D3DC" ]]; then
    mkdir -p "$(dirname "$PREFIX_D3DC")"
    cp -f "$NATIVE_D3DC" "$PREFIX_D3DC"
  fi
fi

to_wine_z() {
  local p="$1"
  printf 'Z:%s' "${p//\//\\}"
}

export ROSETTA_ADVERTISE_AVX=1
export WINE_GPTK_LIBD3DSHARED_PATH="$GPTK_ROOT/external/libd3dshared.dylib"
export WINEDLLPATH="$GPTK_ROOT/wine:$WINE_ROOT/lib/wine"
export DYLD_FALLBACK_LIBRARY_PATH="$GPTK_ROOT/external:$WINE_ROOT/lib:$WINE_ROOT/lib/wine/x86_64-unix:${DYLD_FALLBACK_LIBRARY_PATH:-}"
export PATH="$WINE_ROOT/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export WINEDEBUG="${WINEDEBUG:--all}"
export WINEDLLOVERRIDES="${WINEDLLOVERRIDES:-d3dcompiler_47=n,b}"
export VICTORY_CACHE_DIR="${VICTORY_CACHE_DIR:-$(to_wine_z "$CACHE_UNIX")}"
export VICTORY_DATA_DIR="${VICTORY_DATA_DIR:-$(to_wine_z "$DATA_UNIX")}"
unset VICTORY_INPLACE

exec "$WINE_ROOT/bin/wine" "$@"
