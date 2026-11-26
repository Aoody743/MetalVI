#!/bin/bash
# victorymetal: MetalVI's Wine unixlib for Metal (PE victorymetal.dll + x86_64
# Mach-O victorymetal.so). The Metal object layer is dxmt's winemetal
# (LGPL-2.1-or-later, see LICENSE.dxmt and NOTICE.md); airconv is stubbed out
# (shaders go through Metal Shader Converter instead).
#
# usage: build.sh [--install]      WINE_ROOT=<wine install> (default: runtime-env/wine-gamesir)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT=$(cd "$HERE/../../.." && pwd)
WINE="${WINE_ROOT:-$PROJECT/runtime-env/wine-gamesir}"
OUT="${OUT:-$HERE/build}"
SRC="$HERE/src"
mkdir -p "$OUT"
[ -x "$WINE/bin/winebuild" ] || { echo "no winebuild in $WINE" >&2; exit 1; }
# Headers: the target wine's, or another Wine 11 install's when it lacks the
# development headers (runtime-env/wine-gamesir has no wine/unixlib.h).
HDR="${WINE_HEADERS:-$WINE}"
if [ ! -f "$HDR/include/wine/unixlib.h" ]; then
  HDR="$HOME/Library/Application Support/com.gamemac.www/wine-engine/containers/wine_installations/10000073"
fi
[ -f "$HDR/include/wine/unixlib.h" ] || { echo "no Wine headers (set WINE_HEADERS)" >&2; exit 1; }

DEFS=(-D__WINE_PE_BUILD -D__WINESRC__ -DNOMINMAX -DNDEBUG -D_WIN32_WINNT=0x0a00)
INC=("-I$SRC" "-I$SRC/airconv" "-I$HDR/include" "-I$HDR/include/wine" "-I$HDR/include/wine/windows")

# PE side. mingw's intrin.h emits a global __fastfail in every TU: give each its own name.
i=0; OBJS=()
for f in "$SRC/main.c" "$SRC/winemetal_thunks.c" "$SRC/airconv_thunks.c" "$SRC/msc_thunks.c" "$HERE/third_party/wine_unix_lib.c"; do
  i=$((i+1)); o="$OUT/pe_$(basename "$f" .c).o"
  zig cc -target x86_64-windows-gnu -c -O2 "${DEFS[@]}" -D"__fastfail=__vm_fastfail_$i" "${INC[@]}" -o "$o" "$f"
  OBJS+=("$o")
done
zig cc -target x86_64-windows-gnu -shared -O2 -o "$OUT/victorymetal.dll" "${OBJS[@]}" -lntdll
"$WINE/bin/winebuild" --builtin "$OUT/victorymetal.dll"

# Metal Shader Converter under a name of its own: D3DMetal ships an older
# libmetalirconverter with the same install name, and dyld would hand one
# process one of the two for both (D3DMetal then fails to load, or we get
# the old API). Ad-hoc signed after the rename.
MSC_LIB="$OUT/libvictorymsc.dylib"
if [ ! -f "$MSC_LIB" ] || [ /usr/local/lib/libmetalirconverter.dylib -nt "$MSC_LIB" ]; then
  cp /usr/local/lib/libmetalirconverter.dylib "$MSC_LIB"
  chmod u+w "$MSC_LIB"
  install_name_tool -id @rpath/libvictorymsc.dylib "$MSC_LIB"
  codesign -f -s - "$MSC_LIB"
fi

# Unix side: x86_64 (the whole wine runs under Rosetta); finds winemac.so/ntdll.so next to it.
clang -arch x86_64 -dynamiclib -fblocks -ObjC -O2 -DNDEBUG -I"$SRC" -I"$SRC/airconv" \
  -o "$OUT/victorymetal.so" "$SRC/unix/winemetal_unix.c" "$SRC/unix/cache.c" "$SRC/unix/airconv_stubs.c" "$SRC/unix/msc_unix.c" "$SRC/unix/vm_unix.m" \
  -I/usr/local/include "$MSC_LIB" \
  -Wl,-install_name,@rpath/victorymetal.so -Wl,-rpath,@loader_path \
  "$WINE/lib/wine/x86_64-unix/winemac.so" "$WINE/lib/wine/x86_64-unix/ntdll.so" \
  -weak_framework CoreFoundation -weak_framework Metal -framework MetalFX -weak_framework Foundation \
  -weak_framework CoreGraphics -weak_framework QuartzCore -weak_framework ColorSync -weak_framework Cocoa -lsqlite3

# C smoke test (zero-copy compute through the unixlib)
zig cc -target x86_64-windows-gnu -O1 -I"$SRC" -I"$SRC/airconv" -o "$OUT/victorymetal_smoke.exe" "$HERE/test/smoke.c" "$OUT/victorymetal.dll"

if [ "${1:-}" = "--install" ]; then
  cp "$OUT/victorymetal.dll" "$WINE/lib/wine/x86_64-windows/"
  cp "$OUT/victorymetal.so" "$OUT/libvictorymsc.dylib" "$WINE/lib/wine/x86_64-unix/"
  echo "installed into $WINE"
fi
ls -l "$OUT/victorymetal.dll" "$OUT/victorymetal.so" "$OUT/victorymetal_smoke.exe"
