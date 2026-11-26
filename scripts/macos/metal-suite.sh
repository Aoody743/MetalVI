#!/bin/bash
# Synthetic GPU regression suite. Never launches the game.
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: METAL_SUITE_WINE=/path/to/installed/wine bash scripts/macos/metal-suite.sh [--quick]' \
    'The suite copies Wine and creates its own prefix under TMPDIR.' \
    'METAL_SUITE_ASSETS overrides the sibling metal-backend directory.' \
    'METAL_SUITE_LOG_DIR saves logs under an existing or new directory.' \
    'METAL_SUITE_PREFIX_SEED may point to a completed synthetic Wine prefix to clone.'
}
quick=0
case "${1:-}" in
  "") ;;
  --quick) quick=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

root=$(cd "$(dirname "$0")/../.." && pwd -P)
assets="${METAL_SUITE_ASSETS:-$root/../metal-backend}"
install="${METAL_SUITE_WINE:-}"
if [[ ! -x "$install/bin/wine64" || ! -x "$install/bin/wineserver" ||
      ! -f "$install/lib/wine/x86_64-windows/victorymetal.dll" ||
      ! -f "$install/lib/wine/x86_64-unix/victorymetal.so" ]]; then
  usage >&2
  printf 'METAL_SUITE_WINE must point to a Wine tree with victorymetal installed.\n' >&2
  exit 2
fi
install=$(cd "$install" && pwd -P)
if [[ -d "$assets" ]]; then assets=$(cd "$assets" && pwd -P); fi
export PATH="/opt/homebrew/opt/llvm@21/bin:/opt/homebrew/opt/lld@21/bin:$HOME/.cargo/bin:$PATH"
if [[ $quick == 0 ]]; then
  dxc="$assets/tools/dxc/pkg/bin/x64/dxc.exe"
  [[ -f "$dxc" ]] || { printf 'DXC fixture not found: %s\n' "$dxc" >&2; exit 2; }
fi
command -v jq >/dev/null

tmp=$(mktemp -d "${TMPDIR:-/tmp}/metal-suite.XXXXXXXX")
prefix="$tmp/prefix"
wine="$tmp/wine/bin/wine64"
server="$tmp/wine/bin/wineserver"
cleanup() {
  if [[ -d "$prefix" && -x "$server" ]]; then
    WINEPREFIX="$prefix" arch -x86_64 "$server" -k >/dev/null 2>&1 || true
    WINEPREFIX="$prefix" /usr/bin/perl -e 'alarm shift; exec @ARGV or die "$!\n"' \
      5 arch -x86_64 "$server" -w >/dev/null 2>&1 || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
logs="$tmp/logs"
if [[ -n "${METAL_SUITE_LOG_DIR:-}" ]]; then
  mkdir -p "$METAL_SUITE_LOG_DIR"
  logs=$(mktemp -d "$METAL_SUITE_LOG_DIR/metal-suite.XXXXXXXX")
  logs=$(cd "$logs" && pwd -P)
  printf 'Logs: %s\n' "$logs"
else
  mkdir "$logs"
fi
cp -cR "$install" "$tmp/wine"
if [[ -n "${METAL_SUITE_PREFIX_SEED:-}" ]]; then
  [[ -f "$METAL_SUITE_PREFIX_SEED/system.reg" && -d "$METAL_SUITE_PREFIX_SEED/drive_c/windows" ]] ||
    { printf 'METAL_SUITE_PREFIX_SEED is not an initialized prefix.\n' >&2; exit 2; }
  cp -cR "$(cd "$METAL_SUITE_PREFIX_SEED" && pwd -P)" "$prefix"
fi

export WINEPREFIX="$prefix" WINEDEBUG=-all MTL_DEBUG_LAYER=1 METAL_DEVICE_WRAPPER_TYPE=1
export WINEDLLOVERRIDES='mscoree,mshtml=;winemenubuilder.exe='
export WINEDLLPATH="$tmp/wine/lib/wine/x86_64-unix:$tmp/wine/lib/wine/x86_64-windows"
unset WINELOADER WINESERVER WINEARCH
target=x86_64-pc-windows-gnullvm
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/target}"
mkdir -p "$CARGO_TARGET_DIR"
export CARGO_TARGET_DIR
CARGO_TARGET_DIR=$(cd "$CARGO_TARGET_DIR" && pwd -P)
bin="$CARGO_TARGET_DIR/$target/debug"
fixtures="$prefix/drive_c/suite"
mkdir -p "$fixtures"

run_wine() {
  /usr/bin/perl -e 'alarm shift; exec @ARGV or die "$!\n"' 90 arch -x86_64 "$wine" "$@"
}

failed=0
printf '%-15s %-6s %s\n' Probe Status Detail
check() {
  local name="$1" pattern="$2" status detail rc=0
  shift 2
  "$@" >"$logs/$name.log" 2>&1 || rc=$?
  detail=$(grep -E "$pattern" "$logs/$name.log" | tail -1 || true)
  if (( rc == 0 )) && [[ -n "$detail" ]]; then
    status=PASS
  else
    if (( rc == 0 )); then
      detail='expected validation output missing'
    else
      detail=$(grep -Ei 'error|failed|panicked|assertion' "$logs/$name.log" | tail -1 || true)
      detail="exit $rc${detail:+: $detail}"
    fi
    status=FAIL
    failed=1
  fi
  printf '%-15s %-6s %s\n' "$name" "$status" "$detail"
}

cd "$root"
bins=(--bin victory-metal-smoke --bin victory-metal-events --bin victory-metal-blit --bin present)
if [[ $quick == 0 ]]; then
  bins+=(--bin victory-metal-chunks --bin victory-metal-textures --bin victory-metal-bind
    --bin stage_in --bin rectangle_probe --bin render)
fi
build() {
  cargo zigbuild --locked --target "$target" -p victory-metal "${bins[@]}" &&
    cargo-zigbuild test --no-run --message-format=json --locked --target "$target" -p victory-metal --lib >"$tmp/test.json" &&
    jq -er 'select(.reason == "compiler-artifact" and .profile.test and .target.kind == ["lib"]) | .executable' \
      "$tmp/test.json" >"$tmp/test.path" &&
    [[ -f "$(cat "$tmp/test.path")" ]] &&
    printf 'Windows probes and unit tests built\n'
}
check build 'Windows probes' build
if (( failed )); then tail -12 "$logs/build.log" >&2; exit 1; fi
test_exe=$(cat "$tmp/test.path")
cd "$tmp"
check smoke 'words written' run_wine "$bin/victory-metal-smoke.exe"
check events 'returned true' run_wine "$bin/victory-metal-events.exe"
check blit 'pixels differ' run_wine "$bin/victory-metal-blit.exe"
check present 'PASS:' run_wine "$bin/present.exe" --offscreen
check unit 'test result:' run_wine "$test_exe"
if (( quick )); then exit "$failed"; fi

compile() {
  local source="$1" entry="$2" profile="$3" out="$4" input="$4.hlsl"
  shift 4
  cp "$source" "$fixtures/$input" &&
    run_wine "$dxc" -E "$entry" -T "$profile" "$@" -Fo "C:\\suite\\$out" "C:\\suite\\$input" &&
    [[ -s "$fixtures/$out" ]]
}
stage_in() {
  local src="$assets/flash/F35/stage_in.hlsl"
  compile "$src" VSMain vs_6_6 stage-vs.dxil &&
    compile "$src" PSMain ps_6_6 stage-ps.dxil &&
    run_wine "$bin/stage_in.exe" 'C:\suite\stage-vs.dxil' 'C:\suite\stage-ps.dxil'
}
render() {
  local src="$assets/flash/F19/shader"
  compile "$src/fullscreen.hlsl" vs_main vs_6_0 fullscreen.dxil &&
    compile "$src/ps_const.hlsl" ps_main ps_6_0 const.dxil -O0 &&
    compile "$src/ps_sample.hlsl" ps_main ps_6_0 sample.dxil -O0 &&
    run_wine "$bin/render.exe" 'C:\suite\fullscreen.dxil' 'C:\suite\const.dxil' 'C:\suite\sample.dxil'
}
bind() {
  compile "$assets/tools/dxc/test/bind.hlsl" main cs_6_6 bind.dxil &&
    run_wine "$bin/victory-metal-bind.exe" 'C:\suite\bind.dxil'
}
indirect() {
  local src="$assets/flash/F42/shader.hlsl"
  compile "$src" vs_main vs_6_0 vs.dxil &&
    compile "$src" ps_main ps_6_0 ps.dxil &&
    F42_SHADER_DIR='C:\suite' run_wine "$test_exe" --ignored --exact indirect::tests::wine_gpu_readback --nocapture
}
rectangles() {
  local dir="$assets/flash/F37" s
  compile "$dir/rectangles.hlsl" VSMain vs_6_6 vs.dxil &&
    compile "$dir/rectangles.hlsl" GSMain gs_6_6 gs.dxil &&
    compile "$dir/rectangles.hlsl" PSMain ps_6_6 ps.dxil || return 1
  for s in A B C; do
    local lower
    lower=$(printf '%s' "$s" | tr A-Z a-z)
    compile "$dir/rect_real.hlsl" "VSMain$s" vs_6_6 "vs_$lower.dxil" &&
      compile "$dir/rect_real.hlsl" "PSMain$s" ps_6_6 "ps_$lower.dxil" &&
      cp "$dir/rect_gs_$lower.dxil" "$fixtures/" || return 1
  done
  cp "$dir/rect_gs_depth.dxil" "$fixtures/" &&
    run_wine "$bin/rectangle_probe.exe" --suite 'C:\suite'
}
render33() {
  local src="$assets/integration/f33-probe"
  # Rebuild the integration probe against this checkout, not its old worktree.
  mkdir "$tmp/f33" &&
    ln -s "$root" "$tmp/repo" &&
    ln -s "$src/src" "$tmp/f33/src" &&
    cp "$src/Cargo.lock" "$tmp/f33/Cargo.lock" || return 1
  cat >"$tmp/f33/Cargo.toml" <<'EOF'
[package]
name = "f33-integration-probe"
version = "0.1.0"
edition = "2024"
publish = false

[target.'cfg(windows)'.dependencies]
victory-metal = { path = "../repo/crates/victory-metal" }
EOF
  (cd "$root" && cargo zigbuild --locked --target "$target" --manifest-path "$tmp/f33/Cargo.toml") &&
    compile "$src/probe.hlsl" near_vs vs_6_6 near.dxil &&
    compile "$src/probe.hlsl" far_vs vs_6_6 far.dxil &&
    compile "$src/probe.hlsl" ps_main ps_6_6 f33-ps.dxil &&
    run_wine "$bin/f33-integration-probe.exe" 'C:\suite\near.dxil' 'C:\suite\far.dxil' 'C:\suite\f33-ps.dxil'
}
check stage_in 'PASS:' stage_in
check render33 'PASS:' render33
check rectangles 'ALL 5' rectangles
check indirect 'MSC indirect and direct draws matched' indirect
check chunks 'ALL CHECKS' run_wine "$bin/victory-metal-chunks.exe"
check render 'timing:' render
check bind 'outputs:' bind
check textures 'ALL CHECKS' run_wine "$bin/victory-metal-textures.exe"
exit "$failed"
