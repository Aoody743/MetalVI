#!/bin/bash
# Pre-push guards for the MetalVI repository (F70).
#
# Runs four read-only checks in order and refuses to push if any fails:
#   1. asset policy      `cargo run -q -p xtask -- policy`
#   2. Windows target    `cargo check --workspace --target x86_64-pc-windows-gnullvm`
#   3. shader tests      `cargo test -p victory-shader --lib` (native macOS)
#   4. F58 quick suite   `scripts/macos/metal-suite.sh --quick`
#
# Never launches the game. The F58 suite copies its own Wine tree and keeps
# its private prefix under TMPDIR; it needs a Wine tree with victorymetal
# installed, passed via METAL_SUITE_WINE. If that variable is unset, step 4 is
# skipped with a warning rather than failing.
#
# Usage:
#   METAL_SUITE_WINE=/path/to/wine bash scripts/macos/pre-push.sh
#
# Environment overrides:
#   CARGO_TARGET_DIR  build dir (default: $REPO_ROOT/target/pre-push)
#   METAL_SUITE_WINE  Wine tree for the F58 suite (step 4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
TARGET_TRIPLE="x86_64-pc-windows-gnullvm"

# Project build convention (flash README, common rules): Homebrew llvm/lld on
# PATH alongside cargo, same as the F58 suite and the zigbuild builds.
export PATH="/opt/homebrew/opt/llvm@21/bin:/opt/homebrew/opt/lld@21/bin:$HOME/.cargo/bin:$PATH"

# Keep pre-push artifacts out of the shared game build; /target/ is ignored.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target/pre-push}"
mkdir -p "$CARGO_TARGET_DIR"

log="$(mktemp -t pre-push.XXXXXX)"
cleanup() { rm -f "$log"; }
trap cleanup EXIT INT TERM

failures=0
passed=0
skipped=0

# run_step <name> <command...>
run_step() {
  local name="$1"; shift
  local start end rc=0
  start=$(date +%s)
  if "$@" >"$log" 2>&1; then
    end=$(date +%s)
    printf 'PASS  %-22s %6.1f s\n' "$name" "$((end - start))"
    passed=$((passed + 1))
  else
    rc=$?
    end=$(date +%s)
    printf 'FAIL  %-22s %6.1f s (exit %d)\n' "$name" "$((end - start))" "$rc" >&2
    # Readable tail of the failing step for the developer.
    tail -20 "$log" >&2 || true
    failures=$((failures + 1))
  fi
}

printf 'pre-push checks (repo: %s)\n' "$ROOT"

# 1. Asset policy: allowlist over tracked + untracked-but-not-ignored files.
run_step "asset policy" cargo run -q -p xtask -- policy

# 2. Windows target type check, whole workspace, gnullvm triple.
run_step "windows check" cargo check --workspace --target "$TARGET_TRIPLE" --locked

# 3. victory-shader unit tests, native on macOS (no Windows APIs in this crate).
run_step "victory-shader tests" cargo test -p victory-shader --lib --locked

# 4. F58 quick synthetic suite. Skipped unless METAL_SUITE_WINE points at a
#    Wine tree with victorymetal installed; the suite itself creates and
#    tears down its own private prefix under TMPDIR.
if [[ -n "${METAL_SUITE_WINE:-}" ]]; then
  run_step "F58 quick suite" bash "$ROOT/scripts/macos/metal-suite.sh" --quick
else
  printf 'SKIP  %-22s %6s\n' "F58 quick suite" "n/a"
  printf '      (set METAL_SUITE_WINE to a victorymetal Wine tree to run it)\n' >&2
  skipped=$((skipped + 1))
fi

printf '\nsummary: %d passed, %d skipped, %d failed\n' "$passed" "$skipped" "$failures"
if (( failures > 0 )); then
  printf 'pre-push refused: fix the failing check(s) above before pushing.\n' >&2
  exit 1
fi
exit 0
