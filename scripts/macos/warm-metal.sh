#!/usr/bin/env bash
# Pre-warm Metal shader caches before launching the title:
#   1) Fast cold-cache check (Layer 1: VSH1 Metal-mode translations;
#      Layer 2: F46 MSC .vmsc outputs via victory-warm-metal.exe), printing a
#      direct reminder whenever either layer is cold (even with VICTORY_WARM=0).
#   2) Automatic MSC .vmsc warm-up on all logical cores with progress output
#      and a wall-clock timeout so launch never blocks forever.
#   3) Optional VSH1 Metal-mode warm-up when VICTORY_WARM_VSH1=1 and game_dir
#      is provided (runs `victory.exe warm --metal --no-pipelines`).
#
# Can be sourced by a launcher (`metalvi_warm_metal`) or run directly:
#   scripts/macos/warm-metal.sh [options]
#
# Environment variables:
#   VICTORY_WARM=0|1|force    0/off/false skips warm-up (still warns if cold);
#                             force ignores stamps and warms again.
#   VICTORY_WARM_JOBS=N       Worker threads (default: all logical CPUs).
#   VICTORY_WARM_TIMEOUT=S    Max seconds per Wine step before non-fatal timeout
#                             (default: 300; 0=none).
#   VICTORY_WARM_PROGRESS=N   Progress line interval in shaders (default: 1000).
#   VICTORY_WARM_BIN=PATH     Explicit path to victory-warm-metal.exe.
#   VICTORY_WARM_BUILD=1      Auto-build victory-warm-metal.exe if missing (default: 0).
#   VICTORY_WARM_STRICT=1     Exit non-zero on missing binary/timeout/failure (default: 0).
#   VICTORY_WARM_VSH1=1       Also run `victory.exe warm --metal --no-pipelines` if VSH1 is cold.
#   VICTORY_WARM_GAME_DIR=DIR Game directory for VICTORY_WARM_VSH1=1.
#   VICTORY_WARM_PREFIX=DIR   Optional Wine prefix override for warm-up.
#   VICTORY_CACHE_DIR         Windows (Z:\...) or Unix cache root.
#   VICTORY_METAL_CACHE       Windows (Z:\...) or Unix Metal .vmsc cache directory.
#   WINE_ROOT / WINE_BIN      Wine installation root or wine/wine64 executable.

_mw_from_wine_z() {
  local p="${1:-}"
  case "$p" in
    [Zz]:\\*|[Zz]:/*)
      p="${p#?:}"
      printf '%s\n' "${p//\\//}"
      ;;
    *)
      printf '%s\n' "$p"
      ;;
  esac
}

_mw_to_wine_z() {
  local p="${1:-}"
  case "$p" in
    [A-Za-z]:\\*|[A-Za-z]:/*)
      printf '%s\n' "${p//\//\\}"
      ;;
    /*)
      printf 'Z:%s\n' "${p//\//\\}"
      ;;
    *)
      local abs
      abs="$(cd "$p" 2>/dev/null && pwd -P)" || abs="$p"
      printf 'Z:%s\n' "${abs//\//\\}"
      ;;
  esac
}

_mw_cpu_count() {
  local n=""
  if command -v sysctl >/dev/null 2>&1; then
    n="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
  elif command -v nproc >/dev/null 2>&1; then
    n="$(nproc 2>/dev/null || true)"
  fi
  case "$n" in
    ''|*[!0-9]*|0) printf '4\n' ;;
    *) printf '%s\n' "$n" ;;
  esac
}

_mw_now_ms() {
  /usr/bin/perl -MTime::HiRes=time -e 'printf "%.0f\n", time() * 1000' 2>/dev/null || printf '%s000\n' "$(date +%s)"
}

# Fast directory-level signature over shader_dir and its immediate buckets
# (~10-25 ms for 32k-80k files in 256 buckets).
_mw_shader_dir_sig() {
  local dir="$1"
  local tmp_out
  tmp_out="$(
    cd "$dir" 2>/dev/null || exit 1
    {
      stat -f '%Fm:%l:.' .
      while IFS= read -r sub; do
        [[ -n "$sub" ]] && stat -f '%Fm:%l:%N' "$sub"
      done < <(find . -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | LC_ALL=C sort)
    } | cksum
  )" || return 1
  printf '%s\n' "$tmp_out"
}

# Fast source signature matching `victory_gpu::shader_cache::build()`.
_mw_translator_sig() {
  local repo_root="$1"
  local shader_src="$repo_root/crates/victory-shader/src"
  local sc_file="$repo_root/crates/victory-gpu/src/shader_cache.rs"
  if [[ ! -d "$shader_src" && ! -f "$sc_file" ]]; then
    printf 'external\n'
    return 0
  fi
  (
    cd "$repo_root" 2>/dev/null || exit 1
    {
      [[ -f "crates/victory-gpu/Cargo.toml" ]] && cksum "crates/victory-gpu/Cargo.toml"
      [[ -f "crates/victory-gpu/src/shader_cache.rs" ]] && cksum "crates/victory-gpu/src/shader_cache.rs"
      if [[ -d "crates/victory-shader/src" ]]; then
        while IFS= read -r f; do
          [[ -n "$f" ]] && cksum "$f"
        done < <(find "crates/victory-shader/src" -type f -name '*.rs' 2>/dev/null | LC_ALL=C sort)
      fi
    } | cksum
  ) 2>/dev/null || printf 'unknown\n'
}

_mw_env_sig() {
  local mode="$1"
  printf '%s|w=%s|hs=%s|ds=%s|ob=%s|wl=%s|f16=%s|sj=%s|mo=%s|gs=%s|so=%s|ll=%s\n' \
    "$mode" \
    "${VICTORY_WAVE_WIDE:-}" \
    "${VICTORY_WAVE_HALF_SYNC:-}" \
    "${VICTORY_WAVE_DEEP_SKIPS:-}" \
    "${VICTORY_WAVE_OWN_BITS:-}" \
    "${VICTORY_WAVE_LOCAL:-}" \
    "${VICTORY_F16_NEAREST:-}" \
    "${VICTORY_STRICT_JOINS:-}" \
    "${VICTORY_MESH_OUTPUT_STRICT:-}" \
    "${VICTORY_GROUP_SYNC:-}" \
    "${VICTORY_SHADER_OPT:-}" \
    "${VICTORY_SHADER_LOOP_LIMIT:-}"
}

# Locate victorymetal.dll / victorymetal.so and libvictorymsc.dylib, printing
# "<dll_ok> <msc_sig>" (dll_ok is 1 or 0).
_mw_find_victorymetal() {
  local wine_root="$1"
  local dll="" so="" msc=""
  local -a search_dirs=()
  if [[ -n "${WINEDLLPATH:-}" ]]; then
    local IFS=':'
    local d
    for d in $WINEDLLPATH; do
      [[ -n "$d" ]] && search_dirs+=("$d" "$d/x86_64-windows" "$d/x86_64-unix")
    done
  fi
  if [[ -n "$wine_root" ]]; then
    search_dirs+=(
      "$wine_root/lib/wine/x86_64-windows"
      "$wine_root/lib/wine/x86_64-unix"
      "$wine_root/lib/wine"
    )
  fi
  local d
  for d in "${search_dirs[@]}"; do
    [[ -z "$dll" && -f "$d/victorymetal.dll" ]] && dll="$d/victorymetal.dll"
    [[ -z "$so" && -f "$d/victorymetal.so" ]] && so="$d/victorymetal.so"
    [[ -z "$msc" && -f "$d/libvictorymsc.dylib" ]] && msc="$d/libvictorymsc.dylib"
  done
  local ok=0 msc_sig="none"
  if [[ -n "$dll" && -n "$so" ]]; then
    ok=1
  fi
  if [[ -n "$msc" ]]; then
    msc_sig="$(stat -f '%Fm:%z' "$msc" 2>/dev/null || printf 'none')"
  fi
  printf '%s %s\n' "$ok" "$msc_sig"
}

_mw_find_wine_bin() {
  local wine_root="$1"
  if [[ -n "${WINE_BIN:-}" && -x "${WINE_BIN:-}" ]]; then
    printf '%s\n' "$WINE_BIN"
    return 0
  fi
  if [[ -n "$wine_root" ]]; then
    if [[ -x "$wine_root/bin/wine" ]]; then
      printf '%s\n' "$wine_root/bin/wine"
      return 0
    fi
    if [[ -x "$wine_root/bin/wine64" ]]; then
      printf '%s\n' "$wine_root/bin/wine64"
      return 0
    fi
  fi
  return 1
}

_mw_find_warm_bin() {
  local repo_root="$1"
  local exe_hint="${2:-}"
  if [[ -n "${VICTORY_WARM_BIN:-}" ]]; then
    if [[ -f "$VICTORY_WARM_BIN" ]]; then
      printf '%s\n' "$VICTORY_WARM_BIN"
      return 0
    fi
    return 1
  fi

  local -a candidates=()
  if [[ -n "$exe_hint" ]]; then
    local hint_unix
    hint_unix="$(_mw_from_wine_z "$exe_hint")"
    if [[ -e "$hint_unix" ]]; then
      candidates+=("$(dirname "$hint_unix")/victory-warm-metal.exe")
    fi
  fi
  local target_base="${CARGO_TARGET_DIR:-$repo_root/target}"
  local profile
  for profile in release diag debug; do
    candidates+=("$target_base/x86_64-pc-windows-gnullvm/$profile/victory-warm-metal.exe")
    if [[ "$target_base" != "$repo_root/target" ]]; then
      candidates+=("$repo_root/target/x86_64-pc-windows-gnullvm/$profile/victory-warm-metal.exe")
    fi
  done

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  return 1
}

_mw_find_victory_exe() {
  local repo_root="$1"
  local exe_hint="${2:-}"
  if [[ -n "$exe_hint" ]]; then
    local hint_unix
    hint_unix="$(_mw_from_wine_z "$exe_hint")"
    if [[ -f "$hint_unix" && "$(basename "$hint_unix")" == "victory.exe" ]]; then
      printf '%s\n' "$hint_unix"
      return 0
    fi
  fi
  local target_base="${CARGO_TARGET_DIR:-$repo_root/target}"
  local profile
  for profile in release diag debug; do
    if [[ -f "$target_base/x86_64-pc-windows-gnullvm/$profile/victory.exe" ]]; then
      printf '%s\n' "$target_base/x86_64-pc-windows-gnullvm/$profile/victory.exe"
      return 0
    fi
    if [[ -f "$repo_root/target/x86_64-pc-windows-gnullvm/$profile/victory.exe" ]]; then
      printf '%s\n' "$repo_root/target/x86_64-pc-windows-gnullvm/$profile/victory.exe"
      return 0
    fi
  done
  return 1
}

_mw_read_stamp_field() {
  local stamp_file="$1" key="$2"
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      "${key}="*)
        printf '%s\n' "${line#*=}"
        return 0
        ;;
    esac
  done <"$stamp_file"
  return 1
}

_mw_count_vmsc_entries() {
  local metal_dir="$1"
  local nlink
  nlink="$(stat -f '%l' "$metal_dir" 2>/dev/null || printf '0')"
  if [[ "$nlink" =~ ^[0-9]+$ ]] && (( nlink >= 2 )); then
    # On APFS/HFS+, directory st_nlink equals 2 + direct entries; subtract known non-.vmsc entries.
    local non_vmsc=0
    [[ -f "$metal_dir/.warm-metal.stamp" ]] && non_vmsc=$(( non_vmsc + 1 ))
    [[ -d "$metal_dir/refused" ]] && non_vmsc=$(( non_vmsc + 1 ))
    local est=$(( nlink - 2 - non_vmsc ))
    (( est < 0 )) && est=0
    printf '%s\n' "$est"
    return 0
  fi
  find "$metal_dir" -maxdepth 1 -type f -name '*.vmsc' 2>/dev/null | wc -l | tr -d ' '
}

# Bounded wall-clock runner with clean INT/TERM/ALRM forwarding.
# Exit code: 124 on timeout, 130 on SIGINT, 143 on SIGTERM, else child exit code.
_mw_run_bounded() {
  local timeout_secs="$1"
  shift
  if [[ "$timeout_secs" -le 0 ]]; then
    "$@"
    return $?
  fi
  /usr/bin/perl -e '
    use strict;
    use warnings;
    my $t = shift @ARGV;
    my $pid = fork();
    die "fork failed: $!\n" unless defined $pid;
    if ($pid == 0) {
      exec @ARGV or die "exec failed: $!\n";
    }
    my $timed_out = 0;
    local $SIG{ALRM} = sub {
      $timed_out = 1;
      kill "TERM", $pid;
      select(undef, undef, undef, 1.0);
      kill "KILL", $pid;
    };
    local $SIG{INT} = sub {
      kill "TERM", $pid;
      select(undef, undef, undef, 0.2);
      kill "KILL", $pid;
      waitpid($pid, 0);
      exit 130;
    };
    local $SIG{TERM} = sub {
      kill "TERM", $pid;
      select(undef, undef, undef, 0.2);
      kill "KILL", $pid;
      waitpid($pid, 0);
      exit 143;
    };
    alarm($t);
    waitpid($pid, 0);
    my $st = $?;
    alarm(0);
    exit 124 if $timed_out;
    if ($st & 127) {
      exit(128 + ($st & 127));
    }
    exit($st >> 8);
  ' "$timeout_secs" "$@"
}

_mw_usage() {
  cat <<'EOF'
Usage: warm-metal.sh [options]
  --shader-cache DIR   Shader cache directory (VSH1 files)
  --metal-cache DIR    Metal MSC cache directory (.vmsc output)
  --bin EXE            Path to victory-warm-metal.exe
  --exe-hint EXE       Path to victory.exe (checks sibling directory)
  --game-dir DIR       Optional game directory (for VICTORY_WARM_VSH1=1)
  --wine DIR           Wine installation root
  --prefix DIR         Wine prefix directory
  --jobs N             Worker threads (default: all logical CPUs)
  --timeout SEC        Max wait seconds before non-fatal skip (default: 300, 0=none)
  --progress N         Progress print interval (default: 1000)
  --check              Check cold/warm status only (exit 1 if cold, 0 if warm)
  --force              Ignore stamps and run warm-up anyway
  --strict             Exit non-zero if binary/Wine is missing or warm-up fails
  --dry-run            Check cold/warm status and print planned commands only
  -h, --help           Show this help
EOF
}

# Check VSH1 + MSC cache status. Prints COLD reminders directly to stderr.
# Sets global _MW_VSH1_COLD (0|1) and _MW_MSC_COLD (0|1).
_mw_check_cold_status() {
  local repo_root="$1"
  local cache_root="$2"
  local shader_dir="$3"
  local metal_dir="$4"
  local msc_sig="$5"
  local exe_hint="$6"

  _MW_VSH1_COLD=0
  _MW_MSC_COLD=0
  _MW_VSH1_REASON=""
  _MW_MSC_REASON=""
  _MW_SHADER_SIG=""

  local has_vsh1=0
  if [[ -d "$shader_dir" ]] && find "$shader_dir" -maxdepth 2 -type f ! -name '.*' -print -quit 2>/dev/null | grep -q .; then
    has_vsh1=1
    _MW_SHADER_SIG="$(_mw_shader_dir_sig "$shader_dir" || true)"
  fi

  local src_sig env_sig vsh1_stamp victory_exe exe_sig="none"
  src_sig="$(_mw_translator_sig "$repo_root")"
  env_sig="$(_mw_env_sig "metal")"
  if victory_exe="$(_mw_find_victory_exe "$repo_root" "$exe_hint")"; then
    exe_sig="$(stat -f '%Fm:%z' "$victory_exe" 2>/dev/null || printf 'none')"
  fi
  vsh1_stamp="${cache_root:-$shader_dir}/.warm-metal.stamp"

  if (( has_vsh1 == 0 )); then
    _MW_VSH1_COLD=1
    _MW_VSH1_REASON="shader cache is empty or missing at $shader_dir"
  elif [[ ! -f "$vsh1_stamp" ]]; then
    _MW_VSH1_COLD=1
    _MW_VSH1_REASON="no Metal VSH1 warm stamp ($vsh1_stamp); Metal keys differ (wave=32, no image_atomic64, no attributeAtVertex)"
  else
    local sv ss se sx
    sv="$(_mw_read_stamp_field "$vsh1_stamp" stamp_version || true)"
    ss="$(_mw_read_stamp_field "$vsh1_stamp" src_sig || true)"
    se="$(_mw_read_stamp_field "$vsh1_stamp" env_sig || true)"
    sx="$(_mw_read_stamp_field "$vsh1_stamp" exe_sig || true)"
    if [[ "$sv" != "1" ]]; then
      _MW_VSH1_COLD=1
      _MW_VSH1_REASON="outdated Metal VSH1 warm stamp"
    elif [[ "$src_sig" != "external" && "$ss" != "$src_sig" ]]; then
      _MW_VSH1_COLD=1
      _MW_VSH1_REASON="translator sources changed since last Metal warm ($ss -> $src_sig)"
    elif [[ "$se" != "$env_sig" ]]; then
      _MW_VSH1_COLD=1
      _MW_VSH1_REASON="shader environment switches changed since last Metal warm"
    elif [[ "$exe_sig" != "none" && "$sx" != "none" && "$sx" != "$exe_sig" ]]; then
      _MW_VSH1_COLD=1
      _MW_VSH1_REASON="victory.exe changed since last Metal warm ($sx -> $exe_sig)"
    fi
  fi

  local msc_stamp="$metal_dir/.warm-metal.stamp"
  local vmsc_now=0
  if [[ -d "$metal_dir" ]]; then
    vmsc_now="$(_mw_count_vmsc_entries "$metal_dir")"
  fi

  if [[ ! -d "$metal_dir" ]] || (( vmsc_now == 0 )); then
    _MW_MSC_COLD=1
    _MW_MSC_REASON="metal-cache is empty (0 .vmsc files in $metal_dir)"
  elif [[ ! -f "$msc_stamp" ]]; then
    _MW_MSC_COLD=1
    _MW_MSC_REASON="no MSC warm stamp in $metal_dir (currently $vmsc_now entries)"
  else
    local mv ms mm mc
    mv="$(_mw_read_stamp_field "$msc_stamp" stamp_version || true)"
    ms="$(_mw_read_stamp_field "$msc_stamp" shader_sig || true)"
    mm="$(_mw_read_stamp_field "$msc_stamp" msc_sig || true)"
    mc="$(_mw_read_stamp_field "$msc_stamp" vmsc_count || true)"
    if [[ "$mv" != "1" ]]; then
      _MW_MSC_COLD=1
      _MW_MSC_REASON="outdated MSC warm stamp"
    elif [[ -n "$_MW_SHADER_SIG" && "$ms" != "$_MW_SHADER_SIG" ]]; then
      _MW_MSC_COLD=1
      _MW_MSC_REASON="shader cache updated since last MSC warm"
    elif [[ "$mm" != "$msc_sig" ]]; then
      _MW_MSC_COLD=1
      _MW_MSC_REASON="MSC/victorymetal binary changed since last MSC warm ($mm -> $msc_sig)"
    elif [[ ! "$mc" =~ ^[0-9]+$ ]] || (( vmsc_now < mc )); then
      _MW_MSC_COLD=1
      _MW_MSC_REASON="metal-cache entries missing on disk ($vmsc_now < $mc)"
    fi
  fi

  _MW_SRC_SIG="$src_sig"
  _MW_ENV_SIG="$env_sig"
  _MW_EXE_SIG="$exe_sig"
  _MW_VSH1_STAMP="$vsh1_stamp"
  _MW_MSC_STAMP="$msc_stamp"
  _MW_VMSC_NOW="$vmsc_now"
  _MW_HAS_VSH1="$has_vsh1"
}

metalvi_warm_metal() {
  local script_dir repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  repo_root="$(cd "$script_dir/../.." && pwd -P)"

  local mode="${VICTORY_WARM:-1}"
  local jobs="${VICTORY_WARM_JOBS:-$(_mw_cpu_count)}"
  local timeout_secs="${VICTORY_WARM_TIMEOUT:-300}"
  local progress="${VICTORY_WARM_PROGRESS:-1000}"
  local strict="${VICTORY_WARM_STRICT:-0}"
  local warm_vsh1="${VICTORY_WARM_VSH1:-0}"
  local game_dir="${VICTORY_WARM_GAME_DIR:-${METALVI_GAME_DIR:-}}"
  local dry_run=0
  local check_only=0
  local shader_dir="${VICTORY_WARM_SHADER_CACHE:-${METALVI_SHADER_CACHE:-}}"
  local metal_dir="${VICTORY_WARM_METAL_CACHE:-${METALVI_METAL_CACHE:-}}"
  local wine_root="${WINE_ROOT:-}"
  local wine_prefix="${VICTORY_WARM_PREFIX:-${WINEPREFIX:-}}"
  local warm_bin="${VICTORY_WARM_BIN:-}"
  local exe_hint=""

  while (( $# > 0 )); do
    case "$1" in
      --shader-cache) shader_dir="${2:?missing value for --shader-cache}"; shift 2 ;;
      --metal-cache) metal_dir="${2:?missing value for --metal-cache}"; shift 2 ;;
      --bin) warm_bin="${2:?missing value for --bin}"; shift 2 ;;
      --exe-hint) exe_hint="${2:?missing value for --exe-hint}"; shift 2 ;;
      --game-dir) game_dir="${2:?missing value for --game-dir}"; shift 2 ;;
      --wine) wine_root="${2:?missing value for --wine}"; shift 2 ;;
      --prefix) wine_prefix="${2:?missing value for --prefix}"; shift 2 ;;
      --jobs|-j) jobs="${2:?missing value for --jobs}"; shift 2 ;;
      --timeout) timeout_secs="${2:?missing value for --timeout}"; shift 2 ;;
      --progress) progress="${2:?missing value for --progress}"; shift 2 ;;
      --check) check_only=1; shift ;;
      --force) mode="force"; shift ;;
      --strict) strict=1; shift ;;
      --dry-run) dry_run=1; shift ;;
      -h|--help) _mw_usage; return 0 ;;
      *) printf '[warm-metal] unknown option: %s\n' "$1" >&2; return 2 ;;
    esac
  done

  local mode_lc
  mode_lc="$(printf '%s' "$mode" | tr 'A-Z' 'a-z')"
  local t_start_ms
  t_start_ms="$(_mw_now_ms)"

  local cache_root=""
  if [[ -n "${VICTORY_CACHE_DIR:-}" ]]; then
    cache_root="$(_mw_from_wine_z "$VICTORY_CACHE_DIR")"
  elif [[ -n "${METALVI_CACHE_DIR:-}" ]]; then
    cache_root="$METALVI_CACHE_DIR"
  fi
  if [[ -z "$shader_dir" && -n "$cache_root" ]]; then
    shader_dir="$cache_root/shaders"
  fi
  if [[ -z "$metal_dir" ]]; then
    if [[ -n "${VICTORY_METAL_CACHE:-}" ]]; then
      metal_dir="$(_mw_from_wine_z "$VICTORY_METAL_CACHE")"
    elif [[ -n "$cache_root" ]]; then
      metal_dir="$cache_root/metal"
    fi
  fi

  if [[ -z "$shader_dir" || -z "$metal_dir" ]]; then
    printf '[warm-metal] cache directory not configured; skipping\n' >&2
    [[ "$strict" == "1" ]] && return 1 || return 0
  fi

  local vm_info dll_ok msc_sig
  vm_info="$(_mw_find_victorymetal "$wine_root")"
  dll_ok="${vm_info%% *}"
  msc_sig="${vm_info#* }"

  _mw_check_cold_status "$repo_root" "$cache_root" "$shader_dir" "$metal_dir" "$msc_sig" "$exe_hint"

  if (( _MW_VSH1_COLD == 1 )); then
    printf '[warm-metal] COLD (VSH1 metal shaders): %s\n' "$_MW_VSH1_REASON" >&2
  fi
  if (( _MW_MSC_COLD == 1 )); then
    printf '[warm-metal] COLD (MSC .vmsc cache): %s\n' "$_MW_MSC_REASON" >&2
  fi

  if (( check_only == 1 )); then
    local elapsed_ms=$(( $(_mw_now_ms) - t_start_ms ))
    if (( _MW_VSH1_COLD == 0 && _MW_MSC_COLD == 0 )); then
      printf '[warm-metal] WARM (VSH1+MSC ready, vmsc=%s, checked in %sms)\n' "$_MW_VMSC_NOW" "$elapsed_ms" >&2
      return 0
    fi
    printf '[warm-metal] REMINDER: Metal cache is COLD; run scripts/macos/warm-metal.sh (and `victory.exe warm --metal` if VSH1 is cold)\n' >&2
    return 1
  fi

  case "$mode_lc" in
    0|off|false|no)
      if (( _MW_VSH1_COLD == 1 || _MW_MSC_COLD == 1 )); then
        printf '[warm-metal] REMINDER: cache is COLD, skipped because VICTORY_WARM=%s\n' "$mode" >&2
      fi
      return 0
      ;;
  esac

  if [[ "$mode_lc" != "force" ]] && (( _MW_VSH1_COLD == 0 && _MW_MSC_COLD == 0 )); then
    local elapsed_ms=$(( $(_mw_now_ms) - t_start_ms ))
    printf '[warm-metal] cache up-to-date (vmsc=%s, checked in %sms)\n' "$_MW_VMSC_NOW" "$elapsed_ms" >&2
    return 0
  fi

  local wine_bin=""
  if ! wine_bin="$(_mw_find_wine_bin "$wine_root")"; then
    printf '[warm-metal] Wine binary not found (WINE_ROOT=%s); skipping warm-up\n' "${wine_root:-unset}" >&2
    [[ "$strict" == "1" ]] && return 1 || return 0
  fi

  # Optional Phase 1: warm VSH1 Metal translations via `victory.exe warm --metal --no-pipelines`
  if [[ "$warm_vsh1" == "1" ]] && [[ "$mode_lc" == "force" || "$_MW_VSH1_COLD" == "1" ]]; then
    local victory_exe=""
    if victory_exe="$(_mw_find_victory_exe "$repo_root" "$exe_hint")" && [[ -n "$game_dir" && -d "$game_dir" ]]; then
      local win_victory win_game
      win_victory="$(_mw_to_wine_z "$victory_exe")"
      win_game="$(_mw_to_wine_z "$game_dir")"
      if [[ "$dry_run" == "1" ]]; then
        printf '[warm-metal] DRY-RUN Phase 1 (Metal VSH1): VICTORY_METAL=compute %s %s warm --metal --no-pipelines --jobs %s %s\n' \
          "$wine_bin" "$win_victory" "$jobs" "$win_game" >&2
      else
        printf '[warm-metal] Phase 1: warming Metal VSH1 shader translations (%s, jobs=%s)...\n' "$game_dir" "$jobs" >&2
        local rc1=0
        (
          [[ -n "$wine_prefix" ]] && export WINEPREFIX="$wine_prefix"
          export WINEDEBUG="${WINEDEBUG:--all}"
          export ROSETTA_ADVERTISE_AVX=1
          export VICTORY_METAL=compute
          export VICTORY_CACHE_DIR="${VICTORY_CACHE_DIR:-$(_mw_to_wine_z "${cache_root:-$(dirname "$shader_dir")}")}"
          if [[ -n "${WINEDLLOVERRIDES:-}" ]]; then
            case ";$WINEDLLOVERRIDES;" in
              *";winemenubuilder.exe="* ) export WINEDLLOVERRIDES ;;
              * ) export WINEDLLOVERRIDES="$WINEDLLOVERRIDES;winemenubuilder.exe=" ;;
            esac
          else
            export WINEDLLOVERRIDES="mscoree,mshtml=;winemenubuilder.exe="
          fi
          unset DISPLAY
          _mw_run_bounded "$timeout_secs" \
            "$wine_bin" "$win_victory" warm --no-pipelines --jobs "$jobs" "$win_game"
        ) || rc1=$?
        if (( rc1 == 0 )); then
          mkdir -p "$(dirname "$_MW_VSH1_STAMP")"
          local st1_tmp="${_MW_VSH1_STAMP}.tmp.$$"
          {
            printf 'stamp_version=1\n'
            printf 'mode=metal\n'
            printf 'src_sig=%s\n' "$_MW_SRC_SIG"
            printf 'env_sig=%s\n' "$_MW_ENV_SIG"
            printf 'exe_sig=%s\n' "$_MW_EXE_SIG"
            printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
          } >"$st1_tmp" && mv -f "$st1_tmp" "$_MW_VSH1_STAMP"
          _MW_HAS_VSH1=1
          _MW_SHADER_SIG="$(_mw_shader_dir_sig "$shader_dir" || true)"
        else
          printf '[warm-metal] Phase 1 (Metal VSH1) exited with %s; continuing\n' "$rc1" >&2
          [[ "$strict" == "1" ]] && return "$rc1"
        fi
      fi
    elif [[ "$dry_run" == "0" ]]; then
      printf '[warm-metal] VICTORY_WARM_VSH1=1 set, but victory.exe or VICTORY_WARM_GAME_DIR missing; skipping Phase 1\n' >&2
    fi
  fi

  # Phase 2: warm MSC .vmsc cache via F46 victory-warm-metal.exe
  if [[ "$mode_lc" != "force" && "$_MW_MSC_COLD" == "0" ]]; then
    return 0
  fi

  if (( _MW_HAS_VSH1 == 0 )) && [[ "$dry_run" == "0" ]]; then
    printf '[warm-metal] shader cache at %s is empty; run `victory.exe warm --metal` (or `warm-t8.sh metal`) first\n' "$shader_dir" >&2
    [[ "$strict" == "1" ]] && return 1 || return 0
  fi

  mkdir -p "$metal_dir"

  if [[ "${VICTORY_WARM_SKIP_DLL_CHECK:-0}" != "1" && "$dll_ok" != "1" ]]; then
    printf '[warm-metal] victorymetal.dll/so not found in Wine tree (%s); skipping MSC warm-up\n' "${wine_root:-unset}" >&2
    [[ "$strict" == "1" ]] && return 1 || return 0
  fi

  if [[ -n "$warm_bin" ]]; then
    if [[ ! -f "$warm_bin" ]]; then
      printf '[warm-metal] victory-warm-metal.exe not found at %s; skipping warm-up\n' "$warm_bin" >&2
      [[ "$strict" == "1" ]] && return 1 || return 0
    fi
  else
    if ! warm_bin="$(_mw_find_warm_bin "$repo_root" "$exe_hint")"; then
      if [[ "${VICTORY_WARM_BUILD:-0}" == "1" ]] && command -v cargo >/dev/null 2>&1; then
        printf '[warm-metal] building victory-warm-metal.exe...\n' >&2
        (
          export PATH="/opt/homebrew/opt/llvm@21/bin:/opt/homebrew/opt/lld@21/bin:$HOME/.cargo/bin:$PATH"
          cd "$repo_root" && cargo zigbuild -p victory-gpu --bin victory-warm-metal --target x86_64-pc-windows-gnullvm --locked
        ) || true
        warm_bin="$(_mw_find_warm_bin "$repo_root" "$exe_hint" || true)"
      fi
      if [[ -z "$warm_bin" || ! -f "$warm_bin" ]]; then
        printf '[warm-metal] victory-warm-metal.exe not built; skipping (build with: cargo zigbuild -p victory-gpu --bin victory-warm-metal --target x86_64-pc-windows-gnullvm --locked)\n' >&2
        [[ "$strict" == "1" ]] && return 1 || return 0
      fi
    fi
  fi

  local win_shader_dir win_metal_dir win_warm_bin
  win_shader_dir="$(_mw_to_wine_z "$shader_dir")"
  win_metal_dir="$(_mw_to_wine_z "$metal_dir")"
  win_warm_bin="$(_mw_to_wine_z "$warm_bin")"

  if [[ "$dry_run" == "1" ]]; then
    printf '[warm-metal] DRY-RUN Phase 2 (MSC .vmsc): %s %s --shader-cache %s --metal-cache %s --jobs %s --progress %s (timeout=%ss)\n' \
      "$wine_bin" "$win_warm_bin" "$win_shader_dir" "$win_metal_dir" "$jobs" "$progress" "$timeout_secs" >&2
    return 0
  fi

  printf '[warm-metal] Phase 2: warming MSC .vmsc cache (%s -> %s, jobs=%s, timeout=%ss, VICTORY_WARM=0 to skip)...\n' \
    "$shader_dir" "$metal_dir" "$jobs" "$timeout_secs" >&2

  local log_tmp rc=0
  log_tmp="$(mktemp "${TMPDIR:-/tmp}/warm-metal.XXXXXX")"

  set +o pipefail
  (
    [[ -n "$wine_prefix" ]] && export WINEPREFIX="$wine_prefix"
    export WINEDEBUG="${WINEDEBUG:--all}"
    export ROSETTA_ADVERTISE_AVX=1
    if [[ -n "${WINEDLLOVERRIDES:-}" ]]; then
      case ";$WINEDLLOVERRIDES;" in
        *";winemenubuilder.exe="* ) export WINEDLLOVERRIDES ;;
        * ) export WINEDLLOVERRIDES="$WINEDLLOVERRIDES;winemenubuilder.exe=" ;;
      esac
    else
      export WINEDLLOVERRIDES="mscoree,mshtml=;winemenubuilder.exe="
    fi
    unset DISPLAY
    _mw_run_bounded "$timeout_secs" \
      "$wine_bin" "$win_warm_bin" \
      --shader-cache "$win_shader_dir" \
      --metal-cache "$win_metal_dir" \
      --jobs "$jobs" \
      --progress "$progress"
  ) 2>&1 | tee "$log_tmp" >&2
  rc="${PIPESTATUS[0]}"
  set -o pipefail

  if (( rc == 130 )); then
    rm -f "$log_tmp"
    printf '[warm-metal] interrupted by user\n' >&2
    return 130
  elif (( rc == 124 )); then
    rm -f "$log_tmp"
    printf '[warm-metal] timed out after %ss; continuing with partial .vmsc cache\n' "$timeout_secs" >&2
    [[ "$strict" == "1" ]] && return 124 || return 0
  elif (( rc != 0 )); then
    rm -f "$log_tmp"
    printf '[warm-metal] exited with code %s; continuing launch (set VICTORY_WARM_STRICT=1 to abort)\n' "$rc" >&2
    [[ "$strict" == "1" ]] && return "$rc" || return 0
  fi

  local out_vmsc="" out_scanned=""
  out_vmsc="$(grep -E '^output \.vmsc files[[:space:]]*:' "$log_tmp" | awk -F: '{gsub(/[^0-9]/,"",$2); print $2}' | tail -1 || true)"
  out_scanned="$(grep -E '^files scanned[[:space:]]*:' "$log_tmp" | awk -F: '{gsub(/[^0-9]/,"",$2); print $2}' | tail -1 || true)"
  rm -f "$log_tmp"

  if [[ -z "$out_vmsc" ]]; then
    out_vmsc="$(_mw_count_vmsc_entries "$metal_dir")"
  fi
  [[ -z "$out_scanned" ]] && out_scanned="0"

  _MW_SHADER_SIG="$(_mw_shader_dir_sig "$shader_dir" || printf 'none')"
  local stamp_tmp="${_MW_MSC_STAMP}.tmp.$$"
  {
    printf 'stamp_version=1\n'
    printf 'shader_sig=%s\n' "$_MW_SHADER_SIG"
    printf 'msc_sig=%s\n' "$msc_sig"
    printf 'vmsc_count=%s\n' "$out_vmsc"
    printf 'shader_files=%s\n' "$out_scanned"
    printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"$stamp_tmp" && mv -f "$stamp_tmp" "$_MW_MSC_STAMP"

  local total_ms=$(( $(_mw_now_ms) - t_start_ms ))
  printf '[warm-metal] ready (vmsc=%s, scanned=%s, total=%sms)\n' "$out_vmsc" "$out_scanned" "$total_ms" >&2
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  metalvi_warm_metal "$@"
fi
