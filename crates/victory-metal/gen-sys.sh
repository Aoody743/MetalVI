#!/bin/bash
# Regenerates src/sys.rs from native/victorymetal/src/winemetal.h (x86_64 Windows ABI).
set -eu
cd "$(dirname "$0")"
export PATH="/opt/homebrew/opt/llvm@21/bin:$HOME/.cargo/bin:$PATH" LIBCLANG_PATH=/opt/homebrew/opt/llvm@21/lib
bindgen ../../native/victorymetal/src/winemetal.h --allowlist-file '.*winemetal\.h' \
  --no-layout-tests --use-core --default-enum-style newtype --no-doc-comments \
  -o src/sys.rs.tmp -- --target=x86_64-pc-windows-gnu -x c -std=c11
python3 -I gen-dyn.py src/sys.rs.tmp src/sys.rs
rm src/sys.rs.tmp
