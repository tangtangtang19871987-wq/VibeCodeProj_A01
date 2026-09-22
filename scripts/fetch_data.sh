#!/usr/bin/env bash
# Fetch the public ICCAD 2013 lithography assets and verify their checksums.
#
# Assets: 24-term SOCS kernels (focus/defocus + conjugate variants), the ten M1
# contest layouts, and the contest process configuration. They are redistributed
# by the Apache-2.0 licensed OpenILT project; see docs/source_inventory.md S2.
set -euo pipefail

REPO_URL="https://github.com/OpenOPC/OpenILT"
PINNED_COMMIT="dabb97c6ca3dfd159362e48273c436444c77353b"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${GRIL_DATA_DIR:-$ROOT/data}"
WORK="$DEST/.openilt"

mkdir -p "$DEST"

if [ ! -d "$WORK/.git" ]; then
  echo "Cloning $REPO_URL at $PINNED_COMMIT ..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$REPO_URL" "$WORK"
fi
git -C "$WORK" fetch --depth 1 origin "$PINNED_COMMIT" 2>/dev/null || true
git -C "$WORK" checkout -q "$PINNED_COMMIT" 2>/dev/null || {
  echo "WARNING: could not check out the pinned commit; using the default branch." >&2
}

mkdir -p "$DEST/kernel/kernels" "$DEST/kernel/scales" "$DEST/benchmark/ICCAD2013" "$DEST/config"
cp "$WORK"/kernel/kernels/*.pt        "$DEST/kernel/kernels/"
cp "$WORK"/kernel/scales/*.pt         "$DEST/kernel/scales/"
cp "$WORK"/benchmark/ICCAD2013/*.glp  "$DEST/benchmark/ICCAD2013/"
cp "$WORK"/config/lithoiccad13.txt    "$DEST/config/"

echo "Verifying checksums ..."
( cd "$DEST" && sha256sum -c "$ROOT/data/CHECKSUMS.sha256" )
echo "OK. Assets are in $DEST"
