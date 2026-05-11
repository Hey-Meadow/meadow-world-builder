#!/usr/bin/env bash
# Pipeline: ply -> 36 rotated copies -> qlmanage PNGs -> optimized GIF.
# Args: <input.ply> <output.gif> [n_frames=36] [size=320]
set -eu
PY=/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python
INP="$1"
OUT_GIF="$2"
N="${3:-36}"
SIZE="${4:-320}"

stem=$(basename "$INP" .ply)
WORK=$(mktemp -d "/tmp/qlgif_${stem}_XXXXXX")
PLYDIR="$WORK/ply"
PNGDIR="$WORK/png"
mkdir -p "$PLYDIR" "$PNGDIR"

# 1) Generate N rotated plys (spin around Y axis).
echo "[$stem] generating $N rotated plys ..."
$PY /tmp/rotate_ply_series.py "$INP" "$PLYDIR" --n "$N" --axis y >/dev/null

# 2) Batch qlmanage thumbnail.
echo "[$stem] qlmanage rendering @ ${SIZE}px ..."
qlmanage -x -t -s "$SIZE" -o "$PNGDIR" "$PLYDIR"/*.ply >/dev/null 2>&1 || true
n_png=$(ls "$PNGDIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo "[$stem] qlmanage produced $n_png pngs"

# 3) Stitch PNGs into optimized GIF.
$PY /tmp/pngs_to_gif.py "$PNGDIR" "$OUT_GIF" 80

# 4) Cleanup.
rm -rf "$WORK"
echo "[$stem] $(ls -la "$OUT_GIF" | awk '{print $5}') bytes -> $OUT_GIF"
