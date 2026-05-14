#!/usr/bin/env bash
# stitch_demo.sh — assembles GroceryGPT demo scenes into MP4 + GIF
#
# Usage: bash scripts/stitch_demo.sh
#
# Expects scene clips at docs/scenes/*.webm (or .webp / .mp4)
# and the title card at docs/scenes/title_card.gif
# Outputs:
#   docs/demo_full.mp4  — H.264, 1280×720, 24 fps
#   docs/demo_full.gif  — palette-optimised, 960px wide, 10 fps
#
# Requires: ffmpeg (brew install ffmpeg)

set -euo pipefail

SCENES_DIR="docs/scenes"
OUT_DIR="docs"
CONCAT_LIST="$SCENES_DIR/concat.txt"

echo "=== GroceryGPT Demo Stitcher ==="

# ── Locate scene files ──────────────────────────────────────────────────────────
# Ordered: title_card first, then scene_01..scene_10
# Accept .webm, .webp (video), .mp4, .mov
declare -a SCENE_FILES=()
for name in title_card scene_01 scene_02 scene_03 scene_04 scene_05 \
            scene_06 scene_07 scene_08 scene_09 scene_10; do
    for ext in webm webp mp4 mov gif; do
        f="$SCENES_DIR/${name}.${ext}"
        if [[ -f "$f" ]]; then
            SCENE_FILES+=("$f")
            echo "  Found: $f"
            break
        fi
    done
done

if [[ ${#SCENE_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No scene files found in $SCENES_DIR"
    exit 1
fi

echo ""
echo "Found ${#SCENE_FILES[@]} clips. Building concat list …"

# ── Build ffmpeg concat list ────────────────────────────────────────────────────
rm -f "$CONCAT_LIST"
for f in "${SCENE_FILES[@]}"; do
    # Use absolute path for reliability
    echo "file '$(pwd)/$f'" >> "$CONCAT_LIST"
done

cat "$CONCAT_LIST"

# ── Stitch → MP4 ───────────────────────────────────────────────────────────────
echo ""
echo "Stitching → $OUT_DIR/demo_full.mp4 …"
ffmpeg -y \
    -f concat -safe 0 -i "$CONCAT_LIST" \
    -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=${BG_COLOR:-0e1117},fps=24" \
    -c:v libx264 -preset fast -crf 22 \
    -movflags +faststart \
    -an \
    "$OUT_DIR/demo_full.mp4"

echo "  ✓ MP4 saved: $OUT_DIR/demo_full.mp4"
du -sh "$OUT_DIR/demo_full.mp4"

# ── Convert MP4 → GIF (palette-optimised) ──────────────────────────────────────
echo ""
echo "Converting → $OUT_DIR/demo_full.gif …"
PALETTE="$SCENES_DIR/palette.png"

ffmpeg -y -i "$OUT_DIR/demo_full.mp4" \
    -vf "fps=10,scale=960:-1:flags=lanczos,palettegen=max_colors=128" \
    "$PALETTE"

ffmpeg -y -i "$OUT_DIR/demo_full.mp4" -i "$PALETTE" \
    -filter_complex "fps=10,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" \
    "$OUT_DIR/demo_full.gif"

echo "  ✓ GIF saved: $OUT_DIR/demo_full.gif"
du -sh "$OUT_DIR/demo_full.gif"

# ── Probe output ────────────────────────────────────────────────────────────────
echo ""
echo "=== Output Summary ==="
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 \
    "$OUT_DIR/demo_full.mp4" 2>/dev/null || true
echo "=== Done ==="
