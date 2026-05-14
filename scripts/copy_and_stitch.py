#!/usr/bin/env python3
"""
copy_and_stitch.py — copies the best scene clips to docs/scenes/,
re-encodes each one to a normalised MP4, then concatenates + converts to GIF.

Strategy:
  1. Copy each .webp/.gif clip to docs/scenes/
  2. Re-encode every clip individually → scene_N_norm.mp4  (1280x720, H.264, 24fps)
  3. Concat all norm.mp4 files → docs/demo_full.mp4
  4. Convert MP4 → docs/demo_full.gif  (960px wide, 8fps, palette-optimised)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BRAIN  = Path("/Users/tarunsridhara/.gemini/antigravity/brain/faafcd76-7bdc-411f-a558-d9e821dee3b2")
SCENES = Path("/Users/tarunsridhara/cowork projects/grocerygpt/docs/scenes")
OUT    = Path("/Users/tarunsridhara/cowork projects/grocerygpt/docs")
SCENES.mkdir(parents=True, exist_ok=True)

# ── 1. Source clips ─────────────────────────────────────────────────────────────
# (prefix_glob, dest_name)
RAW_CLIPS: list[tuple[str, str]] = [
    ("scene_01_app_health_",          "scene_01.webm"),
    ("scene_02_suggested_prompts_",   "scene_02.webm"),
    ("scene_03_allergen_query_",      "scene_03.webm"),
    ("scene_04_nutriscore_filter_",   "scene_04.webm"),
    ("scene_05_vegan_query_",         "scene_05.webm"),
    ("scene_06_heinz_brand_",         "scene_06.webm"),
    ("scene_07_palm_oil_query_",      "scene_07.webm"),
    ("scene_08_edge_case_",           "scene_08.webm"),
    ("scene_09_feedback_click_",      "scene_09.webm"),
    ("scene_10_architecture_readme_", "scene_10.webm"),
]

def run(args: list[str], desc: str = "") -> None:
    """Run a subprocess, print stderr on failure."""
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ {desc or args[0]} failed (code {r.returncode})")
        print("    STDERR:", r.stderr[-1500:])
        raise SystemExit(r.returncode)

def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0

# ── 2. Copy raw clips ───────────────────────────────────────────────────────────
print("=== Step 1: Copy raw clips ===")
raw_paths: list[Path] = []

# Title card GIF (already in docs/scenes/)
title_gif = SCENES / "title_card.gif"
if title_gif.exists():
    print(f"  ✓ title_card.gif  ({title_gif.stat().st_size // 1024} KB)")
    raw_paths.append(title_gif)
else:
    print("  ✗ title_card.gif missing — run make_title_card.py first")

for prefix, dest_name in RAW_CLIPS:
    matches = sorted(BRAIN.glob(f"{prefix}*.webp"))
    if not matches:
        print(f"  ✗ MISSING: {prefix}*.webp")
        continue
    src = matches[-1]
    dst = SCENES / dest_name
    shutil.copy2(src, dst)
    sz = dst.stat().st_size / 1024 / 1024
    print(f"  ✓ {src.name}  →  {dst.name}  ({sz:.1f} MB)")
    raw_paths.append(dst)

print(f"\nTotal clips: {len(raw_paths)}")

# ── 3. Re-encode each clip → norm_N.mp4 ────────────────────────────────────────
VFILTER = (
    "scale=1280:720:force_original_aspect_ratio=decrease,"
    "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,"
    "fps=24,format=yuv420p"
)

print("\n=== Step 2: Re-encode to normalised MP4 ===")
norm_paths: list[Path] = []
for i, src in enumerate(raw_paths):
    norm = SCENES / f"norm_{i:02d}.mp4"
    print(f"  [{i:02d}] {src.name}  →  {norm.name}", end=" … ", flush=True)
    run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", VFILTER,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        str(norm),
    ], f"encode {src.name}")
    dur = probe_duration(norm)
    print(f"✓  ({dur:.1f}s)")
    norm_paths.append(norm)

# ── 4. Write concat list ────────────────────────────────────────────────────────
concat_path = SCENES / "concat_norm.txt"
with open(concat_path, "w") as f:
    for p in norm_paths:
        f.write(f"file '{p.absolute()}'\n")
print(f"\nConcat list → {concat_path}")

# ── 5. Stitch → MP4 ────────────────────────────────────────────────────────────
mp4_out = OUT / "demo_full.mp4"
print(f"\n=== Step 3: Stitch → {mp4_out.name} ===")
run([
    "ffmpeg", "-y",
    "-f", "concat", "-safe", "0", "-i", str(concat_path),
    "-c", "copy",
    "-movflags", "+faststart",
    str(mp4_out),
], "stitch MP4")

sz = mp4_out.stat().st_size / 1024 / 1024
dur = probe_duration(mp4_out)
mins, secs = divmod(int(dur), 60)
print(f"  ✓ MP4  ({sz:.1f} MB, {mins}m {secs}s)")

# ── 6. Convert MP4 → GIF ───────────────────────────────────────────────────────
gif_out = OUT / "demo_full.gif"
palette = SCENES / "palette.png"
print(f"\n=== Step 4: Convert → {gif_out.name} ===")

# Palette generation
run([
    "ffmpeg", "-y", "-i", str(mp4_out),
    "-vf", "fps=8,scale=960:-1:flags=lanczos,palettegen=max_colors=128:reserve_transparent=0",
    str(palette),
], "palette gen")

# GIF creation
run([
    "ffmpeg", "-y",
    "-i", str(mp4_out), "-i", str(palette),
    "-filter_complex",
    "fps=8,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
    str(gif_out),
], "GIF conversion")

sz = gif_out.stat().st_size / 1024 / 1024
print(f"  ✓ GIF  ({sz:.1f} MB)")

# ── Summary ─────────────────────────────────────────────────────────────────────
print("\n=== ✅ Done ===")
for f in [mp4_out, gif_out]:
    if f.exists():
        print(f"  {f}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
print("\n  Update README.md: replace docs/demo.png with docs/demo_full.gif")
