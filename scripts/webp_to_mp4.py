#!/usr/bin/env python3
"""
webp_to_mp4.py  — converts animated .webp files → individual .mp4 clips
using Pillow to decode frames and ffmpeg to encode.

Usage: python3 scripts/webp_to_mp4.py

Reads from: docs/scenes/scene_NN.webm  (actually animated WebP)
Writes to:  docs/scenes/scene_NN_norm.mp4  (1280x720 H.264)
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not found — pip install pillow")

SCENES = Path("/Users/tarunsridhara/cowork projects/grocerygpt/docs/scenes")
OUT    = Path("/Users/tarunsridhara/cowork projects/grocerygpt/docs")

# Target resolution
W, H = 1280, 720
BG = (14, 17, 23)   # Streamlit dark bg


def letterbox(frame: Image.Image, target_w: int, target_h: int, bg: tuple) -> Image.Image:
    """Scale frame to fit within target_w x target_h, letterbox with bg colour."""
    frame = frame.convert("RGB")
    fw, fh = frame.size
    scale = min(target_w / fw, target_h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    frame = frame.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas.paste(frame, (off_x, off_y))
    return canvas


def webp_to_mp4(src: Path, dst: Path, fps: int = 24) -> bool:
    """Convert animated WebP at src → MP4 at dst. Returns True on success."""
    print(f"  Converting {src.name} …", flush=True)

    try:
        img = Image.open(src)
    except Exception as e:
        print(f"    ✗ Cannot open: {e}")
        return False

    # Collect all frames
    frames: list[bytes] = []
    frame_idx = 0
    while True:
        try:
            img.seek(frame_idx)
        except EOFError:
            break
        frame = letterbox(img.copy(), W, H, BG)
        frames.append(frame.tobytes("raw", "RGB"))
        frame_idx += 1

    if not frames:
        print(f"    ✗ No frames extracted")
        return False

    print(f"    {len(frames)} frames → encoding at {fps} fps …", flush=True)

    # Pipe raw RGB frames into ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{W}x{H}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for raw in frames:
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        err = proc.stderr.read().decode(errors="replace")[-800:]
        print(f"    ✗ ffmpeg failed (code {proc.returncode}): {err}")
        return False

    sz = dst.stat().st_size / 1024 / 1024
    dur = len(frames) / fps
    print(f"    ✓ {dst.name}  ({sz:.1f} MB, {dur:.1f}s)")
    return True


def gif_to_mp4(src: Path, dst: Path, fps: int = 24) -> bool:
    """Convert animated GIF → MP4 (same approach as WebP)."""
    return webp_to_mp4(src, dst, fps)  # Pillow handles both


def main() -> None:
    # Ordered clip list: (source_path, norm_name)
    sources: list[tuple[Path, str]] = [
        (SCENES / "title_card.gif",  "norm_00.mp4"),
        (SCENES / "scene_01.webm",   "norm_01.mp4"),
        (SCENES / "scene_02.webm",   "norm_02.mp4"),
        (SCENES / "scene_03.webm",   "norm_03.mp4"),
        (SCENES / "scene_04.webm",   "norm_04.mp4"),
        (SCENES / "scene_05.webm",   "norm_05.mp4"),
        (SCENES / "scene_06.webm",   "norm_06.mp4"),
        (SCENES / "scene_07.webm",   "norm_07.mp4"),
        (SCENES / "scene_08.webm",   "norm_08.mp4"),
        (SCENES / "scene_09.webm",   "norm_09.mp4"),
        (SCENES / "scene_10.webm",   "norm_10.mp4"),
    ]

    print("=== Converting clips to normalised MP4 ===")
    norm_paths: list[Path] = []
    for src, norm_name in sources:
        if not src.exists():
            print(f"  ✗ MISSING: {src}")
            continue
        dst = SCENES / norm_name
        ok = webp_to_mp4(src, dst)
        if ok:
            norm_paths.append(dst)

    if not norm_paths:
        sys.exit("No clips converted — aborting")

    print(f"\n=== Converted {len(norm_paths)} clips ===")

    # ── Build concat list ──────────────────────────────────────────────────────
    concat_path = SCENES / "concat_norm.txt"
    with open(concat_path, "w") as f:
        for p in norm_paths:
            f.write(f"file '{p.absolute()}'\n")

    # ── Concat → MP4 ──────────────────────────────────────────────────────────
    mp4_out = OUT / "demo_full.mp4"
    print(f"\n=== Stitching → {mp4_out.name} ===")
    r = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(mp4_out),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg stderr:", r.stderr[-1000:])
        sys.exit("Concat failed")

    sz = mp4_out.stat().st_size / 1024 / 1024
    print(f"  ✓ MP4  ({sz:.1f} MB)")

    # ── MP4 → GIF ─────────────────────────────────────────────────────────────
    gif_out = OUT / "demo_full.gif"
    palette = SCENES / "palette.png"
    print(f"\n=== Converting → {gif_out.name} ===")

    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp4_out),
        "-vf", "fps=8,scale=960:-1:flags=lanczos,palettegen=max_colors=128:reserve_transparent=0",
        str(palette),
    ], capture_output=True)

    r = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(mp4_out), "-i", str(palette),
        "-filter_complex",
        "fps=8,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
        str(gif_out),
    ], capture_output=True, text=True)

    if r.returncode == 0:
        sz = gif_out.stat().st_size / 1024 / 1024
        print(f"  ✓ GIF  ({sz:.1f} MB)")
    else:
        print("  ✗ GIF conversion failed — MP4 is still good")

    print("\n=== ✅ Done ===")
    for f in [mp4_out, gif_out]:
        if f.exists():
            print(f"  {f}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
