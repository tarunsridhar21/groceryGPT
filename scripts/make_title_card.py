#!/usr/bin/env python3
"""
Generate a dark-mode title card for the GroceryGPT demo walkthrough.
Outputs: docs/scenes/title_card.gif  (5 s still, 10 fps = 50 identical frames)
         docs/scenes/title_card.png  (single frame for preview)

Requires: Pillow  (pip install pillow)
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow not found — run: pip install pillow")

# ── Config ─────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
BG_COLOR      = (14, 17, 23)          # Streamlit dark bg
ACCENT        = (255, 75, 75)         # soft red accent
WHITE         = (240, 240, 240)
GREY          = (150, 160, 170)
DURATION_S    = 5
FPS           = 10

TITLE    = "GroceryGPT"
SUBTITLE = "Local RAG over 1,891 UK grocery products"
STACK    = "LangChain · ChromaDB · llama3.2:3b · BAAI/bge-small-en-v1.5 · RAGAS"
CAPTION  = "Zero API keys.  Zero cloud.  Runs entirely on-device."

OUT_DIR  = Path(__file__).parent.parent / "docs" / "scenes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Font helpers ────────────────────────────────────────────────────────────────
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a system font at the given size (falls back to default if not found)."""
    candidates_bold = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    candidates_reg = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in (candidates_bold if bold else candidates_reg):
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ── Draw ────────────────────────────────────────────────────────────────────────
def draw_title_card() -> Image.Image:
    img  = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Subtle gradient overlay (draw horizontal bands from top)
    for y in range(HEIGHT):
        alpha = int(30 * (1 - y / HEIGHT))   # darkens top
        r, g, b = BG_COLOR
        draw.line([(0, y), (WIDTH, y)], fill=(max(0, r - alpha), max(0, g - alpha), max(0, b - alpha)))

    # Accent bar
    bar_h = 6
    draw.rectangle([(0, HEIGHT // 2 - 110), (WIDTH, HEIGHT // 2 - 110 + bar_h)], fill=ACCENT)

    # Title
    f_title = _font(80, bold=True)
    draw.text((WIDTH // 2, HEIGHT // 2 - 80), TITLE, font=f_title, fill=WHITE, anchor="mm")

    # Subtitle
    f_sub = _font(36)
    draw.text((WIDTH // 2, HEIGHT // 2 - 4), SUBTITLE, font=f_sub, fill=WHITE, anchor="mm")

    # Stack line
    f_stack = _font(22)
    draw.text((WIDTH // 2, HEIGHT // 2 + 60), STACK, font=f_stack, fill=GREY, anchor="mm")

    # Caption
    f_cap = _font(26, bold=True)
    draw.text((WIDTH // 2, HEIGHT // 2 + 130), CAPTION, font=f_cap, fill=ACCENT, anchor="mm")

    # Corner watermark
    f_wm = _font(18)
    draw.text((WIDTH - 20, HEIGHT - 20), "github.com/tarunsridharan/grocerygpt",
              font=f_wm, fill=GREY, anchor="rb")

    return img


def main() -> None:
    print("Generating title card …")
    frame = draw_title_card()

    # Save single PNG
    png_path = OUT_DIR / "title_card.png"
    frame.save(png_path)
    print(f"  ✓ PNG saved → {png_path}")

    # Save as animated GIF (all identical frames → still video)
    n_frames = DURATION_S * FPS
    frames   = [frame.copy() for _ in range(n_frames)]
    gif_path = OUT_DIR / "title_card.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),   # ms per frame
        loop=0,
    )
    size_kb = gif_path.stat().st_size / 1024
    print(f"  ✓ GIF saved  → {gif_path}  ({size_kb:.0f} KB, {n_frames} frames)")
    print("Done.")


if __name__ == "__main__":
    main()
