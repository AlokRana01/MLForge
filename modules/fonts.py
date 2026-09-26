"""
MLForge - Offline Typography & Local Font Face Provider
======================================================
Stores and serves local @font-face declarations for:
- Geist Sans (Weights: 400, 500, 600, 700)
- Geist Mono (Weights: 400, 500, 600, 700)

100% offline, zero network / CDN calls, embedded via base64 WOFF2 data URIs.
"""

import base64
from functools import lru_cache
from pathlib import Path

# Directory containing local .woff2 files
FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

_FONT_SPECS = [
    # (font-family name, file name, weight, style)
    ("Geist Sans", "Geist-Regular.woff2", 400, "normal"),
    ("Geist Sans", "Geist-Medium.woff2", 500, "normal"),
    ("Geist Sans", "Geist-SemiBold.woff2", 600, "normal"),
    ("Geist Sans", "Geist-Bold.woff2", 700, "normal"),
    ("Geist Mono", "GeistMono-Regular.woff2", 400, "normal"),
    ("Geist Mono", "GeistMono-Medium.woff2", 500, "normal"),
    ("Geist Mono", "GeistMono-SemiBold.woff2", 600, "normal"),
    ("Geist Mono", "GeistMono-Bold.woff2", 700, "normal"),
]


@lru_cache(maxsize=1)
def get_font_face_css() -> str:
    """
    Generates local @font-face CSS rules embedding local WOFF2 font files as base64 data URIs.
    Guarantees 100% offline execution with zero network dependency or external font services.
    Includes both primary family names and shorthand aliases ('Geist', 'GeistMono').
    """
    rules = []
    for family, filename, weight, style in _FONT_SPECS:
        font_path = FONTS_DIR / filename
        if font_path.exists():
            encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
            src_uri = f"url('data:font/woff2;charset=utf-8;base64,{encoded}') format('woff2')"
            # Primary family definition
            rules.append(
                f"@font-face {{\n"
                f"    font-family: '{family}';\n"
                f"    font-style: {style};\n"
                f"    font-weight: {weight};\n"
                f"    font-display: swap;\n"
                f"    src: {src_uri};\n"
                f"}}"
            )
    return "\n".join(rules)
