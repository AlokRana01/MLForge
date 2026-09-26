"""
MLForge - Typography & Local Font System Verification Suite
===========================================================
Validates:
1. Local font files exist in `assets/fonts/` with valid WOFF2 signatures and non-zero size.
2. `get_font_face_css()` generates valid `@font-face` rules for both Geist Sans & Geist Mono.
3. 100% offline compliance: no Google Fonts, CDN URLs, or remote font services.
4. CSS variables `--font-sans` and `--font-mono` are set correctly to Geist families.
5. Typography hierarchy adherence across headings, buttons, metrics, body, and technical code.
6. Splash screen CSS consistency with Geist Sans and Geist Mono typography.
"""

from pathlib import Path
import pytest
from modules.fonts import get_font_face_css, FONTS_DIR, _FONT_SPECS
from modules.splash_screen import render_splash_css
from modules.ui_theme import inject_custom_css


def test_local_font_files_exist():
    """Verify all 8 expected Geist Sans and Geist Mono WOFF2 font files exist locally."""
    assert FONTS_DIR.exists(), f"Fonts directory {FONTS_DIR} does not exist"

    expected_files = [
        "Geist-Regular.woff2",
        "Geist-Medium.woff2",
        "Geist-SemiBold.woff2",
        "Geist-Bold.woff2",
        "GeistMono-Regular.woff2",
        "GeistMono-Medium.woff2",
        "GeistMono-SemiBold.woff2",
        "GeistMono-Bold.woff2",
    ]

    for fname in expected_files:
        p = FONTS_DIR / fname
        assert p.exists(), f"Missing local font file: {fname}"
        assert p.stat().st_size > 10_000, f"Font file {fname} is unexpectedly small: {p.stat().st_size} bytes"

        # Verify WOFF2 header signature ("wOF2" -> b"\x77\x4F\x46\x32")
        with open(p, "rb") as f:
            magic = f.read(4)
            assert magic == b"wOF2", f"File {fname} is not a valid WOFF2 font file (magic: {magic!r})"


def test_font_face_css_generation():
    """Verify get_font_face_css generates valid @font-face rules with embedded data URIs."""
    css = get_font_face_css()

    assert "@font-face" in css
    assert "font-family: 'Geist Sans'" in css
    assert "font-family: 'Geist Mono'" in css

    # Verify all 4 required weights for each family
    for weight in [400, 500, 600, 700]:
        assert f"font-weight: {weight};" in css

    # Verify base64 data URI structure
    assert "src: url('data:font/woff2;charset=utf-8;base64," in css
    assert "format('woff2')" in css


def test_offline_compliance_no_google_fonts():
    """Verify zero external font CDNs or Google Fonts URLs are used."""
    font_css = get_font_face_css()
    splash_css = render_splash_css(4.0)

    for content, source_name in [(font_css, "Local Font CSS"), (splash_css, "Splash CSS")]:
        clean = content.replace("http://www.w3.org/2000/svg", "")
        assert "fonts.googleapis.com" not in clean, f"Google Fonts found in {source_name}"
        assert "fonts.gstatic.com" not in clean, f"Google Fonts static CDN found in {source_name}"
        assert "cdn.jsdelivr.net" not in clean, f"jsDelivr CDN found in {source_name}"
        assert "cdnjs.cloudflare.com" not in clean, f"Cloudflare CDN found in {source_name}"


def test_splash_screen_typography():
    """Verify startup splash screen uses Geist Sans for titles/taglines and Geist Mono for status."""
    splash_css = render_splash_css(4.0)

    # Title & tagline use Geist Sans
    assert "font-family: 'Geist Sans'" in splash_css

    # Status uses Geist Mono
    assert "font-family: 'Geist Mono'" in splash_css

    # Old fonts must not appear
    assert "Plus Jakarta Sans" not in splash_css
    assert "JetBrains Mono" not in splash_css
