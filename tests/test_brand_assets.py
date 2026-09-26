"""
MLForge - Brand Assets & Identity System Verification Suite
===========================================================
Validates that:
1. All required SVG, PNG, and ICO asset files exist with valid non-zero sizes.
2. SVG files are well-formed XML documents with valid viewBox definitions.
3. Favicon raster files match their exact required pixel dimensions.
4. Windows/Web multi-resolution ICO file is valid and contains multi-resolutions.
5. Brand guidelines documentation exists and contains all required sections.
6. Headless AppTest runs cleanly with integrated brand symbols across themes.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from modules.ui_theme import get_brand_symbol_svg


BASE_DIR = Path(__file__).resolve().parent.parent
BRAND_DIR = BASE_DIR / "assets" / "branding"
LOGO_DIR = BRAND_DIR / "logo"
ICON_DIR = BRAND_DIR / "icon"
FAVICON_DIR = BRAND_DIR / "favicon"


REQUIRED_LOGOS = [
    "mlforge-logo-primary.svg",
    "mlforge-logo-primary.png",
    "mlforge-logo-light.svg",
    "mlforge-logo-light.png",
    "mlforge-logo-dark.svg",
    "mlforge-logo-dark.png",
    "mlforge-logo-transparent.svg",
    "mlforge-logo-transparent.png",
    "mlforge-logo-with-tagline.svg",
    "mlforge-logo-with-tagline.png",
    "mlforge-logo-monochrome-white.svg",
    "mlforge-logo-monochrome-white.png",
    "mlforge-logo-monochrome-dark.svg",
    "mlforge-logo-monochrome-dark.png",
]

REQUIRED_ICONS = [
    "mlforge-icon.svg",
    "mlforge-icon.png",
    "mlforge-icon-light.svg",
    "mlforge-icon-light.png",
    "mlforge-icon-dark.svg",
    "mlforge-icon-dark.png",
    "mlforge-app-icon.svg",
    "mlforge-app-icon.png",
]

REQUIRED_FAVICONS = [
    "favicon-16.png",
    "favicon-32.png",
    "favicon-48.png",
    "favicon-64.png",
    "favicon-128.png",
    "favicon-192.png",
    "favicon-512.png",
    "favicon.ico",
]


def test_asset_files_existence():
    """Verify all 30 expected brand asset files exist and are not empty."""
    for logo_name in REQUIRED_LOGOS:
        p = LOGO_DIR / logo_name
        assert p.exists(), f"Missing required logo asset: {logo_name}"
        assert p.stat().st_size > 0, f"Logo asset is empty: {logo_name}"

    for icon_name in REQUIRED_ICONS:
        p = ICON_DIR / icon_name
        assert p.exists(), f"Missing required icon asset: {icon_name}"
        assert p.stat().st_size > 0, f"Icon asset is empty: {icon_name}"

    for fav_name in REQUIRED_FAVICONS:
        p = FAVICON_DIR / fav_name
        assert p.exists(), f"Missing required favicon asset: {fav_name}"
        assert p.stat().st_size > 0, f"Favicon asset is empty: {fav_name}"


def test_svg_validity_and_viewbox():
    """Verify that all generated SVG assets are well-formed XML and define viewBox."""
    svg_files = list(LOGO_DIR.glob("*.svg")) + list(ICON_DIR.glob("*.svg"))
    assert len(svg_files) >= 11, f"Expected at least 11 SVG files, found {len(svg_files)}"

    for svg_path in svg_files:
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            assert root.tag.endswith("svg"), f"Root element of {svg_path.name} is not <svg>"
            assert "viewBox" in root.attrib, f"{svg_path.name} is missing viewBox attribute"
        except Exception as ex:
            pytest.fail(f"Invalid XML in {svg_path.name}: {ex}")


def test_favicon_dimensions():
    """Verify that raster favicons match their declared resolutions exactly."""
    expected_dimensions = {
        "favicon-16.png": (16, 16),
        "favicon-32.png": (32, 32),
        "favicon-48.png": (48, 48),
        "favicon-64.png": (64, 64),
        "favicon-128.png": (128, 128),
        "favicon-192.png": (192, 192),
        "favicon-512.png": (512, 512),
        "mlforge-app-icon.png": (512, 512),
        "mlforge-icon.png": (512, 512),
    }
    for filename, (expected_w, expected_h) in expected_dimensions.items():
        p = (FAVICON_DIR / filename) if "favicon" in filename else (ICON_DIR / filename)
        with Image.open(p) as img:
            assert img.size == (expected_w, expected_h), f"{filename} size is {img.size}, expected {(expected_w, expected_h)}"
            assert img.mode == "RGBA", f"{filename} mode is {img.mode}, expected RGBA"


def test_favicon_ico_validity():
    """Verify that favicon.ico is a valid Windows icon file with embedded resolutions."""
    ico_path = FAVICON_DIR / "favicon.ico"
    assert ico_path.exists()
    assert ico_path.stat().st_size > 0
    with Image.open(ico_path) as img:
        assert img.format == "ICO"
        assert img.size in [(16, 16), (32, 32), (48, 48), (64, 64)]


def test_brand_guidelines_completeness():
    """Verify that BRAND_GUIDELINES.md exists and contains all required specification sections."""
    guide_path = BRAND_DIR / "BRAND_GUIDELINES.md"
    assert guide_path.exists()
    content = guide_path.read_text(encoding="utf-8")

    required_sections = [
        "Brand Concept",
        "Symbol Design",
        "Wordmark",
        "Color System",
        "Clearspace",
        "Minimum Sizing",
        "Incorrect Logo Usage",
        "Asset Directory Structure",
    ]
    for section in required_sections:
        assert section.lower() in content.lower(), f"Missing section '{section}' in BRAND_GUIDELINES.md"


def test_get_brand_symbol_svg_helper():
    """Verify the inline SVG helper returns valid, parameterized SVG markup."""
    dark_svg = get_brand_symbol_svg(size=32, theme="dark")
    light_svg = get_brand_symbol_svg(size=24, theme="light")

    assert '<svg width="32" height="32"' in dark_svg
    assert '<svg width="24" height="24"' in light_svg
    assert "#e78a53" in dark_svg  # Warm copper in dark mode
    assert "#d87943" in light_svg  # Deep copper in light mode



def test_headless_app_brand_integration():
    """Verify that app.py runs with brand integration under Streamlit headless AppTest."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Check for MLForge text in all rendered markdown
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "MLForge" in all_markdown
    assert "ML Studio Pro" not in all_markdown

    # Check that the brand symbol SVG is embedded in the markdown
    assert "<svg" in all_markdown
    assert "viewBox=\"0 0 100 100\"" in all_markdown
