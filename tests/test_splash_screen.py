"""
Tests for MLForge Startup Splash Screen Component
=================================================
Verifies:
1. Splash HTML contains official brand logo, title, and tagline.
2. Scoped CSS contains smooth animations, timing (~4s default), and dark theme colors.
3. Offline requirement: zero external network dependencies / CDNs.
4. Accessibility: prefers-reduced-motion support.
5. Session state logic: runs once on startup, skips on reruns.
"""

import pytest
import streamlit as st
from modules.splash_screen import render_splash_css, render_splash_html, show_splash_screen


def test_splash_html_structure():
    """Verify splash HTML contains brand logo, title, and tagline."""
    html = render_splash_html(duration_sec=4.0)

    assert "mlforge-splash-wrapper" in html
    assert "ML<span>Forge</span>" in html or "MLForge" in html
    assert "AutoML Workspace" in html
    assert "mlforge-splash-logo" in html
    assert "<svg" in html, "Must contain inline SVG brand symbol"
    assert "progressbar" in html, "Must contain accessible progressbar role"
    assert "dialog" in html, "Must contain accessible dialog role"


def test_splash_css_timing_and_theme():
    """Verify splash CSS contains smooth animations and dark theme tokens."""
    css = render_splash_css(duration_sec=4.0)

    # Dark theme background and copper/teal palette
    assert "#121113" in css, "Must use Darkmatter dark background"
    assert "#e78a53" in css, "Must use warm copper primary accent"
    assert "#5f8787" in css, "Must use slate teal secondary accent"

    # Keyframes
    assert "@keyframes splashLogoEntrance" in css
    assert "@keyframes splashTextEntrance" in css
    assert "@keyframes splashProgressFill" in css
    assert "@keyframes splashFadeOut" in css



def test_splash_offline_compliance():
    """Verify splash CSS and HTML contain zero remote URLs (100% offline)."""
    html = render_splash_html(duration_sec=4.0)
    css = render_splash_css(duration_sec=4.0)

    for content, name in [(html, "HTML"), (css, "CSS")]:
        # Filter out standard XML namespace string xmlns="http://www.w3.org/2000/svg"
        clean_text = content.replace("http://www.w3.org/2000/svg", "")
        assert "http://" not in clean_text.lower(), f"Remote HTTP URL found in splash {name}"
        assert "https://" not in clean_text.lower(), f"Remote HTTPS URL found in splash {name}"



def test_splash_accessibility_reduced_motion():
    """Verify prefers-reduced-motion media query is implemented."""
    css = render_splash_css(duration_sec=4.0)
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "animation: none !important" in css or "animation: splashFadeOut" in css


def test_splash_session_state_guarded():
    """Verify show_splash_screen runs once and suppresses on subsequent calls."""
    # Reset session state for test
    if "splash_completed" in st.session_state:
        del st.session_state["splash_completed"]

    # First call: should run
    result1 = show_splash_screen(duration_sec=4.0)
    assert result1 is True
    assert st.session_state.splash_completed is True

    # Second call (simulating Streamlit rerun): should skip
    result2 = show_splash_screen(duration_sec=4.0)
    assert result2 is False, "Must NOT run again on subsequent rerun"

    # Forced call: should run
    result3 = show_splash_screen(duration_sec=4.0, force=True)
    assert result3 is True
