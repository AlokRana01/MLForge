"""
Tests for MLForge - Professional UI/UX Transformation
=====================================================
Validates design system tokens, CSS generation, Plotly layouts,
empty state containers, and headless AppTest flows.
"""

import pytest
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest
from modules.ui_theme import (
    THEME_DARK,
    THEME_LIGHT,
    get_tokens,
    get_plotly_layout,
    render_empty_state,
    render_dataset_kpi_bar,
    render_platform_header,
    render_step_header,
    render_structured_error
)
from modules.data_audit import run_data_audit


def test_theme_tokens_integrity():
    """Verify dark and light theme tokens define required UI keys."""
    required_keys = [
        "bg", "surface", "surface_card", "border", "text_primary",
        "text_secondary", "text_muted", "accent_primary", "accent_hover",
        "accent_secondary", "success", "warning", "error"
    ]
    for key in required_keys:
        assert key in THEME_DARK, f"Key {key} missing from THEME_DARK"
        assert key in THEME_LIGHT, f"Key {key} missing from THEME_LIGHT"
        assert THEME_DARK[key] != THEME_LIGHT[key], f"Key {key} has identical color across dark/light"


def test_get_tokens_helper():
    """Verify token resolution for specified themes."""
    assert get_tokens("dark")["bg"] == THEME_DARK["bg"]
    assert get_tokens("light")["bg"] == THEME_LIGHT["bg"]


def test_plotly_theme_layout():
    """Verify Plotly layout styling aligns with dark and light themes."""
    dark_layout = get_plotly_layout("dark")
    light_layout = get_plotly_layout("light")

    assert dark_layout["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert light_layout["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert dark_layout["font"]["color"] == THEME_DARK["text_secondary"]
    assert light_layout["font"]["color"] == THEME_LIGHT["text_secondary"]
    assert dark_layout["xaxis"]["gridcolor"] == THEME_DARK["border"]
    assert light_layout["xaxis"]["gridcolor"] == THEME_LIGHT["border"]


def test_headless_app_initial_load():
    """Verify that app.py loads cleanly without uncaught exceptions on Step 1 and displays MLForge brand."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest threw an exception: {at.exception}"
    assert at.session_state["step"] == 1
    assert at.session_state["df_raw"] is None
    assert at.session_state["theme"] in ("dark", "light")
    
    # Verify brand display in markdown
    all_markdown = " ".join(m.value for m in at.markdown)
    assert "MLForge" in all_markdown
    assert "ML Studio Pro" not in all_markdown


def test_headless_app_benchmark_loading():
    """Verify that selecting and loading the Iris benchmark populates dataset and advances step."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Find the benchmark load button and trigger it
    btn_benchmark = at.button(key="btn_load_benchmark")
    assert btn_benchmark is not None, "Benchmark load button not found"
    btn_benchmark.click().run()

    assert not at.exception
    assert at.session_state["step"] == 2
    assert at.session_state["df_raw"] is not None
    assert len(at.session_state["df_raw"]) == 150
    assert at.session_state["target_col"] == "target"
    assert at.session_state["problem_type"] == "classification"


def test_headless_app_step_guards():
    """Verify that jumping to step 7 without data triggers an empty state without crashing."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Force step 7 navigation while df_clean is None
    at.session_state["step"] = 7
    at.run()
    assert not at.exception
    # Verify no crash and markdown empty state displayed
    markdowns = [m.value for m in at.markdown]
    has_empty = any("No Dataset Available for Training" in m for m in markdowns)
    assert has_empty, "Step 7 did not render expected empty state"


def test_headless_app_step10_empty_guard():
    """Verify that jumping to step 10 (Prediction Playground) without trained models shows empty state."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    at.session_state["step"] = 10
    at.run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    has_empty = any("No Trained Pipeline Available" in m for m in markdowns)
    assert has_empty, "Step 10 did not render expected empty state"


def test_headless_theme_dark_only():
    """Verify that MLForge strictly enforces dark mode and removes light theme toggle."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert at.session_state["theme"] == "dark"
    # Ensure no theme switchers exist in the UI
    theme_radios = [r for r in at.radio if r.key == "sidebar_theme_toggle"]
    assert len(theme_radios) == 0, "Theme toggle radio must not exist in Dark Mode Only mode"
    theme_selects = [s for s in at.selectbox if s.key == "global_theme_selector"]
    assert len(theme_selects) == 0, "Theme selectbox must not exist in Dark Mode Only mode"


def test_headless_app_step11_explainability_empty_guard():
    """Verify that jumping to step 11 (Explainability) without trained models shows empty state."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    at.session_state["step"] = 11
    at.run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    has_empty = any("No Trained Model Available for Explainability" in m for m in markdowns)
    assert has_empty, "Step 11 did not render expected empty state"

