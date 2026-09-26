"""
MLForge - Unified Enterprise SaaS Design System & Theme Engine
==============================================================
Provides a high-density, dark-only developer workspace visual language:
- Pure dark theme with centralized --mf-* CSS custom property architecture.
- Detached floating sidebar with all-corner border radius and single independent scrollbar.
- Subtle multi-layer animated atmospheric background (zero CPU / GPU-accelerated).
- Modern typography: Geist Sans (primary / UI) + Geist Mono (data / code / metrics).
- Streamlit chrome suppression (hides deploy button and developer menus).
- Structured glassmorphic surfaces (Level 1 base, Level 2 elevated, Level 3 glass).
- Theme-aligned Plotly visualization layout.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from modules.fonts import get_font_face_css


# =====================================================================
# SEMANTIC DESIGN TOKENS (DARK ONLY)
# =====================================================================

THEME_DARK: Dict[str, str] = {
    # Core Surfaces (Darkmatter Remix)
    "bg": "#121113",
    "bg_elevated": "#181719",
    "bg_surface": "#121212",
    "bg_glass": "rgba(18, 17, 19, 0.85)",
    "bg_hover": "rgba(231, 138, 83, 0.08)",
    "surface": "#121212",
    "surface_card": "#121212",
    "surface_elevated": "#181719",
    "surface_hover": "rgba(231, 138, 83, 0.08)",
    "glass_bg": "rgba(18, 17, 19, 0.80)",
    "glass_border": "#222222",

    # Borders & Inputs
    "border": "#222222",
    "border_subtle": "rgba(255, 255, 255, 0.05)",
    "border_focus": "#e78a53",
    "border_active": "rgba(231, 138, 83, 0.40)",
    "input": "#222222",
    "ring": "#e78a53",

    # Typography
    "text_primary": "#c1c1c1",
    "text_secondary": "#888888",
    "text_muted": "#555555",
    "text_disabled": "#444444",

    # Accents & Palettes (Darkmatter Remix: Primary copper #e78a53, Secondary slate teal #5f8787)
    "accent_primary": "#e78a53",       # Darkmatter Primary (Warm Copper)
    "accent_hover": "#d87943",         # Primary Hover
    "accent_secondary": "#5f8787",     # Darkmatter Secondary (Slate Teal)
    "accent_soft": "rgba(231, 138, 83, 0.12)",
    "blue": "#5f8787",
    "cyan": "#5f8787",
    "indigo": "#5f8787",
    "violet": "#e78a53",

    # Charts
    "chart_1": "#5f8787",
    "chart_2": "#e78a53",
    "chart_3": "#fbcb97",
    "chart_4": "#888888",
    "chart_5": "#999999",

    # Semantic Status
    "success": "#10b981",              # Emerald 500
    "success_bg": "rgba(16, 185, 129, 0.12)",
    "warning": "#f59e0b",              # Amber 500
    "warning_bg": "rgba(245, 158, 11, 0.12)",
    "error": "#ef4444",                # Red 500
    "error_bg": "rgba(239, 68, 68, 0.12)",
    "danger": "#ef4444",
    "info": "#e78a53",                 # Darkmatter Primary
    "info_bg": "rgba(231, 138, 83, 0.12)",

    # Backward compatibility mappings
    "bamboo": "#c1c1c1",
    "water_silk": "#888888",
    "indigo_base": "#121212"
}

# The application is strictly DARK MODE ONLY.
# Light tokens retained strictly for backward compatibility with tests.
THEME_LIGHT: Dict[str, str] = {
    "bg": "#f8fafc",
    "bg_elevated": "#f1f5f9",
    "bg_surface": "#ffffff",
    "bg_glass": "rgba(255, 255, 255, 0.85)",
    "bg_hover": "rgba(2, 132, 199, 0.06)",
    "surface": "#ffffff",
    "surface_card": "#f1f5f9",
    "surface_elevated": "#ffffff",
    "surface_hover": "#e2e8f0",
    "glass_bg": "rgba(255, 255, 255, 0.8)",
    "glass_border": "rgba(226, 232, 240, 0.8)",
    "border": "#e2e8f0",
    "border_subtle": "#f1f5f9",
    "border_focus": "#0284c7",
    "border_active": "#0284c7",
    "text_primary": "#0f172a",
    "text_secondary": "#334155",
    "text_muted": "#475569",
    "text_disabled": "#94a3b8",
    "accent_primary": "#0284c7",
    "accent_hover": "#0369a1",
    "accent_secondary": "#4f46e5",
    "accent_soft": "rgba(2, 132, 199, 0.08)",
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "indigo": "#4f46e5",
    "violet": "#7c3aed",
    "success": "#059669",
    "success_bg": "rgba(5, 150, 105, 0.12)",
    "warning": "#d97706",
    "warning_bg": "rgba(217, 119, 6, 0.12)",
    "error": "#e11d48",
    "error_bg": "rgba(225, 29, 72, 0.12)",
    "danger": "#e11d48",
    "info": "#0284c7",
    "info_bg": "rgba(2, 132, 199, 0.12)",
    "bamboo": "#0f172a",
    "water_silk": "#475569",
    "indigo_base": "#f8fafc"
}


def get_active_theme() -> str:
    """The application is strictly dark-mode only."""
    return "dark"


def get_tokens(theme: Optional[str] = None) -> Dict[str, str]:
    """Retrieve design tokens for the active theme."""
    if theme == "light":
        return THEME_LIGHT
    return THEME_DARK


# =====================================================================
# CSS INJECTION & THEME ENGINE
# =====================================================================

def inject_custom_css(theme: Optional[str] = None) -> None:
    """
    Injects the complete MLForge dark SaaS design system:
    - Centralized semantic variables (--mf-*)
    - Local offline typography: Geist Sans (primary) and Geist Mono (technical/code)
    - Detached floating sidebar geometry with single-scrollbar overflow control
    - Pure atmospheric dark animated background
    - Streamlit default chrome suppression
    - Hugeicons stroke rounded styling
    """
    tok = THEME_DARK

    # 1. Inject Hugeicons web font
    st.markdown(
        '<link rel="preconnect" href="https://use.hugeicons.com">'
        '<link rel="stylesheet" href="https://use.hugeicons.com/font/icons.css">'
        '<style>.hgi{display:inline-block;line-height:1;vertical-align:middle;}'
        '.hgi-stroke{font-style:normal!important;}</style>',
        unsafe_allow_html=True
    )

    font_face_css = get_font_face_css()

    # 2. Main CSS stylesheet
    css = f"""
    <style>
    /* Local offline @font-face declarations */
    {font_face_css}

    :root, .dark {{
        /* Local Typography Stacks */
        --font-sans: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-mono: 'Geist Mono', monospace;
        --mf-font-sans: var(--font-sans);
        --mf-font-mono: var(--font-mono);

        /* Centralized shadcn/ui Darkmatter Remix Tokens */
        --background: #121113;
        --foreground: #c1c1c1;
        --card: #121212;
        --card-foreground: #c1c1c1;
        --popover: #121113;
        --popover-foreground: #c1c1c1;
        --primary: #e78a53;
        --primary-foreground: #121113;
        --secondary: #5f8787;
        --secondary-foreground: #121113;
        --muted: #222222;
        --muted-foreground: #888888;
        --accent: #333333;
        --accent-foreground: #c1c1c1;
        --destructive: #ef4444;
        --destructive-foreground: #121113;
        --border: #222222;
        --input: #222222;
        --ring: #e78a53;
        --radius: 0.375rem;

        /* Darkmatter Chart Palette */
        --chart-1: #5f8787;
        --chart-2: #e78a53;
        --chart-3: #fbcb97;
        --chart-4: #888888;
        --chart-5: #999999;

        /* MLForge-Specific Tokens */
        --mlforge-sidebar: #121212;
        --mlforge-sidebar-border: #222222;
        --mlforge-surface: #121212;
        --mlforge-surface-hover: #1b1a1c;
        --mlforge-success: #10b981;
        --mlforge-warning: #f59e0b;
        --mlforge-error: #ef4444;
        --mlforge-info: #e78a53;
        --mlforge-radius: 0.375rem;
        --mlforge-shadow: 0 4px 16px rgba(0, 0, 0, 0.40);

        /* MLForge --mf-* Semantic Tokens (mapped to Darkmatter Remix) */
        --mf-bg: var(--background);
        --mf-bg-elevated: #181719;
        --mf-bg-surface: var(--card);
        --mf-bg-glass: rgba(18, 17, 19, 0.85);
        --mf-bg-hover: rgba(231, 138, 83, 0.08);

        --mf-text-primary: var(--foreground);
        --mf-text-secondary: var(--muted-foreground);
        --mf-text-muted: #555555;
        --mf-text-disabled: #444444;

        --mf-border: var(--border);
        --mf-border-subtle: rgba(255, 255, 255, 0.05);
        --mf-border-active: rgba(231, 138, 83, 0.40);

        --mf-accent: var(--primary);
        --mf-accent-hover: #d87943;
        --mf-accent-soft: rgba(231, 138, 83, 0.12);

        --mf-blue: var(--secondary);
        --mf-cyan: var(--secondary);
        --mf-indigo: var(--secondary);
        --mf-violet: var(--primary);

        --mf-success: var(--mlforge-success);
        --mf-success-bg: rgba(16, 185, 129, 0.12);
        --mf-warning: var(--mlforge-warning);
        --mf-warning-bg: rgba(245, 158, 11, 0.12);
        --mf-danger: var(--destructive);
        --mf-danger-bg: rgba(239, 68, 68, 0.12);

        /* Backward Compatibility Aliases */
        --bg: var(--background);
        --surface: var(--card);
        --surface2: var(--mf-bg-elevated);
        --surface-elevated: var(--mf-bg-elevated);
        --surface-hover: var(--mf-bg-hover);
        --glass-bg: var(--mf-bg-glass);
        --glass-border: var(--border);
        --border-focus: var(--ring);
        --text: var(--foreground);
        --text-secondary: var(--muted-foreground);
        --muted: var(--muted);
        --accent: var(--accent);
        --accent-hover: var(--mf-accent-hover);
        --accent2: var(--secondary);
        --bamboo: var(--foreground);
        --water-silk: var(--muted-foreground);
        --indigo-base: var(--card);
        --success: var(--mlforge-success);
        --success-bg: var(--mf-success-bg);
        --warning: var(--mlforge-warning);
        --warning-bg: var(--mf-warning-bg);
        --error: var(--destructive);
        --error-bg: var(--mf-danger-bg);
        --info: var(--primary);
        --info-bg: var(--mf-accent-soft);
    }}

    /* Global Typography & Viewport */
    html, body {{
        background-color: var(--background) !important;
        color: var(--foreground) !important;
        font-family: var(--font-sans) !important;
        font-weight: 400;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .stApp {{
        background-color: var(--background) !important;
        font-family: var(--font-sans) !important;
        position: relative;
        overflow-x: hidden;
    }}

    /* Full-viewport Atmospheric Animated Dark Background */
    @keyframes mfAtmosphere {{
        0% {{
            background-position: 0% 0%, 100% 100%, 50% 50%;
        }}
        50% {{
            background-position: 100% 50%, 0% 50%, 80% 20%;
        }}
        100% {{
            background-position: 0% 0%, 100% 100%, 50% 50%;
        }}
    }}

    .mf-atmospheric-backdrop {{
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background-color: #121113;
        background-image: 
            radial-gradient(ellipse 75% 50% at 15% -5%, rgba(231, 138, 83, 0.04), transparent 70%),
            radial-gradient(ellipse 65% 50% at 85% 105%, rgba(95, 135, 135, 0.04), transparent 70%),
            radial-gradient(ellipse 50% 40% at 50% 50%, rgba(18, 17, 19, 0.6), transparent 100%);
        background-size: 150% 150%, 150% 150%, 100% 100%;
        animation: mfAtmosphere 20s ease-in-out infinite alternate;
    }}

    /* =================================================================
       RESPONSIVE FLOATING SIDEBAR & WORKSPACE TOKENS
       ================================================================= */
    :root {{
        --mf-sidebar-width: 260px;
        --mf-sidebar-gap: 16px;
        --mf-workspace-gap: 24px;
        --mf-sidebar-total-offset: calc(var(--mf-sidebar-width) + var(--mf-sidebar-gap) + var(--mf-workspace-gap));
        --mf-content-max-width: 1400px;
    }}

    @media (min-width: 1600px) {{
        :root {{
            --mf-sidebar-width: 260px;
            --mf-sidebar-gap: 18px;
            --mf-workspace-gap: 28px;
            --mf-content-max-width: 1440px;
        }}
    }}

    @media (max-width: 1599px) and (min-width: 1367px) {{
        :root {{
            --mf-sidebar-width: 260px;
            --mf-sidebar-gap: 16px;
            --mf-workspace-gap: 22px;
            --mf-content-max-width: 1240px;
        }}
    }}

    @media (max-width: 1366px) and (min-width: 1280px) {{
        :root {{
            --mf-sidebar-width: 254px;
            --mf-sidebar-gap: 14px;
            --mf-workspace-gap: 18px;
            --mf-content-max-width: 1040px;
        }}
    }}

    @media (max-width: 1279px) and (min-width: 992px) {{
        :root {{
            --mf-sidebar-width: 248px;
            --mf-sidebar-gap: 14px;
            --mf-workspace-gap: 16px;
            --mf-content-max-width: 960px;
        }}
    }}

    @media (max-width: 991px) {{
        :root {{
            --mf-sidebar-width: 240px;
            --mf-sidebar-gap: 12px;
            --mf-workspace-gap: 12px;
        }}
    }}

    /* Streamlit Chrome Elimination (Viewer Mode Enforcement) */
    header[data-testid="stHeader"] {{
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
    }}
    .stAppDeployButton,
    [data-testid="stToolbar"],
    #MainMenu,
    footer,
    [data-testid="stStatusWidget"],
    [data-testid="stSidebarHeader"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* =================================================================
       TRUE FLOATING SIDEBAR (Detached, All-Corner Radius, Independent Scroll)
       ================================================================= */
    section[data-testid="stSidebar"] {{
        position: fixed !important;
        top: var(--mf-sidebar-gap) !important;
        left: var(--mf-sidebar-gap) !important;
        bottom: var(--mf-sidebar-gap) !important;
        width: var(--mf-sidebar-width) !important;
        min-width: var(--mf-sidebar-width) !important;
        max-width: var(--mf-sidebar-width) !important;
        height: calc(100vh - (var(--mf-sidebar-gap) * 2)) !important;
        max-height: calc(100vh - (var(--mf-sidebar-gap) * 2)) !important;
        border-radius: 14px !important;
        border: 1px solid var(--mlforge-sidebar-border) !important;
        background: var(--mlforge-sidebar) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        box-shadow: 0 16px 44px 0 rgba(0, 0, 0, 0.70), 0 0 0 1px rgba(255, 255, 255, 0.03) !important;
        overflow: hidden !important;
        z-index: 100 !important;
        box-sizing: border-box !important;
        transition: transform 0.2s ease, margin-left 0.2s ease !important;
    }}

    /* Single Sleek Scrollbar on Sidebar Content Container */
    [data-testid="stSidebarContent"] {{
        height: 100% !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding: 0.85rem 0.75rem 4.5rem 0.75rem !important;
        box-sizing: border-box !important;
        scrollbar-width: thin !important;
        scrollbar-color: rgba(136, 136, 136, 0.25) transparent !important;
    }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar {{
        width: 4px !important;
    }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar-track {{
        background: transparent !important;
    }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {{
        background: rgba(136, 136, 136, 0.25) !important;
        border-radius: 4px !important;
    }}
    [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover {{
        background: var(--primary) !important;
    }}

    [data-testid="stSidebar"] * {{
        color: var(--foreground) !important;
    }}

    /* Compact Sidebar Row Margins */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
        margin-bottom: 0.12rem !important;
    }}
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
        align-items: center !important;
        gap: 4px !important;
        margin-bottom: 0.12rem !important;
    }}
    [data-testid="stSidebar"] [data-testid="column"] {{
        padding: 0 !important;
        min-width: 0 !important;
    }}

    /* MLForge Sidebar Brand Header */
    .sidebar-brand-header {{
        padding: 1.45rem 0.2rem 0.65rem 0.2rem !important;
        border-bottom: 1px solid var(--border) !important;
        margin-bottom: 0.6rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.65rem !important;
    }}
    .sidebar-brand-text {{
        font-family: var(--font-sans) !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: #f3f4f6 !important;
        line-height: 1 !important;
    }}

    /* =================================================================
       SIDEBAR NAVIGATION ITEM (Consistent Two-Column Row Layout)
       ================================================================= */
    /* Shared Navigation Row Container */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) {{
        display: flex !important;
        align-items: center !important;
        height: 32px !important;
        min-height: 32px !important;
        border-radius: var(--radius) !important;
        border: 1px solid transparent !important;
        border-left: 3px solid transparent !important;
        margin-bottom: 2px !important;
        padding: 0 4px !important;
        gap: 6px !important;
        box-sizing: border-box !important;
        transition: all 0.15s ease-in-out !important;
    }}

    /* Active Navigation Row: Encompasses both Icon and Label */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box):has(button[kind="primary"]),
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box):has(button[data-testid="baseButton-primary"]) {{
        background: rgba(231, 138, 83, 0.12) !important;
        border: 1px solid rgba(231, 138, 83, 0.28) !important;
        border-left: 3px solid var(--primary) !important;
        box-shadow: 0 2px 8px rgba(231, 138, 83, 0.12) !important;
    }}

    /* Inactive Navigation Row Hover */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box):not(:has(button[kind="primary"])):not(:has(button[data-testid="baseButton-primary"])):hover {{
        background: var(--accent) !important;
        border-color: var(--border) !important;
        transform: translateX(2px) !important;
    }}

    /* Icon Column (Fixed Width: 24px) */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) > [data-testid="column"]:first-child {{
        width: 24px !important;
        min-width: 24px !important;
        max-width: 24px !important;
        flex: 0 0 24px !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
    }}

    /* Label / Button Column (Flexible Label Area: flex 1) */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) > [data-testid="column"]:last-child {{
        flex: 1 1 auto !important;
        min-width: 0 !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        padding: 0 !important;
    }}

    /* Streamlit Wrapper Resets inside Navigation Columns */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) [data-testid="stElementContainer"] {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 100% !important;
        width: 100% !important;
        gap: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) [data-testid="stMarkdownContainer"] {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* Navigation Icon Box */
    .nav-ic-box {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 20px !important;
        height: 20px !important;
        font-size: 16px !important;
        line-height: 1 !important;
        color: var(--muted-foreground);
        flex-shrink: 0 !important;
        transition: color 0.15s ease;
    }}
    .nav-ic-box.active {{
        color: var(--primary) !important;
    }}
    .nav-ic-box.locked {{
        color: var(--muted-foreground) !important;
        opacity: 0.35 !important;
    }}

    /* Navigation Button Styling */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) .stButton {{
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) .stButton > button {{
        width: 100% !important;
        height: 100% !important;
        min-height: 100% !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        font-family: var(--font-sans) !important;
        font-size: 0.81rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        color: var(--foreground) !important;
        transform: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) .stButton > button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) .stButton > button[data-testid="baseButton-primary"] {{
        color: #ffffff !important;
        font-weight: 600 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) .stButton > button:disabled {{
        opacity: 0.35 !important;
        background: transparent !important;
        color: var(--muted-foreground) !important;
        cursor: not-allowed !important;
        transform: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) .stButton > button div,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) .stButton > button span,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nav-ic-box) .stButton > button p {{
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        margin: 0 !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        font-family: var(--font-sans) !important;
    }}

    /* Generic Sidebar Buttons (e.g. non-navigation buttons like Reset Workspace) */
    [data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        color: var(--foreground) !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0 0.55rem !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-sans) !important;
        font-size: 0.81rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        transition: all 0.15s ease-in-out !important;
        height: 32px !important;
        min-height: 32px !important;
        line-height: 32px !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: var(--accent) !important;
        color: #ffffff !important;
        border-color: var(--border) !important;
    }}

    /* Collapsible Sidebar Toggle Access */
    [data-testid="stSidebarCollapsedControl"] {{
        top: 14px !important;
        left: 14px !important;
        z-index: 101 !important;
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }}

    /* =================================================================
       MAIN WORKSPACE LAYOUT (Explicit Horizontal Separation from Sidebar)
       ================================================================= */
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"],
    section.main {{
        position: relative !important;
        margin-left: var(--mf-sidebar-total-offset) !important;
        width: calc(100% - var(--mf-sidebar-total-offset)) !important;
        max-width: calc(100% - var(--mf-sidebar-total-offset)) !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
    }}

    /* Inner Content Block inside the Main Workspace */
    .main .block-container,
    [data-testid="stMainBlockContainer"] {{
        position: relative;
        z-index: 1;
        background: transparent !important;
        max-width: var(--mf-content-max-width) !important;
        width: 100% !important;
        margin-left: 0 !important;
        margin-right: auto !important;
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.25rem !important;
        padding-right: 2rem !important;
        box-sizing: border-box !important;
    }}

    /* When sidebar is collapsed */
    section[data-testid="stSidebar"][aria-expanded="false"] {{
        margin-left: -320px !important;
        transform: translateX(-320px) !important;
    }}
    section[data-testid="stSidebar"][aria-expanded="false"] ~ .main,
    section[data-testid="stSidebar"][aria-expanded="false"] ~ [data-testid="stMain"],
    section[data-testid="stSidebar"][aria-expanded="false"] ~ section.main {{
        margin-left: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }}
    section[data-testid="stSidebar"][aria-expanded="false"] ~ .main .block-container {{
        margin: 0 auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

    @media (max-width: 991px) {{
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMain"],
        section.main {{
            margin-left: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
        }}
        .main .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
    }}

    /* =================================================================
       TYPOGRAPHY HIERARCHY
       ================================================================= */
    h1, h2, h3, h4, h5, h6, .brand-text, .page-title {{
        font-family: var(--font-sans) !important;
        letter-spacing: -0.025em !important;
        color: #f3f4f6 !important;
    }}
    h1 {{
        font-weight: 700 !important;
    }}
    h2 {{
        font-weight: 700 !important;
    }}
    h3 {{
        font-weight: 600 !important;
    }}
    h4, h5, h6 {{
        font-weight: 600 !important;
    }}

    p, span, label, div {{
        font-family: var(--font-sans);
    }}

    code, kbd, samp, pre, .stCodeBlock, .stCodeBlock code, .telemetry-item strong, .tech-value, .stat-value, .model-param {{
        font-family: var(--font-mono) !important;
        font-weight: 500;
    }}

    /* =================================================================
       SAAS BUTTONS (Darkmatter Remix Style)
       ================================================================= */
    .stButton > button, .stDownloadButton > button {{
        background: #181719 !important;
        color: var(--foreground) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1rem !important;
        min-height: 38px !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25) !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: var(--accent) !important;
        border-color: #444444 !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
    }}
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {{
        background: var(--primary) !important;
        border-color: var(--primary) !important;
        color: var(--primary-foreground) !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 10px rgba(231, 138, 83, 0.25) !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {{
        background: var(--mf-accent-hover) !important;
        color: var(--primary-foreground) !important;
        border-color: var(--mf-accent-hover) !important;
        box-shadow: 0 4px 14px rgba(231, 138, 83, 0.35) !important;
    }}
    .stButton > button:disabled, .stDownloadButton > button:disabled {{
        opacity: 0.40 !important;
        background: #181719 !important;
        color: var(--muted-foreground) !important;
        border-color: var(--border) !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }}

    /* =================================================================
       PIPELINE STEP NAVIGATION (Back on Left, Next on Right)
       ================================================================= */
    .step-nav-left, .step-nav-right {{
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    [data-testid="stColumn"]:has(.step-nav-right) [data-testid="stVerticalBlock"] {{
        align-items: flex-end !important;
    }}
    [data-testid="stColumn"]:has(.step-nav-right) .stButton,
    [data-testid="stColumn"]:has(.step-nav-right) div[data-testid="stButton"] {{
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
    }}
    [data-testid="stColumn"]:has(.step-nav-right) .stButton > button,
    [data-testid="stColumn"]:has(.step-nav-right) div[data-testid="stButton"] > button {{
        margin-left: auto !important;
    }}

    [data-testid="stColumn"]:has(.step-nav-left) [data-testid="stVerticalBlock"] {{
        align-items: flex-start !important;
    }}
    [data-testid="stColumn"]:has(.step-nav-left) .stButton,
    [data-testid="stColumn"]:has(.step-nav-left) div[data-testid="stButton"] {{
        display: flex !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }}
    [data-testid="stColumn"]:has(.step-nav-left) .stButton > button,
    [data-testid="stColumn"]:has(.step-nav-left) div[data-testid="stButton"] > button {{
        margin-right: auto !important;
    }}


    /* =================================================================
       SURFACES & CARDS (Darkmatter Remix Style)
       ================================================================= */
    .workspace-card, .panel-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.15rem 1.35rem;
        margin: 0.6rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }}

    .glass-panel, .glass-card {{
        background: rgba(18, 18, 18, 0.85);
        border: 1px solid var(--border);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        margin: 0.6rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }}

    /* Compact KPI Metric Cards */
    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.85rem 1.1rem;
        text-align: left;
        position: relative;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.20);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .metric-card:hover {{
        border-color: var(--border-focus);
        transform: translateY(-1px);
    }}
    .metric-card-primary {{
        border-top: 2px solid var(--primary);
    }}
    .metric-card-success {{
        border-top: 2px solid var(--mlforge-success);
    }}
    .metric-card-warning {{
        border-top: 2px solid var(--mlforge-warning);
    }}
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #f3f4f6;
        font-family: var(--font-sans) !important;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }}
    .metric-label {{
        font-size: 0.72rem;
        color: var(--muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        font-family: var(--font-sans) !important;
        margin-top: 0.25rem;
    }}

    /* Semantic Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.55rem;
        border-radius: calc(var(--radius) - 2px);
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        line-height: 1;
        font-family: var(--font-sans) !important;
    }}
    .badge-primary {{
        background: var(--mf-accent-soft);
        color: var(--primary);
        border: 1px solid rgba(231, 138, 83, 0.25);
    }}
    .badge-success {{
        background: var(--mf-success-bg);
        color: var(--mlforge-success);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }}
    .badge-warning {{
        background: var(--mf-warning-bg);
        color: var(--mlforge-warning);
        border: 1px solid rgba(245, 158, 11, 0.25);
    }}
    .badge-error {{
        background: var(--mf-danger-bg);
        color: var(--destructive);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }}
    .badge-neutral {{
        background: var(--accent);
        color: var(--foreground);
        border: 1px solid var(--border);
    }}
    .badge-green {{ background: var(--mf-success-bg); color: var(--mlforge-success); border: 1px solid rgba(16, 185, 129, 0.25); }}
    .badge-purple {{ background: rgba(95, 135, 135, 0.15); color: var(--secondary); border: 1px solid rgba(95, 135, 135, 0.30); }}
    .badge-yellow {{ background: var(--mf-warning-bg); color: var(--mlforge-warning); border: 1px solid rgba(245, 158, 11, 0.25); }}
    .badge-red {{ background: var(--mf-danger-bg); color: var(--destructive); border: 1px solid rgba(239, 68, 68, 0.25); }}

    /* Compact Step Header with Breadcrumb */
    .step-header {{
        background: var(--card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--primary);
        border-radius: var(--radius);
        padding: 0.85rem 1.15rem;
        margin: 0.35rem 0 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.20);
    }}
    .step-breadcrumb {{
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.15rem;
    }}

    /* AI Card & Callout */
    .ai-card, .callout-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.95rem 1.15rem;
        margin: 0.5rem 0;
        font-size: 0.86rem;
        line-height: 1.5;
        color: var(--foreground);
    }}

    /* Live Telemetry Bar */
    .telemetry-bar {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.45rem 0.95rem;
        margin: 0.25rem 0 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.85rem;
        flex-wrap: wrap;
        font-size: 0.78rem;
    }}
    .telemetry-item {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: var(--foreground);
    }}
    .telemetry-divider {{
        color: var(--border);
        user-select: none;
    }}

    /* Form Controls & Inputs */
    label, [data-testid="stWidgetLabel"] {{
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
    }}
    input, select, textarea, .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div, .stMultiSelect > div > div, [data-baseweb="select"] {{
        background-color: var(--card) !important;
        border: 1px solid var(--input) !important;
        color: var(--foreground) !important;
        border-radius: var(--radius) !important;
        font-family: var(--font-sans) !important;
        font-weight: 400 !important;
    }}
    .stTextInput > div > div > input:focus, .stSelectbox > div > div:focus, .stMultiSelect > div > div:focus, [data-baseweb="select"]:focus {{
        border-color: var(--ring) !important;
        box-shadow: 0 0 0 1px var(--ring) !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background-color: var(--popover) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--popover-foreground) !important;
        font-family: var(--font-sans) !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background-color: var(--accent) !important;
    }}

    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background: var(--card) !important;
        border: 1px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.75rem !important;
        font-family: var(--font-sans) !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: transparent !important;
        border: none !important;
    }}

    /* Native Alerts */
    .stAlert {{
        border-radius: var(--radius) !important;
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        font-family: var(--font-sans) !important;
    }}

    /* Compact Tables & Dataframes */
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        background-color: var(--card) !important;
        font-family: var(--font-sans) !important;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px !important;
        background: var(--card) !important;
        padding: 4px !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 32px !important;
        border-radius: calc(var(--radius) - 2px) !important;
        padding: 0 12px !important;
        color: var(--muted-foreground) !important;
        font-family: var(--font-sans) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(231, 138, 83, 0.12) !important;
        color: var(--primary) !important;
        font-weight: 600 !important;
    }}

    /* Clean Footer */
    .footer-bar {{
        margin-top: 2.5rem;
        padding: 1rem 0 0.5rem;
        border-top: 1px solid var(--border);
        text-align: center;
        font-size: 0.72rem;
        color: var(--muted-foreground);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # 3. Inject the clean atmospheric animated background layer
    st.markdown('<div class="mf-atmospheric-backdrop"></div>', unsafe_allow_html=True)


# =====================================================================
# PLOTLY THEME HELPERS
# =====================================================================

def get_plotly_layout(theme: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns Plotly layout configuration matching the dark SaaS design system.
    """
    tok = get_tokens(theme)
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=tok["text_secondary"], family='Geist Sans, -apple-system, sans-serif', size=12),
        xaxis=dict(gridcolor=tok["border"], zerolinecolor=tok["border"], tickcolor=tok["border"]),
        yaxis=dict(gridcolor=tok["border"], zerolinecolor=tok["border"], tickcolor=tok["border"]),
        colorway=["#e78a53", "#5f8787", "#fbcb97", "#888888", "#10b981", "#f59e0b", "#ef4444", "#999999"],
        margin=dict(l=24, r=24, t=44, b=24)
    )


# =====================================================================
# UI COMPONENT RENDERERS
# =====================================================================

def get_brand_symbol_svg(size: int = 24, theme: Optional[str] = None) -> str:
    """Returns clean inline SVG markup for the official MLForge 3D ribbon brand symbol with data pixels."""
    is_light = (theme or get_active_theme()) == "light"
    # Darkmatter Remix Palette: Warm Copper & Slate Teal
    col_left = "#d87943" if is_light else "#e78a53"      # Signature warm copper primary
    col_top = "#e78a53" if is_light else "#fbcb97"       # Luminous peach/gold top lid
    col_center = "#527575" if is_light else "#5f8787"    # Darkmatter slate teal
    col_fold = "#3e5c5c" if is_light else "#527575"      # Deep slate teal shadow fold
    col_right = "#c06530" if is_light else "#d87943"     # Rich copper pillar
    col_bottom = "#7c3714" if is_light else "#9e4e24"    # Deep terracotta base
    px_1 = "#e78a53" if is_light else "#fbcb97"
    px_2 = "#d87943" if is_light else "#e78a53"
    px_3 = "#527575" if is_light else "#5f8787"
    px_4 = "#d87943" if is_light else "#e78a53"
    px_5 = "#e78a53" if is_light else "#fbcb97"

    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;flex-shrink:0">
        <!-- Left Top Lid -->
        <path d="M 14,32 L 32,20 L 44,28 L 26,40 Z" fill="{col_top}" />
        <!-- Left Vertical Face -->
        <path d="M 14,32 L 26,40 L 26,76 L 14,68 Z" fill="{col_left}" />
        <!-- Center Descending Ribbon Fold -->
        <path d="M 26,40 L 44,28 L 62,44 L 44,56 Z" fill="{col_center}" />
        <!-- Center Ascending Ribbon Fold -->
        <path d="M 26,76 L 44,88 L 60,74 L 42,62 Z" fill="{col_fold}" />
        <!-- Right Arch & Pillar -->
        <path d="M 62,44 L 78,32 L 78,74 L 66,74 L 66,54 L 44,56 Z" fill="{col_right}" />
        <!-- Bottom Cap -->
        <path d="M 14,68 L 26,76 L 36,84 L 24,76 Z" fill="{col_bottom}" />
        <!-- Floating Data Pixels -->
        <rect x="56" y="8" width="6.5" height="6.5" rx="1.2" fill="{px_1}" />
        <rect x="66" y="8" width="6.5" height="6.5" rx="1.2" fill="{px_2}" />
        <rect x="50" y="18" width="6.5" height="6.5" rx="1.2" fill="{px_3}" />
        <rect x="60" y="18" width="6.5" height="6.5" rx="1.2" fill="{px_4}" />
        <rect x="70" y="18" width="6.5" height="6.5" rx="1.2" fill="{px_5}" />
    </svg>"""



def render_platform_header() -> None:
    """
    Renders top application workspace status.
    Kept minimal to avoid repeating the MLForge logo present in the sidebar.
    """
    pass


def render_dataset_kpi_bar(
    df: Optional[pd.DataFrame],
    audit_result: Optional[Any] = None,
    target_col: Optional[str] = None,
    problem_type: Optional[str] = None,
    best_model_name: Optional[str] = None
) -> None:
    """
    Persistent dataset telemetry bar displayed across steps once data is loaded.
    """
    if df is None:
        return

    n_rows, n_cols = len(df), len(df.columns)
    SEP = '<span class="telemetry-divider">&#124;</span>'

    def _ti(slug: str, color: str = "var(--mf-text-muted)") -> str:
        return (
            f'<i class="hgi hgi-stroke hgi-{slug}" '
            f'style="font-size:14px;color:{color};vertical-align:middle;'
            f'display:inline-block;margin-right:0.25rem" aria-hidden="true"></i>'
        )

    score_html = ""
    if audit_result is not None:
        score = audit_result.score
        grade = getattr(audit_result, "grade", "").split()[0]
        badge_variant = "success" if score >= 85 else ("primary" if score >= 70 else ("warning" if score >= 50 else "error"))
        health_icon = "tick-02" if score >= 85 else ("alert-01" if score >= 70 else "alert-02")
        score_html = (
            f'<span class="telemetry-item">{_ti(health_icon)}Health: '
            f'<span class="badge badge-{badge_variant}">{score}/100 ({grade})</span></span>{SEP}'
        )

    target_html = ""
    if target_col:
        p_type = problem_type or "classification"
        target_html = (
            f'<span class="telemetry-item">{_ti("target-01")}Target: '
            f'<strong>{target_col}</strong> ({p_type[:5]})</span>{SEP}'
        )

    model_html = ""
    if best_model_name:
        model_html = (
            f'<span class="telemetry-item">{_ti("cpu", "var(--mf-accent)")}Best Model: '
            f'<span class="badge badge-success">{best_model_name}</span></span>{SEP}'
        )

    html = (
        '<div class="telemetry-bar">'
        + f'<span class="telemetry-item">{_ti("database-01")}Records: <strong>{n_rows:,}</strong></span>'
        + SEP
        + f'<span class="telemetry-item">{_ti("table-01")}Features: <strong>{n_cols}</strong></span>'
        + SEP
        + score_html
        + target_html
        + model_html
        + f'<span class="telemetry-item" style="color:var(--mf-text-muted);font-size:0.75rem">'
        + f'{_ti("play-circle")}Pipeline Active</span>'
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_step_header(step_num: int, title: str, subtitle: str = "") -> None:
    """Standardized modern SaaS step header with section breadcrumb and Hugeicons icon."""
    breadcrumb_map = {
        1: "WORKSPACE / DATA INGESTION",
        2: "WORKSPACE / PROFILING & HEALTH",
        3: "WORKSPACE / MISSING VALUES",
        4: "WORKSPACE / DUPLICATE RECORDS",
        5: "WORKSPACE / PREPROCESSING",
        6: "OUTPUTS / EXPORT CENTER",
        7: "MODELING / EXPERIMENT WORKSPACE",
        8: "MODELING / CLUSTER VISUALIZER",
        9: "OUTPUTS / EXECUTIVE REPORTS",
        10: "MODELING / PREDICTION PLAYGROUND",
        11: "MODELING / EXPLAINABILITY"
    }
    # Per-step icon slugs (verified Hugeicons Stroke Rounded names)
    icon_map = {
        1:  "database-import",
        2:  "analytics-01",
        3:  "filter-remove",
        4:  "layers-01",
        5:  "sliders-horizontal",
        6:  "file-export",
        7:  "neural-network",
        8:  "chart-scatter",
        9:  "analytics-02",
        10: "target-01",
        11: "eye"
    }
    bc = breadcrumb_map.get(step_num, f"PIPELINE / STEP {step_num}")
    slug = icon_map.get(step_num, "workflow-square-01")
    icon_html = (
        f'<i class="hgi hgi-stroke hgi-{slug}" '
        f'style="font-size:22px;color:var(--mf-accent);vertical-align:middle;'
        f'margin-right:0.5rem;opacity:0.9;" aria-hidden="true"></i>'
    )
    sub_html = f"<div style='color:var(--mf-text-secondary);font-size:0.85rem;margin-top:0.25rem'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f'<div class="step-header">'
        f'<div class="step-breadcrumb">{bc}</div>'
        f'<div style="font-size:1.25rem;font-weight:700;color:var(--mf-text-primary);'
        f'letter-spacing:-0.02em;margin-top:0.1rem;display:flex;align-items:center">'
        f'{icon_html}{title}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True
    )


def render_empty_state(
    title: str,
    description: str,
    cta_label: Optional[str] = None,
    cta_step: Optional[int] = None,
    icon: str = "database-import"
) -> None:
    """
    Renders a clean, actionable empty state container with a verified Hugeicon.
    """
    icon_html = (
        f'<i class="hgi hgi-stroke hgi-{icon}" '
        f'style="font-size:36px;color:var(--mf-text-muted);opacity:0.6;" aria-hidden="true"></i>'
    )
    st.markdown(
        f'<div style="background:var(--mf-bg-elevated);border:1px dashed var(--mf-border);'
        f'border-radius:10px;padding:2.5rem 1.5rem;text-align:center;margin:1.5rem 0">'
        f'<div style="margin-bottom:0.75rem">{icon_html}</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:var(--mf-text-primary);margin-bottom:0.4rem">{title}</div>'
        f'<div style="font-size:0.875rem;color:var(--mf-text-secondary);max-width:500px;margin:0 auto 1.25rem">{description}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if cta_label and cta_step is not None:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button(cta_label, key=f"empty_cta_{cta_step}", use_container_width=True, type="primary"):
                st.session_state.step = cta_step
                st.rerun()


def render_structured_error(
    title: str,
    what_happened: str,
    why: str,
    what_to_do: str,
    debug_trace: Optional[str] = None
) -> None:
    """
    Displays an actionable, empathetic error state without raw stack traces.
    """
    st.markdown(f"""<div class="workspace-card" style="border-left:3px solid var(--mf-danger);margin:1rem 0">
        <div style="font-size:1rem;font-weight:700;color:var(--mf-danger);margin-bottom:0.5rem">{title}</div>
        <div style="font-size:0.875rem;margin-bottom:0.4rem;color:var(--mf-text-primary)"><strong>What happened:</strong> {what_happened}</div>
        <div style="font-size:0.875rem;color:var(--mf-text-secondary);margin-bottom:0.4rem"><strong>Potential cause:</strong> {why}</div>
        <div style="font-size:0.875rem;color:var(--mf-accent)"><strong>Recommended action:</strong> {what_to_do}</div>
    </div>""", unsafe_allow_html=True)

    if debug_trace:
        with st.expander("Technical Traceback (For Diagnostics)", expanded=False):
            st.code(debug_trace, language="python")


def render_step_navigation(
    back_step: Optional[int] = None,
    back_label: str = "◀ Back",
    back_key: Optional[str] = None,
    next_step: Optional[int] = None,
    next_label: Optional[str] = None,
    next_key: Optional[str] = None,
    next_type: str = "primary",
    extra_action: Optional[Tuple[str, int, str]] = None,
    on_next_click: Optional[Callable[[], None]] = None,
    on_back_click: Optional[Callable[[], None]] = None,
) -> None:
    """
    Renders standardized bottom navigation for MLForge pipeline steps.
    Anchors 'Back' action to the far left and 'Next' action to the far right.
    """
    st.markdown("<div style='margin-top:1.75rem;margin-bottom:0.75rem'></div>", unsafe_allow_html=True)

    if extra_action:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown('<div class="step-nav-left"></div>', unsafe_allow_html=True)
            if back_step is not None:
                b_key = back_key or f"btn_nav_back_{back_step}_{st.session_state.get('step', 0)}"
                if st.button(back_label, key=b_key):
                    if on_back_click:
                        on_back_click()
                    st.session_state.step = back_step
                    st.rerun()
        with c2:
            extra_lbl, extra_step, extra_key = extra_action
            if st.button(extra_lbl, key=extra_key):
                st.session_state.step = extra_step
                st.rerun()
        with c3:
            st.markdown('<div class="step-nav-right"></div>', unsafe_allow_html=True)
            if next_label:
                n_key = next_key or f"btn_nav_next_{next_step}_{st.session_state.get('step', 0)}"
                if st.button(next_label, key=n_key, type=next_type):
                    if on_next_click:
                        on_next_click()
                    if next_step is not None:
                        st.session_state.step = next_step
                    st.rerun()
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="step-nav-left"></div>', unsafe_allow_html=True)
            if back_step is not None:
                b_key = back_key or f"btn_nav_back_{back_step}_{st.session_state.get('step', 0)}"
                if st.button(back_label, key=b_key):
                    if on_back_click:
                        on_back_click()
                    st.session_state.step = back_step
                    st.rerun()
        with c2:
            st.markdown('<div class="step-nav-right"></div>', unsafe_allow_html=True)
            if next_label:
                n_key = next_key or f"btn_nav_next_{next_step}_{st.session_state.get('step', 0)}"
                if st.button(next_label, key=n_key, type=next_type):
                    if on_next_click:
                        on_next_click()
                    if next_step is not None:
                        st.session_state.step = next_step
                    st.rerun()

