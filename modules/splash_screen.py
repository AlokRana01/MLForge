"""
MLForge - Professional Startup Splash Screen
============================================
Provides an enterprise SaaS desktop-grade startup splash experience:
- High-performance, GPU-accelerated CSS keyframe animations.
- Official MLForge 3D ribbon isometric brand logo.
- Polished Darkmatter Remix aesthetic (#121113, #e78a53, #5f8787).
- Session-state guarded: runs exactly once per user session, never on reruns.
- Accessible: respects prefers-reduced-motion.
- 100% offline: zero external network dependencies, CDNs, or remote assets.
"""

from typing import Optional
import streamlit as st
from modules.fonts import get_font_face_css
from modules.ui_theme import get_brand_symbol_svg


def _clean_markup(text: str) -> str:
    """Strips leading and trailing whitespace from every line so Markdown does not treat lines as code blocks."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def render_splash_css(duration_sec: float = 4.0) -> str:
    """Generates scoped, self-contained CSS for the startup splash screen."""
    fade_start = max(0.5, duration_sec - 0.55)
    reveal_start = max(0.5, duration_sec - 0.50)
    fill_duration = max(1.0, duration_sec - 1.5)
    font_face_rules = get_font_face_css()

    raw_css = f"""
<style id="mlforge-splash-styles">
/* -------------------------------------------------------------
   LOCAL OFFLINE TYPOGRAPHY (GEIST SANS & GEIST MONO)
   ------------------------------------------------------------- */
{font_face_rules}

/* -------------------------------------------------------------
   MLFORGE STARTUP SPLASH SCREEN STYLING
   ------------------------------------------------------------- */
#mlforge-splash-wrapper {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    width: 100vw;
    height: 100vh;
    z-index: 999999999;
    background-color: #121113;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    user-select: none;
    pointer-events: all;
    animation: splashFadeOut 0.55s cubic-bezier(0.16, 1, 0.3, 1) {fade_start:.2f}s forwards;
    overflow: hidden;
}}

/* Atmospheric ambient backdrop radial glow */
#mlforge-splash-wrapper::before {{
    content: '';
    position: absolute;
    width: 700px;
    height: 500px;
    border-radius: 50%;
    background: radial-gradient(
        ellipse at center,
        rgba(231, 138, 83, 0.08) 0%,
        rgba(95, 135, 135, 0.05) 45%,
        transparent 70%
    );
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    z-index: 0;
}}

.mlforge-splash-content {{
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}}

/* Logo Entrance */
.mlforge-splash-logo {{
    opacity: 0;
    transform: scale(0.86);
    filter: drop-shadow(0 0 20px rgba(231, 138, 83, 0.28));
    animation: splashLogoEntrance 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.05s both;
    margin-bottom: 1.1rem;
}}

/* Brand Name */
.mlforge-splash-title {{
    font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 2.35rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    color: #f3f4f6 !important;
    line-height: 1;
    margin: 0 0 0.45rem 0;
    opacity: 0;
    transform: translateY(12px);
    animation: splashTextEntrance 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.20s both;
}}

.mlforge-splash-title span {{
    color: #e78a53 !important;
}}

/* Tagline */
.mlforge-splash-tagline {{
    font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #888888 !important;
    margin: 0 0 2.2rem 0;
    opacity: 0;
    transform: translateY(8px);
    animation: splashTextEntrance 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.35s both;
}}

/* Progress Track */
.mlforge-splash-progress-track {{
    width: 190px;
    height: 3px;
    background: #1c1b1e;
    border-radius: 999px;
    overflow: hidden;
    position: relative;
    opacity: 0;
    border: 1px solid rgba(255, 255, 255, 0.05);
    animation: splashProgressTrackFade 0.4s ease 0.50s both;
}}

/* Progress Fill Bar */
.mlforge-splash-progress-bar {{
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #5f8787 0%, #e78a53 60%, #fbcb97 100%);
    border-radius: 999px;
    box-shadow: 0 0 10px rgba(231, 138, 83, 0.55);
    animation: splashProgressFill {fill_duration:.2f}s cubic-bezier(0.4, 0, 0.2, 1) 0.55s both;
}}

/* Status Label */
.mlforge-splash-status {{
    font-family: 'Geist Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    color: #666666 !important;
    margin-top: 0.9rem;
    height: 14px;
    opacity: 0;
    animation: splashStatusText {fill_duration:.2f}s ease 0.55s both;
}}

/* While splash wrapper is present: completely hide sidebar, collapsed control, header */
body:has(#mlforge-splash-wrapper) section[data-testid="stSidebar"],
body:has(#mlforge-splash-wrapper) [data-testid="stSidebar"],
body:has(#mlforge-splash-wrapper) [data-testid="stSidebarCollapsedControl"],
body:has(#mlforge-splash-wrapper) [data-testid="stHeader"] {{
    display: none !important;
    visibility: hidden !important;
}}

body:has(#mlforge-splash-wrapper) [data-testid="stAppViewContainer"] > .main,
body:has(#mlforge-splash-wrapper) [data-testid="stMain"],
body:has(#mlforge-splash-wrapper) section.main {{
    margin-left: 0 !important;
    width: 100vw !important;
    max-width: 100vw !important;
}}


/* -------------------------------------------------------------

   KEYFRAME ANIMATIONS
   ------------------------------------------------------------- */
@keyframes splashLogoEntrance {{
    0% {{
        opacity: 0;
        transform: scale(0.86);
    }}
    100% {{
        opacity: 1;
        transform: scale(1);
    }}
}}

@keyframes splashTextEntrance {{
    0% {{
        opacity: 0;
        transform: translateY(12px);
    }}
    100% {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes splashProgressTrackFade {{
    0% {{ opacity: 0; }}
    100% {{ opacity: 1; }}
}}

@keyframes splashProgressFill {{
    0% {{
        width: 0%;
    }}
    30% {{
        width: 42%;
    }}
    70% {{
        width: 82%;
    }}
    100% {{
        width: 100%;
    }}
}}

@keyframes splashStatusText {{
    0% {{
        opacity: 0;
    }}
    20% {{
        opacity: 1;
    }}
    80% {{
        opacity: 1;
    }}
    100% {{
        opacity: 0.8;
    }}
}}

@keyframes splashFadeOut {{
    0% {{
        opacity: 1;
        transform: scale(1);
        pointer-events: all;
        visibility: visible;
    }}
    99% {{
        opacity: 0;
        transform: scale(1.015);
        pointer-events: none;
        visibility: visible;
    }}
    100% {{
        opacity: 0;
        transform: scale(1.015);
        pointer-events: none;
        visibility: hidden;
        display: none;
    }}
}}

/* -------------------------------------------------------------
   ACCESSIBILITY: PREFERS REDUCED MOTION
   ------------------------------------------------------------- */
@media (prefers-reduced-motion: reduce) {{
    #mlforge-splash-wrapper {{
        animation: splashFadeOut 0.3s ease 1.0s forwards !important;
    }}
    .mlforge-splash-logo,
    .mlforge-splash-title,
    .mlforge-splash-tagline,
    .mlforge-splash-progress-track,
    .mlforge-splash-progress-bar,
    .mlforge-splash-status {{
        animation: none !important;
        opacity: 1 !important;
        transform: none !important;
        transition: none !important;
    }}
    .mlforge-splash-progress-bar {{
        width: 100% !important;
    }}

}}
</style>
"""
    return _clean_markup(raw_css)


def render_splash_html(duration_sec: float = 4.0) -> str:
    """Generates the semantic HTML markup for the startup splash screen."""
    brand_logo_svg = get_brand_symbol_svg(size=76, theme="dark")
    css = render_splash_css(duration_sec)

    raw_html = f"""
<div id="mlforge-splash-container">
{css}
<div id="mlforge-splash-wrapper" role="dialog" aria-label="MLForge Startup Screen" aria-modal="true">
<div class="mlforge-splash-content">
<div class="mlforge-splash-logo">
{brand_logo_svg}
</div>
<h1 class="mlforge-splash-title">ML<span>Forge</span></h1>
<div class="mlforge-splash-tagline">AutoML Workspace</div>
<div class="mlforge-splash-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="100">
<div class="mlforge-splash-progress-bar"></div>
</div>
<div class="mlforge-splash-status">Initializing workspace engine...</div>
</div>
</div>
</div>
"""
    return _clean_markup(raw_html)


def show_splash_screen(duration_sec: float = 4.0, force: bool = False) -> bool:
    """
    Renders the MLForge startup splash screen if not already completed in this session.

    Parameters:
    - duration_sec (float): How long the splash screen stays active before fading out (~3.5-4.0s).
    - force (bool): If True, forces the splash to show even if already completed.

    Returns:
    - bool: True if splash was rendered, False if skipped because it already ran.
    """
    qp = st.query_params
    query_force = qp.get("splash_demo") in ["1", "true", "True"] or qp.get("splash") in ["1", "true", "True"]

    if "splash_completed" not in st.session_state:
        st.session_state.splash_completed = False

    if not st.session_state.splash_completed or force or query_force:
        # Mark as completed so any future interaction / widget rerun will NOT re-trigger the splash
        st.session_state.splash_completed = True
        html = render_splash_html(duration_sec=duration_sec)
        st.markdown(html, unsafe_allow_html=True)
        return True

    return False
