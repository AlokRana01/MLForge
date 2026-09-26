"""
MLForge - Centralized Hugeicons Icon System
============================================
Single source of truth for all application icons.

Icon font: Hugeicons Stroke Rounded
CDN:       https://use.hugeicons.com/font/icons.css
CSS class: hgi hgi-stroke hgi-{icon-name}

All icon names verified against the official Hugeicons catalog.
Do NOT add icons without verifying the exact class name in the catalog.

MLForge SaaS Color Palette:
  Muted (default):  var(--mf-text-muted)       #64748b
  Normal:           var(--mf-text-secondary)  #94a3b8
  Active/Accent:    var(--mf-accent)          #38bdf8
  Strong:           var(--mf-accent-hover)    #0284c7
"""

from typing import Optional


# =========================================================================
# VERIFIED ICON MAPPING  (all names checked against icons.css)
# =========================================================================

MLFORGE_ICONS: dict = {
    # --- Pipeline Steps ---
    "data_ingestion":    "database-import",
    "profiling":         "analytics-01",
    "missing_values":    "filter-remove",
    "duplicate_records": "layers-01",
    "preprocessing":     "sliders-horizontal",
    "export_clean":      "folder-export",
    "model_training":    "neural-network",
    "automl":            "magic-wand-01",
    "cross_validation":  "refresh-ccw",
    "hyperparameter":    "sliders-vertical",
    "cluster":           "chart-scatter",
    "explainability":    "eye",
    "shap":              "chart-bar-increasing",
    "prediction":        "target-01",
    "reports":           "analytics-02",
    "export_center":     "file-export",
    "settings":          "settings-01",

    # --- Actions ---
    "upload":            "upload-01",
    "download":          "download-01",
    "search":            "search-01",
    "filter":            "filter-horizontal",
    "refresh":           "refresh-ccw",
    "delete":            "trash",
    "edit":              "pencil-edit-01",
    "run":               "play-circle",
    "stop":              "stop-circle",
    "loading":           "loading-01",
    "import":            "import",
    "rocket":            "rocket-01",

    # --- Entities ---
    "dataset":           "database-01",
    "model":             "cpu",
    "feature":           "table-01",
    "chart_bar":         "chart-bar-line",
    "chart_line":        "chart-line",
    "chart_pie":         "pie-chart-01",
    "chart_scatter":     "chart-scatter",
    "cluster_grid":      "grid2x2",
    "workflow":          "workflow-square-01",
    "ai":                "ai-sparkles",
    "layers":            "layers-01",
    "test_tube":         "test-tube-01",
    "atom":              "atom-01",
    "wand":              "magic-wand-01",
    "grid":              "grid-view",
    "folder":            "folder-open",

    # --- Status ---
    "success":           "tick-02",
    "warning":           "alert-01",
    "error":             "alert-02",
    "info":              "information-circle",
    "locked":            "lock",
    "completed":         "tick-double-02",
    "active":            "play-circle",
    "inactive":          "stop-circle",

    # --- Navigation ---
    "arrow_right":       "arrow-right-01",
    "arrow_down":        "arrow-down-01",
    "home":              "home-09",
    "back":              "arrow-left-01",
}


# =========================================================================
# SIZE PRESETS  (MLForge SaaS Design System)
# =========================================================================

ICON_SIZE = {
    "nav":     19,   # Sidebar navigation
    "status":  16,   # Status / badge indicators
    "card":    22,   # Card / panel icons
    "header":  24,   # Page header icons
    "hero":    28,   # Hero / large display
    "button":  15,   # Inline button icons
    "inline":  13,   # Text-inline micro icons
}


# =========================================================================
# COLOR PRESETS
# =========================================================================

ICON_COLOR = {
    "muted":     "var(--muted)",
    "secondary": "var(--text-secondary)",
    "accent":    "var(--accent)",
    "strong":    "var(--accent-hover)",
    "success":   "var(--success)",
    "warning":   "var(--warning)",
    "error":     "var(--error)",
    "info":      "var(--accent)",
    "text":      "var(--text)",
    "inherit":   "inherit",
}


# =========================================================================
# CORE RENDER FUNCTION
# =========================================================================

def hgi(
    name: str,
    size: int = 18,
    color: Optional[str] = None,
    extra_style: str = "",
    title: str = "",
    css_class: str = "",
) -> str:
    """
    Render a Hugeicons Stroke Rounded icon as an HTML <i> element.

    Args:
        name:        Icon key from MLFORGE_ICONS, or raw Hugeicons slug.
        size:        Font-size in pixels.
        color:       Key from ICON_COLOR dict, or any CSS color value.
        extra_style: Additional inline CSS string.
        title:       Accessibility title/aria-label.
        css_class:   Extra CSS classes to add.

    Returns:
        HTML string for the icon <i> element.

    Examples:
        hgi("dataset")                     # uses MLFORGE_ICONS mapping
        hgi("database-01", size=22)        # direct Hugeicons slug
        hgi("success", color="success")    # named color preset
    """
    slug = MLFORGE_ICONS.get(name, name)

    if color is None:
        color_css = ICON_COLOR["muted"]
    elif color in ICON_COLOR:
        color_css = ICON_COLOR[color]
    else:
        color_css = color

    parts = [
        f"font-size:{size}px",
        f"color:{color_css}",
        "line-height:1",
        "vertical-align:middle",
        "flex-shrink:0",
        "display:inline-block",
    ]
    if extra_style:
        parts.append(extra_style.rstrip(";"))

    style = ";".join(parts)
    title_attr = f' title="{title}"' if title else ""
    aria_attr = ' aria-hidden="true"' if not title else f' role="img" aria-label="{title}"'
    classes = f"hgi hgi-stroke hgi-{slug}"
    if css_class:
        classes += f" {css_class}"

    return f'<i class="{classes}" style="{style}"{title_attr}{aria_attr}></i>'


# =========================================================================
# CONVENIENCE HELPERS
# =========================================================================

def nav_icon(step_key: str, active: bool = False) -> str:
    """Sidebar navigation icon."""
    color = "accent" if active else "muted"
    return hgi(step_key, size=ICON_SIZE["nav"], color=color,
               extra_style="margin-right:0.45rem")


def status_icon(kind: str) -> str:
    """Status indicator (success/warning/error/info/locked/completed)."""
    color_map = {"success": "success", "warning": "warning",
                 "error": "error", "info": "info",
                 "locked": "muted", "completed": "success"}
    return hgi(kind, size=ICON_SIZE["status"],
               color=color_map.get(kind, "muted"),
               extra_style="margin-right:0.25rem")


def card_icon(key: str, color: str = "accent") -> str:
    """Card / panel header icon."""
    return hgi(key, size=ICON_SIZE["card"], color=color,
               extra_style="margin-right:0.45rem")


def header_icon(key: str, color: str = "accent") -> str:
    """Page step-header icon."""
    return hgi(key, size=ICON_SIZE["header"], color=color,
               extra_style="margin-right:0.5rem;opacity:0.9")


def inline_icon(key: str, color: str = "muted") -> str:
    """Tiny inline icon next to text."""
    return hgi(key, size=ICON_SIZE["inline"], color=color,
               extra_style="margin-right:0.2rem;opacity:0.85")


def icon_label(icon_key: str, text: str, size: int = 18,
               color: str = "muted", gap: str = "0.4rem") -> str:
    """
    Returns HTML for an icon + text pair in a flex row.
    Use inside st.markdown(..., unsafe_allow_html=True).
    """
    icon_html = hgi(icon_key, size=size, color=color)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:{gap}">'
        f'{icon_html}<span>{text}</span></span>'
    )


# =========================================================================
# FONT INJECTION  (call once at app startup)
# =========================================================================

HUGEICONS_CDN = "https://use.hugeicons.com/font/icons.css"


def inject_hugeicons_css() -> str:
    """
    Returns the HTML block to load Hugeicons web font.
    Inject once via st.markdown(..., unsafe_allow_html=True).
    """
    return (
        '<link rel="preconnect" href="https://use.hugeicons.com">'
        f'<link rel="stylesheet" href="{HUGEICONS_CDN}">'
        '<style>'
        '.hgi{display:inline-block;line-height:1;vertical-align:middle;}'
        '.hgi-stroke{font-style:normal!important;}'
        '</style>'
    )
