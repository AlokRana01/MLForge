import os
from pathlib import Path

BASE_DIR = Path(r"c:\Users\alok rana\Desktop\ML-Studio-Pro")
BRAND_DIR = BASE_DIR / "assets" / "branding"

# Build the exact vector symbol paths matching the user's uploaded logo
def get_mlforge_hex_ribbon_svg(color_left="url(#gradLeftPillar)", 
                               color_center="url(#gradCenterRibbon)", 
                               color_right="url(#gradRightPillar)",
                               color_top="url(#gradLeftTop)",
                               mono=False):
    if mono:
        return """
    <!-- Left Pillar -->
    <path d="M 14,32 L 32,20 L 44,28 L 26,40 L 26,76 L 14,68 Z" fill="#ffffff" />
    <!-- Center Ribbon -->
    <path d="M 26,40 L 44,28 L 62,44 L 44,56 Z" fill="#ffffff" opacity="0.8" />
    <path d="M 26,76 L 44,88 L 60,74 L 42,62 Z" fill="#ffffff" opacity="0.6" />
    <!-- Right Pillar -->
    <path d="M 62,44 L 78,32 L 78,74 L 66,74 L 66,54 L 44,56 Z" fill="#ffffff" opacity="0.9" />
    <!-- Floating Data Cubes -->
    <rect x="52" y="8" width="6.5" height="6.5" rx="1.2" fill="#ffffff" />
    <rect x="62" y="8" width="6.5" height="6.5" rx="1.2" fill="#ffffff" opacity="0.8" />
    <rect x="46" y="18" width="6.5" height="6.5" rx="1.2" fill="#ffffff" />
    <rect x="56" y="18" width="6.5" height="6.5" rx="1.2" fill="#ffffff" opacity="0.9" />
    <rect x="66" y="18" width="6.5" height="6.5" rx="1.2" fill="#ffffff" opacity="0.7" />
        """

    return f"""
    <!-- Left Top Facet -->
    <path d="M 14,32 L 32,20 L 44,28 L 26,40 Z" fill="{color_top}" />
    <!-- Left Vertical Face -->
    <path d="M 14,32 L 26,40 L 26,76 L 14,68 Z" fill="{color_left}" />
    <!-- Center Descending Ribbon Fold -->
    <path d="M 26,40 L 44,28 L 62,44 L 44,56 Z" fill="{color_center}" />
    <!-- Center Ascending Ribbon Fold -->
    <path d="M 26,76 L 44,88 L 60,74 L 42,62 Z" fill="url(#gradBottomFold)" />
    <!-- Right Arch & Pillar -->
    <path d="M 62,44 L 78,32 L 78,74 L 66,74 L 66,54 L 44,56 Z" fill="{color_right}" />
    <!-- Bottom Left Cap -->
    <path d="M 14,68 L 26,76 L 36,84 L 24,76 Z" fill="#0052cc" />

    <!-- 5 Floating Data / Transformation Pixels -->
    <rect x="56" y="8" width="6.5" height="6.5" rx="1.2" fill="#00e5ff" />
    <rect x="66" y="8" width="6.5" height="6.5" rx="1.2" fill="#a855f7" />
    <rect x="50" y="18" width="6.5" height="6.5" rx="1.2" fill="#00d2ff" />
    <rect x="60" y="18" width="6.5" height="6.5" rx="1.2" fill="#38bdf8" />
    <rect x="70" y="18" width="6.5" height="6.5" rx="1.2" fill="#c084fc" />
    """

print("Helper defined.")
