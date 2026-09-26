"""
MLForge - Automated Brand Identity & Asset Generator
===================================================
Generates the complete, production-grade vector (SVG) and raster (PNG/ICO)
asset suite for the MLForge platform matching the Darkmatter Remix theme:
- Colors: Warm Copper (#e78a53), Slate Teal (#5f8787), Luminous Peach (#fbcb97), Terracotta (#9e4e24)
- Typography: Geist Sans
- Primary logos (horizontal, dark, light, transparent, tagline, monochrome)
- Standalone icons (light, dark, app icon squircle)
- Multi-resolution favicons (16, 32, 48, 64, 128, 192, 512 + multi-res ICO)
"""

import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter


BASE_DIR = Path(__file__).resolve().parent.parent
BRAND_DIR = BASE_DIR / "assets" / "branding"
LOGO_DIR = BRAND_DIR / "logo"
ICON_DIR = BRAND_DIR / "icon"
FAVICON_DIR = BRAND_DIR / "favicon"

for d in [LOGO_DIR, ICON_DIR, FAVICON_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =====================================================================
# DARKMATTER REMIX BRAND COLOR PALETTES
# =====================================================================

COLOR_PRIMARY = "#e78a53"          # Signature Warm Copper Primary
COLOR_PRIMARY_HOVER = "#d87943"    # Rich Copper Accent
COLOR_SECONDARY = "#5f8787"        # Darkmatter Slate Teal
COLOR_SECONDARY_DARK = "#527575"   # Deep Slate Teal Shadow
COLOR_ACCENT_PEACH = "#fbcb97"     # Luminous Peach / Gold Highlight
COLOR_TERRACOTTA = "#9e4e24"       # Deep Terracotta Base
COLOR_TEXT_WHITE = "#f3f4f6"       # Pure Crisp White
COLOR_TEXT_DARK = "#121113"        # Dark Surface
COLOR_MUTED = "#888888"            # Slate Muted
COLOR_BG_DARK = "#121113"          # Dark Canvas Background
COLOR_BG_LIGHT = "#ffffff"         # Pure White
COLOR_SQUIRCLE_BG = "#121113"      # App Icon Surface
COLOR_SQUIRCLE_BORDER = "#e78a53"  # Copper Glow


# =====================================================================
# VECTOR SVG ASSETS
# =====================================================================

def get_symbol_svg_paths(color_left, color_center, color_right, color_top=None, color_bottom=None, stroke=None, stroke_width=0):
    stroke_attr = f'stroke="{stroke}" stroke-width="{stroke_width}" stroke-linejoin="round"' if stroke else ""
    c_top = color_top or COLOR_ACCENT_PEACH
    c_bottom = color_bottom or COLOR_TERRACOTTA
    return f"""
    <!-- Left Top Lid -->
    <path d="M 14,32 L 32,20 L 44,28 L 26,40 Z" fill="{c_top}" />
    <!-- Left Vertical Face -->
    <path d="M 14,32 L 26,40 L 26,76 L 14,68 Z" fill="{color_left}" {stroke_attr} />
    <!-- Center Descending Ribbon Fold -->
    <path d="M 26,40 L 44,28 L 62,44 L 44,56 Z" fill="{color_center}" {stroke_attr} />
    <!-- Center Ascending Ribbon Fold -->
    <path d="M 26,76 L 44,88 L 60,74 L 42,62 Z" fill="{COLOR_SECONDARY_DARK}" />
    <!-- Right Arch & Pillar -->
    <path d="M 62,44 L 78,32 L 78,74 L 66,74 L 66,54 L 44,56 Z" fill="{color_right}" {stroke_attr} />
    <!-- Bottom Cap -->
    <path d="M 14,68 L 26,76 L 36,84 L 24,76 Z" fill="{c_bottom}" />
    <!-- Floating Data Pixels -->
    <rect x="56" y="8" width="6.5" height="6.5" rx="1.2" fill="{COLOR_ACCENT_PEACH}" />
    <rect x="66" y="8" width="6.5" height="6.5" rx="1.2" fill="{COLOR_PRIMARY}" />
    <rect x="50" y="18" width="6.5" height="6.5" rx="1.2" fill="{COLOR_SECONDARY}" />
    <rect x="60" y="18" width="6.5" height="6.5" rx="1.2" fill="{COLOR_PRIMARY_HOVER}" />
    <rect x="70" y="18" width="6.5" height="6.5" rx="1.2" fill="{COLOR_ACCENT_PEACH}" />
    """


def get_symbol_svg_paths_monochrome(fill_color):
    return f"""
    <path d="M 14,32 L 32,20 L 44,28 L 26,40 Z" fill="{fill_color}" />
    <path d="M 14,32 L 26,40 L 26,76 L 14,68 Z" fill="{fill_color}" opacity="0.9" />
    <path d="M 26,40 L 44,28 L 62,44 L 44,56 Z" fill="{fill_color}" opacity="0.75" />
    <path d="M 26,76 L 44,88 L 60,74 L 42,62 Z" fill="{fill_color}" opacity="0.6" />
    <path d="M 62,44 L 78,32 L 78,74 L 66,74 L 66,54 L 44,56 Z" fill="{fill_color}" />
    <path d="M 14,68 L 26,76 L 36,84 L 24,76 Z" fill="{fill_color}" opacity="0.5" />
    <rect x="56" y="8" width="6.5" height="6.5" rx="1.2" fill="{fill_color}" />
    <rect x="66" y="8" width="6.5" height="6.5" rx="1.2" fill="{fill_color}" opacity="0.85" />
    <rect x="50" y="18" width="6.5" height="6.5" rx="1.2" fill="{fill_color}" />
    <rect x="60" y="18" width="6.5" height="6.5" rx="1.2" fill="{fill_color}" opacity="0.9" />
    <rect x="70" y="18" width="6.5" height="6.5" rx="1.2" fill="{fill_color}" opacity="0.75" />
    """


def generate_svg_logos():
    # Exact sidebar ribbon symbol paths (viewBox 0 0 100 100)
    exact_sidebar_symbol = f"""
    <!-- Left Top Lid -->
    <path d="M 14,32 L 32,20 L 44,28 L 26,40 Z" fill="{COLOR_ACCENT_PEACH}" />
    <!-- Left Vertical Face -->
    <path d="M 14,32 L 26,40 L 26,76 L 14,68 Z" fill="{COLOR_PRIMARY}" />
    <!-- Center Descending Ribbon Fold -->
    <path d="M 26,40 L 44,28 L 62,44 L 44,56 Z" fill="{COLOR_SECONDARY}" />
    <!-- Center Ascending Ribbon Fold -->
    <path d="M 26,76 L 44,88 L 60,74 L 42,62 Z" fill="{COLOR_SECONDARY_DARK}" />
    <!-- Right Arch & Pillar -->
    <path d="M 62,44 L 78,32 L 78,74 L 66,74 L 66,54 L 44,56 Z" fill="{COLOR_PRIMARY_HOVER}" />
    <!-- Bottom Cap -->
    <path d="M 14,68 L 26,76 L 36,84 L 24,76 Z" fill="{COLOR_TERRACOTTA}" />
    <!-- Floating Data Pixels -->
    <rect x="56" y="8" width="6.5" height="6.5" rx="1.2" fill="{COLOR_ACCENT_PEACH}" />
    <rect x="66" y="8" width="6.5" height="6.5" rx="1.2" fill="{COLOR_PRIMARY}" />
    <rect x="50" y="18" width="6.5" height="6.5" rx="1.2" fill="{COLOR_SECONDARY}" />
    <rect x="60" y="18" width="6.5" height="6.5" rx="1.2" fill="{COLOR_PRIMARY_HOVER}" />
    <rect x="70" y="18" width="6.5" height="6.5" rx="1.2" fill="{COLOR_ACCENT_PEACH}" />"""

    # 1. Primary Logo (Matches application sidebar logo lockup exactly)
    svg_primary = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 100" width="360" height="100" fill="none">
  <style>
    .brand-title {{
      font-family: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-weight: 700;
      font-size: 48px;
      letter-spacing: -0.03em;
      fill: {COLOR_TEXT_WHITE};
    }}
    .brand-forge {{
      fill: {COLOR_PRIMARY};
    }}
  </style>
  <g transform="translate(6, 0)">{exact_sidebar_symbol}
  </g>
  <text x="110" y="66" class="brand-title">ML<tspan class="brand-forge">Forge</tspan></text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-primary.svg").write_text(svg_primary, encoding="utf-8")
    (LOGO_DIR / "mlforge-logo-transparent.svg").write_text(svg_primary, encoding="utf-8")

    # 2. Dark Logo (For Light Backgrounds: Dark Surface Text)
    svg_dark = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 100" width="360" height="100" fill="none">
  <g transform="translate(6, 0)">{exact_sidebar_symbol}
  </g>
  <text x="110" y="66" font-family="'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="700" font-size="48" fill="{COLOR_TEXT_DARK}" letter-spacing="-0.03em">MLForge</text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-dark.svg").write_text(svg_dark, encoding="utf-8")

    # 3. Light Logo (White surface card with themed logo inside)
    svg_light = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 100" width="360" height="100" fill="none">
  <rect width="360" height="100" fill="{COLOR_BG_LIGHT}" rx="8" />
  <g transform="translate(6, 0)">{exact_sidebar_symbol}
  </g>
  <text x="110" y="66" font-family="'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="700" font-size="48" fill="{COLOR_TEXT_DARK}" letter-spacing="-0.03em">MLForge</text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-light.svg").write_text(svg_light, encoding="utf-8")

    # Gradient defs used by tagline logo
    defs_gradients = f"""  <defs>
    <linearGradient id="gradPillarLeft" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_ACCENT_PEACH}" />
      <stop offset="100%" stop-color="{COLOR_PRIMARY}" />
    </linearGradient>
    <linearGradient id="gradCrucible" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_SECONDARY}" />
      <stop offset="100%" stop-color="{COLOR_SECONDARY_DARK}" />
    </linearGradient>
    <linearGradient id="gradPillarRight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY}" />
      <stop offset="100%" stop-color="{COLOR_PRIMARY_HOVER}" />
    </linearGradient>
    <linearGradient id="gradWordForge" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY}" />
      <stop offset="100%" stop-color="{COLOR_ACCENT_PEACH}" />
    </linearGradient>
  </defs>"""

    # 4. Logo with Tagline
    svg_tagline = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 120" width="520" height="120">
{defs_gradients}
  <g transform="translate(10, 10)">
    {get_symbol_svg_paths("url(#gradPillarLeft)", "url(#gradCrucible)", "url(#gradPillarRight)")}
  </g>
  <text x="128" y="62" font-family="'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="700" font-size="42" fill="{COLOR_TEXT_WHITE}" letter-spacing="-0.03em">ML<tspan fill="url(#gradWordForge)">Forge</tspan></text>
  <text x="130" y="86" font-family="'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="600" font-size="11.5" fill="{COLOR_MUTED}" letter-spacing="0.14em">MACHINE LEARNING DEVELOPMENT PLATFORM</text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-with-tagline.svg").write_text(svg_tagline, encoding="utf-8")

    # 5. Monochrome White Logo
    svg_mono_white = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 100" width="420" height="100">
  <g transform="translate(10, 0)">
    {get_symbol_svg_paths_monochrome("#ffffff")}
  </g>
  <text x="125" y="66" font-family="'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="700" font-size="44" fill="#ffffff" letter-spacing="-0.03em">MLForge</text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-monochrome-white.svg").write_text(svg_mono_white, encoding="utf-8")

    # 6. Monochrome Dark Logo
    svg_mono_dark = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 100" width="420" height="100">
  <g transform="translate(10, 0)">
    {get_symbol_svg_paths_monochrome(COLOR_TEXT_DARK)}
  </g>
  <text x="125" y="66" font-family="'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="700" font-size="44" fill="{COLOR_TEXT_DARK}" letter-spacing="-0.03em">MLForge</text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-monochrome-dark.svg").write_text(svg_mono_dark, encoding="utf-8")


def generate_svg_icons():
    # 1. Standalone Symbol (Transparent)
    svg_icon = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <defs>
    <linearGradient id="iconLeft" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_ACCENT_PEACH}" />
      <stop offset="100%" stop-color="{COLOR_PRIMARY}" />
    </linearGradient>
    <linearGradient id="iconCrucible" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_SECONDARY}" />
      <stop offset="100%" stop-color="{COLOR_SECONDARY_DARK}" />
    </linearGradient>
    <linearGradient id="iconRight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY}" />
      <stop offset="100%" stop-color="{COLOR_PRIMARY_HOVER}" />
    </linearGradient>
  </defs>
  {get_symbol_svg_paths("url(#iconLeft)", "url(#iconCrucible)", "url(#iconRight)")}
</svg>"""
    (ICON_DIR / "mlforge-icon.svg").write_text(svg_icon, encoding="utf-8")

    # 2. Light Symbol (For Dark Background)
    svg_icon_light = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  {get_symbol_svg_paths(COLOR_PRIMARY, COLOR_SECONDARY, COLOR_PRIMARY_HOVER, color_top=COLOR_ACCENT_PEACH)}
</svg>"""
    (ICON_DIR / "mlforge-icon-light.svg").write_text(svg_icon_light, encoding="utf-8")

    # 3. Dark Symbol (For Light Background)
    svg_icon_dark = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  {get_symbol_svg_paths(COLOR_PRIMARY_HOVER, COLOR_SECONDARY_DARK, COLOR_PRIMARY_HOVER, color_top=COLOR_PRIMARY, color_bottom="#7c3714")}
</svg>"""
    (ICON_DIR / "mlforge-icon-dark.svg").write_text(svg_icon_dark, encoding="utf-8")

    # 4. App Icon (Squircle container with gradient copper border)
    svg_app_icon = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="appBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#181719" />
      <stop offset="100%" stop-color="#121113" />
    </linearGradient>
    <linearGradient id="appBorder" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY}" stop-opacity="0.6" />
      <stop offset="100%" stop-color="{COLOR_SECONDARY}" stop-opacity="0.3" />
    </linearGradient>
    <linearGradient id="appLeft" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_ACCENT_PEACH}" />
      <stop offset="100%" stop-color="{COLOR_PRIMARY}" />
    </linearGradient>
    <linearGradient id="appCrucible" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_SECONDARY}" />
      <stop offset="100%" stop-color="{COLOR_SECONDARY_DARK}" />
    </linearGradient>
    <linearGradient id="appRight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY}" />
      <stop offset="100%" stop-color="{COLOR_PRIMARY_HOVER}" />
    </linearGradient>
  </defs>
  <!-- Squircle Base -->
  <rect x="16" y="16" width="480" height="480" rx="112" fill="url(#appBg)" stroke="url(#appBorder)" stroke-width="4" />
  <!-- Center Symbol scaled to 340px centered at 256 -->
  <g transform="translate(86, 86) scale(3.4)">
    {get_symbol_svg_paths("url(#appLeft)", "url(#appCrucible)", "url(#appRight)")}
  </g>
</svg>"""
    (ICON_DIR / "mlforge-app-icon.svg").write_text(svg_app_icon, encoding="utf-8")


# =====================================================================
# RASTER ASSETS GENERATION (HIGH-RES PNG & MULTI-RES ICO)
# =====================================================================

def draw_symbol_on_image(draw, box, color_left, color_center, color_right, color_top=(251, 203, 151, 255), color_bottom=(158, 78, 36, 255)):
    """
    Renders the 3D ribbon mark with the Darkmatter Remix warm copper & slate teal palette.
    """
    bx0, by0, bx1, by1 = box
    bw = bx1 - bx0
    bh = by1 - by0

    def pt(norm_x, norm_y):
        return (bx0 + norm_x * bw / 100.0, by0 + norm_y * bh / 100.0)

    # 1. Left Vertical Face
    poly_left = [pt(14, 32), pt(26, 40), pt(26, 76), pt(14, 68)]
    # 2. Top Lid
    poly_top = [pt(14, 32), pt(32, 20), pt(44, 28), pt(26, 40)]
    # 3. Center Descending Ribbon Fold
    poly_center = [pt(26, 40), pt(44, 28), pt(62, 44), pt(44, 56)]
    # 4. Center Ascending Ribbon Fold
    poly_fold = [pt(26, 76), pt(44, 88), pt(60, 74), pt(42, 62)]
    # 5. Right Arch & Pillar
    poly_right = [pt(62, 44), pt(78, 32), pt(78, 74), pt(66, 74), pt(66, 54), pt(44, 56)]
    # 6. Bottom Cap
    poly_bottom = [pt(14, 68), pt(26, 76), pt(36, 84), pt(24, 76)]

    # Draw primary facets
    draw.polygon(poly_bottom, fill=color_bottom)
    draw.polygon(poly_left, fill=color_left)
    draw.polygon(poly_top, fill=color_top)
    draw.polygon(poly_fold, fill=(82, 117, 117, 255))
    draw.polygon(poly_center, fill=color_center)
    draw.polygon(poly_right, fill=color_right)

    # Floating data pixels
    def draw_pixel(x, y, w, h, fill):
        p0 = pt(x, y)
        p1 = pt(x + w, y + h)
        draw.rounded_rectangle([p0[0], p0[1], p1[0], p1[1]], radius=max(1, int(bw * 0.012)), fill=fill)

    draw_pixel(56, 8, 6.5, 6.5, (251, 203, 151, 255))
    draw_pixel(66, 8, 6.5, 6.5, (231, 138, 83, 255))
    draw_pixel(50, 18, 6.5, 6.5, (95, 135, 135, 255))
    draw_pixel(60, 18, 6.5, 6.5, (216, 121, 67, 255))
    draw_pixel(70, 18, 6.5, 6.5, (251, 203, 151, 255))


def generate_png_icons_and_favicons():
    """Generates all PNG icons, app squircle, and favicons from 16px to 512px."""
    master_dim = 1024

    # Theme RGB tuples
    rgb_copper = (231, 138, 83, 255)
    rgb_copper_dark = (216, 121, 67, 255)
    rgb_teal = (95, 135, 135, 255)
    rgb_peach = (251, 203, 151, 255)
    rgb_terracotta = (158, 78, 36, 255)

    # 1. Standalone Icon (Transparent)
    im_icon = Image.new("RGBA", (master_dim, master_dim), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im_icon)
    margin = 80
    draw_symbol_on_image(
        draw,
        (margin, margin, master_dim - margin, master_dim - margin),
        color_left=rgb_copper,
        color_center=rgb_teal,
        color_right=rgb_copper_dark,
        color_top=rgb_peach,
        color_bottom=rgb_terracotta
    )
    im_icon.resize((512, 512), Image.Resampling.LANCZOS).save(ICON_DIR / "mlforge-icon.png", "PNG")

    # Light & Dark icon variants
    im_icon_light = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d_l = ImageDraw.Draw(im_icon_light)
    draw_symbol_on_image(d_l, (40, 40, 472, 472), rgb_copper, rgb_teal, rgb_copper_dark, rgb_peach)
    im_icon_light.save(ICON_DIR / "mlforge-icon-light.png", "PNG")

    im_icon_dark = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d_d = ImageDraw.Draw(im_icon_dark)
    draw_symbol_on_image(d_d, (40, 40, 472, 472), rgb_copper_dark, (82, 117, 117, 255), rgb_copper_dark, rgb_copper)
    im_icon_dark.save(ICON_DIR / "mlforge-icon-dark.png", "PNG")

    # 2. App Icon (Squircle container with gradient copper border)
    im_app = Image.new("RGBA", (master_dim, master_dim), (0, 0, 0, 0))
    d_app = ImageDraw.Draw(im_app)
    # Squircle background #121113 with #e78a53 outline
    d_app.rounded_rectangle([32, 32, master_dim - 32, master_dim - 32], radius=220, fill=(18, 17, 19, 255), outline=(231, 138, 83, 140), width=8)
    draw_symbol_on_image(
        d_app,
        (190, 190, master_dim - 190, master_dim - 190),
        color_left=rgb_copper,
        color_center=rgb_teal,
        color_right=rgb_copper_dark,
        color_top=rgb_peach,
        color_bottom=rgb_terracotta
    )
    im_app.resize((512, 512), Image.Resampling.LANCZOS).save(ICON_DIR / "mlforge-app-icon.png", "PNG")

    # 3. Dedicated Favicon Family
    favicon_sizes = [16, 32, 48, 64, 128, 192, 512]
    ico_images = []

    for size in favicon_sizes:
        im_fav = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d_fav = ImageDraw.Draw(im_fav)
        pad = max(1, int(size * 0.08))
        draw_symbol_on_image(
            d_fav,
            (pad, pad, size - pad, size - pad),
            color_left=rgb_copper,
            color_center=rgb_teal,
            color_right=rgb_copper_dark,
            color_top=rgb_peach,
            color_bottom=rgb_terracotta
        )
        im_fav.save(FAVICON_DIR / f"favicon-{size}.png", "PNG")
        if size in (16, 32, 48, 64):
            ico_images.append(im_fav)

    # 4. Multi-resolution Windows/Web ICO file
    ico_images[0].save(
        FAVICON_DIR / "favicon.ico",
        format="ICO",
        sizes=[(img.width, img.height) for img in ico_images]
    )


def generate_png_logos():
    """Generates high-resolution PNG exports for all logo variants."""
    rgb_copper = (231, 138, 83, 255)
    rgb_copper_dark = (216, 121, 67, 255)
    rgb_teal = (95, 135, 135, 255)
    rgb_peach = (251, 203, 151, 255)
    rgb_terracotta = (158, 78, 36, 255)

    def render_horizontal_logo(bg_color, text_ml_color, text_forge_color, symbol_colors, tagline=None):
        w, h = (1040, 240) if not tagline else (1040, 260)
        im = Image.new("RGBA", (w, h), bg_color if bg_color else (0, 0, 0, 0))
        d = ImageDraw.Draw(im)

        # Draw symbol
        sym_box = (30, 30, 210, 210)
        draw_symbol_on_image(
            d, sym_box,
            color_left=symbol_colors[0],
            color_center=symbol_colors[1],
            color_right=symbol_colors[2],
            color_top=symbol_colors[3] if len(symbol_colors) > 3 else rgb_peach,
            color_bottom=symbol_colors[4] if len(symbol_colors) > 4 else rgb_terracotta
        )

        # Draw wordmark text via PIL
        font_path = "C:/Windows/Fonts/segoeui.ttf"
        try:
            f_main = ImageFont.truetype(font_path, 96)
            f_tag = ImageFont.truetype(font_path, 22)
        except Exception:
            f_main = ImageFont.load_default()
            f_tag = ImageFont.load_default()

        # Wordmark: "MLForge"
        text_x = 245
        text_y = 60 if not tagline else 50

        # Draw "ML"
        d.text((text_x, text_y), "ML", fill=text_ml_color, font=f_main)
        # Bounding width of "ML"
        ml_bbox = d.textbbox((text_x, text_y), "ML", font=f_main)
        forge_x = ml_bbox[2] + 2
        d.text((forge_x, text_y), "Forge", fill=text_forge_color, font=f_main)

        if tagline:
            d.text((text_x + 2, text_y + 115), tagline, fill=(136, 136, 136, 255), font=f_tag)

        return im

    # 1. Primary & Transparent (Transparent background, white + warm copper text)
    im_prim = render_horizontal_logo(
        None,
        (243, 244, 246, 255),
        rgb_copper,
        (rgb_copper, rgb_teal, rgb_copper_dark, rgb_peach, rgb_terracotta)
    )
    im_prim.save(LOGO_DIR / "mlforge-logo-primary.png", "PNG")
    im_prim.save(LOGO_DIR / "mlforge-logo-transparent.png", "PNG")

    # 2. Dark Logo (for light backgrounds, dark surface text + warm copper)
    im_dark = render_horizontal_logo(
        None,
        (18, 17, 19, 255),
        rgb_copper,
        (rgb_copper_dark, (82, 117, 117, 255), rgb_copper_dark, rgb_copper, rgb_terracotta)
    )
    im_dark.save(LOGO_DIR / "mlforge-logo-dark.png", "PNG")

    # 3. Light Logo (clean white card background)
    im_light = render_horizontal_logo(
        (255, 255, 255, 255),
        (18, 17, 19, 255),
        rgb_copper,
        (rgb_copper_dark, (82, 117, 117, 255), rgb_copper_dark, rgb_copper, rgb_terracotta)
    )
    im_light.save(LOGO_DIR / "mlforge-logo-light.png", "PNG")

    # 4. Logo with Tagline
    im_tag = render_horizontal_logo(
        None,
        (243, 244, 246, 255),
        rgb_copper,
        (rgb_copper, rgb_teal, rgb_copper_dark, rgb_peach, rgb_terracotta),
        tagline="MACHINE LEARNING DEVELOPMENT PLATFORM"
    )
    im_tag.save(LOGO_DIR / "mlforge-logo-with-tagline.png", "PNG")

    # 5. Monochrome White
    im_mono_w = render_horizontal_logo(
        None,
        (255, 255, 255, 255),
        (255, 255, 255, 255),
        ((255, 255, 255, 255), (255, 255, 255, 255), (255, 255, 255, 255), (255, 255, 255, 255), (255, 255, 255, 255))
    )
    im_mono_w.save(LOGO_DIR / "mlforge-logo-monochrome-white.png", "PNG")

    # 6. Monochrome Dark
    im_mono_d = render_horizontal_logo(
        None,
        (18, 17, 19, 255),
        (18, 17, 19, 255),
        ((18, 17, 19, 255), (18, 17, 19, 255), (18, 17, 19, 255), (18, 17, 19, 255), (18, 17, 19, 255))
    )
    im_mono_d.save(LOGO_DIR / "mlforge-logo-monochrome-dark.png", "PNG")


def main():
    print("Generating SVG vector logos with Darkmatter Remix palette...")
    generate_svg_logos()
    print("Generating SVG vector icons with Darkmatter Remix palette...")
    generate_svg_icons()
    print("Generating raster icons and multi-resolution favicons...")
    generate_png_icons_and_favicons()
    print("Generating raster PNG logos...")
    generate_png_logos()
    print("All MLForge brand assets regenerated successfully!")


if __name__ == "__main__":
    main()
