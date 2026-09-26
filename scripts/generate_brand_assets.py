"""
MLForge - Automated Brand Identity & Asset Generator
===================================================
Generates the complete, production-grade vector (SVG) and raster (PNG/ICO)
asset suite for the MLForge platform:
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
# BRAND COLOR PALETTES
# =====================================================================

COLOR_PRIMARY_SKY = "#38bdf8"       # Electric Sky 400
COLOR_DEEP_SKY = "#0284c7"          # Sky 600
COLOR_INDIGO = "#6366f1"            # Indigo 500
COLOR_DARK_INDIGO = "#4338ca"       # Indigo 700
COLOR_TEXT_WHITE = "#f8fafc"        # Slate 50
COLOR_TEXT_DARK = "#0f172a"         # Slate 900
COLOR_MUTED = "#94a3b8"             # Slate 400
COLOR_BG_DARK = "#090d16"           # Slate 950 / Dark Surface
COLOR_BG_LIGHT = "#ffffff"          # Pure White
COLOR_SQUIRCLE_BG = "#0f172a"       # Slate 900
COLOR_SQUIRCLE_BORDER = "#1e293b"   # Slate 800


# =====================================================================
# VECTOR SVG ASSETS
# =====================================================================

# Standalone Symbol SVG Path geometry (ViewBox 0 0 100 100)
# Official 3D Hexagonal Ribbon M with Floating Data Transformation Pixels
def get_symbol_svg_paths(color_left, color_center, color_right, stroke=None, stroke_width=0):
    stroke_attr = f'stroke="{stroke}" stroke-width="{stroke_width}" stroke-linejoin="round"' if stroke else ""
    return f"""
    <!-- Left Top Lid -->
    <path d="M 14,32 L 32,20 L 44,28 L 26,40 Z" fill="#38bdf8" />
    <!-- Left Vertical Face -->
    <path d="M 14,32 L 26,40 L 26,76 L 14,68 Z" fill="{color_left}" {stroke_attr} />
    <!-- Center Descending Ribbon Fold -->
    <path d="M 26,40 L 44,28 L 62,44 L 44,56 Z" fill="{color_center}" {stroke_attr} />
    <!-- Center Ascending Ribbon Fold -->
    <path d="M 26,76 L 44,88 L 60,74 L 42,62 Z" fill="#2563eb" />
    <!-- Right Arch & Pillar -->
    <path d="M 62,44 L 78,32 L 78,74 L 66,74 L 66,54 L 44,56 Z" fill="{color_right}" {stroke_attr} />
    <!-- Bottom Cap -->
    <path d="M 14,68 L 26,76 L 36,84 L 24,76 Z" fill="#0052cc" />
    <!-- Floating Data Pixels -->
    <rect x="56" y="8" width="6.5" height="6.5" rx="1.2" fill="#00e5ff" />
    <rect x="66" y="8" width="6.5" height="6.5" rx="1.2" fill="#a855f7" />
    <rect x="50" y="18" width="6.5" height="6.5" rx="1.2" fill="#00d2ff" />
    <rect x="60" y="18" width="6.5" height="6.5" rx="1.2" fill="#38bdf8" />
    <rect x="70" y="18" width="6.5" height="6.5" rx="1.2" fill="#c084fc" />
    """


def generate_svg_logos():
    # 1. Primary Logo (Default Dark Theme: Sky/Indigo on Transparent, White Text)
    svg_primary = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 100" width="420" height="100">
  <defs>
    <linearGradient id="gradPillarLeft" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY_SKY}" />
      <stop offset="100%" stop-color="{COLOR_DEEP_SKY}" />
    </linearGradient>
    <linearGradient id="gradCrucible" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY_SKY}" />
      <stop offset="100%" stop-color="{COLOR_INDIGO}" />
    </linearGradient>
    <linearGradient id="gradPillarRight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_INDIGO}" />
      <stop offset="100%" stop-color="{COLOR_DARK_INDIGO}" />
    </linearGradient>
  </defs>
  <g transform="translate(10, 0)">
    {get_symbol_svg_paths("url(#gradPillarLeft)", "url(#gradCrucible)", "url(#gradPillarRight)")}
  </g>
  <text x="125" y="66" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="44" fill="{COLOR_TEXT_WHITE}" letter-spacing="-0.03em">ML<tspan fill="{COLOR_PRIMARY_SKY}">Forge</tspan></text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-primary.svg").write_text(svg_primary, encoding="utf-8")
    (LOGO_DIR / "mlforge-logo-transparent.svg").write_text(svg_primary, encoding="utf-8")

    # 2. Light Theme Logo (For Light Backgrounds: Deep Sky/Indigo, Slate-900 Text)
    svg_dark = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 100" width="420" height="100">
  <defs>
    <linearGradient id="gradLightLeft" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_DEEP_SKY}" />
      <stop offset="100%" stop-color="#0369a1" />
    </linearGradient>
    <linearGradient id="gradLightCrucible" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_DEEP_SKY}" />
      <stop offset="100%" stop-color="{COLOR_INDIGO}" />
    </linearGradient>
    <linearGradient id="gradLightRight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_INDIGO}" />
      <stop offset="100%" stop-color="{COLOR_DARK_INDIGO}" />
    </linearGradient>
  </defs>
  <g transform="translate(10, 0)">
    {get_symbol_svg_paths("url(#gradLightLeft)", "url(#gradLightCrucible)", "url(#gradLightRight)")}
  </g>
  <text x="125" y="66" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="44" fill="{COLOR_TEXT_DARK}" letter-spacing="-0.03em">ML<tspan fill="{COLOR_DEEP_SKY}">Forge</tspan></text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-dark.svg").write_text(svg_dark, encoding="utf-8")

    # 3. Light Logo (White surface container with dark logo inside)
    svg_light = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 100" width="420" height="100">
  <rect width="420" height="100" fill="{COLOR_BG_LIGHT}" rx="8" />
  <g transform="translate(10, 0)">
    {get_symbol_svg_paths(COLOR_DEEP_SKY, COLOR_INDIGO, COLOR_DARK_INDIGO)}
  </g>
  <text x="125" y="66" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="44" fill="{COLOR_TEXT_DARK}" letter-spacing="-0.03em">ML<tspan fill="{COLOR_DEEP_SKY}">Forge</tspan></text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-light.svg").write_text(svg_light, encoding="utf-8")

    # 4. Logo with Secondary Tagline
    svg_tagline = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 120" width="520" height="120">
  <defs>
    <linearGradient id="gradTagLeft" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY_SKY}" />
      <stop offset="100%" stop-color="{COLOR_DEEP_SKY}" />
    </linearGradient>
    <linearGradient id="gradTagCrucible" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY_SKY}" />
      <stop offset="100%" stop-color="{COLOR_INDIGO}" />
    </linearGradient>
    <linearGradient id="gradTagRight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_INDIGO}" />
      <stop offset="100%" stop-color="{COLOR_DARK_INDIGO}" />
    </linearGradient>
  </defs>
  <g transform="translate(10, 10)">
    {get_symbol_svg_paths("url(#gradTagLeft)", "url(#gradTagCrucible)", "url(#gradTagRight)")}
  </g>
  <text x="128" y="62" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="42" fill="{COLOR_TEXT_WHITE}" letter-spacing="-0.03em">ML<tspan fill="{COLOR_PRIMARY_SKY}">Forge</tspan></text>
  <text x="130" y="86" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="600" font-size="11.5" fill="{COLOR_MUTED}" letter-spacing="0.14em">MACHINE LEARNING DEVELOPMENT PLATFORM</text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-with-tagline.svg").write_text(svg_tagline, encoding="utf-8")

    # 5. Monochrome White Logo
    svg_mono_white = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 100" width="420" height="100">
  <g transform="translate(10, 0)">
    {get_symbol_svg_paths("#ffffff", "#ffffff", "#ffffff")}
  </g>
  <text x="125" y="66" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="44" fill="#ffffff" letter-spacing="-0.03em">MLForge</text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-monochrome-white.svg").write_text(svg_mono_white, encoding="utf-8")

    # 6. Monochrome Dark Logo
    svg_mono_dark = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 100" width="420" height="100">
  <g transform="translate(10, 0)">
    {get_symbol_svg_paths(COLOR_TEXT_DARK, COLOR_TEXT_DARK, COLOR_TEXT_DARK)}
  </g>
  <text x="125" y="66" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="44" fill="{COLOR_TEXT_DARK}" letter-spacing="-0.03em">MLForge</text>
</svg>"""
    (LOGO_DIR / "mlforge-logo-monochrome-dark.svg").write_text(svg_mono_dark, encoding="utf-8")


def generate_svg_icons():
    # 1. Standalone Symbol (Transparent)
    svg_icon = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <defs>
    <linearGradient id="iconLeft" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY_SKY}" />
      <stop offset="100%" stop-color="{COLOR_DEEP_SKY}" />
    </linearGradient>
    <linearGradient id="iconCrucible" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY_SKY}" />
      <stop offset="100%" stop-color="{COLOR_INDIGO}" />
    </linearGradient>
    <linearGradient id="iconRight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_INDIGO}" />
      <stop offset="100%" stop-color="{COLOR_DARK_INDIGO}" />
    </linearGradient>
  </defs>
  {get_symbol_svg_paths("url(#iconLeft)", "url(#iconCrucible)", "url(#iconRight)")}
</svg>"""
    (ICON_DIR / "mlforge-icon.svg").write_text(svg_icon, encoding="utf-8")

    # 2. Light Symbol (For Dark Background)
    svg_icon_light = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  {get_symbol_svg_paths(COLOR_PRIMARY_SKY, COLOR_PRIMARY_SKY, COLOR_INDIGO)}
</svg>"""
    (ICON_DIR / "mlforge-icon-light.svg").write_text(svg_icon_light, encoding="utf-8")

    # 3. Dark Symbol (For Light Background)
    svg_icon_dark = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  {get_symbol_svg_paths(COLOR_DEEP_SKY, COLOR_DEEP_SKY, COLOR_DARK_INDIGO)}
</svg>"""
    (ICON_DIR / "mlforge-icon-dark.svg").write_text(svg_icon_dark, encoding="utf-8")

    # 4. App Icon (Squircle container with gradient micro-border)
    svg_app_icon = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="appBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#020617" />
    </linearGradient>
    <linearGradient id="appBorder" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.5" />
      <stop offset="100%" stop-color="#6366f1" stop-opacity="0.2" />
    </linearGradient>
    <linearGradient id="appLeft" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY_SKY}" />
      <stop offset="100%" stop-color="{COLOR_DEEP_SKY}" />
    </linearGradient>
    <linearGradient id="appCrucible" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_PRIMARY_SKY}" />
      <stop offset="100%" stop-color="{COLOR_INDIGO}" />
    </linearGradient>
    <linearGradient id="appRight" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{COLOR_INDIGO}" />
      <stop offset="100%" stop-color="{COLOR_DARK_INDIGO}" />
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
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

def draw_symbol_on_image(draw, box, color_left, color_center, color_right):
    """
    Renders the MLForge geometric symbol within the bounding box (x0, y0, x1, y1).
    """
    bx0, by0, bx1, by1 = box
    bw = bx1 - bx0
    bh = by1 - by0

    def pt(norm_x, norm_y):
        return (bx0 + norm_x * bw / 100.0, by0 + norm_y * bh / 100.0)

    # 1. Left Ingestion Pillar with Precision Chamfer Base
    poly_left = [pt(12, 18), pt(34, 18), pt(46, 52), pt(34, 84), pt(22, 84), pt(12, 74)]
    # 2. Central Crucible / Interlocking Core
    poly_center = [pt(50, 24), pt(62, 54), pt(50, 78), pt(38, 54)]
    # 3. Right Deployment Pillar with Precision Chamfer Base
    poly_right = [pt(88, 18), pt(66, 18), pt(54, 52), pt(66, 84), pt(78, 84), pt(88, 74)]

    # Draw primary facets
    draw.polygon(poly_left, fill=color_left)
    draw.polygon(poly_center, fill=color_center)
    draw.polygon(poly_right, fill=color_right)

    # 4. Architectural Chamfer Depth Facets
    def dim_color(c, factor=0.75):
        if isinstance(c, tuple) and len(c) >= 3:
            return (int(c[0] * factor), int(c[1] * factor), int(c[2] * factor), c[3] if len(c) > 3 else 255)
        return c

    poly_left_chamfer = [pt(12, 74), pt(22, 84), pt(34, 84), pt(24, 74)]
    poly_right_chamfer = [pt(88, 74), pt(78, 84), pt(66, 84), pt(76, 74)]
    draw.polygon(poly_left_chamfer, fill=dim_color(color_left, 0.7))
    draw.polygon(poly_right_chamfer, fill=dim_color(color_right, 0.7))


def generate_png_icons_and_favicons():
    """Generates all PNG icons, app squircle, and favicons from 16px to 512px."""
    # Master resolution 1024x1024 for clean downsampling
    master_dim = 1024
    
    # 1. Standalone Icon (Transparent)
    im_icon = Image.new("RGBA", (master_dim, master_dim), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im_icon)
    margin = 80
    draw_symbol_on_image(
        draw,
        (margin, margin, master_dim - margin, master_dim - margin),
        color_left=(56, 189, 248, 255),    # Sky 400
        color_center=(99, 102, 241, 255),  # Indigo 500
        color_right=(67, 56, 202, 255)     # Indigo 700
    )
    im_icon.resize((512, 512), Image.Resampling.LANCZOS).save(ICON_DIR / "mlforge-icon.png", "PNG")

    # Light & Dark icon variants
    im_icon_light = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d_l = ImageDraw.Draw(im_icon_light)
    draw_symbol_on_image(d_l, (40, 40, 472, 472), (56, 189, 248, 255), (56, 189, 248, 255), (99, 102, 241, 255))
    im_icon_light.save(ICON_DIR / "mlforge-icon-light.png", "PNG")

    im_icon_dark = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d_d = ImageDraw.Draw(im_icon_dark)
    draw_symbol_on_image(d_d, (40, 40, 472, 472), (2, 132, 199, 255), (2, 132, 199, 255), (67, 56, 202, 255))
    im_icon_dark.save(ICON_DIR / "mlforge-icon-dark.png", "PNG")

    # 2. App Icon (Squircle container with gradient and subtle border)
    im_app = Image.new("RGBA", (master_dim, master_dim), (0, 0, 0, 0))
    d_app = ImageDraw.Draw(im_app)
    # Squircle background
    d_app.rounded_rectangle([32, 32, master_dim - 32, master_dim - 32], radius=220, fill=(15, 23, 42, 255), outline=(56, 189, 248, 90), width=8)
    draw_symbol_on_image(
        d_app,
        (190, 190, master_dim - 190, master_dim - 190),
        color_left=(56, 189, 248, 255),
        color_center=(99, 102, 241, 255),
        color_right=(79, 70, 229, 255)
    )
    im_app.resize((512, 512), Image.Resampling.LANCZOS).save(ICON_DIR / "mlforge-app-icon.png", "PNG")

    # 3. Dedicated Favicon Family
    favicon_sizes = [16, 32, 48, 64, 128, 192, 512]
    ico_images = []

    for size in favicon_sizes:
        # Create icon optimized for the target size
        im_fav = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d_fav = ImageDraw.Draw(im_fav)
        
        pad = max(1, int(size * 0.08))
        draw_symbol_on_image(
            d_fav,
            (pad, pad, size - pad, size - pad),
            color_left=(56, 189, 248, 255),
            color_center=(99, 102, 241, 255),
            color_right=(79, 70, 229, 255)
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
    def render_horizontal_logo(bg_color, text_ml_color, text_forge_color, symbol_colors, tagline=None):
        w, h = (1040, 240) if not tagline else (1040, 260)
        im = Image.new("RGBA", (w, h), bg_color if bg_color else (0, 0, 0, 0))
        d = ImageDraw.Draw(im)

        # Draw symbol
        sym_box = (30, 30, 210, 210)
        draw_symbol_on_image(d, sym_box, symbol_colors[0], symbol_colors[1], symbol_colors[2])

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
            d.text((text_x + 2, text_y + 115), tagline, fill=(148, 163, 184, 255), font=f_tag)

        return im

    # 1. Primary & Transparent (Transparent background, white + sky text)
    im_prim = render_horizontal_logo(
        None,
        (248, 250, 252, 255),
        (56, 189, 248, 255),
        ((56, 189, 248, 255), (99, 102, 241, 255), (79, 70, 229, 255))
    )
    im_prim.save(LOGO_DIR / "mlforge-logo-primary.png", "PNG")
    im_prim.save(LOGO_DIR / "mlforge-logo-transparent.png", "PNG")

    # 2. Dark Logo (for light backgrounds, slate-900 text + deep sky)
    im_dark = render_horizontal_logo(
        None,
        (15, 23, 42, 255),
        (2, 132, 199, 255),
        ((2, 132, 199, 255), (99, 102, 241, 255), (67, 56, 202, 255))
    )
    im_dark.save(LOGO_DIR / "mlforge-logo-dark.png", "PNG")

    # 3. Light Logo (clean white card background)
    im_light = render_horizontal_logo(
        (255, 255, 255, 255),
        (15, 23, 42, 255),
        (2, 132, 199, 255),
        ((2, 132, 199, 255), (99, 102, 241, 255), (67, 56, 202, 255))
    )
    im_light.save(LOGO_DIR / "mlforge-logo-light.png", "PNG")

    # 4. Logo with Tagline
    im_tag = render_horizontal_logo(
        None,
        (248, 250, 252, 255),
        (56, 189, 248, 255),
        ((56, 189, 248, 255), (99, 102, 241, 255), (79, 70, 229, 255)),
        tagline="MACHINE LEARNING DEVELOPMENT PLATFORM"
    )
    im_tag.save(LOGO_DIR / "mlforge-logo-with-tagline.png", "PNG")

    # 5. Monochrome White
    im_mono_w = render_horizontal_logo(
        None,
        (255, 255, 255, 255),
        (255, 255, 255, 255),
        ((255, 255, 255, 255), (255, 255, 255, 255), (255, 255, 255, 255))
    )
    im_mono_w.save(LOGO_DIR / "mlforge-logo-monochrome-white.png", "PNG")

    # 6. Monochrome Dark
    im_mono_d = render_horizontal_logo(
        None,
        (15, 23, 42, 255),
        (15, 23, 42, 255),
        ((15, 23, 42, 255), (15, 23, 42, 255), (15, 23, 42, 255))
    )
    im_mono_d.save(LOGO_DIR / "mlforge-logo-monochrome-dark.png", "PNG")


def main():
    print("Generating SVG vector logos...")
    generate_svg_logos()
    print("Generating SVG vector icons...")
    generate_svg_icons()
    print("Generating raster icons and multi-resolution favicons...")
    generate_png_icons_and_favicons()
    print("Generating raster PNG logos...")
    generate_png_logos()
    print("All MLForge brand assets generated successfully!")


if __name__ == "__main__":
    main()
