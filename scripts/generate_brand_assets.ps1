# MLForge - Enterprise Brand Asset Generator (Exact User Reference Style)
param()

$baseDir = Split-Path -Parent $PSScriptRoot
$brandDir = Join-Path $baseDir "assets\branding"
$logoDir = Join-Path $brandDir "logo"
$iconDir = Join-Path $brandDir "icon"
$favDir  = Join-Path $brandDir "favicon"

New-Item -ItemType Directory -Force -Path $logoDir, $iconDir, $favDir | Out-Null
Add-Type -AssemblyName System.Drawing

# ---------------------------------------------------------------------
# SVG SYMBOL GENERATOR (Exact Hexagonal 3D Ribbon M + Floating Pixels)
# ---------------------------------------------------------------------
function Get-SymbolSvgPaths($colLeft, $colCenter, $colRight, $colTop, $isMono = $false) {
    if ($isMono) {
        return @"
    <!-- Left Top Lid -->
    <path d="M 14,32 L 32,20 L 44,28 L 26,40 Z" fill="$colLeft" />
    <!-- Left Vertical Face -->
    <path d="M 14,32 L 26,40 L 26,76 L 14,68 Z" fill="$colLeft" opacity="0.9" />
    <!-- Center Descending Ribbon Fold -->
    <path d="M 26,40 L 44,28 L 62,44 L 44,56 Z" fill="$colCenter" opacity="0.75" />
    <!-- Center Ascending Ribbon Fold -->
    <path d="M 26,76 L 44,88 L 60,74 L 42,62 Z" fill="$colCenter" opacity="0.6" />
    <!-- Right Arch & Pillar -->
    <path d="M 62,44 L 78,32 L 78,74 L 66,74 L 66,54 L 44,56 Z" fill="$colRight" />
    <!-- Bottom Cap -->
    <path d="M 14,68 L 26,76 L 36,84 L 24,76 Z" fill="$colLeft" opacity="0.5" />
    <!-- Floating Data Pixels -->
    <rect x="56" y="8" width="6.5" height="6.5" rx="1.2" fill="$colLeft" />
    <rect x="66" y="8" width="6.5" height="6.5" rx="1.2" fill="$colRight" opacity="0.85" />
    <rect x="50" y="18" width="6.5" height="6.5" rx="1.2" fill="$colLeft" />
    <rect x="60" y="18" width="6.5" height="6.5" rx="1.2" fill="$colLeft" opacity="0.9" />
    <rect x="70" y="18" width="6.5" height="6.5" rx="1.2" fill="$colRight" opacity="0.75" />
"@
    }

    return @"
    <!-- Left Top Lid -->
    <path d="M 14,32 L 32,20 L 44,28 L 26,40 Z" fill="$colTop" />
    <!-- Left Vertical Face -->
    <path d="M 14,32 L 26,40 L 26,76 L 14,68 Z" fill="$colLeft" />
    <!-- Center Descending Ribbon Fold -->
    <path d="M 26,40 L 44,28 L 62,44 L 44,56 Z" fill="$colCenter" />
    <!-- Center Ascending Ribbon Fold -->
    <path d="M 26,76 L 44,88 L 60,74 L 42,62 Z" fill="url(#gradBottomFold)" />
    <!-- Right Arch & Pillar -->
    <path d="M 62,44 L 78,32 L 78,74 L 66,74 L 66,54 L 44,56 Z" fill="$colRight" />
    <!-- Bottom Cap -->
    <path d="M 14,68 L 26,76 L 36,84 L 24,76 Z" fill="#0052cc" />
    <!-- Floating Data Pixels -->
    <rect x="56" y="8" width="6.5" height="6.5" rx="1.2" fill="#00e5ff" />
    <rect x="66" y="8" width="6.5" height="6.5" rx="1.2" fill="#a855f7" />
    <rect x="50" y="18" width="6.5" height="6.5" rx="1.2" fill="#00d2ff" />
    <rect x="60" y="18" width="6.5" height="6.5" rx="1.2" fill="#38bdf8" />
    <rect x="70" y="18" width="6.5" height="6.5" rx="1.2" fill="#c084fc" />
"@
}

# Master Defs for Full-Color Assets
$masterDefs = @"
  <defs>
    <linearGradient id="gradLeftTop" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="100%" stop-color="#00d2ff" />
    </linearGradient>
    <linearGradient id="gradLeftPillar" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00e5ff" />
      <stop offset="100%" stop-color="#0066ff" />
    </linearGradient>
    <linearGradient id="gradCenterRibbon" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0072ff" />
      <stop offset="100%" stop-color="#4f46e5" />
    </linearGradient>
    <linearGradient id="gradBottomFold" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#2563eb" />
      <stop offset="100%" stop-color="#7c3aed" />
    </linearGradient>
    <linearGradient id="gradRightPillar" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7c3aed" />
      <stop offset="60%" stop-color="#a855f7" />
      <stop offset="100%" stop-color="#c084fc" />
    </linearGradient>
    <linearGradient id="gradWordForge" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="45%" stop-color="#60a5fa" />
      <stop offset="80%" stop-color="#a855f7" />
      <stop offset="100%" stop-color="#c084fc" />
    </linearGradient>
    <linearGradient id="gradDarkLeft" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0284c7" />
      <stop offset="100%" stop-color="#0369a1" />
    </linearGradient>
  </defs>
"@

# 1. Primary & Transparent Logos
$primarySvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 105" width="440" height="105">
$masterDefs
  <g transform="translate(6, 2)">
$(Get-SymbolSvgPaths "url(#gradLeftPillar)" "url(#gradCenterRibbon)" "url(#gradRightPillar)" "url(#gradLeftTop)")
  </g>
  <text x="120" y="68" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="48" fill="#ffffff" letter-spacing="-0.03em">ML<tspan fill="url(#gradWordForge)">Forge</tspan></text>
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $logoDir "mlforge-logo-primary.svg"), $primarySvg)
[System.IO.File]::WriteAllText((Join-Path $logoDir "mlforge-logo-transparent.svg"), $primarySvg)

# 2. Logo with Official Tagline
$taglineSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 120" width="540" height="120">
$masterDefs
  <g transform="translate(6, 10)">
$(Get-SymbolSvgPaths "url(#gradLeftPillar)" "url(#gradCenterRibbon)" "url(#gradRightPillar)" "url(#gradLeftTop)")
  </g>
  <text x="122" y="66" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="46" fill="#ffffff" letter-spacing="-0.03em">ML<tspan fill="url(#gradWordForge)">Forge</tspan></text>
  <text x="125" y="93" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="500" font-size="13" fill="#cbd5e1" letter-spacing="0.14em">Build. Train. Evaluate. Deploy.</text>
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $logoDir "mlforge-logo-with-tagline.svg"), $taglineSvg)

# 3. Dark Theme Logo (for Light Surfaces)
$darkSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 105" width="440" height="105">
$masterDefs
  <g transform="translate(6, 2)">
$(Get-SymbolSvgPaths "url(#gradDarkLeft)" "#6366f1" "#7c3aed" "#0284c7")
  </g>
  <text x="120" y="68" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="48" fill="#0f172a" letter-spacing="-0.03em">ML<tspan fill="url(#gradWordForge)">Forge</tspan></text>
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $logoDir "mlforge-logo-dark.svg"), $darkSvg)

# 4. Light Theme Logo (White Enclosure)
$lightSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 105" width="440" height="105">
  <rect width="440" height="105" fill="#ffffff" rx="8" />
$masterDefs
  <g transform="translate(6, 2)">
$(Get-SymbolSvgPaths "url(#gradDarkLeft)" "#6366f1" "#7c3aed" "#0284c7")
  </g>
  <text x="120" y="68" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="48" fill="#0f172a" letter-spacing="-0.03em">ML<tspan fill="url(#gradWordForge)">Forge</tspan></text>
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $logoDir "mlforge-logo-light.svg"), $lightSvg)

# 5. Monochrome White & Dark
$monoWhiteSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 105" width="440" height="105">
  <g transform="translate(6, 2)">
$(Get-SymbolSvgPaths "#ffffff" "#ffffff" "#ffffff" "#ffffff" $true)
  </g>
  <text x="120" y="68" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="48" fill="#ffffff" letter-spacing="-0.03em">MLForge</text>
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $logoDir "mlforge-logo-monochrome-white.svg"), $monoWhiteSvg)

$monoDarkSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 440 105" width="440" height="105">
  <g transform="translate(6, 2)">
$(Get-SymbolSvgPaths "#0f172a" "#0f172a" "#0f172a" "#0f172a" $true)
  </g>
  <text x="120" y="68" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="48" fill="#0f172a" letter-spacing="-0.03em">MLForge</text>
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $logoDir "mlforge-logo-monochrome-dark.svg"), $monoDarkSvg)

# 6. Standalone Icons
$iconSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
$masterDefs
$(Get-SymbolSvgPaths "url(#gradLeftPillar)" "url(#gradCenterRibbon)" "url(#gradRightPillar)" "url(#gradLeftTop)")
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $iconDir "mlforge-icon.svg"), $iconSvg)

$iconLightSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
$masterDefs
$(Get-SymbolSvgPaths "#38bdf8" "#6366f1" "#a855f7" "#67e8f9")
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $iconDir "mlforge-icon-light.svg"), $iconLightSvg)

$iconDarkSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
$masterDefs
$(Get-SymbolSvgPaths "#0284c7" "#4f46e5" "#7c3aed" "#0284c7")
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $iconDir "mlforge-icon-dark.svg"), $iconDarkSvg)

# App Squircle Icon
$appIconSvg = @"
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="appBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#050b18" />
      <stop offset="100%" stop-color="#0f172a" />
    </linearGradient>
    <linearGradient id="appBorder" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.6" />
      <stop offset="100%" stop-color="#a855f7" stop-opacity="0.3" />
    </linearGradient>
    <linearGradient id="gradLeftTop" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="100%" stop-color="#00d2ff" />
    </linearGradient>
    <linearGradient id="gradLeftPillar" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#00e5ff" />
      <stop offset="100%" stop-color="#0066ff" />
    </linearGradient>
    <linearGradient id="gradCenterRibbon" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0072ff" />
      <stop offset="100%" stop-color="#4f46e5" />
    </linearGradient>
    <linearGradient id="gradBottomFold" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#2563eb" />
      <stop offset="100%" stop-color="#7c3aed" />
    </linearGradient>
    <linearGradient id="gradRightPillar" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7c3aed" />
      <stop offset="60%" stop-color="#a855f7" />
      <stop offset="100%" stop-color="#c084fc" />
    </linearGradient>
  </defs>
  <rect x="16" y="16" width="480" height="480" rx="112" fill="url(#appBg)" stroke="url(#appBorder)" stroke-width="4" />
  <g transform="translate(86, 86) scale(3.4)">
$(Get-SymbolSvgPaths "url(#gradLeftPillar)" "url(#gradCenterRibbon)" "url(#gradRightPillar)" "url(#gradLeftTop)")
  </g>
</svg>
"@
[System.IO.File]::WriteAllText((Join-Path $iconDir "mlforge-app-icon.svg"), $appIconSvg)


# ---------------------------------------------------------------------
# RASTER GDI RENDERING (System.Drawing)
# ---------------------------------------------------------------------
function Draw-ExactSymbolGDI($gfx, $x0, $y0, $x1, $y1, $colLeft, $colCenter, $colRight, $colTop) {
    $w = [float]($x1 - $x0)
    $h = [float]($y1 - $y0)

    function P([float]$nx, [float]$ny) {
        $px = [float]($x0 + $nx * $w / 100.0)
        $py = [float]($y0 + $ny * $h / 100.0)
        return [System.Drawing.PointF]::new($px, $py)
    }

    $brushTop    = [System.Drawing.SolidBrush]::new($colTop)
    $brushL      = [System.Drawing.SolidBrush]::new($colLeft)
    $brushC      = [System.Drawing.SolidBrush]::new($colCenter)
    $brushBot    = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 37, 99, 235))
    $brushR      = [System.Drawing.SolidBrush]::new($colRight)
    $brushCap    = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 0, 82, 204))

    # 1. Left Top Lid
    $ptsTop = [System.Drawing.PointF[]]@( (P 14 32), (P 32 20), (P 44 28), (P 26 40) )
    $gfx.FillPolygon($brushTop, $ptsTop)

    # 2. Left Pillar Vertical Face
    $ptsL = [System.Drawing.PointF[]]@( (P 14 32), (P 26 40), (P 26 76), (P 14 68) )
    $gfx.FillPolygon($brushL, $ptsL)

    # 3. Bottom Left Cap
    $ptsCap = [System.Drawing.PointF[]]@( (P 14 68), (P 26 76), (P 36 84), (P 24 76) )
    $gfx.FillPolygon($brushCap, $ptsCap)

    # 4. Center Descending Ribbon Fold
    $ptsC = [System.Drawing.PointF[]]@( (P 26 40), (P 44 28), (P 62 44), (P 44 56) )
    $gfx.FillPolygon($brushC, $ptsC)

    # 5. Bottom Ascending Ribbon Fold
    $ptsBot = [System.Drawing.PointF[]]@( (P 26 76), (P 44 88), (P 60 74), (P 42 62) )
    $gfx.FillPolygon($brushBot, $ptsBot)

    # 6. Right Arch & Pillar
    $ptsR = [System.Drawing.PointF[]]@( (P 62 44), (P 78 32), (P 78 74), (P 66 74), (P 66 54), (P 44 56) )
    $gfx.FillPolygon($brushR, $ptsR)

    # 7. Floating Data Cubes
    $bCyan   = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 0, 229, 255))
    $bPurple = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 168, 85, 247))
    $bBlue   = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 56, 189, 248))
    $bLilac  = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 192, 132, 252))

    $cw = [float](6.5 * $w / 100.0)
    $ch = [float](6.5 * $h / 100.0)
    $gfx.FillRectangle($bCyan,   (P 56 8).X,  (P 56 8).Y,  $cw, $ch)
    $gfx.FillRectangle($bPurple, (P 66 8).X,  (P 66 8).Y,  $cw, $ch)
    $gfx.FillRectangle($bCyan,   (P 50 18).X, (P 50 18).Y, $cw, $ch)
    $gfx.FillRectangle($bBlue,   (P 60 18).X, (P 60 18).Y, $cw, $ch)
    $gfx.FillRectangle($bLilac,  (P 70 18).X, (P 70 18).Y, $cw, $ch)
}

# Color Definitions
$cCyanTop = [System.Drawing.Color]::FromArgb(255, 56, 189, 248)
$cCyanFace= [System.Drawing.Color]::FromArgb(255, 0, 210, 255)
$cBlueMid = [System.Drawing.Color]::FromArgb(255, 0, 114, 255)
$cPurple  = [System.Drawing.Color]::FromArgb(255, 168, 85, 247)
$cWhite   = [System.Drawing.Color]::FromArgb(255, 255, 255, 255)
$cDarkSl  = [System.Drawing.Color]::FromArgb(255, 15, 23, 42)

# Favicons
$favSizes = @(16, 32, 48, 64, 128, 192, 512)
foreach ($sz in $favSizes) {
    $bmp = [System.Drawing.Bitmap]::new($sz, $sz)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $pad = [Math]::Max(1, [int]($sz * 0.06))
    Draw-ExactSymbolGDI $g $pad $pad ($sz - $pad) ($sz - $pad) $cCyanFace $cBlueMid $cPurple $cCyanTop
    $g.Dispose()
    $p = Join-Path $favDir "favicon-$sz.png"
    $bmp.Save($p, [System.Drawing.Imaging.ImageFormat]::Png)
    if ($sz -eq 32) {
        $hIcon = $bmp.GetHicon()
        $icon = [System.Drawing.Icon]::FromHandle($hIcon)
        $fs = [System.IO.File]::OpenWrite((Join-Path $favDir "favicon.ico"))
        $icon.Save($fs)
        $fs.Close()
    }
    $bmp.Dispose()
}

# Standalone Icons
$iconSizes = @(
    @("mlforge-icon.png", 512, 40, $cCyanFace, $cBlueMid, $cPurple, $cCyanTop),
    @("mlforge-icon-light.png", 512, 40, $cCyanFace, $cCyanTop, $cPurple, $cCyanTop),
    @("mlforge-icon-dark.png", 512, 40, [System.Drawing.Color]::FromArgb(255, 2, 132, 199), [System.Drawing.Color]::FromArgb(255, 79, 70, 229), [System.Drawing.Color]::FromArgb(255, 124, 58, 237), [System.Drawing.Color]::FromArgb(255, 2, 132, 199))
)
foreach ($item in $iconSizes) {
    $name = $item[0]
    $dim = $item[1]
    $pad = $item[2]
    $bmp = [System.Drawing.Bitmap]::new($dim, $dim)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    Draw-ExactSymbolGDI $g $pad $pad ($dim - $pad) ($dim - $pad) $item[3] $item[4] $item[5] $item[6]
    $g.Dispose()
    $bmp.Save((Join-Path $iconDir $name), [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}

# App Squircle Icon
$bmpApp = [System.Drawing.Bitmap]::new(512, 512)
$gApp = [System.Drawing.Graphics]::FromImage($bmpApp)
$gApp.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$brushBg = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 5, 11, 24))
$penBorder = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(140, 56, 189, 248), 4)

$rect = [System.Drawing.Rectangle]::new(16, 16, 480, 480)
$path = [System.Drawing.Drawing2D.GraphicsPath]::new()
$r = 112
$d = $r * 2
$path.AddArc($rect.X, $rect.Y, $d, $d, 180, 90)
$path.AddArc($rect.Right - $d, $rect.Y, $d, $d, 270, 90)
$path.AddArc($rect.Right - $d, $rect.Bottom - $d, $d, $d, 0, 90)
$path.AddArc($rect.X, $rect.Bottom - $d, $d, $d, 90, 90)
$path.CloseFigure()
$gApp.FillPath($brushBg, $path)
$gApp.DrawPath($penBorder, $path)
Draw-ExactSymbolGDI $gApp 100 100 412 412 $cCyanFace $cBlueMid $cPurple $cCyanTop
$gApp.Dispose()
$bmpApp.Save((Join-Path $iconDir "mlforge-app-icon.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$bmpApp.Dispose()

# Logo PNGs
$logoItems = @(
    @("mlforge-logo-primary.png", $null, $cWhite, $false),
    @("mlforge-logo-transparent.png", $null, $cWhite, $false),
    @("mlforge-logo-dark.png", $null, $cDarkSl, $false),
    @("mlforge-logo-light.png", [System.Drawing.Color]::White, $cDarkSl, $false),
    @("mlforge-logo-monochrome-white.png", $null, $cWhite, $true),
    @("mlforge-logo-monochrome-dark.png", $null, $cDarkSl, $true),
    @("mlforge-logo-with-tagline.png", $null, $cWhite, $false)
)

foreach ($item in $logoItems) {
    $fname = $item[0]
    $bgColor = $item[1]
    $txtColor = $item[2]
    $isMono = $item[3]

    $w = 1040
    $h = if ($fname -like "*tagline*") { 260 } else { 240 }
    $bmp = [System.Drawing.Bitmap]::new($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

    if ($bgColor -ne $null) {
        $g.Clear($bgColor)
    }

    if ($isMono) {
        Draw-ExactSymbolGDI $g 30 30 210 210 $txtColor $txtColor $txtColor $txtColor
    } else {
        Draw-ExactSymbolGDI $g 30 30 210 210 $cCyanFace $cBlueMid $cPurple $cCyanTop
    }

    $fontMain = [System.Drawing.Font]::new("Segoe UI", 68.0, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $brushML = [System.Drawing.SolidBrush]::new($txtColor)

    if ($isMono) {
        $brushForge = [System.Drawing.SolidBrush]::new($txtColor)
    } else {
        $brushForge = [System.Drawing.Drawing2D.LinearGradientBrush]::new(
            [System.Drawing.PointF]::new(355, 70),
            [System.Drawing.PointF]::new(580, 70),
            $cCyanTop,
            $cPurple
        )
    }

    $g.DrawString("ML", $fontMain, $brushML, 250, 70)
    $g.DrawString("Forge", $fontMain, $brushForge, 355, 70)

    if ($fname -like "*tagline*") {
        $fontTag = [System.Drawing.Font]::new("Segoe UI", 16.0, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
        $brushTag = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 203, 213, 225))
        $g.DrawString("Build. Train. Evaluate. Deploy.", $fontTag, $brushTag, 256, 155)
    }

    $g.Dispose()
    $bmp.Save((Join-Path $logoDir $fname), [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}

Write-Output "Exact user style brand assets generated successfully."
