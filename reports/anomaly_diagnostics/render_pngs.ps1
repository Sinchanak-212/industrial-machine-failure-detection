# render_pngs.ps1 - Generates 4 publication-quality PNG diagnostic figures using System.Drawing
Add-Type -AssemblyName System.Drawing

$plotDataJson = Get-Content -Raw "reports/anomaly_diagnostics/plot_data.json" | ConvertFrom-Json
$data = $plotDataJson.data
$thresholds = $plotDataJson.thresholds

# Helper function to compute percentiles
function Get-Percentile($sorted, [double]$p) {
    if ($sorted.Count -eq 0) { return 0 }
    $idx = ($p / 100.0) * ($sorted.Count - 1)
    $low = [Math]::Floor($idx)
    $high = [Math]::Ceiling($idx)
    $w = $idx - $low
    return $sorted[$low] * (1.0 - $w) + $sorted[$high] * $w
}

function Get-Stats($arr) {
    $sorted = $arr | Sort-Object
    $n = $sorted.Count
    $sum = ($sorted | Measure-Object -Sum).Sum
    $mean = $sum / $n
    $sqDiff = 0.0
    foreach ($v in $sorted) { $sqDiff += [Math]::Pow($v - $mean, 2) }
    $std = [Math]::Sqrt($sqDiff / $n)
    return [PSCustomObject]@{
        Count = $n
        Mean = $mean
        Std = $std
        Median = (Get-Percentile $sorted 50)
        Min = $sorted[0]
        Max = $sorted[-1]
        P25 = (Get-Percentile $sorted 25)
        P75 = (Get-Percentile $sorted 75)
        P90 = (Get-Percentile $sorted 90)
        P95 = (Get-Percentile $sorted 95)
        Sorted = $sorted
    }
}

# -------------------------------------------------------------
# FIGURE 1: Overlapping Histograms / Density Plots
# -------------------------------------------------------------
Write-Host "Rendering 1. reconstruction_error_histograms.png..."
$w = 1400; $h = 900
$bmp1 = [System.Drawing.Bitmap]::new($w, $h)
$g1 = [System.Drawing.Graphics]::FromImage($bmp1)
$g1.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g1.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g1.Clear([System.Drawing.Color]::White)

$titleFont = [System.Drawing.Font]::new("Segoe UI", 16, [System.Drawing.FontStyle]::Bold)
$subFont = [System.Drawing.Font]::new("Segoe UI", 9.5)
$boldFont = [System.Drawing.Font]::new("Segoe UI", 9, [System.Drawing.FontStyle]::Bold)
$font = [System.Drawing.Font]::new("Segoe UI", 8)
$labelFont = [System.Drawing.Font]::new("Segoe UI", 9)

# Header
$sfCenter = [System.Drawing.StringFormat]::new()
$sfCenter.Alignment = [System.Drawing.StringAlignment]::Center
$g1.DrawString("Reconstruction Error Distributions: Normal vs Anomalous Audio", $titleFont, [System.Drawing.Brushes]::Black, ($w / 2), 15, $sfCenter)
$g1.DrawString("Section-Specific CNN Autoencoder (MFCC Features) | Solid vertical line = Validation P95 Threshold", $subFont, [System.Drawing.Brushes]::Gray, ($w / 2), 45, $sfCenter)

# Legend
$g1.FillRectangle([System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(120, 37, 99, 235)), $w - 470, 20, 16, 12)
$g1.DrawRectangle([System.Drawing.Pens]::Blue, $w - 470, 20, 16, 12)
$g1.DrawString("Normal Test (n=100)", $font, [System.Drawing.Brushes]::Black, $w - 450, 20)

$g1.FillRectangle([System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(120, 239, 68, 68)), $w - 320, 20, 16, 12)
$g1.DrawRectangle([System.Drawing.Pens]::Red, $w - 320, 20, 16, 12)
$g1.DrawString("Anomalous Test (n=100)", $font, [System.Drawing.Brushes]::Black, $w - 300, 20)

$threshPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(124, 58, 237), 2)
$threshPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
$g1.DrawLine($threshPen, $w - 150, 26, $w - 125, 26)
$g1.DrawString("Threshold", $font, [System.Drawing.Brushes]::Purple, $w - 120, 20)

$marginL = 80; $marginT = 80
$plotW = [Math]::Floor(($w - $marginL - 40 - 80) / 3)
$plotH = [Math]::Floor(($h - $marginT - 70 - 70) / 2)
$sections = @("section_00", "section_01", "section_02")
$splits = @("source_test", "target_test")

for ($r = 0; $r -lt 2; $r++) {
    $split = $splits[$r]
    $splitName = if ($r -eq 0) { "Source Domain" } else { "Target Domain" }
    for ($c = 0; $c -lt 3; $c++) {
        $sec = $sections[$c]
        $normals = $data.$split.$sec.normals
        $anomalies = $data.$split.$sec.anomalies
        $thresh = [double]$thresholds.$sec

        $x0 = $marginL + $c * ($plotW + 40)
        $y0 = $marginT + $r * ($plotH + 65)

        $allVals = @($normals) + @($anomalies) + @($thresh)
        $minX = ($allVals | Measure-Object -Minimum).Minimum * 0.95
        $maxX = ($allVals | Measure-Object -Maximum).Maximum * 1.05

        $nBins = 24
        $binW = ($maxX - $minX) / $nBins
        $nCounts = [int[]]::new($nBins)
        $aCounts = [int[]]::new($nBins)

        foreach ($v in $normals) {
            $b = [Math]::Floor(($v - $minX) / $binW)
            if ($b -ge $nBins) { $b = $nBins - 1 }
            if ($b -ge 0) { $nCounts[$b]++ }
        }
        foreach ($v in $anomalies) {
            $b = [Math]::Floor(($v - $minX) / $binW)
            if ($b -ge $nBins) { $b = $nBins - 1 }
            if ($b -ge 0) { $aCounts[$b]++ }
        }

        $maxC = [Math]::Max(($nCounts | Measure-Object -Maximum).Maximum, ($aCounts | Measure-Object -Maximum).Maximum)
        if ($maxC -lt 20) { $maxC = 20 }

        # Panel background
        $g1.FillRectangle([System.Drawing.Brushes]::GhostWhite, $x0, $y0, $plotW, $plotH)
        $g1.DrawRectangle([System.Drawing.Pens]::LightGray, $x0, $y0, $plotW, $plotH)

        # Grid lines
        for ($g = 0; $g -le 4; $g++) {
            $gy = $y0 + $plotH - ($g / 4.0) * $plotH
            $g1.DrawLine([System.Drawing.Pens]::Gainsboro, $x0, $gy, $x0 + $plotW, $gy)
            $cVal = [Math]::Round(($g / 4.0) * $maxC)
            $g1.DrawString("$cVal", $font, [System.Drawing.Brushes]::Gray, $x0 - 24, $gy - 7)
        }

        # Subplot Title
        $g1.DrawString("$splitName | $sec", $boldFont, [System.Drawing.Brushes]::Black, $x0 + $plotW / 2, $y0 + 8, $sfCenter)

        # Draw bars
        $nBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(120, 37, 99, 235))
        $aBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(120, 239, 68, 68))
        for ($b = 0; $b -lt $nBins; $b++) {
            $bx = $x0 + ($b / [double]$nBins) * $plotW
            $bw = ($plotW / [double]$nBins) - 1
            $nh = ($nCounts[$b] / [double]$maxC) * $plotH
            $ah = ($aCounts[$b] / [double]$maxC) * $plotH

            if ($nCounts[$b] -gt 0) {
                $g1.FillRectangle($nBrush, $bx, ($y0 + $plotH - $nh), $bw, $nh)
                $g1.DrawRectangle([System.Drawing.Pens]::RoyalBlue, $bx, ($y0 + $plotH - $nh), $bw, $nh)
            }
            if ($aCounts[$b] -gt 0) {
                $g1.FillRectangle($aBrush, $bx, ($y0 + $plotH - $ah), $bw, $ah)
                $g1.DrawRectangle([System.Drawing.Pens]::Crimson, $bx, ($y0 + $plotH - $ah), $bw, $ah)
            }
        }

        # Threshold line
        $tx = $x0 + (($thresh - $minX) / ($maxX - $minX)) * $plotW
        if ($tx -ge $x0 -and $tx -le ($x0 + $plotW)) {
            $g1.DrawLine($threshPen, $tx, $y0, $tx, $y0 + $plotH)
            $g1.DrawString("Thresh: $($thresh.ToString('F5'))", $font, [System.Drawing.Brushes]::Purple, $tx + 3, $y0 + 26)
        }

        # X-axis ticks
        for ($t = 0; $t -le 4; $t++) {
            $val = $minX + ($t / 4.0) * ($maxX - $minX)
            $rx = $x0 + ($t / 4.0) * $plotW
            $g1.DrawLine([System.Drawing.Pens]::DimGray, $rx, $y0 + $plotH, $rx, $y0 + $plotH + 4)
            $g1.DrawString($val.ToString("F4"), $font, [System.Drawing.Brushes]::DimGray, $rx - 18, $y0 + $plotH + 6)
        }

        # Stats box
        $tp = ($anomalies | Where-Object { $_ -gt $thresh }).Count
        $fp = ($normals | Where-Object { $_ -gt $thresh }).Count
        $ovl = 0.0
        for ($b = 0; $b -lt $nBins; $b++) {
            $ovl += [Math]::Min($nCounts[$b] / [double]$normals.Count, $aCounts[$b] / [double]$anomalies.Count)
        }
        $g1.FillRectangle([System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(235, 255, 255, 255)), $x0 + 8, $y0 + 25, 125, 48)
        $g1.DrawRectangle([System.Drawing.Pens]::LightGray, $x0 + 8, $y0 + 25, 125, 48)
        $g1.DrawString("Overlap (OVL): $([Math]::Round($ovl * 100))%", $font, [System.Drawing.Brushes]::Firebrick, $x0 + 12, $y0 + 28)
        $g1.DrawString("Recall (TPR): $tp%", $font, [System.Drawing.Brushes]::Black, $x0 + 12, $y0 + 42)
        $g1.DrawString("Normal FPR: $fp%", $font, [System.Drawing.Brushes]::Black, $x0 + 12, $y0 + 56)
    }
}

$g1.DrawString("Reconstruction Error (Mean Squared Error)", $boldFont, [System.Drawing.Brushes]::DarkSlateGray, ($w / 2), $h - 25, $sfCenter)
$bmp1.Save("reports/anomaly_diagnostics/reconstruction_error_histograms.png", [System.Drawing.Imaging.ImageFormat]::Png)
$g1.Dispose(); $bmp1.Dispose()

# -------------------------------------------------------------
# FIGURE 2: Box Plots
# -------------------------------------------------------------
Write-Host "Rendering 2. reconstruction_error_boxplots.png..."
$w = 1400; $h = 800
$bmp2 = [System.Drawing.Bitmap]::new($w, $h)
$g2 = [System.Drawing.Graphics]::FromImage($bmp2)
$g2.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g2.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g2.Clear([System.Drawing.Color]::White)

$g2.DrawString("Box Plot Comparison: Normal vs Anomalous Reconstruction Error", $titleFont, [System.Drawing.Brushes]::Black, ($w / 2), 15, $sfCenter)
$g2.DrawString("Quartiles, Medians (line), Means (green diamond), and Validation P95 Threshold (dashed purple line)", $subFont, [System.Drawing.Brushes]::Gray, ($w / 2), 45, $sfCenter)

# Legend
$g2.FillRectangle([System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(100, 37, 99, 235)), $w - 480, 20, 14, 14)
$g2.DrawRectangle([System.Drawing.Pens]::RoyalBlue, $w - 480, 20, 14, 14)
$g2.DrawString("Normal", $font, [System.Drawing.Brushes]::Black, $w - 460, 20)

$g2.FillRectangle([System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(100, 239, 68, 68)), $w - 380, 20, 14, 14)
$g2.DrawRectangle([System.Drawing.Pens]::Crimson, $w - 380, 20, 14, 14)
$g2.DrawString("Anomaly", $font, [System.Drawing.Brushes]::Black, $w - 360, 20)

$meanBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(4, 120, 87))
$g2.FillPolygon($meanBrush, @([System.Drawing.Point]::new($w - 280, 27), [System.Drawing.Point]::new($w - 275, 22), [System.Drawing.Point]::new($w - 270, 27), [System.Drawing.Point]::new($w - 275, 32)))
$g2.DrawString("Mean", $font, [System.Drawing.Brushes]::Black, $w - 265, 20)

$g2.DrawLine($threshPen, $w - 200, 27, $w - 170, 27)
$g2.DrawString("Val Threshold", $font, [System.Drawing.Brushes]::Purple, $w - 165, 20)

$panelW = [Math]::Floor(($w - 90 - 50 - 50) / 2)
$panelH = $h - 90 - 70

for ($p = 0; $p -lt 2; $p++) {
    $split = $splits[$p]
    $splitName = if ($p -eq 0) { "SOURCE DOMAIN TEST" } else { "TARGET DOMAIN TEST" }
    $px0 = 90 + $p * ($panelW + 50)
    $py0 = 90

    # Determine global Y scale for panel
    $allVals = @()
    foreach ($sec in $sections) {
        $allVals += @($data.$split.$sec.normals) + @($data.$split.$sec.anomalies) + @([double]$thresholds.$sec)
    }
    $minY = ($allVals | Measure-Object -Minimum).Minimum * 0.94
    $maxY = ($allVals | Measure-Object -Maximum).Maximum * 1.05

    $g2.FillRectangle([System.Drawing.Brushes]::GhostWhite, $px0, $py0, $panelW, $panelH)
    $g2.DrawRectangle([System.Drawing.Pens]::LightGray, $px0, $py0, $panelW, $panelH)
    $g2.DrawString($splitName, $boldFont, [System.Drawing.Brushes]::Black, $px0 + $panelW / 2, $py0 + 10, $sfCenter)

    # Y ticks
    for ($t = 0; $t -le 6; $t++) {
        $val = $minY + ($t / 6.0) * ($maxY - $minY)
        $gy = $py0 + $panelH - ($t / 6.0) * ($panelH - 50) - 15
        $g2.DrawLine([System.Drawing.Pens]::Gainsboro, $px0, $gy, $px0 + $panelW, $gy)
        $g2.DrawString($val.ToString("F4"), $font, [System.Drawing.Brushes]::Gray, $px0 - 45, $gy - 6)
    }

    $secW = $panelW / 3.0
    for ($s = 0; $s -lt 3; $s++) {
        $sec = $sections[$s]
        $sx0 = $px0 + $s * $secW
        $thresh = [double]$thresholds.$sec

        if ($s -gt 0) {
            $g2.DrawLine([System.Drawing.Pens]::LightGray, $sx0, $py0 + 35, $sx0, $py0 + $panelH)
        }
        $g2.DrawString($sec, $boldFont, [System.Drawing.Brushes]::DarkSlateGray, $sx0 + $secW / 2, $py0 + $panelH - 16, $sfCenter)

        function Get-Y([double]$v) {
            return $py0 + $panelH - (($v - $minY) / ($maxY - $minY)) * ($panelH - 50) - 15
        }

        # Draw threshold
        $ty = Get-Y $thresh
        $g2.DrawLine($threshPen, $sx0 + 8, $ty, $sx0 + $secW - 8, $ty)
        $g2.DrawString("Thresh: $($thresh.ToString('F4'))", $font, [System.Drawing.Brushes]::Purple, $sx0 + $secW - 65, $ty - 14)

        # Draw box function
        function Draw-Box($stats, $cx, $bColor, $pColor, $lbl) {
            $bw = 30
            $bx = $cx - $bw / 2

            $yMin = Get-Y $stats.Min
            $yMax = Get-Y $stats.Max
            $y25 = Get-Y $stats.P25
            $y75 = Get-Y $stats.P75
            $yMed = Get-Y $stats.Median
            $yMean = Get-Y $stats.Mean

            $pen = [System.Drawing.Pen]::new($pColor, 1.5)
            # Whiskers
            $g2.DrawLine($pen, $cx, $yMin, $cx, $y75)
            $g2.DrawLine($pen, $cx, $y25, $cx, $yMax)
            $g2.DrawLine($pen, $cx - 8, $yMin, $cx + 8, $yMin)
            $g2.DrawLine($pen, $cx - 8, $yMax, $cx + 8, $yMax)

            # Box
            $boxH = [Math]::Max(2, [Math]::Abs($y25 - $y75))
            $boxY = [Math]::Min($y25, $y75)
            $brush = [System.Drawing.SolidBrush]::new($bColor)
            $g2.FillRectangle($brush, $bx, $boxY, $bw, $boxH)
            $g2.DrawRectangle($pen, $bx, $boxY, $bw, $boxH)

            # Median
            $medPen = [System.Drawing.Pen]::new($pColor, 2.5)
            $g2.DrawLine($medPen, $bx, $yMed, $bx + $bw, $yMed)

            # Mean diamond
            $g2.FillPolygon($meanBrush, @([System.Drawing.Point]::new($cx, $yMean - 4), [System.Drawing.Point]::new($cx + 4, $yMean), [System.Drawing.Point]::new($cx, $yMean + 4), [System.Drawing.Point]::new($cx - 4, $yMean)))

            # Label
            $g2.DrawString($lbl, $font, [System.Drawing.SolidBrush]::new($pColor), $cx - 10, $py0 + $panelH - 32)
        }

        $nStats = Get-Stats $data.$split.$sec.normals
        $aStats = Get-Stats $data.$split.$sec.anomalies

        Draw-Box $nStats ($sx0 + $secW * 0.32) ([System.Drawing.Color]::FromArgb(100, 37, 99, 235)) ([System.Drawing.Color]::RoyalBlue) "Norm"
        Draw-Box $aStats ($sx0 + $secW * 0.68) ([System.Drawing.Color]::FromArgb(100, 239, 68, 68)) ([System.Drawing.Color]::Crimson) "Anom"
    }
}

$bmp2.Save("reports/anomaly_diagnostics/reconstruction_error_boxplots.png", [System.Drawing.Imaging.ImageFormat]::Png)
$g2.Dispose(); $bmp2.Dispose()

# -------------------------------------------------------------
# FIGURE 3: Threshold Visualization & ECDFs
# -------------------------------------------------------------
Write-Host "Rendering 3. threshold_calibration_visualization.png..."
$w = 1400; $h = 800
$bmp3 = [System.Drawing.Bitmap]::new($w, $h)
$g3 = [System.Drawing.Graphics]::FromImage($bmp3)
$g3.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g3.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g3.Clear([System.Drawing.Color]::White)

$g3.DrawString("Threshold Calibration & Empirical Cumulative Distributions (ECDF)", $titleFont, [System.Drawing.Brushes]::Black, ($w / 2), 15, $sfCenter)
$g3.DrawString("Cumulative probability curves show near-complete normal/anomaly overlap, rendering the P95 threshold blind to failure sounds", $subFont, [System.Drawing.Brushes]::Gray, ($w / 2), 45, $sfCenter)

# Legend
$g3.DrawLine([System.Drawing.Pen]::new([System.Drawing.Color]::Blue, 2.5), $w - 620, 26, $w - 595, 26)
$g3.DrawString("Normal Source", $font, [System.Drawing.Brushes]::Black, $w - 590, 20)

$tgtNormPen = [System.Drawing.Pen]::new([System.Drawing.Color]::DeepSkyBlue, 2)
$tgtNormPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
$g3.DrawLine($tgtNormPen, $w - 490, 26, $w - 465, 26)
$g3.DrawString("Normal Target", $font, [System.Drawing.Brushes]::Black, $w - 460, 20)

$g3.DrawLine([System.Drawing.Pen]::new([System.Drawing.Color]::Red, 2.5), $w - 360, 26, $w - 335, 26)
$g3.DrawString("Anomaly Source", $font, [System.Drawing.Brushes]::Black, $w - 330, 20)

$tgtAnomPen = [System.Drawing.Pen]::new([System.Drawing.Color]::OrangeRed, 2)
$tgtAnomPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
$g3.DrawLine($tgtAnomPen, $w - 220, 26, $w - 195, 26)
$g3.DrawString("Anomaly Target", $font, [System.Drawing.Brushes]::Black, $w - 190, 20)

$g3.DrawLine($threshPen, $w - 90, 26, $w - 65, 26)
$g3.DrawString("Thresh", $font, [System.Drawing.Brushes]::Purple, $w - 60, 20)

$colW = [Math]::Floor(($w - 85 - 45 - 60) / 3)
$colH = $h - 90 - 65

for ($c = 0; $c -lt 3; $c++) {
    $sec = $sections[$c]
    $x0 = 85 + $c * ($colW + 30)
    $y0 = 90
    $thresh = [double]$thresholds.$sec

    $srcN = $data.source_test.$sec.normals | Sort-Object
    $srcA = $data.source_test.$sec.anomalies | Sort-Object
    $tgtN = $data.target_test.$sec.normals | Sort-Object
    $tgtA = $data.target_test.$sec.anomalies | Sort-Object

    $all = @($srcN) + @($srcA) + @($tgtN) + @($tgtA) + @($thresh)
    $minX = ($all | Measure-Object -Minimum).Minimum * 0.95
    $maxX = ($all | Measure-Object -Maximum).Maximum * 1.05

    $g3.FillRectangle([System.Drawing.Brushes]::GhostWhite, $x0, $y0, $colW, $colH)
    $g3.DrawRectangle([System.Drawing.Pens]::LightGray, $x0, $y0, $colW, $colH)
    $g3.DrawString($sec, $boldFont, [System.Drawing.Brushes]::Black, $x0 + $colW / 2, $y0 + 12, $sfCenter)

    # Grid for ECDF (0.0 to 1.0)
    for ($p = 0; $p -le 10; $p += 2) {
        $gy = $y0 + $colH - 40 - ($p / 10.0) * ($colH - 80)
        $g3.DrawLine([System.Drawing.Pens]::Gainsboro, $x0, $gy, $x0 + $colW, $gy)
        $g3.DrawString(($p / 10.0).ToString("F1"), $font, [System.Drawing.Brushes]::Gray, $x0 - 24, $gy - 6)
    }

    # Draw 95% target horizontal line
    $p95Y = $y0 + $colH - 40 - 0.95 * ($colH - 80)
    $p95Pen = [System.Drawing.Pen]::new([System.Drawing.Color]::SeaGreen, 1.5)
    $p95Pen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
    $g3.DrawLine($p95Pen, $x0, $p95Y, $x0 + $colW, $p95Y)
    $g3.DrawString("95% Normal Cutoff (P95)", $font, [System.Drawing.Brushes]::SeaGreen, $x0 + 8, $p95Y - 14)

    # Threshold line
    $tx = $x0 + (($thresh - $minX) / ($maxX - $minX)) * $colW
    if ($tx -ge $x0 -and $tx -le ($x0 + $colW)) {
        # Shaded undetected zone
        $g3.FillRectangle([System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(25, 239, 68, 68)), $x0, $y0 + 35, ($tx - $x0), ($colH - 75))
        $g3.DrawString("UNDETECTED ZONE (FN)", $font, [System.Drawing.Brushes]::Firebrick, ($x0 + $tx) / 2, $y0 + 45, $sfCenter)
        $g3.DrawString("92% - 97% Anomalies Missed", $font, [System.Drawing.Brushes]::Firebrick, ($x0 + $tx) / 2, $y0 + 60, $sfCenter)

        $g3.DrawLine($threshPen, $tx, $y0 + 35, $tx, $y0 + $colH - 40)
        $g3.DrawString("Thresh: $($thresh.ToString('F5'))", $font, [System.Drawing.Brushes]::Purple, $tx + 4, $y0 + $colH - 52)
    }

    function Draw-ECDF($sortedArr, $pen) {
        $n = $sortedArr.Count
        $pts = [System.Drawing.PointF[]]::new($n)
        for ($i = 0; $i -lt $n; $i++) {
            $px = $x0 + (($sortedArr[$i] - $minX) / ($maxX - $minX)) * $colW
            $py = $y0 + $colH - 40 - (($i + 1) / [double]$n) * ($colH - 80)
            $pts[$i] = [System.Drawing.PointF]::new($px, $py)
        }
        $g3.DrawLines($pen, $pts)
    }

    Draw-ECDF $srcN ([System.Drawing.Pen]::new([System.Drawing.Color]::Blue, 2.5))
    Draw-ECDF $tgtN $tgtNormPen
    Draw-ECDF $srcA ([System.Drawing.Pen]::new([System.Drawing.Color]::Red, 2.5))
    Draw-ECDF $tgtA $tgtAnomPen

    # X-axis ticks
    for ($t = 0; $t -le 3; $t++) {
        $val = $minX + ($t / 3.0) * ($maxX - $minX)
        $rx = $x0 + ($t / 3.0) * $colW
        $g3.DrawLine([System.Drawing.Pens]::DimGray, $rx, $y0 + $colH - 40, $rx, $y0 + $colH - 35)
        $g3.DrawString($val.ToString("F4"), $font, [System.Drawing.Brushes]::DimGray, $rx - 18, $y0 + $colH - 30)
    }
}

$g3.DrawString("Reconstruction Error MSE", $boldFont, [System.Drawing.Brushes]::DarkSlateGray, ($w / 2), $h - 20, $sfCenter)
$bmp3.Save("reports/anomaly_diagnostics/threshold_calibration_visualization.png", [System.Drawing.Imaging.ImageFormat]::Png)
$g3.Dispose(); $bmp3.Dispose()

# -------------------------------------------------------------
# FIGURE 4: Per-Section Performance Comparison
# -------------------------------------------------------------
Write-Host "Rendering 4. per_section_performance_comparison.png..."
$w = 1400; $h = 800
$bmp4 = [System.Drawing.Bitmap]::new($w, $h)
$g4 = [System.Drawing.Graphics]::FromImage($bmp4)
$g4.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g4.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g4.Clear([System.Drawing.Color]::White)

$g4.DrawString("Per-Section Anomaly Detection Performance: Source vs Target Domain", $titleFont, [System.Drawing.Brushes]::Black, ($w / 2), 15, $sfCenter)
$g4.DrawString("Performance Comparison Highlights Divergence Between Rank AUC (0.50 to 0.71) and Actual Recall (3 to 12 pct)", $subFont, [System.Drawing.Brushes]::Gray, ($w / 2), 45, $sfCenter)

$subW = [Math]::Floor(($w - 80 - 40 - 60) / 2)
$subH = $h - 90 - 60

# LEFT: ROC-AUC
$lx0 = 80; $ly0 = 90
$g4.FillRectangle([System.Drawing.Brushes]::GhostWhite, $lx0, $ly0, $subW, $subH)
$g4.DrawRectangle([System.Drawing.Pens]::LightGray, $lx0, $ly0, $subW, $subH)
$g4.DrawString("Threshold-Independent Ranking (ROC-AUC)", $boldFont, [System.Drawing.Brushes]::Black, $lx0 + $subW / 2, $ly0 + 12, $sfCenter)

for ($p = 0; $p -le 10; $p += 2) {
    $val = $p / 10.0
    $gy = $ly0 + $subH - 40 - $val * ($subH - 80)
    $g4.DrawLine([System.Drawing.Pens]::Gainsboro, $lx0, $gy, $lx0 + $subW, $gy)
    $g4.DrawString($val.ToString("F1"), $font, [System.Drawing.Brushes]::Gray, $lx0 - 24, $gy - 6)
}

$chanceY = $ly0 + $subH - 40 - 0.5 * ($subH - 80)
$chancePen = [System.Drawing.Pen]::new([System.Drawing.Color]::Crimson, 1.5)
$chancePen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
$g4.DrawLine($chancePen, $lx0, $chanceY, $lx0 + $subW, $chanceY)
$g4.DrawString("Random Guessing (AUC=0.50)", $font, [System.Drawing.Brushes]::Crimson, $lx0 + $subW - 160, $chanceY - 14)

# Legend Left
$g4.FillRectangle([System.Drawing.Brushes]::RoyalBlue, $lx0 + 20, $ly0 + 35, 14, 12)
$g4.DrawString("Source Test AUC", $font, [System.Drawing.Brushes]::Black, $lx0 + 38, $ly0 + 35)
$g4.FillRectangle([System.Drawing.Brushes]::DeepSkyBlue, $lx0 + 150, $ly0 + 35, 14, 12)
$g4.DrawString("Target Test AUC", $font, [System.Drawing.Brushes]::Black, $lx0 + 168, $ly0 + 35)

$metrics = @(
    @{ Sec = "section_00"; SAuc = 0.5796; TAuc = 0.7126; SRec = 0.05; TRec = 0.08; TFpr = 0.05; TF1 = 0.1416 },
    @{ Sec = "section_01"; SAuc = 0.5029; TAuc = 0.5323; SRec = 0.03; TRec = 0.03; TFpr = 0.06; TF1 = 0.0550 },
    @{ Sec = "section_02"; SAuc = 0.6269; TAuc = 0.5824; SRec = 0.12; TRec = 0.06; TFpr = 0.05; TF1 = 0.1081 },
    @{ Sec = "Macro Mean"; SAuc = 0.5698; TAuc = 0.6091; SRec = 0.0667; TRec = 0.0567; TFpr = 0.0533; TF1 = 0.1016 }
)

$grpW = $subW / 4.0
for ($i = 0; $i -lt 4; $i++) {
    $m = $metrics[$i]
    $gx = $lx0 + $i * $grpW
    $bw = $grpW * 0.32
    $b1x = $gx + $grpW * 0.16
    $b2x = $gx + $grpW * 0.52

    $h1 = $m.SAuc * ($subH - 80)
    $h2 = $m.TAuc * ($subH - 80)
    $y1 = $ly0 + $subH - 40 - $h1
    $y2 = $ly0 + $subH - 40 - $h2

    $g4.FillRectangle([System.Drawing.Brushes]::RoyalBlue, $b1x, $y1, $bw, $h1)
    $g4.DrawString($m.SAuc.ToString("F3"), $font, [System.Drawing.Brushes]::MidnightBlue, $b1x + $bw / 2, $y1 - 14, $sfCenter)

    $g4.FillRectangle([System.Drawing.Brushes]::DeepSkyBlue, $b2x, $y2, $bw, $h2)
    $g4.DrawString($m.TAuc.ToString("F3"), $font, [System.Drawing.Brushes]::DarkCyan, $b2x + $bw / 2, $y2 - 14, $sfCenter)

    $g4.DrawString($m.Sec, $boldFont, [System.Drawing.Brushes]::Black, $gx + $grpW / 2, $ly0 + $subH - 24, $sfCenter)
}

# RIGHT: Operational Metrics
$rx0 = 80 + $subW + 40; $ry0 = 90
$g4.FillRectangle([System.Drawing.Brushes]::GhostWhite, $rx0, $ry0, $subW, $subH)
$g4.DrawRectangle([System.Drawing.Pens]::LightGray, $rx0, $ry0, $subW, $subH)
$g4.DrawString("Operational Metrics at P95 Threshold", $boldFont, [System.Drawing.Brushes]::Black, $rx0 + $subW / 2, $ry0 + 12, $sfCenter)

$maxOp = 0.25
for ($p = 0; $p -le 5; $p++) {
    $val = ($p / 5.0) * $maxOp
    $gy = $ry0 + $subH - 40 - ($val / $maxOp) * ($subH - 80)
    $g4.DrawLine([System.Drawing.Pens]::Gainsboro, $rx0, $gy, $rx0 + $subW, $gy)
    $pctStr = "$([Math]::Round($val * 100))%"
    $g4.DrawString($pctStr, $font, [System.Drawing.Brushes]::Gray, $rx0 - 28, $gy - 6)
}

# Legend Right
$g4.FillRectangle([System.Drawing.Brushes]::Crimson, $rx0 + 20, $ry0 + 35, 12, 12)
$g4.DrawString("Source Recall", $font, [System.Drawing.Brushes]::Black, $rx0 + 35, $ry0 + 35)
$g4.FillRectangle([System.Drawing.Brushes]::OrangeRed, $rx0 + 125, $ry0 + 35, 12, 12)
$g4.DrawString("Target Recall", $font, [System.Drawing.Brushes]::Black, $rx0 + 140, $ry0 + 35)
$g4.FillRectangle([System.Drawing.Brushes]::SlateGray, $rx0 + 230, $ry0 + 35, 12, 12)
$g4.DrawString("Target FPR", $font, [System.Drawing.Brushes]::Black, $rx0 + 245, $ry0 + 35)
$g4.FillRectangle([System.Drawing.Brushes]::MediumSeaGreen, $rx0 + 325, $ry0 + 35, 12, 12)
$g4.DrawString("Target F1", $font, [System.Drawing.Brushes]::Black, $rx0 + 340, $ry0 + 35)

for ($i = 0; $i -lt 4; $i++) {
    $m = $metrics[$i]
    $gx = $rx0 + $i * $grpW
    $bw = $grpW * 0.18
    $b1x = $gx + $grpW * 0.10
    $b2x = $gx + $grpW * 0.31
    $b3x = $gx + $grpW * 0.52
    $b4x = $gx + $grpW * 0.73

    $h1 = ($m.SRec / $maxOp) * ($subH - 80)
    $h2 = ($m.TRec / $maxOp) * ($subH - 80)
    $h3 = ($m.TFpr / $maxOp) * ($subH - 80)
    $h4 = ($m.TF1 / $maxOp) * ($subH - 80)

    $y1 = $ry0 + $subH - 40 - $h1
    $y2 = $ry0 + $subH - 40 - $h2
    $y3 = $ry0 + $subH - 40 - $h3
    $y4 = $ry0 + $subH - 40 - $h4

    $g4.FillRectangle([System.Drawing.Brushes]::Crimson, $b1x, $y1, $bw, $h1)
    $g4.DrawString("$([Math]::Round($m.SRec * 100))%", $font, [System.Drawing.Brushes]::DarkRed, $b1x + $bw / 2, $y1 - 13, $sfCenter)

    $g4.FillRectangle([System.Drawing.Brushes]::OrangeRed, $b2x, $y2, $bw, $h2)
    $g4.DrawString("$([Math]::Round($m.TRec * 100))%", $font, [System.Drawing.Brushes]::Chocolate, $b2x + $bw / 2, $y2 - 13, $sfCenter)

    $g4.FillRectangle([System.Drawing.Brushes]::SlateGray, $b3x, $y3, $bw, $h3)
    $g4.DrawString("$([Math]::Round($m.TFpr * 100))%", $font, [System.Drawing.Brushes]::Black, $b3x + $bw / 2, $y3 - 13, $sfCenter)

    $g4.FillRectangle([System.Drawing.Brushes]::MediumSeaGreen, $b4x, $y4, $bw, $h4)
    $g4.DrawString("$([Math]::Round($m.TF1 * 100, 1))%", $font, [System.Drawing.Brushes]::DarkGreen, $b4x + $bw / 2, $y4 - 13, $sfCenter)

    $g4.DrawString($m.Sec, $boldFont, [System.Drawing.Brushes]::Black, $gx + $grpW / 2, $ry0 + $subH - 24, $sfCenter)
}

$bmp4.Save("reports/anomaly_diagnostics/per_section_performance_comparison.png", [System.Drawing.Imaging.ImageFormat]::Png)
$g4.Dispose(); $bmp4.Dispose()

Write-Host "Successfully rendered all 4 PNG figures in reports/anomaly_diagnostics/!"
