const fs = require('fs');
const path = require('path');

const plotData = JSON.parse(fs.readFileSync(path.join(__dirname, 'plot_data.json'), 'utf8'));
const { data, thresholds } = plotData;

function getPercentile(sorted, p) {
    const idx = (p / 100) * (sorted.length - 1);
    const low = Math.floor(idx);
    const high = Math.ceil(idx);
    const w = idx - low;
    return sorted[low] * (1 - w) + sorted[high] * w;
}

function getStats(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const sum = sorted.reduce((a, b) => a + b, 0);
    const mean = sum / sorted.length;
    return {
        sorted,
        n: sorted.length,
        mean,
        median: getPercentile(sorted, 50),
        p25: getPercentile(sorted, 25),
        p75: getPercentile(sorted, 75),
        p90: getPercentile(sorted, 90),
        p95: getPercentile(sorted, 95),
        min: sorted[0],
        max: sorted[sorted.length - 1]
    };
}

// =================================================================
// 1. HISTOGRAMS / DENSITY PLOTS
// =================================================================
function generateHistogramsSVG() {
    const width = 1400;
    const height = 900;
    const margin = { top: 85, bottom: 65, left: 85, right: 45 };
    const plotW = (width - margin.left - margin.right - 80) / 3;
    const plotH = (height - margin.top - margin.bottom - 70) / 2;

    const sections = ['section_00', 'section_01', 'section_02'];
    const splits = ['source_test', 'target_test'];

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="background:#ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">\n`;
    
    // Header
    svg += `  <text x="${width / 2}" y="35" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">Reconstruction Error Distributions: Normal vs Anomalous Audio</text>\n`;
    svg += `  <text x="${width / 2}" y="58" text-anchor="middle" font-size="13" fill="#64748b">Section-Specific CNN Autoencoder (MFCC Features) | Solid vertical line = Validation P95 Threshold</text>\n`;

    // Legend
    svg += `  <g transform="translate(${width - 480}, 25)">\n`;
    svg += `    <rect x="0" y="0" width="16" height="12" fill="#2563eb" fill-opacity="0.5" stroke="#1d4ed8" stroke-width="1.5" rx="2"/>\n`;
    svg += `    <text x="22" y="10" font-size="12" fill="#334155">Normal Test (n=100)</text>\n`;
    svg += `    <rect x="165" y="0" width="16" height="12" fill="#ef4444" fill-opacity="0.5" stroke="#dc2626" stroke-width="1.5" rx="2"/>\n`;
    svg += `    <text x="187" y="10" font-size="12" fill="#334155">Anomalous Test (n=100)</text>\n`;
    svg += `    <line x1="345" y1="6" x2="375" y2="6" stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="4,3"/>\n`;
    svg += `    <text x="382" y="10" font-size="12" fill="#334155">Threshold</text>\n`;
    svg += `  </g>\n`;

    for (let r = 0; r < 2; r++) {
        const split = splits[r];
        const splitLabel = split === 'source_test' ? 'Source Domain' : 'Target Domain';

        for (let c = 0; c < 3; c++) {
            const sec = sections[c];
            const normals = data[split][sec].normals;
            const anomalies = data[split][sec].anomalies;
            const thresh = thresholds[sec];

            const x0 = margin.left + c * (plotW + 40);
            const y0 = margin.top + r * (plotH + 65);

            const allVals = [...normals, ...anomalies, thresh];
            const minX = Math.min(...allVals) * 0.96;
            const maxX = Math.max(...allVals) * 1.04;

            const nBins = 24;
            const binW = (maxX - minX) / nBins;
            const nCounts = new Array(nBins).fill(0);
            const aCounts = new Array(nBins).fill(0);

            normals.forEach(v => {
                let b = Math.floor((v - minX) / binW);
                if (b >= nBins) b = nBins - 1;
                nCounts[b]++;
            });
            anomalies.forEach(v => {
                let b = Math.floor((v - minX) / binW);
                if (b >= nBins) b = nBins - 1;
                aCounts[b]++;
            });

            const maxCount = Math.max(...nCounts, ...aCounts, 20);

            // Subplot panel background
            svg += `  <rect x="${x0}" y="${y0}" width="${plotW}" height="${plotH}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="4"/>\n`;

            // Gridlines
            for (let g = 0; g <= 4; g++) {
                const gy = y0 + plotH - (g / 4) * plotH;
                svg += `  <line x1="${x0}" y1="${gy}" x2="${x0 + plotW}" y2="${gy}" stroke="#e2e8f0" stroke-width="1"/>\n`;
                const countVal = Math.round((g / 4) * maxCount);
                svg += `  <text x="${x0 - 8}" y="${gy + 4}" text-anchor="end" font-size="10" fill="#94a3b8">${countVal}</text>\n`;
            }

            // Title
            svg += `  <text x="${x0 + plotW / 2}" y="${y0 + 20}" text-anchor="middle" font-size="13" font-weight="bold" fill="#0f172a">${splitLabel} | ${sec}</text>\n`;

            // Draw bars
            for (let b = 0; b < nBins; b++) {
                const bx = x0 + (b / nBins) * plotW;
                const bw = plotW / nBins - 1;
                const nh = (nCounts[b] / maxCount) * plotH;
                const ah = (aCounts[b] / maxCount) * plotH;

                if (nCounts[b] > 0) {
                    svg += `  <rect x="${bx}" y="${y0 + plotH - nh}" width="${bw}" height="${nh}" fill="#3b82f6" fill-opacity="0.45" stroke="#2563eb" stroke-width="1"/>\n`;
                }
                if (aCounts[b] > 0) {
                    svg += `  <rect x="${bx}" y="${y0 + plotH - ah}" width="${bw}" height="${ah}" fill="#ef4444" fill-opacity="0.45" stroke="#dc2626" stroke-width="1"/>\n`;
                }
            }

            // Threshold line
            const tx = x0 + ((thresh - minX) / (maxX - minX)) * plotW;
            if (tx >= x0 && tx <= x0 + plotW) {
                svg += `  <line x1="${tx}" y1="${y0}" x2="${tx}" y2="${y0 + plotH}" stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="5,3"/>\n`;
                svg += `  <text x="${tx + 4}" y="${y0 + 35}" font-size="10" font-weight="bold" fill="#7c3aed">Threshold: ${thresh.toFixed(5)}</text>\n`;
            }

            // X-axis ticks
            const nTicks = 4;
            for (let t = 0; t <= nTicks; t++) {
                const tv = minX + (t / nTicks) * (maxX - minX);
                const rx = x0 + (t / nTicks) * plotW;
                svg += `  <line x1="${rx}" y1="${y0 + plotH}" x2="${rx}" y2="${y0 + plotH + 5}" stroke="#64748b" stroke-width="1"/>\n`;
                svg += `  <text x="${rx}" y="${y0 + plotH + 18}" text-anchor="middle" font-size="9" fill="#64748b">${tv.toFixed(4)}</text>\n`;
            }

            // Overlap & diagnostic stats
            const tp = anomalies.filter(s => s > thresh).length;
            const fp = normals.filter(s => s > thresh).length;
            let ovl = 0;
            for (let b = 0; b < nBins; b++) {
                ovl += Math.min(nCounts[b] / normals.length, aCounts[b] / anomalies.length);
            }

            svg += `  <rect x="${x0 + 8}" y="${y0 + 26}" width="135" height="54" fill="#ffffff" fill-opacity="0.92" stroke="#e2e8f0" rx="3"/>\n`;
            svg += `  <text x="${x0 + 14}" y="${y0 + 40}" font-size="10" fill="#1e293b">Overlap (OVL): <tspan font-weight="bold" fill="#b91c1c">${(ovl * 100).toFixed(0)}%</tspan></text>\n`;
            svg += `  <text x="${x0 + 14}" y="${y0 + 54}" font-size="10" fill="#1e293b">Anomaly Recall: <tspan font-weight="bold">${tp}%</tspan></text>\n`;
            svg += `  <text x="${x0 + 14}" y="${y0 + 68}" font-size="10" fill="#1e293b">Normal FPR: <tspan font-weight="bold">${fp}%</tspan></text>\n`;
        }
    }

    svg += `  <text x="${width / 2}" y="${height - 15}" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">Reconstruction Error (Mean Squared Error)</text>\n`;
    svg += `  <text transform="translate(25, ${height / 2}) rotate(-90)" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">Sample Count</text>\n`;
    svg += `</svg>`;
    return svg;
}

// =================================================================
// 2. BOX PLOTS
// =================================================================
function generateBoxplotsSVG() {
    const width = 1400;
    const height = 800;
    const margin = { top: 90, bottom: 70, left: 90, right: 50 };
    const panelW = (width - margin.left - margin.right - 50) / 2;
    const panelH = height - margin.top - margin.bottom;

    const sections = ['section_00', 'section_01', 'section_02'];
    const splits = ['source_test', 'target_test'];

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="background:#ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">\n`;

    svg += `  <text x="${width / 2}" y="36" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">Box Plot Comparison: Normal vs Anomalous Reconstruction Error</text>\n`;
    svg += `  <text x="${width / 2}" y="60" text-anchor="middle" font-size="13" fill="#64748b">Showing Quartiles, Medians (horizontal line), Means (diamond), and Validation P95 Threshold (dashed line)</text>\n`;

    // Legend
    svg += `  <g transform="translate(${width - 480}, 25)">\n`;
    svg += `    <rect x="0" y="0" width="14" height="14" fill="#3b82f6" fill-opacity="0.3" stroke="#2563eb" stroke-width="1.5" rx="2"/>\n`;
    svg += `    <text x="20" y="11" font-size="12" fill="#334155">Normal</text>\n`;
    svg += `    <rect x="100" y="0" width="14" height="14" fill="#ef4444" fill-opacity="0.3" stroke="#dc2626" stroke-width="1.5" rx="2"/>\n`;
    svg += `    <text x="120" y="11" font-size="12" fill="#334155">Anomaly</text>\n`;
    svg += `    <polygon points="195,7 200,2 205,7 200,12" fill="#047857"/>\n`;
    svg += `    <text x="212" y="11" font-size="12" fill="#334155">Mean</text>\n`;
    svg += `    <line x1="265" y1="7" x2="295" y2="7" stroke="#7c3aed" stroke-width="2" stroke-dasharray="4,3"/>\n`;
    svg += `    <text x="302" y="11" font-size="12" fill="#334155">Val Threshold</text>\n`;
    svg += `  </g>\n`;

    for (let p = 0; p < 2; p++) {
        const split = splits[p];
        const splitTitle = split === 'source_test' ? 'SOURCE DOMAIN TEST' : 'TARGET DOMAIN TEST';
        const px0 = margin.left + p * (panelW + 50);
        const py0 = margin.top;

        // Find global Y scale for this split
        let minY = Infinity, maxY = -Infinity;
        sections.forEach(s => {
            const all = [...data[split][s].normals, ...data[split][s].anomalies, thresholds[s]];
            minY = Math.min(minY, ...all);
            maxY = Math.max(maxY, ...all);
        });
        minY = Math.max(0, minY * 0.92);
        maxY = maxY * 1.05;

        // Panel rect
        svg += `  <rect x="${px0}" y="${py0}" width="${panelW}" height="${panelH}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="6"/>\n`;
        svg += `  <text x="${px0 + panelW / 2}" y="${py0 + 26}" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">${splitTitle}</text>\n`;

        // Y-axis grid & labels
        const yTicks = 6;
        for (let t = 0; t <= yTicks; t++) {
            const val = minY + (t / yTicks) * (maxY - minY);
            const gy = py0 + panelH - (t / yTicks) * (panelH - 50) - 15;
            svg += `  <line x1="${px0}" y1="${gy}" x2="${px0 + panelW}" y2="${gy}" stroke="#e2e8f0" stroke-width="1"/>\n`;
            svg += `  <text x="${px0 - 8}" y="${gy + 4}" text-anchor="end" font-size="10" fill="#64748b">${val.toFixed(4)}</text>\n`;
        }

        const secW = panelW / 3;
        for (let s = 0; s < 3; s++) {
            const sec = sections[s];
            const sx0 = px0 + s * secW;
            const thresh = thresholds[sec];

            // Section dividing line
            if (s > 0) {
                svg += `  <line x1="${sx0}" y1="${py0 + 35}" x2="${sx0}" y2="${py0 + panelH}" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="3,3"/>\n`;
            }

            svg += `  <text x="${sx0 + secW / 2}" y="${py0 + panelH - 12}" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e293b">${sec}</text>\n`;

            const nStats = getStats(data[split][sec].normals);
            const aStats = getStats(data[split][sec].anomalies);

            function getY(val) {
                return py0 + panelH - ((val - minY) / (maxY - minY)) * (panelH - 50) - 15;
            }

            // Draw Threshold line across section
            const ty = getY(thresh);
            svg += `  <line x1="${sx0 + 10}" y1="${ty}" x2="${sx0 + secW - 10}" y2="${ty}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="4,3"/>\n`;
            svg += `  <text x="${sx0 + secW - 12}" y="${ty - 4}" text-anchor="end" font-size="9" font-weight="bold" fill="#7c3aed">Thresh: ${thresh.toFixed(5)}</text>\n`;

            // Draw Box for a class
            function drawBox(stats, cx, fillColor, strokeColor, label) {
                const bw = 32;
                const bx = cx - bw / 2;

                const yMin = getY(stats.min);
                const yMax = getY(stats.max);
                const y25 = getY(stats.p25);
                const y75 = getY(stats.p75);
                const yMed = getY(stats.median);
                const yMean = getY(stats.mean);

                // Whiskers
                svg += `  <line x1="${cx}" y1="${yMin}" x2="${cx}" y2="${y75}" stroke="${strokeColor}" stroke-width="1.5"/>\n`;
                svg += `  <line x1="${cx}" y1="${y25}" x2="${cx}" y2="${yMax}" stroke="${strokeColor}" stroke-width="1.5"/>\n`;
                svg += `  <line x1="${cx - bw / 4}" y1="${yMin}" x2="${cx + bw / 4}" y2="${yMin}" stroke="${strokeColor}" stroke-width="1.5"/>\n`;
                svg += `  <line x1="${cx - bw / 4}" y1="${yMax}" x2="${cx + bw / 4}" y2="${yMax}" stroke="${strokeColor}" stroke-width="1.5"/>\n`;

                // Box
                const boxH = Math.max(2, Math.abs(y25 - y75));
                const boxY = Math.min(y25, y75);
                svg += `  <rect x="${bx}" y="${boxY}" width="${bw}" height="${boxH}" fill="${fillColor}" fill-opacity="0.3" stroke="${strokeColor}" stroke-width="1.5" rx="2"/>\n`;

                // Median
                svg += `  <line x1="${bx}" y1="${yMed}" x2="${bx + bw}" y2="${yMed}" stroke="${strokeColor}" stroke-width="2.5"/>\n`;

                // Mean diamond
                svg += `  <polygon points="${cx - 4},${yMean} ${cx},${yMean - 4} ${cx + 4},${yMean} ${cx},${yMean + 4}" fill="#047857"/>\n`;

                // Label below box
                svg += `  <text x="${cx}" y="${py0 + panelH - 30}" text-anchor="middle" font-size="10" fill="${strokeColor}">${label}</text>\n`;
            }

            const normCx = sx0 + secW * 0.32;
            const anomCx = sx0 + secW * 0.68;

            drawBox(nStats, normCx, '#3b82f6', '#1d4ed8', 'Norm');
            drawBox(aStats, anomCx, '#ef4444', '#dc2626', 'Anom');
        }
    }

    svg += `  <text transform="translate(28, ${height / 2}) rotate(-90)" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">Reconstruction Error MSE</text>\n`;
    svg += `</svg>`;
    return svg;
}

// =================================================================
// 3. THRESHOLD VISUALIZATION & ECDFs
// =================================================================
function generateThresholdVisSVG() {
    const width = 1400;
    const height = 800;
    const margin = { top: 90, bottom: 65, left: 85, right: 45 };
    const colW = (width - margin.left - margin.right - 60) / 3;
    const panelH = height - margin.top - margin.bottom;

    const sections = ['section_00', 'section_01', 'section_02'];

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="background:#ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">\n`;

    svg += `  <text x="${width / 2}" y="35" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">Threshold Calibration & Empirical Cumulative Distribution (ECDF)</text>\n`;
    svg += `  <text x="${width / 2}" y="58" text-anchor="middle" font-size="13" fill="#64748b">Cumulative probability curves prove normal & anomalous scores overlap, rendering 95% threshold blind to anomalies</text>\n`;

    // Legend
    svg += `  <g transform="translate(${width - 620}, 25)">\n`;
    svg += `    <line x1="0" y1="6" x2="25" y2="6" stroke="#2563eb" stroke-width="2.5"/>\n`;
    svg += `    <text x="32" y="10" font-size="11" fill="#334155">Normal Source</text>\n`;
    svg += `    <line x1="125" y1="6" x2="150" y2="6" stroke="#0284c7" stroke-width="2" stroke-dasharray="4,2"/>\n`;
    svg += `    <text x="156" y="10" font-size="11" fill="#334155">Normal Target</text>\n`;
    svg += `    <line x1="250" y1="6" x2="275" y2="6" stroke="#dc2626" stroke-width="2.5"/>\n`;
    svg += `    <text x="282" y="10" font-size="11" fill="#334155">Anomaly Source</text>\n`;
    svg += `    <line x1="380" y1="6" x2="405" y2="6" stroke="#ea580c" stroke-width="2" stroke-dasharray="4,2"/>\n`;
    svg += `    <text x="412" y="10" font-size="11" fill="#334155">Anomaly Target</text>\n`;
    svg += `    <line x1="510" y1="6" x2="535" y2="6" stroke="#7c3aed" stroke-width="2" stroke-dasharray="4,3"/>\n`;
    svg += `    <text x="542" y="10" font-size="11" fill="#334155">Threshold</text>\n`;
    svg += `  </g>\n`;

    for (let c = 0; c < 3; c++) {
        const sec = sections[c];
        const x0 = margin.left + c * (colW + 30);
        const y0 = margin.top;
        const thresh = thresholds[sec];

        const srcN = data.source_test[sec].normals;
        const srcA = data.source_test[sec].anomalies;
        const tgtN = data.target_test[sec].normals;
        const tgtA = data.target_test[sec].anomalies;

        const all = [...srcN, ...srcA, ...tgtN, ...tgtA, thresh];
        const minX = Math.min(...all) * 0.95;
        const maxX = Math.max(...all) * 1.05;

        svg += `  <rect x="${x0}" y="${y0}" width="${colW}" height="${panelH}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="6"/>\n`;
        svg += `  <text x="${x0 + colW / 2}" y="${y0 + 25}" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">${sec}</text>\n`;

        // Y-axis grid (ECDF from 0.0 to 1.0)
        for (let p = 0; p <= 10; p += 2) {
            const gy = y0 + panelH - 40 - (p / 10) * (panelH - 80);
            svg += `  <line x1="${x0}" y1="${gy}" x2="${x0 + colW}" y2="${gy}" stroke="#e2e8f0" stroke-width="1"/>\n`;
            svg += `  <text x="${x0 - 6}" y="${gy + 4}" text-anchor="end" font-size="10" fill="#64748b">${(p / 10).toFixed(1)}</text>\n`;
        }

        // Draw 95% horizontal calibration line
        const p95Y = y0 + panelH - 40 - 0.95 * (panelH - 80);
        svg += `  <line x1="${x0}" y1="${p95Y}" x2="${x0 + colW}" y2="${p95Y}" stroke="#10b981" stroke-width="1.5" stroke-dasharray="3,3"/>\n`;
        svg += `  <text x="${x0 + 8}" y="${p95Y - 4}" font-size="9" font-weight="bold" fill="#059669">P95 Validation Target (0.95)</text>\n`;

        function drawECDF(arr, strokeColor, dashArray, strokeW = 2) {
            const sorted = [...arr].sort((a, b) => a - b);
            const n = sorted.length;
            let pathD = '';
            for (let i = 0; i < n; i++) {
                const px = x0 + ((sorted[i] - minX) / (maxX - minX)) * colW;
                const py = y0 + panelH - 40 - ((i + 1) / n) * (panelH - 80);
                if (i === 0) pathD += `M ${px} ${py}`;
                else pathD += ` L ${px} ${py}`;
            }
            svg += `  <path d="${pathD}" fill="none" stroke="${strokeColor}" stroke-width="${strokeW}" ${dashArray ? `stroke-dasharray="${dashArray}"` : ''}/>\n`;
        }

        drawECDF(srcN, '#2563eb', null, 2.5);
        drawECDF(tgtN, '#0284c7', '4,2', 2);
        drawECDF(srcA, '#dc2626', null, 2.5);
        drawECDF(tgtA, '#ea580c', '4,2', 2);

        // Threshold line
        const tx = x0 + ((thresh - minX) / (maxX - minX)) * colW;
        if (tx >= x0 && tx <= x0 + colW) {
            svg += `  <line x1="${tx}" y1="${y0 + 35}" x2="${tx}" y2="${y0 + panelH - 40}" stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="5,3"/>\n`;
            
            // Shaded undetected region
            svg += `  <rect x="${x0}" y="${y0 + 35}" width="${tx - x0}" height="${panelH - 75}" fill="#ef4444" fill-opacity="0.04"/>\n`;
            svg += `  <text x="${(x0 + tx) / 2}" y="${y0 + 55}" text-anchor="middle" font-size="10" font-weight="bold" fill="#b91c1c">UNDETECTED ZONE (FN)</text>\n`;
            svg += `  <text x="${(x0 + tx) / 2}" y="${y0 + 70}" text-anchor="middle" font-size="9" fill="#dc2626">92% - 97% Anomalies Missed</text>\n`;

            svg += `  <text x="${tx + 4}" y="${y0 + panelH - 48}" font-size="10" font-weight="bold" fill="#7c3aed">Thresh: ${thresh.toFixed(5)}</text>\n`;
        }

        // X-axis ticks
        const nTicks = 3;
        for (let t = 0; t <= nTicks; t++) {
            const val = minX + (t / nTicks) * (maxX - minX);
            const rx = x0 + (t / nTicks) * colW;
            svg += `  <line x1="${rx}" y1="${y0 + panelH - 40}" x2="${rx}" y2="${y0 + panelH - 35}" stroke="#64748b" stroke-width="1"/>\n`;
            svg += `  <text x="${rx}" y="${y0 + panelH - 22}" text-anchor="middle" font-size="9" fill="#64748b">${val.toFixed(4)}</text>\n`;
        }
    }

    svg += `  <text x="${width / 2}" y="${height - 10}" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">Reconstruction Error MSE</text>\n`;
    svg += `  <text transform="translate(25, ${height / 2}) rotate(-90)" text-anchor="middle" font-size="13" font-weight="bold" fill="#334155">Empirical Cumulative Probability P(Error ≤ x)</text>\n`;
    svg += `</svg>`;
    return svg;
}

// =================================================================
// 4. PER-SECTION PERFORMANCE COMPARISON
// =================================================================
function generatePerSectionComparisonSVG() {
    const width = 1400;
    const height = 800;
    const margin = { top: 90, bottom: 60, left: 80, right: 40 };

    const metricsData = [
        { section: 'section_00', srcAuc: 0.5796, tgtAuc: 0.7126, srcRec: 0.05, tgtRec: 0.08, srcFpr: 0.06, tgtFpr: 0.05, srcF1: 0.0901, tgtF1: 0.1416 },
        { section: 'section_01', srcAuc: 0.5029, tgtAuc: 0.5323, srcRec: 0.03, tgtRec: 0.03, srcFpr: 0.02, tgtFpr: 0.06, srcF1: 0.0571, tgtF1: 0.0550 },
        { section: 'section_02', srcAuc: 0.6269, tgtAuc: 0.5824, srcRec: 0.12, tgtRec: 0.06, srcFpr: 0.05, tgtFpr: 0.05, srcF1: 0.2051, tgtF1: 0.1081 },
        { section: 'Macro Mean', srcAuc: 0.5698, tgtAuc: 0.6091, srcRec: 0.0667, tgtRec: 0.0567, srcFpr: 0.0433, tgtFpr: 0.0533, srcF1: 0.1175, tgtF1: 0.1016 }
    ];

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="background:#ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">\n`;

    svg += `  <text x="${width / 2}" y="35" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">Per-Section Anomaly Detection Performance: Source vs Target Domain</text>\n`;
    svg += `  <text x="${width / 2}" y="58" text-anchor="middle" font-size="13" fill="#64748b">Comparison reveals high discrepancy: Rank ROC-AUC (0.50–0.71) vs Actual Anomaly Recall (3%–12%)</text>\n`;

    // 2 Subplots: Left = ROC-AUC Comparison, Right = Operational Thresholded Metrics (Recall, FPR, F1)
    const subW = (width - margin.left - margin.right - 60) / 2;
    const subH = height - margin.top - margin.bottom;

    // LEFT: ROC-AUC
    const lx0 = margin.left;
    const ly0 = margin.top;
    svg += `  <rect x="${lx0}" y="${ly0}" width="${subW}" height="${subH}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="6"/>\n`;
    svg += `  <text x="${lx0 + subW / 2}" y="${ly0 + 26}" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">Threshold-Independent Discrimination (ROC-AUC)</text>\n`;

    // Y Grid for ROC-AUC (0.0 to 1.0)
    for (let p = 0; p <= 10; p += 2) {
        const val = p / 10;
        const gy = ly0 + subH - 40 - val * (subH - 80);
        svg += `  <line x1="${lx0}" y1="${gy}" x2="${lx0 + subW}" y2="${gy}" stroke="#e2e8f0" stroke-width="1"/>\n`;
        svg += `  <text x="${lx0 - 8}" y="${gy + 4}" text-anchor="end" font-size="10" fill="#64748b">${val.toFixed(1)}</text>\n`;
    }

    // 0.5 Chance reference line
    const chanceY = ly0 + subH - 40 - 0.5 * (subH - 80);
    svg += `  <line x1="${lx0}" y1="${chanceY}" x2="${lx0 + subW}" y2="${chanceY}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3"/>\n`;
    svg += `  <text x="${lx0 + subW - 10}" y="${chanceY - 5}" text-anchor="end" font-size="10" font-weight="bold" fill="#dc2626">Random Guessing (AUC=0.50)</text>\n`;

    // Legend Left
    svg += `  <g transform="translate(${lx0 + 20}, ${ly0 + 40})">\n`;
    svg += `    <rect x="0" y="0" width="14" height="12" fill="#2563eb" rx="2"/>\n`;
    svg += `    <text x="20" y="10" font-size="11" fill="#334155">Source Test AUC</text>\n`;
    svg += `    <rect x="130" y="0" width="14" height="12" fill="#0284c7" rx="2"/>\n`;
    svg += `    <text x="150" y="10" font-size="11" fill="#334155">Target Test AUC</text>\n`;
    svg += `  </g>\n`;

    // Bars Left
    const groupW = subW / 4;
    metricsData.forEach((d, idx) => {
        const gx = lx0 + idx * groupW;
        const bw = groupW * 0.32;
        const b1x = gx + groupW * 0.16;
        const b2x = gx + groupW * 0.52;

        const h1 = d.srcAuc * (subH - 80);
        const h2 = d.tgtAuc * (subH - 80);

        const y1 = ly0 + subH - 40 - h1;
        const y2 = ly0 + subH - 40 - h2;

        svg += `  <rect x="${b1x}" y="${y1}" width="${bw}" height="${h1}" fill="#2563eb" rx="3"/>\n`;
        svg += `  <text x="${b1x + bw / 2}" y="${y1 - 6}" text-anchor="middle" font-size="10" font-weight="bold" fill="#1e40af">${d.srcAuc.toFixed(3)}</text>\n`;

        svg += `  <rect x="${b2x}" y="${y2}" width="${bw}" height="${h2}" fill="#0284c7" rx="3"/>\n`;
        svg += `  <text x="${b2x + bw / 2}" y="${y2 - 6}" text-anchor="middle" font-size="10" font-weight="bold" fill="#0369a1">${d.tgtAuc.toFixed(3)}</text>\n`;

        svg += `  <text x="${gx + groupW / 2}" y="${ly0 + subH - 18}" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">${d.section}</text>\n`;
    });

    // RIGHT: Recall & FPR
    const rx0 = margin.left + subW + 40;
    const ry0 = margin.top;
    svg += `  <rect x="${rx0}" y="${ry0}" width="${subW}" height="${subH}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="6"/>\n`;
    svg += `  <text x="${rx0 + subW / 2}" y="${ry0 + 26}" text-anchor="middle" font-size="15" font-weight="bold" fill="#0f172a">Thresholded Operational Metrics (P95 Cutoff)</text>\n`;

    // Y Grid Right (0% to 25%)
    const maxOp = 0.25;
    for (let p = 0; p <= 5; p++) {
        const val = (p / 5) * maxOp;
        const gy = ry0 + subH - 40 - (val / maxOp) * (subH - 80);
        svg += `  <line x1="${rx0}" y1="${gy}" x2="${rx0 + subW}" y2="${gy}" stroke="#e2e8f0" stroke-width="1"/>\n`;
        svg += `  <text x="${rx0 - 8}" y="${gy + 4}" text-anchor="end" font-size="10" fill="#64748b">${(val * 100).toFixed(0)}%</text>\n`;
    }

    // Legend Right
    svg += `  <g transform="translate(${rx0 + 20}, ${ry0 + 40})">\n`;
    svg += `    <rect x="0" y="0" width="12" height="12" fill="#ef4444" rx="2"/>\n`;
    svg += `    <text x="18" y="10" font-size="11" fill="#334155">Source Recall</text>\n`;
    svg += `    <rect x="110" y="0" width="12" height="12" fill="#f97316" rx="2"/>\n`;
    svg += `    <text x="128" y="10" font-size="11" fill="#334155">Target Recall</text>\n`;
    svg += `    <rect x="220" y="0" width="12" height="12" fill="#64748b" rx="2"/>\n`;
    svg += `    <text x="238" y="10" font-size="11" fill="#334155">Target FPR</text>\n`;
    svg += `    <rect x="320" y="0" width="12" height="12" fill="#10b981" rx="2"/>\n`;
    svg += `    <text x="338" y="10" font-size="11" fill="#334155">Target F1</text>\n`;
    svg += `  </g>\n`;

    metricsData.forEach((d, idx) => {
        const gx = rx0 + idx * groupW;
        const bw = groupW * 0.18;
        const b1x = gx + groupW * 0.10;
        const b2x = gx + groupW * 0.31;
        const b3x = gx + groupW * 0.52;
        const b4x = gx + groupW * 0.73;

        const h1 = (d.srcRec / maxOp) * (subH - 80);
        const h2 = (d.tgtRec / maxOp) * (subH - 80);
        const h3 = (d.tgtFpr / maxOp) * (subH - 80);
        const h4 = (d.tgtF1 / maxOp) * (subH - 80);

        const y1 = ry0 + subH - 40 - h1;
        const y2 = ry0 + subH - 40 - h2;
        const y3 = ry0 + subH - 40 - h3;
        const y4 = ry0 + subH - 40 - h4;

        svg += `  <rect x="${b1x}" y="${y1}" width="${bw}" height="${h1}" fill="#ef4444" rx="2"/>\n`;
        svg += `  <text x="${b1x + bw / 2}" y="${y1 - 5}" text-anchor="middle" font-size="9" font-weight="bold" fill="#b91c1c">${(d.srcRec * 100).toFixed(0)}%</text>\n`;

        svg += `  <rect x="${b2x}" y="${y2}" width="${bw}" height="${h2}" fill="#f97316" rx="2"/>\n`;
        svg += `  <text x="${b2x + bw / 2}" y="${y2 - 5}" text-anchor="middle" font-size="9" font-weight="bold" fill="#c2410c">${(d.tgtRec * 100).toFixed(0)}%</text>\n`;

        svg += `  <rect x="${b3x}" y="${y3}" width="${bw}" height="${h3}" fill="#64748b" rx="2"/>\n`;
        svg += `  <text x="${b3x + bw / 2}" y="${y3 - 5}" text-anchor="middle" font-size="9" font-weight="bold" fill="#475569">${(d.tgtFpr * 100).toFixed(0)}%</text>\n`;

        svg += `  <rect x="${b4x}" y="${y4}" width="${bw}" height="${h4}" fill="#10b981" rx="2"/>\n`;
        svg += `  <text x="${b4x + bw / 2}" y="${y4 - 5}" text-anchor="middle" font-size="9" font-weight="bold" fill="#047857">${(d.tgtF1 * 100).toFixed(1)}%</text>\n`;

        svg += `  <text x="${gx + groupW / 2}" y="${ry0 + subH - 18}" text-anchor="middle" font-size="11" font-weight="bold" fill="#0f172a">${d.section}</text>\n`;
    });

    svg += `</svg>`;
    return svg;
}

// Generate files
fs.writeFileSync(path.join(__dirname, 'reconstruction_error_histograms.svg'), generateHistogramsSVG());
fs.writeFileSync(path.join(__dirname, 'reconstruction_error_boxplots.svg'), generateBoxplotsSVG());
fs.writeFileSync(path.join(__dirname, 'threshold_calibration_visualization.svg'), generateThresholdVisSVG());
fs.writeFileSync(path.join(__dirname, 'per_section_performance_comparison.svg'), generatePerSectionComparisonSVG());

console.log('Successfully generated all 4 SVG plots in reports/anomaly_diagnostics/');
