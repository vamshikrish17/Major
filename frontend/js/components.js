/**
 * VisionExtract 2.0 — UI Components
 * Rendering functions for all dashboard sections.
 */

const Components = {
    modeNames: {
        general: 'General Vision',
        bio: 'BioVision',
        space: 'Geo-Spatial Analysis',
    },

    // ─── Insight Banner ────────────────────────────────────
    renderInsightBanner(insight, mode) {
        const banner = document.getElementById('insightBanner');
        const iconEl = document.getElementById('insightIcon');
        const titleEl = document.getElementById('insightTitle');
        const textEl = document.getElementById('insightText');
        const flagsEl = document.getElementById('insightFlags');

        const modeIcons = { general: '📊', bio: '🧬', space: '🛰️' };
        const modeTitles = {
            general: 'General Vision',
            bio: 'BioVision',
            space: 'Geo-Spatial Analysis',
        };

        iconEl.textContent = modeIcons[mode] || '📊';
        titleEl.textContent = modeTitles[mode] || 'Analysis Results';
        textEl.textContent = insight.summary || 'Analysis complete.';

        flagsEl.innerHTML = '';
        if (insight.flags && insight.flags.length > 0) {
            insight.flags.forEach(flag => {
                const div = document.createElement('div');
                div.className = 'flag-item';
                div.textContent = flag;
                flagsEl.appendChild(div);
            });
        }

        banner.style.display = 'flex';
        banner.classList.add('fade-in-up');
    },

    // ─── Stats Panel ───────────────────────────────────────
    renderStats(data) {
        const coverage = data.features_summary?.coverage_percent
            || data.domain_insight?.metrics?.coverage_percent
            || (data.features_summary?.total_area_px > 0
                ? Math.min(100, (data.features_summary.total_area_px / 1000000 * 100).toFixed(1))
                : 0);

        const meanConf = data.features_summary?.detection_metrics?.mean_confidence
            || (data.detections?.length > 0
                ? (data.detections.reduce((s, d) => s + d.confidence, 0) / data.detections.length)
                : 0);

        document.getElementById('statObjects').textContent = data.num_objects || 0;
        document.getElementById('statCoverage').textContent = `${parseFloat(coverage).toFixed(1)}%`;
        document.getElementById('statConfidence').textContent = `${(meanConf * 100).toFixed(0)}%`;
        document.getElementById('statTime').textContent = `${Math.round(data.timing?.total_ms || 0)}ms`;

        document.getElementById('timingBadge').textContent =
            `⚡ ${Math.round(data.timing?.total_ms || 0)}ms`;
    },

    // ─── Timing Bars ───────────────────────────────────────
    renderTimingBars(timing) {
        const container = document.getElementById('timingBars');
        container.innerHTML = '';

        if (!timing) return;

        const stages = [
            { label: 'Quality Scan', ms: timing.quality_analysis_ms, color: '#f59e0b' },
            { label: 'Detection', ms: timing.detection_ms, color: '#6366f1' },
            { label: 'Segmentation', ms: timing.segmentation_ms, color: '#06b6d4' },
            { label: 'Feature Extraction', ms: timing.feature_extraction_ms, color: '#8b5cf6' },
            { label: 'Domain Analysis', ms: timing.domain_analysis_ms, color: '#34d399' },
        ];

        const maxMs = Math.max(...stages.map(s => s.ms), 1);

        stages.forEach(stage => {
            const pct = (stage.ms / maxMs) * 100;
            const row = document.createElement('div');
            row.className = 'timing-bar-row';
            row.innerHTML = `
                <span class="timing-bar-label">${stage.label}</span>
                <div class="timing-bar-track">
                    <div class="timing-bar-fill" style="width: 0%; background: ${stage.color};"></div>
                </div>
                <span class="timing-bar-value">${stage.ms.toFixed(1)}ms</span>
            `;
            container.appendChild(row);

            // Animate bar fill
            requestAnimationFrame(() => {
                setTimeout(() => {
                    row.querySelector('.timing-bar-fill').style.width = `${pct}%`;
                }, 100);
            });
        });
    },

    // ─── Objects Table ─────────────────────────────────────
    renderObjectsTable(detections) {
        const card = document.getElementById('objectsTableCard');
        const tbody = document.getElementById('objectsTableBody');
        const badge = document.getElementById('objectCountBadge');

        if (!detections || detections.length === 0) {
            card.style.display = 'none';
            return;
        }

        card.style.display = 'block';
        badge.textContent = `${detections.length} object${detections.length !== 1 ? 's' : ''}`;
        tbody.innerHTML = '';

        detections.forEach((det, i) => {
            const f = det.features || {};
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${det.segment_id?.slice(0, 6) || i + 1}</td>
                <td class="label-cell">${det.label}</td>
                <td>${(det.confidence * 100).toFixed(1)}%</td>
                <td>${f.area_px?.toFixed(0) || '—'}</td>
                <td>${f.perimeter_px?.toFixed(0) || '—'}</td>
                <td>${f.shape || '—'}</td>
                <td>${f.circularity?.toFixed(3) || '—'}</td>
                <td>${f.solidity?.toFixed(3) || '—'}</td>
                <td>${f.aspect_ratio?.toFixed(2) || '—'}</td>
            `;
            tbody.appendChild(tr);
        });
    },

    // ─── Image Quality ────────────────────────────────────
    renderQualityPanel(report) {
        const card = document.getElementById('qualityCard');
        const summary = document.getElementById('qualitySummary');
        const issues = document.getElementById('qualityIssues');
        const metrics = document.getElementById('qualityMetrics');

        if (!report || (!report.issues?.length && !Object.keys(report.metrics || {}).length)) {
            card.style.display = 'none';
            return;
        }

        card.style.display = 'block';
        summary.textContent = report.issues?.length
            ? `${report.issues.length} issue${report.issues.length !== 1 ? 's' : ''} detected`
            : 'No major quality issues detected';

        issues.innerHTML = '';
        if (report.issues?.length) {
            report.issues.forEach(issue => {
                const badge = document.createElement('div');
                badge.className = 'quality-issue';
                badge.textContent = issue;
                issues.appendChild(badge);
            });
        } else {
            const badge = document.createElement('div');
            badge.className = 'quality-issue quality-issue-ok';
            badge.textContent = 'Quality looks good';
            issues.appendChild(badge);
        }

        metrics.innerHTML = '';
        Object.entries(report.metrics || {}).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'metric-item';
            item.innerHTML = `
                <div class="metric-label">${this._formatKey(key)}</div>
                <div class="metric-value">${Number(value).toFixed(3)}</div>
            `;
            metrics.appendChild(item);
        });
    },

    // ─── Domain Metrics ────────────────────────────────────
    renderDomainMetrics(insight, mode) {
        const container = document.getElementById('domainMetrics');
        const title = document.getElementById('domainMetricsTitle');
        const grid = document.getElementById('metricsGrid');

        if (!insight || !insight.metrics) {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'block';
        const modeTitles = {
            general: '📊 General Vision Metrics',
            bio: '🧬 BioVision Metrics',
            space: '🛰️ Geo-Spatial Metrics',
        };
        title.textContent = modeTitles[mode] || 'Domain Metrics';
        grid.innerHTML = '';

        const metrics = insight.metrics;
        const items = this._flattenMetrics(metrics);

        items.slice(0, 16).forEach(item => {
            const div = document.createElement('div');
            div.className = 'metric-item';
            div.innerHTML = `
                <div class="metric-label">${item.label}</div>
                <div class="metric-value">${item.value}</div>
            `;
            grid.appendChild(div);
        });
    },

    _flattenMetrics(obj, prefix = '') {
        const items = [];
        for (const [key, val] of Object.entries(obj)) {
            const label = prefix ? `${prefix} › ${this._formatKey(key)}` : this._formatKey(key);
            if (val === null || val === undefined) continue;
            if (typeof val === 'object' && !Array.isArray(val)) {
                items.push(...this._flattenMetrics(val, this._formatKey(key)));
            } else if (Array.isArray(val)) {
                items.push({ label, value: val.length + ' items' });
            } else if (typeof val === 'number') {
                items.push({ label, value: Number.isInteger(val) ? val.toString() : val.toFixed(3) });
            } else {
                items.push({ label, value: String(val) });
            }
        }
        return items;
    },

    _formatKey(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    },

    // ─── Segments Grid ─────────────────────────────────────
    renderSegments(segments) {
        const section = document.getElementById('segmentsSection');
        const grid = document.getElementById('segmentsGrid');

        if (!segments || segments.length === 0) {
            section.style.display = 'none';
            return;
        }

        section.style.display = 'block';
        grid.innerHTML = '';

        segments.forEach((seg, i) => {
            const card = document.createElement('div');
            card.className = 'glass-card segment-card fade-in-up';
            card.style.animationDelay = `${i * 80}ms`;
            card.innerHTML = `
                <div class="segment-card-header">
                    <span class="segment-label">${seg.label}</span>
                    <span class="segment-score">${(seg.score * 100).toFixed(0)}%</span>
                </div>
                <div class="segment-image-wrapper">
                    <img src="${seg.url}" alt="${seg.label}" loading="lazy">
                </div>
                <a class="segment-download" href="${seg.url}" download>⬇ Download PNG</a>
            `;
            grid.appendChild(card);
        });
    },

    // ─── History List ──────────────────────────────────────
    renderHistory(data) {
        const list = document.getElementById('historyList');

        if (!data.analyses || data.analyses.length === 0) {
            list.innerHTML = '<p class="text-muted text-center">No analyses yet — upload an image to get started.</p>';
            return;
        }

        list.innerHTML = '';
        const modeIcons = { general: '🔍', bio: '🧬', space: '🛰️' };

        data.analyses.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.onclick = () => loadPastResult(item.analysis_id);

            const time = new Date(item.timestamp).toLocaleString();
            const original = item.original_url
                ? `<img src="${item.original_url}" alt="Input for ${item.filename || 'analysis'}" loading="lazy">`
                : '<div class="history-thumb-fallback">No input</div>';
            const overlay = item.overlay_url
                ? `<img src="${item.overlay_url}" alt="Output for ${item.filename || 'analysis'}" loading="lazy">`
                : '<div class="history-thumb-fallback">No output</div>';

            div.innerHTML = `
                <div class="history-topline">
                    <span class="history-mode">${modeIcons[item.mode] || '🔍'}</span>
                    <div class="history-info">
                        <div class="history-filename">${item.filename || 'Unknown'}</div>
                        <div class="history-meta">${time} · ${item.processing_time_ms.toFixed(0)}ms</div>
                    </div>
                    <span class="history-objects">${item.num_objects} obj</span>
                </div>
                <div class="history-preview-grid">
                    <div class="history-preview-card">
                        <div class="history-preview-label">Input</div>
                        <div class="history-preview-frame">${original}</div>
                    </div>
                    <div class="history-preview-card">
                        <div class="history-preview-label">Output</div>
                        <div class="history-preview-frame">${overlay}</div>
                    </div>
                </div>
            `;
            list.appendChild(div);
        });
    },

    // ─── Analytics Charts ──────────────────────────────────
    renderCharts(data) {
        this._renderMetricsChart(data);
        this._renderDistributionChart(data);
        this._renderSizeChart(data);
        this._renderTimingChart(data);
    },

    _renderMetricsChart(data) {
        const canvas = document.getElementById('metricsChart');
        const ctx = canvas.getContext('2d');
        const metrics = data.features_summary?.detection_metrics || {};

        const labels = ['Precision', 'Recall', 'F1 Score', 'Mean IoU'];
        const values = [
            metrics.precision || 0,
            metrics.recall || 0,
            metrics.f1_score || 0,
            metrics.mean_iou || 0,
        ];

        this._drawBarChart(ctx, canvas, labels, values, [
            '#6366f1', '#06b6d4', '#8b5cf6', '#34d399'
        ]);
    },

    _renderDistributionChart(data) {
        const canvas = document.getElementById('distributionChart');
        const ctx = canvas.getContext('2d');
        const counts = data.features_summary?.label_counts || {};

        const labels = Object.keys(counts);
        const values = Object.values(counts);

        if (labels.length === 0) {
            this._drawEmptyChart(ctx, canvas, 'No objects detected');
            return;
        }

        const colors = ['#6366f1', '#06b6d4', '#8b5cf6', '#34d399', '#fb7185', '#fbbf24', '#38bdf8'];
        this._drawBarChart(ctx, canvas, labels, values, colors);
    },

    _renderSizeChart(data) {
        const canvas = document.getElementById('sizeChart');
        const ctx = canvas.getContext('2d');
        const dist = data.features_summary?.size_distribution || {};

        const labels = ['Small', 'Medium', 'Large'];
        const values = [dist.small || 0, dist.medium || 0, dist.large || 0];

        this._drawBarChart(ctx, canvas, labels, values, ['#06b6d4', '#6366f1', '#8b5cf6']);
    },

    _renderTimingChart(data) {
        const canvas = document.getElementById('timingChart');
        const ctx = canvas.getContext('2d');
        const t = data.timing || {};

        const labels = ['Detection', 'Segmentation', 'Features', 'Domain'];
        const values = [
            t.detection_ms || 0,
            t.segmentation_ms || 0,
            t.feature_extraction_ms || 0,
            t.domain_analysis_ms || 0,
        ];

        this._drawBarChart(ctx, canvas, labels, values, [
            '#6366f1', '#06b6d4', '#8b5cf6', '#34d399'
        ], 'ms');
    },

    // ─── Canvas Chart Helpers ──────────────────────────────
    _drawBarChart(ctx, canvas, labels, values, colors, unit = '') {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);

        ctx.clearRect(0, 0, w, h);

        const padding = { top: 20, right: 20, bottom: 40, left: 50 };
        const chartW = w - padding.left - padding.right;
        const chartH = h - padding.top - padding.bottom;

        const maxVal = Math.max(...values, 0.01);
        const barWidth = Math.min(40, (chartW / labels.length) * 0.6);
        const gap = chartW / labels.length;

        // Grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (chartH * i / 4);
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(w - padding.right, y);
            ctx.stroke();

            // Y-axis labels
            ctx.fillStyle = '#64748b';
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            const val = maxVal * (1 - i / 4);
            ctx.fillText(
                (val < 1 && unit === '') ? val.toFixed(2) : Math.round(val) + unit,
                padding.left - 8,
                y + 4
            );
        }

        // Bars
        labels.forEach((label, i) => {
            const x = padding.left + gap * i + (gap - barWidth) / 2;
            const barH = (values[i] / maxVal) * chartH;
            const y = padding.top + chartH - barH;

            // Bar gradient
            const grad = ctx.createLinearGradient(x, y, x, y + barH);
            grad.addColorStop(0, colors[i % colors.length]);
            grad.addColorStop(1, colors[i % colors.length] + '66');

            ctx.fillStyle = grad;
            ctx.beginPath();
            const radius = 4;
            ctx.moveTo(x + radius, y);
            ctx.lineTo(x + barWidth - radius, y);
            ctx.quadraticCurveTo(x + barWidth, y, x + barWidth, y + radius);
            ctx.lineTo(x + barWidth, y + barH);
            ctx.lineTo(x, y + barH);
            ctx.lineTo(x, y + radius);
            ctx.quadraticCurveTo(x, y, x + radius, y);
            ctx.fill();

            // Value label
            ctx.fillStyle = '#f1f5f9';
            ctx.font = '11px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            const displayVal = values[i] < 1 && unit === ''
                ? values[i].toFixed(2)
                : Math.round(values[i]) + unit;
            ctx.fillText(displayVal, x + barWidth / 2, y - 6);

            // X-axis label
            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px "Inter", sans-serif';
            ctx.fillText(label, x + barWidth / 2, h - padding.bottom + 18);
        });
    },

    _drawEmptyChart(ctx, canvas, message) {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#475569';
        ctx.font = '13px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(message, w / 2, h / 2);
    }
};
