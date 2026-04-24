/**
 * VisionExtract 2.0 — Main Application Controller
 * Handles user interactions, state management, and pipeline orchestration.
 */

// ─── State ────────────────────────────────────────────────────
let currentMode = 'general';
let selectedFile = null;
let webcamStream = null;
let isAnalyzing = false;
let lastResult = null;
let lockedMode = null;

const MODE_CONTENT = {
    general: {
        name: 'General Vision',
        uploadTitle: 'Upload for General Vision',
        uploadSubtitle: 'Objects, scenes, and broad visual analysis',
        webcamTitle: 'General Vision Capture',
        webcamSubtitle: 'Real-time scene segmentation',
        analyzeText: 'Analyze in General Vision',
        captureText: 'Capture for General Vision',
        loadingTitle: 'Analyzing in General Vision...',
        heroKicker: 'Open-world image workspace',
        heroCopy: 'A cinematic, flexible analysis surface for scenes, products, and mixed visual content.',
        heroPillars: ['Scene Regions', 'Object Highlights', 'Downloadable Segments'],
    },
    bio: {
        name: 'BioVision',
        uploadTitle: 'Upload for BioVision',
        uploadSubtitle: 'Cells, specimens, and biological structures',
        webcamTitle: 'BioVision Capture',
        webcamSubtitle: 'Microscopy-style structure analysis',
        analyzeText: 'Analyze in BioVision',
        captureText: 'Capture for BioVision',
        loadingTitle: 'Analyzing in BioVision...',
        heroKicker: 'Biological imaging workspace',
        heroCopy: 'A cleaner laboratory interface for microscopy, cell structures, specimen regions, and morphology review.',
        heroPillars: ['Cell Regions', 'Morphology Metrics', 'Bio-safe Labels'],
    },
    space: {
        name: 'Geo-Spatial Analysis',
        uploadTitle: 'Upload for Geo-Spatial Analysis',
        uploadSubtitle: 'Terrain, satellite views, and mapped regions',
        webcamTitle: 'Geo-Spatial Capture',
        webcamSubtitle: 'Geospatial feature extraction',
        analyzeText: 'Analyze in Geo-Spatial Analysis',
        captureText: 'Capture for Geo-Spatial Analysis',
        loadingTitle: 'Analyzing in Geo-Spatial Analysis...',
        heroKicker: 'Cartographic analysis workspace',
        heroCopy: 'A map-inspired interface for terrain review, geospatial features, remote sensing images, and regional segmentation.',
        heroPillars: ['Terrain Regions', 'Feature Layers', 'Spatial Summaries'],
    },
};

// ─── Initialization ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    lockedMode = getLockedModeFromUrl();
    if (lockedMode) {
        currentMode = lockedMode;
    }
    initModeSelector();
    initDropZone();
    initSlider();
    applyModeCopy(currentMode);
    checkHealth();
    loadHistory();
});

// ─── Health Check ─────────────────────────────────────────────
async function checkHealth() {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    const deviceBadge = document.getElementById('deviceBadge');

    try {
        const health = await VisionAPI.getHealth();
        statusDot.classList.add('online');
        statusText.textContent = 'System Online';
        deviceBadge.textContent = health.device.toUpperCase();

        if (health.gpu_available) {
            deviceBadge.style.background = 'rgba(52, 211, 153, 0.15)';
            deviceBadge.style.color = '#34d399';
            deviceBadge.style.borderColor = 'rgba(52, 211, 153, 0.2)';
        }
    } catch (e) {
        statusText.textContent = 'Connecting...';
        // Retry health check
        setTimeout(checkHealth, 3000);
    }
}

// ─── Mode Selector ────────────────────────────────────────────
function initModeSelector() {
    const buttons = document.querySelectorAll('.mode-btn');
    if (lockedMode) {
        const selector = document.getElementById('modeSelector');
        if (selector) selector.style.display = 'none';
    }
    buttons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === currentMode);
        btn.addEventListener('click', () => {
            if (lockedMode) return;
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            applyModeCopy(currentMode);
        });
    });
}

function getLockedModeFromUrl() {
    const bodyMode = document.body?.dataset?.lockedMode;
    if (bodyMode && MODE_CONTENT[bodyMode]) {
        return bodyMode;
    }
    const params = new URLSearchParams(window.location.search);
    const mode = params.get('mode');
    if (mode && MODE_CONTENT[mode]) {
        return mode;
    }
    return null;
}

function applyModeCopy(mode) {
    const content = MODE_CONTENT[mode] || MODE_CONTENT.general;
    document.body.classList.remove('mode-general', 'mode-bio', 'mode-space');
    document.body.classList.add(`mode-${mode}`);
    document.getElementById('uploadTitle').textContent = content.uploadTitle;
    document.getElementById('uploadSubtitle').textContent = content.uploadSubtitle;
    document.getElementById('webcamTitle').textContent = content.webcamTitle;
    document.getElementById('webcamSubtitle').textContent = content.webcamSubtitle;
    document.getElementById('analyzeBtnText').textContent = content.analyzeText;
    document.getElementById('captureBtnText').textContent = content.captureText;
    document.getElementById('loadingTitle').textContent = content.loadingTitle;
    const heroKicker = document.getElementById('modeHeroKicker');
    const heroTitle = document.getElementById('modeHeroTitle');
    const heroCopy = document.getElementById('modeHeroCopy');
    const heroPillars = document.getElementById('modeHeroPillars');
    if (heroKicker) heroKicker.textContent = content.heroKicker;
    if (heroTitle) heroTitle.textContent = content.name;
    if (heroCopy) heroCopy.textContent = content.heroCopy;
    if (heroPillars) {
        heroPillars.innerHTML = '';
        content.heroPillars.forEach(item => {
            const chip = document.createElement('span');
            chip.className = 'mode-hero-pill';
            chip.textContent = item;
            heroPillars.appendChild(chip);
        });
    }
}

// ─── Drop Zone ────────────────────────────────────────────────
function initDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    selectedFile = file;

    const preview = document.getElementById('filePreview');
    const dropZone = document.getElementById('dropZone');
    const previewImg = document.getElementById('previewImg');
    const fileName = document.getElementById('fileName');

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        preview.style.display = 'block';
        dropZone.style.display = 'none';
    };
    reader.readAsDataURL(file);

    fileName.textContent = file.name;
}

function clearFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('filePreview').style.display = 'none';
    document.getElementById('dropZone').style.display = 'block';
}

// ─── Slider ───────────────────────────────────────────────────
function initSlider() {
    const slider = document.getElementById('confSlider');
    const value = document.getElementById('confValue');

    slider.addEventListener('input', () => {
        value.textContent = (slider.value / 100).toFixed(2);
    });
}

// ─── Webcam ───────────────────────────────────────────────────
async function startWebcam() {
    if (webcamStream) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        webcamStream = stream;
        const video = document.getElementById('webcamVideo');
        video.srcObject = stream;
        video.classList.add('active');
        document.getElementById('webcamPlaceholder').style.display = 'none';
    } catch (err) {
        alert('Cannot access camera: ' + err.message);
    }
}

async function captureAndAnalyze() {
    const video = document.getElementById('webcamVideo');
    if (!video || video.readyState !== 4) {
        alert("Camera not ready. Click 'Start Camera' first.");
        return;
    }

    const canvas = document.getElementById('webcamCanvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    const dataURL = canvas.toDataURL('image/png');

    showLoading();

    try {
        const confidence = document.getElementById('confSlider').value / 100;
        const result = await VisionAPI.analyzeWebcam(dataURL, currentMode, confidence);
        lastResult = result;
        showResults(result);
    } catch (err) {
        hideLoading();
        alert('Analysis failed: ' + err.message);
    }
}

// ─── Analyze ──────────────────────────────────────────────────
async function analyzeImage() {
    if (isAnalyzing) return;
    if (!selectedFile) {
        alert('Please select an image file first.');
        return;
    }

    isAnalyzing = true;
    showLoading();

    try {
        const confidence = document.getElementById('confSlider').value / 100;
        const result = await VisionAPI.analyzeImage(
            selectedFile,
            currentMode,
            confidence,
            (pct) => updateLoadingProgress(pct),
        );
        lastResult = result;
        showResults(result);
        loadHistory();
    } catch (err) {
        hideLoading();
        alert('Analysis failed: ' + err.message);
    } finally {
        isAnalyzing = false;
    }
}

// ─── Loading States ───────────────────────────────────────────
function showLoading() {
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';

    // Animate stages
    const stages = ['stageInput', 'stageQuality', 'stageDetect', 'stageSeg', 'stageFeatures', 'stageDomain'];
    stages.forEach((id, i) => {
        const el = document.getElementById(id);
        el.className = 'stage';
    });

    let currentStage = 0;
    const interval = setInterval(() => {
        if (currentStage > 0) {
            document.getElementById(stages[currentStage - 1]).className = 'stage done';
        }
        if (currentStage < stages.length) {
            document.getElementById(stages[currentStage]).className = 'stage active';
            currentStage++;
        } else {
            clearInterval(interval);
        }
    }, 600);
}

function hideLoading() {
    document.getElementById('loadingSection').style.display = 'none';
}

function updateLoadingProgress(pct) {
    // Could add a progress bar here in the future
}

// ─── Display Results ──────────────────────────────────────────
function showResults(data) {
    hideLoading();
    const section = document.getElementById('resultsSection');
    section.style.display = 'block';
    section.classList.add('fade-in');

    // Summary
    const modeName = MODE_CONTENT[data.mode]?.name || data.mode;
    document.getElementById('resultsSummary').textContent =
        `${modeName} · ${data.num_objects} results · ${data.device}`;

    // Images
    document.getElementById('originalImage').src = data.original_url;
    document.getElementById('overlayImage').src = data.overlay_url;
    document.getElementById('downloadOverlay').href = data.overlay_url;

    // Insight Banner
    if (data.domain_insight) {
        Components.renderInsightBanner(data.domain_insight, data.mode);
    }

    // Stats
    Components.renderStats(data);

    // Timing
    Components.renderTimingBars(data.timing);

    // Image quality
    Components.renderQualityPanel(data.quality_report || data.features_summary?.quality_report);

    // Objects Table
    Components.renderObjectsTable(data.detections);

    // Domain Metrics
    Components.renderDomainMetrics(data.domain_insight, data.mode);

    // Segments
    Components.renderSegments(data.segment_urls);

    // Charts
    setTimeout(() => Components.renderCharts(data), 300);

    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearResults() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('insightBanner').style.display = 'none';
    document.getElementById('qualityCard').style.display = 'none';
    lastResult = null;
    clearFile();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─── History ──────────────────────────────────────────────────
async function loadHistory() {
    try {
        const data = await VisionAPI.getResults(1, 10, lockedMode || currentMode);
        Components.renderHistory(data);
    } catch (e) {
        // History not available yet
    }
}

async function loadPastResult(id) {
    try {
        const data = await VisionAPI.getResult(id);
        // Build a compatible result object
        const result = {
            analysis_id: data.analysis_id,
            timestamp: data.timestamp,
            mode: data.mode,
            original_url: data.original_url,
            overlay_url: data.overlay_url,
            num_objects: data.num_objects,
            detections: [],
            features_summary: data.features,
            quality_report: data.quality_report || data.features?.quality_report || {},
            domain_insight: data.domain_insight,
            segment_urls: data.segments || [],
            timing: {
                total_ms: data.processing_time_ms,
                quality_analysis_ms: 0,
                detection_ms: 0,
                segmentation_ms: 0,
                feature_extraction_ms: 0,
                domain_analysis_ms: 0,
            },
            device: data.device,
        };

        // Set mode selector
        if (!lockedMode) {
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === data.mode);
            });
            currentMode = data.mode;
        } else {
            currentMode = lockedMode;
        }
        applyModeCopy(currentMode);

        showResults(result);
    } catch (e) {
        alert('Failed to load past result');
    }
}
