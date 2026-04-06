/**
 * HAR Mobile App — Real-time Activity Recognition UI
 * 
 * Simulates smartphone sensor reading → windowing → model inference,
 * with live sensor charts, confidence bars, activity timeline, and stats.
 * 
 * In production, the TFLite model runs via tf.lite.Interpreter with
 * real accelerometer/gyroscope data from the DeviceMotion API.
 */

// ═══════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════

const ACTIVITIES = [
    { id: 0, name: 'Lying',    emoji: '🛌', color: '#a78bfa' },
    { id: 1, name: 'Sitting',  emoji: '🪑', color: '#818cf8' },
    { id: 2, name: 'Standing', emoji: '🧍', color: '#60a5fa' },
    { id: 3, name: 'Walking',  emoji: '🚶', color: '#34d399' },
    { id: 4, name: 'Running',  emoji: '🏃', color: '#f87171' },
    { id: 5, name: 'Cycling',  emoji: '🚴', color: '#fbbf24' },
    { id: 6, name: 'Stairs',   emoji: '🪜', color: '#fb923c' },
];

const SAMPLE_RATE = 50;     // Hz (phone)
const WINDOW_SIZE = 256;
const INFERENCE_INTERVAL = 2560;  // ms — one inference per window
const CHART_POINTS = 120;

// ═══════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════

let isRunning = false;
let startTime = null;
let inferenceCount = 0;
let totalConfidence = 0;
let currentActivity = 3;  // Walking by default
let confidences = new Array(7).fill(0);
let history = [];
let latencies = [];

// Sensor data buffers for chart
let sensorBufferX = new Array(CHART_POINTS).fill(0);
let sensorBufferY = new Array(CHART_POINTS).fill(0);
let sensorBufferZ = new Array(CHART_POINTS).fill(0);
let activeSensor = 'accel';

// Animation
let animFrame = null;
let lastInference = 0;

// ═══════════════════════════════════════════════════════════
// DOM Elements
// ═══════════════════════════════════════════════════════════

const $ = id => document.getElementById(id);

const els = {
    activityEmoji:    $('activityEmoji'),
    activityLabel:    $('activityLabel'),
    activityConfidence: $('activityConfidence'),
    ringProgress:     $('ringProgress'),
    statusDot:        $('statusDot'),
    statusText:       $('statusText'),
    confidenceBars:   $('confidenceBars'),
    sensorCanvas:     $('sensorCanvas'),
    historyList:      $('historyList'),
    axisX:            $('axisX'),
    axisY:            $('axisY'),
    axisZ:            $('axisZ'),
    statDuration:     $('statDuration'),
    statInferences:   $('statInferences'),
    statLatency:      $('statLatency'),
    statAvgConf:      $('statAvgConf'),
    statusTime:       $('statusTime'),
    btnStart:         $('btnStart'),
    btnPause:         $('btnPause'),
    btnReset:         $('btnReset'),
    tabAccel:         $('tabAccel'),
    tabGyro:          $('tabGyro'),
};

const ctx = els.sensorCanvas.getContext('2d');

// ═══════════════════════════════════════════════════════════
// Realistic Sensor Simulation
// ═══════════════════════════════════════════════════════════

/**
 * Generate realistic sensor data based on the current activity.
 * Each activity has characteristic acceleration and gyroscope patterns.
 */
function generateSensorData(activity, t) {
    const patterns = {
        0: () => ({ // Lying — very low, stable
            ax: Math.random() * 0.05 - 0.025,
            ay: Math.random() * 0.05 - 0.025 + 9.8,
            az: Math.random() * 0.05 - 0.025,
            gx: Math.random() * 0.02 - 0.01,
            gy: Math.random() * 0.02 - 0.01,
            gz: Math.random() * 0.02 - 0.01,
        }),
        1: () => ({ // Sitting — minimal motion, gravity on y
            ax: Math.random() * 0.1 - 0.05,
            ay: 9.8 + Math.random() * 0.1 - 0.05,
            az: Math.random() * 0.1 - 0.05,
            gx: Math.random() * 0.05 - 0.025,
            gy: Math.random() * 0.05 - 0.025,
            gz: Math.random() * 0.05 - 0.025,
        }),
        2: () => ({ // Standing — slight sway
            ax: Math.sin(t * 0.5) * 0.2 + Math.random() * 0.1 - 0.05,
            ay: 9.8 + Math.sin(t * 0.3) * 0.1,
            az: Math.cos(t * 0.4) * 0.15 + Math.random() * 0.1 - 0.05,
            gx: Math.sin(t * 0.3) * 0.08,
            gy: Math.random() * 0.05 - 0.025,
            gz: Math.cos(t * 0.4) * 0.06,
        }),
        3: () => ({ // Walking — periodic ~2Hz steps
            ax: Math.sin(t * 4 * Math.PI) * 2.5 + Math.random() * 0.3,
            ay: 9.8 + Math.sin(t * 4 * Math.PI + 0.5) * 3.0 + Math.random() * 0.5,
            az: Math.cos(t * 2 * Math.PI) * 1.2 + Math.random() * 0.3,
            gx: Math.sin(t * 4 * Math.PI) * 0.8 + Math.random() * 0.1,
            gy: Math.cos(t * 4 * Math.PI) * 0.5,
            gz: Math.sin(t * 2 * Math.PI) * 0.4,
        }),
        4: () => ({ // Running — higher amplitude ~3Hz
            ax: Math.sin(t * 6 * Math.PI) * 6.0 + Math.random() * 1.0,
            ay: 9.8 + Math.sin(t * 6 * Math.PI + 0.3) * 8.0 + Math.random() * 1.5,
            az: Math.cos(t * 3 * Math.PI) * 3.5 + Math.random() * 0.8,
            gx: Math.sin(t * 6 * Math.PI) * 2.5 + Math.random() * 0.3,
            gy: Math.cos(t * 6 * Math.PI) * 1.8,
            gz: Math.sin(t * 3 * Math.PI) * 1.2,
        }),
        5: () => ({ // Cycling — smooth circular, ~1.5Hz
            ax: Math.sin(t * 3 * Math.PI) * 1.8 + Math.random() * 0.2,
            ay: 9.8 + Math.sin(t * 3 * Math.PI) * 1.5 + Math.random() * 0.3,
            az: Math.cos(t * 3 * Math.PI) * 1.2 + Math.random() * 0.2,
            gx: Math.sin(t * 3 * Math.PI) * 1.5,
            gy: Math.cos(t * 1.5 * Math.PI) * 0.3,
            gz: Math.cos(t * 3 * Math.PI) * 1.0,
        }),
        6: () => ({ // Stairs — irregular high-impact steps
            ax: Math.sin(t * 3.5 * Math.PI) * 3.5 + Math.random() * 0.5,
            ay: 9.8 + Math.abs(Math.sin(t * 3.5 * Math.PI)) * 5.0 + Math.random() * 0.8,
            az: Math.cos(t * 1.75 * Math.PI) * 2.0 + Math.random() * 0.5,
            gx: Math.sin(t * 3.5 * Math.PI) * 1.2,
            gy: Math.sin(t * 7 * Math.PI) * 0.6,
            gz: Math.cos(t * 3.5 * Math.PI) * 0.8,
        }),
    };

    return (patterns[activity] || patterns[3])();
}

// ═══════════════════════════════════════════════════════════
// Simulated Inference
// ═══════════════════════════════════════════════════════════

/**
 * Simulate model inference.
 * In production, this calls tf.lite.Interpreter.run() with real sensor window.
 */
function simulateInference() {
    const t0 = performance.now();

    // Simulate activity changes occasionally
    if (Math.random() < 0.08) {
        const weights = [0.06, 0.08, 0.10, 0.30, 0.18, 0.14, 0.14];
        let r = Math.random();
        for (let i = 0; i < weights.length; i++) {
            r -= weights[i];
            if (r <= 0) { currentActivity = i; break; }
        }
    }

    // Generate confidence distribution (realistic softmax-like)
    const rawConf = ACTIVITIES.map((_, i) => {
        if (i === currentActivity) return 3.0 + Math.random() * 2.0;
        return Math.random() * 0.5;
    });

    // Softmax
    const maxVal = Math.max(...rawConf);
    const exps = rawConf.map(v => Math.exp(v - maxVal));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    confidences = exps.map(v => v / sumExp);

    const topConf = confidences[currentActivity];
    const latency = performance.now() - t0 + Math.random() * 3;

    inferenceCount++;
    totalConfidence += topConf;
    latencies.push(latency);
    if (latencies.length > 50) latencies.shift();

    // Add to history (if activity changed)
    if (history.length === 0 || history[0].activity !== currentActivity) {
        history.unshift({
            activity: currentActivity,
            confidence: topConf,
            time: new Date(),
        });
        if (history.length > 20) history.pop();
    }
}

// ═══════════════════════════════════════════════════════════
// UI Rendering
// ═══════════════════════════════════════════════════════════

function updateActivityDisplay() {
    const act = ACTIVITIES[currentActivity];
    const conf = confidences[currentActivity];

    els.activityEmoji.textContent = act.emoji;
    els.activityLabel.textContent = act.name;
    els.activityConfidence.textContent = Math.round(conf * 100) + '%';

    // Update ring progress (534 = circumference of r=85)
    const offset = 534 * (1 - conf);
    els.ringProgress.style.strokeDashoffset = offset;
    els.ringProgress.style.stroke = act.color;
}

function updateConfidenceBars() {
    const sorted = confidences
        .map((c, i) => ({ index: i, conf: c }))
        .sort((a, b) => b.conf - a.conf);
    
    const topIdx = sorted[0].index;

    els.confidenceBars.innerHTML = sorted.map(({ index, conf }) => {
        const act = ACTIVITIES[index];
        const pct = Math.round(conf * 100);
        const isTop = index === topIdx;
        return `
            <div class="confidence-row">
                <span class="conf-emoji">${act.emoji}</span>
                <span class="conf-name">${act.name}</span>
                <div class="conf-bar-container">
                    <div class="conf-bar ${isTop ? 'highest' : ''}" 
                         style="width: ${pct}%; background: ${act.color}; color: ${act.color}"></div>
                </div>
                <span class="conf-value">${pct}%</span>
            </div>
        `;
    }).join('');
}

function updateSensorChart(t) {
    const data = generateSensorData(currentActivity, t);
    
    const vals = activeSensor === 'accel'
        ? [data.ax, data.ay - 9.8, data.az]  // Remove gravity for display
        : [data.gx, data.gy, data.gz];

    sensorBufferX.push(vals[0]);
    sensorBufferY.push(vals[1]);
    sensorBufferZ.push(vals[2]);
    sensorBufferX.shift();
    sensorBufferY.shift();
    sensorBufferZ.shift();

    els.axisX.textContent = vals[0].toFixed(2);
    els.axisY.textContent = vals[1].toFixed(2);
    els.axisZ.textContent = vals[2].toFixed(2);

    drawChart();
}

function drawChart() {
    const canvas = els.sensorCanvas;
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(1, 1);

    ctx.clearRect(0, 0, w, h);

    // Background grid
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.06)';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += h / 6) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Center line
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Determine scale based on activity amplitude
    const allVals = [...sensorBufferX, ...sensorBufferY, ...sensorBufferZ];
    const maxAbs = Math.max(1, Math.max(...allVals.map(Math.abs)));
    const scale = (h * 0.4) / maxAbs;

    // Draw each axis
    const drawLine = (buffer, color) => {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';

        for (let i = 0; i < buffer.length; i++) {
            const x = (i / (buffer.length - 1)) * w;
            const y = h / 2 - buffer[i] * scale;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Glow effect
        ctx.strokeStyle = color.replace(')', ', 0.15)').replace('rgb', 'rgba');
        ctx.lineWidth = 6;
        ctx.stroke();
    };

    drawLine(sensorBufferX, 'rgb(248, 113, 113)');  // X — red
    drawLine(sensorBufferY, 'rgb(74, 222, 128)');    // Y — green
    drawLine(sensorBufferZ, 'rgb(96, 165, 250)');    // Z — blue
}

function updateHistory() {
    els.historyList.innerHTML = history.map(entry => {
        const act = ACTIVITIES[entry.activity];
        const timeStr = entry.time.toLocaleTimeString('en-US', {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
        return `
            <div class="history-item">
                <span class="history-emoji">${act.emoji}</span>
                <div class="history-info">
                    <div class="history-activity">${act.name}</div>
                    <div class="history-time">${timeStr}</div>
                </div>
                <span class="history-conf">${Math.round(entry.confidence * 100)}%</span>
            </div>
        `;
    }).join('');
}

function updateStats() {
    if (startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        els.statDuration.textContent = `${mins}:${secs}`;
    }

    els.statInferences.textContent = inferenceCount;

    if (latencies.length > 0) {
        const avgLat = latencies.reduce((a, b) => a + b, 0) / latencies.length;
        els.statLatency.textContent = avgLat.toFixed(1) + 'ms';
    }

    if (inferenceCount > 0) {
        els.statAvgConf.textContent = Math.round((totalConfidence / inferenceCount) * 100) + '%';
    }
}

function updateClock() {
    const now = new Date();
    els.statusTime.textContent = now.toLocaleTimeString('en-US', {
        hour: 'numeric', minute: '2-digit'
    });
}

// ═══════════════════════════════════════════════════════════
// Main Loop
// ═══════════════════════════════════════════════════════════

function mainLoop(timestamp) {
    if (!isRunning) return;

    const t = timestamp / 1000;

    // Run inference at interval
    if (timestamp - lastInference >= INFERENCE_INTERVAL) {
        simulateInference();
        updateActivityDisplay();
        updateConfidenceBars();
        updateHistory();
        lastInference = timestamp;
    }

    // Update sensor chart at ~30fps
    updateSensorChart(t);
    updateStats();
    updateClock();

    animFrame = requestAnimationFrame(mainLoop);
}

// ═══════════════════════════════════════════════════════════
// Controls
// ═══════════════════════════════════════════════════════════

function startRecognition() {
    if (isRunning) return;
    isRunning = true;
    if (!startTime) startTime = Date.now();

    els.statusDot.classList.add('active');
    els.statusText.textContent = 'Recognizing activities...';
    els.btnStart.querySelector('span').textContent = 'Running';

    animFrame = requestAnimationFrame(mainLoop);
}

function pauseRecognition() {
    isRunning = false;
    if (animFrame) cancelAnimationFrame(animFrame);

    els.statusDot.classList.remove('active');
    els.statusText.textContent = 'Paused';
    els.btnStart.querySelector('span').textContent = 'Start';
}

function resetSession() {
    pauseRecognition();
    startTime = null;
    inferenceCount = 0;
    totalConfidence = 0;
    history = [];
    latencies = [];
    confidences = new Array(7).fill(0);
    currentActivity = 3;

    sensorBufferX = new Array(CHART_POINTS).fill(0);
    sensorBufferY = new Array(CHART_POINTS).fill(0);
    sensorBufferZ = new Array(CHART_POINTS).fill(0);

    els.statDuration.textContent = '00:00';
    els.statInferences.textContent = '0';
    els.statLatency.textContent = '0ms';
    els.statAvgConf.textContent = '0%';
    els.historyList.innerHTML = '';
    els.activityEmoji.textContent = '🚶';
    els.activityLabel.textContent = 'Walking';
    els.activityConfidence.textContent = '0%';
    els.statusText.textContent = 'Ready';
    
    drawChart();
    updateConfidenceBars();
}

// ═══════════════════════════════════════════════════════════
// Event Listeners
// ═══════════════════════════════════════════════════════════

els.btnStart.addEventListener('click', startRecognition);
els.btnPause.addEventListener('click', pauseRecognition);
els.btnReset.addEventListener('click', resetSession);

els.tabAccel.addEventListener('click', () => {
    activeSensor = 'accel';
    els.tabAccel.classList.add('active');
    els.tabGyro.classList.remove('active');
    sensorBufferX = new Array(CHART_POINTS).fill(0);
    sensorBufferY = new Array(CHART_POINTS).fill(0);
    sensorBufferZ = new Array(CHART_POINTS).fill(0);
});

els.tabGyro.addEventListener('click', () => {
    activeSensor = 'gyro';
    els.tabGyro.classList.add('active');
    els.tabAccel.classList.remove('active');
    sensorBufferX = new Array(CHART_POINTS).fill(0);
    sensorBufferY = new Array(CHART_POINTS).fill(0);
    sensorBufferZ = new Array(CHART_POINTS).fill(0);
});

// ═══════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════

function init() {
    updateClock();
    updateConfidenceBars();
    drawChart();
    
    // Add SVG gradient for the ring
    const svgNS = 'http://www.w3.org/2000/svg';
    const ring = document.querySelector('.ring-svg');
    const defs = document.createElementNS(svgNS, 'defs');
    const grad = document.createElementNS(svgNS, 'linearGradient');
    grad.id = 'progressGradient';
    grad.setAttribute('x1', '0%');
    grad.setAttribute('y1', '0%');
    grad.setAttribute('x2', '100%');
    grad.setAttribute('y2', '100%');
    
    const stop1 = document.createElementNS(svgNS, 'stop');
    stop1.setAttribute('offset', '0%');
    stop1.setAttribute('stop-color', '#6366f1');
    
    const stop2 = document.createElementNS(svgNS, 'stop');
    stop2.setAttribute('offset', '100%');
    stop2.setAttribute('stop-color', '#06b6d4');
    
    grad.appendChild(stop1);
    grad.appendChild(stop2);
    defs.appendChild(grad);
    ring.insertBefore(defs, ring.firstChild);

    // Auto-start after a brief delay
    setTimeout(startRecognition, 800);
}

init();
