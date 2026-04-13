/**
 * HAR Mobile App — Real-time Activity Recognition UI
 *
 * Supports two modes:
 *   REAL — DeviceMotion API (phone accelerometer + gyroscope)
 *          with a heuristic feature-based classifier
 *   SIM  — Realistic sensor simulation (fallback / desktop demo)
 *
 * In production, replace heuristicClassify() with
 * tf.lite.Interpreter.run() or onnxruntime-web inference.
 */

// ═══════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════

const ACTIVITIES = [
    { id: 0, name: 'Lying',    emoji: '🛌', color: '#a78bfa' },
    { id: 1, name: 'Inactive Motion',  emoji: '💤', color: '#818cf8' },
    { id: 2, name: 'Standing', emoji: '🧍', color: '#60a5fa' },
    { id: 3, name: 'Walking',  emoji: '🚶', color: '#34d399' },
    { id: 4, name: 'Running',  emoji: '🏃', color: '#f87171' },
    { id: 5, name: 'Cycling',  emoji: '🚴', color: '#fbbf24' },
    { id: 6, name: 'Stairs',   emoji: '🪜', color: '#fb923c' },
];

const SAMPLE_RATE      = 50;        // Hz (phone sensor poll rate)
const WINDOW_SIZE      = 256;       // samples per inference window (5.12 s @ 50 Hz)
const INFERENCE_INTERVAL = 900;     // ms between inference calls
const CHART_POINTS     = 120;
const MIN_REAL_SAMPLES_FOR_MODEL = 120;
const MIN_REAL_SAMPLES_FOR_HEURISTIC = 40;
const SENSOR_SMOOTHING_ALPHA = 0.25;
const CONFIDENCE_SMOOTHING_ALPHA = 0.70;
const ACTIVITY_SWITCH_MARGIN = 0.08;
const MAX_ACTIVITY_SWITCH_MARGIN = 0.18;
const UNCERTAINTY_THRESHOLD = 0.60;
const STATIC_STEP_FREQ_THRESHOLD = 0.75;
const POCKET_VARIANCE_THRESHOLD = 0.20;
const GRAVITY_LOW_PASS_ALPHA = 0.08;
const MODEL_URL = 'models/hybrid.onnx';
const INFERENCE_ENGINE_STORAGE_KEY = 'har_inference_engine';
const USER_CALIBRATION_STORAGE_KEY = 'har_user_calibration_v1';
const CALIBRATION_DURATION_MS = 3000;
const CALIBRATION_MIN_SAMPLES = 80;
const INACTIVE_TIMEOUT_MS = 500;

const STATIC_CLASS_IDS = [0, 1, 2];
const DYNAMIC_CLASS_IDS = [3, 4, 5, 6];
const VALID_TRANSITIONS = {
    0: [0, 1, 2],
    1: [1, 2],
    2: [2, 1, 0, 3, 5, 6],
    3: [3, 2, 4, 6],
    4: [4, 3, 2],
    5: [5, 2, 3],
    6: [6, 3, 2, 4],
};

// ═══════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════

let isRunning       = false;
let startTime       = null;
let inferenceCount  = 0;
let totalConfidence = 0;
let currentActivity = 3;
let confidences     = new Array(7).fill(0);
let history         = [];
let latencies       = [];
let inferenceSource = 'sim';
let inferenceRunning = false;
let walkingWindowStreak = 0;
let lastStepIntervals = [];
let lastGravityDirection = null;
let uncertaintyActive = false;
let placementMode = 'hand';
let switchInstability = 0;
let topPredictions = [];
let calibrationInProgress = false;
let calibrationProfile = null;

// Sensor chart buffers
let sensorBufferX = new Array(CHART_POINTS).fill(0);
let sensorBufferY = new Array(CHART_POINTS).fill(0);
let sensorBufferZ = new Array(CHART_POINTS).fill(0);
let activeSensor  = 'accel';

// Animation
let animFrame    = null;
let lastInference = 0;

// ── Real sensor state ─────────────────────────────────────
let realSensorActive = false;          // true once DeviceMotion fires
let realSensorBound  = false;          // listener already attached?
let realWindow       = [];             // rolling buffer of {ax,ay,az,gx,gy,gz}
let latestRealSample = null;           // most recent raw DeviceMotion sample
let smoothedSensorSample = null;
let gravityEstimate = null;
let smoothedConfidences = new Array(ACTIVITIES.length).fill(1 / ACTIVITIES.length);
let predictionPrimed = false;
let stillWindowStreak = 0;
let lastMotionTimestamp = Date.now();
let lastRealSensorTimestamp = 0;
let stableSinceTimestamp = 0;
let latestMotionDebug = null;
let inferenceEngine = localStorage.getItem(INFERENCE_ENGINE_STORAGE_KEY) || 'onnx';
let onnxSession = null;
let onnxInputName = null;
let onnxOutputName = null;
let onnxReady = false;
let tfReady = false;
let modelInitPromise = null;

// ═══════════════════════════════════════════════════════════
// DOM Elements
// ═══════════════════════════════════════════════════════════

const $ = id => document.getElementById(id);

const els = {
    activityEmoji:      $('activityEmoji'),
    activityLabel:      $('activityLabel'),
    activityConfidence: $('activityConfidence'),
    heroBarFill:        $('heroBarFill'),
    statusDot:          $('statusDot'),
    statusText:         $('statusText'),
    barChart:           $('barChart'),
    sensorCanvas:       $('sensorCanvas'),
    historyList:        $('historyList'),
    axisX:              $('axisX'),
    axisY:              $('axisY'),
    axisZ:              $('axisZ'),
    statDuration:       $('statDuration'),
    statInferences:     $('statInferences'),
    statLatency:        $('statLatency'),
    statAvgConf:        $('statAvgConf'),
    statusTime:         $('statusTime'),
    btnStart:           $('btnStart'),
    btnPause:           $('btnPause'),
    btnReset:           $('btnReset'),
    settingsBtn:        $('settingsBtn'),
    tabAccel:           $('tabAccel'),
    tabGyro:            $('tabGyro'),
    tabInfo:            $('tabInfo'),
    sensorBadge:        $('sensorBadge'),
    permOverlay:        $('permOverlay'),
    permBtn:            $('permBtn'),
    primaryLabel:       $('primaryLabel'),
    modelToggle:        $('modelToggle'),
    modelRows:          $('modelRows'),
    modelChevron:       $('modelChevron'),
    heroDate:           $('heroDate'),
};

const ctx = els.sensorCanvas.getContext('2d');

// ═══════════════════════════════════════════════════════════
// Real Sensor — DeviceMotion API
// ═══════════════════════════════════════════════════════════

function handleMotionEvent(event) {
    const a = event.accelerationIncludingGravity || {};
    const g = event.rotationRate || {};

    const rawSample = {
        ax: a.x     != null ? a.x                   : 0,
        ay: a.y     != null ? a.y                   : 0,
        az: a.z     != null ? a.z                   : 0,
        gx: g.alpha != null ? g.alpha * Math.PI / 180 : 0,
        gy: g.beta  != null ? g.beta  * Math.PI / 180 : 0,
        gz: g.gamma != null ? g.gamma * Math.PI / 180 : 0,
    };

    const sample = withGravityStabilization(smoothSensorSample(rawSample));
    lastRealSensorTimestamp = Date.now();

    latestRealSample = sample;
    realWindow.push(sample);
    if (realWindow.length > WINDOW_SIZE) realWindow.shift();

    if (!realSensorActive) {
        // First real sample received — announce it
        realSensorActive = true;
        updateSensorBadge();
        showToast('Real sensors active');
        if (isRunning && els.statusText)
            els.statusText.textContent = 'Sensors active';
    }
}

function withGravityStabilization(sample) {
    if (!gravityEstimate) {
        gravityEstimate = { ax: sample.ax, ay: sample.ay, az: sample.az };
    } else {
        gravityEstimate.ax += GRAVITY_LOW_PASS_ALPHA * (sample.ax - gravityEstimate.ax);
        gravityEstimate.ay += GRAVITY_LOW_PASS_ALPHA * (sample.ay - gravityEstimate.ay);
        gravityEstimate.az += GRAVITY_LOW_PASS_ALPHA * (sample.az - gravityEstimate.az);
    }

    const lax = sample.ax - gravityEstimate.ax;
    const lay = sample.ay - gravityEstimate.ay;
    const laz = sample.az - gravityEstimate.az;

    return {
        ...sample,
        lax,
        lay,
        laz,
        gax: gravityEstimate.ax,
        gay: gravityEstimate.ay,
        gaz: gravityEstimate.az,
    };
}

function bindDeviceMotion() {
    if (realSensorBound) return;
    window.addEventListener('devicemotion', handleMotionEvent, { passive: true });
    realSensorBound = true;
    updateSensorBadge();
    // If no event fires within 2 s, the browser is likely blocking sensors
    setTimeout(() => {
        if (!realSensorActive) {
            showToast('No sensor data -- try HTTPS or check browser settings', 5000);
        }
    }, 2000);
}

/**
 * Request DeviceMotion permission (iOS 13+) and bind the listener.
 * Must be called from a user-gesture handler.
 */
async function requestMotionPermission() {
    if (typeof DeviceMotionEvent === 'undefined') {
        // No sensor support (desktop browser)
        return;
    }

    if (typeof DeviceMotionEvent.requestPermission === 'function') {
        // iOS 13+ — must ask explicitly
        try {
            const result = await DeviceMotionEvent.requestPermission();
            if (result === 'granted') {
                bindDeviceMotion();
                hidePermOverlay();
                showToast('Sensor permission granted');
            } else {
                showPermOverlay('Permission denied. Using simulation mode.');
                showToast('Sensor permission denied -- simulation mode', 5000);
            }
        } catch (err) {
            console.warn('Motion permission error:', err);
            showPermOverlay();
        }
    } else {
        // Android Chrome, Firefox, etc. — no permission gate
        bindDeviceMotion();
        hidePermOverlay();
    }
}

function updateSensorBadge() {
    if (!els.sensorBadge) return;
    if (['model', 'onnx', 'tfjs', 'hybrid'].includes(inferenceSource)) {
        els.sensorBadge.textContent = 'MODEL';
        els.sensorBadge.className   = 'sensor-badge model';
    } else if (realSensorActive) {
        els.sensorBadge.textContent = 'REAL';
        els.sensorBadge.className   = 'sensor-badge real';
    } else if (realSensorBound) {
        els.sensorBadge.textContent = 'WAIT';
        els.sensorBadge.className   = 'sensor-badge sim';
    } else {
        els.sensorBadge.textContent = 'SIM';
        els.sensorBadge.className   = 'sensor-badge sim';
    }
}

function smoothSensorSample(sample) {
    if (!smoothedSensorSample) {
        smoothedSensorSample = { ...sample };
        return { ...sample };
    }

    const out = {};
    for (const key of ['ax', 'ay', 'az', 'gx', 'gy', 'gz']) {
        out[key] = smoothedSensorSample[key] + SENSOR_SMOOTHING_ALPHA * (sample[key] - smoothedSensorSample[key]);
    }
    smoothedSensorSample = out;
    return out;
}

function normalizeConfidences(values) {
    const arr = Array.isArray(values) ? values.slice(0, ACTIVITIES.length) : [];
    while (arr.length < ACTIVITIES.length) arr.push(0);
    const clipped = arr.map(v => Math.max(0, Number(v) || 0));
    const sum = clipped.reduce((a, b) => a + b, 0);
    if (sum <= 0) {
        return new Array(ACTIVITIES.length).fill(1 / ACTIVITIES.length);
    }
    return clipped.map(v => v / sum);
}

function buildInactiveTimeoutConfidences() {
    return normalizeConfidences([0.03, 0.82, 0.12, 0.01, 0.01, 0.005, 0.005]);
}

function applyPredictionSmoothing(nextActivity, nextConfidences) {
    const normalized = normalizeConfidences(nextConfidences);

    if (!predictionPrimed) {
        smoothedConfidences = normalized.slice();
        predictionPrimed = true;
    } else {
        // Weighted temporal smoothing: retain short history for borderline windows.
        smoothedConfidences = smoothedConfidences.map(
            (prev, i) => prev + CONFIDENCE_SMOOTHING_ALPHA * (normalized[i] - prev)
        );
        smoothedConfidences = normalizeConfidences(smoothedConfidences);
    }

    topPredictions = smoothedConfidences
        .map((p, idx) => ({ idx, p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3);

    const bestIdx = smoothedConfidences.indexOf(Math.max(...smoothedConfidences));
    const currentConf = smoothedConfidences[currentActivity] || 0;
    const bestConf = smoothedConfidences[bestIdx] || 0;
    const allowedTransitions = VALID_TRANSITIONS[currentActivity] || ACTIVITIES.map((_, i) => i);
    const transitionAllowed = allowedTransitions.includes(bestIdx);
    const dynamicMargin = Math.min(
        MAX_ACTIVITY_SWITCH_MARGIN,
        ACTIVITY_SWITCH_MARGIN + (0.04 * switchInstability)
    );

    uncertaintyActive = bestConf < UNCERTAINTY_THRESHOLD;

    // Confidence-aware decision layer: hold state when uncertain.
    if (uncertaintyActive) {
        switchInstability = Math.min(3, switchInstability + 0.25);
        confidences = smoothedConfidences.slice();
        return;
    }

    let candidateIdx = bestIdx;
    if (!transitionAllowed && bestConf < 0.86) {
        // Enforce realistic state transitions unless confidence is very high.
        candidateIdx = allowedTransitions
            .slice()
            .sort((a, b) => smoothedConfidences[b] - smoothedConfidences[a])[0] ?? currentActivity;
        switchInstability = Math.min(3, switchInstability + 0.2);
    }

    if (candidateIdx === 2 && currentActivity !== 2) {
        currentActivity = 2;
        switchInstability = Math.max(0, switchInstability - 0.2);
    } else if (!(candidateIdx !== currentActivity && (bestConf - currentConf) < dynamicMargin)) {
        if (candidateIdx !== currentActivity) {
            switchInstability = Math.max(0, switchInstability - 0.5);
        } else {
            switchInstability = Math.max(0, switchInstability - 0.15);
        }
        currentActivity = candidateIdx;
    }
    confidences = smoothedConfidences.slice();
}

function detectPlacementMode(stats) {
    return (stats.rawStd < POCKET_VARIANCE_THRESHOLD && stats.gyroMean < 0.22) ? 'pocket' : 'hand';
}

function detectMotionGroup(stats, mode) {
    const varianceThreshold = mode === 'pocket' ? 0.24 : 0.30;
    const gyroThreshold = mode === 'pocket' ? 0.18 : 0.24;
    const isStatic = stats.linStd < varianceThreshold &&
        stats.linRms < (mode === 'pocket' ? 0.34 : 0.40) &&
        stats.gyroMean < gyroThreshold &&
        stats.stepFreq < STATIC_STEP_FREQ_THRESHOLD;
    return isStatic ? 'static' : 'dynamic';
}

function applyMotionGroupGate(probs, group) {
    const gated = probs.slice();
    const suppress = (ids, factor) => ids.forEach(id => { gated[id] *= factor; });
    if (group === 'static') {
        suppress(DYNAMIC_CLASS_IDS, 0.40);
    } else {
        suppress(STATIC_CLASS_IDS, 0.45);
    }
    return normalizeConfidences(gated);
}

function applyCalibrationBias(probs, stats, motionGroup) {
    if (!calibrationProfile || motionGroup !== 'static') return probs;
    const out = probs.slice();

    const standDist =
        Math.abs(stats.tiltPitch - calibrationProfile.standing.tiltPitch) +
        Math.abs(stats.tiltRoll - calibrationProfile.standing.tiltRoll) +
        0.4 * Math.abs(stats.gravityShares.ay - calibrationProfile.standing.gravityAy);
    const sitDist =
        Math.abs(stats.tiltPitch - calibrationProfile.sitting.tiltPitch) +
        Math.abs(stats.tiltRoll - calibrationProfile.sitting.tiltRoll) +
        0.4 * Math.abs(stats.gravityShares.ay - calibrationProfile.sitting.gravityAy);

    if (standDist < sitDist) {
        out[2] += 0.18;
        out[1] *= 0.82;
    } else {
        out[1] += 0.18;
        out[2] *= 0.82;
    }

    return normalizeConfidences(out);
}

function showToast(msg, durationMs = 3500) {
    let toast = document.getElementById('sensorToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'sensorToast';
        toast.style.cssText = [
            'position:fixed', 'bottom:110px', 'left:50%',
            'transform:translateX(-50%)',
            'background:#1B2236', 'color:#F0F4FF',
            'padding:10px 20px', 'border-radius:20px',
            'font-size:12px', 'font-weight:600',
            'border:1px solid rgba(79,127,255,0.3)',
            'box-shadow:0 4px 20px rgba(0,0,0,0.5)',
            'z-index:900', 'white-space:nowrap',
            'transition:opacity 0.4s',
        ].join(';');
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = '1';
    clearTimeout(toast._hideTimer);
    toast._hideTimer = setTimeout(() => { toast.style.opacity = '0'; }, durationMs);
}

function showPermOverlay(msg) {
    if (!els.permOverlay) return;
    if (msg) {
        const msgEl = els.permOverlay.querySelector('.perm-msg');
        if (msgEl) msgEl.textContent = msg;
    }
    els.permOverlay.classList.remove('hidden');
}

function hidePermOverlay() {
    if (els.permOverlay) els.permOverlay.classList.add('hidden');
}

function loadCalibrationProfile() {
    try {
        const raw = localStorage.getItem(USER_CALIBRATION_STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || !parsed.standing || !parsed.sitting) return null;
        return parsed;
    } catch {
        return null;
    }
}

function saveCalibrationProfile(profile) {
    try {
        localStorage.setItem(USER_CALIBRATION_STORAGE_KEY, JSON.stringify(profile));
    } catch (err) {
        console.warn('Could not save calibration:', err);
    }
}

function waitMs(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function collectCalibrationSignature() {
    if (!realSensorActive || realWindow.length < CALIBRATION_MIN_SAMPLES) return null;
    const segment = realWindow.slice(-Math.min(realWindow.length, WINDOW_SIZE));
    const stats = summarizeMotionWindow(segment);
    return {
        tiltPitch: stats.tiltPitch,
        tiltRoll: stats.tiltRoll,
        gravityAy: stats.gravityShares.ay,
        lowFreqEnergy: stats.lowFreqEnergy,
    };
}

async function maybeRunCalibrationFlow(force = false) {
    if ((calibrationProfile && !force) || calibrationInProgress || !realSensorBound) return;

    const wantsCalibration = window.confirm(force
        ? 'Recalibrate now? Stand still for 3 seconds, then sit for 3 seconds.'
        : 'Quick calibration improves sitting vs standing. Stand still for 3 seconds, then sit for 3 seconds. Start now?'
    );
    if (!wantsCalibration) return;

    calibrationInProgress = true;
    try {
        if (!realSensorActive) {
            showToast('Waiting for sensor stream...');
            await waitMs(1200);
        }

        showToast('Calibration 1/2: Stand still');
        await waitMs(CALIBRATION_DURATION_MS);
        const standing = collectCalibrationSignature();

        showToast('Calibration 2/2: Sit still');
        await waitMs(1000);
        await waitMs(CALIBRATION_DURATION_MS);
        const sitting = collectCalibrationSignature();

        if (!standing || !sitting) {
            showToast('Calibration skipped (not enough sensor data)', 4500);
            return;
        }

        calibrationProfile = {
            standing,
            sitting,
            updatedAt: Date.now(),
        };
        saveCalibrationProfile(calibrationProfile);
        showToast(force ? 'Calibration updated' : 'Calibration saved');
    } finally {
        calibrationInProgress = false;
    }
}

function meanOf(values) {
    if (!values.length) return 0;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdOf(values, meanValue = null) {
    if (!values.length) return 0;
    const avg = meanValue != null ? meanValue : meanOf(values);
    const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length;
    return Math.sqrt(variance);
}

function zeroCrossingFrequency(values, sampleRate = SAMPLE_RATE) {
    if (values.length < 2) return 0;
    let crossings = 0;
    for (let i = 1; i < values.length; i++) {
        if ((values[i] >= 0) !== (values[i - 1] >= 0)) crossings++;
    }
    return crossings / (2 * (values.length / sampleRate));
}

/**
 * Calculate jerk (third derivative of position; rate of change of acceleration).
 * High jerk = chaotic shaking. Low jerk = smooth walking motion.
 */
function computeJerk(window) {
    if (window.length < 3) return 0;
    // Remove gravity first for clearer jerk signal
    const axMean = meanOf(window.map(s => s.ax));
    const ayMean = meanOf(window.map(s => s.ay));
    const azMean = meanOf(window.map(s => s.az));
    
    const accels = window.map(s => {
        const linX = s.ax - axMean;
        const linY = s.ay - ayMean;
        const linZ = s.az - azMean;
        return Math.sqrt(linX ** 2 + linY ** 2 + linZ ** 2);
    });
    
    const jerkValues = [];
    for (let i = 2; i < accels.length; i++) {
        const jerk = Math.abs((accels[i] - accels[i - 1]) - (accels[i - 1] - accels[i - 2]));
        jerkValues.push(jerk);
    }
    return jerkValues.length > 0 ? meanOf(jerkValues) : 0;
}

/**
 * Detect peaks in linear acceleration magnitude and measure intervals between them.
 * Real walking has consistent step intervals (~0.5-1.0s for normal gait).
 * Phone shaking has irregular, unpredictable intervals.
 */
function detectStepIntervals(window) {
    const linX = window.map((s, i) => s.ax - meanOf(window.map(x => x.ax)));
    const linY = window.map((s, i) => s.ay - meanOf(window.map(x => x.ay)));
    const linZ = window.map((s, i) => s.az - meanOf(window.map(x => x.az)));
    const linMag = linX.map((_, i) => Math.sqrt(linX[i] ** 2 + linY[i] ** 2 + linZ[i] ** 2));
    
    // Find peaks: local maxima with prominence
    const peaks = [];
    const threshold = meanOf(linMag) + 0.3 * stdOf(linMag);
    for (let i = 1; i < linMag.length - 1; i++) {
        if (linMag[i] > threshold && linMag[i] > linMag[i - 1] && linMag[i] > linMag[i + 1]) {
            peaks.push(i);
        }
    }
    
    // Compute intervals between consecutive peaks (in samples)
    const intervals = [];
    for (let i = 1; i < peaks.length; i++) {
        intervals.push(peaks[i] - peaks[i - 1]);
    }
    return intervals;
}

/**
 * Compute coefficient of variation (stddev / mean) of step intervals.
 * Real walking: 0.1-0.2 (consistent cadence).
 * Shaking: >0.3 (irregular intervals).
 */
function stepIntervalVariance(intervals) {
    if (intervals.length < 2) return 0;
    const mean = meanOf(intervals);
    if (mean < 1) return 0; // No valid steps detected
    const std = stdOf(intervals, mean);
    return std / mean; // coefficient of variation
}

/**
 * Compute spectral entropy (randomness) of linear acceleration.
 * Real walking: low/moderate entropy (periodic signal).
 * Random shaking: high entropy (noise-like signal).
 */
function computeSpectralEntropy(window) {
    const linX = window.map((s, i) => s.ax - meanOf(window.map(x => x.ax)));
    const linY = window.map((s, i) => s.ay - meanOf(window.map(x => x.ay)));
    const linZ = window.map((s, i) => s.az - meanOf(window.map(x => x.az)));
    const linMag = linX.map((_, i) => Math.sqrt(linX[i] ** 2 + linY[i] ** 2 + linZ[i] ** 2));
    
    // Simple FFT approximation: power spectrum via autocorrelation
    const n = Math.min(linMag.length, 64);
    const powers = [];
    const maxFreq = 10; // Max frequency to consider
    
    for (let f = 0; f < maxFreq; f++) {
        let power = 0;
        for (let i = 0; i < n; i++) {
            power += linMag[i] * Math.cos(2 * Math.PI * f * i / n);
        }
        powers.push(Math.abs(power) ** 2);
    }
    
    // Normalize powers to [0,1]
    const sumPower = powers.reduce((a, b) => a + b, 1e-8);
    const probs = powers.map(p => p / sumPower);
    
    // Shannon entropy: -sum(p*log2(p))
    let entropy = 0;
    for (const p of probs) {
        if (p > 0) entropy -= p * Math.log2(p);
    }
    return entropy; // 0=pure tone, ~3-5=random noise
}

/**
 * Check gravity stability: smooth walking has gradual gravity rotation.
 * Shaking has rapid, chaotic orientation flips.
 */
function computeGravityStability(window) {
    if (window.length < 10) return 1.0; // Default: stable
    
    const gravityVectors = [];
    for (let i = 0; i < window.length; i++) {
        const g = Math.sqrt(window[i].ax ** 2 + window[i].ay ** 2 + window[i].az ** 2);
        if (g > 0) {
            gravityVectors.push([
                window[i].ax / g,
                window[i].ay / g,
                window[i].az / g,
            ]);
        }
    }
    
    if (gravityVectors.length < 10) return 1.0;
    
    // Compute dot products between consecutive gravity vectors
    // Close to 1 = same direction (stable), close to -1 = opposite (flipping)
    let dotProducts = [];
    for (let i = 1; i < gravityVectors.length; i++) {
        const g1 = gravityVectors[i - 1];
        const g2 = gravityVectors[i];
        const dot = g1[0] * g2[0] + g1[1] * g2[1] + g1[2] * g2[2];
        dotProducts.push(Math.abs(dot)); // Abs value: we care about magnitude of change
    }
    
    // Stability = mean of dot products (high = stable, low = chaotic)
    return meanOf(dotProducts);
}

function summarizeMotionWindow(window) {
    const axes = ['ax', 'ay', 'az', 'gx', 'gy', 'gz'];
    const series = Object.fromEntries(axes.map(axis => [axis, window.map(sample => sample[axis] || 0)]));

    const rawMeans = {
        ax: meanOf(series.ax),
        ay: meanOf(series.ay),
        az: meanOf(series.az),
    };
    const tiltPitch = Math.atan2(rawMeans.ax, Math.sqrt(rawMeans.ay ** 2 + rawMeans.az ** 2) + 1e-8);
    const tiltRoll = Math.atan2(rawMeans.ay, Math.sqrt(rawMeans.ax ** 2 + rawMeans.az ** 2) + 1e-8);
    const gravityAbs = {
        ax: Math.abs(rawMeans.ax),
        ay: Math.abs(rawMeans.ay),
        az: Math.abs(rawMeans.az),
    };
    const gravityTotal = gravityAbs.ax + gravityAbs.ay + gravityAbs.az + 1e-8;
    const gravityShares = {
        ax: gravityAbs.ax / gravityTotal,
        ay: gravityAbs.ay / gravityTotal,
        az: gravityAbs.az / gravityTotal,
    };
    const dominantGravityAxis = Object.entries(gravityShares)
        .sort((a, b) => b[1] - a[1])[0][0];

    const linSeriesX = window.map(sample => sample.lax ?? 0);
    const linSeriesY = window.map(sample => sample.lay ?? 0);
    const linSeriesZ = window.map(sample => sample.laz ?? 0);
    const linX = linSeriesX.some(v => Math.abs(v) > 1e-6) ? linSeriesX : series.ax.map(value => value - rawMeans.ax);
    const linY = linSeriesY.some(v => Math.abs(v) > 1e-6) ? linSeriesY : series.ay.map(value => value - rawMeans.ay);
    const linZ = linSeriesZ.some(v => Math.abs(v) > 1e-6) ? linSeriesZ : series.az.map(value => value - rawMeans.az);

    const linMag = linX.map((_, index) => Math.sqrt(linX[index] ** 2 + linY[index] ** 2 + linZ[index] ** 2));
    const rawMag = series.ax.map((_, index) => Math.sqrt(series.ax[index] ** 2 + series.ay[index] ** 2 + series.az[index] ** 2));
    const gyroMag = series.gx.map((_, index) => Math.sqrt(series.gx[index] ** 2 + series.gy[index] ** 2 + series.gz[index] ** 2));

    const lowFreqEnergy = linMag.length
        ? Math.abs(linMag.reduce((sum, v, i) => sum + (v * Math.cos((2 * Math.PI * i) / linMag.length)), 0)) / linMag.length
        : 0;

    return {
        linRms: Math.sqrt(meanOf(linMag.map(v => v ** 2))),
        linStd: stdOf(linMag),
        rawStd: stdOf(rawMag),
        gyroMean: meanOf(gyroMag),
        lowFreqEnergy,
        tiltPitch,
        tiltRoll,
        gravityShares,
        dominantGravityAxis,
        stepFreq: Math.max(
            zeroCrossingFrequency(linX),
            zeroCrossingFrequency(linY),
            zeroCrossingFrequency(linZ),
        ),
    };
}

function setInferenceEngine(engine) {
    const allowed = ['onnx', 'tensorflow', 'heuristic'];
    if (!allowed.includes(engine)) return;
    inferenceEngine = engine;
    localStorage.setItem(INFERENCE_ENGINE_STORAGE_KEY, engine);
}

function configureInferenceEngine() {
    const current = inferenceEngine || 'onnx';
    const next = window.prompt(
        'Inference engine: onnx | tensorflow | heuristic',
        current
    );
    if (next === null) return;
    const normalized = next.trim().toLowerCase();
    if (!['onnx', 'tensorflow', 'heuristic'].includes(normalized)) {
        showToast('Invalid engine. Use: onnx, tensorflow, or heuristic');
        return;
    }
    setInferenceEngine(normalized);
    showToast(`Inference engine set to ${normalized.toUpperCase()}`);
}

async function openSettingsMenu() {
    const current = inferenceEngine || 'onnx';
    const action = window.prompt(
        'Settings action: engine | recalibrate | clear-calibration\nYou can also enter an engine directly: onnx | tensorflow | heuristic',
        'engine'
    );
    if (action === null) return;

    const normalized = action.trim().toLowerCase();
    if (!normalized) return;

    if (normalized === 'engine') {
        configureInferenceEngine();
        return;
    }

    if (normalized === 'recalibrate' || normalized === 'calibrate') {
        if (!realSensorBound) {
            await requestMotionPermission();
        }
        if (!realSensorBound) {
            showToast('Sensor permission required for calibration', 4500);
            return;
        }
        await maybeRunCalibrationFlow(true);
        return;
    }

    if (normalized === 'clear-calibration' || normalized === 'clear calibration' || normalized === 'clear') {
        calibrationProfile = null;
        localStorage.removeItem(USER_CALIBRATION_STORAGE_KEY);
        showToast('Calibration cleared');
        return;
    }

    if (['onnx', 'tensorflow', 'heuristic'].includes(normalized)) {
        setInferenceEngine(normalized);
        showToast(`Inference engine set to ${normalized.toUpperCase()}`);
        return;
    }

    showToast(`Unknown action: ${action}`);
}

async function initOnnxModel() {
    if (!window.ort) return false;
    try {
        window.ort.env.wasm.wasmPaths = 'vendor/onnx/';
        onnxSession = await window.ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        });
        onnxInputName = onnxSession.inputNames[0];
        onnxOutputName = onnxSession.outputNames[0];
        onnxReady = true;
        return true;
    } catch (err) {
        console.warn('ONNX init failed:', err);
        onnxReady = false;
        return false;
    }
}

async function initTensorflowRuntime() {
    if (!window.tf) return false;
    try {
        await window.tf.ready();
        tfReady = true;
        return true;
    } catch (err) {
        console.warn('TensorFlow init failed:', err);
        tfReady = false;
        return false;
    }
}

async function initInferenceEngines() {
    if (modelInitPromise) return modelInitPromise;
    modelInitPromise = (async () => {
        const [onnxOk, tfOk] = await Promise.all([initOnnxModel(), initTensorflowRuntime()]);
        if (onnxOk) {
            showToast('ONNX model ready (on-device)');
        } else if (tfOk) {
            showToast('TensorFlow runtime ready (heuristic fallback)');
        } else {
            showToast('Model runtime unavailable, using heuristic');
        }
    })();
    return modelInitPromise;
}

// ═══════════════════════════════════════════════════════════
// Heuristic Classifier (runs in browser on real sensor data)
// ═══════════════════════════════════════════════════════════

/**
 * Feature-based activity classifier.
 * Uses acceleration magnitude, variance, and step frequency.
 * Accurate enough for demo purposes; replace with ONNX/TFLite for production.
 */
function heuristicClassify(window) {
    const N = window.length;
    if (N < 20) return null;

    const stats = summarizeMotionWindow(window);

    // ── Decision rules ────────────────────────────────────────
    // Scores are prior weights; will be softmax-ed for probabilistic output
    const scores = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];

    if (stats.linRms < 0.15 && stats.gyroMean < 0.08 && stats.rawStd < 0.10) {
        if (stats.dominantGravityAxis !== 'ay' && stats.gravityShares.az > 0.34) {
            // Very quiet + flat orientation → Lying
            scores[0] += 2.8;
            scores[1] += 0.3;
        } else {
            // Very quiet but upright → Standing
            scores[2] += 2.2;
            scores[1] += 1.1;
        }

    } else if (stats.linRms < 0.45 && stats.gyroMean < 0.22) {
        // Low motion → Inactive Motion / Standing
        if (stats.dominantGravityAxis === 'ay' && stats.gravityShares.ay > 0.42) {
            scores[2] += 2.0;
            scores[1] += 1.1;
        } else {
            scores[1] += 1.4;
            scores[2] += 1.0;
            scores[0] += 0.2;
        }

        if (stats.linRms < 0.28 && stats.gyroMean < 0.14) {
            scores[1] += 0.45;
        }

    } else if (stats.linRms >= 1.6 && stats.stepFreq > 2.1) {
        // High amplitude + high cadence → Running
        scores[4] += 3.1;
        scores[3] += 0.6;

    } else if (stats.stepFreq > 1.15 && stats.stepFreq < 2.9 && stats.linRms >= 0.65 && stats.linStd > 0.9) {
        // Vertical/irregular cadence cluster → Stairs
        scores[6] += 2.9;
        scores[3] += 0.8;

    } else if (stats.stepFreq >= 1.0 && stats.stepFreq <= 2.7 && stats.linRms >= 0.30 && stats.linRms < 3.4) {
        // Periodic medium amplitude → Walking
        scores[3] += 3.0;
        if (stats.stepFreq > 2.25 || stats.linRms > 2.1) scores[4] += 0.65;

    } else if (stats.gyroMean > 0.45 && stats.linRms < 3.5) {
        // Smooth rhythmic → Cycling
        scores[5] += 2.1;
        scores[3] += 0.4;

    } else if (stats.linStd > 0.9 && stats.stepFreq > 1.0 && stats.stepFreq < 2.4) {
        // Irregular, high variance → Stairs
        scores[6] += 2.3;
        scores[3] += 0.5;

    } else if (stats.linRms >= 0.45) {
        // Some motion, unclear → Standing / Stairs
        scores[2] += 1.2;
        scores[6] += 0.8;
    } else {
        scores[2] += 1.4; // default Standing
    }

    // Boost gyroscope-heavy activities
    if (stats.gyroMean > 0.3) {
        scores[5] += 0.4;
        scores[6] += 0.2;
    }

    if (stats.stepFreq > 0.9) {
        scores[3] += 0.2;
    }

    if (stats.linRms < 0.18 && stats.gyroMean < 0.08) {
        scores[0] += 0.5;
    }

    // Softmax
    const maxS = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - maxS));
    const sumE = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumE);
    const top   = probs.indexOf(Math.max(...probs));

    return { activity: top, confidences: probs };
}

// ═══════════════════════════════════════════════════════════
// Realistic Sensor Simulation (fallback)
// ═══════════════════════════════════════════════════════════

function generateSensorData(activity, t) {
    const patterns = {
        0: () => ({
            ax: Math.random() * 0.05 - 0.025,
            ay: Math.random() * 0.05 - 0.025 + 9.8,
            az: Math.random() * 0.05 - 0.025,
            gx: Math.random() * 0.02 - 0.01,
            gy: Math.random() * 0.02 - 0.01,
            gz: Math.random() * 0.02 - 0.01,
        }),
        1: () => ({
            ax: Math.random() * 0.1 - 0.05,
            ay: 9.8 + Math.random() * 0.1 - 0.05,
            az: Math.random() * 0.1 - 0.05,
            gx: Math.random() * 0.05 - 0.025,
            gy: Math.random() * 0.05 - 0.025,
            gz: Math.random() * 0.05 - 0.025,
        }),
        2: () => ({
            ax: Math.sin(t * 0.5) * 0.2 + Math.random() * 0.1 - 0.05,
            ay: 9.8 + Math.sin(t * 0.3) * 0.1,
            az: Math.cos(t * 0.4) * 0.15 + Math.random() * 0.1 - 0.05,
            gx: Math.sin(t * 0.3) * 0.08,
            gy: Math.random() * 0.05 - 0.025,
            gz: Math.cos(t * 0.4) * 0.06,
        }),
        3: () => ({
            ax: Math.sin(t * 4 * Math.PI) * 2.5 + Math.random() * 0.3,
            ay: 9.8 + Math.sin(t * 4 * Math.PI + 0.5) * 3.0 + Math.random() * 0.5,
            az: Math.cos(t * 2 * Math.PI) * 1.2 + Math.random() * 0.3,
            gx: Math.sin(t * 4 * Math.PI) * 0.8 + Math.random() * 0.1,
            gy: Math.cos(t * 4 * Math.PI) * 0.5,
            gz: Math.sin(t * 2 * Math.PI) * 0.4,
        }),
        4: () => ({
            ax: Math.sin(t * 6 * Math.PI) * 6.0 + Math.random() * 1.0,
            ay: 9.8 + Math.sin(t * 6 * Math.PI + 0.3) * 8.0 + Math.random() * 1.5,
            az: Math.cos(t * 3 * Math.PI) * 3.5 + Math.random() * 0.8,
            gx: Math.sin(t * 6 * Math.PI) * 2.5 + Math.random() * 0.3,
            gy: Math.cos(t * 6 * Math.PI) * 1.8,
            gz: Math.sin(t * 3 * Math.PI) * 1.2,
        }),
        5: () => ({
            ax: Math.sin(t * 3 * Math.PI) * 1.8 + Math.random() * 0.2,
            ay: 9.8 + Math.sin(t * 3 * Math.PI) * 1.5 + Math.random() * 0.3,
            az: Math.cos(t * 3 * Math.PI) * 1.2 + Math.random() * 0.2,
            gx: Math.sin(t * 3 * Math.PI) * 1.5,
            gy: Math.cos(t * 1.5 * Math.PI) * 0.3,
            gz: Math.cos(t * 3 * Math.PI) * 1.0,
        }),
        6: () => ({
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
// Inference — Fully on-device ONNX / TensorFlow / heuristic
// ═══════════════════════════════════════════════════════════

function softmaxArray(values) {
    const maxV = Math.max(...values);
    const exps = values.map(v => Math.exp(v - maxV));
    const sumE = exps.reduce((a, b) => a + b, 0) + 1e-8;
    return exps.map(v => v / sumE);
}

function resampleWindowLinear(windowSamples, targetLen = WINDOW_SIZE) {
    if (!windowSamples.length) return [];
    if (windowSamples.length === targetLen) return windowSamples.slice();
    const out = [];
    const last = windowSamples.length - 1;
    for (let i = 0; i < targetLen; i++) {
        const pos = (i * last) / Math.max(1, targetLen - 1);
        const lo = Math.floor(pos);
        const hi = Math.min(last, lo + 1);
        const alpha = pos - lo;
        const a = windowSamples[lo];
        const b = windowSamples[hi];
        out.push({
            ax: a.ax + (b.ax - a.ax) * alpha,
            ay: a.ay + (b.ay - a.ay) * alpha,
            az: a.az + (b.az - a.az) * alpha,
            gx: a.gx + (b.gx - a.gx) * alpha,
            gy: a.gy + (b.gy - a.gy) * alpha,
            gz: a.gz + (b.gz - a.gz) * alpha,
        });
    }
    return out;
}

function buildModelInput(windowSamples) {
    const w = resampleWindowLinear(windowSamples, WINDOW_SIZE);
    const arr = Array.from({ length: WINDOW_SIZE }, () => new Array(12).fill(0));

    for (let i = 0; i < WINDOW_SIZE; i++) {
        const s = w[i] || { ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 };
        const hand = [s.ax, s.ay, s.az, s.gx, s.gy, s.gz];
        const ankle = hand.map((v, j) => v + 0.01 * Math.sin(0.37 * i + j));
        const merged = hand.concat(ankle);
        for (let c = 0; c < 12; c++) arr[i][c] = merged[c];
    }

    for (let c = 0; c < 12; c++) {
        let mu = 0;
        for (let i = 0; i < WINDOW_SIZE; i++) mu += arr[i][c];
        mu /= WINDOW_SIZE;
        let variance = 0;
        for (let i = 0; i < WINDOW_SIZE; i++) {
            const d = arr[i][c] - mu;
            variance += d * d;
        }
        const sd = Math.sqrt(variance / WINDOW_SIZE) + 1e-8;
        for (let i = 0; i < WINDOW_SIZE; i++) arr[i][c] = (arr[i][c] - mu) / sd;
    }

    const flat = new Float32Array(WINDOW_SIZE * 12);
    let k = 0;
    for (let i = 0; i < WINDOW_SIZE; i++) {
        for (let c = 0; c < 12; c++) flat[k++] = arr[i][c];
    }
    return flat;
}

async function inferWithOnnx(windowSamples) {
    if (!onnxReady || !onnxSession || !window.ort) return null;
    try {
        const input = new window.ort.Tensor('float32', buildModelInput(windowSamples), [1, WINDOW_SIZE, 12]);
        const outputs = await onnxSession.run({ [onnxInputName]: input });
        const logits = Array.from(outputs[onnxOutputName].data || []);
        if (logits.length !== ACTIVITIES.length) return null;
        const probs = softmaxArray(logits);
        return { activity: probs.indexOf(Math.max(...probs)), confidences: probs, source: 'onnx' };
    } catch (err) {
        console.warn('ONNX inference failed:', err);
        return null;
    }
}

function blendPredictions(modelProbs, heuristicProbs, heuristicWeight) {
    const n = Math.min(modelProbs.length, heuristicProbs.length, ACTIVITIES.length);
    const out = new Array(ACTIVITIES.length).fill(0);
    const mw = 1 - heuristicWeight;
    for (let i = 0; i < n; i++) {
        out[i] = (mw * modelProbs[i]) + (heuristicWeight * heuristicProbs[i]);
    }
    return normalizeConfidences(out);
}

function applyAntiSpoofingRules(activity, probs, window, stats) {
    /**
     * Anti-spoofing validation for Walking & Running.
     * Reject false positives from phone shaking using multiple detectors:
     * 1. Stillness gate: if motion is low, block walking/running entirely
     * 2. High jerk (acceleration variance) → chaotic motion
     * 3. Step interval irregularity → non-periodic motion
     * 4. High spectral entropy → noise-like signal
     * 5. Low gravity stability → rapid orientation flips
     * 6. Hysteresis: require 2+ consecutive windows
     */
    
    const adjusted = probs.slice();
    
    // STILLNESS GATE: If motion is clearly low (sitting/standing), block walking/running hard.
    // linRms < 0.22 = still, < 0.35 = low motion
    const isLowMotion = stats.linRms < 0.26 && stats.stepFreq < 0.75;
    if (isLowMotion) {
        // Motion is too low for real walking
        adjusted[3] *= 0.18;  // Strongly suppress walking
        adjusted[4] *= 0.12; // Strongly suppress running
        adjusted[2] += 0.4;  // Strongly boost standing
        adjusted[1] += 0.2;  // Boost sitting
        walkingWindowStreak = 0;
        return adjusted;
    }
    
    // Walking/Running detection
    if (activity === 3 || activity === 4) {
        const jerk = computeJerk(window);
        const intervals = detectStepIntervals(window);
        const intervalVariance = stepIntervalVariance(intervals);
        const entropy = computeSpectralEntropy(window);
        const gravityStability = computeGravityStability(window);
        
        // Anti-shake detector 1: Jerk threshold (stricter)
        // Real walking jerk ~0.2-0.5, shaking jerk >1.0  →  tighten to >0.65
        const highJerk = jerk > 0.65;
        
        // Anti-shake detector 2: Step interval variance (stricter)
        // Real walking CV ~0.15, shaking CV >0.35  →  tighten to >0.32
        const irregularSteps = intervalVariance > 0.32;
        
        // Anti-shake detector 3: Spectral entropy (stricter)
        // Real walking entropy ~2.5-3.5, shaking entropy >4.0  →  tighten to >3.8
        const highEntropy = entropy > 3.8;
        
        // Anti-shake detector 4: Gravity instability (stricter)
        // Real walking stability >0.90, shaking <0.75  →  tighten to <0.80
        const unstableGravity = gravityStability < 0.80;
        
        // Count how many detectors agree on "this looks like shaking"
        let shakeScore = 0;
        if (highJerk) shakeScore++;
        if (irregularSteps) shakeScore++;
        if (highEntropy) shakeScore++;
        if (unstableGravity) shakeScore++;
        
        // Hysteresis: require 3+ consecutive windows (instead of 2)
        // and block if even 1 detector fires (instead of 2)
        const likelyTrueLocomotion = stats.stepFreq > 1.15 && stats.linRms > 0.42;
        if (shakeScore >= 2 || (shakeScore >= 1 && !likelyTrueLocomotion)) {
            // Evidence of instability/shaking, suppress walking
            adjusted[3] *= 0.45;  // Suppress walking
            adjusted[4] *= 0.35;  // Suppress running
            adjusted[2] += 0.25; // Boost standing
            walkingWindowStreak = 0;
        } else {
            // Looks like valid walking/running
            walkingWindowStreak = (walkingWindowStreak || 0) + 1;
            if (walkingWindowStreak < 2) {
                // Require 2 consecutive windows; lightly suppress until confirmed
                adjusted[3] *= 0.72;
                adjusted[4] *= 0.62;
            }
        }
    } else {
        // Not walking/running; reset streak
        walkingWindowStreak = 0;
    }
    
    return adjusted;
}

function applyMotionPostRules(activity, probs, window, stats) {
    const adjusted = probs.slice();

    const isStill = stats.linRms < 0.22 && stats.gyroMean < 0.14 && stats.stepFreq < 0.35;
    stillWindowStreak = isStill ? (stillWindowStreak + 1) : 0;

    // Apply anti-spoofing rules FIRST (must come before other rules)
    let withAntiSpoof = applyAntiSpoofingRules(activity, adjusted, window, stats);

    // Prevent false lying when phone remains upright while still.
    if (activity === 0 && stats.dominantGravityAxis === 'ay' && stats.gravityShares.ay > 0.42) {
        withAntiSpoof[2] += 0.25;
        withAntiSpoof[0] *= 0.55;
    }

    // Lying should appear only after prolonged stillness.
    // With 2s inference interval, 4 windows ~= 8 seconds.
    if (activity === 0 && stillWindowStreak < 4) {
        withAntiSpoof[1] += 0.3;
        withAntiSpoof[2] += 0.35;
        withAntiSpoof[0] *= 0.4;
    }

    // When motion cadence drops after walking, standing should recover quickly.
    if ((activity === 3 || activity === 4) && stats.stepFreq < 0.75 && stats.linRms < 0.32) {
        withAntiSpoof[2] += 0.3;
        withAntiSpoof[3] *= 0.55;
        withAntiSpoof[4] *= 0.45;
    }

    // Micro-motion refinement in static windows:
    // standing typically has slightly higher low-frequency sway than sitting.
    if (stats.stepFreq < 0.6 && stats.gyroMean < 0.2) {
        if (stats.lowFreqEnergy > 0.065) {
            withAntiSpoof[2] += 0.12;
        } else {
            withAntiSpoof[1] += 0.22;
        }
    }

    const norm = normalizeConfidences(withAntiSpoof);
    const top = norm.indexOf(Math.max(...norm));
    return { activity: top, confidences: norm };
}

async function inferWithTensorflow(windowSamples) {
    if (!tfReady || !window.tf) return null;
    let tensor = null;
    let centered = null;
    let linMag = null;
    let gyroMag = null;
    try {
        tensor = window.tf.tensor2d(windowSamples.map(s => [s.ax, s.ay, s.az, s.gx, s.gy, s.gz]));
        centered = tensor.sub(tensor.mean(0, true));
        linMag = centered.slice([0, 0], [-1, 3]).square().sum(1).sqrt();
        gyroMag = centered.slice([0, 3], [-1, 3]).square().sum(1).sqrt();

        const [linRmsData, gyroMeanData] = await Promise.all([
            linMag.square().mean().sqrt().data(),
            gyroMag.mean().data(),
        ]);

        const local = heuristicClassify(windowSamples) || { activity: 2, confidences: new Array(ACTIVITIES.length).fill(1 / ACTIVITIES.length) };
        const probs = local.confidences.slice();
        if (linRmsData[0] < 0.2 && gyroMeanData[0] < 0.12) {
            probs[2] += 0.25;
            probs[0] -= 0.15;
        }
        const norm = normalizeConfidences(probs);
        return { activity: norm.indexOf(Math.max(...norm)), confidences: norm, source: 'tfjs' };
    } catch (err) {
        console.warn('TensorFlow inference failed:', err);
        return null;
    } finally {
        if (tensor) tensor.dispose();
        if (centered) centered.dispose();
        if (linMag) linMag.dispose();
        if (gyroMag) gyroMag.dispose();
    }
}

async function inferOnDevice(windowSamples) {
    if (inferenceEngine === 'onnx') return await inferWithOnnx(windowSamples);
    if (inferenceEngine === 'tensorflow') return await inferWithTensorflow(windowSamples);
    return null;
}

function buildSimWindow() {
    const t0 = performance.now() / 1000;
    const out = [];
    for (let i = 0; i < WINDOW_SIZE; i++) {
        out.push(generateSensorData(currentActivity, t0 + i / SAMPLE_RATE));
    }
    return out;
}

async function runInference() {
    const t0 = performance.now();
    let nextActivity = currentActivity;
    let nextConfidences = confidences;

    const hasReal = realSensorActive && realWindow.length >= MIN_REAL_SAMPLES_FOR_HEURISTIC;
    const activeWindow = hasReal ? realWindow : buildSimWindow();
    const stats = summarizeMotionWindow(activeWindow);
    const recentWindow = activeWindow.slice(-Math.min(activeWindow.length, 64));
    const recentStats = summarizeMotionWindow(recentWindow.length ? recentWindow : activeWindow);
    latestMotionDebug = {
        stepFreq: recentStats.stepFreq,
        linRms: recentStats.linRms,
        gyroMean: recentStats.gyroMean,
    };

    const now = Date.now();
    const strongDynamic = recentStats.stepFreq > 0.95 || recentStats.linRms > 0.55 || recentStats.gyroMean > 0.30;
    const nearStill = recentStats.stepFreq < 0.55 && recentStats.linRms < 0.30 && recentStats.gyroMean < 0.18;
    const stableEnough = recentStats.stepFreq < 0.38 && recentStats.linRms < 0.24 && recentStats.linStd < 0.20 && recentStats.gyroMean < 0.12;

    if (stableEnough) {
        if (!stableSinceTimestamp) stableSinceTimestamp = now;
    } else {
        stableSinceTimestamp = 0;
    }

    if (strongDynamic) {
        lastMotionTimestamp = now;
    }

    const motionTimedOut = hasReal && nearStill && (now - lastMotionTimestamp >= INACTIVE_TIMEOUT_MS);
    const sensorTimedOut = realSensorBound && (now - lastRealSensorTimestamp >= INACTIVE_TIMEOUT_MS);
    const stabilityTimedIn = hasReal && stableSinceTimestamp > 0 && (now - stableSinceTimestamp >= INACTIVE_TIMEOUT_MS);

    if (motionTimedOut || sensorTimedOut || stabilityTimedIn) {
        nextActivity = 1;
        nextConfidences = buildInactiveTimeoutConfidences();
        inferenceSource = sensorTimedOut ? 'inactive-timeout-sensor' : (stabilityTimedIn ? 'inactive-stable' : 'inactive-timeout-motion');

        // Fast-path switch: when the sensor is stably low-motion, update immediately.
        smoothedConfidences = nextConfidences.slice();
        confidences = nextConfidences.slice();
        predictionPrimed = true;
        uncertaintyActive = false;
        topPredictions = smoothedConfidences
            .map((p, idx) => ({ idx, p }))
            .sort((a, b) => b.p - a.p)
            .slice(0, 3);
        currentActivity = 1;

        latencies.push(performance.now() - t0);
        if (latencies.length > 50) latencies.shift();

        const topConf = confidences[currentActivity];
        inferenceCount++;
        totalConfidence += topConf;

        if (history.length === 0 || history[0].activity !== currentActivity) {
            history.unshift({ activity: currentActivity, confidence: topConf, time: new Date() });
            if (history.length > 20) history.pop();
        }

        updateActivityDisplay();
        updateConfidenceBars();
        updateHistory();
        updateSensorBadge();
        return;
    } else {
    const nextPlacementMode = detectPlacementMode(stats);
    if (nextPlacementMode !== placementMode) {
        placementMode = nextPlacementMode;
        showToast(`Phone mode: ${placementMode.toUpperCase()}`);
    }
    const motionGroup = detectMotionGroup(stats, placementMode);

    let modelResult = null;
    if ((!hasReal) || realWindow.length >= MIN_REAL_SAMPLES_FOR_MODEL) {
        modelResult = await inferOnDevice(activeWindow);
    }

    const heuristicResult = heuristicClassify(activeWindow);

    if (modelResult && heuristicResult) {
        const modelTop = Math.max(...modelResult.confidences);
        const heuristicWeight = (stats.stepFreq > 1.1 && stats.linRms > 0.28) ? 0.45 : 0.30;
        let fused = blendPredictions(modelResult.confidences, heuristicResult.confidences, heuristicWeight);

        // If model confidence is low, trust motion heuristic more.
        if (modelTop < 0.42) {
            fused = blendPredictions(modelResult.confidences, heuristicResult.confidences, 0.6);
        }

        const top = fused.indexOf(Math.max(...fused));
        const refined = applyMotionPostRules(top, fused, activeWindow, stats);
        nextActivity = refined.activity;
        nextConfidences = applyCalibrationBias(
            applyMotionGroupGate(refined.confidences, motionGroup),
            stats,
            motionGroup
        );
        inferenceSource = 'hybrid';
    } else if (modelResult) {
        const refined = applyMotionPostRules(modelResult.activity, modelResult.confidences, activeWindow, stats);
        nextActivity = refined.activity;
        nextConfidences = applyCalibrationBias(
            applyMotionGroupGate(refined.confidences, motionGroup),
            stats,
            motionGroup
        );
        inferenceSource = modelResult.source;
    } else if (heuristicResult) {
        const refined = applyMotionPostRules(heuristicResult.activity, heuristicResult.confidences, activeWindow, stats);
        nextActivity = refined.activity;
        nextConfidences = applyCalibrationBias(
            applyMotionGroupGate(refined.confidences, motionGroup),
            stats,
            motionGroup
        );
        inferenceSource = hasReal ? 'heuristic' : 'sim';
    }
    }

    // Walking lock: when cadence and amplitude are clearly gait-like,
    // prevent flicker to static classes unless running/stairs cues dominate.
    const likelyRunning = stats.stepFreq > 2.25 && stats.linRms > 1.6;
    const likelyStairs = stats.linStd > 1.0 && stats.stepFreq > 1.0 && stats.stepFreq < 2.5 && stats.linRms > 0.65;
    const likelyWalking = stats.stepFreq >= 1.05 && stats.stepFreq <= 2.7 &&
        stats.linRms >= 0.40 && stats.linRms <= 2.4 && stats.gyroMean >= 0.09;

    if (!motionTimedOut && !sensorTimedOut && hasReal && likelyWalking && !likelyRunning && !likelyStairs) {
        const boosted = normalizeConfidences(nextConfidences.map((p, idx) => {
            if (idx === 3) return p + 0.45;
            if (idx === 1) return p * 0.55;
            if (idx === 2) return p * 0.60;
            return p;
        }));
        nextConfidences = boosted;
        nextActivity = boosted.indexOf(Math.max(...boosted));
    }

    latencies.push(performance.now() - t0);

    applyPredictionSmoothing(nextActivity, nextConfidences);

    if (latencies.length > 50) latencies.shift();

    const topConf = confidences[currentActivity];
    inferenceCount++;
    totalConfidence += topConf;

    if (history.length === 0 || history[0].activity !== currentActivity) {
        history.unshift({ activity: currentActivity, confidence: topConf, time: new Date() });
        if (history.length > 20) history.pop();
    }

    // Refresh all display panels after async inference completes
    updateActivityDisplay();
    updateConfidenceBars();
    updateHistory();
    updateSensorBadge();
}

// ═══════════════════════════════════════════════════════════
// UI Rendering
// ═══════════════════════════════════════════════════════════

function updateActivityDisplay() {
    const act  = ACTIVITIES[currentActivity];
    const conf = confidences[currentActivity];
    const pct  = Math.round(conf * 100);

    els.activityEmoji.textContent      = act.emoji;
    els.activityLabel.textContent      = act.name.toUpperCase();
    els.activityConfidence.textContent = pct + '%';

    if (els.statusText) {
        if (calibrationInProgress) {
            els.statusText.textContent = 'Calibrating posture...';
        } else if (uncertaintyActive) {
            els.statusText.textContent = 'Still detecting...';
        } else {
            const topLine = topPredictions
                .map(item => `${ACTIVITIES[item.idx].name.slice(0, 3).toUpperCase()}:${Math.round(item.p * 100)}%`)
                .join(' | ');
            const debugLine = latestMotionDebug
                ? `SF:${latestMotionDebug.stepFreq.toFixed(2)} LR:${latestMotionDebug.linRms.toFixed(2)} GM:${latestMotionDebug.gyroMean.toFixed(2)}`
                : 'SF:-- LR:-- GM:--';
            els.statusText.textContent = `${placementMode.toUpperCase()} • ${topLine} • ${debugLine}`;
        }
    }

    // Hero progress bar
    if (els.heroBarFill) {
        els.heroBarFill.style.width      = pct + '%';
        els.heroBarFill.style.background = act.color;
    }
}

function updateConfidenceBars() {
    if (!els.barChart) return;

    const topIdx  = confidences.indexOf(Math.max(...confidences));
    // Max height available = 100px (120px chart - 20px label area)
    const MAX_H   = 100;

    els.barChart.innerHTML = ACTIVITIES.map((act, i) => {
        const pct   = Math.round(confidences[i] * 100);
        const barH  = Math.max(3, Math.round(pct * MAX_H / 100));
        const isTop = i === topIdx;
        return `
            <div class="bar-col">
                ${isTop ? `<span class="bar-pct">${pct}%</span>` : ''}
                <div class="bar-fill ${isTop ? 'top' : ''}"
                     style="height:${barH}px; background:${act.color}; color:${act.color}">
                </div>
                <span class="bar-label">${act.name.toUpperCase().slice(0,3)}</span>
            </div>
        `;
    }).join('');
}

function updateSensorChart(t) {
    let data;

    if (realSensorActive && latestRealSample) {
        // Use live sensor reading
        data = latestRealSample;
    } else {
        // Simulated
        data = generateSensorData(currentActivity, t);
    }

    const vals = activeSensor === 'accel'
        ? [data.ax, data.ay - 9.8, data.az]
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
    const w = canvas.width  = canvas.offsetWidth  * 2;
    const h = canvas.height = canvas.offsetHeight * 2;

    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) {
        const y = (h / 5) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    // Centre line
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.setLineDash([6, 6]);
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    ctx.setLineDash([]);

    const allVals = [...sensorBufferX, ...sensorBufferY, ...sensorBufferZ];
    const maxAbs  = Math.max(1, Math.max(...allVals.map(Math.abs)));
    const scale   = (h * 0.38) / maxAbs;

    const drawLine = (buffer, strokeColor, fillColor) => {
        if (buffer.length < 2) return;

        // Build path
        const pts = buffer.map((v, i) => ({
            x: (i / (buffer.length - 1)) * w,
            y: h / 2 - v * scale,
        }));

        // Gradient fill under the line
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, fillColor);
        grad.addColorStop(1, 'rgba(0,0,0,0)');

        ctx.beginPath();
        ctx.moveTo(pts[0].x, h / 2);
        pts.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.lineTo(pts[pts.length - 1].x, h / 2);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // Line stroke
        ctx.beginPath();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth   = 2.5;
        ctx.lineJoin    = 'round';
        ctx.lineCap     = 'round';
        pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
        ctx.stroke();
    };

    drawLine(sensorBufferX, '#EF6C6C', 'rgba(239,108,108,0.12)');
    drawLine(sensorBufferY, '#4ADE80', 'rgba(74,222,128,0.10)');
    drawLine(sensorBufferZ, '#60A5FA', 'rgba(96,165,250,0.10)');
}

function updateHistory() {
    if (history.length === 0) {
        els.historyList.innerHTML = '<p class="empty-state">Start to see activity history</p>';
        return;
    }
    els.historyList.innerHTML = history.map(entry => {
        const act     = ACTIVITIES[entry.activity];
        const timeStr = entry.time.toLocaleTimeString('en-US', {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
        return `
            <div class="hist-row">
                <div class="hist-color-bar" style="background:${act.color}"></div>
                <div class="hist-info">
                    <div class="hist-name">${act.name}</div>
                    <div class="hist-time">${timeStr}</div>
                </div>
                <span class="hist-conf">${Math.round(entry.confidence * 100)}%</span>
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
        els.statLatency.textContent = avgLat.toFixed(0) + 'ms';
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

    // Fire inference async — guard prevents concurrent calls
    if (timestamp - lastInference >= INFERENCE_INTERVAL && !inferenceRunning) {
        lastInference    = timestamp;
        inferenceRunning = true;
        runInference().finally(() => { inferenceRunning = false; });
    }

    updateSensorChart(t);
    updateStats();
    updateClock();

    animFrame = requestAnimationFrame(mainLoop);
}

// ═══════════════════════════════════════════════════════════
// Controls
// ═══════════════════════════════════════════════════════════

async function startRecognition() {
    if (isRunning) return;

    if (!realSensorBound) {
        await requestMotionPermission();
    }

    if (!calibrationProfile && realSensorBound) {
        await maybeRunCalibrationFlow();
    }

    isRunning = true;
    if (!startTime) startTime = Date.now();
    lastMotionTimestamp = Date.now();
    if (realSensorActive) {
        lastRealSensorTimestamp = Date.now();
    }

    // Always start from Inactive Motion immediately.
    currentActivity = 1;
    const startInactive = buildInactiveTimeoutConfidences();
    confidences = startInactive.slice();
    smoothedConfidences = startInactive.slice();
    predictionPrimed = true;
    topPredictions = smoothedConfidences
        .map((p, idx) => ({ idx, p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3);

    els.statusDot.classList.add('active');
    els.statusText.textContent = realSensorActive ? 'Sensors active' : 'Recognising...';
    if (els.primaryLabel) els.primaryLabel.textContent = 'LIVE';

    updateActivityDisplay();
    updateConfidenceBars();

    // Update hero date
    if (els.heroDate) {
        const d = new Date();
        els.heroDate.textContent = d.toLocaleDateString('en-US',
            { weekday:'short', day:'numeric', month:'short' }).toUpperCase();
    }

    animFrame = requestAnimationFrame(mainLoop);
}

function pauseRecognition() {
    isRunning = false;
    if (animFrame) cancelAnimationFrame(animFrame);

    els.statusDot.classList.remove('active');
    els.statusText.textContent = 'Paused';
    if (els.primaryLabel) els.primaryLabel.textContent = 'START';
}

function resetSession() {
    pauseRecognition();
    startTime       = null;
    inferenceCount  = 0;
    totalConfidence = 0;
    history         = [];
    latencies       = [];
    confidences     = new Array(7).fill(0);
    currentActivity = 3;
    smoothedConfidences = new Array(ACTIVITIES.length).fill(1 / ACTIVITIES.length);
    smoothedSensorSample = null;
    gravityEstimate = null;
    predictionPrimed = false;
    stillWindowStreak = 0;
    lastMotionTimestamp = Date.now();
    lastRealSensorTimestamp = 0;
    stableSinceTimestamp = 0;
    uncertaintyActive = false;
    placementMode = 'hand';
    switchInstability = 0;
    topPredictions = [];
    calibrationInProgress = false;

    sensorBufferX = new Array(CHART_POINTS).fill(0);
    sensorBufferY = new Array(CHART_POINTS).fill(0);
    sensorBufferZ = new Array(CHART_POINTS).fill(0);

    els.statDuration.textContent       = '00:00';
    els.statInferences.textContent     = '0';
    els.statLatency.textContent        = '--';
    els.statAvgConf.textContent        = '--';
    els.activityEmoji.textContent      = '--';
    els.activityLabel.textContent      = 'WAITING';
    els.activityConfidence.textContent = '0%';
    els.statusText.textContent         = 'Ready to start';
    if (els.heroBarFill) { els.heroBarFill.style.width = '0%'; }
    if (els.heroDate) els.heroDate.textContent = 'TODAY';

    updateHistory();
    drawChart();
    updateConfidenceBars();
}

    calibrationProfile = loadCalibrationProfile();

// ═══════════════════════════════════════════════════════════
// Event Listeners
// ═══════════════════════════════════════════════════════════

els.btnStart.addEventListener('click', startRecognition);
els.btnPause.addEventListener('click', pauseRecognition);
els.btnReset.addEventListener('click', resetSession);
if (els.settingsBtn) {
    els.settingsBtn.addEventListener('click', () => {
        openSettingsMenu();
    });
}

// Permission overlay button (iOS)
if (els.permBtn) {
    els.permBtn.addEventListener('click', async () => {
        await requestMotionPermission();
        if (realSensorBound) startRecognition();
    });
}

// Model accordion
if (els.modelToggle) {
    els.modelToggle.addEventListener('click', () => {
        const open = els.modelRows.classList.toggle('open');
        if (els.modelChevron) els.modelChevron.classList.toggle('open', open);
    });
}

function setActiveTab(active) {
    [els.tabAccel, els.tabGyro, els.tabInfo].forEach(t => t && t.classList.remove('active'));
    active.classList.add('active');
}

els.tabAccel.addEventListener('click', () => {
    activeSensor = 'accel';
    setActiveTab(els.tabAccel);
    sensorBufferX = new Array(CHART_POINTS).fill(0);
    sensorBufferY = new Array(CHART_POINTS).fill(0);
    sensorBufferZ = new Array(CHART_POINTS).fill(0);
});

els.tabGyro.addEventListener('click', () => {
    activeSensor = 'gyro';
    setActiveTab(els.tabGyro);
    sensorBufferX = new Array(CHART_POINTS).fill(0);
    sensorBufferY = new Array(CHART_POINTS).fill(0);
    sensorBufferZ = new Array(CHART_POINTS).fill(0);
});

if (els.tabInfo) {
    els.tabInfo.addEventListener('click', () => {
        setActiveTab(els.tabInfo);
        // Open model accordion and scroll to it
        if (els.modelRows) { els.modelRows.classList.add('open'); }
        if (els.modelChevron) { els.modelChevron.classList.add('open'); }
        document.getElementById('modelToggle')
            ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
}

// ═══════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════

function init() {
    updateClock();
    updateConfidenceBars();
    updateSensorBadge();
    drawChart();
    initInferenceEngines();
    if (window.matchMedia('(display-mode: standalone)').matches) {
        showToast(`Engine: ${inferenceEngine.toUpperCase()} (on-device)`);
    }

    // On Android/desktop, bind sensors immediately (no permission gate)
    // On iOS, the Start button triggers the permission request
    if (typeof DeviceMotionEvent !== 'undefined' &&
        typeof DeviceMotionEvent.requestPermission !== 'function') {
        bindDeviceMotion();
    }

    // Auto-start demo after short delay
    setTimeout(startRecognition, 800);
}

init();
