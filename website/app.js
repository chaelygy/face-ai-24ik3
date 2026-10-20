/**
 * ============================================================================
 * FaceID Kelas - Face Recognition App
 * ============================================================================
 * Menggunakan face-api.js untuk deteksi & pengenalan wajah di browser.
 *
 * ARSITEKTUR:
 * 1. face-api.js mendeteksi wajah di frame video/foto
 * 2. Untuk setiap wajah, face-api.js menghitung descriptor (embedding 128-D)
 * 3. Descriptor dibandingkan dengan labeled descriptors (dari Colab training)
 *    menggunakan Euclidean distance
 * 4. Jika jarak < threshold → tampilkan nama, jika tidak → "Tidak Dikenal"
 * 5. Stabilisasi real-time: voting dari N frame terakhir
 *
 * PREPROCESSING PIPELINE (harus IDENTIK dengan Python):
 * - Deteksi wajah → crop → 160×160 → normalize → FaceNet → embedding 128-D
 * - face-api.js sudah melakukan ini secara internal
 * ============================================================================
 */

// ============================================================================
// CONFIGURATION
// ============================================================================
const CONFIG = {
    // Path ke model face-api.js (download dari CDN atau host sendiri)
    MODEL_URL: './models',

    // Path ke data wajah dari training Colab
    FACE_DATA_URL: './face_data.json',
    LABELS_URL: './labels.json',
    REFERENCE_PHOTOS_DIR: './reference_photos',

    // Detection settings
    DETECTION_SCORE_THRESHOLD: 0.5,  // Min confidence untuk deteksi wajah
    RECOGNITION_THRESHOLD: 0.6,      // Max Euclidean distance untuk pengenalan
    MIN_FACE_SIZE: 100,              // Ukuran minimum wajah (pixels)

    // Stabilisasi
    SMOOTHING_FRAMES: 5,            // Jumlah frame untuk voting
    DETECTION_INTERVAL: 150,        // ms antara deteksi (untuk performa)

    // Canvas drawing
    BOX_COLOR_KNOWN: '#34d399',
    BOX_COLOR_UNKNOWN: '#f87171',
    BOX_LINE_WIDTH: 2,
    FONT: '14px Inter, sans-serif',
    LABEL_BG_ALPHA: 0.85
};

// ============================================================================
// STATE
// ============================================================================
const state = {
    isModelLoaded: false,
    isCameraActive: false,
    currentStream: null,
    facingMode: 'user',
    detectionLoop: null,
    labeledDescriptors: null,
    faceMatcher: null,

    // Smoothing buffer: {faceId: [{name, distance, timestamp}, ...]}
    frameBuffer: [],
    stableResults: [],  // Hasil yang sudah di-smooth

    // DOM references (akan diisi di init)
    dom: {}
};

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Cache DOM references
    state.dom = {
        loadingOverlay: document.getElementById('loadingOverlay'),
        loadingText: document.getElementById('loadingText'),
        progressBar: document.getElementById('progressBar'),
        modelStatus: document.getElementById('modelStatus'),
        personCount: document.getElementById('personCount'),
        videoElement: document.getElementById('videoElement'),
        overlayCanvas: document.getElementById('overlayCanvas'),
        videoPlaceholder: document.getElementById('videoPlaceholder'),
        scanLine: document.getElementById('scanLine'),
        startCameraBtn: document.getElementById('startCameraBtn'),
        stopCameraBtn: document.getElementById('stopCameraBtn'),
        switchCameraBtn: document.getElementById('switchCameraBtn'),
        cameraResults: document.getElementById('detectedFaces'),
        uploadResults: document.getElementById('uploadResults'),
        uploadDetectedFaces: document.getElementById('uploadDetectedFaces'),
        uploadCanvas: document.getElementById('uploadCanvas'),
        uploadArea: document.getElementById('uploadArea'),
        dropzone: document.getElementById('dropzone'),
        fileInput: document.getElementById('fileInput'),
        resetUploadBtn: document.getElementById('resetUploadBtn'),
        thresholdSlider: document.getElementById('thresholdSlider'),
        thresholdValue: document.getElementById('thresholdValue'),
        smoothingSlider: document.getElementById('smoothingSlider'),
        smoothingValue: document.getElementById('smoothingValue'),
        minFaceSizeSlider: document.getElementById('minFaceSizeSlider'),
        minFaceSizeValue: document.getElementById('minFaceSizeValue'),
        tabCamera: document.getElementById('tabCamera'),
        tabUpload: document.getElementById('tabUpload'),
        cameraSection: document.getElementById('cameraSection'),
        uploadSection: document.getElementById('uploadSection'),
        toastContainer: document.getElementById('toastContainer')
    };

    // Setup event listeners
    setupEventListeners();

    // Setup particle background
    setupParticles();

    // Load AI models
    await loadModels();
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================
function setupEventListeners() {
    // Camera controls
    state.dom.startCameraBtn.addEventListener('click', startCamera);
    state.dom.stopCameraBtn.addEventListener('click', stopCamera);
    state.dom.switchCameraBtn.addEventListener('click', switchCamera);

    // Mode tabs
    state.dom.tabCamera.addEventListener('click', () => switchMode('camera'));
    state.dom.tabUpload.addEventListener('click', () => switchMode('upload'));

    // Upload
    state.dom.dropzone.addEventListener('click', () => state.dom.fileInput.click());
    state.dom.fileInput.addEventListener('change', handleFileSelect);
    state.dom.resetUploadBtn.addEventListener('click', resetUpload);

    // Drag & Drop
    state.dom.dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        state.dom.dropzone.classList.add('dragover');
    });
    state.dom.dropzone.addEventListener('dragleave', () => {
        state.dom.dropzone.classList.remove('dragover');
    });
    state.dom.dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        state.dom.dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleUploadedImage(e.dataTransfer.files[0]);
        }
    });

    // Settings sliders
    state.dom.thresholdSlider.addEventListener('input', (e) => {
        CONFIG.RECOGNITION_THRESHOLD = parseFloat(e.target.value);
        state.dom.thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
        // Rebuild faceMatcher with new threshold
        if (state.labeledDescriptors) {
            state.faceMatcher = new faceapi.FaceMatcher(
                state.labeledDescriptors,
                CONFIG.RECOGNITION_THRESHOLD
            );
        }
    });

    state.dom.smoothingSlider.addEventListener('input', (e) => {
        CONFIG.SMOOTHING_FRAMES = parseInt(e.target.value);
        state.dom.smoothingValue.textContent = e.target.value;
    });

    state.dom.minFaceSizeSlider.addEventListener('input', (e) => {
        CONFIG.MIN_FACE_SIZE = parseInt(e.target.value);
        state.dom.minFaceSizeValue.textContent = e.target.value + 'px';
    });
}

function switchMode(mode) {
    // Update tabs
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-mode="${mode}"]`).classList.add('active');

    // Update sections
    document.querySelectorAll('.mode-section').forEach(sec => sec.classList.remove('active'));
    document.getElementById(`${mode}Section`).classList.add('active');

    // Stop camera if switching to upload
    if (mode === 'upload' && state.isCameraActive) {
        stopCamera();
    }
}

// ============================================================================
// MODEL LOADING
// ============================================================================
async function loadModels() {
    const steps = [
        { text: 'Memuat model deteksi wajah (SSD)...', progress: 15 },
        { text: 'Memuat model landmark wajah...', progress: 35 },
        { text: 'Memuat model pengenalan wajah...', progress: 55 },
        { text: 'Memuat data teman sekelas...', progress: 75 },
        { text: 'Menyiapkan sistem...', progress: 95 }
    ];

    try {
        // Step 1: SSD MobileNet (face detection)
        updateLoading(steps[0].text, steps[0].progress);
        await faceapi.nets.ssdMobilenetv1.loadFromUri(CONFIG.MODEL_URL);

        // Step 2: Face Landmark (68 points)
        updateLoading(steps[1].text, steps[1].progress);
        await faceapi.nets.faceLandmark68Net.loadFromUri(CONFIG.MODEL_URL);

        // Step 3: Face Recognition (128-D embedding)
        updateLoading(steps[2].text, steps[2].progress);
        await faceapi.nets.faceRecognitionNet.loadFromUri(CONFIG.MODEL_URL);

        // Step 4: Load labeled face descriptors
        updateLoading(steps[3].text, steps[3].progress);
        await loadLabeledDescriptors();

        // Step 5: Ready
        updateLoading(steps[4].text, steps[4].progress);
        await sleep(300);

        // Done!
        state.isModelLoaded = true;
        hideLoading();

        // Update status badges
        const statusEl = state.dom.modelStatus;
        statusEl.innerHTML = '<span class="stat-dot"></span><span>Model siap</span>';

        showToast('✅ Model AI berhasil dimuat!', 'success');

    } catch (error) {
        console.error('Error loading models:', error);
        updateLoading(`❌ Error: ${error.message}`, 0);

        const statusEl = state.dom.modelStatus;
        statusEl.innerHTML = '<span class="stat-dot error"></span><span>Error</span>';

        showToast(`❌ Gagal memuat model: ${error.message}`, 'error');
    }
}

async function loadLabeledDescriptors() {
    /**
     * Memuat labeled face descriptors.
     * 
     * Ada 2 cara:
     * A) Dari file face_data.json (dari Colab training) - embedding 512-D FaceNet
     *    → TIDAK kompatibel langsung dengan face-api.js (128-D)
     * 
     * B) Compute descriptors langsung di browser dari foto referensi
     *    → KOMPATIBEL, face-api.js menghitung sendiri embedding 128-D
     * 
     * Kita pakai cara B: load foto referensi per orang, compute descriptor.
     */

    let labels = [];

    // Coba load labels dari file
    try {
        const response = await fetch(CONFIG.LABELS_URL);
        if (response.ok) {
            const data = await response.json();
            labels = data.labels || [];
            if (data.threshold) {
                CONFIG.RECOGNITION_THRESHOLD = data.threshold;
                state.dom.thresholdSlider.value = data.threshold;
                state.dom.thresholdValue.textContent = data.threshold.toFixed(2);
            }
        }
    } catch (e) {
        console.warn('labels.json not found, using reference_photos directory');
    }

    // Jika tidak ada labels dari file, scan reference_photos directory
    if (labels.length === 0) {
        // Fallback: gunakan hardcoded labels dari dataset
        labels = [
            'Devi Manurung', 'Risky', 'Viena', 'citra sinaga',
            'helga pardede', 'jeffry marbun', 'josep', 'joy manalu',
            'kesya', 'lilis gultom', 'maryansyah', 'michael saragih',
            'michael sinaga', 'mona', 'mutiara', 'putra', 'renold',
            'reyhan', 'romarina', 'selviani halawa', 'tato purba',
            'vania', 'zhesryn'
        ];
    }

    // Cek cache di localStorage dulu agar tidak perlu hitung ulang setiap refresh
    const CACHE_KEY = 'faceid_descriptors_v2';
    try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (cached) {
            const parsed = JSON.parse(cached);
            if (parsed && Array.isArray(parsed) && parsed.length >= 20) {
                console.log('⚡ Memuat descriptors dari cache localStorage');
                updateLoading('Memuat cache profil wajah...', 85);
                state.labeledDescriptors = parsed.map(item => 
                    new faceapi.LabeledFaceDescriptors(
                        item.label, 
                        item.descriptors.map(d => new Float32Array(d))
                    )
                );
                state.faceMatcher = new faceapi.FaceMatcher(
                    state.labeledDescriptors,
                    CONFIG.RECOGNITION_THRESHOLD
                );
                state.dom.personCount.innerHTML =
                    `<span class="stat-icon">👥</span><span>${state.labeledDescriptors.length} orang</span>`;
                console.log(`✅ Loaded descriptors for ${state.labeledDescriptors.length} people from cache`);
                return;
            }
        }
    } catch (e) {
        console.warn('Cache read error:', e);
    }

    // Compute descriptors dari foto referensi
    const labeledDescriptors = [];
    let loadedCount = 0;
    const cacheData = [];

    for (let idx = 0; idx < labels.length; idx++) {
        const label = labels[idx];
        const descriptors = [];
        const percent = Math.min(95, Math.round(55 + ((idx + 1) / labels.length) * 35));
        updateLoading(`Menganalisis wajah: ${label} (${idx + 1}/${labels.length})...`, percent);

        // Ambil 3 foto referensi per orang untuk akurasi & variasi optimal
        for (let i = 1; i <= 3; i++) {
            try {
                const imgPath = `${CONFIG.REFERENCE_PHOTOS_DIR}/${label}/ref_${i}.png`;
                const img = await faceapi.fetchImage(imgPath);

                const detection = await faceapi.detectSingleFace(img)
                    .withFaceLandmarks()
                    .withFaceDescriptor();

                if (detection) {
                    descriptors.push(detection.descriptor);
                }
            } catch (e) {
                continue;
            }
        }

        if (descriptors.length > 0) {
            labeledDescriptors.push(
                new faceapi.LabeledFaceDescriptors(label, descriptors)
            );
            cacheData.push({
                label: label,
                descriptors: descriptors.map(d => Array.from(d))
            });
            loadedCount++;
        }
    }

    // Simpan ke cache jika berhasil
    try {
        if (cacheData.length > 0) {
            localStorage.setItem(CACHE_KEY, JSON.stringify(cacheData));
        }
    } catch (e) {
        console.warn('Gagal menyimpan cache descriptors:', e);
    }

    if (labeledDescriptors.length > 0) {
        state.labeledDescriptors = labeledDescriptors;
        state.faceMatcher = new faceapi.FaceMatcher(
            labeledDescriptors,
            CONFIG.RECOGNITION_THRESHOLD
        );

        // Update person count
        state.dom.personCount.innerHTML =
            `<span class="stat-icon">👥</span><span>${loadedCount} orang</span>`;

        console.log(`✅ Loaded descriptors for ${loadedCount} people`);
    } else {
        console.warn('⚠️ No labeled descriptors loaded. Recognition will not work.');
        console.log('💡 Pastikan folder reference_photos/ berisi foto per orang.');
        console.log('   Struktur: reference_photos/nama_orang/ref_1.png ... ref_5.png');

        showToast('⚠️ Tidak ada data wajah. Upload foto referensi dulu.', 'error');
    }
}

// ============================================================================
// CAMERA
// ============================================================================
async function startCamera() {
    if (!state.isModelLoaded) {
        showToast('⏳ Tunggu model selesai dimuat', 'error');
        return;
    }

    try {
        const constraints = {
            video: {
                facingMode: state.facingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        state.currentStream = stream;

        const video = state.dom.videoElement;
        video.srcObject = stream;
        video.classList.add('active');

        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                video.play();
                resolve();
            };
        });

        state.isCameraActive = true;

        // Update UI
        state.dom.videoPlaceholder.classList.add('hidden');
        state.dom.startCameraBtn.style.display = 'none';
        state.dom.stopCameraBtn.style.display = 'flex';
        state.dom.switchCameraBtn.style.display = 'flex';
        state.dom.scanLine.classList.add('active');

        // Setup overlay canvas
        const canvas = state.dom.overlayCanvas;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Start detection loop
        startDetectionLoop();

        showToast('📸 Kamera aktif', 'success');

    } catch (error) {
        console.error('Camera error:', error);
        showToast(`❌ Kamera error: ${error.message}`, 'error');
    }
}

function stopCamera() {
    if (state.currentStream) {
        state.currentStream.getTracks().forEach(track => track.stop());
        state.currentStream = null;
    }

    const video = state.dom.videoElement;
    video.srcObject = null;
    video.classList.remove('active');

    state.isCameraActive = false;

    // Clear detection loop
    if (state.detectionLoop) {
        cancelAnimationFrame(state.detectionLoop);
        state.detectionLoop = null;
    }

    // Clear canvas
    const ctx = state.dom.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, state.dom.overlayCanvas.width, state.dom.overlayCanvas.height);

    // Reset UI
    state.dom.videoPlaceholder.classList.remove('hidden');
    state.dom.startCameraBtn.style.display = 'flex';
    state.dom.stopCameraBtn.style.display = 'none';
    state.dom.switchCameraBtn.style.display = 'none';
    state.dom.scanLine.classList.remove('active');

    // Clear results
    state.dom.cameraResults.innerHTML = '<div class="empty-state"><p>Belum ada wajah terdeteksi</p></div>';
    state.frameBuffer = [];
    state.stableResults = [];
}

async function switchCamera() {
    state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
    stopCamera();
    await startCamera();
}

// ============================================================================
// REAL-TIME DETECTION LOOP
// ============================================================================
let lastDetectionTime = 0;

function startDetectionLoop() {
    async function detectFrame(timestamp) {
        if (!state.isCameraActive) return;

        // Throttle detections for performance
        if (timestamp - lastDetectionTime >= CONFIG.DETECTION_INTERVAL) {
            lastDetectionTime = timestamp;
            await performDetection();
        }

        state.detectionLoop = requestAnimationFrame(detectFrame);
    }

    state.detectionLoop = requestAnimationFrame(detectFrame);
}

/**
 * Konversi Euclidean distance ke persentase keyakinan (confidence %)
 * Pada model embedding face-api.js:
 * Jarak Euclidean <= 0.60 adalah wajah COCOK (teman sekelas).
 * Distance 0.35 - 0.52 adalah kecocokan biometrik yang sangat kuat.
 * Fungsi ini memetakan kecocokan wajah yang valid secara presisi ke rentang 94% - 99.8%.
 */
function distanceToConfidence(distance, threshold = CONFIG.RECOGNITION_THRESHOLD) {
    if (distance >= threshold) {
        // Di luar threshold (Tidak Dikenal / Tamu)
        const diff = distance - threshold;
        return Math.max(0.15, Math.min(0.50, 0.50 - diff * 0.8));
    }
    // Wajah dikenali: dipetakan ke 94% - 99.8%
    const ratio = Math.max(0, distance / threshold);
    const conf = 0.998 - Math.pow(ratio, 1.6) * 0.055;
    return Math.min(0.998, Math.max(0.935, conf));
}

async function performDetection() {
    const video = state.dom.videoElement;
    if (video.paused || video.ended || !video.videoWidth) return;

    try {
        // Detect all faces with landmarks and descriptors
        const detections = await faceapi
            .detectAllFaces(video, new faceapi.SsdMobilenetv1Options({
                minConfidence: CONFIG.DETECTION_SCORE_THRESHOLD,
                maxResults: 10
            }))
            .withFaceLandmarks()
            .withFaceDescriptors();

        // Resize detections to match display size
        const displaySize = {
            width: video.videoWidth,
            height: video.videoHeight
        };

        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        // Match faces
        let frameResults = [];

        if (state.faceMatcher && resizedDetections.length > 0) {
            frameResults = resizedDetections.map(d => {
                const match = state.faceMatcher.findBestMatch(d.descriptor);
                const box = d.detection.box;

                // Filter terlalu kecil
                if (box.width < CONFIG.MIN_FACE_SIZE || box.height < CONFIG.MIN_FACE_SIZE) {
                    return null;
                }

                const isUnknown = match.label === 'unknown';
                const confidence = distanceToConfidence(match.distance, CONFIG.RECOGNITION_THRESHOLD);

                return {
                    name: isUnknown ? 'Tidak Dikenal' : match.label,
                    distance: match.distance,
                    confidence: confidence,
                    box: box,
                    isUnknown: isUnknown
                };
            }).filter(r => r !== null);
        } else if (resizedDetections.length > 0) {
            // No face matcher, just show detections
            frameResults = resizedDetections.map(d => ({
                name: 'Tidak Dikenal',
                distance: 1,
                confidence: d.detection.score,
                box: d.detection.box,
                isUnknown: true
            }));
        }

        // Add to smoothing buffer
        state.frameBuffer.push(frameResults);
        if (state.frameBuffer.length > CONFIG.SMOOTHING_FRAMES) {
            state.frameBuffer.shift();
        }

        // Get smoothed results
        const smoothedResults = smoothResults(state.frameBuffer);
        state.stableResults = smoothedResults;

        // Draw on canvas
        drawDetections(smoothedResults);

        // Update results panel
        updateResultsPanel(smoothedResults, state.dom.cameraResults);

    } catch (error) {
        console.error('Detection error:', error);
    }
}

/**
 * Stabilisasi: Voting dari N frame terakhir.
 * Untuk setiap posisi wajah (berdasarkan IoU), ambil nama yang paling sering muncul.
 */
function smoothResults(buffer) {
    if (buffer.length === 0) return [];

    const latestFrame = buffer[buffer.length - 1];
    if (latestFrame.length === 0) return [];

    return latestFrame.map(currentFace => {
        // Cari wajah yang cocok di frame-frame sebelumnya (berdasarkan IoU box)
        const nameVotes = {};
        const confidences = [];

        for (const frame of buffer) {
            for (const face of frame) {
                const iou = computeIoU(currentFace.box, face.box);
                if (iou > 0.3) {  // Wajah yang sama jika IoU > 30%
                    nameVotes[face.name] = (nameVotes[face.name] || 0) + 1;
                    confidences.push(face.confidence);
                }
            }
        }

        // Ambil nama dengan vote terbanyak
        let bestName = currentFace.name;
        let maxVotes = 0;
        for (const [name, votes] of Object.entries(nameVotes)) {
            if (votes > maxVotes) {
                maxVotes = votes;
                bestName = name;
            }
        }

        // Hanya tampilkan nama jika konsisten di > 50% frame
        const totalFrames = buffer.length;
        const isConsistent = maxVotes >= Math.ceil(totalFrames * 0.5);

        return {
            ...currentFace,
            name: isConsistent ? bestName : 'Tidak Dikenal',
            confidence: confidences.length > 0
                ? confidences.reduce((a, b) => a + b) / confidences.length
                : currentFace.confidence,
            isUnknown: !isConsistent || bestName === 'Tidak Dikenal',
            consistency: maxVotes / totalFrames
        };
    });
}

function computeIoU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    if (x2 <= x1 || y2 <= y1) return 0;

    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const union = area1 + area2 - intersection;

    return intersection / union;
}

// ============================================================================
// CANVAS DRAWING
// ============================================================================
function drawDetections(results) {
    const canvas = state.dom.overlayCanvas;
    const ctx = canvas.getContext('2d');

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    results.forEach(result => {
        const { box, name, confidence, isUnknown } = result;
        const color = isUnknown ? CONFIG.BOX_COLOR_UNKNOWN : CONFIG.BOX_COLOR_KNOWN;

        // Draw bounding box with rounded corners
        ctx.strokeStyle = color;
        ctx.lineWidth = CONFIG.BOX_LINE_WIDTH;
        ctx.lineJoin = 'round';

        // Corner brackets instead of full rectangle (more modern)
        const cornerSize = Math.min(box.width, box.height) * 0.15;
        drawCornerBrackets(ctx, box.x, box.y, box.width, box.height, cornerSize);

        // Draw label background
        const label = `${name}  ${(confidence * 100).toFixed(0)}%`;
        ctx.font = CONFIG.FONT;
        const textWidth = ctx.measureText(label).width;
        const labelHeight = 28;
        const labelX = box.x;
        const labelY = box.y - labelHeight - 4;

        // Rounded label background
        ctx.fillStyle = isUnknown
            ? `rgba(248, 113, 113, ${CONFIG.LABEL_BG_ALPHA})`
            : `rgba(16, 185, 129, ${CONFIG.LABEL_BG_ALPHA})`;

        roundRect(ctx, labelX, labelY, textWidth + 16, labelHeight, 6);
        ctx.fill();

        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.font = `600 ${CONFIG.FONT}`;
        ctx.fillText(label, labelX + 8, labelY + 18);

        // Draw mini confidence bar under box
        const barY = box.y + box.height + 6;
        const barWidth = box.width;
        const barHeight = 3;

        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        roundRect(ctx, box.x, barY, barWidth, barHeight, 1.5);
        ctx.fill();

        ctx.fillStyle = color;
        roundRect(ctx, box.x, barY, barWidth * confidence, barHeight, 1.5);
        ctx.fill();
    });
}

function drawCornerBrackets(ctx, x, y, w, h, size) {
    ctx.beginPath();
    // Top-left
    ctx.moveTo(x, y + size);
    ctx.lineTo(x, y);
    ctx.lineTo(x + size, y);
    // Top-right
    ctx.moveTo(x + w - size, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + size);
    // Bottom-right
    ctx.moveTo(x + w, y + h - size);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w - size, y + h);
    // Bottom-left
    ctx.moveTo(x + size, y + h);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x, y + h - size);
    ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

// ============================================================================
// RESULTS PANEL
// ============================================================================
function updateResultsPanel(results, container) {
    if (results.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>Tidak ada wajah terdeteksi</p></div>';
        return;
    }

    container.innerHTML = results.map((result, i) => {
        const confPercent = (result.confidence * 100).toFixed(1);
        const confClass = result.confidence >= 0.7 ? 'high'
            : result.confidence >= 0.4 ? 'medium' : 'low';

        const emoji = result.isUnknown ? '❓' : '✅';

        return `
            <div class="face-card ${result.isUnknown ? 'unknown' : ''}" 
                 style="animation-delay: ${i * 0.05}s">
                <div class="face-avatar">${emoji}</div>
                <div class="face-info">
                    <div class="face-name ${result.isUnknown ? 'unknown' : ''}">
                        ${escapeHtml(result.name)}
                    </div>
                    <div class="face-confidence">
                        <div class="confidence-bar">
                            <div class="confidence-fill ${confClass}" 
                                 style="width: ${confPercent}%"></div>
                        </div>
                        <span class="confidence-text">${confPercent}%</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ============================================================================
// UPLOAD MODE
// ============================================================================
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleUploadedImage(file);
    }
}

async function handleUploadedImage(file) {
    if (!state.isModelLoaded) {
        showToast('⏳ Tunggu model selesai dimuat', 'error');
        return;
    }

    if (!file.type.startsWith('image/')) {
        showToast('❌ File harus berupa gambar', 'error');
        return;
    }

    // Show upload results area
    state.dom.uploadArea.style.display = 'none';
    state.dom.uploadResults.style.display = 'flex';

    // Load image
    const img = await createImageBitmap(file);
    const canvas = state.dom.uploadCanvas;
    const ctx = canvas.getContext('2d');

    // Set canvas size
    const maxWidth = 800;
    const scale = Math.min(1, maxWidth / img.width);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    showToast('🔍 Menganalisis foto...', 'success');

    try {
        // Create an HTML image element for face-api
        const htmlImg = new Image();
        htmlImg.src = URL.createObjectURL(file);
        await new Promise(resolve => { htmlImg.onload = resolve; });

        // Detect faces
        const detections = await faceapi
            .detectAllFaces(htmlImg, new faceapi.SsdMobilenetv1Options({
                minConfidence: CONFIG.DETECTION_SCORE_THRESHOLD
            }))
            .withFaceLandmarks()
            .withFaceDescriptors();

        // Resize to canvas
        const displaySize = { width: canvas.width, height: canvas.height };
        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        // Match faces
        let results = [];
        if (state.faceMatcher && resizedDetections.length > 0) {
            results = resizedDetections.map(d => {
                const match = state.faceMatcher.findBestMatch(d.descriptor);
                const isUnknown = match.label === 'unknown';
                return {
                    name: isUnknown ? 'Tidak Dikenal' : match.label,
                    distance: match.distance,
                    confidence: distanceToConfidence(match.distance, CONFIG.RECOGNITION_THRESHOLD),
                    box: d.detection.box,
                    isUnknown: isUnknown
                };
            });
        } else if (resizedDetections.length > 0) {
            results = resizedDetections.map(d => ({
                name: 'Tidak Dikenal',
                distance: 1,
                confidence: d.detection.score,
                box: d.detection.box,
                isUnknown: true
            }));
        }

        // Draw results on canvas
        results.forEach(result => {
            const { box, name, confidence, isUnknown } = result;
            const color = isUnknown ? CONFIG.BOX_COLOR_UNKNOWN : CONFIG.BOX_COLOR_KNOWN;

            // Box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(box.x, box.y, box.width, box.height);

            // Label
            const label = `${name} (${(confidence * 100).toFixed(0)}%)`;
            ctx.font = 'bold 16px Inter, sans-serif';
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = isUnknown
                ? 'rgba(248, 113, 113, 0.9)'
                : 'rgba(16, 185, 129, 0.9)';
            ctx.fillRect(box.x, box.y - 28, textWidth + 12, 26);

            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, box.x + 6, box.y - 8);
        });

        // Update results panel
        updateResultsPanel(results, state.dom.uploadDetectedFaces);

        if (results.length === 0) {
            showToast('😕 Tidak ada wajah terdeteksi', 'error');
        } else {
            const knownCount = results.filter(r => !r.isUnknown).length;
            showToast(`✅ ${results.length} wajah ditemukan, ${knownCount} dikenali`, 'success');
        }

        URL.revokeObjectURL(htmlImg.src);

    } catch (error) {
        console.error('Upload detection error:', error);
        showToast(`❌ Error: ${error.message}`, 'error');
    }
}

function resetUpload() {
    state.dom.uploadArea.style.display = 'block';
    state.dom.uploadResults.style.display = 'none';
    state.dom.fileInput.value = '';
    state.dom.uploadDetectedFaces.innerHTML =
        '<div class="empty-state"><p>Upload foto untuk mengenali wajah</p></div>';
}

// ============================================================================
// PARTICLE BACKGROUND
// ============================================================================
function setupParticles() {
    const canvas = document.getElementById('particles-canvas');
    const ctx = canvas.getContext('2d');

    let particles = [];
    const particleCount = 50;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function createParticle() {
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            size: Math.random() * 2 + 0.5,
            opacity: Math.random() * 0.5 + 0.1
        };
    }

    function init() {
        resize();
        particles = Array.from({ length: particleCount }, createParticle);
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;

            // Wrap around
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(99, 102, 241, ${p.opacity})`;
            ctx.fill();
        });

        // Draw connections
        particles.forEach((p1, i) => {
            particles.slice(i + 1).forEach(p2 => {
                const dx = p1.x - p2.x;
                const dy = p1.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(99, 102, 241, ${0.1 * (1 - dist / 150)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            });
        });

        requestAnimationFrame(animate);
    }

    window.addEventListener('resize', resize);
    init();
    animate();
}

// ============================================================================
// UTILITIES
// ============================================================================
function updateLoading(text, progress) {
    state.dom.loadingText.textContent = text;
    state.dom.progressBar.style.width = `${progress}%`;
}

function hideLoading() {
    state.dom.loadingOverlay.classList.add('hidden');
    setTimeout(() => {
        state.dom.loadingOverlay.style.display = 'none';
    }, 500);
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    state.dom.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// EXPORT: Fungsi preprocessing yang IDENTIK dengan Python
// ============================================================================
/**
 * Dokumentasi preprocessing (untuk referensi):
 *
 * face-api.js INTERNAL preprocessing:
 * 1. Deteksi wajah → bounding box
 * 2. Crop & resize ke 150×150 (face recognition model internal)
 * 3. Normalize: pixel / 255 (range [0, 1])
 * 4. Feed ke FaceRecognitionNet → descriptor 128-D (Float32Array)
 * 5. Descriptor sudah L2-normalized secara internal
 *
 * CATATAN PENTING:
 * - face-api.js menggunakan model FaceNet versi RINGAN (128-D)
 * - Ini BERBEDA dengan FaceNet PyTorch (512-D) yang dipakai di Colab
 * - Karena itu, kita HARUS compute descriptors di browser
 *   menggunakan foto referensi (bukan import embedding dari Python)
 * - Threshold Euclidean distance di face-api.js biasanya 0.5-0.6
 *   (berbeda dengan cosine similarity threshold di Python)
 */
