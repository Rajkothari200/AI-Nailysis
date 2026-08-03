// Remote console log helper for USB debugging without devtools
function sendReportToBackend(level, message) {
    fetch('/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level: level, message: message })
    }).catch(() => {});
}

// Override console helpers
const originalConsoleError = console.error;
console.error = function(...args) {
    originalConsoleError.apply(console, args);
    sendReportToBackend('error', args.join(' '));
};

const originalConsoleLog = console.log;
console.log = function(...args) {
    originalConsoleLog.apply(console, args);
    sendReportToBackend('log', args.join(' '));
};

window.onerror = function(message, source, lineno, colno, error) {
    const errMsg = `${message} at ${source}:${lineno}:${colno}` + (error && error.stack ? `\nStack: ${error.stack}` : '');
    sendReportToBackend('window-error', errMsg);
};

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
    const reason = event.reason;
    const errMsg = reason ? (reason.message || reason) + (reason.stack ? `\nStack: ${reason.stack}` : '') : 'Unknown promise rejection';
    sendReportToBackend('unhandled-rejection', errMsg);
});

// Helper to set diagnostics steps status with appropriate icons
function setStepState(el, state, text) {
    if (!el) return;
    el.className = state;
    let iconClass = 'fa-solid fa-clock';
    if (state === 'running') {
        iconClass = 'fa-solid fa-spinner';
    } else if (state === 'success') {
        iconClass = 'fa-solid fa-circle-check';
    } else if (state === 'warning') {
        iconClass = 'fa-solid fa-triangle-exclamation';
    } else if (state === 'danger') {
        iconClass = 'fa-solid fa-circle-xmark';
    } else if (state === 'pending') {
        iconClass = 'fa-solid fa-clock';
    }
    el.innerHTML = `<i class="${iconClass}"></i> ${text}`;
}

// App State
let activeTab = 'webcam';
let webcamStream = null;
let selectedFile = null;
let batchResults = null; // Stores results for 5 nails
let selectedFingerIdx = 0; // Currently selected finger index in UI

// MediaPipe variables
let hands = null;
let cameraHelper = null;
let detectedNailBoxes = []; // Current frame's nail coordinates
let autoCaptureTimer = null;
let autoCaptureCounter = 0;
let isCountingDown = false;
let countdownVal = 3;
let lastFiveNailsTime = 0; // Timestamp of last successful 5-nail tracking frame

// Elements
const videoEl = document.getElementById('webcam-feed');
const handsCanvas = document.getElementById('hands-canvas');
const captureCanvas = document.getElementById('capture-canvas');
const fileInputEl = document.getElementById('file-input');
const dropzoneEl = document.getElementById('dropzone');
const previewContainerEl = document.getElementById('preview-container');
const imagePreviewEl = document.getElementById('image-preview');
const dropzonePromptEl = document.getElementById('dropzone-prompt');

// HUD & Overlay elements
const scannerHud = document.getElementById('scanner-hud');
const scanLaser = document.getElementById('scan-laser-line');
const scanInstructions = document.getElementById('scan-instruction-text');
const countdownOverlay = document.getElementById('countdown-overlay');
const countdownNumber = document.getElementById('countdown-number');

// Buttons
const btnStartCamera = document.getElementById('btn-start-camera');
const btnCapture = document.getElementById('btn-capture');
const btnAnalyzeUpload = document.getElementById('btn-analyze-upload');
const btnClearPreview = document.getElementById('btn-clear-preview');
const btnExportPdf = document.getElementById('btn-export-pdf');

// Panel elements
const diagnosticBox = document.getElementById('analysis-status-box');
const resultsEmpty = document.getElementById('results-empty');
const resultsContent = document.getElementById('results-content');
const fingerTabsWrapper = document.getElementById('finger-tabs-wrapper');
const fingerTabs = document.getElementById('finger-tabs');

// Diagnostic Steps
const stepPolish = document.getElementById('step-polish');
const stepBinary = document.getElementById('step-binary');
const stepMulticlass = document.getElementById('step-multiclass');

// Tab Switching
function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    document.getElementById(`tab-${tab}`).classList.add('active');
    document.getElementById(`content-${tab}`).classList.add('active');
    
    if (tab === 'upload') {
        stopWebcam();
    }
}

// MediaPipe Hands Initialization
function initializeHands() {
    if (typeof Hands === "undefined") {
        console.warn("MediaPipe Hands CDN not loaded. Retrying in 1s...");
        setTimeout(initializeHands, 1000);
        return;
    }
    
    hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    
    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6
    });
    
    hands.onResults(onHandResults);
}

// Hand positioning guidelines checks
function checkHandPositioning(landmarks) {
    const wrist = landmarks[0];
    const middleMCP = landmarks[9];
    
    // 1. Check Hand Orientation (Roll/Tilt)
    const dx = middleMCP.x - wrist.x;
    const dy = middleMCP.y - wrist.y;
    const angle = Math.atan2(dy, dx) * 180 / Math.PI; // angle in degrees
    // Straight vertical is -90 degrees. Allow [-120, -60]
    if (angle < -120 || angle > -60) {
        return { isOptimal: false, suggestion: "⚠️ Hold hand straight up and vertical" };
    }
    
    // 2. Check Pitch/Foreshortening (Fingers bent or tilted forward/backward)
    const palmLength = Math.sqrt(dx*dx + dy*dy);
    const middleTip = landmarks[12];
    const fingerLength = Math.sqrt((middleTip.x - middleMCP.x)**2 + (middleTip.y - middleMCP.y)**2);
    const lengthRatio = fingerLength / palmLength;
    if (lengthRatio < 0.55) {
        return { isOptimal: false, suggestion: "⚠️ Flatten your fingers and point straight up" };
    }
    
    // 3. Check Finger Spread (Index to Pinky tip distance relative to palm length)
    const indexTip = landmarks[8];
    const pinkyTip = landmarks[20];
    const indexPinkyDist = Math.sqrt((indexTip.x - pinkyTip.x)**2 + (indexTip.y - pinkyTip.y)**2);
    const spreadRatio = indexPinkyDist / palmLength;
    if (spreadRatio < 0.65) {
        return { isOptimal: false, suggestion: "⚠️ Spread your fingers wider" };
    }
    
    // 4. Check Hand Distance (Wrist to middle tip distance in screen coordinates)
    const middleTipY = middleTip.y;
    const wristY = wrist.y;
    const handHeightOnScreen = Math.abs(wristY - middleTipY);
    if (handHeightOnScreen < 0.35) {
        return { isOptimal: false, suggestion: "⚠️ Bring your hand closer to the camera" };
    }
    if (handHeightOnScreen > 0.78) {
        return { isOptimal: false, suggestion: "⚠️ Move your hand further back" };
    }
    
    return { isOptimal: true, suggestion: "Optimal angle! Hold hand steady..." };
}

// Draw overlays & compute nail boxes
function onHandResults(results) {
    if (activeTab !== 'webcam' || !webcamStream) return;
    
    const ctx = handsCanvas.getContext('2d');
    const width = handsCanvas.width;
    const height = handsCanvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Show/hide hand silhouette guide overlay based on hand presence
    const handSilhouette = document.getElementById('hand-silhouette');
    
    // Finger landmark mapping (Tip index & Joint below tip index)
    const fingerConfig = [
        { name: "Thumb", tip: 4, joint: 3 },
        { name: "Index", tip: 8, joint: 7 },
        { name: "Middle", tip: 12, joint: 11 },
        { name: "Ring", tip: 16, joint: 15 },
        { name: "Pinky", tip: 20, joint: 19 }
    ];
    
    detectedNailBoxes = [];
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        // Fade out the silhouette outline since a hand is detected
        if (handSilhouette) handSilhouette.classList.add('detected');
        
        const landmarks = results.multiHandLandmarks[0];
        
        fingerConfig.forEach((cfg) => {
            const tip = landmarks[cfg.tip];
            const joint = landmarks[cfg.joint];
            
            // Calculate distance for bounding box scale
            const tx = tip.x * width;
            const ty = tip.y * height;
            const jx = joint.x * width;
            const jy = joint.y * height;
            
            const dist = Math.sqrt((tx - jx)**2 + (ty - jy)**2);
            // Nail box scale factor
            const boxSize = dist * 1.4;
            
            const bx = tx - boxSize / 2;
            const by = ty - boxSize / 2;
            
            // Store coordinates normalized for crop slicing from video stream
            const normBox = {
                name: cfg.name,
                x: tip.x - (tip.x - joint.x) * 0.7 - (boxSize / width) / 2, // slightly center nail
                y: tip.y - (tip.y - joint.y) * 0.5 - (boxSize / height) / 2,
                w: boxSize / width,
                h: boxSize / height
            };
            
            // Limit bounds
            normBox.x = Math.max(0, Math.min(1 - normBox.w, normBox.x));
            normBox.y = Math.max(0, Math.min(1 - normBox.h, normBox.y));
            
            detectedNailBoxes.push(normBox);
            
            // Draw neon bounding box on canvas
            const isAllFive = detectedNailBoxes.length === 5;
            ctx.strokeStyle = isAllFive ? '#00e676' : '#00f2fe';
            ctx.lineWidth = 3;
            ctx.shadowBlur = 8;
            ctx.shadowColor = ctx.strokeStyle;
            
            ctx.strokeRect(bx, by, boxSize, boxSize);
            
            // Draw text label
            ctx.fillStyle = '#fff';
            ctx.shadowBlur = 0;
            ctx.font = '600 11px Montserrat';
            ctx.fillText(cfg.name, bx, by - 6);
        });
        
        // Handle Auto-capture flow with 1.0s grace period for tracking dropouts
        if (detectedNailBoxes.length === 5) {
            const positioning = checkHandPositioning(landmarks);
            
            if (positioning.isOptimal) {
                lastFiveNailsTime = Date.now();
                scanInstructions.textContent = "Optimal angle! Hold hand steady...";
                scanInstructions.style.color = "#00e676";
                scanInstructions.style.borderColor = "rgba(0,230,118,0.3)";
                
                if (!isCountingDown) {
                    startAutoCaptureCountdown();
                }
            } else {
                scanInstructions.textContent = positioning.suggestion;
                scanInstructions.style.color = "#ffb199";
                scanInstructions.style.borderColor = "rgba(255,177,153,0.3)";
                
                if (isCountingDown) {
                    // Grace period check: allow 1.0s of bad positioning or missing tracking
                    const elapsed = Date.now() - lastFiveNailsTime;
                    if (elapsed > 1000) {
                        cancelAutoCaptureCountdown();
                    }
                }
            }
        } else {
            if (isCountingDown) {
                const elapsed = Date.now() - lastFiveNailsTime;
                if (elapsed > 1000) {
                    cancelAutoCaptureCountdown();
                } else {
                    scanInstructions.textContent = "Hold steady... Re-acquiring fingers";
                    scanInstructions.style.color = "#ffb199";
                }
            } else {
                scanInstructions.textContent = "Show 1 hand with 5 visible fingernails";
                scanInstructions.style.color = "#00f2fe";
                scanInstructions.style.borderColor = "rgba(0,242,254,0.2)";
            }
        }
    } else {
        // Show silhouette guide if no hand is detected
        if (handSilhouette) handSilhouette.classList.remove('detected');
        
        if (isCountingDown) {
            const elapsed = Date.now() - lastFiveNailsTime;
            if (elapsed > 1000) {
                cancelAutoCaptureCountdown();
                scanInstructions.textContent = "Show 1 hand with 5 visible fingernails";
                scanInstructions.style.color = "#00f2fe";
                scanInstructions.style.borderColor = "rgba(0,242,254,0.2)";
            } else {
                scanInstructions.textContent = "Hold steady... Re-acquiring hand";
                scanInstructions.style.color = "#ffb199";
            }
        } else {
            scanInstructions.textContent = "Show 1 hand with 5 visible fingernails";
            scanInstructions.style.color = "#00f2fe";
            scanInstructions.style.borderColor = "rgba(0,242,254,0.2)";
        }
    }
}

// Auto-capture timer
function startAutoCaptureCountdown() {
    isCountingDown = true;
    countdownVal = 3;
    countdownOverlay.style.display = 'flex';
    countdownNumber.textContent = countdownVal;
    
    autoCaptureTimer = setInterval(() => {
        countdownVal--;
        if (countdownVal > 0) {
            countdownNumber.textContent = countdownVal;
        } else {
            clearInterval(autoCaptureTimer);
            countdownOverlay.style.display = 'none';
            isCountingDown = false;
            // Trigger capture!
            captureAndAnalyzeWebcamBatch();
        }
    }, 800);
}

function cancelAutoCaptureCountdown() {
    if (autoCaptureTimer) {
        clearInterval(autoCaptureTimer);
    }
    isCountingDown = false;
    countdownOverlay.style.display = 'none';
}

// Robust constraints lookup for mobile cameras fallback
async function getWebcamStream() {
    const constraintsList = [
        { video: { facingMode: { exact: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } } },
        { video: { facingMode: { exact: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } },
        { video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } },
        { video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } },
        { video: true }
    ];
    
    for (const constraints of constraintsList) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            if (stream) return stream;
        } catch (e) {
            console.warn("Failed constraints:", constraints, e);
        }
    }
    throw new Error("All camera constraints failed.");
}

// Webcam start
btnStartCamera.addEventListener('click', async () => {
    try {
        webcamStream = await getWebcamStream();
        
        videoEl.srcObject = webcamStream;
        btnStartCamera.style.display = 'none';
        btnCapture.disabled = false;
        
        // Wait for video metadata to establish canvas scales and aspect ratios
        videoEl.onloadedmetadata = () => {
            const vW = videoEl.videoWidth;
            const vH = videoEl.videoHeight;
            console.log(`Camera stream loaded: ${vW}x${vH}`);
            
            // Dynamically set aspect ratio of the wrapper to match the camera stream
            // This prevents object-fit cropping and keeps the canvas 1:1 aligned
            const wrapper = videoEl.parentElement;
            if (wrapper) {
                wrapper.style.aspectRatio = `${vW} / ${vH}`;
            }
            
            // Match canvas width/height to displayed element dimensions
            handsCanvas.width = videoEl.clientWidth;
            handsCanvas.height = videoEl.clientHeight;
            
            // Check if stream is mirrored (front vs rear camera)
            let isMirrored = true;
            try {
                const track = webcamStream.getVideoTracks()[0];
                const settings = track.getSettings();
                if (settings.facingMode === 'environment') {
                    isMirrored = false;
                }
            } catch (e) {
                console.log("Could not detect facingMode from track settings, defaulting to mirrored:", e);
            }
            
            if (isMirrored) {
                videoEl.classList.add('mirror');
                handsCanvas.classList.add('mirror');
            } else {
                videoEl.classList.remove('mirror');
                handsCanvas.classList.remove('mirror');
            }
            
            // Custom high-performance loop for MediaPipe Hands processing
            let active = true;
            async function processVideoFrame() {
                if (!active || !webcamStream) return;
                
                // Keep canvas resized to container in case layout changes
                if (handsCanvas.width !== videoEl.clientWidth || handsCanvas.height !== videoEl.clientHeight) {
                    handsCanvas.width = videoEl.clientWidth;
                    handsCanvas.height = videoEl.clientHeight;
                }
                
                if (videoEl.readyState >= 2 && !videoEl.paused && !videoEl.ended) {
                    if (hands) {
                        try {
                            await hands.send({ image: videoEl });
                        } catch (err) {
                            console.error("MediaPipe Hands prediction error:", err);
                        }
                    }
                }
                requestAnimationFrame(processVideoFrame);
            }
            
            videoEl.play()
                .then(() => {
                    requestAnimationFrame(processVideoFrame);
                })
                .catch((e) => {
                    console.error("Video play failed:", e);
                });
            
            cameraHelper = {
                stop: () => {
                    active = false;
                }
            };
        };
        
    } catch (err) {
        console.error("Camera access failed:", err);
        alert("Could not access camera. Please verify you are on HTTPS (secure connection), and have allowed camera permissions for this site in your phone's browser settings.");
    }
});

function stopWebcam() {
    cancelAutoCaptureCountdown();
    if (cameraHelper) {
        cameraHelper.stop();
        cameraHelper = null;
    }
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
        videoEl.srcObject = null;
        btnStartCamera.style.display = 'flex';
        btnCapture.disabled = true;
    }
    const ctx = handsCanvas.getContext('2d');
    ctx.clearRect(0, 0, handsCanvas.width, handsCanvas.height);
}

// Drag & Drop
dropzoneEl.addEventListener('click', () => {
    if (!selectedFile) fileInputEl.click();
});

fileInputEl.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

dropzoneEl.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzoneEl.classList.add('dragover');
});

dropzoneEl.addEventListener('dragleave', () => {
    dropzoneEl.classList.remove('dragover');
});

dropzoneEl.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzoneEl.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload a valid image file.");
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreviewEl.src = e.target.result;
        dropzonePromptEl.style.display = 'none';
        previewContainerEl.style.display = 'block';
        btnAnalyzeUpload.disabled = false;
    };
    reader.readAsDataURL(file);
}

btnClearPreview.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInputEl.value = '';
    imagePreviewEl.src = '';
    previewContainerEl.style.display = 'none';
    dropzonePromptEl.style.display = 'flex';
    btnAnalyzeUpload.disabled = true;
});

// Capture action button manual override
btnCapture.addEventListener('click', () => {
    captureAndAnalyzeWebcamBatch();
});

btnAnalyzeUpload.addEventListener('click', () => {
    if (selectedFile) {
        runSingleImageAnalysis(selectedFile);
    }
});

// Crop and analyze multi-nails batch
async function captureAndAnalyzeWebcamBatch() {
    console.log("captureAndAnalyzeWebcamBatch started");
    if (!webcamStream) {
        console.log("No webcamStream found, aborting capture");
        return;
    }
    
    // Draw full image to capture-canvas
    const ctx = captureCanvas.getContext('2d');
    const vW = videoEl.videoWidth;
    const vH = videoEl.videoHeight;
    console.log(`Video dimensions: ${vW}x${vH}`);
    captureCanvas.width = vW;
    captureCanvas.height = vH;
    
    // Draw video frame to canvas, mirroring only if the display is mirrored
    const isMirrored = videoEl.classList.contains('mirror');
    console.log(`isMirrored: ${isMirrored}`);
    if (isMirrored) {
        ctx.translate(vW, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoEl, 0, 0, vW, vH);
        ctx.setTransform(1, 0, 0, 1, 0, 0); // reset scale
    } else {
        ctx.drawImage(videoEl, 0, 0, vW, vH);
    }
    console.log("Frame drawn to captureCanvas");
    
    // Copy the detected boxes before we clear the state
    const boxesCopy = [...detectedNailBoxes];
    console.log(`Nail boxes copy length: ${boxesCopy.length}`);
    
    // Stop the webcam stream immediately to halt the tracking loop
    console.log("Stopping webcam...");
    stopWebcam();
    console.log("Webcam stopped successfully");
    
    // Check if we have tracked nail boxes
    if (boxesCopy.length > 0) {
        console.log("Starting cropping promises");
        const cropPromises = boxesCopy.map((box) => {
            return new Promise((resolve) => {
                try {
                    console.log(`Cropping box for finger: ${box.name}`);
                    let rx = isMirrored ? (1 - box.x - box.w) * vW : box.x * vW;
                    let ry = box.y * vH;
                    let rw = box.w * vW;
                    let rh = box.h * vH;
                    
                    // Clamp to source canvas dimensions to prevent IndexSizeError in drawImage
                    rx = Math.max(0, Math.min(vW - 1, rx));
                    ry = Math.max(0, Math.min(vH - 1, ry));
                    rw = Math.max(1, Math.min(vW - rx, rw));
                    rh = Math.max(1, Math.min(vH - ry, rh));
                    console.log(`Clamped coords for ${box.name}: rx=${rx}, ry=${ry}, rw=${rw}, rh=${rh}`);
                    
                    const cropCanvas = document.createElement('canvas');
                    cropCanvas.width = 224;
                    cropCanvas.height = 224;
                    const cropCtx = cropCanvas.getContext('2d');
                    
                    // Crop from capture canvas
                    cropCtx.drawImage(captureCanvas, rx, ry, rw, rh, 0, 0, 224, 224);
                    console.log(`Nail cropped onto cropCanvas for ${box.name}`);
                    
                    cropCanvas.toBlob((blob) => {
                        console.log(`toBlob callback fired for ${box.name}, blob size: ${blob ? blob.size : 'null'}`);
                        resolve({ name: box.name, blob });
                    }, 'image/jpeg', 0.9);
                } catch (e) {
                    console.error(`Failed to crop nail for ${box.name}:`, e);
                    // Resolve with a fallback: just draw the center of the full captured frame
                    try {
                        const cropCanvas = document.createElement('canvas');
                        cropCanvas.width = 224;
                        cropCanvas.height = 224;
                        const cropCtx = cropCanvas.getContext('2d');
                        cropCtx.drawImage(captureCanvas, vW / 4, vH / 4, vW / 2, vH / 2, 0, 0, 224, 224);
                        cropCanvas.toBlob((blob) => {
                            resolve({ name: box.name, blob });
                        }, 'image/jpeg', 0.9);
                    } catch (err) {
                        resolve({ name: box.name, blob: null });
                    }
                }
            });
        });
        
        console.log("Awaiting Promise.all(cropPromises)");
        const crops = await Promise.all(cropPromises);
        console.log("All crops resolved, invoking runBatchAnalysis");
        runBatchAnalysis(crops);
        
    } else {
        console.log("No tracked nail boxes found, running fallback full frame analysis");
        captureCanvas.toBlob((blob) => {
            console.log(`Fallback toBlob callback fired, blob size: ${blob ? blob.size : 'null'}`);
            const file = new File([blob], "capture_full.jpg", { type: "image/jpeg" });
            runSingleImageAnalysis(file);
        }, 'image/jpeg');
    }
}

// Submit batch upload
async function runBatchAnalysis(crops) {
    setStepState(stepPolish, 'running', 'Batch checking: Nail Paint & Art...');
    setStepState(stepBinary, 'pending', 'Batch inspecting: Anomalies');
    setStepState(stepMulticlass, 'pending', 'Batch classifying: Pathologies');
    
    diagnosticBox.style.display = 'block';
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'none';
    fingerTabsWrapper.style.display = 'none';
    
    diagnosticBox.scrollIntoView({ behavior: 'smooth' });
    
    const formData = new FormData();
    crops.forEach((crop) => {
        // Filename contains the finger label (e.g. Index.jpg)
        formData.append("files", crop.blob, `${crop.name}.jpg`);
    });
    
    try {
        await sleep(1000);
        
        const response = await fetch("/analyze_batch", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) throw new Error("Server error");
        
        batchResults = await response.json();
        
        // Associate crop image URLs with the results
        batchResults.forEach((result) => {
            const crop = crops.find(c => c.name === result.finger);
            if (crop) {
                result.imageUrl = URL.createObjectURL(crop.blob);
            }
        });
        
        // Render steps logs based on overall batch state
        const anyPolish = batchResults.some(r => r.polish_detected);
        const anyDiseased = batchResults.some(r => !r.healthy);
        
        if (anyPolish) {
            setStepState(stepPolish, 'warning', 'Nail Paint / Art detected on some fingers');
        } else {
            setStepState(stepPolish, 'success', 'All fingernails are bare');
        }
        
        await sleep(800);
        setStepState(stepBinary, 'running', 'Batch anomaly inspections...');
        await sleep(800);
        
        if (anyDiseased) {
            setStepState(stepBinary, 'danger', 'Anomalous signatures found on hand');
            setStepState(stepMulticlass, 'running', 'Evaluating pathology classifications...');
            await sleep(1000);
            setStepState(stepMulticlass, 'danger', 'Batch classifications complete');
        } else {
            setStepState(stepBinary, 'success', 'All scanned nails appear healthy');
            setStepState(stepMulticlass, 'success', 'All matrices normal');
        }
        
        await sleep(500);
        diagnosticBox.style.display = 'none';
        
        // Show Results Panel
        resultsContent.style.display = 'block';
        fingerTabsWrapper.style.display = 'block';
        
        // Setup finger indicator dots
        setupFingerResultTabs();
        
        // Select first index to display
        selectFingerResult(0);
        
    } catch (err) {
        console.error("Batch analysis failed:", err);
        setStepState(stepPolish, 'danger', 'Batch Aborted');
        alert("Failed to analyze hand nails. Please try again.");
        resultsEmpty.style.display = 'flex';
        diagnosticBox.style.display = 'none';
    }
}

// Upload Single image flow
async function runSingleImageAnalysis(file) {
    batchResults = null;
    
    setStepState(stepPolish, 'running', 'Detecting Nail Paint & Art...');
    setStepState(stepBinary, 'pending', 'Inspecting Pathological Anomalies');
    setStepState(stepMulticlass, 'pending', 'Classifying Disease Signatures');
    
    diagnosticBox.style.display = 'block';
    resultsEmpty.style.display = 'none';
    resultsContent.style.display = 'none';
    fingerTabsWrapper.style.display = 'none';
    
    diagnosticBox.scrollIntoView({ behavior: 'smooth' });
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        await sleep(1000);
        
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) throw new Error("Server error");
        const data = await response.json();
        
        if (data.polish_detected) {
            setStepState(stepPolish, 'warning', `Nail Paint/Art Detected (${data.polish_confidence}%)`);
        } else {
            setStepState(stepPolish, 'success', 'Bare Nails Confirmed');
        }
        
        await sleep(800);
        setStepState(stepBinary, 'running', 'Inspecting pathology...');
        await sleep(800);
        
        if (data.healthy) {
            setStepState(stepBinary, 'success', `Normal Nail Matrix Confirmed (${data.disease_probability}% healthy)`);
            setStepState(stepMulticlass, 'success', 'No disease signatures matched');
        } else {
            setStepState(stepBinary, 'danger', `Pathological Signatures Detected (${data.disease_probability}% anomalous)`);
            setStepState(stepMulticlass, 'running', 'Running multi-class disease model...');
            await sleep(1000);
            setStepState(stepMulticlass, 'danger', `Diagnosis: ${data.disease.toUpperCase()}`);
        }
        
        await sleep(500);
        diagnosticBox.style.display = 'none';
        
        resultsContent.style.display = 'block';
        
        const reader = new FileReader();
        reader.onload = (e) => {
            data.uploadedImageUrl = e.target.result;
            renderResults(data);
        };
        reader.readAsDataURL(file);
        
    } catch (err) {
        console.error("Single image analysis failed:", err);
        setStepState(stepPolish, 'danger', 'Network Error');
        alert("Failed to analyze image.");
        resultsEmpty.style.display = 'flex';
        diagnosticBox.style.display = 'none';
    }
}

// Multi-finger tab selector setup
function setupFingerResultTabs() {
    const tabButtons = document.querySelectorAll('.finger-tab-btn');
    
    tabButtons.forEach((btn, idx) => {
        const dot = document.getElementById(`f-dot-${idx}`);
        // Find if this finger exists in batchResults
        const nameMap = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
        const match = batchResults.find(r => r.finger === nameMap[idx]);
        
        if (match) {
            btn.style.display = 'flex';
            // Set dot color
            if (match.polish_detected) {
                dot.className = 'status-dot-sm bg-yellow';
            } else if (match.healthy) {
                dot.className = 'status-dot-sm bg-green';
            } else {
                dot.className = 'status-dot-sm bg-red';
            }
        } else {
            // Hide tabs for fingers not detected in capture
            btn.style.display = 'none';
        }
    });
}

function selectFingerResult(idx) {
    selectedFingerIdx = idx;
    document.querySelectorAll('.finger-tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`f-tab-${idx}`).classList.add('active');
    
    const nameMap = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
    const match = batchResults.find(r => r.finger === nameMap[idx]);
    if (match) {
        renderResults(match);
    }
}

// Render diagnostic report card
function renderResults(data) {
    // Crop/Upload Image Preview
    const imgPreviewWrapper = document.getElementById('result-image-preview-wrapper');
    const imgPreview = document.getElementById('result-image-preview');
    if (imgPreviewWrapper && imgPreview) {
        const url = data.imageUrl || data.uploadedImageUrl;
        if (url) {
            imgPreview.src = url;
            imgPreviewWrapper.style.display = 'flex';
        } else {
            imgPreviewWrapper.style.display = 'none';
        }
    }
    // Polish warning
    const polishWarningEl = document.getElementById('polish-warning');
    if (data.polish_detected) {
        polishWarningEl.style.display = 'flex';
    } else {
        polishWarningEl.style.display = 'none';
    }
    
    // Status Badge
    const statusGlow = document.getElementById('status-glow');
    const statusIcon = document.getElementById('status-icon');
    const statusTitle = document.getElementById('result-status-title');
    const statusMeta = document.getElementById('result-status-meta');
    
    const detailsAccordion = document.getElementById('diagnosis-details');
    const confidenceContainer = document.getElementById('confidence-container');
    
    if (data.healthy) {
        statusGlow.className = 'status-icon-glow status-glow-healthy';
        statusIcon.className = 'fa-solid fa-shield-halved';
        statusTitle.textContent = data.finger ? `${data.finger}: Healthy` : 'Healthy Nails';
        statusMeta.textContent = `Clear Nail Matrix (Confidence: ${data.disease_probability}%)`;
        
        detailsAccordion.style.display = 'none';
        confidenceContainer.style.display = 'none';
    } else {
        statusGlow.className = 'status-icon-glow status-glow-diseased';
        statusIcon.className = 'fa-solid fa-stethoscope';
        
        const titleName = data.disease.charAt(0).toUpperCase() + data.disease.slice(1);
        statusTitle.textContent = data.finger ? `${data.finger}: ${titleName}` : `${titleName} Detected`;
        statusMeta.textContent = `Pathology Match Confidence: ${data.disease_confidence}%`;
        
        // Populate Accordions
        document.getElementById('result-description').textContent = data.info.description;
        document.getElementById('result-prevention').textContent = data.info.prevention;
        document.getElementById('result-treatment').textContent = data.info.treatment;
        
        detailsAccordion.style.display = 'flex';
        
        // Populate Progress Bars
        const confidenceList = document.getElementById('confidence-list');
        confidenceList.innerHTML = '';
        
        if (data.all_confidences) {
            const sortedConf = Object.entries(data.all_confidences)
                .sort((a, b) => b[1] - a[1]);
                
            sortedConf.forEach(([name, score]) => {
                const row = document.createElement('div');
                row.className = 'confidence-bar-row';
                
                const isMatch = name.toLowerCase() === data.disease.toLowerCase();
                const fillClass = isMatch ? 'bar-fill-rose' : 'bar-fill-cyan';
                const opacityStyle = isMatch ? 'opacity: 1;' : 'opacity: 0.6;';
                
                row.innerHTML = `
                    <div class="bar-labels" style="${opacityStyle}">
                        <span>${name.toUpperCase()}</span>
                        <span>${score.toFixed(1)}%</span>
                    </div>
                    <div class="bar-bg" style="${opacityStyle}">
                        <div class="bar-fill ${fillClass}" style="width: ${score}%"></div>
                    </div>
                `;
                confidenceList.appendChild(row);
            });
            confidenceContainer.style.display = 'block';
        } else {
            confidenceContainer.style.display = 'none';
        }
    }
}

// Accordion Control
function toggleAccordion(trigger) {
    const item = trigger.parentElement;
    const isActive = item.classList.contains('active');
    
    document.querySelectorAll('.accordion-item').forEach(i => i.classList.remove('active'));
    
    if (!isActive) {
        item.classList.add('active');
    }
}

// PDF Export Report
btnExportPdf.addEventListener('click', () => {
    const reportContent = document.getElementById('results-section').cloneNode(true);
    const btn = reportContent.querySelector('#btn-export-pdf');
    if (btn) btn.remove();
    
    const opt = {
        margin:       1,
        filename:     'AI_Nailysis_Report.pdf',
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2, useCORS: true },
        jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    
    const pdfWrapper = document.createElement('div');
    pdfWrapper.style.padding = '40px';
    pdfWrapper.style.backgroundColor = '#07070f';
    pdfWrapper.style.color = '#f0f0f7';
    pdfWrapper.style.fontFamily = 'Montserrat, sans-serif';
    
    const pdfHeader = document.createElement('div');
    pdfHeader.innerHTML = `
        <h1 style="color:#00f2fe;font-family:Outfit,sans-serif;margin-bottom:5px;">AI NAILYSIS</h1>
        <p style="color:#8e8eaf;font-size:12px;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:15px;margin-bottom:20px;">CLINICAL-GRADE PATHOLOGICAL NAIL DIAGNOSIS REPORT</p>
    `;
    pdfWrapper.appendChild(pdfHeader);
    pdfWrapper.appendChild(reportContent);
    
    const pdfFooter = document.createElement('div');
    pdfFooter.innerHTML = `
        <p style="color:#8e8eaf;font-size:9px;margin-top:40px;border-top:1px solid rgba(255,255,255,0.1);padding-top:15px;">Disclaimer: This screening report is generated by an artificial intelligence model and is intended for informational and educational purposes only. It does not replace professional clinical diagnosis or consultation from a certified healthcare provider.</p>
    `;
    pdfWrapper.appendChild(pdfFooter);
    
    html2pdf().from(pdfWrapper).set(opt).save();
});

// Care checklist
function resetPlanner() {
    document.querySelectorAll('.todo-item input').forEach(input => {
        input.checked = false;
    });
}

// Glossary Database loading
async function loadGlossary() {
    try {
        const response = await fetch("/db");
        if (response.ok) {
            const diseases = await response.json();
            const glossaryGrid = document.getElementById('glossary-grid');
            glossaryGrid.innerHTML = '';
            
            Object.entries(diseases).forEach(([id, info]) => {
                const card = document.createElement('div');
                card.className = 'glossary-card';
                card.setAttribute('data-name', id.toLowerCase());
                card.innerHTML = `
                    <h4>${id.charAt(0).toUpperCase() + id.slice(1)}</h4>
                    <p>${info.description}</p>
                    <p style="margin-top: 8px; font-size: 11px; color:#00f2fe;"><strong>Prevention:</strong> ${info.prevention}</p>
                `;
                glossaryGrid.appendChild(card);
            });
        }
    } catch (e) {
        console.error("Glossary fetch error:", e);
    }
}

function filterGlossary() {
    const query = document.getElementById('glossary-search').value.toLowerCase();
    document.querySelectorAll('.glossary-card').forEach(card => {
        const name = card.getAttribute('data-name');
        if (name.includes(query)) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}

// Chatbot functionality
const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const chatBadge = document.getElementById('chat-badge');

let chatOpened = false;

function toggleChat() {
    chatOpened = !chatOpened;
    if (chatOpened) {
        chatWindow.classList.add('open');
        chatBadge.style.display = 'none';
    } else {
        chatWindow.classList.remove('open');
    }
}

function handleChatKey(e) {
    if (e.key === 'Enter') {
        sendChatMessage();
    }
}

async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    addOutgoingMessage(text);
    chatInput.value = '';
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text })
        });
        
        if (response.ok) {
            const data = await response.json();
            addIncomingMessage(data.reply);
        } else {
            addIncomingMessage("I am having trouble connecting to my diagnostic system. Please try again.");
        }
    } catch (err) {
        addIncomingMessage("Network error. Make sure the backend server is running.");
    }
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addOutgoingMessage(text) {
    const el = document.createElement('div');
    el.className = 'message outgoing';
    el.textContent = text;
    chatMessages.appendChild(el);
}

function addIncomingMessage(text) {
    const el = document.createElement('div');
    el.className = 'message incoming';
    el.textContent = text;
    chatMessages.appendChild(el);
    
    if (!chatOpened) {
        chatBadge.style.display = 'flex';
        chatBadge.textContent = parseInt(chatBadge.textContent || 0) + 1;
    }
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Load initialization
window.addEventListener('DOMContentLoaded', () => {
    loadGlossary();
    initializeHands();
});
