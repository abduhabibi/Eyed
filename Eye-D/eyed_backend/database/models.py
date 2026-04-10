// AUTO-REGISTRATION FLOW
// User enters name + clicks ONE button → webcam auto-starts → monitoring auto-starts
// No more buttons needed, just squeeze 3 times

let regState = {
    step: 'idle', // idle → cameraReady → monitoring → uploading → done
    squeezeCount: 0,
    recordings: [],
    cooldown: false,
    earHistory: [],
    calibrated: false
};

function updateRegisterUI() {
    const startBtn = document.getElementById('startRegisterMonitor');
    const manualBtn = document.getElementById('manualRegisterSqueeze');
    const statusDiv = document.getElementById('registerStatus');
    const overlay = document.getElementById('registerSqueezeOverlay');
    
    if (regState.step === 'idle') {
        startBtn.style.display = 'block';
        startBtn.innerHTML = '<i class="fas fa-play"></i> START REGISTRATION';
        startBtn.disabled = false;
        manualBtn.style.display = 'none';
        statusDiv.innerHTML = '⏳ Enter your name & click START to begin';
        if (overlay) overlay.innerText = '👁️ Ready';
    } else if (regState.step === 'monitoring') {
        startBtn.style.display = 'none';
        manualBtn.style.display = 'block';
        manualBtn.innerHTML = `<i class="fas fa-hand-fist"></i> MANUAL SQUEEZE (${regState.squeezeCount}/3)`;
        statusDiv.innerHTML = `🟢 MONITORING | Squeezes: ${regState.squeezeCount}/3. JUST SQUEEZE - NO BUTTON NEEDED!`;
        if (overlay) overlay.innerText = `${regState.squeezeCount}/3 squeezes`;
    } else if (regState.step === 'uploading') {
        startBtn.style.display = 'none';
        manualBtn.style.display = 'none';
        statusDiv.innerHTML = '⏳ Uploading to server...';
    } else if (regState.step === 'done') {
        startBtn.style.display = 'block';
        startBtn.innerHTML = '<i class="fas fa-plus"></i> REGISTER ANOTHER USER';
        startBtn.disabled = false;
        manualBtn.style.display = 'none';
        statusDiv.innerHTML = '✅ Registration complete!';
        if (overlay) overlay.innerText = '👁️ Done';
    }
}

async function startRegisterFlow() {
    const nameField = document.querySelector('.field-group input[name="name"]');
    const name = nameField ? nameField.value.trim() : '';
    
    if (regState.step === 'idle') {
        // User clicks button → activate camera → auto-start monitoring
        if (!name) {
            alert('⚠️ Please enter your name first!');
            nameField.focus();
            return;
        }
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            currentStream = stream;
            const video = document.getElementById('registerVideo');
            video.srcObject = stream;
            await video.play();
            
            // Immediately start monitoring (no intermediate step)
            regState.step = 'monitoring';
            regState.squeezeCount = 0;
            regState.recordings = [];
            regState.calibrated = false;
            regState.earHistory = [];
            document.getElementById('registerStatus').innerHTML = '✅ Camera active! Now initiating squeeze detection...';
            
            setTimeout(() => {
                initFaceMeshForRegister();
                updateRegisterUI();
            }, 500);
            
        } catch (err) {
            alert('❌ Camera access denied: ' + err.message);
        }
    } else if (regState.step === 'done') {
        // Reset for next user
        regState.step = 'idle';
        regState.squeezeCount = 0;
        regState.recordings = [];
        nameField.value = '';
        if (currentStream) {
            currentStream.getTracks().forEach(t => t.stop());
            currentStream = null;
        }
        updateRegisterUI();
    }
}

function initFaceMeshForRegister() {
    if (!faceMesh) {
        faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });
        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
    }
    
    const video = document.getElementById('registerVideo');
    
    faceMesh.onResults((results) => {
        if (regState.step !== 'monitoring') return;
        
        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
            document.getElementById('registerStatus').innerHTML = '⚠️ Face not detected. Look at camera!';
            return;
        }
        
        const landmarks = results.multiFaceLandmarks[0];
        const ear = calculateEAR(landmarks, [33, 133, 157, 158, 159, 160, 161, 173]);
        
        // Calibration: establish what "open eye" looks like
        if (!regState.calibrated) {
            regState.earHistory.push(ear);
            if (regState.earHistory.length > 30) {
                const avgEar = regState.earHistory.reduce((a,b) => a+b) / regState.earHistory.length;
                regState.calibrationThreshold = avgEar * 0.65; // 65% of open = closed/squeeze
                regState.calibrated = true;
                document.getElementById('registerStatus').innerHTML = `✅ Calibrated! Squeeze threshold: ${regState.calibrationThreshold.toFixed(3)}. Now squeeze 3 times!`;
            }
            return;
        }
        
        const isSqueezed = ear < regState.calibrationThreshold;
        const overlay = document.getElementById('registerSqueezeOverlay');
        
        if (isSqueezed && !regState.cooldown && regState.squeezeCount < 3) {
            // AUTO SQUEEZE DETECTION - NO BUTTON NEEDED
            regState.cooldown = true;
            recordSqueeze();
            overlay.innerText = `👁️ SQUEEZE DETECTED! (${regState.squeezeCount + 1}/3)`;
            overlay.classList.add('active');
            
            setTimeout(() => {
                overlay.classList.remove('active');
                regState.cooldown = false;
            }, 1500);
        } else if (ear > regState.calibrationThreshold && !isSqueezed) {
            overlay.innerText = `🟢 Ready (${regState.squeezeCount}/3)`;
            overlay.classList.remove('active');
        }
    });
    
    // Process frames
    async function processFrame() {
        if (video.videoWidth && regState.step === 'monitoring') {
            await faceMesh.send({ image: video });
        }
        if (regState.step === 'monitoring') {
            requestAnimationFrame(processFrame);
        }
    }
    processFrame();
}

function recordSqueeze() {
    if (regState.squeezeCount >= 3 || !currentStream) return;
    
    regState.squeezeCount++;
    const overlay = document.getElementById('registerSqueezeOverlay');
    overlay.innerText = `Recording ${regState.squeezeCount}/3...`;
    
    // 2.5 second recording per squeeze
    const chunks = [];
    const miniRecorder = new MediaRecorder(currentStream, { mimeType: 'video/webm' });
    
    miniRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
    };
    
    miniRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        regState.recordings.push(blob);
        
        if (regState.squeezeCount === 3) {
            // All 3 squeezes recorded → upload immediately
            setTimeout(() => completeRegisterAndUpload(), 500);
        } else {
            // Feedback for remaining squeezes
            document.getElementById('registerStatus').innerHTML = `✅ Squeeze ${regState.squeezeCount}/3 recorded. ${3 - regState.squeezeCount} more!`;
            setTimeout(() => {
                overlay.innerText = `🟢 Ready (${regState.squeezeCount}/3)`;
            }, 800);
        }
    };
    
    miniRecorder.start();
    setTimeout(() => miniRecorder.stop(), 2500);
}

async function completeRegisterAndUpload() {
    regState.step = 'uploading';
    updateRegisterUI();
    
    const nameField = document.querySelector('.field-group input[name="name"]');
    const name = nameField ? nameField.value.trim() : 'User';
    
    const formData = new FormData();
    formData.append('name', name);
    
    for (let i = 0; i < regState.recordings.length; i++) {
        formData.append('videos', regState.recordings[i], `squeeze_${i}.webm`);
    }
    
    // Add custom metadata
    document.querySelectorAll('.field-group').forEach(field => {
        const input = field.querySelector('input');
        if (input && input.name !== 'name' && input.value.trim()) {
            formData.append(input.name, input.value.trim());
        }
    });
    
    try {
        const res = await fetch('http://localhost:8000/register/', { method: 'POST', body: formData });
        const data = await res.json();
        
        if (res.ok) {
            regState.step = 'done';
            document.getElementById('registerStatus').innerHTML = `✅ SUCCESS! ${name} registered (ID: ${data.user_id})`;
            updateRegisterUI();
        } else {
            throw new Error(data.detail || 'Unknown error');
        }
    } catch (err) {
        alert('❌ Error: ' + err.message);
        regState.step = 'monitoring';
        regState.squeezeCount = 0;
        regState.recordings = [];
        updateRegisterUI();
    }
}

// Handlers
document.getElementById('startRegisterMonitor').onclick = startRegisterFlow;
document.getElementById('manualRegisterSqueeze').onclick = () => {
    if (regState.step === 'monitoring' && regState.squeezeCount < 3 && !regState.cooldown) {
        recordSqueeze();
    }
};

// Initialize
updateRegisterUI();
