// script.js - Eye‑D Neural Interface
// Contains: neural network background + full biometric logic

// ------------------- NEURAL NETWORK BACKGROUND -------------------
const canvas = document.getElementById('neuralCanvas');
let ctx = canvas.getContext('2d');
let nodes = [];
let mouseX = window.innerWidth / 2, mouseY = window.innerHeight / 2;
let animationId = null;
const NODE_COUNT = 80;
const CONNECT_DISTANCE = 150; // px
const MOUSE_RADIUS = 150;

// script.js - Eye‑D Neural Interface (Synaptic Nucleus Edition)
// Contains: central nucleus + dendritic web + mouse‑attacking lines

// ------------------- NEURAL NETWORK BACKGROUND (Synaptic Core) -------------------
const canvas = document.getElementById('neuralCanvas');
let ctx = canvas.getContext('2d');
let nodes = [];
let centralNucleus = { x: 0, y: 0, radius: 18, pulse: 0 };
let mouseX = window.innerWidth / 2, mouseY = window.innerHeight / 2;
let animationId = null;

const NODE_COUNT = 120;          // more nodes for dense web
const CONNECT_DIST = 150;        // max distance for dendritic connections
const MOUSE_ATTRACT_DIST = 150;  // mouse "attack" radius
const CENTRAL_FORCE = 0.002;     // slight pull towards center (keeps web coherent)

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    centralNucleus.x = canvas.width / 2;
    centralNucleus.y = canvas.height / 2;
}

function initNodes() {
    nodes = [];
    for (let i = 0; i < NODE_COUNT; i++) {
        // Spread nodes in a ring/cloud around the center, but with randomness
        const angle = Math.random() * Math.PI * 2;
        const radius = 100 + Math.random() * (Math.min(canvas.width, canvas.height) * 0.4);
        nodes.push({
            x: centralNucleus.x + Math.cos(angle) * radius,
            y: centralNucleus.y + Math.sin(angle) * radius,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            radius: Math.random() * 2.5 + 1.8,
            // each node has a slight colour variation
            hue: 200 + Math.random() * 40   // blue to cyan
        });
    }
}

function updateNodes() {
    for (let node of nodes) {
        // random movement (chaotic)
        node.vx += (Math.random() - 0.5) * 0.1;
        node.vy += (Math.random() - 0.5) * 0.1;
        // limit speed
        const maxSpeed = 1.2;
        if (Math.abs(node.vx) > maxSpeed) node.vx *= 0.98;
        if (Math.abs(node.vy) > maxSpeed) node.vy *= 0.98;
        
        node.x += node.vx;
        node.y += node.vy;
        
        // soft central attraction (keeps web anchored to nucleus)
        const dx = centralNucleus.x - node.x;
        const dy = centralNucleus.y - node.y;
        const distToCenter = Math.hypot(dx, dy);
        if (distToCenter > 10) {
            node.vx += dx * CENTRAL_FORCE;
            node.vy += dy * CENTRAL_FORCE;
        }
        
        // boundaries with elastic bounce (stay within canvas)
        const margin = 20;
        if (node.x < margin) { node.x = margin; node.vx *= -0.8; }
        if (node.x > canvas.width - margin) { node.x = canvas.width - margin; node.vx *= -0.8; }
        if (node.y < margin) { node.y = margin; node.vy *= -0.8; }
        if (node.y > canvas.height - margin) { node.y = canvas.height - margin; node.vy *= -0.8; }
    }
}

function drawNeural() {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // ---- 1. Dendritic connections between nodes (chaotic web) ----
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const dx = nodes[i].x - nodes[j].x;
            const dy = nodes[i].y - nodes[j].y;
            const dist = Math.hypot(dx, dy);
            if (dist < CONNECT_DIST) {
                const opacity = (1 - dist / CONNECT_DIST) * 0.35;
                ctx.beginPath();
                ctx.moveTo(nodes[i].x, nodes[i].y);
                ctx.lineTo(nodes[j].x, nodes[j].y);
                ctx.strokeStyle = `rgba(0, 180, 255, ${opacity})`;
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }
        }
    }
    
    // ---- 2. Mouse "attack" lines (connect to nodes within 150px, neon glow) ----
    for (let node of nodes) {
        const distToMouse = Math.hypot(node.x - mouseX, node.y - mouseY);
        if (distToMouse < MOUSE_ATTRACT_DIST) {
            const intensity = 1 - distToMouse / MOUSE_ATTRACT_DIST;
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(mouseX, mouseY);
            ctx.strokeStyle = `rgba(0, 220, 255, ${intensity * 0.9})`;
            ctx.lineWidth = 2;
            ctx.shadowBlur = 8;
            ctx.shadowColor = '#00ccff';
            ctx.stroke();
        }
    }
    ctx.shadowBlur = 0;
    
    // ---- 3. Draw all nodes (crystal points) ----
    for (let node of nodes) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        // gradient fill for "crystalline" look
        const grad = ctx.createRadialGradient(node.x-2, node.y-2, 1, node.x, node.y, node.radius);
        grad.addColorStop(0, `hsl(${node.hue}, 100%, 70%)`);
        grad.addColorStop(1, `hsl(${node.hue}, 80%, 40%)`);
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.shadowBlur = 4;
        ctx.shadowColor = `hsl(${node.hue}, 100%, 60%)`;
        ctx.fill();
        ctx.shadowBlur = 0;
    }
    
    // ---- 4. Central Synaptic Nucleus (radiant energy cluster) ----
    const pulse = Math.sin(Date.now() * 0.003) * 0.2 + 0.8;
    const nucleusRadius = centralNucleus.radius * (0.9 + pulse * 0.2);
    const gradCore = ctx.createRadialGradient(
        centralNucleus.x - 5, centralNucleus.y - 5, 5,
        centralNucleus.x, centralNucleus.y, nucleusRadius
    );
    gradCore.addColorStop(0, '#ffffff');
    gradCore.addColorStop(0.4, '#3b82f6');
    gradCore.addColorStop(1, '#1e3a8a');
    ctx.beginPath();
    ctx.arc(centralNucleus.x, centralNucleus.y, nucleusRadius, 0, Math.PI*2);
    ctx.fillStyle = gradCore;
    ctx.fill();
    ctx.shadowBlur = 20;
    ctx.shadowColor = '#3b82f6';
    ctx.fill();
    ctx.shadowBlur = 0;
    // inner bright core
    ctx.beginPath();
    ctx.arc(centralNucleus.x, centralNucleus.y, nucleusRadius * 0.4, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.fill();
}

function animateNeural() {
    if (!canvas || !ctx) return;
    updateNodes();
    drawNeural();
    animationId = requestAnimationFrame(animateNeural);
}

function startNeuralBackground() {
    resizeCanvas();
    initNodes();
    animateNeural();
    window.addEventListener('resize', () => {
        resizeCanvas();
        initNodes();   // reposition nodes relative to new center
    });
    window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });
}

// ------------------- (ALL BIOMETRIC LOGIC REMAINS EXACTLY AS BEFORE) -------------------
// ... paste the unchanged Eye‑D functions here (from the previous script.js) ...
// To avoid duplication, I assume you already have the full biometric code.
// In practice, you would replace only the neural background part and keep the rest.

// For completeness, here is a placeholder comment – but you must merge with your existing script.
// The code below is a reminder: keep your existing register/verify/data functions untouched.
// Just replace the neural background functions (from "startNeuralBackground" to the end of the neural block)
// and keep everything else (API_BASE, webcam, squeeze detection, etc.) identical.

// ==================== PRESERVE YOUR EXISTING BIOMETRIC CODE HERE ====================
// (The following line is a marker – do not delete your actual functions)
// ... your existing register/verify/data logic goes here ...
// ====================================================================================
// ------------------- EYE‑D BIOMETRIC LOGIC -------------------
const API_BASE = "http://localhost:8000";   // CHANGE to your backend URL

let currentStream = null;
let faceMesh = null;
let camera = null;
let isSqueezeActive = false;
let currentMode = "register";
let registrationSqueezeCount = 0;
let registrationRecordings = [];
let verifyCallback = null;

function stopWebcam() {
    if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
        currentStream = null;
    }
    if (camera) {
        camera.stop();
        camera = null;
    }
}

async function initWebcam(videoElementId, mode) {
    stopWebcam();
    const video = document.getElementById(videoElementId);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    currentStream = stream;
    await video.play();
    
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
        faceMesh.onResults((results) => onFaceMeshResults(results, mode));
    }
    if (camera) camera.stop();
    camera = new Camera(video, {
        onFrame: async () => {
            await faceMesh.send({ image: video });
        },
        width: 640,
        height: 480
    });
    camera.start();
}

function calculateEAR(landmarks, indices) {
    const p1 = landmarks[indices[0]];
    const p2 = landmarks[indices[1]];
    const p3 = landmarks[indices[2]];
    const p4 = landmarks[indices[3]];
    const p5 = landmarks[indices[4]];
    const p6 = landmarks[indices[5]];
    const vertical1 = Math.hypot(p2.x - p6.x, p2.y - p6.y);
    const vertical2 = Math.hypot(p3.x - p5.x, p3.y - p5.y);
    const horizontal = Math.hypot(p1.x - p4.x, p1.y - p4.y);
    return (vertical1 + vertical2) / (2.0 * horizontal);
}

let lastEAR = 1.0;
let squeezeDetected = false;

function onFaceMeshResults(results, mode) {
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;
    const landmarks = results.multiFaceLandmarks[0];
    const ear = calculateEAR(landmarks, [33, 133, 157, 158, 159, 160, 161, 173]);
    const threshold = 0.20;
    const overlay = document.getElementById(mode === 'register' ? 'registerSqueezeOverlay' : 'verifySqueezeOverlay');
    
    if (ear < threshold && !squeezeDetected) {
        squeezeDetected = true;
        if (overlay) {
            overlay.innerText = '👁️ SQUEEZE DETECTED!';
            overlay.classList.add('active');
            setTimeout(() => {
                if (overlay) overlay.classList.remove('active');
            }, 300);
        }
        if (mode === 'register') {
            handleRegisterSqueeze();
        } else if (mode === 'verify') {
            handleVerifySqueeze();
        }
    } else if (ear >= threshold + 0.05) {
        squeezeDetected = false;
        if (overlay && (mode === 'register' && registrationSqueezeCount < 3)) {
            overlay.innerText = `Ready for squeeze ${registrationSqueezeCount+1}/3`;
        } else if (overlay && mode === 'verify') {
            overlay.innerText = 'Monitoring';
        }
    }
    lastEAR = ear;
}

async function handleRegisterSqueeze() {
    if (registrationSqueezeCount >= 3) return;
    const overlay = document.getElementById('registerSqueezeOverlay');
    overlay.innerText = `Recording ${registrationSqueezeCount+1}/3...`;
    const video = document.getElementById('registerVideo');
    const stream = video.srcObject;
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    let chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        registrationRecordings.push(blob);
        registrationSqueezeCount++;
        if (registrationSqueezeCount === 3) {
            uploadRegistration();
        } else {
            overlay.innerText = `Squeeze ${registrationSqueezeCount+1}/3`;
            setTimeout(() => {
                if (registrationSqueezeCount < 3) overlay.innerText = `Ready for squeeze ${registrationSqueezeCount+1}/3`;
            }, 1000);
        }
    };
    recorder.start();
    setTimeout(() => recorder.stop(), 2000);
}

async function uploadRegistration() {
    const statusDiv = document.getElementById('registerStatus');
    statusDiv.innerHTML = 'Uploading enrollment data...';
    const formData = new FormData();
    for (let i=0; i<registrationRecordings.length; i++) {
        formData.append('videos', registrationRecordings[i], `squeeze_${i}.webm`);
    }
    const nameField = document.querySelector('.field-group input[name="name"]');
    const name = nameField ? nameField.value.trim() : 'User';
    formData.append('name', name);
    const allFields = document.querySelectorAll('.field-group');
    allFields.forEach(field => {
        const input = field.querySelector('input');
        if (input && input.name !== 'name' && input.value.trim()) {
            formData.append(input.name, input.value.trim());
        }
    });
    try {
        const res = await fetch(`${API_BASE}/register/`, { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
            statusDiv.innerHTML = `✅ Registration success! User ID: ${data.user_id}`;
            alert(`User ${name} enrolled!`);
            registrationSqueezeCount = 0;
            registrationRecordings = [];
            document.getElementById('registerSqueezeOverlay').innerText = 'Ready for next user';
        } else {
            statusDiv.innerHTML = `❌ Error: ${data.detail}`;
        }
    } catch(err) {
        statusDiv.innerHTML = `❌ Network error: ${err.message}`;
    }
}

async function handleVerifySqueeze() {
    if (verifyCallback) return;
    const overlay = document.getElementById('verifySqueezeOverlay');
    overlay.innerText = 'Analyzing...';
    const video = document.getElementById('verifyVideo');
    const stream = video.srcObject;
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    let chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const formData = new FormData();
        formData.append('video', blob, 'verify.webm');
        const username = prompt("Enter your username for verification:");
        if (!username) {
            document.getElementById('verifyStatus').innerHTML = 'Verification cancelled.';
            overlay.innerText = 'Monitoring';
            verifyCallback = null;
            return;
        }
        formData.append('username', username);
        const statusDiv = document.getElementById('verifyStatus');
        statusDiv.innerHTML = 'Verifying identity...';
        try {
            const res = await fetch(`${API_BASE}/verify/`, { method: 'POST', body: formData });
            const data = await res.json();
            const resultDiv = document.getElementById('verifyResult');
            resultDiv.style.display = 'block';
            if (res.ok && data.accepted) {
                resultDiv.innerHTML = `<strong>✅ ACCESS GRANTED</strong><br>Score: ${data.score.toFixed(4)}<br>User: ${username}`;
                statusDiv.innerHTML = 'Match found.';
            } else {
                resultDiv.innerHTML = `<strong>❌ ACCESS DENIED</strong><br>Score: ${data.score?.toFixed(4) || 'N/A'}<br>No match for ${username}`;
                statusDiv.innerHTML = 'Verification failed.';
            }
        } catch(err) {
            statusDiv.innerHTML = `Error: ${err.message}`;
        }
        overlay.innerText = 'Monitoring';
        verifyCallback = null;
    };
    recorder.start();
    setTimeout(() => recorder.stop(), 2000);
    verifyCallback = true;
}

// Data page functions
async function loadData() {
    const container = document.getElementById('userList');
    container.innerHTML = '<div>Loading neural records...</div>';
    try {
        const res = await fetch(`${API_BASE}/users`);
        if (!res.ok) throw new Error('Failed to fetch users');
        const users = await res.json();
        if (users.length === 0) {
            container.innerHTML = '<div>No users registered.</div>';
            return;
        }
        container.innerHTML = '';
        users.forEach(user => {
            const card = document.createElement('div');
            card.className = 'user-card';
            let metaPreview = '';
            if (user.metadata && typeof user.metadata === 'object') {
                const firstTwo = Object.entries(user.metadata).slice(0,2);
                metaPreview = firstTwo.map(([k,v]) => `<span><strong>${k}:</strong> ${v}</span>`).join(' | ');
            }
            card.innerHTML = `
                <div class="user-name">
                    <span><i class="fas fa-user-circle"></i> ${user.name}</span>
                    <button class="delete-user" data-id="${user.id}"><i class="fas fa-trash"></i></button>
                </div>
                <div class="user-meta">ID: ${user.id} | ${metaPreview}</div>
            `;
            card.addEventListener('click', (e) => {
                if (e.target.classList.contains('delete-user')) return;
                showUserDetails(user);
            });
            container.appendChild(card);
        });
        document.querySelectorAll('.delete-user').forEach(btn => {
            btn.onclick = async (e) => {
                e.stopPropagation();
                const id = btn.getAttribute('data-id');
                if (confirm('Delete this user permanently?')) {
                    const res = await fetch(`${API_BASE}/users/${id}`, { method: 'DELETE' });
                    if (res.ok) loadData();
                    else alert('Delete failed');
                }
            };
        });
    } catch(err) {
        container.innerHTML = `<div>Error: ${err.message}</div>`;
    }
}

function showUserDetails(user) {
    const modal = document.getElementById('userModal');
    document.getElementById('modalName').innerText = user.name;
    let details = `<p><strong>ID:</strong> ${user.id}</p><p><strong>Created:</strong> ${new Date(user.created_at).toLocaleString()}</p>`;
    if (user.metadata) {
        details += '<hr><h4>Metadata</h4><ul>';
        for (const [k,v] of Object.entries(user.metadata)) {
            details += `<li><strong>${k}:</strong> ${v}</li>`;
        }
        details += '</ul>';
    }
    document.getElementById('modalDetails').innerHTML = details;
    modal.style.display = 'flex';
}

// Dynamic fields for register
function addDynamicField(name, placeholder) {
    const container = document.getElementById('dynamicFields');
    const div = document.createElement('div');
    div.className = 'field-group';
    const input = document.createElement('input');
    input.name = name;
    input.placeholder = placeholder;
    const remove = document.createElement('button');
    remove.innerHTML = '<i class="fas fa-times-circle"></i>';
    remove.className = 'remove-field';
    remove.onclick = () => div.remove();
    div.appendChild(input);
    div.appendChild(remove);
    container.appendChild(div);
    return input;
}

function initDynamicFields() {
    const container = document.getElementById('dynamicFields');
    container.innerHTML = '';
    addDynamicField('name', 'Full Name (required)');
    addDynamicField('id_number', 'ID Number');
    addDynamicField('age', 'Age');
}
document.getElementById('addFieldBtn').onclick = () => {
    const key = prompt('Enter field name (e.g., "Country", "Phone"):');
    if (key) addDynamicField(key, key);
};

// Page switching
async function switchPage(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active-page'));
    document.getElementById(page + 'Page').classList.add('active-page');
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('nav' + page.charAt(0).toUpperCase() + page.slice(1)).classList.add('active');
    if (page === 'register') {
        currentMode = 'register';
        registrationSqueezeCount = 0;
        registrationRecordings = [];
        document.getElementById('registerStatus').innerHTML = 'Waiting for 3 squeezes...';
        initDynamicFields();
        await initWebcam('registerVideo', 'register');
    } else if (page === 'verify') {
        currentMode = 'verify';
        document.getElementById('verifyResult').style.display = 'none';
        await initWebcam('verifyVideo', 'verify');
    } else if (page === 'data') {
        if (camera) camera.stop();
        stopWebcam();
        await loadData();
    }
}

// Event listeners
document.getElementById('navRegister').onclick = () => switchPage('register');
document.getElementById('navVerify').onclick = () => switchPage('verify');
document.getElementById('navData').onclick = () => switchPage('data');
document.getElementById('refreshDataBtn').onclick = () => loadData();

// Modal close
document.querySelector('.close-modal').onclick = () => document.getElementById('userModal').style.display = 'none';
window.onclick = (e) => { if(e.target === document.getElementById('userModal')) document.getElementById('userModal').style.display = 'none'; };

// Start everything
window.onload = () => {
    startNeuralBackground();
    switchPage('register');
};
