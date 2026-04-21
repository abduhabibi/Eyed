// ======================= NEURAL BACKGROUND =======================
const canvas = document.getElementById('neuralCanvas');
const ctx = canvas.getContext('2d');
let nodes = [];
let mouseX = 0, mouseY = 0;
const CELL_SIZE = 200;
let cols = 0, rows = 0;

function resizeAndInit() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    cols = Math.ceil(canvas.width / CELL_SIZE);
    rows = Math.ceil(canvas.height / CELL_SIZE);
    initNodes();
}

function initNodes() {
    nodes = [];
    for (let col = 0; col < cols; col++) {
        for (let row = 0; row < rows; row++) {
            const cellX = col * CELL_SIZE;
            const cellY = row * CELL_SIZE;
            const numNodes = Math.floor(Math.random() * 3) + 2;
            for (let i = 0; i < numNodes; i++) {
                nodes.push({
                    x: cellX + Math.random() * CELL_SIZE,
                    y: cellY + Math.random() * CELL_SIZE,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    radius: Math.random() * 2 + 1.5,
                    cellCol: col,
                    cellRow: row,
                    baseHue: 200 + Math.random() * 40
                });
            }
        }
    }
}

function updateNodes() {
    for (let node of nodes) {
        const cellMinX = node.cellCol * CELL_SIZE;
        const cellMaxX = cellMinX + CELL_SIZE;
        const cellMinY = node.cellRow * CELL_SIZE;
        const cellMaxY = cellMinY + CELL_SIZE;
        node.x += node.vx;
        node.y += node.vy;
        if (node.x < cellMinX) { node.x = cellMinX; node.vx *= -0.8; }
        if (node.x > cellMaxX) { node.x = cellMaxX; node.vx *= -0.8; }
        if (node.y < cellMinY) { node.y = cellMinY; node.vy *= -0.8; }
        if (node.y > cellMaxY) { node.y = cellMaxY; node.vy *= -0.8; }
    }
}

function drawNeural() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const dist = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y);
            if (dist < 150) {
                const opacity = (1 - dist / 150) * 0.4;
                ctx.beginPath();
                ctx.moveTo(nodes[i].x, nodes[i].y);
                ctx.lineTo(nodes[j].x, nodes[j].y);
                ctx.strokeStyle = `rgba(0, 180, 255, ${opacity})`;
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }
        }
    }
    const mouseCol = Math.floor(mouseX / CELL_SIZE);
    const mouseRow = Math.floor(mouseY / CELL_SIZE);
    for (let node of nodes) {
        if (node.cellCol === mouseCol && node.cellRow === mouseRow) {
            const distToMouse = Math.hypot(node.x - mouseX, node.y - mouseY);
            if (distToMouse < 150) {
                const intensity = 1 - distToMouse / 150;
                ctx.beginPath();
                ctx.moveTo(node.x, node.y);
                ctx.lineTo(mouseX, mouseY);
                ctx.strokeStyle = `rgba(0, 220, 255, ${intensity * 0.9})`;
                ctx.lineWidth = 2.2;
                ctx.shadowBlur = 8;
                ctx.shadowColor = '#00ccff';
                ctx.stroke();
            }
        }
    }
    ctx.shadowBlur = 0;
    for (let node of nodes) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(node.x-2, node.y-2, 1, node.x, node.y, node.radius);
        grad.addColorStop(0, `hsl(${node.baseHue}, 100%, 70%)`);
        grad.addColorStop(1, `hsl(${node.baseHue}, 80%, 40%)`);
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.shadowBlur = 3;
        ctx.shadowColor = `hsl(${node.baseHue}, 100%, 60%)`;
        ctx.fill();
        ctx.shadowBlur = 0;
    }
}

function animateNeural() {
    updateNodes();
    drawNeural();
    requestAnimationFrame(animateNeural);
}

window.addEventListener('resize', () => { resizeAndInit(); });
window.addEventListener('mousemove', (e) => { mouseX = e.clientX; mouseY = e.clientY; });
resizeAndInit();
animateNeural();

// ======================= EYE-D BIOMETRIC =======================
const API_BASE = "http://localhost:8000";
let currentStream = null;
let faceMesh = null;
let animationId = null;
let lastLandmarks = null;

const EAR_THRESHOLD = 0.22;

let regState = {
    active: false,
    squeezeCount: 0,
    recordings: []
};

function showStatus(elementId, message, isError = false) {
    const el = document.getElementById(elementId);
    if (el) {
        el.innerHTML = message;
        el.style.color = isError ? '#f87171' : '#a5f3fc';
    }
}

function calculateEAR(landmarks, indices) {
    const pts = indices.map(i => landmarks[i]);
    const v1 = Math.hypot(pts[1].x - pts[5].x, pts[1].y - pts[5].y);
    const v2 = Math.hypot(pts[2].x - pts[4].x, pts[2].y - pts[4].y);
    const h = Math.hypot(pts[0].x - pts[3].x, pts[0].y - pts[3].y);
    return h === 0 ? 1.0 : (v1 + v2) / (2 * h);
}

function drawEyeLandmarks(videoId, canvasId, landmarks) {
    const video = document.getElementById(videoId);
    const canvas = document.getElementById(canvasId);
    if (!video || !canvas || !landmarks) return;

    const w = video.clientWidth;
    const h = video.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#00ff88';
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.shadowBlur = 6;
    ctx.shadowColor = '#00ff88';

    const leftEye = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
    const rightEye = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382];

    [leftEye, rightEye].forEach(eye => {
        eye.forEach(i => {
            const pt = landmarks[i];
            ctx.beginPath();
            ctx.arc(pt.x * w, pt.y * h, 2.5, 0, 2 * Math.PI);
            ctx.fill();
        });
        ctx.beginPath();
        eye.forEach((i, j) => {
            const pt = landmarks[i];
            if (j === 0) ctx.moveTo(pt.x * w, pt.y * h);
            else ctx.lineTo(pt.x * w, pt.y * h);
        });
        ctx.closePath();
        ctx.stroke();
    });
    ctx.shadowBlur = 0;
}

function stopWebcam() {
    if (animationId) cancelAnimationFrame(animationId);
    if (currentStream) currentStream.getTracks().forEach(t => t.stop());
    if (faceMesh) faceMesh.close();
    currentStream = null;
    faceMesh = null;
    lastLandmarks = null;
}

async function initWebcam(videoId, mode) {
    stopWebcam();
    const video = document.getElementById(videoId);
    if (!video) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
        currentStream = stream;
        await video.play();

        faceMesh = new FaceMesh({
            locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`
        });
        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.4,
            minTrackingConfidence: 0.4
        });

        faceMesh.onResults((results) => {
            const statusId = mode === 'register' ? 'registerStatus' : 'verifyStatus';
            const displayId = mode === 'register' ? 'registerEarDisplay' : 'verifyEarDisplay';
            const canvasId = mode === 'register' ? 'registerOverlay' : 'verifyOverlay';

            if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                showStatus(statusId, '⚠️ No face detected', false);
                document.getElementById(canvasId).getContext('2d').clearRect(0, 0, 10, 10);
                return;
            }

            lastLandmarks = results.multiFaceLandmarks[0];
            const ear = calculateEAR(lastLandmarks, [33, 160, 158, 133, 153, 144]);
            const isOpen = ear >= EAR_THRESHOLD;

            document.getElementById(displayId).innerHTML = `👀 EAR: ${ear.toFixed(3)} | ${isOpen ? '🟢 Open' : '🔴 Closed'}`;
            drawEyeLandmarks(videoId, canvasId, lastLandmarks);
        });

        async function processFrame() {
            if (video.videoWidth && currentStream) {
                await faceMesh.send({ image: video });
            }
            animationId = requestAnimationFrame(processFrame);
        }
        processFrame();

        showStatus(mode === 'register' ? 'registerStatus' : 'verifyStatus', '✅ Camera ready', false);
    } catch (err) {
        showStatus(mode === 'register' ? 'registerStatus' : 'verifyStatus', '❌ Camera error', true);
    }
}

async function startRegister() {
    const name = document.querySelector('input[name="name"]')?.value.trim();
    if (!name) {
        alert('⚠️ Please enter your name');
        return;
    }

    if (!regState.active) {
        regState.active = true;
        regState.squeezeCount = 0;
        regState.recordings = [];

        document.getElementById('startRegisterBtn').style.display = 'none';
        document.getElementById('recordSqueezeBtn').style.display = 'block';
        document.getElementById('uploadRegistrationBtn').style.display = 'none';

        await initWebcam('registerVideo', 'register');
        showStatus('registerStatus', '✅ Recording ready. Click RECORD SQUEEZE 3 times', false);
    }
}

async function recordSqueeze() {
    if (regState.squeezeCount >= 3) return;

    const stream = document.getElementById('registerVideo').srcObject;
    if (!stream) {
        showStatus('registerStatus', '❌ Camera inactive', true);
        return;
    }

    regState.squeezeCount++;
    showStatus('registerStatus', `Recording squeeze ${regState.squeezeCount}/3...`);

    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    let chunks = [];

    recorder.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
    };

    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        regState.recordings.push(blob);

        document.getElementById('recordSqueezeBtn').innerHTML =
            `<i class="fas fa-record-vinyl"></i> RECORD SQUEEZE (${regState.squeezeCount}/3)`;

        if (regState.squeezeCount === 3) {
            showStatus('registerStatus', '✅ 3 squeezes recorded. Click UPLOAD', false);
            document.getElementById('recordSqueezeBtn').style.display = 'none';
            document.getElementById('uploadRegistrationBtn').style.display = 'block';
        } else {
            showStatus('registerStatus', `✅ Squeeze ${regState.squeezeCount}/3 done. Record ${3 - regState.squeezeCount} more`, false);
        }
    };

    recorder.start();
    setTimeout(() => recorder.stop(), 2000);
}

async function uploadRegistration() {
    if (regState.recordings.length !== 3) {
        alert('Need 3 recordings');
        return;
    }

    const nameField = document.querySelector('input[name="name"]');
    const name = nameField?.value.trim();
    if (!name) {
        showStatus('registerStatus', '❌ Name required', true);
        return;
    }

    showStatus('registerStatus', '📤 Uploading...');

    const formData = new FormData();
    formData.append('name', name);

    regState.recordings.forEach((blob, i) => {
        formData.append('videos', blob, `squeeze_${i}.webm`);
    });

    document.querySelectorAll('.field-group input').forEach(input => {
        if (input.name !== 'name' && input.value.trim()) {
            formData.append(input.name, input.value.trim());
        }
    });

    try {
        const res = await fetch(`${API_BASE}/register/`, { method: 'POST', body: formData });

        if (!res.ok) {
            const data = await res.json();
            showStatus('registerStatus', `❌ ${data.detail || 'Upload failed'}`, true);
            return;
        }

        const data = await res.json();
        alert(`✅ ${name} registered! ID: ${data.user_id}`);
        regState.active = false;
        regState.squeezeCount = 0;
        regState.recordings = [];
        nameField.value = '';
        document.getElementById('recordSqueezeBtn').style.display = 'none';
        document.getElementById('uploadRegistrationBtn').style.display = 'none';
        document.getElementById('startRegisterBtn').style.display = 'block';
        stopWebcam();
        showStatus('registerStatus', '✅ Ready for next user', false);
    } catch (err) {
        showStatus('registerStatus', `❌ Backend not running: ${err.message}`, true);
    }
}

async function verifyNow() {
    const stream = document.getElementById('verifyVideo').srcObject;
    if (!stream) {
        showStatus('verifyStatus', '❌ Camera not active', true);
        return;
    }

    const username = prompt('Enter username:');
    if (!username) return;

    showStatus('verifyStatus', '📸 Recording 2 seconds...');

    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    let chunks = [];

    recorder.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
    };

    recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const formData = new FormData();
        formData.append('video', blob, 'verify.webm');
        formData.append('username', username);

        showStatus('verifyStatus', '🔄 Verifying...');

        try {
            const res = await fetch(`${API_BASE}/verify/`, { method: 'POST', body: formData });
            const data = await res.json();
            const resultDiv = document.getElementById('verifyResult');
            resultDiv.style.display = 'block';

            if (res.ok && data.accepted) {
                resultDiv.innerHTML = `<strong>✅ ACCESS GRANTED</strong><br>Score: ${data.score.toFixed(4)}<br>User: ${username}`;
                showStatus('verifyStatus', '✅ Match found!', false);
            } else {
                resultDiv.innerHTML = `<strong>❌ ACCESS DENIED</strong><br>Score: ${data.score?.toFixed(4) || 'N/A'}`;
                showStatus('verifyStatus', '❌ No match', true);
            }
        } catch (err) {
            showStatus('verifyStatus', `❌ ${err.message}`, true);
        }
    };

    recorder.start();
    setTimeout(() => recorder.stop(), 2000);
}

async function loadData() {
    const container = document.getElementById('userList');
    if (!container) return;

    container.innerHTML = '<div>Loading...</div>';

    try {
        const res = await fetch(`${API_BASE}/users`, { timeout: 5000 });
        if (!res.ok) {
            container.innerHTML = '<div>⚠️ Backend not running at ' + API_BASE + '</div>';
            return;
        }

        const users = await res.json();

        if (users.length === 0) {
            container.innerHTML = '<div>No users registered yet</div>';
            return;
        }

        container.innerHTML = '';
        users.forEach(user => {
            const card = document.createElement('div');
            card.className = 'user-card';

            let metaPreview = '';
            if (user.metadata && typeof user.metadata === 'object') {
                const entries = Object.entries(user.metadata).slice(0, 2);
                metaPreview = entries.map(([k, v]) => `${k}: ${v}`).join(' | ');
            }

            card.innerHTML = `
                <div class="user-name">
                    <span><i class="fas fa-user-circle"></i> ${user.name}</span>
                    <button class="delete-user" data-id="${user.id}"><i class="fas fa-trash"></i></button>
                </div>
                <div class="user-meta">ID: ${user.id} ${metaPreview ? '| ' + metaPreview : ''}</div>
            `;

            card.onclick = (e) => {
                if (!e.target.classList.contains('delete-user')) showUserDetails(user);
            };

            container.appendChild(card);
        });

        document.querySelectorAll('.delete-user').forEach(btn => {
            btn.onclick = async (e) => {
                e.stopPropagation();
                const id = btn.getAttribute('data-id');
                if (confirm('Delete user?')) {
                    const res = await fetch(`${API_BASE}/users/${id}`, { method: 'DELETE' });
                    if (res.ok) loadData();
                }
            };
        });
    } catch (err) {
        container.innerHTML = '<div>⚠️ Backend not running: ' + API_BASE + '</div>';
    }
}

function showUserDetails(user) {
    const modal = document.getElementById('userModal');
    document.getElementById('modalName').innerText = user.name;

    let details = `<p><strong>ID:</strong> ${user.id}</p>`;
    details += `<p><strong>Created:</strong> ${new Date(user.created_at).toLocaleString()}</p>`;

    if (user.metadata && Object.keys(user.metadata).length > 0) {
        details += '<hr><h4>Metadata</h4><ul>';
        for (const [k, v] of Object.entries(user.metadata)) {
            details += `<li><strong>${k}:</strong> ${v}</li>`;
        }
        details += '</ul>';
    }

    document.getElementById('modalDetails').innerHTML = details;
    modal.style.display = 'flex';
}

function addDynamicField(name, placeholder) {
    const container = document.getElementById('dynamicFields');
    const div = document.createElement('div');
    div.className = 'field-group';

    const input = document.createElement('input');
    input.type = 'text';
    input.name = name;
    input.placeholder = placeholder;

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.innerHTML = '<i class="fas fa-times-circle"></i>';
    remove.className = 'remove-field';
    remove.onclick = (e) => {
        e.preventDefault();
        div.remove();
    };

    div.appendChild(input);
    div.appendChild(remove);
    container.appendChild(div);
}

function initDynamicFields() {
    const container = document.getElementById('dynamicFields');
    if (!container) return;
    container.innerHTML = '';
    addDynamicField('name', 'Full Name (required)');
    addDynamicField('id_number', 'ID Number');
    addDynamicField('age', 'Age');
}

async function switchPage(page) {
    stopWebcam();

    document.querySelectorAll('.page').forEach(p => p.classList.remove('active-page'));
    document.getElementById(page + 'Page').classList.add('active-page');

    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('nav' + page.charAt(0).toUpperCase() + page.slice(1)).classList.add('active');

    if (page === 'register') {
        initDynamicFields();
        showStatus('registerStatus', '⏳ Enter your name, then click START', false);
    } else if (page === 'verify') {
        showStatus('verifyStatus', 'Click VERIFY to identify', false);
        document.getElementById('verifyResult').style.display = 'none';
        await initWebcam('verifyVideo', 'verify');
    } else if (page === 'data') {
        await loadData();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('welcomeRegister').onclick = () => {
        document.getElementById('welcomeOverlay').style.display = 'none';
        switchPage('register');
    };
    document.getElementById('welcomeVerify').onclick = () => {
        document.getElementById('welcomeOverlay').style.display = 'none';
        switchPage('verify');
    };
    document.getElementById('welcomeData').onclick = () => {
        document.getElementById('welcomeOverlay').style.display = 'none';
        switchPage('data');
    };

    document.getElementById('navRegister').onclick = () => switchPage('register');
    document.getElementById('navVerify').onclick = () => switchPage('verify');
    document.getElementById('navData').onclick = () => switchPage('data');

    document.getElementById('startRegisterBtn').onclick = startRegister;
    document.getElementById('recordSqueezeBtn').onclick = recordSqueeze;
    document.getElementById('uploadRegistrationBtn').onclick = uploadRegistration;
    document.getElementById('verifyNowBtn').onclick = verifyNow;

    document.getElementById('addFieldBtn').onclick = () => {
        const key = prompt('Field name (e.g., Country, Phone):');
        if (key) addDynamicField(key, key);
    };

    document.getElementById('refreshDataBtn').onclick = loadData;

    document.querySelector('.close-modal').onclick = () => {
        document.getElementById('userModal').style.display = 'none';
    };
    window.onclick = (e) => {
        const modal = document.getElementById('userModal');
        if (e.target === modal) modal.style.display = 'none';
    };

    initDynamicFields();
});
