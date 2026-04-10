<script>
    // ======================= NEURAL NETWORK BACKGROUND (GRID‑BASED) =======================
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
    
    window.addEventListener('resize', resizeAndInit);
    window.addEventListener('mousemove', (e) => { mouseX = e.clientX; mouseY = e.clientY; });
    resizeAndInit();
    animateNeural();
    
    // ======================= EYE‑D BIOMETRIC LOGIC (FIXED FOR MOBILE) =======================
    const API_BASE = "http://localhost:8000";
    let currentStream = null;
    let faceMesh = null;
    let animationId = null;
    let currentMode = "register";
    let registrationSqueezeCount = 0;
    let registrationRecordings = [];
    let verifyCallback = null;
    let lastEar = 1.0;
    let squeezeCooldown = false;
    let earHistory = [];
    const EAR_THRESHOLD = 0.22;
    const EAR_MIN_DURATION = 3; // frames

    // Helper: show status message with optional auto-clear
    function showStatus(elementId, message, isError = false, autoClear = 3000) {
        const el = document.getElementById(elementId);
        if (el) {
            el.innerHTML = message;
            el.style.color = isError ? '#f87171' : '#a5f3fc';
            if (autoClear && !isError) {
                setTimeout(() => {
                    if (el.innerHTML === message) el.innerHTML = '';
                }, autoClear);
            }
        }
    }

    // Stop webcam and all processing
    function stopWebcam() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
        if (faceMesh) {
            faceMesh.close();
            faceMesh = null;
        }
    }

    // Initialize webcam and FaceMesh with manual frame capture (more reliable on iOS)
    async function initWebcam(videoElementId, mode) {
        stopWebcam();
        currentMode = mode;
        const video = document.getElementById(videoElementId);
        if (!video) return;
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
            video.srcObject = stream;
            currentStream = stream;
            await video.play();
            
            // Initialize FaceMesh
            faceMesh = new FaceMesh({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
            });
            faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            // Process frames manually using requestAnimationFrame
            async function processFrame() {
                if (!video.videoWidth || !currentStream) return;
                await faceMesh.send({ image: video });
                animationId = requestAnimationFrame(processFrame);
            }
            faceMesh.onResults((results) => onFaceMeshResults(results, mode));
            animationId = requestAnimationFrame(processFrame);
            
            showStatus(mode === 'register' ? 'registerStatus' : 'verifyStatus', '✅ Camera ready. Look at the screen and squeeze your eye.', false, 4000);
        } catch (err) {
            console.error(err);
            showStatus(mode === 'register' ? 'registerStatus' : 'verifyStatus', '❌ Cannot access camera. Please grant permission.', true, 5000);
        }
    }

    // Eye Aspect Ratio
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
        if (horizontal === 0) return 1.0;
        return (vertical1 + vertical2) / (2.0 * horizontal);
    }

    // Squeeze detection logic
    function onFaceMeshResults(results, mode) {
        if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;
        const landmarks = results.multiFaceLandmarks[0];
        const ear = calculateEAR(landmarks, [33, 133, 157, 158, 159, 160, 161, 173]);
        
        // Smooth with moving average
        earHistory.push(ear);
        if (earHistory.length > 5) earHistory.shift();
        const avgEar = earHistory.reduce((a,b) => a+b, 0) / earHistory.length;
        
        const overlay = document.getElementById(mode === 'register' ? 'registerSqueezeOverlay' : 'verifySqueezeOverlay');
        const isSqueezing = avgEar < EAR_THRESHOLD;
        
        if (isSqueezing && !squeezeCooldown) {
            squeezeCooldown = true;
            if (overlay) {
                overlay.innerText = '👁️ SQUEEZE DETECTED!';
                overlay.classList.add('active');
                setTimeout(() => overlay.classList.remove('active'), 500);
            }
            // Trigger action
            if (mode === 'register') {
                handleRegisterSqueeze();
            } else if (mode === 'verify') {
                handleVerifySqueeze();
            }
            // Cooldown to avoid multiple triggers in one squeeze
            setTimeout(() => { squeezeCooldown = false; }, 1500);
        } else if (!isSqueezing && overlay && !squeezeCooldown) {
            if (mode === 'register' && registrationSqueezeCount < 3) {
                overlay.innerText = `Ready for squeeze ${registrationSqueezeCount+1}/3`;
            } else if (mode === 'verify') {
                overlay.innerText = '🔍 Monitoring';
            }
        }
    }

    // Register squeeze recording (3 times)
    async function handleRegisterSqueeze() {
        if (registrationSqueezeCount >= 3) {
            showStatus('registerStatus', 'Already 3 squeezes recorded. Uploading...', false);
            return;
        }
        const overlay = document.getElementById('registerSqueezeOverlay');
        overlay.innerText = `Recording ${registrationSqueezeCount+1}/3...`;
        const video = document.getElementById('registerVideo');
        const stream = video.srcObject;
        if (!stream) {
            showStatus('registerStatus', 'Camera not active', true);
            return;
        }
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
        statusDiv.innerHTML = '📤 Uploading enrollment data...';
        const formData = new FormData();
        for (let i=0; i<registrationRecordings.length; i++) {
            formData.append('videos', registrationRecordings[i], `squeeze_${i}.webm`);
        }
        const nameField = document.querySelector('.field-group input[name="name"]');
        const name = nameField ? nameField.value.trim() : '';
        if (!name) {
            showStatus('registerStatus', '❌ Name is required!', true);
            return;
        }
        formData.append('name', name);
        document.querySelectorAll('.field-group').forEach(field => {
            const input = field.querySelector('input');
            if (input && input.name !== 'name' && input.value.trim()) {
                formData.append(input.name, input.value.trim());
            }
        });
        try {
            const res = await fetch(`${API_BASE}/register/`, { method: 'POST', body: formData });
            const data = await res.json();
            if (res.ok) {
                statusDiv.innerHTML = `✅ Registration successful! User ID: ${data.user_id}`;
                alert(`User ${name} enrolled successfully!`);
                registrationSqueezeCount = 0;
                registrationRecordings = [];
                document.getElementById('registerSqueezeOverlay').innerText = 'Ready for next user';
            } else {
                statusDiv.innerHTML = `❌ Server error: ${data.detail || 'Unknown'}`;
            }
        } catch(err) {
            statusDiv.innerHTML = `❌ Network error: ${err.message}`;
        }
    }

    // Verify logic
    async function handleVerifySqueeze() {
        if (verifyCallback) return;
        const overlay = document.getElementById('verifySqueezeOverlay');
        overlay.innerText = '📸 Analyzing...';
        const video = document.getElementById('verifyVideo');
        const stream = video.srcObject;
        if (!stream) {
            showStatus('verifyStatus', 'Camera not active', true);
            overlay.innerText = '🔍 Monitoring';
            return;
        }
        const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        let chunks = [];
        recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
        recorder.onstop = async () => {
            const blob = new Blob(chunks, { type: 'video/webm' });
            const formData = new FormData();
            formData.append('video', blob, 'verify.webm');
            const username = prompt("Enter your username for verification:");
            if (!username) {
                showStatus('verifyStatus', 'Verification cancelled', true);
                overlay.innerText = '🔍 Monitoring';
                verifyCallback = null;
                return;
            }
            formData.append('username', username);
            const statusDiv = document.getElementById('verifyStatus');
            statusDiv.innerHTML = '🔄 Verifying...';
            try {
                const res = await fetch(`${API_BASE}/verify/`, { method: 'POST', body: formData });
                const data = await res.json();
                const resultDiv = document.getElementById('verifyResult');
                resultDiv.style.display = 'block';
                if (res.ok && data.accepted) {
                    resultDiv.innerHTML = `<strong>✅ ACCESS GRANTED</strong><br>Score: ${data.score.toFixed(4)}<br>User: ${username}`;
                    statusDiv.innerHTML = '✅ Match found.';
                } else {
                    resultDiv.innerHTML = `<strong>❌ ACCESS DENIED</strong><br>Score: ${data.score?.toFixed(4) || 'N/A'}<br>No match for ${username}`;
                    statusDiv.innerHTML = '❌ Verification failed.';
                }
            } catch(err) {
                statusDiv.innerHTML = `❌ Error: ${err.message}`;
            }
            overlay.innerText = '🔍 Monitoring';
            verifyCallback = null;
        };
        recorder.start();
        setTimeout(() => recorder.stop(), 2000);
        verifyCallback = true;
    }

    // Data page functions (unchanged but added error guidance)
    async function loadData() {
        const container = document.getElementById('userList');
        container.innerHTML = '<div>Loading neural records...</div>';
        try {
            const res = await fetch(`${API_BASE}/users`);
            if (!res.ok) throw new Error('Server returned ' + res.status);
            const users = await res.json();
            if (users.length === 0) {
                container.innerHTML = '<div>No users registered. Go to Register page first.</div>';
                return;
            }
            container.innerHTML = '';
            users.forEach(user => {
                const card = document.createElement('div');
                card.className = 'user-card';
                let metaPreview = '';
                if (user.metadata && typeof user.metadata === 'object') {
                    const entries = Object.entries(user.metadata).slice(0,2);
                    metaPreview = entries.map(([k,v]) => `<span><strong>${k}:</strong> ${v}</span>`).join(' | ');
                }
                card.innerHTML = `
                    <div class="user-name">
                        <span><i class="fas fa-user-circle"></i> ${user.name}</span>
                        <button class="delete-user" data-id="${user.id}"><i class="fas fa-trash"></i></button>
                    </div>
                    <div class="user-meta">ID: ${user.id} | ${metaPreview}</div>
                `;
                card.onclick = (e) => { if (!e.target.classList.contains('delete-user')) showUserDetails(user); };
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
            container.innerHTML = `<div>Error: ${err.message}. Make sure backend is running at ${API_BASE}</div>`;
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
    }
    function initDynamicFields() {
        const container = document.getElementById('dynamicFields');
        if (container) {
            container.innerHTML = '';
            addDynamicField('name', 'Full Name (required)');
            addDynamicField('id_number', 'ID Number');
            addDynamicField('age', 'Age');
        }
    }
    document.getElementById('addFieldBtn').onclick = () => {
        const key = prompt('Enter field name (e.g., "Country", "Phone"):');
        if (key) addDynamicField(key, key);
    };

    // Page navigation
    async function switchPage(page) {
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active-page'));
        document.getElementById(page + 'Page').classList.add('active-page');
        document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById('nav' + page.charAt(0).toUpperCase() + page.slice(1)).classList.add('active');
        if (page === 'register') {
            registrationSqueezeCount = 0;
            registrationRecordings = [];
            document.getElementById('registerStatus').innerHTML = 'Waiting for 3 squeezes...';
            initDynamicFields();
            await initWebcam('registerVideo', 'register');
        } else if (page === 'verify') {
            document.getElementById('verifyResult').style.display = 'none';
            await initWebcam('verifyVideo', 'verify');
        } else if (page === 'data') {
            stopWebcam();
            await loadData();
        }
    }

    // Welcome overlay
    const welcomeOverlay = document.getElementById('welcomeOverlay');
    function dismissWelcomeAndGoTo(page) {
        welcomeOverlay.style.opacity = '0';
        setTimeout(() => {
            welcomeOverlay.style.display = 'none';
            switchPage(page);
        }, 500);
    }
    document.getElementById('welcomeRegister').onclick = () => dismissWelcomeAndGoTo('register');
    document.getElementById('welcomeVerify').onclick = () => dismissWelcomeAndGoTo('verify');
    document.getElementById('welcomeData').onclick = () => dismissWelcomeAndGoTo('data');

    document.getElementById('navRegister').onclick = () => switchPage('register');
    document.getElementById('navVerify').onclick = () => switchPage('verify');
    document.getElementById('navData').onclick = () => switchPage('data');
    document.getElementById('refreshDataBtn').onclick = () => loadData();

    document.querySelector('.close-modal').onclick = () => document.getElementById('userModal').style.display = 'none';
    window.onclick = (e) => { if (e.target === document.getElementById('userModal')) document.getElementById('userModal').style.display = 'none'; };

    // Start with welcome overlay active, no camera until button click
    // The welcome overlay is visible, container is behind it. Camera will start when user picks a page.
</script>
