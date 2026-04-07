// script.js - handles file upload, API calls, dynamic UI updates, data panel
// Backend API base URL (adjust if your backend runs on different port)
const API_BASE = "http://localhost:8000";

// DOM elements
const registerBtn = document.getElementById('registerBtn');
const verifyBtn = document.getElementById('verifyBtn');
const dataBtn = document.getElementById('dataBtn');
const registerVideoInput = document.getElementById('registerVideo');
const verifyVideoInput = document.getElementById('verifyVideo');
const statusMessageDiv = document.getElementById('statusMessage');
const dataPanel = document.getElementById('dataPanel');
const closeDataPanel = document.getElementById('closeDataPanel');
const dataContent = document.getElementById('dataContent');

// Helper: update status message with optional loading indicator
function setStatus(message, isLoading = false, isError = false) {
    if (isLoading) {
        statusMessageDiv.innerHTML = `<span class="loading-spinner"></span> ${message}`;
    } else {
        statusMessageDiv.innerHTML = message;
        if (isError) {
            statusMessageDiv.style.color = "#f85149";
        } else {
            statusMessageDiv.style.color = "#e6edf3";
        }
    }
    // reset color after 5 seconds if error
    if (isError) {
        setTimeout(() => {
            if (!statusMessageDiv.innerHTML.includes("loading-spinner")) {
                statusMessageDiv.style.color = "#e6edf3";
            }
        }, 5000);
    }
}

// Helper: upload video to a specific endpoint
async function uploadVideo(file, endpoint, formDataBuilder) {
    if (!file) {
        setStatus("❌ No video selected.", false, true);
        return null;
    }
    if (file.size > 50 * 1024 * 1024) {
        setStatus("❌ Video too large (max 50MB).", false, true);
        return null;
    }
    const formData = new FormData();
    formDataBuilder(formData, file);

    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Request failed");
        }
        return data;
    } catch (err) {
        setStatus(`❌ ${err.message}`, false, true);
        return null;
    }
}

// REGISTER: send video(s) - we send one video per call (multiple calls possible)
// but backend expects list of videos. For simplicity, we'll send one video per registration.
// Actually register endpoint expects multiple videos? In routes/register.py it expects List[UploadFile].
// We'll send one video but the backend accepts list, so we wrap in array.
async function registerUser(videoFile, userName) {
    // prompt for username if not provided
    let name = userName;
    if (!name) {
        name = prompt("Enter a unique username for registration:", "user_" + Date.now());
        if (!name) return null;
    }
    const formData = new FormData();
    formData.append("name", name);
    // The backend expects field "videos" as list. We'll append same key multiple times.
    formData.append("videos", videoFile, videoFile.name);
    // If you want to send multiple videos, you could loop, but for demo one is enough.
    try {
        setStatus(`📝 Registering user "${name}"...`, true);
        const response = await fetch(`${API_BASE}/register/`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Registration failed");
        }
        setStatus(`✅ Registration successful! User ID: ${data.user_id}. Feature dimension: ${data.feature_dim}.`, false);
        // Store registration info in localStorage for data panel
        localStorage.setItem("eyed_last_user", JSON.stringify({ id: data.user_id, name: data.name, feature_dim: data.feature_dim }));
        return data;
    } catch (err) {
        setStatus(`❌ Registration error: ${err.message}`, false, true);
        return null;
    }
}

// VERIFY: send video + username/user_id
async function verifyUser(videoFile, userIdOrName) {
    let identifier = userIdOrName;
    if (!identifier) {
        identifier = prompt("Enter your username or user ID:", "");
        if (!identifier) return null;
    }
    const formData = new FormData();
    formData.append("video", videoFile, videoFile.name);
    // Determine if it's numeric ID or username
    if (!isNaN(parseInt(identifier)) && String(parseInt(identifier)) === identifier) {
        formData.append("user_id", identifier);
    } else {
        formData.append("username", identifier);
    }
    try {
        setStatus(`🔍 Verifying identity...`, true);
        const response = await fetch(`${API_BASE}/verify/`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Verification failed");
        }
        const resultText = data.accepted ? "✅ ACCESS GRANTED" : "❌ ACCESS DENIED";
        setStatus(`${resultText} (score: ${data.score.toFixed(4)}, threshold: ${data.threshold.toFixed(4)})`, false);
        // Store last verification result in localStorage for data panel
        localStorage.setItem("eyed_last_verification", JSON.stringify(data));
        return data;
    } catch (err) {
        setStatus(`❌ Verification error: ${err.message}`, false, true);
        return null;
    }
}

// DATA: show panel with stored information and optionally fetch from backend
async function showDataPanel() {
    // Clear previous content
    dataContent.innerHTML = '<p>Loading biometric data...</p>';
    dataPanel.classList.add('show');
    
    // Gather information from localStorage + any extra API calls if available
    let html = `<div style="margin-bottom: 1rem;"><i class="fas fa-chart-simple"></i> <strong>Recent Activity</strong></div>`;
    
    const lastUser = localStorage.getItem("eyed_last_user");
    if (lastUser) {
        const user = JSON.parse(lastUser);
        html += `<div class="data-card"><i class="fas fa-user-check"></i> Last Registered User: <strong>${user.name}</strong> (ID: ${user.id})<br>Feature dimension: ${user.feature_dim}</div>`;
    } else {
        html += `<div>No registration recorded yet.</div>`;
    }
    
    const lastVerif = localStorage.getItem("eyed_last_verification");
    if (lastVerif) {
        const verif = JSON.parse(lastVerif);
        const status = verif.accepted ? "Granted" : "Denied";
        html += `<div class="data-card" style="margin-top: 0.8rem;"><i class="fas fa-fingerprint"></i> Last Authentication: <strong>${status}</strong><br>Score: ${verif.score.toFixed(4)} | Threshold: ${verif.threshold.toFixed(4)}</div>`;
    } else {
        html += `<div>No verification attempts yet.</div>`;
    }
    
    // Optionally try to fetch all users (if backend had /users endpoint, but not required)
    html += `<div style="margin-top: 1rem; font-size: 0.8rem; color: #8b949e;"><i class="fas fa-info-circle"></i> Data is stored locally. For full server logs, backend must provide additional endpoints.</div>`;
    dataContent.innerHTML = html;
}

// Event listeners
registerBtn.addEventListener('click', () => {
    registerVideoInput.click();
});

verifyBtn.addEventListener('click', () => {
    verifyVideoInput.click();
});

registerVideoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await registerUser(file, null);
    registerVideoInput.value = ''; // allow re-upload same file
});

verifyVideoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await verifyUser(file, null);
    verifyVideoInput.value = '';
});

dataBtn.addEventListener('click', () => {
    showDataPanel();
});

closeDataPanel.addEventListener('click', () => {
    dataPanel.classList.remove('show');
});

// Optional: close panel when clicking outside (simple)
window.addEventListener('click', (e) => {
    if (dataPanel.classList.contains('show') && !dataPanel.contains(e.target) && e.target !== dataBtn) {
        dataPanel.classList.remove('show');
    }
});

// Initial status
setStatus("Eye‑D ready. Use Register (enroll) or Verify (authenticate).", false);
