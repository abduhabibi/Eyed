# 👁️ Eye‑D: Authenticate with a Single Squeeze

**Eye‑D** is a new kind of biometric system. Instead of your fingerprint or your whole face, it recognises you by the **unique way you squeeze your eyes shut**.

> *No more taking off gloves in winter. No more removing your niqab or scarf. Just squeeze – and you're in.*

---

## 🧠 The Idea

When you forcefully squeeze your eyes, your eyelid doesn't just close. It follows a **personal sequence**:

- How fast it accelerates and brakes (your *kinetic signature*)
- The 3D curve your inner eye corner draws
- The order in which tiny wrinkles appear around your eye
- Which part of your eyelid closes first (inner, middle, or outer)

Even identical twins have different squeeze patterns – because the sequence depends on your unique muscle wiring, skin elasticity, and bone structure.

Eye‑D captures these movements with a **normal camera** (webcam or phone), turns them into a mathematical fingerprint, and uses an **adaptive AI** that gets better every time you log in.

---

## ✨ Why Eye‑D?

| Problem | Eye‑D Solution |
|---------|----------------|
| ❄️ Cold climates – gloves block fingerprints, scarves block Face ID | Works with face partially covered |
| 🧕 Niqab / hijab – traditional face recognition impossible | Only needs the eye region |
| 📱 Smart glasses & AR/VR – no fingerprint sensor, no front camera | Tiny inward camera is enough |
| 🔁 Aging – fingerprints wear, faces change | Model updates with each successful login |
| 🎭 Deepfake / replay attacks | Unique *dynamic* sequence is hard to fake |

---

## 🔬 How It Works (Simplified)

1. **You squeeze** your eye in front of any camera.
2. Eye‑D tracks **five feature families** in real time:
   - Kinetic magnitude (work, power, braking)
   - Main sequence (amplitude vs. peak velocity)
   - 3D path of the medial canthus (inner eye corner)
   - Wrinkle appearance order
   - Wave‑like closure delays
3. All measurements are **normalised** (works near or far) and **encrypted**.
4. A **Gaussian Mixture Model** (GMM) decides: is this you or an impostor?
5. After each successful login, the AI **updates itself** – so it ages with you.

---

## 🧰 Tech Stack

- **Backend**: FastAPI (Python), SQLite, SQLAlchemy
- **AI**: scikit‑learn (GMM), incremental learning
- **Computer Vision**: MediaPipe, OpenCV
- **Encryption**: AES‑256‑CBC
- **Frontend**: HTML, CSS, JavaScript (GitHub‑style dark theme)
- **Deployment**: uvicorn, any cloud (or local)

---

## 🚀 Quick Start

## 📄 License
For research and personal use only. Commercial licensing available – contact the author.

## 🙌 Acknowledgements
MediaPipe team for face landmarks

scikit‑learn for GMM implementation

OpenCV & FastAPI communities
cd eyed-biometric
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
