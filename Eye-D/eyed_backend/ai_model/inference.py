# inference.py
# Loads the trained model and uses it to verify a new eyelid squeeze feature vector.
# Outputs: ACCEPT (genuine) or REJECT (impostor).

import numpy as np
import pickle
import sys
import os

# -------------------------------
# 1. LOAD THE TRAINED MODEL
# -------------------------------
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please run 'train.py' first to enroll a user.")
    sys.exit(1)

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

gmm = model_data['gmm']
threshold = model_data['threshold']
feature_dim = model_data['feature_dim']

print(f"Model loaded. Feature dimension: {feature_dim}")
print(f"Authentication threshold: {threshold:.4f}")

# -------------------------------
# 2. LOAD THE NEW FEATURE VECTOR (from extraction script output)
# -------------------------------
# The new feature vector should be a .npy file (e.g., 'final_feature_vector.npy').
# You can also pass the file path as a command line argument.
if len(sys.argv) > 1:
    feature_path = sys.argv[1]
else:
    # Default: look for a file named 'new_squeeze.npy' in current directory
    feature_path = "new_squeeze.npy"

if not os.path.exists(feature_path):
    print(f"Error: Feature file '{feature_path}' not found.")
    print("Usage: python inference.py <path_to_feature_vector.npy>")
    sys.exit(1)

new_feature = np.load(feature_path)

# Check dimension consistency
if new_feature.shape[0] != feature_dim:
    print(f"Error: Feature dimension mismatch. Expected {feature_dim}, got {new_feature.shape[0]}")
    sys.exit(1)

print(f"Loaded new feature vector from {feature_path}")

# -------------------------------
# 3. COMPUTE LOG‑LIKELIHOOD AND MAKE DECISION
# -------------------------------
# Higher log-likelihood = more likely to be genuine.
score = gmm.score_samples(new_feature.reshape(1, -1))[0]

print(f"Log-likelihood score: {score:.4f}")

if score >= threshold:
    print("DECISION: ACCEPT (genuine user)")
    # Optionally, save this successful sample for incremental learning
    # (the incremental_update.py script will handle that)
    sys.exit(0)
else:
    print("DECISION: REJECT (impostor or unknown)")
    sys.exit(1)
