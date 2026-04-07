# incremental_update.py
# Updates the Gaussian Mixture Model with a new genuine feature vector (online learning).
# This allows the system to adapt to the user over time (e.g., aging, seasonal changes).
# Should be called ONLY after a successful authentication.

import numpy as np
import pickle
import os
import sys
from sklearn.mixture import GaussianMixture

# -------------------------------
# 1. LOAD EXISTING MODEL
# -------------------------------
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please run 'train.py' first.")
    sys.exit(1)

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

old_gmm = model_data['gmm']
old_threshold = model_data['threshold']
feature_dim = model_data['feature_dim']
old_n_samples = model_data['n_samples']

print(f"Loaded existing model. Previously trained on {old_n_samples} samples.")

# -------------------------------
# 2. LOAD THE NEW GENUINE FEATURE VECTOR
# -------------------------------
if len(sys.argv) > 1:
    feature_path = sys.argv[1]
else:
    feature_path = "new_squeeze.npy"

if not os.path.exists(feature_path):
    print(f"Error: Feature file '{feature_path}' not found.")
    print("Usage: python incremental_update.py <path_to_genuine_feature.npy>")
    sys.exit(1)

new_feature = np.load(feature_path).reshape(1, -1)

if new_feature.shape[1] != feature_dim:
    print(f"Error: Feature dimension mismatch. Expected {feature_dim}, got {new_feature.shape[1]}")
    sys.exit(1)

print(f"Loaded new genuine feature vector from {feature_path}")

# -------------------------------
# 3. PERFORM INCREMENTAL UPDATE
# -------------------------------
# scikit-learn's GaussianMixture does not have a direct partial_fit method.
# However, we can implement a simple moving window approach:
#   Keep a buffer of the last N genuine samples, retrain the GMM on that buffer.
# This is more robust than trying to update the mixture parameters directly.

# Maximum number of recent samples to retain (memory buffer)
MAX_BUFFER_SIZE = 100

# Path to store the buffer of recent genuine feature vectors
BUFFER_FILE = "genuine_buffer.npy"

# Load existing buffer if present
if os.path.exists(BUFFER_FILE):
    buffer = np.load(BUFFER_FILE)
    # buffer is a list of feature vectors; we need to handle shape correctly
    if buffer.ndim == 1:
        buffer = buffer.reshape(1, -1)
    print(f"Loaded existing buffer with {buffer.shape[0]} samples.")
else:
    buffer = np.empty((0, feature_dim))
    print("No existing buffer found, starting new buffer.")

# Append the new feature vector
buffer = np.vstack([buffer, new_feature])

# Keep only the most recent MAX_BUFFER_SIZE samples
if buffer.shape[0] > MAX_BUFFER_SIZE:
    buffer = buffer[-MAX_BUFFER_SIZE:, :]

print(f"Buffer now contains {buffer.shape[0]} samples.")

# Retrain the GMM on the buffer
new_gmm = GaussianMixture(
    n_components=old_gmm.n_components,
    covariance_type=old_gmm.covariance_type,
    random_state=42,
    max_iter=100
)
new_gmm.fit(buffer)

# Update threshold: use mean - 2*std of the buffer's scores
new_scores = new_gmm.score_samples(buffer)
new_threshold = np.mean(new_scores) - 2 * np.std(new_scores)

# -------------------------------
# 4. SAVE UPDATED MODEL AND BUFFER
# -------------------------------
updated_model_data = {
    'gmm': new_gmm,
    'threshold': new_threshold,
    'feature_dim': feature_dim,
    'n_samples': buffer.shape[0]
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(updated_model_data, f)

np.save(BUFFER_FILE, buffer)

print(f"Model updated. New threshold: {new_threshold:.4f}")
print(f"Buffer saved with {buffer.shape[0]} samples.")
print("Incremental update complete.")
