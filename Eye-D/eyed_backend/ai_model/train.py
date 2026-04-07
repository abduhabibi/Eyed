# train.py
# Trains an initial Gaussian Mixture Model (GMM) for a user using their enrollment feature vectors.
# The GMM learns the probability distribution of the user's genuine squeezes.
# Saves the trained model as 'model.pkl' for later use in inference and incremental updates.

import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
import os
import glob

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
# Folder containing enrollment feature vectors for the user.
# Each .npy file should contain a single feature vector (e.g., from your extraction scripts).
ENROLLMENT_FOLDER = "enrollment_data/"   # CHANGE THIS

# Number of Gaussian components in the mixture model.
# More components = more expressive, but requires more data. 2-4 is typical for biometrics.
N_COMPONENTS = 3

# Random seed for reproducibility (ensures same results each run)
RANDOM_SEED = 42

# -------------------------------
# 2. LOAD ALL ENROLLMENT FEATURE VECTORS
# -------------------------------
# Find all .npy files in the enrollment folder
feature_files = glob.glob(os.path.join(ENROLLMENT_FOLDER, "*.npy"))

if len(feature_files) == 0:
    print("Error: No .npy feature files found in", ENROLLMENT_FOLDER)
    print("Please run feature extraction scripts first.")
    exit(1)

print(f"Found {len(feature_files)} enrollment feature vectors.")

# Load each feature vector and stack them into a 2D array (samples x features)
feature_list = []
for fpath in feature_files:
    vec = np.load(fpath)
    feature_list.append(vec)
X_train = np.vstack(feature_list)   # shape: (n_samples, n_features)

print(f"Feature vector dimension: {X_train.shape[1]}")
print(f"Number of enrollment samples: {X_train.shape[0]}")

# -------------------------------
# 3. TRAIN THE GAUSSIAN MIXTURE MODEL
# -------------------------------
# GMM learns the underlying distribution of genuine samples.
# During inference, we compute the log-likelihood of a new sample.
# Low likelihood (outlier) suggests an impostor.
gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    covariance_type='full',      # full covariance matrix captures feature correlations
    random_state=RANDOM_SEED,
    max_iter=200,                # maximum EM iterations
    tol=1e-3                     # convergence tolerance
)

print("Training GMM...")
gmm.fit(X_train)
print("Training complete.")

# Compute the average log-likelihood on the training data to establish a baseline.
# This will help set a threshold for authentication later.
train_scores = gmm.score_samples(X_train)
mean_train_score = np.mean(train_scores)
std_train_score = np.std(train_scores)

print(f"Mean training log-likelihood: {mean_train_score:.4f}")
print(f"Std training log-likelihood: {std_train_score:.4f}")

# Store threshold as mean - 2*std (allows for normal variation).
# This threshold can be adjusted later based on security needs.
threshold = mean_train_score - 2 * std_train_score
print(f"Initial authentication threshold: {threshold:.4f}")

# -------------------------------
# 4. SAVE THE MODEL AND THRESHOLD TO DISK
# -------------------------------
# We save both the GMM and the threshold in a dictionary.
model_data = {
    'gmm': gmm,
    'threshold': threshold,
    'feature_dim': X_train.shape[1],
    'n_samples': X_train.shape[0]
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved as 'model.pkl'.")
print("\nEnrollment complete. You can now use 'inference.py' to authenticate.")
