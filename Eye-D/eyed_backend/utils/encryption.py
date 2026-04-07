# utils/encryption.py
# Provides symmetric encryption (AES) for numpy arrays and pickled models.
# Uses a key derived from a passphrase (or stored in an environment variable).
# For a real deployment, the key should be managed by a secure key management service.

import os
import hashlib
import base64
import pickle
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# -------------------------------
# 1. KEY MANAGEMENT
# -------------------------------
# In production, do NOT hardcode the key. Use environment variables or a secrets manager.
# For development, generate a key from a fixed passphrase.
ENCRYPTION_PASSPHRASE = os.environ.get("EYED_ENCRYPTION_KEY", "change-this-secret-key-for-production")

def _derive_key(passphrase: str) -> bytes:
    """Derives a 32‑byte AES‑256 key from a passphrase using SHA‑256."""
    return hashlib.sha256(passphrase.encode('utf-8')).digest()

def _pad(data: bytes) -> bytes:
    """Adds PKCS#7 padding to make data length a multiple of 16 bytes (AES block size)."""
    pad_len = 16 - (len(data) % 16)
    return data + bytes([pad_len] * pad_len)

def _unpad(data: bytes) -> bytes:
    """Removes PKCS#7 padding."""
    pad_len = data[-1]
    if pad_len > 16 or pad_len == 0:
        raise ValueError("Invalid padding")
    return data[:-pad_len]

# -------------------------------
# 2. ENCRYPT / DECRYPT FUNCTIONS
# -------------------------------
def encrypt_numpy(array: np.ndarray) -> bytes:
    """
    Encrypts a numpy array using AES‑256‑CBC.
    Returns the concatenated (IV + ciphertext) as bytes.
    """
    key = _derive_key(ENCRYPTION_PASSPHRASE)
    # Serialize numpy array to bytes using pickle
    plain_bytes = pickle.dumps(array)
    padded = _pad(plain_bytes)
    # Generate a random 16‑byte Initialization Vector
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    # Return IV + ciphertext (IV is needed for decryption)
    return iv + ciphertext

def decrypt_numpy(encrypted_bytes: bytes) -> np.ndarray:
    """
    Decrypts data returned by encrypt_numpy back to a numpy array.
    """
    key = _derive_key(ENCRYPTION_PASSPHRASE)
    # First 16 bytes are the IV
    iv = encrypted_bytes[:16]
    ciphertext = encrypted_bytes[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_plain = decryptor.update(ciphertext) + decryptor.finalize()
    plain_bytes = _unpad(padded_plain)
    return pickle.loads(plain_bytes)

def encrypt_model(model_data: dict) -> bytes:
    """
    Encrypts a model dictionary (as used in model.pkl) directly.
    """
    import pickle
    plain = pickle.dumps(model_data)
    return encrypt_numpy(np.frombuffer(plain, dtype=np.uint8))  # reuse numpy encryption

def decrypt_model(encrypted_bytes: bytes) -> dict:
    """
    Decrypts a model dictionary.
    """
    arr = decrypt_numpy(encrypted_bytes)
    return pickle.loads(arr.tobytes())

# -------------------------------
# 3. CONVENIENCE FOR DATABASE STORAGE
# -------------------------------
def encrypt_feature_vector(vector: np.ndarray) -> bytes:
    """Encrypts a feature vector for safe storage in the database."""
    return encrypt_numpy(vector)

def decrypt_feature_vector(blob: bytes) -> np.ndarray:
    """Decrypts a feature vector from the database."""
    return decrypt_numpy(blob)
