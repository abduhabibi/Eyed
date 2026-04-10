# models.py
# Defines the database tables for the Eye-D biometric system.
# Uses SQLAlchemy ORM to map Python classes to SQLite tables.
# Tables:
#   1. User – stores each enrolled user's identity and model metadata.
#   2. FeatureVector – stores individual feature vectors (enrollment or updates).
#   3. AuthenticationLog – records every authentication attempt (success/failure).

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

# Base class for all table definitions
Base = declarative_base()

class User(Base):
    """
    Represents a single enrolled user.
    Each user has a unique ID, name, and the path to their trained model file (.pkl).
    Also stores the feature dimension (for consistency checks) and creation timestamp.
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)                     # Auto-incrementing user ID
    name = Column(String(100), nullable=False)                 # User's display name
    model_path = Column(String(255), nullable=False)           # File path to model.pkl
    feature_dim = Column(Integer, nullable=False)              # Number of features in feature vector
    metadata = Column(String(2000), nullable=True)             # JSON string for custom fields
    created_at = Column(DateTime, default=datetime.utcnow)    # Enrollment date/time

    # Relationships (links to other tables)
    feature_vectors = relationship("FeatureVector", back_populates="user", cascade="all, delete-orphan")
    auth_logs = relationship("AuthenticationLog", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}', feature_dim={self.feature_dim})>"


class FeatureVector(Base):
    """
    Stores individual feature vectors (numpy arrays) for a user.
    These are used for enrollment and incremental updates.
    The buffer for incremental learning is reconstructed from recent vectors.
    """
    __tablename__ = 'feature_vectors'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    # Store the numpy array as binary large object (BLOB)
    vector_blob = Column(LargeBinary, nullable=False)          # Pickled numpy array
    timestamp = Column(DateTime, default=datetime.utcnow)      # When this vector was captured
    # Optional: type of vector ('enrollment' or 'update')
    vector_type = Column(String(20), default='enrollment')

    # Relationship back to User
    user = relationship("User", back_populates="feature_vectors")

    def __repr__(self):
        return f"<FeatureVector(id={self.id}, user_id={self.user_id}, type='{self.vector_type}', timestamp={self.timestamp})>"


class AuthenticationLog(Base):
    """
    Logs every authentication attempt (both successes and failures).
    Useful for auditing, debugging, and detecting brute‑force attacks.
    """
    __tablename__ = 'auth_logs'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    # The feature vector that was presented (optional, can be large – store only if needed)
    # For brevity, we store the score and decision only.
    score = Column(Float, nullable=False)                     # Log-likelihood score from GMM
    threshold = Column(Float, nullable=False)                 # Threshold used at that time
    accepted = Column(Boolean, nullable=False)                # True if accepted, False if rejected
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="auth_logs")

    def __repr__(self):
        return f"<AuthLog(user_id={self.user_id}, accepted={self.accepted}, score={self.score:.4f})>"
