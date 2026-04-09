# db_handler.py
# Handles all database operations for the Eye-D system.
# Uses SQLite (file-based database) – no separate database server needed.
# Functions:
#   - init_db()           : creates database file and tables
#   - add_user()          : registers a new user after enrollment
#   - get_user()          : retrieves user by ID or name
#   - add_feature_vector(): stores a feature vector (enrollment or update)
#   - get_recent_vectors(): fetches last N vectors for a user (for incremental update)
#   - log_authentication(): records an authentication attempt
#   - get_auth_stats()    : returns statistics (e.g., false accept rate)

import os
import numpy as np
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base, User, FeatureVector, AuthenticationLog

# -------------------------------
# 1. DATABASE SETUP
# -------------------------------
# Database file path (change if you want a different location)
DATABASE_PATH = "eyed_database.sqlite"
# Create engine (connects SQLAlchemy to SQLite)
engine = create_engine(f'sqlite:///{DATABASE_PATH}', echo=False)   # echo=True prints SQL for debugging
# Create all tables if they don't exist
Base.metadata.create_all(engine)
# Session factory for interacting with the database
SessionLocal = sessionmaker(bind=engine)

def get_db_session() -> Session:
    """Returns a new database session. Always close the session after use."""
    return SessionLocal()

# -------------------------------
# 2. USER MANAGEMENT
# -------------------------------
def add_user(name: str, model_path: str, feature_dim: int) -> int:
    """
    Adds a new user to the database after successful enrollment.
    Returns the user's ID.
    """
    session = get_db_session()
    try:
        user = User(name=name, model_path=model_path, feature_dim=feature_dim)
        session.add(user)
        session.commit()
        user_id = user.id
        print(f"User '{name}' added with ID {user_id}")
        return user_id
    except Exception as e:
        session.rollback()
        print(f"Error adding user: {e}")
        return -1
    finally:
        session.close()

def get_user(user_id: int = None, name: str = None):
    """
    Retrieves a User object by either ID or name.
    Returns None if not found.
    """
    session = get_db_session()
    try:
        if user_id is not None:
            user = session.query(User).filter(User.id == user_id).first()
        elif name is not None:
            user = session.query(User).filter(User.name == name).first()
        else:
            user = None
        return user
    finally:
        session.close()

def list_all_users():
    """Returns a list of all enrolled users (for admin purposes)."""
    session = get_db_session()
    try:
        users = session.query(User).all()
        return users
    finally:
        session.close()

# -------------------------------
# 3. FEATURE VECTOR STORAGE
# -------------------------------
def add_feature_vector(user_id: int, vector: np.ndarray, vector_type: str = 'enrollment'):
    """
    Stores a feature vector (numpy array) in the database.
    The vector is pickled and stored as a BLOB.
    """
    session = get_db_session()
    try:
        # Serialize numpy array to bytes
        vector_blob = pickle.dumps(vector)
        fv = FeatureVector(
            user_id=user_id,
            vector_blob=vector_blob,
            vector_type=vector_type
        )
        session.add(fv)
        session.commit()
        print(f"Feature vector added for user_id={user_id}, type='{vector_type}'")
        return fv.id
    except Exception as e:
        session.rollback()
        print(f"Error adding feature vector: {e}")
        return -1
    finally:
        session.close()

def get_recent_vectors(user_id: int, limit: int = 100, vector_type: str = None):
    """
    Retrieves the most recent feature vectors for a user.
    If vector_type is specified (e.g., 'enrollment' or 'update'), filters by type.
    Returns a list of numpy arrays.
    """
    session = get_db_session()
    try:
        query = session.query(FeatureVector).filter(FeatureVector.user_id == user_id)
        if vector_type:
            query = query.filter(FeatureVector.vector_type == vector_type)
        query = query.order_by(FeatureVector.timestamp.desc()).limit(limit)
        fvs = query.all()
        # Deserialize each blob back to numpy array
        vectors = [pickle.loads(fv.vector_blob) for fv in fvs]
        return vectors
    finally:
        session.close()

def delete_old_vectors(user_id: int, keep_last: int = 100):
    """
    Deletes feature vectors older than the most recent 'keep_last' for a user.
    Useful for limiting database growth. Call this after incremental updates.
    """
    session = get_db_session()
    try:
        # Get IDs of vectors to keep (most recent 'keep_last')
        subquery = session.query(FeatureVector.id).filter(
            FeatureVector.user_id == user_id
        ).order_by(FeatureVector.timestamp.desc()).limit(keep_last).subquery()
        # Delete all not in subquery
        deleted = session.query(FeatureVector).filter(
            FeatureVector.user_id == user_id,
            FeatureVector.id.notin_(subquery)
        ).delete(synchronize_session=False)
        session.commit()
        print(f"Deleted {deleted} old feature vectors for user_id={user_id}")
    except Exception as e:
        session.rollback()
        print(f"Error deleting old vectors: {e}")
    finally:
        session.close()

# -------------------------------
# 4. AUTHENTICATION LOGGING
# -------------------------------
def log_authentication(user_id: int, score: float, threshold: float, accepted: bool):
    """
    Records an authentication attempt in the database.
    This is crucial for auditing and improving the system over time.
    """
    session = get_db_session()
    try:
        log = AuthenticationLog(
            user_id=user_id,
            score=score,
            threshold=threshold,
            accepted=accepted
        )
        session.add(log)
        session.commit()
        print(f"Auth log: user_id={user_id}, accepted={accepted}, score={score:.4f}")
    except Exception as e:
        session.rollback()
        print(f"Error logging authentication: {e}")
    finally:
        session.close()

def get_auth_stats(user_id: int):
    """
    Returns authentication statistics for a user:
        - total attempts
        - successful attempts
        - false reject rate (FRR) = rejects / (accepts+rejects)
    (False accepts are not tracked here because they would be attempts on other users.)
    """
    session = get_db_session()
    try:
        total = session.query(AuthenticationLog).filter(AuthenticationLog.user_id == user_id).count()
        accepted = session.query(AuthenticationLog).filter(
            AuthenticationLog.user_id == user_id,
            AuthenticationLog.accepted == True
        ).count()
        rejected = total - accepted
        frr = rejected / total if total > 0 else 0.0
        return {
            'total_attempts': total,
            'accepted': accepted,
            'rejected': rejected,
            'false_reject_rate': frr
        }
    finally:
        session.close()

# -------------------------------
# 5. DATABASE INITIALIZATION (already done at module load)
# -------------------------------
# The tables are created when this module is first imported.
# You can also call init_db() explicitly if needed.
def init_db():
    """Explicitly creates all tables (safe to call multiple times)."""
    Base.metadata.create_all(engine)
    print("Database initialized.")
