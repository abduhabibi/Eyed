# database/db_handler.py
import sqlite3
import uuid

class DBHandler:
    def __init__(self, db_path="eyed.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        # Users table stores only name and template_work (Feature 1)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                template_work REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Metadata table for extra fields (id_number, age, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_metadata (
                user_id TEXT,
                key TEXT,
                value TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        self.conn.commit()

    def insert_user(self, name, template_work, metadata=None):
        """Insert a new user with work template and optional metadata."""
        user_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO users (id, name, template_work) VALUES (?, ?, ?)",
            (user_id, name, template_work)
        )
        if metadata:
            for key, value in metadata.items():
                cursor.execute(
                    "INSERT INTO user_metadata (user_id, key, value) VALUES (?, ?, ?)",
                    (user_id, key, value)
                )
        self.conn.commit()
        return user_id

    def get_user_template(self, username):
        """Return the stored work template for a given username."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT template_work FROM users WHERE name = ?", (username,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_all_users(self):
        """Return list of all users with basic info."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, template_work, created_at FROM users ORDER BY created_at DESC")
        users = []
        for row in cursor.fetchall():
            user = {
                "id": row[0],
                "name": row[1],
                "template_work": row[2],
                "created_at": row[3],
                "metadata": {}
            }
            # Fetch metadata
            meta_cursor = self.conn.cursor()
            meta_cursor.execute("SELECT key, value FROM user_metadata WHERE user_id = ?", (row[0],))
            for meta_row in meta_cursor.fetchall():
                user["metadata"][meta_row[0]] = meta_row[1]
            users.append(user)
        return users

    def delete_user(self, user_id):
        """Delete a user and their metadata."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM user_metadata WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0
