import os
import uuid
import json

DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    class DBHandler:
        def _conn(self):
            return psycopg2.connect(DATABASE_URL)

        def __init__(self):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            template_work REAL NOT NULL,
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_metadata (
                            user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                            key TEXT,
                            value TEXT
                        )
                    """)
                conn.commit()

        def insert_user(self, name, template_work, metadata=None):
            user_id = str(uuid.uuid4())
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (id, name, template_work) VALUES (%s, %s, %s)",
                        (user_id, name, template_work)
                    )
                    if metadata:
                        for key, value in metadata.items():
                            cur.execute(
                                "INSERT INTO user_metadata (user_id, key, value) VALUES (%s, %s, %s)",
                                (user_id, key, str(value))
                            )
                conn.commit()
            return user_id

        def get_user_template(self, username):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT template_work FROM users WHERE name = %s", (username,))
                    row = cur.fetchone()
                    return row[0] if row else None

        def get_all_users(self):
            with self._conn() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT id, name, template_work, created_at FROM users ORDER BY created_at DESC")
                    users = []
                    for row in cur.fetchall():
                        user = dict(row)
                        user["created_at"] = user["created_at"].isoformat()
                        user["metadata"] = {}
                        cur2 = conn.cursor()
                        cur2.execute("SELECT key, value FROM user_metadata WHERE user_id = %s", (user["id"],))
                        for meta in cur2.fetchall():
                            user["metadata"][meta[0]] = meta[1]
                        users.append(user)
                    return users

        def delete_user(self, user_id):
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
                    deleted = cur.rowcount > 0
                conn.commit()
            return deleted

else:
    import sqlite3

    class DBHandler:
        def __init__(self, db_path="eyed.db"):
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self._create_tables()

        def _create_tables(self):
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    template_work REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_metadata (
                    user_id TEXT,
                    key TEXT,
                    value TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            self.conn.commit()

        def insert_user(self, name, template_work, metadata=None):
            user_id = str(uuid.uuid4())
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO users (id, name, template_work) VALUES (?, ?, ?)",
                (user_id, name, template_work)
            )
            if metadata:
                for key, value in metadata.items():
                    cur.execute(
                        "INSERT INTO user_metadata (user_id, key, value) VALUES (?, ?, ?)",
                        (user_id, key, str(value))
                    )
            self.conn.commit()
            return user_id

        def get_user_template(self, username):
            cur = self.conn.cursor()
            cur.execute("SELECT template_work FROM users WHERE name = ?", (username,))
            row = cur.fetchone()
            return row[0] if row else None

        def get_all_users(self):
            cur = self.conn.cursor()
            cur.execute("SELECT id, name, template_work, created_at FROM users ORDER BY created_at DESC")
            users = []
            for row in cur.fetchall():
                user = {"id": row[0], "name": row[1], "template_work": row[2], "created_at": row[3], "metadata": {}}
                meta_cur = self.conn.cursor()
                meta_cur.execute("SELECT key, value FROM user_metadata WHERE user_id = ?", (row[0],))
                for meta in meta_cur.fetchall():
                    user["metadata"][meta[0]] = meta[1]
                users.append(user)
            return users

        def delete_user(self, user_id):
            cur = self.conn.cursor()
            cur.execute("DELETE FROM user_metadata WHERE user_id = ?", (user_id,))
            cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
            self.conn.commit()
            return cur.rowcount > 0
