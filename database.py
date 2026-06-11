"""
SQLite database for users and prediction history.
"""

import csv
import io
import sqlite3
from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash

from config import DATABASE_PATH, CONFIDENCE_THRESHOLD, DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_user_id_column(conn):
    """Add user_id to predictions if upgrading old database."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    if "user_id" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")


def init_db():
    """Create tables and default admin account."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            filepath TEXT,
            emotion TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    _ensure_user_id_column(conn)

    admin = conn.execute(
        "SELECT id FROM users WHERE username = ?", (DEFAULT_ADMIN_USERNAME,)
    ).fetchone()
    if not admin:
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, role, created_at)
            VALUES (?, ?, ?, 'admin', ?)
            """,
            (
                DEFAULT_ADMIN_USERNAME,
                "admin@voice-sentiment.local",
                generate_password_hash(DEFAULT_ADMIN_PASSWORD),
                datetime.utcnow().isoformat(),
            ),
        )

    conn.commit()
    conn.close()


def create_user(username, password, email=None, role="user"):
    """Register a new user. Returns user id or None if username exists."""
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                username,
                email,
                generate_password_hash(password),
                role,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        uid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return uid
    except sqlite3.IntegrityError:
        conn.close()
        return None


def get_user_by_username(username):
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    if not user_id:
        return None
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def verify_user(username, password):
    user = get_user_by_username(username)
    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


def get_all_users():
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT u.id, u.username, u.email, u.role, u.created_at,
               COUNT(p.id) as prediction_count
        FROM users u
        LEFT JOIN predictions p ON p.user_id = u.id
        GROUP BY u.id
        ORDER BY u.id
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_prediction(filename, filepath, emotion, sentiment, confidence, user_id=None):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO predictions (user_id, filename, filepath, emotion, sentiment, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            filename,
            filepath,
            emotion,
            sentiment,
            float(confidence),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    pred_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return pred_id


def get_predictions_filtered(sentiment=None, search=None, user_id=None, limit=100):
    conn = get_connection()
    query = """
        SELECT p.*, u.username as owner_username
        FROM predictions p
        LEFT JOIN users u ON u.id = p.user_id
        WHERE 1=1
    """
    params = []

    if user_id is not None:
        query += " AND p.user_id = ?"
        params.append(user_id)

    if sentiment and sentiment in ("Positive", "Neutral", "Negative"):
        query += " AND p.sentiment = ?"
        params.append(sentiment)

    if search:
        query += " AND p.filename LIKE ?"
        params.append(f"%{search}%")

    query += " ORDER BY p.id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_predictions(limit=100):
    return get_predictions_filtered(limit=limit)


def get_prediction_by_id(pred_id):
    conn = get_connection()
    row = conn.execute(
        """
        SELECT p.*, u.username as owner_username
        FROM predictions p
        LEFT JOIN users u ON u.id = p.user_id
        WHERE p.id = ?
        """,
        (pred_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def user_can_access_prediction(pred, user_id, role):
    if role == "admin":
        return True
    if pred.get("user_id") is None:
        return True
    return pred.get("user_id") == user_id


def get_statistics(user_id=None):
    conn = get_connection()
    if user_id is not None:
        total = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        by_sentiment = conn.execute(
            """
            SELECT sentiment, COUNT(*) as count FROM predictions
            WHERE user_id = ? GROUP BY sentiment
            """,
            (user_id,),
        ).fetchall()
        by_emotion = conn.execute(
            """
            SELECT emotion, COUNT(*) as count FROM predictions
            WHERE user_id = ? GROUP BY emotion
            """,
            (user_id,),
        ).fetchall()
        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM predictions WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        low_conf = conn.execute(
            """
            SELECT COUNT(*) FROM predictions
            WHERE user_id = ? AND confidence < ?
            """,
            (user_id, CONFIDENCE_THRESHOLD),
        ).fetchone()[0]
    else:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        by_sentiment = conn.execute(
            "SELECT sentiment, COUNT(*) as count FROM predictions GROUP BY sentiment"
        ).fetchall()
        by_emotion = conn.execute(
            "SELECT emotion, COUNT(*) as count FROM predictions GROUP BY emotion"
        ).fetchall()
        avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
        low_conf = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE confidence < ?",
            (CONFIDENCE_THRESHOLD,),
        ).fetchone()[0]

    user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return {
        "total": total,
        "by_sentiment": {r["sentiment"]: r["count"] for r in by_sentiment},
        "by_emotion": {r["emotion"]: r["count"] for r in by_emotion},
        "avg_confidence": round(avg_conf or 0, 2),
        "low_confidence_count": low_conf,
        "user_count": user_count,
    }


def export_predictions_csv(sentiment=None, search=None, user_id=None):
    rows = get_predictions_filtered(
        sentiment=sentiment, search=search, user_id=user_id, limit=5000
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "ID",
            "User",
            "Filename",
            "Emotion",
            "Sentiment",
            "Confidence (%)",
            "Date (UTC)",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r["id"],
                r.get("owner_username") or "—",
                r["filename"],
                r["emotion"],
                r["sentiment"],
                round(r["confidence"], 2),
                r["created_at"],
            ]
        )
    return output.getvalue()
