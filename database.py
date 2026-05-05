"""
database.py — SQLite Database Setup for Alzheimer's Detection System
Handles user authentication storage with secure password hashing.
"""

import sqlite3
import os
from datetime import datetime

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alzheimer.db')


def get_db():
    """Get a database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_db()
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create predictions table to track user predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_filename TEXT NOT NULL,
            model_used TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"[OK] Database initialized at {DB_PATH}")


def add_user(full_name, email, password_hash):
    """Add a new user to the database. Returns True on success, False if email exists."""
    conn = get_db()
    try:
        conn.execute(
            'INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)',
            (full_name, email, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_email(email):
    """Retrieve a user by email address."""
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    return user


def add_prediction(user_id, image_filename, model_used, predicted_class, confidence):
    """Log a prediction to the database."""
    conn = get_db()
    conn.execute(
        'INSERT INTO predictions (user_id, image_filename, model_used, predicted_class, confidence) VALUES (?, ?, ?, ?, ?)',
        (user_id, image_filename, model_used, predicted_class, confidence)
    )
    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
