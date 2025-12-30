# database.py
import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "farm_data.db"

def init_db():
    """Creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            batch_name TEXT,
            substrate_type TEXT,
            strain_id TEXT,
            predicted_yield REAL,
            issue_prob REAL,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_batch(data):
    """Saves a new batch entry."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO batches (timestamp, batch_name, substrate_type, strain_id, predicted_yield, issue_prob, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        data['batch_name'],
        data['substrate_type'],
        data['strain_id'],
        data['pred_yield'],
        data['pred_prob'],
        "Active"
    ))
    conn.commit()
    conn.close()

def get_all_batches():
    """Returns a DataFrame of all history."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM batches ORDER BY id DESC", conn)
    conn.close()
    return df