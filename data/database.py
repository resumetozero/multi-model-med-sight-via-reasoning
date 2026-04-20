import os
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional


def connect_sqlite(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def already_ingested(
    conn: sqlite3.Connection,
    table_name: str,
    file_hash: str,
) -> Optional[str]:
    row = conn.execute(
        f"SELECT id FROM {table_name} WHERE file_hash = ?",
        (file_hash,),
    ).fetchone()
    return row["id"] if row else None


def initialize_reports_store(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reports (
            id            TEXT PRIMARY KEY,
            patient_id    TEXT,
            filename      TEXT NOT NULL,
            file_hash     TEXT NOT NULL UNIQUE,
            upload_ts     TEXT NOT NULL,
            page_count    INTEGER,
            raw_text      TEXT,
            modality      TEXT DEFAULT 'Unknown',
            anatomy       TEXT DEFAULT 'General',
            summary       TEXT
        );

        CREATE TABLE IF NOT EXISTS report_chunks (
            id            TEXT PRIMARY KEY,
            report_id     TEXT NOT NULL REFERENCES reports(id),
            chunk_index   INTEGER NOT NULL,
            chunk_text    TEXT NOT NULL,
            qdrant_id     TEXT,
            UNIQUE(report_id, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS report_tables (
            id            TEXT PRIMARY KEY,
            report_id     TEXT NOT NULL REFERENCES reports(id),
            page_number   INTEGER,
            table_index   INTEGER,
            table_text    TEXT
        );
    """)
    conn.commit()


def initialize_scans_store(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scans (
            id            TEXT PRIMARY KEY,
            patient_id    TEXT,
            filename      TEXT NOT NULL,
            file_hash     TEXT NOT NULL UNIQUE,
            upload_ts     TEXT NOT NULL,
            width         INTEGER,
            height        INTEGER,
            color_mode    TEXT,
            modality      TEXT DEFAULT 'Unknown',
            anatomy       TEXT DEFAULT 'General',
            caption       TEXT,
            qdrant_id     TEXT
        );
    """)
    conn.commit()
