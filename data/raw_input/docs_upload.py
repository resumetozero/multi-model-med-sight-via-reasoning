import os
import sys
import uuid
import sqlite3
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient, models

from data.database import (
    already_ingested,
    compute_file_hash,
    connect_sqlite,
    initialize_reports_store,
)
from data.pdf_utils import parse_pdf, chunk_text as pdf_chunk_text
from data.metadata import extract_report_metadata
from data.qdrant_utils import get_qdrant_client, ensure_collection
from data.processed_IU import generate_embeddings  # returns DataFrame with 'embedding' col

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "medical_reports"
DB_PATH         = os.getenv("MEDSIGHT_DB_PATH", "data/database/medsight_personal.db")
CHUNK_SIZE      = 600   # characters per text chunk
CHUNK_OVERLAP   = 80
VECTOR_SIZE     = 512   # BiomedCLIP PubMedBERT-ViT-B16

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("docs_upload")


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE  –  Local Personalization Store
# ─────────────────────────────────────────────────────────────────────────────

def _get_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = connect_sqlite(db_path)
    initialize_reports_store(conn)
    return conn


def _file_hash(path: str) -> str:
    return compute_file_hash(path)


def _already_ingested(conn: sqlite3.Connection, file_hash: str) -> Optional[str]:
    return already_ingested(conn, "reports", file_hash)


# ─────────────────────────────────────────────────────────────────────────────
# METADATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_report_metadata(text: str) -> dict:
    """Infer modality, anatomy, and summary from free-text report content."""
    t = text.lower()
    meta = {"modality": "Unknown", "anatomy": "General", "summary": ""}

    # Modality detection
    if any(x in t for x in ["computed tomography", " ct ", "ct scan", "ct:"]):
        meta["modality"] = "CT"
    elif any(x in t for x in ["magnetic resonance", " mri ", "mri:", "nmr"]):
        meta["modality"] = "MRI"
    elif any(x in t for x in ["x-ray", "radiograph", "chest pa", "ap view"]):
        meta["modality"] = "X-ray"
    elif any(x in t for x in ["ultrasound", "sonography", "echography", " us "]):
        meta["modality"] = "Ultrasound"
    elif any(x in t for x in ["pet scan", "positron emission", "nuclear medicine"]):
        meta["modality"] = "PET"
    elif any(x in t for x in ["histopathology", "biopsy", "cytology", "pathology"]):
        meta["modality"] = "Pathology"

    # Anatomy detection
    if any(x in t for x in ["chest", "lung", "pulmonary", "pleural", "thorax",
                              "bronchi", "trachea", "cardiac", "heart"]):
        meta["anatomy"] = "Chest"
    elif any(x in t for x in ["brain", "cerebral", "cranial", "skull", "head",
                                "intracranial", "meninges", "ventricle"]):
        meta["anatomy"] = "Head"
    elif any(x in t for x in ["abdomen", "liver", "hepatic", "renal", "kidney",
                                "spleen", "pancreas", "gallbladder", "pelvis",
                                "bowel", "colon", "rectum"]):
        meta["anatomy"] = "Abdomen"
    elif any(x in t for x in ["spine", "vertebr", "disc", "lumbar", "cervical",
                                "bone", "fracture", "joint", "femur", "tibia",
                                "humer", "shoulder", "knee", "hip"]):
        meta["anatomy"] = "Musculoskeletal"
    elif any(x in t for x in ["breast", "mammograph"]):
        meta["anatomy"] = "Breast"
    elif any(x in t for x in ["prostate", "uterus", "ovarian", "testicular"]):
        meta["anatomy"] = "Pelvis"

    # Crude summary: first meaningful line
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) > 20:
            meta["summary"] = stripped[:200]
            break

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# QDRANT  –  Embedding + Upsert
# ─────────────────────────────────────────────────────────────────────────────

def _get_qdrant_client() -> QdrantClient:
    return get_qdrant_client()


def _ensure_collection(client: QdrantClient):
    ensure_collection(client, COLLECTION_NAME, VECTOR_SIZE)


def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Generate BiomedCLIP text embeddings via project's generate_embeddings()."""
    import pandas as pd
    df_in  = pd.DataFrame({"text": chunks, "image_path": [""] * len(chunks)})
    df_out = generate_embeddings(df_in, device="cpu", batch_size=4)
    return [
        np.asarray(v, dtype=np.float32).flatten().tolist()
        for v in df_out["embedding"].tolist()
    ]


def _upsert_to_qdrant(
    client:     QdrantClient,
    chunks:     list[str],
    embeddings: list[list[float]],
    report_id:  str,
    patient_id: str,
    filename:   str,
    modality:   str,
    anatomy:    str,
) -> list[str]:
    """Upsert chunk embeddings into Qdrant; return list of point IDs."""
    points, point_ids = [], []
    for chunk_content, vec in zip(chunks, embeddings):
        if any(np.isnan(vec)):
            continue
        pid = str(uuid.uuid4())
        point_ids.append(pid)
        points.append(models.PointStruct(
            id=pid,
            vector=vec,
            payload={
                "dataset":    "personal_report",
                "patient_id": patient_id,
                "report_id":  report_id,
                "filename":   filename,
                "text":       chunk_content,
                "modality":   modality,
                "anatomy":    anatomy,
                "image_ref":  "",
                "image_type": "none",
                "findings":   "unspecified",
            },
        ))

    batch_size = 50
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i: i + batch_size])

    log.info("Upserted %d chunk vectors to Qdrant (report=%s)", len(points), report_id)
    return point_ids


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def ingest_report(
    pdf_path:    str,
    patient_id:  str = "anonymous",
    db_path:     str = DB_PATH,
    skip_qdrant: bool = False,
) -> dict:
    """
    Full pipeline: parse PDF → store in SQLite → embed chunks → upsert to Qdrant.

    Parameters
    ----------
    pdf_path    : absolute or relative path to the PDF file
    patient_id  : patient identifier (used for personalised Qdrant filtering)
    db_path     : path to the local SQLite personalisation database
    skip_qdrant : if True, persist locally only (useful for offline/testing mode)

    Returns
    -------
    dict: report_id, patient_id, filename, page_count,
          modality, anatomy, chunk_count, qdrant_ids, status
    """
    pdf_path = str(Path(pdf_path).resolve())
    filename = Path(pdf_path).name

    # Deduplication guard
    fhash = _file_hash(pdf_path)
    conn  = _get_db(db_path)

    existing_id = _already_ingested(conn, fhash)
    if existing_id:
        log.info("Already ingested (id=%s) – skipping.", existing_id)
        conn.close()
        return {"report_id": existing_id, "status": "already_ingested"}

    # Parse
    parsed    = parse_pdf(pdf_path)
    full_text = parsed["full_text"]
    meta      = extract_report_metadata(full_text)
    report_id = str(uuid.uuid4())
    now       = datetime.utcnow().isoformat()

    # Persist: reports row
    conn.execute(
        """INSERT INTO reports
           (id, patient_id, filename, file_hash, upload_ts,
            page_count, raw_text, modality, anatomy, summary)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (report_id, patient_id, filename, fhash, now,
         parsed["page_count"], full_text,
         meta["modality"], meta["anatomy"], meta["summary"]),
    )

    # Persist: tables
    for tbl in parsed["tables"]:
        conn.execute(
            """INSERT INTO report_tables
               (id, report_id, page_number, table_index, table_text)
               VALUES (?,?,?,?,?)""",
            (str(uuid.uuid4()), report_id, tbl["page"], tbl["index"], tbl["text"]),
        )

    # Chunk → embed → Qdrant
    chunker = pdf_chunk_text
    chunks  = chunker(full_text)
    qdrant_ids: list[str] = []

    if not skip_qdrant and chunks:
        try:
            log.info("Embedding %d chunks …", len(chunks))
            embeddings = _embed_chunks(chunks)
            qclient    = _get_qdrant_client()
            _ensure_collection(qclient)
            qdrant_ids = _upsert_to_qdrant(
                qclient, chunks, embeddings,
                report_id, patient_id, filename,
                meta["modality"], meta["anatomy"],
            )
        except Exception as exc:
            log.warning("Qdrant upsert failed (local-only): %s", exc)

    # Persist: chunks
    pad_ids = qdrant_ids + [""] * max(0, len(chunks) - len(qdrant_ids))
    for idx, (chunk_content, qid) in enumerate(zip(chunks, pad_ids)):
        conn.execute(
            """INSERT OR IGNORE INTO report_chunks
               (id, report_id, chunk_index, chunk_text, qdrant_id)
               VALUES (?,?,?,?,?)""",
            (str(uuid.uuid4()), report_id, idx, chunk_content, qid),
        )

    conn.commit()
    conn.close()

    result = {
        "report_id":   report_id,
        "patient_id":  patient_id,
        "filename":    filename,
        "page_count":  parsed["page_count"],
        "modality":    meta["modality"],
        "anatomy":     meta["anatomy"],
        "chunk_count": len(chunks),
        "qdrant_ids":  qdrant_ids,
        "status":      "success",
    }
    log.info("✓ Report ingested → %s", result)
    return result


def ingest_reports_bulk(
    pdf_paths:   list[str],
    patient_id:  str = "anonymous",
    db_path:     str = DB_PATH,
    skip_qdrant: bool = False,
) -> list[dict]:
    """Ingest multiple PDFs; returns list of result dicts."""
    return [
        ingest_report(p, patient_id=patient_id,
                      db_path=db_path, skip_qdrant=skip_qdrant)
        for p in pdf_paths
    ]


def query_local_reports(
    patient_id: Optional[str] = None,
    modality:   Optional[str] = None,
    anatomy:    Optional[str] = None,
    db_path:    str = DB_PATH,
) -> list[dict]:
    """
    Query the local SQLite store for ingested reports.
    All filters are optional and ANDed together.
    """
    conn   = _get_db(db_path)
    where  = []
    params = []
    if patient_id:
        where.append("patient_id = ?"); params.append(patient_id)
    if modality:
        where.append("modality = ?");   params.append(modality)
    if anatomy:
        where.append("anatomy = ?");    params.append(anatomy)

    sql = "SELECT * FROM reports"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY upload_ts DESC"

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Med-Sight | Ingest clinical PDF reports"
    )
    parser.add_argument("pdfs",         nargs="+",            help="PDF file path(s)")
    parser.add_argument("--patient-id", default="anonymous",  help="Patient identifier")
    parser.add_argument("--db",         default=DB_PATH,      help="SQLite DB path")
    parser.add_argument("--no-qdrant",  action="store_true",  help="Skip Qdrant upsert")
    args = parser.parse_args()

    for pdf in args.pdfs:
        res = ingest_report(
            pdf,
            patient_id=args.patient_id,
            db_path=args.db,
            skip_qdrant=args.no_qdrant,
        )
        print(res)