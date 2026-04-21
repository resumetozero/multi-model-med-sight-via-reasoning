"""
images_upload.py  –  Med-Sight | Medical Scan Image Ingestion Pipeline
======================================================================
Accepts PNG / JPG / JPEG medical scan images (X-ray, CT, MRI, etc.),
runs BiomedCLIP vision embeddings, extracts DICOM-style metadata where
possible, stores records in a local SQLite database, and upserts into
the shared Qdrant collection as dataset="personal_scan".

Enhanced with disease analysis: queries Qdrant for semantically related
reports and generates descriptive disease analysis with image-report
correlation scoring.

Flow
----
  Image file(s)  (PNG / JPG / JPEG)
      └─► PIL / OpenCV  (decode + normalise)
      └─► EXIF / filename heuristics  (modality + anatomy metadata)
      └─► BiomedCLIP ViT-B/16  (512-d vision embedding)
              └─► SQLite  (local personalization store)
              └─► Qdrant  (dataset = "personal_scan")
              └─► Disease analysis  (infer conditions, find related reports)

Usage
-----
  # Basic ingestion
  from data.raw_input import ingest_scan
  result = ingest_scan("chest_xray.png", patient_id="P001", caption="PA chest X-ray")

  # With disease analysis
  from data.raw_input import ingest_scan_with_analysis
  result = ingest_scan_with_analysis("chest_xray.png", patient_id="P001")
  print(result["analysis"]["composite_analysis"])

  # from CLI
  python images_upload.py scan1.jpg scan2.png --patient-id P001
"""

import os
import sys
import io
import uuid
import sqlite3
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ExifTags

# ── Qdrant ────────────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient, models

# ── BiomedCLIP ────────────────────────────────────────────────────────────────
# Uses the same model as the rest of the pipeline (open_clip / transformers)
import torch
import open_clip

# ── Project internals ─────────────────────────────────────────────────────────
from data.database import (
    already_ingested,
    compute_file_hash,
    connect_sqlite,
    initialize_scans_store,
)
from data.metadata import extract_scan_metadata
from data.image_utils import preprocess_medical_image
from data.qdrant_utils import get_qdrant_client, ensure_collection
from data.disease_analysis import analyze_image_with_reports

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_COLLECTION = "patient_reports"


DB_PATH           = "data/database/medsight_personal.db"
VECTOR_SIZE       = 512   # BiomedCLIP ViT-B/16 image embedding dim
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg"}

BIOMEDCLIP_MODEL  = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("images_upload")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  –  BiomedCLIP (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: dict = {}


def _load_biomedclip(device: str = "cpu"):
    """Load BiomedCLIP model + preprocessor (cached after first call)."""
    if "model" not in _model_cache:
        log.info("Loading BiomedCLIP model …")
        model, _, preprocess = open_clip.create_model_and_transforms(
            BIOMEDCLIP_MODEL
        )
        model.eval().to(device)
        _model_cache["model"]     = model
        _model_cache["preprocess"] = preprocess
        _model_cache["device"]    = device
    return _model_cache["model"], _model_cache["preprocess"], _model_cache["device"]


def embed_image(image: Image.Image, device: str = "cpu") -> list[float]:
    """
    Generate a 512-d BiomedCLIP vision embedding for a PIL Image.
    The image is preprocessed (resize, normalise) before encoding.
    """
    model, preprocess, dev = _load_biomedclip(device)

    # Convert to RGB (handles greyscale DICOM-style PNG, RGBA, etc.)
    rgb = image.convert("RGB")

    tensor = preprocess(rgb).unsqueeze(0).to(dev)   # [1, 3, 224, 224]
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalise

    return features.squeeze(0).cpu().numpy().astype(np.float32).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE  –  Local Personalization Store
# ─────────────────────────────────────────────────────────────────────────────

def _get_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = connect_sqlite(db_path)
    initialize_scans_store(conn)
    return conn


def _file_hash(path: str) -> str:
    return compute_file_hash(path)


def _already_ingested(conn: sqlite3.Connection, file_hash: str) -> Optional[str]:
    return already_ingested(conn, "scans", file_hash)


# ─────────────────────────────────────────────────────────────────────────────
# METADATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_scan_metadata(
    image:    Image.Image,
    filename: str,
    caption:  str = "",
) -> dict:
    """
    Infer modality and anatomy from:
      1. Explicit caption / description text
      2. Filename heuristics (e.g. "chest_xray_pa.png")
      3. Image properties (greyscale → likely X-ray / CT / MRI)
    """
    text = (caption + " " + filename).lower()
    meta = {
        "modality":   "Unknown",
        "anatomy":    "General",
        "color_mode": image.mode,
        "width":      image.width,
        "height":     image.height,
    }

    # ── Modality ──────────────────────────────────────────────────────────
    if any(x in text for x in ["ct", "computed", "tomography"]):
        meta["modality"] = "CT"
    elif any(x in text for x in ["mri", "magnetic", "resonance", "t1", "t2", "flair"]):
        meta["modality"] = "MRI"
    elif any(x in text for x in ["xray", "x-ray", "x_ray", "radiograph",
                                   "chest pa", "ap view", "cxr"]):
        meta["modality"] = "X-ray"
    elif any(x in text for x in ["ultrasound", "us_", "_us.", "echo", "sonograph"]):
        meta["modality"] = "Ultrasound"
    elif any(x in text for x in ["pet", "nuclear"]):
        meta["modality"] = "PET"
    elif image.mode in ("L", "I"):
        # Greyscale → most likely X-ray or CT if no other hint
        meta["modality"] = "X-ray"

    # ── Anatomy ───────────────────────────────────────────────────────────
    if any(x in text for x in ["chest", "lung", "thorax", "cxr", "pleural",
                                 "pulmonary", "cardiac", "heart"]):
        meta["anatomy"] = "Chest"
    elif any(x in text for x in ["brain", "head", "skull", "cranial",
                                   "cerebral", "neuro"]):
        meta["anatomy"] = "Head"
    elif any(x in text for x in ["abdomen", "abdom", "liver", "renal",
                                   "kidney", "spleen", "pelvis", "bowel"]):
        meta["anatomy"] = "Abdomen"
    elif any(x in text for x in ["spine", "vertebr", "lumbar", "cervical",
                                   "bone", "fracture", "knee", "hip",
                                   "shoulder", "femur", "tibia"]):
        meta["anatomy"] = "Musculoskeletal"
    elif any(x in text for x in ["breast", "mammo"]):
        meta["anatomy"] = "Breast"

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_medical_image(image: Image.Image) -> Image.Image:
    """
    Normalise raw medical scan images for BiomedCLIP:
    - Convert 16-bit greyscale (common in exported DICOMs) to 8-bit
    - Ensure consistent colour mode before further transforms
    """
    # 16-bit → 8-bit rescale
    if image.mode == "I":
        arr = np.array(image, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        image = Image.fromarray(arr.astype(np.uint8), mode="L")

    # Keep greyscale; BiomedCLIP preprocess handles RGB conversion internally
    return image


# ─────────────────────────────────────────────────────────────────────────────
# QDRANT  –  Upsert
# ─────────────────────────────────────────────────────────────────────────────

def _get_qdrant_client() -> QdrantClient:
    return get_qdrant_client()


def _ensure_collection(client: QdrantClient):
    ensure_collection(client, PATIENT_COLLECTION, VECTOR_SIZE)


def _upsert_scan_to_qdrant(
    client:     QdrantClient,
    vec:        list[float],
    scan_id:    str,
    patient_id: str,
    filename:   str,
    caption:    str,
    modality:   str,
    anatomy:    str,
    image_path: str,
    analysis_json: str = "",
) -> str:
    """Upsert a single scan embedding; returns the Qdrant point ID."""
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=PATIENT_COLLECTION,
        points=[models.PointStruct(
            id=point_id,
            vector=vec,
            payload={
                "dataset":    "personal_scan",
                "patient_id": patient_id,
                "scan_id":    scan_id,
                "filename":   filename,
                "text":       caption or filename,
                "modality":   modality,
                "anatomy":    anatomy,
                "image_ref":  image_path,
                "image_type": "local_path",
                "findings":   "Analysis pending",
                "analysis":   analysis_json,
            },
        )],
    )
    log.info("Upserted scan vector to Qdrant (scan=%s, point=%s)", scan_id, point_id)
    return point_id


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API – INGESTION ONLY
# ─────────────────────────────────────────────────────────────────────────────

def ingest_scan(
    image_path:  str,
    patient_id:  str = "anonymous",
    caption:     str = "",
    device:      str = "cpu",
    db_path:     str = DB_PATH,
    skip_qdrant: bool = False,
) -> dict:
    """
    Full pipeline: load image → extract metadata → embed → store locally → upsert.

    Parameters
    ----------
    image_path  : path to PNG / JPG / JPEG scan file
    patient_id  : patient identifier string
    caption     : optional free-text description (improves metadata inference)
    device      : torch device string ("cpu" or "cuda")
    db_path     : local SQLite personalisation DB path
    skip_qdrant : if True, skip the Qdrant upsert (offline / test mode)

    Returns
    -------
    dict: scan_id, patient_id, filename, modality, anatomy,
          width, height, qdrant_id, status
    """
    image_path = str(Path(image_path).resolve())
    filename   = Path(image_path).name
    ext        = Path(image_path).suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {SUPPORTED_FORMATS}"
        )

    # Deduplication guard
    fhash = _file_hash(image_path)
    conn  = _get_db(db_path)
    existing_id = _already_ingested(conn, fhash)
    if existing_id:
        log.info("Scan already ingested (id=%s) – skipping.", existing_id)
        conn.close()
        return {"scan_id": existing_id, "status": "already_ingested"}

    # Load + preprocess
    log.info("Loading image: %s", image_path)
    raw_image  = Image.open(image_path)
    image      = preprocess_medical_image(raw_image)

    # Metadata
    meta    = extract_scan_metadata(image, filename, caption)
    scan_id = str(uuid.uuid4())
    now     = datetime.utcnow().isoformat()

    # BiomedCLIP vision embedding
    log.info("Embedding scan with BiomedCLIP …")
    vec = embed_image(image, device=device)

    # Qdrant upsert
    qdrant_id = ""
    if not skip_qdrant:
        try:
            qclient = _get_qdrant_client()
            _ensure_collection(qclient)
            qdrant_id = _upsert_scan_to_qdrant(
                qclient, vec, scan_id, patient_id,
                filename, caption, meta["modality"], meta["anatomy"], image_path,
            )
        except Exception as exc:
            log.warning("Qdrant upsert failed (local-only): %s", exc)

    # SQLite persist
    conn.execute(
        """INSERT INTO scans
           (id, patient_id, filename, file_hash, upload_ts,
            width, height, color_mode, modality, anatomy, caption, qdrant_id, analysis)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (scan_id, patient_id, filename, fhash, now,
         meta["width"], meta["height"], meta["color_mode"],
         meta["modality"], meta["anatomy"], caption, qdrant_id, ""),
    )
    conn.commit()
    conn.close()

    result = {
        "scan_id":    scan_id,
        "patient_id": patient_id,
        "filename":   filename,
        "modality":   meta["modality"],
        "anatomy":    meta["anatomy"],
        "width":      meta["width"],
        "height":     meta["height"],
        "qdrant_id":  qdrant_id,
        "status":     "success",
    }
    log.info("✓ Scan ingested → %s", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API – INGESTION WITH DISEASE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def ingest_scan_with_analysis(
    image_path:  str,
    patient_id:  str = "anonymous",
    caption:     str = "",
    device:      str = "cpu",
    db_path:     str = DB_PATH,
    skip_qdrant: bool = False,
) -> dict:
    """
    Full pipeline with disease analysis: integrate scan, analyze for diseases,
    and correlate with patient's existing reports.

    Parameters
    ----------
    image_path  : path to PNG / JPG / JPEG scan file
    patient_id  : patient identifier string
    caption     : optional free-text description
    device      : torch device string ("cpu" or "cuda")
    db_path     : local SQLite personalisation DB path
    skip_qdrant : if True, skip Qdrant operations

    Returns
    -------
    dict containing:
        - ingestion_result: basic scan metadata (scan_id, filename, modality, anatomy)
        - analysis: DiseaseAnalysis object (diseases, related_reports, composite_analysis)
        - status: "success" if both ingestion and analysis completed
    """
    # Import here to avoid circular imports
    from data.disease_analysis import analyze_image_with_reports
    
    # First, ingest the scan normally
    ingestion = ingest_scan(
        image_path,
        patient_id=patient_id,
        caption=caption,
        device=device,
        db_path=db_path,
        skip_qdrant=skip_qdrant,
    )

    # If ingestion failed or was a duplicate, skip analysis
    if ingestion.get("status") != "success":
        return {
            "ingestion_result": ingestion,
            "analysis": None,
            "status": "skipped",
        }

    # Now perform disease analysis
    try:
        image_path_abs = str(Path(image_path).resolve())
        raw_image = Image.open(image_path_abs)
        image = preprocess_medical_image(raw_image)
        
        # Re-embed for analysis
        vec = embed_image(image, device=device)

        log.info("Starting disease analysis for scan %s", ingestion["scan_id"])
        analysis = analyze_image_with_reports(
            image_embedding=vec,
            scan_id=ingestion["scan_id"],
            patient_id=patient_id,
            modality=ingestion["modality"],
            anatomy=ingestion["anatomy"],
            image_array=np.array(image),
        )

        analysis_dict = analysis.to_dict()
        analysis_json = analysis.to_json_str()
        
        log.info("Analysis result: %d diseases detected, confidence: %.0f%%",
                 len(analysis.detected_diseases), analysis.disease_confidence * 100)

        # Update SQLite with analysis
        try:
            conn = _get_db(db_path)
            conn.execute(
                "UPDATE scans SET analysis = ? WHERE id = ?",
                (analysis_json, ingestion["scan_id"])
            )
            conn.commit()
            conn.close()
            log.info("✓ Analysis stored in SQLite (size: %d bytes)", len(analysis_json))
        except Exception as exc:
            log.warning("Failed to store analysis in SQLite: %s", exc)

        # Update Qdrant with analysis if we have a qdrant_id
        if not skip_qdrant and ingestion.get("qdrant_id"):
            try:
                qclient = _get_qdrant_client()
                qclient.update_payload(
                    collection_name=PATIENT_COLLECTION,
                    payload_diff={"analysis": analysis_json},
                    points=[ingestion["qdrant_id"]],
                )
                log.info("✓ Analysis updated in Qdrant (point: %s)", ingestion["qdrant_id"])
            except Exception as exc:
                log.warning("Failed to update Qdrant with analysis: %s", exc)

        log.info("✓ Disease analysis complete - found: %s", analysis.detected_diseases)

        return {
            "ingestion_result": ingestion,
            "analysis": analysis_dict,
            "status": "success",
        }

    except Exception as exc:
        log.warning("Disease analysis failed: %s", exc)
        return {
            "ingestion_result": ingestion,
            "analysis": None,
            "analysis_error": str(exc),
            "status": "ingestion_success_analysis_failed",
        }


def ingest_scans_bulk(
    image_paths: list[str],
    patient_id:  str = "anonymous",
    caption_map: Optional[dict[str, str]] = None,
    device:      str = "cpu",
    db_path:     str = DB_PATH,
    skip_qdrant: bool = False,
) -> list[dict]:
    """
    Ingest multiple scan images.

    Parameters
    ----------
    caption_map : optional dict mapping filename → caption string
    """
    caption_map = caption_map or {}
    return [
        ingest_scan(
            p,
            patient_id=patient_id,
            caption=caption_map.get(Path(p).name, ""),
            device=device,
            db_path=db_path,
            skip_qdrant=skip_qdrant,
        )
        for p in image_paths
    ]


def query_local_scans(
    patient_id: Optional[str] = None,
    modality:   Optional[str] = None,
    anatomy:    Optional[str] = None,
    db_path:    str = DB_PATH,
) -> list[dict]:
    """
    Query the local SQLite store for ingested scans.
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

    sql = "SELECT * FROM scans"
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
        description="Med-Sight | Ingest medical scan images (PNG/JPG/JPEG)"
    )
    parser.add_argument("images",       nargs="+",           help="Image file path(s)")
    parser.add_argument("--patient-id", default="anonymous", help="Patient identifier")
    parser.add_argument("--caption",    default="",          help="Shared caption for all images")
    parser.add_argument("--device",     default="cpu",       help="Torch device (cpu / cuda)")
    parser.add_argument("--db",         default=DB_PATH,     help="SQLite DB path")
    parser.add_argument("--no-qdrant",  action="store_true", help="Skip Qdrant upsert")
    parser.add_argument("--with-analysis", action="store_true", help="Include disease analysis")
    args = parser.parse_args()

    func = ingest_scan_with_analysis if args.with_analysis else ingest_scan

    for img in args.images:
        res = func(
            img,
            patient_id=args.patient_id,
            caption=args.caption,
            device=args.device,
            db_path=args.db,
            skip_qdrant=args.no_qdrant,
        )
        print(f"Result: {res}")


def ingest_scans_bulk(
    image_paths: list[str],
    patient_id:  str = "anonymous",
    caption_map: Optional[dict[str, str]] = None,
    device:      str = "cpu",
    db_path:     str = DB_PATH,
    skip_qdrant: bool = False,
) -> list[dict]:
    """
    Ingest multiple scan images.

    Parameters
    ----------
    caption_map : optional dict mapping filename → caption string
    """
    caption_map = caption_map or {}
    return [
        ingest_scan(
            p,
            patient_id=patient_id,
            caption=caption_map.get(Path(p).name, ""),
            device=device,
            db_path=db_path,
            skip_qdrant=skip_qdrant,
        )
        for p in image_paths
    ]


def query_local_scans(
    patient_id: Optional[str] = None,
    modality:   Optional[str] = None,
    anatomy:    Optional[str] = None,
    db_path:    str = DB_PATH,
) -> list[dict]:
    """
    Query the local SQLite store for ingested scans.
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

    sql = "SELECT * FROM scans"
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
        description="Med-Sight | Ingest medical scan images (PNG/JPG/JPEG)"
    )
    parser.add_argument("images",       nargs="+",           help="Image file path(s)")
    parser.add_argument("--patient-id", default="anonymous", help="Patient identifier")
    parser.add_argument("--caption",    default="",          help="Shared caption for all images")
    parser.add_argument("--device",     default="cpu",       help="Torch device (cpu / cuda)")
    parser.add_argument("--db",         default=DB_PATH,     help="SQLite DB path")
    parser.add_argument("--no-qdrant",  action="store_true", help="Skip Qdrant upsert")
    args = parser.parse_args()

    for img in args.images:
        res = ingest_scan(
            img,
            patient_id=args.patient_id,
            caption=args.caption,
            device=args.device,
            db_path=args.db,
            skip_qdrant=args.no_qdrant,
        )
        print(res)