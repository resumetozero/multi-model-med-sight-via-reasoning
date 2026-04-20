
"""
qdrant_vdata.py  –  Med-Sight | Unified Medical Vector DB Ingestion
====================================================================
Ingests three data sources into a single Qdrant collection:

  dataset = "indiana"          Indiana University chest X-ray reports
  dataset = "rocov2"           ROCO v2 radiology image captions
  dataset = "personal_report"  Patient-uploaded PDF reports  (→ docs_upload.py)
  dataset = "personal_scan"    Patient-uploaded scan images  (→ images_upload.py)

The personal datasets are written by docs_upload.py and images_upload.py
at upload time; this file handles the bulk re-index of the base corpora
and exposes helper functions for cross-dataset querying.
"""

import os
import sys
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
from qdrant_client import models
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.metadata import extract_clinical_metadata
from data.qdrant_utils import (
    delete_points_by_field,
    ensure_collection,
    get_qdrant_client,
)
from data.processed_IU import generate_embeddings, split_docs, load_and_process_data
from data.rocov2_data import load_rocov2_embeddings

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "medical_multimodal"
PATIENT_COLLECTION = "patient_reports"

CACHE_DIR       = "data/cache"
VECTOR_SIZE     = 512   # BiomedCLIP PubMedBERT-ViT-B16
os.makedirs(CACHE_DIR, exist_ok=True)

# All valid dataset tags in the collection
VALID_DATASETS = frozenset({
    "indiana",
    "rocov2",
    "personal_report",
    "personal_scan",
})


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION SETUP
# ─────────────────────────────────────────────────────────────────────────────

def ensure_collections() -> None:
    client = get_qdrant_client()
    ensure_collection(client, COLLECTION_NAME, VECTOR_SIZE)
    ensure_collection(client, PATIENT_COLLECTION, VECTOR_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION  (base corpora)
# ─────────────────────────────────────────────────────────────────────────────

INDIANA_CACHE = os.path.join(CACHE_DIR, "biomed_embeddings_cache.pkl")
ROCO_CACHE = os.path.join(CACHE_DIR, "rocov2_embeddings.pkl")


def _load_or_build_cache(cache_path: str, loader, dataset_name: str):
    if os.path.exists(cache_path):
        print(f"✓ Loading {dataset_name} from cache …")
        return pd.read_pickle(cache_path)

    print(f"! Generating {dataset_name} embeddings (BiomedCLIP) …")
    data = loader()
    pd.to_pickle(data, cache_path)
    return data


def load_indiana_data():
    return _load_or_build_cache(
        INDIANA_CACHE,
        lambda: generate_embeddings(
            split_docs(load_and_process_data(), chunk_size=800, chunk_overlap=50),
            device="cpu",
            batch_size=4,
        ),
        "Indiana",
    )


def load_rocov2_data():
    return _load_or_build_cache(
        ROCO_CACHE,
        lambda: load_rocov2_embeddings(device="cpu", batch_size=8),
        "ROCOv2",
    )


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION ENGINE  (base corpora only)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_data(
    indiana_df=None,
    roco_data=None,
) -> None:
    """
    Bulk-ingest Indiana + ROCOv2 points into the public medical corpus.
    Personal data is managed by the patient ingestion pipelines.
    """
    ensure_collections()
    client = get_qdrant_client()

    indiana_df = indiana_df if indiana_df is not None else load_indiana_data()
    roco_data = roco_data if roco_data is not None else load_rocov2_data()

    all_points = []

    # ── Indiana (detailed X-ray reports) ─────────────────────────────────
    for _, row in tqdm(indiana_df.iterrows(), total=len(indiana_df), desc="Processing Indiana"):
        vec = np.asarray(row["embedding"], dtype=np.float32).flatten().tolist()
        if np.isnan(vec).any():
            continue

        all_points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "dataset":    "indiana",
                "patient_id": "",
                "text":       row["text"],
                "image_ref":  row["image_path"],
                "image_type": "local_path",
                "modality":   "X-ray",
                "anatomy":    "Chest",
                "findings":   row.get("pathology", "unspecified"),
            },
        ))

    # ── ROCOv2 (diverse radiology captions) ──────────────────────────────
    for i in range(len(roco_data["embeddings"])):
        caption = roco_data["captions"][i]
        meta = extract_clinical_metadata(caption)
        vec = np.asarray(roco_data["embeddings"][i], dtype=np.float32).flatten().tolist()
        if np.isnan(vec).any():
            continue

        all_points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "dataset":    "rocov2",
                "patient_id": "",
                "text":       caption,
                "image_ref":  str(roco_data["image_ids"][i]),
                "image_type": "huggingface_id",
                "modality":   meta["modality"],
                "anatomy":    meta["anatomy"],
                "findings":   "unspecified",
            },
        ))

    batch_size = 100
    for i in tqdm(range(0, len(all_points), batch_size), desc="Uploading to Qdrant"):
        batch = all_points[i: i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)


# ─────────────────────────────────────────────────────────────────────────────
# QUERY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def search_collection(
    query_vector:  list[float],
    top_k:         int = 10,
    modality_filter: Optional[str] = None,
    anatomy_filter:  Optional[str] = None,
    patient_id: Optional[str] = None,
) -> dict:
    """
    Semantic search over Qdrant collections with optional metadata filters.

    Parameters
    ----------
    query_vector     : 512-d BiomedCLIP embedding of the query
    top_k            : number of results to return
    modality_filter  : e.g. "CT", "X-ray", "MRI"
    anatomy_filter   : e.g. "Chest", "Head"
    patient_id       : optional patient identifier for personal data search
    """

    client = get_qdrant_client()
    shared_filters = []

    if modality_filter:
        shared_filters.append(
            models.FieldCondition(
                key="modality",
                match=models.MatchValue(value=modality_filter),
            )
        )
    if anatomy_filter:
        shared_filters.append(
            models.FieldCondition(
                key="anatomy",
                match=models.MatchValue(value=anatomy_filter),
            )
        )

    personal_results = []
    if patient_id:
        patient_must = shared_filters + [
            models.FieldCondition(
                key="patient_id",
                match=models.MatchValue(value=patient_id),
            )
        ]
        personal_results = client.search(
            collection_name=PATIENT_COLLECTION,
            query_vector=query_vector,
            query_filter=models.Filter(must=patient_must),
            limit=top_k,
            with_payload=True,
        )

    global_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=models.Filter(must=shared_filters) if shared_filters else None,
        limit=top_k,
        with_payload=True,
    )

    return {
        "personal_history": personal_results,
        "medical_reference": global_results,
    }


def delete_patient_data(patient_id: str) -> None:
    """
    Remove all personal scan data points for a given patient.
    """
    client = get_qdrant_client()
    delete_points_by_field(client, PATIENT_COLLECTION, "patient_id", patient_id)
    print(f"Deleted personal data for patient_id={patient_id!r}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingest_data()
    print("\n✅ Unified Medical Vector DB Ready.")
    print(f"Collection : {COLLECTION_NAME}")
    print("Datasets   : indiana | rocov2 | personal_report | personal_scan")
    print("Filterable : dataset, modality, anatomy, patient_id, findings")