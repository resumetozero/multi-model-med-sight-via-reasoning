import os
import sys
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv, find_dotenv

# Ensure the system path includes the project root for local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processed import generate_embeddings, split_docs, load_and_process_data
from data.rocov2_data import load_rocov2_embeddings

load_dotenv(find_dotenv())

# --- CONFIGURATION ---
COLLECTION_NAME = "medical_multimodal"
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Connection with robust timeout for cloud environments
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=180
)

def extract_clinical_metadata(text):
    """
    Standardizes unstructured captions into searchable metadata.
    This bridges the gap between Indiana's structured data and ROCO's short text.
    """
    text = text.lower()
    meta = {"modality": "Other", "anatomy": "General"}
    
    # Modality Logic
    if any(x in text for x in ["ct", "computed tomography"]): meta["modality"] = "CT"
    elif any(x in text for x in ["x-ray", "radiograph", "radiography"]): meta["modality"] = "X-ray"
    elif any(x in text for x in ["mri", "magnetic resonance", "nmr"]): meta["modality"] = "MRI"
    elif any(x in text for x in ["ultrasound", "us ", "sonography", "echo"]): meta["modality"] = "Ultrasound"
    
    # Anatomy Logic
    if any(x in text for x in ["chest", "lung", "pleural", "thorax", "rib", "heart"]): meta["anatomy"] = "Chest"
    elif any(x in text for x in ["head", "brain", "skull", "cth", "neck"]): meta["anatomy"] = "Head"
    elif any(x in text for x in ["abdomen", "pelvis", "liver", "renal", "kidney"]): meta["anatomy"] = "Abdomen"
    elif any(x in text for x in ["bone", "fracture", "spine", "leg", "arm"]): meta["anatomy"] = "Musculoskeletal"
    
    return meta

# 1. DATA PREPARATION (Using BiomedCLIP-consistent Embeddings)
# -----------------------------------------------------------

# Load Indiana (Detailed Reports)
INDIANA_CACHE = os.path.join(CACHE_DIR, "biomed_embeddings_cache.pkl")
if os.path.exists(INDIANA_CACHE):
    print("✓ Loading Indiana embeddings from cache...")
    indiana_df = pd.read_pickle(INDIANA_CACHE)
else:
    print("! Generating Indiana embeddings (BiomedCLIP)...")
    raw_data = load_and_process_data()
    chunks = split_docs(raw_data, chunk_size=800, chunk_overlap=50)
    indiana_df = generate_embeddings(chunks, device="cpu", batch_size=4)
    indiana_df.to_pickle(INDIANA_CACHE)

# Load ROCOv2 (Captions)
ROCO_CACHE = os.path.join(CACHE_DIR, "rocov2_embeddings.pkl")
if os.path.exists(ROCO_CACHE):
    print("✓ Loading ROCOv2 embeddings from cache...")
    roco_data = pd.read_pickle(ROCO_CACHE)
else:
    print("! Generating ROCOv2 embeddings (BiomedCLIP)...")
    roco_data = load_rocov2_embeddings(device="cpu", batch_size=8)
    pd.to_pickle(roco_data, ROCO_CACHE)

# 2. COLLECTION SETUP & INDEXING
# ------------------------------

vector_size = 512  # Standard for BiomedCLIP PubMedBERT-ViT-B16

if not client.collection_exists(COLLECTION_NAME):
    print(f"Creating unified collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )
    
    # CRITICAL: Create Payload Indexes for fast medical filtering
    print("Creating payload indexes for modality and anatomy...")
    client.create_payload_index(COLLECTION_NAME, "modality", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "anatomy", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "dataset", models.PayloadSchemaType.KEYWORD)

# 3. UNIFIED INGESTION ENGINE
# ---------------------------

def ingest_data():
    all_points = []
    
    # Process Indiana Points
    for _, row in tqdm(indiana_df.iterrows(), total=len(indiana_df), desc="Processing Indiana"):
        vec = np.asarray(row["embedding"], dtype=np.float32).flatten().tolist()
        if np.isnan(vec).any(): continue
        
        all_points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "dataset": "indiana",
                "text": row["text"],
                "image_ref": row["image_path"],
                "image_type": "local_path",
                "modality": "X-ray",
                "anatomy": "Chest",
                "findings": row.get("pathology", "unspecified")
            }
        ))

    # Process ROCOv2 Points
    for i in range(len(roco_data["embeddings"])):
        caption = roco_data["captions"][i]
        meta = extract_clinical_metadata(caption)
        vec = roco_data["embeddings"][i].tolist()
        
        all_points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "dataset": "rocov2",
                "text": caption,
                "image_ref": str(roco_data["image_ids"][i]),
                "image_type": "huggingface_id",
                "modality": meta["modality"],
                "anatomy": meta["anatomy"],
                "findings": "unspecified"
            }
        ))

    # Batch Upload
    batch_size = 100
    for i in tqdm(range(0, len(all_points), batch_size), desc="Uploading to Qdrant"):
        batch = all_points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

if __name__ == "__main__":
    ingest_data()
    print("\n✅ Unified Medical Vector DB Ready.")
    # print(f"Collection: {COLLECTION_NAME}")
    print("Metadata Searchable via: dataset, modality, anatomy, findings")