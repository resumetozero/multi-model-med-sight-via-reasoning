from qdrant_client import QdrantClient, models
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
import uuid
import os, sys
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processed import generate_embeddings, split_docs, load_and_process_data
from data.rocov2_data import load_rocov2_embeddings

load_dotenv(find_dotenv())

# Ensure cache directory exists
os.makedirs("data/cache", exist_ok=True)

# Longer timeout for cloud uploads
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=180
)

# ---------------------------------------------------
# 1️⃣ Indiana Chest X-ray Embedding Cache
# ---------------------------------------------------

CACHE_FILE = "data/cache/biomed_embeddings_cache.pkl"

if os.path.exists(CACHE_FILE):
    print("Loading Chest X-ray embeddings from cache...")
    embeddings = pd.read_pickle(CACHE_FILE)
else:
    print("Generating Chest X-ray embeddings...")
    data = load_and_process_data()
    chunks = split_docs(data, chunk_size=800, chunk_overlap=50)
    embeddings = generate_embeddings(chunks, device="cpu", batch_size=4)
    embeddings.to_pickle(CACHE_FILE)
    print(f"Embeddings cached to {CACHE_FILE}")

# ---------------------------------------------------
# 2️⃣ ROCOv2 Embedding Cache
# ---------------------------------------------------

ROCO_CACHE = "data/cache/rocov2_embeddings.pkl"

if os.path.exists(ROCO_CACHE):
    print("Loading ROCOv2 embeddings from cache...")
    roco_data = pd.read_pickle(ROCO_CACHE)
else:
    print("Generating ROCOv2 embeddings (first run only)...")
    roco_data = load_rocov2_embeddings(device="cpu", batch_size=8)
    pd.to_pickle(roco_data, ROCO_CACHE)
    print(f"ROCOv2 embeddings cached to {ROCO_CACHE}")

# ---------------------------------------------------
# 3️⃣ Chest X-ray Collection
# ---------------------------------------------------

chest_collection = "chestxray_reports"

if client.collection_exists(chest_collection):
    info = client.get_collection(chest_collection)
    if info.config.params.vectors.size != 512:
        print("Vector dimension mismatch. Recreating collection...")
        client.delete_collection(chest_collection)

if not client.collection_exists(chest_collection):
    vector_size = len(embeddings.iloc[0]["embedding"])

    client.create_collection(
        collection_name=chest_collection,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

# Smaller batch avoids timeout
batch_size = 32

points = []

for _, row in tqdm(embeddings.iterrows(), total=len(embeddings), desc="Uploading Chest X-ray"):

    vector = np.asarray(row["embedding"], dtype=np.float32).flatten()

    if np.isnan(vector).any():
        continue

    points.append(
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "dataset": "indiana",
                "text": row["text"],
                "image": row["image_path"]
                # "image_id": None,
                # "uid": str(row["uid"])
            }
        )
    )

    if len(points) >= batch_size:
        client.upsert(
            collection_name=chest_collection,
            points=points,
            wait=False
        )
        points = []

# Final batch
if points:
    client.upsert(
        collection_name=chest_collection,
        points=points,
        wait=True
    )

print(f"Successfully indexed {len(embeddings)} Chest X-ray chunks")

# ---------------------------------------------------
# 4️⃣ ROCOv2 Collection
# ---------------------------------------------------

roco_collection = "rocov2_captions"

if not client.collection_exists(roco_collection):

    vector_size = len(roco_data["embeddings"][0])

    client.create_collection(
        collection_name=roco_collection,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

points = []
batch_size = 64

for idx, emb in tqdm(enumerate(roco_data["embeddings"]),
                     total=len(roco_data["embeddings"]),
                     desc="Uploading ROCOv2"):

    vector = emb.tolist() if isinstance(emb, np.ndarray) else emb

    points.append(
        models.PointStruct(
            id=idx,
            vector=vector,
            payload={
                "dataset": "rocov2",
                "text": roco_data["captions"][idx],
                "image": str(roco_data["image_ids"][idx])
                # "image_path": None,
                # "uid": None
            }
        )
    )

    if len(points) >= batch_size:
        client.upsert(
            collection_name=roco_collection,
            points=points,
            wait=False
        )
        points = []

# Final batch
if points:
    client.upsert(
        collection_name=roco_collection,
        points=points,
        wait=True
    )

print(f"Successfully indexed {len(roco_data['embeddings'])} ROCOv2 captions")