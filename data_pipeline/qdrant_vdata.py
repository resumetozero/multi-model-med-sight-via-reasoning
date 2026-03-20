from qdrant_client import QdrantClient, models
from langchain_community.vectorstores import Qdrant
from processed import generate_embeddings, split_docs, load_and_process_data
import os

# 1. Connect to Qdrant (The Librarian)
client = QdrantClient(url=os.getenv("QDRANT_URL"), 
                        api_key=os.getenv("QDRANT_API_KEY"))


data = load_and_process_data()
chunks = split_docs(data, chunk_size=800, chunk_overlap=50)
embeddings = generate_embeddings(chunks, device="cpu") 


collection_name = "chestxray_reports"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=len(embeddings["embedding"][0]),
            distance=models.Distance.COSINE
        )
    )

batch_size = 256

for i in range(0, len(embeddings), batch_size):
    batch = embeddings.iloc[i:i+batch_size]

    points = [
        models.PointStruct(
            id=f"{row['uid']}_{idx}",
            vector=row["embedding"].tolist(),
            payload={
                    "uid": row["uid"],
                    "text": row["text"],
                    "image_path": row["image_path"]
                })
        for idx, row in batch.iterrows()
    ]

    client.upsert(
        collection_name=collection_name,
        points=points
    ) 