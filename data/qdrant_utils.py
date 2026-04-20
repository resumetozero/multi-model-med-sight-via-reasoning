import os
from typing import Mapping
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DEFAULT_PAYLOAD_INDEXES = {
    "modality": models.PayloadSchemaType.KEYWORD,
    "anatomy": models.PayloadSchemaType.KEYWORD,
    "dataset": models.PayloadSchemaType.KEYWORD,
    "patient_id": models.PayloadSchemaType.KEYWORD,
}


def get_qdrant_client(
    url_env: str = "QDRANT_URL",
    api_key_env: str = "QDRANT_API_KEY",
    timeout: int = 180,
) -> QdrantClient:
    return QdrantClient(
        url=os.getenv(url_env),
        api_key=os.getenv(api_key_env),
        timeout=timeout,
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 512,
    payload_indexes: Mapping[str, models.PayloadSchemaType] = None,
) -> None:
    payload_indexes = payload_indexes or DEFAULT_PAYLOAD_INDEXES
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    for field, schema in payload_indexes.items():
        try:
            client.create_payload_index(collection_name, field, schema)
        except Exception:
            continue


def delete_points_by_field(
    client: QdrantClient,
    collection_name: str,
    field: str,
    value: str,
) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value),
                    )
                ]
            )
        ),
    )
