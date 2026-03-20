from sentence_transformers import SentenceTransformer, models

model = SentenceTransformer("clip-ViT-B-32")

image_embeddings = []
text_embeddings = []
points = []

for idx, item in enumerate(ds["train"]):
    image = item["image"]
    caption = item["caption"]

    img_emb = model.encode(image)
    txt_emb = model.encode(caption)

    points.append(
        models.PointStruct(
            id=f"roco_{idx}",
            vector=txt_emb.tolist(),  # or img_emb
            payload={
                "caption": caption,
                "type": "roco"
            }
        )
    )