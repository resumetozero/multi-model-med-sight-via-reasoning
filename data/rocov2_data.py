from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from tqdm import tqdm


def load_rocov2_embeddings(device="cpu", batch_size=32):

    ds = load_dataset("eltorio/ROCOv2-radiology", split="train")

    device = torch.device(device)

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)

    model.eval()

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    embeddings = []
    captions = []
    image_ids = []

    for batch in tqdm(
        ds.iter(batch_size=batch_size),
        desc="Encoding ROCOv2"
    ):

        batch_captions = batch["caption"]
        batch_images = [img.convert("RGB") for img in batch["image"]]

        inputs = processor(
            text=batch_captions,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = model(**inputs)

            text_features = outputs.text_embeds
            image_features = outputs.image_embeds

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            combined = (text_features + image_features) / 2
            combined = combined / combined.norm(dim=-1, keepdim=True)

        embeddings.extend(combined.cpu().numpy())
        captions.extend(batch_captions)
        image_ids.extend(batch["image_id"])

    return {
        "embeddings": embeddings,
        "captions": captions,
        "image_ids": image_ids
    }