from datasets import load_dataset
import torch
import open_clip
import numpy as np
from tqdm import tqdm
from PIL import Image

from data.metadata import extract_clinical_metadata as get_clinical_metadata

def load_rocov2_embeddings(device="cpu", batch_size=8):
    ds = load_dataset("eltorio/ROCOv2-radiology", split="train")
    device = torch.device(device)

    # Use BiomedCLIP to match the Indiana pipeline
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.to(device).eval()

    results = {"embeddings": [], "captions": [], "image_ids": [], "metadata": []}

    for batch in tqdm(ds.iter(batch_size=batch_size), desc="Encoding ROCOv2 (BiomedCLIP)"):
        captions = batch["caption"]
        # Normalize text style by adding a medical context prefix
        processed_texts = [f"Medical imaging showing: {c}" for c in captions]
        text_tokens = tokenizer(processed_texts).to(device)

        images = [img.convert("RGB") for img in batch["image"]]
        image_tensors = torch.stack([preprocess(img) for img in images]).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            image_features = model.encode_image(image_tensors)
            
            # Normalize
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Late Fusion
            combined = (text_features + image_features) / 2
            combined /= combined.norm(dim=-1, keepdim=True)

        results["embeddings"].extend(combined.cpu().numpy())
        results["captions"].extend(captions)
        results["image_ids"].extend(batch["image_id"])
        results["metadata"].extend([get_clinical_metadata(c) for c in captions])

    return results