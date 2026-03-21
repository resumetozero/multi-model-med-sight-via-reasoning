import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import open_clip
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
USE_HF_API = False
load_dotenv(find_dotenv())
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if USE_HF_API:
    hf_client = InferenceClient(token=HF_API_TOKEN)

def load_and_process_data():
    """
    Load Indiana Chest X-ray reports and projections,
    merge, filter frontal images, and clean missing values.
    """
    reports = pd.read_csv('data/chestxray_IU/indiana_reports.csv')
    projections = pd.read_csv('data/chestxray_IU/indiana_projections.csv')

    merged_df = pd.merge(projections, reports, on='uid')
    frontal_only = merged_df[merged_df['projection'] == 'Frontal'].copy()

    image_folder = "data/chestxray_IU/images_normalized/"
    frontal_only['image_path'] = frontal_only['filename'].apply(lambda x: os.path.join(image_folder, x))

    # Clean missing values
    df_final = frontal_only.dropna(subset=['findings', 'indication']).copy()
    df_final['text'] = df_final['indication'].fillna('') + ". " + df_final['findings'].fillna('')

    print(f"Dataset ready! Total valid cases: {len(df_final)}")
    return df_final

def split_docs(df, chunk_size=800, chunk_overlap=100):
    """
    Convert dataframe text into langchain Documents and split into chunks.
    """
    docs = [
        Document(
            page_content=text,
            metadata={"uid": uid, "image_path": img_path}
        )
        for text, uid, img_path in zip(df['text'], df['uid'], df['image_path'])
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ". ", "; "],
        length_function=len
    )

    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks

def generate_embeddings(chunks, device="cpu", embedding_model="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", batch_size=8):
    """
    Generate embeddings using HF InferenceClient or local SentenceTransformer.
    """
    embeddings, texts, uids, image_paths = [], [], [], []

    if USE_HF_API:
        for doc in tqdm(chunks, desc="Generating multimodal embeddings via HF API"):
            text = doc.page_content
            img_path = doc.metadata["image_path"]

            try:
                # 1. Get Text Embedding (Using CLIP so dimensions match the image)
                # Note: CLIP text model produces 512-dim vectors
                txt_emb = np.array(hf_client.feature_extraction(
                    text, 
                    model="openai/clip-vit-base-patch32"
                ))

                # 2. Get Image Embedding
                # We open the file as bytes to send to the API
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                
                img_emb = np.array(hf_client.feature_extraction(
                    img_bytes, 
                    model="openai/clip-vit-base-patch32"
                ))

                # 3. Combine Embeddings (Late Fusion)
                # Both are now (512,) so we can average them safely
                combined_emb = (txt_emb + img_emb) / 2
                
            except Exception as e:
                print(f"Error at UID {doc.metadata['uid']} with path {img_path}: {e}")
                continue

            embeddings.append(combined_emb)
            texts.append(text)
            uids.append(doc.metadata["uid"])
            image_paths.append(img_path)

    else:
        # model = SentenceTransformer(embedding_model, device=device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        device = torch.device(device)
        model.to(device)
        model.eval()

        for i in tqdm(range(0, len(chunks), batch_size), desc="Generating BiomedCLIP embeddings"):
            batch = chunks[i:i+batch_size]

            # --- Prepare Text ---
            batch_texts = [doc.page_content for doc in batch]
            text_tokens = tokenizer(batch_texts).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # --- Prepare Images ---
            batch_imgs_tensors = []
            valid_mask = []
            
            for doc in batch:
                try:
                    img = Image.open(doc.metadata["image_path"]).convert("RGB")
                    # Preprocess applies the specific 224x224 resize and normalization
                    batch_imgs_tensors.append(preprocess(img))
                    valid_mask.append(True)
                except Exception as e:
                    valid_mask.append(False)

            # Create final batch embedding container
            final_batch_embs = text_features.clone()

            if any(valid_mask):
                image_stack = torch.stack(batch_imgs_tensors).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_stack)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                # --- Late Fusion (Average Text + Image) ---
                img_idx = 0
                for j in range(len(batch)):
                    if valid_mask[j]:
                        # Average the two normalized vectors
                        combined = (text_features[j] + image_features[img_idx]) / 2
                        # Re-normalize so the result is a unit vector for Qdrant
                        final_batch_embs[j] = combined / combined.norm(dim=-1)
                        img_idx += 1

            # Move to CPU and store
            batch_np = final_batch_embs.cpu().numpy()
            for j, doc in enumerate(batch):
                embeddings.append(batch_np[j])
                texts.append(doc.page_content)
                uids.append(doc.metadata["uid"])
                image_paths.append(doc.metadata["image_path"])

        return pd.DataFrame({
            "uid": uids,
            "text": texts,
            "image_path": image_paths,
            "embedding": embeddings
        })

if __name__ == "__main__":
    df_final = load_and_process_data()
    chunks = split_docs(df_final, chunk_size=800, chunk_overlap=50)
    
    # Reducing batch_size for stability
    df_chunks = generate_embeddings(chunks, device="cpu", batch_size=4) 
    print(df_chunks.head())