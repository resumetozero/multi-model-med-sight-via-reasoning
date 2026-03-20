import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 

def load_and_process_data():
    """
    Load Indiana Chest X-ray reports and projections,
    merge, filter frontal images, and clean missing values.
    """
    reports = pd.read_csv('data/chestxray_IU/indiana_reports.csv')
    projections = pd.read_csv('data/chestxray_IU/indiana_projections.csv')

    merged_df = pd.merge(projections, reports, on='uid')

    # Filter for frontal images
    frontal_only = merged_df[merged_df['projection'] == 'Frontal'].copy()

    # Create full image path
    image_folder = "data/chestxray_IU/images_normalized/"
    frontal_only['image_path'] = frontal_only['filename'].apply(lambda x: os.path.join(image_folder, x))

    # Remove rows with missing findings or indications
    df_final = frontal_only.dropna(subset=['findings', 'indication']).copy()

    # Combine text fields safely
    df_final['text'] = df_final['indication'].fillna('') + ". " + df_final['findings'].fillna('')

    print(f"Dataset ready! Total valid cases: {len(df_final)}")
    return df_final

def split_docs(df, chunk_size=800, chunk_overlap=100):
    """
    Convert dataframe text into langchain Documents and split into chunks.
    Returns list of chunk Documents.
    """
    # Create Document objects with uid metadata
    docs = [
    Document(
        page_content=text,
        metadata={
            "uid": uid,
            "image_path": img_path
        }
    )
    for text, uid, img_path in zip(df['text'], df['uid'], df['image_path'])]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ". ", "; "],
        length_function=len
    )

    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks

def generate_embeddings(chunks, device="cpu", embedding_model = "clip-ViT-B-32"):  #microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224   
    """
    Generate embeddings for a list of Document chunks using SentenceTransformer.
    Returns a dataframe with uid, chunk text, and embedding.
    """
    model = SentenceTransformer(embedding_model, device=device)

    embeddings = []
    texts = []
    uids = []
    image_paths = []

    for doc in tqdm(chunks, desc="Generating embeddings"):
        text = doc.page_content
        img_path = doc.metadata["image_path"]

        # Load image safely
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue  # skip broken images

        # Encode
        text_emb = model.encode(text)
        image_emb = model.encode(image)

        # Combine embeddings (simple + effective)
        combined_emb = (text_emb + image_emb) / 2

        embeddings.append(combined_emb)
        texts.append(text)
        uids.append(doc.metadata["uid"])
        image_paths.append(img_path)

    df_chunks = pd.DataFrame({
        "uid": uids,
        "text": texts,
        "image_path": image_paths,
        "embedding": embeddings
    })

    return df_chunks

if __name__ == "__main__":
    df_final = load_and_process_data()
    chunks = split_docs(df_final, chunk_size=800, chunk_overlap=50)
    df_chunks = generate_embeddings(chunks, device="cpu") 
    print(df_chunks.head())