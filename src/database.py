"""
database.py — ChromaDB Vector Database Module
Stores and searches product vectors using HNSW algorithm
"""

import chromadb
import pandas as pd
from tqdm import tqdm
import numpy as np


class ProductDatabase:

    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="abo_products",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ ChromaDB ready — {self.collection.count()} products indexed")

    def index_catalog(self, df: pd.DataFrame, embedder, force=False, embed_images=False):
        """
        embed_images=False (default, FAST): text-only embeddings.
        Product photos are still shown in the UI via their real image_url
        (Streamlit loads that live) — they just aren't pre-embedded during
        indexing, which is what makes indexing take seconds instead of
        hours on a large catalog. Set embed_images=True only for small
        catalogs where you want true multimodal (image+text) indexing.
        """
        if self.collection.count() > 0 and not force:
            print(f"Already indexed {self.collection.count()} products.")
            return

        if force and self.collection.count() > 0:
            self.client.delete_collection("abo_products")
            self.collection = self.client.create_collection(
                name="abo_products",
                metadata={"hnsw:space": "cosine"}
            )

        print(f"\nIndexing {len(df)} products ({'multimodal' if embed_images else 'text-only, fast'})...")
        ids, embeddings, metadatas, documents = [], [], [], []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = (
                f"{row['name']}. {row['description']} "
                f"Color: {row['color']}. Category: {row['category']}. "
                f"Material: {row.get('material', '')}. "
                f"Brand: {row.get('brand', '')}. "
                f"Price: Rs.{row['price']}."
            )
            img_emb = None
            if embed_images and pd.notna(row.get('image_url')):
                img_emb = embedder.embed_image_from_url(row['image_url'])

            if img_emb:
                t = np.array(embedder.embed_text(text))
                i = np.array(img_emb)
                c = (t + i) / 2
                final = (c / np.linalg.norm(c)).tolist()
            else:
                final = embedder.embed_text(text)

            ids.append(str(row['product_id']))
            embeddings.append(final)
            documents.append(text)
            metadatas.append({
                "product_id":  str(row['product_id']),
                "name":        str(row['name']),
                "category":    str(row['category']),
                "description": str(row['description']),
                "price":       int(row['price']),
                "color":       str(row['color']),
                "material":    str(row.get('material', '')),
                "brand":       str(row.get('brand', '')),
                "stock":       str(row['stock']),
                "image_url":   str(row.get('image_url', ''))
            })

        self.collection.add(
            ids=ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas
        )
        print(f"✅ Indexed {self.collection.count()} products!")

    def search(self, query_vector: list, top_k: int = 5) -> list:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),
            include=["metadatas", "distances"]
        )
        products = []
        if results and results['metadatas']:
            for i, meta in enumerate(results['metadatas'][0]):
                p = dict(meta)
                p['similarity_score'] = round(
                    (1 - results['distances'][0][i]) * 100, 1
                )
                products.append(p)
        return products

    def get_count(self): return self.collection.count()
