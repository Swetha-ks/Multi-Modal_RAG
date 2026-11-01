"""
pipeline.py — RAG Pipeline
Connects CLIP → ChromaDB → LLaMA 3 → Answer
"""

from groq import Groq
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:

    def __init__(self, embedder, database):
        self.embedder = embedder
        self.database = database
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.history = []
        print("✅ RAG Pipeline ready!")

    def retrieve(self, query: str, image: Image.Image = None,
                 max_price: int = None, color: str = None, k: int = 5):
        vec = self.embedder.embed_combined(query, image)
        products = self.database.search(vec, top_k=k * 2)
        if max_price:
            products = [p for p in products if p['price'] <= max_price]
        if color and color != "Any":
            match = [p for p in products if color.lower() in p['color'].lower()]
            if match:
                products = match
        return products[:k]

    def generate(self, query: str, products: list) -> str:
        if not products:
            return ("I could not find matching products. "
                    "Please try a different description.")

        context = ""
        for i, p in enumerate(products, 1):
            context += f"""
Product {i}: {p['name']}
  Price: Rs.{p['price']} | Color: {p['color']} | Stock: {p['stock']}
  Brand: {p.get('brand','')} | Material: {p.get('material','')}
  Description: {p['description']}
"""

        system = """You are a helpful, friendly e-commerce shopping assistant.
Rules:
- Answer ONLY using the product data provided below
- Never invent prices, colors, or specifications
- Be concise, warm, and natural — like a real salesperson
- Always mention price and availability
- If multiple products match, briefly list them"""

        msg = f"Customer: {query}\n\nAvailable products:\n{context}\nAnswer:"

        self.history.append({"role": "user", "content": msg})
        recent = self.history[-8:]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *recent],
            max_tokens=350,
            temperature=0.7
        )
        answer = resp.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def chat(self, query: str, image: Image.Image = None,
             max_price: int = None, color: str = None) -> dict:
        products = self.retrieve(query, image, max_price, color)
        answer = self.generate(query, products)
        return {"answer": answer, "products": products}

    def reset(self):
        self.history = []
