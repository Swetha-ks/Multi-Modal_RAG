"""
evaluate.py — Lightweight retrieval evaluation for the RAG pipeline.

Runs a fixed set of (query, expected_category) test cases through the
pipeline's retriever and reports standard IR metrics:
  - Precision@k        — fraction of top-k results in the expected category
  - Category Hit Rate  — fraction of queries with >=1 correct-category
                          result in the top-k
  - Avg Similarity     — mean CLIP cosine similarity of returned results

Can be run standalone:
    python evaluation/evaluate.py
or imported and called from the Streamlit Analytics tab.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_QUERIES = [
    ("blue denim jacket", "Jacket"),
    ("floral dress for a party", "Dress"),
    ("white sneakers", "Shoes"),
    ("running shoes", "Shoes"),
    ("cotton polo t-shirt", "T-Shirt"),
    ("warm hoodie", "Hoodie"),
    ("formal black trousers", "Trousers"),
    ("blue jeans", "Jeans"),
    ("cargo pants", "Pants"),
    ("gym leggings", "Leggings"),
    ("white cotton kurta", "Kurta"),
    ("printed kurti", "Kurti"),
    ("silk saree", "Saree"),
    ("leather handbag", "Bag"),
    ("formal blazer", "Blazer"),
    ("casual shorts", "Shorts"),
    ("linen shirt", "Shirt"),
    ("maxi skirt", "Skirt"),
]


def run_evaluation(pipeline, k: int = 5) -> dict:
    per_query = []
    total_precision, total_similarity, hits = 0.0, 0.0, 0

    for query, expected_category in TEST_QUERIES:
        products = pipeline.retrieve(query, k=k)
        if not products:
            per_query.append({
                "query": query, "expected": expected_category,
                "precision": 0.0, "hit": False, "avg_similarity": 0.0,
            })
            continue

        correct = sum(1 for p in products if p.get("category") == expected_category)
        precision = correct / len(products)
        avg_sim = sum(p.get("similarity_score", 0) for p in products) / len(products)
        hit = correct > 0

        total_precision += precision
        total_similarity += avg_sim
        hits += 1 if hit else 0

        per_query.append({
            "query": query, "expected": expected_category,
            "precision": round(precision, 2), "hit": hit,
            "avg_similarity": round(avg_sim, 1),
        })

    n = len(TEST_QUERIES)
    return {
        "k": k,
        "num_queries": n,
        "mean_precision_at_k": round(total_precision / n, 3) if n else 0,
        "category_hit_rate": round(hits / n, 3) if n else 0,
        "mean_similarity": round(total_similarity / n, 1) if n else 0,
        "per_query": per_query,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from src.embedder import MultimodalEmbedder
    from src.database import ProductDatabase
    from src.pipeline import RAGPipeline

    embedder = MultimodalEmbedder()
    db = ProductDatabase(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    pipe = RAGPipeline(embedder, db)

    results = run_evaluation(pipe)
    print("\n" + "=" * 50)
    print("  RAG Retrieval Evaluation")
    print("=" * 50)
    print(f"Queries evaluated:     {results['num_queries']}")
    print(f"Mean Precision@{results['k']}:      {results['mean_precision_at_k']}")
    print(f"Category Hit Rate:     {results['category_hit_rate']}")
    print(f"Mean Similarity Score: {results['mean_similarity']}%")
    print("\nPer-query breakdown:")
    for r in results["per_query"]:
        mark = "✓" if r["hit"] else "✗"
        print(f"  {mark} '{r['query']}' → expected {r['expected']}: "
              f"precision={r['precision']}, sim={r['avg_similarity']}%")
