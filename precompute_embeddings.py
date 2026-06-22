"""
precompute_embeddings.py

Run this ONCE offline before the competition.
This is NOT part of the 5-minute ranking constraint.
It pre-encodes all 100K candidates using BAAI/bge-small-en-v1.5.

Output:
  precompute/embeddings.npy  — float32 array, shape (100000, 384)
  precompute/cand_ids.npy    — string array, shape (100000,)
  precompute/jd_embedding.npy — float32 array, shape (384,)

Total disk: ~150MB uncompressed.

Usage:
  python precompute_embeddings.py --candidates data/candidates.jsonl
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Install: pip install sentence-transformers")
    raise

# ─────────────────────────────────────────────
# JOB DESCRIPTION TEXT (key phrases, not full JD)
# ─────────────────────────────────────────────
JD_TEXT = """
Senior AI Engineer, 5-9 years experience at product companies.
Production embeddings-based retrieval systems: sentence-transformers, BGE, E5, OpenAI embeddings.
Vector databases: Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch.
Hybrid search, BM25, semantic search, information retrieval, RAG, reranking.
Evaluation frameworks for ranking: NDCG, MRR, MAP, offline benchmarks, A/B testing.
Strong Python, production code quality, not pure research.
NLP, natural language processing, text ranking, candidate matching.
LLM fine-tuning: LoRA, QLoRA, PEFT desirable.
Learning-to-rank, XGBoost, neural ranking models desirable.
Scrappy product engineering, shipping ranking systems to real users.
Based in Pune or Noida India, or willing to relocate.
Short notice period preferred.
"""


def build_candidate_text(candidate: dict) -> str:
    """
    Build a rich text representation of the candidate.
    Weights: title > career descriptions > skills > summary.
    """
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Title repeated 3x for higher weight
    title_text = f"{p.get('current_title', '')} " * 3

    # Career descriptions (most signal-rich part)
    career_texts = []
    for role in career:
        role_text = (
            f"{role.get('title', '')} at {role.get('company', '')} "
            f"({role.get('industry', '')}): {role.get('description', '')}"
        )
        career_texts.append(role_text)

    # Only include skills used for >6 months (not keyword stuffers)
    real_skills = " ".join(
        s["name"]
        for s in skills
        if s.get("duration_months", 0) > 6
    )

    # Summary (less trusted — self-written, can be inflated)
    summary = p.get("summary", "")[:500]  # clip

    full_text = (
        title_text
        + " ".join(career_texts)
        + " "
        + real_skills
        + " "
        + summary
    )

    return full_text[:3000]  # clip total to avoid memory issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--model-path", default="./models/bge-small")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs("precompute", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load or download model
    print("Loading model...")
    if os.path.exists(args.model_path):
        model = SentenceTransformer(args.model_path)
    else:
        print("Downloading BAAI/bge-small-en-v1.5 (first time only)...")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        model.save(args.model_path)
        print(f"Saved to {args.model_path}")

    # Encode JD
    print("Encoding job description...")
    jd_emb = model.encode(JD_TEXT, normalize_embeddings=True)
    np.save("precompute/jd_embedding.npy", jd_emb.astype(np.float32))
    print(f"JD embedding shape: {jd_emb.shape}")

    # Load candidates
    print(f"Reading candidates from {args.candidates}...")
    candidates_text = []
    cand_ids = []

    with open(args.candidates) as f:
        for line in tqdm(f, desc="Reading", total=100000):
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cand_ids.append(c["candidate_id"])
            candidates_text.append(build_candidate_text(c))

    print(f"Loaded {len(candidates_text)} candidates")

    # Encode in batches
    print(f"Encoding candidates (batch_size={args.batch_size})...")
    all_embeddings = []

    for i in tqdm(range(0, len(candidates_text), args.batch_size), desc="Encoding"):
        batch = candidates_text[i : i + args.batch_size]
        embs = model.encode(
            batch,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        all_embeddings.append(embs.astype(np.float32))

    all_embeddings = np.vstack(all_embeddings)
    print(f"Embedding matrix shape: {all_embeddings.shape}")

    # Save
    np.save("precompute/embeddings.npy", all_embeddings)
    np.save("precompute/cand_ids.npy", np.array(cand_ids))

    # Quick sanity check
    sims = all_embeddings @ jd_emb
    top5_idx = np.argsort(sims)[::-1][:5]
    print("\nTop 5 by semantic similarity to JD:")
    for idx in top5_idx:
        print(f"  {cand_ids[idx]}: {sims[idx]:.4f}")

    print("\nDone! Files saved to precompute/")
    print(f"  embeddings.npy: {all_embeddings.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
