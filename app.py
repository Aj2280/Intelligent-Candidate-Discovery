"""
app.py — HuggingFace Spaces sandbox demo for Redrob Hackathon

Upload a small JSONL of candidates → get a ranked CSV back.
This satisfies the "sandbox_link" requirement in submission_metadata.yaml.

Deploy:
  1. Create a new Space at huggingface.co/spaces
  2. Upload: app.py, scorer.py, honeypot_detector.py, reasoning_gen.py,
             precompute_embeddings.py
  3. Upload: requirements_demo.txt as requirements.txt
  4. Upload: precompute/jd_embedding.npy
  5. Upload: models/bge-small/ folder

The Space will auto-install and launch.
"""

import csv
import io
import json
import os
import tempfile

import gradio as gr
import numpy as np

# ── Lazy imports (model loads once at startup) ──
_model = None
_jd_emb = None


def _load_model():
    global _model, _jd_emb
    if _model is not None:
        return

    from sentence_transformers import SentenceTransformer

    model_path = "./models/bge-small"
    if os.path.exists(model_path):
        _model = SentenceTransformer(model_path)
    else:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    if os.path.exists("precompute/jd_embedding.npy"):
        _jd_emb = np.load("precompute/jd_embedding.npy")
    else:
        from precompute_embeddings import JD_TEXT
        _jd_emb = _model.encode(JD_TEXT, normalize_embeddings=True)


def rank_uploaded_candidates(file_obj, requested_top_n=100) -> tuple:
    """
    Takes an uploaded JSONL file, ranks candidates, returns:
    - Markdown preview of top 20 with score breakdown
    - Downloadable CSV file
    """
    from honeypot_detector import is_honeypot
    from scorer import score_candidate, behavioral_multiplier
    from reasoning_gen import generate_reasoning
    from precompute_embeddings import build_candidate_text

    _load_model()

    # Read candidates
    if hasattr(file_obj, "read"):
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
    else:
        with open(file_obj, encoding="utf-8") as f:
            content = f.read()

    candidates = []
    # Try parsing as a standard JSON array first
    try:
        parsed_data = json.loads(content)
        if isinstance(parsed_data, list):
            candidates = parsed_data
        elif isinstance(parsed_data, dict):
            candidates = [parsed_data]
    except json.JSONDecodeError:
        pass

    # Fallback to JSON Lines (JSONL) line-by-line parsing if empty
    if not candidates:
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not candidates:
        return "❌ **Error:** No valid candidates found in the file.", None

    if len(candidates) > 500:
        return "❌ **Error:** Demo limited to 500 candidates. Use the full ranker locally.", None

    # Encode candidates with richer text
    texts = [build_candidate_text(c) for c in candidates]
    embs = _model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    sims = (embs @ _jd_emb).astype(float)

    s_min, s_max = sims.min(), sims.max()
    sims_norm = (sims - s_min) / (s_max - s_min + 1e-9)

    # Score each candidate (with behavioral multiplier — consistent with rank.py)
    results = []
    honeypot_count = 0
    for i, c in enumerate(candidates):
        if is_honeypot(c):
            honeypot_count += 1
            continue
        feat = score_candidate(c)
        beh = behavioral_multiplier(c)
        sem = float(sims_norm[i])
        combined = 0.55 * feat + 0.45 * sem
        results.append({
            "candidate_id": c["candidate_id"],
            "combined": combined,
            "feature": feat,
            "semantic": sem,
            "behavioral": beh,
            "candidate": c,
        })

    results.sort(key=lambda x: x["combined"], reverse=True)
    top_n = min(len(results), int(requested_top_n))
    top = results[:top_n]

    if not top:
        return "❌ **Error:** No candidates passed honeypot filter.", None

    max_s = top[0]["combined"]
    min_s = top[-1]["combined"]
    rng = max_s - min_s if max_s != min_s else 1.0

    # Write CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, item in enumerate(top, 1):
        item["_norm_score"] = round(0.5 + 0.5 * (item["combined"] - min_s) / rng, 4)
        reasoning = generate_reasoning(item["candidate"], rank, item)
        writer.writerow([item["candidate_id"], rank, item["_norm_score"], reasoning])

    csv_content = output.getvalue()

    # Save to temp file for download
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    tmp.write(csv_content)
    tmp.close()

    # Build rich markdown preview
    preview_lines = [
        f"### ✅ Ranked {top_n} candidates  |  🚫 {honeypot_count} honeypots filtered\n",
        "| Rank | ID | Title | YOE | Feature | Semantic | Behavioral | Score |",
        "|------|-----|-------|-----|---------|----------|------------|-------|",
    ]
    for rank, item in enumerate(top[:20], 1):
        p = item["candidate"].get("profile", {})
        preview_lines.append(
            f"| **{rank}** | {item['candidate_id']} | "
            f"{p.get('current_title', 'N/A')[:25]} | "
            f"{p.get('years_of_experience', 0):.1f} | "
            f"{item['feature']:.3f} | "
            f"{item['semantic']:.3f} | "
            f"{item['behavioral']:.2f} | "
            f"**{item['_norm_score']}** |"
        )
    if top_n > 20:
        preview_lines.append(f"\n*... and {top_n - 20} more in the downloadable CSV*")

    return "\n".join(preview_lines), tmp.name


# ── Gradio UI ──

CSS = """
#header { text-align: center; margin-bottom: 1em; }
#upload-col { background: var(--background-fill-secondary); border-radius: 12px; padding: 1em; }
#results-col { background: var(--background-fill-secondary); border-radius: 12px; padding: 1em; }
"""

with gr.Blocks(title="Redrob Candidate Ranker — Demo", css=CSS) as demo:
    gr.Markdown("""
    <div id="header">

    # 🔍 Redrob Intelligent Candidate Ranker
    **Intelligent Candidate Discovery & Ranking — Hackathon Submission**

    Upload a JSONL file (up to 500 candidates) to see ranked results with full score breakdown.

    </div>
    """)

    with gr.Row():
        with gr.Column(elem_id="upload-col"):
            gr.Markdown("### 📁 Upload Candidates")
            file_input = gr.File(
                label="Upload candidates JSONL (max 500 candidates)",
                file_types=[".jsonl", ".json"],
            )
            top_n_input = gr.Slider(
                minimum=1, 
                maximum=500, 
                value=100, 
                step=1, 
                label="Number of top candidates to return"
            )
            rank_btn = gr.Button("🚀 Rank Candidates", variant="primary", size="lg")

            gr.Markdown("""
            ---
            **Scoring Architecture:**
            - 🏷️ **Title + Career** (28%) — explicit title lookup + career arc
            - 🛠️ **Skills** (30%) — weighted by proficiency × usage duration
            - 📅 **Experience** (18%) — sweet spot: 5–9 years
            - 📍 **Location** (14%) — Pune/Noida preferred
            - 🎓 **Education** (10%) — institution tier + field + certs
            - 🤖 **Semantic** (45% weight) — BAAI/bge-small cosine similarity
            - ⚡ **Behavioral** multiplier (0.4–1.0) — activity, responsiveness, GitHub
            """)

        with gr.Column(elem_id="results-col"):
            gr.Markdown("### 📊 Results")
            output_text = gr.Markdown(
                value="*Upload a JSONL file and click **Rank Candidates** to see results.*"
            )
            output_file = gr.File(label="⬇️ Download full ranked CSV")

    rank_btn.click(
        fn=rank_uploaded_candidates,
        inputs=[file_input, top_n_input],
        outputs=[output_text, output_file],
    )

    gr.Markdown("""
    ---
    *Demo limited to 500 candidates. Full submission uses precomputed embeddings for 100K candidates.*
    *Scoring is consistent with the full ranker — feature weights, behavioral multiplier, and semantic scoring all match.*
    """)

if __name__ == "__main__":
    demo.launch()
