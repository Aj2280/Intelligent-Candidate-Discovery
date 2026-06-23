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


def rank_uploaded_candidates(file_obj, requested_top_n=100, preview_count=20) -> tuple:
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
        return "-", "-", "-", "-", "❌ **Error:** No valid candidates found in the file.", None

    if len(candidates) > 500:
        return "-", "-", "-", "-", "❌ **Error:** Demo limited to 500 candidates. Use the full ranker locally.", None

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
        return "-", "-", "-", "-", "❌ **Error:** No candidates passed honeypot filter.", None

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
        "| Rank | ID | Title | YOE | Feature | Semantic | Behavioral | Score |",
        "|------|-----|-------|-----|---------|----------|------------|-------|",
    ]
    for rank, item in enumerate(top[:int(preview_count)], 1):
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
    if top_n > int(preview_count):
        preview_lines.append(f"\n*... and {top_n - int(preview_count)} more in the downloadable CSV*")

    # KPIs
    total_cands = len(candidates) + honeypot_count
    avg_score = sum(item["_norm_score"] for item in top) / len(top) if top else 0.0

    return (
        f"{total_cands:,}",
        f"{honeypot_count:,}",
        f"{top_n:,}",
        f"{avg_score:.3f}",
        "\n".join(preview_lines), 
        tmp.name
    )


# ── Gradio UI ──

CSS = """
:root {
    --primary: #7c3aed;
    --primary-hover: #6d28d9;
    --bg-main: #fafafa;
    --card-bg: #ffffff;
    --border-color: #f3f4f6;
}
body { background-color: var(--bg-main) !important; font-family: 'Inter', sans-serif; }
#top-nav { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background: white; border-bottom: 1px solid #eaeaea; }
#nav-links { display: flex; gap: 20px; font-weight: 500; color: #4b5563; }
#nav-links span.active { color: var(--primary); border-bottom: 2px solid var(--primary); padding-bottom: 5px; }
.hero-title { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
.hero-title span { color: var(--primary); }
.hero-subtitle { color: #6b7280; margin-bottom: 20px; }
.card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
.card-title { font-weight: 600; font-size: 1.2em; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
.step-circle { background: var(--primary); color: white; border-radius: 50%; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.9em; font-weight: bold; }
.rank-button { background: linear-gradient(90deg, #8b5cf6, #7c3aed) !important; color: white !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; }
.kpi-row { display: flex; justify-content: space-around; background: #fafafa; border: 1px solid #f3f4f6; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
.kpi-box { text-align: center; border-right: 1px solid #e5e7eb; flex: 1; }
.kpi-box:last-child { border-right: none; }
.kpi-val { font-size: 1.5em; font-weight: bold; color: var(--primary); }
.kpi-label { font-size: 0.8em; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-icon { font-size: 1.2em; margin-bottom: 5px; }
"""

with gr.Blocks(title="Redrob Candidate Ranker", css=CSS) as demo:
    gr.HTML("""
    <div id="top-nav">
        <div style="font-weight:bold; font-size:1.2em; display:flex; align-items:center; gap:8px;">
            <span style="font-size:1.5em;">🔍</span> Redrob <span style="color:#7c3aed;">Ranker</span>
        </div>
        <div id="nav-links">
            <span class="active">🏠 Home</span>
            <span>💡 How It Works</span>
            <span>🎯 Scoring</span>
            <span>ℹ️ About</span>
        </div>
        <div>
            <button style="background:#f3e8ff; color:#7c3aed; padding:8px 16px; border-radius:6px; font-weight:bold; border:none;">✨ AI-Powered Matching</button>
        </div>
    </div>
    <div style="padding: 30px 20px;">
        <div class="hero-title">Redrob Intelligent Candidate <span>Ranker</span></div>
        <div class="hero-subtitle">Intelligent Candidate Discovery & Ranking — Hackathon Submission<br><br>Upload a JSONL file (up to 500 candidates) to see ranked results with full score breakdown.</div>
    </div>
    """)

    with gr.Row(elem_classes="container", style="padding: 0 20px;"):
        with gr.Column(scale=1, elem_classes="card"):
            gr.HTML('<div class="card-title"><span class="step-circle">1</span> Upload Candidates</div>')
            gr.Markdown("<p style='color:#6b7280; font-size:0.9em;'>Upload a JSONL file containing candidate data (max 500 candidates).</p>")
            
            file_input = gr.File(
                label="Drag & drop your JSONL file here",
                file_types=[".jsonl", ".json"],
            )
            top_n_input = gr.Slider(minimum=1, maximum=500, value=100, step=1, label="Number of top candidates to return (in CSV)")
            preview_count_input = gr.Slider(minimum=10, maximum=100, value=25, step=1, label="Top N to display")
            rank_btn = gr.Button("🚀 Rank Candidates", elem_classes="rank-button", size="lg")

            gr.Markdown("""
            <div style="font-size:0.85em; color:#6b7280; margin-top:20px;">
            <strong>Scoring Architecture:</strong><br>
            • Title + Career (28%)<br>
            • Skills (30%)<br>
            • Experience (18%)<br>
            • Location (14%)<br>
            • Education (10%)<br>
            • Semantic Search (45% weight)<br>
            • Behavioral multiplier (0.4–1.0)
            </div>
            """)

        with gr.Column(scale=2, elem_classes="card"):
            with gr.Row():
                gr.HTML('<div class="card-title"><span class="step-circle">2</span> Results</div>')
                output_file = gr.File(label="⬇️ Download Full CSV", interactive=False)
            
            gr.Markdown("<p style='color:#6b7280; font-size:0.9em;'>Upload a file and click <strong>Rank Candidates</strong> to see results here.</p>")

            with gr.Row(elem_classes="kpi-row"):
                kpi_total = gr.HTML('<div class="kpi-box"><div class="kpi-icon">👥</div><div class="kpi-val">0</div><div class="kpi-label">Total Candidates</div></div>')
                kpi_honey = gr.HTML('<div class="kpi-box"><div class="kpi-icon">🛡️</div><div class="kpi-val" style="color:#ef4444;">0</div><div class="kpi-label">Honeypots Filtered</div></div>')
                kpi_top = gr.HTML('<div class="kpi-box"><div class="kpi-icon">🏆</div><div class="kpi-val" style="color:#10b981;">0</div><div class="kpi-label">Top Ranked</div></div>')
                kpi_avg = gr.HTML('<div class="kpi-box"><div class="kpi-icon">📊</div><div class="kpi-val" style="color:#f59e0b;">0.000</div><div class="kpi-label">Avg. Score (Top N)</div></div>')

            output_text = gr.Markdown(
                value="<div style='text-align:center; padding:40px; color:#9ca3af;'>📄<br><strong>No results yet</strong><br><small>Upload a JSONL file and click Rank Candidates to see ranked results.</small></div>"
            )

    def process_and_update_kpis(*args):
        # We need to wrap the return to format the HTML KPIs properly
        res = rank_uploaded_candidates(*args)
        if len(res) == 2:  # Error returned as (string, None)
            return (
                '<div class="kpi-box"><div class="kpi-icon">👥</div><div class="kpi-val">-</div><div class="kpi-label">Total Candidates</div></div>',
                '<div class="kpi-box"><div class="kpi-icon">🛡️</div><div class="kpi-val" style="color:#ef4444;">-</div><div class="kpi-label">Honeypots Filtered</div></div>',
                '<div class="kpi-box"><div class="kpi-icon">🏆</div><div class="kpi-val" style="color:#10b981;">-</div><div class="kpi-label">Top Ranked</div></div>',
                '<div class="kpi-box"><div class="kpi-icon">📊</div><div class="kpi-val" style="color:#f59e0b;">-</div><div class="kpi-label">Avg. Score (Top N)</div></div>',
                res[0],
                None
            )
        else:
            t, h, tp, a, md, csv = res
            return (
                f'<div class="kpi-box"><div class="kpi-icon">👥</div><div class="kpi-val">{t}</div><div class="kpi-label">Total Candidates</div></div>',
                f'<div class="kpi-box"><div class="kpi-icon">🛡️</div><div class="kpi-val" style="color:#ef4444;">{h}</div><div class="kpi-label">Honeypots Filtered</div></div>',
                f'<div class="kpi-box"><div class="kpi-icon">🏆</div><div class="kpi-val" style="color:#10b981;">{tp}</div><div class="kpi-label">Top Ranked</div></div>',
                f'<div class="kpi-box"><div class="kpi-icon">📊</div><div class="kpi-val" style="color:#f59e0b;">{a}</div><div class="kpi-label">Avg. Score (Top N)</div></div>',
                md,
                csv
            )

    rank_btn.click(
        fn=process_and_update_kpis,
        inputs=[file_input, top_n_input, preview_count_input],
        outputs=[kpi_total, kpi_honey, kpi_top, kpi_avg, output_text, output_file],
    )

    gr.HTML("<div style='text-align:center; padding:20px; color:#9ca3af; font-size:0.85em; border-top:1px solid #eaeaea; margin-top:20px;'>Built for <strong>IndiaRuns Hackathon 2025</strong> | AI-Powered Candidate Ranking System</div>")

if __name__ == "__main__":
    demo.launch()
