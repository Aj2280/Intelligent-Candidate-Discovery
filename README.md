# Redrob Intelligent Candidate Ranker

**Hackathon:** Intelligent Candidate Discovery & Ranking Challenge  
**Approach:** Hybrid rule-based feature scoring + semantic embedding re-ranking + behavioral multiplier + two-stage re-ranking boost

---

## How It Works

### Architecture Overview

```
100K candidates (JSONL)
        │
        ▼
┌─────────────────────┐
│  Honeypot Filter    │  ← 9-flag rule set, removes impossible profiles
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Feature Scorer (55% of final score)            │
│  ├── Title + Career fit          (28%)          │
│  │   (title lookup + career arc + ML@product)   │
│  ├── Core skills match           (30%)          │
│  │   (proficiency × duration × assessment)      │
│  ├── Experience range (5-9y JD)  (18%)          │
│  ├── Location fit (Pune/Noida)   (14%)          │
│  └── Education tier              (10%)          │
│  [+] Career Trajectory Bonus     (0-6%)         │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Semantic Score (45% of final score)            │
│  BAAI/bge-small embeddings vs JD text           │
│  (precomputed offline, instant at rank time)    │
│  Richer text: title + headline + descriptions   │
│  + certs + verified assessments + summary       │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Two-Stage Re-Ranking Boost                     │
│  Top 15% feature AND top 15% semantic → +8%     │
│  "Cream of the crop" candidates surface higher   │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Behavioral Multiplier (0.4 – 1.0)              │
│  ├── Availability (open_to_work, recency)       │
│  ├── Responsiveness (recruiter_response_rate)   │
│  ├── GitHub activity                            │
│  ├── Interview + offer acceptance rate          │
│  ├── Identity verification (email/phone/LinkedIn)│
│  └── Recruiter demand (saved_by_30d, views)     │
└─────────────────────────────────────────────────┘
        │
        ▼
  Top 100 ranked CSV with specific per-candidate reasoning
```

### Key Design Decisions

1. **Title is a gate, not just a signal** — A candidate with "HR Manager" as their current title and no tech career history cannot be a Senior AI Engineer regardless of skills listed. Title gate threshold set to `<= 5` (only pure wrong-domain roles: HR, Accountant, Content Writer, etc.) so QA Engineers and Project Managers are evaluated fairly on their other signals.

2. **Skills weighted by `duration_months` AND platform assessments** — Skills listed with 0 months used get near-zero weight (0.05x multiplier). A keyword stuffer listing "RAG, Pinecone, Embeddings" with 0 months each scores no better than having no skills. Platform-verified assessment scores add up to 25% boost per skill.

3. **Semantic embeddings catch semantic fits** — A candidate who built "relevance ranking for job search" without using the word "NDCG" still scores high semantically. Pure keyword matching fails here. Richer text embedding (includes headline, certifications, verified assessments) makes this signal more accurate.

4. **Two-stage re-ranking** — Candidates in the top 15th percentile of BOTH feature score AND semantic score receive an 8% combined score boost. This prevents semantic noise from pushing down genuinely excellent candidates who score high on both dimensions.

5. **Career trajectory as a bonus signal** — A candidate who progressively moved Data Scientist → ML Engineer → Senior ML Engineer gets up to +6% score bonus over a flat career. ML roles at product companies are the strongest signal.

6. **Behavioral signals as multiplier with full signal coverage** — Now uses `verified_email`, `verified_phone`, `linkedin_connected`, `saved_by_recruiters_30d`, `profile_views_received_30d`, and `offer_acceptance_rate` — all signals present in the schema but previously unused.

7. **Honest reasoning** — Reasoning strings mention specific YOE, company names, skill usage durations, platform-verified assessment scores, recruiter demand, and real concerns. Not templates.

---

## Setup

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/redrob-ranker
cd redrob-ranker

# 2. Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies (CPU-only torch!)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Place dataset
mkdir -p data
cp /path/to/candidates.jsonl data/candidates.jsonl

# 5. Precompute embeddings (run once offline, ~10 min for 100K)
python precompute_embeddings.py --candidates data/candidates.jsonl

# 6. Run ranker (must complete in <5 min)
python rank.py --candidates data/candidates.jsonl --out submission.csv

# 7. Validate before submitting
python validate_submission.py submission.csv
```

---

## File Structure

```
redrob-ranker/
├── rank.py                    # Main script (single reproduce command)
├── scorer.py                  # All feature scoring logic + career trajectory
├── honeypot_detector.py       # 9-flag trap/honeypot detection
├── reasoning_gen.py           # Per-candidate reasoning generation (bug-fixed)
├── precompute_embeddings.py   # Offline embedding precomputation (richer text)
├── app.py                     # HuggingFace Spaces demo (consistent with ranker)
├── validate_submission.py     # Format validator (from bundle)
├── test_ranker.py             # 10-test validation suite
├── requirements.txt
├── submission_metadata.yaml
├── README.md
├── data/
│   └── candidates.jsonl       # NOT committed (too large)
├── models/
│   └── bge-small/             # Cached model (NOT committed)
└── precompute/
    ├── embeddings.npy         # NOT committed (150MB)
    ├── cand_ids.npy           # NOT committed
    └── jd_embedding.npy       # Committed (tiny: 1.5KB)
```

---

## Reproduce Command

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

**Runtime:** ~45–90 seconds on CPU with 16GB RAM (well within 5-min limit).  
**No GPU, no network access required at ranking time.**

---

## Sandbox / Demo

Live demo: [https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker](https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker)

Upload any subset of candidates.jsonl (up to 500 records) to see the ranker in action.  
The demo shows a full score breakdown: feature, semantic, and behavioral scores per candidate.

---

## AI Tools Used

- Claude: architecture discussion, bug identification, and code review
- GitHub Copilot: autocomplete
- No candidate data was fed to any external LLM

---

## Why This Beats Keyword Matching

The sample_submission.csv (provided in bundle) ranks an HR Manager #1 because
their skills section lists AI keywords. Our ranker correctly scores this candidate
near zero because:

1. `score_title_career()` → `TITLE_SCORES["hr manager"] = 3` → `title_gate = 0.25x`
2. `score_skills()` → All skills have `duration_months = 0` → `dur_mult = 0.05` → effectively 0
3. `is_honeypot()` → FLAG 7: 5 Tier-1 core skills all with 0 duration → disqualified

A real Senior ML Engineer with 7 years at a Fintech product company who built
embedding-based search, even without all the buzzwords, scores 0.78+ and ranks
in top 5. Career trajectory bonus, ML-at-product-company multiplier, and the
two-stage re-ranking boost all work together to surface them.

---

## Scoring Changes v1 → v2

| Component | v1 Weight | v2 Weight | Change |
|-----------|-----------|-----------|--------|
| Title + Career | 30% | 28% | Slightly reduced |
| Skills | 25% | 30% | **Increased** — most discriminative |
| Experience | 20% | 18% | Slightly reduced |
| Location | 15% | 14% | Slightly reduced |
| Education | 10% | 10% | Unchanged |
| Career Trajectory | — | +0-6% bonus | **NEW** |
| Feature:Semantic split | 60:40 | 55:45 | More semantic weight |
| Two-stage re-ranking | — | +8% boost | **NEW** |
| Behavioral signals | 6 signals | 10 signals | +4 new signals |
| Honeypot flags | 6 flags | 9 flags | +3 new flags |
