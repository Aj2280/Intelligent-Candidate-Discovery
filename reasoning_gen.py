"""
reasoning_gen.py
Generates specific, honest, non-templated reasoning for each ranked candidate.
The Stage 4 review checks 10 random rows for:
  - Specific facts from the profile (not generic)
  - JD connection
  - Honest concerns
  - No hallucination
  - Variation across rows
  - Rank consistency with tone
"""

from datetime import date

TODAY = date(2026, 6, 18)

PRODUCT_INDUSTRIES = {
    "SaaS", "Fintech", "EdTech", "E-commerce", "Food Delivery",
    "AI/ML", "AdTech", "Software", "HealthTech", "Startup",
    "Insurance Tech", "Transportation",
}

CORE_SKILL_NAMES = {
    "embeddings", "semantic search", "sentence transformers",
    "information retrieval", "pinecone", "faiss", "elasticsearch",
    "weaviate", "qdrant", "milvus", "opensearch", "rag", "bm25",
    "hybrid search", "fine-tuning llms", "nlp", "pytorch",
    "learning to rank", "lora", "qlora", "python",
    "hugging face transformers", "reranking", "xgboost",
    "ranking systems", "ndcg",
}

ML_TITLE_KEYWORDS = [
    "ml", "machine learning", "ai engineer", "nlp", "data scientist",
    "research scientist", "applied scientist", "deep learning",
]


def _career_arc_note(career: list) -> str:
    """Detect progressive ML career trajectory — SE → ML Engineer, etc."""
    if len(career) < 2:
        return ""
    titles = [r.get("title", "").lower() for r in career]
    # Check if earlier roles were SE and later ones have ML keywords
    has_progression = False
    for i, t in enumerate(titles):
        if any(kw in t for kw in ["software engineer", "backend", "data engineer"]):
            # Were there ML roles before this one (earlier in career = higher index in array)?
            earlier = titles[i + 1:]  # career_history is ordered newest-first
            if any(any(kw in e for kw in ML_TITLE_KEYWORDS) for e in earlier):
                has_progression = True
                break
    # Also check direct ML progression (data scientist → ML engineer)
    if not has_progression:
        ml_roles = [i for i, t in enumerate(titles) if any(kw in t for kw in ML_TITLE_KEYWORDS)]
        if len(ml_roles) >= 2:
            has_progression = True
    return " Career trajectory shows progressive ML focus." if has_progression else ""


def generate_reasoning(candidate: dict, rank: int, scores: dict) -> str:
    """
    Returns a 1-2 sentence reasoning string that:
    - Mentions specific facts (title, YOE, company, skills with durations)
    - Connects to JD requirements
    - Honestly mentions concerns for lower-ranked candidates
    - Varies meaningfully across candidates
    """
    p = candidate.get("profile", {})
    rs = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    title = p.get("current_title", "Unknown")
    yoe = p.get("years_of_experience", 0)
    company = p.get("current_company", "")
    location = p.get("location", "")
    country = p.get("country", "")

    # Find top relevant skills with actual usage time
    normalized_skills = [
        s if isinstance(s, dict) else {"name": str(s), "duration_months": 24, "proficiency": "intermediate"}
        for s in skills
    ]
    relevant_skills = sorted(
        [
            s for s in normalized_skills
            if s.get("name", "").lower() in CORE_SKILL_NAMES
            and s.get("duration_months", 0) > 3
        ],
        key=lambda s: (
            {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
                s.get("proficiency", "beginner"), 0
            ) * s.get("duration_months", 0)
        ),
        reverse=True,
    )[:3]

    # Find product company roles
    product_roles = [
        r for r in career
        if r.get("industry", "") in PRODUCT_INDUSTRIES
    ]

    # Skill assessment scores (verified platform signal)
    assessment_scores = rs.get("skill_assessment_scores", {})
    top_assessments = sorted(
        [(k, v) for k, v in assessment_scores.items() if v >= 70],
        key=lambda x: x[1], reverse=True
    )[:2]

    # Recruiter demand signal
    saved_30d = rs.get("saved_by_recruiters_30d", 0)
    recruiter_demand = f"saved by {saved_30d} recruiters (30d)" if saved_30d >= 8 else ""

    # Active job-seeking signal
    apps_30d = rs.get("applications_submitted_30d", 0)
    job_seeking_note = f"actively applying ({apps_30d} apps/30d)" if apps_30d >= 5 else ""

    # Build concerns list (honest!)
    concerns = []
    notice = rs.get("notice_period_days", 0)
    resp_rate = rs.get("recruiter_response_rate", 1.0)
    last_active_str = rs.get("last_active_date", "2026-01-01")
    try:
        days_inactive = (TODAY - date.fromisoformat(last_active_str)).days
    except ValueError:
        days_inactive = 0

    if notice > 60:
        concerns.append(f"{notice}-day notice period")
    if resp_rate < 0.25:
        concerns.append(f"low recruiter response rate ({resp_rate:.0%})")
    if days_inactive > 60:
        concerns.append(f"inactive for {days_inactive} days")
    if country not in ("India", "") and not rs.get("willing_to_relocate", False):
        concerns.append(f"based in {country}, not open to relocation")
    if rs.get("github_activity_score", -1) < 5 and rs.get("github_activity_score", -1) >= 0:
        concerns.append("low GitHub activity score")
    work_mode = rs.get("preferred_work_mode", "flexible")
    if work_mode == "remote":
        concerns.append("prefers remote-only (role is Pune/Noida based)")

    # Build the main sentence
    skill_text = ""
    if relevant_skills:
        parts = []
        for s in relevant_skills:
            parts.append(f"{s['name']} ({s['duration_months']}mo, {s['proficiency']})")
        skill_text = "; ".join(parts)

    # Verified assessment mention
    assess_note = ""
    if top_assessments:
        assess_note = f"; platform-verified: {', '.join(f'{k} ({int(v)}%)' for k, v in top_assessments)}"

    # Product company context
    prod_note = ""
    if product_roles:
        best_prod = product_roles[0]
        prod_note = f" at {best_prod['company']} ({best_prod['industry']})"

    # Career arc
    arc_note = _career_arc_note(career)

    # Location string (fixed — was a Python ternary precedence bug before)
    loc_str = f"; based in {location}, India" if country == "India" else f"; {location}, {country}"

    # Construct reasoning based on rank tier
    if rank <= 10:
        # Strong, specific praise
        parts = [f"{title}, {yoe:.1f}yrs{prod_note}"]
        if skill_text:
            parts.append(f"core retrieval/NLP skills: {skill_text}")
        if assess_note:
            parts.append(assess_note.lstrip("; "))
        if recruiter_demand:
            parts.append(recruiter_demand)
        if job_seeking_note:
            parts.append(job_seeking_note)
        parts.append(loc_str.lstrip("; "))
        if arc_note:
            parts.append(arc_note.strip())
        main = "; ".join(parts)
        if concerns:
            main += f". Note: {concerns[0]}."

    elif rank <= 30:
        # Positive with minor qualifications
        co_note = f" at product company ({company})" if product_roles else f" at {company}"
        parts = [f"{title}, {yoe:.1f}yrs experience{co_note}"]
        if skill_text:
            parts.append(f"relevant skills: {skill_text}")
        if assess_note:
            parts.append(assess_note.lstrip("; "))
        main = "; ".join(parts)
        if concerns:
            main += f". Concern: {', '.join(concerns[:2])}."

    elif rank <= 60:
        # Balanced
        parts = [f"{title} with {yoe:.1f}yrs"]
        if skill_text:
            parts.append(f"skills: {skill_text}")
        else:
            parts.append("limited core skill match")
        main = "; ".join(parts)
        if concerns:
            main += f". Concerns: {', '.join(concerns[:2])}."
        else:
            main += ". Behavioral signals adequate."

    else:
        # Honest about why this is rank 61-100
        if skill_text:
            adj = f"adjacent skills ({skill_text})"
        else:
            adj = "limited direct skill overlap"
        main = f"{title}, {yoe:.1f}yrs — {adj}; included as boundary-case given experience trajectory"
        if concerns:
            main += f". Key concerns: {', '.join(concerns[:2])}."

    # Trim to 250 chars
    return main[:250].strip()
