"""
honeypot_detector.py
Detects the ~80 impossible/trap profiles in the dataset.
Disqualification rule: >10 honeypots in top 100 = instant DQ.

9 flag rules, threshold >= 2 flags.
"""

TIER1_CORE_SKILLS = {
    "embeddings", "semantic search", "sentence transformers",
    "information retrieval", "pinecone", "faiss", "elasticsearch",
    "weaviate", "qdrant", "milvus", "opensearch", "rag", "bm25",
    "hybrid search", "vector database",
}

TODAY_STR = "2026-06-18"


def is_honeypot(candidate: dict) -> bool:
    """
    Returns True if the candidate profile has signs of being a honeypot.
    Conservative threshold — better to miss a honeypot than to false-positive
    a real good candidate.
    """
    flags = 0
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # --- FLAG 1: Career duration >> stated YOE (impossible timeline) ---
    total_career_months = sum(r.get("duration_months", 0) for r in career_history)
    yoe_months = profile.get("years_of_experience", 0) * 12
    if total_career_months > yoe_months * 1.6 and total_career_months > 36:
        flags += 2  # Strong signal

    # --- FLAG 2: Expert/Advanced proficiency with 0 months used ---
    expert_zero = [
        s["name"]
        for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 1) == 0
    ]
    if len(expert_zero) >= 2:
        flags += 2

    # --- FLAG 3: Many high-proficiency skills with < 3 months each ---
    quick_experts = [
        s
        for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and 0 < s.get("duration_months", 999) < 3
    ]
    if len(quick_experts) >= 4:
        flags += 1

    # --- FLAG 4: End date before start date (impossible timeline) ---
    for role in career_history:
        start = role.get("start_date", "")
        end = role.get("end_date", "")
        if start and end and end < start:
            flags += 3  # Definitive impossibility

    # --- FLAG 5: Future end dates beyond today ---
    for role in career_history:
        end = role.get("end_date", "")
        if end and end > TODAY_STR and not role.get("is_current", False):
            flags += 2

    # --- FLAG 6: Skill endorsements absurdly high (>200) with beginner level ---
    absurd_endorsements = [
        s for s in skills
        if s.get("endorsements", 0) > 200 and s.get("proficiency") == "beginner"
    ]
    if len(absurd_endorsements) >= 2:
        flags += 1

    # --- FLAG 7: All Tier-1 core skills with 0 duration AND ≥4 of them ---
    # Classic keyword stuffer: RAG, Pinecone, FAISS, Embeddings all = 0 months
    zero_duration_tier1 = [
        s for s in skills
        if s.get("name", "").lower() in TIER1_CORE_SKILLS
        and s.get("duration_months", 1) == 0
    ]
    if len(zero_duration_tier1) >= 4:
        flags += 2  # Clear keyword stuffer

    # --- FLAG 8: YOE < 1 but career history total > 12 months ---
    yoe = profile.get("years_of_experience", 0)
    if yoe < 1 and total_career_months > 12:
        flags += 2  # Impossible: claimed <1yr experience but career entries sum to >1yr

    # --- FLAG 9: Multiple non-current roles at the same company with overlapping dates ---
    company_roles: dict = {}
    for role in career_history:
        if role.get("is_current", False):
            continue
        co = role.get("company", "").lower().strip()
        if not co:
            continue
        start = role.get("start_date", "")
        end = role.get("end_date", "") or TODAY_STR
        if co not in company_roles:
            company_roles[co] = []
        company_roles[co].append((start, end))

    for co, periods in company_roles.items():
        if len(periods) < 2:
            continue
        periods.sort()
        for i in range(len(periods) - 1):
            _, end_i = periods[i]
            start_next, _ = periods[i + 1]
            # Overlap: end of role i is after start of role i+1
            if end_i > start_next and start_next:
                flags += 2
                break  # One overlap per company is enough

    return flags >= 2  # Conservative threshold
