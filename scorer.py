"""
scorer.py
Core feature scoring logic for the Redrob Intelligent Candidate Ranker.
Returns a normalized score in [0, 1] for each candidate.

Score breakdown:
  30% Title + Career fit
  25% Core skills match (weighted by proficiency + duration)
  20% Experience years
  15% Location fit
  10% Education

Then multiplied by behavioral_multiplier() [0.4 - 1.0].
"""

from datetime import date

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TODAY = date(2026, 6, 18)

# Titles scored directly
TITLE_SCORES = {
    # Perfect fit — 100 pts
    "ml engineer": 100,
    "machine learning engineer": 100,
    "senior machine learning engineer": 100,
    "principal machine learning engineer": 100,
    "staff machine learning engineer": 100,
    "ai engineer": 100,
    "senior ai engineer": 100,
    "lead ai engineer": 100,
    "principal ai engineer": 100,
    "nlp engineer": 95,
    "senior nlp engineer": 95,
    "applied ml engineer": 95,
    "ai research engineer": 90,
    "research engineer": 88,
    "senior software engineer (ml)": 88,
    "software engineer (ml)": 85,
    "ml platform engineer": 85,
    "mlops engineer": 82,
    "senior mlops engineer": 82,
    "machine learning platform engineer": 82,
    # Strong fit
    "research scientist": 78,
    "senior research scientist": 80,
    "data scientist": 70,
    "senior data scientist": 75,
    "principal data scientist": 78,
    "applied scientist": 72,
    "senior applied scientist": 75,
    "ai specialist": 65,
    "ml specialist": 68,
    "nlp scientist": 78,
    "computer vision engineer": 72,
    "deep learning engineer": 85,
    "junior ml engineer": 55,  # too junior but correct domain
    "associate ml engineer": 55,
    "data engineer": 35,       # adjacent — infra skill, not modeling
    "senior data engineer": 38,
    # Tech adjacent — needs career history check
    "software engineer": 40,
    "senior software engineer": 45,
    "staff software engineer": 48,
    "principal software engineer": 48,
    "backend engineer": 35,
    "full stack developer": 30,
    "cloud engineer": 28,
    "devops engineer": 20,
    "java developer": 15,
    ".net developer": 15,
    "mobile developer": 10,
    "frontend engineer": 10,
    "qa engineer": 8,
    # Wrong domain — heavy penalty
    "hr manager": 2,
    "accountant": 2,
    "content writer": 2,
    "graphic designer": 2,
    "civil engineer": 2,
    "mechanical engineer": 3,
    "operations manager": 3,
    "customer support": 2,
    "sales executive": 3,
    "marketing manager": 3,
    "project manager": 5,
    "business analyst": 8,
}

# Fuzzy keyword fallback: if title not in TITLE_SCORES, scan for these keywords.
# Returns the score of the best-matching keyword pair found.
TITLE_KEYWORD_SCORES = [
    # (keyword_must_contain, score)  — checked in order, first match wins
    (["ml", "machine learning", "ai engineer", "nlp engineer"], 90),
    (["deep learning", "neural"], 85),
    (["mlops", "ml platform", "ml infra"], 80),
    (["research scientist", "applied scientist"], 78),
    (["data scientist"], 70),
    (["nlp", "natural language"], 68),
    (["computer vision", "cv engineer"], 65),
    (["ml", "machine learning"], 60),   # catches "junior ml" variants
    (["ai"], 55),
    (["data engineer"], 35),
    (["software engineer", "software developer"], 40),
    (["backend", "back-end", "back end"], 35),
    (["full stack", "fullstack"], 30),
]

# Skills: name → base points
CORE_SKILLS = {
    # Tier 1 — "absolutely need" per JD (3 pts base)
    "embeddings": 3,
    "semantic search": 3,
    "sentence transformers": 3,
    "information retrieval": 3,
    "pinecone": 3,
    "faiss": 3,
    "elasticsearch": 3,
    "weaviate": 3,
    "qdrant": 3,
    "milvus": 3,
    "opensearch": 3,
    "rag": 3,
    "bm25": 3,
    "hybrid search": 3,
    "vector database": 3,
    # Tier 2 — nice to have (2 pts base)
    "fine-tuning llms": 2,
    "nlp": 2,
    "pytorch": 2,
    "learning to rank": 2,
    "lora": 2,
    "qlora": 2,
    "python": 2,
    "hugging face transformers": 2,
    "reranking": 2,
    "xgboost": 2,
    "ranking systems": 2,
    "ndcg": 2,
    "information retrieval systems": 2,
    # Tier 3 — adjacent (1 pt base)
    "docker": 1,
    "aws": 1,
    "gcp": 1,
    "spark": 1,
    "mlflow": 1,
    "weights & biases": 1,
    "redis": 1,
    "kafka": 1,
}

PROFICIENCY_MULT = {
    "expert": 1.5,
    "advanced": 1.2,
    "intermediate": 1.0,
    "beginner": 0.5,
}

# Product companies where AI/ML work actually matters
PRODUCT_INDUSTRIES = {
    "SaaS", "Fintech", "EdTech", "E-commerce", "Food Delivery",
    "AI/ML", "AdTech", "Transportation", "Insurance Tech",
    "HealthTech", "Startup", "Software",
}

# Full-consulting careers get penalized
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mphasis",
}

# India locations mapping
LOCATION_SCORES = {
    "pune": 100, "noida": 100,
    "delhi": 90, "ncr": 90, "gurgaon": 88, "gurugram": 88,
    "bangalore": 80, "bengaluru": 80,
    "hyderabad": 75, "mumbai": 72, "chennai": 65,
    "kolkata": 60, "ahmedabad": 55,
}

EDUCATION_TIER_SCORES = {
    "tier_1": 100,
    "tier_2": 70,
    "tier_3": 40,
    "tier_4": 20,
    "unknown": 20,
}

FIELD_BONUS = {
    "computer science": 15,
    "artificial intelligence": 15,
    "machine learning": 15,
    "information technology": 10,
    "mathematics": 10,
    "statistics": 10,
    "electronics": 5,
    "electrical engineering": 5,
}


# ─────────────────────────────────────────────
# COMPONENT SCORERS
# ─────────────────────────────────────────────

def _lookup_title_score(title: str) -> int:
    """Exact lookup first, then fuzzy keyword fallback, then default 20."""
    if title in TITLE_SCORES:
        return TITLE_SCORES[title]
    # Fuzzy fallback: scan for keyword groups
    for keywords, score in TITLE_KEYWORD_SCORES:
        if any(kw in title for kw in keywords):
            return score
    return 20  # default for truly unknown titles


def score_title_career(candidate: dict) -> float:
    """Returns 0-100. Combines title score + career history context."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    title = profile.get("current_title", "").lower().strip()
    title_score = _lookup_title_score(title)

    # Career history adjustments
    has_product_co = any(
        r.get("industry", "") in PRODUCT_INDUSTRIES for r in career
    )
    has_ml_role_in_history = any(
        any(kw in r.get("title", "").lower() for kw in
            ["ml", "machine learning", "ai", "nlp", "data scientist",
             "research engineer", "applied scientist"])
        for r in career
    )
    all_consulting = len(career) > 0 and all(
        any(co in r.get("company", "").lower() for co in CONSULTING_COMPANIES)
        for r in career
    )

    bonus = 0
    if has_product_co:
        bonus += 12
    if has_ml_role_in_history and title_score < 60:
        bonus += 15  # Redeems bad current title if they had ML roles before
    if all_consulting:
        bonus -= 20  # Penalty for pure consulting career

    return min(100, max(0, title_score + bonus))


# Relevant ML/AI certifications (name fragment → bonus points)
CERT_BONUSES = {
    "aws certified machine learning": 8,
    "google professional machine learning": 8,
    "tensorflow developer": 7,
    "pytorch": 5,
    "deep learning specialization": 7,
    "machine learning": 5,
    "nlp": 5,
    "hugging face": 6,
    "databricks": 5,
    "azure ai": 6,
    "gcp ai": 6,
    "mlops": 6,
    "data science": 4,
}


def score_skills(candidate: dict) -> float:
    """Returns 0-100. Weighted by proficiency + duration + endorsements.
    Also applies a bonus for platform skill_assessment_scores."""
    skills = candidate.get("skills", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    total = 0.0
    max_possible = 0.0

    for skill in skills:
        sname = skill.get("name", "").lower().strip()
        base = CORE_SKILLS.get(sname, 0)
        if base == 0:
            continue

        prof = PROFICIENCY_MULT.get(skill.get("proficiency", "beginner"), 0.5)
        dur_months = skill.get("duration_months", 0)

        # Trust skills more if they've actually used them for a while.
        # 0 months = keyword stuffer flag → near-zero weight (0.05)
        # 1-6 months = low weight, ramps up linearly to full at 12+ months
        if dur_months == 0:
            dur_mult = 0.05   # practically zero — keyword stuffer signal
        elif dur_months < 6:
            dur_mult = 0.1 + 0.15 * (dur_months / 6)  # ramp 0.1 → 0.25
        else:
            dur_mult = min(dur_months / 12.0, 2.0)

        # Small endorsement bonus (capped)
        endorsements = skill.get("endorsements", 0)
        endorse_mult = min(1.0 + endorsements / 80.0, 1.5)

        # Platform assessment score bonus: if the candidate passed a Redrob
        # skill test, that's verified signal — boost by up to 20%
        assess_key = next(
            (k for k in assessment_scores if k.lower() == sname), None
        )
        assess_mult = 1.0
        if assess_key:
            assess_pct = assessment_scores[assess_key] / 100.0  # 0-1
            assess_mult = 1.0 + 0.20 * assess_pct  # up to +20%

        score = base * prof * dur_mult * endorse_mult * assess_mult
        total += score
        max_possible += base * 1.5 * 2.0 * 1.5 * 1.2  # theoretical max

    # Normalize — a perfect candidate gets ~80-90% of max
    if max_possible == 0:
        return 0.0
    return min(100.0, (total / max_possible) * 150)  # scale up so realistic scores hit 80+


def score_experience(candidate: dict) -> float:
    """Returns 0-100. Sweet spot is 5-9 years per JD."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if 5 <= yoe <= 9:
        return 100.0
    elif 4 <= yoe < 5:
        return 85.0
    elif 9 < yoe <= 11:
        return 80.0
    elif 3 <= yoe < 4:
        return 60.0
    elif 11 < yoe <= 13:
        return 65.0
    elif 2 <= yoe < 3:
        return 30.0
    elif 13 < yoe <= 16:
        return 40.0
    else:
        return 5.0


def score_location(candidate: dict) -> float:
    """Returns 0-100. Pune/Noida preferred, India + willing to relocate acceptable."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing = signals.get("willing_to_relocate", False)

    # Check for specific city matches
    for city, score in LOCATION_SCORES.items():
        if city in location:
            return float(score)

    # India but not a known city
    if country == "india":
        return 70.0 if willing else 55.0

    # Outside India
    if willing:
        return 30.0
    return 5.0


def score_education(candidate: dict) -> float:
    """Returns 0-100. Tier 1 institutions get bonus. Relevant fields get bonus.
    Relevant ML/AI certifications add a small bonus (capped at 10 pts)."""
    education = candidate.get("education", [])
    certifications = candidate.get("certifications", [])

    if not education:
        base_edu = 25.0  # Unknown, not penalized heavily
    else:
        base_edu = 0.0
        for edu in education:
            tier = edu.get("tier", "unknown")
            tier_score = EDUCATION_TIER_SCORES.get(tier, 20)

            field = edu.get("field_of_study", "").lower()
            field_bonus = 0
            for f_key, f_bonus in FIELD_BONUS.items():
                if f_key in field:
                    field_bonus = f_bonus
                    break

            combined = min(100, tier_score + field_bonus)
            base_edu = max(base_edu, combined)

    # Certifications bonus: relevant ML/AI certs add credibility
    cert_bonus = 0
    for cert in certifications:
        cert_name = cert.get("name", "").lower()
        for cert_kw, bonus in CERT_BONUSES.items():
            if cert_kw in cert_name:
                cert_bonus += bonus
                break
    cert_bonus = min(cert_bonus, 10)  # cap at +10 pts

    return min(100.0, base_edu + cert_bonus)


def behavioral_multiplier(candidate: dict) -> float:
    """
    Returns 0.4 - 1.0.
    Even a terrible behavioral profile only reduces score by 60%.
    This prevents punishing candidates unfairly while rewarding active ones.
    """
    rs = candidate.get("redrob_signals", {})

    # Availability component (0-1)
    last_active_str = rs.get("last_active_date", "2020-01-01")
    try:
        last_active = date.fromisoformat(last_active_str)
        days_inactive = (TODAY - last_active).days
    except ValueError:
        days_inactive = 365
    recency = max(0.0, 1.0 - days_inactive / 180.0)

    notice = rs.get("notice_period_days", 90)
    notice_score = max(0.0, 1.0 - notice / 90.0)  # 0 days = 1.0, 90+ days = 0.0

    availability = (
        0.5 * float(rs.get("open_to_work_flag", False))
        + 0.3 * recency
        + 0.2 * notice_score
    )

    # Responsiveness component (0-1)
    resp_rate = rs.get("recruiter_response_rate", 0.0)
    avg_resp_hrs = rs.get("avg_response_time_hours", 48)
    resp_speed = max(0.0, 1.0 - avg_resp_hrs / 48.0)  # 0h = 1.0, 48h+ = 0.0

    responsiveness = 0.7 * resp_rate + 0.3 * resp_speed

    # GitHub activity (proxy for technical engagement)
    github = rs.get("github_activity_score", -1)
    github_score = max(0.0, github / 100.0) if github >= 0 else 0.2

    # Interview credibility
    interview_rate = rs.get("interview_completion_rate", 0.5)

    # Profile completeness micro-boost: a 90%+ complete profile gets a small
    # signal boost — indicates serious, engaged candidates
    completeness = rs.get("profile_completeness_score", 50.0)
    completeness_bonus = max(0.0, (completeness - 70.0) / 30.0) * 0.05  # 0 → +0.05

    # Combine
    raw = (
        0.35 * availability
        + 0.30 * responsiveness
        + 0.20 * github_score
        + 0.15 * interview_rate
        + completeness_bonus
    )

    # Scale to 0.4 – 1.0 range
    return min(1.0, 0.4 + 0.6 * raw)


# ─────────────────────────────────────────────
# MAIN SCORER
# ─────────────────────────────────────────────

def score_candidate(candidate: dict) -> float:
    """
    Returns a feature score in [0, 1] for a single candidate.
    Weights: title_career=30%, skills=25%, experience=20%, location=15%, education=10%.
    Then multiplied by behavioral_multiplier [0.4-1.0].

    Title gate: if the raw title score is < 10 (completely wrong domain —
    HR Manager, Accountant, etc.), a hard 0.25x multiplier is applied AFTER
    all other scoring. This prevents experience/location/behavioral signals
    from rescuing a fundamentally wrong-domain candidate.
    """
    t = score_title_career(candidate)
    s = score_skills(candidate)
    e = score_experience(candidate)
    l = score_location(candidate)
    edu = score_education(candidate)

    # Weighted average (all components 0-100)
    feature_score = (
        0.30 * t
        + 0.25 * s
        + 0.20 * e
        + 0.15 * l
        + 0.10 * edu
    ) / 100.0  # normalize to 0-1

    beh = behavioral_multiplier(candidate)

    raw_title_score = _lookup_title_score(
        candidate.get("profile", {}).get("current_title", "").lower().strip()
    )
    # Hard domain gate: completely wrong-domain titles get a severe penalty.
    # Even great experience/location/behavioral signals cannot save an HR Manager.
    title_gate = 0.25 if raw_title_score < 10 else 1.0

    return feature_score * beh * title_gate
