"""
test_ranker.py

Quick validation script. Run this to confirm your scorer logic works
correctly on a few representative candidates BEFORE running on 100K.

Checks:
  1. Keyword stuffer scores LOW
  2. Good ML engineer scores HIGH
  3. Honeypots are correctly detected
  4. Behavioral dead candidates score lower than their active twins
  5. Career trajectory bonus works
  6. Reasoning fix: no truncated strings for top candidates
  7. New honeypot flags (FLAG 7: bulk zero-duration Tier-1 skills)
  8. Verified identity signals affect behavioral multiplier
  9. Score ratio: good ML vs keyword stuffer > 5x (raised bar from 4x)

Usage:
  python test_ranker.py
"""

import json
from scorer import (
    score_candidate, score_title_career, score_skills,
    behavioral_multiplier, score_career_trajectory,
)
from honeypot_detector import is_honeypot
from reasoning_gen import generate_reasoning

# ─────────────────────────────────────────────
# TEST CANDIDATES (synthetic mini-profiles)
# ─────────────────────────────────────────────

KEYWORD_STUFFER = {
    "candidate_id": "CAND_TEST001",
    "profile": {
        "anonymized_name": "Test Person",
        "headline": "HR Manager | AI Enthusiast",
        "summary": "Passionate about AI transformation and digital innovation.",
        "location": "Pune",
        "country": "India",
        "years_of_experience": 7.0,
        "current_title": "HR Manager",
        "current_company": "MegaCorp",
        "current_company_size": "1001-5000",
        "current_industry": "IT Services",
    },
    "career_history": [
        {
            "company": "MegaCorp",
            "title": "HR Manager",
            "start_date": "2019-01-01",
            "end_date": None,
            "duration_months": 65,
            "is_current": True,
            "industry": "IT Services",
            "company_size": "1001-5000",
            "description": "Managed hiring processes and employee relations. Led digital transformation initiatives.",
        }
    ],
    "education": [{"institution": "Mumbai University", "degree": "MBA", "field_of_study": "Human Resources", "start_year": 2015, "end_year": 2017, "grade": None, "tier": "tier_3"}],
    "skills": [
        {"name": "RAG", "proficiency": "advanced", "endorsements": 40, "duration_months": 0},
        {"name": "Embeddings", "proficiency": "advanced", "endorsements": 35, "duration_months": 0},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 50, "duration_months": 0},
        {"name": "Semantic Search", "proficiency": "expert", "endorsements": 45, "duration_months": 0},
        {"name": "NLP", "proficiency": "advanced", "endorsements": 30, "duration_months": 0},
    ],
    "certifications": [],
    "redrob_signals": {
        "profile_completeness_score": 82.0,
        "signup_date": "2024-01-01",
        "last_active_date": "2026-05-15",
        "open_to_work_flag": True,
        "profile_views_received_30d": 50,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.80,
        "avg_response_time_hours": 5.0,
        "skill_assessment_scores": {},
        "connection_count": 400,
        "endorsements_received": 90,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 15.0, "max": 25.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": False,
        "github_activity_score": -1,
        "search_appearance_30d": 200,
        "saved_by_recruiters_30d": 8,
        "interview_completion_rate": 0.90,
        "offer_acceptance_rate": 0.70,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}

GOOD_ML_ENGINEER = {
    "candidate_id": "CAND_TEST002",
    "profile": {
        "anonymized_name": "Good Engineer",
        "headline": "ML Engineer | Embeddings & Retrieval | Fintech",
        "summary": "7 years building production ML systems. Led retrieval stack at a fintech company — embedding-based search, hybrid BM25+dense, Pinecone at scale. Strong Python, shipped models to millions of users.",
        "location": "Pune",
        "country": "India",
        "years_of_experience": 7.2,
        "current_title": "ML Engineer",
        "current_company": "RazorpayML",
        "current_company_size": "1001-5000",
        "current_industry": "Fintech",
    },
    "career_history": [
        {
            "company": "RazorpayML",
            "title": "ML Engineer",
            "start_date": "2021-06-01",
            "end_date": None,
            "duration_months": 60,
            "is_current": True,
            "industry": "Fintech",
            "company_size": "1001-5000",
            "description": "Built and deployed embedding-based retrieval systems for merchant and transaction search. Designed hybrid search pipeline combining BM25 with dense retrievers (sentence-transformers) indexed in Pinecone. Implemented NDCG-based offline evaluation framework. Fine-tuned E5 models on domain-specific data.",
        },
        {
            "company": "Startup XYZ",
            "title": "Software Engineer",
            "start_date": "2018-07-01",
            "end_date": "2021-05-01",
            "duration_months": 34,
            "is_current": False,
            "industry": "SaaS",
            "company_size": "51-200",
            "description": "Built backend APIs and data pipelines in Python. Early NLP work on text classification for content moderation.",
        },
    ],
    "education": [{"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "Computer Science", "start_year": 2014, "end_year": 2018, "grade": "8.9 CGPA", "tier": "tier_1"}],
    "skills": [
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 60, "duration_months": 48},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 45, "duration_months": 36},
        {"name": "Semantic Search", "proficiency": "advanced", "endorsements": 40, "duration_months": 48},
        {"name": "BM25", "proficiency": "advanced", "endorsements": 30, "duration_months": 30},
        {"name": "Sentence Transformers", "proficiency": "advanced", "endorsements": 35, "duration_months": 42},
        {"name": "Python", "proficiency": "expert", "endorsements": 80, "duration_months": 84},
        {"name": "PyTorch", "proficiency": "advanced", "endorsements": 40, "duration_months": 36},
        {"name": "NLP", "proficiency": "advanced", "endorsements": 50, "duration_months": 60},
        {"name": "Information Retrieval", "proficiency": "advanced", "endorsements": 35, "duration_months": 36},
    ],
    "certifications": [
        {"name": "Deep Learning Specialization", "issuer": "Coursera", "year": 2021}
    ],
    "redrob_signals": {
        "profile_completeness_score": 94.0,
        "signup_date": "2024-03-01",
        "last_active_date": "2026-06-10",
        "open_to_work_flag": True,
        "profile_views_received_30d": 120,
        "applications_submitted_30d": 5,
        "recruiter_response_rate": 0.75,
        "avg_response_time_hours": 8.0,
        "skill_assessment_scores": {"Python": 88.0, "NLP": 82.0},
        "connection_count": 520,
        "endorsements_received": 200,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 35.0, "max": 55.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": False,
        "github_activity_score": 72.0,
        "search_appearance_30d": 350,
        "saved_by_recruiters_30d": 18,
        "interview_completion_rate": 0.95,
        "offer_acceptance_rate": 0.60,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}

HONEYPOT = {
    "candidate_id": "CAND_TEST003",
    "profile": {
        "anonymized_name": "Impossible Person",
        "headline": "Expert ML Engineer",
        "summary": "Expert in everything.",
        "location": "Bangalore",
        "country": "India",
        "years_of_experience": 3.0,  # Only 3 years
        "current_title": "ML Engineer",
        "current_company": "FakeCorp",
        "current_company_size": "51-200",
        "current_industry": "AI/ML",
    },
    "career_history": [
        {
            "company": "FakeCorp",
            "title": "ML Engineer",
            "start_date": "2021-01-01",
            "end_date": None,
            "duration_months": 80,  # 80 months but only 3 YOE — impossible!
            "is_current": True,
            "industry": "AI/ML",
            "company_size": "51-200",
            "description": "Built everything.",
        }
    ],
    "education": [{"institution": "IIT Delhi", "degree": "B.Tech", "field_of_study": "CS", "start_year": 2018, "end_year": 2022, "grade": None, "tier": "tier_1"}],
    "skills": [
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 100, "duration_months": 0},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 90, "duration_months": 0},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 80, "duration_months": 0},
        {"name": "RAG", "proficiency": "expert", "endorsements": 70, "duration_months": 0},
    ],
    "certifications": [],
    "redrob_signals": {
        "profile_completeness_score": 90.0,
        "signup_date": "2024-01-01",
        "last_active_date": "2026-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 50,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.80,
        "avg_response_time_hours": 5.0,
        "skill_assessment_scores": {},
        "connection_count": 300,
        "endorsements_received": 100,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 50.0,
        "search_appearance_30d": 200,
        "saved_by_recruiters_30d": 10,
        "interview_completion_rate": 0.90,
        "offer_acceptance_rate": 0.80,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}

# Keyword stuffer with FLAG 7: 4+ Tier-1 core skills all with 0 duration
KEYWORD_STUFFER_TIER1 = {
    "candidate_id": "CAND_TEST007",
    "profile": {
        "anonymized_name": "Tier1 Stuffer",
        "headline": "AI Expert",
        "summary": "Expert in RAG, embeddings, Pinecone, FAISS, semantic search.",
        "location": "Mumbai",
        "country": "India",
        "years_of_experience": 5.0,
        "current_title": "Software Engineer",
        "current_company": "SomeCorp",
        "current_company_size": "201-500",
        "current_industry": "Software",
    },
    "career_history": [
        {
            "company": "SomeCorp",
            "title": "Software Engineer",
            "start_date": "2021-01-01",
            "end_date": None,
            "duration_months": 60,
            "is_current": True,
            "industry": "Software",
            "company_size": "201-500",
            "description": "General software development.",
        }
    ],
    "education": [],
    "skills": [
        {"name": "RAG", "proficiency": "expert", "endorsements": 20, "duration_months": 0},
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 18, "duration_months": 0},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 22, "duration_months": 0},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 15, "duration_months": 0},
        {"name": "Semantic Search", "proficiency": "advanced", "endorsements": 10, "duration_months": 0},
    ],
    "certifications": [],
    "redrob_signals": {
        "profile_completeness_score": 60.0,
        "signup_date": "2024-01-01",
        "last_active_date": "2026-05-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 10,
        "applications_submitted_30d": 1,
        "recruiter_response_rate": 0.5,
        "avg_response_time_hours": 10.0,
        "skill_assessment_scores": {},
        "connection_count": 100,
        "endorsements_received": 20,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 15.0, "max": 25.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": False,
        "github_activity_score": 20.0,
        "search_appearance_30d": 50,
        "saved_by_recruiters_30d": 2,
        "interview_completion_rate": 0.70,
        "offer_acceptance_rate": 0.50,
        "verified_email": True,
        "verified_phone": False,
        "linkedin_connected": False,
    },
}

# Career trajectory test: progressive ML career
CAREER_TRAJECTORY_CANDIDATE = {
    **GOOD_ML_ENGINEER,
    "candidate_id": "CAND_TEST008",
    "career_history": [
        {
            "company": "DeepMind India",
            "title": "Senior ML Engineer",
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": 53,
            "is_current": True,
            "industry": "AI/ML",
            "company_size": "1001-5000",
            "description": "Leading retrieval and ranking for candidate search systems.",
        },
        {
            "company": "Razorpay",
            "title": "ML Engineer",
            "start_date": "2019-06-01",
            "end_date": "2022-01-01",
            "duration_months": 31,
            "is_current": False,
            "industry": "Fintech",
            "company_size": "1001-5000",
            "description": "Built embedding-based retrieval systems.",
        },
        {
            "company": "Startup A",
            "title": "Data Scientist",
            "start_date": "2017-01-01",
            "end_date": "2019-06-01",
            "duration_months": 29,
            "is_current": False,
            "industry": "SaaS",
            "company_size": "51-200",
            "description": "NLP and text classification.",
        },
    ],
}

BEHAVIORAL_DEAD = {**GOOD_ML_ENGINEER, "candidate_id": "CAND_TEST004"}
BEHAVIORAL_DEAD["redrob_signals"] = {
    **GOOD_ML_ENGINEER["redrob_signals"],
    "last_active_date": "2025-10-01",   # 8 months inactive
    "open_to_work_flag": False,
    "recruiter_response_rate": 0.04,
    "avg_response_time_hours": 120.0,
    "github_activity_score": 2.0,
    "interview_completion_rate": 0.30,
    "verified_email": False,
    "verified_phone": False,
    "linkedin_connected": False,
    "offer_acceptance_rate": 0.1,
}

VERIFIED_CANDIDATE = {**GOOD_ML_ENGINEER, "candidate_id": "CAND_TEST009"}
VERIFIED_CANDIDATE["redrob_signals"] = {
    **GOOD_ML_ENGINEER["redrob_signals"],
    "verified_email": True,
    "verified_phone": True,
    "linkedin_connected": True,
    "saved_by_recruiters_30d": 20,
    "profile_views_received_30d": 150,
}
UNVERIFIED_CANDIDATE = {**GOOD_ML_ENGINEER, "candidate_id": "CAND_TEST010"}
UNVERIFIED_CANDIDATE["redrob_signals"] = {
    **GOOD_ML_ENGINEER["redrob_signals"],
    "verified_email": False,
    "verified_phone": False,
    "linkedin_connected": False,
    "saved_by_recruiters_30d": 0,
    "profile_views_received_30d": 5,
}


# ─────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────

def run_tests():
    print("\n" + "═" * 65)
    print("  REDROB RANKER — UNIT TESTS v2")
    print("═" * 65)

    tests_passed = 0
    tests_total = 0

    def check(name, condition, expected, got):
        nonlocal tests_passed, tests_total
        tests_total += 1
        status = "✅ PASS" if condition else "❌ FAIL"
        if condition:
            tests_passed += 1
        print(f"  {status}  {name}")
        if not condition:
            print(f"         Expected: {expected}")
            if isinstance(got, float):
                print(f"         Got:      {got:.4f}")
            else:
                print(f"         Got:      {got}")

    # ── Test 1: Keyword stuffer scores low ──
    stuffer_score = score_candidate(KEYWORD_STUFFER)
    stuffer_title = score_title_career(KEYWORD_STUFFER)
    stuffer_skills = score_skills(KEYWORD_STUFFER)
    print(f"\n[Keyword Stuffer] title={stuffer_title:.1f}, skills={stuffer_skills:.1f}, total={stuffer_score:.4f}")
    check("Keyword stuffer final score < 0.15", stuffer_score < 0.15, "< 0.15", stuffer_score)
    check("Keyword stuffer title score <= 5 (HR Manager)", stuffer_title <= 5, "<= 5", stuffer_title)
    check("Keyword stuffer skills score < 5 (0-duration)", stuffer_skills < 5, "< 5", stuffer_skills)

    # ── Test 2: Good ML Engineer scores high ──
    good_score = score_candidate(GOOD_ML_ENGINEER)
    good_title = score_title_career(GOOD_ML_ENGINEER)
    good_skills = score_skills(GOOD_ML_ENGINEER)
    good_beh = behavioral_multiplier(GOOD_ML_ENGINEER)
    print(f"\n[Good ML Engineer] title={good_title:.1f}, skills={good_skills:.1f}, beh={good_beh:.3f}, total={good_score:.4f}")
    check("Good ML engineer final score > 0.65", good_score > 0.65, "> 0.65", good_score)
    check("Good ML engineer title score > 90", good_title > 90, "> 90", good_title)
    check("Good ML engineer skills > 50", good_skills > 50, "> 50", good_skills)
    check("Good ML engineer behavioral > 0.80", good_beh > 0.80, "> 0.80", good_beh)

    # ── Test 3: Honeypot detected (original — impossible timeline) ──
    hp_detected = is_honeypot(HONEYPOT)
    tests_total += 1
    if hp_detected:
        tests_passed += 1
        print(f"\n  ✅ PASS  Honeypot correctly detected (FLAG 1: career_months >> yoe_months)")
    else:
        print(f"\n  ❌ FAIL  Honeypot NOT detected (career_months=80, yoe_months=36)")

    # ── Test 4: Good engineer NOT flagged as honeypot ──
    good_hp = is_honeypot(GOOD_ML_ENGINEER)
    tests_total += 1
    if not good_hp:
        tests_passed += 1
        print(f"  ✅ PASS  Good ML engineer not falsely flagged as honeypot")
    else:
        print(f"  ❌ FAIL  Good ML engineer falsely flagged as honeypot!")

    # ── Test 5: Behavioral dead vs active twin ──
    active_score = score_candidate(GOOD_ML_ENGINEER)
    dead_score = score_candidate(BEHAVIORAL_DEAD)
    ratio_beh = active_score / dead_score if dead_score > 0 else float("inf")
    print(f"\n[Behavioral Twin] active={active_score:.4f}, inactive={dead_score:.4f}, ratio={ratio_beh:.2f}x")
    check("Active twin scores 30%+ higher than behavioral dead",
          active_score > dead_score * 1.30, "> 1.30x dead", ratio_beh)

    # ── Test 6: Score ratio — raised bar to 5x ──
    ratio = good_score / stuffer_score if stuffer_score > 0 else float("inf")
    print(f"\n[Score ratio] good_ml / keyword_stuffer = {ratio:.1f}x")
    check("Good ML engineer scores 5x+ vs keyword stuffer",
          good_score > stuffer_score * 5, "> 5x stuffer", ratio)

    # ── Test 7: Reasoning quality (no truncated strings) ──
    mock_scores = {"combined": good_score, "feature": 0.8, "semantic": 0.7, "_norm_score": 0.95}
    reasoning_top = generate_reasoning(GOOD_ML_ENGINEER, 1, mock_scores)
    reasoning_low = generate_reasoning(KEYWORD_STUFFER, 85, {**mock_scores, "combined": 0.1, "_norm_score": 0.55})
    print(f"\n[Reasoning @rank1]:  {reasoning_top}")
    print(f"[Reasoning @rank85]: {reasoning_low}")
    tests_total += 1
    if len(reasoning_top) > 50 and len(reasoning_top) <= 250:
        tests_passed += 1
        print(f"  ✅ PASS  Reasoning length OK ({len(reasoning_top)} chars)")
    else:
        print(f"  ❌ FAIL  Reasoning length problem ({len(reasoning_top)} chars)")

    # Check reasoning doesn't start with a semicolon (the bug we fixed)
    tests_total += 1
    if not reasoning_top.startswith(";") and "ML Engineer" in reasoning_top:
        tests_passed += 1
        print(f"  ✅ PASS  Reasoning starts correctly (no leading semicolon, contains title)")
    else:
        print(f"  ❌ FAIL  Reasoning has formatting issue: '{reasoning_top[:40]}'")

    # ── Test 8: NEW — FLAG 7 honeypot (bulk zero-duration Tier-1 skills) ──
    hp7_detected = is_honeypot(KEYWORD_STUFFER_TIER1)
    tests_total += 1
    if hp7_detected:
        tests_passed += 1
        print(f"\n  ✅ PASS  FLAG 7 honeypot correctly detected (5 Tier-1 skills all 0 duration)")
    else:
        print(f"\n  ❌ FAIL  FLAG 7 honeypot NOT detected (5 Tier-1 skills all 0 duration)")

    # ── Test 9: Career trajectory bonus ──
    traj_base = score_career_trajectory(GOOD_ML_ENGINEER)    # 2 roles, 1 ML
    traj_rich  = score_career_trajectory(CAREER_TRAJECTORY_CANDIDATE)  # 3 ML roles
    print(f"\n[Career Trajectory] base(2 roles)={traj_base:.1f}, rich(3 ML roles)={traj_rich:.1f}")
    check("Career trajectory: 3-role ML career scores higher than 2-role",
          traj_rich > traj_base, f"> {traj_base:.1f}", traj_rich)
    check("Career trajectory: multi-ML bonus >= 10", traj_rich >= 10, ">= 10", traj_rich)

    # ── Test 10: Verified identity signals affect behavioral multiplier ──
    beh_verified   = behavioral_multiplier(VERIFIED_CANDIDATE)
    beh_unverified = behavioral_multiplier(UNVERIFIED_CANDIDATE)
    print(f"\n[Identity Signals] verified={beh_verified:.4f}, unverified={beh_unverified:.4f}")
    check("Verified identity scores higher than unverified",
          beh_verified > beh_unverified, f"> {beh_unverified:.4f}", beh_verified)

    # ── Summary ──
    print(f"\n{'═'*65}")
    print(f"  Results: {tests_passed}/{tests_total} tests passed")
    if tests_passed == tests_total:
        print("  🎉 All tests passed! Ready to run on full dataset.")
    else:
        failed = tests_total - tests_passed
        print(f"  ⚠️  {failed} test(s) failed. Review the scorer logic above.")
    print("═" * 65 + "\n")

    return tests_passed == tests_total


if __name__ == "__main__":
    success = run_tests()
    import sys
    sys.exit(0 if success else 1)
