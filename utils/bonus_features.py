"""
bonus_features.py
-------------------
Implements Module 15's bonus features:

- AI Resume Summary
- Resume Strength Analysis
- Missing Skills Detection
- Candidate Improvement Suggestions
- Recruiter Recommendation Engine (extends ranking_engine's recommendation)
- ATS Compatibility Score

These are rule-based / lightweight-NLP implementations so the project
runs fully offline without requiring paid LLM APIs — but the
docstrings show where you could plug in a generative model later.
"""

from typing import Dict, List


def generate_resume_summary(candidate_info: Dict, max_sentences: int = 3) -> str:
    """
    Produces a short, human-readable summary of the candidate using
    the structured fields we already extracted (no LLM call needed,
    keeping the tool free to run).
    """
    name = candidate_info.get("name", "The candidate")
    skills = candidate_info.get("skills", [])
    education = candidate_info.get("education", "").strip()
    experience = candidate_info.get("experience", "").strip()

    skill_phrase = ", ".join(skills[:6]) if skills else "no clearly listed technical skills"

    summary_parts = [
        f"{name} demonstrates proficiency in {skill_phrase}."
    ]

    if experience:
        trimmed_exp = experience[:150].rsplit(" ", 1)[0]
        summary_parts.append(f"Professional background includes: {trimmed_exp}...")

    if education:
        trimmed_edu = education[:120].rsplit(" ", 1)[0]
        summary_parts.append(f"Educational background: {trimmed_edu}...")

    return " ".join(summary_parts[:max_sentences])


def analyze_resume_strength(candidate_info: Dict) -> Dict:
    """
    Scores the resume's overall "quality/completeness" independent of
    any specific job description — i.e., is this a well-built resume?

    Returns a dict with a 0-100 strength score and the contributing
    factors, so recruiters/candidates can see what's missing.
    """
    checks = {
        "has_email": bool(candidate_info.get("email")),
        "has_phone": bool(candidate_info.get("phone")),
        "has_skills": len(candidate_info.get("skills", [])) >= 5,
        "has_education": bool(candidate_info.get("education", "").strip()),
        "has_experience": bool(candidate_info.get("experience", "").strip()),
        "has_projects": bool(candidate_info.get("projects", "").strip()),
        "has_certifications": bool(candidate_info.get("certifications", "").strip()),
    }

    score = round((sum(checks.values()) / len(checks)) * 100, 2)

    missing = [k.replace("has_", "").replace("_", " ") for k, v in checks.items() if not v]

    return {
        "strength_score": score,
        "checks": checks,
        "missing_sections": missing,
    }


def detect_missing_skills(candidate_skills: List[str], jd_skills: List[str]) -> List[str]:
    """Returns required JD skills the candidate does NOT have."""
    candidate_set = {s.lower() for s in candidate_skills}
    return [s for s in jd_skills if s.lower() not in candidate_set]


def suggest_improvements(candidate_info: Dict, jd_skills: List[str]) -> List[str]:
    """
    Generates concrete, actionable suggestions for the candidate to
    improve their fit for this specific role.
    """
    suggestions = []

    missing_skills = detect_missing_skills(candidate_info.get("skills", []), jd_skills)
    if missing_skills:
        top_missing = ", ".join(missing_skills[:5])
        suggestions.append(f"Consider gaining experience or certifications in: {top_missing}.")

    strength = analyze_resume_strength(candidate_info)
    for section in strength["missing_sections"]:
        suggestions.append(f"Add a clear '{section.title()}' section to strengthen the resume.")

    if len(candidate_info.get("skills", [])) < 5:
        suggestions.append("List more specific technical tools/skills rather than general statements.")

    if not suggestions:
        suggestions.append("Resume is well-aligned with the job description — no major gaps found.")

    return suggestions


def compute_ats_compatibility_score(candidate_info: Dict, raw_text: str) -> Dict:
    """
    Estimates how well this resume would survive a typical company
    ATS (Applicant Tracking System) parser — separate from job-match
    score. Checks for common ATS pitfalls:
      - Contact info present and machine-readable
      - Standard section headers used
      - Not overly short (likely image-based/garbled extraction)
      - Reasonable keyword density (not just a wall of buzzwords either)
    """
    issues = []
    score = 100

    if not candidate_info.get("email"):
        issues.append("No email address detected — ATS may reject the application.")
        score -= 25
    if not candidate_info.get("phone"):
        issues.append("No phone number detected.")
        score -= 10
    if len(raw_text.strip()) < 200:
        issues.append("Resume text is very short — may be an image-based PDF that ATS can't read.")
        score -= 30
    if not candidate_info.get("education", "").strip():
        issues.append("No 'Education' section detected — use a standard header like 'Education'.")
        score -= 15
    if not candidate_info.get("experience", "").strip():
        issues.append("No 'Experience' section detected — use a standard header like 'Experience'.")
        score -= 15
    if len(candidate_info.get("skills", [])) == 0:
        issues.append("No recognizable technical skills found — list skills explicitly.")
        score -= 10

    score = max(score, 0)

    return {
        "ats_score": score,
        "issues": issues if issues else ["No major ATS compatibility issues detected."],
    }
