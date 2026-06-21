"""
ranking_engine.py
-------------------
Takes a list of scored candidates and turns them into a ranked,
recruiter-friendly table with fit labels and hiring recommendations.
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


FIT_LABELS = [
    (85, "Excellent Fit"),
    (70, "Strong Fit"),
    (50, "Moderate Fit"),
    (0, "Weak Fit"),
]


def get_fit_label(score: float) -> str:
    """Map a numeric final_score to a human-readable fit label."""
    for threshold, label in FIT_LABELS:
        if score >= threshold:
            return label
    return "Weak Fit"


def get_recommendation(score: float, missing_skills: List[str]) -> str:
    """
    Generate a short, recruiter-friendly recommendation sentence
    based on the score and any missing critical skills.
    """
    label = get_fit_label(score)

    if label == "Excellent Fit":
        return "Highly recommended — schedule an interview as a priority candidate."
    if label == "Strong Fit":
        msg = "Recommended for interview."
        if missing_skills:
            msg += f" Note: candidate may need upskilling in {', '.join(missing_skills[:3])}."
        return msg
    if label == "Moderate Fit":
        return (
            "Consider for interview if the candidate pool is limited. "
            f"Key gaps: {', '.join(missing_skills[:3]) if missing_skills else 'general experience depth'}."
        )
    return "Not recommended at this time — significant skill/experience gaps identified."


def rank_candidates(
    candidates: List[Dict],
    jd_skills: List[str],
) -> pd.DataFrame:
    """
    Build the master ranking table.

    Args:
        candidates: list of dicts, each containing the extracted
            resume info (from extractor.py) MERGED with the score
            breakdown (from skill_matcher.py). Each dict is expected
            to have at least: name, email, phone, skills, final_score,
            skill_match, experience_score, education_score,
            certification_score, semantic_score, keyword_score.
        jd_skills: the list of skills required by the job description,
            used to compute each candidate's "missing skills".

    Returns:
        A pandas DataFrame sorted by final_score descending, with an
        added Rank, Fit Label, and Recommendation column.
    """
    if not candidates:
        return pd.DataFrame()

    rows = []
    jd_skill_set = {s.lower() for s in jd_skills}

    for c in candidates:
        candidate_skills = {s.lower() for s in c.get("skills", [])}
        missing = sorted(jd_skill_set - candidate_skills)

        rows.append({
            "Name": c.get("name", "Unknown"),
            "Email": c.get("email", ""),
            "Phone": c.get("phone", ""),
            "Final Score": c.get("final_score", 0.0),
            "Semantic Score": c.get("semantic_score", 0.0),
            "Keyword Score": c.get("keyword_score", 0.0),
            "Skill Match %": c.get("skill_match", 0.0),
            "Experience Score": c.get("experience_score", 0.0),
            "Education Score": c.get("education_score", 0.0),
            "Certification Score": c.get("certification_score", 0.0),
            "Matched Skills": ", ".join(c.get("skills", [])),
            "Missing Skills": ", ".join(missing) if missing else "None",
            "Fit Label": get_fit_label(c.get("final_score", 0.0)),
            "Recommendation": get_recommendation(c.get("final_score", 0.0), missing),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def get_top_candidates(ranked_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top N candidates from an already-ranked DataFrame."""
    return ranked_df.head(top_n)
