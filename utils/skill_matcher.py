"""
skill_matcher.py
-----------------
The scoring brain of the tool. Combines several signals into one
final 0-100 match score per candidate:

  Keyword Score (TF-IDF + cosine similarity + skill overlap)
        combined with
  Semantic Score (Sentence-Transformers embeddings)
        combined with
  Experience / Education / Certification weighting

Final formulas (as specified by the project spec):

  Keyword-based composite score:
      40% Skill Match
    + 30% Experience
    + 20% Education
    + 10% Certifications

  Overall Final Score:
      0.6 * Semantic Score
    + 0.4 * Keyword Score
"""

import logging
import re
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Sentence-Transformers model is loaded lazily (only once) because
# loading it is somewhat slow. If the package/model isn't available,
# we gracefully degrade to keyword-only scoring.
_SEMANTIC_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_semantic_model():
    """Lazily load (and cache) the Sentence-Transformers model."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEMANTIC_MODEL = SentenceTransformer(_MODEL_NAME)
        except Exception as e:
            logger.warning(f"Could not load Sentence-Transformers model: {e}")
            _SEMANTIC_MODEL = False  # sentinel meaning "unavailable"
    return _SEMANTIC_MODEL if _SEMANTIC_MODEL is not False else None


def compute_tfidf_similarity(resume_text: str, job_description: str) -> float:
    """
    Classic keyword-overlap similarity: turns both texts into TF-IDF
    vectors and measures the cosine angle between them.
    Returns a score between 0 and 100.
    """
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity) * 100, 2)
    except Exception as e:
        logger.error(f"TF-IDF similarity failed: {e}")
        return 0.0


def compute_skill_overlap(candidate_skills: List[str], jd_skills: List[str]) -> float:
    """
    What fraction of the job description's required skills does the
    candidate actually have? Returns a 0-100 percentage.
    """
    if not jd_skills:
        return 0.0

    candidate_set = {s.lower() for s in candidate_skills}
    jd_set = {s.lower() for s in jd_skills}

    matched = candidate_set.intersection(jd_set)
    return round((len(matched) / len(jd_set)) * 100, 2)


def compute_semantic_similarity(resume_text: str, job_description: str) -> float:
    """
    Uses Sentence-Transformers (all-MiniLM-L6-v2) to embed both the
    resume and job description into vectors, then measures cosine
    similarity. Captures *meaning*, not just exact keyword overlap
    (e.g. "built dashboards" vs "data visualization").
    """
    model = _get_semantic_model()
    if model is None:
        # Graceful fallback: if the model can't be loaded, reuse TF-IDF
        # so the app still works (just slightly less "semantic").
        return compute_tfidf_similarity(resume_text, job_description)

    if not resume_text.strip() or not job_description.strip():
        return 0.0

    try:
        embeddings = model.encode([job_description, resume_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        # Cosine similarity from sentence embeddings is roughly 0-1;
        # clip to be safe before scaling to 0-100.
        similarity = max(0.0, min(1.0, float(similarity)))
        return round(similarity * 100, 2)
    except Exception as e:
        logger.error(f"Semantic similarity failed: {e}")
        return compute_tfidf_similarity(resume_text, job_description)


def _experience_score(experience_text: str) -> float:
    """
    Rough proxy for experience: counts the number of years mentioned
    (e.g. '3 years', '2+ years') and scales to 0-100.
    More years -> higher score, capped at 10 years = 100.
    """
    if not experience_text:
        return 0.0

    years_found = re.findall(r"(\d+)\+?\s*(?:years|yrs)", experience_text.lower())
    if not years_found:
        # No explicit year count, but some experience text exists ->
        # give partial credit just for having a populated section.
        return 30.0 if len(experience_text.strip()) > 20 else 0.0

    max_years = max(int(y) for y in years_found)
    return round(min(max_years / 10.0, 1.0) * 100, 2)


def _education_score(education_text: str) -> float:
    """
    Rough proxy for education: checks for degree keywords and assigns
    tiered scores (PhD > Master's > Bachelor's > other).
    """
    if not education_text:
        return 0.0

    text = education_text.lower()
    if "phd" in text or "doctorate" in text:
        return 100.0
    if "master" in text or "m.sc" in text or "msc" in text or "mba" in text:
        return 85.0
    if "bachelor" in text or "b.sc" in text or "bsc" in text or "b.tech" in text or "btech" in text:
        return 70.0
    if len(text.strip()) > 10:
        return 40.0  # some education info, but unclear level
    return 0.0


def _certification_score(certifications_text: str) -> float:
    """
    Simple proxy: more certification mentions -> higher score,
    capped at 5 certifications = 100.
    """
    if not certifications_text:
        return 0.0

    # Split on common delimiters to roughly count distinct certifications
    items = [c for c in re.split(r"[\n,;•]", certifications_text) if c.strip()]
    return round(min(len(items) / 5.0, 1.0) * 100, 2)


def compute_keyword_score(
    candidate_info: Dict,
    job_description: str,
    jd_skills: List[str],
) -> Dict[str, float]:
    """
    Combines skill overlap, experience, education, and certifications
    into the weighted "Keyword Score" per the spec:

        40% Skill Match + 30% Experience + 20% Education + 10% Certifications

    Returns a dict with the breakdown AND the combined keyword_score,
    so the UI can show candidates *why* they scored what they did.
    """
    skill_match = compute_skill_overlap(candidate_info.get("skills", []), jd_skills)
    experience = _experience_score(candidate_info.get("experience", ""))
    education = _education_score(candidate_info.get("education", ""))
    certifications = _certification_score(candidate_info.get("certifications", ""))

    keyword_score = (
        0.40 * skill_match
        + 0.30 * experience
        + 0.20 * education
        + 0.10 * certifications
    )

    return {
        "skill_match": round(skill_match, 2),
        "experience_score": round(experience, 2),
        "education_score": round(education, 2),
        "certification_score": round(certifications, 2),
        "keyword_score": round(keyword_score, 2),
    }


def compute_final_score(
    candidate_info: Dict,
    job_description: str,
    jd_skills: List[str],
) -> Dict:
    """
    Master scoring function. Produces the final 0-100 candidate score
    using:

        Final Score = 0.6 * Semantic Score + 0.4 * Keyword Score

    Returns a full breakdown dictionary for transparency/auditability,
    which the dashboard can display to recruiters.
    """
    resume_text = candidate_info.get("raw_text", "")

    keyword_breakdown = compute_keyword_score(candidate_info, job_description, jd_skills)
    semantic_score = compute_semantic_similarity(resume_text, job_description)

    final_score = 0.6 * semantic_score + 0.4 * keyword_breakdown["keyword_score"]
    final_score = round(min(final_score, 100.0), 2)

    breakdown = {
        **keyword_breakdown,
        "semantic_score": semantic_score,
        "final_score": final_score,
    }
    return breakdown
