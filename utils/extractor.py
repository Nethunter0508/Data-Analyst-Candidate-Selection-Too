"""
extractor.py
------------
Pulls structured information out of cleaned resume text:
name, email, phone, location, education, college, experience,
certifications, projects, and technical skills.

Beginner notes:
- We use regex for "pattern shaped" fields (email, phone) because
  these have predictable formats.
- We use SpaCy's Named Entity Recognition (NER) to guess the
  candidate's name and location, since these don't follow a fixed
  pattern but SpaCy's pretrained model recognizes PERSON/GPE entities.
- We use simple section-header detection (e.g. "EDUCATION", "PROJECTS")
  to slice the resume into chunks for education/experience/projects.
- Skills are matched against assets/skills_database.csv using simple
  case-insensitive keyword matching (whole-word, to avoid partial
  matches like "R" inside "Director").
"""

import re
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# --- SpaCy is optional at runtime: app should still work if the model
# isn't downloaded yet, just with reduced name/location detection.
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning(
            "SpaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        )
        _NLP = None
except ImportError:
    logger.warning("spaCy not installed; name/location detection will be limited.")
    _NLP = None


EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{3,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}"
)

# Common resume section headers, used to slice text into chunks.
SECTION_HEADERS = {
    "education": ["education", "academic background", "qualifications"],
    "experience": ["experience", "work experience", "employment history", "professional experience"],
    "certifications": ["certifications", "certificates", "licenses"],
    "projects": ["projects", "academic projects", "personal projects"],
}


def _load_skills_database(csv_path: str) -> List[str]:
    """Load the master skills list from assets/skills_database.csv."""
    try:
        df = pd.read_csv(csv_path)
        return df["skill"].dropna().str.strip().tolist()
    except Exception as e:
        logger.error(f"Could not load skills database at {csv_path}: {e}")
        return []


def extract_email(text: str) -> str:
    """Find the first email-like string in the text."""
    match = EMAIL_REGEX.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    """Find the first phone-number-like string in the text."""
    match = PHONE_REGEX.search(text)
    if not match:
        return ""
    phone = match.group(0).strip()
    # Filter out obvious false positives (too short to be a real number)
    digits_only = re.sub(r"\D", "", phone)
    return phone if len(digits_only) >= 7 else ""


def extract_name(text: str) -> str:
    """
    Guess the candidate's name.
    Strategy:
    1. Try SpaCy NER on the first ~300 characters (names are almost
       always at the very top of a resume).
    2. Fall back to "first non-empty line" if SpaCy isn't available
       or finds nothing.
    """
    header_chunk = text[:300]

    if _NLP is not None:
        doc = _NLP(header_chunk)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()

    # Fallback: first line that looks like a name (no digits, short length)
    for line in text.split("\n"):
        candidate = line.strip()
        if candidate and not any(ch.isdigit() for ch in candidate) and len(candidate.split()) <= 4:
            return candidate

    return "Unknown Candidate"


def extract_location(text: str) -> str:
    """Guess candidate location using SpaCy GPE (geo-political entity) tags."""
    if _NLP is None:
        return ""

    header_chunk = text[:500]
    doc = _NLP(header_chunk)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations[0] if locations else ""


def _extract_section(text: str, keywords: List[str]) -> str:
    """
    Generic helper: finds a section header (e.g. 'EDUCATION') in the
    text and returns the text following it, up until the next section
    header (or end of document).
    """
    lower_text = text.lower()
    all_headers = [kw for group in SECTION_HEADERS.values() for kw in group]

    for kw in keywords:
        idx = lower_text.find(kw)
        if idx == -1:
            continue

        start = idx + len(kw)
        # Find the next header after this one to know where to stop
        end = len(text)
        for other_kw in all_headers:
            if other_kw == kw:
                continue
            other_idx = lower_text.find(other_kw, start)
            if other_idx != -1:
                end = min(end, other_idx)

        return text[start:end].strip(" :\n-")

    return ""


def extract_education(text: str) -> str:
    return _extract_section(text, SECTION_HEADERS["education"])


def extract_experience(text: str) -> str:
    return _extract_section(text, SECTION_HEADERS["experience"])


def extract_certifications(text: str) -> str:
    return _extract_section(text, SECTION_HEADERS["certifications"])


def extract_projects(text: str) -> str:
    return _extract_section(text, SECTION_HEADERS["projects"])


def extract_college(education_text: str) -> str:
    """
    Very simple heuristic: looks for common college/university keywords
    inside the education section text.
    """
    match = re.search(
        r"([A-Z][a-zA-Z.&,'\s]*?(University|College|Institute|Polytechnic)[a-zA-Z.&,'\s]*)",
        education_text,
    )
    return match.group(1).strip() if match else ""


def extract_skills(text: str, skills_csv_path: str) -> List[str]:
    """
    Match resume text against the master skills database.
    Uses whole-word, case-insensitive matching so short skills like
    "R" or "GO" don't accidentally match inside other words.
    """
    skills_master_list = _load_skills_database(skills_csv_path)
    found_skills = []

    lower_text = text.lower()
    for skill in skills_master_list:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, lower_text):
            found_skills.append(skill)

    return sorted(set(found_skills))


def extract_resume_info(text: str, skills_csv_path: str) -> Dict:
    """
    Master function: runs all individual extractors and returns one
    structured dictionary describing the candidate.

    This is the function app.py / ranking_engine.py should call.
    """
    education_text = extract_education(text)

    info = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "location": extract_location(text),
        "education": education_text,
        "college": extract_college(education_text),
        "experience": extract_experience(text),
        "certifications": extract_certifications(text),
        "projects": extract_projects(text),
        "skills": extract_skills(text, skills_csv_path),
        "raw_text": text,
    }
    return info
