"""
app.py
-------
Main Streamlit entry point for the Data Analyst Candidate Selection Tool.

Run with:
    streamlit run app.py

Workflow:
1. Recruiter uploads a Job Description (txt/pdf/docx) in the sidebar.
2. Recruiter uploads one or more candidate resumes (PDF/DOCX).
3. The app parses, extracts, scores, and ranks every candidate.
4. Results are shown in an interactive dashboard with charts,
   a searchable/filterable ranking table, candidate profile viewer,
   and CSV/Excel export.
"""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import parser, extractor, skill_matcher, ranking_engine, visualization, bonus_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Data Analyst Candidate Selection Tool",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
SKILLS_CSV_PATH = str(BASE_DIR / "assets" / "skills_database.csv")
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def read_uploaded_text(uploaded_file) -> str:
    """Extract clean text from an uploaded PDF/DOCX/TXT file."""
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return parser.clean_text(uploaded_file.read().decode("utf-8", errors="ignore"))
    return parser.detect_and_extract(uploaded_file, uploaded_file.name)


def process_resumes(resume_files, job_description: str, jd_skills: list) -> list:
    """
    Runs the full pipeline (parse -> extract -> score) for every
    uploaded resume and returns a list of merged candidate dicts.
    """
    candidates = []
    progress = st.progress(0, text="Processing resumes...")

    for i, file in enumerate(resume_files):
        try:
            text = parser.detect_and_extract(file, file.name)
            if not text:
                st.warning(f"⚠️ Could not extract text from {file.name}. Skipping.")
                continue

            info = extractor.extract_resume_info(text, SKILLS_CSV_PATH)
            score_breakdown = skill_matcher.compute_final_score(info, job_description, jd_skills)

            candidate = {**info, **score_breakdown, "filename": file.name}
            candidates.append(candidate)

        except Exception as e:
            logger.error(f"Failed processing {file.name}: {e}")
            st.error(f"❌ Error processing {file.name}: {e}")

        progress.progress((i + 1) / len(resume_files), text=f"Processed {file.name}")

    progress.empty()
    return candidates


# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
st.sidebar.title("📊 ATS Navigator")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home & Upload", "📈 Ranking Dashboard", "🔍 Candidate Profile", "📤 Export Reports"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("1️⃣ Job Description")
jd_input_method = st.sidebar.radio("Provide JD via:", ["Upload File", "Paste Text"])

job_description = ""
if jd_input_method == "Upload File":
    jd_file = st.sidebar.file_uploader("Upload Job Description", type=["txt", "pdf", "docx"])
    if jd_file:
        job_description = read_uploaded_text(jd_file)
else:
    job_description = st.sidebar.text_area("Paste Job Description text here", height=200)

st.sidebar.subheader("2️⃣ Candidate Resumes")
resume_files = st.sidebar.file_uploader(
    "Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True
)

run_button = st.sidebar.button("🚀 Run Candidate Matching", use_container_width=True, type="primary")

# Persist results across page navigation using session_state
if "ranked_df" not in st.session_state:
    st.session_state.ranked_df = pd.DataFrame()
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "jd_skills" not in st.session_state:
    st.session_state.jd_skills = []


# ---------------------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------------------
if run_button:
    if not job_description.strip():
        st.sidebar.error("Please provide a Job Description first.")
    elif not resume_files:
        st.sidebar.error("Please upload at least one resume.")
    else:
        with st.spinner("Analyzing job description..."):
            jd_skills = extractor.extract_skills(job_description, SKILLS_CSV_PATH)
            st.session_state.jd_skills = jd_skills

        candidates = process_resumes(resume_files, job_description, jd_skills)
        st.session_state.candidates = candidates

        if candidates:
            st.session_state.ranked_df = ranking_engine.rank_candidates(candidates, jd_skills)
            st.sidebar.success(f"✅ Processed {len(candidates)} candidate(s) successfully.")
        else:
            st.sidebar.error("No candidates could be processed. Check file formats.")


# ---------------------------------------------------------------------
# PAGE: HOME
# ---------------------------------------------------------------------
if page == "🏠 Home & Upload":
    st.title("📊 Data Analyst Candidate Selection Tool")
    st.markdown(
        """
        Welcome! This tool acts like an **AI-powered ATS (Applicant Tracking System)**
        specifically tuned for **Data Analyst** hiring.

        ### How to use it
        1. Add a **Job Description** in the sidebar (upload a file or paste text).
        2. Upload one or more candidate **resumes** (PDF or DOCX).
        3. Click **🚀 Run Candidate Matching**.
        4. Explore results in the **Ranking Dashboard**, drill into individual
           candidates in **Candidate Profile**, and export results from
           **Export Reports**.

        ### What's happening under the hood
        - Resumes are parsed and key fields (name, email, phone, skills, education,
          experience, certifications) are extracted automatically.
        - Candidate skills are compared to the job description using both
          **TF-IDF keyword matching** and **AI semantic similarity**
          (Sentence-Transformers).
        - A weighted **Final Score (0–100)** is computed per candidate and used
          to rank everyone from best to worst fit.
        """
    )

    if job_description:
        with st.expander("📄 Job Description Preview"):
            st.write(job_description[:2000] + ("..." if len(job_description) > 2000 else ""))

    if resume_files:
        st.info(f"📁 {len(resume_files)} resume(s) ready to process. Click **Run Candidate Matching** in the sidebar.")


# ---------------------------------------------------------------------
# PAGE: RANKING DASHBOARD
# ---------------------------------------------------------------------
elif page == "📈 Ranking Dashboard":
    st.title("📈 Candidate Ranking Dashboard")

    df = st.session_state.ranked_df
    if df.empty:
        st.info("No results yet. Upload a job description + resumes and click 'Run Candidate Matching'.")
    else:
        # --- Interactive Filters ---
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("Minimum Final Score", 0, 100, 0)
        with col2:
            fit_filter = st.multiselect(
                "Fit Label", options=df["Fit Label"].unique().tolist(),
                default=df["Fit Label"].unique().tolist(),
            )
        with col3:
            search_term = st.text_input("🔍 Search Candidate by Name")

        filtered_df = df[
            (df["Final Score"] >= min_score) & (df["Fit Label"].isin(fit_filter))
        ]
        if search_term:
            filtered_df = filtered_df[filtered_df["Name"].str.contains(search_term, case=False, na=False)]

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📊 Visual Insights")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(visualization.plot_top_candidates_bar(filtered_df), use_container_width=True)
        with c2:
            st.plotly_chart(visualization.plot_fit_label_pie(filtered_df), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(visualization.plot_score_distribution(filtered_df), use_container_width=True)
        with c4:
            all_skills = [c.get("skills", []) for c in st.session_state.candidates]
            st.plotly_chart(visualization.plot_skill_frequency(all_skills), use_container_width=True)


# ---------------------------------------------------------------------
# PAGE: CANDIDATE PROFILE VIEWER
# ---------------------------------------------------------------------
elif page == "🔍 Candidate Profile":
    st.title("🔍 Candidate Profile Viewer")

    df = st.session_state.ranked_df
    if df.empty:
        st.info("No candidates available yet. Run candidate matching first.")
    else:
        selected_name = st.selectbox("Select a candidate", df["Name"].tolist())
        row = df[df["Name"] == selected_name].iloc[0]
        candidate_info = next(
            (c for c in st.session_state.candidates if c.get("name") == selected_name), {}
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Score", f"{row['Final Score']:.1f}/100")
        col2.metric("Fit Label", row["Fit Label"])
        col3.metric("Skill Match", f"{row['Skill Match %']:.1f}%")

        st.markdown("### 📋 Contact & Background")
        st.write(f"**Email:** {row['Email'] or 'Not found'}")
        st.write(f"**Phone:** {row['Phone'] or 'Not found'}")
        st.write(f"**Matched Skills:** {row['Matched Skills'] or 'None'}")
        st.write(f"**Missing Skills:** {row['Missing Skills']}")

        st.markdown("### 💡 Recruiter Recommendation")
        st.success(row["Recommendation"])

        st.markdown("### 📊 Score Breakdown")
        st.plotly_chart(visualization.plot_score_breakdown_radar(row), use_container_width=True)

        # --- Bonus Features ---
        st.markdown("---")
        st.markdown("### ✨ AI Resume Summary")
        st.write(bonus_features.generate_resume_summary(candidate_info))

        st.markdown("### 💪 Resume Strength Analysis")
        strength = bonus_features.analyze_resume_strength(candidate_info)
        st.write(f"Strength Score: **{strength['strength_score']}/100**")
        if strength["missing_sections"]:
            st.write("Missing sections:", ", ".join(strength["missing_sections"]))

        st.markdown("### 🛠️ Improvement Suggestions")
        for s in bonus_features.suggest_improvements(candidate_info, st.session_state.jd_skills):
            st.write(f"- {s}")

        st.markdown("### ✅ ATS Compatibility Score")
        ats = bonus_features.compute_ats_compatibility_score(
            candidate_info, candidate_info.get("raw_text", "")
        )
        st.write(f"ATS Score: **{ats['ats_score']}/100**")
        for issue in ats["issues"]:
            st.write(f"- {issue}")


# ---------------------------------------------------------------------
# PAGE: EXPORT REPORTS
# ---------------------------------------------------------------------
elif page == "📤 Export Reports":
    st.title("📤 Export Reports")

    df = st.session_state.ranked_df
    if df.empty:
        st.info("No results to export yet. Run candidate matching first.")
    else:
        csv_path = OUTPUTS_DIR / "candidate_ranking.csv"
        excel_path = OUTPUTS_DIR / "candidate_ranking.xlsx"
        top10_path = OUTPUTS_DIR / "top_candidates_report.csv"

        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)
        ranking_engine.get_top_candidates(df, 10).to_csv(top10_path, index=False)

        st.success("Reports generated in the /outputs folder.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "⬇️ Download Full Ranking (CSV)",
                data=df.to_csv(index=False),
                file_name="candidate_ranking.csv",
                mime="text/csv",
            )
        with col2:
            with open(excel_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Full Ranking (Excel)",
                    data=f,
                    file_name="candidate_ranking.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        with col3:
            top10_df = ranking_engine.get_top_candidates(df, 10)
            st.download_button(
                "⬇️ Download Top 10 Report (CSV)",
                data=top10_df.to_csv(index=False),
                file_name="top_candidates_report.csv",
                mime="text/csv",
            )

        st.markdown("### Preview: Top 10 Candidates")
        st.dataframe(ranking_engine.get_top_candidates(df, 10), use_container_width=True, hide_index=True)
