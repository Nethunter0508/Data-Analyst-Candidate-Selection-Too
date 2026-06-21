# 📊 Data Analyst Candidate Selection Tool

An AI-powered ATS (Applicant Tracking System) that automatically parses resumes, extracts candidate information, matches candidates against a job description using **TF-IDF keyword matching** + **Sentence-Transformer semantic similarity**, and ranks candidates through an interactive Streamlit dashboard.

---

## 🚀 Features

- 📄 **Resume Parsing** — Upload PDF/DOCX resumes; text is extracted with PyPDF2, pdfminer.six, and python-docx (with automatic fallback for tricky PDFs).
- 🧠 **Information Extraction** — Name, email, phone, location, education, college, experience, certifications, projects, and skills extracted using Regex + spaCy NER.
- 🛠️ **100+ Skill Database** — A curated CSV of Data Analyst skills (Python, SQL, Power BI, Tableau, Cloud platforms, ML, and more).
- 🎯 **Hybrid Matching Engine**
  - Keyword Score = 40% Skill Match + 30% Experience + 20% Education + 10% Certifications
  - Final Score = 0.6 × Semantic Score (Sentence-Transformers `all-MiniLM-L6-v2`) + 0.4 × Keyword Score
- 🏆 **Candidate Ranking Engine** — Sorts all candidates, assigns fit labels (Excellent / Strong / Moderate / Weak Fit), and generates recruiter recommendations.
- 📈 **Interactive Dashboard** — Ranking table, score distribution, top-candidates chart, skill-frequency chart, fit-label pie chart, and a per-candidate radar breakdown — all built with Plotly.
- 🔍 **Candidate Profile Viewer** — Deep dive into any single candidate with an AI-style summary, strength analysis, missing-skill detection, and improvement suggestions.
- ✅ **ATS Compatibility Score** — Flags formatting/parsing issues that would hurt a resume in a real-world ATS.
- 📤 **Exports** — Download full rankings and Top-10 reports as CSV or Excel.
- 🛡️ **Robust Error Handling** — Corrupted files, unsupported formats, and empty uploads are all handled gracefully with user-friendly messages and logging.

---

## 🧰 Technology Stack

| Layer | Tools |
|---|---|
| UI / Dashboard | Streamlit, Plotly |
| NLP / Extraction | spaCy, NLTK, Regex |
| Document Parsing | PyPDF2, pdfminer.six, python-docx |
| Matching | Scikit-Learn (TF-IDF, Cosine Similarity), Sentence-Transformers |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |

---

## 🏗️ Architecture

```
Resume (PDF/DOCX)        Job Description (txt/pdf/docx)
       │                            │
       ▼                            ▼
 utils/parser.py            utils/extractor.py
 (extract raw text)         (extract JD required skills)
       │                            │
       ▼                            │
 utils/extractor.py                 │
 (name, email, skills, etc.)        │
       │                            │
       └────────────┬───────────────┘
                     ▼
           utils/skill_matcher.py
   (TF-IDF + Semantic Similarity + Weighted Score)
                     ▼
           utils/ranking_engine.py
        (rank, fit labels, recommendations)
                     ▼
              app.py (Streamlit UI)
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
 Ranking Table   Charts (viz.py)  Exports (CSV/Excel)
```

---

## 📂 Project Structure

```
DataAnalystCandidateSelectionTool/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── resumes/
│   └── job_descriptions/
├── models/
├── utils/
│   ├── parser.py
│   ├── extractor.py
│   ├── skill_matcher.py
│   ├── ranking_engine.py
│   ├── visualization.py
│   └── bonus_features.py
├── assets/
│   └── skills_database.csv
└── outputs/
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/DataAnalystCandidateSelectionTool.git
cd DataAnalystCandidateSelectionTool

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy English model (needed for name/location NER)
python -m spacy download en_core_web_sm
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` Sentence-Transformers model (~80MB) automatically. If it can't be downloaded (e.g. no internet), the app automatically falls back to TF-IDF-only matching so it keeps working.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in your browser.

1. In the sidebar, upload or paste a **Job Description**.
2. Upload one or more candidate **resumes** (PDF/DOCX).
3. Click **🚀 Run Candidate Matching**.
4. Browse the **Ranking Dashboard**, **Candidate Profile**, and **Export Reports** pages.

A sample job description and resume are included in `data/job_descriptions/` and `data/resumes/` so you can try the tool immediately.

---

## 📸 Screenshots

> Add your own screenshots here after running the app locally:


---

## 🔮 Future Enhancements

- Integrate a generative LLM for richer AI resume summaries and cover-letter style insights.
- Add multi-language resume support.
- Add authentication for multi-recruiter, multi-job-description workspaces.
- Persist results in a database (PostgreSQL) instead of session state.
- Add a REST API layer so other HR tools can call the scoring engine.

---

## 💼 LinkedIn Project Description

> 🚀 Built an **AI-Powered Data Analyst Candidate Selection Tool** — an end-to-end ATS that parses resumes (PDF/DOCX), extracts structured candidate data with NLP (spaCy, Regex), and ranks candidates against any job description using a hybrid **TF-IDF + Sentence-Transformer semantic similarity** scoring engine. Built a full interactive **Streamlit dashboard** with Plotly visualizations, candidate profile drill-downs, ATS-compatibility scoring, and one-click CSV/Excel exports. #DataScience #Python #NLP #MachineLearning #Streamlit #ATS

---

## 📝 Resume Bullet Points

- Designed and built an end-to-end AI-powered ATS in Python that parses PDF/DOCX resumes, extracts structured candidate data via spaCy NER and Regex, and ranks candidates using a hybrid TF-IDF + Sentence-Transformer semantic matching engine.
- Engineered a weighted scoring system (skills, experience, education, certifications) achieving transparent, explainable 0–100 candidate-job fit scores.
- Developed an interactive Streamlit dashboard with Plotly visualizations (ranking tables, score distributions, skill-frequency charts) supporting real-time candidate filtering, search, and CSV/Excel export.
- Implemented robust error handling and logging for corrupted files, unsupported formats, and malformed resumes, ensuring production-grade reliability.

---

## 🏷️ GitHub Tags

`python` `streamlit` `nlp` `machine-learning` `ats` `resume-parser` `data-analyst` `sentence-transformers` `tf-idf` `spacy` `plotly` `hr-tech` `recruitment-tech` `portfolio-project`

---

## 📄 License

This project is open-source and available under the MIT License.
