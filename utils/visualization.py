"""
visualization.py
------------------
All chart-building functions for the Streamlit dashboard. Centralizing
charts here keeps app.py focused on layout/logic rather than plotting
code.

Uses Plotly for interactive charts (the dashboard's primary charting
library) and Matplotlib/Seaborn for a couple of static fallback charts.
"""

from collections import Counter
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_score_distribution(ranked_df: pd.DataFrame) -> go.Figure:
    """Histogram showing how candidate Final Scores are distributed."""
    fig = px.histogram(
        ranked_df,
        x="Final Score",
        nbins=10,
        title="Candidate Score Distribution",
        labels={"Final Score": "Final Match Score"},
        color_discrete_sequence=["#4C78A8"],
    )
    fig.update_layout(bargap=0.1, yaxis_title="Number of Candidates")
    return fig


def plot_top_candidates_bar(ranked_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of the top N candidates by Final Score."""
    top_df = ranked_df.head(top_n).sort_values("Final Score")
    fig = px.bar(
        top_df,
        x="Final Score",
        y="Name",
        orientation="h",
        title=f"Top {top_n} Candidates by Match Score",
        color="Final Score",
        color_continuous_scale="Blues",
        text="Final Score",
    )
    fig.update_layout(yaxis_title="", xaxis_title="Final Score")
    return fig


def plot_skill_frequency(all_candidate_skills: List[List[str]], top_n: int = 15) -> go.Figure:
    """
    Bar chart showing the most common skills across ALL uploaded
    candidates (helps recruiters see talent-pool strengths/gaps).
    """
    flat_skills = [skill for skills in all_candidate_skills for skill in skills]
    counts = Counter(flat_skills)
    most_common = counts.most_common(top_n)

    if not most_common:
        fig = go.Figure()
        fig.update_layout(title="No skills detected yet")
        return fig

    skills, freq = zip(*most_common)
    fig = px.bar(
        x=list(freq),
        y=list(skills),
        orientation="h",
        title=f"Top {top_n} Most Common Skills in Candidate Pool",
        labels={"x": "Number of Candidates", "y": "Skill"},
        color=list(freq),
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def plot_fit_label_pie(ranked_df: pd.DataFrame) -> go.Figure:
    """Pie chart of candidates grouped by Fit Label."""
    counts = ranked_df["Fit Label"].value_counts().reset_index()
    counts.columns = ["Fit Label", "Count"]
    fig = px.pie(
        counts,
        names="Fit Label",
        values="Count",
        title="Candidate Pool by Fit Label",
        color="Fit Label",
        color_discrete_map={
            "Excellent Fit": "#2E7D32",
            "Strong Fit": "#66BB6A",
            "Moderate Fit": "#FBC02D",
            "Weak Fit": "#E53935",
        },
    )
    return fig


def plot_score_breakdown_radar(candidate_row: pd.Series) -> go.Figure:
    """
    Radar chart showing one candidate's score breakdown across all
    scoring dimensions. Useful in the "Candidate Profile Viewer".
    """
    categories = [
        "Skill Match %", "Experience Score", "Education Score",
        "Certification Score", "Semantic Score",
    ]
    values = [candidate_row.get(cat, 0) for cat in categories]
    # Close the loop for the radar shape
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories, fill="toself", name=candidate_row.get("Name", "")
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Score Breakdown: {candidate_row.get('Name', '')}",
        showlegend=False,
    )
    return fig
