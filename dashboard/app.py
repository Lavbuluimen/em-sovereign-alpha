from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from coverage import render as render_coverage


DATA_DIR = Path("data/processed")


def load_optional_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def render_portfolio():
    st.header("Portfolio")

    portfolio = load_optional_parquet(DATA_DIR / "portfolio_daily.parquet")
    if portfolio is None or portfolio.empty:
        st.info("portfolio_daily.parquet not found.")
        return

    portfolio["date"] = pd.to_datetime(portfolio["date"])
    latest_date = portfolio["date"].max()
    latest = portfolio[portfolio["date"] == latest_date].sort_values("weight", ascending=False)

    st.subheader(f"Latest Portfolio Snapshot — {latest_date.date()}")
    st.dataframe(
        latest[["country", "score", "weight", "hard_w", "local_w", "local_share", "duration_tilt_years"]],
        use_container_width=True,
    )

    fig = px.bar(latest, x="country", y="weight", title="Country Weights")
    st.plotly_chart(fig, use_container_width=True)

    hard_local = pd.DataFrame(
        {
            "bucket": ["Hard Currency", "Local Currency"],
            "weight": [latest["hard_w"].sum(), latest["local_w"].sum()],
        }
    )
    fig2 = px.pie(hard_local, names="bucket", values="weight", title="Hard vs Local Allocation")
    st.plotly_chart(fig2, use_container_width=True)


def render_scores():
    st.header("Scores")

    scores = load_optional_parquet(DATA_DIR / "country_scores_daily.parquet")
    if scores is None or scores.empty:
        st.info("country_scores_daily.parquet not found.")
        return

    scores["date"] = pd.to_datetime(scores["date"])
    latest_date = scores["date"].max()
    latest = scores[scores["date"] == latest_date].sort_values("score", ascending=False)

    st.subheader(f"Latest Scores — {latest_date.date()}")
    score_cols = [c for c in ["country", "score", "score_pct", "score_scaled", "score_raw"] if c in latest.columns]
    st.dataframe(latest[score_cols], use_container_width=True)

    fig = px.bar(latest, x="country", y="score", title="Country Scores")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score History")
    country = st.selectbox("Select a country", sorted(scores["country"].dropna().unique().tolist()))
    hist = scores[scores["country"] == country].sort_values("date")
    fig2 = px.line(hist, x="date", y="score", title=f"Score History — {country}")
    st.plotly_chart(fig2, use_container_width=True)


def render_weekly_actions():
    st.header("Weekly Actions")

    actions = load_optional_parquet(DATA_DIR / "weekly_actions.parquet")
    if actions is None or actions.empty:
        st.info("weekly_actions.parquet not found.")
        return

    if "date" in actions.columns:
        actions["date"] = pd.to_datetime(actions["date"])
        latest_date = actions["date"].max()
        st.subheader(f"Weekly Actions — {latest_date.date()}")

    cols = [c for c in ["country", "action", "weight", "w_change", "hard_w", "local_w", "local_share", "duration_tilt_years"] if c in actions.columns]
    st.dataframe(actions[cols], use_container_width=True)


def main():
    st.set_page_config(page_title="EM Sovereign Dashboard", layout="wide")
    st.title("EM Sovereign Portfolio Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Scores", "Weekly Actions", "Coverage"])

    with tab1:
        render_portfolio()

    with tab2:
        render_scores()

    with tab3:
        render_weekly_actions()

    with tab4:
        render_coverage()


if __name__ == "__main__":
    main()