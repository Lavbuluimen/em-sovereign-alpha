from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_DIR = Path("data/processed")


def load_optional_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def coverage_table(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    if group_col and group_col in work.columns:
        numeric_like = [c for c in work.columns if c != group_col]
        cov = (
            work.groupby(group_col)[numeric_like]
            .apply(lambda x: x.notna().mean())
            .reset_index()
        )
        return cov

    cov = pd.DataFrame({
        "column": work.columns,
        "coverage": [work[c].notna().mean() for c in work.columns],
        "missing_pct": [1 - work[c].notna().mean() for c in work.columns],
    })
    return cov.sort_values("coverage")


def plot_country_coverage(df: pd.DataFrame, value_col: str, title: str):
    if df.empty or value_col not in df.columns:
        st.info(f"No data available for {title}.")
        return

    fig = px.bar(
        df.sort_values(value_col),
        x="country",
        y=value_col,
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.header("Coverage")

    country_daily = load_optional_parquet(DATA_DIR / "country_daily.parquet")
    country_scores = load_optional_parquet(DATA_DIR / "country_scores_daily.parquet")
    portfolio_daily = load_optional_parquet(DATA_DIR / "portfolio_daily.parquet")
    mkt_features = load_optional_parquet(DATA_DIR / "mkt_features.parquet")

    st.subheader("Available files")
    available = {
        "country_daily.parquet": country_daily is not None,
        "country_scores_daily.parquet": country_scores is not None,
        "portfolio_daily.parquet": portfolio_daily is not None,
        "mkt_features.parquet": mkt_features is not None,
    }
    st.write(pd.DataFrame({"file": list(available.keys()), "exists": list(available.values())}))

    # ---- country_daily coverage ----
    st.subheader("Country Daily Coverage")
    if country_daily is not None:
        cov_country_daily = coverage_table(country_daily, group_col="country")
        st.dataframe(cov_country_daily, use_container_width=True)

        for col in ["y10y", "fx_usd_ret", "hard_spread_proxy", "local_ret_proxy_usd"]:
            if col in cov_country_daily.columns:
                plot_country_coverage(
                    cov_country_daily,
                    col,
                    f"country_daily coverage: {col}",
                )
    else:
        st.info("country_daily.parquet not found.")

    # ---- country_scores coverage ----
    st.subheader("Country Scores Coverage")
    if country_scores is not None:
        cov_scores = coverage_table(country_scores, group_col="country")
        st.dataframe(cov_scores, use_container_width=True)

        for col in ["score", "score_raw", "score_pct", "score_scaled"]:
            if col in cov_scores.columns:
                plot_country_coverage(
                    cov_scores,
                    col,
                    f"country_scores coverage: {col}",
                )
    else:
        st.info("country_scores_daily.parquet not found.")

    # ---- portfolio coverage ----
    st.subheader("Portfolio Coverage")
    if portfolio_daily is not None:
        cov_port = coverage_table(portfolio_daily, group_col="country")
        st.dataframe(cov_port, use_container_width=True)

        for col in ["weight", "hard_w", "local_w", "local_share", "duration_tilt_years"]:
            if col in cov_port.columns:
                plot_country_coverage(
                    cov_port,
                    col,
                    f"portfolio_daily coverage: {col}",
                )
    else:
        st.info("portfolio_daily.parquet not found.")

    # ---- market features coverage ----
    st.subheader("Market Features Coverage")
    if mkt_features is not None:
        cov_mkt = coverage_table(mkt_features)
        st.dataframe(cov_mkt, use_container_width=True)

        fig = px.bar(
            cov_mkt.sort_values("coverage"),
            x="column",
            y="coverage",
            title="mkt_features column coverage",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("mkt_features.parquet not found.")

    # ---- file provenance / source mapping ----
    st.subheader("Data Source Mapping")
    source_map = pd.DataFrame(
        [
            {"output_file": "country_daily.parquet", "built_by": "run/build_country_panel.py", "source_module": "src/em/country/data.py"},
            {"output_file": "country_scores_daily.parquet", "built_by": "run/build_country_scores.py", "source_module": "src/em/country/score.py"},
            {"output_file": "portfolio_daily.parquet", "built_by": "run/build_portfolio.py", "source_module": "src/em/portfolio/allocator.py"},
            {"output_file": "mkt_features.parquet", "built_by": "run/ingest_all.py", "source_module": "src/em/ingestion/market.py"},
        ]
    )
    st.dataframe(source_map, use_container_width=True)