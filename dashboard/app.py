import streamlit as st
import pandas as pd
import plotly.express as px

st.title("EM Sovereign Portfolio Dashboard")

portfolio = pd.read_parquet("data/processed/portfolio_daily.parquet")
scores = pd.read_parquet("data/processed/country_scores_daily.parquet")

latest_date = portfolio["date"].max()

st.header("Portfolio Weights")

latest_port = portfolio[portfolio["date"] == latest_date]

fig = px.bar(
    latest_port,
    x="country",
    y="weight",
    title="Country Portfolio Weights"
)

st.plotly_chart(fig)

st.header("Country Scores")

latest_scores = scores[scores["date"] == latest_date]

fig = px.bar(
    latest_scores,
    x="country",
    y="score",
    title="Country Alpha Scores"
)

st.plotly_chart(fig)