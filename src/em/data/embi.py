from __future__ import annotations

import pandas as pd

from em.data.fred import fetch_many_fred_series


# Freely available on FRED — used to track the global EM credit environment.
# These are *not* country-level EMBI spreads; they provide a daily cross-asset
# anchor that correlates closely with sovereign spreads.
#
#   BAMLEMCBPIOAS  — ICE BofA EM Corporate Plus Index OAS (basis points)
#                    Broad daily proxy for EM credit risk premium.
#   BAMLH0A0HYM2   — ICE BofA US HY Master II OAS (basis points)
#                    Global risk-off signal; leads EM spread widening.
EMBI_GLOBAL_FRED: dict[str, str] = {
    "em_oas": "BAMLEMCBPIOAS",
    "us_hy_oas": "BAMLH0A0HYM2",
}


def fetch_global_em_credit(start: str = "2015-01-01") -> pd.DataFrame:
    """
    Fetch global EM credit environment proxies from FRED.

    Returns a daily DataFrame with columns:
      - em_oas     : ICE BofA EM Corporate Plus OAS (bp)
      - us_hy_oas  : ICE BofA US HY OAS — global risk-off anchor (bp)
    """
    return fetch_many_fred_series(series_map=EMBI_GLOBAL_FRED, start=start)


def build_embi_spread_panel(country_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build a country-level EMBI / sovereign spread proxy from freely available data.

    The sovereign yield spread over US Treasuries (hard_spread_proxy = y10y − us10y)
    is the standard EM-fund approach when direct CDS or EMBI strip data is unavailable.
    It captures both credit risk and FX risk premium and typically carries 0.85–0.95
    correlation with 5Y CDS for liquid EM sovereigns.

    Adds to the panel:
      - embi_spread_proxy    : yield spread level (% points), aliased from hard_spread_proxy
      - embi_spread_20d_chg  : 20-day change in spread (momentum)
      - has_embi_data        : float flag — 1.0 when spread is available, else 0.0

    Parameters
    ----------
    country_panel : pd.DataFrame
        Daily country panel that already contains ``date``, ``country``,
        and ``hard_spread_proxy``.

    Returns
    -------
    pd.DataFrame
        Input panel with the three embi_* columns appended.
    """
    df = country_panel.copy().sort_values(["country", "date"])

    df["embi_spread_proxy"] = df["hard_spread_proxy"]

    df["embi_spread_20d_chg"] = df.groupby("country")["embi_spread_proxy"].transform(
        lambda x: x.diff(20)
    )

    df["has_embi_data"] = df["embi_spread_proxy"].notna().astype(float)

    return df
