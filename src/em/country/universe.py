from __future__ import annotations

# ── Country Universe Version History ─────────────────────────────────────────
#
#  Each entry records the countries live in the model during that period.
#  Reason for retirement is noted so the rationale is never lost.
#
COUNTRY_UNIVERSE_ARCHIVED: list[dict] = [
    {
        "version": 1,
        "effective_from": "2015-01-01",
        "effective_to":   "2026-03-13",
        "retired_reason": (
            "Peru and Thailand removed: no reliable daily 10Y yield coverage "
            "from Stooq (10YPEY.B / 10YTHY.B return empty series), causing "
            "signal_confidence = 0 for both countries throughout the backtest."
        ),
        "countries": [
            "Brazil", "Mexico", "Colombia", "Chile", "Peru",
            "South Africa", "Poland", "Hungary", "Romania",
            "Indonesia", "Malaysia", "Philippines", "Thailand",
        ],
    },
]

# Active universe — countries currently in the model.
# Last updated: 2026-03-13
COUNTRY_UNIVERSE: list[str] = [
    "Brazil",
    "Mexico",
    "Colombia",
    "Chile",
    "South Africa",
    "Poland",
    "Hungary",
    "Romania",
    "Indonesia",
    "Malaysia",
    "Philippines",
]

# ── Per-country market data config ───────────────────────────────────────────

FX_TICKERS = {
    "Brazil":       "BRL=X",
    "Mexico":       "MXN=X",
    "Colombia":     "COP=X",
    "Chile":        "CLP=X",
    "South Africa": "ZAR=X",
    "Poland":       "PLN=X",
    "Hungary":      "HUF=X",
    "Romania":      "RON=X",
    "Indonesia":    "IDR=X",
    "Malaysia":     "MYR=X",
    "Philippines":  "PHP=X",
}

# Stooq daily 10Y government bond yield tickers.
# Format: 10Y{ISO2}Y.B  (e.g. 10YUSY.B for the US 10Y)
YIELD10Y_STOOQ = {
    "Brazil":       "10YBRY.B",
    "Mexico":       "10YMXY.B",
    "Colombia":     "10YCOY.B",
    "Chile":        "10YCLY.B",
    "South Africa": "10YZAY.B",
    "Poland":       "10YPLY.B",
    "Hungary":      "10YHUY.B",
    "Romania":      "10YROY.B",
    "Indonesia":    "10YIDY.B",
    "Malaysia":     "10YMYY.B",
    "Philippines":  "10YPHY.B",
}

# ── Global macro data config ──────────────────────────────────────────────────

GLOBAL_MACRO_FRED = {
    "us10y": "DGS10",
    "us2y":  "DGS2",
    "DXY":   "DTWEXEMEGS",             # Fed broad trade-weighted USD index
    # Global EM credit environment (ICE BofA indices, daily, freely available on FRED)
    "us_hy_oas": "BAMLH0A0HYM2",       # US HY Master II OAS — global risk-off signal (bp)
    "em_oas":    "BAMLEMCBPIOAS",       # EM Corporate Plus OAS — broad EM credit (bp)
    "em_hy_oas": "BAMLEMHBHYCRPIOAS",   # EM HY Corporate Plus OAS (bp)
    # Derived: em_hy_ig_spread = em_hy_oas - em_oas, computed in build_country_panel.py
}

GLOBAL_MACRO_YAHOO = {
    "VIX":    "^VIX",
    "Brent":  "BZ=F",
    "WTI":    "CL=F",
    "Copper": "HG=F",
    "Gold":   "GC=F",
}

# Monthly CPI YoY (%) per country — OECD via FRED.
# Template: CPALTT01{CC}M659N  (CC = 2-letter ISO code)
# Used to compute real_yield = y10y - cpi_yoy (in build_country_panel.py).
CPI_YOY_FRED: dict[str, str] = {
    "Brazil":       "CPALTT01BRM659N",
    "Mexico":       "CPALTT01MXM659N",
    "Colombia":     "CPALTT01COM659N",
    "Chile":        "CPALTT01CLM659N",
    "South Africa": "CPALTT01ZAM659N",
    "Poland":       "CPALTT01PLM659N",
    "Hungary":      "CPALTT01HUM659N",
    "Romania":      "CPALTT01ROM659N",
    "Indonesia":    "CPALTT01IDM659N",
    "Malaysia":     "CPALTT01MYM659N",
    "Philippines":  "CPALTT01PHM659N",
    "US":           "CPALTT01USM659N",
}

# Monthly short-term interbank rate (%) per country — OECD via FRED.
# Template: IRSTCI01{CC}M156N  (CC = 2-letter ISO code)
# "US" entry provides us_short_rate for fx_carry = local_short_rate - us_short_rate.
SHORT_RATE_FRED: dict[str, str] = {
    "Brazil":       "IRSTCI01BRM156N",
    "Mexico":       "IRSTCI01MXM156N",
    "Colombia":     "IRSTCI01COM156N",
    "Chile":        "IRSTCI01CLM156N",
    "South Africa": "IRSTCI01ZAM156N",
    "Poland":       "IRSTCI01PLM156N",
    "Hungary":      "IRSTCI01HUM156N",
    "Romania":      "IRSTCI01ROM156N",
    "Indonesia":    "IRSTCI01IDM156N",
    "Malaysia":     "IRSTCI01MYM156N",
    "Philippines":  "IRSTCI01PHM156N",
    "US":           "IRSTCI01USM156N",
}

# Global EM credit environment — daily series freely available on FRED.
# Used as a cross-asset anchor for the EMBI spread proxy model.
#   BAMLEMCBPIOAS  ICE BofA EM Corporate Plus Index OAS (bp)
#   BAMLH0A0HYM2   ICE BofA US HY Master II OAS — global risk-off signal (bp)
EMBI_GLOBAL_FRED: dict[str, str] = {
    "em_oas":    "BAMLEMCBPIOAS",
    "us_hy_oas": "BAMLH0A0HYM2",
}

# J.P. Morgan EMBI Global Diversified index weights (approximate, as of early 2026).
# Source: JPMorgan index factsheets. Sum of raw weights ≈ 0.345 (11-country sub-universe);
# allocator.py normalises these to sum to 1.0 within the active universe.
# Update when index rebalances materially (>1pp change on any country).
EMBI_WEIGHTS: dict[str, float] = {
    "Brazil":       0.0640,
    "Mexico":       0.0600,
    "Colombia":     0.0270,
    "Chile":        0.0120,
    "South Africa": 0.0200,
    "Poland":       0.0160,
    "Hungary":      0.0100,
    "Romania":      0.0130,
    "Indonesia":    0.0480,
    "Malaysia":     0.0290,
    "Philippines":  0.0360,
}

# ── Commodity sensitivity per country ─────────────────────────────────────────
# Coefficient in [-1, +1]: positive = net exporter (benefits from commodity rallies),
# negative = net importer (hurt by commodity rallies).
# Proxy commodity: Brent crude (primary global pricing signal; correlated with
# metals/agri cycles that also drive EM sovereign spreads).
COMMODITY_SENSITIVITY: dict[str, float] = {
    "Brazil":       +1.0,   # crude oil, iron ore, soybeans, sugar
    "Mexico":       +0.5,   # oil exporter, but large manufacturing base
    "Colombia":     +1.0,   # oil ~40% of exports
    "Chile":        +0.8,   # copper ~50% of exports
    "South Africa": +0.8,   # gold, platinum, coal
    "Poland":       -0.5,   # net energy importer
    "Hungary":      -0.5,   # net energy importer
    "Romania":      -0.3,   # small domestic oil, net importer overall
    "Indonesia":    +0.5,   # coal, palm oil, nickel
    "Malaysia":     +0.5,   # crude oil, LNG, palm oil
    "Philippines":  -0.3,   # net commodity importer
}

# ── Sovereign credit rating composite ─────────────────────────────────────────
# Numeric composite of S&P and Moody's long-term foreign-currency ratings.
# Scale: AAA=20, AA+=19, AA=18, AA-=17, A+=16, A=15, A-=14,
#        BBB+=13, BBB=12, BBB-=11, BB+=9, BB=7, BB-=5,
#        B+=3, B=2, B-=1, CCC+=0 (and below).
# Averaged across the two agencies; update when ≥1-notch change occurs.
# Source: S&P Global Ratings / Moody's Investors Service — public ratings, 2026-Q1.
SOVEREIGN_RATINGS: dict[str, float] = {
    "Brazil":       5.0,   # S&P BB-, Moody's Ba3
    "Mexico":       9.0,   # S&P BBB-, Moody's Baa2
    "Colombia":     7.0,   # S&P BB+, Moody's Ba2
    "Chile":       14.0,   # S&P A-, Moody's A2
    "South Africa": 5.0,   # S&P BB-, Moody's Ba2
    "Poland":      13.0,   # S&P A-, Moody's A2
    "Hungary":     11.0,   # S&P BBB, Moody's Baa2
    "Romania":      9.0,   # S&P BBB-, Moody's Baa3
    "Indonesia":   11.0,   # S&P BBB, Moody's Baa2
    "Malaysia":    14.0,   # S&P A-, Moody's A3
    "Philippines": 11.0,   # S&P BBB+, Moody's Baa2
}

# ── World Bank indicators for fiscal fundamentals ─────────────────────────────
# Annual series, forward-filled to daily in build_country_panel.py.
# Fetched via wbdata (pip install wbdata). No API key required.
FISCAL_WB_INDICATORS: dict[str, str] = {
    "fiscal_balance_gdp": "GC.BAL.CASH.GD.ZS",  # general govt net lending/borrowing (% GDP)
    "debt_gdp":           "GC.DOD.TOTL.GD.ZS",  # general govt gross debt (% GDP)
    "reserves_months":    "FI.RES.TOTL.MO",      # total reserves in months of imports
}

# ISO 3166-1 alpha-3 codes for the active universe (needed by wbdata).
COUNTRY_ISO3: dict[str, str] = {
    "Brazil":       "BRA",
    "Mexico":       "MEX",
    "Colombia":     "COL",
    "Chile":        "CHL",
    "South Africa": "ZAF",
    "Poland":       "POL",
    "Hungary":      "HUN",
    "Romania":      "ROU",
    "Indonesia":    "IDN",
    "Malaysia":     "MYS",
    "Philippines":  "PHL",
}

# ── FOMC meeting dates ────────────────────────────────────────────────────────
# End date of each scheduled 2-day FOMC meeting.
# Source: federalreserve.gov. Update list at the start of each calendar year.
FOMC_DATES: list[str] = [
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-10",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]
