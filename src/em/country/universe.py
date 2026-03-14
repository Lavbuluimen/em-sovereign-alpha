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
