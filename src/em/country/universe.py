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
    {
        "version": 2,
        "effective_from": "2026-03-13",
        "effective_to":   "2026-04-01",
        "retired_reason": (
            "Romania, Indonesia, Malaysia, Philippines removed: replaced with "
            "Turkey and China to improve universe quality and data coverage. "
            "Romania lacks OECD/FRED yield series and has thin sovereign bond "
            "market. Indonesia, Malaysia, Philippines have no FRED coverage and "
            "thin daily yield data. Turkey (OECD member, deep bond market) and "
            "China (world's second-largest economy, growing USD bond market) "
            "added as more liquid, analytically significant replacements."
        ),
        "countries": [
            "Brazil", "Mexico", "Colombia", "Chile",
            "South Africa", "Poland", "Hungary", "Romania",
            "Indonesia", "Malaysia", "Philippines",
        ],
    },
]

# Active universe — countries currently in the model.
# Last updated: 2026-04-03
# Turkey removed: no 10Y yield data available from any free public source
# (not in OECD IRLT dataset; CBRT EVDS requires API key; IMF IFS times out).
COUNTRY_UNIVERSE: list[str] = [
    "Brazil",
    "Mexico",
    "Colombia",
    "Chile",
    "South Africa",
    "Poland",
    "Hungary",
    "China",
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
    "China":        "CNY=X",
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
    "China":        "10YCNY.B",
}

# IMF IFS fallback: countries that have no OECD/FRED 10Y yield series.
# FIGB_PA = Government Bond Yield (benchmark, % p.a.) from IMF IFS SDMX API.
# Used as the third-tier fallback (after Stooq and FRED fail).
# IFS fallback retained for Brazil/Colombia/China; Turkey removed (no coverage).
YIELD10Y_IFS = {
    "Brazil":       "BR",
    "Colombia":     "CO",
    "China":        "CN",
}

# OECD SDMX DF_FINMARK fallback — IRLT measure (long-term govt bond yield, monthly).
# Confirmed working for BRA, COL, CHN. Turkey is absent from this dataset.
YIELD10Y_OECD = {
    "Brazil":   "BRA",
    "Colombia": "COL",
    "China":    "CHN",
}

# FRED fallback: OECD monthly long-term (10Y) government bond yields.
# Series template: IRLTLT01{CC}M156N  (OECD via FRED, monthly, %).
# Used when Stooq is unavailable/blocked; linearly interpolated to daily.
YIELD10Y_FRED = {
    "Brazil":       "IRLTLT01BRM156N",   # OECD key partner — tracked in MEI
    "Mexico":       "IRLTLT01MXM156N",
    "Colombia":     "IRLTLT01COM156N",   # OECD member since 2020
    "Chile":        "IRLTLT01CLM156N",
    "South Africa": "IRLTLT01ZAM156N",
    "Poland":       "IRLTLT01PLM156N",
    "Hungary":      "IRLTLT01HUM156N",
    "China":        "INTGSBCNM193N",     # OECD MEI long-term govt bond yield
    "US":           "IRLTLT01USM156N",
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
# Non-OECD countries (Brazil, Colombia, China) fall back to BIS/World Bank via IFS_COUNTRIES.
CPI_YOY_FRED: dict[str, str] = {
    "Mexico":       "CPALTT01MXM659N",
    "Chile":        "CPALTT01CLM659N",
    "South Africa": "CPALTT01ZAM659N",
    "Poland":       "CPALTT01PLM659N",
    "Hungary":      "CPALTT01HUM659N",
    "US":           "CPALTT01USM659N",
}

# Monthly short-term interbank rate (%) per country — OECD via FRED.
# Template: IRSTCI01{CC}M156N  (CC = 2-letter ISO code)
# "US" entry provides us_short_rate for fx_carry = local_short_rate - us_short_rate.
# Non-OECD countries fall back to BIS policy rates via IFS_COUNTRIES.
SHORT_RATE_FRED: dict[str, str] = {
    "Mexico":       "IRSTCI01MXM156N",
    "Chile":        "IRSTCI01CLM156N",
    "South Africa": "IRSTCI01ZAM156N",
    "Poland":       "IRSTCI01PLM156N",
    "Hungary":      "IRSTCI01HUM156N",
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
# Source: JPMorgan index factsheets. Sum of raw weights ≈ 0.271 (9-country sub-universe);
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
    "China":        0.0250,
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
    "China":        -0.4,   # world's largest oil importer, net commodity importer overall
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
    "China":       15.0,   # S&P A+, Moody's A1
}

# ── World Bank indicators ─────────────────────────────────────────────────────
# reserves_months only — fiscal_balance_gdp and debt_gdp come from IMF WEO
# (broader country coverage). Annual series, forward-filled to daily.
FISCAL_WB_INDICATORS: dict[str, str] = {
    "reserves_months": "FI.RES.TOTL.MO",  # total reserves in months of imports
}

# ── IMF WEO indicators ────────────────────────────────────────────────────────
# Fetched via the `weo` library (pip install weo). No API key required.
# Covers all 9 universe countries; World Bank GC.DOD.TOTL.GD.ZS misses several.
IMF_WEO_FISCAL_COLS: tuple[str, ...] = ("fiscal_balance_gdp", "debt_gdp")

# ── IMF IFS countries ─────────────────────────────────────────────────────────
# Countries missing CPI and short-rate coverage on FRED (non-OECD).
# Fetched via IMF IFS SDMX JSON API / BIS / World Bank. No API key required.
# ISO 3166-1 alpha-2 codes as used by the IMF SDMX service.
IFS_COUNTRIES: dict[str, str] = {
    "Brazil":    "BR",
    "Colombia":  "CO",
    "China":     "CN",
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
    "China":        "CHN",
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

# ── LatAm central bank meeting dates ────────────────────────────────────────
# Source: BCB (Copom), Banxico, BanRep — public calendars.
# Update at the start of each calendar year.

COPOM_DATES: list[str] = [
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-08-06", "2025-09-17", "2025-11-05", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-08-05", "2026-09-16", "2026-10-28", "2026-12-09",
]

BANXICO_DATES: list[str] = [
    # 2025
    "2025-02-06", "2025-03-27", "2025-05-15", "2025-06-26",
    "2025-08-14", "2025-09-25", "2025-11-13", "2025-12-18",
    # 2026
    "2026-02-12", "2026-03-26", "2026-05-14", "2026-06-25",
    "2026-08-13", "2026-09-24", "2026-11-12", "2026-12-17",
]
