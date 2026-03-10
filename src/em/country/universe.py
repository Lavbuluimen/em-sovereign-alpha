COUNTRY_UNIVERSE = [
    "Brazil",
    "Mexico",
    "Colombia",
    "Chile",
    "Peru",
    "South Africa",
    "Poland",
    "Hungary",
    "Romania",
    "Indonesia",
    "Malaysia",
    "Philippines",
    "Thailand",
]

FX_TICKERS = {
    "Brazil": "BRL=X",
    "Mexico": "MXN=X",
    "Colombia": "COP=X",
    "Chile": "CLP=X",
    "Peru": "PEN=X",
    "South Africa": "ZAR=X",
    "Poland": "PLN=X",
    "Hungary": "HUF=X",
    "Romania": "RON=X",
    "Indonesia": "IDR=X",
    "Malaysia": "MYR=X",
    "Philippines": "PHP=X",
    "Thailand": "THB=X",
}

# OECD long-term government bond yields via FRED
# Some may have weaker coverage than others; failures are handled gracefully.
YIELD10Y_FRED = {
    "Brazil": "IRLTLT01BRM156N",
    "Mexico": "IRLTLT01MXM156N",
    "Colombia": "IRLTLT01COM156N",
    "Chile": "IRLTLT01CLM156N",
    "Peru": "IRLTLT01PEM156N",
    "South Africa": "IRLTLT01ZAM156N",
    "Poland": "IRLTLT01PLM156N",
    "Hungary": "IRLTLT01HUM156N",
    "Romania": "IRLTLT01ROM156N",
    "Indonesia": "IRLTLT01IDM156N",
    "Malaysia": "IRLTLT01MYM156N",
    "Philippines": "IRLTLT01PHM156N",
    "Thailand": "IRLTLT01THM156N",
}

GLOBAL_MACRO_FRED = {
    "us10y": "DGS10",
    "us2y": "DGS2",
}

GLOBAL_MACRO_YAHOO = {
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "Brent": "BZ=F",
    "WTI": "CL=F",
    "Copper": "HG=F",
    "Gold": "GC=F",
}