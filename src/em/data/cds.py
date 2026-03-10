from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests


WGB_CDS_URL = "https://www.worldgovernmentbonds.com/cds-historical-data/{slug}/5-years/"


def load_local_cds_csvs(raw_dir: str = "data/raw/cds") -> pd.DataFrame:
    """
    Load sovereign CDS history from local CSV files.

    Expected file naming:
        data/raw/cds/Brazil.csv
        data/raw/cds/Mexico.csv
        ...

    Expected columns:
        date, cds_5y
    """
    base = Path(raw_dir)
    if not base.exists():
        return pd.DataFrame()

    out = {}
    for csv_path in base.glob("*.csv"):
        country = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip().lower() for c in df.columns]
            if "date" not in df.columns or "cds_5y" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"])
            s = pd.to_numeric(df["cds_5y"], errors="coerce")
            s.index = df["date"]
            s.name = country
            out[country] = s.sort_index()
        except Exception:
            continue

    if not out:
        return pd.DataFrame()

    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    return df.sort_index()


def fetch_wgb_cds_series(slug: str, start: str = "2015-01-01") -> pd.Series:
    """
    Optional fallback scraper for WorldGovernmentBonds CDS historical page.
    This is intentionally defensive because HTML tables can change.
    """
    url = WGB_CDS_URL.format(slug=slug)
    try:
        tables = pd.read_html(url)
    except Exception:
        return pd.Series(dtype=float, name=slug)

    for tbl in tables:
        cols = [str(c).strip().lower() for c in tbl.columns]
        lower_map = {str(c).strip().lower(): c for c in tbl.columns}

        if "date" in cols:
            # try to find a value column
            value_col = None
            for candidate in tbl.columns:
                c = str(candidate).strip().lower()
                if "cds" in c or "value" in c or "close" in c or "last" in c:
                    value_col = candidate
                    break

            if value_col is None:
                # fallback: second column if present
                if len(tbl.columns) >= 2:
                    value_col = tbl.columns[1]
                else:
                    continue

            try:
                dates = pd.to_datetime(tbl[lower_map["date"]], errors="coerce")
                vals = pd.to_numeric(tbl[value_col], errors="coerce")
                s = pd.Series(vals.values, index=dates).dropna().sort_index()
                s = s.loc[s.index >= pd.to_datetime(start)]
                s.name = slug
                return s
            except Exception:
                continue

    return pd.Series(dtype=float, name=slug)


def fetch_many_wgb_cds(slug_map: dict[str, str], start: str = "2015-01-01") -> pd.DataFrame:
    out = {}
    for country, slug in slug_map.items():
        try:
            out[country] = fetch_wgb_cds_series(slug=slug, start=start)
        except Exception:
            out[country] = pd.Series(dtype=float, name=country)

    if not out:
        return pd.DataFrame()

    df = pd.concat(out.values(), axis=1)
    df.columns = list(out.keys())
    return df.sort_index()


def build_cds_panel(
    slug_map: dict[str, str],
    start: str = "2015-01-01",
    raw_dir: str = "data/raw/cds",
    use_web_fallback: bool = True,
) -> pd.DataFrame:
    """
    Priority:
      1. local CSV data
      2. optional web fallback
    """
    local_df = load_local_cds_csvs(raw_dir=raw_dir)

    if use_web_fallback:
        web_df = fetch_many_wgb_cds(slug_map=slug_map, start=start)
    else:
        web_df = pd.DataFrame()

    countries = sorted(set(local_df.columns).union(web_df.columns))
    idx = local_df.index.union(web_df.index).sort_values()

    local_a = local_df.reindex(idx) if not local_df.empty else pd.DataFrame(index=idx)
    web_a = web_df.reindex(idx) if not web_df.empty else pd.DataFrame(index=idx)

    cds = pd.DataFrame(index=idx, columns=countries, dtype=float)
    source = pd.DataFrame(index=idx, columns=countries, dtype=object)

    for c in countries:
        local_col = local_a[c] if c in local_a.columns else pd.Series(index=idx, dtype=float)
        web_col = web_a[c] if c in web_a.columns else pd.Series(index=idx, dtype=float)

        cds[c] = local_col.combine_first(web_col)
        source[c] = None
        source.loc[local_col.notna(), c] = "local_csv"
        source.loc[local_col.isna() & web_col.notna(), c] = "web_fallback"

    return cds.sort_index(), source.sort_index()