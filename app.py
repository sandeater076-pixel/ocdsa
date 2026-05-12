#!/usr/bin/env python3
"""
Orange County Elections Dashboard
Precinct-level election results and turnout analytics.

────────────────────────────────────────────────
DATA DIRECTORY STRUCTURE
────────────────────────────────────────────────
  data/
    2024/
      *.codes               ← code-to-candidate lookup (tab-separated: code, name, total)
      sov_main.csv          ← wide-format precinct data  (c059_gXX_sov_data_by_*.csv)
      *sov_local_county*.csv
      *sov_local_city*.csv
      *sov_local_ccd*.csv
      *sov_local_school*.csv
    2022/  2020/  (same structure)

Naming rules (auto-detected by glob):
  Main file   : *sov_data*.csv  or  sov_main.csv
  County races: *sov_local_county_by_g*.csv
  City races  : *sov_local_city_by_g*.csv
  CCD races   : *sov_local_ccd_by_g*.csv
  School races: *sov_local_school_by_g*.csv
  GeoJSON     : srprec.geojson  or  *.geojson
    Download  : https://statewidedatabase.org/pub/data/G24/c059/srprec_059_g24_v01.geojson.zip

────────────────────────────────────────────────
RUN
────────────────────────────────────────────────
  pip install dash dash-bootstrap-components plotly pandas
  python app.py
  → open http://localhost:8050
"""

import re
import json
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

PARTY_COLORS = {
    "DEM": "#2166ac",
    "REP": "#d6604d",
    "AIP": "#f4a582",
    "GRN": "#4dac26",
    "LIB": "#d4b200",
    "NLP": "#9970ab",
    "PAF": "#bf812d",
    "IND": "#878787",
    "Y":   "#1a9641",
    "N":   "#d7191c",
    "REF": "#555555",
    "MSC": "#aaaaaa",
}
DEFAULT_COLOR = "#aaaaaa"

RACE_CATEGORIES = [
    ("federal",   "Federal & Statewide"),
    ("congress",  "US Congressional"),
    ("state_sen", "State Senate"),
    ("state_asm", "State Assembly"),
    ("measures",  "Ballot Measures"),
    ("county",    "County"),
    ("city",      "City"),
    ("ccd",       "Community College District"),
    ("school",    "School District"),
]

# Long-format category ids (sourced from separate local CSV files)
LOCAL_CATEGORIES = {"county", "city", "ccd", "school"}

# Column names that are NOT vote counts
META_COLS = {
    "county", "srprec", "addist", "cddist", "sddist", "bedist",
    "TOTREG", "DEMREG", "REPREG", "AIPREG", "GRNREG", "LIBREG",
    "NLPREG", "REFREG", "DCLREG", "MSCREG", "TOTVOTE", "DEMVOTE",
    "REPVOTE", "AIPVOTE", "GRNVOTE", "LIBVOTE", "NLPVOTE", "REFVOTE",
    "DCLVOTE", "MSCVOTE", "PRCVOTE", "ABSVOTE",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def find_file(directory: Path, *patterns: str) -> Path | None:
    for pat in patterns:
        matches = sorted(directory.glob(pat))
        if matches:
            return matches[0]
    return None


def load_codes(path: Path) -> dict:
    """Returns {code: (display_name, countywide_total)}"""
    codes = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                code  = parts[0]
                name  = parts[1]
                total = int(parts[2]) if len(parts) >= 3 else 0
                codes[code] = (name, total)
    return codes


def load_year(year_dir: Path) -> dict | None:
    year = int(year_dir.name)

    main_path   = find_file(year_dir, "*sov_data*.csv", "sov_main.csv", "*sov*.csv")
    county_path = find_file(year_dir, "*sov_local_county_by_g*.csv")
    city_path   = find_file(year_dir, "*sov_local_city_by_g*.csv")
    ccd_path    = find_file(year_dir, "*sov_local_ccd_by_g*.csv")
    school_path = find_file(year_dir, "*sov_local_school_by_g*.csv")
    codes_path  = find_file(year_dir, "*.codes", "*codes*.txt", "*.code")

    if not main_path:
        return None

    main_df = pd.read_csv(main_path)
    main_df["srprec"] = pd.to_numeric(main_df["srprec"], errors="coerce")
    main_df = main_df[main_df["srprec"].notna()].copy()
    main_df = main_df[main_df["srprec"] < 900_000].copy()
    main_df["TOTREG"] = pd.to_numeric(main_df["TOTREG"], errors="coerce").fillna(0)
    main_df = main_df[main_df["TOTREG"] < 100_000].copy()

    county_df = pd.read_csv(county_path) if county_path else pd.DataFrame()
    city_df   = pd.read_csv(city_path)   if city_path   else pd.DataFrame()
    ccd_df    = pd.read_csv(ccd_path)    if ccd_path    else pd.DataFrame()
    school_df = pd.read_csv(school_path) if school_path else pd.DataFrame()
    codes     = load_codes(codes_path)   if codes_path  else {}

    # GeoJSON precinct boundaries (optional — map silently disabled if missing)
    geo_path = find_file(year_dir, "srprec.geojson", "*.geojson")
    geojson  = None
    if geo_path:
        with open(geo_path) as f:
            geojson = json.load(f)
        # Normalise feature id → srprec integer so Plotly can join on it
        for feat in geojson.get("features", []):
            props = feat.get("properties", {})
            srp   = props.get("SRPREC") or props.get("srprec") or feat.get("id")
            if srp is not None:
                feat["id"] = int(srp)

    return {
        "year": year, "main": main_df,
        "county": county_df, "city": city_df, "ccd": ccd_df, "school": school_df,
        "codes": codes, "geojson": geojson,
    }


def load_all_data() -> dict:
    result = {}
    if not DATA_DIR.exists():
        return result
    for year_dir in sorted(DATA_DIR.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            d = load_year(year_dir)
            if d:
                result[int(year_dir.name)] = d
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RACE INDEX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

PARTY_PAT = r"(DEM|REP|AIP|GRN|LIB|PAF|NLP|IND|REF|MSC)"


def build_race_index(year_data: dict) -> dict:
    """
    Returns {category_id: {race_id: {label, candidates[]}}}

    Wide-format races (federal, congressional, assembly, senate, measures) are
    parsed from the main CSV columns.  Long-format races (county, city, ccd,
    school) come from their own separate CSV files stored as DataFrames in
    year_data.
    """
    df    = year_data["main"]
    codes = year_data["codes"]
    races: dict = {}

    # ── 1. District-based races: Assembly, Congressional, Senate ─────────────
    DISTRICT_RACES = [
        ("state_asm", "ASS", "addist", "Assembly District"),
        ("congress",  "CNG", "cddist", "Congressional District"),
        ("state_sen", "SEN", "sddist", "Senate District"),
    ]
    for cat, prefix, dist_col, label_prefix in DISTRICT_RACES:
        generic_cols = [
            c for c in df.columns
            if re.match(rf"^{prefix}{PARTY_PAT}\d+$", c)
        ]
        if not generic_cols or dist_col not in df.columns:
            continue

        unique_dists = sorted(
            int(d) for d in df[dist_col].dropna().unique()
            if 0 < d < 10_000
        )
        for dist in unique_dists:
            candidates = []
            for col in generic_cols:
                m = re.match(rf"^{prefix}{PARTY_PAT}(\d+)$", col)
                if not m:
                    continue
                party, num = m[1], m[2]
                code_key   = f"{prefix}{dist}{party}{num}"
                if code_key not in codes:
                    continue
                name = codes[code_key][0]
                candidates.append(dict(col=col, party=party, name=name,
                                       dist_col=dist_col, dist_val=dist))
            if candidates:
                races.setdefault(cat, {})[f"{prefix}-{dist}"] = {
                    "label":      f"{label_prefix} {dist}",
                    "candidates": candidates,
                }

    # ── 2. Federal / statewide races ─────────────────────────────────────────
    FEDERAL_RACES = [
        ("Presidential",      rf"^PRS{PARTY_PAT}\d+$", "Presidential Race"),
        ("US Senate",         rf"^USS{PARTY_PAT}\d+$", "US Senate (General)"),
        ("US Senate Primary", rf"^USP{PARTY_PAT}\d+$", "US Senate (Top-2 Primary)"),
    ]
    for rid, pat, label in FEDERAL_RACES:
        candidates = []
        for col in df.columns:
            if not re.match(pat, col):
                continue
            m = re.match(rf"^[A-Z]+{PARTY_PAT}(\d+)$", col)
            if not m:
                continue
            party = m[1]
            name  = codes.get(col, (col, 0))[0]
            candidates.append(dict(col=col, party=party, name=name,
                                   dist_col=None, dist_val=None))
        if candidates:
            races.setdefault("federal", {})[rid] = {
                "label":      label,
                "candidates": candidates,
            }

    # ── 3. Ballot Measures ────────────────────────────────────────────────────
    for col in df.columns:
        m = re.match(r"^PR_(\d+)_(Y|N)$", col)
        if not m:
            continue
        prop, side = m[1], m[2]
        rid  = f"Prop {prop}"
        name = "Yes" if side == "Y" else "No"
        races.setdefault("measures", {}).setdefault(rid, {
            "label":      f"Proposition {prop}",
            "candidates": [],
        })["candidates"].append(dict(col=col, party=side, name=name,
                                    dist_col=None, dist_val=None))

    # ── 4. Long-format local races (county / city / ccd / school) ────────────
    # Each gets its OWN dict — previous bug had all writing to county_races.
    LOCAL_SOURCES = [
        ("county", year_data.get("county", pd.DataFrame())),
        ("city",   year_data.get("city",   pd.DataFrame())),
        ("ccd",    year_data.get("ccd",    pd.DataFrame())),
        ("school", year_data.get("school", pd.DataFrame())),
    ]
    for cat_key, df_local in LOCAL_SOURCES:
        if df_local.empty:
            continue
        cat_races: dict = {}
        for contest in df_local["contest"].unique():
            sub    = df_local[df_local["contest"] == contest].copy()
            office = sub["office"].iloc[0]
            cat_races[contest] = {
                "label":      contest,
                "office":     office,
                "candidates": sub,   # long-format DataFrame
            }
        if cat_races:
            races[cat_key] = cat_races

    return races


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_results(year_data: dict, category: str, race_id: str, race_index: dict) -> dict:
    race = race_index.get(category, {}).get(race_id)
    if not race:
        return {}

    # ── Long-format local races ───────────────────────────────────────────────
    if category in LOCAL_CATEGORIES:
        cands_df = race["candidates"]
        cands_df["votes"] = pd.to_numeric(cands_df["votes"], errors='coerce').fillna(0)
        summary  = cands_df.groupby("candidate")["votes"].sum().reset_index()
        total    = int(summary["votes"].sum())

        cands_out = {}
        for _, row in summary.iterrows():
            pct = row["votes"] / total * 100 if total > 0 else 0
            cands_out[row["candidate"]] = {
                "votes": int(row["votes"]),
                "pct":   pct,
                "party": "IND",
            }

        ranked = sorted(cands_out.items(), key=lambda x: x[1]["votes"], reverse=True)
        margin = ranked[0][1]["pct"] - ranked[1][1]["pct"] if len(ranked) >= 2 else 0

        return {
            "candidates":       cands_out,
            "totreg":           None,
            "totvote":          None,
            "turnout_pct":      None,
            "total_race_votes": total,
            "margin":           margin,
            "winner":           ranked[0][0] if ranked else None,
            "winner_party":     "IND",
        }

    # ── Wide-format (main CSV) ────────────────────────────────────────────────
    df    = year_data["main"]
    cands = race["candidates"]
    if not cands:
        return {}

    dist_col = cands[0]["dist_col"]
    dist_val = cands[0]["dist_val"]
    df_sub   = df[df[dist_col] == dist_val] if (dist_col and dist_val) else df

    cands_out = {}
    for cand in cands:
        col = cand["col"]
        if col not in df_sub.columns:
            continue
        cands_out[cand["name"]] = {
            "votes": int(df_sub[col].fillna(0).sum()),
            "party": cand["party"],
            "col":   col,
        }

    total = sum(c["votes"] for c in cands_out.values())
    for c in cands_out.values():
        c["pct"] = c["votes"] / total * 100 if total > 0 else 0

    totreg  = int(df_sub["TOTREG"].fillna(0).sum())  if "TOTREG"  in df_sub else 0
    totvote = int(df_sub["TOTVOTE"].fillna(0).sum()) if "TOTVOTE" in df_sub else 0

    ranked       = sorted(cands_out.items(), key=lambda x: x[1]["votes"], reverse=True)
    margin       = ranked[0][1]["pct"] - ranked[1][1]["pct"] if len(ranked) >= 2 else 0
    winner_party = ranked[0][1]["party"] if ranked else "IND"

    return {
        "candidates":       cands_out,
        "totreg":           totreg,
        "totvote":          totvote,
        "turnout_pct":      totvote / totreg * 100 if totreg > 0 else None,
        "total_race_votes": total,
        "margin":           margin,
        "winner":           ranked[0][0] if ranked else None,
        "winner_party":     winner_party,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

CHART_BG = "rgba(0,0,0,0)"


def chart_layout(**kwargs) -> dict:
    base = dict(
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        margin=dict(l=10, r=10, t=30, b=30),
    )
    base.update(kwargs)
    return base


def fig_vote_share(results: dict) -> go.Figure:
    if not results or not results.get("candidates"):
        return _empty_fig("No data")

    ranked = sorted(results["candidates"].items(), key=lambda x: x[1]["votes"], reverse=True)
    names  = [r[0] for r in ranked]
    pcts   = [r[1]["pct"] for r in ranked]
    votes  = [r[1]["votes"] for r in ranked]
    colors = [PARTY_COLORS.get(r[1].get("party", ""), DEFAULT_COLOR) for r in ranked]

    fig = go.Figure(go.Bar(
        x=pcts, y=names,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"  {p:.1f}%  ({v:,})" for p, v in zip(pcts, votes)],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.2f}%  (%{customdata:,} votes)<extra></extra>",
        customdata=votes,
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="#888", line_width=1.5, opacity=0.6)

    max_pct = max(pcts) if pcts else 60
    fig.update_layout(
        **chart_layout(height=max(180, len(names) * 56 + 60),
                       margin=dict(l=10, r=150, t=65, b=30)),
        title_text="Vote Share by Candidate",
        title_x=0,
        xaxis=dict(title="% of Votes Cast", range=[0, min(max_pct * 1.35, 100)],
                   ticksuffix="%"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        showlegend=False,
    )
    return fig


def fig_margin_trend(all_data: dict, race_indexes: dict,
                     category: str, race_id: str) -> go.Figure:
    years, margins, parties = [], [], []
    for year in sorted(all_data.keys()):
        idx = race_indexes.get(year, {})
        if category not in idx or race_id not in idx[category]:
            continue
        res = compute_results(all_data[year], category, race_id, idx)
        if res:
            years.append(year)
            margins.append(round(res["margin"], 2))
            parties.append(res.get("winner_party", "IND"))

    if not years:
        return _empty_fig("Add more election years to data/ to see trends")

    # Colour each point by the winning party that year
    point_colors = [
        "#2166ac" if p == "DEM"
        else "#d6604d" if p == "REP"
        else DEFAULT_COLOR
        for p in parties
    ]

    fig = go.Figure(go.Scatter(
        x=[str(y) for y in years],
        y=margins,
        mode="lines+markers+text",
        line=dict(color="#aaaaaa", width=2, dash="dot"),
        marker=dict(color=point_colors, size=12,
                    line=dict(width=2, color="white")),
        text=[f"{m:.1f}%" for m in margins],
        textposition="top center",
        hovertemplate="%{x}: %{y:.2f}% margin<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    # fig.update_layout(
    #     **chart_layout(height=230, margin=dict(l=10, r=10, t=36, b=30)),
    #     title_text="Margin of Victory Over Time",
    #     title_x=0,
    #     xaxis_title="Election Year",
    #     yaxis=dict(title="Margin (%)", ticksuffix="%"),
    #     showlegend=False,
    # )
    max_m = max(margins) if margins else 10
    min_m = min(margins) if margins else -10

    fig.update_layout(
        # Increased 't' (top) from 36 to 60 for title clearance
        # Increased 'l' and 'r' to 35 to prevent label clipping on the sides
        **chart_layout(height=230, margin=dict(l=35, r=35, t=60, b=30)),
        
        title_text="Margin of Victory Over Time",
        title_x=0,
        
        xaxis=dict(
            title="Election Year",
            # Adds padding to the left and right of the x-axis markers
            range=[-0.5, len(years) - 0.5] if years else None
        ),
        
        yaxis=dict(
            title="Margin (%)", 
            ticksuffix="%",
            # range=[min, max] logic: adds 30% extra space above the max value
            range=[min_m - 5, max_m * 1.6] 
        ),
        showlegend=False,
    )
    return fig
    return fig


def fig_turnout_trend(all_data: dict, race_indexes: dict,
                      category: str, race_id: str) -> go.Figure:
    years, turnouts = [], []
    for year in sorted(all_data.keys()):
        idx = race_indexes.get(year, {})
        if category not in idx or race_id not in idx[category]:
            continue
        res = compute_results(all_data[year], category, race_id, idx)
        if res and res.get("turnout_pct") is not None:
            years.append(year)
            turnouts.append(round(res["turnout_pct"], 2))

    if not years:
        return _empty_fig("Turnout data not available for this race type")
    max_m = max(turnouts) if turnouts else 10
    min_m = min(turnouts) if turnouts else -10

    fig = go.Figure(go.Scatter(
        x=[str(y) for y in years],
        y=turnouts,
        mode="lines+markers+text",
        text=[f"{t:.1f}%" for t in turnouts],
        textposition="top center",
        line=dict(color="#4a90d9", width=3),
        marker=dict(size=10, color="#4a90d9", line=dict(width=2, color="white")),
        hovertemplate="%{x}: %{y:.2f}% turnout<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(74,144,217,0.12)",
    ))
    fig.update_layout(
        **chart_layout(height=230, margin=dict(l=35, r=35, t=60, b=30)),
        title_text="Registered Voter Turnout Over Time",
        title_x=0,
        xaxis=dict(
            title="Election Year",
            # Adds padding to the left and right of the x-axis markers
            range=[-0.5, len(years) - 0.5] if years else None
        ),
        yaxis=dict(title="Turnout (%)", ticksuffix="%",
                   range=[min_m - 5, 100]),
        showlegend=False,
    )
    return fig

def fig_precinct_map(year_data: dict, category: str, race_id: str,
                     race_index: dict, map_mode: str = "vote") -> go.Figure:
    """
    Choropleth map supporting both wide-format (Federal/State) and 
    long-format (Local) race results.
    """
    geojson = year_data.get("geojson")
    if not geojson:
        return _empty_fig(
            "Map unavailable — place srprec.geojson in data/&lt;year&gt;/ to enable.<br>"
            "Download from statewidedatabase.org → County 059 → SRPREC GeoJSON"
        )

    race = race_index.get(category, {}).get(race_id)
    if not race:
        return _empty_fig("No race data")

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 1: LOCAL RACES (Long-Format Data)
    # ─────────────────────────────────────────────────────────────────────────
    if category in LOCAL_CATEGORIES:
        df_race = race["candidates"].copy()
        # Clean votes column: turn corrupt zero-strings into actual 0s
        df_race["votes"] = pd.to_numeric(df_race["votes"], errors="coerce").fillna(0)
        race_precincts = df_race["srprec"].unique()

        if map_mode == "turnout":
            # For turnout, we pull data from the main CSV for these specific precincts
            df_main = year_data["main"]
            df_sub = df_main[df_main["srprec"].isin(race_precincts)].copy()
            df_sub = df_sub[df_sub["TOTREG"] > 0].copy()

            df_sub["_val"]   = (df_sub["TOTVOTE"] / df_sub["TOTREG"] * 100).round(1)
            df_sub["_label"] = df_sub["_val"].apply(lambda v: f"{v:.1f}% turnout")
            colorscale       = [[0.0, "#f7fbff"], [0.3, "#9ecae1"],
                                [0.65, "#3182bd"], [1.0, "#08306b"]]
            zmin, zmax       = 0, 100
            colorbar_title   = "Turnout %"
            title_text       = "Voter Turnout by Precinct"
        else:
            # For local votes, we show the margin between the top 2 candidates overall
            overall = df_race.groupby("candidate")["votes"].sum().sort_values(ascending=False)
            
            if len(overall) < 2:
                # Fallback for single-candidate races
                c1 = overall.index[0] if not overall.empty else "N/A"
                prec_totals = df_race.groupby("srprec")["votes"].sum().reset_index(name="total_p")
                c1_votes = df_race[df_race["candidate"] == c1][["srprec", "votes"]].rename(columns={"votes": "c1_v"})
                df_sub = pd.merge(prec_totals, c1_votes, on="srprec", how="left").fillna(0)
                df_sub["_val"] = np.where(df_sub["total_p"] > 0, df_sub["c1_v"] / df_sub["total_p"] * 100, np.nan)
                df_sub["_label"] = df_sub["_val"].apply(lambda v: f"{v:.1f}% {c1}" if not np.isnan(v) else "No data")
                colorscale = [[0, "#f7fbff"], [1, "#2166ac"]]
                colorbar_title = f"{c1[:10]}%"
                title_text = f"{c1} Vote Share"
            else:
                # Leading vs. Runner-up margin
                top_names = overall.index[:2].tolist()
                c1, c2 = top_names[0], top_names[1]
                
                df_pivot = df_race[df_race["candidate"].isin(top_names)].pivot(
                    index="srprec", columns="candidate", values="votes"
                ).fillna(0)
                
                total_top2 = df_pivot[c1] + df_pivot[c2]
                df_pivot["_val"] = np.where(total_top2 > 0, (df_pivot[c1] / total_top2 * 100), np.nan)
                df_pivot["_label"] = [
                    f"{v:.1f}% {c1} / {100-v:.1f}% {c2}" if not np.isnan(v) else "No data"
                    for v in df_pivot["_val"]
                ]
                df_sub = df_pivot.reset_index()
                colorscale = [
                    [0.00, "#d6604d"], [0.42, "#f7c4bb"], [0.50, "#f5f0f0"],
                    [0.58, "#c8d9ef"], [1.00, "#2166ac"]
                ]
                colorbar_title = f"{c1[:10]}..."
                title_text = f"{c1} (Blue) vs {c2} (Red)"
            zmin, zmax = 0, 100

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 2: WIDE-FORMAT RACES (Federal / Statewide)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        df    = year_data["main"].copy()
        cands = race["candidates"]
        if not cands:
            return _empty_fig("No candidate data")

        dist_col = cands[0]["dist_col"]
        dist_val = cands[0]["dist_val"]
        df_sub   = df[df[dist_col] == dist_val].copy() if (dist_col and dist_val) else df.copy()
        df_sub   = df_sub[df_sub["TOTREG"] > 0].copy()

        if map_mode == "turnout":
            df_sub["_val"]   = (df_sub["TOTVOTE"] / df_sub["TOTREG"] * 100).round(1)
            df_sub["_label"] = df_sub["_val"].apply(lambda v: f"{v:.1f}% turnout")
            colorscale       = [[0.0, "#f7fbff"], [0.3, "#9ecae1"],
                                [0.65, "#3182bd"], [1.0, "#08306b"]]
            zmin, zmax       = 0, 100
            colorbar_title   = "Turnout %"
            title_text       = "Voter Turnout by Precinct"
        else:
            is_measure = (category == "measures")
            dem_cols   = [c["col"] for c in cands if c["party"] in ("DEM", "Y")]
            rep_cols   = [c["col"] for c in cands if c["party"] in ("REP", "N")]

            if not dem_cols:
                return _empty_fig("Vote map requires DEM/REP or Yes/No candidates")

            dem_votes = df_sub[[c for c in dem_cols if c in df_sub.columns]].sum(axis=1).fillna(0)
            rep_votes = df_sub[[c for c in rep_cols if c in df_sub.columns]].sum(axis=1).fillna(0)
            total     = dem_votes + rep_votes
            dem_pct   = np.where(total > 0, dem_votes / total * 100, np.nan)

            df_sub["_val"]   = np.round(dem_pct, 1)
            label_a = "Yes" if is_measure else "Dem"
            label_b = "No"  if is_measure else "Rep"
            df_sub["_label"] = [
                f"{v:.1f}% {label_a}  /  {100-v:.1f}% {label_b}"
                if not np.isnan(v) else "No data"
                for v in dem_pct
            ]
            colorscale = [
                [0.00, "#d6604d"], [0.42, "#f7c4bb"], [0.50, "#f5f0f0"],
                [0.58, "#c8d9ef"], [1.00, "#2166ac"]
            ]
            zmin, zmax     = 0, 100
            colorbar_title = f"{'Yes' if is_measure else 'Dem'} %"
            title_text     = f"{'Yes' if is_measure else 'Dem'} Vote Share by Precinct"

    # ─────────────────────────────────────────────────────────────────────────
    # MAP GENERATION (Shared)
    # ─────────────────────────────────────────────────────────────────────────
    relevant_ids = set(df_sub["srprec"].dropna().astype(int))
    filtered_geo = {
        **geojson,
        "features": [f for f in geojson["features"] if f.get("id") in relevant_ids],
    }

    fig = go.Figure(go.Choroplethmapbox(
        geojson=filtered_geo,
        locations=df_sub["srprec"].astype(int),
        z=df_sub["_val"],
        text=df_sub["_label"],
        hovertemplate="Precinct %{location}<br>%{text}<extra></extra>",
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        marker_line_width=0.4,
        marker_line_color="white",
        marker_opacity=0.85,
        colorbar=dict(
            title=dict(text=colorbar_title, side="right"),
            thickness=14, len=0.65, ticksuffix="%",
        ),
    ))
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=9.2,
        mapbox_center={"lat": 33.70, "lon": -117.82},
        **chart_layout(height=480, margin=dict(l=0, r=0, t=36, b=0)),
        title_text=title_text,
        title_x=0,
    )
    return fig

def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=13, color="#888")
    )
    fig.update_layout(
        height=200,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY CARDS
# ─────────────────────────────────────────────────────────────────────────────

def make_stat_card(label: str, value: str, sub: str = "", color: str = "#4a90d9") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="text-muted mb-1",
                   style={"fontSize": "0.78rem", "fontWeight": 600}),
            html.H4(value, className="mb-0 fw-bold", style={"color": color}),
            html.Small(sub, className="text-muted") if sub else html.Span(),
        ], className="py-2 px-3"),
        className="border-0 shadow-sm h-100",
        style={"borderRadius": "10px", "background": "#f8f9fa"},
    )


def make_summary_row(results: dict) -> dbc.Row:
    if not results:
        return dbc.Row()

    winner       = results.get("winner", "—")
    winner_info  = results["candidates"].get(winner, {})
    winner_party = winner_info.get("party", "IND")
    winner_pct   = winner_info.get("pct", 0)
    winner_color = PARTY_COLORS.get(winner_party, DEFAULT_COLOR)
    total_votes  = results.get("total_race_votes", 0)
    margin       = results.get("margin", 0)
    turnout      = results.get("turnout_pct")
    totreg       = results.get("totreg")

    cols = [
        dbc.Col(make_stat_card("🏆 Winner", winner,
                               f"{winner_party}  ·  {winner_pct:.1f}%", winner_color), md=3),
        dbc.Col(make_stat_card("📊 Margin", f"{margin:.1f} pts",
                               "winner – runner-up"), md=3),
        dbc.Col(make_stat_card("🗳 Total Votes", f"{total_votes:,}",
                               f"{totreg:,} registered" if totreg else ""), md=3),
    ]
    if turnout is not None:
        cols.append(dbc.Col(make_stat_card("📈 Turnout", f"{turnout:.1f}%",
                                           "of registered voters"), md=3))
    else:
        cols.append(dbc.Col(make_stat_card("📈 Turnout", "N/A",
                                           "not available for this race"), md=3))
    return dbc.Row(cols, className="g-2 mb-3")


# ─────────────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────────────

ALL_DATA     = load_all_data()
RACE_INDEXES = {yr: build_race_index(d) for yr, d in ALL_DATA.items()}
AVAIL_YEARS  = sorted(ALL_DATA.keys(), reverse=True)

if not AVAIL_YEARS:
    raise RuntimeError(
        "No election data found.\n"
        "Create data/<year>/sov_main.csv (and optionally local CSVs + *.codes)\n"
        "and re-run this script."
    )

DEFAULT_YEAR = AVAIL_YEARS[0]


# ─────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="OC Elections Dashboard",
)

app.layout = dbc.Container(
    fluid=True,
    style={"fontFamily": "Inter, system-ui, sans-serif"},
    children=[
        # ── Header ───────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.H3("🗳 Orange County Elections Dashboard",
                        className="mb-0 fw-bold mt-3"),
            ], md=8),
            dbc.Col([
                dbc.Label("Election Year", html_for="year-selector",
                          className="fw-semibold small mb-1"),
                dcc.Dropdown(
                    id="year-selector",
                    options=[{"label": str(y), "value": y} for y in AVAIL_YEARS],
                    value=DEFAULT_YEAR,
                    clearable=False,
                    style={"fontSize": "0.95rem"},
                ),
            ], md=4, className="d-flex flex-column justify-content-center"),
        ], className="border-bottom pb-2"),

        # ── Category Tabs + Race Dropdown ─────────────────────────────────────
        dbc.Row(className="mt-3", children=[
            dbc.Col([
                dbc.Tabs(
                    id="category-tabs",
                    active_tab="federal",
                    children=[
                        dbc.Tab(label=label, tab_id=cat_id)
                        for cat_id, label in RACE_CATEGORIES
                    ],
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Race", html_for="race-dropdown",
                                  className="fw-semibold small mt-3 mb-1"),
                        dcc.Dropdown(
                            id="race-dropdown",
                            clearable=False,
                            placeholder="Select a race…",
                            style={"fontSize": "0.92rem"},
                        ),
                    ]),
                ], className="my-2"),
            ]),
        ]),

        # ── Summary Cards ─────────────────────────────────────────────────────
        html.Div(id="summary-cards"),

        # ── Row 1: Vote Share  |  Precinct Map ───────────────────────────────
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="chart-vote-share", config={"displayModeBar": False}),
            ], md=5),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Map mode",
                                  className="fw-semibold small me-2 mb-0 align-self-center"),
                        dbc.RadioItems(
                            id="map-mode",
                            options=[
                                {"label": " Vote share", "value": "vote"},
                                {"label": " Turnout",    "value": "turnout"},
                            ],
                            value="vote",
                            inline=True,
                            className="small",
                        ),
                    ], className="d-flex align-items-center mb-1"),
                ]),
                dcc.Graph(
                    id="chart-map",
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        "scrollZoom": True,
                    },
                ),
            ], md=7),
        ], className="mb-3"),

        # ── Row 2: Margin Trend  |  Turnout Trend ────────────────────────────
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="chart-margin-trend", config={"displayModeBar": False}),
            ], md=6),
            dbc.Col([
                dcc.Graph(id="chart-turnout-trend", config={"displayModeBar": False}),
            ], md=6),
        ], className="mb-4"),

        dcc.Store(id="selected-race"),
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("race-dropdown", "options"),
    Output("race-dropdown", "value"),
    Input("category-tabs",  "active_tab"),
    Input("year-selector",  "value"),
)
def update_race_options(active_tab: str, year: int):
    race_index = RACE_INDEXES.get(year, {})
    cat_races  = race_index.get(active_tab, {})
    if not cat_races:
        return [], None
    options = sorted(
        [{"label": v["label"], "value": rid} for rid, v in cat_races.items()],
        key=lambda o: o["label"],
    )
    return options, options[0]["value"]


@app.callback(
    Output("summary-cards",       "children"),
    Output("chart-vote-share",    "figure"),
    Output("chart-map",           "figure"),
    Output("chart-margin-trend",  "figure"),
    Output("chart-turnout-trend", "figure"),
    Input("race-dropdown",        "value"),
    Input("year-selector",        "value"),
    Input("category-tabs",        "active_tab"),
    Input("map-mode",             "value"),
)
def update_dashboard(race_id: str, year: int, category: str, map_mode: str):
    if not race_id or not year or year not in ALL_DATA:
        empty = _empty_fig("Select a race above")
        return html.Div(), empty, empty, empty, empty

    year_data  = ALL_DATA[year]
    race_index = RACE_INDEXES[year]
    results    = compute_results(year_data, category, race_id, race_index)

    summary     = make_summary_row(results)
    vote_fig    = fig_vote_share(results)
    map_fig     = fig_precinct_map(year_data, category, race_id, race_index, map_mode or "vote")
    margin_fig  = fig_margin_trend(ALL_DATA, RACE_INDEXES, category, race_id)
    turnout_fig = fig_turnout_trend(ALL_DATA, RACE_INDEXES, category, race_id)

    return summary, vote_fig, map_fig, margin_fig, turnout_fig


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  Years loaded: {sorted(ALL_DATA.keys())}")
    for yr, idx in RACE_INDEXES.items():
        total_races = sum(len(v) for v in idx.values())
        print(f"  {yr}: {total_races} races across {len(idx)} categories")
    print("\n  → http://localhost:8050\n")
    app.run(debug=True, host="0.0.0.0", port=8050)