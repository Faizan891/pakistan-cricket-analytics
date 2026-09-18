"""
utils/data_processor.py
------------------------
Pure Pandas aggregation and transformation functions.
All functions accept a (pre-filtered) DataFrame and return either a
scalar, a Series, or a summary DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ===========================================================================
# FILTER HELPERS
# ===========================================================================

def filter_matches(
    df: pd.DataFrame,
    formats: list[str]  | None = None,
    year_range: tuple[int, int] | None = None,
    opponents: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply sidebar filters to the matches DataFrame.

    Parameters
    ----------
    df         : Raw / cleaned matches DataFrame.
    formats    : List of formats to keep, e.g. ['T20', 'ODI'].
                 Pass None to keep all.
    year_range : (min_year, max_year) inclusive. Pass None to keep all.
    opponents  : List of opponent team names to keep. Pass None to keep all.

    Returns
    -------
    Filtered copy of the DataFrame.
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if formats:
        mask &= df["format"].isin(formats)

    if year_range:
        lo, hi = year_range
        mask &= (df["year"] >= lo) & (df["year"] <= hi)

    if opponents:
        mask &= df["opponent"].isin(opponents)

    return df[mask].copy()


def filter_players(
    df: pd.DataFrame,
    formats: list[str] | None = None,
    year_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Apply format and year-range filters to the players DataFrame."""
    mask = pd.Series([True] * len(df), index=df.index)

    if formats:
        mask &= df["format"].isin(formats)

    if year_range:
        lo, hi = year_range
        mask &= (df["year"] >= lo) & (df["year"] <= hi)

    return df[mask].copy()


# ===========================================================================
# KPI AGGREGATIONS
# ===========================================================================

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPI metrics from a (possibly filtered) matches DataFrame.

    Returns a dict with:
      total_matches   : int
      wins            : int
      losses          : int
      draws           : int
      win_pct         : float  (0-100)
      highest_score   : int
      avg_run_rate    : float
      avg_runs        : float
    """
    total   = len(df)
    wins    = (df["result"] == "Win").sum()
    losses  = (df["result"] == "Loss").sum()
    draws   = ((df["result"] == "Draw") | (df["result"] == "No Result")).sum()
    win_pct = round(wins / total * 100, 1) if total > 0 else 0.0

    highest_score = int(df["pak_runs"].max()) if total > 0 else 0
    avg_run_rate  = round(df["run_rate"].mean(), 2) if total > 0 else 0.0
    avg_runs      = round(df["pak_runs"].mean(), 1) if total > 0 else 0.0

    return {
        "total_matches" : total,
        "wins"          : int(wins),
        "losses"        : int(losses),
        "draws"         : int(draws),
        "win_pct"       : win_pct,
        "highest_score" : highest_score,
        "avg_run_rate"  : avg_run_rate,
        "avg_runs"      : avg_runs,
    }


# ===========================================================================
# WIN / LOSS BREAKDOWN
# ===========================================================================

def result_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with result counts and percentages.

    Columns: result, count, percentage
    """
    counts = (
        df["result"]
        .value_counts()
        .rename_axis("result")
        .reset_index(name="count")
    )
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


# ===========================================================================
# PERFORMANCE OVER TIME
# ===========================================================================

def performance_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate match data by year.

    Returns columns: year, matches, wins, losses, avg_runs, avg_run_rate, win_pct
    """
    grp = df.groupby("year")

    summary = pd.DataFrame({
        "year"         : grp["year"].first(),
        "matches"      : grp.size(),
        "wins"         : (grp["result"].apply(lambda s: (s == "Win").sum())),
        "losses"       : (grp["result"].apply(lambda s: (s == "Loss").sum())),
        "avg_runs"     : grp["pak_runs"].mean().round(1),
        "avg_run_rate" : grp["run_rate"].mean().round(2),
        "total_runs"   : grp["pak_runs"].sum(),
    }).reset_index(drop=True)

    summary["win_pct"] = (summary["wins"] / summary["matches"] * 100).round(1)
    summary.sort_values("year", inplace=True)
    return summary


# ===========================================================================
# HEAD-TO-HEAD ANALYSIS
# ===========================================================================

def head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate match data by opponent.

    Returns columns: opponent, matches, wins, losses, draws, win_pct, avg_runs
    """
    grp = df.groupby("opponent")

    h2h = pd.DataFrame({
        "matches" : grp.size(),
        "wins"    : grp["result"].apply(lambda s: (s == "Win").sum()),
        "losses"  : grp["result"].apply(lambda s: (s == "Loss").sum()),
        "draws"   : grp["result"].apply(
                        lambda s: ((s == "Draw") | (s == "No Result")).sum()),
        "avg_runs": grp["pak_runs"].mean().round(1),
    }).reset_index()

    h2h["win_pct"] = (h2h["wins"] / h2h["matches"] * 100).round(1)
    h2h.sort_values("matches", ascending=False, inplace=True)
    return h2h


# ===========================================================================
# PLAYER LEADERBOARDS
# ===========================================================================

def top_scorers(df_players: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    Aggregate total runs by player across the filtered period.

    Returns columns: player, runs, batting_avg, strike_rate, hundreds, fifties
    """
    grp = df_players.groupby("player")
    summary = pd.DataFrame({
        "runs"        : grp["runs"].sum(),
        "batting_avg" : grp["batting_avg"].mean().round(2),
        "strike_rate" : grp["strike_rate"].mean().round(1),
        "hundreds"    : grp["hundreds"].sum(),
        "fifties"     : grp["fifties"].sum(),
        "matches"     : grp["matches"].sum(),
    }).reset_index()

    return summary.nlargest(top_n, "runs").reset_index(drop=True)


def top_wicket_takers(df_players: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    Aggregate total wickets by player across the filtered period.

    Returns columns: player, wickets, bowling_avg, economy, five_wickets
    """
    grp = df_players.groupby("player")
    summary = pd.DataFrame({
        "wickets"     : grp["wickets"].sum(),
        "bowling_avg" : grp["bowling_avg"].mean().round(2),
        "economy"     : grp["economy"].mean().round(2),
        "five_wickets": grp["five_wickets"].sum(),
        "matches"     : grp["matches"].sum(),
    }).reset_index()

    return summary.nlargest(top_n, "wickets").reset_index(drop=True)


def win_loss_by_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns win/loss/draw counts broken down by format.

    Useful for grouped bar charts.
    """
    rows = []
    for fmt, grp in df.groupby("format"):
        rows.append({
            "format" : fmt,
            "Win"    : (grp["result"] == "Win").sum(),
            "Loss"   : (grp["result"] == "Loss").sum(),
            "Draw"   : ((grp["result"] == "Draw") | (grp["result"] == "No Result")).sum(),
        })
    return pd.DataFrame(rows)
