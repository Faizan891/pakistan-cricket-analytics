"""
utils/data_processor.py
------------------------
Pure Pandas aggregation and transformation functions for cricket analytics.
Covers:
  - Sidebar filters (Format, Year, Opponent, Innings)
  - Core KPIs & Result breakdowns
  - Toss Impact & Decision Analysis
  - Innings Splits & Venue Chasing Dynamics
  - Phase Analysis (Powerplay, Middle, Death) for Batters & Bowlers
  - Head-to-Head and Leaderboard aggregations
"""

import pandas as pd
import numpy as np


# ===========================================================================
# 1. FILTER HELPERS
# ===========================================================================

def filter_matches(
    df: pd.DataFrame,
    formats: list[str] | None = None,
    year_range: tuple[int, int] | None = None,
    opponents: list[str] | None = None,
    innings: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply sidebar filters to the matches DataFrame.

    Parameters
    ----------
    df         : Cleaned matches DataFrame.
    formats    : List of formats to keep (e.g. ['T20', 'ODI']). None for all.
    year_range : (min_year, max_year) inclusive. None for all.
    opponents  : List of opponent team names to keep. None for all.
    innings    : '1st', '2nd', or list, or 'All'. None for all.
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if formats:
        mask &= df["format"].isin(formats)

    if year_range:
        lo, hi = year_range
        mask &= (df["year"] >= lo) & (df["year"] <= hi)

    if opponents:
        mask &= df["opponent"].isin(opponents)

    if innings and innings not in ("All", "Both", ["1st", "2nd"]):
        if isinstance(innings, str):
            if "1st" in innings:
                mask &= (df["pak_innings"] == "1st")
            elif "2nd" in innings:
                mask &= (df["pak_innings"] == "2nd")
        elif isinstance(innings, (list, tuple)):
            mask &= df["pak_innings"].isin(innings)

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
# 2. CORE KPI AGGREGATIONS
# ===========================================================================

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPI metrics from filtered match data.
    Includes toss win % and 1st vs 2nd innings splits.
    """
    total = len(df)
    if total == 0:
        return {
            "total_matches": 0, "wins": 0, "losses": 0, "draws": 0,
            "win_pct": 0.0, "highest_score": 0, "avg_run_rate": 0.0, "avg_runs": 0.0,
            "toss_win_pct": 0.0, "toss_win_match_win_pct": 0.0,
            "bat_1st_win_pct": 0.0, "chase_win_pct": 0.0,
        }

    wins    = int((df["result"] == "Win").sum())
    losses  = int((df["result"] == "Loss").sum())
    draws   = int(((df["result"] == "Draw") | (df["result"] == "No Result")).sum())
    win_pct = round(wins / total * 100, 1)

    highest_score = int(df["pak_runs"].max())
    avg_run_rate  = round(df["run_rate"].mean(), 2)
    avg_runs      = round(df["pak_runs"].mean(), 1)

    # Toss impact KPIs
    toss_won_df = df[df["toss_won"]]
    toss_win_pct = round(len(toss_won_df) / total * 100, 1)
    toss_win_match_win_pct = (
        round((toss_won_df["result"] == "Win").sum() / len(toss_won_df) * 100, 1)
        if len(toss_won_df) > 0 else 0.0
    )

    # Innings splits
    bat1st_df = df[df["pak_innings"] == "1st"]
    bat2nd_df = df[df["pak_innings"] == "2nd"]

    bat_1st_win_pct = (
        round((bat1st_df["result"] == "Win").sum() / len(bat1st_df) * 100, 1)
        if len(bat1st_df) > 0 else 0.0
    )
    chase_win_pct = (
        round((bat2nd_df["result"] == "Win").sum() / len(bat2nd_df) * 100, 1)
        if len(bat2nd_df) > 0 else 0.0
    )

    return {
        "total_matches"          : total,
        "wins"                   : wins,
        "losses"                 : losses,
        "draws"                  : draws,
        "win_pct"                : win_pct,
        "highest_score"          : highest_score,
        "avg_run_rate"           : avg_run_rate,
        "avg_runs"               : avg_runs,
        "toss_win_pct"           : toss_win_pct,
        "toss_win_match_win_pct" : toss_win_match_win_pct,
        "bat_1st_win_pct"        : bat_1st_win_pct,
        "chase_win_pct"          : chase_win_pct,
    }


# ===========================================================================
# 3. TOSS IMPACT & INNINGS SPLITS
# ===========================================================================

def toss_impact_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze the impact of winning/losing the toss and toss decisions (Bat vs Field).
    Returns a dictionary of structured summary DataFrames.
    """
    if df.empty:
        empty_df = pd.DataFrame(columns=["category", "matches", "wins", "losses", "win_pct"])
        return {"toss_outcome": empty_df, "decision_outcome": empty_df}

    # 1. Toss Won vs Toss Lost
    toss_rows = []
    for toss_status, label in [(True, "Toss Won"), (False, "Toss Lost")]:
        sub = df[df["toss_won"] == toss_status]
        m = len(sub)
        w = int((sub["result"] == "Win").sum())
        l = int((sub["result"] == "Loss").sum())
        pct = round(w / m * 100, 1) if m > 0 else 0.0
        toss_rows.append({"Category": label, "Matches": m, "Wins": w, "Losses": l, "Win %": pct})
    toss_df = pd.DataFrame(toss_rows)

    # 2. Toss Decision (Bat vs Field) for Pakistan
    dec_rows = []
    for dec in ["Bat", "Field"]:
        sub = df[df["toss_decision"].str.capitalize() == dec]
        m = len(sub)
        w = int((sub["result"] == "Win").sum())
        l = int((sub["result"] == "Loss").sum())
        pct = round(w / m * 100, 1) if m > 0 else 0.0
        dec_rows.append({"Decision": f"Chose to {dec}", "Matches": m, "Wins": w, "Losses": l, "Win %": pct})
    dec_df = pd.DataFrame(dec_rows)

    # 3. Batting 1st vs Chasing (2nd Innings)
    inn_rows = []
    for inn, label in [("1st", "Batting 1st (Defending)"), ("2nd", "Batting 2nd (Chasing)")]:
        sub = df[df["pak_innings"] == inn]
        m = len(sub)
        w = int((sub["result"] == "Win").sum())
        l = int((sub["result"] == "Loss").sum())
        avg_s = round(sub["pak_runs"].mean(), 1) if m > 0 else 0.0
        pct = round(w / m * 100, 1) if m > 0 else 0.0
        inn_rows.append({"Innings": label, "Matches": m, "Wins": w, "Losses": l, "Win %": pct, "Avg Runs": avg_s})
    inn_df = pd.DataFrame(inn_rows)

    return {
        "toss_outcome": toss_df,
        "decision_outcome": dec_df,
        "innings_outcome": inn_df,
    }


def venue_innings_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Batting 1st vs Chasing win rates and average scores across venues.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "Venue", "Total Matches", "Bat 1st Matches", "Bat 1st Wins", "Bat 1st Win %",
            "Chase Matches", "Chase Wins", "Chase Win %", "Avg 1st Inn Score", "Avg 2nd Inn Score"
        ])

    records = []
    for venue, grp in df.groupby("venue"):
        tot = len(grp)
        first_grp = grp[grp["pak_innings"] == "1st"]
        second_grp = grp[grp["pak_innings"] == "2nd"]

        m_1st = len(first_grp)
        w_1st = int((first_grp["result"] == "Win").sum())
        pct_1st = round(w_1st / m_1st * 100, 1) if m_1st > 0 else 0.0
        avg_1st = round(grp["first_innings_score"].mean(), 1) if "first_innings_score" in grp.columns else 0.0

        m_2nd = len(second_grp)
        w_2nd = int((second_grp["result"] == "Win").sum())
        pct_2nd = round(w_2nd / m_2nd * 100, 1) if m_2nd > 0 else 0.0
        avg_2nd = round(grp["second_innings_score"].mean(), 1) if "second_innings_score" in grp.columns else 0.0

        records.append({
            "Venue": venue,
            "Total Matches": tot,
            "Bat 1st Matches": m_1st,
            "Bat 1st Wins": w_1st,
            "Bat 1st Win %": pct_1st,
            "Chase Matches": m_2nd,
            "Chase Wins": w_2nd,
            "Chase Win %": pct_2nd,
            "Avg 1st Inn Score": avg_1st,
            "Avg 2nd Inn Score": avg_2nd,
        })

    out_df = pd.DataFrame(records)
    return out_df.sort_values("Total Matches", ascending=False).reset_index(drop=True)


# ===========================================================================
# 4. PHASE ANALYSIS MODULE (Powerplay, Middle, Death)
# ===========================================================================

def phase_overview_summary(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate overall performance across Powerplay (Overs 1-6),
    Middle (Overs 7-15), and Death (Overs 16-20).
    """
    if df_players.empty:
        return pd.DataFrame()

    phases = [
        {
            "Phase": "Powerplay (Overs 1–6)",
            "Total Runs": int(df_players["powerplay_runs"].sum()),
            "Total Balls": int(df_players["powerplay_balls"].sum()),
            "Wickets Taken": int(df_players["powerplay_wickets"].sum()),
            "Avg Economy": round(df_players[df_players["powerplay_economy"] > 0]["powerplay_economy"].mean(), 2),
        },
        {
            "Phase": "Middle Overs (Overs 7–15)",
            "Total Runs": int(df_players["middle_runs"].sum()),
            "Total Balls": int(df_players["middle_balls"].sum()),
            "Wickets Taken": int(df_players["middle_wickets"].sum()),
            "Avg Economy": round(df_players[df_players["middle_economy"] > 0]["middle_economy"].mean(), 2),
        },
        {
            "Phase": "Death Overs (Overs 16–20)",
            "Total Runs": int(df_players["death_runs"].sum()),
            "Total Balls": int(df_players["death_balls"].sum()),
            "Wickets Taken": int(df_players["death_wickets"].sum()),
            "Avg Economy": round(df_players[df_players["death_economy"] > 0]["death_economy"].mean(), 2),
        },
    ]
    summary = pd.DataFrame(phases)
    summary["Batting Strike Rate"] = np.where(
        summary["Total Balls"] > 0,
        (summary["Total Runs"] / summary["Total Balls"] * 100).round(1),
        0.0
    )
    return summary


def top_phase_batters(df_players: pd.DataFrame, phase: str = "Powerplay", top_n: int = 8) -> pd.DataFrame:
    """
    Return top run-scorers and strike-rate performers for a specific phase.
    Phase options: 'Powerplay', 'Middle', 'Death'.
    """
    prefix = phase.lower()
    runs_col = f"{prefix}_runs"
    balls_col = f"{prefix}_balls"

    if runs_col not in df_players.columns:
        return pd.DataFrame()

    grp = df_players.groupby("player")
    res = pd.DataFrame({
        "Runs": grp[runs_col].sum(),
        "Balls": grp[balls_col].sum(),
        "Total Runs": grp["runs"].sum(),
        "Overall SR": grp["strike_rate"].mean().round(1),
        "Matches": grp["matches"].sum(),
    }).reset_index()

    res["Strike Rate"] = np.where(
        res["Balls"] > 0,
        (res["Runs"] / res["Balls"] * 100).round(1),
        0.0
    )
    res["Phase Run Share %"] = np.where(
        res["Total Runs"] > 0,
        (res["Runs"] / res["Total Runs"] * 100).round(1),
        0.0
    )

    # Sort by phase runs and filter positive contributors
    res = res[res["Runs"] > 0].sort_values("Runs", ascending=False).head(top_n).reset_index(drop=True)
    return res


def top_phase_bowlers(df_players: pd.DataFrame, phase: str = "Powerplay", top_n: int = 8) -> pd.DataFrame:
    """
    Return top wicket-takers and economy leaders for a specific phase.
    Phase options: 'Powerplay', 'Middle', 'Death'.
    """
    prefix = phase.lower()
    wkt_col = f"{prefix}_wickets"
    eco_col = f"{prefix}_economy"

    if wkt_col not in df_players.columns:
        return pd.DataFrame()

    # Filter bowlers only or players with wickets
    bowlers = df_players[df_players["role"] == "Bowler"]
    if bowlers.empty:
        bowlers = df_players

    grp = bowlers.groupby("player")
    res = pd.DataFrame({
        "Wickets": grp[wkt_col].sum(),
        "Total Wickets": grp["wickets"].sum(),
        "Economy": grp[eco_col].apply(lambda s: s[s > 0].mean()).round(2).fillna(8.0),
        "Overall Economy": grp["economy"].mean().round(2),
        "Matches": grp["matches"].sum(),
    }).reset_index()

    res["Phase Wkt Share %"] = np.where(
        res["Total Wickets"] > 0,
        (res["Wickets"] / res["Total Wickets"] * 100).round(1),
        0.0
    )

    # Sort primarily by Wickets, secondarily by Economy ascending
    res = res.sort_values(["Wickets", "Economy"], ascending=[False, True]).head(top_n).reset_index(drop=True)
    return res


def middle_overs_spin_vs_pace(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Middle Overs (Overs 7-15) spin control vs pace bowling.
    """
    spinners = ["Shadab Khan", "Imad Wasim", "Abrar Ahmed"]
    pacers   = ["Shaheen Afridi", "Naseem Shah", "Haris Rauf", "Mohammad Wasim Jr", "Zaman Khan"]

    spin_df = df_players[df_players["player"].isin(spinners)]
    pace_df = df_players[df_players["player"].isin(pacers)]

    records = [
        {
            "Bowling Type": "Spin Control",
            "Bowlers": ", ".join(spinners),
            "Middle Wickets": int(spin_df["middle_wickets"].sum()),
            "Avg Middle Economy": round(spin_df[spin_df["middle_economy"] > 0]["middle_economy"].mean(), 2) if not spin_df.empty else 0.0,
        },
        {
            "Bowling Type": "Pace Attack",
            "Bowlers": "Shaheen, Naseem, Haris, Wasim Jr, Zaman",
            "Middle Wickets": int(pace_df["middle_wickets"].sum()),
            "Avg Middle Economy": round(pace_df[pace_df["middle_economy"] > 0]["middle_economy"].mean(), 2) if not pace_df.empty else 0.0,
        }
    ]
    return pd.DataFrame(records)


def death_overs_finishing_metrics(df_players: pd.DataFrame) -> dict:
    """
    Return death overs acceleration metrics for batters and death economy for bowlers.
    """
    batters = top_phase_batters(df_players, phase="Death", top_n=6)
    bowlers = top_phase_bowlers(df_players, phase="Death", top_n=6)

    # Estimate boundary frequency in death overs (approx 65% runs come in boundaries at death)
    if not batters.empty:
        batters["Est. Boundary Runs"] = (batters["Runs"] * 0.65).astype(int)
        batters["Runs/Ball"] = (batters["Runs"] / batters["Balls"].replace(0, 1)).round(2)

    return {
        "batters": batters,
        "bowlers": bowlers,
    }


# ===========================================================================
# 5. GENERAL CHARTS & LEADERBOARDS
# ===========================================================================

def result_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with result counts and percentages."""
    counts = (
        df["result"]
        .value_counts()
        .rename_axis("result")
        .reset_index(name="count")
    )
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts


def performance_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate match data by year."""
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


def head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate match data by opponent."""
    grp = df.groupby("opponent")
    h2h = pd.DataFrame({
        "matches" : grp.size(),
        "wins"    : grp["result"].apply(lambda s: (s == "Win").sum()),
        "losses"  : grp["result"].apply(lambda s: (s == "Loss").sum()),
        "draws"   : grp["result"].apply(lambda s: ((s == "Draw") | (s == "No Result")).sum()),
        "avg_runs": grp["pak_runs"].mean().round(1),
    }).reset_index()

    h2h["win_pct"] = (h2h["wins"] / h2h["matches"] * 100).round(1)
    h2h.sort_values("matches", ascending=False, inplace=True)
    return h2h


def top_scorers(df_players: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Aggregate total runs by player."""
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
    """Aggregate total wickets by player."""
    grp = df_players.groupby("player")
    summary = pd.DataFrame({
        "wickets"     : grp["wickets"].sum(),
        "bowling_avg" : grp["bowling_avg"].mean().round(2),
        "economy"     : grp["economy"].mean().round(2),
        "five_wickets": grp["five_wickets"].sum(),
        "matches"     : grp["matches"].sum(),
    }).reset_index()

    return summary.nlargest(top_n, "wickets").reset_index(drop=True)
