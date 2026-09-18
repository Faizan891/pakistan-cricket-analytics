"""
utils/data_loader.py
--------------------
Responsible for reading, cleaning, and type-correcting the raw CSV files.
All functions return clean DataFrames ready for analysis.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_CSV  = os.path.join(_ROOT, "data", "match_results.csv")
PLAYERS_CSV  = os.path.join(_ROOT, "data", "player_stats.csv")


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_matches() -> pd.DataFrame:
    """
    Load and clean the match_results CSV.

    Cleaning steps
    --------------
    1. Parse 'date' as datetime.
    2. Strip whitespace from string columns.
    3. Coerce numeric columns to correct dtypes.
    4. Drop rows where critical fields (date, format, result) are missing.
    5. Clip run_rate to a sensible range [0, 36] (max T20 RR).
    """
    df = pd.read_csv(MATCHES_CSV)

    # -- Date parsing --------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # -- String cleanup ------------------------------------------------------
    str_cols = ["format", "opponent", "venue", "top_scorer",
                "top_wicket_taker", "result"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Re-normalise the result column to consistent casing
    df["result"] = df["result"].replace({
        "Win": "Win", "Loss": "Loss", "Draw": "Draw", "No Result": "No Result"
    })

    # -- Numeric coercion ----------------------------------------------------
    num_cols = ["pak_runs", "pak_wickets", "opp_wickets",
                "overs_played", "run_rate", "top_scorer_runs", "top_wickets"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["run_rate"] = df["run_rate"].clip(0, 36)

    # -- Drop critical nulls -------------------------------------------------
    df.dropna(subset=["date", "format", "result"], inplace=True)

    # -- Derived columns -----------------------------------------------------
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_players() -> pd.DataFrame:
    """
    Load and clean the player_stats CSV.

    Cleaning steps
    --------------
    1. Strip whitespace from string columns.
    2. Coerce all numeric columns to float/int.
    3. Clip averages and economy to realistic ranges.
    4. Drop rows with missing 'player' or 'year'.
    """
    df = pd.read_csv(PLAYERS_CSV)

    # -- String cleanup ------------------------------------------------------
    for col in ["player", "role", "format"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # -- Numeric coercion ----------------------------------------------------
    num_cols = ["year", "matches", "innings", "runs", "batting_avg",
                "strike_rate", "hundreds", "fifties",
                "wickets", "bowling_avg", "economy", "five_wickets"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip to sensible ranges
    df["batting_avg"]  = df["batting_avg"].clip(0, 200)
    df["bowling_avg"]  = df["bowling_avg"].clip(0, 200)
    df["economy"]      = df["economy"].clip(0, 36)
    df["strike_rate"]  = df["strike_rate"].clip(0, 400)

    # -- Drop critical nulls -------------------------------------------------
    df.dropna(subset=["player", "year"], inplace=True)
    df["year"] = df["year"].astype(int)

    df.reset_index(drop=True, inplace=True)
    return df
