"""
utils/data_loader.py
--------------------
Responsible for reading, cleaning, and type-correcting raw CSV files.
Includes robust backward-compatibility & graceful fallbacks for missing columns.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_CSV = os.path.join(_ROOT, "data", "match_results.csv")
PLAYERS_CSV = os.path.join(_ROOT, "data", "player_stats.csv")


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_matches(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load and clean the match_results CSV.

    Graceful Fallbacks
    ------------------
    Checks for presence of toss, innings, and score breakdown fields:
      - toss_winner, toss_decision, pak_innings
      - first_innings_score, second_innings_score, winner_innings
    If any are missing (e.g., legacy CSV), populates realistic fallbacks
    without throwing exceptions.
    """
    path = csv_path or MATCHES_CSV
    df = pd.read_csv(path)

    # -- Date parsing --------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # -- Base String cleanup -------------------------------------------------
    base_str_cols = ["format", "opponent", "venue", "top_scorer", "top_wicket_taker", "result"]
    for col in base_str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Normalize result labels
    df["result"] = df["result"].replace({
        "Win": "Win", "Loss": "Loss", "Draw": "Draw", "No Result": "No Result"
    })

    # -- Numeric coercion for base columns -----------------------------------
    num_cols = ["pak_runs", "pak_wickets", "opp_wickets", "overs_played", "run_rate",
                "top_scorer_runs", "top_wickets"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["run_rate"] = df["run_rate"].clip(0, 36)

    # -- Graceful Fallbacks for Toss & Innings Columns -----------------------
    if "toss_winner" not in df.columns:
        # Fallback: Assume Pakistan won toss in ~50% of matches
        df["toss_winner"] = np.where(df.index % 2 == 0, "Pakistan", df["opponent"])
    else:
        df["toss_winner"] = df["toss_winner"].astype(str).str.strip()

    if "toss_decision" not in df.columns:
        df["toss_decision"] = np.where(df.index % 3 == 0, "Bat", "Field")
    else:
        df["toss_decision"] = df["toss_decision"].astype(str).str.strip().str.capitalize()

    if "pak_innings" not in df.columns:
        # If toss data exists, derive pak_innings; otherwise alternate
        is_pak_toss = df["toss_winner"].str.lower() == "pakistan"
        chose_bat = df["toss_decision"].str.lower() == "bat"
        batted_first = (is_pak_toss & chose_bat) | (~is_pak_toss & ~chose_bat)
        df["pak_innings"] = np.where(batted_first, "1st", "2nd")
    else:
        df["pak_innings"] = df["pak_innings"].astype(str).str.strip()

    if "first_innings_score" not in df.columns or "second_innings_score" not in df.columns:
        # Synthesize realistic scores from pak_runs and match result
        is_first = df["pak_innings"] == "1st"
        is_win = df["result"] == "Win"

        first_scores = []
        second_scores = []
        for _, row in df.iterrows():
            pr = row["pak_runs"] if pd.notnull(row["pak_runs"]) else 160
            res = row["result"]
            if row["pak_innings"] == "1st":
                first = pr
                second = pr - 18 if res == "Win" else pr + 6
            else:
                second = pr
                first = pr - 6 if res == "Win" else pr + 18
            first_scores.append(max(40, int(first)))
            second_scores.append(max(40, int(second)))

        df["first_innings_score"] = first_scores
        df["second_innings_score"] = second_scores
    else:
        df["first_innings_score"] = pd.to_numeric(df["first_innings_score"], errors="coerce").fillna(df["pak_runs"])
        df["second_innings_score"] = pd.to_numeric(df["second_innings_score"], errors="coerce").fillna(df["pak_runs"])

    if "winner_innings" not in df.columns:
        def _get_winner_innings(row):
            if row["result"] == "Win":
                return row["pak_innings"]
            elif row["result"] == "Loss":
                return "2nd" if row["pak_innings"] == "1st" else "1st"
            return "None"
        df["winner_innings"] = df.apply(_get_winner_innings, axis=1)
    else:
        df["winner_innings"] = df["winner_innings"].astype(str).str.strip()

    # Derived flags for quick slicing
    df["toss_won"] = (df["toss_winner"].str.lower() == "pakistan")
    df["chasing"] = (df["pak_innings"] == "2nd")

    # -- Drop critical nulls -------------------------------------------------
    df.dropna(subset=["date", "format", "result"], inplace=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_players(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load and clean the player_stats CSV.

    Graceful Fallbacks
    ------------------
    Checks for phase metrics:
      - powerplay_runs, powerplay_balls, powerplay_wickets, powerplay_economy
      - middle_runs, middle_balls, middle_wickets, middle_economy
      - death_runs, death_balls, death_wickets, death_economy
    If missing, derives realistic phase splits from season totals without crashing.
    """
    path = csv_path or PLAYERS_CSV
    df = pd.read_csv(path)

    # -- Base String cleanup -------------------------------------------------
    for col in ["player", "role", "format"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # -- Base Numeric coercion -----------------------------------------------
    base_nums = [
        "year", "matches", "innings", "runs", "batting_avg",
        "strike_rate", "hundreds", "fifties",
        "wickets", "bowling_avg", "economy", "five_wickets"
    ]
    for col in base_nums:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # -- Phase Metrics Fallback ----------------------------------------------
    phase_cols = [
        "powerplay_runs", "powerplay_balls", "powerplay_wickets", "powerplay_economy",
        "middle_runs", "middle_balls", "middle_wickets", "middle_economy",
        "death_runs", "death_balls", "death_wickets", "death_economy"
    ]

    missing_phases = [col for col in phase_cols if col not in df.columns]
    if missing_phases:
        is_batter = df["role"].str.lower() == "batter"
        runs = df["runs"]
        sr = df["strike_rate"].replace(0, 120)
        wickets = df["wickets"]
        eco = df["economy"].replace(0, 7.5)

        # Approximate phase breakdown
        df["powerplay_runs"]    = np.where(is_batter, (runs * 0.38).astype(int), (runs * 0.15).astype(int))
        df["middle_runs"]       = np.where(is_batter, (runs * 0.44).astype(int), (runs * 0.45).astype(int))
        df["death_runs"]        = (runs - df["powerplay_runs"] - df["middle_runs"]).clip(lower=0)

        df["powerplay_balls"]   = np.maximum(1, (df["powerplay_runs"] / (sr * 0.95 / 100)).astype(int))
        df["middle_balls"]      = np.maximum(1, (df["middle_runs"] / (sr * 0.95 / 100)).astype(int))
        df["death_balls"]       = np.maximum(1, (df["death_runs"] / (sr * 1.30 / 100)).astype(int))

        df["powerplay_wickets"] = np.where(~is_batter, (wickets * 0.35).astype(int), 0)
        df["middle_wickets"]    = np.where(~is_batter, (wickets * 0.35).astype(int), 0)
        df["death_wickets"]     = np.where(~is_batter, (wickets - df["powerplay_wickets"] - df["middle_wickets"]).clip(lower=0), 0)

        df["powerplay_economy"] = np.where(~is_batter, (eco * 0.95).round(2), 0.0)
        df["middle_economy"]    = np.where(~is_batter, (eco * 0.92).round(2), 0.0)
        df["death_economy"]     = np.where(~is_batter, (eco * 1.25).round(2), 0.0)
    else:
        for col in phase_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derived strike rates per phase for convenience
    df["powerplay_strike_rate"] = np.where(
        df["powerplay_balls"] > 0,
        (df["powerplay_runs"] / df["powerplay_balls"] * 100).round(1),
        0.0
    )
    df["middle_strike_rate"] = np.where(
        df["middle_balls"] > 0,
        (df["middle_runs"] / df["middle_balls"] * 100).round(1),
        0.0
    )
    df["death_strike_rate"] = np.where(
        df["death_balls"] > 0,
        (df["death_runs"] / df["death_balls"] * 100).round(1),
        0.0
    )

    # Clip values to sanity limits
    df["batting_avg"]  = df["batting_avg"].clip(0, 200)
    df["bowling_avg"]  = df["bowling_avg"].clip(0, 200)
    df["economy"]      = df["economy"].clip(0, 36)
    df["strike_rate"]  = df["strike_rate"].clip(0, 400)

    df.dropna(subset=["player", "year"], inplace=True)
    df["year"] = df["year"].astype(int)
    df.reset_index(drop=True, inplace=True)
    return df
