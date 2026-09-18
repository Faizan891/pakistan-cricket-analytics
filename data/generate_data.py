"""
generate_data.py
----------------
Generates realistic sample CSV datasets for the Pakistan Cricket Analytics Dashboard.
Run this script once to populate:
  data/match_results.csv   -- match-level data (one row per match)
  data/player_stats.csv    -- aggregated batting/bowling stats per player per year
"""

import pandas as pd
import numpy as np
import os
import random

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OPPONENTS = [
    "India", "Australia", "England", "South Africa",
    "New Zealand", "Sri Lanka", "West Indies", "Bangladesh",
    "Afghanistan", "Zimbabwe",
]

FORMATS = ["T20", "ODI", "Test"]

VENUES = {
    "Pakistan": ["Karachi (NSK)", "Lahore (Gaddafi)", "Rawalpindi Cricket Stadium", "Multan Cricket Stadium"],
    "Neutral":  ["Dubai International", "Abu Dhabi (ADSS)", "Sharjah Cricket Stadium"],
    "Away":     ["MCG", "Lord's", "Eden Gardens", "Wanderers", "Basin Reserve"],
}

BATTERS = [
    "Babar Azam", "Mohammad Rizwan", "Fakhar Zaman", "Imam-ul-Haq",
    "Abdullah Shafique", "Saud Shakeel", "Shan Masood", "Salman Agha",
]

BOWLERS = [
    "Shaheen Afridi", "Naseem Shah", "Haris Rauf", "Mohammad Wasim Jr",
    "Shadab Khan", "Imad Wasim", "Abrar Ahmed", "Zaman Khan",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _random_venue():
    category = random.choice(list(VENUES.keys()))
    return random.choice(VENUES[category])


def _runs_for_format(fmt):
    if fmt == "T20":
        return int(np.random.normal(loc=160, scale=25))
    elif fmt == "ODI":
        return int(np.random.normal(loc=270, scale=45))
    else:
        return int(np.random.normal(loc=370, scale=80))


def _wickets_taken(fmt):
    if fmt == "T20":
        return min(10, max(3, int(np.random.normal(loc=6, scale=2))))
    elif fmt == "ODI":
        return min(10, max(3, int(np.random.normal(loc=7, scale=2))))
    else:
        return min(10, max(1, int(np.random.normal(loc=8, scale=2))))


def _result(pak_runs, fmt):
    threshold = {"T20": 155, "ODI": 255, "Test": 350}[fmt]
    win_prob  = 0.55 if pak_runs >= threshold else 0.40
    roll = random.random()
    if roll < win_prob:
        return "Win"
    elif roll < win_prob + 0.08:
        return "Draw" if fmt == "Test" else "No Result"
    else:
        return "Loss"


# ---------------------------------------------------------------------------
# DATASET 1 -- MATCH RESULTS
# ---------------------------------------------------------------------------

def generate_match_results(n_matches=400):
    """
    Returns a DataFrame with one row per Pakistan cricket match (2015-2024).

    Columns
    -------
    match_id, date, year, format, opponent, venue,
    pak_runs, pak_wickets, opp_wickets, overs_played, run_rate,
    top_scorer, top_scorer_runs, top_wicket_taker, top_wickets, result
    """
    records = []

    year_weights = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.12, 0.12]
    years = np.random.choice(range(2015, 2025), size=n_matches, p=year_weights)

    for match_id, year in enumerate(years, start=1):
        fmt      = random.choice(FORMATS)
        opponent = random.choice(OPPONENTS)
        venue    = _random_venue()

        pak_runs    = max(50, _runs_for_format(fmt))
        pak_wickets = random.randint(4, 10)

        if fmt == "T20":
            overs = min(20.0, round(pak_wickets / 10 * 20 + np.random.normal(0, 1), 1))
        elif fmt == "ODI":
            overs = min(50.0, round(pak_wickets / 10 * 50 + np.random.normal(0, 2), 1))
        else:
            overs = round(np.random.normal(90, 20), 1)
        overs    = max(5.0, overs)
        run_rate = round(pak_runs / overs, 2)

        opp_wickets      = _wickets_taken(fmt)
        top_scorer       = random.choice(BATTERS)
        top_scorer_runs  = max(5, min(pak_runs - 5, int(np.random.normal(pak_runs * 0.38, 20))))
        top_wicket_taker = random.choice(BOWLERS)
        top_wickets      = min(opp_wickets, random.choices([1,2,3,4,5,6], weights=[10,25,30,20,10,5])[0])

        month  = random.randint(1, 12)
        day    = random.randint(1, 28)
        date   = f"{year}-{month:02d}-{day:02d}"
        result = _result(pak_runs, fmt)

        records.append({
            "match_id"         : match_id,
            "date"             : date,
            "year"             : int(year),
            "format"           : fmt,
            "opponent"         : opponent,
            "venue"            : venue,
            "pak_runs"         : pak_runs,
            "pak_wickets"      : pak_wickets,
            "opp_wickets"      : opp_wickets,
            "overs_played"     : overs,
            "run_rate"         : run_rate,
            "top_scorer"       : top_scorer,
            "top_scorer_runs"  : top_scorer_runs,
            "top_wicket_taker" : top_wicket_taker,
            "top_wickets"      : top_wickets,
            "result"           : result,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# DATASET 2 -- PLAYER STATS
# ---------------------------------------------------------------------------

def generate_player_stats():
    """
    Returns a DataFrame with per-player, per-year, per-format aggregated stats.

    Batting  : matches, innings, runs, batting_avg, strike_rate, hundreds, fifties
    Bowling  : wickets, bowling_avg, economy, five_wickets
    """
    records = []
    for player in BATTERS + BOWLERS:
        is_batter = player in BATTERS
        for year in range(2015, 2025):
            for fmt in FORMATS:
                matches = random.randint(5, 20)
                innings = matches if fmt != "Test" else matches * 2

                runs     = max(20, int(np.random.normal(600 if is_batter else 180, 150)))
                avg      = round(runs / max(innings, 1), 2)
                sr_mean  = {"T20": 135, "ODI": 85, "Test": 55}[fmt]
                sr       = round(np.random.normal(sr_mean, 10), 1)
                hundreds = max(0, int(runs // 350) + random.choice([0, 0, 0, 1]))
                fifties  = max(0, int(runs // 120) + random.choice([0, 1]))

                wickets    = max(0, int(np.random.normal(25 if not is_batter else 5, 8)))
                bowl_avg   = round(np.random.normal(28 if not is_batter else 45, 5), 2)
                economy    = round(np.random.normal({"T20": 7.8, "ODI": 5.2, "Test": 3.1}[fmt], 0.5), 2)
                five_wkts  = max(0, wickets // 25 + random.choice([0, 0, 1]))

                records.append({
                    "player"      : player,
                    "role"        : "Batter" if is_batter else "Bowler",
                    "year"        : year,
                    "format"      : fmt,
                    "matches"     : matches,
                    "innings"     : innings,
                    "runs"        : runs,
                    "batting_avg" : avg,
                    "strike_rate" : sr,
                    "hundreds"    : hundreds,
                    "fifties"     : fifties,
                    "wickets"     : wickets,
                    "bowling_avg" : bowl_avg,
                    "economy"     : economy,
                    "five_wickets": five_wkts,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    print("Generating match_results.csv ...")
    m = generate_match_results(n_matches=400)
    p = os.path.join(base, "match_results.csv")
    m.to_csv(p, index=False)
    print(f"  Saved: {p}  ({len(m)} rows)")

    print("Generating player_stats.csv ...")
    s = generate_player_stats()
    q = os.path.join(base, "player_stats.csv")
    s.to_csv(q, index=False)
    print(f"  Saved: {q}  ({len(s)} rows)")

    print("Data generation complete.")
