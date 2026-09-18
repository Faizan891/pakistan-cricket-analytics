"""
generate_data_stdlib.py
-----------------------
Generates Pakistan Cricket CSV datasets using ONLY Python stdlib
(csv, random, math, os, datetime) -- no pandas/numpy required.
Produces identical schema to generate_data.py.
"""

import csv
import random
import math
import os
from datetime import date

# ── Config ──────────────────────────────────────────────────────────────────
random.seed(42)
N_MATCHES = 400

OPPONENTS = [
    "India","Australia","England","South Africa",
    "New Zealand","Sri Lanka","West Indies","Bangladesh",
    "Afghanistan","Zimbabwe",
]
FORMATS   = ["T20","ODI","Test"]
VENUES    = [
    "Karachi (NSK)","Lahore (Gaddafi)","Rawalpindi Cricket Stadium",
    "Multan Cricket Stadium","Dubai International","Abu Dhabi (ADSS)",
    "Sharjah Cricket Stadium","MCG","Lord's","Eden Gardens",
    "Wanderers","Basin Reserve",
]
BATTERS   = [
    "Babar Azam","Mohammad Rizwan","Fakhar Zaman","Imam-ul-Haq",
    "Abdullah Shafique","Saud Shakeel","Shan Masood","Salman Agha",
]
BOWLERS   = [
    "Shaheen Afridi","Naseem Shah","Haris Rauf","Mohammad Wasim Jr",
    "Shadab Khan","Imad Wasim","Abrar Ahmed","Zaman Khan",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def gauss_int(mu, sigma):
    return int(random.gauss(mu, sigma))

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def runs_for_format(fmt):
    mu = {"T20": 160, "ODI": 270, "Test": 370}[fmt]
    return clamp(gauss_int(mu, 25 if fmt=="T20" else 45 if fmt=="ODI" else 80), 50, 500)

def wickets_taken(fmt):
    mu = {"T20": 6, "ODI": 7, "Test": 8}[fmt]
    return clamp(gauss_int(mu, 2), 1 if fmt=="Test" else 3, 10)

def match_result(pak_runs, fmt):
    thresh  = {"T20": 155, "ODI": 255, "Test": 350}[fmt]
    win_p   = 0.55 if pak_runs >= thresh else 0.40
    roll    = random.random()
    if roll < win_p:              return "Win"
    elif roll < win_p + 0.08:    return "Draw" if fmt=="Test" else "No Result"
    else:                         return "Loss"

def weighted_year():
    years   = list(range(2015, 2025))
    weights = [6,7,8,9,10,11,12,13,12,12]
    cumul   = []
    total   = sum(weights)
    run = 0
    for w in weights:
        run += w
        cumul.append(run / total)
    r = random.random()
    for i, c in enumerate(cumul):
        if r <= c:
            return years[i]
    return years[-1]

# ── Dataset 1: match_results.csv ─────────────────────────────────────────────

def generate_matches(path):
    fieldnames = [
        "match_id","date","year","format","opponent","venue",
        "pak_runs","pak_wickets","opp_wickets","overs_played","run_rate",
        "top_scorer","top_scorer_runs","top_wicket_taker","top_wickets","result",
    ]
    rows = []
    for mid in range(1, N_MATCHES + 1):
        year     = weighted_year()
        fmt      = random.choice(FORMATS)
        opponent = random.choice(OPPONENTS)
        venue    = random.choice(VENUES)

        pak_runs    = runs_for_format(fmt)
        pak_wkts    = random.randint(4, 10)

        if fmt == "T20":
            overs = clamp(round(pak_wkts/10*20 + random.gauss(0,1), 1), 5.0, 20.0)
        elif fmt == "ODI":
            overs = clamp(round(pak_wkts/10*50 + random.gauss(0,2), 1), 5.0, 50.0)
        else:
            overs = clamp(round(random.gauss(90, 20), 1), 10.0, 200.0)

        run_rate = round(pak_runs / overs, 2)
        opp_wkts = wickets_taken(fmt)

        top_scorer      = random.choice(BATTERS)
        ts_runs         = clamp(gauss_int(int(pak_runs*0.38), 20), 5, pak_runs-5)
        top_wkt_taker   = random.choice(BOWLERS)
        top_wkts        = clamp(
            random.choices([1,2,3,4,5,6], weights=[10,25,30,20,10,5])[0],
            1, opp_wkts
        )

        month = random.randint(1, 12)
        day   = random.randint(1, 28)
        dt    = f"{year}-{month:02d}-{day:02d}"
        res   = match_result(pak_runs, fmt)

        rows.append({
            "match_id"        : mid,
            "date"            : dt,
            "year"            : year,
            "format"          : fmt,
            "opponent"        : opponent,
            "venue"           : venue,
            "pak_runs"        : pak_runs,
            "pak_wickets"     : pak_wkts,
            "opp_wickets"     : opp_wkts,
            "overs_played"    : overs,
            "run_rate"        : run_rate,
            "top_scorer"      : top_scorer,
            "top_scorer_runs" : ts_runs,
            "top_wicket_taker": top_wkt_taker,
            "top_wickets"     : top_wkts,
            "result"          : res,
        })

    # Sort by date
    rows.sort(key=lambda r: r["date"])
    for i, r in enumerate(rows, 1):
        r["match_id"] = i

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved: {path}  ({len(rows)} rows)")

# ── Dataset 2: player_stats.csv ──────────────────────────────────────────────

def generate_players(path):
    fieldnames = [
        "player","role","year","format","matches","innings","runs",
        "batting_avg","strike_rate","hundreds","fifties",
        "wickets","bowling_avg","economy","five_wickets",
    ]
    rows = []
    for player in BATTERS + BOWLERS:
        is_batter = player in BATTERS
        for year in range(2015, 2025):
            for fmt in FORMATS:
                matches  = random.randint(5, 20)
                innings  = matches if fmt != "Test" else matches * 2

                runs     = clamp(gauss_int(600 if is_batter else 180, 150), 20, 2000)
                avg      = round(runs / max(innings, 1), 2)
                sr_mu    = {"T20": 135, "ODI": 85, "Test": 55}[fmt]
                sr       = round(clamp(random.gauss(sr_mu, 10), 20, 400), 1)
                hundreds = clamp(runs//350 + random.choice([0,0,0,1]), 0, 20)
                fifties  = clamp(runs//120 + random.choice([0,1]), 0, 30)

                wickets    = clamp(gauss_int(25 if not is_batter else 5, 8), 0, 80)
                bowl_avg   = round(clamp(random.gauss(28 if not is_batter else 45, 5), 10, 100), 2)
                eco_mu     = {"T20": 7.8, "ODI": 5.2, "Test": 3.1}[fmt]
                economy    = round(clamp(random.gauss(eco_mu, 0.5), 1.0, 20.0), 2)
                five_wkts  = clamp(wickets//25 + random.choice([0,0,1]), 0, 10)

                rows.append({
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

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved: {path}  ({len(rows)} rows)")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    print("Generating match_results.csv ...")
    generate_matches(os.path.join(base, "match_results.csv"))

    print("Generating player_stats.csv ...")
    generate_players(os.path.join(base, "player_stats.csv"))

    print("Done.")
