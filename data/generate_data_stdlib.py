"""
generate_data_stdlib.py
-----------------------
Generates expanded Pakistan Cricket CSV datasets using ONLY Python stdlib
(csv, random, math, os, datetime).

Schema Expansions:
1. match_results.csv:
   - toss_winner: "Pakistan" or Opponent name
   - toss_decision: "Bat" or "Field"
   - pak_innings: "1st" or "2nd"
   - first_innings_score: integer runs
   - second_innings_score: integer runs
   - winner_innings: "1st" or "2nd" (or "None")
2. player_stats.csv:
   - Powerplay (Overs 1-6): powerplay_runs, powerplay_balls, powerplay_wickets, powerplay_economy
   - Middle (Overs 7-15): middle_runs, middle_balls, middle_wickets, middle_economy
   - Death (Overs 16-20): death_runs, death_balls, death_wickets, death_economy
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
    "India", "Australia", "England", "South Africa",
    "New Zealand", "Sri Lanka", "West Indies", "Bangladesh",
    "Afghanistan", "Zimbabwe",
]
FORMATS   = ["T20", "ODI", "Test"]
VENUES    = [
    "Karachi (NSK)", "Lahore (Gaddafi)", "Rawalpindi Cricket Stadium",
    "Multan Cricket Stadium", "Dubai International", "Abu Dhabi (ADSS)",
    "Sharjah Cricket Stadium", "MCG", "Lord's", "Eden Gardens",
    "Wanderers", "Basin Reserve",
]

# Player roles and phase specializations
BATTERS_META = {
    "Babar Azam":        {"pos": "Top Order", "pp_share": 0.38, "mid_share": 0.45, "death_share": 0.17, "pp_sr": 132, "mid_sr": 128, "death_sr": 165},
    "Mohammad Rizwan":   {"pos": "Opener",    "pp_share": 0.42, "mid_share": 0.42, "death_share": 0.16, "pp_sr": 128, "mid_sr": 124, "death_sr": 168},
    "Fakhar Zaman":      {"pos": "Opener",    "pp_share": 0.48, "mid_share": 0.38, "death_share": 0.14, "pp_sr": 142, "mid_sr": 134, "death_sr": 160},
    "Imam-ul-Haq":       {"pos": "Opener",    "pp_share": 0.40, "mid_share": 0.46, "death_share": 0.14, "pp_sr": 115, "mid_sr": 118, "death_sr": 145},
    "Abdullah Shafique": {"pos": "Top Order", "pp_share": 0.36, "mid_share": 0.46, "death_share": 0.18, "pp_sr": 126, "mid_sr": 125, "death_sr": 155},
    "Saud Shakeel":      {"pos": "Middle",    "pp_share": 0.18, "mid_share": 0.58, "death_share": 0.24, "pp_sr": 118, "mid_sr": 122, "death_sr": 158},
    "Shan Masood":       {"pos": "Top Order", "pp_share": 0.35, "mid_share": 0.48, "death_share": 0.17, "pp_sr": 124, "mid_sr": 122, "death_sr": 150},
    "Salman Agha":       {"pos": "Finisher",  "pp_share": 0.12, "mid_share": 0.48, "death_share": 0.40, "pp_sr": 120, "mid_sr": 130, "death_sr": 178},
}

BOWLERS_META = {
    "Shaheen Afridi":    {"type": "Pace", "pp_wkt_share": 0.46, "mid_wkt_share": 0.22, "death_wkt_share": 0.32, "pp_eco": 6.8, "mid_eco": 7.4, "death_eco": 9.2},
    "Naseem Shah":       {"type": "Pace", "pp_wkt_share": 0.42, "mid_wkt_share": 0.28, "death_wkt_share": 0.30, "pp_eco": 7.1, "mid_eco": 7.2, "death_eco": 9.0},
    "Haris Rauf":        {"type": "Pace", "pp_wkt_share": 0.22, "mid_wkt_share": 0.34, "death_wkt_share": 0.44, "pp_eco": 8.4, "mid_eco": 7.6, "death_eco": 8.8},
    "Mohammad Wasim Jr": {"type": "Pace", "pp_wkt_share": 0.25, "mid_wkt_share": 0.35, "death_wkt_share": 0.40, "pp_eco": 7.9, "mid_eco": 7.8, "death_eco": 9.4},
    "Shadab Khan":       {"type": "Spin", "pp_wkt_share": 0.12, "mid_wkt_share": 0.68, "death_wkt_share": 0.20, "pp_eco": 7.4, "mid_eco": 6.9, "death_eco": 9.6},
    "Imad Wasim":        {"type": "Spin", "pp_wkt_share": 0.38, "mid_wkt_share": 0.50, "death_wkt_share": 0.12, "pp_eco": 6.4, "mid_eco": 6.8, "death_eco": 9.8},
    "Abrar Ahmed":       {"type": "Spin", "pp_wkt_share": 0.10, "mid_wkt_share": 0.76, "death_wkt_share": 0.14, "pp_eco": 7.6, "mid_eco": 6.6, "death_eco": 10.1},
    "Zaman Khan":        {"type": "Pace", "pp_wkt_share": 0.24, "mid_wkt_share": 0.24, "death_wkt_share": 0.52, "pp_eco": 7.8, "mid_eco": 8.0, "death_eco": 8.5},
}

BATTERS = list(BATTERS_META.keys())
BOWLERS = list(BOWLERS_META.keys())

# ── Helpers ──────────────────────────────────────────────────────────────────

def gauss_int(mu, sigma):
    return int(random.gauss(mu, sigma))

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def runs_for_format(fmt):
    mu = {"T20": 160, "ODI": 270, "Test": 370}[fmt]
    return clamp(gauss_int(mu, 25 if fmt=="T20" else 45 if fmt=="ODI" else 80), 50, 550)

def wickets_taken(fmt):
    mu = {"T20": 6, "ODI": 7, "Test": 8}[fmt]
    return clamp(gauss_int(mu, 2), 1 if fmt=="Test" else 3, 10)

def weighted_year():
    years   = list(range(2015, 2025))
    weights = [6, 7, 8, 9, 10, 11, 12, 13, 12, 12]
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
        "match_id", "date", "year", "format", "opponent", "venue",
        "toss_winner", "toss_decision", "pak_innings",
        "pak_runs", "pak_wickets", "opp_wickets", "overs_played", "run_rate",
        "first_innings_score", "second_innings_score", "winner_innings",
        "top_scorer", "top_scorer_runs", "top_wicket_taker", "top_wickets", "result",
    ]
    rows = []
    for mid in range(1, N_MATCHES + 1):
        year     = weighted_year()
        fmt      = random.choice(FORMATS)
        opponent = random.choice(OPPONENTS)
        venue    = random.choice(VENUES)

        # Toss mechanics
        toss_winner   = "Pakistan" if random.random() < 0.50 else opponent
        toss_decision = random.choices(["Bat", "Field"], weights=[45, 55])[0]

        # Determine which innings Pakistan batted
        if toss_winner == "Pakistan":
            pak_innings = "1st" if toss_decision == "Bat" else "2nd"
        else:
            pak_innings = "2nd" if toss_decision == "Bat" else "1st"

        pak_runs    = runs_for_format(fmt)
        pak_wkts    = random.randint(3, 10)

        if fmt == "T20":
            overs = clamp(round(pak_wkts / 10 * 20 + random.gauss(0, 1), 1), 5.0, 20.0)
        elif fmt == "ODI":
            overs = clamp(round(pak_wkts / 10 * 50 + random.gauss(0, 2), 1), 5.0, 50.0)
        else:
            overs = clamp(round(random.gauss(90, 20), 1), 10.0, 200.0)

        run_rate = round(pak_runs / overs, 2)
        opp_wkts = wickets_taken(fmt)

        # Derive match result with realistic conditions
        thresh   = {"T20": 155, "ODI": 255, "Test": 350}[fmt]
        win_prob = 0.56 if pak_runs >= thresh else 0.41
        # Chasing slightly higher in modern T20s
        if fmt == "T20" and pak_innings == "2nd":
            win_prob += 0.04
        if toss_winner == "Pakistan":
            win_prob += 0.03

        roll = random.random()
        if roll < win_prob:
            res = "Win"
        elif roll < win_prob + 0.08:
            res = "Draw" if fmt == "Test" else "No Result"
        else:
            res = "Loss"

        # Construct realistic first and second innings scores
        if pak_innings == "1st":
            first_innings_score = pak_runs
            if res == "Win":
                second_innings_score = clamp(int(pak_runs - random.gauss(24, 12)), 40, pak_runs - 1)
                winner_innings = "1st"
            elif res == "Loss":
                second_innings_score = clamp(int(pak_runs + random.gauss(6, 4)), pak_runs + 1, pak_runs + 40)
                winner_innings = "2nd"
            else:
                second_innings_score = pak_runs if res == "Draw" else 0
                winner_innings = "None"
        else: # pak_innings == "2nd"
            second_innings_score = pak_runs
            if res == "Win":
                first_innings_score = clamp(int(pak_runs - random.gauss(6, 4)), 40, pak_runs - 1)
                winner_innings = "2nd"
            elif res == "Loss":
                first_innings_score = clamp(int(pak_runs + random.gauss(24, 12)), pak_runs + 1, pak_runs + 70)
                winner_innings = "1st"
            else:
                first_innings_score = pak_runs if res == "Draw" else 0
                winner_innings = "None"

        top_scorer    = random.choice(BATTERS)
        ts_runs       = clamp(gauss_int(int(pak_runs * 0.38), 18), 8, pak_runs - 5)
        top_wkt_taker = random.choice(BOWLERS)
        top_wkts      = clamp(
            random.choices([1, 2, 3, 4, 5, 6], weights=[10, 25, 30, 20, 10, 5])[0],
            1, opp_wkts
        )

        month = random.randint(1, 12)
        day   = random.randint(1, 28)
        dt    = f"{year}-{month:02d}-{day:02d}"

        rows.append({
            "match_id"            : mid,
            "date"                : dt,
            "year"                : year,
            "format"              : fmt,
            "opponent"            : opponent,
            "venue"               : venue,
            "toss_winner"         : toss_winner,
            "toss_decision"       : toss_decision,
            "pak_innings"         : pak_innings,
            "pak_runs"            : pak_runs,
            "pak_wickets"         : pak_wkts,
            "opp_wickets"         : opp_wkts,
            "overs_played"        : overs,
            "run_rate"            : run_rate,
            "first_innings_score" : first_innings_score,
            "second_innings_score": second_innings_score,
            "winner_innings"      : winner_innings,
            "top_scorer"          : top_scorer,
            "top_scorer_runs"     : ts_runs,
            "top_wicket_taker"    : top_wkt_taker,
            "top_wickets"         : top_wkts,
            "result"              : res,
        })

    rows.sort(key=lambda r: r["date"])
    for i, r in enumerate(rows, 1):
        r["match_id"] = i

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved expanded: {path} ({len(rows)} rows)")

# ── Dataset 2: player_stats.csv ──────────────────────────────────────────────

def generate_players(path):
    fieldnames = [
        "player", "role", "year", "format", "matches", "innings", "runs",
        "batting_avg", "strike_rate", "hundreds", "fifties",
        "wickets", "bowling_avg", "economy", "five_wickets",
        # Phase metrics
        "powerplay_runs", "powerplay_balls", "powerplay_wickets", "powerplay_economy",
        "middle_runs", "middle_balls", "middle_wickets", "middle_economy",
        "death_runs", "death_balls", "death_wickets", "death_economy",
    ]
    rows = []
    for player in BATTERS + BOWLERS:
        is_batter = player in BATTERS
        b_meta = BATTERS_META.get(player, {})
        w_meta = BOWLERS_META.get(player, {})

        for year in range(2015, 2025):
            for fmt in FORMATS:
                matches  = random.randint(5, 20)
                innings  = matches if fmt != "Test" else matches * 2

                runs     = clamp(gauss_int(600 if is_batter else 160, 140), 15, 2000)
                avg      = round(runs / max(innings, 1), 2)
                sr_mu    = {"T20": 135, "ODI": 86, "Test": 55}[fmt]
                sr       = round(clamp(random.gauss(sr_mu, 10), 20, 400), 1)
                hundreds = clamp(runs // 350 + random.choice([0, 0, 0, 1]), 0, 20)
                fifties  = clamp(runs // 120 + random.choice([0, 1]), 0, 30)

                wickets   = clamp(gauss_int(26 if not is_batter else 4, 8), 0, 80)
                bowl_avg  = round(clamp(random.gauss(27 if not is_batter else 45, 5), 10, 100), 2)
                eco_mu    = {"T20": 7.8, "ODI": 5.2, "Test": 3.1}[fmt]
                economy   = round(clamp(random.gauss(eco_mu, 0.5), 1.0, 20.0), 2)
                five_wkts = clamp(wickets // 25 + random.choice([0, 0, 1]), 0, 10)

                # ── Phase distributions ─────────────────────────────────────
                if is_batter:
                    pp_s    = b_meta.get("pp_share", 0.35)
                    mid_s   = b_meta.get("mid_share", 0.45)
                    death_s = b_meta.get("death_share", 0.20)

                    pp_runs    = clamp(int(runs * pp_s + random.gauss(0, 15)), 0, runs)
                    mid_runs   = clamp(int(runs * mid_s + random.gauss(0, 20)), 0, runs - pp_runs)
                    death_runs = max(0, runs - pp_runs - mid_runs)

                    # Balls faced per phase derived from phase strike rates
                    pp_sr_base  = b_meta.get("pp_sr", 125) * (sr / 130)
                    mid_sr_base = b_meta.get("mid_sr", 125) * (sr / 130)
                    d_sr_base   = b_meta.get("death_sr", 160) * (sr / 130)

                    pp_balls    = max(1, int(pp_runs / (pp_sr_base / 100))) if pp_runs > 0 else 0
                    mid_balls   = max(1, int(mid_runs / (mid_sr_base / 100))) if mid_runs > 0 else 0
                    death_balls = max(1, int(death_runs / (d_sr_base / 100))) if death_runs > 0 else 0

                    pp_wkts    = 0
                    mid_wkts   = 0
                    death_wkts = 0
                    pp_eco     = 0.0
                    mid_eco    = 0.0
                    death_eco  = 0.0
                else:
                    # Bowler phase metrics
                    pp_s    = w_meta.get("pp_wkt_share", 0.33)
                    mid_s   = w_meta.get("mid_wkt_share", 0.34)
                    death_s = w_meta.get("death_wkt_share", 0.33)

                    pp_wkts    = clamp(int(wickets * pp_s + random.choice([0, 0, 1, -1])), 0, wickets)
                    mid_wkts   = clamp(int(wickets * mid_s + random.choice([0, 0, 1, -1])), 0, wickets - pp_wkts)
                    death_wkts = max(0, wickets - pp_wkts - mid_wkts)

                    # Bowling economy by phase
                    fmt_factor = {"T20": 1.0, "ODI": 0.72, "Test": 0.45}[fmt]
                    pp_eco     = round(clamp(w_meta.get("pp_eco", 7.2) * fmt_factor + random.gauss(0, 0.3), 2.5, 14.0), 2)
                    mid_eco    = round(clamp(w_meta.get("mid_eco", 7.0) * fmt_factor + random.gauss(0, 0.3), 2.5, 14.0), 2)
                    death_eco  = round(clamp(w_meta.get("death_eco", 9.0) * fmt_factor + random.gauss(0, 0.4), 3.0, 18.0), 2)

                    pp_runs     = clamp(int(runs * 0.15), 0, runs)
                    mid_runs    = clamp(int(runs * 0.45), 0, runs)
                    death_runs  = max(0, runs - pp_runs - mid_runs)
                    pp_balls    = max(0, int(pp_runs * 1.1))
                    mid_balls   = max(0, int(mid_runs * 1.1))
                    death_balls = max(0, int(death_runs * 0.9))

                rows.append({
                    "player"            : player,
                    "role"              : "Batter" if is_batter else "Bowler",
                    "year"              : year,
                    "format"            : fmt,
                    "matches"           : matches,
                    "innings"           : innings,
                    "runs"              : runs,
                    "batting_avg"       : avg,
                    "strike_rate"       : sr,
                    "hundreds"          : hundreds,
                    "fifties"           : fifties,
                    "wickets"           : wickets,
                    "bowling_avg"       : bowl_avg,
                    "economy"           : economy,
                    "five_wickets"      : five_wkts,
                    "powerplay_runs"    : pp_runs,
                    "powerplay_balls"   : pp_balls,
                    "powerplay_wickets" : pp_wkts,
                    "powerplay_economy" : pp_eco,
                    "middle_runs"       : mid_runs,
                    "middle_balls"      : mid_balls,
                    "middle_wickets"    : mid_wkts,
                    "middle_economy"    : mid_eco,
                    "death_runs"        : death_runs,
                    "death_balls"       : death_balls,
                    "death_wickets"     : death_wkts,
                    "death_economy"     : death_eco,
                })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved expanded: {path} ({len(rows)} rows)")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    print("Generating expanded match_results.csv ...")
    generate_matches(os.path.join(base, "match_results.csv"))

    print("Generating expanded player_stats.csv ...")
    generate_players(os.path.join(base, "player_stats.csv"))

    print("Data generation complete!")
