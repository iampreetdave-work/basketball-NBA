"""
Hourly Odds Checker
Polls The Odds API for new NBA odds.
Compares against existing predictions in NBA_PREDICTIONS_ML.csv.
If new games with odds are found that don't have predictions yet:
  - Writes 'new_odds_found.flag' file (signals pipeline to run)
  - Prints NEW_ODDS_FOUND
Otherwise:
  - Prints NO_NEW_ODDS
"""

import requests
import pandas as pd
import json
import os
import sys
from datetime import datetime, timezone

# ============================================================================
# THE ODDS API CONFIGURATION (same as Odds_Pre_Match.py)
# ============================================================================

API_KEYS = [
    "2654835dd9c6779ec43efbe938a8ebe3",
    "83dcdaff13977e39bc65141046c993f3",
    "02a80c14ece71bed354b63915e3fb8b3",
    "30d78032b75c0922de70de22f0337b91",
    "8972d0f8f1c909b2791607ed1a29d6a5",
    "7483e0df3726e14cdb152f580291f47d"
]
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

TEAM_ALIASES = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GS",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NO",
    "New York Knicks": "NY",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SA",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTAH",
    "Washington Wizards": "WSH",
}

PREDICTIONS_FILE = "NBA_PREDICTIONS_ML.csv"
FLAG_FILE = "new_odds_found.flag"


def fetch_current_odds():
    """Fetch current NBA odds from The Odds API"""
    for key in API_KEYS:
        params = {
            "apiKey": key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
            "bookmakers": "draftkings,fanduel"
        }

        try:
            response = requests.get(
                f"{BASE_URL}/sports/{SPORT}/odds",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                remaining = response.headers.get("x-requests-remaining", "?")
                print(f"  API call successful (quota remaining: {remaining})")
                return response.json()
            elif response.status_code == 429:
                print(f"  Rate limit on key ...{key[-4:]}, trying next")
                continue
            else:
                print(f"  Error {response.status_code} on key ...{key[-4:]}")
                continue

        except Exception as e:
            print(f"  Request failed: {e}")
            continue

    print("  All API keys exhausted")
    return None


def build_game_identifiers(games):
    """Build game_identifier list from Odds API response"""
    identifiers = set()

    for game in games:
        commence_time = game.get('commence_time', '')
        date = commence_time.split('T')[0] if commence_time else ''
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        home_alias = TEAM_ALIASES.get(home_team, home_team[:3].upper())
        away_alias = TEAM_ALIASES.get(away_team, away_team[:3].upper())

        # Check that game actually has bookmaker data (not just listed)
        bookmakers = game.get('bookmakers', [])
        if not bookmakers:
            continue

        game_id = f"{date}_{away_alias}@{home_alias}"
        identifiers.add(game_id)

    return identifiers


def get_existing_predictions():
    """Load existing predictions and return set of game_identifiers"""
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"  {PREDICTIONS_FILE} not found — treating all odds as new")
        return set()

    try:
        df = pd.read_csv(PREDICTIONS_FILE)
        existing = set(df['game_identifier'].dropna().unique())
        print(f"  Found {len(existing)} existing predictions")
        return existing
    except Exception as e:
        print(f"  Error reading {PREDICTIONS_FILE}: {e}")
        return set()


def main():
    utc_now = datetime.now(timezone.utc)

    print("=" * 70)
    print("HOURLY ODDS CHECKER")
    print("=" * 70)
    print(f"UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Remove old flag if exists
    if os.path.exists(FLAG_FILE):
        os.remove(FLAG_FILE)

    # Step 1: Fetch current odds
    print("\n[1/3] Fetching current odds from The Odds API...")
    games = fetch_current_odds()

    if not games:
        print("\n  No games returned from API")
        print("\nNO_NEW_ODDS")
        return

    print(f"  Games with odds: {len(games)}")

    # Step 2: Build game identifiers from odds
    print("\n[2/3] Building game identifiers...")
    odds_game_ids = build_game_identifiers(games)
    print(f"  Games with valid odds: {len(odds_game_ids)}")

    for gid in sorted(odds_game_ids):
        print(f"    {gid}")

    # Step 3: Compare with existing predictions
    print("\n[3/3] Comparing with existing predictions...")
    existing_predictions = get_existing_predictions()

    new_games = odds_game_ids - existing_predictions
    already_predicted = odds_game_ids & existing_predictions

    print(f"  Already predicted: {len(already_predicted)}")
    print(f"  NEW (need predictions): {len(new_games)}")

    # Build summary for Slack
    summary = {
        "timestamp": utc_now.strftime("%Y-%m-%d %H:%M UTC"),
        "total_with_odds": len(odds_game_ids),
        "already_predicted": len(already_predicted),
        "new_games_count": len(new_games),
        "new_games": sorted(new_games),
        "all_games": sorted(odds_game_ids),
    }

    # Write summary JSON for workflow to read
    with open("checker_summary.json", "w") as f:
        json.dump(summary, f)

    if new_games:
        print(f"\n  New games found:")
        for gid in sorted(new_games):
            print(f"    + {gid}")

        # Write flag file
        with open(FLAG_FILE, 'w') as f:
            f.write(f"timestamp={utc_now.isoformat()}\n")
            f.write(f"new_games={len(new_games)}\n")
            for gid in sorted(new_games):
                f.write(f"game={gid}\n")

        print(f"\n  Flag file written: {FLAG_FILE}")
        print("\nNEW_ODDS_FOUND")
    else:
        print("\n  All games with odds already have predictions")
        print("\nNO_NEW_ODDS")

    print("=" * 70)


if __name__ == "__main__":
    main()
