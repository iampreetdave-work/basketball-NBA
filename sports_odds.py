"""
Fetch Odds for UPCOMING NBA Matches from API-Basketball
Uses v1.basketball.api-sports.io
Provides odds 1-7 days before games (vs ~24hrs from The Odds API)
Output format identical to Odds_Pre_Match.py (upcoming_nba_draftkings_odds.csv)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import time

# ============================================================================
# API-BASKETBALL CONFIGURATION
# ============================================================================

API_KEY = "772d9941ad4acca04ce2bfd0b695f9bb"
BASE_URL = "https://v1.basketball.api-sports.io"
NBA_LEAGUE_ID = 12
NBA_SEASON = "2025-2026"
DAYS_AHEAD = 7

# Bookmaker preference order (sharp books first)
PREFERRED_BOOKMAKERS = ["Pinnacle", "Bet365", "1xBet", "Marathon Bet", "Betano", "Unibet", "Betway"]

# Team name to alias mapping
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
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
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


def get_team_alias(team_name):
    """Get team alias from team name"""
    return TEAM_ALIASES.get(team_name, team_name[:3].upper())


def api_request(endpoint, params=None, retries=3):
    """Make API-Basketball request with retry logic"""
    headers = {"x-apisports-key": API_KEY}
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params or {}, timeout=30)

            if response.status_code == 200:
                data = response.json()
                errors = data.get("errors", [])
                if errors:
                    print(f"  API errors: {errors}")
                    return None
                return data
            elif response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                print(f"  Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  Error {response.status_code}: {response.text[:200]}")
                if attempt < retries - 1:
                    time.sleep(5)
                    continue
                return None

        except Exception as e:
            print(f"  Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return None

    return None


def select_bookmaker(bookmakers):
    """Select best bookmaker from preference list"""
    bm_map = {bm.get("name"): bm for bm in bookmakers}

    for preferred in PREFERRED_BOOKMAKERS:
        if preferred in bm_map:
            return bm_map[preferred], preferred

    # Fallback to first available
    if bookmakers:
        first = bookmakers[0]
        return first, first.get("name", "Unknown")

    return None, None


def extract_odds_from_bookmaker(bookmaker):
    """
    Extract moneyline, spread, and totals from a bookmaker's bets.
    Returns dict matching the Odds_Pre_Match.py output columns.
    """
    odds = {
        "home_winning_odds_decimal": None,
        "away_winning_odds_decimal": None,
        "home_spread": None,
        "away_spread": None,
        "home_spread_odds_decimal": None,
        "away_spread_odds_decimal": None,
        "total_line_o": None,
        "total_line_over_odds_decimal": None,
        "total_line_under_odds_decimal": None,
    }

    bets = bookmaker.get("bets", [])

    for bet in bets:
        bet_name = bet.get("name", "")
        values = bet.get("values", [])

        # --- MONEYLINE (Home/Away) ---
        if bet_name == "Home/Away":
            for v in values:
                val = v.get("value", "")
                odd = v.get("odd")
                if odd:
                    odd = float(odd)
                    if val == "Home":
                        odds["home_winning_odds_decimal"] = odd
                    elif val == "Away":
                        odds["away_winning_odds_decimal"] = odd

        # --- SPREAD (Asian Handicap) ---
        elif bet_name == "Asian Handicap" and odds["home_spread"] is None:
            home_lines = []
            away_lines = []
            for v in values:
                val = str(v.get("value", ""))
                odd = v.get("odd")
                if odd:
                    odd = float(odd)
                    if val.startswith("Home"):
                        try:
                            point = float(val.replace("Home", "").strip())
                            home_lines.append((point, odd))
                        except ValueError:
                            pass
                    elif val.startswith("Away"):
                        try:
                            point = float(val.replace("Away", "").strip())
                            away_lines.append((point, odd))
                        except ValueError:
                            pass

            # Select the line where odds are closest to even (~1.91)
            if home_lines:
                best_home = min(home_lines, key=lambda x: abs(x[1] - 1.91))
                odds["home_spread"] = best_home[0]
                odds["home_spread_odds_decimal"] = best_home[1]

                # Find matching away line (opposite sign)
                target_away = -best_home[0]
                matching_away = [a for a in away_lines if abs(a[0] - target_away) < 0.5]
                if matching_away:
                    best_away = matching_away[0]
                    odds["away_spread"] = best_away[0]
                    odds["away_spread_odds_decimal"] = best_away[1]
                elif away_lines:
                    best_away = min(away_lines, key=lambda x: abs(x[1] - 1.91))
                    odds["away_spread"] = best_away[0]
                    odds["away_spread_odds_decimal"] = best_away[1]

        # --- TOTALS (Over/Under) ---
        elif bet_name == "Over/Under" and odds["total_line_o"] is None:
            over_lines = []
            under_lines = []
            for v in values:
                val = str(v.get("value", ""))
                odd = v.get("odd")
                if odd:
                    odd = float(odd)
                    if val.startswith("Over"):
                        try:
                            point = float(val.replace("Over", "").strip())
                            over_lines.append((point, odd))
                        except ValueError:
                            pass
                    elif val.startswith("Under"):
                        try:
                            point = float(val.replace("Under", "").strip())
                            under_lines.append((point, odd))
                        except ValueError:
                            pass

            # Select the line where over odds are closest to even (~1.91)
            if over_lines:
                best_over = min(over_lines, key=lambda x: abs(x[1] - 1.91))
                odds["total_line_o"] = best_over[0]
                odds["total_line_over_odds_decimal"] = best_over[1]

                # Find matching under line (same point)
                matching_under = [u for u in under_lines if u[0] == best_over[0]]
                if matching_under:
                    odds["total_line_under_odds_decimal"] = matching_under[0][1]
                elif under_lines:
                    best_under = min(under_lines, key=lambda x: abs(x[0] - best_over[0]))
                    odds["total_line_under_odds_decimal"] = best_under[1]

    return odds


def main():
    utc_now = datetime.now(timezone.utc)

    print("\n" + "=" * 80)
    print("API-BASKETBALL ODDS FETCHER (sports_odds.py)")
    print(f"Fetching NBA odds up to {DAYS_AHEAD} days ahead")
    print("=" * 80)
    print(f"UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Bookmaker preference: {PREFERRED_BOOKMAKERS[:4]}")

    # ---- Step 1: Fetch upcoming NBA games ----
    print(f"\n{'=' * 80}")
    print(f"FETCHING NBA GAMES (next {DAYS_AHEAD} days)")
    print(f"{'=' * 80}\n")

    all_games = []
    for day_offset in range(DAYS_AHEAD):
        d = utc_now + timedelta(days=day_offset)
        date_str = d.strftime("%Y-%m-%d")

        data = api_request("games", {"date": date_str, "league": NBA_LEAGUE_ID, "season": NBA_SEASON})
        if not data:
            print(f"  [{date_str}] Failed to fetch")
            continue

        games = data.get("response", [])
        upcoming = [g for g in games if g.get("status", {}).get("short") == "NS"]
        print(f"  [{date_str}] {len(upcoming)} upcoming games (of {len(games)} total)")

        for g in upcoming:
            g["_date"] = date_str
            all_games.append(g)

    print(f"\nTotal upcoming games: {len(all_games)}")

    if not all_games:
        print("\nNo upcoming games found")
        # Create empty CSV so pipeline doesn't break
        empty_df = pd.DataFrame(columns=[
            "game_identifier", "date", "start_time", "home_team", "away_team",
            "status", "bookmaker_used",
            "home_spread", "away_spread",
            "home_spread_odds_decimal", "away_spread_odds_decimal",
            "total_line_o", "total_line_over_odds_decimal", "total_line_under_odds_decimal",
            "away_winning_odds_decimal", "home_winning_odds_decimal",
        ])
        output_file = os.path.join(os.getcwd(), "upcoming_nba_sports_api_odds.csv")
        empty_df.to_csv(output_file, index=False)
        print(f"Saved empty CSV: {output_file}")
        return

    # ---- Step 2: Fetch odds for each game ----
    print(f"\n{'=' * 80}")
    print("FETCHING ODDS FOR EACH GAME")
    print(f"{'=' * 80}\n")

    odds_list = []
    odds_found = 0
    no_odds = 0

    for g in all_games:
        gid = g.get("id")
        home_name = g.get("teams", {}).get("home", {}).get("name", "Unknown")
        away_name = g.get("teams", {}).get("away", {}).get("name", "Unknown")
        home_alias = get_team_alias(home_name)
        away_alias = get_team_alias(away_name)
        date_str = g["_date"]
        start_time = g.get("date", "")

        game_identifier = f"{date_str}_{away_alias}@{home_alias}"

        print(f"  {game_identifier}", end="")

        # Fetch odds
        odds_data = api_request("odds", {"game": gid})
        time.sleep(0.1)

        if not odds_data or not odds_data.get("response") or len(odds_data["response"]) == 0:
            print("  -- No odds")
            no_odds += 1
            continue

        entry = odds_data["response"][0]
        bookmakers = entry.get("bookmakers", [])

        if not bookmakers:
            print("  -- No bookmakers")
            no_odds += 1
            continue

        # Select best bookmaker
        selected_bm, bm_name = select_bookmaker(bookmakers)
        if not selected_bm:
            print("  -- No matching bookmaker")
            no_odds += 1
            continue

        # Extract odds
        extracted = extract_odds_from_bookmaker(selected_bm)

        game_odds = {
            "game_identifier": game_identifier,
            "date": date_str,
            "start_time": start_time,
            "home_team": home_name,
            "away_team": away_name,
            "status": "upcoming",
            "bookmaker_used": bm_name.lower().replace(" ", ""),
            **extracted,
        }

        odds_list.append(game_odds)
        odds_found += 1

        ml_str = f"ML: {extracted['home_winning_odds_decimal']}/{extracted['away_winning_odds_decimal']}"
        spread_str = f"Spread: {extracted['home_spread']}"
        total_str = f"Total: {extracted['total_line_o']}"
        print(f"  -- {bm_name} | {ml_str} | {spread_str} | {total_str}")

    # ---- Step 3: Save to CSV ----
    print(f"\n{'=' * 80}")
    print("SAVING DATA")
    print(f"{'=' * 80}\n")

    if not odds_list:
        print("No odds data extracted")
        empty_df = pd.DataFrame(columns=[
            "game_identifier", "date", "start_time", "home_team", "away_team",
            "status", "bookmaker_used",
            "home_spread", "away_spread",
            "home_spread_odds_decimal", "away_spread_odds_decimal",
            "total_line_o", "total_line_over_odds_decimal", "total_line_under_odds_decimal",
            "away_winning_odds_decimal", "home_winning_odds_decimal",
        ])
        output_file = os.path.join(os.getcwd(), "upcoming_nba_sports_api_odds.csv")
        empty_df.to_csv(output_file, index=False)
        print(f"Saved empty CSV: {output_file}")
        return

    df_odds = pd.DataFrame(odds_list)

    # Ensure column order matches Odds_Pre_Match.py output exactly
    col_order = [
        "game_identifier", "date", "start_time", "home_team", "away_team",
        "status", "bookmaker_used",
        "home_spread", "away_spread",
        "home_spread_odds_decimal", "away_spread_odds_decimal",
        "total_line_o", "total_line_over_odds_decimal", "total_line_under_odds_decimal",
        "away_winning_odds_decimal", "home_winning_odds_decimal",
    ]
    df_odds = df_odds[col_order]

    # Convert odds columns to numeric
    odds_columns = [
        'home_winning_odds_decimal', 'away_winning_odds_decimal',
        'home_spread', 'away_spread',
        'home_spread_odds_decimal', 'away_spread_odds_decimal',
        'total_line_o', 'total_line_over_odds_decimal', 'total_line_under_odds_decimal'
    ]
    for col in odds_columns:
        if col in df_odds.columns:
            df_odds[col] = pd.to_numeric(df_odds[col], errors='coerce')

    output_file = os.path.join(os.getcwd(), "upcoming_nba_sports_api_odds.csv")
    df_odds.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"  Games: {len(df_odds)}")
    print(f"  Columns: {len(df_odds.columns)}")

    # ---- Statistics ----
    print(f"\n{'=' * 80}")
    print("STATISTICS")
    print(f"{'=' * 80}\n")

    print(f"Total upcoming NBA games found: {len(all_games)}")
    print(f"Games with odds: {odds_found}")
    print(f"Games without odds: {no_odds}")

    if 'bookmaker_used' in df_odds.columns:
        bm_counts = df_odds['bookmaker_used'].value_counts()
        print(f"\nBookmaker distribution:")
        for bm, count in bm_counts.items():
            print(f"  {bm}: {count}")

    print(f"\nDates covered: {sorted(df_odds['date'].unique())}")

    print(f"\nOdds availability:")
    print(f"  Moneyline (home): {df_odds['home_winning_odds_decimal'].notna().sum()}")
    print(f"  Moneyline (away): {df_odds['away_winning_odds_decimal'].notna().sum()}")
    print(f"  Spreads: {df_odds['home_spread'].notna().sum()}")
    print(f"  Totals: {df_odds['total_line_o'].notna().sum()}")

    if df_odds['home_spread'].notna().sum() > 0:
        print(f"\nSpread ranges:")
        print(f"  Home spread: {df_odds['home_spread'].min():.1f} to {df_odds['home_spread'].max():.1f}")
        print(f"  Away spread: {df_odds['away_spread'].min():.1f} to {df_odds['away_spread'].max():.1f}")

    if df_odds['total_line_o'].notna().sum() > 0:
        print(f"  Total line: {df_odds['total_line_o'].min():.1f} to {df_odds['total_line_o'].max():.1f}")

    # Sample
    print(f"\n{'=' * 80}")
    print("SAMPLE DATA")
    print(f"{'=' * 80}\n")

    sample_cols = [
        'game_identifier', 'date', 'bookmaker_used',
        'home_spread', 'home_spread_odds_decimal',
        'away_spread', 'away_spread_odds_decimal',
        'total_line_o', 'total_line_over_odds_decimal', 'total_line_under_odds_decimal'
    ]
    available_cols = [c for c in sample_cols if c in df_odds.columns]
    if available_cols:
        print(df_odds[available_cols].head(5).to_string(index=False))

    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
