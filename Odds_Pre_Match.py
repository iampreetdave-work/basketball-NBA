"""
Fetch DraftKings Decimal Odds for UPCOMING NBA Matches
Uses The Odds API v4
"""

import requests
import pandas as pd
from datetime import datetime
import os

# ============================================================================
# THE ODDS API CONFIGURATION
# ============================================================================

API_KEYS = [
    "83dcdaff13977e39bc65141046c993f3",
    "02a80c14ece71bed354b63915e3fb8b3",
    "30d78032b75c0922de70de22f0337b91",
    "8972d0f8f1c909b2791607ed1a29d6a5",
    "7483e0df3726e14cdb152f580291f47d"
]
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
RATE_LIMIT_THRESHOLD = 5

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


class OddsAPIClient:
    """Client for The Odds API with multi-key support"""
    
    def __init__(self, api_keys=API_KEYS):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.rate_limit_count = 0
    
    def _get_current_api_key(self) -> str:
        """Get the current active API key"""
        return self.api_keys[self.current_key_index]
    
    def _switch_api_key(self) -> None:
        """Switch to the next API key"""
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.rate_limit_count = 0
            print(f"  Switching to API key {self.current_key_index + 1}/{len(self.api_keys)}")
        else:
            self.rate_limit_count = 0
            print(f"  All API keys exhausted, resetting rate limit count")
    
    def get_upcoming_nba_odds(self):
        """Fetch upcoming NBA games with DraftKings odds"""
        print(f"\n{'='*80}")
        print(f"FETCHING UPCOMING NBA GAMES WITH DRAFTKINGS ODDS")
        print(f"{'='*80}\n")
        
        endpoint = f"{BASE_URL}/sports/{SPORT}/odds"
        params = {
            "apiKey": self._get_current_api_key(),
            "regions": "us",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "bookmakers": "draftkings"
        }
        
        try:
            print(f"Calling API...", flush=True)
            response = requests.get(endpoint, params=params, timeout=30)
            
            if response.status_code == 200:
                self.rate_limit_count = 0
                data = response.json()
                print(f"✓ Found {len(data)} games\n")
                return data
            elif response.status_code == 429:
                self.rate_limit_count += 1
                print(f"  Rate limit hit ({self.rate_limit_count}/{RATE_LIMIT_THRESHOLD})")
                
                if self.rate_limit_count >= RATE_LIMIT_THRESHOLD:
                    self._switch_api_key()
                    print("  Retrying with new API key...")
                    params["apiKey"] = self._get_current_api_key()
                    response = requests.get(endpoint, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        self.rate_limit_count = 0
                        data = response.json()
                        print(f"✓ Found {len(data)} games\n")
                        return data
                
                print("✗ RATE LIMIT - Please wait and try again")
                return None
            else:
                print(f"✗ Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"✗ Exception: {e}")
            return None


def extract_draftkings_odds(games):
    """Extract DraftKings decimal odds from games"""
    print(f"{'='*80}")
    print(f"EXTRACTING DRAFTKINGS DECIMAL ODDS")
    print(f"{'='*80}\n")
    
    odds_list = []
    
    for game in games:
        # Parse date from commence_time
        commence_time = game.get('commence_time', '')
        date = commence_time.split('T')[0] if commence_time else 'N/A'
        
        home_team = game.get('home_team', 'Unknown')
        away_team = game.get('away_team', 'Unknown')
        
        game_id = f"{date}_{TEAM_ALIASES.get(away_team, away_team[:3].upper())}@{TEAM_ALIASES.get(home_team, home_team[:3].upper())}"
        
        game_odds = {
            'game_identifier': game_id,
            'date': date,
            'start_time': commence_time,
            'home_team': home_team,
            'away_team': away_team,
            'status': 'upcoming',
        }
        
        # Find DraftKings bookmaker
        bookmakers = game.get('bookmakers', [])
        draftkings_data = None
        
        for bookmaker in bookmakers:
            if bookmaker.get('key') == 'draftkings':
                draftkings_data = bookmaker
                break
        
        if not draftkings_data:
            continue
        
        # Extract markets
        markets = draftkings_data.get('markets', [])
        
        for market in markets:
            market_key = market.get('key')
            outcomes = market.get('outcomes', [])
            
            # Moneyline (h2h)
            if market_key == 'h2h':
                for outcome in outcomes:
                    name = outcome.get('name')
                    odds_decimal = outcome.get('price')
                    
                    if name == home_team and odds_decimal:
                        game_odds['home_winning_odds_decimal'] = odds_decimal
                    elif name == away_team and odds_decimal:
                        game_odds['away_winning_odds_decimal'] = odds_decimal
            
            # Totals (over/under)
            elif market_key == 'totals':
                for outcome in outcomes:
                    name = outcome.get('name')
                    odds_decimal = outcome.get('price')
                    point = outcome.get('point')
                    
                    if name == 'Over' and odds_decimal:
                        game_odds['over_odds_decimal'] = odds_decimal
                        if point:
                            game_odds['total_line'] = point
                    elif name == 'Under' and odds_decimal:
                        game_odds['under_odds_decimal'] = odds_decimal
        
        if len(game_odds) > 6:  # Has more than just basic info
            odds_list.append(game_odds)
    
    print(f"✓ Extracted {len(odds_list)} games with DraftKings odds\n")
    return odds_list


def main():
    print("\n" + "="*80)
    print("THE ODDS API - UPCOMING NBA DRAFTKINGS ODDS")
    print("="*80)
    print(f"API keys available: {len(API_KEYS)}")
    print(f"Rate limit threshold: {RATE_LIMIT_THRESHOLD} consecutive hits")
    
    # Initialize API client
    client = OddsAPIClient()
    
    # Step 1: Get upcoming games with odds
    games = client.get_upcoming_nba_odds()
    
    if not games:
        print("\n✗ No games found")
        return
    
    # Step 2: Extract DraftKings odds
    odds_list = extract_draftkings_odds(games)
    
    if not odds_list:
        print("\n✗ No DraftKings odds found")
        return
    
    # Step 3: Create DataFrame and save
    print(f"{'='*80}")
    print("SAVING DATA")
    print(f"{'='*80}\n")
    
    df_odds = pd.DataFrame(odds_list)
    
    # Convert odds columns to numeric
    odds_columns = [
        'home_winning_odds_decimal', 'away_winning_odds_decimal',
        'over_odds_decimal', 'under_odds_decimal', 'total_line'
    ]
    for col in odds_columns:
        if col in df_odds.columns:
            df_odds[col] = pd.to_numeric(df_odds[col], errors='coerce')
    
    current_dir = os.getcwd()
    output_file = os.path.join(current_dir, "upcoming_nba_draftkings_odds.csv")
    
    df_odds.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  Games: {len(df_odds)}")
    print(f"  Columns: {len(df_odds.columns)}\n")
    
    # Step 4: Show sample
    print(f"{'='*80}")
    print("SAMPLE DATA")
    print(f"{'='*80}\n")
    
    sample_cols = [
        'game_identifier', 'date', 'start_time',
        'home_team', 'away_team',
        'home_winning_odds_decimal', 'away_winning_odds_decimal',
        'total_line', 'over_odds_decimal', 'under_odds_decimal'
    ]
    
    available_cols = [c for c in sample_cols if c in df_odds.columns]
    
    if available_cols:
        print(df_odds[available_cols].to_string(index=False))
    
    # Step 5: Statistics
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}\n")
    
    print(f"Total upcoming NBA games with DraftKings odds: {len(df_odds)}")
    print(f"\nOdds availability:")
    print(f"  Home winning: {df_odds['home_winning_odds_decimal'].notna().sum()}")
    print(f"  Away winning: {df_odds['away_winning_odds_decimal'].notna().sum()}")
    print(f"  Over: {df_odds['over_odds_decimal'].notna().sum()}")
    print(f"  Under: {df_odds['under_odds_decimal'].notna().sum()}")
    
    if df_odds['home_winning_odds_decimal'].notna().sum() > 0:
        print(f"\nOdds ranges:")
        print(f"  Home: {df_odds['home_winning_odds_decimal'].min():.2f} - {df_odds['home_winning_odds_decimal'].max():.2f}")
        print(f"  Away: {df_odds['away_winning_odds_decimal'].min():.2f} - {df_odds['away_winning_odds_decimal'].max():.2f}")
    
    if df_odds['over_odds_decimal'].notna().sum() > 0:
        print(f"  Over: {df_odds['over_odds_decimal'].min():.2f} - {df_odds['over_odds_decimal'].max():.2f}")
        print(f"  Under: {df_odds['under_odds_decimal'].min():.2f} - {df_odds['under_odds_decimal'].max():.2f}")
    
    print(f"\n{'='*80}")
    print("✓ COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
