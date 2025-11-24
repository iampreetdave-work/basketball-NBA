"""
NBA Pre-Match Feature Engineering
Fetches upcoming games and enriches them with recent team performance data

Features included:
- Today's and tomorrow's games
- Recent team statistics (last N games)
- Home/away splits
- Team season averages
- Head-to-head history
- Form metrics (win streaks, scoring trends)
- Pre-match features for ML models
"""
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Team Aliases Mapping
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

# API Configuration
API_KEY = "lIIRZjnpCE3GYS834fA6U7xBnj0dKISND5UJf4Lh"
BASE_URL = "https://api.sportradar.us/nba"
ACCESS_LEVEL = "trial"
VERSION = "v8"
LANGUAGE = "en"
FORMAT = "json"
REQUEST_DELAY = 1.5


class PreMatchFeatureEngine:
    """
    Fetches upcoming games and creates pre-match features using recent team data
    """
    
    def __init__(self, api_key=API_KEY):
        self.api_key = api_key
        self.base_url = f"{BASE_URL}/{ACCESS_LEVEL}/{VERSION}/{LANGUAGE}"
        self.request_count = 0
        
        # Cache for team data
        self.team_season_stats = {}
        self.team_recent_games = {}
        self.team_profiles = {}
        
    def _get_team_alias(self, team_name: str) -> str:
        """Get team alias from team name using manual mapping"""
        return TEAM_ALIASES.get(team_name, 'UNK')
        
    def _make_request(self, endpoint: str, retries: int = 3) -> Optional[Dict]:
        """Make API request with retry logic"""
        url = f"{self.base_url}/{endpoint}?api_key={self.api_key}"
        
        for attempt in range(retries):
            try:
                print(f"  Fetching: {endpoint[:70]}...")
                response = requests.get(url, timeout=30)
                self.request_count += 1
                
                if response.status_code == 200:
                    time.sleep(REQUEST_DELAY)
                    return response.json()
                elif response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    print(f"  Rate limit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    print(f"  Not found (404)")
                    return None
                else:
                    print(f"  Error {response.status_code}")
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
    
    def get_daily_schedule(self, date: datetime) -> Optional[Dict]:
        """Get games for a specific date"""
        date_str = date.strftime("%Y/%m/%d")
        endpoint = f"games/{date_str}/schedule.{FORMAT}"
        return self._make_request(endpoint)
    
    def get_season_schedule(self, year: int = 2024, season_type: str = "REG") -> Optional[Dict]:
        """Get full season schedule"""
        endpoint = f"games/{year}/{season_type}/schedule.{FORMAT}"
        return self._make_request(endpoint)
    
    def get_game_summary(self, game_id: str) -> Optional[Dict]:
        """Get detailed game statistics"""
        endpoint = f"games/{game_id}/summary.{FORMAT}"
        return self._make_request(endpoint)
    
    def get_team_profile(self, team_id: str) -> Optional[Dict]:
        """Get team profile with roster and basic stats"""
        if team_id in self.team_profiles:
            return self.team_profiles[team_id]
        
        endpoint = f"teams/{team_id}/profile.{FORMAT}"
        profile = self._make_request(endpoint)
        if profile:
            self.team_profiles[team_id] = profile
        return profile
    
    def get_seasonal_statistics(self, year: int = 2024, season_type: str = "REG") -> Optional[Dict]:
        """Get seasonal statistics for all teams"""
        endpoint = f"seasons/{year}/{season_type}/statistics.{FORMAT}"
        return self._make_request(endpoint)
    
    def get_upcoming_games(self, days_ahead: int = 2) -> List[Dict]:
        """
        Get games for today and the next N days
        
        Args:
            days_ahead: Number of days to look ahead (default 2 = today + tomorrow)
        """
        print(f"\n{'='*60}")
        print(f"Fetching upcoming games (next {days_ahead} days)")
        print(f"{'='*60}\n")
        
        all_games = []
        
        for day_offset in range(days_ahead):
            date = datetime.now() + timedelta(days=day_offset)
            print(f"Checking {date.strftime('%Y-%m-%d')} ({date.strftime('%A')})...")
            
            schedule = self.get_daily_schedule(date)
            
            if schedule and 'games' in schedule:
                games = schedule['games']
                print(f"  Found {len(games)} games")
                
                for game in games:
                    game['fetch_date'] = date.strftime('%Y-%m-%d')
                    all_games.append(game)
            else:
                print(f"  No games found")
        
        return all_games
    
    def get_team_recent_games(self, team_id: str, num_games: int = 5, 
                             year: int = 2024, season_type: str = "REG") -> List[Dict]:
        """
        Get recent completed games for a team
        
        Args:
            team_id: Team identifier
            num_games: Number of recent games to fetch
        """
        cache_key = f"{team_id}_{num_games}"
        if cache_key in self.team_recent_games:
            return self.team_recent_games[cache_key]
        
        print(f"  Fetching recent {num_games} games for team {team_id[:8]}...")
        
        # Get season schedule
        schedule = self.get_season_schedule(year, season_type)
        if not schedule or 'games' not in schedule:
            return []
        
        # Filter games for this team that are completed
        team_games = []
        for game in schedule['games']:
            if game.get('status') != 'closed':
                continue
            
            home_id = game.get('home', {}).get('id')
            away_id = game.get('away', {}).get('id')
            
            if team_id in [home_id, away_id]:
                team_games.append(game)
        
        # Get most recent N games
        recent_games = sorted(
            team_games, 
            key=lambda x: x.get('scheduled', ''), 
            reverse=True
        )[:num_games]
        
        # Fetch detailed stats for each game
        detailed_games = []
        for game in recent_games:
            game_summary = self.get_game_summary(game['id'])
            if game_summary:
                detailed_games.append(game_summary)
        
        self.team_recent_games[cache_key] = detailed_games
        return detailed_games
    
    def calculate_team_recent_stats(self, team_id: str, recent_games: List[Dict]) -> Dict:
        """
        Calculate statistics from recent games for a team
        
        Returns aggregated stats: averages, trends, form
        """
        if not recent_games:
            return {}
        
        stats_list = []
        results = []
        
        for game in recent_games:
            # Determine if team was home or away
            home_id = game.get('home', {}).get('id')
            away_id = game.get('away', {}).get('id')
            
            if team_id == home_id:
                team_stats = game.get('home', {}).get('statistics', {})
                opp_stats = game.get('away', {}).get('statistics', {})
                team_points = game.get('home', {}).get('points', 0)
                opp_points = game.get('away', {}).get('points', 0)
                is_home = True
            else:
                team_stats = game.get('away', {}).get('statistics', {})
                opp_stats = game.get('home', {}).get('statistics', {})
                team_points = game.get('away', {}).get('points', 0)
                opp_points = game.get('home', {}).get('points', 0)
                is_home = False
            
            stats_list.append(team_stats)
            results.append({
                'won': team_points > opp_points,
                'points_for': team_points,
                'points_against': opp_points,
                'is_home': is_home
            })
        
        # Calculate averages
        avg_stats = self._average_stats(stats_list)
        
        # Calculate form metrics
        form_stats = self._calculate_form(results)
        
        # Combine
        recent_stats = {
            'games_played': len(recent_games),
            **{f'recent_{k}': v for k, v in avg_stats.items()},
            **form_stats
        }
        
        return recent_stats
    
    def _average_stats(self, stats_list: List[Dict]) -> Dict:
        """Calculate average statistics from multiple games"""
        if not stats_list:
            return {}
        
        # Key stats to average
        key_metrics = [
            'points', 'field_goals_pct', 'three_points_pct', 'free_throws_pct',
            'rebounds', 'assists', 'turnovers', 'steals', 'blocks',
            'offensive_rebounds', 'defensive_rebounds'
        ]
        
        avg_stats = {}
        for metric in key_metrics:
            values = [s.get(metric, 0) for s in stats_list if metric in s]
            if values:
                avg_stats[metric] = round(np.mean(values), 2)
        
        return avg_stats
    
    def _calculate_form(self, results: List[Dict]) -> Dict:
        """Calculate form metrics (wins, scoring trends, etc.)"""
        if not results:
            return {}
        
        wins = sum(1 for r in results if r['won'])
        
        # Calculate point differential
        point_diffs = [r['points_for'] - r['points_against'] for r in results]
        
        # Calculate trends (using linear regression on points)
        points_scored = [r['points_for'] for r in results]
        
        form = {
            'recent_wins': wins,
            'recent_losses': len(results) - wins,
            'recent_win_pct': round(wins / len(results), 3) if results else 0,
            'recent_ppg': round(np.mean([r['points_for'] for r in results]), 2),
            'recent_opp_ppg': round(np.mean([r['points_against'] for r in results]), 2),
            'recent_point_diff': round(np.mean(point_diffs), 2),
            'recent_scoring_trend': self._calculate_trend(points_scored)
        }
        
        return form
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate if values are trending up, down, or stable"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        if slope > 1:
            return 'up'
        elif slope < -1:
            return 'down'
        else:
            return 'stable'
    
    def create_game_identifier(self, game: Dict) -> str:
        """
        Create a unique identifier that can match between APIs
        
        Format: DATE_AWAYTEAM_HOMETEAM
        This can be used to match with odds data which typically uses date + teams
        """
        scheduled = game.get('scheduled', '')
        if scheduled:
            game_date = scheduled.split('T')[0]  # Get YYYY-MM-DD
        else:
            game_date = 'UNKNOWN'
        
        home_name = game.get('home', {}).get('name', '')
        away_name = game.get('away', {}).get('name', '')
        
        home_alias = self._get_team_alias(home_name)
        away_alias = self._get_team_alias(away_name)
        
        # Create identifier: DATE_AWAY@HOME
        identifier = f"{game_date}_{away_alias}@{home_alias}"
        
        return identifier
    
    def enrich_game_with_features(self, game: Dict, recent_games_count: int = 5) -> Dict:
        """
        Enrich a single game with pre-match features
        
        Args:
            game: Game object from schedule
            recent_games_count: Number of recent games to analyze
        
        Returns:
            Dictionary with game info + pre-match features
        """
        home_id = game.get('home', {}).get('id')
        away_id = game.get('away', {}).get('id')
        home_name = game.get('home', {}).get('name', '')
        away_name = game.get('away', {}).get('name', '')
        home_alias = self._get_team_alias(home_name)
        away_alias = self._get_team_alias(away_name)
        
        print(f"\n{'─'*60}")
        print(f"Processing: {away_alias} @ {home_alias}")
        print(f"{'─'*60}")
        
        # Basic game info
        enriched_data = {
            'match_id': game.get('id', ''),
            'game_identifier': self.create_game_identifier(game),
            'scheduled': game.get('scheduled', ''),
            'status': game.get('status', ''),
            'venue_name': game.get('venue', {}).get('name', ''),
            'venue_city': game.get('venue', {}).get('city', ''),
            'league': 'NBA',
            
            'home_id': home_id,
            'home_name': home_name,
            'home_alias': home_alias,
            'home_market': game.get('home', {}).get('market', ''),
            
            'away_id': away_id,
            'away_name': away_name,
            'away_alias': away_alias,
            'away_market': game.get('away', {}).get('market', ''),
        }
        
        # Fetch recent games for home team
        print(f"Home team ({home_alias})")
        home_recent = self.get_team_recent_games(home_id, recent_games_count)
        home_stats = self.calculate_team_recent_stats(home_id, home_recent)
        
        # Add with prefix
        for key, value in home_stats.items():
            enriched_data[f'home_{key}'] = value
        
        # Fetch recent games for away team
        print(f"Away team ({away_alias})")
        away_recent = self.get_team_recent_games(away_id, recent_games_count)
        away_stats = self.calculate_team_recent_stats(away_id, away_recent)
        
        # Add with prefix
        for key, value in away_stats.items():
            enriched_data[f'away_{key}'] = value
        
        # Calculate comparative features
        enriched_data.update(self._calculate_comparative_features(home_stats, away_stats))
        
        return enriched_data
    
    def _calculate_comparative_features(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """
        Calculate comparative features between teams
        """
        comparative = {}
        
        # Point differential advantage
        home_ppg = home_stats.get('recent_ppg', 0)
        away_ppg = away_stats.get('recent_ppg', 0)
        comparative['scoring_advantage_home'] = round(home_ppg - away_ppg, 2)
        
        # Form comparison
        home_win_pct = home_stats.get('recent_win_pct', 0)
        away_win_pct = away_stats.get('recent_win_pct', 0)
        comparative['form_advantage_home'] = round(home_win_pct - away_win_pct, 3)
        
        # Defensive comparison (lower is better)
        home_opp_ppg = home_stats.get('recent_opp_ppg', 0)
        away_opp_ppg = away_stats.get('recent_opp_ppg', 0)
        comparative['defensive_advantage_home'] = round(away_opp_ppg - home_opp_ppg, 2)
        
        # Assist-to-turnover comparison
        home_assists = home_stats.get('recent_assists', 0)
        home_turnovers = home_stats.get('recent_turnovers', 1)
        away_assists = away_stats.get('recent_assists', 0)
        away_turnovers = away_stats.get('recent_turnovers', 1)
        
        home_ratio = round(home_assists / home_turnovers, 2) if home_turnovers > 0 else 0
        away_ratio = round(away_assists / away_turnovers, 2) if away_turnovers > 0 else 0
        comparative['ball_control_advantage_home'] = round(home_ratio - away_ratio, 2)
        
        return comparative
    
    def process_upcoming_games(self, days_ahead: int = 2, recent_games_count: int = 5) -> List[Dict]:
        """
        Main pipeline: Fetch upcoming games and enrich with features
        
        Args:
            days_ahead: Number of days to look ahead
            recent_games_count: Number of recent games to analyze per team
        """
        print(f"\n{'='*60}")
        print(f"PRE-MATCH FEATURE ENGINEERING")
        print(f"{'='*60}")
        print(f"Looking ahead: {days_ahead} days")
        print(f"Recent games per team: {recent_games_count}")
        print(f"{'='*60}")
        
        # Get upcoming games
        upcoming_games = self.get_upcoming_games(days_ahead)
        
        if not upcoming_games:
            print("\n❌ No upcoming games found")
            return []
        
        print(f"\n{'='*60}")
        print(f"Found {len(upcoming_games)} upcoming games")
        print(f"{'='*60}")
        
        # Enrich each game
        enriched_games = []
        for idx, game in enumerate(upcoming_games, 1):
            print(f"\n[{idx}/{len(upcoming_games)}]")
            enriched = self.enrich_game_with_features(game, recent_games_count)
            enriched_games.append(enriched)
        
        return enriched_games


def main():
    """Main execution"""
    
    # Configuration
    CONFIG = {
        'days_ahead': 2,           # Today + tomorrow
        'recent_games_count': 5,   # Last 5 games for each team
    }
    
    print("="*60)
    print("NBA PRE-MATCH FEATURE ENGINEERING")
    print("="*60)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Initialize engine
    engine = PreMatchFeatureEngine()
    
    # Process upcoming games
    enriched_games = engine.process_upcoming_games(**CONFIG)
    
    if not enriched_games:
        print("\n❌ No data to export")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(enriched_games)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Games processed: {len(df)}")
    print(f"Features per game: {len(df.columns)}")
    print(f"API requests made: {engine.request_count}")
    
    print(f"\n{'='*60}")
    print("FEATURE COLUMNS")
    print(f"{'='*60}")
    print(f"Total columns: {len(df.columns)}")
    print("\nSample columns:")
    for col in list(df.columns)[:20]:
        print(f"  - {col}")
    print(f"  ... and {len(df.columns) - 20} more")
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"nba_prematch_features_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Data saved to: {filename}")
    print(f"{'='*60}")
    
    # Display sample data
    print(f"\n{'='*60}")
    print("SAMPLE DATA (first 2 games)")
    print(f"{'='*60}")
    
    sample_cols = [
        'game_identifier', 'home_alias', 'away_alias',
        'home_recent_ppg', 'away_recent_ppg',
        'home_recent_win_pct', 'away_recent_win_pct',
        'scoring_advantage_home', 'form_advantage_home'
    ]
    
    available_cols = [col for col in sample_cols if col in df.columns]
    if available_cols:
        print(df[available_cols].head(2).to_string())
    
    print(f"\n{'='*60}")
    print("NOTES")
    print(f"{'='*60}")
    print("1. game_identifier can be used to match with odds data")
    print("   Format: DATE_AWAY@HOME (e.g., 2024-11-04_BOS@LAL)")
    print("\n2. For odds matching, use this identifier instead of API IDs")
    print("\n3. All recent_* features are from last N games")
    print("\n4. Comparative features show home team advantage")
    print("="*60)


if __name__ == "__main__":
    main()