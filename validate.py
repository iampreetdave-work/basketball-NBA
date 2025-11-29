"""
NBA Predictions Validation - WORKING VERSION
Uses exact SportRadar API format that works locally
Run this on your LOCAL MACHINE, not in Claude environment
"""

import pandas as pd
import requests
import sys
from datetime import datetime
import time

# ============================================================================
# ML CALCULATION FUNCTIONS
# ============================================================================

def calculate_ml_correct(predicted_winner, actual_winner):
    """Calculate if moneyline prediction was correct."""
    if predicted_winner == actual_winner:
        return 1
    else:
        return 0


def calculate_ml_pnl(ml_correct, moneyline_odds):
    """Calculate P/L on moneyline bet."""
    if ml_correct == 1:
        pnl = round((moneyline_odds * 1.5) - 1, 2)
    else:
        pnl = -1.0
    return pnl


# ============================================================================
# SPORTRADAR API CONFIGURATION
# ============================================================================

# 10 API Keys for rotation
SPORTSRADAR_API_KEYS = [
    'yaVs9ag9ZV7B011YWcbOFuszgN5bdeTai5r8eVWi',
    'dfgSQXX31W4efJ2Nqq71E35eVbtRBth8BYtHRYPc',
    '7iXdsTMLsQpiFV6f1aWUak0BOoYrmuAf4YD99oVE',
#    'key4_replace_with_actual_key',
#    'key5_replace_with_actual_key',
#    'key6_replace_with_actual_key',
#    'key7_replace_with_actual_key',
#    'key8_replace_with_actual_key',
#    'key9_replace_with_actual_key',
#    'key10_replace_with_actual_key',
]

SPORTSRADAR_BASE_URL = "https://api.sportradar.com/nba/trial/v8/en"
REQUEST_DELAY = 1.1


class NBAValidationEngine:
    """Fetches actual NBA game scores using SportRadar API with key rotation"""
    
    def __init__(self, api_keys=SPORTSRADAR_API_KEYS):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.consecutive_rate_limits = 0
        self.base_url = SPORTSRADAR_BASE_URL
        self.request_count = 0
        self.match_cache = {}
    
    @property
    def current_api_key(self):
        """Get current API key"""
        return self.api_keys[self.current_key_index]
    
    def rotate_api_key(self):
        """Rotate to next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.consecutive_rate_limits = 0
        print(f"      [KEY ROTATION] Switched to API key #{self.current_key_index + 1}")
    
    def handle_rate_limit(self):
        """Track consecutive rate limit hits"""
        self.consecutive_rate_limits += 1
        print(f"      [RATE LIMIT] Hit #{self.consecutive_rate_limits} for current key")
        
        if self.consecutive_rate_limits >= 5:
            print(f"      [RATE LIMIT] 5 consecutive hits - rotating key")
            self.rotate_api_key()
    
    def reset_rate_limit_counter(self):
        """Reset counter on successful request"""
        self.consecutive_rate_limits = 0
    
    def fetch_game(self, match_id, retries=3):
        """
        Fetch game details using exact working format
        Endpoint: games/{match_id}/pbp.json
        API key in header (not query parameter)
        Rotates API keys on rate limit
        """
        try:
            if match_id in self.match_cache:
                return self.match_cache[match_id]
            
            # EXACT FORMAT THAT WORKS
            url = f"{self.base_url}/games/{match_id}/pbp.json"
            headers = {
                "accept": "application/json",
                "x-api-key": self.current_api_key
            }
            
            for attempt in range(retries):
                try:
                    print(f"      [API] Key #{self.current_key_index + 1} | Fetching: {match_id}")
                    response = requests.get(url, headers=headers, timeout=15)
                    self.request_count += 1
                    
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get('status', '')
                        
                        if status == 'closed':
                            self.reset_rate_limit_counter()
                            self.match_cache[match_id] = data
                            time.sleep(REQUEST_DELAY)
                            return data
                        else:
                            print(f"      [API] Game status: {status} (not closed)")
                            return None
                    
                    elif response.status_code == 429:
                        # Rate limit hit
                        self.handle_rate_limit()
                        if attempt < retries - 1:
                            time.sleep(2)
                            continue
                        return None
                    
                    elif response.status_code == 404:
                        print(f"      [API] Game not found (404)")
                        self.reset_rate_limit_counter()
                        return None
                    
                    else:
                        print(f"      [API] Error {response.status_code}")
                        self.reset_rate_limit_counter()
                        if attempt < retries - 1:
                            time.sleep(2)
                            continue
                        return None
                
                except Exception as e:
                    print(f"      [ERROR] {str(e)}")
                    self.reset_rate_limit_counter()
                    if attempt < retries - 1:
                        time.sleep(2)
                        continue
                    return None
            
            return None
        
        except Exception as e:
            print(f"      [ERROR] {str(e)}")
            return None
    
    def extract_scores(self, game_data):
        """Extract scores from game data"""
        try:
            home_score = game_data.get('home', {}).get('points', 0)
            away_score = game_data.get('away', {}).get('points', 0)
            
            return {
                'home_score': home_score,
                'away_score': away_score,
                'status': game_data.get('status', '')
            }
        except Exception:
            return None
    
    def determine_winner(self, home_score, away_score):
        """Determine match winner from scores"""
        if home_score > away_score:
            return 'Home Win'
        elif away_score > home_score:
            return 'Away Win'
        else:
            return 'Draw'


def read_csv(csv_file):
    """Read CSV file"""
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df)} rows from {csv_file}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("🏀 NBA PREDICTIONS - VALIDATION & RESULTS")
    print("=" * 80)
    print(f"⏰ Time: {datetime.now().isoformat()}\n")
    
    # Read CSVs
    predictions_df = read_csv('NBA_PREDICTIONS_ML.csv')
    if predictions_df is None or len(predictions_df) == 0:
        print("⚠️  No predictions to validate")
        sys.exit(0)
    
    prematch_df = read_csv('nba_prematch_features.csv')
    if prematch_df is None or len(prematch_df) == 0:
        print("⚠️  No prematch features found")
        sys.exit(0)
    
    # Merge CSVs
    if 'match_id' not in prematch_df.columns:
        print("❌ Error: nba_prematch_features.csv missing 'match_id' column")
        sys.exit(1)
    
    # Try to merge on common column
    common_col = None
    for col in ['game_identifier']:
        if col in predictions_df.columns and col in prematch_df.columns:
            common_col = col
            break
    
    if common_col:
        print(f"✓ Merging on '{common_col}'")
        df_merged = pd.merge(predictions_df, prematch_df[['match_id', common_col]], 
                             on=common_col, how='inner')
    else:
        print("✓ Merging by index")
        df_merged = predictions_df.copy()
        if len(prematch_df) >= len(predictions_df):
            df_merged['match_id'] = prematch_df['match_id'].values[:len(predictions_df)]
        else:
            print("❌ Prematch CSV has fewer rows")
            sys.exit(1)
    
    print(f"✓ Merged to {len(df_merged)} games\n")
    
    # Validate
    engine = NBAValidationEngine(SPORTSRADAR_API_KEYS)
    
    validated = 0
    validated_correct = 0
    skipped = 0
    total_pnl = 0.0
    
    df_results = df_merged.copy()
    
    for idx, row in df_results.iterrows():
        match_id = row.get('match_id')
        ml_prediction = row.get('ml_prediction')
        home_win_odds = row.get('home_win_odds')
        away_win_odds = row.get('away_win_odds')
        home_team = row.get('home_team', 'Home')
        away_team = row.get('away_team', 'Away')
        
        if not match_id:
            print(f"⚠️  Row {idx + 1}: No match_id")
            skipped += 1
            continue
        
        if not ml_prediction:
            print(f"⚠️  Row {idx + 1}: No prediction")
            skipped += 1
            continue
        
        print(f"\n📋 Processing {idx + 1}/{len(df_merged)}: {home_team} vs {away_team}")
        print(f"   Match ID: {match_id}")
        print(f"   Prediction: {ml_prediction}")
        
        # Fetch game
        game_data = engine.fetch_game(match_id)
        
        if game_data is None:
            print(f"   ⏭️  Could not fetch game")
            skipped += 1
            continue
        
        # Extract scores
        scores = engine.extract_scores(game_data)
        
        if scores is None:
            print(f"   ⏭️  Could not extract scores")
            skipped += 1
            continue
        
        home_score = scores['home_score']
        away_score = scores['away_score']
        
        # Determine winner
        actual_winner = engine.determine_winner(home_score, away_score)
        
        # Get odds
        if ml_prediction == 'Home Win':
            odds = home_win_odds
        else:
            odds = away_win_odds
        
        # Calculate metrics
        ml_correct = calculate_ml_correct(ml_prediction, actual_winner)
        ml_pnl = calculate_ml_pnl(ml_correct, odds) if pd.notna(odds) and odds > 0 else None
        
        # Display result
        if ml_correct == 1:
            print(f"   ✓ CORRECT! Predicted: {ml_prediction}, Actual: {actual_winner}")
            print(f"      Score: {away_team} {away_score}-{home_score} {home_team} | P/L: ${ml_pnl}")
            validated_correct += 1
        else:
            print(f"   ✗ INCORRECT. Predicted: {ml_prediction}, Actual: {actual_winner}")
            print(f"      Score: {away_team} {away_score}-{home_score} {home_team} | P/L: ${ml_pnl}")
        
        if ml_pnl:
            total_pnl += ml_pnl
        
        # Update results
        df_results.at[idx, 'home_points_actual'] = home_score
        df_results.at[idx, 'away_points_actual'] = away_score
        df_results.at[idx, 'total_points_actual'] = home_score + away_score
        df_results.at[idx, 'ml_actual'] = actual_winner
        df_results.at[idx, 'ml_correct'] = ml_correct
        df_results.at[idx, 'ml_pnl'] = ml_pnl
        
        validated += 1
        time.sleep(0.5)
    
    # Save results
    if validated > 0:
        output_file = f"NBA_Validation.csv"
        # Remove match_id from output
        df_to_save = df_results.drop(columns=['match_id'], errors='ignore')
        df_to_save.to_csv(output_file, index=False)
        print(f"\n✓ Saved results to: {output_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    print(f"✓ Validated: {validated}")
    print(f"  ├─ Correct: {validated_correct}")
    print(f"  └─ Incorrect: {validated - validated_correct}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"Total Processed: {len(df_merged)}")
    
    if validated > 0:
        accuracy = (validated_correct / validated * 100)
        print(f"\n📈 Accuracy: {accuracy:.1f}%")
        print(f"💰 Total P/L: ${total_pnl:+.2f}")
        print(f"💵 Avg P/L per Bet: ${total_pnl / validated:+.2f}")
        roi = (total_pnl / validated * 100)
        print(f"📊 ROI: {roi:+.2f}%")
    
    print(f"\n🔍 API Requests Made: {engine.request_count}")
    print(f"🎮 Matches cached: {len(engine.match_cache)}")
    print(f"🔑 Final API Key Used: #{engine.current_key_index + 1}")
    print(f"⚠️  Rate Limit Rotations: {engine.current_key_index}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
