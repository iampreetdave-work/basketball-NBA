import os
import requests
import psycopg2
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List
from collections import defaultdict

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_DATABASE')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
TABLE_NAME = 'agility_nba_b1'
HISTORICAL_TABLE = 'historical_features_NBA'

# ============================================================================
# API CONFIGURATION (Same as PreMatchFeatureEngine)
# ============================================================================
API_KEYS = [
    "5Mvm4Nyfl1t0QAxknd3NIU7YBm21hL4zlmmg11S8",
    "ad5pG3KO1Y7OCJoO8IxGCRHn2sbiQsHB1CbUcRi9",
    "IRNmE6PWKVEzvVCGc7EpIHGrhnPoph98k9O2C2OB"
]
BASE_URL = "https://api.sportradar.us/nba"
ACCESS_LEVEL = "trial"
VERSION = "v8"
LANGUAGE = "en"
FORMAT = "json"
REQUEST_DELAY = 1.5
RATE_LIMIT_THRESHOLD = 5


# ============================================================================
# SPORTRADAR API FETCHER (From PreMatchFeatureEngine)
# ============================================================================
class SportradarFetcher:
    """Fetch actual game data from Sportradar using match_ids"""
    
    def __init__(self, api_keys=API_KEYS):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.rate_limit_count = 0
        self.base_url = f"{BASE_URL}/{ACCESS_LEVEL}/{VERSION}/{LANGUAGE}"
        self.request_count = 0
    
    def _get_current_api_key(self) -> str:
        """Get current active API key"""
        return self.api_keys[self.current_key_index]
    
    def _switch_api_key(self) -> None:
        """Switch to next API key"""
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.rate_limit_count = 0
            print(f"    Switching to API key {self.current_key_index + 1}/{len(self.api_keys)}")
        else:
            self.rate_limit_count = 0
    
    def _make_request(self, endpoint: str, retries: int = 3) -> Optional[Dict]:
        """Make API request with retry logic and API key rotation"""
        url = f"{self.base_url}/{endpoint}?api_key={self._get_current_api_key()}"
        
        total_attempts = 0
        max_total_attempts = 50
        
        while total_attempts < max_total_attempts:
            try:
                response = requests.get(url, timeout=30)
                self.request_count += 1
                total_attempts += 1
                
                if response.status_code == 200:
                    self.rate_limit_count = 0
                    time.sleep(REQUEST_DELAY)
                    return response.json()
                elif response.status_code == 429:
                    self.rate_limit_count += 1
                    
                    if self.rate_limit_count >= RATE_LIMIT_THRESHOLD:
                        self._switch_api_key()
                        url = f"{self.base_url}/{endpoint}?api_key={self._get_current_api_key()}"
                        self.rate_limit_count = 0
                    
                    continue
                elif response.status_code == 404:
                    return None
                else:
                    time.sleep(5)
                    continue
                    
            except Exception as e:
                time.sleep(5)
                continue
        
        return None
    
    def get_game_summary(self, game_id: str) -> Optional[Dict]:
        """Get detailed game statistics by match_id"""
        endpoint = f"games/{game_id}/summary.{FORMAT}"
        return self._make_request(endpoint)


# ============================================================================
# PNL CALCULATION FUNCTIONS
# ============================================================================

def calculate_ml_correct(predicted_winner, actual_winner):
    """
    Calculate if moneyline prediction was correct.
    
    Args:
        predicted_winner: str - "Home Win" or "Away Win"
        actual_winner: str - "Home Win" or "Away Win"
    
    Returns:
        int - 1 if correct, 0 if incorrect
    """
    pred_normalized = str(predicted_winner).strip().upper()
    actual_normalized = str(actual_winner).strip().upper()
    
    return 1 if pred_normalized == actual_normalized else 0


def determine_ml_actual(home_points, away_points):
    """
    Determine actual moneyline winner from scores.
    
    Args:
        home_points: int - home team actual points
        away_points: int - away team actual points
    
    Returns:
        str - "Home Win" or "Away Win"
    """
    if home_points is None or away_points is None:
        return None
    
    try:
        h_pts = int(home_points)
        a_pts = int(away_points)
        return "Home Win" if h_pts > a_pts else "Away Win"
    except (ValueError, TypeError):
        return None


def calculate_ml_pnl(ml_correct, moneyline_odds):
    """
    Calculate P/L on moneyline bet.
    
    Logic:
    - If correct: profit = (odds * 1) - 1
    - If incorrect: loss = -1.0
    
    Args:
        ml_correct: int - 1 if prediction correct, 0 if incorrect
        moneyline_odds: float - the odds for the predicted side
    
    Returns:
        float - rounded to 2 decimal places
    """
    try:
        odds = float(moneyline_odds) if pd.notna(moneyline_odds) else None
        
        if odds is None or odds <= 0:
            return None
        
        if ml_correct == 1:
            pnl = round((odds * 1) - 1, 2)
        else:
            pnl = -1.0
        
        return pnl
    except (ValueError, TypeError):
        return None


def calculate_home_spread_covered_actual(home_points_actual, away_points_actual, home_spread):
    """
    Calculate if HOME team covered their spread (actual results)
    Logic: delta > -home_spread
    Works for both positive (underdog) and negative (favored) spreads
    
    Args:
        home_points_actual: int - home team actual points
        away_points_actual: int - away team actual points
        home_spread: float - home team spread (SIGNED)
    
    Returns:
        bool - True if covered, False if not, None if data missing
    """
    if (pd.isna(home_points_actual) or pd.isna(away_points_actual) or pd.isna(home_spread)):
        return None
    
    try:
        delta = int(home_points_actual) - int(away_points_actual)
        spread_val = float(home_spread)
        return delta > -spread_val
    except (ValueError, TypeError):
        return None


def calculate_away_spread_covered_actual(home_points_actual, away_points_actual, away_spread):
    """
    Calculate if AWAY team covered their spread (actual results)
    Logic: delta < -away_spread
    Works for both positive (favored) and negative (underdog) spreads
    
    Args:
        home_points_actual: int - home team actual points
        away_points_actual: int - away team actual points
        away_spread: float - away team spread (SIGNED)
    
    Returns:
        bool - True if covered, False if not, None if data missing
    """
    if (pd.isna(home_points_actual) or pd.isna(away_points_actual) or pd.isna(away_spread)):
        return None
    
    try:
        delta = int(home_points_actual) - int(away_points_actual)
        spread_val = float(away_spread)
        return delta < -spread_val
    except (ValueError, TypeError):
        return None


def calculate_home_spread_pnl(home_covered_predicted, home_covered_actual, home_spread_odds):
    """
    Calculate P/L on HOME spread bet.
    
    Logic:
    - If covered_predicted == covered_actual: profit = (odds * 1) - 1
    - Else: loss = -1.0
    
    Args:
        home_covered_predicted: bool - True/False (prediction)
        home_covered_actual: bool - True/False (actual)
        home_spread_odds: float - odds for home spread
    
    Returns:
        float - rounded to 2 decimal places, or None if data missing
    """
    if (home_covered_predicted is None or home_covered_actual is None):
        return None
    
    try:
        odds = float(home_spread_odds) if pd.notna(home_spread_odds) else None
        
        if odds is None or odds <= 0:
            return None
        
        if home_covered_predicted == home_covered_actual:
            pnl = round((odds * 1) - 1, 2)
            return pnl
        else:
            return -1.0
    
    except (ValueError, TypeError):
        return None


def calculate_away_spread_pnl(away_covered_predicted, away_covered_actual, away_spread_odds):
    """
    Calculate P/L on AWAY spread bet.
    
    Logic:
    - If covered_predicted == covered_actual: profit = (odds * 1) - 1
    - Else: loss = -1.0
    
    Args:
        away_covered_predicted: bool - True/False (prediction)
        away_covered_actual: bool - True/False (actual)
        away_spread_odds: float - odds for away spread
    
    Returns:
        float - rounded to 2 decimal places, or None if data missing
    """
    if (away_covered_predicted is None or away_covered_actual is None):
        return None
    
    try:
        odds = float(away_spread_odds) if pd.notna(away_spread_odds) else None
        
        if odds is None or odds <= 0:
            return None
        
        if away_covered_predicted == away_covered_actual:
            pnl = round((odds * 1) - 1, 2)
            return pnl
        else:
            return -1.0
    
    except (ValueError, TypeError):
        return None


def calculate_ou_correct(predicted_ou, total_points_actual, market_total_line):
    """
    Calculate actual over/under outcome.
    
    Logic:
    - If total_points_actual > market_total_line: return "OVER"
    - If total_points_actual < market_total_line: return "UNDER"
    
    Args:
        predicted_ou: str - "OVER" or "UNDER" (not used for calculation, just for validation)
        total_points_actual: int - actual total points (home + away)
        market_total_line: float - the market total line
    
    Returns:
        str - "OVER" or "UNDER", None if data is missing
    """
    if total_points_actual is None or market_total_line is None:
        return None
    
    try:
        total = int(total_points_actual)
        line = float(market_total_line)
        
        # Determine actual outcome
        if total > line:
            return "OVER"
        elif total < line:
            return "UNDER"
        else:
            return None  # Push scenario
    except (ValueError, TypeError):
        return None


def calculate_ou_pnl(ou_predicted, ou_correct, over_odds, under_odds):
    """
    Calculate P/L on over/under bet.
    
    Logic:
    - If ou_predicted == ou_correct: profit = (ou_odds * 1) - 1
    - If not equal: loss = -1.0
    
    Args:
        ou_predicted: str - "OVER" or "UNDER"
        ou_correct: str - "OVER" or "UNDER" (actual outcome)
        over_odds: float - odds for OVER
        under_odds: float - odds for UNDER
    
    Returns:
        float - rounded to 2 decimal places, or None if data missing
    """
    if ou_predicted is None or ou_correct is None:
        return None
    
    try:
        pred_normalized = str(ou_predicted).strip().upper()
        correct_normalized = str(ou_correct).strip().upper()
        
        # Check if prediction matches actual
        if pred_normalized == correct_normalized:
            # Get the odds for the predicted side
            if pred_normalized == "OVER":
                odds = float(over_odds) if pd.notna(over_odds) else None
            elif pred_normalized == "UNDER":
                odds = float(under_odds) if pd.notna(under_odds) else None
            else:
                return None
            
            if odds is None or odds <= 0:
                return None
            
            pnl = round((odds * 1) - 1, 2)
        else:
            pnl = -1.0
        
        return pnl
    except (ValueError, TypeError):
        return None


def determine_status(home_points_actual, away_points_actual):
    """
    Determine game status based on actual points.
    
    Args:
        home_points_actual: int or None - home team actual points
        away_points_actual: int or None - away team actual points
    
    Returns:
        str - "SETTLED" if both scores exist, "PENDING" otherwise
    """
    if pd.notna(home_points_actual) and pd.notna(away_points_actual):
        return 'SETTLED'
    else:
        return 'PENDING'


# ============================================================================
# MAIN VALIDATION WORKFLOW
# ============================================================================

def validate_with_actual_data():
    """
    Main workflow:
    1. Query agility_nba_b1 table for PENDING status records
    2. Query historical_features_NBA to get match_ids
    3. Fetch actual game data from Sportradar
    4. Calculate ml_actual, ml_correct, ml_pnl, ou_correct, spread_covered_actual, spread_pnl, ou_pnl
    5. Update status based on whether actual data exists
    6. Push to database
    """
    
    print("\n" + "="*100)
    print("NBA VALIDATION WITH ACTUAL DATA FETCH (ML + OU + SPREAD)")
    print("="*100)
    
    # ========================================================================
    # STEP 1: FETCH PENDING RECORDS FROM agility_nba_b1
    # ========================================================================
    print("\n[STEP 1] Fetching PENDING records from database...")
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Query for PENDING records
        query = f"""
            SELECT * FROM "{TABLE_NAME}"
            WHERE status = 'PENDING'
            ORDER BY game_identifier
        """
        
        df_predictions = pd.read_sql(query, conn)
        print(f"  ✓ Loaded {len(df_predictions)} PENDING predictions from {TABLE_NAME}")
        
        if len(df_predictions) == 0:
            print("  ✗ No PENDING records found")
            conn.close()
            return
    
    except psycopg2.Error as e:
        print(f"  ✗ Database error: {e}")
        return
    
    # ========================================================================
    # STEP 2: FETCH MATCH_IDS FROM historical_features_NBA
    # ========================================================================
    print("\n[STEP 2] Fetching match_ids from historical_features_NBA...")
    
    try:
        # Query for match_ids
        query = f"""
            SELECT game_identifier, match_id FROM "{HISTORICAL_TABLE}"
            ORDER BY game_identifier
        """
        
        df_match_ids = pd.read_sql(query, conn)
        print(f"  ✓ Loaded {len(df_match_ids)} records from {HISTORICAL_TABLE}")
        
        # Create match_id lookup by game_identifier
        match_id_lookup = dict(zip(df_match_ids['game_identifier'], df_match_ids['match_id']))
        print(f"  ✓ Created lookup for {len(match_id_lookup)} match_ids")
        
        conn.close()
    
    except psycopg2.Error as e:
        print(f"  ✗ Database error: {e}")
        return
    
    # ========================================================================
    # STEP 3: MERGE AND PREPARE DATA
    # ========================================================================
    print("\n[STEP 3] Merging predictions with match_ids...")
    
    # Add match_id column from lookup
    df_predictions['match_id'] = df_predictions['game_identifier'].map(match_id_lookup)
    
    missing_match_ids = df_predictions['match_id'].isna().sum()
    if missing_match_ids > 0:
        print(f"  ⚠️  {missing_match_ids} records missing match_ids (game not in historical features)")
    
    # Filter out records without match_ids
    df_valid = df_predictions[df_predictions['match_id'].notna()].copy()
    print(f"  ✓ {len(df_valid)} records have match_ids to fetch")
    
    if len(df_valid) == 0:
        print("  ✗ No valid records to process")
        return
    
    # ========================================================================
    # STEP 4: FETCH ACTUAL DATA FROM SPORTRADAR
    # ========================================================================
    print("\n[STEP 4] Fetching actual game data from Sportradar...")
    print(f"  API keys available: {len(API_KEYS)}")
    print(f"  Rate limit threshold: {RATE_LIMIT_THRESHOLD}")
    print()
    
    fetcher = SportradarFetcher()
    
    # Store fetched data
    actual_data = {}
    fetch_success = 0
    fetch_failed = 0
    
    for idx, row in df_valid.iterrows():
        game_id = row['match_id']
        game_identifier = row['game_identifier']
        
        # Fetch game summary
        game_summary = fetcher.get_game_summary(game_id)
        
        if game_summary:
            try:
                home_points = game_summary.get('home', {}).get('points')
                away_points = game_summary.get('away', {}).get('points')
                
                actual_data[game_identifier] = {
                    'home_points_actual': home_points,
                    'away_points_actual': away_points,
                    'status': game_summary.get('status'),
                    'raw_data': game_summary
                }
                fetch_success += 1
                
                if (fetch_success + fetch_failed) % 10 == 0:
                    print(f"  ✓ Fetched {fetch_success}/{len(df_valid)} games")
            except Exception as e:
                actual_data[game_identifier] = {'error': str(e)}
                fetch_failed += 1
        else:
            actual_data[game_identifier] = {'error': 'No data returned'}
            fetch_failed += 1
    
    print(f"\n  ✓ Successfully fetched: {fetch_success}")
    print(f"  ✗ Failed to fetch: {fetch_failed}")
    print(f"  API requests made: {fetcher.request_count}")
    
    # ========================================================================
    # STEP 5: CALCULATE VALIDATION METRICS
    # ========================================================================
    print("\n[STEP 5] Calculating validation metrics...")
    
    df_validation = df_valid.copy()
    
    # Add actual data columns
    df_validation['home_points_actual'] = df_validation['game_identifier'].apply(
        lambda x: actual_data.get(x, {}).get('home_points_actual')
    )
    df_validation['away_points_actual'] = df_validation['game_identifier'].apply(
        lambda x: actual_data.get(x, {}).get('away_points_actual')
    )
    
    # Calculate total actual
    df_validation['total_points_actual'] = (
        df_validation['home_points_actual'] + df_validation['away_points_actual']
    )
    
    # ========== MONEYLINE CALCULATIONS ==========
    # Determine ML actual winner
    df_validation['ml_actual'] = df_validation.apply(
        lambda row: determine_ml_actual(row['home_points_actual'], row['away_points_actual']),
        axis=1
    )
    
    # Calculate ML correctness
    df_validation['ml_correct'] = df_validation.apply(
        lambda row: calculate_ml_correct(row['ml_prediction'], row['ml_actual']),
        axis=1
    )
    
    # Get correct odds based on prediction
    def get_odds_for_prediction(row):
        """Extract odds based on which side was predicted"""
        predicted = str(row.get('ml_prediction', '')).strip().upper()
        home_odds = row.get('home_win_odds')
        away_odds = row.get('away_win_odds')
        
        if pd.notna(home_odds) and pd.notna(away_odds):
            if predicted == 'HOME WIN':
                return home_odds
            elif predicted == 'AWAY WIN':
                return away_odds
        return None
    
    df_validation['odds_used'] = df_validation.apply(get_odds_for_prediction, axis=1)
    
    # Calculate PnL
    df_validation['ml_pnl'] = df_validation.apply(
        lambda row: calculate_ml_pnl(row['ml_correct'], row['odds_used']),
        axis=1
    )
    
    # ========== SPREAD CALCULATIONS ==========
    # Calculate HOME spread covered actual
    df_validation['home_spread_covered_actual'] = df_validation.apply(
        lambda row: calculate_home_spread_covered_actual(
            row['home_points_actual'], 
            row['away_points_actual'],
            row['home_spread']
        ),
        axis=1
    )
    
    # Calculate AWAY spread covered actual
    df_validation['away_spread_covered_actual'] = df_validation.apply(
        lambda row: calculate_away_spread_covered_actual(
            row['home_points_actual'], 
            row['away_points_actual'],
            row['away_spread']
        ),
        axis=1
    )
    
    # Calculate HOME spread PnL
    df_validation['home_spread_pnl'] = df_validation.apply(
        lambda row: calculate_home_spread_pnl(
            row['home_spread_covered_predicted'],
            row['home_spread_covered_actual'],
            row['home_spread_odds']
        ),
        axis=1
    )
    
    # Calculate AWAY spread PnL
    df_validation['away_spread_pnl'] = df_validation.apply(
        lambda row: calculate_away_spread_pnl(
            row['away_spread_covered_predicted'],
            row['away_spread_covered_actual'],
            row['away_spread_odds']
        ),
        axis=1
    )
    
    # ========== OLD SPREAD CALCULATIONS - KEEP FOR BACKWARD COMPATIBILITY ==========
    # Calculate spread covered actual (OLD LOGIC)
    df_validation['spread_covered_actual'] = df_validation.apply(
        lambda row: 'TRUE' if (pd.notna(row['home_points_actual']) and pd.notna(row['away_points_actual']) and 
                              pd.notna(row['home_spread']) and abs(int(row['home_points_actual']) - int(row['away_points_actual'])) >= abs(float(row['home_spread'])))
                  else ('FALSE' if (pd.notna(row['home_points_actual']) and pd.notna(row['away_points_actual']) and pd.notna(row['home_spread']))
                  else None),
        axis=1
    )
    
    # Calculate spread PnL (OLD LOGIC)
    df_validation['spread_pnl'] = df_validation.apply(
        lambda row: (round((float(row['home_spread_odds']) * 1) - 1, 2) if pd.notna(row['home_spread_odds']) and float(row['home_spread_odds']) > 0 else None)
                   if (pd.notna(row['spread_covered_predicted']) and pd.notna(row['spread_covered_actual']) and 
                       str(row['spread_covered_predicted']).strip().upper() == str(row['spread_covered_actual']).strip().upper())
                   else (-1.0 if (pd.notna(row['spread_covered_predicted']) and pd.notna(row['spread_covered_actual'])) else None),
        axis=1
    )
    
    # ========== OVER/UNDER CALCULATIONS ==========
    # Calculate OU outcome (OVER/UNDER) - this is what gets stored in database
    df_validation['ou_correct'] = df_validation.apply(
        lambda row: calculate_ou_correct(row['ou_predicted'], row['total_points_actual'], row['market_total_line']),
        axis=1
    )
    
    # Calculate OU correctness (1/0) for summary statistics only
    df_validation['ou_correct_numeric'] = df_validation.apply(
        lambda row: 1 if (row['ou_predicted'] is not None and row['ou_correct'] is not None 
                         and str(row['ou_predicted']).strip().upper() == str(row['ou_correct']).strip().upper()) else 0,
        axis=1
    )
    
    # Calculate OU PnL
    df_validation['ou_pnl'] = df_validation.apply(
        lambda row: calculate_ou_pnl(
            row['ou_predicted'],
            row['ou_correct'],
            row['over_odds'],
            row['under_odds']
        ),
        axis=1
    )
    
    # Determine status based on actual points
    df_validation['status'] = df_validation.apply(
        lambda row: determine_status(row['home_points_actual'], row['away_points_actual']),
        axis=1
    )
    
    print(f"  ✓ Calculated metrics for {len(df_validation)} records")
    
    # ========================================================================
    # STEP 6: SUMMARY STATISTICS
    # ========================================================================
    print("\n[STEP 6] Validation Summary")
    print("="*100)
    
    total_with_data = df_validation['home_points_actual'].notna().sum()
    
    # ML Stats
    correct_ml = df_validation['ml_correct'].sum()
    accuracy_ml = (correct_ml / total_with_data * 100) if total_with_data > 0 else 0
    total_ml_pnl = df_validation['ml_pnl'].sum()
    
    # HOME Spread Stats
    home_spread_correct = df_validation.apply(
        lambda row: 1 if (pd.notna(row['home_spread_covered_actual']) and 
                         row['home_spread_covered_predicted'] == row['home_spread_covered_actual']) else 0,
        axis=1
    ).sum()
    accuracy_home_spread = (home_spread_correct / total_with_data * 100) if total_with_data > 0 else 0
    total_home_spread_pnl = df_validation['home_spread_pnl'].sum()
    
    # AWAY Spread Stats
    away_spread_correct = df_validation.apply(
        lambda row: 1 if (pd.notna(row['away_spread_covered_actual']) and 
                         row['away_spread_covered_predicted'] == row['away_spread_covered_actual']) else 0,
        axis=1
    ).sum()
    accuracy_away_spread = (away_spread_correct / total_with_data * 100) if total_with_data > 0 else 0
    total_away_spread_pnl = df_validation['away_spread_pnl'].sum()
    
    # OU Stats
    correct_ou = df_validation['ou_correct_numeric'].sum()
    accuracy_ou = (correct_ou / total_with_data * 100) if total_with_data > 0 else 0
    total_ou_pnl = df_validation['ou_pnl'].sum()
    
    # Status Stats
    settled_count = (df_validation['status'] == 'SETTLED').sum()
    pending_count = (df_validation['status'] == 'PENDING').sum()
    
    print(f"  Total records: {len(df_validation)}")
    print(f"  With actual data: {total_with_data}")
    
    print(f"\n  MONEYLINE RESULTS:")
    print(f"    Correct predictions: {int(correct_ml)}")
    print(f"    Accuracy: {accuracy_ml:.1f}%")
    print(f"    Total P/L: ${total_ml_pnl:+.2f}")
    if total_with_data > 0:
        print(f"    Avg P/L per bet: ${total_ml_pnl / total_with_data:+.2f}")
    
    print(f"\n  HOME SPREAD RESULTS:")
    print(f"    Correct predictions: {int(home_spread_correct)}")
    print(f"    Accuracy: {accuracy_home_spread:.1f}%")
    print(f"    Total P/L: ${total_home_spread_pnl:+.2f}")
    if total_with_data > 0:
        print(f"    Avg P/L per bet: ${total_home_spread_pnl / total_with_data:+.2f}")
    
    print(f"\n  AWAY SPREAD RESULTS:")
    print(f"    Correct predictions: {int(away_spread_correct)}")
    print(f"    Accuracy: {accuracy_away_spread:.1f}%")
    print(f"    Total P/L: ${total_away_spread_pnl:+.2f}")
    if total_with_data > 0:
        print(f"    Avg P/L per bet: ${total_away_spread_pnl / total_with_data:+.2f}")
    
    print(f"\n  OVER/UNDER RESULTS:")
    print(f"    Correct predictions: {int(correct_ou)}")
    print(f"    Accuracy: {accuracy_ou:.1f}%")
    print(f"    Total P/L: ${total_ou_pnl:+.2f}")
    if total_with_data > 0:
        print(f"    Avg P/L per bet: ${total_ou_pnl / total_with_data:+.2f}")
    
    print(f"\n  STATUS:")
    print(f"    SETTLED: {settled_count}")
    print(f"    PENDING: {pending_count}")
    
    # Show sample records
    print(f"\n[SAMPLE DATA] First 5 records with calculations:")
    print("-"*100)
    
    sample_cols = [
        'game_identifier', 'ml_prediction', 'ml_actual', 'ml_correct', 'ml_pnl',
        'home_spread_covered_predicted', 'home_spread_covered_actual', 'home_spread_pnl',
        'away_spread_covered_predicted', 'away_spread_covered_actual', 'away_spread_pnl',
        'ou_predicted', 'ou_correct', 'ou_pnl',
        'home_points_actual', 'away_points_actual', 'total_points_actual', 'status'
    ]
    available_cols = [col for col in sample_cols if col in df_validation.columns]
    
    if available_cols:
        print(df_validation[available_cols].head(5).to_string(index=False))
    
    # ========================================================================
    # STEP 7: PUSH TO DATABASE
    # ========================================================================
    print("\n[STEP 7] Database Push")
    print("="*100)
    
    push_df = df_validation[[
        'game_identifier',
        'home_points_actual',
        'away_points_actual',
        'total_points_actual',
        'ml_actual',
        'ml_correct',
        'ml_pnl',
        'spread_covered_actual',
        'spread_pnl',
        'home_spread_covered_actual',
        'away_spread_covered_actual',
        'home_spread_pnl',
        'away_spread_pnl',
        'ou_correct',
        'ou_pnl',
        'status'
    ]].copy()
    
    print(f"\nRecords to push: {len(push_df)}")
    print(f"\nSample push data:")
    print("-"*100)
    print(push_df.head(10).to_string(index=False))
    print("-"*100)
    print()
    
    # Connect and push
    print("\n[CONNECTING] To database...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        print(f"  ✓ Connected to {TABLE_NAME}")
    except psycopg2.Error as e:
        print(f"  ✗ Database error: {e}")
        return
    
    updated = 0
    failed = 0
    
    print(f"\n[PUSHING] {len(push_df)} records...")
    
    for idx, row in push_df.iterrows():
        game_id = row['game_identifier']
        
        try:
            update_query = f"""
            UPDATE "{TABLE_NAME}"
            SET home_points_actual = %s,
                away_points_actual = %s,
                total_points_actual = %s,
                ml_actual = %s,
                ml_correct = %s::boolean,
                ml_pnl = %s,
                spread_covered_actual = %s,
                spread_pnl = %s,
                home_spread_covered_actual = %s,
                away_spread_covered_actual = %s,
                home_spread_pnl = %s,
                away_spread_pnl = %s,
                ou_correct = %s,
                ou_pnl = %s,
                status = %s
            WHERE game_identifier = %s
            """
            
            cursor.execute(update_query, (
                int(row['home_points_actual']) if pd.notna(row['home_points_actual']) else None,
                int(row['away_points_actual']) if pd.notna(row['away_points_actual']) else None,
                int(row['total_points_actual']) if pd.notna(row['total_points_actual']) else None,
                row['ml_actual'],
                int(row['ml_correct']) if pd.notna(row['ml_correct']) else None,
                float(row['ml_pnl']) if pd.notna(row['ml_pnl']) else None,
                row['spread_covered_actual'],
                float(row['spread_pnl']) if pd.notna(row['spread_pnl']) else None,
                row['home_spread_covered_actual'],
                row['away_spread_covered_actual'],
                float(row['home_spread_pnl']) if pd.notna(row['home_spread_pnl']) else None,
                float(row['away_spread_pnl']) if pd.notna(row['away_spread_pnl']) else None,
                row['ou_correct'],
                float(row['ou_pnl']) if pd.notna(row['ou_pnl']) else None,
                row['status'],
                game_id
            ))
            
            rows_affected = cursor.rowcount
            if rows_affected > 0:
                updated += rows_affected
                if (idx + 1) % 20 == 0:
                    print(f"  ✓ Processed {idx + 1}/{len(push_df)}")
            else:
                print(f"  ⚠️  Row {idx + 1}: No record found for {game_id}")
        
        except Exception as e:
            print(f"  ✗ Row {idx + 1} ({game_id}): {str(e)}")
            failed += 1
    
    # Commit
    conn.commit()
    
    print(f"\n{'='*100}")
    print("✓ PUSH COMPLETE")
    print(f"{'='*100}")
    print(f"✓ Successfully updated: {updated} records")
    print(f"✗ Failed: {failed} records")
    print(f"⭐ Skipped: {len(push_df) - updated - failed} records (no match)")
    
    cursor.close()
    conn.close()
    
    print(f"\n{'='*100}")


if __name__ == "__main__":
    validate_with_actual_data()
