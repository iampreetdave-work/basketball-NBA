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
TABLE_NAME = 'predictions_nba_b1_ourmodel'
HISTORICAL_TABLE = 'model_training_nba'

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_KEYS = ["dWZbZ9jJB0Of8EqTWrFqi1xuHa9ORqRyldJxHEQq"]
BASE_URL = "https://api.sportradar.us/nba"
ACCESS_LEVEL = "trial"
VERSION = "v8"
LANGUAGE = "en"
FORMAT = "json"
REQUEST_DELAY = 1.5
RATE_LIMIT_THRESHOLD = 5


# ============================================================================
# SPORTRADAR API FETCHER
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
        return self.api_keys[self.current_key_index]

    def _switch_api_key(self) -> None:
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.rate_limit_count = 0
            print(f"    Switching to API key {self.current_key_index + 1}/{len(self.api_keys)}")
        else:
            self.rate_limit_count = 0

    def _make_request(self, endpoint: str, retries: int = 3) -> Optional[Dict]:
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
        endpoint = f"games/{game_id}/summary.{FORMAT}"
        return self._make_request(endpoint)


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def calculate_ml_correct(predicted_winner, actual_winner):
    """1 if ML prediction correct, 0 if wrong, None if no actual data."""
    if actual_winner is None or pd.isna(actual_winner):
        return None
    pred = str(predicted_winner).strip().upper()
    actual = str(actual_winner).strip().upper()
    return 1 if pred == actual else 0


def determine_ml_actual(home_points, away_points):
    """'Home Win' or 'Away Win' from actual scores."""
    if home_points is None or away_points is None:
        return None
    try:
        return "Home Win" if int(home_points) > int(away_points) else "Away Win"
    except (ValueError, TypeError):
        return None


def calculate_ml_pnl(ml_correct, moneyline_odds):
    """PnL on moneyline bet. Win: odds - 1. Lose: -1."""
    if ml_correct is None or pd.isna(ml_correct):
        return None
    try:
        odds = float(moneyline_odds) if pd.notna(moneyline_odds) else None
        if odds is None or odds <= 0:
            return None
        return round(odds - 1, 2) if ml_correct == 1 else -1.0
    except (ValueError, TypeError):
        return None


def calculate_spread_covered_actual(home_points_actual, away_points_actual, home_spread):
    """
    Did the home team cover the spread? ONE simple check.

    Formula: home_points_actual + home_spread > away_points_actual

    Returns:
        'TRUE'  — home covered
        'FALSE' — away covered
        'PUSH'  — exact tie on spread
        None    — missing data
    """
    if pd.isna(home_points_actual) or pd.isna(away_points_actual) or pd.isna(home_spread):
        return None

    try:
        home_adj = int(home_points_actual) + float(home_spread)
        away_pts = int(away_points_actual)

        if home_adj > away_pts:
            return 'TRUE'
        elif home_adj < away_pts:
            return 'FALSE'
        else:
            return 'PUSH'
    except (ValueError, TypeError):
        return None


def calculate_spread_pnl(spread_covered_predicted, spread_covered_actual, home_spread_odds, away_spread_odds):
    """
    PnL on spread bet.

    - spread_covered_predicted = 'TRUE' means we bet HOME covers → use home_spread_odds
    - spread_covered_predicted = 'FALSE' means we bet AWAY covers → use away_spread_odds
    - If prediction matches actual: profit = odds - 1
    - If prediction is wrong: loss = -1.0
    """
    if pd.isna(spread_covered_predicted) or pd.isna(spread_covered_actual):
        return None

    try:
        pred = str(spread_covered_predicted).strip().upper()
        actual = str(spread_covered_actual).strip().upper()

        # Skip pushes
        if actual == 'PUSH':
            return 0.0

        # Pick odds based on which side we BET ON (the predicted side)
        if pred == 'TRUE':
            odds = float(home_spread_odds) if pd.notna(home_spread_odds) else None
        else:
            odds = float(away_spread_odds) if pd.notna(away_spread_odds) else None

        if odds is None or odds <= 0:
            return None

        if pred == actual:
            return round(odds - 1, 2)
        else:
            return -1.0
    except (ValueError, TypeError):
        return None


def calculate_ou_correct(predicted_ou, total_points_actual, market_total_line):
    """Actual O/U outcome: 'OVER', 'UNDER', or None (push/missing)."""
    if total_points_actual is None or market_total_line is None:
        return None
    try:
        total = int(total_points_actual)
        line = float(market_total_line)
        if total > line:
            return "OVER"
        elif total < line:
            return "UNDER"
        else:
            return None
    except (ValueError, TypeError):
        return None


def calculate_ou_pnl(ou_predicted, ou_correct, over_odds, under_odds):
    """PnL on O/U bet. Uses odds for the predicted side."""
    if ou_predicted is None or ou_correct is None:
        return None
    try:
        pred = str(ou_predicted).strip().upper()
        actual = str(ou_correct).strip().upper()

        if pred == actual:
            if pred == "OVER":
                odds = float(over_odds) if pd.notna(over_odds) else None
            else:
                odds = float(under_odds) if pd.notna(under_odds) else None
            if odds is None or odds <= 0:
                return None
            return round(odds - 1, 2)
        else:
            return -1.0
    except (ValueError, TypeError):
        return None


def determine_status(home_points_actual, away_points_actual):
    """'SETTLED' if both scores exist, 'PENDING' otherwise."""
    if pd.notna(home_points_actual) and pd.notna(away_points_actual):
        return 'SETTLED'
    return 'PENDING'


# ============================================================================
# MAIN VALIDATION WORKFLOW
# ============================================================================

def validate_with_actual_data():
    """
    1. Query predictions_nba_b1_ourmodel for PENDING records
    2. Resolve match_ids (sportsradar_game_id or fallback to model_training_nba)
    3. Fetch actual scores from Sportradar
    4. Calculate: ml_actual, ml_correct, ml_pnl, spread_covered_actual, spread_pnl, ou_correct, ou_pnl
    5. Push SETTLED records back to DB
    """

    print("\n" + "="*100)
    print("NBA VALIDATION — ML + OU + SPREAD (SIMPLIFIED)")
    print("="*100)

    # ========================================================================
    # STEP 1: FETCH PENDING RECORDS
    # ========================================================================
    print("\n[STEP 1] Fetching PENDING records from database...")

    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD
        )

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
    # STEP 2: RESOLVE MATCH_IDS
    # ========================================================================
    print("\n[STEP 2] Resolving match_ids...")

    if 'sportsradar_game_id' in df_predictions.columns:
        df_predictions['match_id'] = df_predictions['sportsradar_game_id'].replace('', np.nan)
        has_id = df_predictions['match_id'].notna().sum()
        print(f"  ✓ {has_id}/{len(df_predictions)} records have sportsradar_game_id")
    else:
        df_predictions['match_id'] = np.nan
        print(f"  ⚠️  Column sportsradar_game_id not found in {TABLE_NAME}")

    missing_mask = df_predictions['match_id'].isna()
    missing_count = missing_mask.sum()

    if missing_count > 0:
        print(f"  ⚠️  {missing_count} records missing, falling back to {HISTORICAL_TABLE}...")
        try:
            query = f'SELECT game_identifier, match_id FROM "{HISTORICAL_TABLE}" ORDER BY game_identifier'
            df_match_ids = pd.read_sql(query, conn)
            match_id_lookup = dict(zip(df_match_ids['game_identifier'], df_match_ids['match_id']))
            print(f"  ✓ Loaded {len(match_id_lookup)} records from {HISTORICAL_TABLE}")

            df_predictions.loc[missing_mask, 'match_id'] = (
                df_predictions.loc[missing_mask, 'game_identifier'].map(match_id_lookup)
            )
            filled = missing_count - df_predictions['match_id'].isna().sum()
            print(f"  ✓ Filled {filled} match_ids from {HISTORICAL_TABLE}")
        except psycopg2.Error as e:
            print(f"  ⚠️  Could not query {HISTORICAL_TABLE}: {e}")
    else:
        print(f"  ✓ All records have sportsradar_game_id")

    conn.close()

    # ========================================================================
    # STEP 3: PREPARE DATA
    # ========================================================================
    print("\n[STEP 3] Preparing data...")

    missing_match_ids = df_predictions['match_id'].isna().sum()
    if missing_match_ids > 0:
        print(f"  ⚠️  {missing_match_ids} records still missing match_ids")

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
    print()

    fetcher = SportradarFetcher()
    actual_data = {}
    fetch_success = 0
    fetch_failed = 0

    for idx, row in df_valid.iterrows():
        game_id = row['match_id']
        game_identifier = row['game_identifier']

        game_summary = fetcher.get_game_summary(game_id)

        if game_summary:
            try:
                home_points = game_summary.get('home', {}).get('points')
                away_points = game_summary.get('away', {}).get('points')
                game_status = game_summary.get('status')

                # Skip games with 0-0 or null scores (not yet played)
                if (home_points is None or away_points is None or
                    (int(home_points) == 0 and int(away_points) == 0)):
                    actual_data[game_identifier] = {
                        'home_points_actual': None,
                        'away_points_actual': None,
                    }
                    fetch_success += 1
                    if (fetch_success + fetch_failed) % 10 == 0:
                        print(f"  ✓ Fetched {fetch_success}/{len(df_valid)} games")
                    continue

                actual_data[game_identifier] = {
                    'home_points_actual': home_points,
                    'away_points_actual': away_points,
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

    # Add actual data
    df_validation['home_points_actual'] = df_validation['game_identifier'].apply(
        lambda x: actual_data.get(x, {}).get('home_points_actual')
    )
    df_validation['away_points_actual'] = df_validation['game_identifier'].apply(
        lambda x: actual_data.get(x, {}).get('away_points_actual')
    )

    # Total actual
    df_validation['total_points_actual'] = (
        df_validation['home_points_actual'] + df_validation['away_points_actual']
    )

    # ========== MONEYLINE ==========
    df_validation['ml_actual'] = df_validation.apply(
        lambda row: determine_ml_actual(row['home_points_actual'], row['away_points_actual']),
        axis=1
    )

    df_validation['ml_correct'] = df_validation.apply(
        lambda row: calculate_ml_correct(row['ml_prediction'], row['ml_actual']),
        axis=1
    )

    def get_odds_for_prediction(row):
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

    df_validation['ml_pnl'] = df_validation.apply(
        lambda row: calculate_ml_pnl(row['ml_correct'], row['odds_used']),
        axis=1
    )

    # ========== SPREAD — ONE INDICATOR ==========
    # spread_covered_actual: did the home team actually cover?
    # TRUE = home covered, FALSE = away covered, PUSH = tie on spread
    df_validation['spread_covered_actual'] = df_validation.apply(
        lambda row: calculate_spread_covered_actual(
            row['home_points_actual'],
            row['away_points_actual'],
            row['home_spread']
        ),
        axis=1
    )

    # Derive home/away booleans from the single indicator
    df_validation['home_spread_covered_actual'] = df_validation['spread_covered_actual'].apply(
        lambda x: True if x == 'TRUE' else (False if x == 'FALSE' else None)
    )
    df_validation['away_spread_covered_actual'] = df_validation['spread_covered_actual'].apply(
        lambda x: True if x == 'FALSE' else (False if x == 'TRUE' else None)
    )

    # Home/away spread PnL derived from the single spread_pnl
    # spread_pnl: uses odds for the side we BET ON (predicted side)
    df_validation['spread_pnl'] = df_validation.apply(
        lambda row: calculate_spread_pnl(
            row['spread_covered_predicted'],
            row['spread_covered_actual'],
            row['home_spread_odds'],
            row['away_spread_odds']
        ),
        axis=1
    )

    # ========== OVER/UNDER ==========
    df_validation['ou_correct'] = df_validation.apply(
        lambda row: calculate_ou_correct(row['ou_predicted'], row['total_points_actual'], row['market_total_line']),
        axis=1
    )

    df_validation['ou_pnl'] = df_validation.apply(
        lambda row: calculate_ou_pnl(
            row['ou_predicted'],
            row['ou_correct'],
            row['over_odds'],
            row['under_odds']
        ),
        axis=1
    )

    # Status
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

    # Spread Stats
    spread_with_data = df_validation[
        df_validation['spread_covered_actual'].notna() &
        df_validation['spread_covered_predicted'].notna() &
        (df_validation['spread_covered_actual'] != 'PUSH')
    ]
    spread_correct = (
        spread_with_data['spread_covered_predicted'].str.strip().str.upper() ==
        spread_with_data['spread_covered_actual'].str.strip().str.upper()
    ).sum()
    spread_total = len(spread_with_data)
    accuracy_spread = (spread_correct / spread_total * 100) if spread_total > 0 else 0
    total_spread_pnl = df_validation['spread_pnl'].sum()

    # OU Stats
    ou_with_data = df_validation[
        df_validation['ou_correct'].notna() & df_validation['ou_predicted'].notna()
    ]
    correct_ou = (
        ou_with_data['ou_predicted'].str.strip().str.upper() ==
        ou_with_data['ou_correct'].str.strip().str.upper()
    ).sum()
    ou_total = len(ou_with_data)
    accuracy_ou = (correct_ou / ou_total * 100) if ou_total > 0 else 0
    total_ou_pnl = df_validation['ou_pnl'].sum()

    # Status Stats
    settled_count = (df_validation['status'] == 'SETTLED').sum()
    pending_count = (df_validation['status'] == 'PENDING').sum()

    print(f"  Total records: {len(df_validation)}")
    print(f"  With actual data: {total_with_data}")

    print(f"\n  MONEYLINE:")
    print(f"    Correct: {int(correct_ml)}/{total_with_data}")
    print(f"    Accuracy: {accuracy_ml:.1f}%")
    print(f"    Total P/L: ${total_ml_pnl:+.2f}")

    print(f"\n  SPREAD:")
    print(f"    Correct: {int(spread_correct)}/{spread_total}")
    print(f"    Accuracy: {accuracy_spread:.1f}%")
    print(f"    Total P/L: ${total_spread_pnl:+.2f}")

    print(f"\n  OVER/UNDER:")
    print(f"    Correct: {int(correct_ou)}/{ou_total}")
    print(f"    Accuracy: {accuracy_ou:.1f}%")
    print(f"    Total P/L: ${total_ou_pnl:+.2f}")

    print(f"\n  STATUS: {settled_count} SETTLED, {pending_count} PENDING")

    # Sample
    print(f"\n[SAMPLE DATA] First 5 records:")
    print("-"*100)
    sample_cols = [
        'game_identifier', 'ml_prediction', 'ml_actual', 'ml_correct', 'ml_pnl',
        'spread_covered_predicted', 'spread_covered_actual', 'spread_pnl',
        'ou_predicted', 'ou_correct', 'ou_pnl',
        'home_points_actual', 'away_points_actual', 'status'
    ]
    available_cols = [col for col in sample_cols if col in df_validation.columns]
    if available_cols:
        print(df_validation[available_cols].head(5).to_string(index=False))

    # ========================================================================
    # STEP 7: PUSH TO DATABASE
    # ========================================================================
    print("\n[STEP 7] Database Push")
    print("="*100)

    df_settled = df_validation[df_validation['status'] == 'SETTLED'].copy()

    if len(df_settled) == 0:
        print("\n  No SETTLED records to push — all games still PENDING")
        return

    push_df = df_settled[[
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
        'ou_correct',
        'ou_pnl',
        'status'
    ]].copy()

    skipped_pending = len(df_validation) - len(push_df)
    print(f"\nRecords to push: {len(push_df)} SETTLED (skipped {skipped_pending} PENDING)")
    print(f"\nSample push data:")
    print("-"*100)
    print(push_df.head(10).to_string(index=False))
    print("-"*100)

    # Connect and push
    print("\n[CONNECTING] To database...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD
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
                print(f"  ⚠️  No record found for {game_id}")

        except Exception as e:
            print(f"  ✗ {game_id}: {str(e)}")
            failed += 1

    conn.commit()

    print(f"\n{'='*100}")
    print("PUSH COMPLETE")
    print(f"{'='*100}")
    print(f"  Updated: {updated} records")
    print(f"  Failed: {failed} records")
    print(f"  Skipped: {len(push_df) - updated - failed} records (no match)")

    cursor.close()
    conn.close()

    print(f"\n{'='*100}")


if __name__ == "__main__":
    validate_with_actual_data()
