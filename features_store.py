import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
import os
import sys

# ============================================================================
# DATABASE CREDENTIALS FROM ENVIRONMENT VARIABLES
# ============================================================================
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_DATABASE')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Validate credentials
if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
    print("❌ Missing database credentials in environment variables")
    print("Required: DB_HOST, DB_DATABASE, DB_USER, DB_PASSWORD")
    sys.exit(1)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================
def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

# ============================================================================
# LOAD DATA FROM CSV
# ============================================================================
def load_data():
    """Load data from CSV file"""
    try:
        # Try multiple possible filenames
        possible_files = ['Future.csv', 'Future__1_.csv']
        df = None
        
        for file in possible_files:
            try:
                df = pd.read_csv(file, on_bad_lines='skip')
                print(f"✓ Loaded data from {file}")
                break
            except:
                continue
        
        if df is None:
            raise FileNotFoundError("Could not find Future.csv or Future__1_.csv")
        
        print(f"✓ Total games loaded: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

# ============================================================================
# CALCULATE DERIVED FEATURES
# ============================================================================
def calculate_derived_features(row):
    """Calculate derived features for a single row"""
    try:
        h_ppg = float(row['home_recent_ppg']) if pd.notna(row['home_recent_ppg']) else 0
        a_ppg = float(row['away_recent_ppg']) if pd.notna(row['away_recent_ppg']) else 0
        h_opp_ppg = float(row['home_recent_opp_ppg']) if pd.notna(row['home_recent_opp_ppg']) else 0
        a_opp_ppg = float(row['away_recent_opp_ppg']) if pd.notna(row['away_recent_opp_ppg']) else 0
        h_win_pct = float(row['home_recent_win_pct']) if pd.notna(row['home_recent_win_pct']) else 0
        a_win_pct = float(row['away_recent_win_pct']) if pd.notna(row['away_recent_win_pct']) else 0
        h_steals = float(row['home_recent_steals']) if pd.notna(row['home_recent_steals']) else 0
        a_steals = float(row['away_recent_steals']) if pd.notna(row['away_recent_steals']) else 0
        h_blocks = float(row['home_recent_blocks']) if pd.notna(row['home_recent_blocks']) else 0
        a_blocks = float(row['away_recent_blocks']) if pd.notna(row['away_recent_blocks']) else 0
        h_odds = float(row['home_winning_odds_decimal']) if pd.notna(row['home_winning_odds_decimal']) else 1
        total_line = float(row['total_line_o']) if pd.notna(row['total_line_o']) else 0
        
        ppg_diff = h_ppg - a_ppg
        ppg_sum = h_ppg + a_ppg
        net_rating_diff = (h_ppg - h_opp_ppg) - (a_ppg - a_opp_ppg)
        win_pct_diff = h_win_pct - a_win_pct
        implied_home_prob = 1 / h_odds if h_odds > 0 else 0
        steals_diff = h_steals - a_steals
        blocks_diff = h_blocks - a_blocks
        defense_diff = steals_diff + blocks_diff
        line_bias = total_line - ppg_sum
        
        return {
            'ppg_diff': ppg_diff,
            'ppg_sum': ppg_sum,
            'net_rating_diff': net_rating_diff,
            'win_pct_diff': win_pct_diff,
            'implied_home_prob': implied_home_prob,
            'steals_diff': steals_diff,
            'blocks_diff': blocks_diff,
            'defense_diff': defense_diff,
            'line_bias': line_bias
        }
    except Exception as e:
        return {k: None for k in ['ppg_diff', 'ppg_sum', 'net_rating_diff', 'win_pct_diff', 
                                  'implied_home_prob', 'steals_diff', 'blocks_diff', 'defense_diff', 'line_bias']}

# ============================================================================
# PREPARE DATA FOR DATABASE
# ============================================================================
def prepare_data_for_db(df):
    """Prepare data for database insertion"""
    
    print("\n[2/4] Calculating derived features...")
    
    # Parse date from scheduled column
    df['date'] = pd.to_datetime(df['scheduled']).dt.date
    
    # Calculate derived features for each row
    derived_features = df.apply(calculate_derived_features, axis=1, result_type='expand')
    
    # Rename home_name and away_name columns if they exist
    if 'home_name' in df.columns:
        df['home_team'] = df['home_name']
    if 'away_name' in df.columns:
        df['away_team'] = df['away_name']
    
    # All columns to include
    all_columns = [
        'game_identifier', 'match_id', 'date', 'league', 'status',
        'home_id', 'home_team', 'away_id', 'away_team',
        'home_games_played', 'home_recent_points', 'home_recent_field_goals_pct',
        'home_recent_three_points_pct', 'home_recent_free_throws_pct', 'home_recent_assists',
        'home_recent_steals', 'home_recent_blocks', 'home_recent_offensive_rebounds',
        'home_recent_defensive_rebounds', 'home_recent_wins', 'home_recent_losses',
        'home_recent_win_pct', 'home_recent_ppg', 'home_recent_opp_ppg',
        'home_recent_point_diff', 'home_recent_scoring_trend',
        'away_games_played', 'away_recent_points', 'away_recent_field_goals_pct',
        'away_recent_three_points_pct', 'away_recent_free_throws_pct', 'away_recent_assists',
        'away_recent_steals', 'away_recent_blocks', 'away_recent_offensive_rebounds',
        'away_recent_defensive_rebounds', 'away_recent_wins', 'away_recent_losses',
        'away_recent_win_pct', 'away_recent_ppg', 'away_recent_opp_ppg',
        'away_recent_point_diff', 'away_recent_scoring_trend',
        'scoring_advantage_home', 'form_advantage_home', 'defensive_advantage_home',
        'ball_control_advantage_home',
        'home_winning_odds_decimal', 'away_winning_odds_decimal',
        'home_spread', 'away_spread', 'home_spread_odds_decimal', 'away_spread_odds_decimal',
        'total_line_o', 'total_line_over_odds_decimal', 'total_line_under_odds_decimal'
    ]
    
    # Build result dataframe with all columns
    result_data = {}
    for col in all_columns:
        if col in df.columns:
            result_data[col] = df[col].values
        else:
            result_data[col] = [None] * len(df)
    
    # Add derived features
    for col in derived_features.columns:
        result_data[col] = derived_features[col].values
    
    result_df = pd.DataFrame(result_data)
    
    # Handle NaN and infinity
    result_df = result_df.where(pd.notna(result_df), None)
    result_df = result_df.replace([np.inf, -np.inf], None)
    
    # Remove duplicates from CSV based on game_identifier
    print("\n[2.5/4] Checking for duplicates in CSV...")
    initial_count = len(result_df)
    
    # Drop rows with missing game_identifier first
    result_df = result_df[result_df['game_identifier'].notna()]
    missing_id_count = initial_count - len(result_df)
    if missing_id_count > 0:
        print(f"⚠️  Removed {missing_id_count} row(s) with missing game_identifier")
    
    # Drop duplicate game_identifiers, keeping the first occurrence
    result_df = result_df.drop_duplicates(subset=['game_identifier'], keep='first')
    duplicates_removed = initial_count - missing_id_count - len(result_df)
    
    if duplicates_removed > 0:
        print(f"⚠️  Removed {duplicates_removed} duplicate game_identifier(s) from CSV")
    
    print(f"✓ Prepared {len(result_df)} unique records with all features")
    return result_df

# ============================================================================
# INSERT DATA INTO DATABASE (ROW BY ROW)
# ============================================================================
def insert_data_to_db(df_clean):
    """Insert data in database row by row, skip if already exists in DB"""
    
    print("\n[3/4] Inserting into database (checking DB for each row)...")
    print("Note: CSV duplicates already removed\n")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # All columns to insert
    db_columns = [
        'game_identifier', 'match_id', 'date', 'league', 'status',
        'home_id', 'home_team', 'away_id', 'away_team',
        'home_games_played', 'home_recent_points', 'home_recent_field_goals_pct',
        'home_recent_three_points_pct', 'home_recent_free_throws_pct', 'home_recent_assists',
        'home_recent_steals', 'home_recent_blocks', 'home_recent_offensive_rebounds',
        'home_recent_defensive_rebounds', 'home_recent_wins', 'home_recent_losses',
        'home_recent_win_pct', 'home_recent_ppg', 'home_recent_opp_ppg',
        'home_recent_point_diff', 'home_recent_scoring_trend',
        'away_games_played', 'away_recent_points', 'away_recent_field_goals_pct',
        'away_recent_three_points_pct', 'away_recent_free_throws_pct', 'away_recent_assists',
        'away_recent_steals', 'away_recent_blocks', 'away_recent_offensive_rebounds',
        'away_recent_defensive_rebounds', 'away_recent_wins', 'away_recent_losses',
        'away_recent_win_pct', 'away_recent_ppg', 'away_recent_opp_ppg',
        'away_recent_point_diff', 'away_recent_scoring_trend',
        'scoring_advantage_home', 'form_advantage_home', 'defensive_advantage_home',
        'ball_control_advantage_home',
        'home_winning_odds_decimal', 'away_winning_odds_decimal',
        'home_spread', 'away_spread', 'home_spread_odds_decimal', 'away_spread_odds_decimal',
        'total_line_o', 'total_line_over_odds_decimal', 'total_line_under_odds_decimal',
        'ppg_diff', 'ppg_sum', 'net_rating_diff', 'win_pct_diff', 'implied_home_prob',
        'steals_diff', 'blocks_diff', 'defense_diff', 'line_bias'
    ]
    
    inserted_count = 0
    skipped_count = 0
    error_count = 0
    
    try:
        for idx, row in df_clean.iterrows():
            try:
                game_id = row.get('game_identifier', None)
                
                if not game_id or pd.isna(game_id):
                    print(f"⚠️  Row {idx + 1}: Skipping - no game_identifier")
                    skipped_count += 1
                    continue
                
                # Check if record already exists in database
                cur.execute(
                    'SELECT game_identifier FROM "historical_features_NBA" WHERE game_identifier = %s',
                    (game_id,)
                )
                exists = cur.fetchone()
                
                if exists:
                    # SKIP - record already exists
                    skipped_count += 1
                    print(f"⊘ Row {idx + 1}: Skipped (already exists) - {game_id}")
                    continue
                
                # Prepare values for this row
                values = []
                for col in db_columns:
                    val = row.get(col, None)
                    # Convert numpy types to Python types
                    if pd.notna(val):
                        if isinstance(val, (np.integer, np.floating)):
                            val = float(val) if isinstance(val, np.floating) else int(val)
                        values.append(val)
                    else:
                        values.append(None)
                
                # INSERT new record
                columns_str = ', '.join([f'"{col}"' for col in db_columns])
                placeholders = ', '.join(['%s'] * len(db_columns))
                
                query = f"""
                    INSERT INTO "historical_features_NBA" ({columns_str})
                    VALUES ({placeholders})
                """
                
                cur.execute(query, values)
                conn.commit()
                inserted_count += 1
                print(f"✓ Row {idx + 1}: Inserted - {game_id}")
                    
            except Exception as row_error:
                conn.rollback()
                error_count += 1
                print(f"❌ Row {idx + 1}: Error - {str(row_error)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✓ Processing complete:")
        print(f"  - Inserted: {inserted_count}")
        print(f"  - Skipped: {skipped_count}")
        print(f"  - Errors: {error_count}")
        print(f"{'='*60}")
        
        return inserted_count
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise
    finally:
        cur.close()
        conn.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*80)
    print("STORING HISTORICAL FEATURES TO DATABASE (INSERT ONLY)")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Database: {DB_NAME} @ {DB_HOST}")
    
    try:
        # Load data
        print("\n[1/4] Loading data...")
        df = load_data()
        
        # Prepare data
        df_clean = prepare_data_for_db(df)
        
        # Test connection
        print("\n[2/4] Testing database connection...")
        conn = get_db_connection()
        conn.close()
        print("✓ Database connection successful")
        
        # Insert data
        rows_processed = insert_data_to_db(df_clean)
        
        print("\n[4/4] Finalizing...")
        print("\n" + "="*80)
        print("✅ COMPLETE - Historical features stored successfully")
        print("="*80)
        print(f"Records processed: {rows_processed}")
        print(f"Table: historical_features_NBA")
        print(f"Timestamp: {datetime.now().isoformat()}\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ FAILED - {str(e)}")
        print("="*80 + "\n")
        raise

if __name__ == "__main__":
    main()
