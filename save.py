import psycopg2
import pandas as pd
from psycopg2 import sql

# Database configuration
DB_CONFIG = {
    'host': 'winbets-predictions.postgres.database.azure.com',
    'port': 5432,
    'database': 'postgres',
    'user': 'winbets',
    'password': 'Constantinople@1900'
}

TABLE_NAME = 'agility_NBA_b1'
CSV_FILE = 'NBA_PREDICTIONS_ML.csv'

# Columns to extract and push
COLUMNS_TO_PUSH = [
    'date',
    'league',
    'game_identifier',
    'home_id',
    'home_team',
    'away_id',
    'away_team',
    'home_points_predicted',
    'away_points_predicted',
    'total_points_predicted',
    'ml_prediction',
    'ml_probability',
    'home_win_odds',
    'away_win_odds',
    'ml_confidence'
]

def get_existing_game_identifiers(connection):
    """Fetch all existing game_identifiers from database"""
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SELECT DISTINCT game_identifier FROM {};").format(
                sql.Identifier(TABLE_NAME)
            ))
            existing = {row[0] for row in cursor.fetchall()}
            print(f"✓ Found {len(existing)} existing game identifiers in database")
            return existing
    except psycopg2.Error as e:
        print(f"✗ Database error while fetching existing records: {e}")
        raise

def push_data():
    """Read CSV and push only new rows (by game_identifier) to database"""
    try:
        # Read CSV
        print(f"Reading {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        print(f"✓ Loaded {len(df)} rows from CSV")
        
        # Select only required columns
        df = df[COLUMNS_TO_PUSH]
        print(f"✓ Selected {len(COLUMNS_TO_PUSH)} columns")
        
        # Connect to database
        print("Connecting to PostgreSQL...")
        connection = psycopg2.connect(**DB_CONFIG)
        print("✓ Connected to database")
        
        # Get existing game identifiers
        existing_identifiers = get_existing_game_identifiers(connection)
        
        # Filter for new rows only
        new_rows = df[~df['game_identifier'].isin(existing_identifiers)]
        
        if len(new_rows) == 0:
            print("✓ No new data found. CSV data matches database.")
            connection.close()
            return
        
        print(f"✓ Found {len(new_rows)} new rows to insert (out of {len(df)} total)")
        
        # Insert new data only
        with connection.cursor() as cursor:
            for index, row in new_rows.iterrows():
                # Build dynamic INSERT query with column names
                columns = sql.SQL(', ').join([sql.Identifier(col) for col in COLUMNS_TO_PUSH])
                placeholders = ', '.join(['%s'] * len(COLUMNS_TO_PUSH))
                
                insert_query = sql.SQL(
                    "INSERT INTO {} ({}) VALUES ({})"
                ).format(
                    sql.Identifier(TABLE_NAME),
                    columns,
                    sql.SQL(placeholders)
                )
                
                # Handle NaN values as None for NULL insertion, preserving column order
                values = tuple(
                    None if pd.isna(row[col]) else row[col]
                    for col in COLUMNS_TO_PUSH
                )
                
                cursor.execute(insert_query, values)
        
        connection.commit()
        print(f"✓ Inserted {len(new_rows)} new rows into '{TABLE_NAME}'")
        
        # Verify
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(
                sql.Identifier(TABLE_NAME)
            ))
            count = cursor.fetchone()[0]
            print(f"✓ Verification: {count} total rows in {TABLE_NAME}")
        
        connection.close()
        print("✓ Database connection closed")
        print(f"\n✓ Success! {len(new_rows)} new rows pushed to {TABLE_NAME}")
        
    except FileNotFoundError:
        print(f"✗ Error: {CSV_FILE} not found")
        raise
    except KeyError as e:
        print(f"✗ Error: Column {e} not found in CSV")
        raise
    except psycopg2.Error as e:
        print(f"✗ Database error: {e}")
        raise
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        raise

if __name__ == "__main__":
    push_data()
