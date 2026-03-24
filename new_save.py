import os
import psycopg2
import pandas as pd
from psycopg2 import sql

# Database configuration from environment variables (GitHub Secrets)
DB_CONFIG = {
    'host': os.environ.get('DB_HOST'),
    'port': int(os.environ.get('DB_PORT', 5432)),
    'database': os.environ.get('DB_DATABASE'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD')
}

TABLE_NAME = 'predictions_nba_b1_ourmodel'
CSV_FILE = 'NBA_PREDICTIONS_ML.csv'

# All columns from generated CSV (excluding id)
# Matches the output of the fixed predict.py — single spread_covered_predicted column
CSV_COLUMNS = [
    'date',
    'league',
    'game_identifier',
    'home_id',
    'home_team',
    'away_id',
    'away_team',
    'home_points_predicted',
    'home_points_actual',
    'away_points_predicted',
    'away_points_actual',
    'total_points_predicted',
    'total_points_actual',
    'ml_prediction',
    'ml_actual',
    'ml_probability',
    'home_win_odds',
    'away_win_odds',
    'ml_correct',
    'ml_pnl',
    'ml_confidence',
    'grade',
    'ou_grade',
    'spread_grade',
    'status',
    'market_total_line',
    'ou_predicted',
    'ou_correct',
    'ou_pnl',
    'home_spread',
    'away_spread',
    'home_spread_odds',
    'away_spread_odds',
    'over_odds',
    'under_odds',
    'spread_covered_predicted',
    'home_spread_covered_predicted',
    'away_spread_covered_predicted',
]

COLUMN_MAPPING = {}

def push_data():
    """Read CSV and push to database — insert new rows, update existing ones by game_identifier"""
    try:
        # Validate credentials are loaded
        if not all(DB_CONFIG.values()):
            raise ValueError("Missing database credentials in environment variables")

        # Read CSV
        print(f"Reading {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        print(f"✓ Loaded {len(df)} rows from CSV")

        # Select only required columns
        df = df[CSV_COLUMNS]
        print(f"✓ Selected {len(CSV_COLUMNS)} columns")

        # Connect to database
        print("Connecting to PostgreSQL...")
        connection = psycopg2.connect(**DB_CONFIG)
        print("✓ Connected to database")

        # Build queries once
        db_columns = [COLUMN_MAPPING.get(col, col) for col in CSV_COLUMNS]

        # UPDATE query: SET all columns except game_identifier, WHERE game_identifier = %s
        update_columns = [c for c in db_columns if c != 'game_identifier']
        update_set = ', '.join(f"{col} = %s" for col in update_columns)
        update_query = f"UPDATE {TABLE_NAME} SET {update_set} WHERE game_identifier = %s"

        # INSERT query
        columns_str = ', '.join(db_columns)
        placeholders = ', '.join(['%s'] * len(CSV_COLUMNS))
        insert_query = f"INSERT INTO {TABLE_NAME} ({columns_str}) VALUES ({placeholders})"

        inserted_count = 0
        updated_count = 0

        with connection.cursor() as cursor:
            for index, row in df.iterrows():
                game_id = row['game_identifier']

                # Handle NaN values as None
                values = {
                    col: None if pd.isna(row[col]) else row[col]
                    for col in CSV_COLUMNS
                }

                # Check if game_identifier exists
                cursor.execute(
                    f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE game_identifier = %s",
                    (game_id,)
                )
                exists = cursor.fetchone()[0] > 0

                if exists:
                    # UPDATE: values for SET columns + game_identifier for WHERE clause
                    update_values = tuple(
                        values[col] for col in CSV_COLUMNS if col != 'game_identifier'
                    ) + (game_id,)
                    cursor.execute(update_query, update_values)
                    print(f"  ↻ Updated: {game_id}")
                    updated_count += 1
                else:
                    # INSERT: all columns
                    insert_values = tuple(values[col] for col in CSV_COLUMNS)
                    cursor.execute(insert_query, insert_values)
                    print(f"  ✓ Inserted: {game_id}")
                    inserted_count += 1

        connection.commit()
        print(f"\n✓ Inserted {inserted_count} new rows into '{TABLE_NAME}'")
        print(f"✓ Updated {updated_count} existing rows in '{TABLE_NAME}'")

        # Verify
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
            count = cursor.fetchone()[0]
            print(f"✓ Verification: {count} total rows in {TABLE_NAME}")

        connection.close()
        print("✓ Database connection closed")
        print(f"\n✓ Success! Data pushed to {TABLE_NAME}")

    except FileNotFoundError:
        print(f"✗ Error: {CSV_FILE} not found")
        raise
    except KeyError as e:
        print(f"✗ Error: Column {e} not found in CSV")
        raise
    except psycopg2.Error as e:
        print(f"✗ Database error: {e}")
        if 'connection' in locals():
            connection.rollback()
        raise
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        raise

if __name__ == "__main__":
    push_data()
