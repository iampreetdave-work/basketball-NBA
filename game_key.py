import psycopg2
from psycopg2 import sql
from datetime import datetime
import os

# Database connection credentials
DB_CONFIG = {
    'host': os.environ.get('DB_HOST'),
    'port': int(os.environ.get('DB_PORT', 5432)),
    'database': os.environ.get('DB_DATABASE'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD')
}
TABLE_NAME = 'agility_nba_b1'

# Team name to abbreviation mapping
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

def get_team_abbreviation(team_name):
    """Get team abbreviation from full team name."""
    abbr = TEAM_ALIASES.get(team_name)
    if not abbr:
        print(f"⚠️  Warning: No abbreviation found for team '{team_name}'")
        return None
    return abbr

def get_unique_gamekey(cursor, base_gamekey, existing_gamekeys):
    """
    Generate a unique gamekey by checking if it exists and appending _2, _3, etc.
    Starting from _2 for the first duplicate.
    """
    gamekey = base_gamekey
    
    # Check if base gamekey exists
    if gamekey not in existing_gamekeys:
        return gamekey
    
    # If exists, start from _2
    counter = 2
    while True:
        gamekey = f"{base_gamekey}_{counter}"
        if gamekey not in existing_gamekeys:
            return gamekey
        counter += 1

def generate_gamekeys():
    """
    Generate gamekey values for rows where gamekey is NULL.
    Format: YYYYAWAYHOME (e.g., 2026CLEIND)
    """
    conn = None
    cursor = None
    
    try:
        # Get current year dynamically
        current_year = datetime.now().year
        print(f"Using current year: {current_year}")
        
        # Connect to the database
        print("\nConnecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✓ Connected successfully!")
        
        # First, get all existing gamekeys to check for duplicates
        print("\nFetching existing gamekeys...")
        cursor.execute(f"""
            SELECT gamekey 
            FROM {TABLE_NAME} 
            WHERE gamekey IS NOT NULL;
        """)
        existing_gamekeys = set([row[0] for row in cursor.fetchall()])
        print(f"✓ Found {len(existing_gamekeys)} existing gamekeys")
        
        # Get all rows where gamekey is NULL
        print("\nFetching rows with NULL gamekey...")
        cursor.execute(f"""
            SELECT date, home_team, away_team, game_identifier
            FROM {TABLE_NAME} 
            WHERE gamekey IS NULL
            ORDER BY date;
        """)
        
        null_rows = cursor.fetchall()
        total_rows = len(null_rows)
        
        if total_rows == 0:
            print("✓ No rows with NULL gamekey found. Nothing to update!")
            return
        
        print(f"✓ Found {total_rows} rows with NULL gamekey")
        print("\n" + "="*80)
        print("Starting gamekey generation...")
        print("="*80)
        
        updated_count = 0
        skipped_count = 0
        
        for idx, row in enumerate(null_rows, 1):
            date_val, home_team, away_team, game_identifier = row
            
            # Get abbreviations
            away_abbr = get_team_abbreviation(away_team)
            home_abbr = get_team_abbreviation(home_team)
            
            if not away_abbr or not home_abbr:
                print(f"\n[{idx}/{total_rows}] ❌ Skipping - Missing abbreviation")
                print(f"  Game: {away_team} @ {home_team}")
                skipped_count += 1
                continue
            
            # Generate base gamekey
            base_gamekey = f"{current_year}{away_abbr}{home_abbr}"
            
            # Get unique gamekey (handles duplicates)
            unique_gamekey = get_unique_gamekey(cursor, base_gamekey, existing_gamekeys)
            
            # Update the row
            cursor.execute(f"""
                UPDATE {TABLE_NAME} 
                SET gamekey = %s 
                WHERE game_identifier = %s AND gamekey IS NULL;
            """, (unique_gamekey, game_identifier))
            
            # Add to existing gamekeys set
            existing_gamekeys.add(unique_gamekey)
            
            # Show progress
            suffix = "" if unique_gamekey == base_gamekey else f" (duplicate handled)"
            print(f"[{idx}/{total_rows}] ✓ {away_team} @ {home_team}")
            print(f"          → gamekey: {unique_gamekey}{suffix}")
            
            updated_count += 1
        
        # Commit all changes
        conn.commit()
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"✓ Total rows processed: {total_rows}")
        print(f"✓ Successfully updated: {updated_count}")
        if skipped_count > 0:
            print(f"⚠️  Skipped (missing abbreviation): {skipped_count}")
        print(f"✓ All changes committed to database!")
        
    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        if conn:
            conn.rollback()
            print("Changes rolled back.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if conn:
            conn.rollback()
            print("Changes rolled back.")
    finally:
        # Close connections
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    print("="*80)
    print(f"GAMEKEY GENERATOR FOR {TABLE_NAME}")
    print("="*80)
    generate_gamekeys()
    print("="*80)
    print("Script completed.")
