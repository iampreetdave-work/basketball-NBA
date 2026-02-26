import psycopg2
import pandas as pd
import os

# ============================================================================
# DATABASE CONFIG
# ============================================================================
DB_CONFIG = {
    'host': os.environ.get('DB_HOST'),
    'port': int(os.environ.get('DB_PORT', 5432)),
    'database': os.environ.get('DB_DATABASE'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD')
}

print("="*80)
print("NBA GRADES CALCULATION - ML, OU, SPREAD GRADES")
print("="*80)

# ============================================================================
# 1. FETCH RECORDS WITH NULL GRADES
# ============================================================================
print("\n[1/3] Fetching predictions with null grades from database...")

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Fetch all records where any grade is NULL
    fetch_query = """
        SELECT 
            game_identifier, ml_confidence, spread_covered_predicted
        FROM predictions_nba_b1_ourmodel
        WHERE grade IS NULL 
           OR ou_grade IS NULL 
           OR spread_grade IS NULL
        ORDER BY game_identifier ASC
    """
    
    cursor.execute(fetch_query)
    rows = cursor.fetchall()
    
    print(f"  ✓ Found {len(rows)} records with null grades")
    
    if len(rows) == 0:
        print("  ℹ️  No records to update. Exiting.")
        cursor.close()
        conn.close()
        exit()
    
    # Convert to DataFrame for easier processing
    grades_df = pd.DataFrame(
        rows,
        columns=['game_identifier', 'ml_confidence', 'spread_covered_predicted']
    )
    
except Exception as e:
    print(f"  ❌ Database fetch error: {e}")
    exit()

# ============================================================================
# 2. CALCULATE GRADES
# ============================================================================
print("\n[2/3] Calculating grades...")

def assign_ml_grade(confidence):
    """
    Assign ML grade based on confidence
    >= 85: B
    69-84.99: D
    52-68.99: A
    < 52: C
    """
    if confidence >= 85:
        return 'B'
    elif confidence >= 69:
        return 'D'
    elif confidence >= 52:
        return 'A'
    else:
        return 'C'

def assign_ou_grade(confidence):
    """
    Assign Over/Under grade based on confidence
    >= 80: B
    40-53.99: A
    30-44.99: C
    < 30: D
    """
    if confidence >= 80:
        return 'B'
    elif confidence >= 40 and confidence < 54:
        return 'A'
    elif confidence >= 30 and confidence < 45:
        return 'C'
    else:
        return 'D'

def assign_spread_grade(ml_grade, spread_covered_predicted):
    """
    Assign spread grade based on ML grade and spread coverage prediction
    
    if spread_covered_predicted == TRUE and grade in ['A', 'B']:
        spread_grade = 'A'
    elif spread_covered_predicted == TRUE and grade in ['C', 'D']:
        spread_grade = 'B'
    elif spread_covered_predicted == FALSE and grade in ['A', 'B']:
        spread_grade = 'C'
    else:  # FALSE and C/D
        spread_grade = 'D'
    """
    # Handle null spread_covered_predicted
    if pd.isna(spread_covered_predicted) or spread_covered_predicted is None:
        return None
    
    pred_is_true = str(spread_covered_predicted).upper() == 'TRUE'
    
    if pred_is_true:
        if ml_grade in ['A', 'B']:
            return 'A'
        else:  # C, D
            return 'B'
    else:
        if ml_grade in ['A', 'B']:
            return 'C'
        else:  # C, D
            return 'D'

# Calculate grades
grades_df['grade'] = grades_df['ml_confidence'].apply(assign_ml_grade)
grades_df['ou_grade'] = grades_df['ml_confidence'].apply(assign_ou_grade)
grades_df['spread_grade'] = grades_df.apply(
    lambda row: assign_spread_grade(row['grade'], row['spread_covered_predicted']),
    axis=1
)

print(f"  ✓ Grades calculated")
print(f"    - ML grades (grade):     {grades_df['grade'].value_counts().to_dict()}")
print(f"    - OU grades (ou_grade):  {grades_df['ou_grade'].value_counts().to_dict()}")
print(f"    - Spread grades:         {grades_df['spread_grade'].value_counts(dropna=False).to_dict()}")

# ============================================================================
# 3. UPDATE DATABASE
# ============================================================================
print("\n[3/3] Updating database with grades...")

try:
    update_count = 0
    error_count = 0
    
    for idx, row in grades_df.iterrows():
        try:
            update_query = """
                UPDATE predictions_nba_b1_ourmodel
                SET 
                    grade = %s,
                    ou_grade = %s,
                    spread_grade = %s
                WHERE game_identifier = %s
            """
            
            cursor.execute(update_query, (
                row['grade'],
                row['ou_grade'],
                row['spread_grade'],
                row['game_identifier']
            ))
            
            update_count += 1
            
        except Exception as e:
            print(f"  ⚠️  Error updating game_identifier {row['game_identifier']}: {e}")
            error_count += 1
            conn.rollback()
            continue
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"  ✓ Database updated: {update_count} records updated, {error_count} errors")
    
except Exception as e:
    print(f"  ❌ Database update error: {e}")
    exit()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ GRADES CALCULATION COMPLETE")
print("="*80)
print(f"\nTotal records processed:    {len(grades_df)}")
print(f"Records updated:            {update_count}")
print(f"Errors:                     {error_count}")
print("\nGrade Distribution:")
print(f"  ML Grades:      {grades_df['grade'].value_counts().to_dict()}")
print(f"  OU Grades:      {grades_df['ou_grade'].value_counts().to_dict()}")
print(f"  Spread Grades:  {grades_df['spread_grade'].value_counts(dropna=False).to_dict()}")

print("\n" + "="*80)

# Display sample records
print("\n📋 SAMPLE RECORDS WITH GRADES (first 10):")
print("-"*80)
display_cols = ['game_identifier', 'ml_confidence', 'grade', 'ou_grade', 'spread_grade', 'spread_covered_predicted']
print(grades_df[display_cols].head(10).to_string(index=False))
print("-"*80)
