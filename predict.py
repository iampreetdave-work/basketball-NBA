import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

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

print("="*80)
print("NBA PREDICTIONS - MONEYLINE + OVER/UNDER + SPREAD")
print("="*80)

# ============================================================================
# 1. LOAD TRAINED MODELS
# ============================================================================
print("\n[1/5] Loading trained models...")

model_dir = './model'

try:
    with open(f'{model_dir}/hybrid_home_xgb.pkl', 'rb') as f:
        home_model = pickle.load(f)

    with open(f'{model_dir}/hybrid_away_xgb.pkl', 'rb') as f:
        away_model = pickle.load(f)

    with open(f'{model_dir}/hybrid_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    print("  ✓ All models loaded successfully")

except FileNotFoundError as e:
    print(f"  ❌ Error: Could not find model files in '{model_dir}'")
    print(f"  Missing file: {e.filename}")
    exit()

# ============================================================================
# 2. LOAD DATA
# ============================================================================
print("\n[2/5] Loading data...")

try:
    df = pd.read_csv('Future.csv', on_bad_lines='skip')
except:
    try:
        df = pd.read_csv('/content/NBANBA.csv', on_bad_lines='skip')
    except:
        df = pd.read_csv('NBANBA__1_.csv')

print(f"  ✓ Loaded {len(df)} total games")

if 'home_insufficient_data' in df.columns and 'away_insufficient_data' in df.columns:
    df = df[(df['home_insufficient_data'] == False) & (df['away_insufficient_data'] == False)].copy()
    print(f"  ✓ Filtered by data sufficiency flags")

critical_cols = ['home_recent_ppg', 'away_recent_ppg',
                 'home_winning_odds_decimal', 'away_winning_odds_decimal', 'total_line_o']
df = df.dropna(subset=critical_cols).copy()
df = df.reset_index(drop=True)
cols_to_convert = [
    'home_recent_points', 'home_recent_field_goals_pct', 'home_recent_three_points_pct',
    'home_recent_free_throws_pct', 'home_recent_assists', 'home_recent_steals',
    'home_recent_blocks', 'home_recent_offensive_rebounds', 'home_recent_defensive_rebounds',
    'home_recent_wins', 'home_recent_losses', 'home_recent_win_pct', 'home_recent_ppg',
    'home_recent_opp_ppg', 'home_recent_point_diff', 'home_recent_scoring_trend',
    'away_recent_points', 'away_recent_field_goals_pct', 'away_recent_three_points_pct',
    'away_recent_free_throws_pct', 'away_recent_assists', 'away_recent_steals',
    'away_recent_blocks', 'away_recent_offensive_rebounds', 'away_recent_defensive_rebounds',
    'away_recent_wins', 'away_recent_losses', 'away_recent_win_pct', 'away_recent_ppg',
    'away_recent_opp_ppg', 'away_recent_point_diff', 'away_recent_scoring_trend',
    'scoring_advantage_home', 'form_advantage_home', 'defensive_advantage_home',
    'ball_control_advantage_home', 'home_games_played', 'away_games_played', 'total_line_o'
]

for col in cols_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print(f"  ✓ {len(df)} games ready for prediction")

# ============================================================================
# 3. CREATE FEATURES
# ============================================================================
print("\n[3/5] Creating features...")

def create_defense_features(df_input):
    """Defense features - same as training"""
    df = df_input.copy()

    h_ppg = df['home_recent_ppg'].values
    a_ppg = df['away_recent_ppg'].values

    features = {
        'ppg_diff': h_ppg - a_ppg,
        'ppg_sum': h_ppg + a_ppg,
        'net_rating_diff': (h_ppg - df['home_recent_opp_ppg']) - (a_ppg - df['away_recent_opp_ppg']),
        'win_pct_diff': df['home_recent_win_pct'] - df['away_recent_win_pct'],
        'implied_home_prob': 1 / df['home_winning_odds_decimal'],
        'steals_diff': df['home_recent_steals'] - df['away_recent_steals'],
        'blocks_diff': df['home_recent_blocks'] - df['away_recent_blocks'],
        'defense_diff': (df['home_recent_steals'] - df['away_recent_steals']) +
                       (df['home_recent_blocks'] - df['away_recent_blocks']),
        'line_bias': df['total_line_o'] - (h_ppg + a_ppg)
    }

    feat_df = pd.DataFrame(features)
    feat_df = feat_df.fillna(0).replace([np.inf, -np.inf], 0)
    return feat_df.values

X_all = create_defense_features(df)
X_all_scaled = scaler.transform(X_all)

print(f"  ✓ Features created and scaled")

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================
print("\n[4/5] Making predictions...")

pred_home = home_model.predict(X_all_scaled)
pred_away = away_model.predict(X_all_scaled)

# ============================================================================
# HANDLE TIES: Add random 0-7 to the team with better odds (lower odds decimal)
# ============================================================================
tie_mask = np.abs(pred_home - pred_away) < 0.01  # Detect near-ties
tie_indices = np.where(tie_mask)[0]

if len(tie_indices) > 0:
    print(f"  ⚠️  Detected {len(tie_indices)} tie predictions, applying random adjustment...")

    for idx in tie_indices:
        home_odds = df.loc[idx, 'home_winning_odds_decimal']
        away_odds = df.loc[idx, 'away_winning_odds_decimal']
        random_points = np.random.uniform(0, 7)

        # Add random points to the team with better odds (lower odds decimal)
        if home_odds < away_odds:
            pred_home[idx] += random_points
        else:
            pred_away[idx] += random_points

    print(f"  ✓ Tie adjustments applied to {len(tie_indices)} games")

pred_total = pred_home + pred_away
pred_winner = (pred_home > pred_away).astype(int)
pred_margin = pred_home - pred_away
pred_confidence = np.tanh(np.abs(pred_margin) / 5) * 100

print(f"  ✓ Predictions complete for {len(df)} games")

# ============================================================================
# 5. CREATE RESULTS DATAFRAME
# ============================================================================
print("\n[5/5] Compiling results...")

results_df = pd.DataFrame()

results_df['id'] = range(1, len(df) + 1)

# Generate game_identifier
if 'game_identifier' in df.columns:
    results_df['game_identifier'] = df['game_identifier'].values
else:
    results_df['game_identifier'] = results_df['id'].astype(str) + '_' + (df['game_date'].astype(str) if 'game_date' in df.columns else pd.Series(index=df.index, dtype=str))

# Generate team IDs as league_teamalias format
league = (df['league'].iloc[0] if 'league' in df.columns else 'NBA').lower()

home_team_ids = []
away_team_ids = []
home_teams = df['home_name'].values if 'home_name' in df.columns else df['home_alias'].values
away_teams = df['away_name'].values if 'away_name' in df.columns else df['away_alias'].values

for home, away in zip(home_teams, away_teams):
    home_alias = TEAM_ALIASES.get(home, home.replace(' ', '_').lower()[:3])
    away_alias = TEAM_ALIASES.get(away, away.replace(' ', '_').lower()[:3])

    home_team_ids.append(f"{league}_{home_alias.lower()}")
    away_team_ids.append(f"{league}_{away_alias.lower()}")

results_df['home_id'] = home_team_ids
results_df['away_id'] = away_team_ids

# Date and league
if 'game_date' in df.columns:
    results_df['date'] = df['game_date'].values
elif 'date' in df.columns:
    results_df['date'] = df['date'].values
elif 'scheduled' in df.columns:
    results_df['date'] = df['scheduled'].values
else:
    results_df['date'] = pd.NaT

results_df['league'] = league.upper()

# Team names
results_df['home_team'] = home_teams
results_df['away_team'] = away_teams

# Points predictions
results_df['home_points_predicted'] = pred_home.round().astype(int)
results_df['away_points_predicted'] = pred_away.round().astype(int)
results_df['total_points_predicted'] = pred_total.round().astype(int)

# Actual results (null initially, filled during validation)
results_df['home_points_actual'] = None
results_df['away_points_actual'] = None
results_df['total_points_actual'] = None

# ML prediction
results_df['ml_prediction'] = ['Home Win' if x == 1 else 'Away Win' for x in pred_winner]
results_df['ml_actual'] = None

# ML probability
ml_prob = 1 / (1 + np.exp(-pred_margin / 5))
results_df['ml_probability'] = ml_prob.round(4)

# Over/Under prediction
results_df['ou_predicted'] = np.where(
    (results_df['total_points_predicted'].notna()) & (df['total_line_o'].notna()),
    np.where(results_df['total_points_predicted'] > df['total_line_o'].values, 'OVER', 'UNDER'),
    None
)

# Odds
results_df['home_win_odds'] = df['home_winning_odds_decimal'].values.round(2) if 'home_winning_odds_decimal' in df.columns else 0.0
results_df['away_win_odds'] = df['away_winning_odds_decimal'].values.round(2) if 'away_winning_odds_decimal' in df.columns else 0.0

# Spread lines and odds from Future.csv
results_df['home_spread'] = df['home_spread'].values if 'home_spread' in df.columns else None
results_df['away_spread'] = df['away_spread'].values if 'away_spread' in df.columns else None
results_df['home_spread_odds_decimal'] = df['home_spread_odds_decimal'].values.round(2) if 'home_spread_odds_decimal' in df.columns else None
results_df['away_spread_odds_decimal'] = df['away_spread_odds_decimal'].values.round(2) if 'away_spread_odds_decimal' in df.columns else None
results_df['total_line_o'] = df['total_line_o'].values if 'total_line_o' in df.columns else None
results_df['total_line_over_odds_decimal'] = df['total_line_over_odds_decimal'].values.round(2) if 'total_line_over_odds_decimal' in df.columns else None
results_df['total_line_under_odds_decimal'] = df['total_line_under_odds_decimal'].values.round(2) if 'total_line_under_odds_decimal' in df.columns else None

# ML correct / PnL (null initially)
results_df['ml_correct'] = None
results_df['ml_pnl'] = None
results_df['ou_correct'] = None
results_df['ou_pnl'] = None

# Confidence
results_df['ml_confidence'] = pred_confidence.round(2)

# Status
results_df['status'] = 'PENDING'

# ============================================================================
# GRADING LOGIC
# ============================================================================
def assign_grade(confidence, grade_type='ml'):
    if grade_type == 'ml':
        if confidence >= 85:
            return 'B'
        elif confidence >= 69:
            return 'D'
        elif confidence >= 52:
            return 'A'
        else:
            return 'C'
    elif grade_type == 'ou':
        if confidence >= 80:
            return 'B'
        elif confidence >= 40 and confidence < 54:
            return 'A'
        elif confidence >= 30 and confidence < 45:
            return 'C'
        else:
            return 'D'
    return 'D'

results_df['grade'] = results_df['ml_confidence'].apply(lambda x: assign_grade(x, 'ml'))
results_df['ou_grade'] = results_df['ml_confidence'].apply(lambda x: assign_grade(x, 'ou'))

# ============================================================================
# SPREAD PREDICTION — ONE SIMPLE INDICATOR
# ============================================================================
# spread_covered_predicted: does the model predict the HOME team covers?
#   TRUE  = home team predicted to cover (home_pts_predicted + home_spread > away_pts_predicted)
#   FALSE = away team predicted to cover
#
# This is the ONLY spread prediction column. No separate home/away columns.

def calculate_spread_covered_predicted(row):
    """
    Does the model predict the home team covers the spread?

    Formula: home_points_predicted + home_spread > away_points_predicted

    This is equivalent to: predicted_margin > -home_spread

    Examples:
        home_spread = -5.5 (home favored by 5.5)
        pred_home=110, pred_away=105, margin=5
        110 + (-5.5) = 104.5, 104.5 > 105? NO → FALSE (home doesn't cover)

        home_spread = +8.5 (home underdog by 8.5)
        pred_home=100, pred_away=106, margin=-6
        100 + 8.5 = 108.5, 108.5 > 106? YES → TRUE (home covers)
    """
    if pd.isna(row['home_spread']):
        return None

    home_adj = row['home_points_predicted'] + float(row['home_spread'])
    if home_adj > row['away_points_predicted']:
        return 'TRUE'
    elif home_adj < row['away_points_predicted']:
        return 'FALSE'
    else:
        return 'PUSH'

results_df['spread_covered_predicted'] = results_df.apply(calculate_spread_covered_predicted, axis=1)

# Derive home/away booleans from the single indicator
# home_spread_covered_predicted = TRUE means home covers
# away_spread_covered_predicted = TRUE means away covers (i.e. home did NOT cover)
results_df['home_spread_covered_predicted'] = results_df['spread_covered_predicted'].apply(
    lambda x: True if x == 'TRUE' else (False if x == 'FALSE' else None)
)
results_df['away_spread_covered_predicted'] = results_df['spread_covered_predicted'].apply(
    lambda x: True if x == 'FALSE' else (False if x == 'TRUE' else None)
)

# ============================================================================
# SPREAD GRADE
# ============================================================================
def calculate_spread_grade(row):
    confidence = row['ml_confidence']
    spread_pred = row['spread_covered_predicted']

    if confidence >= 85:
        ml_grade = 'B'
    elif confidence >= 69:
        ml_grade = 'D'
    elif confidence >= 52:
        ml_grade = 'A'
    else:
        ml_grade = 'C'

    if pd.isna(spread_pred) or spread_pred is None:
        return None

    pred_home_covers = str(spread_pred).upper() == 'TRUE'

    if pred_home_covers:
        return 'A' if ml_grade in ['A', 'B'] else 'B'
    else:
        return 'C' if ml_grade in ['A', 'B'] else 'D'

results_df['spread_grade'] = results_df.apply(calculate_spread_grade, axis=1)

# Add market_total_line
results_df['market_total_line'] = df['total_line_o'].values if 'total_line_o' in df.columns else None

# Rename odds columns
results_df.rename(columns={
    'home_spread_odds_decimal': 'home_spread_odds',
    'away_spread_odds_decimal': 'away_spread_odds',
    'total_line_over_odds_decimal': 'over_odds',
    'total_line_under_odds_decimal': 'under_odds'
}, inplace=True)

# ============================================================================
# FINAL COLUMN ORDER — clean, no home/away split for spread coverage
# ============================================================================
final_columns = [
    'id', 'date', 'league', 'game_identifier', 'home_id', 'home_team', 'away_id', 'away_team',
    'home_points_predicted', 'home_points_actual',
    'away_points_predicted', 'away_points_actual',
    'total_points_predicted', 'total_points_actual',
    'ml_prediction', 'ml_actual', 'ml_probability',
    'home_win_odds', 'away_win_odds',
    'ml_correct', 'ml_pnl',
    'ml_confidence', 'grade', 'ou_grade', 'spread_grade', 'status',
    'market_total_line',
    'ou_predicted', 'ou_correct', 'ou_pnl',
    'home_spread', 'away_spread',
    'home_spread_odds', 'away_spread_odds',
    'over_odds', 'under_odds',
    'spread_covered_predicted',
    'home_spread_covered_predicted', 'away_spread_covered_predicted',
]

results_df = results_df[final_columns]

# Save to CSV
output_file = 'NBA_PREDICTIONS_ML.csv'
results_df.to_csv(output_file, index=False)

print(f"  ✓ Results saved to '{output_file}'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PREDICTION SUMMARY")
print("="*80)

print(f"\n  MONEYLINE: Pending validation")
print(f"  OVER/UNDER: Pending validation")
print(f"  SPREAD: Pending validation")

print(f"\n  Total Games:          {len(results_df)}")
print(f"  Avg ML Confidence:    {results_df['ml_confidence'].mean():.1f}%")
print(f"  Output Columns:       {len(results_df.columns)}")

print("\n" + "="*80)
print(f"COMPLETE - Predictions saved to '{output_file}'")
print("="*80)

# Show sample predictions
print("\nSAMPLE PREDICTIONS (first 10 games):")
print("-"*80)
display_cols = ['home_team', 'away_team', 'home_points_predicted', 'away_points_predicted',
                'total_points_predicted', 'ml_prediction', 'ml_probability', 'ml_confidence',
                'grade', 'ou_grade', 'spread_grade', 'ou_predicted', 'home_spread', 'away_spread',
                'spread_covered_predicted', 'status']

print(results_df[display_cols].head(10).to_string(index=False))
print("-"*80)
