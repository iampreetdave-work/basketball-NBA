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
print("NBA PREDICTIONS - USING TRAINED HYBRID MODEL")
print("="*80)

# ============================================================================
# 1. LOAD TRAINED MODELS
# ============================================================================
print("\n[1/5] Loading trained models...")

model_dir = './'  # Change this if your models are in a different directory

try:
    # Load HOME ensemble models
    with open(f'{model_dir}hybrid_home_xgb.pkl', 'rb') as f:
        h_xgb = pickle.load(f)
    with open(f'{model_dir}hybrid_home_lgb.pkl', 'rb') as f:
        h_lgb = pickle.load(f)
    with open(f'{model_dir}hybrid_home_gb.pkl', 'rb') as f:
        h_gb = pickle.load(f)
    with open(f'{model_dir}hybrid_home_rf.pkl', 'rb') as f:
        h_rf = pickle.load(f)
    
    # Load HOME weights
    with open(f'{model_dir}hybrid_home_weights.pkl', 'rb') as f:
        h_weights = pickle.load(f)
    
    # Load AWAY model
    with open(f'{model_dir}hybrid_away_xgb.pkl', 'rb') as f:
        away_model = pickle.load(f)
    
    # Load scaler
    with open(f'{model_dir}hybrid_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("  ✓ All models loaded successfully")
    print(f"  ✓ HOME weights: {', '.join([f'{k}={v:.3f}' for k, v in h_weights.items()])}")
    
except FileNotFoundError as e:
    print(f"  ❌ Error: Could not find model files in '{model_dir}'")
    print(f"  Please make sure you've trained the models first and they're in the correct directory")
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

# FIXED: Only filter by insufficient_data columns if they exist
# For live prediction data, these columns typically won't be present
if 'home_insufficient_data' in df.columns and 'away_insufficient_data' in df.columns:
    df = df[(df['home_insufficient_data'] == False) & (df['away_insufficient_data'] == False)].copy()
    print(f"  ✓ Filtered by data sufficiency flags")

critical_cols = ['home_recent_ppg', 'away_recent_ppg', 
                 'home_winning_odds_decimal', 'away_winning_odds_decimal', 'total_line']
df = df.dropna(subset=critical_cols).copy()

# Convert to numeric
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
    'ball_control_advantage_home', 'home_games_played', 'away_games_played'
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
        'line_bias': df['total_line'] - (h_ppg + a_ppg),
        'steals_diff': df['home_recent_steals'] - df['away_recent_steals'],
        'blocks_diff': df['home_recent_blocks'] - df['away_recent_blocks'],
        'defense_diff': (df['home_recent_steals'] - df['away_recent_steals']) + 
                       (df['home_recent_blocks'] - df['away_recent_blocks'])
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

# HOME predictions (ensemble)
h_models = {'XGB': h_xgb, 'LGB': h_lgb, 'GB': h_gb, 'RF': h_rf}
pred_home = np.zeros(len(X_all_scaled))

for name, model in h_models.items():
    pred_home += h_weights[name] * model.predict(X_all_scaled)

# AWAY predictions (single XGB)
pred_away = away_model.predict(X_all_scaled)

# Derived predictions
pred_total = pred_home + pred_away
pred_winner = (pred_home > pred_away).astype(int)  # 1 = home wins, 0 = away wins
pred_margin = pred_home - pred_away
pred_confidence = np.tanh(np.abs(pred_margin) / 5) * 100

print(f"  ✓ Predictions complete for {len(df)} games")

# ============================================================================
# 5. CREATE RESULTS DATAFRAME
# ============================================================================
print("\n[5/5] Compiling results...")

# Extract key columns from original data
results_df = pd.DataFrame()

# Pre-match feature columns (all available from input data)
prematch_columns = [
    'match_id', 'game_identifier', 'scheduled', 'status', 'venue_name', 'venue_city', 'league',
    'home_id', 'home_name', 'home_alias', 'home_market',
    'away_id', 'away_name', 'away_alias', 'away_market',
    'home_games_played', 'away_games_played',
    'home_recent_points', 'home_recent_field_goals_pct', 'home_recent_three_points_pct',
    'home_recent_free_throws_pct', 'home_recent_rebounds', 'home_recent_assists',
    'home_recent_turnovers', 'home_recent_steals', 'home_recent_blocks',
    'home_recent_offensive_rebounds', 'home_recent_defensive_rebounds',
    'home_recent_wins', 'home_recent_losses', 'home_recent_win_pct',
    'home_recent_ppg', 'home_recent_opp_ppg', 'home_recent_point_diff',
    'home_recent_scoring_trend',
    'away_recent_points', 'away_recent_field_goals_pct', 'away_recent_three_points_pct',
    'away_recent_free_throws_pct', 'away_recent_rebounds', 'away_recent_assists',
    'away_recent_turnovers', 'away_recent_steals', 'away_recent_blocks',
    'away_recent_offensive_rebounds', 'away_recent_defensive_rebounds',
    'away_recent_wins', 'away_recent_losses', 'away_recent_win_pct',
    'away_recent_ppg', 'away_recent_opp_ppg', 'away_recent_point_diff',
    'away_recent_scoring_trend',
    'scoring_advantage_home', 'form_advantage_home', 'defensive_advantage_home',
    'ball_control_advantage_home'
]

# Add pre-match feature columns from input data
for col in prematch_columns:
    if col in df.columns:
        results_df[col] = df[col].values

# Add game_date at the beginning if available
if 'game_date' in df.columns:
    results_df.insert(0, 'game_date', df['game_date'].values)

# Add predictions
results_df['home_predicted'] = pred_home.round().astype(int)
results_df['away_predicted'] = pred_away.round().astype(int)
results_df['total_predicted'] = pred_total.round().astype(int)
results_df['moneyline_predicted'] = ['HOME WIN' if x == 1 else 'AWAY WIN' for x in pred_winner]
results_df['predicted_margin'] = pred_margin.round(1)
results_df['moneyline_confidence'] = pred_confidence.round().astype(int)

# Actual results (if available)
if 'home_points' in df.columns:
    results_df['home_actual'] = df['home_points'].values.astype(int)
    results_df['home_error'] = np.abs(pred_home - df['home_points'].values).round(1)
    results_df['home_within_5'] = (np.abs(pred_home - df['home_points'].values) <= 5).astype(int)

if 'away_points' in df.columns:
    results_df['away_actual'] = df['away_points'].values.astype(int)
    results_df['away_error'] = np.abs(pred_away - df['away_points'].values).round(1)
    results_df['away_within_5'] = (np.abs(pred_away - df['away_points'].values) <= 5).astype(int)

if 'home_points' in df.columns and 'away_points' in df.columns:
    actual_total = df['home_points'].values + df['away_points'].values
    results_df['total_actual'] = actual_total.astype(int)
    results_df['total_error'] = np.abs(pred_total - actual_total).round(1)
    results_df['total_within_10'] = (np.abs(pred_total - actual_total) <= 10).astype(int)
    
    actual_winner = (df['home_points'] > df['away_points']).astype(int)
    results_df['moneyline_actual'] = ['HOME WIN' if x == 1 else 'AWAY WIN' for x in actual_winner]
    results_df['moneyline_correct'] = (pred_winner == actual_winner).astype(int)

# Additional info (odds, lines if available)
if 'home_winning_odds_decimal' in df.columns:
    results_df['home_odds'] = df['home_winning_odds_decimal'].values.round(2)
if 'away_winning_odds_decimal' in df.columns:
    results_df['away_odds'] = df['away_winning_odds_decimal'].values.round(2)
if 'total_line' in df.columns:
    results_df['total_line'] = df['total_line'].values

# Save to CSV
output_file = 'NBA_PREDICTIONS.csv'
results_df.to_csv(output_file, index=False)

print(f"  ✓ Results saved to '{output_file}'")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("📊 PREDICTION SUMMARY")
print("="*80)

if 'home_error' in results_df.columns:
    print(f"\n🏀 HOME POINTS")
    print(f"  Average Error:        {results_df['home_error'].mean():.2f}")
    print(f"  Within ±5:            {results_df['home_within_5'].mean()*100:.2f}%")
    
if 'away_error' in results_df.columns:
    print(f"\n🚗 AWAY POINTS")
    print(f"  Average Error:        {results_df['away_error'].mean():.2f}")
    print(f"  Within ±5:            {results_df['away_within_5'].mean()*100:.2f}%")
    
if 'total_error' in results_df.columns:
    print(f"\n📊 TOTAL POINTS")
    print(f"  Average Error:        {results_df['total_error'].mean():.2f}")
    print(f"  Within ±10:           {results_df['total_within_10'].mean()*100:.2f}%")
    
if 'moneyline_correct' in results_df.columns:
    print(f"\n💰 WINNER PREDICTION")
    print(f"  Accuracy:             {results_df['moneyline_correct'].mean()*100:.2f}%")
    print(f"  Correct:              {results_df['moneyline_correct'].sum()}/{len(results_df)}")

print(f"\n📈 OVERALL")
print(f"  Total Games:          {len(results_df)}")
print(f"  Avg Confidence:       {results_df['moneyline_confidence'].mean():.1f}%")
print(f"  Output Columns:       {len(results_df.columns)}")

print("\n" + "="*80)
print(f"✅ COMPLETE - All predictions saved to '{output_file}'")
print("="*80)

# Show sample predictions
print("\n📋 SAMPLE PREDICTIONS (first 10 games):")
print("-"*80)
display_cols = ['home_predicted', 'away_predicted', 'total_predicted', 
                'moneyline_predicted', 'moneyline_confidence']
if 'home_team' in results_df.columns:
    display_cols = ['home_team', 'away_team'] + display_cols
if 'home_actual' in results_df.columns:
    display_cols += ['home_actual', 'away_actual']

print(results_df[display_cols].head(10).to_string(index=False))
print("-"*80)
