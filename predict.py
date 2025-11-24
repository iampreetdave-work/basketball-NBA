import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBRegressor
import pickle
import warnings
import time

warnings.filterwarnings('ignore')

print("="*80)
print("NBA HYBRID MODEL - SINGLE MODEL ARCHITECTURE")
print("="*80)
print("HOME: Single XGB (1000/0.009/7/0.9/0.9) → ~37.07%")
print("AWAY: Single XGB (600/0.009/6/0.85/0.85) → ~41.22%")
print("="*80)

start_time = time.time()

# ============================================================================
# 1. LOAD & CLEAN
# ============================================================================
print("\n[1/8] Loading data...")

try:
    df = pd.read_csv('/content/NBANBA.csv', on_bad_lines='skip')
except:
    try:
        df = pd.read_csv('NBANBA.csv', on_bad_lines='skip')
    except:
        df = pd.read_csv('NBANBA__1_.csv')

df = df[(df['home_insufficient_data'] == False) & (df['away_insufficient_data'] == False)].copy()

critical_cols = ['home_points', 'away_points', 'home_recent_ppg', 'away_recent_ppg', 
                 'home_winning_odds_decimal', 'away_winning_odds_decimal', 'total_line']
df = df.dropna(subset=critical_cols).copy()

df = df[(df['home_points'] > 70) & (df['home_points'] < 145)].copy()
df = df[(df['away_points'] > 70) & (df['away_points'] < 145)].copy()

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

print(f"  ✓ {len(df)} games loaded")

# ============================================================================
# 2. SPLIT DATA
# ============================================================================
print("[2/8] Splitting data...")

df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

print(f"  ✓ Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# ============================================================================
# 3. FEATURE ENGINEERING (DEFENSE FEATURES)
# ============================================================================
print("[3/8] Creating defense features...")

def create_defense_features(df_input):
    """Defense features - proven best"""
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

X_train = create_defense_features(df_train)
X_val = create_defense_features(df_val)
X_test = create_defense_features(df_test)

print(f"  ✓ 9 defense features created")

# ============================================================================
# 4. SCALE
# ============================================================================
print("[4/8] Scaling features...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

y_h_train = df_train['home_points'].values
y_h_val = df_val['home_points'].values
y_h_test = df_test['home_points'].values

y_a_train = df_train['away_points'].values
y_a_val = df_val['away_points'].values
y_a_test = df_test['away_points'].values

y_w_test = (df_test['home_won'].astype(int)).values if 'home_won' in df_test.columns else (df_test['home_points'] > df_test['away_points']).astype(int).values

print(f"  ✓ Scaling complete")

# ============================================================================
# 5. TRAIN HOME MODEL (SINGLE XGB: 1000/0.009/7/0.9/0.9)
# ============================================================================
print("[5/8] Training HOME single XGB (1000/0.009/7/0.9/0.9)...")

home_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.009,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=0.5,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

home_model.fit(X_train_scaled, y_h_train)
pred_home = home_model.predict(X_test_scaled)

print("  ✓ HOME model trained")

# ============================================================================
# 6. TRAIN AWAY MODEL (SINGLE XGB: 600/0.009/6/0.85/0.85)
# ============================================================================
print("[6/8] Training AWAY single XGB (600/0.009/6/0.85/0.85)...")

away_model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.009,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.5,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

away_model.fit(X_train_scaled, y_a_train)
pred_away = away_model.predict(X_test_scaled)

print("  ✓ AWAY model trained")

# ============================================================================
# 7. GENERATE PREDICTIONS
# ============================================================================
print("[7/8] Generating predictions...")

ml_pred = (pred_home > pred_away).astype(int)
ml_correct = (ml_pred == y_w_test).astype(int)
ml_acc = accuracy_score(y_w_test, ml_pred)
ml_confidence = np.tanh(np.abs(pred_home - pred_away) / 5) * 100

print("  ✓ Predictions generated")

# ============================================================================
# EVALUATE
# ============================================================================
print("\n" + "="*80)
print("🎯 SINGLE MODEL RESULTS")
print("="*80)

# HOME
home_mae = mean_absolute_error(y_h_test, pred_home)
home_within_3 = np.mean(np.abs(pred_home - y_h_test) <= 3)
home_within_5 = np.mean(np.abs(pred_home - y_h_test) <= 5)
home_within_7 = np.mean(np.abs(pred_home - y_h_test) <= 7)

print(f"\n🏀 HOME POINTS (Single XGB)")
print("-" * 80)
print(f"  Within ±3:            {home_within_3*100:.2f}%")
print(f"  Within ±5:            {home_within_5*100:.2f}%")
print(f"  Within ±7:            {home_within_7*100:.2f}%")
print(f"  MAE:                  {home_mae:.2f}")

# AWAY
away_mae = mean_absolute_error(y_a_test, pred_away)
away_within_3 = np.mean(np.abs(pred_away - y_a_test) <= 3)
away_within_5 = np.mean(np.abs(pred_away - y_a_test) <= 5)
away_within_7 = np.mean(np.abs(pred_away - y_a_test) <= 7)

print(f"\n🚗 AWAY POINTS (Single XGB)")
print("-" * 80)
print(f"  Within ±3:            {away_within_3*100:.2f}%")
print(f"  Within ±5:            {away_within_5*100:.2f}%")
print(f"  Within ±7:            {away_within_7*100:.2f}%")
print(f"  MAE:                  {away_mae:.2f}")

# TOTAL
pred_total = pred_home + pred_away
actual_total = y_h_test + y_a_test
total_mae = mean_absolute_error(actual_total, pred_total)
total_within_5 = np.mean(np.abs(pred_total - actual_total) <= 5)
total_within_10 = np.mean(np.abs(pred_total - actual_total) <= 10)

print(f"\n📊 TOTAL POINTS")
print("-" * 80)
print(f"  Within ±5:            {total_within_5*100:.2f}%")
print(f"  Within ±10:           {total_within_10*100:.2f}%")
print(f"  MAE:                  {total_mae:.2f}")

# COMBINED
combined_within_5 = (home_within_5 + away_within_5) / 2
combined_mae = (home_mae + away_mae) / 2

print(f"\n💰 MONEYLINE")
print("-" * 80)
print(f"  Accuracy:             {ml_acc*100:.2f}%")
print(f"  Correct:              {np.sum(ml_correct)}/{len(ml_correct)}")
print(f"  Avg Confidence:       {ml_confidence.mean():.1f}%")

print(f"\n📈 COMBINED (SINGLE MODEL ARCHITECTURE)")
print("="*80)
print(f"  Average ±5:           {combined_within_5*100:.2f}%")
print(f"  Average MAE:          {combined_mae:.2f}")

print("\n" + "="*80)
if combined_within_5 * 100 >= 40:
    print(f"✅ GOAL ACHIEVED: {combined_within_5*100:.2f}%")
    print(f"   Improvement: +{combined_within_5*100 - 40:.2f}% above target!")
elif combined_within_5 * 100 >= 39:
    print(f"🎯 EXCELLENT: {combined_within_5*100:.2f}%")
    print(f"   Gap to 40%: {40 - combined_within_5*100:.2f}%")
else:
    print(f"Result: {combined_within_5*100:.2f}%")
    print(f"Gap to 40%: {40 - combined_within_5*100:.2f}%")
print("="*80)

elapsed = (time.time() - start_time) / 60
print(f"\n⏱️  Complete in {elapsed:.2f} minutes\n")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("💾 SAVING MODELS...")

output_dir = 'C:/Users/Ansh Thakkar/Downloads/NBA_F/'

# Save HOME model
with open(f'{output_dir}hybrid_home_xgb.pkl', 'wb') as f:
    pickle.dump(home_model, f)

# Save AWAY model
with open(f'{output_dir}hybrid_away_xgb.pkl', 'wb') as f:
    pickle.dump(away_model, f)

# Save scaler
with open(f'{output_dir}hybrid_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save config
config_df = pd.DataFrame([{
    'target': 'HOME',
    'model_type': 'Single XGBoost',
    'n_estimators': 1000,
    'learning_rate': 0.009,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'within5_accuracy': home_within_5 * 100,
    'mae': home_mae
}, {
    'target': 'AWAY',
    'model_type': 'Single XGBoost',
    'n_estimators': 600,
    'learning_rate': 0.009,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'within5_accuracy': away_within_5 * 100,
    'mae': away_mae
}])
config_df.to_csv(f'{output_dir}FINAL_HYBRID_CONFIG.csv', index=False)

# Save predictions
predictions_df = pd.DataFrame({
    'home_actual': y_h_test.astype(int),
    'home_predicted': pred_home.round().astype(int),
    'home_error': np.abs(pred_home - y_h_test).round(1),
    'away_actual': y_a_test.astype(int),
    'away_predicted': pred_away.round().astype(int),
    'away_error': np.abs(pred_away - y_a_test).round(1),
    'total_actual': (y_h_test + y_a_test).astype(int),
    'total_predicted': (pred_home + pred_away).round().astype(int),
    'total_error': np.abs((pred_home + pred_away) - (y_h_test + y_a_test)).round(1),
    'moneyline_actual': y_w_test,
    'moneyline_predicted': ml_pred,
    'moneyline_confidence': ml_confidence.round().astype(int),
    'moneyline_correct': ml_correct
})
predictions_df.to_csv(f'{output_dir}FINAL_HYBRID_PREDICTIONS.csv', index=False)

print(f"✓ Saved all models to {output_dir}")
print(f"✓ HOME: Single XGBoost model")
print(f"✓ AWAY: Single XGBoost model")
print(f"✓ Configuration and predictions saved")

print("\n" + "="*80)
print("✅ COMPLETE - SINGLE MODEL ARCHITECTURE DEPLOYED!")
print("="*80)