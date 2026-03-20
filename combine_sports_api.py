"""
Merge Pre_Match CSV and Sports API Odds CSV
Identical logic to Combine.py but uses sports_api odds source
Input:  nba_prematch_features.csv + upcoming_nba_sports_api_odds.csv
Output: Future_sports_api.csv
"""

import pandas as pd
import os
import sys
from datetime import datetime


def merge_csvs(prematch_file: str, odds_file: str, output_file: str = "Future_sports_api.csv"):
    """
    Merge Pre_Match CSV with Sports API Odds CSV
    Identical logic to Combine.py
    """

    print(f"\n{'='*80}")
    print("MERGING CSV FILES (Sports API Pipeline)")
    print(f"{'='*80}\n")

    # Step 1: Load CSVs
    print(f"Loading Pre_Match CSV: {prematch_file}")
    try:
        df_prematch = pd.read_csv(prematch_file)
        print(f"  Loaded {len(df_prematch)} games with {len(df_prematch.columns)} columns")
    except FileNotFoundError:
        print(f"  Pre_Match file not found: {prematch_file}")
        return
    except Exception as e:
        print(f"  Error loading Pre_Match: {e}")
        return

    print(f"\nLoading Sports API Odds CSV: {odds_file}")
    try:
        df_odds = pd.read_csv(odds_file)
        print(f"  Loaded {len(df_odds)} games with {len(df_odds.columns)} columns")
    except FileNotFoundError:
        print(f"  Odds file not found: {odds_file}")
        return
    except Exception as e:
        print(f"  Error loading Odds: {e}")
        return

    # Step 2: Validate data
    print(f"\n{'='*80}")
    print("VALIDATING DATA")
    print(f"{'='*80}\n")

    if 'game_identifier' not in df_prematch.columns:
        print("  'game_identifier' not found in Pre_Match CSV")
        return

    if 'game_identifier' not in df_odds.columns:
        print("  'game_identifier' not found in Odds CSV")
        return

    print(f"  Both files have 'game_identifier' column")

    # Step 3: Select odds columns
    print(f"\n{'='*80}")
    print("EXTRACTING ODDS COLUMNS")
    print(f"{'='*80}\n")

    odds_columns_to_extract = ['game_identifier']
    column_rename_map = {}

    if 'home_winning_odds_decimal' in df_odds.columns:
        odds_columns_to_extract.append('home_winning_odds_decimal')
        print("  Found home_winning_odds_decimal")

    if 'away_winning_odds_decimal' in df_odds.columns:
        odds_columns_to_extract.append('away_winning_odds_decimal')
        print("  Found away_winning_odds_decimal")

    if 'total_line' in df_odds.columns:
        odds_columns_to_extract.append('total_line')
        print("  Found total_line")

    if 'home_spread' in df_odds.columns:
        odds_columns_to_extract.append('home_spread')
        print("  Found home_spread")

    if 'away_spread' in df_odds.columns:
        odds_columns_to_extract.append('away_spread')
        print("  Found away_spread")

    if 'home_spread_odds_decimal' in df_odds.columns:
        odds_columns_to_extract.append('home_spread_odds_decimal')
        print("  Found home_spread_odds_decimal")

    if 'away_spread_odds_decimal' in df_odds.columns:
        odds_columns_to_extract.append('away_spread_odds_decimal')
        print("  Found away_spread_odds_decimal")

    if 'total_line_o' in df_odds.columns:
        odds_columns_to_extract.append('total_line_o')
        print("  Found total_line_o")

    if 'total_line_over_odds_decimal' in df_odds.columns:
        odds_columns_to_extract.append('total_line_over_odds_decimal')
        print("  Found total_line_over_odds_decimal")

    if 'total_line_under_odds_decimal' in df_odds.columns:
        odds_columns_to_extract.append('total_line_under_odds_decimal')
        print("  Found total_line_under_odds_decimal")

    df_odds_subset = df_odds[odds_columns_to_extract].copy()
    df_odds_subset.rename(columns=column_rename_map, inplace=True)

    print(f"\n  Extracting {len(odds_columns_to_extract) - 1} odds columns")

    # Step 4: Merge DataFrames
    print(f"\n{'='*80}")
    print("MERGING DATA")
    print(f"{'='*80}\n")

    df_merged = pd.merge(
        df_prematch,
        df_odds_subset,
        on='game_identifier',
        how='left'
    )

    print(f"  Merged on 'game_identifier'")
    print(f"  Total games: {len(df_merged)}")

    if 'home_winning_odds_decimal' in df_merged.columns:
        print(f"  Games with moneyline odds: {df_merged['home_winning_odds_decimal'].notna().sum()}")

    if 'home_spread' in df_merged.columns:
        print(f"  Games with spreads: {df_merged['home_spread'].notna().sum()}")

    if 'total_line' in df_merged.columns:
        print(f"  Games with total_line (source 1): {df_merged['total_line'].notna().sum()}")

    if 'total_line_o' in df_merged.columns:
        print(f"  Games with total_line_o: {df_merged['total_line_o'].notna().sum()}")

    # Step 5: Reorganize columns
    print(f"\n{'='*80}")
    print("ORGANIZING COLUMNS")
    print(f"{'='*80}\n")

    core_cols = [
        'game_identifier', 'match_id', 'scheduled', 'status',
        'home_alias', 'away_alias', 'home_name', 'away_name',
        'venue_name', 'venue_city', 'league'
    ]

    odds_cols = [
        'home_winning_odds_decimal', 'away_winning_odds_decimal',
        'home_spread', 'away_spread',
        'home_spread_odds_decimal', 'away_spread_odds_decimal',
        'total_line', 'total_line_o',
        'total_line_over_odds_decimal', 'total_line_under_odds_decimal'
    ]

    final_col_order = []

    for col in core_cols:
        if col in df_merged.columns:
            final_col_order.append(col)

    for col in df_merged.columns:
        if col not in core_cols + odds_cols:
            final_col_order.append(col)

    for col in odds_cols:
        if col in df_merged.columns:
            final_col_order.append(col)

    df_merged = df_merged[final_col_order]

    print(f"  Final column order:")
    print(f"  1. Core info: {len([c for c in final_col_order if c in core_cols])} columns")
    print(f"  2. Team stats & features: {len([c for c in final_col_order if c not in core_cols + odds_cols])} columns")
    print(f"  3. Betting odds (at end): {len([c for c in final_col_order if c in odds_cols])} columns")

    # Step 6: Save to CSV
    print(f"\n{'='*80}")
    print("SAVING OUTPUT")
    print(f"{'='*80}\n")

    df_merged.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    print(f"  Total games: {len(df_merged)}")
    print(f"  Total columns: {len(df_merged.columns)}")

    # Step 7: Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print(f"Pre-Match games: {len(df_prematch)}")
    print(f"Odds games: {len(df_odds)}")
    print(f"Merged games: {len(df_merged)}")

    print(f"\nMatch results:")
    if 'home_winning_odds_decimal' in df_merged.columns:
        matched_moneyline = df_merged['home_winning_odds_decimal'].notna().sum()
        print(f"  Games with moneyline: {matched_moneyline}")

    if 'home_spread' in df_merged.columns:
        matched_spreads = df_merged['home_spread'].notna().sum()
        print(f"  Games with spreads: {matched_spreads}")

    if 'total_line' in df_merged.columns:
        matched_total_1 = df_merged['total_line'].notna().sum()
        print(f"  Games with total_line (source 1): {matched_total_1}")

    if 'total_line_o' in df_merged.columns:
        matched_totals = df_merged['total_line_o'].notna().sum()
        print(f"  Games with total_line_o: {matched_totals}")

    print(f"\nOdds statistics:")

    if 'home_winning_odds_decimal' in df_merged.columns and df_merged['home_winning_odds_decimal'].notna().sum() > 0:
        print(f"  Moneyline:")
        print(f"    Home odds: {df_merged['home_winning_odds_decimal'].min():.2f} - {df_merged['home_winning_odds_decimal'].max():.2f}")
        print(f"    Away odds: {df_merged['away_winning_odds_decimal'].min():.2f} - {df_merged['away_winning_odds_decimal'].max():.2f}")

    if 'home_spread' in df_merged.columns and df_merged['home_spread'].notna().sum() > 0:
        print(f"  Spreads:")
        print(f"    Home points: {df_merged['home_spread'].min():.1f} to {df_merged['home_spread'].max():.1f}")

    if 'total_line_o' in df_merged.columns and df_merged['total_line_o'].notna().sum() > 0:
        print(f"  Totals:")
        print(f"    Line: {df_merged['total_line_o'].min():.1f} - {df_merged['total_line_o'].max():.1f}")

    # Sample
    print(f"\n{'='*80}")
    print("SAMPLE DATA (First 3 games)")
    print(f"{'='*80}\n")

    sample_cols = [
        'game_identifier', 'home_alias', 'away_alias',
        'home_spread', 'home_spread_odds_decimal',
        'away_spread', 'away_spread_odds_decimal',
        'total_line', 'total_line_o', 'total_line_over_odds_decimal', 'total_line_under_odds_decimal'
    ]
    available_sample_cols = [c for c in sample_cols if c in df_merged.columns]
    if available_sample_cols:
        print(df_merged[available_sample_cols].head(3).to_string(index=False))

    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}\n")


def main():
    """Main execution - fully automated, no user interaction"""

    print("\n" + "="*80)
    print("CSV MERGE UTILITY - Pre_Match + Sports API Odds")
    print("="*80)

    print("\nRunning in automated mode")

    # Use same prematch file as original pipeline
    prematch_file = None
    for file in sorted(os.listdir('.'), reverse=True):
        if file.startswith('nba_prematch_features') and file.endswith('.csv'):
            prematch_file = file
            break
    if not prematch_file:
        prematch_file = "nba_prematch_features.csv"

    odds_file = "upcoming_nba_sports_api_odds.csv"
    output_file = "Future_sports_api.csv"

    print(f"  Using: {prematch_file}")
    print(f"  Using: {odds_file}")
    print(f"  Output: {output_file}\n")

    # Execute merge
    merge_csvs(prematch_file, odds_file, output_file)


if __name__ == "__main__":
    main()
