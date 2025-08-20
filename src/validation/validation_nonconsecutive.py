import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from data_preparation import prepare_prediction_data

def add_temporal_features(df_full, X):
    """Add temporal features to the data."""
    df = df_full.copy()
    # Patch ordinalization
    patches = sorted(df['patch'].unique(), key=lambda x: [int(p) for p in x.split('.')])
    patch_map = {p: i for i, p in enumerate(patches)}
    df['patch_idx'] = df['patch'].map(patch_map)

    # Rolling and previous
    df.sort_values(['champion_name', 'patch_idx'], inplace=True)
    df['champ_prev_win'] = df.groupby('champion_name')['winrate'].shift(1)
    df['champ_roll3'] = (
        df.groupby('champion_name')['winrate']
          .rolling(3, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )
    
    # Trend over 1 and 2 patches
    df['win_trend_1'] = df.groupby('champion_name')['winrate'].diff(1)
    df['win_trend_2'] = df.groupby('champion_name')['winrate'].diff(2)
    
    # Volatility
    df['win_volatility'] = (
        df.groupby('champion_name')['winrate']
          .rolling(3, min_periods=2)
          .std()
          .reset_index(level=0, drop=True)
    )

    # Means and relative positions
    champ_mean = df.groupby('champion_name')['winrate'].transform('mean')
    patch_mean = df.groupby('patch')['winrate'].transform('mean')
    global_mean = df['winrate'].mean()
    
    df['rel_to_champ_mean'] = df['winrate'] - champ_mean
    df['rel_to_patch_mean'] = df['winrate'] - patch_mean
    df['rel_to_global_mean'] = df['winrate'] - global_mean

    # Transfer features to X
    X = X.copy()
    X['patch_idx'] = df.loc[X.index, 'patch_idx']
    X['champ_prev_win'] = df.loc[X.index, 'champ_prev_win'].fillna(global_mean)
    X['champ_roll3'] = df.loc[X.index, 'champ_roll3']
    X['win_trend_1'] = df.loc[X.index, 'win_trend_1'].fillna(0)
    X['win_trend_2'] = df.loc[X.index, 'win_trend_2'].fillna(0)
    X['win_volatility'] = df.loc[X.index, 'win_volatility'].fillna(0)
    X['rel_to_champ_mean'] = df.loc[X.index, 'rel_to_champ_mean']
    X['rel_to_patch_mean'] = df.loc[X.index, 'rel_to_patch_mean']
    X['rel_to_global_mean'] = df.loc[X.index, 'rel_to_global_mean']
    
    return X

def aggregate_ability_changes(X):
    """Aggregate ability changes to reduce dimensionality."""
    ability_types = ['Passive', 'Q', 'W', 'E', 'R']
    
    agg_features = {}
    
    # Aggregation by ability type
    for ability in ability_types:
        # Identify columns by type
        damage_cols = [col for col in X.columns if f'ability_{ability}_base_damage' in col]
        cooldown_cols = [col for col in X.columns if f'ability_{ability}_cooldown' in col]
        mana_cols = [col for col in X.columns if f'ability_{ability}_mana_cost' in col]
        ap_ratio_cols = [col for col in X.columns if f'ability_{ability}_ap_ratio' in col]
        ad_ratio_cols = [col for col in X.columns if f'ability_{ability}_ad_ratio' in col or 
                        f'ability_{ability}_bonus_ad_ratio' in col]
        
        # Aggregate by sum or mean depending on the nature of the feature
        if damage_cols:
            agg_features[f'{ability}_damage_change'] = X[damage_cols].sum(axis=1)
        
        if cooldown_cols:
            agg_features[f'{ability}_cooldown_change'] = X[cooldown_cols].mean(axis=1)
        
        if mana_cols:
            agg_features[f'{ability}_mana_change'] = X[mana_cols].mean(axis=1)
            
        if ap_ratio_cols:
            agg_features[f'{ability}_ap_ratio_change'] = X[ap_ratio_cols].sum(axis=1)
            
        if ad_ratio_cols:
            agg_features[f'{ability}_ad_ratio_change'] = X[ad_ratio_cols].sum(axis=1)
    
    # Aggregate base stats
    base_stat_cols = [col for col in X.columns if 'base_stat_' in col]
    per_level_cols = [col for col in X.columns if 'per_level_' in col]
    item_cols = [col for col in X.columns if 'item_' in col]
    
    agg_features['base_stat_total_change'] = X[base_stat_cols].sum(axis=1)
    agg_features['per_level_total_change'] = X[per_level_cols].sum(axis=1)
    agg_features['item_total_change'] = X[item_cols].sum(axis=1)
    
    return pd.DataFrame(agg_features, index=X.index)

def run_nonconsecutive_validation():
    print("="*50)
    print("CROSS-EPOCH VALIDATION")
    print("="*50)
    
    # Load data
    data = prepare_prediction_data(temporal_split=False)  # Pas de split pour ce test
    df_full = data['full_data']
    X_train = data['X_train'] 
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    w_train = data['w_train']
    w_test = data['w_test']

    # Combine train and test for this validation
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test]) 
    w = pd.concat([w_train, w_test])
    patches = df_full['patch']
    
    # Add temporal features
    X = add_temporal_features(df_full, X)
    
    # Aggregate ability changes
    X_aggregated = aggregate_ability_changes(X)
    
    # Remove individual columns and replace with aggregated
    cols_to_remove = []
    for col in X.columns:
        if any(ability in col for ability in ['ability_Passive', 'ability_Q', 'ability_W', 'ability_E', 'ability_R']):
            cols_to_remove.append(col)
    
    X_reduced = X.drop(columns=cols_to_remove)
    X_final = pd.concat([X_reduced, X_aggregated], axis=1)
    
    # Define epoch groups IN ENGLISH
    epoch_groups = {
        'Season 13 start': lambda p: p.startswith('13.') and int(p.split('.')[1]) <= 5,
        'Season 13 mid': lambda p: p.startswith('13.') and 6 <= int(p.split('.')[1]) <= 15,
        'Season 13 end': lambda p: p.startswith('13.') and int(p.split('.')[1]) >= 16,
        'Season 14 start': lambda p: p.startswith('14.') and int(p.split('.')[1]) <= 8,
        'Season 14 mid': lambda p: p.startswith('14.') and 9 <= int(p.split('.')[1]) <= 16,
        'Season 14 end': lambda p: p.startswith('14.') and int(p.split('.')[1]) >= 17
    }
    
    results = {}
    
    # Test all combinations
    for train_group, train_filter in epoch_groups.items():
        for test_group, test_filter in epoch_groups.items():
            if train_group == test_group:
                continue
            
            print(f"\nTraining on {train_group}, Testing on {test_group}")
            
            # Create masks
            train_mask = patches.apply(train_filter)
            test_mask = patches.apply(test_filter)
            
            if train_mask.sum() < 10 or test_mask.sum() < 10:
                print(f"  Insufficient data (train: {train_mask.sum()}, test: {test_mask.sum()})")
                continue
            
            X_train = X_final[train_mask]
            X_test = X_final[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            w_train = w[train_mask]
            w_test = w[test_mask]
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=1.0,
                gamma=0.05,
                min_child_weight=3,
                objective='reg:squarederror',
                random_state=42,
                verbosity=0
            )
            
            model.fit(X_train_scaled, y_train, sample_weight=w_train)
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            r2 = r2_score(y_test, y_pred, sample_weight=w_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))
            mae = mean_absolute_error(y_test, y_pred, sample_weight=w_test)
            
            results[f"{train_group} -> {test_group}"] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'n_train': train_mask.sum(),
                'n_test': test_mask.sum()
            }
            
            print(f"  Training samples: {train_mask.sum()}")
            print(f"  Test samples: {test_mask.sum()}")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Sort by R²
    combinations = sorted(results.keys(), key=lambda x: results[x]['r2'], reverse=True)
    r2_values = [results[c]['r2'] for c in combinations]
    
    bar_positions = np.arange(len(combinations))
    bars = plt.barh(bar_positions, r2_values, height=0.6)
    
    plt.yticks(bar_positions, combinations)
    plt.xlabel('R²')
    plt.title('Model Performance on Different Game Epochs')
    
    # Add values on bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{r2_values[i]:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig('nonconsecutive_validation_results.png')
    plt.close()
    
    # Export results
    pd.DataFrame(results).T.to_csv('nonconsecutive_validation_results.csv')
    
    return results

if __name__ == "__main__":
    run_nonconsecutive_validation()