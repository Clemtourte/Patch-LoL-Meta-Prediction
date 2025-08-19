import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
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

def run_ablation_study():
    print("="*50)
    print("ABLATION STUDY")
    print("="*50)
    
    # Load data
    df_full, X, y, w, patches = prepare_prediction_data()
    patches = df_full['patch']
    
    # Add temporal features
    X = add_temporal_features(df_full, X)
    
    # Aggregate ability changes
    aggregate_cols = list(aggregate_ability_changes(X).columns)
    X_aggregated = aggregate_ability_changes(X)
    
    # Remove individual columns and replace with aggregated
    cols_to_remove = []
    for col in X.columns:
        if any(ability in col for ability in ['ability_Passive', 'ability_Q', 'ability_W', 'ability_E', 'ability_R']):
            cols_to_remove.append(col)
    
    X_reduced = X.drop(columns=cols_to_remove)
    X_final = pd.concat([X_reduced, X_aggregated], axis=1)
    
    # Temporal split
    patches_sorted = sorted(patches.unique(), key=lambda x: [int(p) for p in x.split('.')])
    n_train = int(len(patches_sorted) * 0.8)
    train_patches = patches_sorted[:n_train]
    test_patches = patches_sorted[n_train:]
    
    train_mask = patches.isin(train_patches)
    test_mask = patches.isin(test_patches)
    
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
    
    # All feature columns
    feature_cols = list(X_final.columns)
    
    # Define feature groups IN ENGLISH
    feature_groups = {
        'All features': feature_cols,
        'Without temporal features': [col for col in feature_cols if not any(temp in col for temp in ['patch_idx', 'prev_win', 'roll3', 'trend', 'volatility', 'rel_to'])],
        'Without relative features': [col for col in feature_cols if 'rel_to' not in col],
        'Without champion statistics': [col for col in feature_cols if not any(stat in col for stat in ['base_stat_', 'per_level_'])],
        'Without item changes': [col for col in feature_cols if 'item_' not in col],
        'Without ability changes': aggregate_cols,
        'Base stats + per level only': [col for col in feature_cols if 'base_stat_' in col or 'per_level_' in col],
        'Temporal features only': [col for col in feature_cols if any(temp in col for temp in ['patch_idx', 'prev_win', 'roll3', 'trend', 'volatility', 'rel_to'])]
    }
    
    results = {}
    
    for group_name, features in feature_groups.items():
        if len(features) == 0:
            print(f"\n{group_name}: No features to test")
            continue
        
        # Select features
        feature_indices = [i for i, col in enumerate(feature_cols) if col in features]
        X_train_subset = X_train_scaled[:, feature_indices]
        X_test_subset = X_test_scaled[:, feature_indices]
        
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
        
        model.fit(X_train_subset, y_train, sample_weight=w_train)
        y_pred = model.predict(X_test_subset)
        
        # Metrics
        r2 = r2_score(y_test, y_pred, sample_weight=w_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))
        
        results[group_name] = {
            'r2': r2,
            'rmse': rmse,
            'n_features': len(features)
        }
        
        print(f"\n{group_name}:")
        print(f"  Number of features: {len(features)}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Sort by R²
    groups = sorted(results.keys(), key=lambda x: results[x]['r2'], reverse=True)
    r2_values = [results[g]['r2'] for g in groups]
    
    bar_positions = np.arange(len(groups))
    bars = plt.barh(bar_positions, r2_values, height=0.6)
    
    plt.yticks(bar_positions, groups)
    plt.xlabel('R²')
    plt.title('Impact of Different Feature Groups on R²')
    
    # Add values on bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{r2_values[i]:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png')
    plt.close()
    
    # Export results
    pd.DataFrame(results).T.to_csv('ablation_study_results.csv')
    
    return results

if __name__ == "__main__":
    run_ablation_study()