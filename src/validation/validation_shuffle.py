# validation_shuffling.py
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from data_preparation import prepare_prediction_data

def analyze_by_change_status(y_test, y_pred, test_df):
    """Analyse performance par statut de changement."""
    # Détecte qui a eu des changements
    patch_change_cols = [col for col in test_df.columns 
                        if any(s in col for s in ['ability_', 'item_', 'base_stat_', 'per_level_'])]
    
    has_changes = (test_df[patch_change_cols].abs().sum(axis=1) > 0)
    
    print(f"\n📊 ANALYSE PAR STATUT DE CHANGEMENT:")
    print(f"Champions AVEC changements: {has_changes.sum()}")
    if has_changes.sum() > 0:
        mae_with = mean_absolute_error(y_test[has_changes], y_pred[has_changes])
        mean_change_with = y_test[has_changes].mean()
        print(f"  - MAE: {mae_with:.4f}")
        print(f"  - Changement moyen: {mean_change_with:.4f}")
    
    print(f"Champions SANS changements: {(~has_changes).sum()}")
    if (~has_changes).sum() > 0:
        mae_without = mean_absolute_error(y_test[~has_changes], y_pred[~has_changes])
        mean_change_without = y_test[~has_changes].mean()
        print(f"  - MAE: {mae_without:.4f}")
        print(f"  - Changement moyen: {mean_change_without:.4f}")

def add_temporal_features(df_full, X):
    """Ajoute des caractéristiques temporelles aux données."""
    df = df_full.copy()
    # Ordinalisation des patches
    patches = sorted(df['patch'].unique(), key=lambda x: [int(p) for p in x.split('.')])
    patch_map = {p: i for i, p in enumerate(patches)}
    df['patch_idx'] = df['patch'].map(patch_map)

    # Rolling et précédent
    df.sort_values(['champion_name', 'patch_idx'], inplace=True)
    df['champ_prev_win'] = df.groupby('champion_name')['winrate'].shift(1)
    df['champ_roll3'] = (
        df.groupby('champion_name')['winrate']
          .rolling(3, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )
    
    # Tendance sur 1 et 2 patches
    df['win_trend_1'] = df.groupby('champion_name')['winrate'].diff(1)
    df['win_trend_2'] = df.groupby('champion_name')['winrate'].diff(2)
    
    # Volatilité
    df['win_volatility'] = (
        df.groupby('champion_name')['winrate']
          .rolling(3, min_periods=2)
          .std()
          .reset_index(level=0, drop=True)
    )

    # Moyennes et positions relatives
    champ_mean = df.groupby('champion_name')['winrate'].transform('mean')
    patch_mean = df.groupby('patch')['winrate'].transform('mean')
    global_mean = df['winrate'].mean()
    
    df['rel_to_champ_mean'] = df['winrate'] - champ_mean
    df['rel_to_patch_mean'] = df['winrate'] - patch_mean
    df['rel_to_global_mean'] = df['winrate'] - global_mean

    # Transfert des caractéristiques dans X
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
    """Agrège les changements des compétences pour réduire la dimensionnalité."""
    ability_types = ['Passive', 'Q', 'W', 'E', 'R']
    
    agg_features = {}
    
    # Agrégation par type d'abilité
    for ability in ability_types:
        # Identification des colonnes par type
        damage_cols = [col for col in X.columns if f'ability_{ability}_base_damage' in col]
        cooldown_cols = [col for col in X.columns if f'ability_{ability}_cooldown' in col]
        mana_cols = [col for col in X.columns if f'ability_{ability}_mana_cost' in col]
        ap_ratio_cols = [col for col in X.columns if f'ability_{ability}_ap_ratio' in col]
        ad_ratio_cols = [col for col in X.columns if f'ability_{ability}_ad_ratio' in col or 
                        f'ability_{ability}_bonus_ad_ratio' in col]
        
        # Agrégation par somme ou moyenne selon la nature de la caractéristique
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
    
    # Agrégation des statistiques de base
    base_stat_cols = [col for col in X.columns if 'base_stat_' in col]
    per_level_cols = [col for col in X.columns if 'per_level_' in col]
    item_cols = [col for col in X.columns if 'item_' in col]
    
    agg_features['base_stat_total_change'] = X[base_stat_cols].sum(axis=1)
    agg_features['per_level_total_change'] = X[per_level_cols].sum(axis=1)
    agg_features['item_total_change'] = X[item_cols].sum(axis=1)
    
    return pd.DataFrame(agg_features, index=X.index)

def run_shuffle_test(n_iterations=10):
    """Effectue un test de shuffling des étiquettes pour détecter les fuites de données."""
    print("Test de Shuffling des Étiquettes")
    print("===============================")
    
    # Chargement des données
    data = prepare_prediction_data(temporal_split=True)
    full_df = data['full_data']
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    w_train, w_test = data['w_train'], data['w_test']
    
    # Feature engineering
    X_train = add_temporal_features(full_df.loc[X_train.index], X_train)
    X_test = add_temporal_features(full_df.loc[X_test.index], X_test)
    
    for col in ['pickrate', 'total_games']:
        X_train[col] = full_df.loc[X_train.index, col]
        X_test[col] = full_df.loc[X_test.index, col]
    
    X_train_combined = X_train.copy()
    X_test_combined = X_test.copy()
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined.fillna(0))
    X_test_scaled = scaler.transform(X_test_combined.fillna(0))
    
    # Résultats réels
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
    real_r2 = r2_score(y_test, y_pred, sample_weight=w_test)
    
    # Test avec étiquettes mélangées
    shuffle_r2_values = []
    
    for i in range(n_iterations):
        # Mélanger les étiquettes
        y_train_shuffled = y_train.copy().sample(frac=1).reset_index(drop=True)
        y_train_shuffled.index = y_train.index
        
        # Entraîner avec étiquettes mélangées
        shuffle_model = xgb.XGBRegressor(
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
        
        shuffle_model.fit(X_train_scaled, y_train_shuffled, sample_weight=w_train)
        y_pred_shuffled = shuffle_model.predict(X_test_scaled)
        shuffle_r2 = r2_score(y_test, y_pred_shuffled, sample_weight=w_test)
        shuffle_r2_values.append(shuffle_r2)
        
        print(f"Itération {i+1}: R² avec étiquettes mélangées = {shuffle_r2:.4f}")
    
    # Résumé des résultats
    avg_shuffle_r2 = np.mean(shuffle_r2_values)
    max_shuffle_r2 = np.max(shuffle_r2_values)
    
    print("\nRésultats:")
    print(f"R² réel: {real_r2:.4f}")
    print(f"R² moyen avec étiquettes mélangées: {avg_shuffle_r2:.4f}")
    print(f"R² maximum avec étiquettes mélangées: {max_shuffle_r2:.4f}")
    
    if avg_shuffle_r2 > 0.1:
        print("\nAVERTISSEMENT: Le modèle semble apprendre même avec des étiquettes mélangées,")
        print("ce qui pourrait indiquer une fuite de données ou des caractéristiques problématiques.")
    else:
        print("\nLe modèle ne semble pas apprendre de motifs significatifs avec des étiquettes mélangées,")
        print("ce qui suggère que le R² réel est probablement valide.")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.hist(shuffle_r2_values, bins=20, alpha=0.7)
    plt.axvline(real_r2, color='r', linestyle='--', label=f'R² réel: {real_r2:.4f}')
    plt.axvline(avg_shuffle_r2, color='g', linestyle='--', label=f'R² moyen des shuffles: {avg_shuffle_r2:.4f}')
    plt.title('Distribution des R² après shuffling des étiquettes')
    plt.xlabel('R²')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.savefig('shuffle_test_results.png')
    plt.close()

if __name__ == "__main__":
    run_shuffle_test(n_iterations=10)