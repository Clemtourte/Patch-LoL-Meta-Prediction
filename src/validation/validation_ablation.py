# validation_ablation.py
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

# Import des mêmes fonctions d'ajout de caractéristiques et d'agrégation
# (Utilisez les mêmes fonctions que dans le script précédent)

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

def run_ablation_study():
    """Effectue une étude d'ablation pour évaluer l'importance de différents groupes de caractéristiques."""
    print("Étude d'Ablation")
    print("===============")
    
    # Chargement des données
    data = prepare_prediction_data(temporal_split=True)
    full_df = data['full_data']
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    w_train, w_test = data['w_train'], data['w_test']
    
    # Feature engineering complet
    X_train = add_temporal_features(full_df.loc[X_train.index], X_train)
    X_test = add_temporal_features(full_df.loc[X_test.index], X_test)
    
    for col in ['pickrate', 'total_games']:
        X_train[col] = full_df.loc[X_train.index, col]
        X_test[col] = full_df.loc[X_test.index, col]
    
    agg_train = aggregate_ability_changes(X_train)
    agg_test = aggregate_ability_changes(X_test)
    
    X_train_combined = pd.concat([X_train, agg_train], axis=1)
    X_test_combined = pd.concat([X_test, agg_test], axis=1)
    
    # Définition des groupes de caractéristiques à tester
    feature_groups = {
        "Toutes les caractéristiques": X_train_combined.columns.tolist(),
        "Sans caractéristiques temporelles": [col for col in X_train_combined.columns if col not in 
                                            ['patch_idx', 'champ_prev_win', 'champ_roll3', 
                                            'win_trend_1', 'win_trend_2', 'win_volatility']],
        "Sans caractéristiques relatives": [col for col in X_train_combined.columns if col not in 
                                          ['rel_to_champ_mean', 'rel_to_patch_mean', 'rel_to_global_mean']],
        "Sans statistiques de champion": [col for col in X_train_combined.columns if not any(s in col for s in 
                                         ['base_stat_', 'per_level_'])],
        "Sans changements d'objets": [col for col in X_train_combined.columns if not 'item_' in col],
        "Sans changements d'aptitudes": [col for col in X_train_combined.columns if not any(s in col for s in 
                                       ['Passive_', 'Q_', 'W_', 'E_', 'R_'])],
        "Base stats + per level uniquement": [col for col in X_train_combined.columns if any(s in col for s in 
                                           ['base_stat_', 'per_level_'])],
        "Caractéristiques temporelles uniquement": ['patch_idx', 'champ_prev_win', 'champ_roll3', 
                                                  'win_trend_1', 'win_trend_2', 'win_volatility']
    }
    
    # Exécution des tests pour chaque groupe
    results = {}
    
    for group_name, features in feature_groups.items():
        # Filtrer les caractéristiques qui existent dans le DataFrame
        features = [f for f in features if f in X_train_combined.columns]
        
        # Normalisation
        scaler = StandardScaler()
        X_train_subset = X_train_combined[features].fillna(0)
        X_test_subset = X_test_combined[features].fillna(0)
        
        X_train_scaled = scaler.fit_transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)
        
        # Entraînement
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
        
        # Métriques
        r2 = r2_score(y_test, y_pred, sample_weight=w_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))
        
        results[group_name] = {
            'r2': r2,
            'rmse': rmse,
            'n_features': len(features)
        }
        
        print(f"\n{group_name}:")
        print(f"  Nombre de caractéristiques: {len(features)}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    
    # Tri par R²
    groups = sorted(results.keys(), key=lambda x: results[x]['r2'], reverse=True)
    r2_values = [results[g]['r2'] for g in groups]
    
    bar_positions = np.arange(len(groups))
    bars = plt.barh(bar_positions, r2_values, height=0.6)
    
    plt.yticks(bar_positions, groups)
    plt.xlabel('R²')
    plt.title('Impact des Différents Groupes de Caractéristiques sur le R²')
    
    # Ajout des valeurs sur les barres
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{r2_values[i]:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png')
    plt.close()
    
    # Export des résultats
    pd.DataFrame(results).T.to_csv('ablation_study_results.csv')
    
    return results

if __name__ == "__main__":
    run_ablation_study()