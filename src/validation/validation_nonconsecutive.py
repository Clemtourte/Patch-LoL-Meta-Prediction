# validation_nonconsecutive.py
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
# (Utilisez les mêmes fonctions que dans les scripts précédents)

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

# Suite du script validation_nonconsecutive.py

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

def run_nonconsecutive_validation():
    """
    Teste le modèle sur des patchs non consécutifs pour évaluer sa robustesse
    à différentes 'époques' du jeu.
    """
    print("Validation sur Patchs Non-consécutifs")
    print("====================================")
    
    # Chargement des données complètes
    data = prepare_prediction_data(temporal_split=False)  # Pas de split temporel ici
    full_df = data['full_data']
    
    # Obtenir la liste complète des patchs
    all_patches = sorted(full_df['patch'].unique(), key=lambda x: [int(p) for p in x.split('.')])
    print(f"Total des patchs disponibles: {len(all_patches)}")
    print(f"Liste des patchs: {all_patches}")
    
    # Diviser les patchs en groupes
    patch_groups = {
        "Season 13 start": [p for p in all_patches if p.startswith("13.") and int(p.split('.')[1]) <= 8],
        "Season 13 end": [p for p in all_patches if p.startswith("13.") and int(p.split('.')[1]) > 8],
        "Season 14 start": [p for p in all_patches if p.startswith("14.") and int(p.split('.')[1]) <= 8],
        "Season 14 end": [p for p in all_patches if p.startswith("14.") and int(p.split('.')[1]) > 8]
    }
    
    # Afficher les groupes de patchs
    for group, patches in patch_groups.items():
        print(f"\n{group}: {patches}")
    
    # Combinaisons de test: entraîner sur un groupe, tester sur un autre
    test_combinations = [
        ("Season 13 start", "Season 13 end"),
        ("Season 13 start", "Season 14 start"),
        ("Season 13 start", "Season 14 end"),
        ("Season 13 end", "Season 14 start"),
        ("Season 13 end", "Season 14 end"),
        ("Season 14 start", "Season 14 end")
    ]
    
    results = {}
    
    # Exécuter les tests pour chaque combinaison
    for train_group, test_group in test_combinations:
        print(f"\nTest: Entraînement sur {train_group}, Test sur {test_group}")
        
        # Créer des masques pour les patchs d'entraînement et de test
        train_mask = full_df['patch'].isin(patch_groups[train_group])
        test_mask = full_df['patch'].isin(patch_groups[test_group])
        
        # Extraire les données
        X_train = full_df.loc[train_mask, data['feature_names']]
        y_train = full_df.loc[train_mask, 'delta_winrate']
        w_train = full_df.loc[train_mask, 'total_games'] / full_df.loc[train_mask, 'total_games'].mean()
        
        X_test = full_df.loc[test_mask, data['feature_names']]
        y_test = full_df.loc[test_mask, 'delta_winrate']
        w_test = full_df.loc[test_mask, 'total_games'] / full_df.loc[test_mask, 'total_games'].mean()
        
        # Feature engineering
        X_train = add_temporal_features(full_df.loc[train_mask], X_train)
        X_test = add_temporal_features(full_df.loc[test_mask], X_test)
        
        for col in ['pickrate', 'total_games']:
            X_train[col] = full_df.loc[train_mask, col]
            X_test[col] = full_df.loc[test_mask, col]
        
        X_train_combined = X_train.copy()
        X_test_combined = X_test.copy()
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_combined.fillna(0))
        X_test_scaled = scaler.transform(X_test_combined.fillna(0))
        
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
        mae = mean_absolute_error(y_test, y_pred, sample_weight=w_test)
        
        results[f"{train_group} -> {test_group}"] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_train': train_mask.sum(),
            'n_test': test_mask.sum()
        }
        
        print(f"  Échantillons d'entraînement: {train_mask.sum()}")
        print(f"  Échantillons de test: {test_mask.sum()}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    
    # Tri par R²
    combinations = sorted(results.keys(), key=lambda x: results[x]['r2'], reverse=True)
    r2_values = [results[c]['r2'] for c in combinations]
    
    bar_positions = np.arange(len(combinations))
    bars = plt.barh(bar_positions, r2_values, height=0.6)
    
    plt.yticks(bar_positions, combinations)
    plt.xlabel('R²')
    plt.title('Performance du Modèle sur Différentes Époques du Jeu')
    
    # Ajout des valeurs sur les barres
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{r2_values[i]:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig('nonconsecutive_validation_results.png')
    plt.close()
    
    # Export des résultats
    pd.DataFrame(results).T.to_csv('nonconsecutive_validation_results.csv')

    analyze_by_change_status(y_test, y_pred, full_df.loc[X_test.index])

    return results

if __name__ == "__main__":
    run_nonconsecutive_validation()