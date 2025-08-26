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

def save_change_status_analysis(y_test, y_pred, test_df):
    """Sauvegarde l'analyse par statut de changement dans un CSV."""
    
    # Détecte qui a eu des changements
    patch_change_cols = [col for col in test_df.columns 
                        if any(s in col for s in ['ability_', 'item_', 'base_stat_', 'per_level_'])]
    
    has_changes = (test_df[patch_change_cols].abs().sum(axis=1) > 0)
    
    # Calcul des métriques par groupe
    results = []
    
    # Groupe AVEC changements
    if has_changes.sum() > 0:
        with_changes_mask = has_changes
        mae_with = mean_absolute_error(y_test[with_changes_mask], y_pred[with_changes_mask])
        mean_change_with = y_test[with_changes_mask].mean()
        std_change_with = y_test[with_changes_mask].std()
        
        results.append({
            'group': 'With Changes',
            'count': has_changes.sum(),
            'mae': mae_with,
            'mean_change': mean_change_with,
            'std_change': std_change_with,
            'percentage': (has_changes.sum() / len(y_test)) * 100
        })
    
    # Groupe SANS changements
    if (~has_changes).sum() > 0:
        without_changes_mask = ~has_changes
        mae_without = mean_absolute_error(y_test[without_changes_mask], y_pred[without_changes_mask])
        mean_change_without = y_test[without_changes_mask].mean()
        std_change_without = y_test[without_changes_mask].std()
        
        results.append({
            'group': 'Without Changes',
            'count': (~has_changes).sum(),
            'mae': mae_without,
            'mean_change': mean_change_without,
            'std_change': std_change_without,
            'percentage': ((~has_changes).sum() / len(y_test)) * 100
        })
    
    # Sauvegarde en CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('change_status_analysis.csv', index=False)
    
    print(f"💾 Analyse sauvegardée dans change_status_analysis.csv")
    
    return results_df

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
    
    # Feature engineering complet (SANS agrégation)
    X_train = add_temporal_features(full_df.loc[X_train.index], X_train)
    X_test = add_temporal_features(full_df.loc[X_test.index], X_test)
    
    for col in ['pickrate', 'total_games']:
        X_train[col] = full_df.loc[X_train.index, col]
        X_test[col] = full_df.loc[X_test.index, col]
    
    X_train_combined = X_train.copy()
    X_test_combined = X_test.copy()
    
    print(f"Nombre total de features: {len(X_train_combined.columns)}")
    
    # Définition des groupes de caractéristiques à tester
    feature_groups = {
        "All features": X_train_combined.columns.tolist(),
        
        "Without ability changes": [col for col in X_train_combined.columns 
                                  if not any(s in col for s in ['ability_'])],
        
        "Without item changes": [col for col in X_train_combined.columns 
                               if not any(s in col for s in ['item_'])],
        
        "Without champion statistics": [col for col in X_train_combined.columns 
                                      if not any(s in col for s in ['base_stat_', 'per_level_'])],
        
        "Patch changes only": [col for col in X_train_combined.columns 
                             if any(s in col for s in ['ability_', 'item_', 'base_stat_', 'per_level_'])],
        
        "Context only": [col for col in X_train_combined.columns 
                       if col in ['pickrate', 'total_games']],
        
        "Without temporal features": [col for col in X_train_combined.columns if col not in 
                                    ['patch_idx', 'champ_prev_win', 'champ_roll3', 
                                     'win_trend_1', 'win_trend_2', 'win_volatility',
                                     'rel_to_champ_mean', 'rel_to_patch_mean', 'rel_to_global_mean']],
        
        "Temporal features only": ['patch_idx', 'champ_prev_win', 'champ_roll3', 
                                 'win_trend_1', 'win_trend_2', 'win_volatility',
                                 'rel_to_champ_mean', 'rel_to_patch_mean', 'rel_to_global_mean']
    }
    
    # Affichage du nombre de features par groupe
    for group_name, features in feature_groups.items():
        available_features = [f for f in features if f in X_train_combined.columns]
        print(f"{group_name}: {len(available_features)} features")
    
    # Exécution des tests pour chaque groupe
    results = {}
    all_features_pred = None  # Pour stocker les prédictions du test "All features"
    
    for group_name, features in feature_groups.items():
        print(f"\nTest: {group_name}")
        
        # Filtrer les caractéristiques qui existent dans le DataFrame
        features = [f for f in features if f in X_train_combined.columns]
        
        if len(features) == 0:
            print("  Aucune feature disponible, skip")
            continue
        
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
        
        # 🔥 GARDE LES PRÉDICTIONS DU TEST "All features"
        if group_name == "All features":
            all_features_pred = y_pred.copy()
        
        # Métriques
        r2 = r2_score(y_test, y_pred, sample_weight=w_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))
        
        results[group_name] = {
            'r2': r2,
            'rmse': rmse,
            'n_features': len(features)
        }
        
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
    
    print("\n" + "="*50)
    print("ANALYSE GLOBALE - TEST AVEC TOUTES LES FEATURES:")
    print("="*50)
    
    if all_features_pred is not None:
        analyze_by_change_status(y_test, all_features_pred, full_df.loc[X_test.index])
        
        # 🔥 NOUVEAU : Sauvegarde pour visualisation
        change_analysis = save_change_status_analysis(y_test, all_features_pred, full_df.loc[X_test.index])
        
    else:
        print("Erreur: Pas de prédictions pour 'All features'")
    
    return results

if __name__ == "__main__":
    run_ablation_study()