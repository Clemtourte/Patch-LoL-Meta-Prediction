# model_benchmark.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
import sys
import os

# Ajouter le chemin vers le module de données
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from data_preparation import prepare_prediction_data

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

def aggregate_ability_changes(X, ability_types=None):
    """Agrège les changements des compétences pour réduire la dimensionnalité."""
    if ability_types is None:
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
    
    # Calcul global par type d'effet
    agg_features['total_damage_change'] = sum(agg_features.get(f'{a}_damage_change', 0) for a in ability_types)
    agg_features['total_cooldown_change'] = sum(agg_features.get(f'{a}_cooldown_change', 0) for a in ability_types)
    agg_features['total_mana_change'] = sum(agg_features.get(f'{a}_mana_change', 0) for a in ability_types)
    
    return pd.DataFrame(agg_features, index=X.index)

def prepare_features(data):
    """Prépare toutes les features pour l'entraînement."""
    full_df = data['full_data']
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    w_train, w_test = data['w_train'], data['w_test']

    # AJOUTE CES LIGNES pour être cohérent avec validation_ablation :
    X_train = add_temporal_features(full_df.loc[X_train.index], X_train)
    X_test = add_temporal_features(full_df.loc[X_test.index], X_test)
    
    for col in ['pickrate', 'total_games']:
        X_train[col] = full_df.loc[X_train.index, col]
        X_test[col] = full_df.loc[X_test.index, col]
    
    # Agrégation des caractéristiques liées aux compétences
    agg_train = aggregate_ability_changes(X_train)
    agg_test = aggregate_ability_changes(X_test)
    
    X_train_combined = pd.concat([X_train, agg_train], axis=1)
    X_test_combined = pd.concat([X_test, agg_test], axis=1)
    
    return X_train_combined, X_test_combined, y_train, y_test, w_train, w_test

def define_models():
    """Définit les modèles à comparer."""
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        ),
        
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        
        'Ridge Regression': Ridge(
            alpha=1.0,
            random_state=42
        )
    }
    
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test, w_train, w_test, model_name):
    """Évalue un modèle et retourne les métriques."""
    print(f"\nEntraînement de {model_name}...")
    
    start_time = time.time()
    
    try:
        # Entraînement avec ou sans sample_weight selon le modèle
        if model_name in ['Ridge Regression']:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=w_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques avec ou sans sample_weight
        if model_name in ['Ridge Regression']:
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
        else:
            r2 = r2_score(y_test, y_pred, sample_weight=w_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))
            mae = mean_absolute_error(y_test, y_pred, sample_weight=w_test)
        
        training_time = time.time() - start_time
        
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Temps d'entraînement: {training_time:.2f}s")
        
        return {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Training_Time': training_time,
            'Status': 'Success'
        }
        
    except Exception as e:
        print(f"  Erreur: {str(e)}")
        return {
            'R²': np.nan,
            'RMSE': np.nan,
            'MAE': np.nan,
            'Training_Time': np.nan,
            'Status': f'Error: {str(e)}'
        }

def create_benchmark_visualizations(results_df):
    """Crée les visualisations du benchmark."""
    
    # Configuration du style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure avec 3 sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Benchmark Results', fontsize=16, fontweight='bold')
    
    # Couleurs pour 3 modèles
    colors = ['darkblue', 'darkgreen', 'orange']
    
    # 1. R² Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(results_df['Model'], results_df['R²'], color=colors)
    ax1.set_title('R² Score Comparison', fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.grid(True, alpha=0.3)
    
    # Annotations pour R²
    for bar, r2 in zip(bars1, results_df['R²']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. RMSE Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(results_df['Model'], results_df['RMSE'], color=colors)
    ax2.set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
    ax2.set_ylabel('RMSE')
    ax2.grid(True, alpha=0.3)
    
    # Annotations pour RMSE
    for bar, rmse in zip(bars2, results_df['RMSE']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rmse:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. MAE Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(results_df['Model'], results_df['MAE'], color=colors)
    ax3.set_title('MAE Comparison (Lower is Better)', fontweight='bold')
    ax3.set_ylabel('MAE')
    ax3.grid(True, alpha=0.3)
    
    # Annotations pour MAE
    for bar, mae in zip(bars3, results_df['MAE']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mae:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Training Time Comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(results_df['Model'], results_df['Training_Time'], color=colors)
    ax4.set_title('Training Time Comparison', fontweight='bold')
    ax4.set_ylabel('Time (seconds)')
    ax4.grid(True, alpha=0.3)
    
    # Annotations pour Training Time
    for bar, time_val in zip(bars4, results_df['Training_Time']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Rotation des labels pour tous les graphiques
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_table(results_df):
    """Crée un tableau récapitulatif des performances."""
    
    # Tri par R² décroissant
    results_sorted = results_df.sort_values('R²', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Données du tableau
    table_data = []
    for _, row in results_sorted.iterrows():
        table_data.append([
            row['Model'],
            f"{row['R²']:.4f}",
            f"{row['RMSE']:.4f}",
            f"{row['MAE']:.4f}",
            f"{row['Training_Time']:.1f}s",
            "✓" if row['Status'] == 'Success' else "✗"
        ])
    
    headers = ['Model', 'R²', 'RMSE', 'MAE', 'Training Time', 'Status']
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.15, 0.15, 0.15, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style du header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E8B57')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style des lignes alternées (sans couleur spéciale pour le meilleur)
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Model Performance Comparison Summary', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('model_benchmark_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_sorted

def run_model_benchmark():
    """Fonction principale pour exécuter le benchmark."""
    print("=" * 60)
    print("BENCHMARK DE MODÈLES - LEAGUE OF LEGENDS PATCH PREDICTION")
    print("=" * 60)
    
    # Chargement des données
    print("\n1. Chargement et préparation des données...")
    data = prepare_prediction_data(temporal_split=True)
    X_train, X_test, y_train, y_test, w_train, w_test = prepare_features(data)
    
    print(f"   Données d'entraînement: {X_train.shape}")
    print(f"   Données de test: {X_test.shape}")
    print(f"   Nombre de features: {X_train.shape[1]}")
    
    # Normalisation
    print("\n2. Normalisation des données...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_test_scaled = scaler.transform(X_test.fillna(0))
    
    # Définition des modèles
    models = define_models()
    print(f"\n3. Modèles à évaluer: {list(models.keys())}")
    
    # Évaluation de chaque modèle
    print("\n4. Évaluation des modèles...")
    results = {}
    
    for model_name, model in models.items():
        result = evaluate_model(model, X_train_scaled, X_test_scaled, 
                              y_train, y_test, w_train, w_test, model_name)
        results[model_name] = result
    
    # Création du DataFrame de résultats
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'Model'}, inplace=True)
    
    # Sauvegarde des résultats
    results_df.to_csv('model_benchmark_results.csv', index=False)
    print(f"\n5. Résultats sauvegardés dans 'model_benchmark_results.csv'")
    
    # Création des visualisations
    print("\n6. Création des visualisations...")
    create_benchmark_visualizations(results_df)
    
    print("\n7. Création du tableau récapitulatif...")
    final_table = create_performance_table(results_df)
    
    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ DU BENCHMARK")
    print("=" * 60)
    
    best_model = final_table.iloc[0]
    print(f"🏆 Meilleur modèle: {best_model['Model']}")
    print(f"   R²: {best_model['R²']:.4f}")
    print(f"   RMSE: {best_model['RMSE']:.4f}")
    print(f"   MAE: {best_model['MAE']:.4f}")
    
    print(f"\n📊 Fichiers générés:")
    print(f"   - model_benchmark_results.csv")
    print(f"   - model_benchmark_comparison.png")
    print(f"   - model_benchmark_table.png")
    
    return results_df

if __name__ == "__main__":
    results = run_model_benchmark()