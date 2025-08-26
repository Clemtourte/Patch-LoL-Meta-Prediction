# validation_error_distribution.py
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
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

# Définir les classes de champions
def assign_champion_class(champion_name):
    tanks = [
        'Maokai', 'Malphite', 'Sion', 'Ornn', 'Shen', 'Nautilus', 'Leona', 'Braum', 'Sejuani', 'Alistar',
        'Amumu', 'Blitzcrank', 'Cho\'Gath', 'Dr. Mundo', 'Galio', 'Gragas', 'Poppy', 'Rammus', 'Tahm Kench', 
        'Taric', 'Thresh', 'Volibear', 'Warwick', 'Zac', 'Rell', 'Nunu', 'Ivern', 'DrMundo','KSante','Singed','Skarner','TahmKench'
    ]
    
    fighters = [
        'Darius', 'Garen', 'Renekton', 'Irelia', 'Riven', 'Jax', 'Fiora', 'Camille', 'Aatrox', 'Lee Sin', 'Olaf', 'Wukong',
        'Gnar', 'Hecarim', 'Illaoi', 'Jarvan IV', 'Kayn', 'Kled', 'Mordekaiser', 'Nasus', 'Pantheon', 'Sett', 
        'Tryndamere', 'Udyr', 'Vi', 'Yorick', 'Yone', 'Ambessa', 'Briar', 'Gwen', 'Viego', 'MonkeyKing','LeeSin','Shyvana'
    ]
    
    mages = [
        'Ryze', 'Syndra', 'Orianna', 'Viktor', 'Cassiopeia', 'Annie', 'Veigar', 'Brand', 'Lux', 'Xerath', 'Ziggs', 'Vel\'Koz',
        'Ahri', 'Anivia', 'Azir', 'Elise', 'Fiddlesticks', 'Heimerdinger', 'Karthus', 'Kassadin', 'Katarina',
        'Kennen', 'LeBlanc', 'Lissandra', 'Malzahar', 'Neeko', 'Rumble', 'Swain', 'Taliyah', 'Twisted Fate', 'Vladimir',
        'Zoe', 'Zyra', 'Aurelion Sol', 'Lillia', 'Seraphine', 'Vex', 'Hwei', 'Aurora', 'AurelionSol', 'Sylas','Kayle'
    ]
    
    assassins = [
        'Zed', 'Akali', 'Katarina', 'Talon', 'Fizz', 'Kayn', 'Shaco', 'Evelynn', 'Kha\'Zix', 'Rengar',
        'Diana', 'Ekko', 'Kassadin', 'LeBlanc', 'Nocturne', 'Qiyana', 'Yasuo', 'Yone', 'Master Yi', 
        'Nidalee', 'Kindred','Khazix'
    ]
    
    marksmen = [
        'Ashe', 'Caitlyn', 'Jhin', 'Jinx', 'Vayne', 'Ezreal', 'Draven', 'Varus', 'Tristana', 'Kalista', 'Xayah', 'Kai\'Sa',
        'Aphelios', 'Corki', 'Graves', 'Jayce', 'Kog\'Maw', 'Lucian', 'Miss Fortune', 'Quinn', 'Sivir', 
        'Teemo', 'Twitch', 'Urgot', 'Zeri', 'Nilah', 'Smolder','KogMaw','MissFortune'
    ]
    
    supports = [
        'Soraka', 'Janna', 'Lulu', 'Nami', 'Sona', 'Yuumi', 'Karma', 'Seraphine', 'Bard', 'Thresh', 'Pyke',
        'Alistar', 'Blitzcrank', 'Braum', 'Leona', 'Morgana', 'Nautilus', 'Rakan', 'Senna', 'Shen', 'Taric', 
        'Zilean', 'Zyra', 'Milio', 'Renata Glasc', 'Galio','Renata'
    ]
    
    if champion_name in tanks:
        return 'Tank'
    elif champion_name in fighters:
        return 'Fighter'
    elif champion_name in mages:
        return 'Mage'
    elif champion_name in assassins:
        return 'Assassin'
    elif champion_name in marksmen:
        return 'Marksman'
    elif champion_name in supports:
        return 'Support'
    else:
        return 'Unknown'

def run_error_distribution_analysis():
    """
    Analyse la distribution des erreurs de prédiction par champion et classe.
    """
    print("Analyse de la Distribution des Erreurs")
    print("====================================")
    
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
    
    # Calcul des erreurs
    test_df = full_df.loc[X_test.index].copy()
    test_df['y_true'] = y_test
    test_df['y_pred'] = y_pred
    test_df['error'] = test_df['y_true'] - test_df['y_pred']
    test_df['abs_error'] = np.abs(test_df['error'])
    test_df['squared_error'] = test_df['error'] ** 2
    
    # Assigner les classes de champions
    test_df['champion_class'] = test_df['champion_name'].apply(assign_champion_class)
    
    # Afficher les métriques globales
    r2 = r2_score(y_test, y_pred, sample_weight=w_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))
    mae = mean_absolute_error(y_test, y_pred, sample_weight=w_test)
    
    print(f"Métriques globales:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # Analyse par classe de champion
    class_stats = test_df.groupby('champion_class').agg({
        'y_true': 'count',
        'abs_error': ['mean', 'median', 'std'],
        'squared_error': ['mean', 'std']
    }).reset_index()
    
    class_stats.columns = ['champion_class', 'count', 'mae', 'median_abs_error', 'std_abs_error', 'mse', 'std_squared_error']
    class_stats['rmse'] = np.sqrt(class_stats['mse'])
    
    print("\nAnalyse par Classe de Champion:")
    print(class_stats.to_string(index=False))
    
    # Top champions avec les erreurs les plus élevées
    champion_stats = test_df.groupby('champion_name').agg({
        'y_true': 'count',
        'abs_error': ['mean', 'median', 'max'],
        'squared_error': 'mean'
    }).reset_index()
    
    champion_stats.columns = ['champion_name', 'count', 'mae', 'median_abs_error', 'max_abs_error', 'mse']
    champion_stats['rmse'] = np.sqrt(champion_stats['mse'])
    champion_stats = champion_stats.sort_values('mae', ascending=False)
    
    print("\nTop 10 Champions avec les Erreurs les Plus Élevées:")
    print(champion_stats.head(10).to_string(index=False))
    
    # Top champions avec les erreurs les plus faibles
    print("\nTop 10 Champions avec les Erreurs les Plus Faibles:")
    print(champion_stats.tail(10).to_string(index=False))
    
    # Distribution des erreurs
    plt.figure(figsize=(12, 6))
    sns.histplot(test_df['error'], kde=True)
    plt.title('Distribution des Erreurs de Prédiction')
    plt.xlabel('Erreur (y_true - y_pred)')
    plt.ylabel('Fréquence')
    plt.axvline(0, color='r', linestyle='--')
    plt.savefig('error_distribution.png')
    plt.close()
    
    # Erreurs par classe de champion
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='champion_class', y='abs_error', data=test_df)
    plt.title('Distribution des Erreurs Absolues par Classe de Champion')
    plt.xlabel('Classe de Champion')
    plt.ylabel('Erreur Absolue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('error_by_class.png')
    plt.close()
    
    # Erreurs vs Delta Winrate
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='y_true', y='error', data=test_df, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Erreurs vs Delta Winrate Réel')
    plt.xlabel('Delta Winrate Réel')
    plt.ylabel('Erreur (y_true - y_pred)')
    plt.savefig('error_vs_delta_winrate.png')
    plt.close()
    
    # Erreurs vs Nombre de Jeux
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='total_games', y='abs_error', data=test_df, alpha=0.6)
    plt.title('Erreurs Absolues vs Nombre de Jeux')
    plt.xlabel('Nombre de Jeux')
    plt.ylabel('Erreur Absolue')
    plt.savefig('error_vs_games.png')
    plt.close()
    
    # Export des résultats
    test_df[['champion_name', 'champion_class', 'patch', 'y_true', 'y_pred', 'error', 'abs_error']].to_csv('error_analysis_by_champion.csv')
    class_stats.to_csv('error_analysis_by_class.csv')
    unknown_champions = test_df[test_df['champion_class'] == 'Unknown']['champion_name'].unique()
    print("Champions non classés:", unknown_champions)

    analyze_by_change_status(y_test, y_pred, full_df.loc[X_test.index])
    
    return test_df, class_stats, champion_stats


if __name__ == "__main__":
    run_error_distribution_analysis()
