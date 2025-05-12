import numpy as np
import pandas as pd
import logging
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from data_preparation import prepare_prediction_data

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def add_interaction_features(X):
    """Ajoute des caractéristiques d'interaction ciblées."""
    X = X.copy()
    
    # Interaction entre statistiques de base et tendances
    if all(col in X.columns for col in ['base_stat_total_change', 'win_trend_1']):
        X['base_stat_trend_interaction'] = X['base_stat_total_change'] * np.sign(X['win_trend_1'])
    
    # Interaction entre cooldown et dégâts (ratio efficacité)
    if all(col in X.columns for col in ['total_damage_change', 'total_cooldown_change']):
        X['damage_per_cooldown'] = X['total_damage_change'] / (X['total_cooldown_change'].abs() + 0.01)
    
    # Direction du changement (buff ou nerf)
    total_cols = [col for col in X.columns if col.startswith('total_') and col.endswith('_change')]
    if total_cols:
        X['is_buff'] = X[total_cols].sum(axis=1) > 0
        X['is_buff'] = X['is_buff'].astype(int)
    
    return X

def identify_important_features(X_train, y_train, w_train, threshold=0.005):
    """Identifie les caractéristiques les plus importantes en utilisant XGBoost."""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    
    model.fit(X_train, y_train, sample_weight=w_train)
    
    # Sélection des caractéristiques importantes
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    selected = importances[importances['importance'] > threshold]['feature'].tolist()
    
    return selected, importances

def main():
    # Important: Division temporelle des données
    logger.info("Chargement et préparation des données avec division temporelle...")
    data = prepare_prediction_data(temporal_split=True)
    full_df = data['full_data']
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    w_train, w_test = data['w_train'], data['w_test']

    # 1) Ajout de caractéristiques temporelles
    logger.info("Ajout de caractéristiques temporelles...")
    X_train = add_temporal_features(full_df.loc[X_train.index], X_train)
    X_test = add_temporal_features(full_df.loc[X_test.index], X_test)
    
    # Ajout des métriques de base
    for col in ['pickrate', 'total_games']:
        X_train[col] = full_df.loc[X_train.index, col]
        X_test[col] = full_df.loc[X_test.index, col]
    
    # 2) Agrégation des caractéristiques liées aux compétences
    logger.info("Création de caractéristiques agrégées...")
    agg_train = aggregate_ability_changes(X_train)
    agg_test = aggregate_ability_changes(X_test)
    
    # 3) Ajout des caractéristiques d'interaction
    logger.info("Ajout des caractéristiques d'interaction...")
    X_train_combined = pd.concat([X_train, agg_train], axis=1)
    X_test_combined = pd.concat([X_test, agg_test], axis=1)
    
    X_train_combined = add_interaction_features(X_train_combined)
    X_test_combined = add_interaction_features(X_test_combined)
    
    # 4) Identification des caractéristiques importantes
    logger.info("Identification des caractéristiques importantes...")
    important_features, feature_importance = identify_important_features(X_train_combined, y_train, w_train)
    logger.info(f"Sélection de {len(important_features)} caractéristiques importantes")
    
    # 5) Réduction aux caractéristiques importantes
    X_train_selected = X_train_combined[important_features]
    X_test_selected = X_test_combined[important_features]
    
    logger.info(f"Dimensions finales après sélection: {X_train_selected.shape}")
    
    # 6) Normalisation des données
    logger.info("Normalisation des caractéristiques...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected.fillna(0))
    X_test_scaled = scaler.transform(X_test_selected.fillna(0))
    
    # 7) Validation croisée temporelle
    logger.info("Évaluation par validation croisée temporelle...")
    tscv = TimeSeriesSplit(n_splits=5)
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
    
    # Validation croisée avec découpe temporelle 
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=tscv, scoring='r2', 
        fit_params={'sample_weight': w_train}
    )
    
    logger.info(f"Scores R² de validation croisée: {cv_scores}")
    logger.info(f"R² moyen de validation croisée: {cv_scores.mean():.3f}")
    
    # 8) Entraînement du modèle final
    logger.info("Entraînement du modèle final...")
    model.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    # 9) Évaluation sur l'ensemble de test
    logger.info("Évaluation sur l'ensemble de test...")
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False, sample_weight=w_test)
    mae = mean_absolute_error(y_test, y_pred, sample_weight=w_test)
    r2 = r2_score(y_test, y_pred, sample_weight=w_test)
    
    logger.info(f"Résultats finaux: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
    
    # 10) Analyse des caractéristiques importantes
    logger.info("Top 15 des caractéristiques les plus importantes:")
    for i, row in feature_importance.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 11) Sauvegarde du modèle
    logger.info("Sauvegarde du modèle final...")
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'important_features': important_features,
        'feature_importance': feature_importance.to_dict(),
        'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
        'cv_scores': cv_scores.tolist()
    }
    
    joblib.dump(model_package, 'winrate_model_package.joblib')
    
    logger.info("Modèle sauvegardé avec succès !")
    
    return model_package

if __name__ == '__main__':
    main()