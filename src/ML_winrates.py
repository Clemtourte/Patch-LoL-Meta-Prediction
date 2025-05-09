import numpy as np
import pandas as pd
import logging
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from data_preparation import prepare_prediction_data
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_temporal_features(df_full, X):
    """
    Add rolling, delta features, plus champ_mean and patch_mean.
    """
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
    
    # Ajout de caractéristiques de tendance
    df['win_trend'] = df.groupby('champion_name')['winrate'].diff()
    df['pick_trend'] = df.groupby('champion_name')['pickrate'].diff()
    
    # Volatilité
    df['win_volatility'] = (
        df.groupby('champion_name')['winrate']
          .rolling(3, min_periods=2)
          .std()
          .reset_index(level=0, drop=True)
    )

    # Moyennes
    champ_mean = df.groupby('champion_name')['winrate'].transform('mean')
    patch_mean = df.groupby('patch')['winrate'].transform('mean')
    global_mean = df['winrate'].mean()
    
    # Position relative par rapport à la moyenne globale
    df['relative_position'] = df['winrate'] - global_mean

    # Merge dans X
    X = X.copy()
    X['patch_idx'] = df.loc[X.index, 'patch_idx']
    X['champ_prev_win'] = df.loc[X.index, 'champ_prev_win'].fillna(global_mean)
    X['champ_roll3'] = df.loc[X.index, 'champ_roll3']
    X['champ_mean'] = champ_mean.loc[X.index]
    X['patch_mean'] = patch_mean.loc[X.index]
    X['champ_delta'] = df.loc[X.index, 'winrate'] - champ_mean.loc[X.index]
    X['win_trend'] = df.loc[X.index, 'win_trend'].fillna(0)
    X['pick_trend'] = df.loc[X.index, 'pick_trend'].fillna(0)
    X['win_volatility'] = df.loc[X.index, 'win_volatility'].fillna(0)
    X['relative_position'] = df.loc[X.index, 'relative_position']
    
    return X

def aggregate_ability_changes(X, ability_types=None):
    """
    Agrège les changements des compétences pour réduire la dimensionnalité et
    capturer l'essence des changements sans se perdre dans les détails.
    """
    if ability_types is None:
        ability_types = ['Passive', 'Q', 'W', 'E', 'R']
    
    agg_features = {}
    
    for ability in ability_types:
        # Créer des caractéristiques agrégées par type d'abilité
        damage_cols = [col for col in X.columns if f'ability_{ability}_base_damage' in col]
        cooldown_cols = [col for col in X.columns if f'ability_{ability}_cooldown' in col]
        mana_cols = [col for col in X.columns if f'ability_{ability}_mana_cost' in col]
        ap_ratio_cols = [col for col in X.columns if f'ability_{ability}_ap_ratio' in col]
        ad_ratio_cols = [col for col in X.columns if f'ability_{ability}_ad_ratio' in col or 
                                                    f'ability_{ability}_bonus_ad_ratio' in col]
        
        # Aggrégat de dégâts
        if damage_cols:
            agg_features[f'{ability}_damage_change'] = X[damage_cols].sum(axis=1)
        
        # Aggrégat de cooldown
        if cooldown_cols:
            agg_features[f'{ability}_cooldown_change'] = X[cooldown_cols].mean(axis=1)
        
        # Aggrégat de coût mana
        if mana_cols:
            agg_features[f'{ability}_mana_change'] = X[mana_cols].mean(axis=1)
            
        # Aggrégat de ratios AP/AD
        if ap_ratio_cols:
            agg_features[f'{ability}_ap_ratio_change'] = X[ap_ratio_cols].sum(axis=1)
        if ad_ratio_cols:
            agg_features[f'{ability}_ad_ratio_change'] = X[ad_ratio_cols].sum(axis=1)
    
    # Ajout d'agrégats pour les statistiques de base
    base_stat_cols = [col for col in X.columns if 'base_stat_' in col]
    per_level_cols = [col for col in X.columns if 'per_level_' in col]
    item_cols = [col for col in X.columns if 'item_' in col]
    
    agg_features['base_stat_total_change'] = X[base_stat_cols].sum(axis=1)
    agg_features['per_level_total_change'] = X[per_level_cols].sum(axis=1)
    agg_features['item_total_change'] = X[item_cols].sum(axis=1)
    
    # Calcul des changements nets
    agg_features['total_ability_damage_change'] = sum(agg_features.get(f'{a}_damage_change', 0) 
                                                     for a in ability_types)
    agg_features['total_cooldown_change'] = sum(agg_features.get(f'{a}_cooldown_change', 0) 
                                               for a in ability_types)
    agg_features['total_mana_change'] = sum(agg_features.get(f'{a}_mana_change', 0) 
                                           for a in ability_types)
    
    return pd.DataFrame(agg_features, index=X.index)

def main():
    logger.info("Loading and preparing data...")
    data = prepare_prediction_data()
    full_df = data['full_data']
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    w_train, w_test = data['w_train'], data['w_test']

    # 1) Temporal + basic features
    logger.info("Adding temporal & mean features...")
    X_train = add_temporal_features(full_df.loc[X_train.index], X_train)
    X_test = add_temporal_features(full_df.loc[X_test.index], X_test)
    
    # Ajout de caractéristiques de base
    for col in ['pickrate', 'total_games']:
        X_train[col] = full_df.loc[X_train.index, col]
        X_test[col] = full_df.loc[X_test.index, col]
    
    # 2) Création de caractéristiques agrégées (réduction de dimensionnalité)
    logger.info("Creating aggregated ability features...")
    agg_train = aggregate_ability_changes(X_train)
    agg_test = aggregate_ability_changes(X_test)
    
    # Fusionner avec les caractéristiques existantes
    important_cols = ['patch_idx', 'champ_prev_win', 'champ_roll3', 'champ_mean', 
                      'patch_mean', 'champ_delta', 'win_trend', 'pick_trend', 
                      'win_volatility', 'relative_position', 'pickrate', 'total_games']
    
    X_train_agg = pd.concat([X_train[important_cols], agg_train], axis=1)
    X_test_agg = pd.concat([X_test[important_cols], agg_test], axis=1)
    
    logger.info(f"Shape after aggregation: {X_train_agg.shape}")
    
    # 3) Normalisation des données
    logger.info("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_agg)
    X_test_scaled = scaler.transform(X_test_agg)
    
    # Reconvertir en DataFrame pour conserver les noms de colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_agg.index, columns=X_train_agg.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test_agg.index, columns=X_test_agg.columns)
    
    # 4) Premier modèle: XGBoost optimisé
    logger.info("Training XGBoost model with hyperparameter optimization...")
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 10.0],
    }
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)
    
    # Utilisation de GridSearchCV avec un échantillon des hyperparamètres pour l'efficacité
    quick_params = {
        'n_estimators': [100],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=quick_params,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train, sample_weight=w_train)
    best_xgb = grid_search.best_estimator_
    
    logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
    
    # 5) Deuxième modèle: RandomForest
    logger.info("Training RandomForest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    rf_model.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    # 6) Ensemble des deux modèles
    logger.info("Creating ensemble model...")
    ensemble = VotingRegressor(
        estimators=[
            ('xgb', best_xgb),
            ('rf', rf_model)
        ],
        weights=[0.7, 0.3]  # Donner plus de poids à XGBoost qui est généralement meilleur
    )
    
    ensemble.fit(X_train_scaled, y_train)
    
    # 7) Évaluation
    logger.info("Evaluating models on test set...")
    
    # Évaluation du modèle XGBoost
    y_pred_xgb = best_xgb.predict(X_test_scaled)
    rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False, sample_weight=w_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb, sample_weight=w_test)
    r2_xgb = r2_score(y_test, y_pred_xgb, sample_weight=w_test)
    
    # Évaluation du modèle RandomForest
    y_pred_rf = rf_model.predict(X_test_scaled)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False, sample_weight=w_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf, sample_weight=w_test)
    r2_rf = r2_score(y_test, y_pred_rf, sample_weight=w_test)
    
    # Évaluation de l'ensemble
    y_pred_ensemble = ensemble.predict(X_test_scaled)
    rmse_ensemble = mean_squared_error(y_test, y_pred_ensemble, squared=False, sample_weight=w_test)
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble, sample_weight=w_test)
    r2_ensemble = r2_score(y_test, y_pred_ensemble, sample_weight=w_test)
    
    logger.info(f"XGBoost Results: RMSE={rmse_xgb:.3f}, MAE={mae_xgb:.3f}, R2={r2_xgb:.3f}")
    logger.info(f"RandomForest Results: RMSE={rmse_rf:.3f}, MAE={mae_rf:.3f}, R2={r2_rf:.3f}")
    logger.info(f"Ensemble Results: RMSE={rmse_ensemble:.3f}, MAE={mae_ensemble:.3f}, R2={r2_ensemble:.3f}")
    
    # Analyse des caractéristiques importantes
    logger.info("Analyzing feature importance...")
    feature_importance = pd.DataFrame({
        'feature': X_train_scaled.columns,
        'importance': best_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 most important features:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Sauvegarde du meilleur modèle (l'ensemble) et des features
    joblib.dump(ensemble, 'ensemble_winrate_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    pd.Series(X_train_scaled.columns).to_csv('model_features.csv', index=False)
    
    # Sauvegarder la feature importance pour l'analyse future
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    logger.info("Model, scaler and features saved.")
    
    return {
        'xgb_model': best_xgb,
        'rf_model': rf_model,
        'ensemble_model': ensemble,
        'feature_importance': feature_importance,
        'metrics': {
            'xgb': {'rmse': rmse_xgb, 'mae': mae_xgb, 'r2': r2_xgb},
            'rf': {'rmse': rmse_rf, 'mae': mae_rf, 'r2': r2_rf},
            'ensemble': {'rmse': rmse_ensemble, 'mae': mae_ensemble, 'r2': r2_ensemble}
        }
    }

if __name__ == '__main__':
    main()