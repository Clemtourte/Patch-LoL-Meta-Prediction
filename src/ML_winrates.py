import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from data_preparation import prepare_prediction_data

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features agrégées qui regroupent des changements offensifs, défensifs et utilitaires.
    On ajoute également des features spécifiques par capacité (Q, W, E, R) et des interactions.
    """
    # Exemple : agrégation simple sur des features dont le nom contient 'damage' ou 'attack'
    df['total_offensive_changes'] = df[[col for col in df.columns if 'damage' in col or 'attack' in col]].abs().sum(axis=1)
    df['total_defensive_changes'] = df[[col for col in df.columns if 'hp' in col or 'armor' in col or 'spellblock' in col]].abs().sum(axis=1)
    df['total_utility_changes'] = df[[col for col in df.columns if 'cooldown' in col or 'movespeed' in col]].abs().sum(axis=1)
    
    # Exemple d'interaction simple : produit entre certaines features si disponibles
    for ability in ['Q', 'W', 'E', 'R']:
        damage_col = f'ability_{ability}_base_damage_rank1'
        cooldown_col = f'ability_{ability}_cooldown_rank1'
        if damage_col in df.columns and cooldown_col in df.columns:
            df[f'{ability}_damage_cooldown'] = df[damage_col] * df[cooldown_col]
    
    # Pour les items, on peut agréger certains changements
    if any(col.startswith('item_') for col in df.columns):
        df['total_item_gold'] = df[[col for col in df.columns if col.startswith('item_')]].sum(axis=1)
    
    return df

def get_models():
    # Définition des grilles d'hyperparamètres pour chaque modèle
    param_grid_xgb = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0.1, 1.0]
    }
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    param_grid_ridge = {'alpha': [0.1, 1, 10]}
    param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1]}
    param_grid_en = {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.2, 0.5, 0.8]}

    models = {
        'XGBoost': (xgb.XGBRegressor(objective='reg:squarederror', random_state=42, enable_categorical=True), param_grid_xgb),
        'RandomForest': (RandomForestRegressor(random_state=42), param_grid_rf),
        'Ridge': (Ridge(random_state=42), param_grid_ridge),
        'Lasso': (Lasso(random_state=42), param_grid_lasso),
        'ElasticNet': (ElasticNet(random_state=42), param_grid_en)
    }
    return models

def evaluate_model(model, X_train, y_train, X_test, y_test, w_train, w_test):
    # Prédictions et calcul des métriques
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train, sample_weight=w_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test, sample_weight=w_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train, sample_weight=w_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test, sample_weight=w_test),
        'train_r2': r2_score(y_train, y_pred_train, sample_weight=w_train),
        'test_r2': r2_score(y_test, y_pred_test, sample_weight=w_test)
    }
    return metrics, y_pred_test

def main():
    logger.info("Chargement des données...")
    # La fonction prepare_prediction_data() renvoie désormais le delta winrate comme cible.
    data = prepare_prediction_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']  # delta winrate
    y_test = data['y_test']
    w_train = data['w_train']
    w_test = data['w_test']

    # Appliquer l'ingénierie de features
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)
    feature_names = X_train.columns.tolist()
    logger.info(f"Nombre de features après ingénierie: {len(feature_names)}")
    
    models = get_models()
    results = {}

    # Utiliser 5-fold cross-validation pour chaque modèle
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, (model, param_grid) in models.items():
        logger.info(f"Recherche des meilleurs hyperparamètres pour {model_name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train, sample_weight=w_train)
        best_model = grid_search.best_estimator_
        metrics, y_pred_test = evaluate_model(best_model, X_train, y_train, X_test, y_test, w_train, w_test)
        results[model_name] = {
            'best_params': grid_search.best_params_,
            'metrics': metrics,
            'model': best_model
        }
        logger.info(f"{model_name} - Meilleurs paramètres: {grid_search.best_params_}")
        logger.info(f"{model_name} - Performance sur l'ensemble de test: RMSE = {metrics['test_rmse']:.4f}, MAE = {metrics['test_mae']:.4f}, R² = {metrics['test_r2']:.4f}")
    
    # Comparaison finale des modèles
    summary = pd.DataFrame({
        model_name: result['metrics'] for model_name, result in results.items()
    }).T
    print("\nComparaison des modèles:")
    print(summary)
    
    # Sauvegarde du résumé
    summary.to_csv("model_comparison_summary.csv", index=True)

if __name__ == "__main__":
    main()
