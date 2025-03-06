import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from data_preparation import prepare_prediction_data
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer additional features for the model"""
    # Group related changes
    df['total_offensive_changes'] = df[[col for col in df.columns 
        if any(x in col for x in ['damage', 'attack'])]].abs().sum(axis=1)
    
    df['total_defensive_changes'] = df[[col for col in df.columns 
        if any(x in col for x in ['hp', 'armor', 'spellblock'])]].abs().sum(axis=1)
    
    df['total_utility_changes'] = df[[col for col in df.columns 
        if any(x in col for x in ['cooldown', 'cost', 'movespeed'])]].abs().sum(axis=1)
    
    # Add per-ability changes
    abilities = ['Q', 'W', 'E', 'R']
    for ability in abilities:
        df[f'total_{ability}_changes'] = df[[col for col in df.columns 
            if f'ability_{ability}' in col]].abs().sum(axis=1)
    
    # Add interaction terms for abilities
    for ability in abilities:
        damage_col = f'ability_{ability}_base_damage'
        cooldown_col = f'ability_{ability}_cooldown'
        if damage_col in df.columns and cooldown_col in df.columns:
            df[f'{ability}_damage_cooldown'] = df[damage_col] * df[cooldown_col]
    
    return df

def train_xgboost_model() -> Dict[str, Any]:
    """Train and evaluate XGBoost model"""
    logger.info("Starting model training")
    
    # Get and prepare data
    data = prepare_prediction_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    w_train = data['w_train']
    w_test = data['w_test']
    
    # Engineer features
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)
    feature_names = X_train.columns.tolist()
    
    # Define model parameters
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0.1, 1.0]
    }
    
    # Create base model - removed early stopping and tree_method
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        enable_categorical=True  # Add this for better handling of categorical features
    )
    
    # Perform grid search
    logger.info("Starting grid search")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1  # Reduced verbosity
    )
    
    # Simple fit without extra parameters
    grid_search.fit(X_train, y_train, sample_weight=w_train)
    
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, w_train, w_test)
    logger.info("Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Generate plots
    plot_feature_importance(best_model, feature_names)
    plot_predictions(y_test, y_pred_test)
    plot_error_analysis(y_test, y_pred_test)
    
    return {
        'model': best_model,
        'metrics': metrics,
        'feature_importance': get_feature_importance(best_model, feature_names),
        'predictions': {
            'train': y_pred_train,
            'test': y_pred_test
        }
    }

def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, w_train, w_test) -> Dict[str, float]:
    """Calculate model performance metrics"""
    return {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train, sample_weight=w_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test, sample_weight=w_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train, sample_weight=w_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test, sample_weight=w_test),
        'train_r2': r2_score(y_train, y_pred_train, sample_weight=w_train),
        'test_r2': r2_score(y_test, y_pred_test, sample_weight=w_test)
    }

def plot_feature_importance(model: xgb.XGBRegressor, feature_names: list) -> None:
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_predictions(y_test, y_pred_test) -> None:
    """Plot actual vs predicted values"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Winrate')
    plt.ylabel('Predicted Winrate')
    plt.title('Test Set: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def plot_error_analysis(y_test, y_pred_test) -> None:
    """Plot error distribution"""
    errors = y_pred_test - y_test
    plt.figure(figsize=(10, 6))
    
    plt.subplot(121)
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    
    plt.subplot(122)
    sns.scatterplot(x=y_test, y=errors)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Winrate')
    plt.ylabel('Prediction Error')
    plt.title('Error vs Actual Value')
    
    plt.tight_layout()
    plt.savefig('error_analysis.png')
    plt.close()

def get_feature_importance(model: xgb.XGBRegressor, feature_names: list) -> pd.DataFrame:
    """Get feature importance DataFrame"""
    return pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

if __name__ == "__main__":
    try:
        results = train_xgboost_model()
        
        # Print top features
        print("\nTop 10 most important features:")
        print(results['feature_importance'].head(10))
        
        # Save results
        results['feature_importance'].to_csv('feature_importance.csv', index=False)
        
        # Print key metrics
        print("\nTest Set Metrics:")
        print(f"RMSE: {results['metrics']['test_rmse']:.4f}")
        print(f"MAE: {results['metrics']['test_mae']:.4f}")
        print(f"R²: {results['metrics']['test_r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")