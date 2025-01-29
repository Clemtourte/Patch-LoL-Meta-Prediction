import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from data_preparation import prepare_prediction_data
import matplotlib.pyplot as plt
import seaborn as sns

def train_xgboost_model():
    # Get prepared data
    data = prepare_prediction_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    w_train = data['w_train']
    w_test = data['w_test']
    feature_names = data['feature_names']
    
    # Define XGBoost parameters for grid search
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0]
    }
    
    # Create base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    
    # Fit the model with sample weights
    grid_search.fit(X_train, y_train, sample_weight=w_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print("\nBest parameters:", grid_search.best_params_)
    
    # Make predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train, sample_weight=w_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test, sample_weight=w_test))
    train_r2 = r2_score(y_train, y_pred_train, sample_weight=w_train)
    test_r2 = r2_score(y_test, y_pred_test, sample_weight=w_test)
    
    print("\nModel Performance:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Feature importance analysis
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Winrate')
    plt.ylabel('Predicted Winrate')
    plt.title('Test Set: Actual vs Predicted')
    
    plt.subplot(122)
    plt.hist(y_pred_test - y_test, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'model': best_model,
        'importance': importance_df,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    }

if __name__ == "__main__":
    results = train_xgboost_model()
    print("\nTop 5 most important features:")
    print(results['importance'].head())