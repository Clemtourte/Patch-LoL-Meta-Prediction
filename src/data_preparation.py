import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def format_patch_number(patch: str) -> str:
    parts = patch.split('.')
    if len(parts) == 2:
        return patch
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}"
    return patch

def validate_data(final_df: pd.DataFrame) -> None:
    if final_df.empty:
        raise ValueError("No data after merging")
    if final_df.isna().any().any():
        nan_cols = final_df.columns[final_df.isna().any()].tolist()
        logger.warning(f"Dataset contains NaN values in columns: {nan_cols}")
    logger.info("Features distribution:")
    for col in final_df.columns:
        if col not in ['patch', 'champion_name']:
            non_zero = (final_df[col] != 0).sum()
            logger.info(f"{col}: {non_zero} non-zero values")

def prepare_prediction_data() -> Dict[str, Any]:
    logger.info("Starting data preparation")
    engine = create_engine('sqlite:///../datasets/league_data.db')
    
    patch_changes_query = """
    SELECT 
        to_patch as patch,
        champion_name,
        stat_type,
        stat_name,
        change_value
    FROM patch_changes 
    WHERE stat_type IN ('base_stat', 'per_level', 'ability')
      AND change_value IS NOT NULL
    """
    patch_changes = pd.read_sql(patch_changes_query, engine)
    if patch_changes.empty:
        raise ValueError("No patch changes data found")
    
    winrates_query = """
    SELECT 
        patch,
        champion_name,
        winrate,
        pickrate,
        total_games
    FROM champion_winrates
    """
    winrates = pd.read_sql(winrates_query, engine)
    if winrates.empty:
        raise ValueError("No winrate data found")
    
    patch_changes['patch'] = patch_changes['patch'].apply(format_patch_number)
    winrates['patch'] = winrates['patch'].apply(format_patch_number)
    
    sorted_patch_changes = sorted(patch_changes['patch'].unique())
    sorted_winrates = sorted(winrates['patch'].unique())
    matching_patches = sorted(set(sorted_patch_changes) & set(sorted_winrates))
    
    logger.info(f"Patches in changes: {sorted_patch_changes}")
    logger.info(f"Patches in winrates: {sorted_winrates}")
    logger.info(f"Matching patches: {matching_patches}")
    
    if not matching_patches:
        raise ValueError("No matching patches found between changes and winrates")
    
    logger.info(f"Shape before merge - patch_changes: {patch_changes.shape}")
    logger.info(f"Shape before merge - winrates: {winrates.shape}")
    
    patch_matrix = pd.pivot_table(
        patch_changes,
        index=['patch', 'champion_name'],
        columns=['stat_type', 'stat_name'],
        values='change_value',
        fill_value=0
    )
    patch_matrix.columns = [f'{col[0]}_{col[1]}' for col in patch_matrix.columns]
    patch_matrix = patch_matrix.reset_index()
    
    final_df = pd.merge(
        patch_matrix,
        winrates,
        on=['patch', 'champion_name'],
        how='inner'
    )
    
    logger.info(f"Shape after merge: {final_df.shape}")
    logger.info(f"Final patches: {sorted(final_df['patch'].unique())}")
    
    validate_data(final_df)
    
    feature_cols = [col for col in patch_matrix.columns if col not in ['patch', 'champion_name']]
    X = final_df[feature_cols]
    y = final_df['winrate']
    weights = final_df['total_games'] / final_df['total_games'].mean()
    
    logger.info("Splitting into train/test sets")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
    logger.info("Data preparation completed")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'w_train': w_train,
        'w_test': w_test,
        'feature_names': feature_cols,
        'full_data': final_df
    }

def analyze_features(data: Dict[str, Any]) -> None:
    df = data['full_data']
    features = data['feature_names']
    logger.info("\nFeature Analysis:")
    for feature in features:
        non_zero = (df[feature] != 0).sum()
        if non_zero > 0:
            mean_change = df[feature][df[feature] != 0].mean()
            correlation = df[feature].corr(df['winrate'])
            logger.info(f"{feature}:")
            logger.info(f"  Non-zero changes: {non_zero}")
            logger.info(f"  Mean change: {mean_change:.4f}")
            logger.info(f"  Correlation with winrate: {correlation:.4f}")

if __name__ == "__main__":
    try:
        data = prepare_prediction_data()
        print("\nTraining Data Shape:", data['X_train'].shape)
        print("Number of Features:", len(data['feature_names']))
        print("\nFeatures:", data['feature_names'])
        analyze_features(data)
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
