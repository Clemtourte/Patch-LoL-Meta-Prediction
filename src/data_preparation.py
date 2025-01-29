import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split

def format_patch_number(patch):
    # Split by dots
    parts = patch.split('.')
    
    # If it's already in the correct format (like "13.10")
    if len(parts) == 2:
        return patch
    
    # If it's in format "13.10.1"
    if len(parts) == 3:
        return f"{parts[0]}.{parts[1]}"
    
    return patch

def prepare_prediction_data():
    # Connect to database
    engine = create_engine('sqlite:///../datasets/league_data.db')
    
    # Get patch changes data
    patch_changes_query = """
    SELECT 
        to_patch as patch,
        champion_name,
        stat_type,
        stat_name,
        change_value
    FROM patch_changes 
    WHERE stat_type IN ('base_stat', 'per_level')
    """
    patch_changes = pd.read_sql(patch_changes_query, engine)
    
    # Get win rates data
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
    
    # Fix patch format in patch_changes
    patch_changes['patch'] = patch_changes['patch'].apply(format_patch_number)
    
    # Sort patches for better comparison
    sorted_patch_changes = sorted(patch_changes['patch'].unique())
    sorted_winrates = sorted(winrates['patch'].unique())
    
    print("\nNumber of unique patches in patch_changes:", len(sorted_patch_changes))
    print("Number of unique patches in winrates:", len(sorted_winrates))
    print("\nPatches in patch_changes:", sorted_patch_changes)
    print("\nPatches in winrates:", sorted_winrates)
    print("\nPatches that match:", sorted(set(sorted_patch_changes) & set(sorted_winrates)))
    print("\nShape before merge - patch_changes:", patch_changes.shape)
    print("Shape before merge - winrates:", winrates.shape)
    
    # Pivot patch changes
    patch_matrix = pd.pivot_table(
        patch_changes,
        index=['patch', 'champion_name'],
        columns=['stat_type', 'stat_name'],
        values='change_value',
        fill_value=0
    )
    
    # Flatten column names
    patch_matrix.columns = [f'{col[0]}_{col[1]}' for col in patch_matrix.columns]
    patch_matrix = patch_matrix.reset_index()
    
    # Merge with winrates
    final_df = pd.merge(
        patch_matrix,
        winrates,
        on=['patch', 'champion_name'],
        how='inner'
    )
    
    print("\nShape after merge - final_df:", final_df.shape)
    print("\nFinal patches in dataset:", sorted(final_df['patch'].unique()))
    print("\nSample merged data:")
    print(final_df.head())
    
    # Get feature columns
    feature_cols = [col for col in patch_matrix.columns 
                   if col not in ['patch', 'champion_name']]
    
    X = final_df[feature_cols]
    y = final_df['winrate']
    
    # Optional: Add sample weights based on number of games
    weights = final_df['total_games'] / final_df['total_games'].mean()
    
    # Create train/test split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
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

if __name__ == "__main__":
    data = prepare_prediction_data()
    print("\nFinal training data shape:", data['X_train'].shape)
    print("Number of features:", len(data['feature_names']))
    print("\nFeatures:", data['feature_names'])