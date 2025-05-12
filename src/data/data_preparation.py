import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Any

# Configuration du logging
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
        return patch.strip()
    if len(parts) >= 3:
        return f"{parts[0].strip()}.{parts[1].strip()}"
    return patch.strip()

def patch_to_tuple(patch: str):
    """
    Convertit une chaîne de patch (ex. '14.10') en un tuple d'entiers pour faciliter le tri.
    Si la chaîne ne respecte pas le format 'nombre.nombre', retourne (0, 0) sans lever d'erreur.
    """
    if re.fullmatch(r'\d+\.\d+', patch):
        return tuple(map(int, patch.split('.')))
    else:
        # On renvoie un tuple par défaut et on peut loguer si nécessaire (ici on le supprime pour éviter le bruit)
        return (0, 0)

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

def prepare_prediction_data(temporal_split=True) -> Dict[str, Any]:
    """
    Prépare les données pour la prédiction avec option de séparation temporelle.
    """
    logger.info("Starting data preparation")
    
    engine = create_engine('sqlite:///../datasets/league_data.db')
    
    # Récupération des changements liés aux champions (base_stats, per_level, abilities)
    champ_changes_query = """
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
    champ_changes = pd.read_sql(champ_changes_query, engine)
    if champ_changes.empty:
        raise ValueError("No champion patch changes data found")
    
    # Récupération des changements liés aux items (stat_type = 'item')
    item_changes_query = """
    SELECT 
        to_patch as patch,
        stat_type,
        stat_name,
        change_value
    FROM patch_changes 
    WHERE stat_type = 'item'
      AND change_value IS NOT NULL
    """
    item_changes = pd.read_sql(item_changes_query, engine)
    
    # Récupération des winrates
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
    
    # Formatage et nettoyage des numéros de patch
    champ_changes['patch'] = champ_changes['patch'].apply(format_patch_number)
    item_changes['patch'] = item_changes['patch'].apply(format_patch_number)
    winrates['patch'] = winrates['patch'].apply(format_patch_number)
    
    # Forcer la conversion en chaîne et enlever les espaces
    champ_changes['patch'] = champ_changes['patch'].astype(str).str.strip()
    item_changes['patch'] = item_changes['patch'].astype(str).str.strip()
    winrates['patch'] = winrates['patch'].astype(str).str.strip()
    
    # Ne conserver que les lignes dont le patch respecte exactement le format "nombre.nombre"
    champ_changes = champ_changes[champ_changes['patch'].str.fullmatch(r'\d+\.\d+')]
    item_changes = item_changes[item_changes['patch'].str.fullmatch(r'\d+\.\d+')]
    winrates = winrates[winrates['patch'].str.fullmatch(r'\d+\.\d+')]
    
    # On garde uniquement les patchs présents dans winrates
    sorted_winrates = sorted(winrates['patch'].unique())
    champ_changes = champ_changes[champ_changes['patch'].isin(sorted_winrates)]
    item_changes = item_changes[item_changes['patch'].isin(sorted_winrates)]
    
    logger.info(f"Shape before merge - champ_changes: {champ_changes.shape}")
    logger.info(f"Shape before merge - item_changes: {item_changes.shape}")
    logger.info(f"Shape before merge - winrates: {winrates.shape}")
    
    # Création d'une matrice pivot pour les changements champions
    champ_matrix = pd.pivot_table(
        champ_changes,
        index=['patch', 'champion_name'],
        columns=['stat_type', 'stat_name'],
        values='change_value',
        fill_value=0
    )
    champ_matrix.columns = [f'{col[0]}_{col[1]}' for col in champ_matrix.columns]
    champ_matrix = champ_matrix.reset_index()
    
    # Création d'une matrice pivot pour les changements items (index uniquement par patch)
    item_matrix = pd.pivot_table(
        item_changes,
        index=['patch'],
        columns=['stat_type', 'stat_name'],
        values='change_value',
        fill_value=0
    )
    item_matrix.columns = [f'{col[0]}_{col[1]}' for col in item_matrix.columns]
    item_matrix = item_matrix.reset_index()
    
    # Fusionner les deux matrices sur 'patch'
    merged_changes = pd.merge(champ_matrix, item_matrix, on='patch', how='left')
    merged_changes.fillna(0, inplace=True)
    
    # Fusionner avec les données de winrates
    final_df = pd.merge(
        merged_changes,
        winrates,
        on=['patch', 'champion_name'],
        how='inner'
    )
    
    # Filtrer de nouveau pour s'assurer que la colonne patch est au bon format
    final_df = final_df[final_df['patch'].str.fullmatch(r'\d+\.\d+')]
    
    logger.info(f"Shape after merge: {final_df.shape}")
    logger.info(f"Final patches: {sorted(final_df['patch'].unique())}")
    
    validate_data(final_df)
    
    # Trier par champion et par patch en utilisant patch_to_tuple
    final_df = final_df.sort_values(
        by=["champion_name", "patch"],
        key=lambda x: x.map(patch_to_tuple) if x.name == "patch" else x
    )
    
    # Calculer la différence de winrate (delta_winrate) pour chaque champion
    final_df["delta_winrate"] = final_df.groupby("champion_name")["winrate"].diff()
    
    # Retirer les lignes dont delta_winrate est NaN (première occurrence de chaque champion)
    final_df = final_df.dropna(subset=["delta_winrate"])
    
    feature_cols = [col for col in merged_changes.columns if col not in ['patch', 'champion_name']]
    X = final_df[feature_cols]
    y = final_df['delta_winrate']
    weights = final_df['total_games'] / final_df['total_games'].mean()
    
    # MODIFICATION IMPORTANTE: Division train/test avec option temporelle
    if temporal_split:
        # Tri chronologique des patches
        patches = sorted(final_df['patch'].unique(), key=patch_to_tuple)
        # 80% des patches pour l'entraînement
        split_idx = int(len(patches) * 0.8)
        train_patches = patches[:split_idx]
        
        # Création des masques train/test
        train_mask = final_df['patch'].isin(train_patches)
        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]
        w_train = weights[train_mask]
        w_test = weights[~train_mask]
        
        logger.info(f"Temporal split: {len(train_patches)} patches for training, {len(patches) - len(train_patches)} for testing")
    else:
        # Split classique
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
        logger.info("Random train/test split")
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
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
            correlation = df[feature].corr(df['delta_winrate'])
            logger.info(f"{feature}:")
            logger.info(f"  Non-zero changes: {non_zero}")
            logger.info(f"  Mean change: {mean_change:.4f}")
            logger.info(f"  Correlation with delta winrate: {correlation:.4f}")

if __name__ == "__main__":
    try:
        data = prepare_prediction_data(temporal_split=True)
        print("\nTraining Data Shape:", data['X_train'].shape)
        print("Number of Features:", len(data['feature_names']))
        print("\nFeatures:", data['feature_names'])
        analyze_features(data)
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")