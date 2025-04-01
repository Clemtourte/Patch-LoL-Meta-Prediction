import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from data_preparation import prepare_prediction_data
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_correlation_heatmap(df: pd.DataFrame, stat_columns: list) -> None:
    """Trace la heatmap des corrélations entre les features et le winrate"""
    plt.figure(figsize=(12, 10))
    corr_matrix = df[stat_columns + ['winrate']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../graphs/correlation_heatmap.png')
    plt.close()

def analyze_feature_groups(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyse les features en les regroupant par catégories.
    Ici, on distingue les changements liés aux champions et ceux liés aux items.
    """
    groups = {
        'champion': [col for col in df.columns if col.startswith('base_stat_') 
                     or col.startswith('per_level_') 
                     or col.startswith('ability_')],
        'item': [col for col in df.columns if col.startswith('item_')]
    }
    results = {}
    for group_name, features in groups.items():
        valid_features = [f for f in features if f in df.columns]
        if valid_features:
            changes = df[valid_features].astype(bool).sum()
            correlations = df[valid_features + ['winrate']].corr()['winrate'].sort_values(ascending=False)
            results[group_name] = {
                'total_changes': changes.sum(),
                'correlations': correlations
            }
    return results

def analyze_changes_over_time(df: pd.DataFrame, feature_names: list) -> None:
    """Trace l'évolution du nombre de changements par patch"""
    changes_per_patch = df.groupby('patch')[feature_names].apply(lambda x: (x != 0).sum())
    plt.figure(figsize=(15, 5))
    changes_per_patch.sum(axis=1).plot(kind='bar')
    plt.title('Number of Changes per Patch')
    plt.xlabel('Patch')
    plt.ylabel('Number of Changes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../graphs/changes_per_patch.png')
    plt.close()

def analyze_significance(df: pd.DataFrame, stat_columns: list) -> pd.DataFrame:
    """Analyse statistique de la significativité des changements"""
    significant_changes = []
    for col in stat_columns:
        if df[col].any():
            mask = df[col] != 0
            if mask.sum() < 2:
                continue
            before = df.loc[mask, 'winrate'] - df.loc[mask, col]
            after = df.loc[mask, 'winrate']
            if len(before) > 1:
                t_stat, p_value = stats.ttest_rel(before, after)
                effect_size = (after.mean() - before.mean()) / before.std() if before.std() != 0 else np.nan
                significant_changes.append({
                    'stat': col,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'sample_size': mask.sum()
                })
    return pd.DataFrame(significant_changes)

def analyze_prediction_data() -> Dict[str, Any]:
    logger.info("Starting data analysis")
    
    # Charge les données préparées, qui intègrent désormais les changements d’items
    data = prepare_prediction_data()
    df = data['full_data']
    stat_columns = data['feature_names']
    
    logger.info("\n=== Dataset Size ===")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Unique champions: {df['champion_name'].nunique()}")
    logger.info(f"Unique patches: {df['patch'].nunique()}")
    
    logger.info("\n=== Winrate Statistics ===")
    logger.info(df['winrate'].describe())
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df['winrate'], bins=30)
    plt.title('Winrate Distribution')
    plt.savefig('../graphs/winrate_distribution.png')
    plt.close()
    
    logger.info("\n=== Stat Changes Statistics ===")
    changes_stats = df[stat_columns].describe()
    logger.info(changes_stats)
    
    # Calcul des corrélations pour chaque feature (en ne considérant que les valeurs non nulles)
    correlations = []
    for col in stat_columns:
        mask = df[col] != 0
        if mask.sum() < 2:
            continue
        corr, p_val = stats.pearsonr(df[col][mask], df['winrate'][mask])
        correlations.append({
            'stat': col,
            'correlation': corr,
            'p_value': p_val,
            'non_zero_changes': mask.sum()
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
    
    plot_correlation_heatmap(df, stat_columns)
    group_analysis = analyze_feature_groups(df)
    analyze_changes_over_time(df, stat_columns)
    significance = analyze_significance(df, stat_columns)
    
    if not corr_df.empty:
        top_stats = corr_df.head(5)
        plt.figure(figsize=(15, 10))
        for i, (_, row) in enumerate(top_stats.iterrows(), 1):
            stat = row['stat']
            plt.subplot(2, 3, i)
            non_zero_data = df[df[stat] != 0]
            sns.scatterplot(data=non_zero_data, x=stat, y='winrate')
            plt.title(f'{stat}\nr={row["correlation"]:.3f}')
        plt.tight_layout()
        plt.savefig('../graphs/top_correlations.png')
        plt.close()
    
    results = {
        'correlations': corr_df,
        'stats': changes_stats,
        'winrate_stats': df['winrate'].describe(),
        'sample_sizes': {
            'total_samples': len(df),
            'unique_champions': df['champion_name'].nunique(),
            'unique_patches': df['patch'].nunique()
        },
        'group_analysis': group_analysis,
        'significance': significance
    }
    
    sig_changes = significance[significance['p_value'] < 0.05]
    logger.info("\n=== Key Insights ===")
    logger.info(f"- {len(sig_changes)} out of {len(significance)} changes are statistically significant")
    if not corr_df.empty:
        top_change = corr_df.iloc[0]
        logger.info(f"- Most correlated change: {top_change['stat']}")
        logger.info(f"- Correlation: {top_change['correlation']:.3f}")
        logger.info(f"- Changes analyzed: {top_change['non_zero_changes']}")
    logger.info(f"- Average winrate: {df['winrate'].mean():.2f}%")
    logger.info(f"- Winrate standard deviation: {df['winrate'].std():.2f}%")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_prediction_data()
        print("\n=== Feature Group Analysis ===")
        for group, analysis in results['group_analysis'].items():
            print(f"\n{group.title()} Features:")
            print(f"Total changes: {analysis['total_changes']}")
            print("Correlations with winrate:")
            print(analysis['correlations'])
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
