import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from data_preparation import prepare_prediction_data

def analyze_prediction_data():
    # Get the prepared data
    data = prepare_prediction_data()
    df = data['full_data']
    
    print("\n=== Dataset Size ===")
    print(f"Total samples: {len(df)}")
    print(f"Unique champions: {df['champion_name'].nunique()}")
    print(f"Unique patches: {df['patch'].nunique()}")
    
    print("\n=== Winrate Statistics ===")
    print(df['winrate'].describe())
    
    # Analyze stat changes
    stat_columns = data['feature_names']
    
    print("\n=== Stat Changes Statistics ===")
    changes_stats = df[stat_columns].describe()
    print(changes_stats)
    
    # Calculate correlations with winrate
    correlations = []
    for col in stat_columns:
        # Only calculate correlation for non-zero changes
        mask = df[col] != 0
        if mask.sum() > 0:  # Only if we have non-zero values
            corr = stats.pearsonr(df[col][mask], df['winrate'][mask])
            correlations.append({
                'stat': col,
                'correlation': corr[0],
                'p_value': corr[1],
                'non_zero_changes': mask.sum()
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
    
    print("\n=== Top Correlations with Winrate (Only Significant) ===")
    print(corr_df[corr_df['p_value'] < 0.05].to_string())
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Winrate distribution
    plt.subplot(131)
    sns.histplot(df['winrate'], bins=30)
    plt.title('Winrate Distribution')
    
    # If we have significant correlations, show the top one
    if len(corr_df) > 0:
        top_stat = corr_df.iloc[0]['stat']
        
        # Distribution of top correlated stat (non-zero values)
        plt.subplot(132)
        sns.histplot(df[df[top_stat] != 0][top_stat], bins=30)
        plt.title(f'Distribution of {top_stat}\n(non-zero values)')
        
        # Scatter plot of top correlation
        plt.subplot(133)
        sns.scatterplot(data=df[df[top_stat] != 0], x=top_stat, y='winrate')
        plt.title(f'Winrate vs {top_stat}\n(correlation: {corr_df.iloc[0]["correlation"]:.3f})')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'correlations': corr_df,
        'stats': changes_stats,
        'winrate_stats': df['winrate'].describe(),
        'sample_sizes': {
            'total_samples': len(df),
            'unique_champions': df['champion_name'].nunique(),
            'unique_patches': df['patch'].nunique()
        }
    }

if __name__ == "__main__":
    analysis_results = analyze_prediction_data()
    
    # Additional insights
    print("\n=== Key Insights ===")
    total_changes = analysis_results['correlations'].shape[0]
    significant_changes = analysis_results['correlations'][
        analysis_results['correlations']['p_value'] < 0.05
    ].shape[0]
    
    print(f"- {significant_changes} out of {total_changes} stat changes have significant correlation with winrate")
    if total_changes > 0:
        print(f"- Most impactful stat change: {analysis_results['correlations'].iloc[0]['stat']}")
        print(f"- Correlation: {analysis_results['correlations'].iloc[0]['correlation']:.3f}")
        print(f"- Number of changes: {analysis_results['correlations'].iloc[0]['non_zero_changes']}")
    print(f"- Average winrate: {analysis_results['winrate_stats']['mean']:.2f}%")
    print(f"- Winrate standard deviation: {analysis_results['winrate_stats']['std']:.2f}%")