import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set style for academic paper
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define the results directory path - all CSV files are in src folder
RESULTS_DIR = Path(".")  # Current directory (src)

def load_results_data():
    """Load all results data from CSV files"""
    try:
        # Load ablation study results
        ablation_df = pd.read_csv("ablation_study_results.csv", index_col=0)
        
        # Load error analysis results
        error_by_class_df = pd.read_csv("error_analysis_by_class.csv")
        error_by_champion_df = pd.read_csv("error_analysis_by_champion.csv")
        
        # Load temporal validation results
        temporal_df = pd.read_csv("nonconsecutive_validation_results.csv", index_col=0)
        
        return ablation_df, error_by_class_df, error_by_champion_df, temporal_df
    except FileNotFoundError as e:
        print(f"Error loading results: {e}")
        print("Available CSV files in current directory:")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        for f in csv_files:
            print(f"  - {f}")
        print("\nMake sure you have run the validation scripts first to generate the CSV files.")
        return None, None, None, None

def plot_ablation_study(ablation_df):
    """Visualisation des résultats d'ablation study avec vraies données"""
    if ablation_df is None:
        print("No ablation data available")
        return
    
    # Prepare data
    df = ablation_df.reset_index()
    df.columns = ['Feature Group', 'R²', 'RMSE', 'Features']
    df = df.sort_values('R²', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Couleurs conditionnelles basées sur les vraies valeurs
    colors = ['red' if r2 < 0 else 'lightcoral' if r2 < 0.4 else 'lightblue' if r2 < 0.6 else 'darkblue' 
              for r2 in df['R²']]
    
    bars = ax.barh(df['Feature Group'], df['R²'], color=colors, alpha=0.8)
    
    # Lignes de référence
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    full_model_r2 = df[df['Feature Group'] == 'Toutes les caractéristiques']['R²'].iloc[0]
    ax.axvline(x=full_model_r2, color='green', linestyle='--', alpha=0.7, 
               label=f'Full Model Performance (R²={full_model_r2:.3f})')
    
    # Annotations avec nombre de features
    for i, (bar, features) in enumerate(zip(bars, df['Features'])):
        width = bar.get_width()
        ax.text(width + 0.01 if width > 0 else width - 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}\n({features} features)', 
                ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Ablation Study: Impact on Model Performance', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.3, max(df['R²']) + 0.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_by_class(error_by_class_df, error_by_champion_df):
    """Distribution des erreurs par classe de champion avec vraies données"""
    if error_by_class_df is None:
        print("No error analysis data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: MAE par classe avec taille des échantillons
    df_sorted = error_by_class_df.sort_values('mae')
    
    scatter = ax1.scatter(df_sorted['mae'], range(len(df_sorted)), 
                         s=df_sorted['count']*20, 
                         c=df_sorted['mae'], cmap='Reds', alpha=0.7)
    
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['champion_class'])
    ax1.set_xlabel('Mean Absolute Error (percentage points)', fontweight='bold')
    ax1.set_title('Prediction Error by Champion Class', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Annotations
    for i, (mae, class_name, count) in enumerate(zip(df_sorted['mae'], df_sorted['champion_class'], df_sorted['count'])):
        ax1.annotate(f'n={count}', (mae, i), xytext=(5, 0), 
                    textcoords='offset points', va='center', fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('MAE (percentage points)', rotation=270, labelpad=15)
    
    # Plot 2: Distribution des échantillons
    total_samples = df_sorted['count'].sum()
    colors = plt.cm.Set3(np.linspace(0, 1, len(df_sorted)))
    wedges, texts, autotexts = ax2.pie(df_sorted['count'], labels=df_sorted['champion_class'], 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    
    ax2.set_title(f'Sample Distribution by Champion Class\n(Total: {total_samples} samples)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('error_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_validation(temporal_df):
    """Résultats de validation temporelle avec vraies données"""
    if temporal_df is None:
        print("No temporal validation data available")
        return
    
    # Prepare data
    df = temporal_df.reset_index()
    df.columns = ['Train → Test', 'R²', 'RMSE', 'MAE', 'Train Samples', 'Test Samples']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: R² scores
    colors = ['lightcoral' if r2 < 0.2 else 'orange' if r2 < 0.4 else 'lightgreen' for r2 in df['R²']]
    bars1 = ax1.bar(range(len(df)), df['R²'], color=colors, alpha=0.8)
    
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Train → Test'], rotation=45, ha='right')
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('Cross-Epoch Validation Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Ligne de référence (meilleur score)
    best_r2 = df['R²'].max()
    ax1.axhline(y=best_r2, color='red', linestyle='--', alpha=0.7, 
                label=f'Best Performance (R²={best_r2:.3f})')
    
    # Annotations
    for bar, r2 in zip(bars1, df['R²']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.legend()
    
    # Plot 2: Sample sizes
    x = np.arange(len(df))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, df['Train Samples'], width, label='Training', alpha=0.8, color='steelblue')
    bars3 = ax2.bar(x + width/2, df['Test Samples'], width, label='Testing', alpha=0.8, color='orange')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Train → Test'], rotation=45, ha='right')
    ax2.set_ylabel('Number of Samples', fontweight='bold')
    ax2.set_title('Training and Testing Sample Sizes', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_validation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(ablation_df):
    """Importance des features calculée à partir des résultats d'ablation"""
    if ablation_df is None:
        print("No ablation data available")
        return
    
    # Récupérer le R² du modèle complet
    full_model_r2 = ablation_df.loc['Toutes les caractéristiques', 'r2']
    
    # Calculer l'impact de chaque catégorie (différence quand on l'enlève)
    categories = [
        ('Temporal Features\n(Previous WR, Trends)', 'Sans caractéristiques temporelles'),
        ('Ability Changes\n(Damage, Cooldown)', 'Sans changements d\'aptitudes'),
        ('Champion Base Stats\n(HP, Armor, AD)', 'Sans statistiques de champion'),
        ('Item Modifications\n(Cost, Stats)', 'Sans changements d\'objets'),
        ('Relative Positioning\n(vs Patch Mean)', 'Sans caractéristiques relatives')
    ]
    
    # Calculer l'importance comme différence de performance
    importance_data = []
    colors = ['darkred', 'darkblue', 'darkgreen', 'purple', 'orange']
    
    for (category_name, ablation_name), color in zip(categories, colors):
        if ablation_name in ablation_df.index:
            without_r2 = ablation_df.loc[ablation_name, 'r2']
            importance = full_model_r2 - without_r2
            importance_data.append((category_name, importance, color))
    
    # Trier par importance
    importance_data.sort(key=lambda x: x[1], reverse=True)
    
    categories, importance, colors = zip(*importance_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(categories, importance, color=colors, alpha=0.7)
    
    # Ligne de référence
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    
    # Annotations
    for bar, imp in zip(bars, importance):
        width = bar.get_width()
        ax.text(width + 0.005 if width > 0 else width - 0.005, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left' if width > 0 else 'right', 
                va='center', fontweight='bold')
    
    ax.set_xlabel('Performance Impact (Δ R² when removed)', fontweight='bold')
    ax.set_title('Feature Category Importance Analysis\n(Impact on Model Performance)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_table(ablation_df):
    """Tableau récapitulatif des performances avec vraies données"""
    if ablation_df is None:
        print("No ablation data available")
        return
    
    # Récupérer les métriques du modèle complet
    full_model = ablation_df.loc['Toutes les caractéristiques']
    r2_score = full_model['r2']
    rmse_score = full_model['rmse']
    n_features = int(full_model['n_features'])
    
    performance_data = {
        'Metric': ['R²', 'RMSE', 'MAE', 'Features Used', 'Training Method', 'Validation Method'],
        'Value': [
            f'{r2_score:.4f}', 
            f'{rmse_score:.4f}', 
            'See error analysis', 
            n_features, 
            'First 80% patches (temporal)', 
            'Last 20% patches (temporal)'
        ],
        'Interpretation': [
            f'{r2_score*100:.1f}% of variance explained',
            f'±{rmse_score:.2f} percentage points error',
            'Varies by champion class', 
            f'{n_features} engineered features',
            'Temporal split (chronological)',
            'Temporal split (chronological)'
        ]
    }
    
    df = pd.DataFrame(performance_data)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='left', loc='center', 
                    colWidths=[0.2, 0.3, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style alternating rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_distribution(error_by_champion_df):
    """Visualisation de la distribution des erreurs"""
    if error_by_champion_df is None:
        print("No champion error data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribution des erreurs
    errors = error_by_champion_df['error'].dropna()
    ax1.hist(errors, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax1.axvline(errors.mean(), color='red', linestyle='--', 
                label=f'Mean Error: {errors.mean():.3f}')
    ax1.axvline(0, color='green', linestyle='-', label='Perfect Prediction')
    
    ax1.set_xlabel('Prediction Error (percentage points)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Distribution of Prediction Errors', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Erreurs absolues vs vraies valeurs
    abs_errors = error_by_champion_df['abs_error'].dropna()
    true_values = error_by_champion_df['y_true'].dropna()
    
    ax2.scatter(true_values, abs_errors, alpha=0.6, color='steelblue')
    ax2.set_xlabel('True Winrate Change (percentage points)', fontweight='bold')
    ax2.set_ylabel('Absolute Error (percentage points)', fontweight='bold')
    ax2.set_title('Absolute Error vs True Values', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(true_values, abs_errors, 1)
    p = np.poly1d(z)
    ax2.plot(true_values, p(true_values), "r--", alpha=0.8, label='Trend')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_all_visualizations():
    """Génère toutes les visualisations pour le mémoire avec vraies données"""
    
    print("Loading results data...")
    ablation_df, error_by_class_df, error_by_champion_df, temporal_df = load_results_data()
    
    if ablation_df is None:
        print("Cannot proceed without results data. Please run validation scripts first.")
        return
    
    print("Generating visualizations for thesis...")
    
    print("1. Ablation Study...")
    plot_ablation_study(ablation_df)
    
    print("2. Error Analysis by Champion Class...")
    plot_error_by_class(error_by_class_df, error_by_champion_df)
    
    print("3. Temporal Validation...")
    plot_temporal_validation(temporal_df)
    
    print("4. Feature Importance...")
    plot_feature_importance(ablation_df)
    
    print("5. Performance Summary Table...")
    create_performance_table(ablation_df)
    
    print("6. Error Distribution...")
    plot_error_distribution(error_by_champion_df)
    
    print("All visualizations generated! Files saved as PNG.")

def show_available_data():
    """Affiche un résumé des données disponibles"""
    print("Checking available results files in src directory...")
    
    files_to_check = [
        "ablation_study_results.csv",
        "error_analysis_by_class.csv", 
        "error_analysis_by_champion.csv",
        "nonconsecutive_validation_results.csv"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f"✓ {file}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"✗ {file}: Missing")
    
    # Show all CSV files in current directory
    print("\nAll CSV files in src directory:")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    for f in csv_files:
        print(f"  - {f}")

# Exécuter
if __name__ == "__main__":
    print("=== Dynamic Visualization Script ===")
    show_available_data()
    print("\n" + "="*40 + "\n")
    generate_all_visualizations()