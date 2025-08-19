import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results_data():
    """Load all validation results"""
    try:
        # Load ablation study results
        if os.path.exists("ablation_study_results.csv"):
            ablation_df = pd.read_csv("ablation_study_results.csv", index_col=0)
        else:
            print("Warning: ablation_study_results.csv not found")
            ablation_df = None
        
        # Load error analysis
        if os.path.exists("error_analysis_by_class.csv"):
            error_by_class_df = pd.read_csv("error_analysis_by_class.csv", index_col=0)
        else:
            print("Warning: error_analysis_by_class.csv not found")
            error_by_class_df = None
            
        if os.path.exists("error_analysis_by_champion.csv"):
            error_by_champion_df = pd.read_csv("error_analysis_by_champion.csv")
        else:
            print("Warning: error_analysis_by_champion.csv not found")
            error_by_champion_df = None
            
        # Load temporal validation
        if os.path.exists("nonconsecutive_validation_results.csv"):
            temporal_df = pd.read_csv("nonconsecutive_validation_results.csv", index_col=0)
        else:
            print("Warning: nonconsecutive_validation_results.csv not found")
            temporal_df = None
        
        return ablation_df, error_by_class_df, error_by_champion_df, temporal_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def plot_ablation_study(ablation_df):
    """Ablation study results visualization"""
    if ablation_df is None:
        print("No ablation data available")
        return
    
    # Data is already in English from validation script
    df = ablation_df.reset_index()
    df.columns = ['Feature Group', 'R²', 'RMSE', 'Features']
    df = df.sort_values('R²', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Conditional colors based on R² values
    colors = ['red' if r2 < 0 else 'lightcoral' if r2 < 0.4 else 'lightblue' if r2 < 0.6 else 'darkblue' 
              for r2 in df['R²']]
    
    bars = ax.barh(df['Feature Group'], df['R²'], color=colors, alpha=0.8)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    full_model_r2 = df[df['Feature Group'] == 'All features']['R²'].iloc[0]
    ax.axvline(x=full_model_r2, color='green', linestyle='--', alpha=0.7, 
               label=f'Full Model Performance (R²={full_model_r2:.3f})')
    
    # Annotations with feature count
    for i, (bar, features) in enumerate(zip(bars, df['Features'])):
        width = bar.get_width()
        ax.text(width + 0.01 if width > 0 else width - 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}\n({int(features)} features)', 
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
    """Error distribution by champion class"""
    if error_by_class_df is None:
        print("No error analysis data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: MAE by class with sample sizes
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
    
    # Plot 2: Sample distribution
    total_samples = df_sorted['count'].sum()
    colors = plt.cm.Set3(np.linspace(0, 1, len(df_sorted)))
    wedges, texts, autotexts = ax2.pie(df_sorted['count'], labels=df_sorted['champion_class'], 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    
    ax2.set_title(f'Sample Distribution by Champion Class\n(Total: {total_samples} samples)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('error_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_validation(temporal_df):
    """Temporal validation results"""
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
    
    # Reference line (best score)
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
    """Feature importance from ablation results"""
    if ablation_df is None:
        print("No ablation data available")
        return
    
    # Get full model R²
    full_model_r2 = ablation_df.loc['All features', 'r2']
    
    # Calculate impact of each category
    impact_data = []
    for idx in ablation_df.index:
        if idx != 'All features' and 'only' not in idx:
            if 'Without' in idx:
                without_r2 = ablation_df.loc[idx, 'r2']
                impact = full_model_r2 - without_r2
                clean_label = idx.replace('Without ', '')
                impact_data.append({
                    'Category': clean_label,
                    'Impact': impact,
                    'Percentage': (impact / full_model_r2) * 100
                })
    
    impact_df = pd.DataFrame(impact_data).sort_values('Impact', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['darkgreen' if imp > 0.2 else 'green' if imp > 0.1 else 'orange' if imp > 0.05 else 'red' 
              for imp in impact_df['Impact']]
    
    bars = ax.barh(impact_df['Category'], impact_df['Impact'], color=colors, alpha=0.8)
    
    # Annotations
    for bar, pct in zip(bars, impact_df['Percentage']):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', ha='left', va='center', fontweight='bold')
    
    ax.set_xlabel('Impact on R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Category Importance Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_table(ablation_df):
    """Performance summary table"""
    if ablation_df is None:
        print("No ablation data available")
        return
    
    # Get full model metrics
    full_model = ablation_df.loc['All features']
    r2_score = full_model['r2']
    rmse_score = full_model['rmse']
    n_features = int(full_model['n_features'])
    
    performance_data = {
        'Metric': ['R²', 'RMSE', 'MAE', 'Features Used', 'Training Method', 'Validation Method'],
        'Value': [
            f'{r2_score:.4f}', 
            f'{rmse_score:.4f}', 
            '0.4839',  # From ML_winrates output
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
    """Error distribution visualization"""
    if error_by_champion_df is None:
        print("No champion error data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Error distribution
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
    
    # Plot 2: Absolute errors vs true values
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

def plot_model_benchmark():
    """Model comparison visualization"""
    # Data from your comparison results
    models_data = {
        'Model': ['Random Forest', 'XGBoost', 'Ridge Regression'],
        'R²': [0.7437, 0.7321, 0.5363],
        'RMSE': [0.7716, 0.7889, 1.2533],
        'MAE': [0.4271, 0.4839, 0.7532]
    }
    
    df = pd.DataFrame(models_data)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: R² comparison
    colors = ['green', 'blue', 'orange']
    bars1 = ax1.bar(df['Model'], df['R²'], color=colors, alpha=0.7)
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_ylim(0, 1)
    
    for bar, val in zip(bars1, df['R²']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: RMSE comparison
    bars2 = ax2.bar(df['Model'], df['RMSE'], color=colors, alpha=0.7)
    ax2.set_ylabel('RMSE (percentage points)', fontweight='bold')
    ax2.set_title('Root Mean Squared Error', fontweight='bold')
    
    for bar, val in zip(bars2, df['RMSE']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: MAE comparison
    bars3 = ax3.bar(df['Model'], df['MAE'], color=colors, alpha=0.7)
    ax3.set_ylabel('MAE (percentage points)', fontweight='bold')
    ax3.set_title('Mean Absolute Error', fontweight='bold')
    
    for bar, val in zip(bars3, df['MAE']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Machine Learning Model Comparison for Patch Impact Prediction', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('model_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_all_visualizations():
    """Generate all visualizations for thesis"""
    
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
    
    print("7. Model Benchmark...")
    plot_model_benchmark()
    
    print("\nAll visualizations generated! Files saved as PNG.")
    print("Generated files:")
    print("- ablation_study.png")
    print("- error_by_class.png")
    print("- temporal_validation.png")
    print("- feature_importance.png")
    print("- performance_summary.png")
    print("- error_distribution.png")
    print("- model_benchmark.png")

def show_available_data():
    """Show summary of available data"""
    print("Checking available results files...")
    
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
    print("\nAll CSV files in current directory:")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    for f in csv_files:
        print(f"  - {f}")

# Execute
if __name__ == "__main__":
    print("=== LoL Patch Prediction - Visualization Generator ===")
    show_available_data()
    print("\n" + "="*50 + "\n")
    generate_all_visualizations()