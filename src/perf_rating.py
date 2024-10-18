import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, select, text, inspect, func
from sqlalchemy.orm import sessionmaker
from models import Match, Team, Participant, PerformanceFeatures, Base
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_performance_ratings():
    engine = create_engine("sqlite:///../datasets/league_data.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        query = select(
            Participant.participant_id,
            Participant.champion_name,
            Participant.position,
            Team.win,
            PerformanceFeatures.kill_participation,
            PerformanceFeatures.death_share,
            PerformanceFeatures.damage_share,
            PerformanceFeatures.damage_taken_share,
            PerformanceFeatures.gold_share,
            PerformanceFeatures.heal_share,
            PerformanceFeatures.damage_mitigated_share,
            PerformanceFeatures.cs_share,
            PerformanceFeatures.vision_share,
            PerformanceFeatures.vision_denial_share,
            PerformanceFeatures.xp_share,
            PerformanceFeatures.cc_share
        ).join(Team).join(PerformanceFeatures)

        df = pd.read_sql(query, session.connection())

        if df.empty:
            logging.warning("No data retrieved from the database.")
            return pd.DataFrame(), {}

        features = ['kill_participation', 'death_share', 'damage_share', 'damage_taken_share',
                    'gold_share', 'heal_share', 'damage_mitigated_share', 'cs_share',
                    'vision_share', 'vision_denial_share', 'xp_share', 'cc_share']

        # Data verification and logging
        log_data_info(df, features)

        # Calculate Spearman correlations
        spearman_correlations = df[features].apply(lambda x: x.corr(df['win'], method='spearman'))
        log_correlations(spearman_correlations, "Spearman")

        # Create non-linear transformations
        for feature in features:
            df[f'{feature}_squared'] = df[feature] ** 2
            df[f'{feature}_sqrt'] = np.sqrt(df[feature])
        
        # Create interaction terms
        for i, feature1 in enumerate(features):
            for feature2 in features[i+1:]:
                df[f'{feature1}_{feature2}_interaction'] = df[feature1] * df[feature2]

        # Prepare features for machine learning
        X = df[features + [f'{f}_squared' for f in features] + 
               [f'{f}_sqrt' for f in features] + 
               [col for col in df.columns if 'interaction' in col]]
        y = df['win']

        # Perform feature importance analysis
        feature_importance = perform_feature_importance(X, y)
        log_feature_importance(feature_importance, X.columns)

        # Calculate performance score using feature importance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        df['performance_score'] = np.dot(X_scaled, feature_importance)

        logging.info(f"Performance score stats: Mean = {df['performance_score'].mean():.4f}, Std = {df['performance_score'].std():.4f}")
        logging.info(f"Performance score range: Min = {df['performance_score'].min():.4f}, Max = {df['performance_score'].max():.4f}")

        df['standardized_performance_score'] = stats.zscore(df['performance_score'])

        # Update database
        for index, row in df.iterrows():
            participant = session.get(Participant, row['participant_id'])
            if participant:
                participant.performance_score = row['performance_score']
                participant.standardized_performance_score = row['standardized_performance_score']

        session.commit()

        summary_stats = df['standardized_performance_score'].describe()
        summary = {
            'Count': f"{summary_stats['count']:,.0f}",
            'Mean': f"{summary_stats['mean']:.6f}",
            'Standard Deviation': f"{summary_stats['std']:.6f}",
            'Minimum': f"{summary_stats['min']:.6f}",
            '25th Percentile': f"{summary_stats['25%']:.6f}",
            'Median': f"{summary_stats['50%']:.6f}",
            '75th Percentile': f"{summary_stats['75%']:.6f}",
            'Maximum': f"{summary_stats['max']:.6f}"
        }

        logging.info(f"Processed performance ratings for {len(df)} participants.")
        
        # Create visualizations
        create_visualizations(df, features, spearman_correlations, feature_importance, X.columns)
        
        return df, summary
    except Exception as e:
        logging.error(f"Error in calculate_performance_ratings: {str(e)}")
        return pd.DataFrame(), {}
    finally:
        session.close()

def log_data_info(df, features):
    logging.info("Data types:")
    logging.info(df.dtypes)
    logging.info("Unique values in 'win' column:")
    logging.info(df['win'].unique())
    logging.info("Data ranges:")
    for column in df.columns:
        logging.info(f"{column}: {df[column].min()} to {df[column].max()}")
    logging.info("Mean values grouped by win:")
    logging.info(df.groupby('win')[features].mean())

def log_correlations(correlations, method):
    logging.info(f"{method} correlations with win:")
    for feature, corr in correlations.items():
        logging.info(f"{feature}: {corr}")

def perform_feature_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf.feature_importances_

def log_feature_importance(feature_importance, feature_names):
    logging.info("Feature importances:")
    for feature, importance in zip(feature_names, feature_importance):
        logging.info(f"{feature}: {importance}")

def create_visualizations(df, features, spearman_correlations, feature_importance, feature_names):
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../graphs/correlation_heatmap.png')
    plt.close()

    # Feature importance plot
    plt.figure(figsize=(12, 10))
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(20)  # Top 20 features
    
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('../graphs/feature_importance.png')
    plt.close()

    # Distribution of Standardized Performance Scores
    plt.figure(figsize=(10, 6))
    plt.hist(df['standardized_performance_score'], bins=50, edgecolor='black')
    plt.title('Distribution of Standardized Performance Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig('../graphs/score_distribution.png')
    plt.close()

    # Q-Q plot
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(df['standardized_performance_score'], dist="norm", plot=ax)
    ax.set_title("Q-Q plot of Standardized Performance Scores")
    plt.savefig('../graphs/score_qq_plot.png')
    plt.close()

def calculate_champion_stats(session, position=None):
    metrics = [
        'kill_participation', 'death_share', 'damage_share', 'damage_taken_share',
        'gold_share', 'heal_share', 'damage_mitigated_share', 'cs_share',
        'vision_share', 'vision_denial_share', 'xp_share', 'cc_share'
    ]
    
    query = session.query(
        Participant.champion_name,
        Participant.position,
        func.count(Participant.participant_id).label('games'),
        func.avg(Team.win).label('win_rate')
    )
    
    for metric in metrics:
        query = query.add_columns(
            func.avg(getattr(PerformanceFeatures, metric)).label(f'avg_{metric}'),
            func.avg(func.pow(getattr(PerformanceFeatures, metric), 2)).label(f'avg_sq_{metric}'),
            func.group_concat(getattr(PerformanceFeatures, metric)).label(f'values_{metric}'),
            func.group_concat(Team.win).label(f'wins_{metric}')
        )
    
    query = query.join(Team).join(PerformanceFeatures)
    
    if position:
        query = query.filter(Participant.position == position)
    
    query = query.group_by(Participant.champion_name, Participant.position)
    
    df = pd.read_sql(query.statement, session.bind)
    
    for metric in metrics:
        df[f'std_{metric}'] = np.sqrt(df[f'avg_sq_{metric}'] - df[f'avg_{metric}']**2)
        
        def calc_corr(row):
            values = np.array([float(x) for x in row[f'values_{metric}'].split(',')])
            wins = np.array([float(x) for x in row[f'wins_{metric}'].split(',')])
            return np.corrcoef(values, wins)[0, 1]
        
        df[f'corr_{metric}'] = df.apply(calc_corr, axis=1)
        
        df = df.drop(columns=[f'avg_sq_{metric}', f'values_{metric}', f'wins_{metric}'])
    
    return df

def compare_champions_advanced(champion_stats):
    champions = champion_stats['champion_name'].unique()
    metrics = ['kill_participation', 'death_share', 'damage_share', 'damage_taken_share',
               'gold_share', 'heal_share', 'damage_mitigated_share', 'cs_share',
               'vision_share', 'vision_denial_share', 'xp_share', 'cc_share']
    
    results = []
    for champ_x in champions:
        x_stats = champion_stats[champion_stats['champion_name'] == champ_x].iloc[0]
        for champ_y in champions:
            if champ_x != champ_y:
                y_stats = champion_stats[champion_stats['champion_name'] == champ_y].iloc[0]
                performance = []
                for metric in metrics:
                    if x_stats[f'std_{metric}'] != 0:
                        perf = (y_stats[f'avg_{metric}'] - x_stats[f'avg_{metric}']) / x_stats[f'std_{metric}']
                        perf *= x_stats[f'corr_{metric}']
                        performance.append(perf)
                avg_performance = np.mean(performance) if performance else 0
                results.append({
                    'champion_x': champ_x,
                    'champion_y': champ_y,
                    'performance': avg_performance
                })
    return pd.DataFrame(results)

def analyze_champion_performance(session, min_games=10):
    positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    all_results = []

    for position in positions:
        champion_stats = calculate_champion_stats(session, position)
        
        champion_stats = champion_stats[champion_stats['games'] >= min_games]
        
        if not champion_stats.empty:
            comparisons = compare_champions_advanced(champion_stats)
            
            ratings = comparisons.groupby('champion_x')['performance'].mean().reset_index()
            ratings = ratings.rename(columns={'performance': 'rating'})
            
            final_stats = champion_stats.merge(ratings, left_on='champion_name', right_on='champion_x')
            final_stats = final_stats[['champion_name', 'games', 'win_rate', 'rating', 'position']]
            final_stats['win_rate'] = final_stats['win_rate'] * 100 
            final_stats = final_stats.sort_values('rating', ascending=False)
            
            all_results.append(final_stats)
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

def plot_score_distribution(df):
    if df.empty:
        logging.warning("No data available to plot score distribution.")
        return

    plt.figure(figsize=(10,6))
    plt.hist(df['standardized_performance_score'], bins=50, edgecolor='black')
    plt.title('Distribution of Standardized Performance Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig('../graphs/score_distribution.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,6))
    stats.probplot(df['standardized_performance_score'], dist="norm", plot=ax)
    ax.set_title("Q-Q plot of Standardized Performance Scores")
    plt.savefig('../graphs/score_qq_plot.png')
    plt.close()

def create_performance_score_columns():
    engine = create_engine("sqlite:///../datasets/league_data.db")
    inspector = inspect(engine)
    
    if not inspector.has_table('participants'):
        Base.metadata.create_all(engine)
    else:
        with engine.connect() as conn:
            columns = inspector.get_columns('participants')
            column_names = [col['name'] for col in columns]
        
            if 'performance_score' not in column_names:
                conn.execute(text("ALTER TABLE participants ADD COLUMN performance_score FLOAT"))
            if 'standardized_performance_score' not in column_names:
                conn.execute(text("ALTER TABLE participants ADD COLUMN standardized_performance_score FLOAT"))
            conn.commit()

if __name__ == "__main__":
    create_performance_score_columns()
    df, summary_stats = calculate_performance_ratings()
    if not df.empty:
        print("Summary statistics of standardized performance scores:")
        for key, value in summary_stats.items():
            print(f"{key}: {value}")
        plot_score_distribution(df)
        print("Plots saved as 'score_distribution.png', 'score_qq_plot.png', 'correlation_heatmap.png', and 'feature_importance.png'")
    else:
        print("No data available for analysis.")