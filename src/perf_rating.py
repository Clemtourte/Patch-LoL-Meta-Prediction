import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, select, text, inspect, func
from sqlalchemy.orm import sessionmaker
from models import Match, Team, Participant, PerformanceFeatures, Base
import logging

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

        correlations = df[features].corrwith(df['win'])
        logging.info("Feature correlations with win:")
        for feature, corr in correlations.items():
            logging.info(f"{feature}: {corr}")

        df_standardized = (df[features] - df[features].mean()) / df[features].std()
        
        logging.info("Standardized features stats:")
        for feature in features:
            logging.info(f"{feature}: Mean = {df_standardized[feature].mean():.4f}, Std = {df_standardized[feature].std():.4f}")

        df['performance_score'] = (df_standardized * correlations).sum(axis=1) / correlations.sum()

        logging.info(f"Performance score stats: Mean = {df['performance_score'].mean():.4f}, Std = {df['performance_score'].std():.4f}")
        logging.info(f"Performance score range: Min = {df['performance_score'].min():.4f}, Max = {df['performance_score'].max():.4f}")

        df['standardized_performance_score'] = stats.zscore(df['performance_score'])

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
        return df, summary
    except Exception as e:
        logging.error(f"Error in calculate_performance_ratings: {str(e)}")
        return pd.DataFrame(), {}
    finally:
        session.close()

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
    engine = create_engine("sqlite:///../datasets/matches.db")
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
        conn.close()

if __name__ == "__main__":
    create_performance_score_columns()
    df, summary_stats = calculate_performance_ratings()
    if not df.empty:
        print("Summary statistics of standardized performance scores:")
        for key, value in summary_stats.items():
            print(f"{key}: {value}")
        plot_score_distribution(df)
        print("Plots saved as 'score_distribution.png' and 'score_qq_plot.png'")
    else:
        print("No data available for analysis.")