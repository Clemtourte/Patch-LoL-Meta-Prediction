import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, select, text, inspect, func
from sqlalchemy.orm import sessionmaker
from models import Match, Team, Participant, PerformanceFeatures, Base
import logging

logging.basicConfig(level=logging.INFO)

def calculate_performance_ratings():
    engine = create_engine('sqlite:///matches.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        query = select(
            Participant.participant_id,
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
        df_standardized = (df[features] - df[features].mean()) / df[features].std()
        df['performance_score'] = (df_standardized * correlations).sum(axis=1) / correlations.sum()
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

def calculate_champion_stats(session, summoner_name=None):
    try:
        metrics = [
            'kill_participation', 'death_share', 'damage_share', 'damage_taken_share',
            'gold_share', 'heal_share', 'damage_mitigated_share', 'cs_share',
            'vision_share', 'vision_denial_share', 'xp_share', 'cc_share'
        ]
        
        query = session.query(Participant.champion_name)
        
        for metric in metrics:
            query = query.add_columns(
                func.avg(getattr(PerformanceFeatures, metric)).label(f'avg_{metric}'),
                (func.avg(func.pow(getattr(PerformanceFeatures, metric), 2)) - 
                 func.pow(func.avg(getattr(PerformanceFeatures, metric)), 2)).label(f'var_{metric}')
            )
        
        query = query.join(PerformanceFeatures)
        
        if summoner_name:
            query = query.filter(Participant.summoner_name == summoner_name)
        
        query = query.group_by(Participant.champion_name)

        df = pd.read_sql(query.statement, session.bind)
        
        # Calculate standard deviation from variance
        for metric in metrics:
            df[f'std_{metric}'] = np.sqrt(df[f'var_{metric}'])
            df = df.drop(f'var_{metric}', axis=1)
        
        if df.empty:
            logging.warning(f"No champion stats data retrieved for {'summoner ' + summoner_name if summoner_name else 'all summoners'}.")
        else:
            logging.info(f"Retrieved stats for {len(df)} champions for {'summoner ' + summoner_name if summoner_name else 'all summoners'}.")
        return df
    except Exception as e:
        logging.error(f"Error in calculate_champion_stats: {str(e)}")
        return pd.DataFrame()

def compare_champions(session, champion_x, champion_y, summoner_name=None):
    stats_df = calculate_champion_stats(session, summoner_name)
    
    if stats_df.empty:
        logging.warning("No champion stats available for comparison.")
        return None

    x_stats = stats_df[stats_df['champion_name'] == champion_x]
    if x_stats.empty:
        logging.warning(f"No stats available for champion {champion_x}")
        return None

    x_stats = x_stats.iloc[0]
    
    metrics = [
        'kill_participation', 'death_share', 'damage_share', 'damage_taken_share',
        'gold_share', 'heal_share', 'damage_mitigated_share', 'cs_share',
        'vision_share', 'vision_denial_share', 'xp_share', 'cc_share'
    ]
    
    query = session.query(PerformanceFeatures).join(Participant)
    if summoner_name:
        query = query.filter(Participant.summoner_name == summoner_name)
    y_data = query.filter(Participant.champion_name == champion_y).all()
    
    if not y_data:
        logging.warning(f"No performance data available for champion {champion_y}")
        return None

    y_df = pd.DataFrame([{metric: getattr(pf, metric) for metric in metrics} for pf in y_data])
    
    standardized_performance = pd.DataFrame()
    for metric in metrics:
        avg_col = f'avg_{metric}'
        std_col = f'std_{metric}'
        if x_stats[std_col] == 0:
            standardized_performance[metric] = 0
        else:
            standardized_performance[metric] = (y_df[metric] - x_stats[avg_col]) / x_stats[std_col]
    
    return standardized_performance.mean()

def analyze_champion_performance(session, summoner_name=None):
    if summoner_name:
        champions = session.query(Participant.champion_name).filter(Participant.summoner_name == summoner_name).distinct().all()
    else:
        champions = session.query(Participant.champion_name).distinct().all()
    
    champions = [c[0] for c in champions]
    
    logging.info(f"Analyzing performance for {len(champions)} champions for {'summoner ' + summoner_name if summoner_name else 'all summoners'}")
    
    if len(champions) < 2:
        logging.warning(f"Not enough champions ({len(champions)}) to perform analysis")
        return pd.DataFrame()

    results = []
    for champion_x in champions:
        for champion_y in champions:
            if champion_x != champion_y:
                performance = compare_champions(session, champion_x, champion_y, summoner_name)
                if performance is not None:
                    results.append({
                        'champion_x': champion_x,
                        'champion_y': champion_y,
                        'average_performance': performance.mean()
                    })
    
    logging.info(f"Generated {len(results)} comparison results")
    return pd.DataFrame(results)

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
    engine = create_engine('sqlite:///matches.db')
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
        print(summary_stats)
        plot_score_distribution(df)
        print("Plots saved as 'score_distribution.png' and 'score_qq_plot.png'")
    else:
        print("No data available for analysis.")