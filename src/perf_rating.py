import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, select, text, inspect
from sqlalchemy.orm import sessionmaker
from models import Match, Team, Participant, PerformanceFeatures, Base

def calculate_performance_ratings():
    engine = create_engine('sqlite:///matches.db')
    Session = sessionmaker(bind=engine)
    session = Session()

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

    session.close()
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

    return df, summary

def plot_score_distribution(df):
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
    print("Summary statistics of standardized performance scores:")
    print(summary_stats)
    plot_score_distribution(df)
    print("Plots saved as 'score_distribution.png' and 'score_qq_plot.png'")