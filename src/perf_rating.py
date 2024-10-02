import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Match, Team, Participant, PerformanceFeatures
import matplotlib.pyplot as plt


def calculate_performance_ratings():
    engine = create_engine('sqlite:///matches.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    query = session.query(
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

    df = pd.read_sql(query.statement, session.bind)

    features = ['kill_participation', 'death_share', 'damage_share', 'damage_taken_share',
                'gold_share', 'heal_share', 'damage_mitigated_share', 'cs_share',
                'vision_share', 'vision_denial_share', 'xp_share', 'cc_share']

    correlations = df[features].corrwith(df['win'])


    df_standardized = (df[features] - df[features].mean()) / df[features].std()

    df['performance_score'] = (df_standardized * correlations).sum(axis=1) / correlations.sum()

    df['standardized_performance_score'] = stats.zscore(df['performance_score'])

    for index, row in df.iterrows():
        participant = session.query(Participant).get(row['participant_id'])
        if participant:
            participant.performance_score = row['performance_score']
            participant.standardized_performance_score = row['standardized_performance_score']

    session.commit()

    session.close()

    return df['standardized_performance_score'].describe()
def plot_score_distribution(df):
    plt.figure(figsize=(10,6))
    plt.hist(df['standardized_performance_score'], bins=50, edgecolor='black')
    plt.title('Distribution of Standardized Performance Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig('score_distribution.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,6))
    stats.probplot(df['standardized_performance_score'], dist="norm", plot=ax)
    ax.set_title("Q-Q plot of Standardized Performance Scores")
    plt.savefig('score_qq_plot.png')
    plt.close()

if __name__ == "__main__":
    summary_stats = calculate_performance_ratings()
    print("Summary statistics of standardized performance scores:")
    print(summary_stats)
    df = calculate_performance_ratings()
    plot_score_distribution(df)
    print("Plots saved as 'score_distribution.png' and 'score_qq_plot.png'")
