import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, select, text, inspect, func
from sqlalchemy.orm import sessionmaker
from models import Match, Team, Participant, PerformanceFeatures, Base
import logging
from sklearn.preprocessing import StandardScaler
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_enemy_performance_metrics(session, match, participant):
    try:
        enemy_team = session.query(Team).filter(
            Team.match_id == match.match_id,
            Team.team_id != participant.team_id
        ).first()
        
        if not enemy_team:
            return None
        
        # Calculate enemy team totals
        enemy_stats = {
            'total_kills': sum(p.kills for p in enemy_team.participants),
            'total_deaths': sum(p.deaths for p in enemy_team.participants),
            'total_damage': sum(p.total_damage_dealt for p in enemy_team.participants),
            'total_gold': sum(p.gold_earned for p in enemy_team.participants),
            'total_wards': sum(p.wards_placed for p in enemy_team.participants),
            'total_level': sum(p.champ_level for p in enemy_team.participants)
        }

        # Calculate total team cs
        enemy_total_cs = sum(p.cs for p in enemy_team.participants)
        
        # Calculate metrics with proper ratios
        metrics = {
            'kill_participation': (participant.kills + participant.assists) / 
                                max(1, enemy_stats['total_kills'] + participant.kills + participant.assists),
            'death_share': participant.deaths / max(1, enemy_stats['total_deaths']),
            'damage_share': participant.total_damage_dealt / max(1, enemy_stats['total_damage']),
            'damage_taken_share': participant.damage_taken / max(1, enemy_stats['total_damage']),
            'gold_share': participant.gold_earned / max(1, enemy_stats['total_gold']),
            'heal_share': participant.total_heal / max(1, enemy_stats['total_damage']),
            'damage_mitigated_share': participant.damage_mitigated / max(1, enemy_stats['total_damage']),
            'cs_share': participant.cs / max(1, enemy_total_cs),  # Fixed CS calculation
            'vision_share': participant.wards_placed / max(1, enemy_stats['total_wards']),
            'vision_denial_share': participant.wards_killed / max(1, enemy_stats['total_wards']),
            'xp_share': participant.xp / max(1, enemy_stats['total_level'] * 100),  # Normalized XP
            'cc_share': participant.time_ccing_others / max(1, enemy_stats['total_deaths'] * 10)  # Normalized CC
        }
        
        # Ensure all values are between 0 and 1
        for key in metrics:
            metrics[key] = min(1, max(0, metrics[key]))
        
        return metrics
    except Exception as e:
        logging.error(f"Error calculating enemy performance metrics: {str(e)}")
        return None

def calculate_performance_ratings():
    """Main function to calculate performance ratings"""
    engine = create_engine("sqlite:///../datasets/league_data.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get all matches with teams and participants
        logging.info("Fetching matches with teams and participants...")
        matches = session.query(Match)\
            .join(Team)\
            .join(Participant)\
            .filter(Team.participants.any())\
            .all()
            
        if not matches:
            logging.error("No matches found with teams and participants")
            return pd.DataFrame(), {}
            
        logging.info(f"Found {len(matches)} valid matches")
        all_metrics = []
        
        for match in matches:
            for team in match.teams:
                if not team.participants:
                    logging.warning(f"No participants found for team {team.team_id} in match {match.match_id}")
                    continue
                    
                for participant in team.participants:
                    if participant is None:
                        logging.warning("Found None participant, skipping")
                        continue
                        
                    metrics = calculate_enemy_performance_metrics(session, match, participant)
                    if metrics:
                        metrics.update({
                            'participant_id': participant.participant_id,
                            'champion_name': participant.champion_name,
                            'position': participant.position,
                            'patch': match.patch,
                            'win': team.win
                        })
                        all_metrics.append(metrics)
        
        if not all_metrics:
            logging.error("No metrics could be calculated")
            return pd.DataFrame(), {}
            
        df = pd.DataFrame(all_metrics)
        
        if df.empty:
            logging.warning("No data retrieved from the database.")
            return pd.DataFrame(), {}

        # Create champion-role-patch grouping
        df['champion_role_patch'] = df.apply(
            lambda x: f"{x['champion_name']}-{x['position']}-{x['patch']}", 
            axis=1
        )

        features = [
            'kill_participation', 'death_share', 'damage_share', 'damage_taken_share',
            'gold_share', 'heal_share', 'damage_mitigated_share', 'cs_share',
            'vision_share', 'vision_denial_share', 'xp_share', 'cc_share'
        ]

        log_data_info(df, features)

        # Calculate correlations with victory
        correlations = df[features].corrwith(df['win'])
        log_correlations(correlations, "Pearson")

        # Standardize features
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[features]),
            columns=features
        )

        # Calculate performance score using correlation weights
        total_correlation = correlations.abs().sum()
        df['performance_score'] = 0
        
        for feature in features:
            df['performance_score'] += df_scaled[feature] * correlations[feature]
        
        df['performance_score'] = df['performance_score'] / total_correlation
        df['standardized_performance_score'] = stats.zscore(df['performance_score'])

        # Update database
        updated_count = 0
        for index, row in df.iterrows():
            participant = session.query(Participant).get(row['participant_id'])
            if participant:
                participant.performance_score = row['performance_score']
                participant.standardized_performance_score = row['standardized_performance_score']
                
                # Update or create performance features
                perf_features = participant.performance_features or PerformanceFeatures(participant_id=participant.participant_id)
                for feature in features:
                    setattr(perf_features, feature, row[feature])
                perf_features.champion_role_patch = row['champion_role_patch']
                
                if not participant.performance_features:
                    session.add(perf_features)
                updated_count += 1

        session.commit()
        logging.info(f"Updated {updated_count} participants")

        # Create visualizations
        create_visualizations(df, features, correlations)
        
        # Calculate summary statistics
        summary = create_summary_statistics(df)
        
        return df, summary
        
    except Exception as e:
        logging.error(f"Error in calculate_performance_ratings: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), {}
    finally:
        session.close()

def log_data_info(df, features):
    """Log information about the dataset"""
    logging.info("Data types:")
    logging.info(df.dtypes)
    logging.info("\nUnique values in 'win' column:")
    logging.info(df['win'].unique())
    logging.info("\nData ranges:")
    for feature in features:
        logging.info(f"{feature}: {df[feature].min():.3f} to {df[feature].max():.3f}")
    logging.info("\nMean values grouped by win:")
    logging.info(df.groupby('win')[features].mean())

def log_correlations(correlations, method):
    """Log correlation information"""
    logging.info(f"\n{method} correlations with win:")
    for feature, corr in correlations.items():
        logging.info(f"{feature}: {corr:.3f}")

def create_visualizations(df, features, correlations):
    """Create visualization plots"""
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../graphs/correlation_heatmap.png')
    plt.close()

    # Correlation with victory barplot
    plt.figure(figsize=(12, 6))
    correlations.sort_values().plot(kind='bar')
    plt.title('Feature Correlations with Victory')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../graphs/victory_correlations.png')
    plt.close()

    # Performance score distribution
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
    ax.set_title("Q-Q Plot of Standardized Performance Scores")
    plt.savefig('../graphs/score_qq_plot.png')
    plt.close()

def create_summary_statistics(df):
    """Create summary statistics for performance scores"""
    summary_stats = df['standardized_performance_score'].describe()
    return {
        'Count': f"{summary_stats['count']:,.0f}",
        'Mean': f"{summary_stats['mean']:.6f}",
        'Standard Deviation': f"{summary_stats['std']:.6f}",
        'Minimum': f"{summary_stats['min']:.6f}",
        '25th Percentile': f"{summary_stats['25%']:.6f}",
        'Median': f"{summary_stats['50%']:.6f}",
        '75th Percentile': f"{summary_stats['75%']:.6f}",
        'Maximum': f"{summary_stats['max']:.6f}"
    }

def analyze_champion_performance(session, min_games=10):
    """Analyze champion performance by position"""
    positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    all_results = []

    for position in positions:
        df = pd.read_sql(
            session.query(
                Participant.champion_name,
                Participant.position,
                func.count(Participant.participant_id).label('games'),
                func.avg(Team.win).label('win_rate'),
                func.avg(Participant.standardized_performance_score).label('avg_score')
            )
            .join(Team)
            .filter(Participant.position == position)
            .group_by(Participant.champion_name, Participant.position)
            .statement,
            session.bind
        )
        
        df = df[df['games'] >= min_games]
        
        if not df.empty:
            df['win_rate'] = df['win_rate'] * 100
            df['score_std'] = df.groupby(['champion_name', 'position'])['avg_score'].transform('std')
            df['rating'] = df['avg_score']  # Add the 'rating' column
            df = df.sort_values('avg_score', ascending=False)
            all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

if __name__ == "__main__":
    print("Calculating performance ratings...")
    engine = create_engine("sqlite:///../datasets/league_data.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check data integrity
        match_count = session.query(Match).count()
        team_count = session.query(Team).count()
        participant_count = session.query(Participant).count()
        perf_features_count = session.query(PerformanceFeatures).count()
        
        print(f"\nDatabase contains:")
        print(f"- {match_count} matches")
        print(f"- {team_count} teams")
        print(f"- {participant_count} participants")
        print(f"- {perf_features_count} performance features entries")
        
        # Calculate ratings
        df, summary_stats = calculate_performance_ratings()
        
        if not df.empty:
            print("\nSummary statistics of standardized performance scores:")
            for key, value in summary_stats.items():
                print(f"{key}: {value}")
            print("\nVisualization files have been saved to the ../graphs directory")
        else:
            print("No data available for analysis.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        session.close()