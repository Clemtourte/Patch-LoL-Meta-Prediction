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
import statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_enemy_performance_metrics(session, match, participant):
    try:
        enemy_team = session.query(Team).filter(
            Team.match_id == match.match_id,
            Team.team_id != participant.team_id
        ).first()
        
        if not enemy_team:
            return None

        enemy_stats = {
            'total_kills': sum(p.kills for p in enemy_team.participants),
            'total_deaths': sum(p.deaths for p in enemy_team.participants),
            'total_damage': sum(p.total_damage_dealt for p in enemy_team.participants),
            'total_gold': sum(p.gold_earned for p in enemy_team.participants),
            'total_wards': sum(p.wards_placed for p in enemy_team.participants),
            'total_level': sum(p.champ_level for p in enemy_team.participants)
        }

        metrics = {
            'kill_participation': (participant.kills + participant.assists) / 
                                max(1, enemy_stats['total_kills'] + participant.kills + participant.assists),
            'death_share': participant.deaths / max(1, enemy_stats['total_deaths']),
            'damage_share': participant.total_damage_dealt / max(1, enemy_stats['total_damage']),
            'damage_taken_share': participant.damage_taken / max(1, enemy_stats['total_damage']),
            'gold_share': participant.gold_earned / max(1, enemy_stats['total_gold']),
            'heal_share': participant.total_heal / max(1, enemy_stats['total_damage']),
            'damage_mitigated_share': participant.damage_mitigated / max(1, enemy_stats['total_damage']),
            'cs_share': participant.cs / max(1, enemy_stats['total_gold']),
            'vision_share': participant.wards_placed / max(1, enemy_stats['total_wards']),
            'vision_denial_share': participant.wards_killed / max(1, enemy_stats['total_wards']),
            'xp_share': participant.xp / max(1, enemy_stats['total_level'] * 100),
            'cc_share': participant.time_ccing_others / max(1, enemy_stats['total_deaths'] * 10)
        }
        
        for key in metrics:
            metrics[key] = min(1, max(0, metrics[key]))
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating enemy metrics: {str(e)}")
        return None

def calculate_performance_ratings():
    engine = create_engine("sqlite:///../datasets/league_data.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get participants without performance scores
        unrated_participants = session.query(Participant)\
            .filter(Participant.standardized_performance_score.is_(None))\
            .all()

        if not unrated_participants:
            logging.info("No unrated participants found")
            return

        logging.info(f"Found {len(unrated_participants)} unrated participants")

        # Process participants by position
        positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        for position in positions:
            position_participants = [p for p in unrated_participants if p.position == position]
            if not position_participants:
                continue

            # Calculate features and scores for position
            features = ['kill_participation', 'death_share', 'damage_share', 
                       'damage_taken_share', 'gold_share', 'heal_share',
                       'damage_mitigated_share', 'cs_share', 'vision_share',
                       'vision_denial_share', 'xp_share', 'cc_share']

            for participant in position_participants:
                perf_features = participant.performance_features
                if not perf_features:
                    continue

                # Calculate performance score
                feature_values = [getattr(perf_features, f) for f in features]
                if not all(v is not None for v in feature_values):
                    continue

                # Simple scoring: average of all features
                performance_score = sum(feature_values) / len(features)
                participant.performance_score = performance_score

        # Calculate standardized scores
        session.flush()
        for position in positions:
            position_participants = session.query(Participant)\
                .filter(Participant.position == position,
                       Participant.performance_score.isnot(None))\
                .all()

            if len(position_participants) >= 2:
                scores = [p.performance_score for p in position_participants]
                mean = statistics.mean(scores)
                stdev = statistics.stdev(scores) if len(scores) > 1 else 1.0

                for participant in position_participants:
                    if stdev > 0:
                        participant.standardized_performance_score = (participant.performance_score - mean) / stdev
                    else:
                        participant.standardized_performance_score = 0.0

        session.commit()
        logging.info("Successfully calculated performance ratings")

    except Exception as e:
        logging.error(f"Error calculating performance ratings: {str(e)}")
        session.rollback()
    finally:
        session.close()

def analyze_champion_performance(session, min_games=10):
    try:
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        all_results = []

        for position in roles:
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
                .having(func.count(Participant.participant_id) >= min_games)
                .statement,
                session.bind
            )
            
            if not df.empty:
                df['win_rate'] = df['win_rate'] * 100
                df['rating'] = df['avg_score']
                df = df.sort_values('rating', ascending=False)
                all_results.append(df)
        
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        
    except Exception as e:
        logging.error(f"Error in analyze_champion_performance: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    calculate_performance_ratings()