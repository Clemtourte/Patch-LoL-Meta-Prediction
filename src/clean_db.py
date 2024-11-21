# fix_missing_data.py
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Match, Team, Participant, PerformanceFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_missing_data():
    engine = create_engine("sqlite:///../datasets/league_data.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Find participants missing performance features
        missing_features = session.query(Participant)\
            .outerjoin(PerformanceFeatures)\
            .filter(PerformanceFeatures.id.is_(None))\
            .all()

        logger.info(f"Found {len(missing_features)} participants missing performance features")

        # Process in batches
        batch_size = 1000
        for i in range(0, len(missing_features), batch_size):
            batch = missing_features[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {len(missing_features)//batch_size + 1}")
            
            for participant in batch:
                try:
                    # Check for valid relationships
                    if not participant.team:
                        logger.warning(f"Participant {participant.participant_id} has no team relationship")
                        continue
                        
                    if not participant.team.match:
                        logger.warning(f"Team {participant.team.team_id} has no match relationship")
                        continue

                    team = participant.team
                    match = team.match
                    
                    # Calculate team totals
                    team_totals = {
                        'kills': sum(p.kills for p in team.participants if p.kills is not None),
                        'deaths': sum(p.deaths for p in team.participants if p.deaths is not None),
                        'damage': sum(p.total_damage_dealt for p in team.participants if p.total_damage_dealt is not None),
                        'gold': sum(p.gold_earned for p in team.participants if p.gold_earned is not None),
                        'wards': sum(p.wards_placed for p in team.participants if p.wards_placed is not None),
                        'wards_killed': sum(p.wards_killed for p in team.participants if p.wards_killed is not None),
                        'cs': sum(p.cs for p in team.participants if p.cs is not None),
                        'xp': sum(p.xp for p in team.participants if p.xp is not None),
                        'heal': sum(p.total_heal for p in team.participants if p.total_heal is not None),
                        'damage_taken': sum(p.damage_taken for p in team.participants if p.damage_taken is not None),
                        'damage_mitigated': sum(p.damage_mitigated for p in team.participants if p.damage_mitigated is not None),
                        'cc': sum(p.time_ccing_others for p in team.participants if p.time_ccing_others is not None)
                    }

                    # Create new performance features with null checks
                    perf_features = PerformanceFeatures(
                        participant_id=participant.participant_id,
                        champion_role_patch=f"{participant.champion_name}-{participant.position}-{match.patch}",
                        kill_participation=(participant.kills + participant.assists) / max(1, team_totals['kills']) if participant.kills is not None and participant.assists is not None else 0,
                        death_share=participant.deaths / max(1, team_totals['deaths']) if participant.deaths is not None else 0,
                        damage_share=participant.total_damage_dealt / max(1, team_totals['damage']) if participant.total_damage_dealt is not None else 0,
                        damage_taken_share=participant.damage_taken / max(1, team_totals['damage_taken']) if participant.damage_taken is not None else 0,
                        gold_share=participant.gold_earned / max(1, team_totals['gold']) if participant.gold_earned is not None else 0,
                        heal_share=participant.total_heal / max(1, team_totals['heal']) if participant.total_heal is not None else 0,
                        damage_mitigated_share=participant.damage_mitigated / max(1, team_totals['damage_mitigated']) if participant.damage_mitigated is not None else 0,
                        cs_share=participant.cs / max(1, team_totals['cs']) if participant.cs is not None else 0,
                        vision_share=participant.wards_placed / max(1, team_totals['wards']) if participant.wards_placed is not None else 0,
                        vision_denial_share=participant.wards_killed / max(1, team_totals['wards_killed']) if participant.wards_killed is not None else 0,
                        xp_share=participant.xp / max(1, team_totals['xp']) if participant.xp is not None else 0,
                        cc_share=participant.time_ccing_others / max(1, team_totals['cc']) if participant.time_ccing_others is not None else 0
                    )

                    # Ensure all shares are between 0 and 1
                    for attr in ['kill_participation', 'death_share', 'damage_share', 'damage_taken_share', 
                               'gold_share', 'heal_share', 'damage_mitigated_share', 'cs_share', 
                               'vision_share', 'vision_denial_share', 'xp_share', 'cc_share']:
                        value = getattr(perf_features, attr)
                        setattr(perf_features, attr, min(1.0, max(0.0, float(value if value is not None else 0))))

                    session.add(perf_features)

                except Exception as e:
                    logger.error(f"Error processing participant {participant.participant_id}: {str(e)}")
                    continue

            # Commit each batch
            try:
                session.commit()
                logger.info(f"Committed batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error committing batch: {str(e)}")
                session.rollback()
                continue

        # Now recalculate all performance ratings
        logger.info("Recalculating performance ratings...")
        from perf_rating import calculate_performance_ratings
        calculate_performance_ratings()
        
    except Exception as e:
        logger.error(f"Error fixing data: {str(e)}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    fix_missing_data()