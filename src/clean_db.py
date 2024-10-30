import logging
from sqlalchemy import create_engine, inspect, func, text, delete
from sqlalchemy.orm import sessionmaker
from models import Base, Participant, PerformanceFeatures, Team, Match
from sqlalchemy.exc import OperationalError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_db(uri="sqlite:///../datasets/league_data.db"):
    engine = create_engine(uri)
    Session = sessionmaker(bind=engine)
    return engine, Session()

def safely_reset_sequences(session, engine):
    """Safely reset auto-increment sequences"""
    logger.info("Attempting to reset sequences safely...")
    
    try:
        # First check if sqlite_sequence exists
        result = session.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        )).fetchone()
        
        if result is None:
            logger.info("sqlite_sequence table doesn't exist - creating it")
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS temp_seq (
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                )
            """))
            session.execute(text("DROP TABLE IF EXISTS temp_seq"))
        
        # Get current max IDs
        max_ids = {
            'participants': session.query(func.max(Participant.participant_id)).scalar() or 0,
            'teams': session.query(func.max(Team.team_id)).scalar() or 0,
            'matches': session.query(func.max(Match.match_id)).scalar() or 0,
            'performance_features': session.query(func.max(PerformanceFeatures.id)).scalar() or 0
        }
        
        for table, max_id in max_ids.items():
            try:
                session.execute(text(f"DELETE FROM sqlite_sequence WHERE name = '{table}'"))
                session.execute(text(f"INSERT INTO sqlite_sequence (name, seq) VALUES ('{table}', {max_id})"))
                logger.info(f"Reset sequence for {table} to {max_id}")
            except OperationalError as e:
                logger.warning(f"Could not reset sequence for {table}: {str(e)}")
        
        session.commit()
        
    except Exception as e:
        logger.warning(f"Error resetting sequences: {str(e)}")
        session.rollback()

def fix_orphaned_records(session):
    """Clean up orphaned records and fix relationships"""
    logger.info("Fixing orphaned records...")

    try:
        # Delete orphaned performance features using DELETE statement
        deleted_features = session.execute(
            delete(PerformanceFeatures).where(
                ~PerformanceFeatures.participant_id.in_(
                    session.query(Participant.participant_id)
                )
            )
        ).rowcount
        logger.info(f"Removed {deleted_features} orphaned performance features")

        # Delete orphaned participants using DELETE statement
        deleted_participants = session.execute(
            delete(Participant).where(
                ~Participant.team_id.in_(
                    session.query(Team.team_id)
                )
            )
        ).rowcount
        logger.info(f"Removed {deleted_participants} orphaned participants")

        # Delete orphaned teams using DELETE statement
        deleted_teams = session.execute(
            delete(Team).where(
                ~Team.match_id.in_(
                    session.query(Match.match_id)
                )
            )
        ).rowcount
        logger.info(f"Removed {deleted_teams} orphaned teams")

        session.commit()
        return deleted_participants, deleted_features, deleted_teams

    except Exception as e:
        logger.error(f"Error fixing orphaned records: {str(e)}")
        session.rollback()
        raise

def cleanup_null_values(session):
    """Clean up null values in tables"""
    logger.info("Cleaning up null values...")
    
    try:
        # Delete rows with null in critical columns
        deleted_participants = session.execute(
            delete(Participant).where(
                (Participant.participant_id.is_(None)) |
                (Participant.team_id.is_(None)) |
                (Participant.summoner_name.is_(None)) |
                (Participant.champion_name.is_(None))
            )
        ).rowcount
        logger.info(f"Removed {deleted_participants} participants with null values")

        deleted_features = session.execute(
            delete(PerformanceFeatures).where(
                (PerformanceFeatures.id.is_(None)) |
                (PerformanceFeatures.participant_id.is_(None))
            )
        ).rowcount
        logger.info(f"Removed {deleted_features} performance features with null values")

        deleted_teams = session.execute(
            delete(Team).where(
                (Team.team_id.is_(None)) |
                (Team.match_id.is_(None))
            )
        ).rowcount
        logger.info(f"Removed {deleted_teams} teams with null values")

        session.commit()
        return deleted_participants, deleted_features, deleted_teams

    except Exception as e:
        logger.error(f"Error cleaning up null values: {str(e)}")
        session.rollback()
        raise

def verify_data_integrity(session):
    """Verify data integrity after cleanup"""
    logger.info("Verifying data integrity...")

    try:
        # Check for null values in critical columns
        null_checks = {
            'Participants': {
                'total': session.query(Participant).count(),
                'null_ids': session.query(Participant).filter(Participant.participant_id.is_(None)).count(),
                'null_team_ids': session.query(Participant).filter(Participant.team_id.is_(None)).count()
            },
            'Teams': {
                'total': session.query(Team).count(),
                'null_ids': session.query(Team).filter(Team.team_id.is_(None)).count(),
                'null_match_ids': session.query(Team).filter(Team.match_id.is_(None)).count()
            },
            'Performance Features': {
                'total': session.query(PerformanceFeatures).count(),
                'null_ids': session.query(PerformanceFeatures).filter(PerformanceFeatures.id.is_(None)).count(),
                'null_participant_ids': session.query(PerformanceFeatures).filter(PerformanceFeatures.participant_id.is_(None)).count()
            }
        }

        for table, checks in null_checks.items():
            logger.info(f"\n{table} integrity check:")
            logger.info(f"Total records: {checks['total']}")
            for check_name, count in checks.items():
                if check_name != 'total' and count > 0:
                    logger.warning(f"{check_name}: {count}")
                elif check_name != 'total':
                    logger.info(f"{check_name}: {count}")

        return all(
            all(count == 0 for name, count in checks.items() if name != 'total')
            for checks in null_checks.values()
        )

    except Exception as e:
        logger.error(f"Error verifying data integrity: {str(e)}")
        return False

def main():
    engine, session = connect_db()
    
    try:
        logger.info("Starting database cleanup...")

        # Step 1: Clean up null values
        logger.info("Step 1: Cleaning up null values...")
        null_cleaned = cleanup_null_values(session)
        
        # Step 2: Fix orphaned records
        logger.info("Step 2: Fixing orphaned records...")
        orphans_cleaned = fix_orphaned_records(session)
        
        # Step 3: Safely reset sequences
        logger.info("Step 3: Resetting sequences...")
        safely_reset_sequences(session, engine)

        # Step 4: Verify integrity
        logger.info("Step 4: Verifying data integrity...")
        if verify_data_integrity(session):
            logger.info("Data integrity verification passed")
        else:
            logger.warning("Data integrity verification failed")

        logger.info(f"""
Cleanup Summary:
Null values cleaned:
- Participants: {null_cleaned[0]}
- Performance Features: {null_cleaned[1]}
- Teams: {null_cleaned[2]}

Orphaned records cleaned:
- Participants: {orphans_cleaned[0]}
- Performance Features: {orphans_cleaned[1]}
- Teams: {orphans_cleaned[2]}
        """)

    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main()