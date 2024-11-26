import logging
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from models import Match, Team, Participant, PerformanceFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_database():
    engine = create_engine("sqlite:///../datasets/league_data.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Count participants without performance features
        orphaned_participants = session.query(Participant)\
            .outerjoin(PerformanceFeatures)\
            .filter(PerformanceFeatures.participant_id == None)\
            .count()
            
        logger.info(f"Participants without performance features: {orphaned_participants}")
        
        # This will help us determine if we need to run the cleanup script
        return orphaned_participants > 0
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    verify_database()