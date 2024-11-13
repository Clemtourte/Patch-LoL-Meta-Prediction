from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from models import Base, Match, Team, Participant, PerformanceFeatures
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_duplicate_entries():
    """Clean duplicate entries and align tables"""
    engine = create_engine("sqlite:///../datasets/league_data.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        logging.info("Starting cleanup process...")
        
        # 1. First, let's delete completely empty performance features entries
        logging.info("Deleting empty performance features entries...")
        empty_query = """
        DELETE FROM performance_features 
        WHERE kill_participation IS NULL 
        AND death_share IS NULL 
        AND damage_share IS NULL 
        AND damage_taken_share IS NULL 
        AND gold_share IS NULL 
        AND heal_share IS NULL 
        AND damage_mitigated_share IS NULL 
        AND cs_share IS NULL 
        AND vision_share IS NULL 
        AND vision_denial_share IS NULL 
        AND xp_share IS NULL 
        AND cc_share IS NULL
        """
        result = session.execute(text(empty_query))
        session.commit()
        logging.info(f"Deleted {result.rowcount} empty entries")

        # 2. Find and handle duplicates
        logging.info("Checking for duplicate performance features...")
        duplicate_query = """
        SELECT participant_id, COUNT(*) as count 
        FROM performance_features 
        GROUP BY participant_id 
        HAVING COUNT(*) > 1
        """
        duplicates = session.execute(text(duplicate_query)).fetchall()
        logging.info(f"Found {len(duplicates)} participants with duplicate performance features")

        # 3. For each duplicate, keep only the one with most non-null values
        for participant_id, count in duplicates:
            logging.info(f"Processing duplicates for participant {participant_id}")
            
            # Get all entries with non-null count
            entries_query = """
            SELECT id,
                   (CASE WHEN kill_participation IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN death_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN damage_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN damage_taken_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN gold_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN heal_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN damage_mitigated_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN cs_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN vision_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN vision_denial_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN xp_share IS NOT NULL THEN 1 ELSE 0 END +
                    CASE WHEN cc_share IS NOT NULL THEN 1 ELSE 0 END) as non_null_count
            FROM performance_features
            WHERE participant_id = :participant_id
            ORDER BY non_null_count DESC
            """
            entries = session.execute(text(entries_query), {"participant_id": participant_id}).fetchall()
            
            # Keep the first one (most complete), delete others
            if len(entries) > 1:
                # Get IDs to delete (all except the first one)
                ids_to_delete = [entry[0] for entry in entries[1:]]
                if ids_to_delete:  # Check if there are IDs to delete
                    delete_query = text("DELETE FROM performance_features WHERE id IN :ids")
                    session.execute(delete_query, {"ids": tuple(ids_to_delete)})
                    logging.info(f"Deleted {len(ids_to_delete)} duplicate entries for participant {participant_id}")

        session.commit()

        # 4. Find and delete orphaned entries
        logging.info("Checking for orphaned performance features...")
        orphaned_query = """
        DELETE FROM performance_features 
        WHERE participant_id NOT IN (SELECT participant_id FROM participants)
        """
        result = session.execute(text(orphaned_query))
        session.commit()
        logging.info(f"Deleted {result.rowcount} orphaned entries")

        # 5. Create missing performance features with proper IDs
        logging.info("Checking for missing performance features...")
        
        # First get the current max ID
  # Additional step: Fix any remaining NULL IDs
        logging.info("Fixing remaining NULL IDs...")
        
        # Get current max ID
        max_id_query = """
        SELECT COALESCE(MAX(id), 0) from performance_features WHERE id IS NOT NULL
        """
        current_max_id = session.execute(text(max_id_query)).scalar() or 0
        
        # Get entries with NULL IDs
        null_ids_query = """
        SELECT participant_id FROM performance_features WHERE id IS NULL ORDER BY participant_id
        """
        null_entries = session.execute(text(null_ids_query)).fetchall()
        
        # Update each NULL entry with a new sequential ID
        for i, (participant_id,) in enumerate(null_entries, 1):
            new_id = current_max_id + i
            update_query = """
            UPDATE performance_features 
            SET id = :new_id 
            WHERE participant_id = :participant_id AND id IS NULL
            """
            session.execute(text(update_query), {
                "new_id": new_id,
                "participant_id": participant_id
            })
            
        session.commit()
        logging.info(f"Fixed {len(null_entries)} NULL IDs")


        # 6. Final verification
        null_check_query = """
        SELECT COUNT(*) FROM performance_features WHERE id IS NULL
        """
        remaining_nulls = session.execute(text(null_check_query)).scalar()
        
        participant_count = session.query(Participant).count()
        perf_features_count = session.query(PerformanceFeatures).count()
        
        logging.info("Cleanup completed!")
        logging.info(f"Final counts:")
        logging.info(f"- Participants: {participant_count}")
        logging.info(f"- Performance Features: {perf_features_count}")
        logging.info(f"- Remaining NULL IDs: {remaining_nulls}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")
        session.rollback()
        return False
    finally:
        session.close()

if __name__ == "__main__":
    print("Starting database cleanup process...")
    if clean_duplicate_entries():
        print("Database cleanup completed successfully!")
    else:
        print("Error occurred during database cleanup")