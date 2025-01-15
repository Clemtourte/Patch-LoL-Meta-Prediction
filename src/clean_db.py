from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deep_clean_database():
    engine = create_engine('sqlite:///../datasets/league_data.db')
    
    try:
        with engine.connect() as conn:
            logger.info("Starting deep database cleanup...")

            # 1. Find ALL problematic data
            issues = {}

            # Check for teams with NULL values
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM teams 
                WHERE match_id IS NULL 
                   OR team_name IS NULL 
                   OR win IS NULL
            """))
            issues['null_teams'] = result.fetchone()[0]

            # Check for matches that don't exist in matches table but exist in teams
            result = conn.execute(text("""
                SELECT DISTINCT t.match_id
                FROM teams t
                LEFT JOIN matches m ON t.match_id = m.match_id
                WHERE m.match_id IS NULL
            """))
            issues['orphaned_matches'] = len(list(result))

            # Check for wrong triplet format
            result = conn.execute(text("""
                SELECT DISTINCT m.match_id
                FROM matches m
                JOIN teams t ON m.match_id = t.match_id
                JOIN participants p ON t.team_id = p.team_id
                JOIN performance_features pf ON p.participant_id = pf.participant_id
                WHERE pf.champion_role_patch NOT LIKE '%-%-%.%'
                   OR pf.champion_role_patch IS NULL
            """))
            issues['wrong_triplets'] = len(list(result))

            # Show summary of issues
            logger.info("\nFound issues:")
            logger.info(f"Teams with NULL values: {issues['null_teams']}")
            logger.info(f"Teams referencing non-existent matches: {issues['orphaned_matches']}")
            logger.info(f"Matches with wrong triplet format: {issues['wrong_triplets']}")

            if sum(issues.values()) == 0:
                logger.info("No issues found! Database is clean.")
                return

            confirm = input("\nProceed with cleaning these issues? (yes/no): ")
            if confirm.lower() != 'yes':
                logger.info("Operation cancelled")
                return

            # Clean everything in correct order
            logger.info("\nCleaning database...")

            # 1. Clean teams with NULL values first
            conn.execute(text("""
                DELETE FROM teams 
                WHERE match_id IS NULL 
                   OR team_name IS NULL 
                   OR win IS NULL
            """))

            # 2. Clean teams referencing non-existent matches
            conn.execute(text("""
                DELETE FROM teams
                WHERE match_id NOT IN (SELECT match_id FROM matches)
            """))

            # 3. Clean matches with wrong triplet format
            problematic_matches = conn.execute(text("""
                SELECT DISTINCT m.match_id
                FROM matches m
                JOIN teams t ON m.match_id = t.match_id
                JOIN participants p ON t.team_id = p.team_id
                JOIN performance_features pf ON p.participant_id = pf.participant_id
                WHERE pf.champion_role_patch NOT LIKE '%-%-%.%'
                   OR pf.champion_role_patch IS NULL
            """)).fetchall()

            for row in problematic_matches:
                match_id = row[0]
                # Delete in correct order with explicit cascade
                conn.execute(text("""
                    DELETE FROM performance_features 
                    WHERE participant_id IN (
                        SELECT p.participant_id 
                        FROM participants p 
                        JOIN teams t ON p.team_id = t.team_id 
                        WHERE t.match_id = :match_id
                    )
                """), {"match_id": match_id})

                conn.execute(text("""
                    DELETE FROM participants 
                    WHERE team_id IN (
                        SELECT team_id FROM teams WHERE match_id = :match_id
                    )
                """), {"match_id": match_id})

                conn.execute(text("""
                    DELETE FROM teams WHERE match_id = :match_id
                """), {"match_id": match_id})

                conn.execute(text("""
                    DELETE FROM matches WHERE match_id = :match_id
                """), {"match_id": match_id})

            conn.commit()

            # Verify cleanup
            logger.info("\nVerifying database state after cleanup...")
            
            verification_queries = [
                ("Teams with NULL values", """
                    SELECT COUNT(*) FROM teams 
                    WHERE match_id IS NULL OR team_name IS NULL OR win IS NULL
                """),
                ("Orphaned teams", """
                    SELECT COUNT(*) FROM teams t 
                    LEFT JOIN matches m ON t.match_id = m.match_id 
                    WHERE m.match_id IS NULL
                """),
                ("Wrong triplet format", """
                    SELECT COUNT(*) FROM performance_features 
                    WHERE champion_role_patch NOT LIKE '%-%-%.%' 
                    OR champion_role_patch IS NULL
                """),
                ("Matches count", "SELECT COUNT(*) FROM matches"),
                ("Teams count", "SELECT COUNT(*) FROM teams"),
                ("Participants count", "SELECT COUNT(*) FROM participants"),
                ("Performance features count", "SELECT COUNT(*) FROM performance_features")
            ]

            for check_name, query in verification_queries:
                result = conn.execute(text(query))
                count = result.fetchone()[0]
                logger.info(f"{check_name}: {count}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        raise

if __name__ == "__main__":
    deep_clean_database()