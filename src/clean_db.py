import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_query(cursor, query, description):
    try:
        cursor.execute(query)
        logging.info(f"{description}: {cursor.rowcount} rows affected")
    except sqlite3.Error as e:
        logging.error(f"Error executing {description}: {e}")

def fix_performance_features(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check for discrepancies
        cursor.execute("SELECT COUNT(*) FROM performance_features WHERE id != participant_id")
        discrepancies = cursor.fetchone()[0]
        logging.info(f"Found {discrepancies} discrepancies between id and participant_id")

        if discrepancies > 0:
            # Create a temporary table with corrected ids
            execute_query(cursor, """
                CREATE TABLE temp_performance_features AS
                SELECT 
                    participant_id as id,
                    participant_id,
                    kill_participation,
                    death_share,
                    damage_share,
                    damage_taken_share,
                    gold_share,
                    heal_share,
                    damage_mitigated_share,
                    cs_share,
                    vision_share,
                    vision_denial_share,
                    xp_share,
                    cc_share
                FROM performance_features
                ORDER BY participant_id;
            """, "Creating temporary performance_features table with corrected ids")

            # Replace the original table
            execute_query(cursor, "DROP TABLE performance_features;", "Dropping original performance_features table")
            execute_query(cursor, "ALTER TABLE temp_performance_features RENAME TO performance_features;", "Renaming temporary performance_features table")

            # Reset the autoincrement counter
            execute_query(cursor, "DELETE FROM sqlite_sequence WHERE name='performance_features';", "Resetting performance_features sequence")
            execute_query(cursor, """
                INSERT INTO sqlite_sequence (name, seq) 
                VALUES ('performance_features', (SELECT MAX(id) FROM performance_features));
            """, "Setting performance_features sequence")

            # Verify the changes
            cursor.execute("SELECT COUNT(*) FROM performance_features WHERE id != participant_id")
            remaining_discrepancies = cursor.fetchone()[0]
            logging.info(f"Remaining discrepancies after fix: {remaining_discrepancies}")

        else:
            logging.info("No discrepancies found. No changes made.")

        # Check for any orphaned records
        cursor.execute("""
            SELECT COUNT(*) FROM performance_features
            WHERE participant_id NOT IN (SELECT participant_id FROM participants)
        """)
        orphaned = cursor.fetchone()[0]
        logging.info(f"Found {orphaned} orphaned records in performance_features")

        if orphaned > 0:
            execute_query(cursor, """
                DELETE FROM performance_features
                WHERE participant_id NOT IN (SELECT participant_id FROM participants)
            """, "Deleting orphaned records from performance_features")

        conn.commit()
        logging.info("Performance features cleanup completed successfully")

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        conn.rollback()

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    db_path = "../datasets/league_data.db"  # Update this path to your database file
    fix_performance_features(db_path)