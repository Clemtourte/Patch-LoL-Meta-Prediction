from sqlalchemy import create_engine, text

def clean_wrong_format_matches():
    engine = create_engine('sqlite:///../datasets/league_data.db')
    
    try:
        with engine.connect() as conn:
            # First get just the game_ids
            result = conn.execute(text("""
                SELECT DISTINCT m.match_id, m.game_id
                FROM matches m
                JOIN teams t ON m.match_id = t.match_id
                JOIN participants p ON t.team_id = p.team_id
                JOIN performance_features pf ON p.participant_id = pf.participant_id
                WHERE pf.champion_role_patch NOT LIKE '%-%-%'
            """))
            
            matches = list(result)
            print(f"Found {len(matches)} matches to delete")
            
            # Delete in correct order due to foreign key constraints
            for match_id, game_id in matches:
                # Delete performance_features first
                conn.execute(text("""
                    DELETE FROM performance_features 
                    WHERE participant_id IN (
                        SELECT p.participant_id 
                        FROM participants p
                        JOIN teams t ON p.team_id = t.team_id
                        WHERE t.match_id = :match_id
                    )
                """), {"match_id": match_id})
                
                # Delete participants
                conn.execute(text("""
                    DELETE FROM participants 
                    WHERE team_id IN (
                        SELECT team_id FROM teams WHERE match_id = :match_id
                    )
                """), {"match_id": match_id})
                
                # Delete teams
                conn.execute(text("DELETE FROM teams WHERE match_id = :match_id"), 
                           {"match_id": match_id})
                
                # Finally delete match
                conn.execute(text("DELETE FROM matches WHERE match_id = :match_id"), 
                           {"match_id": match_id})
                
                conn.commit()
                print(f"Deleted match {game_id}")
            
            print("\nDeletion complete!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'conn' in locals():
            conn.rollback()

if __name__ == "__main__":
    confirm = input("This will delete matches with wrong format. Are you sure? (yes/no): ")
    if confirm.lower() == 'yes':
        clean_wrong_format_matches()
    else:
        print("Operation cancelled.")