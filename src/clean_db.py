from sqlalchemy import create_engine, text
import re

engine = create_engine('sqlite:///matches.db')

def extract_minutes(duration):
    match = re.search(r'(\d+)m', duration)
    return int(match.group(1)) if match else 0

with engine.connect() as conn:
    # 1. Remove short games
    result = conn.execute(text("SELECT match_id, game_duration FROM matches"))
    matches_to_delete = [row.match_id for row in result if extract_minutes(row.game_duration) < 15]

    if matches_to_delete:
        conn.execute(text(f"DELETE FROM performance_features WHERE participant_id IN (SELECT participant_id FROM participants WHERE team_id IN (SELECT team_id FROM teams WHERE match_id IN ({','.join(map(str, matches_to_delete))})))"))
        conn.execute(text(f"DELETE FROM participants WHERE team_id IN (SELECT team_id FROM teams WHERE match_id IN ({','.join(map(str, matches_to_delete))}))"))
        conn.execute(text(f"DELETE FROM teams WHERE match_id IN ({','.join(map(str, matches_to_delete))})"))
        conn.execute(text(f"DELETE FROM matches WHERE match_id IN ({','.join(map(str, matches_to_delete))})"))
        print(f"Removed {len(matches_to_delete)} short matches.")

    # 2. Create temporary tables
    conn.execute(text("CREATE TABLE temp_matches AS SELECT * FROM matches"))
    conn.execute(text("CREATE TABLE temp_teams AS SELECT * FROM teams"))
    conn.execute(text("CREATE TABLE temp_participants AS SELECT * FROM participants"))
    conn.execute(text("CREATE TABLE temp_performance_features AS SELECT * FROM performance_features"))
    
    # 3. Update IDs in temporary tables
    conn.execute(text("UPDATE temp_matches SET match_id = (SELECT COUNT(*) FROM temp_matches m2 WHERE m2.match_id <= temp_matches.match_id)"))
    conn.execute(text("UPDATE temp_teams SET team_id = (SELECT COUNT(*) FROM temp_teams t2 WHERE t2.team_id <= temp_teams.team_id)"))
    conn.execute(text("UPDATE temp_participants SET participant_id = (SELECT COUNT(*) FROM temp_participants p2 WHERE p2.participant_id <= temp_participants.participant_id)"))

    # 4. Update foreign keys in temporary tables
    conn.execute(text("UPDATE temp_teams SET match_id = (SELECT match_id FROM temp_matches WHERE temp_matches.game_id = temp_teams.match_id)"))
    conn.execute(text("UPDATE temp_participants SET team_id = (SELECT team_id FROM temp_teams WHERE temp_teams.team_id = temp_participants.team_id)"))

    # 5. Update performance_features with new participant_ids, removing any that no longer have a matching participant
    conn.execute(text("""
    DELETE FROM temp_performance_features 
    WHERE participant_id NOT IN (SELECT participant_id FROM temp_participants)
    """))
    conn.execute(text("""
    UPDATE temp_performance_features 
    SET participant_id = (
        SELECT new.participant_id 
        FROM temp_participants new 
        JOIN participants old ON old.participant_id = temp_performance_features.participant_id 
        WHERE new.summoner_id = old.summoner_id 
        AND new.champion_id = old.champion_id
    )
    """))

    # 6. Replace original tables with temporary tables
    conn.execute(text("DROP TABLE matches"))
    conn.execute(text("DROP TABLE teams"))
    conn.execute(text("DROP TABLE participants"))
    conn.execute(text("DROP TABLE performance_features"))

    conn.execute(text("ALTER TABLE temp_matches RENAME TO matches"))
    conn.execute(text("ALTER TABLE temp_teams RENAME TO teams"))
    conn.execute(text("ALTER TABLE temp_participants RENAME TO participants"))
    conn.execute(text("ALTER TABLE temp_performance_features RENAME TO performance_features"))

    conn.commit()

print("Short matches have been removed and IDs have been resequenced.")