#!/usr/bin/env python
import sqlite3

def print_spell_stats(db_path: str = "../datasets/league_data.db"):
    # Connect to your local SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query the entire spell_stats table (limit to 50 rows if you want less)
    cursor.execute("SELECT * FROM spell_stats LIMIT 50")
    rows = cursor.fetchall()
    
    # Print the column names first
    col_names = [desc[0] for desc in cursor.description]
    print(" | ".join(col_names))
    print("-" * 80)
    
    # Print each row
    for row in rows:
        print(row)
    
    # Close
    conn.close()

if __name__ == "__main__":
    print_spell_stats()
