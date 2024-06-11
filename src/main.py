import sqlite3
from user import User

def create_connection(db_file):
    """ Create a database connection to the SQLite database specified by db_file """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print("SQLite version:", sqlite3.version)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    """ Create a table for storing user information """
    try:
        sql_create_user_table = """ CREATE TABLE IF NOT EXISTS users (
                                        id integer PRIMARY KEY,
                                        username text NOT NULL,
                                        tag text NOT NULL,
                                        region text NOT NULL,
                                        puuid text NOT NULL,
                                        tier text,
                                        rank text,
                                        league_points integer
                                    ); """
        c = conn.cursor()
        c.execute(sql_create_user_table)
        print("User table created")
    except sqlite3.Error as e:
        print(e)

# Main execution
database = "user_data.db"
conn = create_connection(database)
if conn:
    create_table(conn)

    # Create a User object and store its information in the database
    user = User('MenuMaxiBestFlop', 'EUW', 'EUW1')
    user.store_user_info(conn)

    # Close the connection
    conn.close()
