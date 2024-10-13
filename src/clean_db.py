import os
from sqlalchemy import create_engine, inspect, text

# Get the current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Check if the database file exists
db_file = 'matches.db'
if os.path.exists(db_file):
    print(f"Database file '{db_file}' exists.")
    
    # Create an engine and inspect the database
    engine = create_engine(f'sqlite:///{db_file}')
    inspector = inspect(engine)
    
    # Print table names and row counts
    with engine.connect() as connection:
        for table_name in inspector.get_table_names():
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()
            print(f"Table '{table_name}' has {row_count} rows.")
else:
    print(f"Database file '{db_file}' does not exist in the current directory.")