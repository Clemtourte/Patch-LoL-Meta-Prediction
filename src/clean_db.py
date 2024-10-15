from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker
from models import Base, SpellStats
import os

# Database setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'datasets', 'league_data.db')
ENGINE = create_engine(f'sqlite:///{DB_PATH}')
Session = sessionmaker(bind=ENGINE)

def clear_spell_stats():
    session = Session()
    try:
        session.execute(delete(SpellStats))
        session.commit()
        print("spell_stats table has been cleared successfully.")
    except Exception as e:
        print(f"Error clearing spell_stats table: {str(e)}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    clear_spell_stats()