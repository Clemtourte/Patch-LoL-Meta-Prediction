from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import SpellStats, ChampionStats, ItemStats

def clean_tables():
    engine = create_engine('sqlite:///../datasets/league_data.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        print("Deleting records from SpellStats...")
        session.query(SpellStats).delete()
        
        print("Deleting records from ChampionStats...")
        session.query(ChampionStats).delete()
        
        print("Deleting records from ItemStats...")
        session.query(ItemStats).delete()
        
        session.commit()
        print("All tables cleaned successfully!")
        
    except Exception as e:
        print(f"Error while cleaning tables: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    clean_tables()