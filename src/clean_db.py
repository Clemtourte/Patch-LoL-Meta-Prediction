from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import ChampionWinrates

engine = create_engine('sqlite:///../datasets/league_data.db')
Session = sessionmaker(bind=engine)
session = Session()

try:
    # Delete all records where patch is '13.2'
    deleted = session.query(ChampionWinrates)\
        .filter(ChampionWinrates.patch == '13.2')\
        .delete()
    session.commit()
    print(f"Successfully deleted {deleted} records for patch 13.2")
except Exception as e:
    session.rollback()
    print(f"Error deleting records: {str(e)}")
finally:
    session.close()