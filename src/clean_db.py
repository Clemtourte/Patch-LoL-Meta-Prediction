#!/usr/bin/env python
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import SpellStats, ChampionStats, ItemStats, PatchChanges

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def empty_tables():
    engine = create_engine('sqlite:///../datasets/league_data.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        deleted_spellstats = session.query(SpellStats).delete()
        deleted_championstats = session.query(ChampionStats).delete()
        deleted_itemstats = session.query(ItemStats).delete()
        deleted_patchchanges = session.query(PatchChanges).delete()
        session.commit()
        logger.info(f"Emptied SpellStats table. Deleted {deleted_spellstats} rows.")
        logger.info(f"Emptied ChampionStats table. Deleted {deleted_championstats} rows.")
        logger.info(f"Emptied ItemStats table. Deleted {deleted_itemstats} rows.")
        logger.info(f"Emptied PatchChanges table. Deleted {deleted_patchchanges} rows.")
    except Exception as e:
        logger.error(f"Error emptying tables: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    empty_tables()
