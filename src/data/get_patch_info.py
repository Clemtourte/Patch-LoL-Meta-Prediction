#!/usr/bin/env python
import requests
import re
import unicodedata
import logging
from packaging import version
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Base, ChampionStats, ItemStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine("sqlite:///../../datasets/league_data.db")
SessionMaker = sessionmaker(bind=engine)

def get_available_versions() -> list[str]:
    """
    Retrieves the list of available patch versions from DDragon.
    """
    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def get_champion_stats_ddragon(patch: str) -> dict:
    """
    Fetches champion data (including base stats) from Data Dragon for a given patch.
    """
    url = f"https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/championFull.json"
    logger.info(f"Fetching champion stats from DDragon: {url}")
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    # 'data' contains a dictionary with champion keys and their data
    return data.get("data", {})

def get_item_data_ddragon(patch: str) -> dict:
    """
    Fetches item data from Data Dragon for a given patch.
    """
    url = f"https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/item.json"
    logger.info(f"Fetching item data from DDragon: {url}")
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def store_champion_data(session, patch: str, champion_data: dict) -> None:
    """
    Stores champion base stats in the ChampionStats table.
    """
    for champ_key, data in champion_data.items():
        try:
            # 'stats' typically holds base HP, MP, armor, etc.
            stats = data.get('stats', {})
            session.merge(
                ChampionStats(
                    version=patch,
                    champion=champ_key,
                    hp=stats.get('hp', 0),
                    mp=stats.get('mp', 0),
                    armor=stats.get('armor', 0),
                    spellblock=stats.get('spellblock', 0),
                    attackdamage=stats.get('attackdamage', 0),
                    attackspeed=stats.get('attackspeed', 0),
                    hpperlevel=stats.get('hpperlevel', 0),
                    mpperlevel=stats.get('mpperlevel', 0),
                    armorperlevel=stats.get('armorperlevel', 0),
                    spellblockperlevel=stats.get('spellblockperlevel', 0),
                    attackdamageperlevel=stats.get('attackdamageperlevel', 0),
                    attackspeedperlevel=stats.get('attackspeedperlevel', 0),
                    attackrange=stats.get('attackrange', 0),
                    movespeed=stats.get('movespeed', 0),
                    crit=stats.get('crit', 0),
                    critperlevel=stats.get('critperlevel', 0)
                )
            )
            session.commit()
        except Exception as e:
            logger.error(f"Error storing champion data for {champ_key}: {e}")
            session.rollback()

def store_item_data(session, patch: str, item_data: dict) -> None:
    """
    Stores item data in the ItemStats table.
    """
    items = item_data.get("data", {})
    for item_id, data in items.items():
        try:
            gold_info = data.get('gold', {})
            session.merge(
                ItemStats(
                    version=patch,
                    item_id=int(item_id),
                    name=data.get('name', ''),
                    description=data.get('description', ''),
                    plaintext=data.get('plaintext', ''),
                    total_gold=gold_info.get('total', 0),
                    base_gold=gold_info.get('base', 0),
                    sell_gold=gold_info.get('sell', 0),
                    purchasable=gold_info.get('purchasable', False),
                    tags=",".join(data.get('tags', []))
                )
            )
            session.commit()
        except Exception as e:
            logger.error(f"Error storing item {item_id}: {e}")
            session.rollback()

def fetch_and_store_patch_data(start_version: str, end_version: str) -> None:
    """
    Fetches champion + item data (excluding spells) from DDragon for all versions
    between start_version and end_version, then stores them in the database.
    """
    # Ensure tables exist
    Base.metadata.create_all(engine)
    session = SessionMaker()

    try:
        all_versions = get_available_versions()
        start_ver = version.parse(start_version)
        end_ver = version.parse(end_version)

        # Filter versions that match the pattern and are in range
        versions_to_fetch = [
            v for v in all_versions
            if re.match(r'^\d+\.\d+\.\d+$', v)
            and start_ver <= version.parse(v) <= end_ver
        ]
        # Sort them descending so the latest patch is processed first
        versions_to_fetch.sort(key=version.parse, reverse=True)

        for version_str in versions_to_fetch:
            logger.info(f"Processing version {version_str}")
            try:
                # 1) Store champion base stats
                champ_data = get_champion_stats_ddragon(version_str)
                store_champion_data(session, version_str, champ_data)
                logger.info(f"Stored champion base stats for version {version_str}")

                # 2) Store item data
                item_data = get_item_data_ddragon(version_str)
                store_item_data(session, version_str, item_data)
                logger.info(f"Stored item data for version {version_str}")

            except Exception as e:
                logger.error(f"Error fetching/storing data for {version_str}: {e}")

    except Exception as e:
        logger.error(f"Error in patch processing: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    # Example usage: fetch data from patch 13.1.1 up to patch 13.5.1
    fetch_and_store_patch_data("13.1.1", "14.24.1")
