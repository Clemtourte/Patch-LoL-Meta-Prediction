#!/usr/bin/env python
import requests
import json
import logging
import re
import unicodedata
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from packaging import version
from models import Base, ChampionStats, SpellStats, ItemStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine('sqlite:///../datasets/league_data.db')
SessionMaker = sessionmaker(bind=engine)

def get_available_versions() -> list[str]:
    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def get_champion_stats_ddragon(patch: str) -> dict:
    url = f"https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/championFull.json"
    logger.info(f"Fetching champion stats from DDragon: {url}")
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    return data.get("data", {})

def get_item_data_ddragon(patch: str) -> dict:
    url = f"https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/item.json"
    logger.info(f"Fetching item data from DDragon: {url}")
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def urlify(champion: str) -> str:
    normalized = unicodedata.normalize('NFKD', champion).encode('ascii', 'ignore').decode('utf-8')
    return re.sub(r"[^a-z]", "", normalized.lower())

def get_champion_data_cdragon(champion: str, patch: str) -> dict:
    champ_url = urlify(champion)
    url = f"https://raw.communitydragon.org/{patch}/game/data/characters/{champ_url}/{champ_url}.bin.json"
    logger.info(f"Fetching champion data for {champion} from {url}")
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def get_damage_type(spell: dict) -> str:
    t = (spell.get('tooltip', '') + ' ' + spell.get('description', '')).lower()
    if '<magicdamage>' in t or 'magic damage' in t:
        return 'magic'
    elif '<physicaldamage>' in t or 'physical damage' in t:
        return 'physical'
    elif '<truedamage>' in t or 'true damage' in t:
        return 'true'
    return 'UNKNOWN'

def process_spell_data(spell: dict) -> dict:
    # Helper: try primary key and fallback if needed.
    def parse_list(primary_key, fallback_key=None):
        vals = spell.get(primary_key)
        if vals is None and fallback_key:
            vals = spell.get(fallback_key)
        if vals is None:
            return []
        try:
            return [float(x) for x in vals]
        except Exception:
            return vals

    # Generic logic for damage values:
    damage_values = []
    effects = spell.get("effect", [])
    if isinstance(effects, list):
        for eff in effects:
            if isinstance(eff, list):
                try:
                    # Convert each non-null element to float.
                    damage_values.append([float(x) if x is not None else 0.0 for x in eff])
                except Exception as exc:
                    # If conversion fails, append the original value.
                    damage_values.append(eff)
            else:
                # For non-list entries, skip or handle differently.
                damage_values.append(eff)
    else:
        damage_values = []

    return {
        'damage_type': get_damage_type(spell),
        'max_rank': spell.get('maxrank', 5),
        'cooldown': parse_list('cooldown', fallback_key='cooldownTime'),
        'cost': parse_list('cost'),  # May return [] if not present.
        'range': parse_list('range', fallback_key='castRange'),
        'resource_type': spell.get('costType', 'None'),
        'description': spell.get('description', ''),
        'damage_values': damage_values  # New key for damage numbers.
    }

# --- New Helper Functions for Essential Spells ---
def get_spell_type(last_part: str) -> str:
    lp = last_part.lower()
    if "passiveability" in lp:
        return "Passive"
    elif "qability" in lp:
        return "Q"
    elif "wability" in lp:
        return "W"
    elif "eability" in lp:
        return "E"
    elif "rability" in lp:
        return "R"
    else:
        return None

# --- Modified store_spell_data ---
def store_spell_data(session, patch: str, champion_data: dict) -> None:
    """
    Process and store only essential spell data (Q, W, E, R, Passive) from the champion data.
    """
    for champion, data in champion_data.items():
        logger.info(f"Processing spells for {champion}")
        for key, raw_spell_obj in data.items():
            # Only process keys from the Spells folder
            if not key.startswith("Characters/") or "/Spells/" not in key:
                continue
            last_part = key.split("/")[-1]
            spell_type = get_spell_type(last_part)
            if spell_type is None:
                # Skip non-essential spell variants
                continue

            # Determine a standardized spell identifier (champion_spellType)
            standardized_spell_type = spell_type  # e.g., "Q"
            
            # Get spell resource – sometimes the object is nested
            spell_resource = raw_spell_obj.get("mSpell", raw_spell_obj)
            
            # Try to extract a display name from the tooltip data if available
            mClientData = spell_resource.get("mClientData", {})
            tooltip_data = mClientData.get("mTooltipData", {})
            loc_keys = tooltip_data.get("mLocKeys", {})
            spell_name = loc_keys.get("keyName", last_part)
            
            # Process the spell to extract numerical values
            processed = process_spell_data(spell_resource)
            
            # Optionally, check if the mBuff holds an alternate description
            description = ""
            if "mBuff" in raw_spell_obj and "mTooltipData" in raw_spell_obj["mBuff"]:
                desc_data = raw_spell_obj["mBuff"]["mTooltipData"]
                description = desc_data.get("mObjectName", "")
            if not description:
                description = processed.get("description", "")
            
            session.merge(SpellStats(
                version=patch,
                champion=champion,
                spell_type=standardized_spell_type,
                spell_name=spell_name,
                damage_type=processed['damage_type'],
                max_rank=processed['max_rank'],
                cooldown=processed['cooldown'],  # stored as JSON
                cost=processed['cost'],
                range=processed['range'],
                resource_type=processed['resource_type'],
                description=description
            ))
        session.commit()

# The rest of the functions (store_champion_data, store_item_data, etc.) remain the same.
def store_champion_data(session, patch: str, champion_data: dict) -> None:
    for champ_key, data in champion_data.items():
        try:
            stats = data.get('stats', {})
            session.merge(ChampionStats(
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
            ))
            session.commit()
        except Exception as e:
            logger.error(f"Error storing champion data for {champ_key}: {e}")
            session.rollback()

def store_item_data(session, patch: str, item_data: dict) -> None:
    items = item_data.get("data", {})
    for item_id, data in items.items():
        try:
            gold_info = data.get('gold', {})
            session.merge(ItemStats(
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
            ))
            session.commit()
        except Exception as e:
            logger.error(f"Error storing item {item_id}: {e}")
            session.rollback()

def fetch_and_store_patch_data(start_version: str, end_version: str) -> None:
    Base.metadata.create_all(engine)
    session = SessionMaker()
    try:
        all_vers = get_available_versions()
        from packaging import version
        start = version.parse(start_version)
        end = version.parse(end_version)
        versions_to_fetch = [
            v for v in all_vers
            if re.match(r'^\d+\.\d+\.\d+$', v)
            and start <= version.parse(v) <= end
        ]
        versions_to_fetch.sort(key=version.parse, reverse=True)
        for version_str in versions_to_fetch:
            logger.info(f"Processing version {version_str}")
            cdragon_patch = ".".join(version_str.split(".")[:2])
            try:
                champ_stats = get_champion_stats_ddragon(version_str)
                store_champion_data(session, version_str, champ_stats)
                logger.info(f"Stored champion base stats for version {version_str}")
            except Exception as e:
                logger.error(f"Error fetching/storing champion stats: {e}")
            champion_names = [
                "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Ambessa", "Amumu", "Anivia", "Annie",
                "Aphelios", "Ashe", "Aurelion Sol", "Aurora", "Azir", "Bard", "Bel'Veth", "Blitzcrank",
                "Brand", "Braum", "Briar", "Caitlyn", "Camille", "Cassiopeia", "Cho'Gath", "Corki",
                "Darius", "Diana", "Dr. Mundo", "Draven", "Ekko", "Elise", "Evelynn", "Ezreal", "Fiddlesticks",
                "Fiora", "Fizz", "Galio", "Gangplank", "Garen", "Gnar", "Gragas", "Graves", "Gwen", "Hecarim",
                "Heimerdinger", "Hwei", "Illaoi", "Irelia", "Ivern", "Janna", "Jarvan IV", "Jax", "Jayce",
                "Jhin", "Jinx", "K'Santé", "Kai'Sa", "Kalista", "Karma", "Karthus", "Kassadin", "Katarina",
                "Kayle", "Kayn", "Kennen", "Kha'Zix", "Kindred", "Kled", "Kog'Maw", "LeBlanc", "Lee Sin",
                "Leona", "Lillia", "Lissandra", "Lucian", "Lulu", "Lux", "Malphite", "Malzahar", "Maokai",
                "Maître Yi", "Mel", "Milio", "Miss Fortune", "Mordekaiser", "Morgana", "Naafiri", "Nami",
                "Nasus", "Nautilus", "Neeko", "Nidalee", "Nilah", "Nocturne", "Nunu et Willump", "Olaf",
                "Orianna", "Ornn", "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan", "Rammus",
                "Rek'Sai", "Rell", "Renata Glasc", "Renekton", "Rengar", "Riven", "Rumble", "Ryze", "Samira",
                "Sejuani", "Senna", "Seraphine", "Sett", "Shaco", "Shen", "Shyvana", "Singed", "Sion",
                "Sivir", "Skarner", "Smolder", "Sona", "Soraka", "Swain", "Sylas", "Syndra", "Séraphine",
                "Tahm Kench", "Taliyah", "Talon", "Taric", "Teemo", "Thresh", "Tristana", "Trundle",
                "Tryndamere", "Twisted Fate", "Twitch", "Udyr", "Urgot", "Varus", "Vayne", "Veigar",
                "Vel'Koz", "Vex", "Vi", "Viego", "Viktor", "Vladimir", "Volibear", "Warwick", "Wukong",
                "Xayah", "Xerath", "Xin Zhao", "Yasuo", "Yone", "Yorick", "Yuumi", "Zac", "Zed", "Zeri",
                "Ziggs", "Zilean", "Zoé", "Zyra"
            ]
            champion_spell_data = {}
            for champ in champion_names:
                try:
                    cdragon_data = get_champion_data_cdragon(champ, cdragon_patch)
                    champion_spell_data[champ] = cdragon_data
                except Exception as exc:
                    logger.error(f"Error fetching spell data for {champ}: {exc}")
                    continue
            store_spell_data(session, cdragon_patch, champion_spell_data)
            logger.info(f"Stored champion spell data for version {cdragon_patch}")
            try:
                item_data = get_item_data_ddragon(version_str)
                store_item_data(session, version_str, item_data)
                logger.info(f"Stored item data for version {version_str}")
            except Exception as e:
                logger.error(f"Error fetching item data: {e}")
    except Exception as e:
        logger.error(f"Error in patch processing: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    fetch_and_store_patch_data("14.19.1", "14.19.1")
