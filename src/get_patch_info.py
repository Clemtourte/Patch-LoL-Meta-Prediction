import requests
import json
import tarfile
import logging
from io import BytesIO
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session as SQLSession, sessionmaker
from models import Base, ChampionStats, SpellStats, ItemStats
from packaging import version
from typing import Dict, List, Any, Optional, Set
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine('sqlite:///../datasets/league_data.db')
SessionMaker = sessionmaker(bind=engine)

def get_available_versions() -> List[str]:
    """Fetch available versions from Dragon API"""
    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def download_data_dragon(version: str) -> BytesIO:
    """Download Dragon data for a specific version"""
    url = f"https://ddragon.leagueoflegends.com/cdn/dragontail-{version}.tgz"
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)

def extract_data(tar_file: tarfile.TarFile, version: str, file_name: str) -> Dict:
    """Extract data from tar file"""
    file_path = f"{version}/data/en_US/{file_name}.json"
    return json.loads(tar_file.extractfile(file_path).read().decode('utf-8'))['data']

def get_damage_type(spell: Dict[str, Any]) -> List[str]:
    """Extract damage type from spell data"""
    text = (spell.get('tooltip', '') + ' ' + spell.get('description', '')).lower()
    if '<magicdamage>' in text or 'magic damage' in text:
        return ['magic']
    elif '<physicaldamage>' in text or 'physical damage' in text:
        return ['physical']
    elif '<truedamage>' in text or 'true damage' in text:
        return ['true']
    return ['none']

def process_spell_data(spell: Dict[str, Any]) -> Dict[str, Any]:
    """Process spell data focusing on reliable stats"""
    effects = spell.get('effect', [])
    tooltip = spell.get('tooltip', '').lower()
    
    # Get all non-zero effect values
    effect_values = {}
    for i, effect in enumerate(effects):
        if effect and any(v != 0 for v in (effect or [])):
            effect_values[f'e{i}'] = effect
    
    # Find base damage (typically in first few effects with 5 values)
    base_damage = None
    if effects:
        for i in range(1, min(4, len(effects))):
            if effects[i] and len(effects[i]) == 5:
                if all(isinstance(v, (int, float)) and v > 0 for v in effects[i]):
                    if f'e{i}' in tooltip and ('damage' in tooltip or 'deal' in tooltip):
                        base_damage = effects[i]
                        break
    
    # Get CC duration if present
    cc_duration = None
    cc_keywords = ['stun', 'root', 'snare', 'knockup', 'airborne']
    if any(keyword in tooltip for keyword in cc_keywords):
        for i, effect in enumerate(effects):
            if effect and len(effect) == 5:
                if all(0 < v <= 5 for v in effect):  # CC usually 0-5 seconds
                    if f'e{i}' in tooltip:
                        cc_duration = effect
                        break
    
    return {
        'damage_types': get_damage_type(spell),
        'damage_values': effect_values,
        'base_damage': base_damage,
        'cc_duration': cc_duration,
        'resource_type': spell.get('costType', 'None'),
        'cooldown': spell.get('cooldown', []),
        'cost': spell.get('cost', []),
        'range': spell.get('range', []),
        # Keep ratio fields but set to None
        'ratios': {
            'ap_ratio': None,
            'ad_ratio': None,
            'bonus_ad_ratio': None,
            'hp_ratio': None,
            'target_hp_ratio': None
        }
    }

def store_spell_data(session: SQLSession, version: str, champion_data: Dict[str, Any]) -> None:
    """Store spell data with better error handling"""
    for champion, data in champion_data.items():
        logger.info(f"Processing {champion}")
        try:
            # Handle active abilities
            for spell_index, spell in enumerate(data['spells']):
                spell_data = process_spell_data(spell)
                session.merge(SpellStats(
                    version=version,
                    champion=champion,
                    spell_id=spell['id'],
                    standardized_id=f"{champion}_{['Q', 'W', 'E', 'R'][spell_index]}",
                    spell_name=spell['name'],
                    damage_type=','.join(spell_data['damage_types']),
                    damage_values=json.dumps(spell_data['damage_values']),
                    base_damage=json.dumps(spell_data['base_damage']),
                    max_rank=spell.get('maxrank', 5),
                    cooldown=json.dumps(spell_data['cooldown']),
                    cost=json.dumps(spell_data['cost']),
                    range=json.dumps(spell_data['range']),
                    resource_type=spell_data['resource_type'],
                    description=spell.get('description', ''),
                    is_passive=False,
                    ap_ratio=json.dumps(spell_data['ratios']['ap_ratio']),
                    ad_ratio=json.dumps(spell_data['ratios']['ad_ratio']),
                    bonus_ad_ratio=json.dumps(spell_data['ratios']['bonus_ad_ratio']),
                    hp_ratio=json.dumps(spell_data['ratios']['hp_ratio']),
                    target_hp_ratio=json.dumps(spell_data['ratios']['target_hp_ratio']),
                    cc_effects=json.dumps(spell_data['cc_duration'])
                ))
            
            # Handle passive
            passive = data['passive']
            passive_data = process_spell_data({
                'tooltip': passive.get('description', ''),
                'description': passive.get('description', ''),
                'effect': [],
                'costType': 'None'
            })
            
            session.merge(SpellStats(
                version=version,
                champion=champion,
                spell_id=f"{champion}_P",
                standardized_id=f"{champion}_P",
                spell_name=passive['name'],
                damage_type=','.join(passive_data['damage_types']),
                damage_values=json.dumps(passive_data['damage_values']),
                base_damage=json.dumps(passive_data['base_damage']),
                max_rank=1,
                cooldown=json.dumps([]),
                cost=json.dumps([]),
                range=json.dumps([]),
                resource_type='None',
                description=passive.get('description', ''),
                is_passive=True,
                ap_ratio=json.dumps(passive_data['ratios']['ap_ratio']),
                ad_ratio=json.dumps(passive_data['ratios']['ad_ratio']),
                bonus_ad_ratio=json.dumps(passive_data['ratios']['bonus_ad_ratio']),
                hp_ratio=json.dumps(passive_data['ratios']['hp_ratio']),
                target_hp_ratio=json.dumps(passive_data['ratios']['target_hp_ratio']),
                cc_effects=json.dumps(passive_data['cc_duration'])
            ))
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Error processing {champion}: {str(e)}")
            session.rollback()

def store_champion_data(session: SQLSession, version: str, champion_data: Dict[str, Any]) -> None:
    """Store champion base stats"""
    for champion, data in champion_data.items():
        if 'stats' not in data:
            continue
            
        stats = data['stats']
        session.merge(ChampionStats(
            version=version,
            champion=champion,
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
    
    try:
        session.commit()
    except Exception as e:
        logger.error(f"Error storing champion data: {str(e)}")
        session.rollback()

def store_item_data(session: SQLSession, version: str, item_data: Dict[str, Any]) -> None:
    """Store item data"""
    for item_id, data in item_data.items():
        try:
            session.merge(ItemStats(
                version=version,
                item_id=int(item_id),
                name=data['name'],
                description=data.get('description', ''),
                plaintext=data.get('plaintext', ''),
                total_gold=data['gold']['total'],
                base_gold=data['gold']['base'],
                sell_gold=data['gold']['sell'],
                purchasable=data['gold']['purchasable'],
                tags=','.join(data.get('tags', []))
            ))
            session.commit()
        except Exception as e:
            logger.error(f"Error storing item {item_id}: {str(e)}")
            session.rollback()

def fetch_and_store_patch_data(start_version: str, end_version: str) -> None:
    """Main function to fetch and store patch data"""
    Base.metadata.create_all(engine)
    session = SessionMaker()

    try:
        all_versions = get_available_versions()
        start = version.parse(start_version)
        end = version.parse(end_version)
        
        versions_to_fetch = [
            v for v in all_versions 
            if re.match(r'^\d+\.\d+\.\d+$', v) and 
            start <= version.parse(v) <= end
        ]
        
        versions_to_fetch.sort(key=version.parse, reverse=True)
        
        for version_str in versions_to_fetch:
            logger.info(f"Processing version {version_str}")
            try:
                tar_file = tarfile.open(fileobj=download_data_dragon(version_str))
                
                # Process champion data
                champion_data = extract_data(tar_file, version_str, 'championFull')
                store_champion_data(session, version_str, champion_data)
                store_spell_data(session, version_str, champion_data)
                logger.info(f"Stored champion and spell data for version {version_str}")
                
                # Process item data
                item_data = extract_data(tar_file, version_str, 'item')
                store_item_data(session, version_str, item_data)
                logger.info(f"Stored item data for version {version_str}")
                
            except Exception as e:
                logger.error(f"Error processing version {version_str}: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Error in patch processing: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    fetch_and_store_patch_data("13.1.1", "14.24.1")