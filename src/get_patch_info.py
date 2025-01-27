import requests
import json
import tarfile
import logging
from io import BytesIO
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, ChampionStats, SpellStats, ItemStats
from packaging import version
import re


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

engine = create_engine(f'sqlite:///../datasets/league_data.db')
Session = sessionmaker(bind=engine)

def get_available_versions():
    url = "https://ddragon.leagueoflegends.com/api/versions.json"
    response = requests.get(url)
    return response.json()

def download_data_dragon(version):
    url = f"https://ddragon.leagueoflegends.com/cdn/dragontail-{version}.tgz"
    response = requests.get(url)
    return BytesIO(response.content)

def extract_data(tar_file, version, file_name):
    file_path = f"{version}/data/en_US/{file_name}.json"
    data = json.loads(tar_file.extractfile(file_path).read().decode('utf-8'))
    return data['data']

def store_champion_data(session, version, champion_data):
    for champion, data in champion_data.items():
        stats = data['stats']
        champ_stats = ChampionStats(
            version=version,
            champion=champion,
            hp=stats['hp'],
            mp=stats['mp'],
            armor=stats['armor'],
            spellblock=stats['spellblock'],
            attackdamage=stats['attackdamage'],
            attackspeed=stats['attackspeed'],
            hpperlevel=stats['hpperlevel'],
            mpperlevel=stats['mpperlevel'],
            armorperlevel=stats['armorperlevel'],
            spellblockperlevel=stats['spellblockperlevel'],
            attackdamageperlevel=stats['attackdamageperlevel'],
            attackspeedperlevel=stats['attackspeedperlevel'],
            attackrange=stats['attackrange'],
            movespeed=stats['movespeed'],
            crit=stats['crit'],
            critperlevel=stats['critperlevel']
        )
        session.merge(champ_stats)
    session.commit()

def store_spell_data(session, version, champion_data):
    for champion, data in champion_data.items():
        for spell in data['spells']:
            spell_stats = SpellStats(
                version=version,
                champion=champion,
                spell_id=spell['id'],
                spell_name=spell['name'],
                damage_type=get_damage_type(spell),
                damage_values=json.dumps(spell.get('effect', [])[1] if spell.get('effect') else []),
                max_rank=spell.get('maxrank', 5),
                cooldown=json.dumps(spell.get('cooldown', [])),
                cost=json.dumps(spell.get('cost', [])),
                range=json.dumps(spell.get('range', [])),
                resource=spell.get('resource', ''),
                description=spell.get('description', ''),
                is_passive=False
            )
            session.merge(spell_stats)
        
        passive = data['passive']
        passive_stats = SpellStats(
            version=version,
            champion=champion,
            spell_id=f"{champion}_P",
            spell_name=passive['name'],
            damage_type='unknown',
            damage_values=json.dumps([]),
            max_rank=1,
            cooldown=json.dumps([]),
            cost=json.dumps([]),
            range=json.dumps([]),
            resource='',
            description=passive.get('description', ''),
            is_passive=True
        )
        session.merge(passive_stats)
    session.commit()

def store_item_data(session, version, item_data):
    for item_id, data in item_data.items():
        item_stats = ItemStats(
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
        )
        session.merge(item_stats)
    session.commit()

def get_damage_type(spell):
    description = spell.get('description', '').lower()
    if 'magic damage' in description:
        return 'magic'
    elif 'physical damage' in description:
        return 'physical'
    elif 'true damage' in description:
        return 'true'
    else:
        return 'unknown'

def is_valid_version(v):

    return re.match(r'^\d+\.\d+\.\d+$', v) is not None

def fetch_and_store_patch_data(start_version, end_version):
    Base.metadata.create_all(engine)
    session = Session()

    all_versions = get_available_versions()
    
    start = version.parse(start_version)
    end = version.parse(end_version)
    
    versions_to_fetch = []
    for v in all_versions:
        if is_valid_version(v):
            try:
                parsed_v = version.parse(v)
                if start <= parsed_v <= end:
                    versions_to_fetch.append(v)
            except version.InvalidVersion:
                print(f"Skipping invalid version: {v}")
    
    versions_to_fetch.sort(key=version.parse, reverse=True)
    
    for version_str in versions_to_fetch:
        print(f"Processing version {version_str}")
        try:
            tar_file = tarfile.open(fileobj=download_data_dragon(version_str))
            
            champion_data = extract_data(tar_file, version_str, 'championFull')
            store_champion_data(session, version_str, champion_data)
            store_spell_data(session, version_str, champion_data)
            print(f"Stored champion and spell data for version {version_str}")
            
            item_data = extract_data(tar_file, version_str, 'item')
            store_item_data(session, version_str, item_data)
            print(f"Stored item data for version {version_str}")
            
        except Exception as e:
            print(f"Error processing version {version_str}: {str(e)}")
    
    session.close()

if __name__ == "__main__":
    fetch_and_store_patch_data("13.1.1", "13.24.1")