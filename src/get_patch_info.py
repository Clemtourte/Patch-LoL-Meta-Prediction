import requests
import json
import sqlite3
import tarfile
from io import BytesIO

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

def create_database():
    conn = sqlite3.connect("sqlite:///../datasets/league_patches.db")
    c = conn.cursor()
    
    # Champion stats table
    c.execute('''CREATE TABLE IF NOT EXISTS champion_stats
                 (version TEXT, champion TEXT, 
                  hp REAL, mp REAL, armor REAL, spellblock REAL,
                  attackdamage REAL, attackspeed REAL, 
                  hpperlevel REAL, mpperlevel REAL, armorperlevel REAL,
                  spellblockperlevel REAL, attackdamageperlevel REAL,
                  attackspeedperlevel REAL, attackrange REAL,
                  movespeed REAL, crit REAL, critperlevel REAL,
                  PRIMARY KEY (version, champion))''')
    
    # Item stats table
    c.execute('''CREATE TABLE IF NOT EXISTS item_stats
                 (version TEXT, item_id INTEGER, name TEXT,
                  description TEXT, plaintext TEXT,
                  total_gold INTEGER, base_gold INTEGER, sell_gold INTEGER,
                  purchasable BOOLEAN, tags TEXT,
                  PRIMARY KEY (version, item_id))''')
    
    conn.commit()
    return conn

def store_champion_data(conn, version, champion_data):
    c = conn.cursor()
    for champion, data in champion_data.items():
        stats = data['stats']
        c.execute('''INSERT OR REPLACE INTO champion_stats VALUES
                     (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (version, champion, 
                   stats['hp'], stats['mp'], stats['armor'], stats['spellblock'],
                   stats['attackdamage'], stats['attackspeed'],
                   stats['hpperlevel'], stats['mpperlevel'], stats['armorperlevel'],
                   stats['spellblockperlevel'], stats['attackdamageperlevel'],
                   stats['attackspeedperlevel'], stats['attackrange'],
                   stats['movespeed'], stats['crit'], stats['critperlevel']))
    conn.commit()

def store_item_data(conn, version, item_data):
    c = conn.cursor()
    for item_id, data in item_data.items():
        c.execute('''INSERT OR REPLACE INTO item_stats VALUES
                     (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (version, item_id, data['name'],
                   data.get('description', ''), data.get('plaintext', ''),
                   data['gold']['total'], data['gold']['base'], data['gold']['sell'],
                   data['gold']['purchasable'], ','.join(data.get('tags', []))))
    conn.commit()

def fetch_and_store_patch_data(versions_to_fetch=None):
    conn = create_database()
    if versions_to_fetch is None:
        versions_to_fetch = get_available_versions()
    
    for version in versions_to_fetch:
        print(f"Processing version {version}")
        try:
            tar_file = tarfile.open(fileobj=download_data_dragon(version))
            
            # Process champion data
            champion_data = extract_data(tar_file, version, 'champion')
            store_champion_data(conn, version, champion_data)
            print(f"Stored champion data for version {version}")
            
            # Process item data
            item_data = extract_data(tar_file, version, 'item')
            store_item_data(conn, version, item_data)
            print(f"Stored item data for version {version}")
            
        except Exception as e:
            print(f"Error processing version {version}: {str(e)}")
    
    conn.close()
fetch_and_store_patch_data()