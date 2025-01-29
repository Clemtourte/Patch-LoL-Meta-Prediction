import requests
import json
import tarfile
import logging
import re
from io import BytesIO
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, ChampionStats, SpellStats, ItemStats
from packaging import version
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DragonAPI:
    def __init__(self, db_path: str = 'sqlite:///../datasets/league_data.db'):
        self.engine = create_engine(db_path)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def get_available_versions(self) -> List[str]:
        """Fetch available versions from Dragon API"""
        try:
            response = requests.get("https://ddragon.leagueoflegends.com/api/versions.json")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch versions: {e}")
            return []

    def download_dragon_data(self, version: str) -> Optional[BytesIO]:
        """Download Dragon data for a specific version"""
        try:
            url = f"https://ddragon.leagueoflegends.com/cdn/dragontail-{version}.tgz"
            response = requests.get(url)
            response.raise_for_status()
            return BytesIO(response.content)
        except Exception as e:
            logger.error(f"Failed to download version {version}: {e}")
            return None

class DataParser:
    @staticmethod
    def parse_damage_value(text: str) -> Dict[str, Any]:
        """Parse damage values from text including scaling"""
        damage_info = {
            'base_value': None,
            'ap_ratio': None,
            'ad_ratio': None,
            'bonus_ad_ratio': None,
            'hp_ratio': None,
            'target_hp_ratio': None,
            'armor_ratio': None,
            'mr_ratio': None
        }

        # Extract base values and ratios using regex
        base_pattern = r'(\d+(?:\.\d+)?)'
        ratio_patterns = {
            'ap_ratio': r'(\d+(?:\.\d+)?)%?\s*AP',
            'ad_ratio': r'(\d+(?:\.\d+)?)%?\s*(?:AD|Attack Damage)',
            'bonus_ad_ratio': r'(\d+(?:\.\d+)?)%?\s*bonus\s*AD',
            'hp_ratio': r'(\d+(?:\.\d+)?)%?\s*(?:maximum|max)?\s*HP',
            'target_hp_ratio': r'(\d+(?:\.\d+)?)%?\s*target(?:\'s)?\s*(?:maximum|max)?\s*HP',
            'armor_ratio': r'(\d+(?:\.\d+)?)%?\s*Armor',
            'mr_ratio': r'(\d+(?:\.\d+)?)%?\s*Magic\s*Resist'
        }

        # Extract base damage
        base_matches = re.findall(base_pattern, text)
        if base_matches:
            damage_info['base_value'] = float(base_matches[0])

        # Extract ratios
        for ratio_type, pattern in ratio_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                damage_info[ratio_type] = float(matches[0]) / 100

        return damage_info

    @staticmethod
    def parse_tooltip(tooltip: str) -> Dict[str, Any]:
        """Parse ability tooltip for detailed information"""
        tooltip_data = {
            'damage_types': [],
            'scalings': {},
            'cc_effects': {},
            'effects': {}
        }

        # Extract damage types
        damage_tags = {
            'magic': r'<magicDamage>(.*?)</magicDamage>',
            'physical': r'<physicalDamage>(.*?)</physicalDamage>',
            'true': r'<trueDamage>(.*?)</trueDamage>'
        }

        for damage_type, pattern in damage_tags.items():
            if re.search(pattern, tooltip, re.IGNORECASE):
                tooltip_data['damage_types'].append(damage_type)
                # Parse damage values within tags
                matches = re.finditer(pattern, tooltip, re.IGNORECASE)
                for match in matches:
                    damage_text = match.group(1)
                    damage_info = DataParser.parse_damage_value(damage_text)
                    tooltip_data['scalings'][f'{damage_type}_damage'] = damage_info

        # Extract CC effects
        cc_patterns = {
            'stun': r'(?:Stun|Stunned).*?(\d+(?:\.\d+)?)\s*seconds?',
            'root': r'(?:Root|Rooted).*?(\d+(?:\.\d+)?)\s*seconds?',
            'silence': r'(?:Silence|Silenced).*?(\d+(?:\.\d+)?)\s*seconds?',
            'slow': r'(?:Slow|Slowed).*?(\d+)%',
            'knockup': r'(?:Knock|Knocked)\s*Up.*?(\d+(?:\.\d+)?)\s*seconds?',
            'charm': r'(?:Charm|Charmed).*?(\d+(?:\.\d+)?)\s*seconds?',
            'fear': r'(?:Fear|Feared).*?(\d+(?:\.\d+)?)\s*seconds?',
            'taunt': r'(?:Taunt|Taunted).*?(\d+(?:\.\d+)?)\s*seconds?'
        }

        for cc_type, pattern in cc_patterns.items():
            matches = re.finditer(pattern, tooltip, re.IGNORECASE)
            for match in matches:
                duration = float(match.group(1))
                if cc_type not in tooltip_data['cc_effects']:
                    tooltip_data['cc_effects'][cc_type] = []
                tooltip_data['cc_effects'][cc_type].append(duration)

        return tooltip_data

    @staticmethod
    def parse_effect_values(effect_str: str) -> List[float]:
        """Parse effect values into a list of floats"""
        if not effect_str or effect_str == "":
            return []
        try:
            values = effect_str.strip().split('/')
            return [float(x) for x in values if x and x != ""]
        except (ValueError, AttributeError):
            return []

class ChampionProcessor:
    def __init__(self, session):
        self.session = session

    def process_champion(self, champion_data: Dict[str, Any], version: str):
        """Process champion data and store in database"""
        try:
            # Store basic stats
            self._store_champion_stats(champion_data, version)
            # Store abilities
            self._store_champion_abilities(champion_data, version)
        except Exception as e:
            logger.error(f"Error processing champion {champion_data.get('id', 'unknown')}: {e}")
            self.session.rollback()

    def _store_champion_stats(self, champion_data: Dict[str, Any], version: str):
        """Store champion base stats"""
        stats = champion_data.get('stats', {})
        champ_stats = ChampionStats(
            version=version,
            champion=champion_data['id'],
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
        self.session.merge(champ_stats)
        self.session.commit()

    def _store_champion_abilities(self, champion_data: Dict[str, Any], version: str):
        """Store champion abilities including passive"""
        # Process passive
        passive = champion_data.get('passive', {})
        if passive:
            self._store_ability(
                champion_data['id'],
                passive,
                version,
                is_passive=True,
                spell_key='P'
            )

        # Process active abilities
        spells = champion_data.get('spells', [])
        for index, spell in enumerate(spells):
            spell_key = ['Q', 'W', 'E', 'R'][index]
            self._store_ability(
                champion_data['id'],
                spell,
                version,
                is_passive=False,
                spell_key=spell_key
            )

    def _store_ability(self, champion_id: str, ability: Dict[str, Any], version: str, 
                      is_passive: bool, spell_key: str):
        """Store a single ability"""
        try:
            # Parse tooltip for detailed information
            tooltip_data = DataParser.parse_tooltip(ability.get('tooltip', ''))
            
            # Process effect values
            effect_values = {}
            if not is_passive:
                for i, effect in enumerate(ability.get('effectBurn', [])):
                    if effect and effect != "0":
                        effect_values[f'e{i}'] = DataParser.parse_effect_values(effect)

            spell_stats = SpellStats(
                version=version,
                champion=champion_id,
                spell_id=ability.get('id', f"{champion_id}_{spell_key}"),
                standardized_id=f"{champion_id}_{spell_key}",
                spell_name=ability.get('name', ''),
                damage_type=','.join(tooltip_data['damage_types']),
                damage_values=json.dumps(effect_values),
                base_damage=json.dumps(tooltip_data['scalings'].get('base_damage')),
                max_rank=ability.get('maxrank', 1),
                cooldown=json.dumps(ability.get('cooldown', [])),
                cost=json.dumps(ability.get('cost', [])),
                range=json.dumps(ability.get('range', [])),
                ap_ratio=json.dumps(tooltip_data['scalings'].get('ap_ratio')),
                ad_ratio=json.dumps(tooltip_data['scalings'].get('ad_ratio')),
                bonus_ad_ratio=json.dumps(tooltip_data['scalings'].get('bonus_ad_ratio')),
                hp_ratio=json.dumps(tooltip_data['scalings'].get('hp_ratio')),
                target_hp_ratio=json.dumps(tooltip_data['scalings'].get('target_hp_ratio')),
                armor_ratio=json.dumps(tooltip_data['scalings'].get('armor_ratio')),
                mr_ratio=json.dumps(tooltip_data['scalings'].get('mr_ratio')),
                cc_effects=json.dumps(tooltip_data['cc_effects']),
                description=ability.get('description', ''),
                resource_type=ability.get('costType', ''),
                is_passive=is_passive
            )
            self.session.merge(spell_stats)
            self.session.commit()

        except Exception as e:
            logger.error(f"Error storing ability {spell_key} for {champion_id}: {e}")
            raise

class PatchDataManager:
    def __init__(self, db_path: str = 'sqlite:///../datasets/league_data.db'):
        self.dragon_api = DragonAPI(db_path)
        self.data_parser = DataParser()

    def process_patch_range(self, start_version: str, end_version: str):
        """Process all patches between start and end versions"""
        versions = self.dragon_api.get_available_versions()
        
        start = version.parse(start_version)
        end = version.parse(end_version)
        
        valid_versions = [
            v for v in versions 
            if re.match(r'^\d+\.\d+\.\d+$', v) 
            and start <= version.parse(v) <= end
        ]

        for ver in sorted(valid_versions, key=version.parse, reverse=True):
            self.process_single_patch(ver)

    def process_single_patch(self, version_str: str):
        """Process a single patch version"""
        logger.info(f"Processing patch {version_str}")
        
        # Download data
        tar_data = self.dragon_api.download_dragon_data(version_str)
        if not tar_data:
            return

        try:
            with tarfile.open(fileobj=tar_data) as tar_file:
                session = self.dragon_api.Session()
                try:
                    # Process champions
                    champion_data = json.loads(
                        tar_file.extractfile(f"{version_str}/data/en_US/championFull.json")
                        .read().decode('utf-8')
                    )['data']
                    
                    processor = ChampionProcessor(session)
                    for champion_id, champ_data in champion_data.items():
                        processor.process_champion(champ_data, version_str)
                    
                    # Process items (similar structure)
                    self._process_items(tar_file, version_str, session)
                    
                    logger.info(f"Successfully processed patch {version_str}")
                
                except Exception as e:
                    logger.error(f"Error processing patch {version_str}: {e}")
                    session.rollback()
                finally:
                    session.close()
                    
        except Exception as e:
            logger.error(f"Error opening tar file for {version_str}: {e}")

    def _process_items(self, tar_file, version_str: str, session):
        """Process and store item data"""
        try:
            item_data = json.loads(
                tar_file.extractfile(f"{version_str}/data/en_US/item.json")
                .read().decode('utf-8')
            )['data']

            for item_id, data in item_data.items():
                item_stats = ItemStats(
                    version=version_str,
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
            
        except Exception as e:
            logger.error(f"Error processing items for {version_str}: {e}")
            raise

def main():
    manager = PatchDataManager()
    
    # Command line argument handling could be added here
    start_version = "14.19.1"
    end_version = "14.19.1"
    
    try:
        manager.process_patch_range(start_version, end_version)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()