import requests
from bs4 import BeautifulSoup
import logging
import re
import json
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from models import SpellStats, Base

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PatchNotesScraper:
    def __init__(self, db_path="sqlite:///../datasets/league_data.db"):
        """Initialize the scraper with database connection."""
        self.engine = create_engine(db_path)
        self.Session = sessionmaker(bind=self.engine)
        
        # Print schema info for debugging
        self.log_schema_info()
        
    def log_schema_info(self):
        """Log schema information for debugging."""
        try:
            inspector = inspect(self.engine)
            logger.info("SpellStats table columns:")
            for column in inspector.get_columns('spell_stats'):
                logger.info(f"  {column['name']}: {column['type']}")
        except Exception as e:
            logger.error(f"Error inspecting schema: {str(e)}")

    def extract_patch_version(self, url_or_text):
        """Extract patch version from URL or text, e.g. patch-13-3 or patch 13.3."""
        match = re.search(r'patch[- ](\d+)[.-](\d+)', url_or_text.lower())
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return None

    def extract_numeric_values(self, text):
        """
        Extract numeric values from text like '80/100/120' or '14/13/12/11/10 seconds'.
        Returns a list of floats. If no numeric is found, returns [].
        """
        if not text:
            return []
        
        # Remove parenthetical expressions like (+40% AP)
        clean_text = re.sub(r'\([^)]*\)', '', text)

        # If we see multiple slash-separated values, parse them individually
        # e.g. "14/13/12/11/10" -> [14.0, 13.0, 12.0, 11.0, 10.0]
        values = []
        if '/' in clean_text:
            parts = clean_text.split('/')
            for part in parts:
                num_match = re.search(r'(\d+(\.\d+)?)', part.strip())
                if num_match:
                    values.append(float(num_match.group(1)))
            return values
        
        # Otherwise, just look for a single number (like "80" or "12.5")
        num_match = re.search(r'(\d+(\.\d+)?)', clean_text)
        if num_match:
            return [float(num_match.group(1))]
        
        return []

    def extract_scaling_info(self, text):
        """
        Extract AP/AD scaling from text like (+55% AP). You can expand this
        if you need to handle bonus AD, max health, etc. 
        Returns a dict, e.g. {'ap': 0.55}.
        """
        scaling_info = {}
        text_lower = text.lower()
        
        # AP scaling
        ap_match = re.search(r'\(\+([0-9.]+)%\s*ap\)', text_lower)
        if ap_match:
            scaling_info['ap'] = float(ap_match.group(1)) / 100
        
        # AD scaling
        ad_match = re.search(r'\(\+([0-9.]+)%\s*ad\)', text_lower)
        if ad_match and not re.search(r'bonus\s*ad', text_lower):
            scaling_info['ad'] = float(ad_match.group(1)) / 100
        
        # Bonus AD scaling
        bonus_ad_match = re.search(r'\(\+([0-9.]+)%\s*bonus\s*ad\)', text_lower)
        if bonus_ad_match:
            scaling_info['bonus_ad'] = float(bonus_ad_match.group(1)) / 100
        
        return scaling_info

    def determine_change_type(self, label_text, raw_text):
        """
        Devine le type de changement : 'damage', 'cooldown', 'mana_cost', etc.
        """
        label_lower = label_text.lower() if label_text else ''
        raw_lower = raw_text.lower()

        # Pour différencier les bullets de Kassadin R
        if 'bonus damage' in label_lower:
            return 'bonus_damage'
        if 'magic damage' in label_lower:
            return 'magic_damage'
        
        # Pour K’Sante : distinguer knock up et stun
        if 'knock up' in label_lower:
            return 'knockup_duration'
        if 'stun' in label_lower:
            return 'stun_duration'
        
        # Pour Jarvan IV : distinguer shield duration de shield value
        if 'shield' in label_lower:
            if 'duration' in label_lower or 'duration' in raw_lower:
                return 'shield_duration'
            return 'shield'
        
        if 'cooldown' in label_lower:
            return 'cooldown'
        if 'mana cost' in label_lower or 'energy cost' in label_lower:
            return 'mana_cost'
        if 'heal' in label_lower:
            return 'heal'
        if 'slow' in label_lower and '%' in raw_lower:
            return 'slow_percent'
        if 'range' in label_lower:
            return 'range'
        
        # Détection générique
        if 'damage' in raw_lower:
            return 'damage'
        
        return 'unknown'

    def parse_change(self, text):
        """
        Parse a single bullet of text that might look like:
          "Damage: 80/120/160 ⇒ 70/110/150"
        or
          "14/13/12/11/10 seconds ⇒ 12/11/10/9/8 seconds"

        Returns a dict with:
          {
            'label': <str or None>,
            'type': <guessed change type>,
            'before_values': [...],
            'after_values': [...],
            'before_ratios': {...},
            'after_ratios': {...},
            'before_raw': <string>,
            'after_raw': <string>
          }
        or None if we fail to parse numeric data.
        """
        change = {'raw_text': text}
        
        # We require an arrow "⇒" to detect old vs new
        arrow_match = re.search(r'(.*?)⇒(.*)', text)
        if not arrow_match:
            return None  # No arrow => not a numeric change

        before_part = arrow_match.group(1).strip()
        after_part  = arrow_match.group(2).strip()

        # Attempt to see if there's a label like "Damage:" or "Cooldown:"
        label_match = re.search(r'^([^:]+):\s*(.*)$', before_part)
        if label_match:
            label = label_match.group(1).strip()
            before_val_str = label_match.group(2).strip()
        else:
            # No label found
            label = ''
            before_val_str = before_part

        change_type = self.determine_change_type(label, text)
        change['type'] = change_type
        change['label'] = label

        # Extract numeric arrays from each side
        change['before_values'] = self.extract_numeric_values(before_val_str)
        change['after_values']  = self.extract_numeric_values(after_part)

        # Extract possible scaling info
        change['before_ratios'] = self.extract_scaling_info(before_val_str)
        change['after_ratios']  = self.extract_scaling_info(after_part)

        change['before_raw'] = before_val_str
        change['after_raw']  = after_part

        # If neither side has numeric values, skip it
        if not change['before_values'] and not change['after_values']:
            return None

        return change

    def parse_ability_type(self, text):
        """
        Determine if an ability is Passive, Q, W, E, or R based on the heading text.
        e.g. "Q - Bandage Toss" => Q
        """
        t = text.lower()
        if 'passive' in t:
            return 'Passive'
        if 'q -' in t or 'q:' in t or ' q ' in t:
            return 'Q'
        if 'w -' in t or 'w:' in t or ' w ' in t:
            return 'W'
        if 'e -' in t or 'e:' in t or ' e ' in t:
            return 'E'
        if 'r -' in t or 'r:' in t or ' r ' in t:
            return 'R'
        return None

    def scrape_patch_notes(self, html_content):
        """
        Extract champion changes from patch notes HTML. 
        Returns a list of {champion, spell_type, spell_name, changes[]} and a detected patch version.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        patch_version = None
        
        # Attempt to detect patch version from <title> or from headings
        title_tag = soup.find('title')
        if title_tag:
            patch_version = self.extract_patch_version(title_tag.text)
        if not patch_version:
            for header in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                potential = self.extract_patch_version(header.text)
                if potential:
                    patch_version = potential
                    break
        
        if patch_version:
            logger.info(f"Detected patch version: {patch_version}")
        else:
            logger.warning("Could not detect patch version in HTML.")

        # Find champion sections
        champion_changes = []
        champion_blocks = soup.find_all('div', class_='patch-change-block')
        if not champion_blocks:
            logger.warning("No standard champion blocks found in HTML. Attempting fallback.")
            # fallback approach if the patch notes use a different structure
            content_divs = soup.find_all('div', class_='content-border')
            for div in content_divs:
                champion_name_tag = div.find(['h3', 'h4'], text=re.compile(r'[A-Z][a-z]+'))
                if champion_name_tag:
                    champion_blocks.append(div)

        for block in champion_blocks:
            # Attempt to find champion name
            champion_header = block.find(['h3', 'h4'], class_='change-title')
            if not champion_header:
                champion_header = block.find(['h3', 'h4'], text=re.compile(r'[A-Z][a-z]+'))
            if not champion_header:
                continue

            champion_link = champion_header.find('a')
            if champion_link:
                champion_name = champion_link.text.strip()
            else:
                champion_name = re.sub(r'[^a-zA-Z\s]', '', champion_header.text.strip())

            logger.info(f"Processing champion: {champion_name}")

            # Find ability sections
            ability_sections = block.find_all(['h4', 'h5'], class_='change-detail-title ability-title')
            if not ability_sections:
                # fallback if no dedicated class
                ability_sections = block.find_all(['h4', 'h5'], text=re.compile(r'(Q|W|E|R) -|Passive'))
            
            for ability in ability_sections:
                ability_name = ability.text.strip()
                spell_type = self.parse_ability_type(ability_name)
                if not spell_type:
                    logger.warning(f"Could not determine spell type for: {ability_name}")
                    continue

                logger.info(f"  Processing ability: {ability_name} ({spell_type})")
                changes_list = ability.find_next('ul')
                if not changes_list:
                    logger.warning(f"No <ul> with changes found for {champion_name} {spell_type}")
                    continue
                
                ability_changes = []
                for item in changes_list.find_all('li'):
                    line_text = item.text.strip()
                    if '⇒' not in line_text:
                        # skip lines that do not have an arrow
                        continue

                    parsed = self.parse_change(line_text)
                    if parsed:
                        ability_changes.append(parsed)
                        logger.info(f"    Found change: {parsed['type']}")
                    else:
                        logger.debug(f"Skipping line due to no numeric parse: {line_text}")

                if ability_changes:
                    champion_changes.append({
                        'champion': champion_name,
                        'spell_type': spell_type,
                        'spell_name': ability_name,
                        'changes': ability_changes
                    })

        return champion_changes, patch_version

    def update_spell_stats(self, changes, patch_version):
        """
        Met à jour (ou crée) les enregistrements dans SpellStats avec les changements parsés.
        Les anciennes valeurs (gauche) sont stockées dans les colonnes "previous_..."
        et les nouvelles valeurs dans les colonnes correspondantes.
        """
        change_mapping = {
            'damage': 'base_damage',
            'magic_damage': 'base_damage',
            'bonus_damage': 'base_damage',
            'cooldown': 'cooldown',
            'mana_cost': 'mana_cost',
            'shield': 'shield_value',
            'shield_duration': 'shield_duration',
            'heal': 'heal_value',
            'slow_percent': 'slow_percent',
            'stun_duration': 'stun_duration',
            'knockup_duration': 'knockup_duration',
            'range': 'range'
        }
        
        session = self.Session()
        try:
            for champ_change in changes:
                champion = champ_change['champion']
                spell_type = champ_change['spell_type']
                spell_name = champ_change['spell_name']
                
                logger.info(f"Updating DB for {champion} {spell_type} ({spell_name})")
                
                record = session.query(SpellStats).filter_by(
                    version=patch_version,
                    champion=champion,
                    spell_type=spell_type,
                    spell_name=spell_name
                ).first()
                
                if not record:
                    record = SpellStats(
                        version=patch_version,
                        champion=champion,
                        spell_type=spell_type,
                        spell_name=spell_name
                    )
                    session.add(record)
                
                for change in champ_change['changes']:
                    ctype = change.get('type')
                    if ctype in change_mapping:
                        col = change_mapping[ctype]
                        
                        # Si aucune valeur précedente n'est trouvée, on considère 0
                        before_vals = change.get('before_values') or [0]
                        after_vals = change.get('after_values')
                        
                        # Si after_vals est un nombre unique et before_vals est une liste,
                        # on répète la valeur après pour correspondre au nombre d'éléments.
                        if isinstance(after_vals, (int, float)) and isinstance(before_vals, list) and len(before_vals) > 1:
                            after_vals = [after_vals] * len(before_vals)
                        
                        setattr(record, f"previous_{col}", before_vals)
                        setattr(record, col, after_vals)
                        
                        # Pour les ratios (AP, AD, etc.)
                        for ratio_key, ratio_col in [('ap', 'ap_ratio'),
                                                    ('ad', 'ad_ratio'),
                                                    ('bonus_ad', 'bonus_ad_ratio'),
                                                    ('max_health', 'max_health_ratio')]:
                            if ratio_key in change.get('before_ratios', {}):
                                setattr(record, f"previous_{ratio_col}", change['before_ratios'][ratio_key])
                            if ratio_key in change.get('after_ratios', {}):
                                setattr(record, ratio_col, change['after_ratios'][ratio_key])
                        
                        logger.info(f"  - {ctype} updated: previous {col}={before_vals} -> {col}={after_vals}")
                    else:
                        logger.warning(f"Unmapped change type '{ctype}' for {champion} {spell_type} ({spell_name})")
                
                session.commit()
                logger.info(f"Record updated for {champion} {spell_type} ({spell_name})")
            
            logger.info(f"Successfully updated database with changes for patch {patch_version}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating DB: {str(e)}", exc_info=True)
            return False
        finally:
            session.close()


    def scrape_patch_from_url(self, patch_url):
        """
        Fetch patch notes from a URL, parse them, and update DB.
        """
        logger.info(f"Fetching patch notes from {patch_url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        try:
            # Some patch pages may require verify=False if there's a certificate issue
            # but it's recommended to handle SSL properly
            response = requests.get(patch_url, headers=headers, verify=False)
            response.raise_for_status()
            
            # Attempt to parse patch version from URL
            patch_version = self.extract_patch_version(patch_url)
            if not patch_version:
                logger.warning("Couldn't extract patch version from URL; will try from content")

            # Parse HTML
            changes, detected_patch = self.scrape_patch_notes(response.text)
            if not patch_version:
                patch_version = detected_patch

            if not patch_version:
                logger.error("Failed to determine patch version. Aborting.")
                return False
            
            logger.info(f"Extracted {len(changes)} champion changes from patch {patch_version}")
            return self.update_spell_stats(changes, patch_version)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching patch notes: {str(e)}")
            return False

    def scrape_patch_from_file(self, html_filepath, patch_version=None):
        """
        Scrape patch notes from a local HTML file for offline testing.
        """
        logger.info(f"Reading patch notes from file: {html_filepath}")
        try:
            with open(html_filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            changes, detected_patch = self.scrape_patch_notes(html_content)
            if not patch_version:
                patch_version = detected_patch

            if not patch_version:
                logger.error("Failed to determine patch version from file.")
                return False
            
            logger.info(f"Extracted {len(changes)} champion changes from patch {patch_version}")
            return self.update_spell_stats(changes, patch_version)
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return False


def scrape_patch_range(start_patch, end_patch, language="en-gb"):
    """
    Example function to scrape a range of patch notes.
    e.g. scrape_patch_range("13.1", "13.3", language="en-gb")
    """
    scraper = PatchNotesScraper()
    base_url = f"https://www.leagueoflegends.com/{language}/news/game-updates/patch-{{}}-notes/"
    
    # Convert patch strings (e.g. 13.1) to numeric
    start_major, start_minor = map(int, start_patch.split('.'))
    end_major, end_minor     = map(int, end_patch.split('.'))

    patches = []
    for major in range(start_major, end_major + 1):
        if major == start_major:
            minor_start = start_minor
        else:
            minor_start = 1
        if major == end_major:
            minor_end = end_minor
        else:
            minor_end = 24
        
        for minor in range(minor_start, minor_end + 1):
            patches.append(f"{major}-{minor}")

    success_count = 0
    for patch in patches:
        url = base_url.format(patch)
        logger.info(f"Processing patch {patch}")
        success = scraper.scrape_patch_from_url(url)
        if success:
            success_count += 1
        else:
            logger.warning(f"Failed to process patch {patch}")

    logger.info(f"Successfully processed {success_count} out of {len(patches)} patches.")


if __name__ == "__main__":
    # Example usage
    scraper = PatchNotesScraper()
    
    # 1) Scrape a single patch from URL
    scraper.scrape_patch_from_url("https://www.leagueoflegends.com/en-gb/news/game-updates/patch-13-15-notes/")
    
    # 2) Scrape from a local file
    # scraper.scrape_patch_from_file("patch_notes.html", "13.3")
    
    # 3) Scrape a range of patches
    # scrape_patch_range("13.1", "14.24", language="en-gb")
    
    pass
