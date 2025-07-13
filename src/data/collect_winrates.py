import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ChampionStats, ChampionWinrates
import logging
import time


logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   handlers=[
       logging.FileHandler('scraping.log'),
       logging.StreamHandler()
   ]
)
logger = logging.getLogger(__name__)

def generate_patch_list():
    """Generate list of patches from 13.1 to 14.24"""
    patches = []
    # Season 13
    for i in range(1, 25):
        patches.append(f"13.{i}")
    # Season 14
    for i in range(1, 25):
        patches.append(f"14.{i}")
    
    # Skip problematic patches
    patches_to_skip = ['13.2']  # Ajoutez d'autres patches ici si nécessaire
    patches = [p for p in patches if p not in patches_to_skip]
    
    return patches

class MetaScraper:
    def __init__(self):
       self.engine = create_engine('sqlite:///../../datasets/league_data.db')
       self.Session = sessionmaker(bind=self.engine)
       self.base_url = "https://www.metasrc.com/lol/{}/build/{}"

    def get_champion_list(self):
        """Get list of champions from database"""
        session = self.Session()
        try:
            # Get all versions and sort them properly
            all_versions = session.query(ChampionStats.version).distinct().all()
            versions = [v[0] for v in all_versions]
            
            # Sort versions properly (14.24.1 > 14.9.1)
            def version_key(version_str):
                parts = version_str.split('.')
                return tuple(int(part) for part in parts)
            
            latest_patch = max(versions, key=version_key)
            
            champions = session.query(ChampionStats.champion)\
                .filter(ChampionStats.version == latest_patch)\
                .all()
            
            logger.info(f"Found {len(champions)} champions from patch {latest_patch}")
            return [champ[0] for champ in champions]
        finally:
            session.close()
           
    def should_scrape(self, champion, patch):
        """Check if data already exists"""
        session = self.Session()
        try:
            exists = session.query(ChampionWinrates)\
                .filter_by(champion_name=champion, patch=patch)\
                .first()
            return exists is None
        finally:
            session.close()
           
    def scrape_champion_stats(self, champion, patch, max_retries=3):
        """Scrape stats for a specific champion and patch"""
        url = self.base_url.format(patch, champion.lower())
        
        for attempt in range(max_retries):
            try:
                # Add delay to avoid rate limiting
                time.sleep(1)
                
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                stats_div = soup.find('div', {'class': '_fcip6v'})
                stats = {}
                
                if stats_div:
                    stat_spans = stats_div.find_all('span', {'class': '_dxv0e1'})
                    for span in stat_spans:
                        label = span.find('strong').text.strip().lower().replace(':', '')
                        value = span.contents[-1].text.strip()
                        
                        if label in ['win', 'pick', 'ban']:
                            try:
                                stats[label] = float(value.replace('%', ''))
                            except ValueError:
                                logger.error(f"Could not parse {label} value: {value} for {champion} patch {patch}")
                                continue
                        elif label == 'games':
                            try:
                                stats['games'] = int(value.replace(',', ''))
                            except ValueError:
                                logger.error(f"Could not parse games value: {value} for {champion} patch {patch}")
                                continue

                    if all(k in stats for k in ['win', 'pick', 'ban', 'games']):
                        return {
                            'winrate': stats['win'],
                            'pickrate': stats['pick'],
                            'banrate': stats['ban'],
                            'games': stats['games']
                        }
                    else:
                        logger.error(f"Missing required stats for {champion} patch {patch}. Found: {stats}")
                        return None
                else:
                    logger.error(f"Could not find stats div for {champion} patch {patch}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed for {champion} patch {patch}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)  # Wait longer between retries
                continue
            except Exception as e:
                logger.error(f"Error scraping {champion} for patch {patch}: {str(e)}")
                return None
            
    def save_champion_stats(self, champion, patch, stats):
        """Save champion stats to database"""
        if not stats or None in stats.values():
            logger.error(f"Invalid stats for {champion} patch {patch}: {stats}")
            return
            
        session = self.Session()
        try:
            metric = ChampionWinrates(
                patch=patch,
                champion_name=champion,
                winrate=stats['winrate'],
                pickrate=stats['pickrate'],
                banrate=stats['banrate'],
                total_games=stats['games']
            )
            session.add(metric)
            session.commit()
            logger.info(f"Saved stats for {champion} patch {patch}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving stats for {champion} patch {patch}: {str(e)}")
        finally:
            session.close()
            
    def scrape_patch(self, patch):
        """Scrape all champions for a specific patch"""
        champions = self.get_champion_list()
        total_champions = len(champions)
        logger.info(f"Starting scrape for patch {patch} with {total_champions} champions")
        
        processed = 0
        for champion in champions:
            try:
                if not self.should_scrape(champion, patch):
                    logger.info(f"Skipping {champion} patch {patch} - already exists")
                    continue
                    
                stats = self.scrape_champion_stats(champion, patch)
                if stats:
                    self.save_champion_stats(champion, patch, stats)
                
                processed += 1
                if processed % 10 == 0:  # Log progress every 10 champions
                    logger.info(f"Processed {processed}/{total_champions} champions for patch {patch}")
                    
            except Exception as e:
                logger.error(f"Failed to process {champion}: {str(e)}")
                continue

if __name__ == "__main__":
   scraper = MetaScraper()
   patches = generate_patch_list()
   
   logger.info(f"Starting scraping for {len(patches)} patches")
   
   for patch in patches:
       logger.info(f"Starting scrape for patch {patch}")
       scraper.scrape_patch(patch)
       logger.info(f"Completed patch {patch}")
       time.sleep(5)  # Wait between patches
   
   logger.info("Scraping completed")