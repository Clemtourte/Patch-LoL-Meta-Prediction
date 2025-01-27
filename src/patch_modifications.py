from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import ChampionStats, SpellStats, Base
import pandas as pd
import numpy as np
from models import PatchChanges
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('patch_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PatchChangeDetector:
    def __init__(self):
        self.engine = create_engine('sqlite:///../datasets/league_data.db')
        self.Session = sessionmaker(bind=self.engine)

    def get_champion_stats(self, patch_version):
        """Get all champion stats for a specific patch"""
        session = self.Session()
        try:
            stats = session.query(ChampionStats).filter_by(version=patch_version).all()
            return {stat.champion: stat for stat in stats}
        finally:
            session.close()

    def get_spell_stats(self, patch_version):
        """Get all spell stats for a specific patch"""
        session = self.Session()
        try:
            spells = session.query(SpellStats).filter_by(version=patch_version).all()
            # Group spells by champion
            spell_dict = {}
            for spell in spells:
                if spell.champion not in spell_dict:
                    spell_dict[spell.champion] = []
                spell_dict[spell.champion].append(spell)
            return spell_dict
        finally:
            session.close()

    def compute_base_stat_changes(self, old_patch, new_patch):
        """
        Compute changes in base stats between patches
        Returns a matrix of changes
        """
        old_stats = self.get_champion_stats(old_patch)
        new_stats = self.get_champion_stats(new_patch)
        
        changes = {}
        base_stats = ['hp', 'mp', 'armor', 'spellblock', 'attackdamage', 
                     'attackspeed', 'attackrange', 'movespeed']

        for champion in old_stats:
            if champion in new_stats:
                changes[champion] = {}
                for stat in base_stats:
                    old_value = getattr(old_stats[champion], stat)
                    new_value = getattr(new_stats[champion], stat)
                    changes[champion][stat] = new_value - old_value

        return changes

    def compute_per_level_changes(self, old_patch, new_patch):
        """
        Compute changes in per-level stats between patches
        Returns a matrix of changes
        """
        old_stats = self.get_champion_stats(old_patch)
        new_stats = self.get_champion_stats(new_patch)
        
        changes = {}
        per_level_stats = ['hpperlevel', 'mpperlevel', 'armorperlevel', 
                          'spellblockperlevel', 'attackdamageperlevel', 
                          'attackspeedperlevel']

        for champion in old_stats:
            if champion in new_stats:
                changes[champion] = {}
                for stat in per_level_stats:
                    old_value = getattr(old_stats[champion], stat)
                    new_value = getattr(new_stats[champion], stat)
                    changes[champion][stat] = new_value - old_value

        return changes

    def compute_spell_changes(self, old_patch, new_patch):
        """
        Compute changes in abilities between patches
        Returns a matrix of changes
        """
        old_spells = self.get_spell_stats(old_patch)
        new_spells = self.get_spell_stats(new_patch)
        
        changes = {}
        
        for champion in old_spells:
            if champion in new_spells:
                changes[champion] = {}
                old_champ_spells = {s.spell_id: s for s in old_spells[champion]}
                new_champ_spells = {s.spell_id: s for s in new_spells[champion]}
                
                for spell_id, old_spell in old_champ_spells.items():
                    if spell_id in new_champ_spells:
                        new_spell = new_champ_spells[spell_id]
                        spell_key = f"{spell_id}"
                        
                        # Compare damage values
                        old_damage = self.parse_damage_values(old_spell.damage_values)
                        new_damage = self.parse_damage_values(new_spell.damage_values)
                        if old_damage != new_damage:
                            changes[champion][f"{spell_key}_damage"] = new_damage - old_damage
                        
                        # Compare cooldowns
                        old_cd = self.parse_json_list(old_spell.cooldown)
                        new_cd = self.parse_json_list(new_spell.cooldown)
                        if old_cd != new_cd:
                            changes[champion][f"{spell_key}_cooldown"] = new_cd - old_cd

        return changes

    def parse_damage_values(self, damage_values_json):
        """Parse damage values from JSON string"""
        try:
            values = eval(damage_values_json)  # Be careful with eval, ensure input is safe
            return np.mean(values) if values else 0
        except:
            return 0

    def parse_json_list(self, json_str):
        """Parse JSON string into list and return average"""
        try:
            values = eval(json_str)  # Be careful with eval, ensure input is safe
            return np.mean(values) if values else 0
        except:
            return 0

    def analyze_patch_changes(self, old_patch, new_patch):
        """
        Main function to analyze all changes between two patches
        """
        return {
            'base_stats': self.compute_base_stat_changes(old_patch, new_patch),
            'per_level': self.compute_per_level_changes(old_patch, new_patch),
            'abilities': self.compute_spell_changes(old_patch, new_patch)
        }
    
    def save_patch_changes(self, from_patch, to_patch, changes):
        session = self.Session()
        try:
            # Save base stat changes
            for champion, stats in changes['base_stats'].items():
                for stat_name, value in stats.items():
                    change = PatchChanges(
                        from_patch=from_patch,
                        to_patch=to_patch,
                        champion_name=champion,
                        stat_type='base_stat',
                        stat_name=stat_name,
                        change_value=value
                    )
                    session.add(change)
            
            # Save per-level changes
            for champion, stats in changes['per_level'].items():
                for stat_name, value in stats.items():
                    change = PatchChanges(
                        from_patch=from_patch,
                        to_patch=to_patch,
                        champion_name=champion,
                        stat_type='per_level',
                        stat_name=stat_name,
                        change_value=value
                    )
                    session.add(change)

            # Save ability changes
            for champion, abilities in changes['abilities'].items():
                for ability_name, value in abilities.items():
                    change = PatchChanges(
                        from_patch=from_patch,
                        to_patch=to_patch,
                        champion_name=champion,
                        stat_type='ability',
                        stat_name=ability_name,
                        change_value=value
                    )
                    session.add(change)

            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def analyze_all_sequential_patches(self):
        """
        Analyzes changes between sequential patches (14.1.1 -> 14.2.1 -> 14.3.1 etc)
        """
        session = self.Session()
        try:
            # Get all unique patches
            patches = session.query(ChampionStats.version)\
                .distinct()\
                .all()
            patches = [p[0] for p in patches]
            
            # Custom sort for patch versions
            def sort_patches(patch):
                major, minor, micro = map(int, patch.split('.'))
                return (major, minor, micro)
            
            patches.sort(key=sort_patches)
            logger.info(f"Found {len(patches)} patches to analyze")
            
            # Analyze sequential pairs
            for i in range(len(patches) - 1):
                old_patch = patches[i]
                new_patch = patches[i + 1]
                
                # Check if this comparison already exists
                existing = session.query(PatchChanges)\
                    .filter_by(from_patch=old_patch, to_patch=new_patch)\
                    .first()
                
                if existing:
                    logger.info(f"Skipping {old_patch} -> {new_patch}: already analyzed")
                    continue
                
                logger.info(f"Analyzing changes between {old_patch} and {new_patch}")
                
                try:
                    changes = self.analyze_patch_changes(old_patch, new_patch)
                    self.save_patch_changes(old_patch, new_patch, changes)
                    logger.info(f"Successfully saved changes for {old_patch} -> {new_patch}")
                except Exception as e:
                    logger.error(f"Error analyzing patches {old_patch} -> {new_patch}: {str(e)}")
                    continue
                    
        finally:
            session.close()

    def clean_patch_changes(self):
        """Clean all entries from patch_changes table"""
        session = self.Session()
        try:
            session.query(PatchChanges).delete()
            session.commit()
            logger.info("Successfully cleaned patch_changes table")
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning patch_changes table: {str(e)}")
        finally:
            session.close()

if __name__ == "__main__":
    try:
        detector = PatchChangeDetector()
        
        # Clean existing data
        detector.clean_patch_changes()
        
        # Run analysis with correct sorting
        detector.analyze_all_sequential_patches()
        
        # Print summary of results
        session = detector.Session()
        try:
            total_changes = session.query(PatchChanges).count()
            patch_pairs = session.query(PatchChanges.from_patch, PatchChanges.to_patch)\
                .distinct().all()
            
            print("\nAnalysis Complete!")
            print(f"Total changes detected: {total_changes}")
            print(f"Patches analyzed: {len(patch_pairs)}")
            for from_patch, to_patch in patch_pairs:
                print(f"  {from_patch} -> {to_patch}")
                
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")