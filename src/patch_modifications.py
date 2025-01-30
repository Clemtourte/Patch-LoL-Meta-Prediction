from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import ChampionStats, SpellStats, Base, PatchChanges
import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, Any, List

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

    def get_champion_stats(self, patch_version: str) -> Dict[str, ChampionStats]:
        """Get all champion stats for a specific patch"""
        session = self.Session()
        try:
            stats = session.query(ChampionStats).filter_by(version=patch_version).all()
            return {stat.champion: stat for stat in stats}
        finally:
            session.close()

    def get_spell_stats(self, patch_version: str) -> Dict[str, List[SpellStats]]:
        """Get all spell stats for a specific patch"""
        session = self.Session()
        try:
            spells = session.query(SpellStats).filter_by(version=patch_version).all()
            spell_dict = {}
            for spell in spells:
                if spell.champion not in spell_dict:
                    spell_dict[spell.champion] = []
                spell_dict[spell.champion].append(spell)
            return spell_dict
        finally:
            session.close()

    def compute_base_stat_changes(self, old_patch: str, new_patch: str) -> Dict[str, Dict[str, float]]:
        """Compute changes in base stats between patches"""
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
                    if old_value != new_value:  # Only record actual changes
                        changes[champion][stat] = new_value - old_value

        return changes

    def compute_per_level_changes(self, old_patch: str, new_patch: str) -> Dict[str, Dict[str, float]]:
        """Compute changes in per-level stats between patches"""
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
                    if old_value != new_value:  # Only record actual changes
                        changes[champion][stat] = new_value - old_value

        return changes

    def compute_spell_changes(self, old_patch: str, new_patch: str) -> Dict[str, Dict[str, Dict]]:
        """Compute changes in spell stats between patches"""
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
                        
                        # Get standardized spell ID (Q, W, E, R)
                        std_id = old_spell.standardized_id.split('_')[1]  # Gets 'Q','W','E','R'
                        
                        # Compare base damage
                        try:
                            old_damage = json.loads(old_spell.base_damage)
                            new_damage = json.loads(new_spell.base_damage)
                            if old_damage != new_damage and old_damage and new_damage:
                                changes[champion][f"{std_id}_base_damage"] = np.mean([n - o for n, o in zip(new_damage, old_damage)])
                        except (json.JSONDecodeError, TypeError):
                            pass

                        # Compare cooldowns
                        try:
                            old_cd = json.loads(old_spell.cooldown)
                            new_cd = json.loads(new_spell.cooldown)
                            if old_cd != new_cd:
                                changes[champion][f"{std_id}_cooldown"] = np.mean([n - o for n, o in zip(new_cd, old_cd)])
                        except (json.JSONDecodeError, TypeError):
                            pass

                        # Compare costs
                        try:
                            old_cost = json.loads(old_spell.cost)
                            new_cost = json.loads(new_spell.cost)
                            if old_cost != new_cost:
                                changes[champion][f"{std_id}_cost"] = np.mean([n - o for n, o in zip(new_cost, old_cost)])
                        except (json.JSONDecodeError, TypeError):
                            pass

        return changes

    def analyze_patch_changes(self, old_patch: str, new_patch: str) -> Dict[str, Any]:
        """Analyze all changes between two patches"""
        return {
            'base_stats': self.compute_base_stat_changes(old_patch, new_patch),
            'per_level': self.compute_per_level_changes(old_patch, new_patch),
            'abilities': self.compute_spell_changes(old_patch, new_patch)
        }
    
    def save_patch_changes(self, from_patch: str, to_patch: str, changes: Dict[str, Any]) -> None:
        """Save detected changes to database"""
        session = self.Session()
        try:
            # Save base stat changes
            for champion, stats in changes['base_stats'].items():
                for stat_name, value in stats.items():
                    if value != 0:  # Only save non-zero changes
                        session.merge(PatchChanges(
                            from_patch=from_patch,
                            to_patch=to_patch,
                            champion_name=champion,
                            stat_type='base_stat',
                            stat_name=stat_name,
                            change_value=value
                        ))
            
            # Save per-level changes
            for champion, stats in changes['per_level'].items():
                for stat_name, value in stats.items():
                    if value != 0:  # Only save non-zero changes
                        session.merge(PatchChanges(
                            from_patch=from_patch,
                            to_patch=to_patch,
                            champion_name=champion,
                            stat_type='per_level',
                            stat_name=stat_name,
                            change_value=value
                        ))

            # Save ability changes
            for champion, abilities in changes['abilities'].items():
                for ability_name, value in abilities.items():
                    if value != 0:  # Only save non-zero changes
                        session.merge(PatchChanges(
                            from_patch=from_patch,
                            to_patch=to_patch,
                            champion_name=champion,
                            stat_type='ability',
                            stat_name=ability_name,
                            change_value=value
                        ))

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving changes: {str(e)}")
            raise
        finally:
            session.close()

    def analyze_all_sequential_patches(self) -> None:
        """Analyze changes between sequential patches"""
        session = self.Session()
        try:
            patches = [p[0] for p in session.query(ChampionStats.version).distinct().all()]
            patches.sort(key=lambda x: [int(p) for p in x.split('.')])
            logger.info(f"Found {len(patches)} patches to analyze")
            
            for i in range(len(patches) - 1):
                old_patch = patches[i]
                new_patch = patches[i + 1]
                
                if not session.query(PatchChanges).filter_by(
                    from_patch=old_patch, to_patch=new_patch).first():
                    logger.info(f"Analyzing {old_patch} -> {new_patch}")
                    try:
                        changes = self.analyze_patch_changes(old_patch, new_patch)
                        self.save_patch_changes(old_patch, new_patch, changes)
                    except Exception as e:
                        logger.error(f"Error analyzing {old_patch} -> {new_patch}: {str(e)}")
                else:
                    logger.info(f"Skipping {old_patch} -> {new_patch}: already analyzed")
                    
        finally:
            session.close()

    def clean_patch_changes(self) -> None:
        """Clean patch_changes table"""
        session = self.Session()
        try:
            session.query(PatchChanges).delete()
            session.commit()
            logger.info("Cleaned patch_changes table")
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning table: {str(e)}")
        finally:
            session.close()

if __name__ == "__main__":
    try:
        detector = PatchChangeDetector()
        detector.clean_patch_changes()
        detector.analyze_all_sequential_patches()
        
        # Print summary
        with detector.Session() as session:
            total_changes = session.query(PatchChanges).count()
            patch_pairs = session.query(PatchChanges.from_patch, PatchChanges.to_patch).distinct().all()
            
            print(f"\nProcessed {len(patch_pairs)} patch pairs")
            print(f"Found {total_changes} total changes")
            for from_patch, to_patch in patch_pairs:
                changes = session.query(PatchChanges).filter_by(
                    from_patch=from_patch, to_patch=to_patch).count()
                print(f"  {from_patch} -> {to_patch}: {changes} changes")
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")