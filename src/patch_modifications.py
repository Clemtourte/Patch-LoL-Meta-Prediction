#!/usr/bin/env python
import logging
import json
import numpy as np
import re
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from models import ChampionStats, SpellStats, Base, PatchChanges
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

def normalize_patch_version(patch: str) -> str:
    """Normalize a version string to 'major.minor'. For example, '14.19.1' becomes '14.19'."""
    parts = patch.split('.')
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return patch

def load_json(field_val):
    """If field_val is a string, load it as JSON; otherwise return it as is."""
    if isinstance(field_val, str):
        try:
            return json.loads(field_val)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return None
    else:
        return field_val

def slice_field(field, ranks):
    """If field is a list and has more than 'ranks' elements, slice it; else return as is."""
    if isinstance(field, list) and len(field) > ranks:
        return field[:ranks]
    return field

class PatchChangeDetector:
    def __init__(self):
        self.engine = create_engine('sqlite:///../datasets/league_data.db')
        self.Session = sessionmaker(bind=self.engine)

    def get_champion_stats(self, patch_version: str) -> Dict[str, ChampionStats]:
        """Get all champion stats for a specific patch (using normalized version)."""
        session = self.Session()
        norm_patch = normalize_patch_version(patch_version)
        try:
            stats = session.query(ChampionStats).filter(
                ChampionStats.version.like(f"{norm_patch}%")
            ).all()
            return {stat.champion: stat for stat in stats}
        finally:
            session.close()

    def get_spell_stats(self, patch_version: str) -> Dict[str, List[SpellStats]]:
        """Get all spell stats for a specific patch (using normalized version)."""
        session = self.Session()
        norm_patch = normalize_patch_version(patch_version)
        try:
            spells = session.query(SpellStats).filter(
                SpellStats.version.like(f"{norm_patch}%")
            ).all()
            spell_dict = {}
            for spell in spells:
                if spell.champion not in spell_dict:
                    spell_dict[spell.champion] = []
                spell_dict[spell.champion].append(spell)
            return spell_dict
        finally:
            session.close()

    def compute_base_stat_changes(self, old_patch: str, new_patch: str) -> Dict[str, Dict[str, float]]:
        """Compute changes in base stats between patches."""
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
                    if old_value != new_value:
                        changes[champion][stat] = new_value - old_value

        return changes

    def compute_per_level_changes(self, old_patch: str, new_patch: str) -> Dict[str, Dict[str, float]]:
        """Compute changes in per-level stats between patches."""
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
                    if old_value != new_value:
                        changes[champion][stat] = new_value - old_value

        return changes

    def compute_spell_changes(self, old_patch: str, new_patch: str) -> Dict[str, Dict[str, Dict]]:
        """
        Compute per-rank differences in spell stats between patches.
        For each champion and each spell type (Q, W, E, R, Passive),
        compare the fields stored in cooldown, cost, range, and damage_values.
        Returns a dict like:
          { champion: { spell_type: { field: [diff_rank1, diff_rank2, ...], ... }, ... } }
        """
        old_spells = self.get_spell_stats(old_patch)
        new_spells = self.get_spell_stats(new_patch)
        
        changes = {}
        
        for champion in old_spells:
            if champion not in new_spells:
                continue
            changes[champion] = {}
            # Build dictionaries keyed by spell_type.
            old_dict = {s.spell_type: s for s in old_spells[champion]}
            new_dict = {s.spell_type: s for s in new_spells[champion]}
            
            for spell_type in old_dict:
                if spell_type not in new_dict:
                    continue
                old_spell = old_dict[spell_type]
                new_spell = new_dict[spell_type]
                diff = {}
                
                # Process fields 'cooldown', 'cost', 'range'
                for field in ['cooldown', 'cost', 'range']:
                    try:
                        old_val = load_json(getattr(old_spell, field) or "[]")
                        new_val = load_json(getattr(new_spell, field) or "[]")
                    except Exception as e:
                        logger.error(f"Error loading {field} for {champion} {spell_type}: {e}")
                        continue
                    if old_val and new_val:
                        n = min(len(old_val), len(new_val))
                        diff_list = []
                        for i in range(n):
                            try:
                                # Only process if both values are numeric:
                                ov = float(old_val[i])
                                nv = float(new_val[i])
                                diff_list.append(nv - ov)
                            except Exception as e:
                                logger.warning(f"Skipping non-numeric {field} value for {champion} {spell_type} at index {i}: {e}")
                        if diff_list and any(x != 0 for x in diff_list):
                            diff[field] = diff_list
                
                # Process damage_values (expected to be a JSON dict)
                try:
                    old_damage = load_json(getattr(old_spell, "damage_values") or "{}")
                    new_damage = load_json(getattr(new_spell, "damage_values") or "{}")
                except Exception as e:
                    logger.error(f"Error loading damage_values for {champion} {spell_type}: {e}")
                    old_damage = {}
                    new_damage = {}
                diff_damage = {}
                for key in old_damage:
                    if key in new_damage:
                        old_list = old_damage[key]
                        new_list = new_damage[key]
                        n = min(len(old_list), len(new_list))
                        diff_list = []
                        for i in range(n):
                            try:
                                # Process only numeric elements.
                                ov = float(old_list[i])
                                nv = float(new_list[i])
                                diff_list.append(nv - ov)
                            except Exception as e:
                                logger.warning(f"Skipping non-numeric damage value for {champion} {spell_type} key {key} at rank {i}: {e}")
                        if diff_list and any(x != 0 for x in diff_list):
                            diff_damage[key] = diff_list
                if diff_damage:
                    diff["damage_values"] = diff_damage

                if diff:
                    changes[champion][spell_type] = diff

        return changes

    def analyze_patch_changes(self, old_patch: str, new_patch: str) -> Dict[str, Any]:
        """Analyze all changes between two patches."""
        return {
            'base_stats': self.compute_base_stat_changes(old_patch, new_patch),
            'per_level': self.compute_per_level_changes(old_patch, new_patch),
            'abilities': self.compute_spell_changes(old_patch, new_patch)
        }
    
    def save_patch_changes(self, from_patch: str, to_patch: str, changes: Dict[str, Any]) -> None:
        """Save detected changes to the PatchChanges table."""
        session = self.Session()
        try:
            # Save base stat changes.
            for champion, stats in changes['base_stats'].items():
                for stat_name, value in stats.items():
                    if value != 0:
                        session.merge(PatchChanges(
                            from_patch=from_patch,
                            to_patch=to_patch,
                            champion_name=champion,
                            stat_type='base_stat',
                            stat_name=stat_name,
                            change_value=value
                        ))
            
            # Save per-level changes.
            for champion, stats in changes['per_level'].items():
                for stat_name, value in stats.items():
                    if value != 0:
                        session.merge(PatchChanges(
                            from_patch=from_patch,
                            to_patch=to_patch,
                            champion_name=champion,
                            stat_type='per_level',
                            stat_name=stat_name,
                            change_value=value
                        ))

            # Save ability changes.
            for champion, abilities in changes['abilities'].items():
                for ability_name, field_diffs in abilities.items():
                    for field, diff_list in field_diffs.items():
                        for rank, diff_val in enumerate(diff_list, start=1):
                            # Only save if diff_val is numeric.
                            try:
                                diff_numeric = float(diff_val)
                            except Exception:
                                logger.warning(f"Skipping non-numeric diff for {champion} {ability_name} {field} rank {rank}")
                                continue
                            if diff_numeric != 0:
                                session.merge(PatchChanges(
                                    from_patch=from_patch,
                                    to_patch=to_patch,
                                    champion_name=champion,
                                    stat_type='ability',
                                    stat_name=f"{ability_name}_{field}_rank{rank}",
                                    change_value=diff_numeric
                                ))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving changes: {e}")
            raise
        finally:
            session.close()

    def analyze_all_sequential_patches(self) -> None:
        """Analyze changes between sequential patches."""
        session = self.Session()
        try:
            patches = [p[0] for p in session.query(ChampionStats.version).distinct().all()]
            # Normalize versions for sorting.
            patches.sort(key=lambda x: [int(p) for p in normalize_patch_version(x).split('.')])
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
                        logger.error(f"Error analyzing {old_patch} -> {new_patch}: {e}")
                else:
                    logger.info(f"Skipping {old_patch} -> {new_patch}: already analyzed")
        finally:
            session.close()

    def clean_patch_changes(self) -> None:
        """Clean the patch_changes table."""
        session = self.Session()
        try:
            deleted = session.query(PatchChanges).delete()
            session.commit()
            logger.info(f"Cleaned patch_changes table. Deleted {deleted} rows.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning table: {e}")
        finally:
            session.close()


if __name__ == "__main__":
    try:
        detector = PatchChangeDetector()
        detector.clean_patch_changes()
        detector.analyze_all_sequential_patches()
        
        with detector.Session() as session:
            total_changes = session.query(PatchChanges).count()
            patch_pairs = session.query(PatchChanges.from_patch, PatchChanges.to_patch).distinct().all()
            print(f"\nProcessed {len(patch_pairs)} patch pairs")
            print(f"Found {total_changes} total changes")
            for from_patch, to_patch in patch_pairs:
                count = session.query(PatchChanges).filter_by(
                    from_patch=from_patch, to_patch=to_patch).count()
                print(f"  {from_patch} -> {to_patch}: {count} changes")
    except Exception as e:
        logger.error(f"Main execution error: {e}")
