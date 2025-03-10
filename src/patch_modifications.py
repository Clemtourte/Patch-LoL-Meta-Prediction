#!/usr/bin/env python
import logging
import json
import numpy as np
import re
from sqlalchemy import create_engine
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
    """Normalise une version en 'major.minor'. Exemple: '14.19.1' devient '14.19'."""
    parts = patch.split('.')
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return patch

def load_json(field_val):
    """Si field_val est une chaîne, on la charge en JSON ; sinon on la retourne telle quelle."""
    if isinstance(field_val, str):
        try:
            return json.loads(field_val)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return None
    else:
        return field_val

def slice_field(field, ranks):
    """Si field est une liste et contient plus de 'ranks' éléments, on la découpe ; sinon, on la retourne."""
    if isinstance(field, list) and len(field) > ranks:
        return field[:ranks]
    return field

class PatchChangeDetector:
    def __init__(self):
        self.engine = create_engine('sqlite:///../datasets/league_data.db')
        self.Session = sessionmaker(bind=self.engine)

    def get_champion_stats(self, patch_version: str) -> Dict[str, ChampionStats]:
        """Récupère les ChampionStats pour un patch donné (version normalisée)."""
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
        """Récupère les SpellStats pour un patch donné (version normalisée)."""
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
        """Calcule les changements dans les stats de base entre deux patches."""
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
        """Calcule les changements dans les stats par niveau entre deux patches."""
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

    def compute_spell_changes(self, patch_version: str) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """
        Pour un patch donné, parcourt les enregistrements de SpellStats et calcule, pour chaque champ
        d'intérêt, la différence entre la valeur actuelle et la valeur précédente.
        Retourne un dictionnaire de la forme :
          { champion: { spell_type: { field: [diff_rank1, diff_rank2, ...], ... }, ... } }
        """
        spell_stats = self.get_spell_stats(patch_version)
        # Liste des champs numériques à comparer
        fields = ["base_damage", "cooldown", "mana_cost", "range",
                  "shield_value", "heal_value", "slow_percent",
                  "stun_duration", "knockup_duration", "root_duration"]
        # On peut aussi ajouter les ratios si besoin
        ratio_fields = ["ap_ratio", "ad_ratio", "bonus_ad_ratio", "max_health_ratio"]
        
        changes = {}
        for champion, spells in spell_stats.items():
            changes.setdefault(champion, {})
            for spell in spells:
                spell_type = spell.spell_type  # Assurez-vous que ce champ est bien renseigné
                changes[champion].setdefault(spell_type, {})
                for field in fields:
                    # Charger la valeur actuelle et la valeur précédente (on part de 0 si aucune valeur)
                    new_val = load_json(getattr(spell, field) or "[]")
                    prev_val = load_json(getattr(spell, "previous_" + field) or "[]")
                    # Si ce sont des listes, effectuer une soustraction élément par élément
                    if isinstance(new_val, list) and isinstance(prev_val, list):
                        n = min(len(new_val), len(prev_val))
                        diff_list = []
                        for i in range(n):
                            try:
                                diff_list.append(float(new_val[i]) - float(prev_val[i]))
                            except Exception as e:
                                logger.warning(f"Skipping non-numeric value for {champion} {spell_type} {field} at index {i}: {e}")
                        if diff_list and any(x != 0 for x in diff_list):
                            changes[champion][spell_type][field] = diff_list
                    # Si c'est un nombre unique
                    else:
                        try:
                            new_num = float(new_val)
                        except Exception:
                            new_num = 0.0
                        try:
                            prev_num = float(prev_val)
                        except Exception:
                            prev_num = 0.0
                        diff = new_num - prev_num
                        if diff != 0:
                            changes[champion][spell_type][field] = [diff]
                
                # Pour les ratios, on peut faire de même
                for field in ratio_fields:
                    new_val = getattr(spell, field)
                    prev_val = getattr(spell, "previous_" + field)
                    try:
                        new_num = float(new_val) if new_val is not None else 0.0
                    except Exception:
                        new_num = 0.0
                    try:
                        prev_num = float(prev_val) if prev_val is not None else 0.0
                    except Exception:
                        prev_num = 0.0
                    diff = new_num - prev_num
                    if diff != 0:
                        changes[champion][spell.spell_type][field] = [diff]
        
        return changes

    def analyze_patch_changes(self, old_patch: str, new_patch: str) -> Dict[str, Any]:
        """Analyse l'évolution entre deux patches en combinant stats de champions et changements d'abilités.
           Pour les abilities, on utilisera les données du nouveau patch (qui contiennent déjà les valeurs avant/après).
        """
        base = self.compute_base_stat_changes(old_patch, new_patch)
        per_level = self.compute_per_level_changes(old_patch, new_patch)
        # Ici, pour les abilities, nous analysons les changements du patch "new_patch"
        abilities = self.compute_spell_changes(new_patch)
        return {
            'base_stats': base,
            'per_level': per_level,
            'abilities': abilities
        }
    
    def save_patch_changes(self, from_patch: str, to_patch: str, changes: Dict[str, Any]) -> None:
        """Enregistre les changements détectés dans la table PatchChanges."""
        session = self.Session()
        try:
            # Enregistrer les changements de base stats
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
            
            # Enregistrer les changements de per_level
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
            
            # Enregistrer les changements d'abilités
            for champion, spells in changes['abilities'].items():
                for spell_type, fields_diff in spells.items():
                    for field, diff_list in fields_diff.items():
                        for rank, diff_val in enumerate(diff_list, start=1):
                            try:
                                diff_numeric = float(diff_val)
                            except Exception:
                                logger.warning(f"Skipping non-numeric diff for {champion} {spell_type} {field} rank {rank}")
                                continue
                            if diff_numeric != 0:
                                session.merge(PatchChanges(
                                    from_patch=from_patch,
                                    to_patch=to_patch,
                                    champion_name=champion,
                                    stat_type='ability',
                                    stat_name=f"{spell_type}_{field}_rank{rank}",
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
        """Analyse les changements entre patches séquentiels."""
        session = self.Session()
        try:
            patches = [p[0] for p in session.query(ChampionStats.version).distinct().all()]
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
        """Nettoie la table patch_changes."""
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
