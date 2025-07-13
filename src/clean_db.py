# debug_simple.py
import pandas as pd
from sqlalchemy import create_engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from data_preparation import format_patch_number

# UTILISER LE BON CHEMIN
engine = create_engine('sqlite:///../../datasets/league_data.db')  # ← Correction ici

print("=== DIAGNOSTIC SIMPLE ===")

# Tester le formatage des patches
print("\n1. TEST FORMATAGE:")
test_patches = ['14.24.1', '14.24', '14.23']
for patch in test_patches:
    formatted = format_patch_number(patch)
    print(f"{patch} → {formatted}")

# Vérifier après formatage
print("\n2. CHANGEMENTS APRÈS FORMATAGE:")
champ_changes_query = """
SELECT to_patch as patch, champion_name, stat_type, stat_name, change_value
FROM patch_changes 
WHERE champion_name = 'Ambessa'
  AND stat_type IN ('base_stat', 'per_level', 'ability')
  AND change_value IS NOT NULL
"""
champ_changes = pd.read_sql(champ_changes_query, engine)
champ_changes['patch'] = champ_changes['patch'].apply(format_patch_number)
champ_changes = champ_changes[champ_changes['patch'].str.fullmatch(r'\d+\.\d+')]

print("Ambessa après formatage:")
print(champ_changes[['champion_name', 'patch', 'stat_type', 'stat_name']])

# Vérifier winrates
print("\n3. WINRATES:")
winrates_query = """
SELECT patch, champion_name, winrate
FROM champion_winrates 
WHERE champion_name = 'Ambessa'
"""
winrates = pd.read_sql(winrates_query, engine)
winrates['patch'] = winrates['patch'].apply(format_patch_number)
winrates = winrates[winrates['patch'].str.fullmatch(r'\d+\.\d+')]

print("Ambessa winrates après formatage:")
print(winrates)

# Test du merge
print("\n4. TEST MERGE:")
# Simuler votre logique de merge
test_merge = pd.merge(champ_changes, winrates, on=['patch', 'champion_name'], how='left')
print(f"Résultat merge: {len(test_merge)} lignes")
print(test_merge[['champion_name', 'patch', 'winrate']])