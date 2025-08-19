# debug_final.py (dans src/)
import pandas as pd
from sqlalchemy import create_engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from data_preparation import format_patch_number

# BON CHEMIN depuis src/ vers datasets/
engine = create_engine('sqlite:///../datasets/league_data.db')

print("=== DIAGNOSTIC FINAL ===")

try:
    # Test simple
    test_query = "SELECT COUNT(*) as total FROM patch_changes"
    result = pd.read_sql(test_query, engine)
    print(f"Database OK: {result['total'].iloc[0]} patch changes total")
    
    # Test Ambessa
    print("\n1. AMBESSA - Changements bruts:")
    ambessa_raw = pd.read_sql("SELECT * FROM patch_changes WHERE champion_name = 'Ambessa'", engine)
    print(ambessa_raw[['champion_name', 'to_patch', 'stat_type', 'stat_name']])
    
    print("\n2. AMBESSA - Après formatage:")
    ambessa_formatted = ambessa_raw.copy()
    ambessa_formatted['patch'] = ambessa_formatted['to_patch'].apply(format_patch_number)
    print(ambessa_formatted[['champion_name', 'patch', 'stat_type', 'stat_name']])
    
    print("\n3. AMBESSA - Winrates:")
    ambessa_wr = pd.read_sql("SELECT * FROM champion_winrates WHERE champion_name = 'Ambessa'", engine)
    ambessa_wr['patch'] = ambessa_wr['patch'].apply(format_patch_number)
    print(ambessa_wr[['champion_name', 'patch', 'winrate']])
    
    # Simuler le merge exact de votre logique
    print("\n4. SIMULATION DU MERGE:")
    
    # Créer matrice pivot comme dans votre code
    if len(ambessa_formatted) > 0:
        ambessa_pivot = pd.pivot_table(
            ambessa_formatted,
            index=['patch', 'champion_name'],
            columns=['stat_type', 'stat_name'],
            values='change_value',
            fill_value=0
        )
        ambessa_pivot.columns = [f'{col[0]}_{col[1]}' for col in ambessa_pivot.columns]
        ambessa_pivot = ambessa_pivot.reset_index()
        print("Matrice pivot Ambessa:")
        print(ambessa_pivot[['champion_name', 'patch']])
        
        # Test merge
        merge_result = pd.merge(ambessa_pivot, ambessa_wr, on=['patch', 'champion_name'], how='left')
        print(f"Résultat merge: {len(merge_result)} lignes")
        print(merge_result[['champion_name', 'patch', 'winrate']])
    
    # Vérifier si Ambessa est dans "champions perdus"
    print("\n5. TEST CHAMPIONS PERDUS:")
    
    # Simuler votre logique
    all_champ_changes = pd.read_sql("""
        SELECT DISTINCT champion_name FROM patch_changes 
        WHERE stat_type IN ('base_stat', 'per_level', 'ability')
    """, engine)
    
    all_champ_winrates = pd.read_sql("""
        SELECT DISTINCT champion_name FROM champion_winrates
    """, engine)
    
    champions_in_changes = set(all_champ_changes['champion_name'].unique())
    champions_in_winrates = set(all_champ_winrates['champion_name'].unique())
    champions_perdus = champions_in_winrates - champions_in_changes
    
    print(f"Total champions avec changements: {len(champions_in_changes)}")
    print(f"Total champions avec winrates: {len(champions_in_winrates)}")
    print(f"Champions perdus: {len(champions_perdus)}")
    print(f"Ambessa dans changements: {'Ambessa' in champions_in_changes}")
    print(f"Ambessa dans winrates: {'Ambessa' in champions_in_winrates}")
    print(f"Ambessa dans perdus: {'Ambessa' in champions_perdus}")
    
    # Si Ambessa est dans les perdus, vérifier combien de patches elle a
    if 'Ambessa' in champions_perdus:
        print(f"Ambessa patches count: {len(ambessa_wr)}")
        if len(ambessa_wr) >= 2:
            print("✅ Ambessa devrait être ajoutée par la logique 'champions perdus'")
        else:
            print("❌ Ambessa n'a pas assez de patches")

except Exception as e:
    print(f"Erreur: {e}")