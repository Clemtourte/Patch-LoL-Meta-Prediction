# quick_debug.py (dans src/)
import sys
import os

# Ajoute le dossier parent pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_preparation import prepare_prediction_data  # Sans 'src.'
    print("✅ Import réussi")
    
    data = prepare_prediction_data()
    features = data['feature_names']
    
    print(f"Nombre de features: {len(features)}")
    
    # Check du data leakage
    if 'champion_class' in features:
        print("❌ LEAKAGE: champion_class dans les features!")
    else:
        print("✅ Pas de champion_class dans les features")
        
    print(f"Taille dataset: {len(data['full_data'])}")
    
except Exception as e:
    print(f"Erreur: {e}")