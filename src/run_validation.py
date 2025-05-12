# run_validations.py
import time
import pandas as pd
import matplotlib.pyplot as plt
from validation_shuffle import run_shuffle_test
from validation_ablation import run_ablation_study
from validation_nonconsecutive import run_nonconsecutive_validation
from validation_errordistrib import run_error_distribution_analysis

def run_all_validations():
    """Exécute toutes les validations et génère un rapport de synthèse."""
    start_time = time.time()
    
    print("============================================")
    print("Exécution des tests de validation du modèle")
    print("============================================")
    
    # 1. Test de Shuffling des Étiquettes
    print("\n\n1. Test de Shuffling des Étiquettes")
    print("------------------------------------")
    shuffle_results = run_shuffle_test(n_iterations=5)
    
    # 2. Étude d'Ablation
    print("\n\n2. Étude d'Ablation")
    print("-------------------")
    ablation_results = run_ablation_study()
    
    # 3. Validation sur Patchs Non-consécutifs
    print("\n\n3. Validation sur Patchs Non-consécutifs")
    print("----------------------------------------")
    nonconsecutive_results = run_nonconsecutive_validation()
    
    # 4. Analyse de la Distribution des Erreurs
    print("\n\n4. Analyse de la Distribution des Erreurs")
    print("----------------------------------------")
    error_df, class_stats, champion_stats = run_error_distribution_analysis()
    
    # Durée totale
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n\n============================================")
    print(f"Toutes les validations terminées en {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("============================================")
    
    # Créer un rapport de synthèse
    with open('rapport_validation_r2.md', 'w', encoding='utf-8') as f:
        f.write("# Rapport de Validation du R²\n\n")
        f.write(f"Date d'exécution: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Test de Shuffling des Étiquettes\n\n")
        f.write("Ce test vérifie si le modèle peut apprendre à partir d'étiquettes aléatoires, ce qui indiquerait une fuite de données.\n\n")
        f.write("Résultats: [Voir fichier shuffle_test_results.png]\n\n")
        
        f.write("## 2. Étude d'Ablation\n\n")
        f.write("Cette étude mesure l'impact de différents groupes de caractéristiques sur la performance du modèle.\n\n")
        f.write("Résultats: [Voir fichier ablation_study_results.png]\n\n")
        
        f.write("## 3. Validation sur Patchs Non-consécutifs\n\n")
        f.write("Ce test évalue la robustesse du modèle à travers différentes 'époques' du jeu.\n\n")
        f.write("Résultats: [Voir fichier nonconsecutive_validation_results.png]\n\n")
        
        f.write("## 4. Analyse de la Distribution des Erreurs\n\n")
        f.write("Cette analyse examine comment les erreurs de prédiction sont distribuées par champion et classe.\n\n")
        f.write("Résultats: [Voir fichiers error_distribution.png, error_by_class.png]\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("Sur la base des résultats des tests de validation, nous pouvons conclure que :\n\n")
        f.write("1. **Test de Shuffling** : Si le R² avec des étiquettes mélangées est proche de zéro, cela confirme que le modèle ne peut pas apprendre de motifs aléatoires, ce qui renforce la validité du R² obtenu.\n\n")
        f.write("2. **Étude d'Ablation** : L'impact relatif des différentes catégories de caractéristiques montre quels aspects des changements de patch influencent le plus les taux de victoire. Une forte dépendance aux caractéristiques temporelles pourrait indiquer un R² artificiellement élevé.\n\n")
        f.write("3. **Validation Non-consécutive** : Un modèle qui maintient un R² élevé lorsqu'il est entraîné sur une époque et testé sur une autre démontre une robustesse et une généralisation dans le temps, ce qui valide davantage le R².\n\n")
        f.write("4. **Distribution des Erreurs** : Une distribution équilibrée des erreurs entre les classes de champions et les patches suggère que le modèle capture des patterns universels plutôt que de se spécialiser sur certains cas spécifiques.\n\n")
        
        # Ajoutez ici des conclusions spécifiques basées sur les résultats réels des tests.

if __name__ == "__main__":
    run_all_validations()