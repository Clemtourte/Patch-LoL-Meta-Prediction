# Rapport de Validation du R²

Date d'exécution: 2025-05-12 15:20:25

## 1. Test de Shuffling des Étiquettes

Ce test vérifie si le modèle peut apprendre à partir d'étiquettes aléatoires, ce qui indiquerait une fuite de données.

Résultats: [Voir fichier shuffle_test_results.png]

## 2. Étude d'Ablation

Cette étude mesure l'impact de différents groupes de caractéristiques sur la performance du modèle.

Résultats: [Voir fichier ablation_study_results.png]

## 3. Validation sur Patchs Non-consécutifs

Ce test évalue la robustesse du modèle à travers différentes 'époques' du jeu.

Résultats: [Voir fichier nonconsecutive_validation_results.png]

## 4. Analyse de la Distribution des Erreurs

Cette analyse examine comment les erreurs de prédiction sont distribuées par champion et classe.

Résultats: [Voir fichiers error_distribution.png, error_by_class.png]

## Conclusion

Sur la base des résultats des tests de validation, nous pouvons conclure que :

1. **Test de Shuffling** : Si le R² avec des étiquettes mélangées est proche de zéro, cela confirme que le modèle ne peut pas apprendre de motifs aléatoires, ce qui renforce la validité du R² obtenu.

2. **Étude d'Ablation** : L'impact relatif des différentes catégories de caractéristiques montre quels aspects des changements de patch influencent le plus les taux de victoire. Une forte dépendance aux caractéristiques temporelles pourrait indiquer un R² artificiellement élevé.

3. **Validation Non-consécutive** : Un modèle qui maintient un R² élevé lorsqu'il est entraîné sur une époque et testé sur une autre démontre une robustesse et une généralisation dans le temps, ce qui valide davantage le R².

4. **Distribution des Erreurs** : Une distribution équilibrée des erreurs entre les classes de champions et les patches suggère que le modèle capture des patterns universels plutôt que de se spécialiser sur certains cas spécifiques.

