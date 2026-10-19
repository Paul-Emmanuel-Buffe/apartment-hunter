# Analyse et Fonctionnement des Algorithmes de Régression

## 1. Introduction à la Régression et Préparation des Données
### 1.1 Échantillonnage Stratifié : Équilibrer les biens abordables et luxueux
### 1.2 Gérer l'asymétrie du marché : Transformation Logarithmique de la cible et RobustScaler
### 1.3 L'Encodage Intelligent des Catégories : Le Target Encoding sur le Code Postal

## 2. Algorithme 1 : La Régression Ridge (Modèle Linéaire Pénalisé)
### Recherche automatique de la meilleure pénalité avec RidgeCV

## 3. Algorithme 2 : Random Forest Regressor (Méthode Ensembliste - Bagging)

## 4. Algorithme 3 : Gradient Boosting Regressor (Méthode Ensembliste - Boosting)
### L'ajustement fin des hyperparamètres avec GridSearchCV

## 5. Interprétabilité, Choix du Modèle et Déploiement
### 5.1 Ouvrir la « Boîte Noire » : L'analyse par Permutation Importance
### 5.2 Bilan et Choix du Modèle Champion
### 5.3 Sérialisation pour la Mise en Production (Joblib)