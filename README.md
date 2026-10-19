# Analyse et Fonctionnement des Algorithmes de Régression

## 1. Introduction à la Régression et Préparation des Données
Dans le domaine du Machine Learning, notre projet s'inscrit dans le cadre de l'**apprentissage supervisé**. Notre objectif est de prédire la valeur d'un bien immobilier en nous basant sur ses caractéristiques (surface, localisation, qualité de construction, etc.). Puisque la variable que nous cherchons à prédire est une valeur continue (un prix en dollars) et non une catégorie, nous utilisons des algorithmes de **régression**.

Cependant, la performance d'un modèle de régression dépend fortement de la qualité des données qu'on lui fournit. Le marché immobilier présente des spécificités techniques (forte asymétrie, variables géographiques complexes) qui ont nécessité des transformations mathématiques préalables :

*   **L'échantillonnage stratifié :** Pour garantir une évaluation juste de nos modèles, nous avons séparé nos données d'entraînement et de test en veillant à conserver la même proportion de biens abordables, moyens et luxueux dans les deux échantillons (via la méthode `qcut`).
*   **La gestion de l'asymétrie (Logarithme et RobustScaler) :** Le marché immobilier est dominé par une poignée de biens d'exception très onéreux qui faussent les calculs. En prédisant le logarithme du prix (`log_price`) au lieu du prix brut, nous normalisons la distribution. De plus, l'utilisation d'un `RobustScaler` (basé sur la médiane et l'écart interquartile) permet de mettre à l'échelle nos données sans être influencé par ces valeurs extrêmes.
*   **Le Target Encoding du Code Postal :** Les algorithmes ne comprennent pas les zones géographiques. Au lieu de multiplier artificiellement les colonnes (One-Hot Encoding), nous avons remplacé chaque code postal par le log-prix moyen historique de ce quartier. Ce calcul a été effectué strictement sur le jeu d'entraînement pour éviter toute fuite de données (Data Leakage).

Une fois cette fondation solide établie, nous avons pu confronter trois familles d'algorithmes de régression.

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