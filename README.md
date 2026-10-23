# Analyse et Fonctionnement des Algorithmes de Régression

## 1. Introduction à la Régression et Préparation des Données
Dans le domaine du Machine Learning, notre projet s'inscrit dans le cadre de l'**apprentissage supervisé**. Notre objectif est de prédire la valeur d'un bien immobilier en nous basant sur ses caractéristiques (surface, localisation, qualité de construction, etc.). Puisque la variable que nous cherchons à prédire est une valeur continue (un prix en dollars) et non une catégorie, nous utilisons des algorithmes de **régression**.

Cependant, la performance d'un modèle de régression dépend fortement de la qualité des données qu'on lui fournit. Le marché immobilier présente des spécificités techniques (forte asymétrie, variables géographiques complexes) qui ont nécessité des transformations mathématiques préalables :

*   **L'échantillonnage stratifié :** Pour garantir une évaluation juste de nos modèles, nous avons séparé nos données d'entraînement et de test en veillant à conserver la même proportion de biens abordables, moyens et luxueux dans les deux échantillons (via la méthode `qcut`).
*   **La gestion de l'asymétrie (Logarithme et RobustScaler) :** Le marché immobilier est dominé par une poignée de biens d'exception très onéreux qui faussent les calculs. En prédisant le logarithme du prix (`log_price`) au lieu du prix brut, nous normalisons la distribution. De plus, l'utilisation d'un `RobustScaler` (basé sur la médiane et l'écart interquartile) permet de mettre à l'échelle nos données sans être influencé par ces valeurs extrêmes.
*   **Le Target Encoding du Code Postal :** Les algorithmes ne comprennent pas les zones géographiques. Au lieu de multiplier artificiellement les colonnes (One-Hot Encoding), nous avons remplacé chaque code postal par le log-prix moyen historique de ce quartier. Ce calcul a été effectué strictement sur le jeu d'entraînement pour éviter toute fuite de données (Data Leakage).

Une fois cette fondation solide établie, nous avons pu confronter trois familles d'algorithmes de régression.

### 1.1 Échantillonnage Stratifié : Équilibrer les biens abordables et luxueux

Le problème initial :
En Machine Learning, on sépare nos données en un jeu d'entraînement (Train) et un jeu de test (Test). Si l'on fait cela au hasard (aléatoirement), on prend le risque que notre jeu de test ne contienne que des maisons de taille moyenne, et aucune villa de luxe. Le modèle ne sera donc jamais évalué sur sa capacité à prédire les biens très chers.

La solution : L'échantillonnage stratifié
La stratification garantit que la distribution d'une variable cible est identique dans le Train et dans le Test.

Pour une classification (ex: Chien/Chat), c'est facile : on garde 50% de chiens dans le train et 50% dans le test.

Pour une régression (le prix continu d'une maison), c'est plus complexe. On utilise alors une méthode mathématique (pd.qcut en Python) pour découper nos prix continus en "tranches" (quantiles). Par exemple, 5 tranches : Très abordable, Abordable, Moyen, Cher, Très cher.
L'algorithme s'assure ensuite de piocher proportionnellement dans ces 5 boîtes pour créer le Train et le Test.

### 1.2 Gérer l'asymétrie du marché : Transformation Logarithmique de la cible et RobustScaler

**Le problème des valeurs extrêmes (Outliers) :**Le marché immobilier est fortement asymétrique (on parle de Right-Skewed Distribution). La majorité des maisons coûte entre 300 000 $et 800 000$, mais quelques biens à 5 ou 10 millions de dollars "tirent" la moyenne vers le haut. Les modèles linéaires sont très sensibles à ces valeurs extrêmes qui faussent leurs calculs.

**Solution 1 : La transformation Logarithmique (np.log)**
En appliquant le logarithme mathématique sur la variable cible (le prix), on "tasse" les très grandes valeurs. L'échelle passe de millions à des valeurs autour de 12 à 15. La distribution prend alors une forme de "cloche" classique (loi normale). Les modèles apprennent beaucoup mieux sur des courbes normales. Une fois la prédiction faite, on utilise l'exponentielle (np.exp) pour repasser en dollars.

**Solution 2 : Le RobustScaler**
Même avec le log sur le prix, nos features (surfaces, etc.) contiennent des valeurs extrêmes. Le **StandardScaler classique** utilise la moyenne, qui est faussée par les biens les plus chers. Le **RobustScaler**, lui, utilise la médiane et l'écart interquartile (IQR : la zone où se trouvent 50 % des données centrales). **Il ignore donc complètement l'influence des valeurs aberrantes pour mettre toutes les colonnes à la même échelle (entre 0 et 1 environ)**.

### 1.3 L'Encodage Intelligent des Catégories : Le Target Encoding sur le Code Postal

### 1.4 Sélection rigoureuse des caractéristiques (Feature Selection) avec Boruta

Pour s'assurer que les modèles ne s'entraînent que sur des données pertinentes, une étape de sélection des variables (*Feature Selection*) est souvent nécessaire avant la modélisation. Pour répondre à ce besoin, l'algorithme **Boruta** est utilisé comme filtre de validation.

**Le concept :** 
Boruta est un algorithme qui s'appuie sur un modèle d'arbres de décision (généralement un Random Forest) pour comparer le pouvoir prédictif des variables réelles face à du bruit statistique pur. L'objectif n'est pas juste de prendre le "Top 10" des variables, mais de conserver **toutes** les variables statistiquement pertinentes.

**Le fonctionnement (La compétition contre le hasard) :**
1. **Création d'ombres :** Boruta duplique toutes les variables du jeu de données, mais mélange aléatoirement leurs valeurs. Ces « variables ombres » (*shadow features*) perdent tout lien logique avec la variable cible.
2. **Entraînement :** Un modèle Random Forest est entraîné sur ce jeu de données combiné (vraies variables + variables ombres).
3. **Le Jugement :** L'algorithme extrait l'importance de chaque variable. La meilleure variable ombre (le meilleur bruit) sert de seuil de référence.
4. **La Décision :** Toute variable réelle qui n'arrive pas à battre le score du hasard est rejetée. Celles qui performent significativement mieux sont définitivement validées.

◄───────────────────────────── 2p colonnes ─────────────────────────────►

         ┌───────────────────────────────────┬───────────────────────────────────┐ ───
         │          VRAIES VARIABLES         │         VARIABLES OMBRES          │  ▲
         │             (p colonnes)          │       (p colonnes mélangées)      │  │
         ├───────────────────────────────────┼───────────────────────────────────┤  │
Maison 1 │ sqft_living | grade | zipcode_enc │ shadow_sqft | shadow_gr | sh_zip  │  │
Maison 2 │    1 500    │   7   │    13,2     │    3 400    │     9   │   12,1   │  │
Maison 3 │    2 200    │   8   │    14,1     │      850    │     6   │   13,8   │  N lignes
  ...    │      850    │   6   │    12,4     │    1 500    │     7   │   13,2   │  │
Maison N │      ...    │  ...  │     ...     │     ...     │    ...  │   ...    │  │
         └───────────────────────────────────┴───────────────────────────────────┘  ▼
                                       │
                                       ▼
                       [ Entraînement du Random Forest ]
                        (sur les 2p colonnes mélangées)

**L'intérêt pour le projet :** 
L'application de Boruta permet de prouver objectivement que le jeu de données final fourni aux modèles de régression (Gradient Boosting, Ridge) est débarrassé de tout bruit parasite, limitant ainsi fortement le risque de surapprentissage (*overfitting*).

## 2. Algorithme 1 : La Régression Ridge (Modèle Linéaire Pénalisé)

### Pourquoi Ridge plutôt qu'une régression classique ou Lasso ?

Le choix de cet algorithme découle directement des caractéristiques de notre jeu de données :

* **Face à la régression classique (MCO) :** le modèle linéaire classique tente d'isoler l'effet propre de chaque critère en inversant la matrice des recoupements ($X^T X$). En présence de variables corrélées (comme la surface et le nombre de pièces), ce calcul tente de diviser par un chiffre quasi nul. Les poids explosent avec des valeurs démesurées et opposées, rendant le modèle instable. Ridge corrige ce défaut en stabilisant le calcul matriciel.
* **Face à Lasso (L1) :** Lasso a pour propriété d'annuler certains poids (mise à zéro stricte), ce qui en fait un outil de sélection drastique de variables. Notre sélection de variables ayant déjà été opérée en amont (notamment avec Boruta), nous souhaitons conserver toutes les caractéristiques restantes. Ridge (pénalité L2) est donc le candidat idéal : il ne supprime aucune variable, mais réduit et répartit équitablement l'impact entre les critères redondants.

---

### Recherche automatique de la meilleure pénalité avec RidgeCV

L'efficacité de Ridge repose sur la valeur de son amende, le paramètre **Alpha** :

* **Alpha trop faible** : le modèle se comporte comme une régression classique et redevient instable.
* **Alpha trop fort** : les poids sont excessivement écrasés et le modèle perd sa capacité à bien prédire (sous-apprentissage).

Pour identifier la valeur idéale sans réglage manuel arbitraire, nous utilisons `RidgeCV` de Scikit-Learn :

1. **Grille de test :** nous définissons une échelle de valeurs d'Alpha à évaluer (ex. de 0,1 à 1 000).
2. **Validation croisée (CV) :** les données d'entraînement sont découpées en plusieurs blocs. Pour chaque valeur d'Alpha, le modèle est entraîné sur une partie des blocs et testé sur le bloc restant afin d'évaluer son erreur moyenne réelle.
3. **Optimisation :** l'algorithme sélectionne l'Alpha affichant la plus faible marge d'erreur.

C'est cet Alpha optimal qui vient s'ajouter sur la diagonale de la grille via la formule $b = (X^T X + \text{Alpha} \times I)^{-1} X^T y$, garantissant des prédictions stables et robustes sur les futures transactions.

## 3. Algorithme 2 : Random Forest Regressor (Méthode Ensembliste - Bagging)

## 4. Algorithme 3 : Gradient Boosting Regressor (Méthode Ensembliste - Boosting)
### L'ajustement fin des hyperparamètres avec GridSearchCV

## 5. Interprétabilité, Choix du Modèle et Déploiement
### 5.1 Ouvrir la « Boîte Noire » : L'analyse par Permutation Importance
### 5.2 Bilan et Choix du Modèle Champion
### 5.3 Sérialisation pour la Mise en Production (Joblib)