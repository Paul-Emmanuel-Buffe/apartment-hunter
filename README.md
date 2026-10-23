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

---

## 3. Algorithme 2 : Random Forest Regressor (Méthode Ensembliste)

### 3.1 Présentation et Principe Théorique (Bagging)
Le **Random Forest Regressor** (Forêt Aléatoire) est un algorithme d'apprentissage supervisé qui repose sur l'approche ensembliste, et plus précisément sur le **Bagging** (*Bootstrap Aggregating*).

Au lieu de s'appuyer sur un seul arbre de décision (qui a tendance à mémoriser les données par cœur et faire du surapprentissage), la forêt construit des dizaines ou des centaines d'arbres **strictement indépendants et travaillant en parallèle**, puis combine leurs prédictions.

#### Les concepts clés du fonctionnement :
1. **Bootstrap (Aléa sur les lignes) :** Chaque arbre s'entraîne sur un échantillon différent, créé par tirage au sort avec remise parmi les maisons de la base d'entraînement.
2. **Feature Randomness (Aléa sur les colonnes) :** À chaque intersection d'un arbre, l'algorithme tire au sort un petit groupe de critères (ex: uniquement la Surface et l'Année). L'arbre doit se débrouiller avec ces critères imposés pour créer sa branche.
3. **Agrégation (Le Vote) :** La prédiction finale est la moyenne exacte des prédictions de tous les arbres indépendants.

---

### 3.2 Exemple Pratique : Le vote de la Forêt (En parallèle)

Imaginons que nous voulons prédire le prix d'une maison de **120 m²** avec **4 pièces**, et que notre forêt contient **3 arbres** ($N=3$) travaillant chacun dans leur coin, sans se concerter. *Note : Ces valeurs ont été fixées une fois pour toutes lors de l'entraînement, l'algorithme n'a plus besoin du fichier de données pour faire cette prédiction.*

*   **Arbre 1 (Expert Surface) :**
    *   *Contexte :* Lors de sa construction, le tirage au sort lui a imposé de se concentrer sur la Surface.
    *   *Parcours :* Notre maison répond "Oui" à *Surface > 100 m²*. Elle tombe dans un groupe final (feuille) pré-calculé valant un log-prix moyen de : $\mathbf{12.2}$

*   **Arbre 2 (Expert Pièces) :**
    *   *Contexte :* Le tirage au sort l'a forcé à regarder le nombre de pièces.
    *   *Parcours :* Notre maison répond "Oui" à *Pièces > 3* et tombe dans une feuille pré-calculée valant : $\mathbf{12.4}$

*   **Arbre 3 (Expert Mixte) :**
    *   *Parcours :* Croise différents critères et place la maison dans une feuille valant : $\mathbf{12.3}$

**L'Agrégation finale (Bagging) :**
Le Random Forest ramasse les copies et fait la moyenne des 3 experts pour obtenir le log-prix final :
$\hat{y}_{\log} = \frac{12.2 + 12.4 + 12.3}{3} = \mathbf{12.3}$

**Le retour à la réalité ($) :**
Pour connaître le vrai prix en dollars, nous annulons le log avec l'exponentielle (`np.exp`) :
$\text{Prix final} = \exp(12.3) \approx \mathbf{219\,695\ \$}$

---

### 3.3 Avantages et Limites (Random Forest)

#### **Avantages :**
- **Résistance au surapprentissage :** Grâce au double tirage aléatoire, il est beaucoup moins sensible à l'overfitting qu'un arbre unique. C'est la sagesse de la foule : une erreur individuelle est noyée par le groupe.
- **Relations non-linéaires :** Capture des interactions complexes sans nécessiter de transformation préalable des données (pas besoin de standardisation).
- **Robustesse :** Le moyennage encaisse très bien les valeurs extrêmes (outliers).
- **Calcul parallélisable :** Puisque les arbres sont indépendants, l'entraînement peut être réparti sur tous les cœurs du processeur (via le paramètre `n_jobs=-1`).

#### **Limites :**
- **Incapacité d'extrapolation :** Il ne peut pas prédire un prix supérieur au prix maximum vu lors de son entraînement.
- **Boîte Noire :** Impossible de visualiser l'arbre final puisqu'il s'agit d'une moyenne de centaines d'arbres.

---

### 3.4 Hyperparamètres Clés

- **`n_estimators`** : Le nombre d'arbres (ex: 100). Plus il y en a, plus la moyenne est stable (sans risque de surapprentissage), mais plus le calcul est long.
- **`max_depth`** : Profondeur maximale de chaque arbre (ex: 15 questions successives). Limite la complexité.
- **`min_samples_split`** : Nombre minimum de maisons requises dans un nœud pour autoriser une nouvelle division (ex: 5). Agit comme un garde-fou.
- **`random_state`** : Fixe la graine aléatoire pour garantir des résultats identiques à chaque exécution du notebook.

Voici une version enrichie et plus détaillée de ton cours. J'y ai ajouté les fondements mathématiques vulgarisés (notamment le rôle exact du "Gradient" et de la "Fonction de perte") ainsi que des explications plus poussées sur la mécanique interne du modèle.

C'est idéal pour démontrer dans ton notebook que tu ne te contentes pas d'importer une librairie, mais que tu comprends la véritable machinerie de l'algorithme.

---

```markdown
## 4. Algorithme 3 : Gradient Boosting Regressor (Méthode Ensembliste - Boosting)

### 4.1 Présentation et Principes Fondamentaux : Le mariage de deux mondes

Si le Random Forest repose sur l'indépendance et le vote parallèle (Bagging), le **Gradient Boosting** repose sur la **collaboration séquentielle** (Boosting). 

Fondamentalement, cet algorithme est la fusion brillante entre deux grands concepts du Machine Learning :
1. **L'arbre de décision comme "Apprenant Faible" (*Weak Learner*) :** Au lieu de construire des arbres profonds et complexes, le Boosting utilise volontairement des arbres très simples (peu profonds). Pris isolément, un tel arbre est un prédicteur médiocre, mais il a l'avantage de ne pas surapprendre.
2. **La Descente de Gradient (L'optimisation) :** C'est le moteur mathématique de l'algorithme. L'objectif est de minimiser une **Fonction de Perte** (*Loss Function*), généralement l'Erreur Quadratique Moyenne (MSE) pour la régression. 

**Pourquoi "Gradient" ?** 
En mathématiques, le gradient indique la pente de l'erreur. Il s'avère que si l'on cherche à minimiser l'erreur quadratique (MSE), la dérivée de cette erreur (le gradient) correspond exactement au **résidu** (la différence entre le vrai prix et le prix prédit). 
Ainsi, demander à un arbre de "prédire le résidu", c'est littéralement lui demander de calculer le gradient pour savoir dans quelle direction corriger la prédiction globale.

#### La mécanique mathématique de la cascade :
L'algorithme construit un modèle additif pas à pas :

1. **L'initialisation ($F_0$) :** Le modèle fait une première prédiction constante pour toutes les maisons, généralement la moyenne globale des prix.
2. **Le calcul de l'erreur (Pseudo-Résidus) :** Pour chaque maison $i$, on calcule l'erreur actuelle : $r_i = y_i - F_{m-1}(x_i)$.
3. **L'arbre correcteur ($h_m$) :** Un nouvel arbre de décision est entraîné **non pas sur les prix $y$**, mais sur ces résidus $r$.
4. **La mise à jour avec le *Learning Rate* ($\nu$) :** On met à jour la prédiction globale en y ajoutant les trouvailles du nouvel arbre, freinées par le taux d'apprentissage :
   $$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$
5. **Répétition :** On recommence l'opération des centaines de fois ($M$ itérations). Le modèle final est la somme de l'initialisation et de toutes les micro-corrections.

---

### 4.2 Exemple Pratique : L'apprentissage par l'erreur pas à pas

Imaginons une maison dont le vrai log-prix est de **12.5**.

*   **Étape 0 (Prédiction de base $F_0$) :** Le modèle commence par prédire la moyenne globale, disons **12.0**.
    *   *Erreur résiduelle :* $12.5 - 12.0 = \mathbf{+0.5}$ (Le modèle est en dessous).

*   **Arbre 1 (Le premier correcteur $h_1$) :** 
    L'Arbre 1 cherche à prédire cette erreur de **+0.5**. Disons qu'il trouve +0.4.
    Au lieu d'ajouter +0.4 d'un coup, le modèle applique le frein, le **Learning Rate** (ex: $\nu = 0.1$).
    *   *Nouvelle prédiction $F_1$ :* $12.0 + (0.1 \times 0.4) = \mathbf{12.04}$
    *   *Nouveau résidu à corriger :* $12.5 - 12.04 = \mathbf{+0.46}$

*   **Arbre 2 (Le deuxième correcteur $h_2$) :**
    L'Arbre 2 prend le relais. Il tente d'estimer ce nouveau résidu de +0.46. Il trouve +0.5.
    *   *Nouvelle prédiction $F_2$ :* $12.04 + (0.1 \times 0.5) = \mathbf{12.09}$
    *   *Nouveau résidu à corriger :* $12.5 - 12.09 = \mathbf{+0.41}$

... Et ainsi de suite, jusqu'à l'Arbre 200 ! Petit à petit, guidé par la pente de l'erreur, le modèle s'approche de la vraie valeur de 12.5 de manière douce et stable.

---

### 4.3 Décorticage de tes Hyperparamètres (La magie du "Hist")

Le code utilise `HistGradientBoostingRegressor`. Le préfixe **Hist** signifie "Histogramme". 
Dans un arbre classique, pour trouver où diviser une variable continue (ex: la Surface), l'algorithme teste toutes les valeurs possibles (120,5 m², 120,6 m²...). C'est très long. 
La version *Hist* divise l'espace en "bacs" discrets (ex: 256 bacs d'histogrammes). L'algorithme ne cherche plus les coupures sur les valeurs réelles, mais sur ces bacs. Le temps d'entraînement passe d'une complexité $O(n\_données)$ à $O(n\_bacs)$, rendant le calcul **foudroyant de rapidité** sur de grands jeux de données.

*   **`max_iter=200` (Nombre d'arbres) :** Définit le nombre d'étapes de correction. Il faut trouver le juste milieu : trop peu d'arbres = sous-apprentissage ; trop d'arbres = surapprentissage.
*   **`learning_rate=0.1` (Taux d'apprentissage) :** Contrôle l'impact de chaque arbre. **Il est intimement lié à `max_iter`** : si on divise le `learning_rate` par deux, il faudra généralement doubler `max_iter` pour atteindre la même cible, ce qui donne un modèle souvent plus précis, mais plus long à entraîner.
*   **`max_depth=7` (Profondeur) :** Garde les arbres volontairement "faibles". Un arbre profond chercherait à corriger parfaitement les résidus du premier coup, ce qui détruirait la logique d'apprentissage progressif.
*   **`random_state=42` :** Garantit la reproductibilité des résultats.

---

### 4.4 Entraînement, Prédiction et Évaluation

La phase de modélisation respecte les mêmes avantages que la Forêt Aléatoire :
1. **Insensibilité à l'échelle :** L'approche par histogrammes et arbres de décision structure l'espace via des seuils. Standardiser les données (via `RobustScaler`) est strictement inutile et ne ferait que rajouter une étape superflue.
2. **Reconversion de la cible :** Le modèle ayant été entraîné sur la cible transformée (`y_train` en log), il produit une prédiction finale en log-prix. L'application de la fonction inverse `np.exp()` permet de repasser en monnaie réelle pour analyser nos métriques (MAE et RMSE).

---

### 4.5 Avantages et Limites (Gradient Boosting)

#### **Avantages :**
- **Précision chirurgicale :** En ciblant activement et itérativement ses propres erreurs, c'est l'un des algorithmes les plus puissants existants pour les données tabulaires.
- **Optimisation des ressources :** L'approche `Hist` offre un rapport performance/temps d'exécution exceptionnel, optimisant considérablement l'usage de la mémoire RAM.

#### **Limites :**
- **Risque de surapprentissage (Overfitting) :** Si on le laisse tourner trop longtemps (`max_iter` excessif), le modèle finira par considérer le "bruit" aléatoire ou les valeurs aberrantes comme des motifs légitimes à apprendre et à corriger.
- **Séquentiel par nature :** Puisque l'Arbre $m$ a besoin des résidus générés par l'Arbre $m-1$, la construction des arbres ne peut pas être parallélisée (contrairement au `n_jobs=-1` du Random Forest).

---

### L'ajustement fin des hyperparamètres avec GridSearchCV

## 5. Interprétabilité, Choix du Modèle et Déploiement
### 5.1 Ouvrir la « Boîte Noire » : L'analyse par Permutation Importance
### 5.2 Bilan et Choix du Modèle Champion
### 5.3 Sérialisation pour la Mise en Production (Joblib)