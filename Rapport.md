- Exercice 1 :

- Exercice 2 :

- Exercice 3 :

1/ En C on déclare et alloue dynamiquement une matrice pour utiliser BLAS et LAPACK.

On déclare la matrice 2D ou plus, comme un pointeur de type double. On alloue la mémoire en utilisant `malloc` et on stocke les éléments de la matrice de préférence colonne par colonne (column major ordering). Le stocker ligne par ligne (row major ordering) requiert plus de mémoire et de temps que le stockage contigu colonne par colonne.

Exemple matrice \( n \times m \) :
```c
matrix = (double *)malloc(m * n * sizeof(double));
```

2/ La constante `LAPACK_COL_MAJOR` est une variable qui stocke de façon contiguë les éléments colonne par colonne de la matrice (column-major ordering).

3/ La dimension principale (leading dimension) notée `ld`, est un paramètre (variable) dans les bibliothèques BLAS et LAPACK pour spécifier la taille physique de la première dimension de la matrice en mémoire. C'est le nombre de lignes allouées en mémoire pour la matrice, indépendamment du nombre de lignes logiques utilisées dans les calculs.

4/ La fonction `dgbmv` est utilisée par la bibliothèque BLAS qui effectue une opération de multiplication matrice-vecteur pour une matrice bande (band matrix). La fonction `dgbmv` implémente l'opération suivante :

\[
y := \alpha \cdot A \cdot x + \beta \cdot y
\]

5/ La fonction `dgbtrf` est utilisée dans la bibliothèque LAPACK, elle effectue la factorisation LU d'une matrice bande (band matrix). La factorisation LU décompose une matrice \( A \) en deux matrices triangulaires : une matrice triangulaire inférieure \( L \) et une matrice triangulaire supérieure \( U \).

6/ La fonction `dgbtrs` est une routine de la bibliothèque LAPACK qui résout un système d'équations linéaires avec une matrice bande \( A \) en utilisant la factorisation LU obtenue par `dgbtrf`.

La fonction `dgbtrs` utilise les matrices \( L \) et \( U \) obtenues par `dgbtrf` pour résoudre le système \( A \cdot x = b \). Elle effectue d'abord une substitution avant pour résoudre \( L \cdot y = b \), puis une substitution arrière pour résoudre \( U \cdot x = y \).

7/ La fonction `dgbtsv` est une routine de la bibliothèque LAPACK qui résout un système d'équations linéaires avec une matrice bande \( A \) en utilisant une approche combinée de factorisation LU suivie de la résolution du système.

La fonction `dgbtsv` implémente les étapes suivantes :

- **Factorisation LU** : Utilise `dgbtrf` pour factoriser la matrice bande \( A \) en une matrice triangulaire inférieure \( L \) et une matrice triangulaire supérieure \( U \).
- **Résolution du système** : Utilise `dgbtrs` pour résoudre le système \( A \cdot x = b \) en utilisant les matrices \( L \) et \( U \) obtenues lors de la factorisation.

En résumé, `dgbtsv` combine les opérations de `dgbtrf` et `dgbtrs` en une seule fonction pour simplifier le processus de résolution d'un système d'équations linéaires avec une matrice bande.
`dgbtsv = dgbtrf + dgbtrs`

8/ Pour calculer la norme du résidu relatif avec des appels BLAS, vous pouvez suivre les étapes suivantes en utilisant les fonctions `ddot`, `dnrm2`, et `daxpy` de la bibliothèque BLAS.

- **Calcul de \( A \cdot \hat{x} \)** : Utilisez `cblas_dgemv` pour effectuer le produit matriciel-vecteur.
- **Calcul du résidu \( b - A\hat{x} \)** : Soustrayez les éléments de \( A \cdot \hat{x} \) des éléments de \( b \).
- **Calcul des normes** : Utilisez `cblas_dnrm2` pour calculer les normes des vecteurs \( b \), \( A \cdot \hat{x} \), et du résidu.
- **Calcul des erreurs** :
    - Norme du résidu relatif (arrière) : \(\text{resrel} = \frac{| b - A\hat{x} |}{|A| \cdot |\hat{x}|}\)
    - Erreur avant : \(\text{erreur avant} = \frac{| b - A\hat{x} |}{|b|}\)
    - Erreur relative : \(\text{erreur relative} = \frac{| x - \hat{x} |}{|x|}\)

Utilisation de `daxpy` pour calculer \( x - \hat{x} \) :
Pour calculer la différence \( x - \hat{x} \) et sa norme, vous pouvez utiliser `daxpy` :
- **Calcul de la différence \( x_{\text{exact}} - x_{\text{approx}} \)** :
```c
cblas_daxpy(n, -1.0, x_approx, 1, x_exact, 1);
norm_x_diff = norm2(n, x_exact);
```

Exercice 4 : 
1/
### Stockage GB en priorité colonne pour la matrice de Poisson 1D

La matrice de Poisson 1D est une matrice tridiagonale. Pour une matrice de taille \( n \times n \), elle a la forme suivante :

\[
A = \begin{pmatrix}
2 & -1 & 0 & \cdots & 0 \\
-1 & 2 & -1 & \cdots & 0 \\
0 & -1 & 2 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 2
\end{pmatrix}
\]

Pour stocker cette matrice en format bande général (GB) en priorité colonne, nous devons stocker les diagonales de la matrice dans un tableau 2D. Supposons que nous avons \( kl = 1 \) sous-diagonale et \( ku = 1 \) sur-diagonale. Le tableau de stockage aura \( kl + ku + 1 = 3 \) lignes et \( n \) colonnes.

Le tableau de stockage \( AB \) pour la matrice \( A \) sera :

\[
AB = \begin{pmatrix}
0 & -1 & -1 & \cdots & -1 & 0 \\
2 & 2 & 2 & \cdots & 2 & 2 \\
-1 & -1 & -1 & \cdots & -1 & 0
\end{pmatrix}
\]

Chaque colonne de \( AB \) correspond à une colonne de la matrice \( A \), et les lignes de \( AB \) contiennent les éléments des diagonales de \( A \) comme suit :

- La première ligne de \( AB \) contient les éléments de la sur-diagonale de \( A \) (décalée d'une position à droite).
- La deuxième ligne de \( AB \) contient les éléments de la diagonale principale de \( A \).
- La troisième ligne de \( AB \) contient les éléments de la sous-diagonale de \( A \) (décalée d'une position à gauche).

En résumé, le stockage GB en priorité colonne pour la matrice de Poisson 1D est représenté par le tableau \( AB \) ci-dessus.

2/
### Utilisation de la fonction BLAS `dgbmv` avec cette matrice

La fonction BLAS `dgbmv` effectue une opération de multiplication matrice-vecteur pour une matrice bande. L'opération est définie comme suit :

\[
y := \alpha \cdot A \cdot x + \beta \cdot y
\]

où :
- \( \alpha \) et \( \beta \) sont des scalaires.
- \( A \) est la matrice bande stockée en format GB.
- \( x \) est un vecteur de taille \( n \).
- \( y \) est un vecteur de taille \( m \).

3/
### Méthode de validation

Pour valider l'implémentation de la fonction `dgbmv` avec une matrice bande (GB) et un vecteur unitaire, nous utilisons la méthode suivante :

1. **Définir la matrice de Poisson 1D en format bande (GB)** :
   - Utiliser une matrice tridiagonale de taille \( n \times n \) avec des 2 sur la diagonale principale et des -1 sur les sous-diagonales et sur-diagonales.

2. **Définir un vecteur unitaire** :
   - Un vecteur de taille \( n \) avec tous les éléments égaux à 1.

3. **Appliquer la fonction `dgbmv`** :
   - Utiliser la fonction `dgbmv` pour effectuer la multiplication matrice-vecteur.

4. **Vérifier le résultat** :
   - Comparer le résultat obtenu avec le résultat attendu : des 1 aux extrémités et des 0 entre.

Exercice 5:

## Évaluation des performances et complexité des méthodes appelées

### Temps d'exécution

1. **Temps d'exécution mesuré** :
   - `dgbtrf` : 0.000122 secondes
   - `dgbtrs` : 0.000026 secondes
   - `dgbtrftridiag` : 0.000001 secondes
   - `dgbsv` : 0.000035 secondes

2. **Erreur relative** :
   - Pour toutes les méthodes (`dgbtrf`, `dgbtrs`, `dgbtrftridiag`, `dgbsv`), l'erreur relative est `0.000000e+00`, ce qui indique une solution très précise.

### Complexité des méthodes

1. **`dgbtrf`** :
   - Cette fonction effectue la factorisation LU d'une matrice bande. La complexité est généralement \(O(n^2)\) pour une matrice bande avec une largeur de bande fixe.

2. **`dgbtrs`** :
   - Cette fonction résout un système d'équations linéaires utilisant la factorisation LU obtenue par `dgbtrf`. La complexité est \(O(n^2)\) pour une matrice bande.

3. **`dgbtrftridiag`** :
   - Cette fonction est optimisée pour les matrices tridiagonales, et sa complexité est \(O(n)\), ce qui explique son temps d'exécution très rapide.

4. **`dgbsv`** :
   - Cette fonction combine la factorisation LU et la résolution du système d'équations. Sa complexité est également \(O(n^2)\) pour une matrice bande.

### Conclusion

- Les méthodes `dgbtrf` et `dgbtrs` sont efficaces pour les matrices bandes, mais leur complexité est \(O(n^2)\).
- La méthode `dgbtrftridiag` est particulièrement efficace pour les matrices tridiagonales avec une complexité \(O(n)\).
- La méthode `dgbsv` est une solution tout-en-un avec une complexité \(O(n^2)\).

En résumé, les performances des méthodes sont bonnes, avec des temps d'exécution très courts et des erreurs relatives nulles. La complexité des méthodes varie en fonction de la structure de la matrice, avec des méthodes optimisées pour les matrices tridiagonales offrant les meilleures performances.

Exercice 6 :
## Méthode de validation pour `dgbtrftridiag`

Pour valider la fonction `dgbtrftridiag`, nous avons créé une méthode de test qui suit les étapes suivantes :

1. **Définir une matrice tridiagonale connue** :
   - Nous avons créé une matrice tridiagonale avec des valeurs connues pour lesquelles nous pouvons facilement vérifier les résultats.

2. **Effectuer la factorisation LU** :
   - Nous avons appelé la fonction `dgbtrftridiag` sur cette matrice.

3. **Vérifier les résultats** :
   - Nous avons comparé les facteurs LU résultants avec les valeurs attendues.
