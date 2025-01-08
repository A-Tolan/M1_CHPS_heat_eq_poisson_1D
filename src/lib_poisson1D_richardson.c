/**********************************************/
/* lib_poisson1D.c                            */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"

// tableau des valeus propres
void eig_poisson1D(double* eigval, int *la){
  double h=(1.0/((double)(*la)+1.0));
  int n = *la;
  for (int k = 1; k <= n; ++k){
    double sin_theta = sin((double)k * M_PI * h / 2.0);
    eigval[k - 1] = 4.0 * sin_theta * sin_theta;
  }
}

double eigmax_poisson1D(int *la){
  int n = *la;
  double h = 1.0 / ((double)(n) + 1.0);
  double sin_theta = sin((double)n * M_PI * h / 2.0);
  return 4.0 * sin_theta * sin_theta;
}

double eigmin_poisson1D(int *la){
  int n = *la;
  double h = 1.0 / ((double)(n) + 1.0);
  double sin_theta = sin(M_PI * h / 2.0);
  return 4.0 * sin_theta * sin_theta;
}

double richardson_alpha_opt(int *la){
  double richardson_alpha_opt = 2 / (eigmax_poisson1D(la) + eigmin_poisson1D(la));
  return richardson_alpha_opt;
}

void richardson_alpha(double *AB, double *RHS, double *X, double *alpha_rich, int *lab, int *la, int *ku, int *kl, double *tol, int *maxit, double *resvec, int *nbite) {
    int k = 0;
    double *y = (double *)malloc(*la * sizeof(double));
    double norm_residual, res= *tol + 1, ny;

    printf("\nEntering richardson_alpha\n");

    // Calculer la norme de RHS
    ny = cblas_dnrm2(*la, RHS, 1);
    printf("\nNorm of RHS: %lf\n", ny);

    while (res > *tol && k < *maxit) {
        // Calculer le résidu r = b - A * x
        cblas_dgbmv(CblasColMajor, CblasNoTrans, *la, *la, *kl, *ku, -1.0, AB, *lab, X, 1, 1.0, y, 1);
        cblas_dcopy(*la, y, 1, y, 1);

        // Mettre à jour x^(k+1) = x^k + alpha * r
        cblas_daxpy(*la, *alpha_rich, y, 1, X, 1);

        // Calculer la norme du résidu
        norm_residual = cblas_dnrm2(*la, y, 1);
        res = norm_residual / ny;
        resvec[k] = res;

        printf("\nIteration %d, Residual: %lf\n", k, res);

        k++;
    }

    *nbite = k;
    free(y);
    printf("\nExiting richardson_alpha\n");
}

// forme general : x^(k+1) = (x^k)+(M^(-1))*(b-A(x^k))
void richardson_MB(double *AB, double *RHS, double *X, double *MB, int *lab, int *la, int *ku, int *kl, double *tol, int *maxit, double *resvec, int *nbite) {
    
int k = 0;
double *y = (double *)malloc(*la * sizeof(double));
double norm_residual, res = *tol + 1, ny;

printf("\nEntering richardson_MB\n");

// Calculer la norme de RHS
ny = cblas_dnrm2(*la, RHS, 1);
printf("\nNorm of RHS: %lf\n", ny);

while (res > *tol && k < *maxit) {
  // Calculer le résidu r = b - A * x
  cblas_dgbmv(CblasColMajor, CblasNoTrans, *la, *la, *kl, *ku, -1.0, AB, *lab, X, 1, 1.0, y, 1);
  cblas_dcopy(*la, y, 1, y, 1);

  // Résoudre M * z = r pour z
  LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', *la, *kl, *ku, 1, MB, *lab, NULL, y, *la);

  // Mettre à jour x^(k+1) = x^k + z
  cblas_daxpy(*la, 1.0, y, 1, X, 1);

  // Calculer la norme du résidu
  norm_residual = cblas_dnrm2(*la, y, 1);
  res = norm_residual / ny;
  resvec[k] = res;

  printf("\nIteration %d, Residual: %lf\n", k, res);

  k++;
}

*nbite = k;
free(y);
printf("\nExiting richardson_MB\n");
}
// M = D
void extract_MB_jacobi_tridiag(double *AB, double *MB, int *lab, int *la,int *ku, int*kl, int *kv){
  // Initialiser MB à zéro
  for (int i = 0; i < (*lab) * (*la); i++) {
    MB[i] = 0.0;
  }

  // Extraire la diagonale principale de AB et la placer dans MB
  for (int i = 0; i < *la; i++) {
    MB[*kv + i * (*lab)] = AB[*kv + i * (*lab)];
  }
}

//  M = D-E
void extract_MB_gauss_seidel_tridiag(double *AB, double *MB, int *lab, int *la,int *ku, int*kl, int *kv){
  // Initialiser MB à zéro
  for (int i = 0; i < (*lab) * (*la); i++) {
    MB[i] = 0.0;
  }

  // Extraire la diagonale principale et la sous-diagonale de AB et les placer dans MB
  for (int i = 0; i < *la; i++) {
    // Diagonale principale
    MB[*kv + i * (*lab)] = AB[*kv + i * (*lab)];
    // Sous-diagonale
    if (i > 0) {
      MB[(*kv - 1) + i * (*lab)] = -AB[(*kv - 1) + i * (*lab)];
    }
  }
}



