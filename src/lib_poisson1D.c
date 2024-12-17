/**********************************************/
/* lib_poisson1D.c                            */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"

void set_GB_operator_colMajor_poisson1D(double* AB, int *lab, int *la, int *kv){
  //AB = (double *)malloc(lab * la * sizeof(double));
  for (int i = 0; i < *la; i++ ){
    if (*kv){
      AB[i*4] = 0;
      AB[i*4+1] = -1;
      AB[i*4+2] = 2;
      AB[i*4+3] = -1;
    }
    else {
      AB[i*3] = -1;
      AB[i*3+1] = 2;
      AB[i*3+2] = -1;
    }
  }
  AB[*kv] = 0;
  AB[*la * (*lab - 1 + *kv) - 1] = 0;
}

void set_GB_operator_colMajor_poisson1D_Id(double* AB, int *lab, int *la, int *kv){
  for (int i = 0; i < *la; i++ ){
    if (*kv){
      AB[i*4] = 0;
      AB[i*4+1] = 0;
      AB[i*4+2] = 1;
      AB[i*4+3] = 0;
    }
    else {
      AB[i*3] = 0;
      AB[i*3+1] = 1;
      AB[i*3+2] = 0;
    }
  }
}

//vecteur initiale
void set_dense_RHS_DBC_1D(double* RHS, int* la, double* BC0, double* BC1){
  double h = 1.0 / (*la + 1);
  for (int i = 0; i < *la; i++) {
    RHS[i] = 0.0;
  }
  RHS[0] = *BC0;
  RHS[*la - 1] = *BC1;
}

//T(x)=T0 + x(T1-T0)
void set_analytical_solution_DBC_1D(double* EX_SOL, double* X, int* la, double* BC0, double* BC1){
  double T0 = *BC0;
  double T1 = *BC1;
  for (int i = 0; i < *la; i++) {
    EX_SOL[i] = T0 + X[i] * (T1 - T0);
  }
}
//construire vect x pour position de chaque point de température
void set_grid_points_1D(double* x, int* la){
  double h = 1.0 / (*la + 1);
  for (int i = 0; i < *la; i++) {
    x[i] = (i + 1) * h;
  }
}

double relative_forward_error(double* x, double* y, int* la){
  double norm_num = 0.0;
  double norm_den = 0.0;

  for (int i = 0; i < *la; i++) {
    norm_num += (x[i] - y[i]) * (x[i] - y[i]);
    norm_den += x[i] * x[i];
  }

  norm_num = sqrt(norm_num);
  norm_den = sqrt(norm_den);

  return norm_num / norm_den;
}

int indexABCol(int i, int j, int *lab){
  return i + j * (*lab);
}

int dgbtrftridiag(int *la, int *n, int *kl, int *ku, double *AB, int *lab, int *ipiv, int *info) {
  int i;
  double factor;

  *info = 0;

  for (i = 0; i < *n - 1; i++) {
    if (AB[2 + i * (*lab)] == 0.0) {
      *info = i + 1;
      return *info;
    }

    factor = AB[3 + i * (*lab)] / AB[2 + i * (*lab)];
    AB[3 + i * (*lab)] = factor;
    AB[2 + (i + 1) * (*lab)] -= factor * AB[1 + (i + 1) * (*lab)];
  }

  if (AB[2 + (*n - 1) * (*lab)] == 0.0) {
    *info = *n;
  }

  return *info;
}