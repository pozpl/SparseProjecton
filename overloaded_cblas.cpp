#include "overloaded_cblas.hpp"

extern "C" {
    //#include <atlas/atlas_enum.h>
    //#include "clapack.h"
    //#include "cblas.h"
}


//------------------------------------------------------------------------------
/////////////////////////OVERLOADED CBLAS FUNCTIONS//////////////////////////////////

int clapack_potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda){
    return clapack_spotrf(Order, Uplo, N, A, lda);
}

int clapack_potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda){
    return clapack_dpotrf(Order, Uplo, N, A, lda);
}

//--------------------------------------------------------------------------------

int clapack_potri(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda){
     return  clapack_spotri(Order, Uplo, N, A, lda);
}

int clapack_potri(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda){
     return  clapack_dpotri(Order, Uplo, N, A, lda);
}

//--------------------------------------------------------------------------------

int clapack_gesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  double *A, const int lda, int *ipiv,
                  double *B, const int ldb){

    return clapack_dgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
}

int clapack_gesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  float *A, const int lda, int *ipiv,
                  float *B, const int ldb){
    return clapack_sgesv(Order, N, NRHS, A, lda, ipiv, B, ldb);
}

//--------------------------------------------------------------------------------



int clapack_getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   float *A, const int lda, int *ipiv){
    return clapack_sgetrf(Order, M, N, A, lda, ipiv);
}

int clapack_getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   double *A, const int lda, int *ipiv){
    return clapack_dgetrf(Order, M, N, A, lda, ipiv);
}

//--------------------------------------------------------------------------------



int clapack_getri(const enum CBLAS_ORDER Order, const int N, float *A,
                   const int lda, const int *ipiv){
    return clapack_sgetri(Order, N, A, lda, ipiv);
}

int clapack_getri(const enum CBLAS_ORDER Order, const int N, double *A,
                   const int lda, const int *ipiv){
    return clapack_dgetri(Order, N, A, lda, ipiv);
}
//--------------------------------------------------------------------------------


float cblas_dot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY){
    return  cblas_sdot(N,  X, incX, Y, incY);
}

double cblas_dot(const int N, const double *X, const int incX,
                  const double *Y, const int incY){
    return cblas_ddot(N, X, incX, Y, incY);

}

//--------------------------------------------------------------------------------

void cblas_copy(const int N, const float *X, const int incX,
                 float *Y, const int incY){
     cblas_scopy(N, X, incX, Y, incY);

}

void cblas_copy(const int N, const double *X, const int incX,
                 double *Y, const int incY){
    cblas_dcopy(N, X, incX, Y, incY);
}

//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------

void cblas_scal(const int N, const float alpha, float *X, const int incX){
     cblas_sscal(N, alpha, X, incX);
}

void cblas_scal(const int N, const double alpha, double *X, const int incX){
    cblas_dscal(N, alpha, X, incX);
}

//--------------------------------------------------------------------------------

void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc){
    cblas_sgemm(Order,  TransA, TransB,  M,  N, K,  alpha, A, lda, B, ldb, beta, C, ldc);
}

void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc){
    cblas_dgemm(Order,  TransA, TransB,  M,  N, K,  alpha, A, lda, B,  ldb, beta, C, ldc);
}
//--------------------------------------------------------------------------------
void cblas_axpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY){
    cblas_saxpy( N,  alpha, X, incX, Y,  incY);
}

void cblas_axpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY){
    cblas_daxpy(N, alpha, X, incX, Y, incY);

}
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void cblas_gemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY){
    cblas_sgemv(Order,  TransA,  M,  N, alpha,  A, lda, X,  incX,  beta, Y,  incY);
}

void cblas_gemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY){
    cblas_dgemv(Order,  TransA,  M,  N, alpha,  A, lda, X,  incX,  beta, Y,  incY);
}

//--------------------------------------------------------------------------------


