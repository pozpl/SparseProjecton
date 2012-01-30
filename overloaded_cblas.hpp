/* 
 * File:   overloaded_cblas.hpp
 * Author: pozpl
 *
 * Created on 25 Май 2010 г., 12:51
 */

#ifndef _OVERLOADED_CBLAS_HPP
#define	_OVERLOADED_CBLAS_HPP

extern "C" {
#include <atlas/atlas_enum.h>
#include "clapack.h"
#include "cblas.h"
}

//------------------------------------------------------------------------------
/////////////////////////OVERLOADED CBLAS FUNCTIONS//////////////////////////////////

int clapack_potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda);

int clapack_potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);

//--------------------------------------------------------------------------------

int clapack_potri(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda);

int clapack_potri(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);

//--------------------------------------------------------------------------------

float cblas_dot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY);

double cblas_dot(const int N, const double *X, const int incX,
                  const double *Y, const int incY);

//--------------------------------------------------------------------------------

void cblas_copy(const int N, const float *X, const int incX,
                 float *Y, const int incY);

void cblas_copy(const int N, const double *X, const int incX,
                 double *Y, const int incY);

//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------

void cblas_scal(const int N, const float alpha, float *X, const int incX);

void cblas_scal(const int N, const double alpha, double *X, const int incX);
//--------------------------------------------------------------------------------

void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);
//--------------------------------------------------------------------------------
void cblas_axpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY);
void cblas_axpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY);
//--------------------------------------------------------------------------------

//--------------------------------------------------------------------------------
void cblas_gemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY);

void cblas_gemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);
//--------------------------------------------------------------------------------

int clapack_gesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  double *A, const int lda, int *ipiv,
                  double *B, const int ldb);
int clapack_gesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
                  float *A, const int lda, int *ipiv,
                  float *B, const int ldb);
int clapack_getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   float *A, const int lda, int *ipiv);

int clapack_getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   double *A, const int lda, int *ipiv);
int clapack_getri(const enum CBLAS_ORDER Order, const int N, float *A,
                   const int lda, const int *ipiv);

int clapack_getri(const enum CBLAS_ORDER Order, const int N, double *A,
                   const int lda, const int *ipiv);


//#include "overloaded_cblas.cpp"

#endif	/* _OVERLOADED_CBLAS_HPP */

