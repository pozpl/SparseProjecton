/* 
 * File:   min_norm_sparse.hpp
 * Author: pozpl
 *
 * Created on 25 Май 2010 г., 12:54
 */

#ifndef _MIN_NORM_SPARSE_HPP
#define	_MIN_NORM_SPARSE_HPP
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <ctype.h>
#include <unistd.h>
#include <getopt.h>

#include <math.h>


#include <fstream>
#include <iostream>
#include <istream>
#include <vector>
#include <algorithm>
#include <string>
using namespace std;

extern "C" {
    #include <atlas/atlas_enum.h>
    #include "clapack.h"
    #include "cblas.h"
}


#include "sparse_operations_ext_cpu.hpp"
#include "sparse_types.hpp"
//#include "overloaded_cblas.hpp"

//#include "lup_cpu.hpp"

//template <typename int,typename double>
void getMinNormElemOutRepr(csr_matrix &inputSet,  double * minNormVector, int inSetDim, int vectorDim, double tollerance, int* kvec_in, int* basisVecInx_in, int &baselen_in, int &numEqCon);
//template <typename int,typename double>
void findOptZInBasisCPU(csr_matrix& inSet, int &baselen, int vectorDim, int basisInc,
        csr_matrix &basis, csr_matrix &basis_t, int *basisVecInx, int *kvec, double *grammMatr, double *invGrammMatr,
        double *mu, double *mu_old, double *z, double eps, int numEqCon);
void getMinVCosCPU(double* z, csr_matrix &inSet, int inSetDim, int vectorDim, double* minVcos, int *minVecId, double epsilon, int numEqCon);
void evalMuVectorCPU(int baselen, int vectorDim, int basisInc, csr_matrix basis, csr_matrix basis_t, double *grammMatr, double *invGrammMatr, double *mu, bool &isFullBasis, int vecToDelId, double eps);
double getMinVectorElCPU(double *vector, int vectorDim, int &minVecIdx, int numEqCon);
double attractDotCoeffCPU(double *mu_old, double *mu, int *delBasElIndx, int baselen, int numEqCon);
void invGrammAdd_CPU(int rowsInGramOld, int vectorDim, csr_matrix &basis, csr_matrix &basis_t, double *invGrammOldm, double *invGramm);
void inverseMatrixCPU(double* grammMatrix, double* invMatrix, int mtrxRows);
void invGrammDel_CPU(int rowsIG, int collsIG, double *invGrammOld, double *invGramm, int vecToDelId);
void sumMtrxByStr(double *srcMtrx, double *sumVector, int srcMtrxRows, int srcMtrxCols);
void sumVecElmts(double *vector, double *summ, int vectorDim);
void multVectorToValue(double *vector, double ratio, int vectorDim);
void putSubMatrix_CPU(int mtrxRows, int mtrxCols, double* Matrix, int subMtrxX, int subMtrxY, int subMtrxRows, int subMtrxCols, double *subMtrx);
void getMuViaSystemSolve(const csr_matrix& basis, const csr_matrix& basis_t, double *mu, double eps);
void biSoprGradient(const csr_matrix& basis, const csr_matrix& basis_t, double *b, double *x,double eps, double *x0);

//#include "min_norm_sparse.cpp"

#endif	/* _MIN_NORM_SPARSE_HPP */

