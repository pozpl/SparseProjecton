/*
 * File:   min_norm_sparse.hpp
 * Author: pozpl
 *
 * Created on 25 Май 2010 г., 12:54
 */

#ifndef _MIN_NORM_CONE_SPARSE_HPP
#define	_MIN_NORM_CONE_SPARSE_HPP
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
    #include <ldl.h>
}


#include "sparse_operations_ext_cpu.hpp"
#include "sparse_types.hpp"
//#include "overloaded_cblas.hpp"

//#include "lup_cpu.hpp"

//template <typename int,typename double>
void getMinNormElemOutRepr(csc_matrix &inputSet,  double * minNormVector, double tollerance, int* kvec_in, int* basisVecInx_in, int &baselen_in, int &numEqCon);
//template <typename int,typename double>
void findOptZInBasisCPU(csc_matrix& inSet, int &baselen, int vectorDim, int basisInc,
        csr_matrix &basis, csr_matrix &basis_t, int *basisVecInx, int *kvec, csr_matrix& grammMatr, double *invGrammMatr,
        double *mu, double *mu_old, double *z, double eps, int numEqCon, 
        ldl_matrix &grammPartFactor);
void getMinVCosCPU(double* z, csc_matrix &inSet,  double& minVcos, int &minVecId, double epsilon, int numEqCon, double* l_norms);
void evalMuVectorCPU(int baselen, int vectorDim, int basisInc, csr_matrix basis, csr_matrix basis_t, double *grammMatr, double *invGrammMatr, double *mu, double eps);
double getMinVectorElCPU(double *vector, int vectorDim, int &minVecIdx, int numEqCon);
double getMaxVectorElCPU(double *vector, int vectorDim, int &maxVecIdx, int numEqCon);
double attractDotCoeffCPU(csr_matrix &basis, double *mu_old, double *mu, int &delBasElIndx, int baselen, int numEqCon);
void invGrammAdd_CPU(int rowsInGramOld, int vectorDim, csr_matrix &basis, csr_matrix &basis_t, double *invGrammOldm, double *invGramm);
void inverseMatrixCPU(double* grammMatrix, double* invMatrix, int mtrxRows);
void invGrammDel_CPU(int rowsIG, int collsIG, double *invGrammOld, double *invGramm, int vecToDelId);
void sumMtrxByStr(double *srcMtrx, double *sumVector, int srcMtrxRows, int srcMtrxCols);
void sumVecElmts(double *vector, double *summ, int vectorDim);
void multVectorToValue(double *vector, double ratio, int vectorDim);
void putSubMatrix_CPU(int mtrxRows, int mtrxCols, double* Matrix, int subMtrxX, int subMtrxY, int subMtrxRows, int subMtrxCols, double *subMtrx);
void getMuViaSystemSolve(const csr_matrix& basis, const csr_matrix& basis_t, double *mu, double eps);
void biSoprGradient(const csr_matrix& basis, const csr_matrix& basis_t, double *b, double *x,double eps, double *x0);
coo_matrix getShiftCoo(int vectorDim, double shift_val);
void biSoprGradientStoredMatrix(const csr_matrix& basis, const csr_matrix& basis_t, const csr_matrix& gramMtxNotFull, double *b, double *x, double eps, double *x0);
void evalMuVectorCPUwithStoredMatrix(int basisInc, csr_matrix basis, csr_matrix basis_t,
        csr_matrix& grammMatrParted,ldl_matrix &grammPartFactor, double *mu);
//#include "min_norm_sparse.cpp"

void evalCholmodFactor(csr_matrix gMP, ldl_matrix &grammPartFactor);
void addRowCholmodFactor(ldl_matrix &grammPartFactor, csr_matrix grammMatrParted);
void change_sign_in_ldl(ldl_matrix &ldlm, int num_col);

void updateGrammMtxPart_impr(csr_matrix& basis, csr_matrix& gramm_parted);
void addColToCholmodFactor(ldl_matrix &grammPartFactor, csr_matrix grammMatrParted);
void evalCholmodFactorTrans(csr_matrix gMP, ldl_matrix &grammPartFactorTrans);
void change_sign_in_ldl_t(ldl_matrix &ldlm, int num_col);

void evalGrammMtxPart(csr_matrix& basis_t, csr_matrix& gramm_parted);
void updateGrammMtxPartTriagForm_impr(csr_matrix& basis,csr_matrix& gramm_parted);
void evalGrammMtxPartTriangForm(csr_matrix& basis_t, csr_matrix& gramm_parted);
void downgradeGrammMtxPart(csr_matrix& gramm_parted, int col_to_del_idx);
ldl_matrix recompute_l33_d33_for_ldl_col_del(ldl_matrix &ldl33_old, double* l32, double d22);
#endif	/* _MIN_NORM_CONE_SPARSE_HPP */

