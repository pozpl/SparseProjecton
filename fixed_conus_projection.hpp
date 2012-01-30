/* 
 * File:   fixed_conus_projection.hpp
 * Author: pozpl
 *
 * Created on 25 Май 2010 г., 12:53
 */

#ifndef _FIXED_CONUS_PROJECTION_HPP
#define	_FIXED_CONUS_PROJECTION_HPP
#include "sparse_operations_ext_cpu.hpp"
//#include "overloaded_cblas.hpp"
#include "sparse_types.hpp"
//#include "over"
#include "min_norm_sparse.hpp"



void incValsInLstMtxRow(const csr_matrix& csr, int rowIdx, double valToAdd);
void doubleCsrMatrix(csr_matrix& csr);
void csr_get_bibi(const csr_matrix& csr, double* bibi);
void scaleValsInAllColumns(const csr_matrix& csr, double *s);
void buildSmplxFromCon(const csr_matrix& csr, double *s, double valOfShiftLstEl);
void scaleValsInCol(const csr_matrix& csr, int colIdx, double s);
coo_matrix getShiftCoo(int vectorDim, double shift_val);
void projOnFixedSimplex(csr_matrix &inputSet, double *minNormVector, int* kvec, int* basisVecInx, int &baselen, double tollerance, int numEqCon);
void csr_get_bibj(const csr_matrix& A, const csr_matrix& B);
//#include "fixed_conus_projection.cpp"

#endif	/* _FIXED_CONUS_PROJECTION_HPP */

