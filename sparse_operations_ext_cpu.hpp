/* 
 * File:   sparce_operations_ext.hpp
 * Author: pozpl
 *
 * Created on 24 Май 2010 г., 14:07
 */

#ifndef _SPARCE_OPERATIONS_EXT_HPP
#define	_SPARCE_OPERATIONS_EXT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include<math.h>
#include <ctype.h>
#include <unistd.h>
#include <getopt.h>

#include <fstream>
#include <iostream>
#include <istream>
#include <vector>
#include <algorithm>

extern "C" {
//#include <atlas/atlas_enum.h>
//#include "atlas/clapack.h"
    //#include "min_norm_gpu.cuh"
    #include "cblas.h"
        //#include <cholmod.h>
}

#include "sparse_types.hpp"
//#include "overloaded_cblas.hpp"

//!#include <cublas.h>
//#include <cutil.h>
//#include <cuda.h>
//SpMV includes
//#include "spmv_nvidia/sparse_io.h"
//#include "spmv_nvidia/cmdline.h"
//#include "spmv_nvidia/tests.h"
//#include "spmv_nvidia/gallery.h"







void print_csr_matrix(const csr_matrix& csr);
void __get_dense_column(const csr_matrix& csr_t, int col_number, double *dense_column);
void get_dense_column(const csr_matrix& csr, int col_number, double *dense_column);
void printMatrixCPU(int str_num, int coll_num, double* matrix);
int __estimate_max_nonzeros(const csr_matrix& csr, int max_colls);
void print_coo_matrix(const coo_matrix& coo);
coo_matrix __get_coo_column(const csr_matrix& csr_t, int col_number);
csr_matrix get_empty_csr_for_col_add(int est_rows_num, int est_nonzer_num);
csr_matrix get_empty_csr_for_row_add(int est_cols_num, int est_nonzer_num);



void __mm_csr_serial_host(const csr_matrix& A,
        const csr_matrix& B_t,
        double *C);
void del_col_from_csr_mtx(csr_matrix& csr, int col_to_del);
void add_row_to_csr(csr_matrix &csr,
        coo_matrix &coo_row);
void add_col_to_csr_mtx(csr_matrix& csr,
        coo_matrix& coo_column);
void transpose_coo_mv(coo_matrix &coo);
void sum_csr_duplicates(const int num_rows,
const int num_cols,int * Ap,int * Aj,double * Ax);
csr_matrix csr_transpose(const csr_matrix& csr);
void csr_transpose(const int * Ap,const int * Aj,const double * Ax,const int num_rows,const int num_cols,int * Bp,int * Bj,double * Bx);
csr_matrix csr_transpose_fix(const csr_matrix& csr);
csr_matrix coo2csr(const coo_matrix& coo);
int estimate_max_nonzeros(const csc_matrix& csr, int max_colls);
coo_matrix get_coo_column(const csr_matrix& csr, int col_number);
coo_matrix get_coo_column_from_csc(const csc_matrix& csc, int col_number);
void spmv_csr_serial_host(const csr_matrix& csr, const double * x, double * y);
void spmv_csr_t_serial_host(const csr_matrix& csr, const double * x,  double * y);
void del_row_from_csr_mtx(csr_matrix& csr, int row_to_del_idx);
void mm_csr_serial_host(const csr_matrix& A,  const csr_matrix& B,  double *C);
csr_matrix convCsrToDense(const csr_matrix& csr, double* denseMtx);
csr_matrix scaleColInCsrMtx(const csr_matrix& csr, double scaleParam, int colIdx);
void scaleRowInCsrMtx(const csr_matrix& csr, double scaleParam, int rowIdx);
csr_matrix scaleColInCsrMtx_AndAdd2Shift(const csr_matrix& csr, double scaleParam, int colIdx, double shift);
void scaleRowInCsrMtx_AndAdd2Shift(const csr_matrix& csr, double scaleParam, int rowIdx, double shift);
void scale_col_in_csc_mtx(const csc_matrix& csc, double scaleParam, int col_idx);
void get_dense_row(const csr_matrix& csr, int row_number, double *dense_row);
void mulGrammToX(const csr_matrix& basis_t, const csr_matrix& basis, double *x, double *p );
void evalRowOfGrammMtx(csr_matrix basis_t, int row_idx, double* gramm_row);
coo_matrix convDenseToCooNaiv(double* denseMtx, int num_rows, int num_cols);
coo_matrix convDenseColToCooCol(double* denseCol, int num_els);
coo_matrix convDenseRowToCooRow(double* denseRow, int num_els);

void print_ldl_matrix(const ldl_matrix& ldlm);
void add_row_to_ldl(ldl_matrix& ldl,  coo_matrix& coo_row);
void csr_transpose_mv(const csr_matrix& csr, csr_matrix& csr_t);
    
void A_dot_x_csc_serial_host(const csc_matrix& csc, const double * x, double * y);

void print_csr_matrix_to_file(const csr_matrix& csr, const char *file_name, const char *matrix_name);

void spmv_scr_t_coo_serial_host(const csr_matrix& csr_t, const coo_matrix x, double * y);

void ldl_ltsolve_t(int n, double X [ ], int Lp [ ], int Li [ ], double Lx [ ]);
void ldl_dsolve_t(int n, double X [ ], double D [ ]);
void ldl_lsolve_t(int n, double X [ ], int Lp [ ], int Li [ ], double Lx [ ]);
ldl_matrix ldl_transpose(const ldl_matrix& ldl, ldl_matrix& ldl_t);
void add_col_to_ldl(ldl_matrix &ldl,  coo_matrix &coo_col);
double* eval_csc_cols_norms(csc_matrix csc);
void print_csc_matrix(const csc_matrix& csc);

void get_dense_row_from_triangular_gramm(const csr_matrix& csr, int row_number, double *dense_row);
void get_ldl_dense_column_from_l_low(const ldl_matrix& ldl, int col_number, double *dense_column);
void add_last_col_to_ldl_l_low(ldl_matrix& ldl, double *dense_column, int col_id, int dens_col_dim);
void get_ldl_dense_row_from_l_upper(const ldl_matrix& ldl, int row_index, double*dense_row);
ldl_matrix get_ldl33_up_from_ldl_l_upper(const ldl_matrix& ldl, int row_start_idx);
#endif	/* _SPARCE_OPERATIONS_EXT_HPP */

