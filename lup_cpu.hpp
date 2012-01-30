/* 
 * File:   lup_cpu.hpp
 * Author: pozpl
 *
 * Created on 26 Май 2010 г., 12:46
 */

#ifndef _LUP_CPU_HPP
#define	_LUP_CPU_HPP

template <typename T> void LUP(const T* A, T *C, int * P, int rows_num, int col_num);
void swapPivotRows(int *pivotMatrix,int row1, int row2);
template <typename T> void swapMatrixRows(T *Matrix,int row1, int row2, int row_num, int col_num);
void fitIdentityPivot(int * pivMatr, int col_num);
template <typename T> void forwardSubstitution(T *A, T* b, int row_num, int col_num);
template <typename T> void backwardSubstitution(T *A, T *b, int row_num, int col_num);
template <typename T> void inverseMatrixCPURaw(T* matrix,T* invMatrix, int cols, int rows);


//#include "lup_cpu.cpp"
#endif	/* _LUP_CPU_HPP */

