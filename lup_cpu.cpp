#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
extern "C" {
#include <atlas/atlas_enum.h>
#include "clapack.h"
#include "cblas.h"
}

#include "sparse_operations_ext_cpu.hpp"
#include "lup_cpu.hpp"

void fitIdentityPivot(int * pivMatr, int col_num){
    int i;
    for(i = 0; i < col_num; i++){
        pivMatr[i] = i;
    }
}

void swapPivotRows(int *pivotMatrix,int row1, int row2){
    int swapEl = pivotMatrix[row1];
    pivotMatrix[row1] = pivotMatrix[row2];
    pivotMatrix[row2] = swapEl;
}

template <typename T> void swapMatrixRows(T *Matrix,int row1, int row2, int row_num, int col_num){
    T * swapRow = (T*) malloc(col_num * sizeof(T));
    memmove(swapRow, &Matrix[row1 * col_num], col_num * sizeof(T));
    memmove(&Matrix[row1 * col_num], &Matrix[row2 * col_num], col_num * sizeof(T));
    memmove(&Matrix[row2 * col_num], swapRow ,col_num * sizeof(T));
    free(swapRow);
}


template <typename T> void forwardSubstitution(T *A, T* b, int row_num, int col_num){
    b[0] = b[0];
    int row;
    for(row = 1; row < row_num; row++){
        int col;
        T LToB_Store = 0;
        for(col = 0; col <= row -1 ; col++){
            LToB_Store += A[row * col_num + col] * b[col];
            //printf("A[i,j] * b[j] %f %f %f \n",A[row * col_num + col], b[col] , LToB_Store);
        }
        //printf("b[ %i ] -  L*B %f %f \n",row ,b[row] ,LToB_Store);
        b[row] = b[row] - LToB_Store;
        //printf("B[%i] %f \n",row, b[row]);
    }
}

template <typename T> void backwardSubstitution(T *A, T *b, int row_num, int col_num){
    //b[row_num]= b[row_num] / A[row_num * col_num + row_num];
    //printf(" B[%i] = %f ",row_num,b[row_num]);
    int row;
    for(row = row_num - 1; row >= 0; row--){
        int col;
        T UToB_Store = 0;
        for(col = row + 1; col < col_num; col++){
            UToB_Store += A[row * col_num + col] * b[col];
        }
        b[row] = (b[row] - UToB_Store) / A[row * col_num + row];
        //printf(" B[%i] = %f ",row,b[row]);
    }
}


/**LUP разлодение матрицы
 *A - входящая матрица,
 *С - LUP разложение в компактной форме
 *P - вектор перестановок
 */
template <typename T> void LUP(const T* A, T *C, int * P, int rows_num, int col_num) {
    //n - размерность исходной матрицы
    const int n = rows_num;

    memcpy(C, A, rows_num * col_num * sizeof(T));

    //загружаем в матрицу P единичную матрицу
    fitIdentityPivot(P, col_num);//P = IdentityMatrix();
    int i;
    for( i = 0; i < n; i++ ) {
        //поиск опорного элемента
        double pivotValue = 0;
        int pivot = -1;
        int row;
        for( row = i; row < n; row++ ) {
            if( fabs(C[ row * col_num + i]) > pivotValue ) {
            //if( fabs(C[ row ][ i ]) > pivotValue ) {
                pivotValue = fabs(C[ row * col_num + i ]);
                pivot = row;
            }
        }
        if( pivotValue == 0 ) {
            printf("\nМатрица вырождена\n");
            return;
        }

        //меняем местами i-ю строку и строку с опорным элементом
        swapPivotRows(P, pivot, i);    //P.SwapRows(pivot, i);
        swapMatrixRows(C, pivot, i, rows_num, col_num);   //C.SwapRows(pivot, i);
        int j;
        for( j = i+1; j < n; j++ ) {
	    //printMatrixCPU(3,3,C);
            C[ j * col_num + i ] /= C[ i*col_num + i ];
	    //printMatrixCPU(3,3,C);
            int k;
            for(k = i+1; k < n; k++ ){
	      //printMatrixCPU(3,3,C);
	      //std::cout<< "C["<< j * col_num + i << "] = " << C[ j * col_num + i ] << " \n";
	      //std::cout<< "C[" << i * col_num + k <<"] = " << C[ i * col_num + k ] << " \n";
	      //std::cout<< "C[" << j * col_num + k <<" ] = " << C[ j * col_num + k ] << " \n";
	      C[ j*col_num + k ] -= C[ j * col_num + i ] * C[ i * col_num +  k ];
	      //std::cout<< "C[" << j * col_num + k <<" ] = " << C[ j * col_num + k ] << " \n";
	    }

        }
    }
}//теперь матрица C = L + U - E







template <typename T> void inverseMatrixCPURaw(T* matrix,T* invMatrix, int cols, int rows){

    T *C = (T *)malloc(rows * cols * sizeof(T));
    int * P  = (int *)malloc( cols * sizeof(int));
    int row = 0;

    for (row = 0; row < rows; row++) {

        int col;
        for (col = 0; col < cols; col++) {
            if (row == col) {
                invMatrix[row * cols + col] = 1.0f;
            } else {
                invMatrix[row * cols + col] = 0.0f;
            }
        }
    }
    //делаем LUP казложение
    LUP(matrix, C, P, rows, cols);
    int col;
    for(col = 0; col < cols; col++){
        forwardSubstitution(C, &invMatrix[col * rows], rows, cols);
        backwardSubstitution(C, &invMatrix[col * rows], rows, cols);
    }
}


