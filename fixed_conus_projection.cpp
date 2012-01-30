
#include "fixed_conus_projection.hpp"

/*
 * This funcion cut conus and get simplex fom it accoring to a * (a + t * b) >=0
 */
//template <typename int, typename double>

void projOnFixedSimplex(csr_matrix &inputSet, double *minNormVector,
        int* kvec, int* basisVecInx, int &baselen, double tollerance, int numEqCon) {
    double sigma = 0;
    int rowIdx = inputSet.num_rows - 1;
    int activColIdx = 0;


    double *scaleCoefsForColls = new_host_darray(inputSet.num_cols + 1);
    for (int i = 0; i < inputSet.num_cols + 1; i++) {
        scaleCoefsForColls[i] = 0.0;
    }


    double *bi2bi = new_host_darray(inputSet.num_cols);
    for (int i = 0; i < inputSet.num_cols; i++) {
        bi2bi[i] = 0.0;
    }

    //Сдвигаем множество на shift
    incValsInLstMtxRow(inputSet, (inputSet.num_rows - 1), (double) - 1.0);

    csr_get_bibi(inputSet, bi2bi);
    for (int j = inputSet.Ap[rowIdx]; j < inputSet.Ap[rowIdx + 1]; j++) {
        activColIdx = inputSet.Aj[j];
        //std::cout << "Bi'*Bi" << "[ " << activColIdx << "] = " << bi2bi[activColIdx] << "\n";
        //get_dense_column(inputSet, activColIdx, bi);
        sigma = inputSet.Ax[j] / bi2bi[activColIdx];
        if (sigma < 0) {
            sigma = 0;
        }
        scaleCoefsForColls[activColIdx] = sigma  + 1;

    }
    //Сдвигаем множество на -shift
    incValsInLstMtxRow(inputSet, (inputSet.num_rows - 1), (double) + 1);

    //Выбираем наибольший коэффициент
    double maxScaleCoef = 0;
    for (int i = 0; i < inputSet.num_cols + 1; i++) {
        if (maxScaleCoef < scaleCoefsForColls[i]) {
            maxScaleCoef = scaleCoefsForColls[i];
        }
    }
    /*for (int i = 0; i < inputSet.num_cols + 1; i++) {
        scaleCoefsForColls[i] = 1000;

    }*/

    std::cout << "Max Scale Coeff = " << maxScaleCoef << "\n";
    /////////////////////////////////////////////////////////////////////////////////////////////
    //printMatrixCPU(inputSet.num_cols, 1, scaleCoefsForColls);
    //std::cout << "Input set before building simplex\n";
    //print_csr_matrix(inputSet);
    std::cout << "===================== \n";
    buildSmplxFromCon(inputSet, scaleCoefsForColls, -1.0);


    //Добавляем сдвиг в конец [..., shift]
    //std::cout << "Try to add shift to the input set\n";
    coo_matrix shift_col = getShiftCoo(inputSet.num_rows, -1.0);
    add_col_to_csr_mtx(inputSet, shift_col);

    std::cout << "rows " << inputSet.num_rows << " cols " << inputSet.num_cols << " nonzerrows " << inputSet.num_nonzeros << " \n";

    //print_csr_matrix(inputSet);
    //csr_matrix inputSet_t = csr_transpose(inputSet);
    //csr_get_bibj(inputSet_t, inputSet);

    //print_csr_matrix(inputSet);
    std::cout << "Find el IN Simplex\n";
    getMinNormElemOutRepr(inputSet, minNormVector, inputSet.num_cols, inputSet.num_rows, tollerance, kvec, basisVecInx, baselen, numEqCon);

    std::cout << " last ell " << minNormVector[inputSet.num_rows - 1] << " \n";
    printf("last el %e \n", minNormVector[inputSet.num_rows - 1]);
    cblas_dscal(inputSet.num_rows, 1.0 / minNormVector[inputSet.num_rows - 1], minNormVector, 1);
    std::cout << "End Find El in Simplex\n";

    double z_summ = 0.0;
    int nonzer_summ = 0;
    for (int i = 0; i < inputSet.num_rows - 1; i++) {
        z_summ += minNormVector[i];
        if (minNormVector[i] != 0.0) {
            nonzer_summ++;
        }
    }

    std::cout << "Summ of elements " << z_summ << " \n";
    std::cout << "Count of nonzerros " << nonzer_summ << " \n";
    //printMatrixCPU(inputSet.num_rows, 1, minNormVector);
    
    delete_host_matrix(shift_col);
    free(scaleCoefsForColls);
    free(bi2bi);
    //free(bi);

}

//template <typename int, typename double>

coo_matrix getShiftCoo(int vectorDim, double shift_val) {
    coo_matrix coo_col;
    coo_col.num_nonzeros = 1;
    coo_col.num_cols = 1;
    coo_col.num_rows = 1;
    coo_col.I = new_host_iarray(coo_col.num_rows);
    coo_col.J = new_host_iarray(coo_col.num_rows);
    coo_col.V = new_host_darray(coo_col.num_rows);

    coo_col.I[0] = vectorDim - 1;
    coo_col.J[0] = (int) 0;
    coo_col.V[0] = shift_val;

    return coo_col;
}

/**Scale elements all elemant on given matrix column.
 * We assume that we don't need to add elements in csr matrix. #0 * any_num = 0
 * csr - matrix that contain elements
 * colIdx - index of col
 * s - parameter of multiplication.
 *
 */
//template <typename int, typename double>

void scaleValsInCol(const csr_matrix& csr, int colIdx, double s) {
    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            if (csr.Aj[j] == colIdx) {
                csr.Ax[j] *= s;
            }
        }
    }
}


/**Эта функция строит симплекс из конуса, путем применения формулы
 * b_new = a + t * b_old, где a это сдвиг
 * @param csr матрица столбцы которой это коэфециенты при конусе
 * @param s коэффециенты растяжения
 * @param valOfShiftLstEl значение последнего элемента вектора сдвига
 */
//template <typename int, typename double>

void buildSmplxFromCon(const csr_matrix& csr, double *s, double valOfShiftLstEl) {
    for (int i = 0; i < csr.num_rows - 1; i++) {
        for (int j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            //if (csr.Aj[j] == colIdx) {
            csr.Ax[j] *= s[csr.Aj[j]];
            //}            
        }
    }
    //Так как все элементы сдвига равны 0, то только в последней строке надо добавлять
    // shidft last element т.к. мы строим b_new = a + t * b_old, где a это сдвиг
    for (int j = csr.Ap[csr.num_rows - 1]; j < csr.Ap[csr.num_rows]; j++) {
        csr.Ax[j] = valOfShiftLstEl + s[csr.Aj[j]] * csr.Ax[j];
    }
}

/**Scale elements in all columns.
 * We assume that we don't need to add elements in csr matrix. #0 * any_num = 0
 * csr - matrix that contain elements
 * s - vector  parameters of multiplication.
 *
 */
//template <typename int, typename double>

void scaleValsInAllColumns(const csr_matrix& csr, double *s) {
    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            //if (csr.Aj[j] == colIdx) {
            csr.Ax[j] *= s[csr.Aj[j]];
            //}
        }
    }
}

/**Increas all elemant on Last matrix rows. We assume that this row is full of elements, so no zerrow elements present in it and we
 * dont need to add elements in csr matrix.
 * csr - matrix that contain elements
 * rowIdx - index of last row
 * valToAdd - value to add to presented value.
 *
 */
//template <typename int, typename double>

void incValsInLstMtxRow(const csr_matrix& csr, int rowIdx, double valToAdd) {
    for (int j = csr.Ap[rowIdx]; j < csr.Ap[rowIdx + 1]; j++) {
        //std::cout << "before " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
        csr.Ax[j] += valToAdd;
        //std::cout << "after " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
    }
}

/**
 * Document doubleCsrMatrix<class,class>(const csr_matrix<class,class>&) here...
 *
 * Double the size of collumns of csr matrix, and fill new colluns with entaire csr cols
 * Assume that we have all necessary memory
 *
 * @param csr
 * @author pozpl
 */
//template <typename int, typename double>

void doubleCsrMatrix(csr_matrix& csr) {
    /*coo_matrix collToAdd;
    int initColsNum = csr.num_cols;
    for(int i = 0; i < initColsNum; i++){
        collToAdd = get_coo_column(csr, i);
        //add_col_to_csr_mtx(csr, collToAdd);
        add_col_to_csr_mtx(csr, collToAdd);
    }
     */
    //скопируем Aj, Ax

    for (int i = csr.num_rows - 1; i >= 0; i--) {
        int rowBegin = csr.Ap[i];
        int rowEnd = csr.Ap[i + 1];
        int pieceLength = rowEnd - rowBegin;

        memmove(&csr.Aj[rowBegin * 2], &csr.Aj[rowBegin], pieceLength * sizeof (int));
        //memmove(&csr.Aj[rowBegin * 2 + pieceLength], &csr.Aj[rowBegin], pieceLength * sizeof(int));
        for (int j = rowBegin * 2; j < rowBegin * 2 + pieceLength; j++) {
            csr.Aj[j + pieceLength] = csr.Aj[j] + csr.num_cols;
        }

        memmove(&csr.Ax[rowBegin * 2], &csr.Ax[rowBegin], pieceLength * sizeof (double));
        memmove(&csr.Ax[rowBegin * 2 + pieceLength], &csr.Ax[rowBegin], pieceLength * sizeof (double));

    }

    for (int i = 0; i <= csr.num_rows; i++) {
        csr.Ap[i] = csr.Ap[i] * 2;
    }

    csr.num_cols = csr.num_cols * 2;
    csr.num_nonzeros = csr.num_nonzeros * 2;
}

void csr_get_bibi(const csr_matrix& csr, double* bibi) {
    for (int i = 0; i < csr.num_cols; i++) {
        bibi[i] = 0;
    }
    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            //printf("%i %i %f \n", i, csr.Aj[j], csr.Ax[j]);
            bibi[csr.Aj[j]] += csr.Ax[j] * csr.Ax[j];
            //std::cout << i << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
        }
    }
}

void csr_get_bibj(const csr_matrix& A, const csr_matrix& B) {
    double* bibj = new_host_darray(B.num_cols);
    double epsilon = 10e-10;
    int numberOfZerros = 0;
    std::cout << "A.num_rows * B.num_cols: " << A.num_rows * B.num_cols << "\n";
    for (int i = 0; i < B.num_cols; i++) {
        bibj[i] = 0.0;
    }
    for (int aRowNum = 0; aRowNum < A.num_rows; aRowNum++) {
        const int A_row_start = A.Ap[aRowNum];
        const int A_row_end = A.Ap[aRowNum + 1];
        for (int aColIdx = A_row_start; aColIdx < A_row_end; aColIdx++) {
            int aColNum = A.Aj[aColIdx];
            int B_row_start = B.Ap[aColNum];
            int B_row_end = B.Ap[aColNum + 1];
            for (int bColIdx = B_row_start; bColIdx < B_row_end; bColIdx++) {
                int bColNum = B.Aj[bColIdx];
                bibj[bColNum] += A.Ax[aColIdx] * B.Ax[bColIdx];
            }

        }
        for (int zerCntr = 0; zerCntr < B.num_cols; zerCntr++) {
            if (bibj[zerCntr] < epsilon && bibj[zerCntr] > -epsilon) {
                numberOfZerros++;
            }
        }
        std::cout << "Number of zerros = " << numberOfZerros << " \n";
        for (int i = 0; i < B.num_cols; i++) {
            bibj[i] = 0.0;
        }
    }
    std::cout << "Number of zerros = " << numberOfZerros << " the part of zerros = " << A.num_cols * A.num_rows / numberOfZerros << "\n";

}

