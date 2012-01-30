#include "sparse_operations_ext_cpu.hpp"
//#include "conus_projection_sparse.cuh"
#include "sparse_types.hpp"
#include "overloaded_cblas.hpp"
#include "fixed_conus_projection.hpp"

template <typename IndexType, typename ValueType> void proj_to_lp_cpu(csr_matrix<IndexType, ValueType> &inputSet,
        ValueType * minNormVector, IndexType inSetDim, IndexType vectorDim, ValueType tollerance) {
    ValueType lpShift = (ValueType) 10; //Установим начальное смещение, оно должно быть очень большим
    //переместим наш политоп в соответствии с этим смещением, это будет означать
    //что мы переместили начало координат в эту точку, что соответствует тому что мы отнимим
    //от вектора b из уравнения Ax <=b следующую величину b + t * A * c (с = [1])
    ValueType* vshift = (ValueType *) malloc(inSetDim * sizeof (ValueType));
    ValueType* t_C = (ValueType *) malloc((vectorDim - 1) * sizeof (ValueType));
    for (IndexType i = 0; i < vectorDim - 1; i++) {
        t_C[i] = -lpShift;
    }

    IndexType maxApprIters = 10;
    ValueType *x_s = new_host_array<ValueType > (vectorDim);
    ValueType *delta_x = new_host_array<ValueType > (vectorDim);
    ValueType* v_t = new_host_array<ValueType > (vectorDim); //- 1); //(ValueType *) malloc((vectorDim -1) * sizeof (ValueType));

    IndexType *kvec = new_host_array<IndexType>(2 * inputSet.num_cols + 1 );
    IndexType *kvec_old = new_host_array<IndexType>(2 * inputSet.num_cols + 1 );
    IndexType *basisVecInx = new_host_array<IndexType>(2 * inputSet.num_cols + 1);
    IndexType baselen = 0;

    for (IndexType i = 0; i < 2 * inputSet.num_cols + 1; i++) {
        kvec[i] = 0;
        kvec_old[i] = 0;
    }

    //Начальный x_s это ноль
    cblas_scal(vectorDim, 0.0, x_s, 1);
    for (IndexType apprIt = 0; apprIt < maxApprIters; apprIt++) {
        //Получаем новое B
        cblas_copy(vectorDim - 1, t_C, 1, v_t, 1);
        cblas_axpy(vectorDim - 1, -1.0, x_s, 1, v_t, 1);
        //std::cout << "Vector V_t\n";
        //printMatrixCPU(vectorDim - 1, 1, v_t);
        cblas_copy(vectorDim, x_s, 1, delta_x, 1);
        inputSet.num_rows--;
        spmv_csr_t_serial_host(inputSet, v_t, vshift);
        inputSet.num_rows++;

        //printMatrixCPU(inSetDim, 1, vshift);
        incValsInLstMtxRowVec(inputSet, inputSet.num_rows - 1, vshift);

        std::cout << "Input set on iteration\n";
        //print_csr_matrix(inputSet);
        //getProjectionOnConus(inputSet, minNormVector, inSetDim, vectorDim, tollerance);
        csr_matrix<IndexType, ValueType> inSetCopy = getCsrMtxCopy(inputSet);

        if(apprIt > 0){
            memcpy(kvec_old, kvec, (2 * inputSet.num_cols + 1) * sizeof (IndexType));
        }
        //print_csr_matrix(inSetCopy);
        //projOnFixedSimplex(inSetCopy, minNormVector, kvec, basisVecInx, baselen, tollerance );
        getMinNormElemOutRepr(in_a_csc, minNormVector, 0.0001, kvec,  basisVecInx, baselen, numberOfEqConstr);
        delete_host_matrix(inSetCopy);

        IndexType kvec_razn = 0;
        if(apprIt > 0){
            for (IndexType i = 0; i < 2 * inputSet.num_cols + 1; i++) {
                if(kvec[i] == kvec_old[i] && kvec[i] == 1){
                    kvec_razn++;
                }
            }
        }
        std::cout << "KVEC RAZN is = " << kvec_razn <<"\n";
        std::cout << "baselen is = " << baselen <<"\n";

        cblas_axpy(vectorDim, -1.0, v_t, 1, minNormVector, 1);
        //std::cout << "Min norm vector on iteration\n";
        //printMatrixCPU(vectorDim, 1, minNormVector);

        decValsInLstMtxRowVec(inputSet, inputSet.num_rows - 1, vshift);


        cblas_copy(vectorDim, minNormVector, 1, x_s, 1);

        cblas_axpy(vectorDim, -1.0, x_s, 1, delta_x, 1);
        ValueType z_summ = 0.0;        
        for (IndexType i = 0; i < vectorDim - 1; i++) {
            z_summ += minNormVector[i];
        }
        std::cout << "Summ of elements " << z_summ << " on "<< apprIt <<" iteration\n";
        ValueType dist = cblas_dot(vectorDim - 1, delta_x, 1, delta_x, 1);
        if (dist < tollerance * tollerance) {
            std::cout << "iterations count :" << apprIt << "\n";
            break;
        }       
    }

    //cblas_axpy(vectorDim, -1.0, v_t, 1, minNormVector, 1);
    ValueType dist = cblas_dot(vectorDim - 1, minNormVector, 1, minNormVector, 1);
    std::cout << "Min Vector Lengh = " << sqrt(dist) << "\n";
    ValueType z_summ = 0.0;
    IndexType nonzer_summ = 0;
    for (IndexType i = 0; i < vectorDim - 1; i++) {
        z_summ += minNormVector[i];
        if (minNormVector[i] != 0.0) {
            nonzer_summ++;
        }
    }

    std::cout << "Summ of elements " << z_summ << " \n";
    std::cout << "Count of nonzerros " << nonzer_summ << " \n";
    //printMatrixToFileForOctave(vectorDim, 1, minNormVector);
    ValueType* b_contr = new_host_array<ValueType > (inSetDim);
    inputSet.num_rows--;
    spmv_csr_t_serial_host(inputSet, minNormVector, b_contr);
    inputSet.num_rows++;
    //printMatrixCPU(inSetDim, 1, b_contr);
    IndexType b_begin = inputSet.Ap[vectorDim - 1];
    IndexType b_end = inputSet.Ap[vectorDim];
    IndexType inconsCount = 0;
    for (IndexType i = b_begin; i < b_end; i++) {
        IndexType j = inputSet.Aj[i];
        if (b_contr[j] > inputSet.Ax[i] + tollerance * tollerance) {
            //std::cout << "b_contr " << b_contr[j] << " Ax " << inputSet.Ax[i] << " \n";
            ValueType razn = b_contr[j] -  inputSet.Ax[i];
            //printf("bcontr[%i] %e Ax %e  razn %e\n", j ,b_contr[j], inputSet.Ax[i], razn);
            inconsCount++;
        }
    }
    std::cout << "Inconsistent X count: " << inconsCount << " \n";

}

/**Increas all elemant on Last matrix column. We assume that this column is full of elements, so no zerrow elements present in it and we
 * dont need to add elements in csr matrix.
 * csr - matrix that contain elements
 * rowIdx - index of last row
 * valToAdd - value to add to presented value.
 *
 */
template <typename IndexType, typename ValueType> void incValsInLstColVec(const csr_matrix<IndexType, ValueType>& csr, IndexType colIdx, ValueType* valsToAdd) {
    for (IndexType i = 0; i < csr.num_rows; i++) {
        for (IndexType j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            if (csr.Aj[j] == colIdx) {
                //std::cout << "before " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
                csr.Ax[j] += valsToAdd[i];
                //std::cout << "after " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
            }
        }
    }
}

/**Decreas all elemant on Last matrix column. We assume that this column is full of elements, so no zerrow elements present in it and we
 * dont need to add elements in csr matrix.
 * csr - matrix that contain elements
 * rowIdx - index of last row
 * valToAdd - value to add to presented value.
 *
 */
template <typename IndexType, typename ValueType> void decValsInLstColVec(const csr_matrix<IndexType, ValueType>& csr, IndexType colIdx, ValueType* valsToAdd) {
    for (IndexType i = 0; i < csr.num_rows; i++) {
        for (IndexType j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            if (csr.Aj[j] == colIdx) {
                //std::cout << "before " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
                csr.Ax[j] -= valsToAdd[i];
                //std::cout << "after " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
            }
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
template <typename IndexType, typename ValueType> void incValsInLstMtxRowVec(const csr_matrix<IndexType, ValueType>& csr, IndexType rowIdx, ValueType* valToAdd) {
    for (IndexType j = csr.Ap[rowIdx], colIdx = 0; j < csr.Ap[rowIdx + 1]; j++, colIdx++) {
        //std::cout << "before " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
        csr.Ax[j] += valToAdd[colIdx];
        //std::cout << "after " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
    }
}

/**Decreas all elemant on Last matrix rows. We assume that this row is full of elements, so no zerrow elements present in it and we
 * dont need to add elements in csr matrix.
 * csr - matrix that contain elements
 * rowIdx - index of last row
 * valToAdd - value to add to presented value.
 *
 */
template <typename IndexType, typename ValueType> void decValsInLstMtxRowVec(const csr_matrix<IndexType, ValueType>& csr, IndexType rowIdx, ValueType* valToAdd) {
    for (IndexType j = csr.Ap[rowIdx], colIdx = 0; j < csr.Ap[rowIdx + 1]; j++, colIdx++) {
        //std::cout << "before " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
        csr.Ax[j] -= valToAdd[colIdx];
        //std::cout << "after " << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
    }
}

template <typename T> void printMatrixToFileForOctave(int rows, int cols, T* matrix) {
    FILE *file;
    char* file_name = "x_vect.mat";


    file = fopen(file_name, "w");

    //fputs( "string", file );



    printf("\n");
    fprintf(file, "# Created by Octave 3.2.0, Thu Jul 09 16:44:15 2009 VLAST <pozpl@urbis> \n");
    fprintf(file, "# name: X\n");
    fprintf(file, "# type: matrix\n");
    fprintf(file, "# rows: %i\n", rows);
    fprintf(file, "# columns: %i\n", cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j == (cols - 1)) {
                //printf("%f ", matrix[i * coll_num + j]);
                fprintf(file, "%e ", matrix[j * rows + i]);
            } else {
                //printf("%f, ", matrix[i * coll_num + j]);
                fprintf(file, "%e ", matrix[j * rows + i]);
            }
        }
        if (i != (rows - 1)) {
            fprintf(file, "\n");
        } else {
            fprintf(file, "\n");
        }
    }
    fprintf(file, "");
    fclose(file);
}


