#include "sparse_operations_ext_cpu.hpp"
#include "overloaded_cblas.hpp"



//template <typename int, typename double>

void printMatrixCPU(int str_num, int coll_num, double* matrix) {
    std::cout << "\n ";
    for (int i = 0; i < str_num; i++) {
        for (int j = 0; j < coll_num; j++) {
            //printf("%f ", matrix[i * coll_num + j]);
            //printf("%f ", matrix[j * str_num + i]);
            std::cout << matrix[j * str_num + i] << " ";
        }
        std::cout << "\n ";
        //std::cout << " ; ";
    }
}

//template <typename int, typename double>

void print_csr_matrix(const csr_matrix& csr) {
    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            //printf("%i %i %f \n", i, csr.Aj[j], csr.Ax[j]);
            std::cout << i << " " << csr.Aj[j] << " " << csr.Ax[j] << "\n";
        }
    }
}

void print_csr_matrix_to_file(const csr_matrix& csr, const char *file_name, const char *matrix_name) {
    std::ofstream myfile;
    myfile.open(file_name);
    myfile << "# name: " << matrix_name << "\n";
    myfile << "# type: sparse matrix \n";
    myfile << "# nnz: " << csr.num_nonzeros << "\n";
    myfile << "# rows: " << csr.num_cols << "\n";
    myfile << "# columns: " << csr.num_rows << "\n";

    myfile.precision(15);

    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            //printf("%i %i %f \n", i, csr.Aj[j], csr.Ax[j]);
            myfile << csr.Aj[j] + 1 << " " << i + 1 << " " << csr.Ax[j] << "\n";
        }
    }
}

void print_ldl_matrix(const ldl_matrix& ldlm) {
    for (int i = 0; i < ldlm.num_cols; i++) {
        for (int j = ldlm.Lp[i]; j < ldlm.Lp[i + 1]; j++) {
            //printf("%i %i %f \n", i, csr.Aj[j], csr.Ax[j]);
            std::cout << i << " " << ldlm.Li[j] << " " << ldlm.Lx[j] << "\n";
        }
    }
    std::cout << "Diag part\n";
    for (int i = 0; i < ldlm.num_cols; i++) {
        std::cout << i << " " << ldlm.D[i] << "\n";
    }
}
//template <typename int, typename double>

void print_coo_matrix(const coo_matrix& coo) {
    for (int i = 0; i < coo.num_nonzeros; i++) {
        printf("%u %u %f \n", coo.I[i], coo.J[i], coo.V[i]);
    }
}

void print_csc_matrix(const csc_matrix& csc) {
    for (int i = 0; i < csc.num_cols; i++) {
        for (int j = csc.Cp[i]; j < csc.Cp[i + 1]; j++) {
            //printf("%i %i %f \n", i, csr.Aj[j], csr.Ax[j]);
            std::cout << csc.Ri[j] << " " << i << " " << csc.Ex[j] << "\n";
        }
    }
}

//template <typename int, typename double>

void get_dense_row(const csr_matrix& csr, int row_number, double *dense_row) {

    for (int i = 0; i < csr.num_cols; i++) {
        dense_row[i] = 0;
    }
    for (int i = csr.Ap[row_number]; i < csr.Ap[row_number + 1]; i++) {
        dense_row[csr.Aj[i]] = csr.Ax[i];
    }
}

void get_dense_row_from_triangular_gramm(const csr_matrix& csr, int row_number, double *dense_row) {
    for (int i = 0; i < csr.num_cols; i++) {
        dense_row[i] = 0;
    }
    /**
     * Берём элементы по строке до индекса row_number ыключительно, остальные добрать надо по столбцам
     */
    for (int i = csr.Ap[row_number]; i < csr.Ap[row_number + 1]; i++) {
        dense_row[csr.Aj[i]] = csr.Ax[i];
    }
    //NOTE in ideal world then we do not get any rows but last row, code below shuld not work at all -> profit!
    for (int i = csr.Ap[row_number + 1]; i < csr.Ap[csr.num_rows]; i++) {
        int rowBeg = csr.Ap[i];
        int rowEnd = csr.Ap[i + 1];
        for (int j = rowBeg; j < rowEnd; j++) {
            if (csr.Aj[j] == row_number) {
                dense_row[i] = csr.Ax[j];
            }
        }
    }

}

void __get_dense_column(const csr_matrix& csr_t, int col_number, double *dense_column) {

    for (int i = 0; i < csr_t.num_cols; i++) {
        dense_column[i] = 0;
    }
    for (int i = csr_t.Ap[col_number]; i < csr_t.Ap[col_number + 1]; i++) {
        dense_column[csr_t.Aj[i]] = csr_t.Ax[i];
    }
}

//template <typename int, typename double>

void get_dense_column(const csr_matrix& csr, int col_number, double *dense_column) {

    for (int i = 0; i < csr.num_rows; i++) {
        dense_column[i] = 0.0;
    }
    for (int i = 0; i < csr.num_rows; i++) {
        int rowBeg = csr.Ap[i];
        int rowEnd = csr.Ap[i + 1];
        for (int j = rowBeg; j < rowEnd; j++) {
            if (csr.Aj[j] == col_number) {
                dense_column[i] = csr.Ax[j];
            }
        }
    }
    //for(int i=csr_t.Ap[col_number]; i < csr_t.Ap[col_number+1]; i++ ){
    //    dense_column[csr_t.Aj[i]] = csr_t.Ax[i];
    //}
}

void get_ldl_dense_column_from_l_low(const ldl_matrix& ldl, int col_number, double *dense_column) {

    for (int i = 0; i < ldl.num_rows - 1; i++) {
        dense_column[i] = 0.0;
    }

    for (int nonz_i = ldl.Lp[col_number]; nonz_i < ldl.Lp[col_number + 1]; nonz_i++) {
        dense_column[ldl.Li[nonz_i]] = ldl.Lx[nonz_i];
    }
}

void get_ldl_dense_row_from_l_upper(const ldl_matrix& ldl, int row_index, double*dense_row) {
    int row_dem = ldl.num_rows - (row_index + 1);
    for (int col_i = 0; col_i < row_dem; col_i++) {
        dense_row[col_i] = 0.0;
    }
    for (int col_i = row_index; col_i < ldl.num_cols; col_i++) {
        for (int col_j = ldl.Lp[col_i]; col_j < ldl.Lp[col_i + 1]; col_j++) {
            //std::cout << "dense_row   " << i << " " << ldl.Li[j] << " " << ldl.Lx[j] << "\n";
            if (ldl.Li[col_j] == row_index) {
                dense_row[col_i - (row_index + 1)] = ldl.Lx[col_j];
            }
        }
    }
}

ldl_matrix get_ldl33_up_from_ldl_l_upper(const ldl_matrix& ldl, int row_start_idx) {
    //estimate elements number
    int elemets_number = 0;
    for (int col_i = row_start_idx; col_i < ldl.num_cols; col_i++) {
        for (int row_j = ldl.Lp[col_i]; row_j < ldl.Lp[col_i + 1]; row_j++) {
            if (ldl.Li[row_j] >= row_start_idx) {
                elemets_number++;
            }
        }
    }
    
    
    int ldl33_max_dim = ldl.num_rows - row_start_idx;
    
    std::cout << "Elements in L33 = " << elemets_number << " ldl33 max dim = " << ldl33_max_dim << " ldl.num_rows = " << ldl.num_rows << "\n";
    //construct new ldl matrix with upper triangular form
    if(elemets_number == 0){elemets_number++;} // trik for matrix to be st propertly
    ldl_matrix ldl33_up = new_ldl_matrix(ldl33_max_dim, elemets_number);

    //copy L^t_33 to L_33
    ldl33_up.num_cols = ldl.num_cols - row_start_idx;
    ldl33_up.num_rows = ldl.num_rows - row_start_idx;
    elemets_number = 0;
    for (int col_i = row_start_idx; col_i < ldl.num_cols; col_i++) {
        ldl33_up.Lp[col_i - row_start_idx] = elemets_number;
        for (int row_j = ldl.Lp[col_i]; row_j < ldl.Lp[col_i + 1]; row_j++) {
            if (ldl.Li[row_j] >= row_start_idx) {
                ldl33_up.Li[elemets_number] = ldl.Li[row_j] - row_start_idx;
                ldl33_up.Lx[elemets_number] = ldl.Lx[row_j];
                ldl33_up.num_nonzeros++;
                elemets_number++;
            }
        }
    }
    ldl33_up.Lp[ldl.num_cols - row_start_idx] = elemets_number;


    //copy diagonal elements
    for (int diag_idx = row_start_idx; diag_idx < ldl.num_cols; diag_idx++) {
        ldl33_up.D[diag_idx - row_start_idx] = ldl.D[diag_idx];
    }

    return ldl33_up;
}

/***
 * Add column to the end of lower triangular part of matrix
 * i.e. build l33 part
 */
void add_last_col_to_ldl_l_low(ldl_matrix& ldl, double *dense_column, int col_id, int dens_col_dim) {

    //get adress to write information
    int active_col_id = col_id;
    ldl.Lp[ active_col_id ] = ldl.num_nonzeros;
    ldl.Lp[ active_col_id + 1] = ldl.num_nonzeros;

    int ldl_li_index_to_insert = ldl.num_nonzeros;

    for (int den_col_iter = 0; den_col_iter < dens_col_dim; den_col_iter++) {
        if (dense_column[den_col_iter] != 0) {
            ldl.Li[ldl_li_index_to_insert] = den_col_iter + 1 + col_id;
            ldl.Lx[ldl_li_index_to_insert] = dense_column[den_col_iter];

            ldl_li_index_to_insert++;
            ldl.num_nonzeros++;
            ldl.Lp[active_col_id + 1]++;
        }
    }
}

/*
 * @param ldl - ldl matrix
 * col_to_start_count - index of col to star count a number of elements in column.
 */
void compute_ldl_lnz(ldl_matrix& ldl, int col_to_start_count) {
    for (int col_i = col_to_start_count; col_i < ldl.num_cols; col_i++) {
        int row_begin = ldl.Lp[col_i];
        int row_end = ldl.Lp[col_i + 1];
        ldl.Lnz[col_i] = row_end - row_begin;
    }
}

void insert_ldl33_up_into_ldl_up(ldl_matrix& ldl, ldl_matrix ldl33) {
    //compute number of nonzerows elements in ewach column of ldl33
    compute_ldl_lnz(ldl33, 1); //because 0, column is diag element

    //get col in ldl to start insert ldl33
    //+1 is becouse we deal with uptriangle matrix
    //and index in up triangle matrix is 1;
    int col_to_insert = ldl.num_cols - ldl33.num_cols + 1;

    //compute number of nonzerows elements in ewach column of ldl
    compute_ldl_lnz(ldl, col_to_insert);

    int row_idx_to_start_insert = ldl.num_cols - ldl33.num_cols;
    //iterate throught ldl last columns and insert ldl33 columns into right places
    for (int col_i = col_to_insert, ldl33_col_i = 1; col_i < ldl.num_cols; col_i++, ldl33_col_i++) {
        int row_begin = ldl.Lp[col_i];
        int row_end = ldl.Lp[col_i + 1];

        for (int row_i = row_begin; row_i < row_end; row_i++) {
            if (ldl.Li[row_i] >= row_idx_to_start_insert) {
                int shift_elems = (row_end - row_i) - ldl33.Lnz[ldl33_col_i];
                int tail_elements = ldl.num_nonzeros - row_i;
                //shift for new boundares
                memmove(&ldl.Li[row_end + shift_elems], &ldl.Li[row_end], sizeof (int) * tail_elements);
                memmove(&ldl.Lx[row_end + shift_elems], &ldl.Lx[row_end], sizeof (double) * tail_elements);

                //actually copyinformation
                int ldl33_row_begin = ldl33.Lp[ldl33_col_i];
                memmove(&ldl.Li[row_i], &ldl33.Li[ldl33_row_begin], sizeof (int) * ldl33.Lnz[ldl33_col_i]);
                memmove(&ldl.Lx[row_i], &ldl33.Lx[ldl33_row_begin], sizeof (double) * ldl33.Lnz[ldl33_col_i]);
                
                ldl.num_nonzeros += shift_elems;
                
                //correct Lp values for subsequent elements
                for (int col_tail_i = col_i + 1; col_tail_i < ldl.num_cols + 1; col_tail_i++) {
                    ldl.Lp[col_tail_i] += shift_elems;
                }

                //correct Li values
                for (int li_idx = row_i; li_idx < ldl.Lp[col_i + 1]; li_idx++) {
                    ldl.Li[li_idx] += row_idx_to_start_insert;
                }

                break; //go to the next column
            }
        }
    }
    //insert diag part
    memmove(&ldl.D[col_to_insert - 1], &ldl33.D[0], sizeof(double) * ldl33.num_cols);
}

/*
 * Change LDL in upper case.
 * Move ldl33 with new ldl33
 */
void change_ldl_33_up(ldl_matrix& ldl, ldl_matrix ldl33_up) {

}

void del_col_from_ldl_up(ldl_matrix& ldl, int del_idx) {
    if (del_idx < ldl.num_cols - 1) {
        if (del_idx != 0) {
            int col_begin = ldl.Lp[del_idx];
            int col_end = ldl.Lp[del_idx + 1];
            
            memmove(&ldl.Li[col_begin], &ldl.Li[col_end], sizeof (int) * (ldl.num_nonzeros - col_end));
            memmove(&ldl.Lx[col_begin], &ldl.Lx[col_end], sizeof (double) * (ldl.num_nonzeros - col_end));
            
            int els_in_col = col_end - col_begin;
            for (int i = del_idx + 1; i <= ldl.num_cols; i++) {
                std::cout << "Elements in del cplumn = " << els_in_col << " \n";
                ldl.Lp[i] = ldl.Lp[i] - els_in_col;
                std::cout << "ldl.Lp[" << i << "] = " << ldl.Lp[i] << " \n";
            }
            //ldl.Lp[ldl.num_cols] = 0;
            ldl.num_nonzeros -= els_in_col;
        }
        
        std::cout << "move el with index L[" << del_idx << "] = " << ldl.Lp[del_idx] << "  to  " << ldl.Lp[del_idx + 1] << "\n";
        memmove(&ldl.Lp[del_idx], &ldl.Lp[del_idx + 1], sizeof (int) * (ldl.num_cols - del_idx));
    }
    ldl.num_cols--;
}

void del_row_from_ldl_up(ldl_matrix& ldl, int del_idx) {
    //If del_idx = 0, then we del only from diag part, but l_up part is 
    //unchnged, so we need fix this
    int columns_in_ldl_up = ldl.num_rows;//(del_idx == 0) ? ldl.num_cols + 1 : ldl.num_cols;
    //del row
    for (int col_i = 1; col_i < columns_in_ldl_up; col_i++) {
        for (int row_i = ldl.Lp[col_i]; row_i < ldl.Lp[col_i + 1]; row_i++) {
            if (ldl.Li[row_i] == del_idx) {
                //perfom shift
                int num_els_in_tail = ldl.num_nonzeros - (row_i + 1);
                std::cout << "Num elements in tail to mmove = " << num_els_in_tail << "\n";
                memmove(&ldl.Li[row_i], &ldl.Li[row_i + 1], sizeof (int) * num_els_in_tail);
                memmove(&ldl.Lx[row_i], &ldl.Lx[row_i + 1], sizeof (double) * num_els_in_tail);
                ldl.num_nonzeros--;
                
                for (int col_tail_i = col_i + 1; col_tail_i < columns_in_ldl_up + 1; col_tail_i++) {
                    ldl.Lp[col_tail_i]--;
                } 
                
                if(row_i < ldl.Lp[col_i + 1]){                        
                        ldl.Li[row_i]--;
                }
            } else if (ldl.Li[row_i] > del_idx) {
                ldl.Li[row_i]--;
            }
        }
    }
    //Delete element from diagonal
    int num_els_in_tail = ldl.num_rows - (del_idx + 1);
    memmove(&ldl.D[del_idx], &ldl.D[del_idx + 1], sizeof (double) * num_els_in_tail);

    ldl.num_rows--;
}

void del_row_col_from_ldl_up(ldl_matrix& ldl, int del_idx) {
    del_row_from_ldl_up(ldl, del_idx);
    
    del_col_from_ldl_up(ldl, del_idx);
}

//template <typename int, typename double>

coo_matrix __get_coo_column(const csr_matrix& csr_t, int col_number) {

    coo_matrix coo_col;
    coo_col.num_nonzeros = csr_t.Ap[col_number + 1] - csr_t.Ap[col_number];
    coo_col.num_cols = 1;
    coo_col.num_rows = coo_col.num_nonzeros;
    coo_col.I = new_host_iarray(coo_col.num_rows);
    coo_col.J = new_host_iarray(coo_col.num_rows);
    coo_col.V = new_host_darray(coo_col.num_rows);
    for (int i = csr_t.Ap[col_number], ii = 0; i < csr_t.Ap[col_number + 1]; i++, ii++) {
        //dense_column[csr_t.Aj[i]] = csr_t.Ax[i];
        coo_col.I[ii] = csr_t.Aj[i];
        coo_col.J[ii] = 0;
        coo_col.V[ii] = csr_t.Ax[i];
    }
    return coo_col;
}

//template <typename int, typename double>

coo_matrix get_coo_column(const csr_matrix& csr, int col_number) {

    coo_matrix coo_col;
    coo_col.num_nonzeros = 0;
    for (int i = 0; i < csr.num_nonzeros; i++) {
        if (csr.Aj[i] == col_number) {
            coo_col.num_nonzeros++;
        }
    }

    coo_col.num_cols = 1;
    coo_col.num_rows = coo_col.num_nonzeros;
    coo_col.I = new_host_iarray(coo_col.num_rows);
    coo_col.J = new_host_iarray(coo_col.num_rows);
    coo_col.V = new_host_darray(coo_col.num_rows);
    for (int i = 0, nonzer_count = 0; i < csr.num_rows; i++) {
        int rowBeg = csr.Ap[i];
        int rowEnd = csr.Ap[i + 1];
        for (int j = rowBeg; j < rowEnd; j++) {
            if (csr.Aj[j] == col_number) {
                coo_col.I[nonzer_count] = i;
                coo_col.J[nonzer_count] = 0;
                coo_col.V[nonzer_count] = csr.Ax[j];
                //std::cout << "I " << coo_col.I[nonzer_count]  << " J " << coo_col.J[nonzer_count] << " V " << coo_col.V[nonzer_count] << "\n";
                nonzer_count++;
            }
        }
    }
    return coo_col;
}

coo_matrix get_coo_column_from_csc(const csc_matrix& csc, int col_number) {

    coo_matrix coo_col;
    coo_col.num_nonzeros = csc.Cp[col_number + 1] - csc.Cp[col_number];


    coo_col.num_cols = 1;
    coo_col.num_rows = coo_col.num_nonzeros;
    coo_col.I = new_host_iarray(coo_col.num_rows);
    coo_col.J = new_host_iarray(coo_col.num_rows);
    coo_col.V = new_host_darray(coo_col.num_rows);
    int csc_col_begin = csc.Cp[col_number];
    for (int i = 0; i < coo_col.num_nonzeros; i++) {
        coo_col.I[i] = csc.Ri[csc_col_begin + i];
        coo_col.J[i] = 0;
        coo_col.V[i] = csc.Ex[csc_col_begin + i];
    }
    return coo_col;
}

/*
 * Добавляет к матрице в формате csr столбец в формате coo
 * Предполагается, что предварительно выделено памяти для добавления
 * и добавление происходит путём смещения областей памяти в массиве столбцов и
 * значений, в освободившиеся участки втавляются значения.
 */
//template <typename int, typename double>

void add_col_to_csr_mtx(csr_matrix& csr,
        coo_matrix& coo_column) {
    /*csr.num_cols++;
    int row_to_add;
    //std::cout << "Row Numbers " << coo_column.num_rows << " \n";
    for (int i = 0; i < coo_column.num_rows; i++) {
        csr.num_nonzeros++;
        row_to_add = coo_column.I[i];
        //std::cout << "add row " << row_to_add << " \n";
        //int el_count = csr.num_nonzeros - csr.Ap[row_to_add + 1];
        if ((row_to_add + 1) != csr.num_rows) {
            //std::cout << "muve n =  " << csr.num_nonzeros - csr.Ap[row_to_add + 1] << " elements " << csr.Aj[csr.Ap[row_to_add + 1] + 1] << " \n";
            memmove(&csr.Aj[csr.Ap[row_to_add + 1] + 1], &csr.Aj[csr.Ap[row_to_add + 1]], sizeof (int) * (csr.num_nonzeros - csr.Ap[row_to_add + 1]));
            memmove(&csr.Ax[csr.Ap[row_to_add + 1] + 1], &csr.Ax[csr.Ap[row_to_add + 1]], sizeof (double) * (csr.num_nonzeros - csr.Ap[row_to_add + 1]));
        }
        csr.Aj[csr.Ap[row_to_add + 1]] = csr.num_cols - 1;
        csr.Ax[csr.Ap[row_to_add + 1]] = coo_column.V[i];
        //std::cout <<" wtite to position " << csr.Ap[row_to_add + 1] <<  " \n";
        for (int j = row_to_add + 1; j < csr.num_rows + 1; j++) {
            csr.Ap[j]++;
            //std::cout << ">" << csr.Ap[j] << "\n";
        }
    }*/

    csr.num_cols++;
    for (int coo_i = 0; coo_i < coo_column.num_nonzeros; coo_i++) {
        int el_row_idx = coo_column.I[coo_i];
        double el_val = coo_column.V[coo_i];
        //std::cout << "el_val =  " << el_val << " \n";
        int number_of_csr_rows = csr.num_rows;
        if (el_row_idx < number_of_csr_rows - 1) {
            int ldl_next_col_start_idx = csr.Ap[el_row_idx + 1];
            int num_eld_involved = csr.num_nonzeros - ldl_next_col_start_idx;
            //std::cout << "el_col_idx =  " << el_col_idx << " index of array to start "<< ldl_next_col_start_idx << " number of nonzeros " << ldl.num_nonzeros << " number of colums " << ldl.num_cols << " \n";
            memmove(&csr.Aj[ldl_next_col_start_idx + 1], &csr.Aj[ldl_next_col_start_idx], sizeof (int) * num_eld_involved);
            memmove(&csr.Ax[ldl_next_col_start_idx + 1], &csr.Ax[ldl_next_col_start_idx], sizeof (double) * num_eld_involved);
            csr.num_nonzeros++;
            csr.Aj[ldl_next_col_start_idx] = csr.num_cols - 1;
            csr.Ax[ldl_next_col_start_idx] = el_val;
            for (int ldl_col_i = el_row_idx + 1; ldl_col_i < csr.num_rows + 1; ldl_col_i++) {
                csr.Ap[ldl_col_i]++;
            }
        } else if (el_row_idx == number_of_csr_rows - 1) {
            int ldl_next_col_start_idx = csr.Ap[el_row_idx + 1];
            csr.Aj[ldl_next_col_start_idx] = csr.num_cols - 1;
            csr.Ax[ldl_next_col_start_idx] = el_val;

            csr.Ap[el_row_idx + 1]++;
            csr.num_nonzeros++;
            //std::cout << "!el_col_idx =  " << el_col_idx << " index of array to start "<< ldl_next_col_start_idx << " number of nonzeros " << ldl.num_nonzeros << " \n";
        }//else{
        //   int ldl_next_col_start_idx = ldl.Lp[el_col_idx];
        //   ldl.Li[ldl_next_col_start_idx] = ldl.num_rows - 1;
        //   ldl.Lx[ldl_next_col_start_idx] = el_val;
        //   ldl.num_cols++;
        //   ldl.num_nonzeros++;
        //   ldl.Lp[el_col_idx + 1] = ldl.num_nonzeros;            
        //   //std::cout << "!!el_col_idx =  " << el_col_idx << " index of array to start "<< ldl_next_col_start_idx << " number of nonzeros " << ldl.num_nonzeros << " \n";
        //}

    }
    // std::cout<< "<<<<>> csr.num_cols " << csr.num_cols << "csr.num_rows " << csr.num_rows << "\n";  
    // print_csr_matrix(csr);
}

/*Estimate max count of nonzerros elements for entire matrix
 * number of distinct columns that will geted from this matrix
 * csr_t - matrix in csr format that is transposed matrix for wich we need to
 * estimate number of nonzerow elements in max_colls coluns
 * max_colls - maximum of distinct columns that we can get from matrix
 */
//template <typename int, typename double>

int __estimate_max_nonzeros(const csr_matrix& csr_t, int max_colls) {
    int *num_cols_in_row = new int[csr_t.num_rows];
    for (int i = 0; i < csr_t.num_rows; i++) {
        num_cols_in_row[i] = csr_t.Ap[i + 1] - csr_t.Ap[i];
        printf("%u\n", num_cols_in_row[i]);
    }
    std::sort(&num_cols_in_row[0], &num_cols_in_row[csr_t.num_rows]);
    //printf("sorted\n");
    int est_nonzer = 0;
    for (int i = (csr_t.num_rows - max_colls); i < csr_t.num_rows; i++) {
        //printf("%u\n", num_cols_in_row[i]);
        est_nonzer += num_cols_in_row[i];
    }

    std::cout << "Estimated cout " << est_nonzer << "\n";
    return est_nonzer;
}

//template <typename int, typename double>

int estimate_max_nonzeros(const csc_matrix& csc, int max_colls) {
    if (max_colls > csc.num_cols) {
        max_colls = csc.num_cols;
    }
    int *ells_in_coll = new_host_iarray(csc.num_cols);
    //std::cout << "SSS\n";
    for (int i = 0; i < csc.num_cols; i++) {
        ells_in_coll[i] = (int) 0;
    }
    //std::cout << "SSS\n";
    for (int i = 0; i < csc.num_rows; i++) {
        for (int j = csc.Cp[i]; j < csc.Cp[i + 1]; j++) {
            ells_in_coll[csc.Ri[j]]++;
        }
    }
    //std::cout << "SSS\n";
    std::sort(&ells_in_coll[0], &ells_in_coll[csc.num_cols]);
    int est_nonzer = 0;
    for (int i = (csc.num_cols - max_colls); i < csc.num_cols; i++) {
        //printf("%u\n", est_nonzer);
        est_nonzer += ells_in_coll[i];
    }
    //при появлении  базиса полной размерности, нам понадобится забить последнюю его строку единицами
    est_nonzer += max_colls; //csr.num_rows + 1;
    std::cout << "Estimated cout " << est_nonzer << "\n";

    free(ells_in_coll);
    return est_nonzer;
}

//template <typename int, typename double>

csr_matrix get_empty_csr_for_col_add(int est_rows_num, int est_nonzer_num) {
    csr_matrix csr;
    csr.num_nonzeros = 0;
    csr.num_rows = est_rows_num;
    csr.num_cols = 0;
    csr.Ap = new_host_iarray(csr.num_rows + 2); //2 вместо 1 добавляем так как может добавляться ещё одна строка при достижении полного базиса
    csr.Aj = new_host_iarray(est_nonzer_num);
    csr.Ax = new_host_darray(est_nonzer_num);
    for (int i = 0; i <= csr.num_rows; i++) {
        csr.Ap[i] = 0;
    }
    for (int i = 0; i < est_nonzer_num; i++) {
        csr.Aj[i] = 0;
        csr.Ax[i] = (double) 0.0;
    }
    return csr;
}

//template <typename int, typename double>

csr_matrix get_empty_csr_for_row_add(int est_rows_num, int est_nonzer_num) {
    csr_matrix csr;
    csr.num_nonzeros = 0;
    csr.num_rows = 0; //est_rows_num;
    csr.num_cols = 0;

    csr.Ap = new_host_iarray(est_rows_num + 1);
    csr.Aj = new_host_iarray(est_nonzer_num);
    csr.Ax = new_host_darray(est_nonzer_num);
    for (int i = 0; i <= est_rows_num; i++) {
        csr.Ap[i] = 0;
    }
    for (int i = 0; i < est_nonzer_num; i++) {
        csr.Aj[i] = 0;
        csr.Ax[i] = (double) 0.0;
    }
    return csr;
}




////////////////////////////////////////////////////////////////////////////////
//! Compute y += A*x for a sparse CSR matrix A and column vectors x and y
//! @param num_rows   number of rows in A
//! @param Ap         CSR pointer array
//! @param Aj         CSR index array
//! @param Ax         CSR data array
//! @param x          column vector
//! @param y          column vector
////////////////////////////////////////////////////////////////////////////////

/**
 * This Version of procedure multiply matrices A * B
 */
//template <typename int, typename double>

void mm_csr_serial_host(const csr_matrix& A,
        const csr_matrix& B,
        double *C) {
    //std::cout << "A.num_rows * B.num_cols: " << A.num_rows * B.num_cols << "\n";
    for (int i = 0; i < A.num_rows * B.num_cols; i++) {
        //std::cout << "i =  " << i << "\n";
        C[i] = 0.0;
    }
    //print_csr_matrix(A);
    for (int aRowNum = 0; aRowNum < A.num_rows; aRowNum++) {
        const int A_row_start = A.Ap[aRowNum];
        const int A_row_end = A.Ap[aRowNum + 1];
        for (int aColIdx = A_row_start; aColIdx < A_row_end; aColIdx++) {
            int aColNum = A.Aj[aColIdx];
            int B_row_start = B.Ap[aColNum];
            int B_row_end = B.Ap[aColNum + 1];
            for (int bColIdx = B_row_start; bColIdx < B_row_end; bColIdx++) {
                int bColNum = B.Aj[bColIdx];
                //if(bColNum == aColNum){
                C[A.num_rows * bColNum + aRowNum] += A.Ax[aColIdx] * B.Ax[bColIdx];
                //std::cout << "aColIdx " << aRowNum << " bColIdx " << bColNum << " C[A.num_rows * bColNum + aRowNum]: " << C[A.num_rows * bColNum + aRowNum]
                //        << " B.Ax[bColIdx]; " << B.Ax[bColIdx] << " A.Ax[aColIdx] "  <<  A.Ax[aColIdx]<< "\n";
                //}
            }

        }
    }
}

/**
 * This Version of procedure multiply matrices A * B assuming that second argument is
 * B' i.e. second arg is B transporated matrix. This significtly facilitate
 * process of multiplication
 */
//template <typename int, typename double>

void __mm_csr_serial_host(const csr_matrix& A,
        const csr_matrix& B_t,
        double *C) {
    //csr_matrix B_t = csr_transpose(B);
    for (int i = 0; i < A.num_rows * B_t.num_rows; i++) {
        C[i] = (double) 0;
    }
    for (int i = 0; i < A.num_rows; i++) {
        const int A_row_start = A.Ap[i];
        const int A_row_end = A.Ap[i + 1];
        for (int jj = A_row_start; jj < A_row_end; jj++) {
            const int j = A.Aj[jj]; //index of j column
            for (int b_t_row_counter = 0; b_t_row_counter < B_t.num_rows; b_t_row_counter++) {
                const int BT_row_start = B_t.Ap[b_t_row_counter];
                const int BT_row_end = B_t.Ap[b_t_row_counter + 1];
                for (int b_t_col_count = BT_row_start; b_t_col_count < BT_row_end; b_t_col_count++) {
                    int b_t_col_idx = B_t.Aj[b_t_col_count];
                    if (j == b_t_col_idx) {
                        C[A.num_rows * b_t_row_counter + i] += A.Ax[jj] * B_t.Ax[b_t_col_count];
                        std::cout << "i " << i << " j " << b_t_row_counter << " tmp Val: "
                                << C[A.num_rows * b_t_row_counter + i] << " + "
                                << A.Ax[jj] * B_t.Ax[b_t_col_count] << "\n";
                    }
                }
            }
        }
    }
}

//template <typename int, typename double>

void del_col_from_csr_mtx(csr_matrix& csr, int col_to_del) {
    csr.num_cols--;
    //std::cout << "\nDel column " << col_to_del << " \n";
    for (int i = 0; i < csr.num_rows; i++) {
        //std::cout << "row >>> " << i << "\n";
        for (int jj = csr.Ap[i]; jj < csr.Ap[i + 1]; jj++) {
            //std::cout << "col " << csr.Aj[jj] << "\n";
            if (csr.Aj[jj] == col_to_del) {
                //std::cout << "del col " << csr.Aj[jj] << " With index " << jj << " nonzer "<< csr.num_nonzeros <<  "\n";
                memmove(&csr.Aj[jj], &csr.Aj[jj + 1], sizeof (int) * (csr.num_nonzeros - (jj + 1)));
                memmove(&csr.Ax[jj], &csr.Ax[jj + 1], sizeof (double) * (csr.num_nonzeros - (jj + 1)));
                csr.num_nonzeros--;
                //printMatrixCPU((int)1, csr.num_nonzeros, csr.Aj);
                for (int j = i + 1; j < csr.num_rows + 1; j++) {
                    csr.Ap[j]--;
                }

                //std::cout << "TMP-------------------------\n";
                //print_csr_matrix<int,double>(csr);
                //std::cout << "TMP---------------------------\n";
            } //else if (csr.Aj[jj] > col_to_del) {
            //  csr.Aj[jj]--;
            //}
        }
    }

    for (int i = 0; i < csr.num_rows; i++) {
        //std::cout << "row >>> " << i << "\n";
        for (int jj = csr.Ap[i]; jj < csr.Ap[i + 1]; jj++) {
            if (csr.Aj[jj] > col_to_del) {
                csr.Aj[jj]--;
            }
        }
    }
}

//template <typename int, typename double>

void del_row_from_csr_mtx(csr_matrix& csr, int row_to_del_idx) {
    //std::cout << "\nDel row " << row_to_del_idx << " \n";
    //std::cout << "\ncsr.Ap[row_to_del_idx + 1] " << csr.Ap[row_to_del_idx + 1] << " \n";
    int row_begin = csr.Ap[row_to_del_idx];
    int row_end = csr.Ap[row_to_del_idx + 1];
    memmove(&csr.Aj[row_begin], &csr.Aj[row_end], sizeof (int) * (csr.num_nonzeros - row_end));
    memmove(&csr.Ax[row_begin], &csr.Ax[row_end], sizeof (double) * (csr.num_nonzeros - row_end));
    //memmove(&csr.Ap[row_to_del_idx], &csr.Ap[row_to_del_idx + 1], sizeof(int) * (csr.num_rows - (row_to_del_idx - 1)));
    int els_in_row = row_end - row_begin;
    for (int i = row_to_del_idx; i < csr.num_rows; i++) {
        csr.Ap[i] = csr.Ap[i + 1] - els_in_row;
        //std::cout << "\ncsr.Ap[i + 1] " << csr.Ap[i + 1] << " i + 1 " <<  i + 1 << " \n";
    }
    csr.num_rows--;
    csr.num_nonzeros -= els_in_row;
}

//template <typename int, typename double>

void transpose_coo_mv(coo_matrix &coo) {
    int swap;
    int rows = coo.num_rows;
    coo.num_rows = coo.num_cols;
    coo.num_cols = rows;
    for (int i = 0; i < coo.num_nonzeros; i++) {
        swap = coo.I[i];
        coo.I[i] = coo.J[i];
        coo.J[i] = swap;
    }
}

/*Эта процедура добавляет к матрице в csr формате строку. Строка представленна
 * в виде coo матрицы
 */
//template <typename int, typename double>

void add_row_to_csr(csr_matrix &csr,
        coo_matrix &coo_row) {
    if (coo_row.num_nonzeros > 0) {
        int lastRowCsrIndex;
        if (csr.num_rows == 0) {
            lastRowCsrIndex = 1;
        } else {
            lastRowCsrIndex = csr.num_rows + 1;
        }
        csr.num_rows++;
        csr.Ap[lastRowCsrIndex] = csr.Ap[lastRowCsrIndex - 1] + coo_row.num_nonzeros;
        csr.num_nonzeros += coo_row.num_nonzeros;

        //csr.Ap[csr.num_rows + 1] = csr.Ap[csr.num_rows] + coo_row.num_nonzeros;
        //std::cout << "Nonzerrows " << coo_row.num_nonzeros << "\n";
        //std::cout << "csr.num_rows + 1 " << csr.num_rows + 1 << " csr.Ap[lastRowCsrIndex] "<< csr.Ap[csr.num_rows + 1] << "\n";
        //std::cout << "Begin " << csr.Ap[lastRowCsrIndex] << "\n";
        for (int i = 0, jj = csr.Ap[lastRowCsrIndex - 1]; i < coo_row.num_nonzeros; i++, jj++) {
            csr.Aj[jj] = coo_row.J[i];
            //std::cout << "JJ " << jj << "\n";
            if (coo_row.J[i] + 1 > csr.num_cols) {
                csr.num_cols = coo_row.J[i] + 1;
            }
            csr.Ax[jj] = coo_row.V[i];
            //std::cout << "add new ell " << csr.num_rows << " " << csr.Aj[jj] << " " << csr.Ax[jj] << "\n";
        }
        //std::cout << "CSR NUM COLS " << csr.num_cols << " nonzerros " << csr.num_nonzeros << " num rows " << csr.num_rows << "\n";
    } else {
        int lastRowCsrIndex;
        if (csr.num_rows == 0) {
            lastRowCsrIndex = 1;
        } else {
            lastRowCsrIndex = csr.num_rows + 1;
        }
        csr.num_rows++;
        csr.Ap[lastRowCsrIndex] = csr.Ap[lastRowCsrIndex - 1];
    }

}

/*void add_row_to_ldl(ldl_matrix& ldl,  coo_matrix& coo_row) {
    ldl.num_rows++;
    
    //print_ldl_matrix(ldl);
    
    for(int coo_i = 0; coo_i < coo_row.num_nonzeros; coo_i++){
        int el_col_idx = coo_row.J[coo_i];
        double el_val = coo_row.V[coo_i];
        //std::cout << "el_val =  " << el_val << " \n";
        int number_of_l_columns = ldl.num_cols - 1;
        ldl.Lp[number_of_l_columns] = ldl.num_nonzeros;
        if(el_col_idx < number_of_l_columns - 1){
            int ldl_next_col_start_idx = ldl.Lp[el_col_idx + 1];
            int num_eld_involved = ldl.num_nonzeros - ldl_next_col_start_idx;
            //std::cout << "el_col_idx =  " << el_col_idx << " index of array to start "<< ldl_next_col_start_idx << " number of nonzeros " << ldl.num_nonzeros << " number of colums " << ldl.num_cols << " \n";
            memmove(&ldl.Li[ldl_next_col_start_idx + 1], &ldl.Li[ldl_next_col_start_idx], sizeof (int) * num_eld_involved);
            memmove(&ldl.Lx[ldl_next_col_start_idx + 1], &ldl.Lx[ldl_next_col_start_idx], sizeof (double) * num_eld_involved);
            ldl.num_nonzeros++;
            ldl.Li[ldl_next_col_start_idx] = ldl.num_rows - 1;
            ldl.Lx[ldl_next_col_start_idx] = el_val;
            for(int ldl_col_i = el_col_idx + 1; ldl_col_i <= number_of_l_columns; ldl_col_i++){
                ldl.Lp[ldl_col_i]++;
            }
        }else if(el_col_idx == number_of_l_columns - 1){
            int ldl_next_col_start_idx = ldl.Lp[el_col_idx + 1];
            ldl.Li[ldl_next_col_start_idx] = ldl.num_rows - 1;
            ldl.Lx[ldl_next_col_start_idx] = el_val;
            
            ldl.Lp[el_col_idx + 1]++;            
            ldl.num_nonzeros++;
            //std::cout << "!el_col_idx =  " << el_col_idx << " index of array to start "<< ldl_next_col_start_idx << " number of nonzeros " << ldl.num_nonzeros << " \n";
        }else if(el_col_idx == number_of_l_columns){
            int ldl_next_col_start_idx = ldl.Lp[el_col_idx];
            ldl.Li[ldl_next_col_start_idx] = ldl.num_rows - 1;
            ldl.Lx[ldl_next_col_start_idx] = el_val;
            
            //ldl.num_cols++;
            std::cout << "Add element to D and ++ col " << el_col_idx << " cols numbr= " <<  ldl.num_cols << " ldl.Li[" << ldl_next_col_start_idx << "]" << ldl.Li[ldl_next_col_start_idx] << "\n";
            ldl.num_nonzeros++;
            ldl.Lp[el_col_idx + 1] = ldl.num_nonzeros;            
            //std::cout << "!!el_col_idx =  " << el_col_idx << " index of array to start "<< ldl_next_col_start_idx << " number of nonzeros " << ldl.num_nonzeros << " \n";
        }
        
    }
    ldl.num_cols++;
    //std::cout << "Add element to D mtx on index " << coo_row.num_cols << "With value =" << coo_row.V[coo_row.num_nonzeros - 1] << "\n";
    ldl.D[ldl.num_cols - 1] = coo_row.V[coo_row.num_nonzeros - 1];
    //print_ldl_matrix(ldl);
}*/

void add_row_to_ldl(ldl_matrix& ldl, coo_matrix& coo_row) {
    ldl.num_rows++;
    ldl.num_cols++;

    int l_num_rows = ldl.num_rows - 1;
    int l_num_cols = ldl.num_cols - 1;

    ldl.Lp[l_num_cols] = ldl.Lp[l_num_cols - 1];
    //std::cout << "ldl.Lp[" << l_num_cols  << "] = " << ldl.Lp[l_num_cols] << "\n";
    for (int nzr_i = 0; nzr_i < coo_row.num_nonzeros - 1; nzr_i++) {
        int el_col_idx = coo_row.J[nzr_i];
        //if (el_col_idx < l_num_cols) {
        double el_val = coo_row.V[nzr_i];
        int ldl_next_col_start_idx = ldl.Lp[el_col_idx + 1];
        int num_eld_involved = ldl.num_nonzeros - ldl_next_col_start_idx;
        //std::cout << "el_col_idx =  " << el_col_idx << " index of array to start " << ldl_next_col_start_idx << " number of nonzeros " << ldl.num_nonzeros << " number l colums " << l_num_cols << " \n";
        memmove(&ldl.Li[ldl_next_col_start_idx + 1], &ldl.Li[ldl_next_col_start_idx], sizeof (int) * num_eld_involved);
        memmove(&ldl.Lx[ldl_next_col_start_idx + 1], &ldl.Lx[ldl_next_col_start_idx], sizeof (double) * num_eld_involved);
        ldl.num_nonzeros++;
        ldl.Li[ldl_next_col_start_idx] = l_num_rows;
        ldl.Lx[ldl_next_col_start_idx] = el_val;
        for (int ldl_col_i = el_col_idx + 1; ldl_col_i <= l_num_cols; ldl_col_i++) {
            ldl.Lp[ldl_col_i]++;
        }
        //}
    }
    ldl.D[ldl.num_cols - 1] = coo_row.V[coo_row.num_nonzeros - 1];
}

/*
void add_col_to_ldl(ldl_matrix &ldl,
        coo_matrix &coo_col) {
    if (coo_col.num_nonzeros > 0) {
        int lastRowCsrIndex;
        if (ldl.num_cols == 0) {
            lastRowCsrIndex = 1;
        } else {
            lastRowCsrIndex = ldl.num_cols + 1;
        }
        ldl.num_cols++;
        ldl.Lp[lastRowCsrIndex] = ldl.Lp[lastRowCsrIndex - 1] + coo_col.num_nonzeros;
        ldl.num_nonzeros += coo_col.num_nonzeros;

        //csr.Ap[csr.num_rows + 1] = csr.Ap[csr.num_rows] + coo_row.num_nonzeros;
        //std::cout << "Nonzerrows " << coo_row.num_nonzeros << "\n";
        //std::cout << "csr.num_rows + 1 " << csr.num_rows + 1 << " csr.Ap[lastRowCsrIndex] "<< csr.Ap[csr.num_rows + 1] << "\n";
        //std::cout << "Begin " << csr.Ap[lastRowCsrIndex] << "\n";
        for (int i = 0, jj = ldl.Lp[lastRowCsrIndex - 1]; i < coo_col.num_nonzeros - 1; i++, jj++) {
            ldl.Li[jj] = coo_col.J[i];
            //std::cout << "JJ " << jj << "\n";
            if (coo_col.J[i] + 1 > ldl.num_rows) {
                ldl.num_rows = coo_col.J[i] + 1;
            }
            ldl.Lx[jj] = coo_col.V[i];
            //std::cout << "add new ell " << csr.num_rows << " " << csr.Aj[jj] << " " << csr.Ax[jj] << "\n";
        }
        //std::cout << "CSR NUM COLS " << csr.num_cols << " nonzerros " << csr.num_nonzeros << " num rows " << csr.num_rows << "\n";
    }
    
    ldl.D[ldl.num_cols - 1] = coo_col.V[coo_col.num_nonzeros - 1];
}
 */


void add_col_to_ldl(ldl_matrix &ldl, coo_matrix &coo_col) {
    int begin_idx = ldl.Lp[ldl.num_cols];

    if (coo_col.num_nonzeros > 0) {
        for (int coo_i = 0; coo_i < coo_col.num_nonzeros - 1; coo_i++) {
            ldl.Li[begin_idx + coo_i] = coo_col.I[coo_i];
            ldl.Lx[begin_idx + coo_i] = coo_col.V[coo_i];
            //if (coo_col.I[coo_i] + 1 > ldl.num_rows) {
            //    ldl.num_rows = coo_col.I[coo_i] + 1;
            //}
            ldl.num_nonzeros++;
        }

    }

    ldl.num_cols++;
    ldl.num_rows = ldl.num_cols; //becouse of simmetry
    ldl.Lp[ldl.num_cols] = begin_idx + coo_col.num_nonzeros - 1;
    ldl.D[ldl.num_cols - 1] = coo_col.V[coo_col.num_nonzeros - 1];
}

/**
 * Compute y = A^t * x
 */
//template <typename int, typename double>

void spmv_csr_t_serial_host(const csr_matrix& csr,
        const double * x,
        double * y) {
    for (int i = 0; i < csr.num_cols; i++) {
        y[i] = 0;
    }
    for (int i = 0; i < csr.num_rows; i++) {
        const int row_start = csr.Ap[i];
        const int row_end = csr.Ap[i + 1];
        //double sum = y[i];
        for (int jj = row_start; jj < row_end; jj++) {
            const int j = csr.Aj[jj]; //column index
            y[j] += csr.Ax[jj] * x[i];
        }
    }
}

void A_dot_x_csc_serial_host(const csc_matrix& csc,
        const double * x,
        double * y) {
    for (int i = 0; i < csc.num_cols; i++) {
        y[i] = 0;
    }
    for (int col_i = 0; col_i < csc.num_cols; col_i++) {
        const int col_start = csc.Cp[col_i];
        const int col_end = csc.Cp[col_i + 1];
        //double sum = y[i];
        for (int row_i = col_start; row_i < col_end; row_i++) {
            const int row = csc.Ri[row_i]; //column index
            y[col_i] += csc.Ex[row_i] * x[row];
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Compute y += A*x for a sparse CSR matrix A and column vectors x and y
//! @param num_rows   number of rows in A
//! @param Ap         CSR pointer array
//! @param Aj         CSR index array
//! @param Ax         CSR data array
//! @param x          column vector
//! @param y          column vector
////////////////////////////////////////////////////////////////////////////////

//template <typename int, typename double>

void __spmv_csr_serial_host(const int num_rows,
        const int * Ap,
        const int * Aj,
        const double * Ax,
        const double * x,
        double * y) {
    for (int i = 0; i < num_rows; i++) {
        int row_start = Ap[i];
        int row_end = Ap[i + 1];
        double sum = y[i];
        for (int jj = row_start; jj < row_end; jj++) {
            int j = Aj[jj]; //column index
            std::cout << "J  = " << j << " jj= " << jj << " \n";
            sum += x[j] * Ax[jj];
        }
        y[i] = sum;
    }
}

//template <typename int, typename double>

void spmv_csr_serial_host(const csr_matrix& csr,
        const double * x,
        double * y) {
    for (int y_i = 0; y_i < csr.num_rows; y_i++) {
        y[y_i] = 0.0;
    }//;cblas_dscal(csr.num_rows, 0.0, y, 1);
    //__spmv_csr_serial_host(csr.num_rows, csr.Ap, csr.Aj, csr.Ax, x, y);
    for (int i = 0; i < csr.num_rows; i++) {
        int row_start = csr.Ap[i];
        int row_end = csr.Ap[i + 1];
        double sum = y[i];
        for (int jj = row_start; jj < row_end; jj++) {
            int j = csr.Aj[jj]; //column index
            //std::cout << "J  = " << j << " jj= " << jj << " \n";
            sum += x[j] * csr.Ax[jj];
        }
        y[i] = sum;
    }
}

void spmv_scr_t_coo_serial_host(const csr_matrix& csr_t,
        const coo_matrix x,
        double * y) {
    for (int y_i = 0; y_i < csr_t.num_cols; y_i++) {
        y[y_i] = 0.0;
    }
    for (int coo_i = 0; coo_i < x.num_nonzeros; coo_i++) {
        int x_col = x.I[coo_i];
        double x_val = x.V[coo_i];
        int row_start = csr_t.Ap[x_col];
        int row_end = csr_t.Ap[x_col + 1];
        for (int csr_row_i = row_start; csr_row_i < row_end; csr_row_i++) {
            int csr_col = csr_t.Aj[ csr_row_i ]; //column index
            double csr_val = csr_t.Ax[csr_row_i];

            y[csr_col] += csr_val * x_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Convert COO format to CSR format
// Storage for output is assumed to have been allocated
//! @param rows           COO row array
//! @param cols           COO column array
//! @param data           COO data array
//! @param num_rows       number of rows
//! @param num_cols       number of columns
//! @param num_nonzeros   number of nonzeros
//! @param Ap             CSR pointer array
//! @param Ai             CSR index array
//! @param Ax             CSR data array
////////////////////////////////////////////////////////////////////////////////

//template <class int, class double>

void coo2csr(const int * rows,
        const int * cols,
        const double * data,
        const int num_rows,
        const int num_cols,
        const int num_nonzeros,
        int * Ap,
        int * Aj,
        double * Ax) {
    for (int i = 0; i < num_rows; i++)
        Ap[i] = 0;

    for (int i = 0; i < num_nonzeros; i++)
        Ap[rows[i]]++;


    //cumsum the nnz per row to get Bp[]
    for (int i = 0, cumsum = 0; i < num_rows; i++) {
        int temp = Ap[i];
        Ap[i] = cumsum;
        cumsum += temp;
    }
    Ap[num_rows] = num_nonzeros;

    //write Aj,Ax into Bj,Bx
    for (int i = 0; i < num_nonzeros; i++) {
        int row = rows[i];
        int dest = Ap[row];

        Aj[dest] = cols[i];
        Ax[dest] = data[i];

        Ap[row]++;
    }

    for (int i = 0, last = 0; i <= num_rows; i++) {
        int temp = Ap[i];
        Ap[i] = last;
        last = temp;
    }

}


////////////////////////////////////////////////////////////////////////////////
//! Convert COOrdinate format (triplet) to CSR format
//! @param coo        coo_matrix
////////////////////////////////////////////////////////////////////////////////

//template <class int, class double>

csr_matrix coo2csr(const coo_matrix& coo) {
    bool compact = false;
    csr_matrix csr;

    csr.num_rows = coo.num_rows;
    csr.num_cols = coo.num_cols;
    csr.num_nonzeros = coo.num_nonzeros;

    csr.Ap = new_host_iarray(csr.num_rows + 1);
    csr.Aj = new_host_iarray(csr.num_nonzeros);
    csr.Ax = new_host_darray(csr.num_nonzeros);

    coo2csr(coo.I, coo.J, coo.V,
            coo.num_rows, coo.num_cols, coo.num_nonzeros,
            csr.Ap, csr.Aj, csr.Ax);

    if (compact) {
        //sum duplicates together
        sum_csr_duplicates(csr.num_rows, csr.num_cols, csr.Ap, csr.Aj, csr.Ax);
        csr.num_nonzeros = csr.Ap[csr.num_rows];
    }

    return csr;
}



////////////////////////////////////////////////////////////////////////////////
//! Transpose a matrix in CSR format
//! Storage for B is assumed to have been allocated
//! @param Ap         CSR pointer array
//! @param Aj         CSR column index array
//! @param Ax         CSR data array
//! @param num_rows   number of rows in A
//! @param num_cols   number of columns in A
//! @param Bp         CSR pointer array
//! @param Bi         CSR column index array
//! @param Bx         CSR data array
////////////////////////////////////////////////////////////////////////////////

//template <class int, class double>

void csr_transpose_fix(const int * Ap,
        const int * Aj,
        const double * Ax,
        const int num_rows,
        const int num_cols,
        int * Bp,
        int * Bj,
        double * Bx) {
    //TODO use temp-free method
    int * temp = new_host_iarray(num_cols);

    for (int i = 0; i < num_cols; i++) {
        temp[i] = 0;
    }

    int num_nonzeros = Ap[num_rows];

    for (int i = 0; i < num_nonzeros; i++) //count number of entries in each column
        temp[Aj[i]]++;

    Bp[0] = 0;
    for (int i = 0; i < num_cols; i++) {
        Bp[i + 1] = Bp[i] + temp[i]; //cumsum number column entries to form Bp
        temp[i] = 0; //count number of entries in each column
    }

    for (int i = 0; i < num_rows; i++) {
        int row_start = Ap[i];
        int row_end = Ap[i + 1];
        for (int jj = row_start; jj < row_end; jj++) {
            int col = Aj[jj];
            int offset = temp[col] + Bp[col];

            Bj[offset] = i;
            Bx[offset] = Ax[jj];

            temp[col]++;
        }
    }

    free(temp);
}

//template <typename int, typename double>

csr_matrix csr_transpose_fix(const csr_matrix& csr) {
    csr_matrix csr_t;

    csr_t.num_rows = csr.num_cols;
    csr_t.num_cols = csr.num_rows;
    csr_t.num_nonzeros = csr.num_nonzeros;

    csr_t.Ap = new_host_iarray(csr.num_cols + 1);
    csr_t.Aj = new_host_iarray(csr.num_nonzeros + 1);
    csr_t.Ax = new_host_darray(csr.num_nonzeros + 1);
    for (int i = 0; i < csr.num_cols + 1; i++) {
        csr_t.Ap[i] = 0;
    }
    for (int i = 0; i < csr.num_nonzeros + 1; i++) {
        csr_t.Aj[i] = 0;
        csr_t.Ax[i] = 0.0;
    }
    csr_transpose(csr.Ap, csr.Aj, csr.Ax,
            csr.num_rows, csr.num_cols,
            csr_t.Ap, csr_t.Aj, csr_t.Ax);

    return csr_t;
}


////////////////////////////////////////////////////////////////////////////////
//! Sum together the duplicate nonzeros in a CSR format
//! CSR format will be modified *in place*
//! @param num_rows       number of rows
//! @param num_cols       number of columns
//! @param Ap             CSR pointer array
//! @param Ai             CSR index array
//! @param Ax             CSR data array
////////////////////////////////////////////////////////////////////////////////

//template <class int, class double>

void sum_csr_duplicates(const int num_rows,
        const int num_cols,
        int * Ap,
        int * Aj,
        double * Ax) {
    int * next = new_host_iarray(num_cols);
    double * sums = new_host_darray(num_cols);

    for (int i = 0; i < num_cols; i++) {
        next[i] = (int) - 1;
        sums[i] = (double) 0;
    }

    int NNZ = 0;

    int row_start = 0;
    int row_end = 0;


    for (int i = 0; i < num_rows; i++) {
        int head = (int) - 2;

        row_start = row_end; //Ap[i] may have been changed
        row_end = Ap[i + 1]; //Ap[i+1] is safe

        for (int jj = row_start; jj < row_end; jj++) {
            int j = Aj[jj];

            sums[j] += Ax[jj];
            if (next[j] == (int) - 1) {
                next[j] = head;
                head = j;
            }
        }


        while (head != (int) - 2) {
            int curr = head; //current column
            head = next[curr];

            if (sums[curr] != 0) {
                Aj[NNZ] = curr;
                Ax[NNZ] = sums[curr];
                NNZ++;
            }

            next[curr] = (int) - 1;
            sums[curr] = 0;
        }
        Ap[i + 1] = NNZ;
    }

    free(next);
    free(sums);
}

//template <class int, class double>

void sum_csr_duplicates(csr_matrix& A) {
    sum_csr_duplicates(A.num_rows, A.num_cols, A.Ap, A.Aj, A.Ax);
    A.num_nonzeros = A.Ap[A.num_rows];
}



////////////////////////////////////////////////////////////////////////////////
//! Transpose a matrix in CSR format
//! Storage for B is assumed to have been allocated
//! @param Ap         CSR pointer array
//! @param Aj         CSR column index array
//! @param Ax         CSR data array
//! @param num_rows   number of rows in A
//! @param num_cols   number of columns in A
//! @param Bp         CSR pointer array
//! @param Bi         CSR column index array
//! @param Bx         CSR data array
////////////////////////////////////////////////////////////////////////////////

//template <class int, class double>

void csr_transpose(const int * Ap,
        const int * Aj,
        const double * Ax,
        const int num_rows,
        const int num_cols,
        int * Bp,
        int * Bj,
        double * Bx) {
    //TODO use temp-free method
    int * temp = new_host_iarray(num_cols);

    for (int i = 0; i < num_cols; i++) {
        temp[i] = 0;
    }

    int num_nonzeros = Ap[num_rows];

    for (int i = 0; i < num_nonzeros; i++) { //count number of entries in each column
        //if(Aj[i] > num_cols - 1){
        //    std::cout<< "Aj[ " << i << " ] = " << Aj[i] << " Num cols: "<<  num_cols<<"\n";
        //}
        temp[Aj[i]]++;
    }


    Bp[0] = 0;
    for (int i = 0; i < num_cols; i++) {
        Bp[i + 1] = Bp[i] + temp[i]; //cumsum number column entries to form Bp
        temp[i] = 0; //count number of entries in each column
    }

    for (int i = 0; i < num_rows; i++) {
        int row_start = Ap[i];
        int row_end = Ap[i + 1];
        for (int jj = row_start; jj < row_end; jj++) {
            int col = Aj[jj];
            int offset = temp[col] + Bp[col];

            Bj[offset] = i;
            Bx[offset] = Ax[jj];

            temp[col]++;
        }
    }

    free(temp);
}

ldl_matrix ldl_transpose(const ldl_matrix& ldl, ldl_matrix& ldl_t) {
    //ldl_matrix ldl_t;

    ldl_t.num_nonzeros = 0;
    ldl_t.num_cols = 0;
    ldl_t.num_rows = 0;

    ldl_t.num_rows = ldl.num_cols;
    ldl_t.num_cols = ldl.num_rows;
    ldl_t.num_nonzeros = ldl.num_nonzeros;

    /*
    ldl_t.D = new_host_darray(ldl.MAX_D_LP_SIZE);
    ldl_t.Lp = new_host_iarray(ldl.MAX_D_LP_SIZE);
    ldl_t.Li = new_host_iarray(ldl.MAX_Li_Lx_SIZE);
    ldl_t.Lx = new_host_darray(ldl.MAX_Li_Lx_SIZE);
    ldl_t.Lnz = new_host_iarray(ldl.MAX_D_LP_SIZE);
    ldl_t.Parent = new_host_iarray(ldl.MAX_D_LP_SIZE);
     */
    cblas_dcopy(ldl.num_cols, ldl.D, 1, ldl_t.D, 1);

    csr_transpose(ldl.Lp, ldl.Li, ldl.Lx,
            //ldl.num_rows, ldl.num_cols,
            ldl.num_cols, ldl.num_rows,
            ldl_t.Lp, ldl_t.Li, ldl_t.Lx);
    return ldl_t;
}

//template <typename int, typename double>

csr_matrix csr_transpose(const csr_matrix& csr) {
    csr_matrix csr_t;

    csr_t.num_rows = csr.num_cols;
    csr_t.num_cols = csr.num_rows;
    csr_t.num_nonzeros = csr.num_nonzeros;

    csr_t.Ap = new_host_iarray(csr.num_cols + 1);
    csr_t.Aj = new_host_iarray(csr.num_nonzeros);
    csr_t.Ax = new_host_darray(csr.num_nonzeros);

    csr_transpose(csr.Ap, csr.Aj, csr.Ax,
            csr.num_rows, csr.num_cols,
            csr_t.Ap, csr_t.Aj, csr_t.Ax);

    return csr_t;
}

void csr_transpose_mv(const csr_matrix& csr, csr_matrix& csr_t) {
    //csr_matrix csr_t;

    csr_t.num_rows = csr.num_cols;
    csr_t.num_cols = csr.num_rows;
    csr_t.num_nonzeros = csr.num_nonzeros;

    //csr_t.Ap = new_host_iarray(csr.num_cols + 1);
    //csr_t.Aj = new_host_iarray(csr.num_nonzeros);
    //csr_t.Ax = new_host_darray(csr.num_nonzeros);

    csr_transpose(csr.Ap, csr.Aj, csr.Ax,
            csr.num_rows, csr.num_cols,
            csr_t.Ap, csr_t.Aj, csr_t.Ax);

    //return csr_t;
}

void ldl_lsolve_t
(
        int n, /* L is n-by-n, where n >= 0 */
        double X [ ], /* size n.  right-hand-side on input, soln. on output */
        int Lp [ ], /* input of size n+1, not modified */
        int Li [ ], /* input of size lnz=Lp[n], not modified */
        double Lx [ ] /* input of size lnz=Lp[n], not modified */
        ) {
    int j, p, p2;
    //for (j = 0 ; j < n ; j++)
    for (j = n - 1; j >= 0; j--) {
        p2 = Lp [j + 1];
        //for (p = Lp [j] ; p < p2 ; p++)
        for (p = p2 - 1; p >= Lp[j]; p--) {
            //X [Li [p]] -= Lx [p] * X [j] ;
            X [Li [p]] -= Lx [p] * X [j];
        }
    }
}


/* ========================================================================== */
/* === ldl_dsolve:  solve Dx=b ============================================== */

/* ========================================================================== */

void ldl_dsolve_t
(
        int n, /* D is n-by-n, where n >= 0 */
        double X [ ], /* size n.  right-hand-side on input, soln. on output */
        double D [ ] /* input of size n, not modified */
        ) {
    int j;
    for (j = 0; j < n; j++) {
        X [j] /= D [j];
    }
}


/* ========================================================================== */
/* === ldl_ltsolve: solve L'x=b  ============================================ */

/* ========================================================================== */

void ldl_ltsolve_t
(
        int n, /* L is n-by-n, where n >= 0 */
        double X [ ], /* size n.  right-hand-side on input, soln. on output */
        int Lp [ ], /* input of size n+1, not modified */
        int Li [ ], /* input of size lnz=Lp[n], not modified */
        double Lx [ ] /* input of size lnz=Lp[n], not modified */
        ) {
    int j, p, p2;
    //for (j = n-1 ; j >= 0 ; j--)
    for (j = 0; j < n; j++) {
        p2 = Lp [j + 1];
        for (p = Lp [j]; p < p2; p++) {
            X [j] -= Lx [p] * X [Li [p]];
        }
    }
}

/**Get copy of entire csr matrix
 * csr - matrix that contain elements
 */
//template <typename int, typename double>

csr_matrix getCsrMtxCopy(const csr_matrix& csr) {

    csr_matrix csr_copy;

    csr_copy.num_rows = csr.num_rows;
    csr_copy.num_cols = csr.num_cols;
    csr_copy.num_nonzeros = csr.num_nonzeros;

    csr_copy.Ap = new_host_iarray(csr.num_rows + 1);
    csr_copy.Aj = new_host_iarray(csr.num_nonzeros * 2);
    csr_copy.Ax = new_host_darray(csr.num_nonzeros * 2);

    memmove(csr_copy.Ap, csr.Ap, (csr.num_rows + 1) * sizeof (int));
    memmove(csr_copy.Aj, csr.Aj, (csr.num_nonzeros + 1) * sizeof (int));
    memmove(csr_copy.Ax, csr.Ax, (csr.num_nonzeros + 1) * sizeof (double));

    return csr_copy;
}

//template <typename int, typename double>

csr_matrix convCsrToDense(const csr_matrix& csr, double* denseMtx) {
    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = 0; j < csr.num_cols; j++) {
            denseMtx[j * csr.num_rows + i] = 0;
        }
        //std::cout << "\n ";
    }

    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            denseMtx[csr.Aj[j] * csr.num_rows + i] = csr.Ax[j];
        }
    }
}

//template <typename int, typename double>

coo_matrix
convDenseToCooNaiv(double* denseMtx, int num_rows, int num_cols) {
    coo_matrix coo;

    coo.num_rows = num_rows;
    coo.num_cols = num_cols;
    coo.num_nonzeros = num_cols * num_rows;

    coo.I = new_host_iarray(coo.num_nonzeros);
    coo.J = new_host_iarray(coo.num_nonzeros);
    coo.V = new_host_darray(coo.num_nonzeros);

    int nonzIdx = 0;
    for (int i = 0; i < coo.num_rows; i++) {

        for (int j = 0; j < coo.num_cols; j++) {
            coo.I[nonzIdx] = i;
            coo.J[nonzIdx] = j;
            coo.V[nonzIdx] = denseMtx[j * coo.num_rows + i];
            nonzIdx++;
        }
        //std::cout << "\n ";
    }


    return coo;
}

coo_matrix
convDenseColToCooCol(double* denseCol, int num_els) {
    coo_matrix coo;

    int nonzeros_in_dense_col = 0;
    for (int dens_col_i = 0; dens_col_i < num_els; dens_col_i++) {
        if (denseCol[dens_col_i] != 0.0) {
            nonzeros_in_dense_col++;
        }
    }

    coo.num_rows = num_els;
    coo.num_cols = 1;
    coo.num_nonzeros = nonzeros_in_dense_col;

    coo.I = new_host_iarray(coo.num_nonzeros);
    coo.J = new_host_iarray(coo.num_nonzeros);
    coo.V = new_host_darray(coo.num_nonzeros);

    int nonzIdx = 0;
    for (int dens_col_i = 0; dens_col_i < num_els; dens_col_i++) {
        if (denseCol[dens_col_i] != 0.0) {
            coo.I[nonzIdx] = dens_col_i;
            coo.J[nonzIdx] = 0;
            coo.V[nonzIdx] = denseCol[dens_col_i];
            nonzIdx++;
        }
    }
    return coo;
}

coo_matrix convDenseRowToCooRow(double* denseRow, int num_els) {
    coo_matrix coo;

    int nonzeros_in_dense_row = 0;
    for (int dens_col_i = 0; dens_col_i < num_els; dens_col_i++) {
        if (denseRow[dens_col_i] != 0.0) {
            nonzeros_in_dense_row++;
        }
    }

    coo.num_rows = 1;
    coo.num_cols = num_els;
    coo.num_nonzeros = nonzeros_in_dense_row;

    coo.I = new_host_iarray(coo.num_nonzeros);
    coo.J = new_host_iarray(coo.num_nonzeros);
    coo.V = new_host_darray(coo.num_nonzeros);

    int nonzIdx = 0;
    for (int dens_col_i = 0; dens_col_i < num_els; dens_col_i++) {
        if (denseRow[dens_col_i] != 0.0) {
            coo.I[nonzIdx] = 0;
            coo.J[nonzIdx] = dens_col_i;
            coo.V[nonzIdx] = denseRow[dens_col_i];
            nonzIdx++;
        }
    }
    return coo;
}
//template <typename int, typename double>

csr_matrix scaleColInCsrMtx(const csr_matrix& csr, double scaleParam, int colIdx) {
    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            int AcolId = csr.Aj[j];
            if (AcolId == colIdx) {
                csr.Ax[j] *= scaleParam;
            }
        }
    }
}



//template <typename int, typename double>

void scaleRowInCsrMtx(const csr_matrix& csr, double scaleParam, int rowIdx) {
    for (int j = csr.Ap[rowIdx]; j < csr.Ap[rowIdx + 1]; j++) {
        csr.Ax[j] *= scaleParam;
    }

}

void scale_col_in_csc_mtx(const csc_matrix& csc, double scaleParam, int col_idx) {
    for (int j = csc.Cp[col_idx]; j < csc.Cp[col_idx + 1]; j++) {
        csc.Ex[j] *= scaleParam;
    }

}

csr_matrix scaleColInCsrMtx_AndAdd2Shift(const csr_matrix& csr, double scaleParam, int colIdx, double shift) {
    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = csr.Ap[i]; j < csr.Ap[i + 1]; j++) {
            int AcolId = csr.Aj[j];
            if (AcolId == colIdx) {
                csr.Ax[j] *= scaleParam;
            }
        }
    }
    for (int j = csr.Ap[csr.num_rows - 1 ]; j < csr.Ap[csr.num_rows]; j++) {
        int AcolId = csr.Aj[j];
        if (AcolId == colIdx) {
            csr.Ax[j] += 2 * shift;
        }
    }
}

void scaleRowInCsrMtx_AndAdd2Shift(const csr_matrix& csr, double scaleParam, int rowIdx, double shift) {
    for (int j = csr.Ap[rowIdx]; j < csr.Ap[rowIdx + 1]; j++) {
        csr.Ax[j] *= scaleParam;
    }
    csr.Ax[csr.Ap[rowIdx + 1] - 1 ] += 2 * shift;

}


/*
 * Добавляет к матрице в формате csr столбец в формате coo
 * Предполагается, что предварительно выделено памяти для добавления
 * и добавление происходит путём смещения областей памяти в массиве столбцов и
 * значений, в освободившиеся участки втавляются значения.
 */
//template <typename int, typename double>

void add_first_col_to_csr_mtx(csr_matrix& csr,
        coo_matrix& coo_column) {
    csr.num_cols++;
    int row_to_add;
    //std::cout << "Row Numbers " << coo_column.num_rows << " \n";
    for (int i = 0; i < coo_column.num_rows; i++) {
        csr.num_nonzeros++;
        row_to_add = coo_column.I[i];
        //std::cout << "add row " << row_to_add << " \n";
        //int el_count = csr.num_nonzeros - csr.Ap[row_to_add + 1];
        if ((row_to_add + 1) != csr.num_rows) {
            //std::cout << "muve n =  " << csr.num_nonzeros - csr.Ap[row_to_add + 1] << " elements " << csr.Aj[csr.Ap[row_to_add + 1] + 1] << " \n";
            memmove(&csr.Aj[1], &csr.Aj[0], sizeof (int) * (csr.num_nonzeros));
            memmove(&csr.Ax[1], &csr.Ax[0], sizeof (double) * (csr.num_nonzeros));
        }
        csr.Aj[csr.Ap[row_to_add + 1]] = csr.num_cols - 1;
        csr.Ax[csr.Ap[row_to_add + 1]] = coo_column.V[i];
        //std::cout <<" wtite to position " << csr.Ap[row_to_add + 1] <<  " \n";
        for (int j = row_to_add + 1; j < csr.num_rows + 1; j++) {
            csr.Ap[j]++;
            //std::cout << ">" << csr.Ap[j] << "\n";
        }
    }
}

/**
 * Multiply GrammaMatrix on vector.
 * The matrix is given in form of basis' and basis, x vector is x
 * and output vector that is b/
 */

/*This is all the same as for simple gramma matrix becouse matrix of gramma is SYMMETRY
 * void mulGrammTransposedToX(const csr_matrix& basis_t,
        const csr_matrix& basis,
        double *x, double *p ) {
    std::cout << "basis_t.num_rows: " << basis_t.num_rows << "\n";
    for (int i = 0; i < basis_t.num_rows; i++) {
        p[i] = 0.0;
    }
    double *bas_t_row = new_host_darray(basis_t.num_cols);
    double *gramm_row = new_host_darray(basis_t.num_rows);

    for(int bas_t_row_idx = 0; bas_t_row_idx < basis_t.num_rows; bas_t_row_idx++){
        __get_dense_column(basis_t, bas_t_row_idx, bas_t_row);
        for(int gr_i = 0; gr_i < basis_t.num_rows; gr_i++){gramm_row[gr_i] = 0.0;}
        spmv_csr_serial_host(basis_t, bas_t_row, gramm_row);        
        p[bas_t_row_idx] = cblas_dot(basis_t.num_rows, x, 1, gramm_row, 1);
    }
}*/


double evalRowElemOfGrammMtx(csr_matrix basis_t, int A_row_idx, int B_row_idx) {
    int A_row_start = basis_t.Ap[A_row_idx];
    int A_row_end = basis_t.Ap[A_row_idx + 1];

    int B_row_start = basis_t.Ap[B_row_idx];
    int B_row_end = basis_t.Ap[B_row_idx + 1];

    int A_activ_cols = A_row_end - A_row_start;
    int B_activ_cols = B_row_end - B_row_start;

    int row_iter_max = A_activ_cols + B_activ_cols;

    double gramm_row_el = 0.0;

    for (int i = 0, A_iter = 0, B_iter = 0; i < row_iter_max; i++) {
        int A_col_i = basis_t.Aj[A_row_start + A_iter];
        int B_col_i = basis_t.Aj[B_row_start + B_iter];
        if (A_col_i == B_col_i) {
            //multiplication
            double A_i_val = basis_t.Ax[A_row_start + A_iter];
            double B_i_val = basis_t.Ax[B_row_start + B_iter];
            gramm_row_el += A_i_val * B_i_val;
            if ((B_iter < B_activ_cols - 1) && (A_iter < A_activ_cols - 1)) {
                A_iter++;
                B_iter++;
            } else {
                break;
            }
        } else if (A_col_i > B_col_i) {
            if (B_iter < B_activ_cols - 1) {
                B_iter++;
            } else {
                break;
            }
        } else {
            if (A_iter < A_activ_cols - 1) {
                A_iter++;
            } else {
                break;
            }
        }
    }
    return gramm_row_el;
}

void evalRowOfGrammMtx(csr_matrix basis_t, int row_idx, double* gramm_row) {
    //double* gramm_row = new_host_darray(basis_t.num_rows);
    for (int el_i = 0; el_i < basis_t.num_rows; el_i++) {
        gramm_row[el_i] = 0;
    }

    for (int el_i = 0; el_i < basis_t.num_rows; el_i++) {
        gramm_row[el_i] = evalRowElemOfGrammMtx(basis_t, row_idx, el_i);
    }
}

/**
 * Multiply Transposed GrammaMatrix on vector.
 * The matrix is given in form of basis' and basis, x vector is x
 * and output vector that is b/
 */
void mulGrammToX(const csr_matrix& basis_t,
        const csr_matrix& basis,
        double *x, double *p) {
    //std::cout << "basis_t.num_rows: " << basis_t.num_rows << "\n";
    for (int i = 0; i < basis_t.num_rows; i++) {
        p[i] = 0.0;
    }
    double *bas_t_row = new_host_darray(basis_t.num_cols);
    double *gramm_row = new_host_darray(basis_t.num_rows);

    for (int bas_t_row_idx = 0; bas_t_row_idx < basis_t.num_rows; bas_t_row_idx++) {
        //this mingled invocation simply get row of basis_t
        __get_dense_column(basis_t, bas_t_row_idx, bas_t_row);
        for (int gr_i = 0; gr_i < basis_t.num_rows; gr_i++) {
            gramm_row[gr_i] = 0.0;
        }
        spmv_csr_serial_host(basis_t, bas_t_row, gramm_row);
        //evalRowOfGrammMtx(basis_t, bas_t_row_idx, gramm_row);
        //std::cout << ">";
        p[bas_t_row_idx] = cblas_ddot(basis_t.num_rows, x, 1, gramm_row, 1);
    }

    free(bas_t_row);
    free(gramm_row);
}

double* eval_csc_cols_norms(csc_matrix csc) {
    double *norms = new_host_darray(csc.num_cols);
    for (int col_i = 0; col_i < csc.num_cols; col_i++) {
        int bgn_idx = csc.Cp[col_i];
        int end_idx = csc.Cp[col_i + 1];
        double qudr_summ = 0.0;
        for (int row_i = bgn_idx; row_i < end_idx; row_i++) {
            qudr_summ += csc.Ex[row_i] * csc.Ex[row_i];
        }
        norms[col_i] = sqrt(qudr_summ);
    }
    return norms;
}