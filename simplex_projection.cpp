#include "simplex_projection.hpp"
#include "overloaded_cblas.hpp"
/**
 * Данная программа решает задачу min 1/2 ||x||^2, Ax <= b
 */
//template <typename int, typename double>

void getSimplexProjection(csc_matrix &inputSet,
        double * minNormVector, double tollerance,
        int* kvec_in, int* basisVecInx_in, int &baselen_in) {
    int maxIter = 10000; //inSetDim;
    double eps = tollerance; //0.001;
    int numEqCon = 0;
    //coo_matrix shift_col = getShiftCoo(inputSet.num_rows, -1.0);
    //add_col_to_csr_mtx(inputSet, shift_col);
    //inSetDim++;
    //так как базис это разреженная матрица, то надо оценить количество ненулевых элементов
    //для векторов базиса которое равно размерности протранства

    int maxVectorsInBasis = inputSet.num_cols + 2; //+ 2 так как когда размер базиса равен размерности вектора + 1, то к базису добавляем ещё и mu
    if (inputSet.num_cols > (inputSet.num_rows + 2)) {
        maxVectorsInBasis = inputSet.num_rows + 2;

    };
    std::cout << "Number of nonzeros in inSet=" << inputSet.num_nonzeros << "\n";
    std::cout << "MaxVectorsInBasis = " << maxVectorsInBasis << "\n";

    //Вычислим массив норм Евклида для всех столбцов исходного множества
    double* norms = eval_csc_cols_norms(inputSet);

    //массив базисных векторов, вектора задаются своими индексами
    int *kvec = new_host_iarray(inputSet.num_cols + 1); //(int*) malloc(maxVectorsInBasis * sizeof (int)); //вектор принадлежности к базису [0 - не принадлежит,1 - принадлежит]
    int *basisVecInx = new_host_iarray(inputSet.num_cols + 1); //(int*) malloc(maxVectorsInBasis * sizeof (int)); //вектор индексов

    int estim_nonzer_in_basis = estimate_max_nonzeros(inputSet, maxVectorsInBasis);

    kvec[0] = 1; //выбираем первый вектор в качестве базисного
    int baselen = 1;

    for (int i = 1; i < inputSet.num_cols; i++) {
        kvec[i] = 0;
    }

    basisVecInx[0] = 0;


    if (baselen_in != 0 && kvec_in != 0 && basisVecInx_in != 0) {
        //тут бы всё и скопировать
    }

    //double *z = (double *) malloc(vectorDim * sizeof (double)); //allocate memory for el of min norm
    double *z = new_host_darray(inputSet.num_rows); //allocate memory for el of min norm    
    //пусть перый элемент мин нормы будет равен первому вектору симплекса
    //не то чтобы я совсем дурак и умножаю на ноль, но для наглядности не повредит (потом и стереть можно)
    //memcpy(z, &inputSet[0 * vectorDim], vectorDim * sizeof (double));
    //get_dense_column(inputSet, (int) 0, z);

    int iterCounter = 0;
    //std::cout << "Max count of elements in grammMatr: " << maxVectorsInBasis * maxVectorsInBasis << "\n";
    //можно сразу выделить память под матрицу Грамма и ей обратную
    //double *grammMatr = new_host_darray (maxVectorsInBasis * maxVectorsInBasis); //(double*) malloc(maxVectorsInBasis * maxVectorsInBasis * sizeof (double)); //
    //double *invGrammMatr = new_host_darray (maxVectorsInBasis * maxVectorsInBasis); //(double*) malloc(maxVectorsInBasis * maxVectorsInBasis * sizeof (double)); //new_host_darray (maxVectorsInBasis * maxVectorsInBasis);
    //double *grammMatr = new_host_darray(400 * 400);
    csr_matrix grammMatr;

    grammMatr.num_rows = 0;
    grammMatr.num_cols = 0;
    grammMatr.num_nonzeros = 0;

    grammMatr.Ap = new_host_iarray(maxVectorsInBasis + 1);
    grammMatr.Aj = new_host_iarray(inputSet.num_nonzeros * 5);
    grammMatr.Ax = new_host_darray(inputSet.num_nonzeros * 5);

    double *invGrammMatr = new_host_darray(400 * 400);
    //for (int i = 0; i < maxVectorsInBasis * maxVectorsInBasis; i++) {
    for (int i = 0; i < 400 * 400; i++) {
        //grammMatr[i] = 0.0;
        invGrammMatr[i] = 0.0;
    }
    double *mu = new_host_darray(maxVectorsInBasis); //(double*) malloc(maxVectorsInBasis * sizeof (double)); //координаты вектора в новом базисе
    mu[0] = 1.0; //Инициализируем mu тк в начале только 1 вектор в базисе.
    double *mu_old = new_host_darray(maxVectorsInBasis); //(double*) malloc(maxVectorsInBasis * sizeof (double));

    //Indicate if basis decreas or increas
    //Здесь выделим память под базис. В данной программе базис это разреженная матрица
    //double *basis = (double*) malloc(inSetDim * vectorDim * sizeof (double));
    csr_matrix basis = get_empty_csr_for_col_add((int) inputSet.num_rows, estim_nonzer_in_basis);
    csr_matrix basis_t = get_empty_csr_for_row_add((int) maxVectorsInBasis, estim_nonzer_in_basis); //((int) inputSet.num_cols, estim_nonzer_in_basis);
    //добавим первый вектор в базис
    //cblas_scopy(vectorDim, &inputSet[0 * vectorDim], 1, &basis[(0) * vectorDim], 1);
    /////////////////////this is for one vector////////////////////////
    //coo_matrix coo_col_to_add = get_coo_column(inputSet, (int) 0);
    //print_coo_matrix(coo_col_to_add);
    //add_col_to_csr_mtx(basis, coo_col_to_add);
    //transpose_coo_mv(coo_col_to_add);
    //add_row_to_csr(basis_t, coo_col_to_add);
    //delete_host_matrix (coo_col_to_add);
    ///////////////////end this is for one vector/////////////////////
    basis_t.num_rows = 1;
    basis_t.num_cols = inputSet.num_rows;
    basis_t.num_nonzeros = inputSet.Cp[1];
    memmove(&basis_t.Ap[0], &inputSet.Cp[0], (2) * sizeof (int));
    memmove(&basis_t.Aj[0], &inputSet.Ri[0], basis_t.num_nonzeros * sizeof (int));
    memmove(&basis_t.Ax[0], &inputSet.Ex[0], basis_t.num_nonzeros * sizeof (double));

    std::cout << "Number of nonzeros in basis_t=" << basis_t.num_nonzeros << " rows " << basis_t.num_rows << " colls " << basis_t.num_cols << "\n";
    //print_csr_matrix(basis_t);

    coo_matrix coo_col_to_add; //

    kvec[0] = 1;
    basisVecInx[0] = 0;

    csr_transpose_mv(basis_t, basis);
    std::cout << "Number of nonzeros in basis=" << basis.num_nonzeros << "\n";

    //print_csc_matrix(inputSet);

    baselen = 1;


    ldl_matrix grammPartFactor = new_ldl_matrix(maxVectorsInBasis + 1, inputSet.num_nonzeros * 5);
    //grammPartFactor.num_nonzeros = 0;
    //grammPartFactor.num_cols = 0;
    //grammPartFactor.num_rows = 0;
    //grammPartFactor.D = new_host_darray(maxVectorsInBasis + 1);
    //grammPartFactor.Lp = new_host_iarray(maxVectorsInBasis + 1);
    //grammPartFactor.Li = new_host_iarray(inputSet.num_nonzeros * 5);
    //grammPartFactor.Lx = new_host_darray(inputSet.num_nonzeros * 5);
    //grammPartFactor.Lnz = new_host_iarray(maxVectorsInBasis + 1);
    //grammPartFactor.Parent = new_host_iarray(maxVectorsInBasis + 1);
    //grammPartFactor.MAX_D_LP_SIZE = maxVectorsInBasis + 1;
    //grammPartFactor.MAX_Li_Lx_SIZE = inputSet.num_nonzeros * 5;
    ///////Now we mast calculate z for this basis
    int basisInc = 0;
    std::cout << "Get Mu for initial basis\n";
        
    evalMuForSimplexCPUwithStoredMatrix(basisInc, basis, basis_t, grammMatr, grammPartFactor, mu,0);

        

    memcpy(mu_old, mu, baselen * sizeof (double));
    cblas_dscal(basis.num_rows, 0.0, z, 1);
    spmv_csr_serial_host(basis, mu, z);
    
    
    /////////////TEST
    double mu_summ = 0.0;
    for (int sum_i = 0; sum_i < basis.num_cols; sum_i++) {
        mu_summ += mu[sum_i];
    }

    std::cout << "MU summ = " << mu_summ << "\n";
     /* 
    double* test_aX = new_host_darray(baselen);
    double* e_i = new_host_darray(baselen);
    spmv_csr_serial_host(basis_t, z, test_aX);
    get_dense_row(basis, basis.num_rows - 1, e_i);
    double razn_aX_Xtz = 0.0;
    for (int test_i = 0; test_i < baselen; test_i++) {
        razn_aX_Xtz += (e_i[test_i] - test_aX[test_i]) * (e_i[test_i] - test_aX[test_i]);
    }
    std::cout << "TEST aX - X^t * z = " << sqrt(razn_aX_Xtz) << "\n";
    ///////////END TEST
    */
    

    //coo_col_to_add = convDenseColToCooCol(z, basis.num_rows);
    //csr_matrix csrCol = coo2csr(coo_col_to_add);
    //print_csr_matrix_to_file(csrCol, "z.mtx", "z_s");
    //printMatrixCPU(1, basis.num_rows, z);
    /*
    //////////////////Проверка условия на правильность z
    double *x_i = new_host_darray(basis_t.num_rows);
    cblas_dscal(basis_t.num_rows, 0, x_i, 1);
    spmv_csr_serial_host(basis_t, z, x_i);
    //spmv_csr_t_serial_host(basis, z, x_i);
    std::cout << "b_t * Z\n";
    //printMatrixCPU(1, basis.num_rows, x_i);
    double maxb_t_to_z = getMaxVectorElCPU(x_i, basis.num_rows, minMuIdx, 0);
    std::cout << "Max from b_t*Z = " << maxb_t_to_z << " with idx: " << minMuIdx << "\n";
    double minb_t_to_z = getMinVectorElCPU(x_i, basis.num_rows, minMuIdx, 0);
    std::cout << "Mix from b_t*Z = " << minb_t_to_z << " with idx: " << minMuIdx << "\n";
    ////////////////////
     */
    //cblas_dscal(vectorDim, -1.0 / z[vectorDim - 1], z, 1);
    //evalMuVectorCPU(baselen, vectorDim, basisInc, basis, basis_t, grammMatr, invGrammMatr, mu, isFulBasis, delBasElIndx);
    ///////////////////////////////////////////////////
    //==========================OPEN FILE to write A'*z================
    std::ofstream myfile;

    myfile.open("z_decrese_dynamic");


    while (iterCounter < maxIter) {
        std::cout << "Iteration " << iterCounter << "\n";
        //print_csr_matrix(basis);
        //printMatrixCPU(1, vectorDim, z);
        iterCounter++; //увеличиваем счётчик итераций
        double minVcos = 0;
        int minVecId = 0;
        //находим следующего кандидата для добавления в базис руководствуясь критерием
        //наибольшей дистанции от множества
        //int hTimer = 0;
        //CUT_SAFE_CALL(cutCreateTimer(&hTimer)); //init timer
        //CUDA_SAFE_CALL(cudaThreadSynchronize());
        //CUT_SAFE_CALL(cutResetTimer(hTimer));
        //CUT_SAFE_CALL(cutStartTimer(hTimer));
        getMinVCosSimplexCPU(z, inputSet, minVcos, minVecId, eps, norms);
        //CUT_SAFE_CALL(cutStopTimer(hTimer));
        //codePartTimer += cutGetTimerValue(hTimer);
        //double zsqr = cblas_ddot(vectorDim, z, 1, z, 1);
        //printf("Epsilon threshold %e", eps * eps * zsqr);
        double z_norm = cblas_dnrm2(basis.num_rows, z, 1);
        myfile << iterCounter << " " << fabs(minVcos) << " " << z_norm << "\n";

        if (minVcos > -eps * eps) {
            printf("End of algoritm \n");
            printf("Min vCos %e \n", minVcos);
            printf("Iterations num %i \n", iterCounter);
            //print_csr_matrix(basis);
            //coo_col_to_add = get_coo_column(inputSet, minVecId);
            //print_coo_matrix(coo_col_to_add);
            //printMatrixCPU(1, inputSet.num_rows, z);

            cblas_dscal(inputSet.num_rows, 1.0 / z[inputSet.num_rows - 1], z, 1);

            memcpy(minNormVector, z, inputSet.num_rows * sizeof (double));
            memcpy(kvec_in, kvec, (inputSet.num_cols + 1) * sizeof (int));
            memcpy(basisVecInx_in, basisVecInx, (inputSet.num_cols + 1) * sizeof (int));
            baselen_in = baselen;
            break;
        }
        if (kvec[minVecId] == 1) {
            printf("Vector %i aready in basis. Exit\n", minVecId);
            printf("Min vCos %e \n", minVcos);
            printf("Iterations num %i \n", iterCounter);
            for (int basidx_iter = 0; basidx_iter < basis.num_cols; basidx_iter++) {
                if (basisVecInx[basidx_iter] == minVecId) {
                    std::cout << "mu["<< basidx_iter <<"] for min vector=" << mu[basidx_iter] << " \n";
                    print_coo_matrix(get_coo_column(basis, basidx_iter));
                }
            }

            print_csr_matrix_to_file(basis, "basis.csr", "BASIS");

            cblas_dscal(inputSet.num_rows, 1.0 / z[inputSet.num_rows - 1], z, 1);

            memcpy(minNormVector, z, inputSet.num_rows * sizeof (double));
            memcpy(kvec_in, kvec, (inputSet.num_cols + 1) * sizeof (int));
            memcpy(basisVecInx_in, basisVecInx, (inputSet.num_cols + 1) * sizeof (int));
            baselen_in = baselen;
            //for(int j=0;j<baselen;j++){printf("%i ", basisVecInx[j]);} printf("\n");
            //printMatrixCPU(vectorDim, 1, z);
            //double zkv = cblas_sdot(vectorDim, z, 1, z, 1);
            //printf("z norm = %f,  %f\n", zkv, sqrt(zkv));
            //minVecId = 194430;
            break;
        }
        int ix = minVecId; //кандидат для включения в базис.
        //printf("Add %i vec to basis\n", ix);
        baselen++;
        basisInc = 1;
        basisVecInx[baselen - 1] = ix;

        kvec[ix] = 1;
        //копируем координаты вектора в переменную + 0 в новую позицию для добавляемого элемента базиса
        memcpy(mu_old, mu, baselen * sizeof (double));
        mu_old[baselen - 1] = 0.0;
        //тут я думаю стоит выделить память под новый базис

        //double *mu = (double*) malloc(baselen * sizeof(double));//координаты вектора в новом базисе
        //тут можно вызвать процедуру для заполнения выделенной памяти векторами
        //cblas_scopy(vectorDim, &inputSet[ix * vectorDim], 1, &basis[(baselen - 1) * vectorDim], 1);
        coo_col_to_add = get_coo_column_from_csc(inputSet, ix);
        add_col_to_csr_mtx(basis, coo_col_to_add);
        transpose_coo_mv(coo_col_to_add);
        //std::cout << "add row to csr baselen= "<< baselen << " basis_t.num_rows=" << basis_t.num_rows << " basis_t.num_cols="<<basis_t.num_cols <<"\n";
        //print_csr_matrix(basis_t);
        //std::cout << "\n";
        add_row_to_csr(basis_t, coo_col_to_add);
        //print_csr_matrix(basis_t);
        //std::cout << "\n";
        //print_coo_matrix(coo_col_to_add);
        //std::cout << " end add row to csr\n";
        delete_host_matrix(coo_col_to_add);

        //fillBasis_CPU(basis, inputSet, kvec, baselen, inSetDim, vectorDim);
        //transMtrx(basis, basis_t, vectorDim, baselen);
        //printf("New basis");
        //printMatrix(vectorDim, baselen, basis);
        //printMatrix(baselen, vectorDim ,basis_t);
        //выделяем память под вектор для хранения mu = -(X' * X)^-1 * e / e' (X' * X)^-1 * e

        //int retToSimplex = 0;
        //std::cout << "Bazis after addition\n";
        //print_csr_matrix(basis);

        findOptZInBasisCPU(inputSet, baselen, inputSet.num_rows, basisInc, basis, basis_t, basisVecInx,
                kvec, grammMatr, invGrammMatr, mu, mu_old, z, eps, grammPartFactor);


    }
    std::cout << "Z mult el = " << z[inputSet.num_rows - 1] << "\n";
    cblas_dscal(inputSet.num_rows, 1.0 / z[inputSet.num_rows - 1], z, 1);

    //memcpy(minNormVector, z, inputSet.num_rows * sizeof (double));
    cblas_dcopy(inputSet.num_rows, z, 1, minNormVector, 1);
    memcpy(kvec_in, kvec, (inputSet.num_cols + 1) * sizeof (int));
    memcpy(basisVecInx_in, basisVecInx, (inputSet.num_cols + 1) * sizeof (int));

    //cblas_dscal(vectorDim, -1.0 / z[vectorDim - 1], z, 1);
    //printf("Cpu time of procedure: %f msecs.\n", codePartTimer);
    //for(int j=0;j<baselen;j++){printf("%i ", basisVecInx[j]);}printf("\n");
    //printMatrix(vectorDim, baselen,  basis);
    //printMatrix(1,vectorDim,  z);
    double dist = cblas_ddot(inputSet.num_rows, z, 1, z, 1);

    //===================Close file to store A'z
    myfile.close();

    free(mu_old);

    free(invGrammMatr);

    delete_host_matrix(grammMatr);
    delete_ldl_matrix(grammPartFactor);

    free(basisVecInx);
    free(mu);
    free(kvec);
    delete_host_matrix(basis);
    delete_host_matrix(basis_t);
    free(z);
}


//template <typename int, typename double>

void findOptZInBasisCPU(csc_matrix& inSet, int &baselen, int vectorDim, int basisInc,
        csr_matrix &basis, csr_matrix &basis_t, int *basisVecInx, int *kvec, csr_matrix& grammMatr, double *invGrammMatr,
        double *mu, double *mu_old, double *z, double eps, ldl_matrix& grammPartFactor) {
    int baselenFixed = baselen;
    //print_csr_matrix(basis);
    //print_csr_matrix(basis_t);
    bool isFulBasis = false;
    int delBasElIndx = 0; //индекс элемента который мы удаляем из базиса
    for (int i = 0; i < baselenFixed; i++) {
        //решаем задачу проекции нуля на вновь образованное подпространство
        // min||z^2|| = ||Z_s^2||
        evalMuForSimplexCPUwithStoredMatrix(basisInc, basis, basis_t, grammMatr, grammPartFactor, mu, delBasElIndx);
        //evalMuVectorCPUwithStoredMatrix(baselen, vectorDim, basisInc, basis, basis_t, grammMatr, invGrammMatr, mu, eps, numEqCon, 0);
        //std::cout << "basis" << "\n";
        //print_csr_matrix(basis);
        //std::cout << "mu vector " << baselen << "\n";
        //printMatrixCPU((int) 1, baselen, mu);
        //for(int j=0;j<baselen;j++){printf("%i ", basisVecInx[j]);}
        //printf("\n");
        int minMuIdx = 0;

        //Находим нормальное наименьшее MU после исправления нашего базиса
        //Тут необходимо проверить, принадлежит ли получившийся вектор множеству X
        //Похоже это условие min(mu) > -eps
        double minMu = getMinVectorElCPU(mu, baselen, minMuIdx, 0);
        double maxMu = getMaxVectorElCPU(mu, baselen, minMuIdx, 0);
        std::cout << "Min mu = " << minMu << " max mu = " << maxMu << " \n";
        //std:cout <<"minMu - mu_old[minMuIdx] = " << mu_old[minMuIdx] - minMu << "\n";
        if (minMu > -eps * eps) {
            //if (getMinVectorElCPU(mu, baselen) > 0) {
            //Вычисляем координаты получившегося вектора Z_s
            //cblas_sgemv(CblasColMajor, CblasNoTrans, vectorDim, baselen, 1.0, basis, vectorDim, mu, 1, 0, z, 1);

            cblas_dscal(basis.num_rows, 0, z, 1);
            spmv_csr_serial_host(basis, mu, z);

            ///test
            //cblas_dscal(basis.num_rows, 0.001, z, 1);

            

            
            double z_norm = cblas_dnrm2(basis.num_rows, z, 1);
            std::cout << "z_norm = " << z_norm << "\n";
            /*
            double z_dot = 0;
            for(int z_i = 0; z_i < basis.num_rows; z_i++){
                z_dot += z[z_i]* z[z_i];
            }
            std::cout << "z_norm = " << sqrt(z_dot) << "\n";
             */
            //////////////////Проверка условия на правильность z
            //double *x_i = new_host_darray(basis_t.num_rows);
            //cblas_dscal(basis_t.num_rows, 0, x_i, 1);
            //spmv_csr_serial_host(basis_t, z, x_i);
            //std::cout << "b_t * Z\n";
            //printMatrixCPU(1, basis_t.num_rows, x_i);            
            ////////////////////
            

            break;
        }

        //Если Z* находится за перделами X то надо решить вспомогательную задачу
        //спроецируем Z* на X
        //Мы подтягиваем Z(k+1) к Z(k) и получаем новый ветор Z(k+1)
        //то есть u(t) = t * mu + (t-1) mu_old ищим максимум по t так что u принадлежит X

        double t = attractDotCoeffToSimplexCPU(basis, mu_old, mu, delBasElIndx, baselen);
        //Удалим элемент из базиса
        //cblas_scopy(((baselen - 1) - delBasElIndx) * vectorDim, &basis[(delBasElIndx + 1) * vectorDim], 1, &basis[delBasElIndx * vectorDim], 1);
        //print_csr_matrix(basis);
        del_col_from_csr_mtx(basis, delBasElIndx);
        //std::cout << "\n";
        //print_csr_matrix(basis);

        del_row_from_csr_mtx(basis_t, delBasElIndx);
        //Удалим mu
        cblas_dcopy(((baselen - 1) - delBasElIndx), &mu[(delBasElIndx + 1)], 1, &mu[delBasElIndx], 1);
        cblas_dcopy(baselen, mu, 1, mu_old, 1);
        mu_old[baselen - 1] = 0.0;
        //cblas_copy(((baselen - 1) - delBasElIndx), &mu_old[(delBasElIndx + 1)], 1, &mu_old[delBasElIndx], 1);

        kvec[basisVecInx[delBasElIndx]] = 0;
        printf("del %i\n", basisVecInx[delBasElIndx]);
        //double* basisVecInxTmp = (double*) malloc(((baselen - 1) - delBasElIndx) * sizeof (int));
        memmove(&basisVecInx[delBasElIndx], &basisVecInx[delBasElIndx + 1], ((baselen - 1) - delBasElIndx) * sizeof (int));
        //memcpy(basisVecInxTmp, &basisVecInx[delBasElIndx + 1], ((baselen - 1) - delBasElIndx) * sizeof (int));
        //memcpy(&basisVecInx[delBasElIndx], basisVecInxTmp, ((baselen - 1) - delBasElIndx) * sizeof (int));
        //free(basisVecInxTmp);
        baselen--;
        basisInc = -1;
        //Вычисляем координаты получившегося вектора Z_s
        //cblas_sgemv(CblasColMajor, CblasNoTrans, vectorDim, baselen, 1, basis, vectorDim, mu, 1, 0, z, 1);

        //spmv_csr_serial_host(basis, mu, z);

        //multMatrToVector(basis, mu, z, vectorDim, baselen);
        //printf("Coordinates of new Z coordinates\n");
        //printMatrix(vectorDim, 1, z);
        ///////////////////////////////////////////////////
    }
}

void evalMuForSimplexCPUwithStoredMatrix(int basisInc, csr_matrix basis, csr_matrix basis_t,
        csr_matrix& grammMatrParted, ldl_matrix& grammPartFactor, double *mu, int delBasElIndx) {

    if (basisInc == 0) {
        std::cout << "basisInc == 0 New basis\n";
        evalGrammMtxPart(basis_t, grammMatrParted);

        std::cout << "Number of nonzeros in grammMatrParted=" << grammMatrParted.num_nonzeros << "\n";
        evalCholmodFactorTrans(grammMatrParted, grammPartFactor);
        std::cout << "Number of nonzeros in Factor=" << grammPartFactor.num_nonzeros << "\n";
        
        evalGrammMtxPartTriangForm(basis_t, grammMatrParted);
    } else if (basisInc == 1) {
        std::cout << "basisInc == 1 add column to basis\n";

        //updateGrammMtxPart_impr(basis, grammMatrParted);
        updateGrammMtxPartTriagForm_impr(basis, grammMatrParted);
        
        addColToCholmodFactor(grammPartFactor, grammMatrParted);

        //print_csr_matrix(grammMatrParted);
    } else if (basisInc == 2) {
        std::cout << "basisInc == 2 change sign of basis columns\n";        
    } else {
        std::cout << "basisInc == -1 del column from basis\n";
        std::cout << "del column with index " << delBasElIndx << " \n";
        //evalGrammMtxPart(basis_t, grammMatrParted);        
        downgradeGrammMtxPart(grammMatrParted, delBasElIndx);                
        //evalCholmodFactorTrans(grammMatrParted, grammPartFactor);
        //evalGrammMtxPartTriangForm(basis_t, grammMatrParted);
        
        std::cout << "grammPartFactor.num_rows =  " << grammPartFactor.num_rows << "grammPartFactor.num_cols =  " << grammPartFactor.num_cols <<  "grammPartFactor.num_nonzeros =  " << grammPartFactor.num_nonzeros << " \n";
        print_ldl_matrix(grammPartFactor);
        
        delete_col_from_ldl_factor(grammPartFactor, delBasElIndx);
        
        ////////////TEST!!!!!!!!!!!!!!!!!!!!!!!!
        evalCholmodFactorTrans(grammMatrParted, grammPartFactor);
        std::cout << "RECOMPUTED MATRIX!!!\n";
        print_ldl_matrix(grammPartFactor);
    }

    double *z_i = new_host_darray(basis.num_cols); //column of inverse matrix of gramm
    double *v_i = new_host_darray(basis.num_cols); //column of Unar matrix to pass to system solver as right elemetn
    double *y_i = new_host_darray(basis.num_cols); //column of inverse matrix of gramm

    get_dense_row(basis, basis.num_rows - 1, v_i);


    cblas_dcopy(grammMatrParted.num_rows, v_i, 1, z_i, 1);


    ldl_ltsolve_t(grammMatrParted.num_rows, z_i, grammPartFactor.Lp, grammPartFactor.Li, grammPartFactor.Lx);
    ldl_dsolve_t(grammMatrParted.num_rows, z_i, grammPartFactor.D);
    ldl_lsolve_t(grammMatrParted.num_rows, z_i, grammPartFactor.Lp, grammPartFactor.Li, grammPartFactor.Lx);


    for (int ones_i = 0; ones_i < basis.num_cols; ones_i++) {
        y_i[ones_i] = 1.0;
    }



    ldl_ltsolve_t(grammMatrParted.num_rows, y_i, grammPartFactor.Lp, grammPartFactor.Li, grammPartFactor.Lx);
    ldl_dsolve_t(grammMatrParted.num_rows, y_i, grammPartFactor.D);
    ldl_lsolve_t(grammMatrParted.num_rows, y_i, grammPartFactor.Lp, grammPartFactor.Li, grammPartFactor.Lx);

    double yv = cblas_ddot(basis.num_cols, y_i, 1, v_i, 1);
    double vz = cblas_ddot(basis.num_cols, z_i, 1, v_i, 1);


    double betta = yv / (1 + vz);

    cblas_daxpy(basis.num_cols, -betta, z_i, 1, y_i, 1);


    double elSumm = 0;
    sumVecElmts(y_i, &elSumm, basis.num_cols);
    cblas_dscal(basis.num_cols, 1.0 / elSumm, y_i, 1);

    cblas_dcopy(basis.num_cols, y_i, 1, mu, 1);
    //printMatrixCPU(1, basis.num_cols, mu);

    free(z_i);
    free(y_i);
    free(v_i);
    //free(x_row_summ);
}

void getMinVCosSimplexCPU(double* z, csc_matrix &inSet, double& minVcos, int &minVecId, double epsilon, double* l_norms) {
    double* vcos = new_host_darray(inSet.num_cols); //(double *) malloc(inSetDim * sizeof (double));

    //double z_norm = cblas_dnrm2(inSet.num_rows, z, 1);
    double zz = cblas_ddot(inSet.num_rows, z, 1, z, 1);
    A_dot_x_csc_serial_host(inSet, z, vcos);
    for (int col_i = 0; col_i < inSet.num_cols; col_i++) {
        //vcos[col_i] /= l_norms[col_i] * z_norm;
        vcos[col_i] -= zz;
    }
    minVcos = vcos[0] - zz; // - 1];
    minVecId = 0; // - 1;
    for (int elId = 0; elId < inSet.num_cols; elId++) {
        double curentVcos = vcos[elId];
        if (curentVcos < minVcos) {
            minVcos = curentVcos;
            minVecId = elId;
        }
    }

    std::cout << "Min vcos id:" << minVecId << " value" << minVcos << "\n";
    free(vcos);
}

/**
 *Подтянуть выпавшую точку в базис
 **/
//template <typename int, typename double>

double attractDotCoeffToSimplexCPU(csr_matrix &basis, double *mu_old, double *mu, int &delBasElIndx, int baselen) {

    double *mu_razn = (double*) malloc(baselen * sizeof (double));
    double *mu_div = (double*) malloc(baselen * sizeof (double));

    double minPozVal = 100000.0;
    //this for case withiout eq conditions
    //for (int i = 0; i < baselen; i++) {
    for (int i = 0; i < baselen; i++) {
        mu_razn[i] = mu_old[i] - mu[i];
        if (mu_razn[i] <= 0.0) {
            mu_div[i] = 100000;
        } else {
            mu_div[i] = mu_old[i] / mu_razn[i];
            if (mu_div[i] < minPozVal) {
                minPozVal = mu_div[i];
                delBasElIndx = i;
            }
        }
    }
    std::cout << "Parameter of attraction t= " << minPozVal << "and min el inx = " << delBasElIndx << "\n";
    if (minPozVal == 100000.0) {
        minPozVal = 0.0;
    }
    double t = minPozVal;
    double t2 = 1 - t;
    //Case without eq conditions
    //for (int i = 0; i < baselen; i++) {
    for (int i = 0; i < baselen; i++) {
        mu[i] = t * mu[i] + t2 * mu_old[i];
    }

    free(mu_razn);
    free(mu_div);
    return minPozVal;
}
