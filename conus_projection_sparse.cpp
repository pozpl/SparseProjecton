
#include "conus_projection_sparse.hpp"
#include "overloaded_cblas.hpp"
/**
 * Данная программа решает задачу min 1/2 ||x||^2, Ax <= b
 */
//template <typename int, typename double>

void getMinNormElemOutRepr(csc_matrix &inputSet,
        double * minNormVector, double tollerance,
        int* kvec_in, int* basisVecInx_in, int &baselen_in, int &numEqCon) {
    int maxIter = 0; //inSetDim;
    double eps = tollerance; //0.001;

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
    grammMatr.Aj = new_host_iarray(inputSet.num_nonzeros * 10);
    grammMatr.Ax = new_host_darray(inputSet.num_nonzeros * 10);

    double *invGrammMatr = new_host_darray(400 * 400);
    //for (int i = 0; i < maxVectorsInBasis * maxVectorsInBasis; i++) {
    for (int i = 0; i < 400 * 400; i++) {
        //grammMatr[i] = 0.0;
        invGrammMatr[i] = 0.0;
    }
    double *mu = new_host_darray(maxVectorsInBasis); //(double*) malloc(maxVectorsInBasis * sizeof (double)); //координаты вектора в новом базисе
    mu[0] = 1.0; //Инициализируем mu тк в начале только 1 вектор в базисе.
    double *mu_old = new_host_darray(maxVectorsInBasis); //(double*) malloc(maxVectorsInBasis * sizeof (double));

    int basisInc = 0; //Indicate if basis decreas or increas
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
    basis_t.num_rows = numEqCon;
    basis_t.num_cols = inputSet.num_rows;
    basis_t.num_nonzeros = inputSet.Cp[numEqCon ];
    memmove(&basis_t.Ap[0], &inputSet.Cp[0], (numEqCon + 1) * sizeof(int));
    memmove(&basis_t.Aj[0], &inputSet.Ri[0], basis_t.num_nonzeros * sizeof(int));
    memmove(&basis_t.Ax[0], &inputSet.Ex[0], basis_t.num_nonzeros * sizeof(double));
    //print_csr_matrix(basis_t);
    coo_matrix coo_col_to_add;//
    for (int eqVecCount = 0; eqVecCount < numEqCon; eqVecCount++) {
        //coo_col_to_add = get_coo_column_from_csc(inputSet, (int) eqVecCount);
        //print_coo_matrix(coo_col_to_add);
        //add_col_to_csr_mtx(basis, coo_col_to_add);
        //transpose_coo_mv(coo_col_to_add);
        //add_row_to_csr(basis_t, coo_col_to_add);
        //delete_host_matrix(coo_col_to_add);
        kvec[eqVecCount] = 1;
        basisVecInx[eqVecCount] = eqVecCount;
    }
    csr_transpose_mv(basis_t, basis);
    std::cout << "Number of nonzeros in basis=" << basis.num_nonzeros << "\n";
    //print_csr_matrix(basis);
    //print_csr_matrix(basis_t);
    baselen = numEqCon;


    ldl_matrix grammPartFactor;
    grammPartFactor.num_nonzeros = 0;
    grammPartFactor.num_cols = 0;
    grammPartFactor.num_rows = 0;
    grammPartFactor.D = new_host_darray(maxVectorsInBasis + 1);
    grammPartFactor.Lp = new_host_iarray(maxVectorsInBasis + 1);
    grammPartFactor.Li = new_host_iarray(inputSet.num_nonzeros * 10);
    grammPartFactor.Lx = new_host_darray(inputSet.num_nonzeros * 10);
    grammPartFactor.Lnz = new_host_iarray(maxVectorsInBasis + 1);
    grammPartFactor.Parent = new_host_iarray(maxVectorsInBasis + 1);
    grammPartFactor.MAX_D_LP_SIZE = maxVectorsInBasis + 1;
    grammPartFactor.MAX_Li_Lx_SIZE = inputSet.num_nonzeros * 10;
    ///////Now we mast calculate z for this basis
    std::cout<<"Get Mu for initial basis\n";
    evalMuVectorCPUwithStoredMatrix(basisInc, basis, basis_t, grammMatr, grammPartFactor, mu);
    
    ///Коррекция начального базиса для того чтобы он был в перделах конуса
    int minMuIdx = 0;
    double minMuFromEqConstrains = getMinVectorElCPU(mu, numEqCon, minMuIdx, 0);
    std::cout << "Min MU from eq constrains = " << minMuFromEqConstrains << " \n";
    //Если mu отрецательно в переделах условий, то надо корректировать умножением базисного вектора на -1
    while (minMuFromEqConstrains < 0){//-eps * eps * eps) {
        //std::cout << "Negative mu in basis equality constrains: " << minMu << " with index: " << minMuIdx << "\n";
        for (int j = 0; j < numEqCon; j++) {
            if (mu[j] <0){//< -eps * eps * eps ) {
                scaleColInCsrMtx(basis, -1.0, j);
                scaleRowInCsrMtx(basis_t, -1.0, j);
                //print_csr_matrix(basis);

                //scaleColInCsrMtx(inputSet, -1.0, basisVecInx[j]);
                scale_col_in_csc_mtx(inputSet, -1.0, basisVecInx[j]);

                scaleColInCsrMtx(grammMatr, -1.0, j);
                scaleRowInCsrMtx(grammMatr, -1.0, j);
                
                change_sign_in_ldl_t(grammPartFactor, j);
                //std::cout<< "after2\n";
                //print_ldl_matrix(grammPartFactor);
            }
        }
        //evalMuVectorCPUwithStoredMatrix(baselen, vectorDim, basisInc, basis, basis_t, grammMatr, invGrammMatr, mu, eps, numEqCon, 1);
        basisInc = 2;
        evalMuVectorCPUwithStoredMatrix(basisInc, basis, basis_t, grammMatr, grammPartFactor, mu);
        
        
        minMuFromEqConstrains = getMinVectorElCPU(mu, numEqCon, minMuIdx, 0);
        std::cout << "Min MU from eq constrains = " << minMuFromEqConstrains << " \n";
        
        double maxMuFromEqConstrains = getMaxVectorElCPU(mu, numEqCon, minMuIdx, 0);
        std::cout << "Min MU from eq constrains = " << maxMuFromEqConstrains << " \n";
    }

    /*for (int i = 0; i < inputSet.num_rows; i++) {
        for (int j = inputSet.Ap[i]; j < inputSet.Ap[i + 1]; j++) {
            //printf("%i %i %f \n", i, csr.Aj[j], csr.Ax[j]);
            if(inputSet.Aj[j] >0){printf("");}
            if(inputSet.Ax[j] >0){printf("");}
            //std::cout << i << " " << inputSet.Aj[j] << " " << inputSet.Ax[j] << "\n";
        }
    }*/
    
    //print_csr_matrix_to_file(basis_t, "basis.mtx", "basis_s");
    //print_csr_matrix_to_file(grammMatr, "gramm.mtx", "gramm_s");
    
    memcpy(mu_old, mu, baselen * sizeof (double));
    cblas_dscal(basis.num_rows, 0.0, z, 1);
    spmv_csr_serial_host(basis, mu, z);
    
    /////////////TEST
    double mu_summ = 0.0;
    for(int sum_i = 0; sum_i < basis.num_cols; sum_i++){
        mu_summ += mu[sum_i];
    }
    
    std::cout << "MU summ = " << mu_summ << "\n"; 
    
    double* test_aX = new_host_darray(baselen);
    double* e_i = new_host_darray(baselen);
    spmv_csr_serial_host(basis_t, z, test_aX);
    get_dense_row(basis, basis.num_rows - 1, e_i);
    double razn_aX_Xtz = 0.0;
    for(int test_i = 0; test_i < baselen; test_i++){
        razn_aX_Xtz += (e_i[test_i] - test_aX[test_i]) * (e_i[test_i] - test_aX[test_i]);
    }
    std::cout << "TEST aX - X^t * z = " << sqrt(razn_aX_Xtz) << "\n";
    ///////////END TEST
    
    z[basis.num_rows - 1] -= 1.0;
    
    //coo_col_to_add = convDenseColToCooCol(z, basis.num_rows);
    //csr_matrix csrCol = coo2csr(coo_col_to_add);
    //print_csr_matrix_to_file(csrCol, "z.mtx", "z_s");
    //printMatrixCPU(1, basis.num_rows, z);
    //////////////////Проверка условия на правильность z
    double *x_i = new_host_darray(basis_t.num_rows);
    cblas_dscal(basis_t.num_rows, 0, x_i, 1);
    spmv_csr_serial_host(basis_t, z, x_i);
    //spmv_csr_t_serial_host(basis, z, x_i);
    std::cout << "b_t * Z\n";
    //printMatrixCPU(1, basis.num_rows, x_i);
    double maxb_t_to_z = getMaxVectorElCPU(x_i, basis.num_rows, minMuIdx, 0);
    std::cout<< "Max from b_t*Z = " << maxb_t_to_z << " with idx: " << minMuIdx << "\n";
    double minb_t_to_z = getMinVectorElCPU(x_i, basis.num_rows, minMuIdx, 0);
    std::cout<< "Mix from b_t*Z = " << minb_t_to_z << " with idx: " << minMuIdx << "\n";
    ////////////////////

    //cblas_dscal(vectorDim, -1.0 / z[vectorDim - 1], z, 1);
    //evalMuVectorCPU(baselen, vectorDim, basisInc, basis, basis_t, grammMatr, invGrammMatr, mu, isFulBasis, delBasElIndx);
    ///////////////////////////////////////////////////
    //==========================OPEN FILE to write A'*z================
    std::ofstream myfile;
    
    myfile.open ("z_decrese_dynamic");
    
    
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
        getMinVCosCPU(z, inputSet, minVcos, minVecId, eps, numEqCon, norms);
        //CUT_SAFE_CALL(cutStopTimer(hTimer));
        //codePartTimer += cutGetTimerValue(hTimer);
        //double zsqr = cblas_ddot(vectorDim, z, 1, z, 1);
        //printf("Epsilon threshold %e", eps * eps * zsqr);
        double z_norm =  cblas_dnrm2(basis.num_rows, z, 1);
        myfile << iterCounter << " " << fabs(minVcos) << " " << z_norm <<  "\n";
        
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
            for(int basidx_iter = 0; basidx_iter < basis.num_cols; basidx_iter++){
                if(basisVecInx[basidx_iter] == minVecId){
                    std::cout << "mu for min vector=" << mu[basidx_iter] << " \n";
                    print_coo_matrix(get_coo_column(basis, basidx_iter));
                }
            }
            
            
            
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
                kvec, grammMatr, invGrammMatr, mu, mu_old, z, eps, numEqCon, grammPartFactor);


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
        double *mu, double *mu_old, double *z, double eps, int numEqCon,
        ldl_matrix& grammPartFactor) {
    int baselenFixed = baselen;
    //print_csr_matrix(basis);
    //print_csr_matrix(basis_t);
    bool isFulBasis = false;
    int delBasElIndx = 0; //индекс элемента который мы удаляем из базиса
    for (int i = 0; i < baselenFixed; i++) {
        //решаем задачу проекции нуля на вновь образованное подпространство
        // min||z^2|| = ||Z_s^2||
        evalMuVectorCPUwithStoredMatrix(basisInc, basis, basis_t, grammMatr, grammPartFactor, mu);
        //evalMuVectorCPUwithStoredMatrix(baselen, vectorDim, basisInc, basis, basis_t, grammMatr, invGrammMatr, mu, eps, numEqCon, 0);
        //std::cout << "InvGrammMatrix " << "\n";
        //printMatrixCPU(baselen, baselen, invGrammMatr);
        //std::cout << "mu vector before " << baselen << "\n";
        //printMatrixCPU((int) 1, baselen, mu);
        //for(int j=0;j<baselen;j++){printf("%i ", basisVecInx[j]);}
        //printf("\n");
        int minMuIdx = 0;
        ////////////////////Коррекция базиса с начальными условиями в виде равенств///////
        double minMuFromEqConstrains = getMinVectorElCPU(mu, numEqCon, minMuIdx, 0);
        //Если mu отрецательно в переделах условий, то надо корректировать умножением базисного вектора на -1
        while (minMuFromEqConstrains < 0) {//< -eps * eps * eps) {
            //std::cout << "Negative mu in basis equality constrains: " << minMu << " with index: " << minMuIdx << "\n";
            for (int j = 0; j < numEqCon; j++) {
                if (mu[j] < 0){//-eps * eps * eps) {
                    scaleColInCsrMtx(basis, -1.0, j);
                    scaleRowInCsrMtx(basis_t, -1.0, j);

                    //scaleColInCsrMtx(inSet, -1.0, basisVecInx[j]);
                    scale_col_in_csc_mtx(inSet, -1.0, basisVecInx[j]);
                    
                    scaleColInCsrMtx(grammMatr, -1.0, j);
                    scaleRowInCsrMtx(grammMatr, -1.0, j);
                    
                    change_sign_in_ldl_t(grammPartFactor, j);
                }
            }
            basisInc = 2;
            //getMuViaSystemSolve(basis, basis_t, mu, eps);
            evalMuVectorCPUwithStoredMatrix(basisInc, basis, basis_t, grammMatr, grammPartFactor, mu);
            //free(mu_n);
            //std::cout << "Mu from ideal===========================\n";
            //printMatrixCPU(1, baselen, mu);
            //////////////////END TEST
            minMuFromEqConstrains = getMinVectorElCPU(mu, numEqCon, minMuIdx, 0);
            std::cout << "Min MU from eq constrains = " << minMuFromEqConstrains << " with index: " << minMuIdx << " \n";            
        }
        /////////////////////////////////////////////////////////
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
            
            z[basis.num_rows - 1] -= 1.0;
            
            
            double z_norm =  cblas_dnrm2(basis.num_rows, z, 1);
            std::cout << "z_norm = " << z_norm << "\n"; 
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

        double t = attractDotCoeffCPU(basis, mu_old, mu, delBasElIndx, baselen, numEqCon);
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

//template <typename int, typename double>

void evalMuVectorCPU(int baselen, int vectorDim,
        int basisInc, csr_matrix basis, csr_matrix basis_t,
        double *grammMatr, double *invGrammMatr, double *mu, double eps) {
    double *x_i = new_host_darray(basis.num_cols); //column of inverse matrix of gramm
    double *e_i = new_host_darray(basis.num_cols); //column of Unar matrix to pass to system solver as right elemetn
    double *x_row_summ = new_host_darray(basis.num_cols); //vector to store reow summs for inver gramm matrix
    for (int i = 0; i < basis.num_cols; i++) {
        x_i[i] = 0.0;
        x_row_summ[i] = 0.0;
    }

    get_dense_column(basis_t, basis_t.num_cols - 1, e_i);
    //std::cout << "BASIS\n";
    //double *denseBasis = new_host_darray(basis.num_rows * basis.num_rows);
    //convCsrToDense(basis, denseBasis);
    //printMatrixCPU(basis.num_rows, basis.num_cols, denseBasis);
    //print_csr_matrix(basis);
    //std::cout << "B^'a^0\n";
    //printMatrixCPU(1, basis.num_cols, e_i);
    biSoprGradient(basis, basis_t, e_i, x_i, eps, mu);
    //std::cout << "End biSoprGrad\n";
    cblas_dcopy(basis.num_cols, x_i, 1, mu, 1);

    free(x_i);
    free(e_i);
    free(x_row_summ);
}

/*
double evalRowElemOfGrammMtxPart(csr_matrix basis_t, int A_row_idx, int B_row_idx){
    int A_row_start = basis_t.Ap[A_row_idx];
    int A_row_end = basis_t.Ap[A_row_idx + 1];
    
    int B_row_start = basis_t.Ap[B_row_idx];
    int B_row_end = basis_t.Ap[B_row_idx + 1];
    
    int A_activ_cols = A_row_end - A_row_start;
    int B_activ_cols = B_row_end - B_row_start;
    
    int row_iter_max = A_activ_cols + B_activ_cols;
    
    double gramm_row_el = 0.0;
    
    for(int i = 0, A_iter = 0, B_iter = 0; i < row_iter_max; i++){
        int A_col_i = basis_t.Aj[A_row_start + A_iter];
        int B_col_i = basis_t.Aj[B_row_start + B_iter];
        if(A_col_i == B_col_i){
            //multiplication
            double A_i_val = basis_t.Ax[A_row_start + A_iter];
            double B_i_val = basis_t.Ax[B_row_start + B_iter];
            gramm_row_el += A_i_val * B_i_val;
        }else if(A_col_i > B_col_i){
            if(B_iter <= B_activ_cols){
                B_iter++;
            }else{
                break;
            }
        }else{
           if(A_iter <= A_activ_cols){
                A_iter++;
            }else{
                break;
            } 
        }
    }
    return gramm_row_el;
}
 */

void evalRowOfGrammMtxPart(csr_matrix basis_t, int row_idx, double* gramm_row) {
    for (int el_i = 0; el_i < basis_t.num_rows; el_i++) {
        gramm_row[el_i] = 0;
    }
    double *bas_t_row = new_host_darray(basis_t.num_cols);

    get_dense_row(basis_t, row_idx, bas_t_row);

    bas_t_row[basis_t.num_cols - 1] = 0.0;
    
    spmv_csr_serial_host(basis_t, bas_t_row, gramm_row);
    delete_host_array(bas_t_row);
}

void evalGrammMtxPart(csr_matrix& basis_t, csr_matrix& gramm_parted) {
    double* gramm_row = new_host_darray(basis_t.num_rows);

    gramm_parted.Ap[0] = 0;
    gramm_parted.num_cols = basis_t.num_rows;
    gramm_parted.num_rows = basis_t.num_rows;
    
    gramm_parted.num_nonzeros = 0;
    for (int row_idx = 0; row_idx < basis_t.num_rows; row_idx++) {
        evalRowOfGrammMtxPart(basis_t, row_idx, gramm_row);
        int start_row_idx = gramm_parted.Ap[row_idx];
        int nonzer_row_elems = 0;
        for (int el_i = 0; el_i < basis_t.num_rows; el_i++) {
            if (gramm_row[el_i] != 0.0) {
                gramm_parted.num_nonzeros++;
                gramm_parted.Aj[start_row_idx + nonzer_row_elems] = el_i;
                gramm_parted.Ax[start_row_idx + nonzer_row_elems] = gramm_row[el_i];

                nonzer_row_elems++;
            }
        }
        gramm_parted.Ap[row_idx + 1] = start_row_idx + nonzer_row_elems;
    }
    delete_host_array(gramm_row);
}


void evalGrammMtxPartTriangForm(csr_matrix& basis_t, csr_matrix& gramm_parted) {
    double* gramm_row = new_host_darray(basis_t.num_rows);

    gramm_parted.Ap[0] = 0;
    gramm_parted.num_cols = basis_t.num_rows;
    gramm_parted.num_rows = basis_t.num_rows;
    
    gramm_parted.num_nonzeros = 0;
    for (int row_idx = 0; row_idx < basis_t.num_rows; row_idx++) {
        evalRowOfGrammMtxPart(basis_t, row_idx, gramm_row);
        int start_row_idx = gramm_parted.Ap[row_idx];
        int nonzer_row_elems = 0;
        for (int el_i = 0; el_i <= row_idx; el_i++) {
            if (gramm_row[el_i] != 0.0) {
                gramm_parted.num_nonzeros++;
                gramm_parted.Aj[start_row_idx + nonzer_row_elems] = el_i;
                gramm_parted.Ax[start_row_idx + nonzer_row_elems] = gramm_row[el_i];

                nonzer_row_elems++;
            }
        }
        gramm_parted.Ap[row_idx + 1] = start_row_idx + nonzer_row_elems;
    }
    delete_host_array(gramm_row);
}

void updateGrammMtxPart(csr_matrix& basis_t, csr_matrix& gramm_parted) {
    double* add_column = new_host_darray(basis_t.num_rows);
    for(int ac_i = 0; ac_i < basis_t.num_rows; ac_i++){add_column[ac_i] = 0.0;}
    //cblas_dscal(basis_t.num_rows, 0, add_column, 1);
    double *bas_t_row = new_host_darray(basis_t.num_cols);
    get_dense_row(basis_t, basis_t.num_rows - 1, bas_t_row);

    bas_t_row[basis_t.num_cols - 1 ] = 0.0; //cose we only need parted gramm matrix


    spmv_csr_serial_host(basis_t, bas_t_row, add_column);
    coo_matrix add_col_coo = convDenseColToCooCol(add_column, basis_t.num_rows - 1); //convDenseToCooNaiv(add_column, basis_t.num_rows - 1, 1);
    coo_matrix add_row_coo = convDenseRowToCooRow(add_column, basis_t.num_rows); //convDenseToCooNaiv(add_column, 1, basis_t.num_rows);

    add_col_to_csr_mtx(gramm_parted, add_col_coo);
    add_row_to_csr(gramm_parted, add_row_coo);
    
    delete_coo_matrix(add_col_coo);
    delete_coo_matrix(add_row_coo);

    free(bas_t_row);
    free(add_column);
}


void updateGrammMtxPart_impr(csr_matrix& basis,csr_matrix& gramm_parted) {
    double* add_column = new_host_darray(basis.num_cols);
    
    //double* add_column_chk = new_host_darray(basis.num_cols);
    
    for(int ac_i = 0; ac_i < basis.num_cols; ac_i++){add_column[ac_i] = 0.0;}
    
    //double *bas_col_chk = new_host_darray(basis.num_rows);
    
    coo_matrix bas_col;
    //get_dense_column(basis, basis.num_cols - 1, bas_col_chk);
    bas_col = get_coo_column(basis, basis.num_cols - 1);
    if(bas_col.I[bas_col.num_nonzeros - 1] == (basis.num_rows - 1) ){
        bas_col.V[bas_col.num_nonzeros - 1] = 0.0;         
    }
    
    //print_coo_matrix(bas_col);
    
    //bas_col_chk[basis.num_rows - 1 ] = 0.0; //cose we only need parted gramm matrix
    
   // spmv_csr_serial_host(basis_t, bas_col_chk, add_column_chk);
    
    spmv_scr_t_coo_serial_host(basis, bas_col, add_column);
    //printMatrixCPU(1, basis.num_cols, add_column);
    //printMatrixCPU(1, basis.num_cols, add_column_chk);
    //for(int chk_i = 0; chk_i < basis.num_cols; chk_i++){
    //    if(fabs(add_column[chk_i] - add_column_chk[chk_i]) ){
    //        std::cout <<  " add_column[" << chk_i << "] = " << add_column[chk_i] << " add_column_chk[" << chk_i << "] = " << add_column_chk[chk_i]  << "\n";
    //    }
    //}
    
    coo_matrix add_col_coo = convDenseColToCooCol(add_column, basis.num_cols - 1); //convDenseToCooNaiv(add_column, basis_t.num_rows - 1, 1);
    coo_matrix add_row_coo = convDenseRowToCooRow(add_column, basis.num_cols); //convDenseToCooNaiv(add_column, 1, basis_t.num_rows);
    
    add_col_to_csr_mtx(gramm_parted, add_col_coo);
    add_row_to_csr(gramm_parted, add_row_coo);
        
    delete_coo_matrix(add_col_coo);
    delete_coo_matrix(add_row_coo);

    delete_coo_matrix(bas_col);
    free(add_column);
}


void updateGrammMtxPartTriagForm_impr(csr_matrix& basis,csr_matrix& gramm_parted) {
    double* add_column = new_host_darray(basis.num_cols);
    
    for(int ac_i = 0; ac_i < basis.num_cols; ac_i++){add_column[ac_i] = 0.0;}
    
    coo_matrix bas_col;
    
    bas_col = get_coo_column(basis, basis.num_cols - 1);
    if(bas_col.I[bas_col.num_nonzeros - 1] == (basis.num_rows - 1) ){
        bas_col.V[bas_col.num_nonzeros - 1] = 0.0;         
    }
    
    spmv_scr_t_coo_serial_host(basis, bas_col, add_column);
    
    //coo_matrix add_col_coo = convDenseColToCooCol(add_column, basis.num_cols - 1); //convDenseToCooNaiv(add_column, basis_t.num_rows - 1, 1);
    coo_matrix add_row_coo = convDenseRowToCooRow(add_column, basis.num_cols); //convDenseToCooNaiv(add_column, 1, basis_t.num_rows);
    
    //add_col_to_csr_mtx(gramm_parted, add_col_coo);
    add_row_to_csr(gramm_parted, add_row_coo);
        
    //delete_coo_matrix(add_col_coo);
    delete_coo_matrix(add_row_coo);

    delete_coo_matrix(bas_col);
    free(add_column);
}

void downgradeGrammMtxPart(csr_matrix& gramm_parted, int col_to_del_idx){
    del_col_from_csr_mtx(gramm_parted, col_to_del_idx);
    del_row_from_csr_mtx(gramm_parted, col_to_del_idx);    
}


void evalMuVectorCPUwithStoredMatrix(int basisInc, csr_matrix basis, csr_matrix basis_t,
        csr_matrix& grammMatrParted, ldl_matrix& grammPartFactor, double *mu) {
    double *x_i = new_host_darray(basis.num_cols); //column of inverse matrix of gramm
    double *e_i = new_host_darray(basis.num_cols); //column of Unar matrix to pass to system solver as right elemetn
    //double *x_row_summ = new_host_darray(basis.num_cols); //vector to store reow summs for inver gramm matrix

    for (int i = 0; i < basis.num_cols; i++) {
        x_i[i] = 0.0;
    }
    
    if (basisInc == 0) {
        std::cout << "basisInc == 0 New basis\n";
        evalGrammMtxPart(basis_t, grammMatrParted);
        
        ///test
        //cblas_dscal(grammMatrParted.num_nonzeros, 1000.0, grammMatrParted.Ax, 1);
        
        std::cout << "Number of nonzeros in grammMatrParted=" << grammMatrParted.num_nonzeros << "\n";
        evalCholmodFactorTrans(grammMatrParted, grammPartFactor);
        std::cout << "Number of nonzeros in Factor=" << grammPartFactor.num_nonzeros << "\n";
        
        evalGrammMtxPartTriangForm(basis_t, grammMatrParted);
    } else if (basisInc == 1) {
        std::cout << "basisInc == 1 add column to basis\n";
        
        //evalGrammMtxPart(basis_t, grammMatrParted);
        //std::cout << "Number of nonzeros in grammMatrParted=" << grammMatrParted.num_nonzeros << "\n";
        //updateGrammMtxPart(basis_t, grammMatrParted);
        
        //updateGrammMtxPart_impr(basis, grammMatrParted);
        updateGrammMtxPartTriagForm_impr(basis, grammMatrParted);
        
        
        //evalCholmodFactorTrans(grammMatrParted, grammPartFactor);
        //addRowCholmodFactor(grammPartFactor, grammMatrParted);
        addColToCholmodFactor(grammPartFactor, grammMatrParted);
        /*
        ldl_matrix grammPartFactor_new;
        grammPartFactor_new.D = new_host_darray(grammMatrParted.num_cols + 1);
        grammPartFactor_new.Lp = new_host_iarray(grammMatrParted.num_cols + 1);
        grammPartFactor_new.Li = new_host_iarray(grammMatrParted.num_nonzeros);
        grammPartFactor_new.Lx = new_host_darray(grammMatrParted.num_nonzeros);
        grammPartFactor_new.Lnz = new_host_iarray(grammMatrParted.num_cols + 1);
        grammPartFactor_new.Parent = new_host_iarray(grammMatrParted.num_cols + 1);
        
        evalCholmodFactor(grammMatrParted, grammPartFactor_new);
        
        std::cout << "Matrix comparison====================\n";
        for(int d_i = 0; d_i < grammPartFactor.num_nonzeros; d_i++){
            if(fabs(grammPartFactor.Lx[d_i] - grammPartFactor_new.Lx[d_i]) > 0.0000001){
                std:cout << ">>>>>Matrix el[" << d_i << "]= " << fabs(grammPartFactor.Lx[d_i] - grammPartFactor_new.Lx[d_i]) << " grammPartFactor.Lx[" << d_i << "]= " << grammPartFactor.Lx[d_i] << " grammPartFactor_new.Lx[" << d_i << "]= " <<  grammPartFactor_new.Lx[d_i] << " grammPartFactor.Li[" << d_i << "]= " <<  grammPartFactor.Li[d_i] << " grammPartFactor_new.Li[" << d_i << "]= " <<  grammPartFactor_new.Li[d_i] <<"\n";
            }
        }
        for(int d_i = 0; d_i < grammPartFactor.num_cols; d_i++){
            if(fabs(grammPartFactor.D[d_i] - grammPartFactor_new.D[d_i]) > 0.0000001){
                cout << ">>>>>Matrix el[" << d_i << "]= " << fabs(grammPartFactor.D[d_i] - grammPartFactor_new.D[d_i]) << " grammPartFactor.Lx[" << d_i << "]= " << grammPartFactor.D[d_i] << " grammPartFactor_new.Lx[" << d_i << "]= " <<  grammPartFactor_new.D[d_i] << " grammPartFactor.Li[" << d_i << "]= " <<  grammPartFactor.D[d_i] << " grammPartFactor_new.Li[" << d_i << "]= " <<  grammPartFactor_new.D[d_i] <<"\n";
            }
        }
        
        //print_ldl_matrix(grammPartFactor);
        //print_ldl_matrix(grammPartFactor_new);
        
        
        delete_ldl_matrix(grammPartFactor_new);
        */     
        
        //////////////////TEST//////////
        /*
        ldl_matrix grammPartFactor_CHK;
        grammPartFactor_CHK.D = new_host_darray(basis.num_cols + 1);
        grammPartFactor_CHK.Lp = new_host_iarray(basis.num_cols + 1);        
        grammPartFactor_CHK.Li = new_host_iarray(basis.num_nonzeros * 10);
        grammPartFactor_CHK.Lx = new_host_darray(basis.num_nonzeros * 10);
        grammPartFactor_CHK.Lnz = new_host_iarray(basis.num_cols + 1);
        grammPartFactor_CHK.Parent = new_host_iarray(basis.num_cols + 1);
        //evalGrammMtxPart(basis_t, grammMatrParted);        
        evalCholmodFactor(grammMatrParted, grammPartFactor_CHK);
        //std::cout << "rows: " << grammPartFactor_CHK.num_rows << " columns: " <<grammPartFactor_CHK.num_cols <<"\n";
        for (int i = 0; i < grammPartFactor_CHK.num_rows; i++) {
                for (int j = grammPartFactor_CHK.Lp[i]; j < grammPartFactor_CHK.Lp[i + 1]; j++) {
                    if(fabs(grammPartFactor.Lx[j] - grammPartFactor_CHK.Lx[j]) > 0.00001 ){
                        std::cout << "Lx is diffrent!!!!  "<< i << " " << grammPartFactor_CHK.Li[j] << " CHK= " << grammPartFactor_CHK.Lx[j] << " orig=" <<  grammPartFactor.Lx[j] << "\n";
                    }
                    //std::cout << i << " " << grammPartFactor_CHK.Li[j] << " " << grammPartFactor_CHK.Lx[j] << "\n";
                }
        }
        */
        
    } else if (basisInc == 2) {
        std::cout << "basisInc == 2 change sign of basis columns\n";
        //test
        //cblas_dscal(grammMatrParted.num_nonzeros, 1000.0, grammMatrParted.Ax, 1);        
        //evalCholmodFactor(grammMatrParted, grammPartFactor);
        
        //print_ldl_matrix(grammPartFactor);
    } else {
        std::cout << "basisInc == -1 del column from basis\n";
        evalGrammMtxPart(basis_t, grammMatrParted);       
               
        evalCholmodFactorTrans(grammMatrParted, grammPartFactor);
        evalGrammMtxPartTriangForm(basis_t, grammMatrParted);
    }

   
    
    get_dense_row(basis, basis.num_rows - 1, e_i);
    //std::cout<<"e_i : basis.num_rows - 1 = " << basis.num_rows - 1 << "\n";
    //printMatrixCPU(1, grammPartFactor.num_rows, e_i);
    //print_csr_matrix(basis);
    
    cblas_dcopy(grammMatrParted.num_rows, e_i, 1, x_i, 1);
    
    //double min_el_val = 0;
    //int min_el_idx = 0;
    
    //min_el_val = getMinVectorElCPU(grammPartFactor.D, grammMatrParted.num_rows, min_el_idx, 0);
    //std::cout << "Min element of Factor.D=" << min_el_val << " with index " << min_el_idx <<  "\n";
    //y_cholmod = cholmod_solve(CHOLMOD_A, *grammPartFactor, b, cholmod_commons);
    ldl_ltsolve_t(grammMatrParted.num_rows, x_i, grammPartFactor.Lp, grammPartFactor.Li, grammPartFactor.Lx);
    //min_el_val = getMinVectorElCPU(x_i, grammMatrParted.num_rows, min_el_idx, 0);
    //std::cout << "Min element of x_i=" << min_el_val << " with index " << min_el_idx <<  "\n";
    ldl_dsolve_t(grammMatrParted.num_rows, x_i, grammPartFactor.D);
    //min_el_val = getMinVectorElCPU(x_i, grammMatrParted.num_rows, min_el_idx, 0);
    //std::cout << "Min element of x_i=" << min_el_val << " with index " << min_el_idx <<  "\n";
    ldl_lsolve_t(grammMatrParted.num_rows, x_i, grammPartFactor.Lp, grammPartFactor.Li, grammPartFactor.Lx);
    
    //min_el_val = getMinVectorElCPU(x_i, grammMatrParted.num_rows, min_el_idx, 0);
    //std::cout << "Min element of x_i=" << min_el_val << " with index " << min_el_idx <<  "\n";
    /*
    double *x_i2 = new_host_darray(basis.num_cols); //column of inverse matrix of gramm
    std::cout << "bi sopr stated\n";
    biSoprGradientStoredMatrix(basis, basis_t, grammMatrParted, e_i, x_i2, 0.0001, x_i);
    double x_razn = 0;
    for(int i = 0; i < basis.num_cols; i++){
        x_razn += fabs(x_i[i] - x_i2[i]);
        //if(fabs(x_i[i] - x_i2[i]) > 0){
        //        std::cout << "x_raxn[" << i << "]=" << x_razn << "\n";
        //}
    }
    std::cout << "Bi sopr Cop = " << x_razn << "\n";
    cblas_dcopy(basis.num_cols, x_i2, 1, x_i, 1);
    */
    double yv = cblas_ddot(basis.num_cols, x_i, 1, e_i, 1);
    //double alpha = yv / (1 + yv);
    //TEST//////////////////////////////////////////////////////////////////////
    /*
    //int min_el_idx = 0;
    //double min_el_val = getMinVectorElCPU(x_i, grammMatrParted.num_rows, min_el_idx, 0);
    //std::cout << "Min element of x_i=" << min_el_val << " with index " << min_el_idx <<  "\n";
    //min_el_val = getMaxVectorElCPU(x_i, grammMatrParted.num_rows, min_el_idx, 0);
    //std::cout << "Max element of x_i=" << min_el_val << " with index " << min_el_idx <<  "\n";
    
    double* ei_big = new_host_darray(basis.num_cols);
    double* ei_small =  new_host_darray(basis.num_cols);
    for(int ei_idx = 0; ei_idx < basis.num_cols; ei_idx++){
        ei_small[ei_idx] = modf(e_i[ei_idx], &ei_big[ei_idx]);
    }
    double yv_small = cblas_ddot(basis.num_cols, x_i, 1, ei_small, 1);
    double yv_big = cblas_ddot(basis.num_cols, x_i, 1, ei_big, 1);
    
    double alpha_small = yv_small / (1 + yv_small + yv_big);
    double alpha_big = yv_big / (1 + yv_small + yv_big);
    
    //std::cout << "alpha - alpha_big = " << alpha - alpha_big << " alpha -alpha_big - alpha small = " << alpha - alpha_big - alpha_small << "\n";
    
    double* xi_copy =  new_host_darray(basis.num_cols);
    cblas_dcopy(basis.num_cols, x_i, 1, xi_copy, 1);
    cblas_daxpy(basis.num_cols, -1 * alpha_big, x_i, 1, x_i, 1);
    cblas_daxpy(basis.num_cols, -1 * alpha_small, xi_copy, 1, x_i, 1);
    
    free(ei_big);
    free(ei_small);
    free(xi_copy);
     */
    /*
    double sum = 0.0;
    double c = 0.0;          //A running compensation for lost low-order bits.
    for(int dot_i = 1; dot_i < basis.num_cols; dot_i++){
        double y = e_i[dot_i] * x_i[dot_i] - c;    //So far, so good: c is zero.
        double t = sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
        c = (t - sum) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
        sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
        //Next time around, the lost low part will be added to y in a fresh attempt.
    }
    double yv_al = sum;
    alpha = yv_al / (1 + yv_al);*/
    //TEST//////////////////////////////////////////////////////////////////////
    double betta = 1 / (1 + yv);
    cblas_dscal(basis.num_cols, betta, x_i, 1);
    //cblas_daxpy(basis.num_cols, -1 * alpha, x_i, 1, x_i, 1);
    /*
    double *x_i2 = new_host_darray(basis.num_cols); //column of inverse matrix of gramm
    std::cout << "bi sopr stated\n";
    biSoprGradientStoredMatrix(basis, basis_t, grammMatrParted, e_i, x_i2, 0.0001, x_i);
    double x_razn = 0;
    for(int i = 0; i < basis.num_cols; i++){
        x_razn += fabs(x_i[i] - x_i2[i]);
        //if(fabs(x_i[i] - x_i2[i]) > 0){
        //        std::cout << "x_raxn[" << i << "]=" << x_razn << "\n";
        //}
    }
    std::cout << "Bi sopr Cop = " << x_razn << "\n";
    cblas_dcopy(basis.num_cols, x_i2, 1, x_i, 1);
    delete_host_array(x_i2);
     */
     ///test
    //cblas_dscal(grammMatrParted.num_nonzeros, 0.001, grammMatrParted.Ax, 1);
    
    cblas_dcopy(basis.num_cols, x_i, 1, mu, 1);
    //printMatrixCPU(1, basis.num_cols, mu);
    
    free(x_i);
    free(e_i);
    //free(x_row_summ);
}

//template <typename int, typename double>

double getZDistance(double *z, int vectorDim) {
    double summ = 0.0f;
    for (int i = 0; i < vectorDim; i++) {
        summ += z[i] * z[i];
    }

    return sqrt(summ);
}

//template <typename int, typename double>

void invGrammAdd_CPU(int rowsInGramOld, int vectorDim, csr_matrix &basis,
        csr_matrix &basis_t, double *invGrammOldm, double *invGramm) {

    double* B = (double*) malloc((rowsInGramOld + 1) * sizeof (double));
    double* C = (double*) malloc((rowsInGramOld) * sizeof (double));
    double D;

    double* addVect = new_host_darray(vectorDim); //(double*) malloc(vectorDim * sizeof (double));


    //printMatrixFromGPUForOctave(vectorDim, rowsInGramOld + 1, basis);
    //printMatrixFromGPU(rowsInGramOld, rowsInGramOld, invGrammOldm);

    //status = cublasAlloc(rowsInGramOld + 1, sizeof (double), (void**) & B);
    //status = cublasAlloc(rowsInGramOld, sizeof (double), (void**) & C);
    //status = cublasAlloc(1, sizeof (double), (void**) & D);
    //status = cublasAlloc(vectorDim, sizeof (double), (void**) & addVect);
    //cublasScopy(vectorDim, &basis[rowsInGramOld * vectorDim], 1, addVect, 1);
    //memcpy(addVect, &basis[rowsInGramOld * vectorDim], vectorDim * sizeof (double));
    get_dense_column(basis, rowsInGramOld, addVect);

    //printMatrixFromGPU(vectorDim, 1, addVect);
    //cblas_sgemv(CblasColMajor, CblasTrans, vectorDim, rowsInGramOld + 1, 1.0, basis, vectorDim, addVect, 1, 0, B, 1);
    //spmv_csr_t_serial_host(basis, addVect, B);
    spmv_csr_serial_host(basis_t, addVect, B);

    //cublasSgemv('t', vectorDim, rowsInGramOld + 1, 1.0, basis, vectorDim, addVect, 1, 0, B, 1);
    //printMatrixFromGPU(rowsInGramOld + 1, 1, B_dev);
    //transpose_GPU(rowsInGramOld, 1, B, C);
    cblas_dcopy(rowsInGramOld, B, 1, C, 1); //Для более высокой размерности тут болжно быть транспонирование
    //printMatrixCPU(vectorDim, 1, addVect);
    //printMatrixFromGPU(rowsInGramOld, 1, C_dev);
    //double D_host = cublasSdot(rowsInGramOld, B_dev, 1, B_dev, 1);
    //status = cublasSetVector(1, sizeof (double), &D_host, 1, D_dev, 1);
    //cublasScopy((1), &B[rowsInGramOld], 1, D, 1);
    D = B[rowsInGramOld];

    int mtrxCols = rowsInGramOld + 1;
    int mtrxRows = rowsInGramOld + 1;
    int t = rowsInGramOld + 1; // - 1;
    int p = rowsInGramOld; // (t+1) /2
    //printf("P = %i\n T = %i\n",p, t);
    if (t > 1) { //t > 0

        //auxiliary matrices
        double* X = (double*) malloc((t - p) * p * sizeof (double));
        double* Y = (double*) malloc((t - p) * (t - p) * sizeof (double));
        double* N = (double*) malloc((t - p) * (t - p) * sizeof (double));
        double* M = (double*) malloc((t - p) * p * sizeof (double));
        double* L = (double*) malloc(p * (t - p) * sizeof (double));
        for (int i = 0; i < (t - p) * p; i++) {
            M[i] = 0.0;
            L[i] = 0.0;
            X[i] = 0.0;
        }
        //double* K_dev;
        //status = cublasAlloc((t - p) * p, sizeof (double), (void**) & X_dev);
        //status = cublasAlloc((t - p) * (t - p), sizeof (double), (void**) & Y_dev);
        //status = cublasAlloc((t - p) * (t - p), sizeof (double), (void**) & N_dev);
        //status = cublasAlloc((t - p) * p, sizeof (double), (void**) & M_dev);
        //status = cublasAlloc(p * (t - p), sizeof (double), (void**) & L_dev);
        //if (status != CUBLAS_STATUS_SUCCESS) {
        //    fprintf(stderr, "!!!! device memory allocation error (A)\n");
        //}


        //cublasSgemm('N', 'N', (t - p), p, p, 1.0, C, t - p, invGrammOldm, p, 0.0, X_dev, t - p);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (t - p), p, p, 1.0, C, t - p, invGrammOldm, p, 1.0, X, t - p);

        //printMatrixFromGPU(t-p,p, X_dev);
        //printf("Y <- X * B \n");
        //cublasSgemm('N', 'N', t - p, (t - p), p, 1.0, X_dev, t - p, B, p, 0.0, Y_dev, t - p);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, t - p, (t - p), p, 1.0, X, t - p, B, p, 0.0, Y, t - p);
        //printMatrixFromGPU(t-p,t-p, Y_dev);
        //printf("Y <- D - Y \n");
        //cublasSaxpy((t - p) * (t - p), -1.0, D, 1, Y_dev, 1);
        cblas_daxpy((t - p) * (t - p), -1.0, &D, 1, Y, 1);
        //cublasSscal((t - p) * (t - p), -1.0, Y_dev, 1);
        cblas_dscal((t - p) * (t - p), -1.0, Y, 1);
        //printMatrixFromGPU(t-p,t-p, Y_dev);
        //printf("Y <- inv Y \n");
        //blocInvSymMtrx(t - p, Y_dev);
        //*Y = 1 / (*Y);
        Y[0] = 1 / Y[0];
        //std::cout<< "X: \n";
        //printMatrixCPU(1, p, X);
        //printf("M <- - Y * X \n");
        //cublasSgemm('N', 'N', (t - p), p, (t - p), -1.0, Y, (t - p), X, (t - p), 0.0, M, (t - p));
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (t - p), p, (t - p), -1.0, Y, (t - p), X, (t - p), 0.0, M, (t - p));
        //printMatrixFromGPU(t-p,p, M_dev);
        //printf("L <- M' \n");
        //if (t - p > 1 && p > 1) {
        //    transpose_GPU((t - p), p, M, L);
        //} else {
        //cublasScopy((t - p) * p, M, 1, L, 1);
        cblas_dcopy((t - p) * p, M, 1, L, 1);
        //}
        //printMatrixFromGPU(p,t-p, L_dev);
        //printf("A <- A - L * X \n");
        //printMatrixFromGPU(p,p, invGrammOldm);
        //printMatrixFromGPU(t-p,p, X_dev);
        //printMatrixFromGPU(t-p,p, M_dev);
        //cublasSgemm('N', 'N', p, p, t - p, -1.0, L, p, X, t - p, 1.0, invGrammOldm, p);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, p, p, t - p, -1.0, L, p, X, t - p, 1.0, invGrammOldm, p);
        //printMatrixFromGPU(p,p, A_dev);
        putSubMatrix_CPU(mtrxRows, mtrxCols, invGramm, (int) 0, (int) 0, p, p, invGrammOldm);
        //printf("put sub matrix\n");
        putSubMatrix_CPU(mtrxRows, mtrxCols, invGramm, p, (int) 0, (t - p), p, M);
        //putSubMatrix(mtrxRows, mtrxCols, Matrix, 0, p, (t - p), p, M_dev);
        //printf("put sub matrix\n");
        putSubMatrix_CPU(mtrxRows, mtrxCols, invGramm, (int) 0, p, p, (t - p), L);
        //putSubMatrix(mtrxRows, mtrxCols, Matrix, p, 0, p, (t - p), L_dev);
        //printf("put sub matrix\n");
        putSubMatrix_CPU(mtrxRows, mtrxCols, invGramm, p, p, (t - p), (t - p), Y);

        //printMatrixFromGPU(rowsInGramOld + 1, rowsInGramOld + 1, invGramm);

        free(X); //status = cublasFree(X);
        free(Y); //status = cublasFree(Y);
        free(N); //status = cublasFree(N);
        free(M); //status = cublasFree(M);
        free(L); //status = cublasFree(L);
    } else {
        //double a = cublasSdot(vectorDim, basis, 1, basis, 1);
        double *AA = new_host_darray((int) 1);
        //__mm_csr_serial_host(basis, basis_t, AA);
        mm_csr_serial_host(basis, basis_t, AA);
        double a = AA[0];

        a = 1.0 / a;
        invGramm[0] = a;
        //cublasSetVector(1, sizeof (a), &a, 1, invGramm, 1);
    }



    free(C); //status = cublasFree(C);
    free(B); //status = cublasFree(B);
    //free(D); //status = cublasFree(D);

    free(addVect); //status = cublasFree(addVect);
}


//template <typename int, typename double>

void invGrammDel_CPU(int rowsIG, int collsIG, double *invGrammOld, double *invGramm, int vecToDelId) {

    double* swpVec = new_host_darray(rowsIG);

    //tmp_vect = invGrammOld(:, columns(invGrammOld))
    cblas_dcopy(rowsIG, &invGrammOld[(collsIG - 1) * rowsIG], 1, swpVec, 1);
    //invGrammOld(:, columns(invGrammOld)) = (-1)^(columns(invGrammOld) - vecToDelID - 1).* invGrammOld(:, vecToDelID)
    cblas_dcopy(rowsIG, &invGrammOld[vecToDelId * rowsIG], 1, &invGrammOld[(collsIG - 1) * rowsIG], 1);
    cblas_dscal(rowsIG, pow(-1.0, (collsIG - vecToDelId - 1)), &invGrammOld[(collsIG - 1) * rowsIG], 1);
    //invGrammOld(:, vecToDelID) = (-1)^(columns(invGrammOld) - vecToDelID -1).* tmp_vect
    cblas_dcopy(rowsIG, swpVec, 1, &invGrammOld[vecToDelId * rowsIG], 1);
    cblas_dscal(rowsIG, pow(-1.0, (collsIG - vecToDelId - 1)), &invGrammOld[vecToDelId * rowsIG], 1);

    //tmp_vect = invGrammOld(columns(invGrammOld), :)
    for (int i = 0; i < collsIG; i++) {
        swpVec[i] = invGrammOld[i * rowsIG + rowsIG - 1];
    }
    //invGrammOld(columns(invGrammOld), :) = (-1)^(columns(invGrammOld) - vecToDelID - 1).* invGrammOld(vecToDelID, :)
    double scaleCoef = pow(-1.0, collsIG - vecToDelId - 1);
    for (int i = 0; i < collsIG; i++) {
        invGrammOld[i * rowsIG + rowsIG - 1] = scaleCoef * invGrammOld[i * rowsIG + vecToDelId];
    }
    //invGrammOld(vecToDelID, :) = (-1)^(columns(invGrammOld) - vecToDelID -1).* tmp_vect
    for (int i = 0; i < collsIG; i++) {
        invGrammOld[i * rowsIG + vecToDelId] = scaleCoef * swpVec[i];
    }
    //invGr = invGrammOld(1:(columns(invGrammOld) - 1), 1:(columns(invGrammOld) -1)) -  invGrammOld(1:(columns(invGrammOld) -1),(columns(invGrammOld) -1))*
    //(1/invGrammOld(columns(invGrammOld),columns(invGrammOld) )) *
    //invGrammOld((columns(invGrammOld) -1), 1:(columns(invGrammOld) -1) )
    cblas_dcopy(rowsIG, &invGrammOld[(collsIG - 1) * rowsIG], 1, swpVec, 1);


    for (int i = 0; i < collsIG; i++) {
        cblas_dcopy(rowsIG - 1, &invGrammOld[i * rowsIG], 1, &invGramm[i * (rowsIG - 1)], 1);
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, rowsIG, rowsIG, 1, -1.0 / (swpVec[collsIG - 1]), swpVec, rowsIG, swpVec, 1, 1.0, invGramm, rowsIG);
}


//template <typename int, typename double>

void invGrammAddFulBasis_CPU(int rowsInGramOld, int vectorDim, csr_matrix &basis,
        csr_matrix &basis_t, double *invGrammOldm, double *invGramm) {

    double* B = (double*) malloc((rowsInGramOld + 1) * sizeof (double));
    double* C = (double*) malloc((rowsInGramOld + 1) * sizeof (double));
    double D;

    double* addVect = (double*) malloc(basis.num_rows * sizeof (double));


    get_dense_column(basis, rowsInGramOld, B);
    //printMatrixCPU(rowsInGramOld + 1, 1, B);

    get_dense_column(basis_t, rowsInGramOld, C);
    //printMatrixCPU(rowsInGramOld + 1, 1, C);

    D = B[rowsInGramOld];

    int mtrxCols = rowsInGramOld + 1;
    int mtrxRows = rowsInGramOld + 1;
    int t = rowsInGramOld + 1; // - 1;
    int p = rowsInGramOld; // (t+1) /2
    //printf("P = %i\n T = %i\n",p, t);
    if (t > 1) { //t > 0

        //auxiliary matrices
        double* X = (double*) malloc((t - p) * p * sizeof (double));
        double* Y = (double*) malloc((t - p) * (t - p) * sizeof (double));
        double* N = (double*) malloc((t - p) * (t - p) * sizeof (double));
        double* M = (double*) malloc((t - p) * p * sizeof (double));
        double* L = (double*) malloc(p * (t - p) * sizeof (double));


        //X<-InvGramOld * B
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (t - p), p, p, 1.0, invGrammOldm, t - p, B, p, 0.0, X, t - p);
        //Y

        free(X); //status = cublasFree(X);
        free(Y); //status = cublasFree(Y);
        free(N); //status = cublasFree(N);
        free(M); //status = cublasFree(M);
        free(L); //status = cublasFree(L);
    } else {
        //double a = cublasSdot(vectorDim, basis, 1, basis, 1);
        double *AA = new_host_darray((int) 1);
        //__mm_csr_serial_host(basis, basis_t, AA);
        mm_csr_serial_host(basis, basis_t, AA);
        double a = AA[0];

        a = 1.0 / a;
        invGramm[0] = a;
        //cublasSetVector(1, sizeof (a), &a, 1, invGramm, 1);
    }



    free(C); //status = cublasFree(C);
    free(B); //status = cublasFree(B);
    //free(D); //status = cublasFree(D);

    free(addVect); //status = cublasFree(addVect);
}


//template <typename int, typename double>

void putSubMatrix_CPU(int mtrxRows, int mtrxCols, double* Matrix, int subMtrxX,
        int subMtrxY, int subMtrxRows, int subMtrxCols, double *subMtrx) {
    if ((subMtrxY + subMtrxCols) > mtrxCols || (subMtrxX + subMtrxRows) > mtrxRows) {
        printf("Parammeters of submatrix is inconsistent\n");
        printf("matrix rows %i matrix cols %i subMtrxX %i subMtrxY %i subMtrxRows %i subMtrxCols %i \n",
                mtrxRows, mtrxCols, subMtrxX, subMtrxY, subMtrxRows, subMtrxCols);
        //exit;
    }
    //копируем матрицу по столбцам
    int endCol = subMtrxCols + subMtrxY;
    for (int mtrxCol = subMtrxY, subMtrxCol = 0; mtrxCol < endCol; mtrxCol++, subMtrxCol++) {
        //int col =
        //cublasScopy(subMtrxRows, &subMtrx[subMtrxCol * subMtrxRows], 1,
        //        &Matrix[mtrxCol * mtrxRows + subMtrxX], 1);
        cblas_dcopy(subMtrxRows, &subMtrx[subMtrxCol * subMtrxRows], 1,
                &Matrix[mtrxCol * mtrxRows + subMtrxX], 1);
    }

}

/**
 *Подтянуть выпавшую точку в базис
 **/
//template <typename int, typename double>

double attractDotCoeffCPU(csr_matrix &basis, double *mu_old, double *mu, int &delBasElIndx, int baselen, int eqConNum) {
    /*
    double *Bmu = new_host_darray(baselen);
    double *Bmu_old = new_host_darray(baselen);

    spmv_csr_serial_host(basis, mu, Bmu);
    spmv_csr_serial_host(basis, mu_old, Bmu_old);
    
    std::cout << "B * mu = \n";
    printMatrixCPU(1, basis.num_rows, Bmu);

    double phi = 0.0;
     *delBasElIndx = - 1;
    //this for case withiout eq conditions    
    for (int i = eqConNum; i < baselen - 1; i++) {
        double phi_tmp = Bmu_old[i] / (Bmu_old[i] - Bmu[i]);                
        if (  phi_tmp > phi) {
            phi = phi_tmp;
     *delBasElIndx = i;
        }
    }
    double phi_tmp = (-Bmu_old[baselen - 1] +1)/ (Bmu[baselen - 1] - Bmu_old[baselen - 1]);
    if (  phi_tmp > phi) {
            phi = phi_tmp;
     *delBasElIndx = baselen - 1;
    }    
    std::cout << "Parameter of attraction t= " << phi << "and min el inx = " << *delBasElIndx << "\n";
    double t = phi;
    double t2 = 1 - t;
    for (int i = 0; i < baselen; i++) {
        mu[i] = t * mu[i] + t2 * mu_old[i];
    }

    delete_host_array(Bmu);
    delete_host_array(Bmu_old);
    return phi;
     */
    double *mu_razn = (double*) malloc(baselen * sizeof (double));
    double *mu_div = (double*) malloc(baselen * sizeof (double));

    double minPozVal = 100000.0;
    //this for case withiout eq conditions
    //for (int i = 0; i < baselen; i++) {
    for (int i = eqConNum; i < baselen; i++) {
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


/**
 *Делит вектора поэлементно resultVec = vector1 ./ vector2
 *vector1 - первый вектор
 *vector2 - второй вектор
 *resultVector - результирующий вектор
 *vectorDim - размерность векторов
 **/
//void vectDiv(double *vector1, double *vector2, double *resultVect, int vectorDim){
//    for(int i = 0; i < vectorDim; i++){
//        resultVect[i] = vector1[i] / vector2[i];
//    }
//}

/**
 *Вычисляет разность векторов resultVec = vector1 - vector2
 *vector1 - первый вектор
 *vector2 - второй вектор
 *resultVector - результирующий вектор
 *vectorDim - размерность векторов
 **/
//template <typename int, typename double>

void vectRazn(double *vector1, double *vector2, double *resultVect, int vectorDim) {
    for (int i = 0; i < vectorDim; i++) {
        resultVect[i] = vector1[i] - vector2[i];
    }
}

/**
 *Выделяет наименьший элемент вектора
 **/
//template <typename int, typename double>

double getMinVectorElCPU(double *vector, int vectorDim, int &minVecIdx, int numEqCon) {
    double min = vector[0];
    //////////////////////////This is for case without eq constrains
    //for (int i = 1; i < vectorDim; i++) {
    for (int i = numEqCon; i < vectorDim; i++) {
        if (vector[i] < min) {
            min = vector[i];
            minVecIdx = i;
        }
    }
    return min;
}

double getMaxVectorElCPU(double *vector, int vectorDim, int &maxVecIdx, int numEqCon) {
    double max = vector[0];
    //////////////////////////This is for case without eq constrains
    //for (int i = 1; i < vectorDim; i++) {
    for (int i = numEqCon; i < vectorDim; i++) {
        if (vector[i] > max) {
            max = vector[i];
            maxVecIdx = i;
        }
    }
    return max;
}

/**
 *Эта функция вычисляет Умножение матрицы на вектор mtrx * vector = result
 *Матрица задана как массив по столбцам.
 *mtrx - указатель на матрицу
 *vector - указатель на вектор
 *result - результат умножения
 *mtrxStrNum - количество строк в матрице
 *mtrx Col Num - соличество столбцов
 **/
//template <typename int, typename double>

void multMatrToVectorCPU(double *mtrx, double *vector, double *result, int mtrxStrNum, int mtrxColNum) {
    for (int str = 0; str < mtrxStrNum; str++) {
        result[str] = 0;
        for (int col = 0; col < mtrxColNum; col++) {
            result[str] += mtrx[str + col * mtrxStrNum] * vector[col];
        }
    }
}

/*
 *Эта функция копирует векторы из входного множества в память отведённую под базис
 *basis - указатель на участок памяти для хранения базиса
 *inputSet - входное множество
 *rvec - вектор принадлежности к базису, если kvec[i] == 1, товектор принадлежит к базису
 *baselen - количество векторов в базисе
 *inSetDim - размерность входного множества
 *размерность пространства векторов
 */
//template <typename int, typename double>

void fillBasis_CPU(double* basis, double* inputSet, int* kvec, int baselen, int inSetDim, int vectorDim) {
    int basisVec = 0;
    for (int vecIter = 0; vecIter < inSetDim; vecIter++) {
        if (kvec[vecIter] == 1 && basisVec < baselen) {
            memcpy(&basis[basisVec * vectorDim], &inputSet[vecIter * vectorDim], vectorDim * sizeof (double));
            basisVec++;
        }
    }
}

/*
 *Эта функция формирует структуру [basis; E] для решегия системы X * mu = 0; e' * mu = 1
 *Эта функция копирует векторы из входного множества в память отведённую под базис, и
 *заполняет последний элемент каждого вектора 1, тк размерность вектора на 1 меньше
 *количества строк в новой матрице
 *basisAndE - указатель на участок памяти для хранения матрицы
 *inputSet - входное множество
 *rvec - вектор принадлежности к базису, если kvec[i] == 1, товектор принадлежит к базису
 *baselen - количество векторов в базисе
 *inSetDim - размерность входного множества
 *размерность пространства векторов
 */
//template <typename int, typename double>

void fillBasisAndE(double* basis, double* inputSet, int inSetDim, int vectorDim) {
    int strsCount = vectorDim + 1;
    for (int vecIter = 0; vecIter < inSetDim; vecIter++) {
        memcpy(&basis[vecIter * strsCount], &inputSet[vecIter * vectorDim], vectorDim * sizeof (double));
        basis[vecIter * strsCount + vectorDim] = 1.0f;
    }
}

/**
 *Транспонирование матрицы
 *srcMtx - матрица которую транспонируем
 *trgMtrx - транспонированная матрица
 *srcMtrxStrNum - количество строк в исходной матрице
 *srcMtrxColNum - количество столбцов и исходной матрице
 */
//template <typename int, typename double>

void transMtrx(double *srcMtrx, double *trgMtrx, int srcMtrxStrNum, int srcMtrxColNum) {
    for (int colId = 0; colId < srcMtrxColNum; colId++) {
        for (int strId = 0; strId < srcMtrxStrNum; strId++) {
            trgMtrx[strId * srcMtrxColNum + colId] = srcMtrx[colId * srcMtrxStrNum + strId];
        }
    }
}

/**Умножает Транспонированную матрицу на матрицу X' * X
 * matrix1 - матрица1 !!! Не транспонированная
 * matrix2 - матрица2
 * matrixMul - X' * X результат умножения нетранспонированной первой матрицы на вторую
 * m1_rows - количество строк в транспонированной первой матрице
 * m2_cols - количество столбцов во второй матрице
 * m1_cols - количество столбцов в транспонированной второй матрице
 *matrixMul[0:matrix1_rows; 0:matrix2_colls]
 */
//template <typename int, typename double>

void mulTMtrx_to_Mtrx(double *matrix1, double *matrix2, double *matrixMul, int m1_rows, int m2_cols, int m1_cols) {
    int vec1_count = m1_rows;
    int vec2_count = m2_cols;
    int vec_dim = m1_cols;

    for (int j = 0; j < vec2_count; j++) {//цыкл по строкам получаюшейся матрицы
        for (int i = 0; i < vec1_count; i++) {//скалярн произв столбеу стлбец
            matrixMul[j * vec1_count + i] = 0.0;
            for (int k = 0; k < vec_dim; k++) {//цыкл по скалярному произведению
                matrixMul[j * vec1_count + i] += matrix1[i * vec_dim + k] * matrix2[j * vec_dim + k];
            }
        }
    }
}

/*Функция вычисляет сумму матрицы по строкам, результатом является столбец, каждый
 *элемент которого представляет из себя сумму элементов матрицы по соответствующей строке
 *
 */
//template <typename int, typename double>

void sumMtrxByStr(double *srcMtrx, double *sumVector, int srcMtrxRows, int srcMtrxCols) {
    for (int strId = 0; strId < srcMtrxRows; strId++) {
        sumVector[strId] = 0;
        for (int colId = 0; colId < srcMtrxCols; colId++) {
            sumVector[strId] += srcMtrx[colId * srcMtrxRows + strId];
        }
    }
}

/**
 *Вычисляем сумму вектора. Результатом является сумма элементов вектора
 */
//template <typename int, typename double>

void sumVecElmts(double *vector, double *summ, int vectorDim) {
    *summ = 0.0f;
    for (int i = 0; i < vectorDim; i++) {
        *summ += vector[i];
    }
}

/**
 *Умножение вектора на число
 */

//template <typename int, typename double>

void multVectorToValue(double *vector, double ratio, int vectorDim) {

    for (int i = 0; i < vectorDim; i++) {
        vector[i] = ratio * vector[i];
    }
}

//template <typename int, typename double>

void getMinVCosCPU(double* z, csc_matrix &inSet, double& minVcos, int &minVecId, double epsilon, int numEqCon, double* l_norms) {
    double* vcos = new_host_darray(inSet.num_cols); //(double *) malloc(inSetDim * sizeof (double));
    /*for (int i = 0; i < inSet.num_cols; i++) {
        vcos[i] = 0;
    }*/

    //cblas_sgemv(CblasColMajor, CblasTrans, vectorDim, inSetDim, 1.0f, inSet, vectorDim, z, 1, 0, vcos, 1);
    //spmv_csr_t_serial_host(inSet, z, vcos);
    double z_norm = cblas_dnrm2(inSet.num_rows, z, 1);
    A_dot_x_csc_serial_host(inSet, z, vcos);
    for(int col_i = 0; col_i < inSet.num_cols; col_i++){
        vcos[col_i] /= l_norms[col_i] * z_norm;
    }
    //std::cout << "Vcos>>>>>>>>>>>";
    //printMatrixCPU(inSetDim, (int)1,vcos);
    ///////////////////for vase wisout eq constrains
    //*minVcos = vcos[0] - zz;
    minVcos = vcos[numEqCon];// - 1];
    //printMatrixCPU(inSetDim, (int)1,vcos);
    ///////////////////for vase wisout eq constrains
    //*minVecId = 0;
    minVecId = numEqCon;// - 1;
    //printf("init vcos:%f \n ", *minVcos);
    //int oldminId = 0;
    //double oldminVal = 0;
    ////////////////jast for one
    //for (int elId = 0; elId < inSetDim; elId++) {
    //double minRaznost = 0;
    if (inSet.num_cols >= numEqCon) {
        for (int elId = numEqCon; elId < inSet.num_cols; elId++) {
            double curentVcos = vcos[elId];
            //printf("vcos[%i] = %e\n", elId, curentVcos);
            if (curentVcos < minVcos) {
                //if(abs(curentVcos - *minVcos) > epsilon){
                //oldminId = *minVecId;
                //oldminVal = *minVcos;
                //minRaznost = minVcos - curentVcos; 
                minVcos = curentVcos;
                minVecId = elId;
            }
        }
    }

    //printf(" %i  value %e \n ", , *);
    std::cout << "Min vcos id:" << minVecId << " value" << minVcos <<"\n";
    free(vcos);
}

/**
 *Helper function that evaluate scalar prod on CPU
 */
//template <typename int, typename double>

void evalScalarCPU(int x_dimm, //number of vectors in X set
        int vector_size, //numbaer of elements in single vector
        double *x_set, //pointer to x set
        double* z_vector, //pointer to vector Z
        double* dot_products //pointer to store scalar prod out
        ) {
    //for each vector in input x_set
    for (int vectCntr = 0; vectCntr < x_dimm; vectCntr++) {
        double summ = 0;
        int base = vectCntr * vector_size;
        //for each element of vector
        for (int elemCntr = 0; elemCntr < vector_size; elemCntr++) {
            summ += x_set[base + elemCntr] * z_vector[elemCntr];
        }
        dot_products[vectCntr] = summ;
    }
}


//Печатает матрицу представленную в вектороном виде [(вектор1) (вектор2)...}

//template <typename int, typename double>
/*void printMatrixCPU(int str_num, int coll_num, double* matrix) {
    printf("\n");
    for (int i = 0; i < str_num; i++) {
        for (int j = 0; j < coll_num; j++) {
            //printf("%f ", matrix[i * coll_num + j]);
            printf("%f ", matrix[j * str_num + i]);
        }
        printf("\n");
    }
}
 */
//template <typename int, typename double>

void printMatrixForOctaveCPU(int str_num, int coll_num, double* matrix) {
    printf("\n");
    for (int i = 0; i < str_num; i++) {
        for (int j = 0; j < coll_num; j++) {
            if (j == (coll_num - 1)) {
                //printf("%f ", matrix[i * coll_num + j]);
                printf("%f ", matrix[j * str_num + i]);
            } else {
                //printf("%f, ", matrix[i * coll_num + j]);
                printf("%f, ", matrix[j * str_num + i]);
            }
        }
        if (i != (str_num - 1)) {
            printf(";");
        } else {
            printf("");
        }
    }
}

//template <typename int, typename double>

void inverseMatrixCPU(double* grammMatrix, double* invMatrix, int mtrxRows) {
    memcpy(invMatrix, grammMatrix, mtrxRows * mtrxRows * sizeof (double));
    int success = clapack_dpotrf(CblasColMajor, CblasUpper, mtrxRows, invMatrix, mtrxRows);
    if (success == 0) {

        success = clapack_dpotri(CblasColMajor, CblasUpper, mtrxRows, invMatrix, mtrxRows);
        if (success != 0) {
            std::cout << "Matrix is not inversed right\n";
        }
        for (int i = 0; i < mtrxRows; i++) {
            for (int j = i; j < mtrxRows; j++) {
                invMatrix[i * mtrxRows + j] = invMatrix[j * mtrxRows + i];
                //printf("%f ", A[j * row_num + i]);
            }
            //printf("\n");
        }
    } else {
        printf("AHTUNG Matrix is not positive\n");

        memcpy(invMatrix, grammMatrix, mtrxRows * mtrxRows * sizeof (double));
        //printMatrixCPU(mtrxRows, mtrxRows, invMatrix);

        std::cout << "\n";
        int* ipiv = new_host_iarray(mtrxRows);
        int success = clapack_dgetrf(CblasColMajor, mtrxRows, mtrxRows, invMatrix, mtrxRows, ipiv);

        //printMatrixCPU(mtrxRows, mtrxRows, invMatrix);
        if (success != 0) {
            std::cout << "Matrix is not factorized by LU right\n";
        }
        success = clapack_dgetri(CblasColMajor, mtrxRows, invMatrix, mtrxRows, ipiv);
        if (success != 0) {
            std::cout << "Matrix is not inversed right\n";
        }
    }
}

//template <typename int, typename double>

void inverseNotSymMatrixCPU(double* invMatrix, int mtrxRows) {
    int success = clapack_dpotrf(CblasColMajor, CblasUpper, mtrxRows, invMatrix, mtrxRows);
    if (success != 0) {
        printf("AHTUNG Matrix is not positive\n");
    }
    clapack_dpotri(CblasColMajor, CblasUpper, mtrxRows, invMatrix, mtrxRows);
    for (int i = 0; i < mtrxRows; i++) {
        for (int j = i; j < mtrxRows; j++) {
            invMatrix[i * mtrxRows + j] = invMatrix[j * mtrxRows + i];
            //printf("%f ", A[j * row_num + i]);
        }
        //printf("\n");
    }
}

void biSoprGradient(const csr_matrix& basis, const csr_matrix& basis_t, double *b, double *x, double eps, double *x0) {
    double *x_i = new_host_darray(basis.num_cols);

    if (x0 == NULL) {
        for (int x0_i = 0; x0_i < basis.num_cols; x0_i++) {
            x_i[x0_i] = 0.0;
        }
    } else {
        cblas_dcopy(basis.num_cols, x0, 1, x_i, 1);
    }
    double *r_i = new_host_darray(basis.num_cols);

    double *rtild_i = new_host_darray(basis.num_cols);
    double *p_i = new_host_darray(basis.num_cols);
    double *ptild_i = new_host_darray(basis.num_cols);
    //A*x
    double *gramm_to_x = new_host_darray(basis.num_cols);
    mulGrammToX(basis_t, basis, x_i, gramm_to_x);



    //r_0 = b - A* x_0
    cblas_dcopy(basis.num_cols, b, 1, r_i, 1);
    cblas_daxpy(basis.num_cols, -1.0, gramm_to_x, 1, r_i, 1);

    //set rtild so <rtild,r> != 0
    cblas_dcopy(basis.num_cols, r_i, 1, rtild_i, 1);
    //p_0 = r_0
    cblas_dcopy(basis.num_cols, r_i, 1, p_i, 1);
    cblas_dcopy(basis.num_cols, rtild_i, 1, ptild_i, 1);

    double ri_to_rtildi = 0;
    double alpha = 0;
    double betta = 0;
    while (sqrt(cblas_ddot(basis.num_cols, r_i, 1, r_i, 1)) > eps * eps * eps) { //eps * eps * eps * eps
        //aplha = (r_i, rtild_i)/(Ap_i, ptild_i)
        ri_to_rtildi = cblas_ddot(basis.num_cols, r_i, 1, rtild_i, 1);
        /////////////////
        //std::cout << "========================New Iteration==================>\n";
        //std::cout << "===>r_i vector \n";
        //printMatrixCPU(1, basis.num_cols, r_i);

        //std::cout << "===>rtild_i vector \n";
        //printMatrixCPU(1, basis.num_cols, rtild_i);

        //std::cout << "ri_to_rtildi = " << ri_to_rtildi << std::endl;
        //////////////////
        mulGrammToX(basis_t, basis, p_i, gramm_to_x);

        //////////////////////TEST
        /*
        double *grammMatr = new_host_darray(basis.num_cols * basis.num_cols);
        mm_csr_serial_host(basis_t, basis, grammMatr);
        std::cout << "===>Check gramma _ to p_i";
        double *gramm_to_x_temp = new_host_darray(basis.num_cols);
        cblas_dgemv(CblasColMajor, CblasNoTrans, basis.num_cols, basis.num_cols, 1.0, grammMatr, basis.num_cols, p_i, 1, 0, gramm_to_x_temp, 1);

        printMatrixCPU(1, basis.num_cols, gramm_to_x_temp);
        std::cout << "===>Real gramma _ to p_i";
        printMatrixCPU(1, basis.num_cols, gramm_to_x);

        std::cout << "===>p_tild_i = ";
        printMatrixCPU(1, basis.num_cols,  ptild_i);
         */
        ///////////////////////TEST


        alpha = ri_to_rtildi / cblas_ddot(basis.num_cols, gramm_to_x, 1, ptild_i, 1);

        //std::cout << "alpha = " << alpha << std::endl;

        //x_(i+1) = x_i + alpha * p_i
        cblas_daxpy(basis.num_cols, alpha, p_i, 1, x_i, 1);
        //r_(i+1) = r_i - alpha * A * p_j
        cblas_daxpy(basis.num_cols, -1.0 * alpha, gramm_to_x, 1, r_i, 1);
        //std::cout << "===>R_I criteriy vihoda";
        //printMatrixCPU(1, basis.num_cols, r_i);
        double ri_ri = sqrt(cblas_ddot(basis.num_cols, r_i, 1, r_i, 1));
        //std::cout << "sqrt(R_i * R_i) = " << ri_ri << std::endl;
        //rtild_(i+1) = rtild_i - alpha * A^t * ptild_i;
        mulGrammToX(basis_t, basis, ptild_i, gramm_to_x);
        cblas_daxpy(basis.num_cols, -1.0 * alpha, gramm_to_x, 1, rtild_i, 1);
        //betta = (r_(i+1), rtild_(i+1))/(r_i, rtild_i)
        betta = cblas_ddot(basis.num_cols, r_i, 1, rtild_i, 1) / ri_to_rtildi;
        if (betta == 0) {//sqrt(betta * betta) < eps*eps ){
            std::cout << "betta = " << betta << std::endl;
            break;
        }
        //p_(i+1) = r_i + betta * p_i
        //use gramm_to_x as temprorary data storage
        cblas_dcopy(basis.num_cols, r_i, 1, gramm_to_x, 1);
        cblas_daxpy(basis.num_cols, betta, p_i, 1, gramm_to_x, 1);
        cblas_dcopy(basis.num_cols, gramm_to_x, 1, p_i, 1);
        //ptild_(i+1) = rtild_i + betta * ptild_i
        //use gramm_to_x as temprorary data storage
        cblas_dcopy(basis.num_cols, rtild_i, 1, gramm_to_x, 1);
        cblas_daxpy(basis.num_cols, betta, ptild_i, 1, gramm_to_x, 1);
        cblas_dcopy(basis.num_cols, gramm_to_x, 1, ptild_i, 1);
    }
    cblas_dcopy(basis.num_cols, x_i, 1, x, 1);

    free(x_i);
    free(r_i);
    free(rtild_i);
    free(p_i);
    free(ptild_i);
    free(gramm_to_x);
}

/**
 * Multiply Transposed GrammaMatrix on vector.
 * The matrix is given in form of basis' and basis, x vector is x
 * and output vector that is b/
 */
void mulGrammToXStoredGramPart(const csr_matrix& basis_t, const csr_matrix& basis, const csr_matrix& gramm_part, double *x, double *p) {
    //std::cout << "basis_t.num_rows: " << basis_t.num_rows << "\n";
    for (int i = 0; i < basis_t.num_rows; i++) {
        p[i] = 0.0;
    }

    //double *bas_t_row = new_host_darray(basis_t.num_cols);
    double *gramm_row = new_host_darray(basis_t.num_rows);
    //double *gramm_row_test = new_host_darray(basis_t.num_rows);



    for (int bas_t_row_idx = 0; bas_t_row_idx < gramm_part.num_rows; bas_t_row_idx++) {
        //this mingled invocation simply get row of basis_t
        get_dense_row(gramm_part, bas_t_row_idx, gramm_row);

        //std::cout << "Gram row row raw \n";
        //printMatrixCPU(1, basis_t.num_rows, gramm_row);
        double element_1 = basis_t.Ax[basis_t.Ap[bas_t_row_idx + 1] - 1];
        for (int row_el_i = 0; row_el_i < basis_t.num_rows; row_el_i++) {
            //std::cout << "gramm_row[row_el_i] = " << gramm_row[row_el_i] << "\n";
            gramm_row[row_el_i] += element_1 * basis_t.Ax[basis_t.Ap[row_el_i + 1] - 1];
            //std::cout <<  "X = " << basis_t.Ax[basis_t.Ap[row_el_i + 1] - 1]  <<" El_x= " << element_1 <<  " All = " << basis_t.num_cols << "\n";
            //std::cout << "After gramm_row[row_el_i] = " << gramm_row[row_el_i] << "\n";
        }

        //Test
        /*evalRowOfGrammMtx(basis_t, bas_t_row_idx, gramm_row_test);
        double gr_row_nev = 0.0;
        for(int nev_i = 0; nev_i < basis_t.num_rows; nev_i++){
            gr_row_nev += fabs(gramm_row_test[nev_i] - gramm_row[nev_i]);
            if(gr_row_nev > 0){
                std::cout<<"CSR: "<<gramm_part.num_cols << " " <<gramm_part.num_nonzeros << "\n";
                print_csr_matrix(gramm_part);
                std::cout << "Col_i "<< nev_i << " neviaz " << gr_row_nev << "All cols=" << basis_t.num_rows << "X = " << basis_t.Ax[basis_t.Ap[nev_i + 1] - 1] <<" El_x= " << element_1 << " All = " << basis_t.num_cols << "\n";
                //printMatrixCPU(1, basis_t.num_rows, gramm_row_test);
                //printMatrixCPU(1, basis_t.num_rows, gramm_row);
            }
        }
        std::cout << "Row "<< bas_t_row_idx << " neviaz " << gr_row_nev << "\n";
        //std::cout << ">";
         */
        p[bas_t_row_idx] = cblas_ddot(gramm_part.num_cols, x, 1, gramm_row, 1);
    }

    //free(bas_t_row);
    free(gramm_row);
}

void biSoprGradientStoredMatrix(const csr_matrix& basis, const csr_matrix& basis_t, const csr_matrix& gramMtxNotFull, double *b, double *x, double eps, double *x0) {
    double *x_i = new_host_darray(basis.num_cols);

    if (x0 == NULL) {
        for (int x0_i = 0; x0_i < basis.num_cols; x0_i++) {
            x_i[x0_i] = 0.0;
        }
    } else {
        cblas_dcopy(basis.num_cols, x0, 1, x_i, 1);
    }
    double *r_i = new_host_darray(basis.num_cols);

    double *rtild_i = new_host_darray(basis.num_cols);
    double *p_i = new_host_darray(basis.num_cols);
    double *ptild_i = new_host_darray(basis.num_cols);
    //A*x
    double *gramm_to_x = new_host_darray(basis.num_cols);
    //////////////////////////////////////////////////////////////////////////////////
    mulGrammToXStoredGramPart(basis_t, basis, gramMtxNotFull, x_i, gramm_to_x);
    //spmv_csr_serial_host(gramMtxNotFull, x_i, gramm_to_x);
    //////////////////////////////////////////////////////////////////////////////////

    //r_0 = b - A* x_0
    cblas_dcopy(basis.num_cols, b, 1, r_i, 1);
    cblas_daxpy(basis.num_cols, -1.0, gramm_to_x, 1, r_i, 1);

    //set rtild so <rtild,r> != 0
    cblas_dcopy(basis.num_cols, r_i, 1, rtild_i, 1);
    //p_0 = r_0
    cblas_dcopy(basis.num_cols, r_i, 1, p_i, 1);
    cblas_dcopy(basis.num_cols, rtild_i, 1, ptild_i, 1);

    double ri_to_rtildi = 0;
    double alpha = 0;
    double betta = 0;
    int iter_counter = 0;
    while (sqrt(cblas_ddot(basis.num_cols, r_i, 1, r_i, 1)) > eps * eps * eps && iter_counter < 1000) { //eps * eps * eps * eps
        iter_counter++;
        //aplha = (r_i, rtild_i)/(Ap_i, ptild_i)
        ri_to_rtildi = cblas_ddot(basis.num_cols, r_i, 1, rtild_i, 1);
        /////////////////
        //std::cout << "========================New Iteration==================>\n";
        //std::cout << "===>r_i vector \n";
        //printMatrixCPU(1, basis.num_cols, r_i);

        //std::cout << "===>rtild_i vector \n";
        //printMatrixCPU(1, basis.num_cols, rtild_i);

        //std::cout << "ri_to_rtildi = " << ri_to_rtildi << std::endl;
        //////////////////
        //mulGrammToX(basis_t, basis, p_i, gramm_to_x);
        //////////////////////////////////////////////////////////////////////////////////
        mulGrammToXStoredGramPart(basis_t, basis, gramMtxNotFull, p_i, gramm_to_x);
        //spmv_csr_serial_host(gramMtxNotFull, p_i, gramm_to_x);
        //////////////////////////////////////////////////////////////////////////////////
        //////////////////////TEST
        /*
        double *grammMatr = new_host_darray(basis.num_cols * basis.num_cols);
        mm_csr_serial_host(basis_t, basis, grammMatr);
        std::cout << "===>Check gramma _ to p_i";
        double *gramm_to_x_temp = new_host_darray(basis.num_cols);
        cblas_dgemv(CblasColMajor, CblasNoTrans, basis.num_cols, basis.num_cols, 1.0, grammMatr, basis.num_cols, p_i, 1, 0, gramm_to_x_temp, 1);

        printMatrixCPU(1, basis.num_cols, gramm_to_x_temp);
        std::cout << "===>Real gramma _ to p_i";
        printMatrixCPU(1, basis.num_cols, gramm_to_x);

        std::cout << "===>p_tild_i = ";
        printMatrixCPU(1, basis.num_cols,  ptild_i);
         */
        ///////////////////////TEST


        alpha = ri_to_rtildi / cblas_ddot(basis.num_cols, gramm_to_x, 1, ptild_i, 1);

        //std::cout << "alpha = " << alpha << std::endl;

        //x_(i+1) = x_i + alpha * p_i
        cblas_daxpy(basis.num_cols, alpha, p_i, 1, x_i, 1);
        //r_(i+1) = r_i - alpha * A * p_j
        cblas_daxpy(basis.num_cols, -1.0 * alpha, gramm_to_x, 1, r_i, 1);
        //std::cout << "===>R_I criteriy vihoda";
        //printMatrixCPU(1, basis.num_cols, r_i);
        double ri_ri = sqrt(cblas_ddot(basis.num_cols, r_i, 1, r_i, 1));
        //std::cout << "sqrt(R_i * R_i) = " << ri_ri << std::endl;
        //rtild_(i+1) = rtild_i - alpha * A^t * ptild_i;
        //mulGrammToX(basis_t, basis, ptild_i, gramm_to_x);
        //////////////////////////////////////////////////////////////////////////////////
        mulGrammToXStoredGramPart(basis_t, basis, gramMtxNotFull, ptild_i, gramm_to_x);
        //spmv_csr_serial_host(gramMtxNotFull, ptild_i, gramm_to_x);
        /////////////////////////////////////////////////////////////////////////////////
        cblas_daxpy(basis.num_cols, -1.0 * alpha, gramm_to_x, 1, rtild_i, 1);
        //betta = (r_(i+1), rtild_(i+1))/(r_i, rtild_i)
        betta = cblas_ddot(basis.num_cols, r_i, 1, rtild_i, 1) / ri_to_rtildi;
        if (betta == 0) {//sqrt(betta * betta) < eps*eps ){
            std::cout << "betta = " << betta << std::endl;
            break;
        }
        //p_(i+1) = r_i + betta * p_i
        //use gramm_to_x as temprorary data storage
        cblas_dcopy(basis.num_cols, r_i, 1, gramm_to_x, 1);
        cblas_daxpy(basis.num_cols, betta, p_i, 1, gramm_to_x, 1);
        cblas_dcopy(basis.num_cols, gramm_to_x, 1, p_i, 1);
        //ptild_(i+1) = rtild_i + betta * ptild_i
        //use gramm_to_x as temprorary data storage
        cblas_dcopy(basis.num_cols, rtild_i, 1, gramm_to_x, 1);
        cblas_daxpy(basis.num_cols, betta, ptild_i, 1, gramm_to_x, 1);
        cblas_dcopy(basis.num_cols, gramm_to_x, 1, ptild_i, 1);
    }
    cblas_dcopy(basis.num_cols, x_i, 1, x, 1);

    free(x_i);
    free(r_i);
    free(rtild_i);
    free(p_i);
    free(ptild_i);
    free(gramm_to_x);
}

void getMuViaSystemSolve(const csr_matrix& basis, const csr_matrix& basis_t, double *mu, double eps) {
    double *x_i = new_host_darray(basis.num_cols); //column of inverse matrix of gramm
    double *e_i = new_host_darray(basis.num_cols); //column of Unar matrix to pass to system solver as right elemetn
    double *x_row_summ = new_host_darray(basis.num_cols); //vector to store reow summs for inver gramm matrix
    for (int i = 0; i < basis.num_cols; i++) {
        x_i[i] = 0.0;
        e_i[i] = 1.0;
        x_row_summ[i] = 0.0;
    }
    /*for(int i = 0; i < basis.num_cols; i++){
        e_i[i] = 1.0;
        if(i != 0){e_i[i - 1] = 0.0;}

        biSoprGradient(basis, basis_t, e_i, x_i, eps);
        //std::cout << "MatrixCOL from bisopr grad=========================== \n";
        //printMatrixCPU( 1, basis.num_cols, x_i );
        cblas_daxpy(basis.num_cols, 1.0, x_i, 1, x_row_summ, 1);
    }*/

    biSoprGradient(basis, basis_t, e_i, x_i, eps, NULL);
    cblas_dcopy(basis.num_cols, x_i, 1, x_row_summ, 1);

    double elSumm = 0;
    sumVecElmts(x_row_summ, &elSumm, basis.num_cols);
    cblas_dscal(basis.num_cols, 1.0 / elSumm, x_row_summ, 1);

    cblas_dcopy(basis.num_cols, x_row_summ, 1, mu, 1);
    //std::cout << "MU from matrix solver=========================== \n";
    //printMatrixCPU( 1, basis.num_cols, mu );
    free(x_i);
    free(e_i);
    free(x_row_summ);
}

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

void evalCholmodFactor(csr_matrix gMP, ldl_matrix &grammPartFactor) {
    int* Flag = new_host_iarray(gMP.num_rows);
    int* Pattern = new_host_iarray(gMP.num_rows);
    double* Y = new_host_darray(gMP.num_rows);
    /*
    for(int els_iter = 0; els_iter < grammPartFactor.num_nonzeros + 1; els_iter++){
        grammPartFactor.Li[els_iter] = 0.0;
        grammPartFactor.Lx[els_iter] = 0.0;
    }
    for(int els_iter = 0; els_iter < grammPartFactor.num_rows + 1; els_iter++){
        grammPartFactor.Lnz[els_iter] = 0.0;
        grammPartFactor.Parent[els_iter] = 0.0;
        grammPartFactor.Lp[els_iter] = 0.0;
        grammPartFactor.D[els_iter] = 0.0;  
    }
    */
    ldl_symbolic(gMP.num_rows, gMP.Ap, gMP.Aj, grammPartFactor.Lp, grammPartFactor.Parent, grammPartFactor.Lnz, Flag, NULL, NULL);
    printf("Nonzeros in L, excluding diagonal: %d\n", grammPartFactor.Lp[gMP.num_rows]);
    double d = ldl_numeric(gMP.num_rows, gMP.Ap, gMP.Aj, gMP.Ax, grammPartFactor.Lp, grammPartFactor.Parent, grammPartFactor.Lnz,
            grammPartFactor.Li, grammPartFactor.Lx, grammPartFactor.D, Y, Pattern, Flag, NULL, NULL);

    grammPartFactor.num_rows = gMP.num_rows;
    grammPartFactor.num_cols = gMP.num_cols;
    grammPartFactor.num_nonzeros = grammPartFactor.Lp[grammPartFactor.num_cols];
    //print_ldl_matrix(grammPartFactor);
    free(Flag);
    free(Pattern);
    free(Y);
}

void evalCholmodFactorTrans(csr_matrix gMP, ldl_matrix &grammPartFactorTrans) {
    int* Flag = new_host_iarray(gMP.num_rows);
    int* Pattern = new_host_iarray(gMP.num_rows);
    double* Y = new_host_darray(gMP.num_rows);
    /*burn all values with fire*/
    cblas_dscal(grammPartFactorTrans.MAX_D_LP_SIZE, 0.0, grammPartFactorTrans.D, 1);
    cblas_dscal(grammPartFactorTrans.MAX_Li_Lx_SIZE, 0.0, grammPartFactorTrans.Lx, 1);
    
    ldl_matrix grammPartFactor;
    grammPartFactor.num_nonzeros = 0;
    grammPartFactor.num_cols = 0;
    grammPartFactor.num_rows = 0;
    grammPartFactor.D = new_host_darray(grammPartFactorTrans.MAX_D_LP_SIZE);
    grammPartFactor.Lp = new_host_iarray(grammPartFactorTrans.MAX_D_LP_SIZE);
    grammPartFactor.Li = new_host_iarray(grammPartFactorTrans.MAX_Li_Lx_SIZE);
    grammPartFactor.Lx = new_host_darray(grammPartFactorTrans.MAX_Li_Lx_SIZE);
    grammPartFactor.Lnz = new_host_iarray(grammPartFactorTrans.MAX_D_LP_SIZE);
    grammPartFactor.Parent = new_host_iarray(grammPartFactorTrans.MAX_D_LP_SIZE);
    grammPartFactor.MAX_D_LP_SIZE = grammPartFactorTrans.MAX_D_LP_SIZE;
    grammPartFactor.MAX_Li_Lx_SIZE = grammPartFactorTrans.MAX_Li_Lx_SIZE;
    
    
    ldl_symbolic(gMP.num_rows, gMP.Ap, gMP.Aj, grammPartFactor.Lp, grammPartFactor.Parent, grammPartFactor.Lnz, Flag, NULL, NULL);
    printf("Nonzeros in L, excluding diagonal: %d\n", grammPartFactor.Lp[gMP.num_rows]);
    double d = ldl_numeric(gMP.num_rows, gMP.Ap, gMP.Aj, gMP.Ax, grammPartFactor.Lp, grammPartFactor.Parent, grammPartFactor.Lnz,
            grammPartFactor.Li, grammPartFactor.Lx, grammPartFactor.D, Y, Pattern, Flag, NULL, NULL);

    grammPartFactor.num_rows = gMP.num_rows;
    grammPartFactor.num_cols = gMP.num_cols;
    grammPartFactor.num_nonzeros = grammPartFactor.Lp[grammPartFactor.num_cols];
    
    
    ldl_transpose(grammPartFactor, grammPartFactorTrans);
    //print_ldl_matrix(grammPartFactor);
    
    delete_ldl_matrix(grammPartFactor);
    free(Flag);
    free(Pattern);
    free(Y);
}


void addRowCholmodFactor(ldl_matrix &grammPartFactor, csr_matrix grammMatrParted) {
    double* c_12 = new_host_darray(grammMatrParted.num_rows);
    //for(int c_i = 0; c_i < grammMatrParted.num_rows; c_i++){c_12[c_i] = 0.0;}
    //cblas_dscal(grammMatrParted.num_rows, 0.0, c_12, 1);
    int last_row_idx = grammMatrParted.num_rows - 1;
    
    get_dense_row(grammMatrParted, last_row_idx, c_12);
    //cblas_dcopy(grammMatrParted.num_rows, &grammMatrParted.Ax[grammMatrParted.Ap[grammMatrParted.num_rows - 1]], 1, c_12, 1);
    //print_csr_matrix(grammMatrParted);
    //printMatrixCPU(1, grammMatrParted.num_rows, c_12);
    
        
    ldl_lsolve(grammMatrParted.num_rows - 1, c_12, grammPartFactor.Lp, grammPartFactor.Li, grammPartFactor.Lx);
    ldl_dsolve(grammMatrParted.num_rows - 1, c_12, grammPartFactor.D);
    
    
    //d_22 = c_22 - l_21*D*l_12
    for (int l_count = 0; l_count < grammMatrParted.num_rows - 1; l_count++) {
        c_12[grammMatrParted.num_rows - 1] -= c_12[l_count] * c_12[l_count] * grammPartFactor.D[l_count];
    }
    //printMatrixCPU(1, grammMatrParted.num_rows, c_12);
    coo_matrix coo_row = convDenseRowToCooRow(c_12, grammMatrParted.num_rows);
    //print_coo_matrix(coo_row);
    add_row_to_ldl(grammPartFactor, coo_row);
    
    delete_coo_matrix(coo_row);
    free(c_12);
}


void addColToCholmodFactor(ldl_matrix &grammPartFactor, csr_matrix grammMatrParted) {
    double* c_12 = new_host_darray(grammMatrParted.num_rows);
    
    int last_row_idx = grammMatrParted.num_rows - 1;
    
    //get_dense_row(grammMatrParted, last_row_idx, c_12);
    get_dense_row_from_triangular_gramm(grammMatrParted, last_row_idx, c_12);
    
    double c_22 = c_12[grammMatrParted.num_rows - 1];
    
    ldl_ltsolve_t(grammMatrParted.num_rows, c_12, grammPartFactor.Lp, grammPartFactor.Li, grammPartFactor.Lx);
    ldl_dsolve_t(grammMatrParted.num_rows, c_12, grammPartFactor.D);
    
    
    //d_22 = c_22 - l_21*D*l_12
    c_12[grammMatrParted.num_rows - 1] = c_22;
    for (int l_count = 0; l_count < grammMatrParted.num_rows - 1; l_count++) {
        c_12[grammMatrParted.num_rows - 1] -= c_12[l_count] * c_12[l_count] * grammPartFactor.D[l_count];
    }
    //printMatrixCPU(1, grammMatrParted.num_rows, c_12);
    //coo_matrix coo_row = convDenseRowToCooRow(c_12, grammMatrParted.num_rows);
    coo_matrix coo_col = convDenseColToCooCol(c_12, grammMatrParted.num_rows);
    //print_coo_matrix(coo_row);
    //add_row_to_ldl(grammPartFactor, coo_row);
    add_col_to_ldl(grammPartFactor, coo_col);
    
    delete_coo_matrix(coo_col);
    free(c_12);
}


/***
[L, D] = ldlt(a1a1)
w = sqrt(D(del_row_idx, del_row_idx)) * L(del_row_idx + 1:rows(L), del_row_idx)

L33 = L(del_row_idx + 1:rows(L), del_row_idx + 1:columns(L))
D33 = D(del_row_idx + 1:rows(D), del_row_idx + 1:columns(D))

v = L33 \ w
alpha  = 1;

m = columns(L33)
for i=1:(m)
  alpha_  = alpha + v(i)^2 / D33(i,i);
  gamma = v(i) / (alpha_ * D33(i,i));
  D33(i,i) = (alpha_ / alpha) * D33(i,i);
  alpha = alpha_;
  w( i + 1:m ) = w( i + 1:m ) - v(i) * L33(i+1:m,i);
  L33(i+1:m,i) = L33(i+1:m,i) +  gamma * w(i+1:m);
endfor
L33
D33

a2a2_vost = L33 * D33 * L33'
#a2a2
[L1, D1] = ldlt(a2a2)
a2a2_vost = L1 * D1 * L1'
 */
ldl_matrix recompute_l33_d33_for_ldl_col_del(ldl_matrix &ldl33_old, double* l32, double d22){
    
    ldl_matrix ldl33_new = new_ldl_matrix(ldl33_old.MAX_D_LP_SIZE, ldl33_old.num_nonzeros * 3);//this is low triagonal matrix!!!
    ldl33_new.num_cols = ldl33_old.num_cols;
    ldl33_new.num_rows = ldl33_old.num_rows;
    
    
    double *w = new_host_darray(ldl33_old.num_rows);
    cblas_dcopy(ldl33_old.num_rows, l32, 1, w,1);
    std::cout << "W\n";
    printMatrixCPU(1, ldl33_old.num_rows, w);
    //get_ldl_dense_column_from_l_low(ldl33_old, 0, w);
    double sqrt_d22 = sqrt(d22);
    cblas_dscal(ldl33_old.num_rows, sqrt_d22, w, 1);
    
    double* v = new_host_darray(ldl33_old.num_rows);
    cblas_dcopy(ldl33_old.num_rows , w, 1, v, 1);
    ldl_lsolve(ldl33_old.num_rows, v, ldl33_old.Lp, ldl33_old.Li, ldl33_old.Lx);
    
    std::cout << "V\n";
    printMatrixCPU(1, ldl33_old.num_rows, v);
    
    double alpha = 1.0;
    int m = ldl33_old.num_cols;
    
    double* ldl33_col = new_host_darray(ldl33_old.num_rows);
    std::cout << "Start L33 recompution with m="<< m << "\n";
    for(int i = 0; i < m ; i++){
        double alpha_ = alpha + v[i] * v[i] / ldl33_old.D[i];
        double gamma = v[i]/(alpha_ * ldl33_old.D[i]);
        ldl33_new.D[i] = (alpha_ / alpha) * ldl33_old.D[i];
        alpha = alpha_;
        
        get_ldl_dense_column_from_l_low(ldl33_old, i, ldl33_col);
        
        std::cout << "ldl33_col before\n";
        printMatrixCPU(1, ldl33_old.num_cols, ldl33_col);
        
        cblas_daxpy((m - i), -v[i], &ldl33_col[i+1], 1, &w[i + 1],1);
        cblas_daxpy((m - i), gamma, &w[i+1], 1, &ldl33_col[i + 1],1);
        std::cout << "ldl33_col after\n";
        printMatrixCPU(1, ldl33_old.num_cols, ldl33_col);
        add_last_col_to_ldl_l_low(ldl33_new, &ldl33_col[i + 1], m - i);
    }
    
    return ldl33_new;
    
}

void delete_col_from_ldl_factor(ldl_matrix &grammPartFactor, int delColIdx){
    
}

void change_sign_in_ldl(ldl_matrix &ldlm, int num_col){
    //int last_idx = ldlm.num_rows - 1;
    //int activ_row_length = last_idx - num_col;
    
    //std::cout<< "before i="<< num_col << "\n";
    //print_ldl_matrix(ldlm);
    
    for(int col_i = 0; col_i < num_col; col_i++){
        for (int el_p = ldlm.Lp[col_i]; el_p < ldlm.Lp[col_i + 1]; el_p++) {
            if(ldlm.Li[el_p] == num_col){
                ldlm.Lx[el_p] = -ldlm.Lx[el_p];
                break;
            }
        }
    }
    
    for (int j = ldlm.Lp[num_col]; j < ldlm.Lp[num_col + 1]; j++){
        if(ldlm.Li[j] > num_col){
            ldlm.Lx[j] = -ldlm.Lx[j];
        }
    }   
    //std::cout<< "after\n";
    //print_ldl_matrix(ldlm);
}


void change_sign_in_ldl_t(ldl_matrix &ldlm, int num_col){
    //int last_idx = ldlm.num_rows - 1;
    //int activ_row_length = last_idx - num_col;
    
    //std::cout<< "before i="<< num_col << "\n";
    //print_ldl_matrix(ldlm);
    
    //for(int col_i = 0; col_i < num_col; col_i++){
    for(int col_i = num_col; col_i < ldlm.num_cols; col_i++){
        for (int el_p = ldlm.Lp[col_i]; el_p < ldlm.Lp[col_i + 1]; el_p++) {
            if(ldlm.Li[el_p] == num_col){
                ldlm.Lx[el_p] = -ldlm.Lx[el_p];
                break;
            }
        }
    }
    
    for (int j = ldlm.Lp[num_col]; j < ldlm.Lp[num_col + 1]; j++){
        if(ldlm.Li[j] < num_col){
            ldlm.Lx[j] = -ldlm.Lx[j];
        }
    }   
    //std::cout<< "after\n";
    //print_ldl_matrix(ldlm);
}
