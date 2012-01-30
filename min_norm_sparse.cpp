
#include "min_norm_sparse.hpp"
#include "overloaded_cblas.hpp"
/**
 * Данная программа решает задачу min 1/2 ||x||^2, Ax <= b
 */
//template <typename int, typename double>

void getMinNormElemOutRepr(csr_matrix &inputSet,
        double * minNormVector, int inSetDim, int vectorDim, double tollerance,
        int* kvec_in, int* basisVecInx_in, int &baselen_in, int &numEqCon) {
    int maxIter = 1000; //inSetDim;
    double eps = tollerance; //0.001;

    //if (inSetDim > vectorDim + 1) {
    //    printf("слишком много вершин %i это не симплекс", inSetDim);
    //}

    //так как базис это разреженная матрица, то надо оценить количество ненулевых элементов
    //для векторов базиса которое равно размерности протранства

    int maxVectorsInBasis = inSetDim + 2; //+ 2 так как когда размер базиса равен размерности вектора + 1, то к базису добавляем ещё и mu
    if (inSetDim > (vectorDim + 2)) {
        maxVectorsInBasis = vectorDim + 2;

    };
    std::cout << "MaxVectorsInBasis = " << maxVectorsInBasis << "\n";

    //массив базисных векторов, вектора задаются своими индексами
    int *kvec = new_host_iarray(inSetDim + 1); //(int*) malloc(maxVectorsInBasis * sizeof (int)); //вектор принадлежности к базису [0 - не принадлежит,1 - принадлежит]
    int *basisVecInx = new_host_iarray(inSetDim + 1); //(int*) malloc(maxVectorsInBasis * sizeof (int)); //вектор индексов

    int estim_nonzer_in_basis = estimate_max_nonzeros(inputSet, maxVectorsInBasis);

    kvec[0] = 1; //выбираем первый вектор в качестве базисного
    int baselen = 1;

    for (int i = 1; i < inSetDim; i++) {
        kvec[i] = 0;
    }

    basisVecInx[0] = 0;


    if (baselen_in != 0 && kvec_in != 0 && basisVecInx_in != 0) {
        //тут бы всё и скопировать
    }

    //double *z = (double *) malloc(vectorDim * sizeof (double)); //allocate memory for el of min norm
    double *z = new_host_darray(vectorDim); //allocate memory for el of min norm

    //пусть перый элемент мин нормы будет равен первому вектору симплекса
    //не то чтобы я совсем дурак и умножаю на ноль, но для наглядности не повредит (потом и стереть можно)
    //memcpy(z, &inputSet[0 * vectorDim], vectorDim * sizeof (double));
    get_dense_column(inputSet, (int) 0, z);

    int iterCounter = 0;
    //std::cout << "Max count of elements in grammMatr: " << maxVectorsInBasis * maxVectorsInBasis << "\n";
    //можно сразу выделить память под матрицу Грамма и ей обратную
    //double *grammMatr = new_host_darray (maxVectorsInBasis * maxVectorsInBasis); //(double*) malloc(maxVectorsInBasis * maxVectorsInBasis * sizeof (double)); //
    //double *invGrammMatr = new_host_darray (maxVectorsInBasis * maxVectorsInBasis); //(double*) malloc(maxVectorsInBasis * maxVectorsInBasis * sizeof (double)); //new_host_darray (maxVectorsInBasis * maxVectorsInBasis);
    double *grammMatr = new_host_darray (400 * 400);
    double *invGrammMatr = new_host_darray (400 * 400);
    //for (int i = 0; i < maxVectorsInBasis * maxVectorsInBasis; i++) {
    for (int i = 0; i < 400 * 400; i++) {
        grammMatr[i] = 0.0;
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
    coo_matrix coo_col_to_add;
    for (int eqVecCount = 0; eqVecCount < numEqCon; eqVecCount++) {
        coo_col_to_add = get_coo_column(inputSet, (int) eqVecCount);
        //print_coo_matrix(coo_col_to_add);
        add_col_to_csr_mtx(basis, coo_col_to_add);
        transpose_coo_mv(coo_col_to_add);
        add_row_to_csr(basis_t, coo_col_to_add);
        delete_host_matrix(coo_col_to_add);
        kvec[eqVecCount] = 1;
        basisVecInx[eqVecCount] = eqVecCount;
    }
    baselen = numEqCon;
    ///////Now we mast calculate z for this basis
    mm_csr_serial_host(basis_t, basis, grammMatr);
    //std::cout << "GrammaMatr eq constr";
    //printMatrixCPU(baselen, baselen, grammMatr);

    inverseMatrixCPU(grammMatr, invGrammMatr, baselen);
    //std::cout << "InvGrammaMatr eq constr";
    //printMatrixCPU(baselen, baselen, invGrammMatr);
    sumMtrxByStr(invGrammMatr, mu, baselen, baselen);
    //вычисление e' (X' * X)^-1 * e, что есть сумма всех элементов матрицы
    double elSumm = 0;
    sumVecElmts(mu, &elSumm, baselen);
    std::cout << "muSumm=" << elSumm << "\n";
    cblas_dscal(baselen, 1.0 / elSumm, mu, 1);
    memcpy(mu_old, mu, baselen * sizeof (double));
    spmv_csr_serial_host(basis, mu, z);
    //evalMuVectorCPU(baselen, vectorDim, basisInc, basis, basis_t, grammMatr, invGrammMatr, mu, isFulBasis, delBasElIndx);
    ///////////////////////////////////////////////////
    while (iterCounter < maxIter) {
        std::cout << "Iteration "<< iterCounter<< "\n";
        //print_csr_matrix(basis);
        //printMatrixCPU(1, vectorDim, z);
        iterCounter++; //увеличиваем счётчик итераций
        double minVcos;
        int minVecId;
        //находим следующего кандидата для добавления в базис руководствуясь критерием
        //наибольшей дистанции от множества
        //int hTimer = 0;
        //CUT_SAFE_CALL(cutCreateTimer(&hTimer)); //init timer
        //CUDA_SAFE_CALL(cudaThreadSynchronize());
        //CUT_SAFE_CALL(cutResetTimer(hTimer));
        //CUT_SAFE_CALL(cutStartTimer(hTimer));
        getMinVCosCPU(z, inputSet, inSetDim, vectorDim, &minVcos, &minVecId, eps, numEqCon);
        //CUT_SAFE_CALL(cutStopTimer(hTimer));
        //codePartTimer += cutGetTimerValue(hTimer);
        //double zsqr = cblas_ddot(vectorDim, z, 1, z, 1);
        //printf("Epsilon threshold %e", eps * eps * zsqr);
        if (minVcos > -eps * eps ) {
            printf("End of algoritm \n");
            printf("Min vCos %e \n", minVcos);
            printf("Iterations num %i \n", iterCounter);
            //print_csr_matrix(basis);
            //coo_col_to_add = get_coo_column(inputSet, minVecId);
            //print_coo_matrix(coo_col_to_add);
            //printMatrixCPU(vectorDim, 1, z);
            memcpy(minNormVector, z, vectorDim * sizeof (double));
            memcpy(kvec_in, kvec, (inSetDim + 1) * sizeof (int));
            memcpy(basisVecInx_in, basisVecInx, (inSetDim + 1) * sizeof (int));
            baselen_in = baselen;
            break;
        }
        if (kvec[minVecId] == 1) {
            printf("Vector %i aready in basis. Exit\n", minVecId);
            printf("Min vCos %e \n", minVcos);
            printf("Iterations num %i \n", iterCounter);

            memcpy(minNormVector, z, vectorDim * sizeof (double));
            memcpy(kvec_in, kvec, (inSetDim + 1) * sizeof (int));
            memcpy(basisVecInx_in, basisVecInx, (inSetDim + 1) * sizeof (int));
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
        coo_col_to_add = get_coo_column(inputSet, ix);
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

        findOptZInBasisCPU(inputSet, baselen, vectorDim, basisInc, basis, basis_t, basisVecInx,
                kvec, grammMatr, invGrammMatr, mu, mu_old, z, eps, numEqCon);


    }
    //printf("Cpu time of procedure: %f msecs.\n", codePartTimer);
    //for(int j=0;j<baselen;j++){printf("%i ", basisVecInx[j]);}printf("\n");
    //printMatrix(vectorDim, baselen,  basis);
    //printMatrix(1,vectorDim,  z);
    double dist = cblas_ddot(vectorDim, z, 1, z, 1);
    //printf("Distance %f \n", sqrt(dist));
    //printf("min_nor_el\n");
    free(mu_old);

    free(invGrammMatr);
    free(grammMatr);

    free(basisVecInx);
    free(mu);
    free(kvec);
    delete_host_matrix(basis);
    delete_host_matrix(basis_t);
    free(z);
}



//template <typename int, typename double>

void findOptZInBasisCPU(csr_matrix& inSet, int &baselen, int vectorDim, int basisInc,
        csr_matrix &basis, csr_matrix &basis_t, int *basisVecInx, int *kvec, double *grammMatr, double *invGrammMatr,
        double *mu, double *mu_old, double *z, double eps, int numEqCon) {
    int baselenFixed = baselen;
    //print_csr_matrix(basis);
    //print_csr_matrix(basis_t);
    bool isFulBasis = false;
    int delBasElIndx = 0; //индекс элемента который мы удаляем из базиса
    for (int i = 0; i < baselenFixed; i++) {
        //решаем задачу проекции нуля на вновь образованное подпространство
        // min||z^2|| = ||Z_s^2||
        evalMuVectorCPU(baselen, vectorDim, basisInc, basis, basis_t, grammMatr, invGrammMatr, mu, isFulBasis, delBasElIndx, eps);
        //std::cout << "InvGrammMatrix " << "\n";
        //printMatrixCPU(baselen, baselen, invGrammMatr);
        std::cout << "mu vector before " << baselen << "\n";
        printMatrixCPU((int) 1, baselen, mu);
        //for(int j=0;j<baselen;j++){printf("%i ", basisVecInx[j]);}
        //printf("\n");
        int minMuIdx = 0;
        ////////////////////Коррекция базиса с начальными условиями в виде равенств///////
        double minMuFromEqConstrains = getMinVectorElCPU(mu, numEqCon, minMuIdx, 0);
        //Если mu отрецательно в переделах условий, то надо корректировать умножением базисного вектора на -1
        while(minMuFromEqConstrains < -eps * eps ) {
            //std::cout << "Negative mu in basis equality constrains: " << minMu << " with index: " << minMuIdx << "\n";
            for(int j = 0; j < numEqCon; j++){
                if(mu[j] < -eps * eps){
                    scaleColInCsrMtx_AndAdd2Shift(basis, -1.0, j, -1.0);
                    scaleRowInCsrMtx_AndAdd2Shift(basis_t, -1.0, j, -1.0);

                    //print_csr_matrix(basis);

                    scaleColInCsrMtx_AndAdd2Shift(inSet, -1.0, basisVecInx[j], -1.0);
                }
            }
            
            //getMuViaSystemSolve(basis, basis_t, mu, eps);
            evalMuVectorCPU(baselen, vectorDim, basisInc, basis, basis_t, grammMatr, invGrammMatr, mu, isFulBasis, delBasElIndx, eps);
            //free(mu_n);
            std::cout << "Mu from ideal===========================\n";
            printMatrixCPU(1, baselen, mu);
            //////////////////END TEST
            minMuFromEqConstrains = getMinVectorElCPU(mu, numEqCon, minMuIdx, 0);
        }
        /////////////////////////////////////////////////////////
        //Находим нормальное наименьшее MU после исправления нашего базиса
        //Тут необходимо проверить, принадлежит ли получившийся вектор множеству X
        //Похоже это условие min(mu) > -eps        
        double minMu = getMinVectorElCPU(mu, baselen, minMuIdx, 0);    

        //std:cout <<"minMu - mu_old[minMuIdx] = " << mu_old[minMuIdx] - minMu << "\n";
        if (minMu > -eps * eps) {
            //if (getMinVectorElCPU(mu, baselen) > 0) {
            //Вычисляем координаты получившегося вектора Z_s
            //cblas_sgemv(CblasColMajor, CblasNoTrans, vectorDim, baselen, 1.0, basis, vectorDim, mu, 1, 0, z, 1);

            cblas_dscal(basis.num_rows, (double) 0.0, z, 1);
            //std::cout << "baselen "<< baselen << " basis.num_rows = "<< basis.num_rows << "basis.num_cols "<< basis.num_cols << "\n";
            spmv_csr_serial_host(basis, mu, z);
            //std::cout << "Basis \n";
            //print_csr_matrix(basis);
            //std::cout << "Z\n";
            //printMatrixCPU((int)1, vectorDim, z);
            break;
        }
        std::cout << "Min mu = " << minMu << " \n";
        //Если Z* находится за перделами X то надо решить вспомогательную задачу
        //спроецируем Z* на X
        //Мы подтягиваем Z(k+1) к Z(k) и получаем новый ветор Z(k+1)
        //то есть u(t) = t * mu + (t-1) mu_old ищим максимум по t так что u принадлежит X

        double t = attractDotCoeffCPU(mu_old, mu, &delBasElIndx, baselen, numEqCon);
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
        //cblas_copy(((baselen - 1) - delBasElIndx), &mu_old[(delBasElIndx + 1)], 1, &mu_old[delBasElIndx], 1);

        kvec[basisVecInx[delBasElIndx]] = 0;
        printf("del %i\n", basisVecInx[delBasElIndx]);
        //double* basisVecInxTmp = (double*) malloc(((baselen - 1) - delBasElIndx) * sizeof (int));
        memmove(&basisVecInx[delBasElIndx], &basisVecInx[delBasElIndx + 1], ((baselen - 1) - delBasElIndx) * sizeof (int));
        //memcpy(basisVecInxTmp, &basisVecInx[delBasElIndx + 1], ((baselen - 1) - delBasElIndx) * sizeof (int));
        //memcpy(&basisVecInx[delBasElIndx], basisVecInxTmp, ((baselen - 1) - delBasElIndx) * sizeof (int));
        //free(basisVecInxTmp);
        baselen--;
        basisInc = 0;
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
        double *grammMatr, double *invGrammMatr, double *mu, bool &isFullBasis, int vecToDelId, double eps) {
    if (baselen <= vectorDim) {
        isFullBasis = false;
        if(baselen < 1){
        if (basisInc && baselen > 2) {
            //cublasScopy((baselen - 1) * (baselen - 1), invGrammMatr, 1, grammMatr, 1);
            //printf("invMatrix gramma");
            //printMatrixCPU(baselen - 1 , baselen - 1 , invGrammMatr);
            //std::cout << "invGrammAdd_CPU " << baselen << " \n";
            cblas_dcopy((baselen - 1) * (baselen - 1), invGrammMatr, 1, grammMatr, 1);

            //print_csr_matrix(basis);
            //std::cout << "old inv gramma\n";
            //printMatrixCPU(baselen -1 , baselen - 1, invGrammMatr);

            invGrammAdd_CPU(baselen - 1, basis.num_rows, basis, basis_t, grammMatr, invGrammMatr);

            //std::cout << "gramma\n";
            //mm_csr_serial_host(basis_t, basis, grammMatr);
            //printMatrixCPU(baselen, baselen, grammMatr);
            //std::cout << "inverse Matrix gramma reevaluated \n";
            //printMatrixCPU(baselen, baselen, invGrammMatr);

            //double* tstMtrx = new_host_darray(baselen * baselen);
            //cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, baselen, baselen, baselen, 1.0, invGrammMatr, baselen, grammMatr, baselen, 0.0, tstMtrx, baselen);
            //printMatrixCPU(baselen, baselen, tstMtrx);


        } else if (basisInc) {
            std::cout << " basis.rows " << basis.num_rows << " basis.col " << basis.num_cols << " basis.nonz " << basis.num_nonzeros << "\n";
            std::cout << " basis_t.rows " << basis_t.num_rows << " basis_t.col " << basis_t.num_cols << " basis_t.nonz " << basis_t.num_nonzeros << "\n";
            //printMatrixCPU(basis_t.num_rows, basis.num_cols, grammMatr);
            mm_csr_serial_host(basis_t, basis, grammMatr);

            //вычислим обратную матрицу к матрице Грамма
            //std::cout << "inverseMatrixCPU " << baselen << " \n";

            inverseMatrixCPU(grammMatr, invGrammMatr, baselen);
            //inverseMatrixCPURaw(grammMatr, invGrammMatr, baselen, baselen);
            //std::cout << "baslen " << baselen << " \n";
            //printMatrixCPU(baselen, baselen, grammMatr);
            //printMatrixCPU(baselen, baselen, invGrammMatr);
        } else {
            //Решаем через множители Лагранжа
            //mu = -(X' * X)^-1 * e / e' (X' * X)^-1 * e
            cblas_dcopy((baselen + 1) * (baselen + 1), invGrammMatr, 1, grammMatr, 1);
            invGrammDel_CPU(baselen + 1, baselen + 1, grammMatr, invGrammMatr, vecToDelId);
            /*
            mm_csr_serial_host(basis_t, basis, grammMatr);
            
            //вычислим обратную матрицу к матрице Грамма
            //std::cout << "inverseMatrixCPU " << baselen << " \n";

            inverseMatrixCPU(grammMatr ,invGrammMatr, baselen);
            //inverseMatrixCPURaw(grammMatr, invGrammMatr, baselen, baselen);
            //std::cout << "baslen " << baselen << " \n";
            printMatrixCPU(baselen, baselen, grammMatr);
            printMatrixCPU(baselen, baselen, invGrammMatr);
             */
        }
        //printMatrixForOctave(baselen, baselen, invGrammMatr);
        //вычисление вектора (X'*X)^1 * e, что есть суммирование матрицы по строкам
        sumMtrxByStr(invGrammMatr, mu, baselen, baselen);
        //вычисление e' (X' * X)^-1 * e, что есть сумма всех элементов матрицы
        double elSumm = 0;
        sumVecElmts(mu, &elSumm, baselen);
        //std::cout << "muSumm="<< elSumm << "\n";
        /*if(elSumm == NAN || elSumm == -NAN || elSumm == INFINITY || elSumm == -INFINITY){
            std::cout << "NAN detected in MU muSumm="<< elSumm << "\n";
            inverseMatrixCPURaw(grammMatr, invGrammMatr, baselen, baselen);
            sumMtrxByStr(invGrammMatr, mu, baselen, baselen);
            //вычисление e' (X' * X)^-1 * e, что есть сумма всех элементов матрицы
            double elSumm = 0;
            sumVecElmts(mu, &elSumm, baselen);
        }*/
        //std::cout << "elSumm " << elSumm << "\n";
        //вычисление барицентрических координат
        cblas_dscal(baselen, 1.0 / elSumm, mu, 1);
        //multVectorToValue(mu, 1 / elSumm, baselen);
        //printf("\nMu -- new vector baricentric coordinates\n");
        //printMatrix(baselen, 1, mu);
        }else{
            getMuViaSystemSolve(basis, basis_t, mu, eps);
        }
        
    } else {
        //решаем систему X * mu = 0; e' * mu = 1

        std::cout << "Basis dimension is vectorDim + 1 = " << baselen << "\n";
        //print_csr_matrix(basis);
        isFullBasis = true;
        //делаем единичный столбец в формате coo
        coo_matrix coo_col;
        coo_col.num_nonzeros = baselen;
        coo_col.num_cols = 1;
        coo_col.num_rows = coo_col.num_nonzeros;
        coo_col.I = new_host_iarray(coo_col.num_rows);
        coo_col.J = new_host_iarray(coo_col.num_rows);
        coo_col.V = new_host_darray(coo_col.num_rows);

        for (int i = 0; i < baselen; i++) {
            coo_col.I[i] = i;
            coo_col.J[i] = 0;
            coo_col.V[i] = (double) 1.0;
        }

        /*
        basis_t.num_rows--;
        convCsrToDense(basis_t, grammMatr);
        basis_t.num_rows++;

        printMatrixCPU(baselen - 1, baselen - 1, grammMatr);
        printMatrixCPU(baselen -1, baselen -1 , invGrammMatr);
        
        double* inv_old_basis = new_host_darray((baselen -1) * (baselen - 1));
        cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, baselen - 1,
                    baselen - 1 , baselen - 1, (double)1.0, invGrammMatr, baselen - 1, grammMatr,
                    baselen - 1, (double)0.0, inv_old_basis, baselen - 1);

        printMatrixCPU(baselen -1 , baselen -1 , inv_old_basis);
         */

        add_col_to_csr_mtx(basis_t, coo_col);
        transpose_coo_mv(coo_col);
        std::cout << "\nBefore basis.num_rows = " << basis.num_rows << "\n";
        add_row_to_csr(basis, coo_col);
        std::cout << "After basis.num_rows = " << basis.num_rows << "\n";
        delete_host_matrix(coo_col);

        double* b = new_host_darray(baselen);
        for(int b_count = 0; b_count < baselen - 1; b_count++){b[b_count] = 0.0;}
        b[baselen - 1] = 1.0;
        biSoprGradient(basis, basis_t, b, mu, eps, mu);
        delete_host_array(b);

        //printMatrixCPU(baselen,1, mu);
        //invGrammAddFulBasis_CPU(baselen - 1, basis.num_rows, basis, basis_t, inv_old_basis, invGrammMatr);
        //free<double>(inv_old_basis);


        //printMatrixCPU(baselen, baselen, invGrammMatr);
        //printMatrixCPU(baselen, baselen, grammMatr);

        //cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, baselen,
        //            baselen, baselen, 1.0, invGrammMatr, baselen, grammMatr,
        //            baselen, 0.0, grammMatr, baselen);

        //printMatrixCPU(baselen, baselen, grammMatr);

        //memmove(mu, &grammMatr[(baselen - 1) * basis.num_rows],basis.num_rows * sizeof(double));
        //cblas_sgemv(CblasColMajor, CblasNoTrans, baselen, baselen, 1.0, invGrammMatr, vectorDim, mu, 1, 0, z, 1);
        /*
        convCsrToDense(basis, grammMatr);
        int* ipiv = new_host_iarray(baselen);
        cblas_dscal(baselen - 1, (double) 0.0, mu, 1);
        mu[baselen - 1] = 1.0;
        int status = clapack_dgesv(CblasColMajor, baselen, 1, grammMatr, baselen, ipiv, mu, baselen);

        std::cout << "Status of solve = " << status << "\n";
        */
        
        
        //printf("mu of Full basis\n");
        //printMatrixCPU((int)1, baselen, mu);

        del_col_from_csr_mtx(basis_t, baselen - 1);
        del_row_from_csr_mtx(basis, baselen - 1);

        //free(ipiv);
    }
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
    spmv_csr_t_serial_host(basis, addVect, B);

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
    printMatrixCPU(rowsInGramOld + 1, 1, B);

    get_dense_column(basis_t, rowsInGramOld, C);
    printMatrixCPU(rowsInGramOld + 1, 1, C);

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

double attractDotCoeffCPU(double *mu_old, double *mu, int *delBasElIndx, int baselen, int eqConNum) {
    double *mu_razn = (double*) malloc(baselen * sizeof (double));
    double *mu_div = (double*) malloc(baselen * sizeof (double));
    //mu_old - mu
    //for (int i = 0; i < vectorDim; i++) {
    //    mu_razn[i] = mu_old[i] - mu[i];
    //printf("MU_Razn %f\n", mu_razn[i]);
    //}
    //mu_old / mu_razn,
    //for (int i = 0; i < vectorDim; i++) {
    //    mu_div[i] = mu_old[i] / mu_razn[i];
    //}
    //get min pozitive value
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
                *delBasElIndx = i;
            }
        }

        //if (mu_div[i] >= 0.0 && mu_div[i] < minPozVal) {//Err with cof
        //    minPozVal = mu_div[i];
        //    *delBasElIndx = i;
        //}else{printf("mu_old[%i]=%f mu_razn=%f, mu=%f mu_div=%f\n", i,mu_old[i], mu_razn[i], mu[i],mu_div[i]);}
    }
    //printf("Parameter of attraction t=%e and min el inx = %d\n", minPozVal, *delBasElIndx);
    std::cout << "Parameter of attraction t= " << minPozVal << "and min el inx = " << *delBasElIndx << "\n";
    if (minPozVal == 100000.0) {
        minPozVal = 0.0;
        //printMatrixCPU(10, 1, mu);
    }
    double t = minPozVal;
    double t2 = 1 - t;
    //Case without eq conditions
    //for (int i = 0; i < baselen; i++) {
    for (int i = 0; i < baselen; i++) {
        //mu_new[i] = t * mu[i] + t2 * mu_old[i];
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

void getMinVCosCPU(double* z, csr_matrix &inSet, int inSetDim, int vectorDim, double* minVcos, int *minVecId, double epsilon, int numEqCon) {
    double* vcos = new_host_darray(inSetDim); //(double *) malloc(inSetDim * sizeof (double));
    for (int i = 0; i < inSetDim; i++) {
        vcos[i] = 0;
    }
    double zz;
    //evalScalarCPU(1, vectorDim, z, z, &zz);
    //cblas_sdot(vectorDim, z, 1, z, 1);
    zz = cblas_ddot(vectorDim, z, 1, z, 1);

    std::cout << "ZZ " << zz << "\n";

    //cblas_sgemv(CblasColMajor, CblasTrans, vectorDim, inSetDim, 1.0f, inSet, vectorDim, z, 1, 0, vcos, 1);
    spmv_csr_t_serial_host(inSet, z, vcos);
    //std::cout << "Vcos>>>>>>>>>>>";
    //printMatrixCPU(inSetDim, (int)1,vcos);
    ///////////////////for vase wisout eq constrains
    //*minVcos = vcos[0] - zz;
    *minVcos = vcos[numEqCon - 1] - zz;
    //printMatrixCPU(inSetDim, (int)1,vcos);
    ///////////////////for vase wisout eq constrains
    //*minVecId = 0;
    *minVecId = numEqCon;
    //printf("init vcos:%f \n ", *minVcos);
    //int oldminId = 0;
    //double oldminVal = 0;
    ////////////////jast for one
    //for (int elId = 0; elId < inSetDim; elId++) {
    for (int elId = numEqCon; elId < inSetDim; elId++) {
        double curentVcos = vcos[elId] - zz;
        //printf("vcos[%i] = %e\n", elId, curentVcos);
        if (curentVcos < *minVcos) {
            //if(abs(curentVcos - *minVcos) > epsilon){
            //oldminId = *minVecId;
            //oldminVal = *minVcos;
            *minVcos = curentVcos;
            *minVecId = elId;
        }
    }

    //printf(" %i  value %e \n ", , *);
    std::cout << "Min vcos id:" << *minVecId << " value" << *minVcos << "\n";
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

        printMatrixCPU(mtrxRows, mtrxRows, invMatrix);
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


void biSoprGradient(const csr_matrix& basis, const csr_matrix& basis_t, double *b, double *x,double eps, double *x0){
    double *x_i = new_host_darray(basis.num_cols);
    
    if(x0 == NULL){
        for(int x0_i = 0; x0_i < basis.num_cols; x0_i++){x_i[x0_i] = 0.0;}}
    else{
        cblas_dcopy(basis.num_cols, x0, 1, x_i, 1);
    }
    double *r_i = new_host_darray(basis.num_cols);
    
    double *rtild_i = new_host_darray(basis.num_cols);
    double *p_i = new_host_darray(basis.num_cols);
    double *ptild_i = new_host_darray(basis.num_cols);
    //A*x
    double *gramm_to_x = new_host_darray(basis.num_cols);
    mulGrammToX(basis_t, basis, x_i, gramm_to_x );

     

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
    while(sqrt(cblas_ddot(basis.num_cols, r_i, 1, r_i, 1)) > eps * eps * eps){
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
        mulGrammToX(basis_t, basis, p_i, gramm_to_x );
        
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
        mulGrammToX(basis_t, basis, ptild_i, gramm_to_x );
        cblas_daxpy(basis.num_cols, -1.0 * alpha, gramm_to_x, 1, rtild_i, 1);
        //betta = (r_(i+1), rtild_(i+1))/(r_i, rtild_i)
        betta = cblas_ddot(basis.num_cols, r_i, 1, rtild_i, 1) / ri_to_rtildi;
        if(betta == 0 ){//sqrt(betta * betta) < eps*eps ){
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


void getMuViaSystemSolve(const csr_matrix& basis, const csr_matrix& basis_t, double *mu, double eps){
    double *x_i = new_host_darray(basis.num_cols);//column of inverse matrix of gramm
    double *e_i = new_host_darray(basis.num_cols);//column of Unar matrix to pass to system solver as right elemetn
    double *x_row_summ = new_host_darray(basis.num_cols);//vector to store reow summs for inver gramm matrix
    for(int i = 0; i < basis.num_cols; i++){
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



