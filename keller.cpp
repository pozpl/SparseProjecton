#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <ctype.h>
#include <unistd.h>
#include <getopt.h>

#include <math.h>


#include <fstream>
#include <iostream>
#include <istream>
#include <vector>
#include <algorithm>
#include <string>
using namespace std;

extern "C" {
#include <atlas/atlas_enum.h>
#include <clapack.h>
}

//#define GENERATE
#define FROM_FILE

/*
#include <cublas.h>
#include <cutil.h>
#include <cuda.h>
 */
//SpMV includes
//#include "spmv_nvidia/sparse_io.h"
//#include "spmv_nvidia/cmdline.h"
//#include "spmv_nvidia/tests.h"
//#include "spmv_nvidia/gallery.h"
//#include "overloaded_cblas.hpp"
#include "sparse_operations_ext_cpu.hpp"
//#include "min_norm_sparse.hpp"
//#include "fixed_conus_projection.hpp"
#include "conus_projection_sparse.hpp"
#include "sparse_types.hpp"
#include "simplex_projection.hpp"

//template <typename int, typename double>
void prepare_matrix();
//template <typename int, typename double>
double* readRightSide(const char *fname, int &vectorDim);
//template <typename int, typename double>
coo_matrix readCooMatrix(const char *fnameMatrix, const char *fnameRHS, bool fff);
//template <typename int, typename double>
coo_matrix printLPFile(csr_matrix taskMatrix);
//template <typename int, typename double>
coo_matrix parseLPFile(char *file_name);
//template <typename int, typename double>
coo_matrix readAuProblemMatrices(const char *fnameSigma, const char *fnameG, const char *fnameC, const char *fnameP0);
double* calculate_a(int z_dim, double *z, const char *fnameSigma, const char *fnameP0);
double* calculate_a_dynamic(int z_dim, double *z, int sigmaRows, double *Sigma, int G_rows, double *P0, coo_matrix &G_matrix, double* C);
/*
 *Return matrix with additional information about how much rows in matrix presented
 * by equality constrains
 */
//coo_matrix readAuProblemMtxPosCon(const char *fnameSigma, const char *fnameG, const char *fnameC, const char *fnameP0, int &numberOfEqConstr);
coo_matrix readAuProblemMtxPosCon(const char *fnameSigma, const char *fnameG, const char *fnameC, const char *fnameP0, int &numberOfEqConstr, coo_matrix &G_matrix, double* C);
coo_matrix generateAuProblemMtxPosCon(int Sigma_rows, int G_rows, int &numberOfEqConstr, double* Sigma, double* P0, coo_matrix &G_matrix, double* C);
double rand2(double low, double high);

/*int parseArgs(int argc, char** argv) {
    int c;

    //float tollerance = 0.001;
    int vector_dim; //VECTOR_DIM;
    int nvertex; //SIMPLEX_VERTEX_COUNT;


    while (1) {
        //int this_option_optind = optind ? optind : 1;
        int option_index = 0;
        static struct option long_options[] = {
            {"nvertex", 1, 0, 'n'},
            {"vector_dim", 1, 0, 'd'},
            {0, 0, 0, 0}
        };

        c = getopt_long(argc, argv, "abc:d:012",
                long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {

            case 'n':
                //FileName = (char *) malloc(sizeof (char) * (strlen(optarg) + 1));
                //FileName[strlen(optarg)] = '\0';
                //strcpy(FileName, optarg);
                nvertex = atoi(optarg);
                break;

            case 'd':
                vector_dim = atoi(optarg);

                break;
            default:
                printf("?? getopt returned character code 0%o ??\n", c);
        }
    }

    if (optind < argc) {
        printf("non-option ARGV-elements: ");
        while (optind < argc)
            printf("%s ", argv[optind++]);
        printf("\n");
    }
    printf("Vertex count %i, vector dim %i \n", nvertex, vector_dim);


    return 0;
}
 */

//template <typename int, typename double>

void prepare_matrix() {



    //double *rightSide;
    //int rsVecDim;
    //rightSide = readRightSide<int,double>("/home/pozpl/tmp/rhs_test.dat", rsVecDim);
    //char* LPFfile = "/home/pozpl/tmp/keller6.lp";
    //parseLPFile(LPFfile);

    //coo_matrix<int, double> inA_coo = readCooMatrix<int, double > ("/home/pozpl/tmp/lpmtrx_2x4.mtx", "/home/pozpl/tmp/rhs_2x4.dat", true);
    //coo_matrix<int, double> inA_coo = readCooMatrix<int, double > ("/home/pozpl/tmp/simple_2x3.mtx", "/home/pozpl/tmp/rhs_2x3.dat", true);
    //coo_matrix<int, double> inA_coo = readCooMatrix<int, double > ("/home/pozpl/tmp/keller6/matrix.dat", "/home/pozpl/tmp/keller6/rhs.dat", true);
    //coo_matrix<int, double> inA_coo = readCooMatrix<int, double > ("/home/pozpl/workspace/OCTAVE/AS", "/home/pozpl/workspace/OCTAVE/CS", false);

    //coo_matrix inA_coo = readAuProblemMatrices("/home/pozpl/tmp/AuPrSmall/Sigma.mtx",
    //        "/home/pozpl/tmp/AuPrSmall/G.mtx", "/home/pozpl/tmp/AuPrSmall/C.mtx", "/home/pozpl/tmp/AuPrSmall/P0.mtx");
    int numberOfEqConstr = 0;
#ifdef FROM_FILE
    coo_matrix G_matrix;
    double* C = new_host_darray(3000);
#endif    

    //coo_matrix inA_coo = readAuProblemMtxPosCon("/home/pozpl/tmp/AuPrSmall/Sigma.mtx",
    //        "/home/pozpl/tmp/AuPrSmall/G.mtx", "/home/pozpl/tmp/AuPrSmall/C.mtx", "/home/pozpl/tmp/AuPrSmall/P0.mtx", numberOfEqConstr);
#ifdef FROM_FILE
    //coo_matrix inA_coo = readAuProblemMtxPosCon("/home/pozpl/tmp/AuProblem/Sigma.mtx",
    //       "/home/pozpl/tmp/AuProblem/G.mtx", "/home/pozpl/tmp/AuProblem/C.mtx", "/home/pozpl/tmp/AuProblem/P0.mtx", numberOfEqConstr, G_matrix, C);
    coo_matrix inA_coo = readAuProblemMtxPosCon("/home/pozpl/tmp/AuPrSmall/Sigma.mtx",
            "/home/pozpl/tmp/AuPrSmall/G.mtx", "/home/pozpl/tmp/AuPrSmall/C.mtx", "/home/pozpl/tmp/AuPrSmall/P0.mtx", numberOfEqConstr, G_matrix, C);
#endif


    //coo_matrix inA_coo = readAuProblemMatrices("/home/pozpl/tmp/AuProblem/Sigma.mtx",
    //        "/home/pozpl/tmp/AuProblem/G.mtx", "/home/pozpl/tmp/AuProblem/C.mtx", "/home/pozpl/tmp/AuProblem/P0.mtx");
    //print_coo_matrix(inA_coo);
    //coo_matrix<int, double> inA_coo = readCooMatrix<int, double>("/home/pozpl/tmp/keller7/matrix.dat", "/home/pozpl/tmp/keller7/rhs.dat", true);
    //coo_matrix<int, double> inA_coo = readCooMatrix<int, double>("/home/pozpl/tmp/TestMatrices/k6100", "/home/pozpl/tmp/TestMatrices/k6100_rhs", true);

#ifdef GENERATE   
    int Sigma_rows = 1000;
    int G_rows = 600;
    double *Sigma;
    double *P0;
    Sigma = new_host_darray(Sigma_rows);
    P0 = new_host_darray(Sigma_rows);
    coo_matrix G_matrix;
    double* C = new_host_darray(G_rows);

    coo_matrix inA_coo = generateAuProblemMtxPosCon(Sigma_rows, G_rows, numberOfEqConstr, Sigma, P0, G_matrix, C);
#endif         
    csr_matrix inA = coo2csr(inA_coo); //coo_to_csr(inA_coo);
    csc_matrix in_a_csc;
    in_a_csc.num_cols = inA.num_rows;
    in_a_csc.num_rows = inA.num_cols;
    in_a_csc.num_nonzeros = inA.num_nonzeros;
    in_a_csc.Cp = inA.Ap;
    in_a_csc.Ri = inA.Aj;
    in_a_csc.Ex = inA.Ax;

    //print_csr_matrix(inA);
    std::cout << "inA_coo.num_cols = " << inA_coo.num_cols << " inA_coo.num_rows = " << inA_coo.num_rows << " \n";

    delete_host_matrix(inA_coo);

    //std::cout<< "inA.num_cols = " << inA.num_cols << " inA.num_rows = " << inA.num_rows << " \n";
    //print_csr_matrix(inA);
    //printLPFile(inA);

    //csr_matrix inSet = csr_transpose_fix(inA);
    //print_csr_matrix(inSet);
    double * minNormVector = new_host_darray(in_a_csc.num_rows);
    for (int i = 0; i < in_a_csc.num_rows; i++) {
        minNormVector[i] = 0.0;
    }

    std::cout << "inSetRows " << in_a_csc.num_rows << " columns " << in_a_csc.num_cols << " nonzerrows " << in_a_csc.num_nonzeros << "\n";

    int inSetPow = in_a_csc.num_cols;
    int vectorDim = in_a_csc.num_rows;

    std::cout << "Number of equation constrains " << numberOfEqConstr << "\n";
    std::cout << "inSetDimension " << inSetPow << " vectorDim " << vectorDim << "\n";
    //print_csr_matrix(inSet);

    //getProjectionOnConus(inSet, minNormVector, inSetPow, vectorDim, (double) 0.001);
    //proj_to_lp_cpu(inSet, minNormVector, inSetPow, vectorDim, (double) 0.0001);
    int *kvec = new_host_iarray(2 * in_a_csc.num_cols + 1);
    int *basisVecInx = new_host_iarray(2 * in_a_csc.num_cols + 1);
    for (int i = 0; i < 2 * in_a_csc.num_cols + 1; i++) {
        kvec[i] = 0.0;
        basisVecInx[i] = 0.0;
    }
    int baselen = 0;

    //std::cout<< "INPUT SET: \n";
    //print_csr_matrix(inSet);
    //cblas_dscal(in_a_csc.num_nonzeros,  0.001, in_a_csc.Ex, 1);
    //projOnFixedSimplex(inSet, minNormVector, kvec, basisVecInx, baselen, 0.0001, numberOfEqConstr);
    getMinNormElemOutRepr(in_a_csc, minNormVector, 0.0001, kvec, basisVecInx, baselen, numberOfEqConstr);

    //cblas_dscal(in_a_csc.num_nonzeros,1000, in_a_csc.Ex, 1);
    //projOnFixedSimplex(inSet);
    //std:cout << "Min norm vector\n";
    //printMatrixCPU(in_a_csc.num_rows, 1, minNormVector);
#ifdef FROM_FILE
    //double* a_rel = calculate_a(G_matrix.num_cols, minNormVector, "/home/pozpl/tmp/AuProblem/Sigma.mtx", "/home/pozpl/tmp/AuProblem/P0.mtx");
    double* a_rel = calculate_a(in_a_csc.num_cols, minNormVector, "/home/pozpl/tmp/AuPrSmall/Sigma.mtx", "/home/pozpl/tmp/AuPrSmall/P0.mtx");
    double a_norm = cblas_dnrm2(G_matrix.num_cols, a_rel, 1);
    std::cout << "norm2(a_real) = " << a_norm << std::endl;
#endif


    double* C_new = new_host_darray(G_matrix.num_rows);
#ifdef FROM_FILE
    for (int i = 0; i < G_matrix.num_nonzeros; i++) {
        C_new[G_matrix.I[i]] += G_matrix.V[i] * a_rel[G_matrix.J[i]];
    }
#endif
    //printMatrixCPU(1, G_matrix.num_rows, C_new);
#ifdef GENERATE
    double* a_rel = calculate_a_dynamic(G_matrix.num_cols, minNormVector, Sigma_rows, Sigma, G_rows, P0, G_matrix, C_new);
#endif    

    //std::cout << "a_real vector\n";
    //printMatrixCPU(in_a_csc.num_rows, 1, a_rel);

    double nevC = 0; //Вектор невзки С
    for (int i = 0; i < G_matrix.num_rows; i++) {
        //if(fabs(C_new[i] - C[i]) > 0.0001){
        //    std::cout << "C[" << i << "] = " << fabs(C_new[i] - C[i]) << "\n";
        //}
        nevC += fabs(C_new[i] - C[i]);
    }
    std::cout << "Neviazka po C = " << nevC << std::endl;
    nevC = cblas_dnrm2(G_matrix.num_rows, C_new, 1);
    std::cout << "norm2(C) = " << nevC << std::endl;
    /*double * minNormVector = new_host_darray(basis.num_rows);
    std::cout << "inSetRows " << basis.num_rows << " columns " << basis.num_cols << " nonzerrows " << basis.num_nonzeros << "\n";
    print_csr_matrix(basis);
    getMinNormElemOutRepr(basis, minNormVector, basis.num_cols, basis.num_rows, (double) 0.001);*/
    delete_host_matrix(inA);
    //delete_csc_matrix(in_a_csc);
    delete_host_array(minNormVector);
    free(basisVecInx);
    free(kvec);
    free(C);
    free(C_new);
#ifdef GENERATE        
    free(Sigma);
    free(P0);
#endif


    free(a_rel);

}

coo_matrix read_coo_matrix(const char *coo_fn) {
    FILE* coo_file;
    coo_matrix coo_matrix;
    if ((coo_file = fopen(coo_fn, "r")) == NULL) {
        std::cout << "No such file:" << coo_fn << "\n";

    } else {
        int fscanRes;
        char str [100];
        
        fgets(str,100,coo_file);
        fgets(str,100,coo_file);
        fgets(str,100,coo_file);
        printf ("%s\n", str);
        int nonseroz = 0;
        int columns = 0;
        int rows = 0;
        fscanRes = fscanf(coo_file, "# nnz: %d\n", &nonseroz);
        fscanRes = fscanf(coo_file, "# rows: %d\n", &rows);
        fscanRes = fscanf(coo_file, "# columns: %d\n", &columns);
        
        std::cout << "Rows in input matrix " << rows << " columns " << columns << " nonzeros " << nonseroz << "\n";
        
        coo_matrix.num_cols = columns;
        coo_matrix.num_rows = rows;
        coo_matrix.num_nonzeros = nonseroz;
    
        coo_matrix.I = new_host_iarray(coo_matrix.num_nonzeros); //colsInMatrix, добавляется для того чтобы добавить условия x > 0
        coo_matrix.J = new_host_iarray(coo_matrix.num_nonzeros);
        coo_matrix.V = new_host_darray(coo_matrix.num_nonzeros);
        
        int row_idx;
        int col_idx;
        double value;
       
        for (int i = 0; i < nonseroz; i++) {
            fscanRes = fscanf(coo_file, "%d %d %lg\n", &row_idx, &col_idx, &value);
            coo_matrix.I[i] = row_idx - 1;
            coo_matrix.J[i] = col_idx - 1;
            coo_matrix.V[i] = value;
        }
    }

    return coo_matrix;

}

csr_matrix read_ir_prob_matrix(const char *irows_fn, const char *cols_vals_fn) {
    FILE *cols_vals;
    FILE *irows;

    int all_vals_num = 0;
    if ((cols_vals = fopen(cols_vals_fn, "r")) == NULL) {
        std::cout << "No such file:" << cols_vals_fn << "\n";

    } else {
        all_vals_num = 0;
        ifstream in(cols_vals_fn);

        while (in.good()) {
            std::string line;
            std::getline(in, line);
            ++all_vals_num;
        }
        in.close();
        std::cout << "File " << cols_vals_fn << " has " << all_vals_num << " rows\n";
    }

    int irows_number = 0;

    if ((irows = fopen(irows_fn, "r")) == NULL) {
        std::cout << "No such file" << irows_fn << "\n";
    } else {
        irows_number = 0;
        ifstream in(irows_fn);
        //while ( ! in.eof() )
        while (in.good()) {
            std::string line;
            std::getline(in, line);
            ++irows_number;
        }
        in.close();
        std::cout << "File " << irows_fn << " has " << irows_number << " rows\n";
    }



    csr_matrix out_matrix;
    out_matrix.num_cols = 0;
    out_matrix.num_nonzeros = all_vals_num;
    out_matrix.num_rows = irows_number;

    out_matrix.Ap = new_host_iarray(out_matrix.num_rows + 1);
    out_matrix.Aj = new_host_iarray(out_matrix.num_nonzeros);
    out_matrix.Ax = new_host_darray(out_matrix.num_nonzeros);

    int fscanRes;
    int col_idx;
    double value;
    int max_call_num = 0;
    for (int i = 0; i < all_vals_num; i++) {
        fscanRes = fscanf(cols_vals, "%d %lg\n", &col_idx, &value);
        out_matrix.Aj[i] = col_idx - 1;
        out_matrix.Ax[i] = value;
        if (max_call_num < col_idx) {
            max_call_num = col_idx;
        }
    }
    out_matrix.num_cols = max_call_num;

    int row_idx;
    for (int i = 0; i < irows_number; i++) {
        fscanRes = fscanf(irows, "%d\n", &row_idx);
        out_matrix.Ap[i] = row_idx - 1;
    }
    out_matrix.Ap[row_idx + 1] = row_idx - 1;

    std::cout << "Data loaded\n";
    fclose(cols_vals);
    fclose(irows);

    return out_matrix;
}

void test_simplex_projection() {
    coo_matrix inA_coo = read_coo_matrix("/home/pozpl/tmp/SimProBig/oct/spar");
    //csr_matrix inA = read_ir_prob_matrix("/home/pozpl/tmp/SimProBig/new-ir.csr", "/home/pozpl/tmp/SimProBig/new-prob.csr");

    
    csr_matrix inA = coo2csr(inA_coo); //coo_to_csr(inA_coo);
    delete_coo_matrix(inA_coo);
    csr_matrix inA_t = csr_transpose(inA);
    delete_csr_matrix(inA);
       
    csc_matrix in_a_csc;
    in_a_csc.num_cols = inA_t.num_rows;
    in_a_csc.num_rows = inA_t.num_cols;
    in_a_csc.num_nonzeros = inA_t.num_nonzeros;
    in_a_csc.Cp = inA_t.Ap;
    in_a_csc.Ri = inA_t.Aj;
    in_a_csc.Ex = inA_t.Ax;

    
    //std::cout << "inA_coo.num_cols = " << inA_coo.num_cols << " inA_coo.num_rows = " << inA_coo.num_rows << " \n";

    //delete_host_matrix(inA_coo);

    double * minNormVector = new_host_darray(in_a_csc.num_rows);
    for (int i = 0; i < in_a_csc.num_rows; i++) {
        minNormVector[i] = 0.0;
    }

    std::cout << "inSetRows " << in_a_csc.num_rows << " columns " << in_a_csc.num_cols << " nonzerrows " << in_a_csc.num_nonzeros << "\n";


    int inSetPow = in_a_csc.num_cols;
    int vectorDim = in_a_csc.num_rows;


    std::cout << "inSetDimension " << inSetPow << " vectorDim " << vectorDim << "\n";
    //print_csr_matrix(inSet);

    //getProjectionOnConus(inSet, minNormVector, inSetPow, vectorDim, (double) 0.001);
    //proj_to_lp_cpu(inSet, minNormVector, inSetPow, vectorDim, (double) 0.0001);
    int *kvec = new_host_iarray(2 * in_a_csc.num_cols + 1);
    int *basisVecInx = new_host_iarray(2 * in_a_csc.num_cols + 1);
    for (int i = 0; i < 2 * in_a_csc.num_cols + 1; i++) {
        kvec[i] = 0.0;
        basisVecInx[i] = 0.0;
    }
    int baselen = 0;

    getSimplexProjection(in_a_csc, minNormVector, 0.0001, kvec, basisVecInx, baselen);
    /*
    for(int i=0; i<in_a_csc.num_rows; i++){
        if(minNormVector[i]!=0){
            std::cout<<"z[" << i << "] = "<< minNormVector[i] << "\n";
        }
    }
     */ 
}

//template <typename int, typename double>

double* readRightSide(const char *fname, int &vectorDim) {
    FILE *f;
    double *rightSide;
    if ((f = fopen(fname, "r")) == NULL)
        return NULL;
    //read vector dimension
    int rez = fscanf(f, "%d \n", &vectorDim);

    rightSide = new_host_darray(vectorDim);
    int valIndex;

    for (int i = 0; i < vectorDim; i++) {
        rez = fscanf(f, "%d %lg\n", &valIndex, &rightSide[i]);
        std::cout << "Right vector [" << valIndex << "] " << rightSide[i] << "\n";
    }
    fclose(f);
    return rightSide;
}

double* calculate_a_dynamic(int z_dim, double *z, int sigmaRows, double *Sigma, int G_rows, double *P0, coo_matrix &G_matrix, double* C) {

    double* a_result = new_host_darray(z_dim);

    //Sigma = new_host_darray(sigmaRows);
    //P0 = new_host_darray(sigmaRows);

    cblas_dcopy(z_dim, z, 1, a_result, 1);
    /////////////////////////READ SIGMA MATRIX///////////////////////////////////////////////////////////////////////////////////////
    /*Sigma matrix will be readed as vector
     */
    //read vector dimension
    for (int i = 0; i < sigmaRows; i++) {
        a_result[i] *= sqrt(Sigma[i]);
    }

    //////////////////////END READ SIGMA MATRIX///////////////////////////////////////////////////////////////////
    /////////////////////////////////////////START READ P0 matrix///////////////////////////////////////////

    for (int i = 0; i < sigmaRows; i++) {
        a_result[i] += P0[i];
    }
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    ////////////////////////////////END READ P0 MATRIX///////////////////////////////////////////////////////////////////////

    for (int i = 0; i < G_matrix.num_rows; i++) {
        C[i] = 0.0;
    }

    for (int i = 0; i < G_matrix.num_nonzeros; i++) {
        C[G_matrix.I[i]] += G_matrix.V[i] * a_result[G_matrix.J[i]];
    }

    return a_result;
}

double* calculate_a(int z_dim, double *z, const char *fnameSigma, const char *fnameP0) {
    FILE *f_Sigma;
    FILE *f_P0;

    if ((f_Sigma = fopen(fnameSigma, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_P0 = fopen(fnameP0, "r")) == NULL) {
        std::cout << "No such file\n";

    }

    double* a_result = new_host_darray(z_dim);

    cblas_dcopy(z_dim, z, 1, a_result, 1);

    /////////////////////////READ SIGMA MATRIX///////////////////////////////////////////////////////////////////////////////////////
    /*Sigma matrix will be readed as vector
     */
    //read vector dimension
    int SigmaVecDim, SigmaColsInMatrix, SigmaNonzerrosInMatrix;
    int fscanRes = fscanf(f_Sigma, "%d %d %d\n", &SigmaVecDim, &SigmaColsInMatrix, &SigmaNonzerrosInMatrix);
    double* Sigma = new_host_darray(SigmaNonzerrosInMatrix);
    std::cout << "Sigma vectordim:" << SigmaVecDim << " Cols in matrix:" << SigmaColsInMatrix << " Nonzero " << SigmaNonzerrosInMatrix << "\n";
    int rowIndex;
    int colIndex;
    double value;
    for (int i = 0; i < SigmaNonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_Sigma, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        Sigma[rowIndex - 1] = value;
        a_result[rowIndex - 1] *= sqrt(value);
    }

    //////////////////////END READ SIGMA MATRIX///////////////////////////////////////////////////////////////////
    /////////////////////////////////////////START READ P0 matrix///////////////////////////////////////////
    int P0_VecDim, P0_ColsInMatrix, P0_NonzerrosInMatrix;
    fscanRes = fscanf(f_P0, "%d %d %d\n", &P0_VecDim, &P0_ColsInMatrix, &P0_NonzerrosInMatrix);
    std::cout << "P0 vectordim:" << P0_VecDim << " Cols in matrix:" << P0_ColsInMatrix << " Nonzero " << P0_NonzerrosInMatrix << "\n";
    double* P0 = new_host_darray(P0_ColsInMatrix);
    P0_VecDim = P0_ColsInMatrix;
    for (int i = 0; i < P0_ColsInMatrix; i++) {
        P0[i] = 0.0;
    }

    for (int i = 0; i < P0_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_P0, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        P0[colIndex - 1] = value;
        a_result[colIndex - 1] += value;
    }
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    ////////////////////////////////END READ P0 MATRIX///////////////////////////////////////////////////////////////////////

    return a_result;
}


//template <typename int, typename double>

coo_matrix readCooMatrix(const char *fnameMatrix, const char *fnameRHS, bool fillXnonZer) {
    FILE *f_matrix;
    FILE *f_rhs;
    coo_matrix coo_matrix;
    coo_matrix.num_cols = 0;
    coo_matrix.num_rows = 0;
    coo_matrix.num_nonzeros = 0;
    coo_matrix.I = NULL;
    coo_matrix.J = NULL;
    coo_matrix.V = NULL;


    if ((f_matrix = fopen(fnameMatrix, "r")) == NULL) {
        std::cout << "No such file\n";
        return coo_matrix;
    }
    if ((f_rhs = fopen(fnameRHS, "r")) == NULL) {
        std::cout << "No such file\n";
        return coo_matrix;
    }
    //read vector dimension
    int vectorDim, colsInMatrix, nonzerrosInMatrix;
    int fscanRes = fscanf(f_matrix, "%d %d %d\n", &vectorDim, &colsInMatrix, &nonzerrosInMatrix);
    //nonzerrosInMatrix++; //Добавляем фиктивный нулевой элемент, так как D = {0, a^i}, где a^i - строки матрицы
    std::cout << "vectorDim " << vectorDim << " colsInMatrix " << colsInMatrix << " nonzerrosInMatrix " << nonzerrosInMatrix << "\n";

    nonzerrosInMatrix;
    // \/ тут теперь всё не так
    //vectorDim--;//В первом столбце хранится целевая функция, это печально, надо её проигнорировать

    coo_matrix.num_cols = colsInMatrix + 1; //один стодбец под вектор b

    if (fillXnonZer) {
        coo_matrix.num_rows = vectorDim + colsInMatrix; // + 1;//так как добавили нулевой элемент в 0 позиция
        coo_matrix.num_nonzeros = nonzerrosInMatrix + vectorDim + 2 * colsInMatrix;
        coo_matrix.I = new_host_iarray(coo_matrix.num_nonzeros); //colsInMatrix, добавляется для того чтобы добавить условия x > 0
        coo_matrix.J = new_host_iarray(coo_matrix.num_nonzeros);
        coo_matrix.V = new_host_darray(coo_matrix.num_nonzeros);
    } else {
        coo_matrix.num_rows = vectorDim;
        coo_matrix.num_nonzeros = nonzerrosInMatrix + vectorDim;
        coo_matrix.I = new_host_iarray(coo_matrix.num_nonzeros);
        coo_matrix.J = new_host_iarray(coo_matrix.num_nonzeros);
        coo_matrix.V = new_host_darray(coo_matrix.num_nonzeros);
    }
    //coo_matrix.I[0] = 0;
    //coo_matrix.J[0] = colsInMatrix;
    //coo_matrix.V[0] = 0;

    int rowIndex;
    int colIndex;
    double value;
    //Это фиктивный цикл, для того чтобы выбрать элементы целевой функции, хранящейся в первой строчке
    //for(int i = 0; i < colsInMatrix; i++){
    //    fscanRes = fscanf(f_matrix, "%d %d %lg\n", &rowIndex, &colIndex, &value);
    //}
    for (int i = 0; i < nonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_matrix, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        //std::cout<< ">>" << rowIndex << " " << colIndex << " " << value <<  "\n";
        //if(rowIndex != 1){//Пропускаем первую строчку с целевой функцией
        coo_matrix.I[i] = rowIndex - 1; //coo_matrix.I[i] = rowIndex - 1;
        coo_matrix.J[i] = colIndex - 1;
        coo_matrix.V[i] = value;
        //}

    }
    int vectorDimRHS;
    fscanRes = fscanf(f_rhs, "%d\n", &vectorDimRHS);
    std::cout << "vectorDimRHS " << vectorDimRHS << "\n";
    //fscanRes = fscanf(f_rhs, "%d %lg\n", &rowIndex, &value);//читаем желаемое значение целевой функции
    for (int i = nonzerrosInMatrix, j = 0; i < nonzerrosInMatrix + vectorDim; i++, j++) {
        fscanRes = fscanf(f_rhs, "%d %lg\n", &rowIndex, &value);
        //std::cout<< "" << rowIndex << " "  << value <<  "\n";
        //if(j != 0){//Пропускаем первую строчку с целевой функцией
        coo_matrix.I[i] = j; //rowIndex - 1;
        coo_matrix.J[i] = colsInMatrix;
        coo_matrix.V[i] = -value; //так как X = [A, -C]'
        //}
    }

    if (fillXnonZer) {
        for (int i = nonzerrosInMatrix + vectorDim, j = vectorDim, k = 0; i < nonzerrosInMatrix + vectorDim + colsInMatrix; i++, j++, k++) {
            coo_matrix.I[i] = j; //rowIndex - 1;
            coo_matrix.J[i] = k;
            coo_matrix.V[i] = (double) - 1.0;
        }

        for (int i = nonzerrosInMatrix + vectorDim + colsInMatrix, j = vectorDim; i < nonzerrosInMatrix + vectorDim + 2 * colsInMatrix; i++, j++) {
            coo_matrix.I[i] = j; //rowIndex - 1;
            coo_matrix.J[i] = colsInMatrix;
            coo_matrix.V[i] = (double) 0.0;
        }
    }

    fclose(f_matrix);
    fclose(f_rhs);

    return coo_matrix;
}

/*
 *Return matrix with additional information about how much rows in matrix presented
 * by equality constrains
 */
coo_matrix readAuProblemMtxPosCon(const char *fnameSigma, const char *fnameG, const char *fnameC, const char *fnameP0, int &numberOfEqConstr, coo_matrix &G_matrix, double* C) {
    FILE *f_Sigma;
    FILE *f_G;
    FILE *f_C;
    FILE *f_P0;




    if ((f_Sigma = fopen(fnameSigma, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_G = fopen(fnameG, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_C = fopen(fnameC, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_P0 = fopen(fnameP0, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    /////////////////////////READ SIGMA MATRIX///////////////////////////////////////////////////////////////////////////////////////
    /*Sigma matrix will be readed as vector
     */
    //read vector dimension
    int SigmaVecDim, SigmaColsInMatrix, SigmaNonzerrosInMatrix;
    int fscanRes = fscanf(f_Sigma, "%d %d %d\n", &SigmaVecDim, &SigmaColsInMatrix, &SigmaNonzerrosInMatrix);
    double* Sigma = new_host_darray(SigmaNonzerrosInMatrix);
    std::cout << "Sigma vectordim:" << SigmaVecDim << " Cols in matrix:" << SigmaColsInMatrix << " Nonzero " << SigmaNonzerrosInMatrix << "\n";
    int rowIndex;
    int colIndex;
    double value;
    for (int i = 0; i < SigmaNonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_Sigma, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        Sigma[rowIndex - 1] = value;
    }

    //////////////////////END READ SIGMA MATRIX////////READ G MATRIX/////////////////////////////////////////////////////////////
    int G_VecDim, G_ColsInMatrix, G_NonzerrosInMatrix;
    fscanRes = fscanf(f_G, "%d %d %d\n", &G_VecDim, &G_ColsInMatrix, &G_NonzerrosInMatrix);
    std::cout << "G vectordim:" << G_VecDim << " Cols in matrix:" << G_ColsInMatrix << " Nonzero " << G_NonzerrosInMatrix << "\n";

    //coo_matrix G_matrix;
    G_matrix.num_cols = 0;
    G_matrix.num_rows = 0;
    G_matrix.num_nonzeros = 0;
    G_matrix.I = NULL;
    G_matrix.J = NULL;
    G_matrix.V = NULL;

    G_matrix.num_rows = G_VecDim;
    G_matrix.num_nonzeros = G_NonzerrosInMatrix;
    G_matrix.num_cols = G_ColsInMatrix;
    G_matrix.I = new_host_iarray(G_matrix.num_nonzeros);
    G_matrix.J = new_host_iarray(G_matrix.num_nonzeros);
    G_matrix.V = new_host_darray(G_matrix.num_nonzeros);

    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_G, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        //std::cout<< ">>" << rowIndex << " " << colIndex << " " << value <<  "\n";
        //if(rowIndex != 1){//Пропускаем первую строчку с целевой функцией
        G_matrix.I[i] = rowIndex - 1; //coo_matrix.I[i] = rowIndex - 1;
        G_matrix.J[i] = colIndex - 1;
        G_matrix.V[i] = value;
        //}

    }

    //print_coo_matrix(G_matrix);
    //////////////////////////////////END READ G MATRIX/////////START READ C matrix///////////////////////////////////////////
    int C_VecDim, C_ColsInMatrix, C_NonzerrosInMatrix;
    fscanRes = fscanf(f_C, "%d %d %d\n", &C_VecDim, &C_ColsInMatrix, &C_NonzerrosInMatrix);
    std::cout << "C vectordim:" << C_VecDim << " Cols in matrix:" << C_ColsInMatrix << " Nonzero " << C_NonzerrosInMatrix << "\n";
    //C = new_host_darray(C_ColsInMatrix);
    for (int i = 0; i < C_ColsInMatrix; i++) {
        C[i] = 0.0;
    }

    for (int i = 0; i < C_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_C, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        C[colIndex - 1] = value;
    }

    //printMatrixCPU(1, C_ColsInMatrix, C);
    //////////////////////////////////END READ C MATRIX/////////START READ P0 matrix///////////////////////////////////////////
    int P0_VecDim, P0_ColsInMatrix, P0_NonzerrosInMatrix;
    fscanRes = fscanf(f_P0, "%d %d %d\n", &P0_VecDim, &P0_ColsInMatrix, &P0_NonzerrosInMatrix);
    std::cout << "P0 vectordim:" << P0_VecDim << " Cols in matrix:" << P0_ColsInMatrix << " Nonzero " << P0_NonzerrosInMatrix << "\n";
    double* P0 = new_host_darray(P0_ColsInMatrix);
    P0_VecDim = P0_ColsInMatrix;
    for (int i = 0; i < P0_ColsInMatrix; i++) {
        P0[i] = 0.0;
    }

    for (int i = 0; i < P0_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_P0, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        P0[colIndex - 1] = value;
    }
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    ////////////////////////////////END READ P0 MATRIX///////////////////////////////////////////////////////////////////////
    numberOfEqConstr = G_VecDim;
    int outMatrixRows = G_VecDim + P0_VecDim;
    int outMatrixCols = G_ColsInMatrix + 1;
    int outMatrixNonz = (G_NonzerrosInMatrix + P0_NonzerrosInMatrix) + (G_VecDim + P0_VecDim); //first brakets conditions of equality, second positive constrains

    coo_matrix out_matrix;
    //out_matrix.num_cols = G_ColsInMatrix + 1;
    //out_matrix.num_rows = G_VecDim * 2 + P0_VecDim;
    //out_matrix.num_nonzeros = outMatrixNonz;
    out_matrix.I = NULL;
    out_matrix.J = NULL;
    out_matrix.V = NULL;

    out_matrix.num_rows = outMatrixRows;
    out_matrix.num_nonzeros = outMatrixNonz;
    out_matrix.num_cols = outMatrixCols;
    out_matrix.I = new_host_iarray(outMatrixNonz);
    out_matrix.J = new_host_iarray(outMatrixNonz);
    out_matrix.V = new_host_darray(outMatrixNonz);
    for (int i = 0; i < outMatrixNonz; i++) {
        out_matrix.I[i] = 0;
        out_matrix.J[i] = 0;
        out_matrix.V[i] = 0.0;
    }


    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        out_matrix.V[i] = G_matrix.V[i] * sqrt(Sigma[G_matrix.J[i]]);
        out_matrix.I[i] = G_matrix.I[i];
        out_matrix.J[i] = G_matrix.J[i];

        //out_matrix.V[i + G_NonzerrosInMatrix] = -G_matrix.V[i] * sqrt(Sigma[G_matrix.J[i]]);
        //out_matrix.I[i + G_NonzerrosInMatrix] = G_matrix.I[i] + G_VecDim;
        //out_matrix.J[i + G_NonzerrosInMatrix]  = G_matrix.J[i];
    }


    std::cout << "Nonzer constrains\n";

    for (int i = G_NonzerrosInMatrix, j = G_VecDim; i < G_NonzerrosInMatrix + P0_NonzerrosInMatrix; i++, j++) {
        out_matrix.V[i] = -1.0;
        out_matrix.I[i] = j;
        out_matrix.J[i] = j - G_VecDim;
    }


    /////////////////////////FILL RHS/////////////////////////////////////////////////
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    std::cout << "Compute RHS\n";
    double* G1Xa = new_host_darray(G_VecDim);
    for (int i = 0; i < G_VecDim; i++) {
        G1Xa[i] = 0;
    }
    std::cout << "null\n";
    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        G1Xa[G_matrix.I[i]] += G_matrix.V[i] * P0[G_matrix.J[i]];
        //std::cout<<"G1Xa["<< G_matrix.I[i] << "] =  "<< G1Xa[G_matrix.I[i]] << " G " << G_matrix.V[i] << " P0["<< G_matrix.J[i] <<" ]" << P0[G_matrix.J[i]] << "\n";
    }

    //printMatrixCPU(1, G_VecDim, G1Xa);
    //!!!!!!!!!!!NOTE we mast add RHS with -1 coefficient
    std::cout << "Add SigmaRHS\n";
    int outMatrixShift = G_NonzerrosInMatrix + P0_NonzerrosInMatrix;
    for (int i = outMatrixShift, j = 0; i < outMatrixShift + G_VecDim; i++, j++) {
        out_matrix.V[i] = -(C[j] - G1Xa[j]); //with -1 coefficient
        out_matrix.I[i] = j;
        out_matrix.J[i] = G_ColsInMatrix;

        //out_matrix.V[i + G_VecDim] = (C[j] - G1Xa[j]);//with -1 coeff
        //out_matrix.I[i + G_VecDim] = j + G_VecDim;
        //out_matrix.J[i + G_VecDim]  = G_ColsInMatrix;
    }
    std::cout << "Add nonzeros RHS\n";
    outMatrixShift = G_NonzerrosInMatrix + P0_NonzerrosInMatrix + G_VecDim;
    for (int i = outMatrixShift, j = 0; i < outMatrixShift + P0_VecDim; i++, j++) {
        out_matrix.V[i] = -P0[j] / sqrt(Sigma[j]); //with -1 coeff
        out_matrix.I[i] = j + G_VecDim;
        out_matrix.J[i] = G_ColsInMatrix;
    }

    //print_coo_matrix(out_matrix);
    delete_host_array(Sigma);
    delete_host_array(P0);
    //delete_host_matrix(G_matrix);
    free(G1Xa);
    //free(C);
    std::cout << "\n";
    fclose(f_Sigma);
    fclose(f_C);
    fclose(f_G);
    fclose(f_P0);

    return out_matrix;
}

/*
 *Return matrix with additional information about how much rows in matrix presented
 * by equality constrains
 */
coo_matrix generateAuProblemMtxPosCon(int Sigma_rows, int G_rows, int &numberOfEqConstr, double* Sigma, double* P0, coo_matrix &G_matrix, double* C) {

    /////////////////////////READ SIGMA MATRIX///////////////////////////////////////////////////////////////////////////////////////
    /*Sigma matrix will be readed as vector
     */
    //read vector dimension
    //int SigmaVecDim, SigmaColsInMatrix, SigmaNonzerrosInMatrix;
    //int fscanRes = fscanf(f_Sigma, "%d %d %d\n", &SigmaVecDim, &SigmaColsInMatrix, &SigmaNonzerrosInMatrix);
    //Sigma = new_host_darray(Sigma_rows);
    //std::cout << "Sigma vectordim:" << SigmaVecDim << " Cols in matrix:" << SigmaColsInMatrix << " Nonzero " << SigmaNonzerrosInMatrix << "\n";
    //int rowIndex;
    //int colIndex;
    //double value;
    for (int i = 0; i < Sigma_rows; i++) {
        //fscanRes = fscanf(f_Sigma, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        Sigma[i] = rand2(0, 100);
    }

    //////////////////////END READ SIGMA MATRIX////////READ G MATRIX/////////////////////////////////////////////////////////////
    int G_VecDim = G_rows;
    int G_ColsInMatrix = Sigma_rows;
    int G_NonzerrosInMatrix = G_VecDim * G_ColsInMatrix;
    //fscanRes = fscanf(f_G, "%d %d %d\n", &G_VecDim, &G_ColsInMatrix, &G_NonzerrosInMatrix);
    //std::cout << "G vectordim:" << G_VecDim << " Cols in matrix:" << G_ColsInMatrix << " Nonzero " << G_NonzerrosInMatrix << "\n";


    G_matrix.num_cols = 0;
    G_matrix.num_rows = 0;
    G_matrix.num_nonzeros = 0;
    G_matrix.I = NULL;
    G_matrix.J = NULL;
    G_matrix.V = NULL;

    G_matrix.num_rows = G_VecDim;
    G_matrix.num_nonzeros = G_NonzerrosInMatrix;
    G_matrix.num_cols = G_ColsInMatrix;
    G_matrix.I = new_host_iarray(G_matrix.num_nonzeros);
    G_matrix.J = new_host_iarray(G_matrix.num_nonzeros);
    G_matrix.V = new_host_darray(G_matrix.num_nonzeros);

    for (int rowIndex = 0; rowIndex < G_VecDim; rowIndex++) {
        for (int colIndex = 0; colIndex < G_ColsInMatrix; colIndex++) {
            //fscanRes = fscanf(f_G, "%d %d %lg\n", &rowIndex, &colIndex, &value);
            //std::cout<< ">>" << rowIndex << " " << colIndex << " " << value <<  "\n";
            //if(rowIndex != 1){//Пропускаем первую строчку с целевой функцией
            G_matrix.I[colIndex * G_VecDim + rowIndex] = rowIndex; //coo_matrix.I[i] = rowIndex - 1;
            G_matrix.J[colIndex * G_VecDim + rowIndex] = colIndex;
            if (rand2(0.0, 100) > 50) {
                G_matrix.V[colIndex * G_VecDim + rowIndex] = rand2(-1000, 1000);
            } else {
                G_matrix.V[colIndex * G_VecDim + rowIndex] = 0.0;
            }
            //}
        }
    }

    //print_coo_matrix(G_matrix);
    //////////////////////////////////END READ G MATRIX/////////START READ C matrix///////////////////////////////////////////
    //int C_VecDim, C_ColsInMatrix, C_NonzerrosInMatrix;
    //fscanRes = fscanf(f_C, "%d %d %d\n", &C_VecDim, &C_ColsInMatrix, &C_NonzerrosInMatrix);
    //std::cout << "C vectordim:" << C_VecDim << " Cols in matrix:" << C_ColsInMatrix << " Nonzero " << C_NonzerrosInMatrix << "\n";
    double* X_tmp = new_host_darray(G_matrix.num_cols);
    for (int i = 0; i < G_matrix.num_cols; i++) {
        X_tmp[i] = rand2(0, 100);
    }
    //double* C = new_host_darray(G_rows);
    for (int i = 0; i < G_matrix.num_rows; i++) {
        C[i] = 0.0;
    }

    for (int i = 0; i < G_matrix.num_nonzeros; i++) {
        C[G_matrix.I[i]] += G_matrix.V[i] * X_tmp[G_matrix.J[i]];
    }


    //for(int i = 0; i < G_rows; i++){C[i] = rand2(0, 100);}

    //for (int i = 0; i < C_NonzerrosInMatrix; i++) {
    //    fscanRes =numberOfEqConstr fscanf(f_C, "%d %d %lg\n", &rowIndex, &colIndex, &value);
    //    C[colIndex - 1] = value;
    //}

    //printMatrixCPU(1, C_ColsInMatrix, C);
    //////////////////////////////////END READ C MATRIX/////////START READ P0 matrix///////////////////////////////////////////
    //int P0_VecDim, P0_ColsInMatrix, P0_NonzerrosInMatrix;
    //fscanRes = fscanf(f_P0, "%d %d %d\n", &P0_VecDim, &P0_ColsInMatrix, &P0_NonzerrosInMatrix);
    //std::cout << "P0 vectordim:" << P0_VecDim << " Cols in matrix:" << P0_ColsInMatrix << " Nonzero " << P0_NonzerrosInMatrix << "\n";
    //P0 = new_host_darray(Sigma_rows);
    //P0_VecDim = Sigma_rows;
    for (int i = 0; i < Sigma_rows; i++) {
        P0[i] = rand2(0, 100);
    }

    //for (int i = 0; i < P0_NonzerrosInMatrix; i++) {
    //    fscanRes = fscanf(f_P0, "%d %d %lg\n", &rowIndex, &colIndex, &value);
    //    P0[colIndex - 1] = value;
    //}
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    ////////////////////////////////END READ P0 MATRIX///////////////////////////////////////////////////////////////////////
    numberOfEqConstr = G_VecDim;
    int outMatrixRows = G_VecDim + Sigma_rows;
    int outMatrixCols = G_ColsInMatrix + 1;
    int outMatrixNonz = (G_NonzerrosInMatrix + Sigma_rows) + (G_VecDim + Sigma_rows); //first brakets conditions of equality, second positive constrains

    coo_matrix out_matrix;
    //out_matrix.num_cols = G_ColsInMatrix + 1;
    //out_matrix.num_rows = G_VecDim * 2 + P0_VecDim;
    //out_matrix.num_nonzeros = outMatrixNonz;
    out_matrix.I = NULL;
    out_matrix.J = NULL;
    out_matrix.V = NULL;

    out_matrix.num_rows = outMatrixRows;
    out_matrix.num_nonzeros = outMatrixNonz;
    out_matrix.num_cols = outMatrixCols;
    out_matrix.I = new_host_iarray(outMatrixNonz);
    out_matrix.J = new_host_iarray(outMatrixNonz);
    out_matrix.V = new_host_darray(outMatrixNonz);
    for (int i; i < outMatrixNonz; i++) {
        out_matrix.I[i] = 0;
        out_matrix.J[i] = 0;
        out_matrix.V[i] = 0.0;
    }


    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        out_matrix.V[i] = G_matrix.V[i] * sqrt(Sigma[G_matrix.J[i]]);
        out_matrix.I[i] = G_matrix.I[i];
        out_matrix.J[i] = G_matrix.J[i];

        //out_matrix.V[i + G_NonzerrosInMatrix] = -G_matrix.V[i] * sqrt(Sigma[G_matrix.J[i]]);
        //out_matrix.I[i + G_NonzerrosInMatrix] = G_matrix.I[i] + G_VecDim;
        //out_matrix.J[i + G_NonzerrosInMatrix]  = G_matrix.J[i];
    }


    std::cout << "Nonzer constrains\n";

    for (int i = G_NonzerrosInMatrix, j = G_VecDim; i < G_NonzerrosInMatrix + Sigma_rows; i++, j++) {
        out_matrix.V[i] = -1.0;
        out_matrix.I[i] = j;
        out_matrix.J[i] = j - G_VecDim;
    }


    /////////////////////////FILL RHS/////////////////////////////////////////////////
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    std::cout << "Compute RHS\n";
    double* G1Xa = new_host_darray(G_VecDim);
    for (int i = 0; i < G_VecDim; i++) {
        G1Xa[i] = 0;
    }
    std::cout << "null\n";
    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        G1Xa[G_matrix.I[i]] += G_matrix.V[i] * P0[G_matrix.J[i]];
        //std::cout<<"G1Xa["<< G_matrix.I[i] << "] =  "<< G1Xa[G_matrix.I[i]] << " G " << G_matrix.V[i] << " P0["<< G_matrix.J[i] <<" ]" << P0[G_matrix.J[i]] << "\n";
    }

    //printMatrixCPU(1, G_VecDim, G1Xa);
    //!!!!!!!!!!!NOTE we mast add RHS with -1 coefficient
    std::cout << "Add SigmaRHS\n";
    int outMatrixShift = G_NonzerrosInMatrix + Sigma_rows;
    for (int i = outMatrixShift, j = 0; i < outMatrixShift + G_VecDim; i++, j++) {
        out_matrix.V[i] = -(C[j] - G1Xa[j]); //with -1 coefficient
        out_matrix.I[i] = j;
        out_matrix.J[i] = G_ColsInMatrix;

        //out_matrix.V[i + G_VecDim] = (C[j] - G1Xa[j]);//with -1 coeff
        //out_matrix.I[i + G_VecDim] = j + G_VecDim;
        //out_matrix.J[i + G_VecDim]  = G_ColsInMatrix;
    }
    std::cout << "Add nonzeros RHS\n";
    outMatrixShift = G_NonzerrosInMatrix + Sigma_rows + G_VecDim;
    for (int i = outMatrixShift, j = 0; i < outMatrixShift + Sigma_rows; i++, j++) {
        out_matrix.V[i] = -P0[j] / sqrt(Sigma[j]); //with -1 coeff
        out_matrix.I[i] = j + G_VecDim;
        out_matrix.J[i] = G_ColsInMatrix;
    }

    //print_coo_matrix(out_matrix);
    //delete_host_array(Sigma);
    //delete_host_array(P0);
    //delete_host_matrix(G_matrix);
    free(G1Xa);
    free(X_tmp);
    //free(C);
    std::cout << "\n";


    return out_matrix;
}

/*
 *Return matrix with additional information about how much rows in matrix presented
 * by equality constrains
 */
coo_matrix readAuProblemMtxPosConNormEqConstr(const char *fnameSigma, const char *fnameG, const char *fnameC, const char *fnameP0, int &numberOfEqConstr) {
    FILE *f_Sigma;
    FILE *f_G;
    FILE *f_C;
    FILE *f_P0;




    if ((f_Sigma = fopen(fnameSigma, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_G = fopen(fnameG, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_C = fopen(fnameC, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_P0 = fopen(fnameP0, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    /////////////////////////READ SIGMA MATRIX///////////////////////////////////////////////////////////////////////////////////////
    /*Sigma matrix will be readed as vector
     */
    //read vector dimension
    int SigmaVecDim, SigmaColsInMatrix, SigmaNonzerrosInMatrix;
    int fscanRes = fscanf(f_Sigma, "%d %d %d\n", &SigmaVecDim, &SigmaColsInMatrix, &SigmaNonzerrosInMatrix);
    double* Sigma = new_host_darray(SigmaNonzerrosInMatrix);
    std::cout << "Sigma vectordim:" << SigmaVecDim << " Cols in matrix:" << SigmaColsInMatrix << " Nonzero " << SigmaNonzerrosInMatrix << "\n";
    int rowIndex;
    int colIndex;
    double value;
    for (int i = 0; i < SigmaNonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_Sigma, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        Sigma[rowIndex - 1] = value;
    }

    //////////////////////END READ SIGMA MATRIX////////READ G MATRIX/////////////////////////////////////////////////////////////
    int G_VecDim, G_ColsInMatrix, G_NonzerrosInMatrix;
    fscanRes = fscanf(f_G, "%d %d %d\n", &G_VecDim, &G_ColsInMatrix, &G_NonzerrosInMatrix);
    std::cout << "G vectordim:" << G_VecDim << " Cols in matrix:" << G_ColsInMatrix << " Nonzero " << G_NonzerrosInMatrix << "\n";

    coo_matrix G_matrix;
    G_matrix.num_cols = 0;
    G_matrix.num_rows = 0;
    G_matrix.num_nonzeros = 0;
    G_matrix.I = NULL;
    G_matrix.J = NULL;
    G_matrix.V = NULL;

    G_matrix.num_rows = G_VecDim;
    G_matrix.num_nonzeros = G_NonzerrosInMatrix;
    G_matrix.num_cols = G_ColsInMatrix;
    G_matrix.I = new_host_iarray(G_matrix.num_nonzeros);
    G_matrix.J = new_host_iarray(G_matrix.num_nonzeros);
    G_matrix.V = new_host_darray(G_matrix.num_nonzeros);

    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_G, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        //std::cout<< ">>" << rowIndex << " " << colIndex << " " << value <<  "\n";
        //if(rowIndex != 1){//Пропускаем первую строчку с целевой функцией
        G_matrix.I[i] = rowIndex - 1; //coo_matrix.I[i] = rowIndex - 1;
        G_matrix.J[i] = colIndex - 1;
        G_matrix.V[i] = value;
        //}

    }

    //print_coo_matrix(G_matrix);
    //////////////////////////////////END READ G MATRIX/////////START READ C matrix///////////////////////////////////////////
    int C_VecDim, C_ColsInMatrix, C_NonzerrosInMatrix;
    fscanRes = fscanf(f_C, "%d %d %d\n", &C_VecDim, &C_ColsInMatrix, &C_NonzerrosInMatrix);
    std::cout << "C vectordim:" << C_VecDim << " Cols in matrix:" << C_ColsInMatrix << " Nonzero " << C_NonzerrosInMatrix << "\n";
    double* C = new_host_darray(C_ColsInMatrix);
    for (int i = 0; i < C_ColsInMatrix; i++) {
        C[i] = 0.0;
    }

    for (int i = 0; i < C_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_C, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        C[colIndex - 1] = value;
    }

    //printMatrixCPU(1, C_ColsInMatrix, C);
    //////////////////////////////////END READ C MATRIX/////////START READ P0 matrix///////////////////////////////////////////
    int P0_VecDim, P0_ColsInMatrix, P0_NonzerrosInMatrix;
    fscanRes = fscanf(f_P0, "%d %d %d\n", &P0_VecDim, &P0_ColsInMatrix, &P0_NonzerrosInMatrix);
    std::cout << "P0 vectordim:" << P0_VecDim << " Cols in matrix:" << P0_ColsInMatrix << " Nonzero " << P0_NonzerrosInMatrix << "\n";
    double* P0 = new_host_darray(P0_ColsInMatrix);
    P0_VecDim = P0_ColsInMatrix;
    for (int i = 0; i < P0_ColsInMatrix; i++) {
        P0[i] = 0.0;
    }

    for (int i = 0; i < P0_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_P0, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        P0[colIndex - 1] = value;
    }
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    ////////////////////////////////END READ P0 MATRIX///////////////////////////////////////////////////////////////////////
    numberOfEqConstr = 0;
    int outMatrixRows = G_VecDim * 2 + P0_VecDim;
    int outMatrixCols = G_ColsInMatrix + 1;
    int outMatrixNonz = (G_NonzerrosInMatrix * 2 + P0_NonzerrosInMatrix) + (G_VecDim * 2 + P0_VecDim); //first brakets conditions of equality, second positive constrains

    coo_matrix out_matrix;
    //out_matrix.num_cols = G_ColsInMatrix + 1;
    //out_matrix.num_rows = G_VecDim * 2 + P0_VecDim;
    //out_matrix.num_nonzeros = outMatrixNonz;
    out_matrix.I = NULL;
    out_matrix.J = NULL;
    out_matrix.V = NULL;

    out_matrix.num_rows = outMatrixRows;
    out_matrix.num_nonzeros = outMatrixNonz;
    out_matrix.num_cols = outMatrixCols;
    out_matrix.I = new_host_iarray(outMatrixNonz);
    out_matrix.J = new_host_iarray(outMatrixNonz);
    out_matrix.V = new_host_darray(outMatrixNonz);
    for (int i; i < outMatrixNonz; i++) {
        out_matrix.I[i] = 0;
        out_matrix.J[i] = 0;
        out_matrix.V[i] = 0.0;
    }


    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        out_matrix.V[i] = G_matrix.V[i] * sqrt(Sigma[G_matrix.J[i]]);
        out_matrix.I[i] = G_matrix.I[i];
        out_matrix.J[i] = G_matrix.J[i];

        out_matrix.V[i + G_NonzerrosInMatrix] = -G_matrix.V[i] * sqrt(Sigma[G_matrix.J[i]]);
        out_matrix.I[i + G_NonzerrosInMatrix] = G_matrix.I[i] + G_VecDim;
        out_matrix.J[i + G_NonzerrosInMatrix] = G_matrix.J[i];
    }


    std::cout << "Nonzer constrains\n";

    for (int i = 2 * G_NonzerrosInMatrix, j = 2 * G_VecDim; i < 2 * G_NonzerrosInMatrix + P0_NonzerrosInMatrix; i++, j++) {
        out_matrix.V[i] = -1.0;
        out_matrix.I[i] = j;
        out_matrix.J[i] = j - 2 * G_VecDim;
    }


    /////////////////////////FILL RHS/////////////////////////////////////////////////
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    std::cout << "Compute RHS\n";
    double* G1Xa = new_host_darray(G_VecDim);
    for (int i = 0; i < G_VecDim; i++) {
        G1Xa[i] = 0;
    }
    std::cout << "null\n";
    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        G1Xa[G_matrix.I[i]] += G_matrix.V[i] * P0[G_matrix.J[i]];
        //std::cout<<"G1Xa["<< G_matrix.I[i] << "] =  "<< G1Xa[G_matrix.I[i]] << " G " << G_matrix.V[i] << " P0["<< G_matrix.J[i] <<" ]" << P0[G_matrix.J[i]] << "\n";
    }

    //printMatrixCPU(1, G_VecDim, G1Xa);
    //!!!!!!!!!!!NOTE we mast add RHS with -1 coefficient
    std::cout << "Add RHS\n";
    int outMatrixShift = 2 * G_NonzerrosInMatrix + P0_NonzerrosInMatrix;
    for (int i = outMatrixShift, j = 0; i < outMatrixShift + G_VecDim; i++, j++) {
        out_matrix.V[i] = (C[j] - G1Xa[j]); //with -1 coefficient
        out_matrix.I[i] = j;
        out_matrix.J[i] = G_ColsInMatrix;

        out_matrix.V[i + G_VecDim] = (C[j] - G1Xa[j]); //with -1 coeff
        out_matrix.I[i + G_VecDim] = j + G_VecDim;
        out_matrix.J[i + G_VecDim] = G_ColsInMatrix;
    }
    std::cout << "Add nonzeros RHS\n";
    outMatrixShift = 2 * G_NonzerrosInMatrix + P0_NonzerrosInMatrix + 2 * G_VecDim;
    for (int i = outMatrixShift, j = 0; i < outMatrixShift + P0_VecDim; i++, j++) {
        out_matrix.V[i] = -P0[j] / sqrt(Sigma[j]); //with -1 coeff
        out_matrix.I[i] = j + 2 * G_VecDim;
        out_matrix.J[i] = G_ColsInMatrix;
    }

    //print_coo_matrix(out_matrix);
    std::cout << "\n";
    fclose(f_Sigma);
    fclose(f_C);
    fclose(f_G);
    fclose(f_P0);

    return out_matrix;
}

//template <typename int, typename double>

coo_matrix readAuProblemMatrices(const char *fnameSigma, const char *fnameG, const char *fnameC, const char *fnameP0) {
    FILE *f_Sigma;
    FILE *f_G;
    FILE *f_C;
    FILE *f_P0;




    if ((f_Sigma = fopen(fnameSigma, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_G = fopen(fnameG, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_C = fopen(fnameC, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    if ((f_P0 = fopen(fnameP0, "r")) == NULL) {
        std::cout << "No such file\n";

    }
    /////////////////////////READ SIGMA MATRIX///////////////////////////////////////////////////////////////////////////////////////
    /*Sigma matrix will be readed as vector
     */
    //read vector dimension
    int SigmaVecDim, SigmaColsInMatrix, SigmaNonzerrosInMatrix;
    int fscanRes = fscanf(f_Sigma, "%d %d %d\n", &SigmaVecDim, &SigmaColsInMatrix, &SigmaNonzerrosInMatrix);
    double* Sigma = new_host_darray(SigmaNonzerrosInMatrix);
    std::cout << "Sigma vectordim:" << SigmaVecDim << " Cols in matrix:" << SigmaColsInMatrix << " Nonzero " << SigmaNonzerrosInMatrix << "\n";
    int rowIndex;
    int colIndex;
    double value;
    for (int i = 0; i < SigmaNonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_Sigma, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        Sigma[rowIndex - 1] = value;
    }

    //////////////////////END READ SIGMA MATRIX////////READ G MATRIX/////////////////////////////////////////////////////////////
    int G_VecDim, G_ColsInMatrix, G_NonzerrosInMatrix;
    fscanRes = fscanf(f_G, "%d %d %d\n", &G_VecDim, &G_ColsInMatrix, &G_NonzerrosInMatrix);
    std::cout << "G vectordim:" << G_VecDim << " Cols in matrix:" << G_ColsInMatrix << " Nonzero " << G_NonzerrosInMatrix << "\n";

    coo_matrix G_matrix;
    G_matrix.num_cols = 0;
    G_matrix.num_rows = 0;
    G_matrix.num_nonzeros = 0;
    G_matrix.I = NULL;
    G_matrix.J = NULL;
    G_matrix.V = NULL;

    G_matrix.num_rows = G_VecDim;
    G_matrix.num_nonzeros = G_NonzerrosInMatrix;
    G_matrix.num_cols = G_ColsInMatrix;
    G_matrix.I = new_host_iarray(G_matrix.num_nonzeros);
    G_matrix.J = new_host_iarray(G_matrix.num_nonzeros);
    G_matrix.V = new_host_darray(G_matrix.num_nonzeros);

    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_G, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        //std::cout<< ">>" << rowIndex << " " << colIndex << " " << value <<  "\n";
        //if(rowIndex != 1){//Пропускаем первую строчку с целевой функцией
        G_matrix.I[i] = rowIndex - 1; //coo_matrix.I[i] = rowIndex - 1;
        G_matrix.J[i] = colIndex - 1;
        G_matrix.V[i] = value;
        //}

    }

    //print_coo_matrix(G_matrix);
    //////////////////////////////////END READ G MATRIX/////////START READ C matrix///////////////////////////////////////////
    int C_VecDim, C_ColsInMatrix, C_NonzerrosInMatrix;
    fscanRes = fscanf(f_C, "%d %d %d\n", &C_VecDim, &C_ColsInMatrix, &C_NonzerrosInMatrix);
    std::cout << "C vectordim:" << C_VecDim << " Cols in matrix:" << C_ColsInMatrix << " Nonzero " << C_NonzerrosInMatrix << "\n";
    double* C = new_host_darray(C_ColsInMatrix);
    for (int i = 0; i < C_ColsInMatrix; i++) {
        C[i] = 0.0;
    }

    for (int i = 0; i < C_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_C, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        C[colIndex - 1] = value;
    }

    //printMatrixCPU(1, C_ColsInMatrix, C);
    //////////////////////////////////END READ G MATRIX/////////START READ P0 matrix///////////////////////////////////////////
    int P0_VecDim, P0_ColsInMatrix, P0_NonzerrosInMatrix;
    fscanRes = fscanf(f_P0, "%d %d %d\n", &P0_VecDim, &P0_ColsInMatrix, &P0_NonzerrosInMatrix);
    std::cout << "P0 vectordim:" << P0_VecDim << " Cols in matrix:" << P0_ColsInMatrix << " Nonzero " << P0_NonzerrosInMatrix << "\n";
    double* P0 = new_host_darray(P0_ColsInMatrix);
    P0_VecDim = P0_ColsInMatrix;
    for (int i = 0; i < P0_ColsInMatrix; i++) {
        P0[i] = 0.0;
    }

    for (int i = 0; i < P0_NonzerrosInMatrix; i++) {
        fscanRes = fscanf(f_P0, "%d %d %lg\n", &rowIndex, &colIndex, &value);
        P0[colIndex - 1] = value;
    }
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    ////////////////////////////////END READ P0 MATRIX///////////////////////////////////////////////////////////////////////
    int outMatrixRows = G_VecDim * 2 + P0_VecDim;
    int outMatrixCols = G_ColsInMatrix + 1;
    int outMatrixNonz = G_NonzerrosInMatrix * 2 + P0_NonzerrosInMatrix + G_VecDim * 2 + P0_VecDim;

    coo_matrix out_matrix;
    //out_matrix.num_cols = G_ColsInMatrix + 1;
    //out_matrix.num_rows = G_VecDim * 2 + P0_VecDim;
    //out_matrix.num_nonzeros = outMatrixNonz;
    out_matrix.I = NULL;
    out_matrix.J = NULL;
    out_matrix.V = NULL;

    out_matrix.num_rows = outMatrixRows;
    out_matrix.num_nonzeros = outMatrixNonz;
    out_matrix.num_cols = outMatrixCols;
    out_matrix.I = new_host_iarray(outMatrixNonz);
    out_matrix.J = new_host_iarray(outMatrixNonz);
    out_matrix.V = new_host_darray(outMatrixNonz);
    for (int i; i < outMatrixNonz; i++) {
        out_matrix.I[i] = 0;
        out_matrix.J[i] = 0;
        out_matrix.V[i] = 0.0;
    }

    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        out_matrix.V[i] = G_matrix.V[i] * sqrt(Sigma[G_matrix.J[i]]);
        out_matrix.I[i] = G_matrix.I[i];
        out_matrix.J[i] = G_matrix.J[i];

        out_matrix.V[i + G_NonzerrosInMatrix] = -G_matrix.V[i] * sqrt(Sigma[G_matrix.J[i]]);
        out_matrix.I[i + G_NonzerrosInMatrix] = G_matrix.I[i] + G_VecDim;
        out_matrix.J[i + G_NonzerrosInMatrix] = G_matrix.J[i];
    }


    std::cout << "Nonzer constrains\n";

    for (int i = 2 * G_NonzerrosInMatrix, j = 2 * G_VecDim; i < 2 * G_NonzerrosInMatrix + P0_NonzerrosInMatrix; i++, j++) {
        out_matrix.V[i] = -1.0;
        out_matrix.I[i] = j;
        out_matrix.J[i] = j - 2 * G_VecDim;
    }


    /////////////////////////FILL RHS/////////////////////////////////////////////////
    //printMatrixCPU(1, P0_ColsInMatrix, P0);
    std::cout << "Compute RHS\n";
    double* G1Xa = new_host_darray(G_VecDim);
    for (int i = 0; i < G_VecDim; i++) {
        G1Xa[i] = 0;
    }
    std::cout << "null\n";
    for (int i = 0; i < G_NonzerrosInMatrix; i++) {
        G1Xa[G_matrix.I[i]] += G_matrix.V[i] * P0[G_matrix.J[i]];
        //std::cout<<"G1Xa["<< G_matrix.I[i] << "] =  "<< G1Xa[G_matrix.I[i]] << " G " << G_matrix.V[i] << " P0["<< G_matrix.J[i] <<" ]" << P0[G_matrix.J[i]] << "\n";
    }

    //printMatrixCPU(1, G_VecDim, G1Xa);
    //!!!!!!!!!!!NOTE we mast add RHS with -1 coefficient
    std::cout << "Add RHS\n";
    int outMatrixShift = 2 * G_NonzerrosInMatrix + P0_NonzerrosInMatrix;
    for (int i = outMatrixShift, j = 0; i < outMatrixShift + G_VecDim; i++, j++) {
        out_matrix.V[i] = -(C[j] - G1Xa[j]); //with -1 coefficient
        out_matrix.I[i] = j;
        out_matrix.J[i] = G_ColsInMatrix;

        out_matrix.V[i + G_VecDim] = (C[j] - G1Xa[j]); //with -1 coeff
        out_matrix.I[i + G_VecDim] = j + G_VecDim;
        out_matrix.J[i + G_VecDim] = G_ColsInMatrix;
    }
    std::cout << "Add nonzeros RHS\n";
    outMatrixShift = 2 * G_NonzerrosInMatrix + P0_NonzerrosInMatrix + 2 * G_VecDim;
    for (int i = outMatrixShift, j = 0; i < outMatrixShift + P0_VecDim; i++, j++) {
        out_matrix.V[i] = -P0[j] / sqrt(Sigma[j]); //with -1 coeff
        out_matrix.I[i] = j + 2 * G_VecDim;
        out_matrix.J[i] = G_ColsInMatrix;
    }

    //print_coo_matrix(out_matrix);
    std::cout << "\n";
    fclose(f_Sigma);
    fclose(f_C);
    fclose(f_G);
    fclose(f_P0);

    return out_matrix;
}



//template <typename int, typename double>

coo_matrix printLPFile(csr_matrix taskMatrix) {

    std::ofstream myfile;
    myfile.open("task.lp");
    myfile << "Maximize\n";
    for (int i = 1; i < taskMatrix.num_cols - 1; i++) {
        myfile << "x" << i << "+\n";
    }
    myfile << "x" << taskMatrix.num_cols - 1 << "\n";

    myfile << "Subject To\n";

    //print_csr_matrix(taskMatrix);

    for (int i = 0; i < taskMatrix.num_rows; i++) {
        myfile << taskMatrix.Ax[taskMatrix.Ap[i]] << " x" << taskMatrix.Aj[taskMatrix.Ap[i]] + 1;
        for (int j = taskMatrix.Ap[i] + 1; j < taskMatrix.Ap[i + 1] - 1; j++) {
            //printf("%i %i %f \n", i, csr.Aj[j], csr.Ax[j]);
            //std::cout << i << " " << taskMatrix.Aj[j] << " " << taskMatrix.Ax[j] << "\n";
            myfile << " + " << taskMatrix.Ax[j] << " x" << taskMatrix.Aj[j] + 1;
        }
        //myfile << taskMatrix.Ax[taskMatrix.Ap[i + 1] - 1] << " * x" << taskMatrix.Aj[taskMatrix.Ap[i + 1] - 1] +  1;
        myfile << " <= " << taskMatrix.Ax[ taskMatrix.Ap[i + 1] - 1 ];
        myfile << "\n";
    }

    myfile << "End\n";

    myfile.close();

}

//template <typename int, typename double>

coo_matrix parseLPFile(char* file_name) {
    string lpFileTags;
    ifstream infile(file_name);
    if (!infile) {
        cout << "Cannot open file.\n";
    }
    getline(infile, lpFileTags);
    //string
    while (getline(infile, lpFileTags)) {
        //while(infile >> lpFileTags){
        if (lpFileTags == "Subject To") {
            cout << "AGAGAY\n";
        }
    }
    infile.close();
}


//template <typename T>

double rand2(double low, double high) {
    //srandom(seed);
    //srandom (1593);
    double t = double(rand()) / double(RAND_MAX);

    //printf("%e\n", t);
    //printf(">>> %e, %e\n", (T) rand(), (T)RAND_MAX);
    return (1.0 - t) * low + t * high;

}


//template <typename T>

void genInputSet(int vector_dim, int inputSetDim, double* inputSet) {
    FILE *stream;
    if ((stream = fopen("/dev/random", "r")) == NULL)
        printf("We all gonna die!!! UnixWare doesn't have /dev/random as well as Windows\n");

    unsigned int seed = 0;
    /*while (!feof(stream)) {
        fread((void*) & seed, sizeof (unsigned int), 1, stream);
        printf("Got seed %u\n", seed);
    }*/
    int rez = fread((void*) & seed, sizeof (unsigned int), 1, stream);
    srandom(seed);
    fclose(stream);


    for (int vecCount = 0; vecCount < inputSetDim - 1; vecCount++) {
        int vector_base_el = vecCount * vector_dim;
        for (int elemCount = 0; elemCount < vector_dim - 1; elemCount++) {
            if (elemCount == vecCount) {
                inputSet[vector_base_el + elemCount] = 100.0 * rand2(0.0, 1.0);
            } else {
                inputSet[vector_base_el + elemCount] = 0.0;
            }
            //seed = (int)(randT2(0,1, seed) *1000);
        }
    }


    for (int elemCount = 0; elemCount < vector_dim; elemCount++) {
        inputSet[vector_dim * (inputSetDim - 1) + elemCount] = 0.0;
    }
    //printMatrix(vector_dim, inputSetDim, inputSet);
    double *c = (double*) malloc(vector_dim * sizeof (double));
    /*for (int elemCount = 0; elemCount < vector_dim; elemCount++) {
        c[elemCount] = inputSet[elemCount * inputSetDim + elemCount] / inputSetDim;
        if(c[elemCount] != 0.0){
            printf("%f %i %f \n", c[elemCount], elemCount, inputSet[elemCount * inputSetDim + elemCount]);
        }
    }*/
    for (int elemCount = 0; elemCount < inputSetDim; elemCount++) {
        c[elemCount] = inputSet[vector_dim * elemCount + elemCount] / inputSetDim;
        //if(c[elemCount] != 0.0){
        //printf("%f %i %f \n", c[elemCount], elemCount, inputSet[vector_dim * elemCount + elemCount]);
        //}
    }
    //printMatrixCPU(vector_dim, 1, c);

    /*for (int elemCount = 0; elemCount < vector_dim; elemCount++) {
        int vector_base_el = elemCount * inputSetDim;
        for (int vecCount = 0; vecCount < inputSetDim; vecCount++) {
            inputSet[vector_base_el + vecCount] -= c[elemCount];
        }
    }*/
    for (int colId = 0; colId < inputSetDim; colId++) {
        int vector_base = colId * vector_dim;
        for (int strId = 0; strId < vector_dim; strId++) {
            inputSet[vector_base + strId] -= c[strId];
        }
    }
    //printMatrix(vector_dim, inputSetDim, inputSet);
    for (int vecCount = 0; vecCount < inputSetDim; vecCount++) {
        inputSet[(vector_dim - 1) + vector_dim * vecCount] = 1e-3;
    }
    //printMatrixCPU(vector_dim, inputSetDim, inputSet);
    free(c);
}

//template <typename T>

void genAnotherInputSet(int vector_dim, int inputSetDim, double* inputSet) {
    FILE *stream;
    if ((stream = fopen("/dev/random", "r")) == NULL)
        printf("We all gonna die!!! UnixWare doesn't have /dev/random as well as Windows\n");

    unsigned int seed = 0;
    /*while (!feof(stream)) {
        fread((void*) & seed, sizeof (unsigned int), 1, stream);
        printf("Got seed %u\n", seed);
    }*/
    int rez = fread((void*) & seed, sizeof (unsigned int), 1, stream);
    srandom(seed);
    //srandom (1593);
    //T t = (T) rand() / (T) RAND_MAX;
    //return (1.0 - t) * low + t * hight;
    fclose(stream);

    double scale1 = 100.0;
    double scale2 = 100.0;
    double shift = 1;
    for (int vecCount = 0; vecCount < inputSetDim; vecCount++) {
        int vector_base_el = vecCount * vector_dim;
        for (int elemCount = 0; elemCount < vector_dim; elemCount++) {
            if (elemCount != (vector_dim - 1)) {
                //fread((void*) & seed, sizeof (unsigned int), 1, stream);
                inputSet[vector_base_el + elemCount] = rand2(0.0, 1.0) * scale1;
            } else {
                //fread((void*) & seed, sizeof (unsigned int), 1, stream);

                inputSet[vector_base_el + elemCount] = rand2(0.0, 1.0) * scale2 + shift;
            }
            seed = (int) (rand2(0.0, 1.0) *1000);
        }
    }


    //printMatrix(vector_dim, inputSetDim, inputSet);

    //printMatrix(vector_dim, inputSetDim, inputSet);
    //for (int elemCount = 0; elemCount < vector_dim; elemCount++) {
    //    inputSet[vector_dim * (inputSetDim - 1) + elemCount] = 0.0;
    //}
    //printMatrix(vector_dim, inputSetDim, inputSet);
    double *sum2 = (double*) malloc((vector_dim - 1) * sizeof (double));
    for (int elemCount = 0; elemCount < vector_dim - 1; elemCount++) {
        sum2[elemCount] = 0;
    }
    for (int j = 0; j < inputSetDim; j++) {
        int bEl = j * vector_dim;
        for (int i = 0; i < vector_dim - 1; i++) {
            sum2[i] += inputSet[bEl + i];
        }
    }

    for (int j = 0; j < inputSetDim; j++) {
        int bEl = j * vector_dim;
        for (int i = 0; i < vector_dim - 1; i++) {
            inputSet[bEl + i] = scale1 * (inputSet[bEl + i] - sum2[i] / (double) inputSetDim);
        }
    }


    //printMatrix(vector_dim, inputSetDim, inputSet);
    free(sum2);
}

void test_get_ldl33_up_from_ldl_l_upper(){
    int cols = 4;
    int rows = 4;
    ldl_matrix ldlm = new_ldl_matrix(4, 10);
    
    
    ldlm.num_cols = cols;
    ldlm.num_rows = rows;
    int elts_num = 0;
    for (int i = 0; i < cols; i++) {
        ldlm.Lp[i] = elts_num;
        for (int j = i; j < rows; j++) {            
            ldlm.Li[elts_num] = j;
            ldlm.Lx[elts_num] = (double)elts_num;            
            elts_num++;
        }
    }
    ldlm.Lp[cols] = elts_num;
    
    std::cout << "Diag part\n";
    for (int i = 0; i < ldlm.num_cols; i++) {
        ldlm.D[i] = i;
    }
    ldl_matrix ldlmt = new_ldl_matrix(4, 10);
    ldl_transpose(ldlm, ldlmt);
    print_ldl_matrix(ldlm);
    
    
    ldl_matrix ldl33 = get_ldl33_up_from_ldl_l_upper(ldlmt, 1);
    
    print_ldl_matrix(ldl33);
}


void test_recompute_l33_d33_for_ldl_col_del(){
    
    int cols = 4;
    int rows = 4;
    ldl_matrix ldlm = new_ldl_matrix(cols, 10);
    
    
    ldlm.num_cols = cols;
    ldlm.num_rows = rows;
    int elts_num = 0;
    for (int i = 0; i < cols - 1; i++) {
        ldlm.Lp[i] = elts_num;
        for (int j = i + 1; j < rows; j++) {            
            ldlm.Li[elts_num] = j;
            ldlm.Lx[elts_num] = rand2(1,100);   
            ldlm.num_nonzeros++;
            elts_num++;
        }
    }
    ldlm.Lp[cols - 1] = elts_num;
    ldlm.Lp[cols] = elts_num; //This is bug, but I will dill with it
    
    std::cout << "Diag part\n";
    for (int i = 0; i < ldlm.num_cols; i++) {
        ldlm.D[i] = i + 1;
    }
    
    std::cout << "LDL\n";
    print_ldl_matrix(ldlm);
    //for(int i = 0; i < ldlm.num_nonzeros; i++){ std::cout << ldlm.Li[i] << " ";   }
    
    ldl_matrix ldlmt = new_ldl_matrix(cols, 10);
    ldl_transpose(ldlm, ldlmt);
    
    std::cout << "LDL^t  "<< "nonzeros ->" <<ldlmt.num_nonzeros << " cols-> " << ldlmt.num_cols << "\n";
    //printMatrixCPU(1, ldlmt.num_cols+1, ldlmt.Lx);
    //for(int i = 0; i < ldlmt.num_nonzeros; i++){ std::cout << ldlmt.Li[i] << " ";   }
    //std::cout << "\n";
    print_ldl_matrix(ldlmt);
    
    ldl_matrix ldl33_up = get_ldl33_up_from_ldl_l_upper(ldlmt, 1);
    ldl_matrix ldl33_low = new_ldl_matrix(ldl33_up.num_cols, ldl33_up.num_nonzeros);
    ldl_transpose(ldl33_up, ldl33_low);
    
    std::cout << "LDL33_low\n";
    print_ldl_matrix(ldl33_low);
    
    
    int l32_id = 0;
    int l32_dem = ldlm.num_rows - (l32_id + 1);
    double* dense_row = new_host_darray(l32_dem);
    get_ldl_dense_row_from_l_upper(ldlmt, l32_id, dense_row);
    
    std::cout<<"l32 \n";
    printMatrixCPU(1, l32_dem, dense_row);
    
    ldl_matrix ldl33_new = recompute_l33_d33_for_ldl_col_del(ldl33_low, dense_row, ldlmt.D[l32_id]);
    
    
    std::cout << "LDL33 recomputed\n";
    print_ldl_matrix(ldl33_new);
}
    

void try_ldl_engine(){
   const int N = 10;
    const int ANZ = 19;
    const int LNZ = 13;
     int    Ap [N+1] = {0, 1, 2, 3, 4,   6, 7,   9,   11,      15,     ANZ},
           Ai [ANZ] = {0, 1, 2, 3, 1,4, 5, 4,6, 4,7, 0,4,7,8, 1,4,6,9 } ;
    double Ax [ANZ] = {1.7, 1., 1.5, 1.1, .02,2.6, 1.2, .16,1.3, .09,1.6,
                     .13,.52,.11,1.4, .01,.53,.56,3.1},
           b [N] = {.287, .22, .45, .44, 2.486, .72, 1.55, 1.424, 1.621, 3.759};
    double Lx [LNZ], D [N], Y [N] ;
    int Li [LNZ], Lp [N+1], Parent [N], Lnz [N], Flag [N], Pattern [N], d, i ;

    /* factorize A into LDL' (P and Pinv not used) */
    ldl_symbolic (N, Ap, Ai, Lp, Parent, Lnz, Flag, NULL, NULL) ;
    printf ("Nonzeros in L, excluding diagonal: %d\n", Lp [N]) ;
    d = ldl_numeric (N, Ap, Ai, Ax, Lp, Parent, Lnz, Li, Lx, D, Y, Pattern,
        Flag, NULL, NULL) ;
    
    for(int i = 0; i < N+1; i++){
        std::cout << " " << Lp[i] << "";
    }
    std::cout<<"\n";
    for (int i = 0; i < N; i++) {
        for (int j = Lp[i]; j < Lp[i + 1]; j++) {
            //printf("%i %i %f \n", i, csr.Aj[j], csr.Ax[j]);
            std::cout << i << " " << Li[j] << " " << Lx[j] << "\n";
        }
    }
    std::cout << "Diag part\n";
    for (int i = 0; i < N; i++) {
        std::cout << i << " " << D[i] << "\n";
    }

}


void test_del_row_from_ldl_up(){
    int cols = 4;
    int rows = 4;
    ldl_matrix ldlm = new_ldl_matrix(cols, 10);
    
    
    ldlm.num_cols = cols;
    ldlm.num_rows = rows;
    int elts_num = 0;
    for (int i = 0; i < cols - 1; i++) {
        ldlm.Lp[i] = elts_num;
        for (int j = i + 1; j < rows; j++) {            
            ldlm.Li[elts_num] = j;
            ldlm.Lx[elts_num] = rand2(1,100);   
            ldlm.num_nonzeros++;
            elts_num++;
        }
    }
    ldlm.Lp[cols - 1] = elts_num;
    ldlm.Lp[cols] = elts_num; //This is bug, but I will dill with it
    
    std::cout << "Diag part\n";
    for (int i = 0; i < ldlm.num_cols; i++) {
        ldlm.D[i] = i + 1;
    }
    
    std::cout << "LDL\n";
    print_ldl_matrix(ldlm);
    //for(int i = 0; i < ldlm.num_nonzeros; i++){ std::cout << ldlm.Li[i] << " ";   }
    
    ldl_matrix ldlmt = new_ldl_matrix(cols, 10);
    ldl_transpose(ldlm, ldlmt);
    
    std::cout << "LDL^t  "<< "nonzeros ->" <<ldlmt.num_nonzeros << " cols-> " << ldlmt.num_cols << "\n";
    //printMatrixCPU(1, ldlmt.num_cols+1, ldlmt.Lx);
    //for(int i = 0; i < ldlmt.num_nonzeros; i++){ std::cout << ldlmt.Li[i] << " ";   }
    //std::cout << "\n";
    print_ldl_matrix(ldlmt);
    
    del_row_col_from_ldl_up(ldlmt, 1);
    
    print_ldl_matrix(ldlmt);
}


int main(int argc, char **argv) {
    //prepare_matrix();//<int, double>();
    //test_simplex_projection();
    
    //test_get_ldl33_up_from_ldl_l_upper();
    //test_recompute_l33_d33_for_ldl_col_del();
    
    //try_ldl_engine();
    test_del_row_from_ldl_up();
}

