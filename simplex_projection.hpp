/* 
 * File:   simplex_projection.hpp
 * Author: pozpl
 *
 * Created on 2 Декабрь 2011 г., 13:12
 */

#ifndef SIMPLEX_PROJECTION_HPP
#define	SIMPLEX_PROJECTION_HPP
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

extern "C" {
    #include <atlas/atlas_enum.h>
    #include "clapack.h"
    #include "cblas.h"
    #include <ldl.h>
}


#include "sparse_operations_ext_cpu.hpp"
#include "sparse_types.hpp"
#include "conus_projection_sparse.hpp"

void getSimplexProjection(csc_matrix &inputSet,
        double * minNormVector, double tollerance,
        int* kvec_in, int* basisVecInx_in, int &baselen_in);
void findOptZInBasisCPU(csc_matrix& inSet, int &baselen, int vectorDim, int basisInc,
        csr_matrix &basis, csr_matrix &basis_t, int *basisVecInx, int *kvec, csr_matrix& grammMatr, double *invGrammMatr,
        double *mu, double *mu_old, double *z, double eps, ldl_matrix& grammPartFactor);
void evalMuForSimplexCPUwithStoredMatrix(int basisInc, csr_matrix basis, csr_matrix basis_t,
        csr_matrix& grammMatrParted, ldl_matrix& grammPartFactor, double *mu, int delBasElIndx);
void getMinVCosSimplexCPU(double* z, csc_matrix &inSet, double& minVcos, int &minVecId, double epsilon, double* l_norms);
double attractDotCoeffToSimplexCPU(csr_matrix &basis, double *mu_old, double *mu, int &delBasElIndx, int baselen);

#endif	/* SIMPLEX_PROJECTION_HPP */

