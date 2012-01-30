/* 
 * File:   sparce_types.hpp
 * Author: pozpl
 *
 * Created on 24 Май 2010 г., 14:09
 */

#ifndef _SPARCE_TYPES_HPP
#define	_SPARCE_TYPES_HPP

#include <stdio.h>
#include <stdlib.h>







////////////////////////////////////////////////////////////////////////////////
//! Defines the following sparse matrix formats
//
// DIA - Diagonal
// ELL - ELLPACK/ITPACK
// CSR - Compressed Sparse Row
// CSC - Compressed Sparse Column
// COO - Coordinate
// PKT - Packet
////////////////////////////////////////////////////////////////////////////////

//template<typename int>
struct matrix_shape
{
    typedef int index_type;
    int num_rows, num_cols, num_nonzeros;
};

// DIAgonal matrix
//template <typename int, typename double>
struct dia_matrix : public matrix_shape
{
    typedef int index_type;
    typedef double value_type;

    int stride;
    int num_diags;

    int       * diag_offsets;  //diagonal offsets (must be a signed type)
    double * diag_data;     //nonzero values stored in a (num_diags x num_cols) matrix
};

// ELLPACK/ITPACK matrix format
//template <typename int, typename double>
struct ell_matrix : public matrix_shape
{
    typedef int index_type;
    typedef double value_type;

    int stride;
    int num_cols_per_row;

    int * Aj;           //column indices stored in a (cols_per_row x stride) matrix
    double * Ax;           //nonzero values stored in a (cols_per_row x stride) matrix
};

// COOrdinate matrix (aka IJV or Triplet format)
//template <typename int, typename double>
struct coo_matrix : public matrix_shape
{
    typedef int index_type;
    typedef double value_type;

    int * I;  //row indices
    int * J;  //column indices
    double * V;  //nonzero values
};


/*
 *  Compressed Sparse Row matrix (aka CRS)
 */
//template <typename int, typename double>
struct csr_matrix : public matrix_shape
{
    typedef int index_type;
    typedef double value_type;

    int * Ap;  //row pointer
    int * Aj;  //column indices
    double * Ax;  //nonzeros
};

struct csc_matrix : public matrix_shape
{
    typedef int index_type;
    typedef double value_type;

    int * Cp;  //row pointer
    int * Ri;  //column indices
    double * Ex;  //nonzeros
};



struct ldl_matrix : public matrix_shape
{
    typedef int index_type;
    typedef double value_type;
    
    int MAX_D_LP_SIZE;
    int MAX_Li_Lx_SIZE;
    
    int * Lp;  //row pointer Lp holds the cumulative sum of Lnz.
    int * Li;  //column indices
    double * Lx;  //nonzeros
    int* Lnz; //holds the counts of the number of entries in each column of L
    int* Parent;//elimination tree Parent - n dimention
    double* D; //identity matrix ( n-dimension vector actually)
};

/*
 *  Hybrid ELL/COO format
 */
//template <typename int, typename double>
struct hyb_matrix : public matrix_shape
{
    typedef int index_type;
    typedef double value_type;

    ell_matrix ell; //ELL portion
    coo_matrix coo; //COO portion
};



/*
 *  Packet matrix
 */
typedef unsigned int Packedint;
//template <typename int, typename double>
struct packet_array : public matrix_shape
{
    typedef int index_type;
    typedef double value_type;

    Packedint * index_array;  // compressed row/col indices
    double * data_array;         // nonzero values

    int * pos_start;          // start ptr into index and data arrays for each thread
    int * pos_end;

    int total_cycles;         // total amount of work in each thread lane
};

//template <typename int, typename double>
struct pkt_matrix : public matrix_shape
{
    typedef int index_type;
    typedef double value_type;

    int threads_per_packet;    // # of threads in a block, e.g. 256
    int max_rows_per_packet;   // maximum over all packets

    int num_packets;
    int * row_ptr;             // packet i corresponds to rows row_ptr[i] through row_ptr[i+1] - 1
    int * permute_old_to_new;
    int * permute_new_to_old;

    packet_array packets;
    coo_matrix coo;
};

// store row index in upper 16 bits and col index in lower 16 bits
#define pkt_pack_indices(row,col)          (  (row << 16) + col  )
#define pkt_unpack_row_index(packed_index) ( packed_index >> 16  )
#define pkt_unpack_col_index(packed_index) (packed_index & 0xFFFF)


////////////////////////////////////////////////////////////////////////////////
//! sparse matrix memory management
////////////////////////////////////////////////////////////////////////////////
ldl_matrix new_ldl_matrix(int max_dim, int max_nonzer);

double* new_host_darray(const size_t N);
int* new_host_iarray(const size_t N);


//template <typename int, typename double>
void delete_dia_matrix(dia_matrix &dia);

//template <typename int, typename double>
void delete_ell_matrix(ell_matrix & ell);

//template <typename int, typename double>
void delete_csr_matrix(csr_matrix& csr);

void delete_csc_matrix(csc_matrix& csc);
//template <typename int, typename double>
void delete_coo_matrix(coo_matrix& coo);

//template <typename int, typename double>
void delete_packet_array(packet_array& pa);

//template <typename int, typename double>
void delete_hyb_matrix(hyb_matrix& hyb);

//template <typename int, typename double>
void delete_pkt_matrix(pkt_matrix& pm);


////////////////////////////////////////////////////////////////////////////////
//! host functions
////////////////////////////////////////////////////////////////////////////////

//template <typename int, typename double>
void delete_host_matrix(dia_matrix& dia);

//template <typename int, typename double>
void delete_host_matrix(ell_matrix& ell);
//template <typename int, typename double>
void delete_host_matrix(coo_matrix& coo);

//template <typename int, typename double>
void delete_host_matrix(csr_matrix& csr);
//template <class int, class double>
void delete_host_matrix(hyb_matrix& hyb);
//template <typename int, typename double>
void delete_host_matrix(pkt_matrix& pm);



void delete_host_array(double *p);

void delete_ldl_matrix(ldl_matrix& ldl_m);




#endif	/* _SPARCE_TYPES_HPP */

