#include "sparse_types.hpp"

////////////////////////////////////////////////////////////////////////////////
//! sparse matrix memory management
////////////////////////////////////////////////////////////////////////////////

//template <typename int, typename double>
void delete_dia_matrix(dia_matrix &dia){
    free(dia.diag_offsets);  free(dia.diag_data);
}

//template <typename int, typename double>
void delete_ell_matrix(ell_matrix & ell){
    free(ell.Aj);  free(ell.Ax);
}

//template <typename int, typename double>
void delete_csr_matrix(csr_matrix& csr){
    free(csr.Ap);  free(csr.Aj);   free(csr.Ax);
}

//template <typename int, typename double>
void delete_csc_matrix(csc_matrix& csc){
    free(csc.Cp);  free(csc.Ri);   free(csc.Ex);
}

//template <typename int, typename double>
void delete_coo_matrix(coo_matrix& coo){
    free(coo.I);   free(coo.J);   free(coo.V);
}

//template <typename int, typename double>
void delete_packet_array(packet_array& pa){
    free(pa.index_array); free(pa.data_array);
    free(pa.pos_start);   free(pa.pos_end);
}

//template <typename int, typename double>
void delete_hyb_matrix(hyb_matrix& hyb){
    delete_ell_matrix(hyb.ell);
    delete_coo_matrix(hyb.coo);
}

//template <typename int, typename double>
void delete_pkt_matrix(pkt_matrix& pm)
{
    free(pm.row_ptr);
    free(pm.permute_new_to_old);
    free(pm.permute_old_to_new);

    delete_packet_array(pm.packets);
    delete_coo_matrix(pm.coo);
}


void delete_ldl_matrix(ldl_matrix& ldl_m){
    free(ldl_m.Lp);
    free(ldl_m.Li);
    free(ldl_m.Lx);
    free(ldl_m.Lnz);
    free(ldl_m.Parent);
    free(ldl_m.D);
}
////////////////////////////////////////////////////////////////////////////////
//! host functions
////////////////////////////////////////////////////////////////////////////////

//template <typename int, typename double>
void delete_host_matrix(dia_matrix& dia){ delete_dia_matrix(dia); }

//template <typename int, typename double>
void delete_host_matrix(ell_matrix& ell){ delete_ell_matrix(ell); }

//template <typename int, typename double>
void delete_host_matrix(coo_matrix& coo){ delete_coo_matrix(coo); }

//template <typename int, typename double>
void delete_host_matrix(csr_matrix& csr){ delete_csr_matrix(csr); }

//template <class int, class double>
void delete_host_matrix(hyb_matrix& hyb){  delete_hyb_matrix(hyb); }

//template <typename int, typename double>
void delete_host_matrix(pkt_matrix& pm){ delete_pkt_matrix(pm); }


double* new_host_darray(const size_t N){
    double* darray = (double*) malloc(N * sizeof(double));
    for(int i=0; i<N; i++){
        darray[i] = 0.0;
    }
    return darray;
}

int* new_host_iarray(const size_t N){
    int* iarray = (int*) malloc(N * sizeof(int));
    for(int i=0; i<N; i++){
        iarray[i] = 0.0;
    }
    return iarray;
}


ldl_matrix new_ldl_matrix(int max_dim, int max_nonzer){
    ldl_matrix ldl;
    ldl.num_nonzeros = 0;
    ldl.num_cols = 0;
    ldl.num_rows = 0;
    ldl.D = new_host_darray(max_dim);
    ldl.Lp = new_host_iarray(max_dim);
    ldl.Li = new_host_iarray(max_nonzer);
    ldl.Lx = new_host_darray(max_nonzer);
    ldl.Lnz = new_host_iarray(max_dim);
    ldl.Parent = new_host_iarray(max_dim);
    ldl.MAX_D_LP_SIZE = max_dim;
    ldl.MAX_Li_Lx_SIZE = max_nonzer;
    
    return ldl;
}

void delete_host_array(double *p)   { free(p); }