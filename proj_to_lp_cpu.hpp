/* 
 * File:   proj_to_lp_cpu.hpp
 * Author: pozpl
 *
 * Created on 25 Май 2010 г., 12:46
 */

#ifndef _PROJ_TO_LP_CPU_HPP
#define	_PROJ_TO_LP_CPU_HPP

template <typename IndexType, typename ValueType> void proj_to_lp_cpu(csr_matrix<IndexType, ValueType> &inputSet,
        ValueType * minNormVector, IndexType inSetDim, IndexType vectorDim, ValueType tollerance);

#include "proj_to_lp_cpu.cpp"
#endif	/* _PROJ_TO_LP_CPU_HPP */

