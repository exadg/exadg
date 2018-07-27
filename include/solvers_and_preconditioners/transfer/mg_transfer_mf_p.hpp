#include "mg_transfer_mf_p.h"

#include <navier-constants.h>

typedef dealii::LinearAlgebra::distributed::Vector<float> vfloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> vdouble;

#if DIM_2 && DEGREE_9 && DEGREE_4 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 9, 4, float, vfloat>;
#endif

#if DIM_2 && DEGREE_9 && DEGREE_4 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 9, 4, double, vdouble>;
#endif

#if DIM_2 && DEGREE_8 && DEGREE_4 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 8, 4, float, vfloat>;
#endif

#if DIM_2 && DEGREE_8 && DEGREE_4 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 8, 4, double, vdouble>;
#endif

#if DIM_2 && DEGREE_7 && DEGREE_3 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 7, 3, float, vfloat>;
#endif

#if DIM_2 && DEGREE_7 && DEGREE_3 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 7, 3, double, vdouble>;
#endif

#if DIM_2 && DEGREE_6 && DEGREE_3 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 6, 3, float, vfloat>;
#endif

#if DIM_2 && DEGREE_6 && DEGREE_3 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 6, 3, double, vdouble>;
#endif

#if DIM_2 && DEGREE_5 && DEGREE_2 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 5, 2, float, vfloat>;
#endif

#if DIM_2 && DEGREE_5 && DEGREE_2 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 5, 2, double, vdouble>;
#endif

#if DIM_2 && DEGREE_4 && DEGREE_2 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 4, 2, float, vfloat>;
#endif

#if DIM_2 && DEGREE_4 && DEGREE_2 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 4, 2, double, vdouble>;
#endif

#if DIM_2 && DEGREE_3 && DEGREE_1 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 3, 1, float, vfloat>;
#endif

#if DIM_2 && DEGREE_3 && DEGREE_1 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 3, 1, double, vdouble>;
#endif

#if DIM_2 && DEGREE_2 && DEGREE_1 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 2, 1, float, vfloat>;
#endif

#if DIM_2 && DEGREE_2 && DEGREE_1 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 2, 1, double, vdouble>;
#endif