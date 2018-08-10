#include "mg_transfer_mf_p.h"

#include <navierstokes/config.h>

typedef dealii::LinearAlgebra::distributed::Vector<float>  vfloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> vdouble;

#if DIM_2 && DEGREE_15 && DEGREE_7 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 15, 7, float, vfloat>;
#endif

#if DIM_2 && DEGREE_15 && DEGREE_7 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 15, 7, double, vdouble>;
#endif

#if DIM_2 && DEGREE_14 && DEGREE_7 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 14, 7, float, vfloat>;
#endif

#if DIM_2 && DEGREE_14 && DEGREE_7 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 14, 7, double, vdouble>;
#endif

#if DIM_2 && DEGREE_13 && DEGREE_4 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 13, 6, float, vfloat>;
#endif

#if DIM_2 && DEGREE_13 && DEGREE_6 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 13, 6, double, vdouble>;
#endif

#if DIM_2 && DEGREE_12 && DEGREE_6 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 12, 6, float, vfloat>;
#endif

#if DIM_2 && DEGREE_12 && DEGREE_6 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 12, 6, double, vdouble>;
#endif

#if DIM_2 && DEGREE_11 && DEGREE_5 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 11, 5, float, vfloat>;
#endif

#if DIM_2 && DEGREE_11 && DEGREE_5 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 11, 5, double, vdouble>;
#endif

#if DIM_2 && DEGREE_10 && DEGREE_5 && MG_FLOAT
template class MGTransferMatrixFreeP<2, 10, 5, float, vfloat>;
#endif

#if DIM_2 && DEGREE_10 && DEGREE_5 && MG_DOUBLE
template class MGTransferMatrixFreeP<2, 10, 5, double, vdouble>;
#endif

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


#if DIM_3 && DEGREE_15 && DEGREE_7 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 15, 7, float, vfloat>;
#endif

#if DIM_3 && DEGREE_15 && DEGREE_7 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 15, 7, double, vdouble>;
#endif

#if DIM_3 && DEGREE_14 && DEGREE_7 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 14, 7, float, vfloat>;
#endif

#if DIM_3 && DEGREE_14 && DEGREE_7 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 14, 7, double, vdouble>;
#endif

#if DIM_3 && DEGREE_13 && DEGREE_4 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 13, 6, float, vfloat>;
#endif

#if DIM_3 && DEGREE_13 && DEGREE_6 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 13, 6, double, vdouble>;
#endif

#if DIM_3 && DEGREE_12 && DEGREE_6 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 12, 6, float, vfloat>;
#endif

#if DIM_3 && DEGREE_12 && DEGREE_6 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 12, 6, double, vdouble>;
#endif

#if DIM_3 && DEGREE_11 && DEGREE_5 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 11, 5, float, vfloat>;
#endif

#if DIM_3 && DEGREE_11 && DEGREE_5 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 11, 5, double, vdouble>;
#endif

#if DIM_3 && DEGREE_10 && DEGREE_5 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 10, 5, float, vfloat>;
#endif

#if DIM_3 && DEGREE_10 && DEGREE_5 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 10, 5, double, vdouble>;
#endif

#if DIM_3 && DEGREE_9 && DEGREE_4 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 9, 4, float, vfloat>;
#endif

#if DIM_3 && DEGREE_9 && DEGREE_4 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 9, 4, double, vdouble>;
#endif

#if DIM_3 && DEGREE_8 && DEGREE_4 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 8, 4, float, vfloat>;
#endif

#if DIM_3 && DEGREE_8 && DEGREE_4 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 8, 4, double, vdouble>;
#endif

#if DIM_3 && DEGREE_7 && DEGREE_3 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 7, 3, float, vfloat>;
#endif

#if DIM_3 && DEGREE_7 && DEGREE_3 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 7, 3, double, vdouble>;
#endif

#if DIM_3 && DEGREE_6 && DEGREE_3 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 6, 3, float, vfloat>;
#endif

#if DIM_3 && DEGREE_6 && DEGREE_3 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 6, 3, double, vdouble>;
#endif

#if DIM_3 && DEGREE_5 && DEGREE_2 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 5, 2, float, vfloat>;
#endif

#if DIM_3 && DEGREE_5 && DEGREE_2 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 5, 2, double, vdouble>;
#endif

#if DIM_3 && DEGREE_4 && DEGREE_2 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 4, 2, float, vfloat>;
#endif

#if DIM_3 && DEGREE_4 && DEGREE_2 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 4, 2, double, vdouble>;
#endif

#if DIM_3 && DEGREE_3 && DEGREE_1 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 3, 1, float, vfloat>;
#endif

#if DIM_3 && DEGREE_3 && DEGREE_1 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 3, 1, double, vdouble>;
#endif

#if DIM_3 && DEGREE_2 && DEGREE_1 && MG_FLOAT
template class MGTransferMatrixFreeP<3, 2, 1, float, vfloat>;
#endif

#if DIM_3 && DEGREE_2 && DEGREE_1 && MG_DOUBLE
template class MGTransferMatrixFreeP<3, 2, 1, double, vdouble>;
#endif