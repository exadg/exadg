#include "mg_transfer_mf_p.h"

#include <navierstokes/config.h>

typedef dealii::LinearAlgebra::distributed::Vector<float>  vfloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> vdouble;

#if DIM_2 && MG_DOUBLE
template class MGTransferMFP<2, double, vdouble>;
#endif

#if DIM_2 && MG_FLOAT
template class MGTransferMFP<2, float, vfloat>;
#endif

#if DIM_3 && MG_DOUBLE
template class MGTransferMFP<3, double, vdouble>;
#endif

#if DIM_3 && MG_FLOAT
template class MGTransferMFP<3, float, vfloat>;
#endif
