#include "mg_coarse_ml.h"

#include <navier-constants.h>

#include "../../operators/matrix_operator_base_new.h"

#if DIM_2 && OP_FLOAT && MG_FLOAT
template class MGCoarseML<MatrixOperatorBaseNew<2, float>, float>;
#endif
#if DIM_2 && OP_DOUBLE && MG_FLOAT
template class MGCoarseML<MatrixOperatorBaseNew<2, float>, double>;
#endif
#if DIM_2 && OP_DOUBLE && MG_DOUBLE
template class MGCoarseML<MatrixOperatorBaseNew<2, double>, double>;
#endif

#if DIM_3 && OP_FLOAT && MG_FLOAT
template class MGCoarseML<MatrixOperatorBaseNew<3, float>, float>;
#endif
#if DIM_3 && OP_DOUBLE && MG_FLOAT
template class MGCoarseML<MatrixOperatorBaseNew<3, float>, double>;
#endif
#if DIM_3 && OP_DOUBLE && MG_DOUBLE
template class MGCoarseML<MatrixOperatorBaseNew<3, double>, double>;
#endif
