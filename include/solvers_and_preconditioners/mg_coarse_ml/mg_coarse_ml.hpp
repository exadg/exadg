#include "mg_coarse_ml.h"

#include "../../operators/matrix_operator_base_new.h"

template class MGCoarseML<MatrixOperatorBaseNew<2, float>, float>;
template class MGCoarseML<MatrixOperatorBaseNew<2, float>, double>;
