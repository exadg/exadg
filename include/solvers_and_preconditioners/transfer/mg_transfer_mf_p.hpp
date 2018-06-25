#include "mg_transfer_mf_p.h"

typedef dealii::LinearAlgebra::distributed::Vector<float> vfloat;

template class MGTransferMatrixFreeP<2, 9, 4, float, vfloat>;
template class MGTransferMatrixFreeP<2, 8, 4, float, vfloat>;
template class MGTransferMatrixFreeP<2, 7, 3, float, vfloat>;
template class MGTransferMatrixFreeP<2, 6, 3, float, vfloat>;
template class MGTransferMatrixFreeP<2, 5, 2, float, vfloat>;
template class MGTransferMatrixFreeP<2, 4, 2, float, vfloat>;
template class MGTransferMatrixFreeP<2, 3, 1, float, vfloat>;
template class MGTransferMatrixFreeP<2, 2, 1, float, vfloat>;
