#include "mass_operator.h"

namespace ConvDiff
{
template class MassMatrixOperator<2, float>;
template class MassMatrixOperator<2, double>;

template class MassMatrixOperator<3, float>;
template class MassMatrixOperator<3, double>;

} // namespace ConvDiff
