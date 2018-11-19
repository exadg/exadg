/*
 * operator.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: fehn
 */

#include "operator.h"

namespace IncNS
{

namespace Interface
{
// nothing to implement here because all member functions of interface class Operator are pure virtual.

// instantiations
#include <navierstokes/config.h>

#if OP_FLOAT
template class OperatorBase<float>;
template class OperatorCoupled<float>;
template class OperatorDualSplitting<float>;
template class OperatorPressureCorrection<float>;
template class OperatorOIF<float>;
#endif

#if OP_DOUBLE
template class OperatorBase<double>;
template class OperatorCoupled<double>;
template class OperatorDualSplitting<double>;
template class OperatorPressureCorrection<double>;
template class OperatorOIF<double>;
#endif

}

}

