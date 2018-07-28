#include "multigrid_preconditioner_adapter_base.h"

#include <navier-constants.h>

#if DIM_2 && MG_FLOAT
template class MyMultigridPreconditionerBase<2,float, MatrixOperatorBaseNew<2,float>>;
template class MyMultigridPreconditionerBase<2,double, MatrixOperatorBaseNew<2,float>>;
#endif

//#if DIM_2 && MG_DOUBLE
//template class MyMultigridPreconditionerBase<2,double, MatrixOperatorBaseNew<2,double>>;
//#endif

#if DIM_3 && MG_FLOAT
template class MyMultigridPreconditionerBase<3,float, MatrixOperatorBaseNew<3,float>>;
template class MyMultigridPreconditionerBase<3,double, MatrixOperatorBaseNew<3,float>>;
#endif

//#if DIM_3 && MG_DOUBLE
//template class MyMultigridPreconditionerBase<3,double, MatrixOperatorBaseNew<3,double>>;
//#endif