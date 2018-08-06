#include "multigrid_preconditioner_adapter_base.h"

#include <navier_constants.h>

#if DIM_2 && OP_FLOAT && MG_FLOAT
template class MyMultigridPreconditionerBase<2,float, MatrixOperatorBaseNew<2,float>>;
#endif
#if DIM_2 && OP_DOUBLE && MG_FLOAT
template class MyMultigridPreconditionerBase<2,double, MatrixOperatorBaseNew<2,float>>;
#endif
#if DIM_2 && OP_DOUBLE && MG_DOUBLE
template class MyMultigridPreconditionerBase<2,double, MatrixOperatorBaseNew<2,double>>;
#endif

#if DIM_3 && OP_FLOAT && MG_FLOAT
template class MyMultigridPreconditionerBase<3,float, MatrixOperatorBaseNew<3,float>>;
#endif
#if DIM_3 && OP_DOUBLE && MG_FLOAT
template class MyMultigridPreconditionerBase<3,double, MatrixOperatorBaseNew<3,float>>;
#endif
#if DIM_3 && OP_DOUBLE && MG_DOUBLE
template class MyMultigridPreconditionerBase<3,double, MatrixOperatorBaseNew<3,double>>;
#endif
