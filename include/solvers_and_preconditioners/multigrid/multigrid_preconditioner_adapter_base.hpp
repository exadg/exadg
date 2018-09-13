#include "multigrid_preconditioner_adapter_base.h"

#include <navierstokes/config.h>

#if DIM_2 && OP_FLOAT && MG_FLOAT
template class MyMultigridPreconditionerBase<2, float, MultigridOperatorBase<2, float>>;
#endif
#if DIM_2 && OP_DOUBLE && MG_FLOAT
template class MyMultigridPreconditionerBase<2, double, MultigridOperatorBase<2, float>>;
#endif
#if DIM_2 && OP_DOUBLE && MG_DOUBLE
template class MyMultigridPreconditionerBase<2, double, MultigridOperatorBase<2, double>>;
#endif

#if DIM_3 && OP_FLOAT && MG_FLOAT
template class MyMultigridPreconditionerBase<3, float, MultigridOperatorBase<3, float>>;
#endif
#if DIM_3 && OP_DOUBLE && MG_FLOAT
template class MyMultigridPreconditionerBase<3, double, MultigridOperatorBase<3, float>>;
#endif
#if DIM_3 && OP_DOUBLE && MG_DOUBLE
template class MyMultigridPreconditionerBase<3, double, MultigridOperatorBase<3, double>>;
#endif
