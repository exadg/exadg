#include <navierstokes/config.h>
#include "multigrid_preconditioner_base.h"

#if DIM_2 && OP_FLOAT && MG_FLOAT
template class MultigridPreconditionerBase<2, float, float>;
#endif
#if DIM_2 && OP_DOUBLE && MG_FLOAT
template class MultigridPreconditionerBase<2, double, float>;
#endif
#if DIM_2 && OP_DOUBLE && MG_DOUBLE
template class MultigridPreconditionerBase<2, double, double>;
#endif

#if DIM_3 && OP_FLOAT && MG_FLOAT
template class MultigridPreconditionerBase<3, float, float>;
#endif
#if DIM_3 && OP_DOUBLE && MG_FLOAT
template class MultigridPreconditionerBase<3, double, float>;
#endif
#if DIM_3 && OP_DOUBLE && MG_DOUBLE
template class MultigridPreconditionerBase<3, double, double>;
#endif
