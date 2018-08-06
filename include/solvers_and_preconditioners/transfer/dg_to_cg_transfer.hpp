#include "dg_to_cg_transfer.h"

#include <navier_constants.h>

#if DIM_2 && MG_FLOAT
template class CGToDGTransfer<2, float>;
#endif

#if DIM_2 && MG_DOUBLE
template class CGToDGTransfer<2, double>;
#endif

#if DIM_3 && MG_FLOAT
template class CGToDGTransfer<3, float>;
#endif

#if DIM_3 && MG_DOUBLE
template class CGToDGTransfer<3, double>;
#endif
