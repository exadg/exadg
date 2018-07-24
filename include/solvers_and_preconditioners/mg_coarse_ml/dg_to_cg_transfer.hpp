#include "dg_to_cg_transfer.h"

#include "../../../applications/macros/constants.h"

#ifdef DIM_2
template class CGToDGTransfer<2, float>;
template class CGToDGTransfer<2, double>;
#endif

#ifdef DIM_3
template class CGToDGTransfer<3, float>;
template class CGToDGTransfer<3, double>;
#endif
