#include "mg_transfer_mf_c.h"

#include <navierstokes/config.h>

#if DIM_2
template class MGTransferMFC<2, float>;
template class MGTransferMFC<2, double>;
#endif

#if DIM_3
template class MGTransferMFC<3, float>;
template class MGTransferMFC<3, double>;
#endif
