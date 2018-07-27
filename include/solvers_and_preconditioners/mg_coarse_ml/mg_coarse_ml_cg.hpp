#include "mg_coarse_ml_cg.h"

#include <navier-constants.h>

#if DIM_2
template class MGCoarseMLCG<2, float>;
template class MGCoarseMLCG<2, double>;
#endif

#if DIM_3
template class MGCoarseMLCG<3, float>;
template class MGCoarseMLCG<3, double>;
#endif
