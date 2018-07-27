#include "mg_coarse_ml_cg.h"

#include <navier-constants.h>

#ifdef DIM_2
template class MGCoarseMLCG<2, float>;
template class MGCoarseMLCG<2, double>;
#endif

#ifdef DIM_3
template class MGCoarseMLCG<3, float>;
template class MGCoarseMLCG<3, double>;
#endif
