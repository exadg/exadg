#include "mg_coarse_ml_dg.h"

#include <navier-constants.h>

#ifdef DIM_2
template class MGCoarseMLDG<2, float>;
template class MGCoarseMLDG<2, double>;
#endif

#ifdef DIM_3
template class MGCoarseMLDG<3, float>;
template class MGCoarseMLDG<3, double>;
#endif
