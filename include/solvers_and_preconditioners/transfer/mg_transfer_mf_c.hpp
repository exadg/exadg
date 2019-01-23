#include "mg_transfer_mf_c.h"

#include <navierstokes/config.h>

#if DIM_2
template class MGTransferMFC<2, float, LinearAlgebra::distributed::Vector<float>, 1>;
template class MGTransferMFC<2, double, LinearAlgebra::distributed::Vector<double>, 1>;

template class MGTransferMFC<2, float, LinearAlgebra::distributed::Vector<float>, 2>;
template class MGTransferMFC<2, double, LinearAlgebra::distributed::Vector<double>, 2>;
#endif

#if DIM_3
template class MGTransferMFC<3, float, LinearAlgebra::distributed::Vector<float>, 1>;
template class MGTransferMFC<3, double, LinearAlgebra::distributed::Vector<double>, 1>;

template class MGTransferMFC<3, float, LinearAlgebra::distributed::Vector<float>, 3>;
template class MGTransferMFC<3, double, LinearAlgebra::distributed::Vector<double>, 3>;
#endif
